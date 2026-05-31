"""
HardNet-D PPO — Intervention D (auxiliary feasibility loss)
============================================================
A PPO subclass that adds an auxiliary feasibility loss to the standard PPO
objective, plus instrumentation for the gradient-degradation hypothesis.

Total loss:
    L = L_PPO  +  λ · mean‖u_raw − u_safe‖²

where u_raw is the policy network's raw mean and u_safe is its CBF
projection. The term pulls raw means toward the safe set so the projection
rarely activates → its Jacobian stays ≈ identity (full rank) → gradients
flow cleanly. The executed action is ALWAYS u_safe regardless (the env
sees the projected action), so this term changes only the network's
internal representation, not behavior — it is a gradient-conditioning
device, not a behavioral constraint.

λ kept small (default 0.05) to avoid a "lazy policy" that settles for a
feasible-but-mediocre action because the aux term dominates the (relatively
down-weighted) reward gradient.

Instrumentation (logged to TensorBoard each train() call):
    feas/aux_loss              mean ‖u_raw−u_safe‖² over minibatches
    feas/proj_active_frac      fraction of samples where projection moved u
    feas/raw_safe_dist_mean    mean ‖u_raw−u_safe‖
    feas/active_when_fov       proj-active frac on FOV-ACTIVE samples
                               (n_active_constraints == 4: nominal-like)
    feas/active_when_nofov     proj-active frac on FOV-INACTIVE samples
                               (n_active_constraints == 2: worst-like, target
                                out of frame so only pitch/roll constrained)
The fov-split is the key diagnostic: the gradient hypothesis predicts the
projection fires far less on FOV-inactive (worst-like) steps, which would
explain why a projection-targeting fix helps nominal more than worst-case.
"""

import numpy as np
import torch as th
from torch.nn import functional as F
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance


class HardNetDPPO(PPO):
    """PPO + auxiliary feasibility loss for the HardNet policy."""

    def __init__(self, *args, feasibility_coef: float = 0.05, **kwargs):
        self.feasibility_coef = float(feasibility_coef)
        super().__init__(*args, **kwargs)

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        # D instrumentation accumulators
        feas_losses, proj_active, raw_safe_dist = [], [], []
        active_fov, active_nofov = [], []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf)
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # --- Intervention D: auxiliary feasibility loss ---
                feas = self.policy.feasibility_terms(rollout_data.observations)
                feas_loss = feas['sq_dist_mean']

                loss = (policy_loss
                        + self.ent_coef * entropy_loss
                        + self.vf_coef * value_loss
                        + self.feasibility_coef * feas_loss)

                # --- instrumentation (no grad) ---
                with th.no_grad():
                    feas_losses.append(feas_loss.item())
                    active = feas['active']
                    proj_active.append(active.float().mean().item())
                    raw_safe_dist.append(feas['dist'].mean().item())
                    n_act = feas['n_active_constraints']
                    fov_mask = n_act >= 4         # all 4 constraints → in-FOV
                    nofov_mask = n_act <= 2        # only pitch/roll → out-of-FOV
                    if fov_mask.any():
                        active_fov.append(active[fov_mask].float().mean().item())
                    if nofov_mask.any():
                        active_nofov.append(active[nofov_mask].float().mean().item())

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Standard PPO logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        # D instrumentation
        self.logger.record("feas/coef", self.feasibility_coef)
        self.logger.record("feas/aux_loss", float(np.mean(feas_losses)))
        self.logger.record("feas/proj_active_frac", float(np.mean(proj_active)))
        self.logger.record("feas/raw_safe_dist_mean", float(np.mean(raw_safe_dist)))
        if active_fov:
            self.logger.record("feas/active_when_fov", float(np.mean(active_fov)))
        if active_nofov:
            self.logger.record("feas/active_when_nofov", float(np.mean(active_nofov)))
