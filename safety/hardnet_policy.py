"""
HardNet Policy — Stage 4a Phase 4
==================================
A custom Stable-Baselines3 ActorCriticPolicy that embeds the differentiable
CBF projection layer (safety/hardnet_projection.py) on the action mean.

The policy consumes an augmented observation:

    obs = [ base_obs (16) | cbf_context (20) ]

where cbf_context = [ A.flatten() (16) , b (4) ] are the per-step CBF
constraint coefficients (A = scaled L_g h, b = −L_f h − α·h), supplied by
the CBFContextWrapper. The base_obs is the usual 16-D IBVS observation that
the network actually conditions on; the context is consumed only by the
projection layer (which has no learned parameters).

Network wiring:
  base_obs (16) ─► SlicingFlattenExtractor ─► MLP ─► action_net ─► mean_raw (4)
                                                       │
            cbf_context (20) ─► (A, b) ──────────────► CBFProjection ─► mean_safe (4)
                                                       │
                                          DiagGaussian(mean_safe, log_std)

Because the projection is differentiable and sits between action_net and the
action distribution, the PPO policy-gradient update backpropagates through
it — the policy learns to emit raw means whose *projection* maximizes
return, with gradients that respect the safe-set geometry. The deterministic
output (eval / deployment) is mean_safe, which is hard-constrained to the
CBF safe set (modulo the infeasible-corner fallback noted in
hardnet_projection.py).

Warm-starting: the MLP / action_net / value_net see only the 16-D base obs
(via the slicing extractor), so their shapes match a vanilla MlpPolicy
trained on the 16-D observation. Weights transfer directly with
`load_state_dict(..., strict=False)` — see train.py's HardNet warm-start.
"""

import torch
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import DiagGaussianDistribution

from safety.hardnet_projection import CBFProjection


class SlicingFlattenExtractor(BaseFeaturesExtractor):
    """Features extractor that keeps only the first `n_base` obs dims.

    The augmented observation carries the CBF context in its tail; the
    network must not condition its features on those coefficients (they're
    used only by the projection), and keeping the feature dim at `n_base`
    makes weights interchangeable with a vanilla 16-D MlpPolicy.
    """

    def __init__(self, observation_space: spaces.Box, n_base: int = 16):
        super().__init__(observation_space, features_dim=n_base)
        self.n_base = int(n_base)
        self.flatten = nn.Flatten()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.flatten(observations)[:, :self.n_base]


class HardNetActorCriticPolicy(ActorCriticPolicy):
    """ActorCriticPolicy with a differentiable CBF projection on the mean.

    Extra kwargs (passed via policy_kwargs):
        n_base:        Dimension of the base observation (default 16).
        n_constraints: Number of CBF half-space constraints (default 4).
        proj_iters:    Dykstra iterations in the projection (default 20).
    """

    def __init__(self, *args, n_base: int = 16, n_constraints: int = 4,
                 proj_iters: int = 20, max_log_std: float = None, **kwargs):
        self._n_base = int(n_base)
        self._n_constraints = int(n_constraints)
        self._proj_iters = int(proj_iters)
        # Intervention C: optional hard cap on log_std. The baseline HardNet
        # run saw action std blow up to 2.68 (log_std≈0.99), diffusing the
        # stochastic rollouts PPO learns from. Capping log_std at, e.g., 0.0
        # (std≤1.0) prevents that runaway while still allowing exploration.
        # None = no cap (original behavior).
        self._max_log_std = max_log_std
        # Force our slicing extractor (keeps feature dim = n_base).
        kwargs['features_extractor_class'] = SlicingFlattenExtractor
        fe_kwargs = dict(kwargs.get('features_extractor_kwargs', {}) or {})
        fe_kwargs.setdefault('n_base', self._n_base)
        kwargs['features_extractor_kwargs'] = fe_kwargs
        super().__init__(*args, **kwargs)
        # Action dim m (inferred from the action space).
        self._m = int(self.action_space.shape[0])
        self.projection = CBFProjection(n_iters=self._proj_iters)
        # Scratch slot for the per-call CBF context (set in the public
        # methods, consumed in _get_action_dist_from_latent).
        self._cbf_context = None

    # ---------------------------------------------------------------- #
    # Context plumbing                                                  #
    # ---------------------------------------------------------------- #
    def _split_context(self, obs: torch.Tensor):
        """Extract (A, b) from the tail of the augmented observation.

        Args:
            obs: (B, n_base + n_constraints*m + n_constraints) tensor.

        Returns:
            (A, b): A is (B, k, m), b is (B, k).
        """
        k, m = self._n_constraints, self._m
        context = obs[:, self._n_base:]
        A = context[:, :k * m].reshape(-1, k, m)
        b = context[:, k * m: k * m + k]
        return A, b

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        """Compute the action distribution, projecting the mean onto the
        CBF safe set using the stashed context."""
        mean_raw = self.action_net(latent_pi)
        if self._cbf_context is not None:
            A, b = self._cbf_context
            mean_safe = self.projection(mean_raw, A, b)
        else:
            mean_safe = mean_raw
        if isinstance(self.action_dist, DiagGaussianDistribution):
            log_std = self.log_std
            if self._max_log_std is not None:
                # Differentiable upper clamp: keeps gradient below the cap,
                # zeroes it above. Prevents the std runaway (Intervention C).
                log_std = torch.clamp(log_std, max=self._max_log_std)
            return self.action_dist.proba_distribution(mean_safe, log_std)
        raise ValueError(
            "HardNetActorCriticPolicy only supports DiagGaussianDistribution "
            "(continuous actions)."
        )

    # ---------------------------------------------------------------- #
    # Intervention D: feasibility-loss support                          #
    # ---------------------------------------------------------------- #
    def feasibility_terms(self, obs: torch.Tensor) -> dict:
        """Recompute raw vs projected means and projection stats for a batch.

        Used by the D-PPO subclass to (a) add the auxiliary feasibility loss
        λ·mean‖u_raw − u_safe‖² and (b) log instrumentation. Mirrors the
        actor forward path: extract pi-features → mlp_extractor.forward_actor
        → action_net → projection.

        Returns dict with:
            'dist'        : (B,) per-sample ‖u_raw − u_safe‖ (Euclidean).
            'sq_dist_mean': scalar mean ‖u_raw − u_safe‖² (the aux loss term).
            'active'      : (B,) bool — projection moved the action
                            (dist > tol), i.e. u_raw was infeasible.
            'n_active_constraints': (B,) count of CBF rows with ‖A_i‖>0
                            (FOV constraints drop out when target not in FOV).
        """
        # Actor feature path (shared or pi-specific extractor)
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        u_raw = self.action_net(latent_pi)
        A, b = self._split_context(obs)
        u_safe = self.projection(u_raw, A, b)
        diff = u_raw - u_safe
        sq = (diff * diff).sum(dim=-1)              # (B,)
        dist = torch.sqrt(sq + 1e-12)
        active = dist > 1e-4
        a_norm2 = (A * A).sum(dim=-1)               # (B, k)
        n_active = (a_norm2 > 1e-8).sum(dim=-1)     # (B,)
        return {
            'dist': dist,
            'sq_dist_mean': sq.mean(),
            'active': active,
            'n_active_constraints': n_active,
        }

    # ---------------------------------------------------------------- #
    # Public overrides: stash context, then defer to the base impl     #
    # ---------------------------------------------------------------- #
    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        self._cbf_context = self._split_context(obs)
        try:
            return super().forward(obs, deterministic=deterministic)
        finally:
            self._cbf_context = None

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        self._cbf_context = self._split_context(obs)
        try:
            return super().evaluate_actions(obs, actions)
        finally:
            self._cbf_context = None

    def get_distribution(self, obs: torch.Tensor):
        self._cbf_context = self._split_context(obs)
        try:
            return super().get_distribution(obs)
        finally:
            self._cbf_context = None
