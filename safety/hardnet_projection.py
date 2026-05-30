"""
Differentiable CBF Projection Layer — Stage 4a Phase 4 (HardNet)
================================================================
A PyTorch module that projects a raw action onto the CBF-defined safe set,
**differentiably**, so it can be embedded inside an RL policy network and
trained end-to-end (gradients flow through the projection during the policy
update).

This is the "HardNet" component in the safe-RL sense: a hard-constraint
output layer that guarantees the (deterministic) policy output satisfies
the CBF constraints, while letting the policy *learn* — through gradients
that respect the constraint geometry — how to produce raw actions whose
projection achieves high reward. Contrast with the external CBF filter
(safety/cbf_qp_hocbf.py), which corrects actions at inference time with no
gradient signal to the policy.

Safe set (per sample in the batch):
    C(x) = { u ∈ R^4 :  A_i · u ≥ b_i  for i=1..4   (CBF half-spaces)
                        −1 ≤ u_j ≤ 1   for j=1..4   (action box) }

where A = L_g h (scaled to normalized action units) and b = −L_f h − α·h
are supplied per-step by the environment (see CBFContextWrapper). Rows with
‖A_i‖ ≈ 0 are inactive (e.g. FOV constraints when the target is out of
frame) and are skipped.

Projection algorithm: **Dykstra's alternating projection**. Unlike plain
POCS (alternating projections), Dykstra converges to the *true* Euclidean
projection (closest feasible point) onto the intersection of convex sets,
which gives well-behaved gradients (∂u_safe/∂u_raw is the identity on the
interior, and the projection operator on the active faces). Each elementary
projection (onto one half-space, or onto the box) is closed-form and
differentiable, so unrolling K iterations is differentiable via autograd.

Infeasible case: if C(x) is empty (the CBF half-spaces and box don't
intersect — rare, happens deep in a corner), Dykstra does not converge to a
feasible point; it oscillates. We cap iterations and end with a box
projection, so the output is always at least in the action box. The
environment-side HOCBF filter remains available as a hard backstop during
rollout if desired.
"""

import torch
import torch.nn as nn


class CBFProjection(nn.Module):
    """Differentiable projection onto CBF half-spaces ∩ action box.

    Args:
        n_iters:    Number of Dykstra sweeps. Each sweep projects onto all
                    active half-spaces and the box once. 15–25 is plenty for
                    a 4-D problem with ≤4 half-spaces.
        eps:        Rows of A with squared-norm below this are treated as
                    inactive (skipped). Also guards division.
        box_low,
        box_high:   Action box bounds (defaults ∓1 for the [-1,1]^4 space).
    """

    def __init__(self, n_iters: int = 20, eps: float = 1e-8,
                 box_low: float = -1.0, box_high: float = 1.0):
        super().__init__()
        self.n_iters = int(n_iters)
        self.eps = float(eps)
        self.box_low = float(box_low)
        self.box_high = float(box_high)

    def forward(self, u_raw: torch.Tensor, A: torch.Tensor,
                b: torch.Tensor) -> torch.Tensor:
        """Project a batch of raw actions onto their per-sample safe sets.

        Args:
            u_raw: (B, m) raw actions from the policy network.
            A:     (B, k, m) CBF constraint normals (k constraints, m action
                   dims). Row i encodes A_i · u ≥ b_i.
            b:     (B, k) CBF constraint right-hand sides.

        Returns:
            u_safe: (B, m) projected actions, each in the safe set (or in the
                    box if the safe set is empty).
        """
        B, k, m = A.shape
        # Active-constraint mask: ‖A_i‖² > eps (inactive rows are skipped).
        a_norm2 = (A * A).sum(dim=-1)                      # (B, k)
        active = (a_norm2 > self.eps).to(u_raw.dtype)      # (B, k)
        # Avoid div-by-zero on inactive rows (their coeff is masked to 0).
        a_norm2_safe = torch.clamp(a_norm2, min=self.eps)  # (B, k)

        u = u_raw
        # Dykstra correction terms — one per set: k half-spaces + 1 box.
        p = torch.zeros(B, k, m, dtype=u_raw.dtype, device=u_raw.device)
        q = torch.zeros(B, m, dtype=u_raw.dtype, device=u_raw.device)

        for _ in range(self.n_iters):
            # --- Project onto each active half-space (Dykstra) ---
            for i in range(k):
                a_i = A[:, i, :]                            # (B, m)
                y = u + p[:, i, :]                          # add correction
                # Half-space projection of y onto {u : a_i·u ≥ b_i}:
                #   if a_i·y < b_i:  y + (b_i − a_i·y)/‖a_i‖² · a_i
                #   else:            y   (already satisfied)
                viol = b[:, i] - (a_i * y).sum(dim=-1)      # (B,) ; >0 = violated
                coeff = torch.clamp(viol, min=0.0) / a_norm2_safe[:, i]
                coeff = coeff * active[:, i]                # zero on inactive rows
                u_new = y + coeff.unsqueeze(-1) * a_i
                p[:, i, :] = y - u_new                      # update correction
                u = u_new

            # --- Project onto the action box (Dykstra) ---
            y = u + q
            u_new = torch.clamp(y, self.box_low, self.box_high)
            q = y - u_new
            u = u_new

        return u
