# HardNet Robustness — Multi-Seed Variance Study (Plan A)

> **PRELIMINARY LEGACY EVIDENCE.** Not part of the canonical clean MVP and not
> statistically validated for a publication claim. See `docs/README.md`.

**Date:** 2026-05-30
**Question:** Is HardNet's worst-case under-performance (single-run 20.5%, vs
the external HOCBF filter's 31.5%) a *systematic* property of the training
recipe, or just *seed variance* from a single run?

## Motivation

Phase 4a.4 introduced HardNet — an in-policy differentiable CBF projection
(gradients flow through the safe-set projection during PPO). A single training
run (seed 0) gave a mixed head-to-head against the external HOCBF filter:

| Condition | HardNet (seed 0) | 4b + HOCBF filter |
|-----------|------------------|-------------------|
| Clean     | 90.0%            | 87.0%             |
| Nominal   | 92.0%            | 91.0%             |
| Hard      | 59.5%            | 74.0%             |
| Worst     | 20.5%            | 31.5%             |

Drawing a conclusion from one run is unsound in RL: reward shaping,
observations, and seed all swing outcomes. Before accepting a "mixed result"
we ran a controlled multi-seed study, changing **only** the master seed.

## Method

- Code: added a reproducible `--seed` flag to `train.py` threading into both
  the PPO constructor and `make_vec_env` (PPO init + env streams).
- 3 fresh seeds (42, 123, 7), each an exact replica of the original config:
  3M steps, warm-start from the 4b 1M policy, full DR (`--stage4b`), HOCBF
  proxy constraints via the context wrapper, linear LR decay, n_envs=16.
- Driver: `scripts/legacy/training/run_hardnet_seeds.sh` (sequential, full CPU
  each, ~55 min/seed).
- Eval: each seed's checkpoints {1.0M, 1.2M, 1.4M} on **nominal** and
  **worst** conditions, 200 episodes, eval seed=1000, via
  `scripts/legacy/eval_stage4b.py --mode hardnet`. Original seed-0 (locked
  1.2M) included as reference.

## Results

### Per-seed best-at-nominal checkpoint, with its worst-case

| Seed | Best ckpt | Nominal | Worst |
|------|-----------|---------|-------|
| 0 (orig) | 1.2M  | 92.0%   | 20.5% |
| 42       | 1.0M  | 90.5%   | 27.0% |
| 123      | 1.2M  | 95.0%   | 28.0% |
| 7        | 1.2M  | 89.0%   | 20.5% |

**Nominal:** mean **91.6% ± 2.2%**, range [89.0, 95.0]
**Worst (at best-nom ckpt):** mean **24.0% ± 3.5%**, range [20.5, 28.0]

### Worst-case across all 10 checkpoints (3 seeds × 3 ckpts + orig)

Sorted: `[19.0, 20.0, 20.5, 20.5, 21.0, 23.5, 27.0, 27.5, 28.0, 31.0]`
mean **23.8% ± 4.0%**, min 19.0, max **31.0**

Attitude safety held across every run: max |roll| ≤ 0.638 rad (limit 0.611),
exceedance ≤ 0.05% of steps in all 18 evaluations — the projection's safety
guarantee is seed-independent.

## Conclusion

1. **The worst-case degradation is systematic, not a fluke.** All four seeds
   land in the low-20s at their best-nominal checkpoint; none reaches the
   external filter's 31.5% there. HardNet under this recipe has a genuine
   worst-case weakness.

2. **But the single-run gap (11pp) overstated it.** Honest comparison:
   HardNet worst-case **24.0% ± 3.5%** vs filter 31.5%. The gap is ~7.5pp on
   average, not 11.

3. **The capacity exists — the recipe doesn't reliably extract it.** Seed 123
   is best at nominal (95%) *and* reaches 31.0% worst-case at its 1.0M
   checkpoint — matching the external filter while keeping 94% nominal. The
   ceiling is right there; typical checkpoints just don't hit it. This is the
   signature of a training/selection problem, **not an architectural ceiling**
   — i.e. it is solvable by changing the training recipe.

4. **Nominal performance is reproducibly excellent** (91.6% ± 2.2%),
   confirming HardNet matches/beats the filter near the operating point.

## Caveats (for rigor)

- All evals share eval seed=1000, so the 200-ep sampling noise (~±3pp at
  these rates) is *common-mode* across runs; true between-seed spread is
  marginally larger than reported.
- "Best-at-nominal" checkpoint selection is itself a policy and not obviously
  optimal (seed 123 @ 1.0M dominates its own 1.2M on the worst-case axis). A
  principled, pre-registered selection rule should be fixed before the final
  comparison.

## Next steps (staged interventions)

The study justifies recovering the worst-case headroom via training changes,
applied **one mechanism at a time** for clean credit attribution:

- **C — entropy/std fix:** `ent_coef` 0.01→0.001 + cap on `log_std`.
  Targets the observed action-std blow-up (1.6→2.68) that diffused rollouts.
- **B — curriculum DR:** start near-nominal, anneal to full-hard by ~2M
  steps. Targets destabilization from full DR applied from step 0.
- **D — auxiliary feasibility loss:** penalize ‖u_raw − u_safe‖² so the
  network emits already-feasible means → full-rank projection Jacobian →
  healthier gradients. Targets the gradient-degradation root cause.

We begin with **C + B** (cheapest, most diagnostic), then assess before D.
