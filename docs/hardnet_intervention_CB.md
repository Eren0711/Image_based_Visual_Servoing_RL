# HardNet Robustness — Intervention C+B (Entropy/Std Fix + Curriculum DR)

**Date:** 2026-05-31
**Question:** Does stabilizing exploration (C) and smoothing domain
randomization (B) recover HardNet's worst-case gap vs the external filter?

## Interventions

- **C — entropy/std fix:** `ent_coef` 0.01→0.001; hard cap `log_std ≤ 0`
  (effective std ≤ 1.0). Targets the baseline std blowup (raw log_std rose
  until std≈2.68), which was diffusing the rollouts PPO learns from.
- **B — curriculum DR:** anneal the DR band easy→full-hard over the first
  2M of 3M steps (then hold full-hard). `CurriculumCallback` broadcasts
  frac=min(1, t/2M) to all envs; `WindWrapper`/`IntermittentDetectionWrapper`
  interpolate their sampling bands. Targets early destabilization from full
  DR at step 0.

Identical to baseline otherwise: 3M steps, warm-start from 4b 1M, HOCBF
context, linear LR decay, 3 seeds (42, 123, 7).

## Results (200-ep eval, seed=1000, vs baseline variance study)

|                          | Baseline HardNet | C+B HardNet | Δ      |
|--------------------------|------------------|-------------|--------|
| Nominal (best-nom ckpt)  | 91.5% ± 2.5      | 87.7% ± 0.8 | −3.8pp |
| Worst   (best-nom ckpt)  | 25.2% ± 3.3      | 21.8% ± 2.5 | −3.4pp |
| Nominal (all 9 ckpts)    | 90.2% ± 3.1      | 85.2% ± 2.6 |        |
| Worst   (all 9 ckpts)    | 24.2% ± 4.1      | 21.2% ± 2.3 |        |
| Worst-case CEILING       | 31% (=filter)    | 24%         | −7pp   |

External HOCBF filter worst-case reference: 31.5%.

## Conclusion — NEGATIVE RESULT (informative)

C+B **did not recover** the worst-case gap; it slightly **reduced** both
nominal and worst-case performance.

What it *did* do, as designed: **lowered variance** (worst-case std
4.1→2.3, nominal 2.5→0.8) and held the std cap (effective std ≤1.0). The
interventions genuinely stabilized training — but stabilized it to a
**lower plateau**, and collapsed the worst-case ceiling from 31% to 24%.

**Mechanism (why it backfired):**
- C reduced exploration (lower entropy + capped std), so the policy
  explored less of the action space.
- B spent the high-LR early phase on *easy* conditions; by the time the
  curriculum reached the hard distribution, LR had decayed and little
  adaptation capacity remained for worst-case.

**Attribution value:** this **rules out "exploration instability" as the
bottleneck.** The std blowup was a *symptom*, not the cause. Suppressing it
did not unlock worst-case performance. The remaining hypothesis — gradient
degradation through the projection on near-infeasible constraints (90%+
detection dropout) — is what **Intervention D (auxiliary feasibility loss)**
targets directly, and is the natural next experiment.

## Next step
Proceed to **D**: penalize ‖u_raw − u_safe‖² so the network emits
already-feasible means → full-rank projection Jacobian → healthier
gradients at worst-case. Keep C/B OFF (they hurt) for the D run.
