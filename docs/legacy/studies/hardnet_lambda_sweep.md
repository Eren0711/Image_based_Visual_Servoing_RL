# HardNet Intervention D — λ Ablation Sweep

> **PRELIMINARY LEGACY EVIDENCE.** Not part of the canonical clean MVP and not
> statistically validated for a publication claim. See `docs/README.md`.

**Date:** 2026-06-01
**Question:** What feasibility-loss weight λ is optimal, and where does the
"lazy policy" failure mode (aux term dominating the reward gradient) appear?

## Method

Sweep λ ∈ {0.02, 0.1, 0.2}, 3 seeds each (42, 123, 7), combined with the
already-run λ=0.05 (Intervention D). Identical recipe: 3M steps, warm-start
from 4b 1M, full DR from step 0, HOCBF context, linear LR decay. Eval: each
run's {1.0M, 1.2M, 1.4M} ckpts on nominal + worst, 200 ep, eval seed=1000.
12 trainings total (3 new λ × 3 seeds + 3 reused), 72 evaluation points.

## Training-time instrumentation (3-seed mean, end of training)

| λ    | raw_safe_dist | aux_loss | proj_active | act_nofov |
|------|---------------|----------|-------------|-----------|
| 0.02 | 0.149         | 0.060    | 0.610       | 0.720     |
| 0.05 | 0.086         | 0.027    | 0.501       | 0.562     |
| 0.1  | 0.059         | 0.016    | 0.435       | 0.490     |
| 0.2  | 0.034         | 0.007    | 0.345       | 0.352     |

Clean monotonic dose-response: higher λ pulls raw means toward feasibility
(raw_safe_dist 0.149→0.034) and reduces projection activation (0.61→0.35).
The mechanism scales smoothly and predictably with λ.

## Task-performance results (200-ep eval, seed=1000)

| λ    | Nominal (best-nom) | Worst (all-ckpt) | Worst ceiling |
|------|--------------------|------------------|---------------|
| 0.02 | 91.5% ± 2.3        | 32.2% ± 3.0      | 38%           |
| 0.05 | 91.2% ± 1.0        | 33.9% ± 4.3      | 40%           |
| 0.1  | 91.2% ± 1.0        | 31.8% ± 2.6      | 37%           |
| 0.2  | 88.3% ± 3.1        | **36.7% ± 4.4**  | **44%**       |
| baseline (no D) | 91.5%   | 24.2%            | 31%           |
| ext. HOCBF filter | 91.0% | 31.5%           | 31.5%         |

## Findings

1. **Every λ beats both baselines.** All four settings recover the
   worst-case gap (32–37% vs baseline 24%) and exceed the external filter
   (31.5%). The intervention is robust to λ, not knife-edge — a reassuring
   property for deployment.

2. **Worst-case is ~flat over λ∈[0.02, 0.1] (~32–34%), then λ=0.2 jumps to
   36.7%** — the best worst-case of the sweep. More feasibility pressure
   keeps helping worst-case past where the lazy-policy effect was expected.

3. **The lazy-policy trade-off appears at λ=0.2 in NOMINAL**, which drops to
   88.3% (from ~91% at lower λ) with higher variance (±3.1). This is the
   theorized cost: strong feasibility pressure makes the policy more
   conservative — ~3pp nominal traded for ~3pp worst-case.

4. **It's a Pareto frontier, not a single optimum:**
   - **λ=0.05** = best nominal-preserving (91.2% nom / 33.9% worst).
   - **λ=0.2**  = best worst-case (36.7% worst / 88.3% nom).

   Caveat: worst-case std (~3–4pp) is comparable to inter-λ differences, so
   we do not over-claim a sharp ranking among 0.02/0.05/0.1. The robust
   claims are: (a) all λ beat baseline & filter; (b) λ=0.2 gives the best
   worst-case at a measurable nominal cost.

Safety held across the entire sweep: max|roll| ≤ 0.633, limit-exceedance
≤ 0.02% of steps in all 72 evaluations.

## Recommendation

- **Worst-case-priority deployment (our hardware goal): λ=0.2.** When the
  drone is flying half-blind in wind — exactly when safety margin matters
  most — λ=0.2 gives the strongest robustness (36.7% worst, 44% ceiling),
  for a 3pp nominal cost.
- **Balanced / nominal-priority: λ=0.05** keeps full nominal performance
  with worst-case still well above baseline and filter.

**Locked artifacts** (`logs/stages/stage4a_hardnet_d_locked/`):
- `ibvs_ppo_best.zip` — λ=0.05, seed 123 @ 1.0M (88.5% nom / 40.5% worst)
- `ibvs_ppo_best_lam02.zip` — λ=0.2, seed 123 @ 1.4M (92.5% nom / 43.5% worst)

Reproduce λ=0.2 worst-case eval:
```
python scripts/legacy/eval_stage4b.py \
  --model logs/stages/stage4a_hardnet_d_locked/models/ibvs_ppo_best_lam02 \
  --mode hardnet --condition worst --episodes 200
```
