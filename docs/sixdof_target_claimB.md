# Claim B — Curriculum Training vs 6-DOF Equal-Agility Evader

**Date:** 2026-06-02

## Setup
Separate, clearly-labeled experiment from Claim A. We train (not just
evaluate) against the 6-DOF symmetric-agility target via a cumulative
maneuver curriculum: start at `cruise`, append the next level
(`steady_turn` -> `weave` -> `break_turn` -> `random_evasive`) once rolling
training success crosses 60%. Warm-started from HardNet-D (locked), 5M steps,
λ=0.2 feasibility loss, linear LR decay.

**Confound control (v2):** the first attempt (v1) accidentally enabled the
full Stage-4b stack (wind + 90% detection dropout + DR) *on top of* the 6-DOF
target — stacking two hard changes — and the action std ran away (1.77->4.69).
v2 fixes both: (a) CLEAN perception stack identical to the Claim A eval
(noise + delay + DKF + in-policy CBF; NO wind/dropout), so the only change
from zero-shot is that we now *train* on the 6-DOF target; (b) `max_log_std=0`
caps the std (held stable at ~1.64 the whole run).

## Curriculum progression (the headline diagnostic)
```
start  level 0: cruise
60%  -> level 1: steady_turn   (active set: [cruise, steady_turn])
       ... never reached 60% on [cruise, steady_turn] -> STALLED at level 1
```
The curriculum unlocked exactly one level. It mastered straight cruise,
advanced to sustained banked turns, and could not master those — so `weave`,
`break_turn`, `random_evasive` were never unlocked during training.

## Per-level evaluation (100 ep/level, FOV-retention harness)

| Level          | Claim A succ | Claim B succ | Claim A FOVret | Claim B FOVret |
|----------------|--------------|--------------|----------------|----------------|
| cruise         | 75%          | 78%          | 97.2%          | 98.2%          |
| steady_turn    |  0%          |  0%          | 90.9%          | 93.7%          |
| weave          |  5%          |  9%          | 92.3%          | 93.6%          |
| break_turn     | 12%          |  5%          | 93.0%          | 94.8%          |
| random_evasive | 12%          |  5%          | 93.1%          | 94.6%          |
(Claim B = 5M checkpoint; 2M checkpoint is within noise.)
Attitude-limit exceedance: 0.000% across all levels.

## Verdict — NEGATIVE result (and an informative one)
1. **Curriculum training did NOT solve maneuvering interception.** Interception
   on every maneuvering level is within noise of the zero-shot baseline
   (0 / 9 / 5 / 5 %). The curriculum stalled at `steady_turn`. This is *not* a
   training-instability artifact (std stayed at 1.64) nor a confound (clean
   stack) — it is a genuine capability ceiling.
2. **FOV retention stayed excellent (93-98%) and slightly improved.** The
   safety + perception layer is robust whether or not we train on evasion; it
   is not the bottleneck.
3. **`steady_turn` (sustained constant banked turn) is the hardest case — 0%.**
   Harder than the reactive `break_turn`. Mechanically: the pursuer needs a
   continuous matching turn, but its FORWARD-FIXED camera couples look-
   direction to flight-direction, so it cannot simultaneously sustain the
   turn-to-pursue AND keep the nose on a target curving inside its own turn
   radius. With equal agility, the geometry is unwinnable for pure pursuit.

## Interpretation
Phase 1 (Claim A) + Phase 2 (Claim B) together isolate a clean boundary:
the framework **keeps an equally-agile maneuvering target in FOV ~94%** of the
time (perception/safety generalizes), but **cannot intercept a sustained-
maneuvering equal-agility target**, and curriculum training does not fix it —
identifying the limitation as architectural (platform + fixed sensor +
pure-pursuit guidance), not a data/training shortfall.

## Motivated next steps (not hand-waving — directly implied by the mechanism)
- **Gimbaled / stabilized camera:** decouple visual tracking from the flight
  path, removing the look-vs-go coupling that makes sustained turns unwinnable.
- **Lead-pursuit / predictive guidance:** aim at the target's predicted
  intercept point rather than its current bearing (pure pursuit), reducing the
  required terminal turn rate.
- **Asymmetric agility (if mission-appropriate):** the equal-agility constraint
  was deliberately fair; a faster/more-agile interceptor changes the geometry.
These belong to the hardware / next-architecture phase.
