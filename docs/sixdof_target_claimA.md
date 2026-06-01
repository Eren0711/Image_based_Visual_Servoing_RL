# 6-DOF Equal-Agility Target & Claim A (Zero-Shot Generalization)

**Date:** 2026-06-01

## Motivation
Through Stage 4b the interceptor flew a realistic 6-DOF model while the target
was a kinematic point mass — an asymmetry that let the target sustain
unphysical instantaneous turns. We built a 6-DOF target with the SAME flight
model (`Multicopter6DOFLite`) and the SAME agility limits as the interceptor
(v_max, a_max, omega_max, attitude), so the adversary is exactly equally
capable. It is driven by an evasion guidance law over a difficulty curriculum.

## Evasion curriculum (simplest -> hardest)
- **L1 cruise**: straight line at cruise speed.
- **L2 steady_turn**: sustained coordinated banked turn.
- **L3 weave**: periodic alternating banked turns (S-curves).
- **L4 break_turn**: reactive — banks away from the pursuer line of sight.
- **L5 random_evasive**: reactive break-turns with randomized flip timing.
Turns are coordinated (lateral accel + matching yaw-rate omega = a_lat/v) so
the bank settles rather than tumbling; all commands respect the shared a_max.

## A confound we caught and fixed
The first 6-DOF target cruised at 0.8*v_max = 12 m/s, but the policy trained
against point-mass targets moving only `uniform(0, 0.5*v_max_pm)` ≈ 0-5 m/s.
That made every episode a tail-chase the guidance never learned (~0-8% even on
a straight target). Fix: the 6-DOF target's cruise speed now defaults to its
SPAWNED initial speed (sampled by the env in the training regime), so Claim A
isolates the *dynamics-model* change from a *speed* change.

## Claim A — zero-shot, fixed reward/training, only the target changes
HardNet-D (locked, λ=0.05), 100 episodes/level, full realistic stack
(noise + delay + DKF + in-policy CBF), 6-DOF target at training speed regime:

| Level          | Success | FOV-loss | FOV retention | Attitude exceed |
|----------------|---------|----------|---------------|-----------------|
| cruise (L1)    | 75.0%   | 24.0%    | 97.2%         | 0.006%          |
| steady_turn    |  0.0%   | 99.0%    | 90.9%         | 0.000%          |
| weave          |  5.0%   | 95.0%    | 92.3%         | 0.000%          |
| break_turn     | 12.0%   | 84.0%    | 93.0%         | 0.004%          |
| random_evasive | 12.0%   | 84.0%    | 93.1%         | 0.004%          |

## Findings (honest)
1. **FOV-tracking generalizes.** FOV retention is 91-97% across ALL levels,
   including the ones where interception fails. The CBF + HardNet + perception
   stack keeps the target in frame ~92% of the time even when guidance cannot
   close — the IBVS-specific objective transfers zero-shot.
2. **Interception of a maneuvering 6-DOF target does NOT generalize.**
   Straight cruise recovers to 75%, but any sustained turn drops interception
   to 0-12%. The guidance, trained only on straight/mild point-mass targets,
   cannot close on an equally-agile banking evader without training.
3. **Safety holds.** Attitude-limit exceedance <= 0.006% of steps everywhere.

## Interpretation
This cleanly separates two competencies the project built: the safety+perception
layer (generalizes) vs the guidance policy (does not transfer to maneuvering
equal-agility targets). It motivates Claim B: train against the curriculum and
measure whether the framework CAN learn to intercept the 6-DOF evader. Claim B
is a separately-labeled experiment with its own (possibly adapted) training
setup; it will not be conflated with the Claim-A zero-shot numbers.
