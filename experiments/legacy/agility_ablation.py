"""
Asymmetric Agility Ablation on the Equal-Agility 6-DOF Evader
============================================================
HISTORICAL / NON-CANONICAL: this operating-envelope study is outside
fixed_camera_intercept_v1 and does not establish an impossibility theorem.

Pure evaluation experiment (NO retraining, NO reward/obs/target changes). The
only thing that varies across runs is the INTERCEPTOR's physical limits.

Question: the equal-agility evader experiment shows ~0% interception on every
turning maneuver. Is that failure caused by insufficient interceptor agility, or
by a fundamental guidance-law limitation? We answer it by multiplying ONLY the
interceptor's agility limits and re-running the standard evader evaluation.

What is scaled (interceptor only), per the task spec:
  * max body-frame acceleration magnitude  -> env.a_max AND interceptor.a_max
  * max thrust force                        -> interceptor.f_max AND attitude_ctrl.f_max
  * max angular rate                        -> interceptor.omega_max AND attitude_ctrl.omega_max

Why each is touched in two places:
  - The env scales the normalized action by ``env.a_max`` before the interceptor
    re-clips it to ``interceptor.a_max`` (envs/interception_env.py); both must
    move or the smaller of the two caps the result.
  - Thrust and rate limits are cached independently inside the attitude
    controller (models/attitude_controller.py), so the model copy and the
    controller copy must both be scaled.

The TARGET is a separate ``SixDOFTarget`` object owning its own
``Multicopter6DOFLite`` and attitude controller, so scaling ``base.interceptor``
and ``env.a_max`` provably leaves the target's agility unchanged (verified: at
3.0x, target.a_max stays 10.0 while interceptor.a_max becomes 30.0).

Checkpoint: the SAME policy used by the standard evader evaluation (Claim A) ---
HardNet-D locked, lambda=0.05 (36-D observation, in-policy CBF projection), with
the full noise+delay+DKF+CBF-context stack from
``scripts/legacy/eval_evasion.py``.

IMPORTANT confound (flagged, not fixed --- see interpretation.txt): the action
space is normalized to [-1, 1] and scaled by a_max, so raising the limits also
amplifies the *trained* policy's effective control gains and pushes it outside
its training distribution. v_max, max_pitch, and max_roll are NOT scaled, so the
attitude/speed-normalized observation components can saturate at high multipliers.

Run:  python experiments/legacy/agility_ablation.py
Out:  results/agility_ablation/{raw_results.json, interpretation.txt, fig1..fig5}
"""

import os
import sys
import json
import copy
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from stable_baselines3 import PPO
from envs.interception_env import InterceptionEnv
from envs.wrappers.noise_delay_wrapper import NoiseDelayWrapper
from envs.wrappers.dkf_wrapper import DKFWrapper
from envs.wrappers.cbf_context_wrapper import CBFContextWrapper
from safety.hardnet_policy import HardNetActorCriticPolicy  # noqa: register policy

OUTDIR = os.path.join(ROOT, "results", "agility_ablation")
os.makedirs(OUTDIR, exist_ok=True)

MODEL_PATH = os.path.join(ROOT, "logs/stages/stage4a_hardnet_d_locked/models/ibvs_ppo_best.zip")
MULTIPLIERS = [1.0, 1.2, 1.5, 2.0, 3.0]
MODES = ["cruise", "steady_turn", "weave", "break_turn", "random_evasive"]
MODE_SHORT = {"cruise": "Cruise", "steady_turn": "Turn", "weave": "Weave",
              "break_turn": "Break", "random_evasive": "Random"}
N_SEEDS = 100  # seeds 0..99, identical across multipliers for fair comparison

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 11})


# ------------------------------------------------------------------ #
# Config helpers
# ------------------------------------------------------------------ #
def _find_key(cfg, key, default=None):
    if isinstance(cfg, dict):
        if key in cfg:
            return cfg[key]
        for v in cfg.values():
            r = _find_key(v, key, None)
            if r is not None:
                return r
    return default


def apply_interceptor_multiplier(base_env, m):
    """Scale ONLY the interceptor's agility limits in place (target untouched)."""
    base_env.a_max *= m                         # env-level action scaling
    it = base_env.interceptor
    it.a_max *= m                               # interceptor internal clip
    it.f_max *= m                               # max thrust (model)
    it.omega_max *= m                           # max body angular rate (model)
    ac = getattr(it, "attitude_ctrl", None)
    if ac is not None:                          # controller-cached copies
        ac.f_max *= m
        ac.omega_max *= m


def build_env(cfg, level, m, hardnet):
    """eval_evasion stack + interceptor-only agility multiplier (before wrappers)."""
    cfg = copy.deepcopy(cfg)
    cfg["target"]["model"] = "sixdof"
    cfg["target"]["inherit_interceptor_limits"] = True
    cfg["target"]["maneuver_modes"] = [level]
    env = InterceptionEnv(config=cfg)
    apply_interceptor_multiplier(env, m)        # BEFORE wrappers cache params
    nd, dkf = cfg["noise_delay"], cfg["dkf"]
    env = NoiseDelayWrapper(env, delay=nd["delay"], sigma_noise=nd["sigma_noise"])
    env = DKFWrapper(env, delay=nd["delay"], dt=cfg["interceptor"]["dt"],
                     sigma_pos_process=dkf["sigma_pos_process"],
                     sigma_vel_process=dkf["sigma_vel_process"],
                     sigma_measurement=dkf["sigma_measurement"], use_imu=True)
    if hardnet:
        env = CBFContextWrapper(env, alpha_fov=100.0, alpha_attitude=100.0,
                                attitude_safety_margin=0.10)
    return env


# ------------------------------------------------------------------ #
# One episode
# ------------------------------------------------------------------ #
def run_episode(model, env, seed, dt, multiplier, mode):
    obs, info = env.reset(seed=seed)
    base = env.unwrapped
    dists, in_fov_flags, speeds, pitches = [], [], [], []

    def record(info):
        dists.append(float(info["relative_distance"]))
        in_fov_flags.append(bool(info.get("in_fov", False)))
        speeds.append(float(np.linalg.norm(base.interceptor.velocity)))
        pitches.append(abs(float(base.interceptor.pitch)))

    record(info)
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        record(info)
        if term or trunc:
            break

    dists = np.array(dists)
    n_steps = len(dists) - 1  # transitions
    in_fov_steps = int(np.sum(in_fov_flags))
    outcome = base._episode_outcome
    # mean closure rate: d(range)/dt averaged over the episode (negative = closing)
    closure = float(np.mean(np.diff(dists)) / dt) if len(dists) > 1 else 0.0
    return {
        "multiplier": float(multiplier),
        "maneuver_mode": mode,
        "seed": int(seed),
        "success": bool(outcome == "success"),
        "fov_retained": bool(in_fov_steps == len(in_fov_flags)),
        "fov_retention_pct": float(100.0 * in_fov_steps / max(len(in_fov_flags), 1)),
        "final_range": float(dists[-1]),
        "min_range": float(np.min(dists)),
        "episode_steps": int(n_steps),
        "outcome": str(outcome),
        "max_speed": float(np.max(speeds)),
        "max_pitch": float(np.max(pitches)),
        "closure_rate_mean": closure,
    }


# ------------------------------------------------------------------ #
# Experiment
# ------------------------------------------------------------------ #
def run_all():
    with open(os.path.join(ROOT, "configs", "legacy", "stage3_stage4.yaml")) as f:
        cfg = yaml.safe_load(f)
    dt = float(cfg["interceptor"]["dt"])
    model = PPO.load(MODEL_PATH)
    hardnet = model.observation_space.shape[0] == 36
    print(f"Model: {MODEL_PATH}\n  obs={model.observation_space.shape} HardNet={hardnet}")
    print(f"Baseline interceptor a_max={cfg['interceptor']['a_max']} "
          f"v_max={cfg['interceptor']['v_max']} "
          f"(v_max held fixed at all multipliers)\n")

    results = []
    for m in MULTIPLIERS:
        for mode in MODES:
            env = build_env(cfg, mode, m, hardnet)
            base = env.unwrapped
            for s in range(N_SEEDS):
                results.append(run_episode(model, env, s, dt, m, mode))
            env.close()
            succ = 100.0 * np.mean([r["success"] for r in results[-N_SEEDS:]])
            minr = np.mean([r["min_range"] for r in results[-N_SEEDS:]])
            print(f"  mult={m:>3} {mode:<15} success={succ:5.1f}%  "
                  f"mean_min_range={minr:5.1f}m  "
                  f"(int.a_max={base.interceptor.a_max:.0f}, target.a_max={base.target.a_max:.0f})")
    with open(os.path.join(OUTDIR, "raw_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} episodes -> {os.path.join(OUTDIR, 'raw_results.json')}")
    return results, _find_key(cfg, "d_success", 2.0)


# ------------------------------------------------------------------ #
# Aggregation
# ------------------------------------------------------------------ #
def grid(results, field, reducer=np.mean):
    """[mode][mult] aggregate of `field`."""
    out = {mode: [] for mode in MODES}
    for mode in MODES:
        for m in MULTIPLIERS:
            vals = [r[field] for r in results if r["maneuver_mode"] == mode and r["multiplier"] == m]
            out[mode].append(reducer(vals) if vals else float("nan"))
    return out


def success_grid(results):
    return grid(results, "success", lambda v: 100.0 * np.mean(v))


# ------------------------------------------------------------------ #
# Figures
# ------------------------------------------------------------------ #
def fig1_heatmap(results):
    sg = success_grid(results)
    M = np.array([sg[mode] for mode in MODES])  # rows=modes, cols=mults
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(MULTIPLIERS))); ax.set_xticklabels([f"{m}x" for m in MULTIPLIERS])
    ax.set_yticks(range(len(MODES))); ax.set_yticklabels([MODE_SHORT[m] for m in MODES])
    ax.set_xlabel("Interceptor agility multiplier")
    ax.set_ylabel("Maneuver mode")
    ax.set_title("Interception Success Rate (%) vs Interceptor Agility")
    for i in range(len(MODES)):
        for j in range(len(MULTIPLIERS)):
            v = M[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color="black" if 25 < v < 80 else "white", fontweight="bold")
    cb = fig.colorbar(im, ax=ax); cb.set_label("Success rate (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig1_success_heatmap.pdf"))
    fig.savefig(os.path.join(OUTDIR, "fig1_success_heatmap.png"))
    plt.close(fig)


def fig2_lines(results):
    sg = success_grid(results)
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in MODES:
        ax.plot(MULTIPLIERS, sg[mode], "o-", lw=2, ms=6, label=MODE_SHORT[mode])
    ax.axvline(1.0, ls="--", color="gray", lw=1.5)
    ax.text(1.02, 5, "equal agility", rotation=90, va="bottom", color="gray", fontsize=9)
    ax.set_xlabel("Interceptor agility multiplier")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Interception Success vs Interceptor Agility, by Maneuver Mode")
    ax.set_ylim(-3, 103); ax.legend(title="Maneuver mode")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig2_success_by_mode.pdf"))
    fig.savefig(os.path.join(OUTDIR, "fig2_success_by_mode.png"))
    plt.close(fig)


def fig3_range_dist(results, d_success):
    fig, axes = plt.subplots(1, len(MODES), figsize=(20, 4.2), sharey=True)
    for ax, mode in zip(axes, MODES):
        data = [[r["final_range"] for r in results
                 if r["maneuver_mode"] == mode and r["multiplier"] == m] for m in MULTIPLIERS]
        ax.boxplot(data, labels=[f"{m}x" for m in MULTIPLIERS], showfliers=False)
        ax.axhline(d_success, ls="--", color="red", lw=1.4)
        ax.set_title(MODE_SHORT[mode]); ax.set_xlabel("Agility multiplier")
    axes[0].set_ylabel(r"Final range $\|p_r\|$ (m)")
    axes[-1].text(0.98, 0.95, f"success radius = {d_success:g} m", transform=axes[-1].transAxes,
                  ha="right", va="top", color="red", fontsize=9)
    fig.suptitle("Final-Range Distribution vs Interceptor Agility (per maneuver mode)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig3_range_distribution.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUTDIR, "fig3_range_distribution.png"), bbox_inches="tight")
    plt.close(fig)


def fig4_closure(results):
    cg = grid(results, "closure_rate_mean", np.mean)
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in MODES:
        ax.plot(MULTIPLIERS, cg[mode], "o-", lw=2, ms=6, label=MODE_SHORT[mode])
    ax.axhline(0.0, ls=":", color="black", lw=1)
    ax.set_xlabel("Interceptor agility multiplier")
    ax.set_ylabel("Mean closure rate (m/s, negative = closing)")
    ax.set_title("Closure Rate vs Interceptor Agility")
    ax.legend(title="Maneuver mode")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig4_closure_rate.pdf"))
    fig.savefig(os.path.join(OUTDIR, "fig4_closure_rate.png"))
    plt.close(fig)


def fig5_min_range(results):
    mg = grid(results, "min_range", np.mean)
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in MODES:
        ax.plot(MULTIPLIERS, mg[mode], "o-", lw=2, ms=6, label=MODE_SHORT[mode])
    ax.set_xlabel("Interceptor agility multiplier")
    ax.set_ylabel("Mean minimum range achieved (m)")
    ax.set_title("Closest Approach vs Interceptor Agility")
    ax.legend(title="Maneuver mode")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig5_min_range.pdf"))
    fig.savefig(os.path.join(OUTDIR, "fig5_min_range.png"))
    plt.close(fig)


# ------------------------------------------------------------------ #
# Summary table + interpretation
# ------------------------------------------------------------------ #
def print_summary(results):
    sg = success_grid(results)
    lines = []
    lines.append("=" * 60)
    lines.append("AGILITY ABLATION SUMMARY")
    lines.append("=" * 60)
    lines.append("Multiplier | Cruise | Turn  | Weave | Break | Random | Mean")
    lines.append("-----------|--------|-------|-------|-------|--------|------")
    for j, m in enumerate(MULTIPLIERS):
        row = [sg[mode][j] for mode in MODES]
        mean = np.mean(row)
        lines.append(f"{m:<4}x      | {row[0]:5.0f}% | {row[1]:4.0f}% | {row[2]:4.0f}% | "
                     f"{row[3]:4.0f}% | {row[4]:5.0f}% | {mean:4.1f}%")
    lines.append("=" * 60)
    lines.append("INTERPRETATION:")
    lines.append("- If turn success > 30% at 1.5x: failure is AGILITY LIMITED")
    lines.append("- If turn success < 10% at 2.0x: failure is GUIDANCE LIMITED")
    out = "\n".join(lines)
    print("\n" + out)
    return out, sg


def write_interpretation(results, sg, d_success):
    cg = grid(results, "closure_rate_mean", np.mean)
    mg = grid(results, "min_range", np.mean)
    turn = sg["steady_turn"]  # success per multiplier
    # first multiplier achieving >30% steady-turn success
    first_30 = next((MULTIPLIERS[i] for i, v in enumerate(turn) if v > 30), None)
    turn_at_15 = turn[MULTIPLIERS.index(1.5)]
    turn_at_20 = turn[MULTIPLIERS.index(2.0)]
    # monotonic in agility? (mean success across modes)
    mean_succ = [np.mean([sg[mode][j] for mode in MODES]) for j in range(len(MULTIPLIERS))]
    monotonic = all(mean_succ[i] <= mean_succ[i + 1] + 1e-9 for i in range(len(mean_succ) - 1))
    # closure improves with agility on turns even if success doesn't?
    turn_closure = cg["steady_turn"]
    closure_improves = turn_closure[-1] < turn_closure[0] - 1e-6  # more negative = closing more
    turn_minrange = mg["steady_turn"]
    minrange_improves = turn_minrange[-1] < turn_minrange[0] - 1e-6

    if turn_at_15 > 30:
        conclusion = "AGILITY-LIMITED"
        rec = ("Retrain the interceptor with a >=1.5x agility advantage; the "
               "zero-shot result already shows turning interception recovering "
               "above 30% at 1.5x, so a policy trained at that envelope should "
               "close the gap.")
    elif turn_at_20 < 10:
        conclusion = "GUIDANCE-LIMITED"
        rec = ("Proceed to the lead-pursuit / predictive-guidance experiment. "
               "Even 2.0x interceptor agility does not lift turning interception "
               "above 10%, so simply giving the existing guidance more muscle is "
               "not the fix; the guidance law itself must anticipate the turn.")
    else:
        conclusion = "MIXED"
        rec = ("Characterize the boundary: turning interception is partially "
               "agility-sensitive but not fully recovered by 2.0x. Sweep finer "
               "multipliers around the transition and, in parallel, prototype a "
               "lead-pursuit guidance term.")

    txt = []
    txt.append("ASYMMETRIC AGILITY ABLATION — INTERPRETATION")
    txt.append("=" * 60)
    txt.append("")
    txt.append("Setup: HardNet-D (lambda=0.05) locked policy, the SAME checkpoint")
    txt.append("used in the standard evader evaluation (Claim A). 100 seeds (0-99)")
    txt.append("per maneuver mode, identical across multipliers, deterministic")
    txt.append("policy. Only the interceptor's a_max, f_max, and omega_max are")
    txt.append("scaled; the target is untouched (verified: target.a_max stays")
    txt.append("10.0 m/s^2 while interceptor.a_max reaches 30.0 at 3.0x).")
    txt.append("")
    txt.append("Steady-turn success by multiplier (%):")
    txt.append("  " + "  ".join(f"{m}x={v:.0f}" for m, v in zip(MULTIPLIERS, turn)))
    txt.append("")
    txt.append(f"1) First multiplier with >30% steady-turn success: "
               f"{(str(first_30) + 'x') if first_30 else 'NONE (no multiplier reached 30%)'}")
    txt.append("")
    txt.append(f"2) Is mean success monotonic in agility? {'YES' if monotonic else 'NO'}")
    txt.append(f"   Mean success across modes per multiplier: "
               + ", ".join(f"{m}x={v:.1f}%" for m, v in zip(MULTIPLIERS, mean_succ)))
    if not monotonic:
        txt.append("   -> Non-monotonic: extra agility does not help and can hurt,")
        txt.append("      consistent with the trained policy being driven out of")
        txt.append("      distribution when its control gains are amplified.")
    txt.append("")
    txt.append("3) Does closure improve with agility on steady turns even when")
    txt.append("   success does not?")
    txt.append("   mean closure rate (m/s, neg=closing): "
               + ", ".join(f"{m}x={v:.2f}" for m, v in zip(MULTIPLIERS, turn_closure)))
    txt.append("   mean min-range (m): "
               + ", ".join(f"{m}x={v:.1f}" for m, v in zip(MULTIPLIERS, turn_minrange)))
    txt.append(f"   closure improves with agility: {'YES' if closure_improves else 'NO'}; "
               f"closest-approach improves: {'YES' if minrange_improves else 'NO'}")
    txt.append("")
    txt.append(f"4) CONCLUSION: the failure is {conclusion}.")
    txt.append("")
    txt.append(f"5) RECOMMENDED NEXT EXPERIMENT: {rec}")
    txt.append("")
    txt.append("-" * 60)
    txt.append("CONFOUNDS FLAGGED (not fixed, per task spec):")
    txt.append("- Normalized action space: actions are in [-1,1] and scaled by")
    txt.append("  a_max, so raising the limits also amplifies the TRAINED policy's")
    txt.append("  effective control gains. The policy was calibrated at 1.0x, so")
    txt.append("  high multipliers push it out of distribution (e.g. cruise itself")
    txt.append("  can degrade). A clean agility-limit test would require RETRAINING")
    txt.append("  at each multiplier; this zero-shot ablation cannot fully separate")
    txt.append("  'more agility' from 'amplified, miscalibrated commands'.")
    txt.append("- v_max, max_pitch, and max_roll are NOT scaled. Top speed stays")
    txt.append(f"  capped (interceptor already out-cruises the ~12 m/s target), and")
    txt.append("  achievable lateral acceleration from tilting remains bounded by")
    txt.append("  the fixed attitude limits, partially gating the benefit of larger")
    txt.append("  a_max/f_max.")
    txt.append("- Observation space is unchanged: body velocity is normalized by the")
    txt.append("  unscaled v_max and attitude by the unscaled max_pitch/max_roll, so")
    txt.append("  those observation components can saturate at high multipliers,")
    txt.append("  changing their effective meaning to the policy.")
    txt.append(f"- Success radius for reference: d_success = {d_success:g} m.")
    out = "\n".join(txt)
    with open(os.path.join(OUTDIR, "interpretation.txt"), "w") as f:
        f.write(out + "\n")
    print(f"\nWrote {os.path.join(OUTDIR, 'interpretation.txt')}")
    return conclusion


def main():
    results, d_success = run_all()
    fig1_heatmap(results)
    fig2_lines(results)
    fig3_range_dist(results, d_success)
    fig4_closure(results)
    fig5_min_range(results)
    print(f"\nFigures saved to {OUTDIR}")
    _, sg = print_summary(results)
    conclusion = write_interpretation(results, sg, d_success)
    print(f"\nDONE. Conclusion: {conclusion}")


if __name__ == "__main__":
    main()
