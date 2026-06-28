"""
Lead-Pursuit Guidance Evaluation on the Equal-Agility 6-DOF Evader
=================================================================
Follow-on to the agility ablation, which proved the equal-agility turn failure
is GUIDANCE-limited (steady-turn success stays 0% from 1.0x to 3.0x interceptor
agility). Here we test a guidance fix that requires NO retraining: a lead-pursuit
observation transform that servos on the PREDICTED future image error instead of
the current one (see envs/lead_pursuit_wrapper.py for the unit-correct math).

Design mirrors experiments/agility_ablation.py exactly so the two are directly
comparable:
  * SAME checkpoint   : HardNet-D locked, lambda=0.05 (36-D, in-policy CBF)
  * SAME stack        : noise + delay + DKF + CBF context (+ LeadPursuitWrapper)
  * SAME seeds        : 0..99 per maneuver mode, identical across lead times
  * deterministic policy, evaluation only

The only change across runs is the lead horizon T (seconds). T=0.0 is the pure
pursuit baseline and reproduces the agility-ablation 1.0x row.

The image-plane velocity used for the lead is the DKF *filtered* estimate
(obs[2:4] is overwritten by the DKF wrapper), not the raw noisy measurement.

Run:  python experiments/lead_pursuit_eval.py
Out:  results/lead_pursuit/{raw_results.json, interpretation.txt, fig1..fig6}
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stable_baselines3 import PPO
from envs.interception_env import InterceptionEnv
from envs.wrappers.noise_delay_wrapper import NoiseDelayWrapper
from envs.wrappers.dkf_wrapper import DKFWrapper
from envs.wrappers.cbf_context_wrapper import CBFContextWrapper
from envs.lead_pursuit_wrapper import LeadPursuitWrapper
from safety.hardnet_policy import HardNetActorCriticPolicy  # noqa: register policy

OUTDIR = os.path.join(ROOT, "results", "lead_pursuit")
AGILITY_JSON = os.path.join(ROOT, "results", "agility_ablation", "raw_results.json")
os.makedirs(OUTDIR, exist_ok=True)

MODEL_PATH = os.path.join(ROOT, "logs/stages/stage4a_hardnet_d_locked/models/ibvs_ppo_best.zip")
LEAD_TIMES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
MODES = ["cruise", "steady_turn", "weave", "break_turn", "random_evasive"]
MODE_SHORT = {"cruise": "Cruise", "steady_turn": "Turn", "weave": "Weave",
              "break_turn": "Break", "random_evasive": "Random"}
MODE_COLORS = {"cruise": "#1f77b4", "steady_turn": "#d62728", "weave": "#2ca02c",
               "break_turn": "#ff7f0e", "random_evasive": "#9467bd"}
N_SEEDS = 100

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 11})


def _find_key(cfg, key, default=None):
    if isinstance(cfg, dict):
        if key in cfg:
            return cfg[key]
        for v in cfg.values():
            r = _find_key(v, key, None)
            if r is not None:
                return r
    return default


def build_env(cfg, level, lead_time, hardnet):
    cfg = copy.deepcopy(cfg)
    cfg["target"]["model"] = "sixdof"
    cfg["target"]["maneuver_modes"] = [level]
    env = InterceptionEnv(config=cfg)
    nd, dkf = cfg["noise_delay"], cfg["dkf"]
    env = NoiseDelayWrapper(env, delay=nd["delay"], sigma_noise=nd["sigma_noise"])
    env = DKFWrapper(env, delay=nd["delay"], dt=cfg["interceptor"]["dt"],
                     sigma_pos_process=dkf["sigma_pos_process"],
                     sigma_vel_process=dkf["sigma_vel_process"],
                     sigma_measurement=dkf["sigma_measurement"], use_imu=True)
    if hardnet:
        env = CBFContextWrapper(env, alpha_fov=100.0, alpha_attitude=100.0,
                                attitude_safety_margin=0.10)
    env = LeadPursuitWrapper(env, lead_time=lead_time)  # outermost: pure obs transform
    return env


def run_episode(model, env, seed, dt, lead_time, mode):
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
    in_fov_steps = int(np.sum(in_fov_flags))
    outcome = base._episode_outcome
    closure = float(np.mean(np.diff(dists)) / dt) if len(dists) > 1 else 0.0
    return {
        "lead_time": float(lead_time),
        "maneuver_mode": mode,
        "seed": int(seed),
        "success": bool(outcome == "success"),
        "fov_retained": bool(in_fov_steps == len(in_fov_flags)),
        "fov_retention_pct": float(100.0 * in_fov_steps / max(len(in_fov_flags), 1)),
        "final_range": float(dists[-1]),
        "min_range": float(np.min(dists)),
        "episode_steps": int(len(dists) - 1),
        "outcome": str(outcome),
        "max_speed": float(np.max(speeds)),
        "max_pitch": float(np.max(pitches)),
        "closure_rate_mean": closure,
    }


def run_all():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    dt = float(cfg["interceptor"]["dt"])
    model = PPO.load(MODEL_PATH)
    hardnet = model.observation_space.shape[0] == 36
    print(f"Model: {MODEL_PATH}\n  obs={model.observation_space.shape} HardNet={hardnet}")
    print("Velocity source for lead: DKF filtered estimate (obs[2:4]).\n")

    results = []
    for T in LEAD_TIMES:
        for mode in MODES:
            env = build_env(cfg, mode, T, hardnet)
            for s in range(N_SEEDS):
                results.append(run_episode(model, env, s, dt, T, mode))
            env.close()
            succ = 100.0 * np.mean([r["success"] for r in results[-N_SEEDS:]])
            minr = np.mean([r["min_range"] for r in results[-N_SEEDS:]])
            print(f"  T={T:<4} {mode:<15} success={succ:5.1f}%  mean_min_range={minr:5.1f}m")
    with open(os.path.join(OUTDIR, "raw_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} episodes -> {os.path.join(OUTDIR, 'raw_results.json')}")
    return results, _find_key(cfg, "d_success", 2.0)


# ------------------------------------------------------------------ #
# Aggregation
# ------------------------------------------------------------------ #
def grid(results, field, xs, xkey, reducer=np.mean):
    out = {mode: [] for mode in MODES}
    for mode in MODES:
        for x in xs:
            vals = [r[field] for r in results if r["maneuver_mode"] == mode and r[xkey] == x]
            out[mode].append(reducer(vals) if vals else float("nan"))
    return out


def success_grid(results, xs, xkey):
    return grid(results, "success", xs, xkey, lambda v: 100.0 * np.mean(v))


# ------------------------------------------------------------------ #
# Figures
# ------------------------------------------------------------------ #
def fig1_heatmap(results):
    sg = success_grid(results, LEAD_TIMES, "lead_time")
    M = np.array([sg[m] for m in MODES])
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(LEAD_TIMES))); ax.set_xticklabels([f"{t:g}s" for t in LEAD_TIMES])
    ax.set_yticks(range(len(MODES))); ax.set_yticklabels([MODE_SHORT[m] for m in MODES])
    ax.set_xlabel("Lead time T (s)"); ax.set_ylabel("Maneuver mode")
    ax.set_title("Interception Success Rate (%) vs Lead Time")
    for i in range(len(MODES)):
        for j in range(len(LEAD_TIMES)):
            v = M[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color="black" if 25 < v < 80 else "white", fontweight="bold")
    fig.colorbar(im, ax=ax, label="Success rate (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig1_success_heatmap.pdf"))
    fig.savefig(os.path.join(OUTDIR, "fig1_success_heatmap.png"))
    plt.close(fig)


def fig2_by_mode(results):
    sg = success_grid(results, LEAD_TIMES, "lead_time")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for mode in MODES:
        ys = sg[mode]
        ax.plot(LEAD_TIMES, ys, "o-", lw=2, ms=5, color=MODE_COLORS[mode], label=MODE_SHORT[mode])
        j = int(np.argmax(ys))
        ax.plot(LEAD_TIMES[j], ys[j], "*", ms=15, color=MODE_COLORS[mode],
                markeredgecolor="black", markeredgewidth=0.5, zorder=5)
    ax.axvline(0.0, ls="--", color="gray", lw=1.5)
    ax.text(0.02, 3, "pure pursuit baseline", rotation=90, va="bottom", color="gray", fontsize=9)
    ax.set_xlabel("Lead time T (s)"); ax.set_ylabel("Success rate (%)")
    ax.set_title("Interception Success vs Lead Time (stars = per-mode optimum)")
    ax.set_ylim(-3, 103); ax.legend(title="Maneuver mode")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig2_success_by_mode.pdf"))
    fig.savefig(os.path.join(OUTDIR, "fig2_success_by_mode.png"))
    plt.close(fig)


def fig3_turn_focus(results):
    sg = success_grid(results, LEAD_TIMES, "lead_time")
    ys = sg["steady_turn"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(LEAD_TIMES, ys, "o-", lw=2.5, ms=10, color=MODE_COLORS["steady_turn"])
    j = int(np.argmax(ys))
    ax.annotate(f"peak {ys[j]:.0f}% at T={LEAD_TIMES[j]:g}s",
                xy=(LEAD_TIMES[j], ys[j]), xytext=(0.4, max(ys) + 6 if max(ys) < 90 else 90),
                arrowprops=dict(arrowstyle="->", color="black"), fontsize=10)
    ax.axhline(30, ls=":", color="green", lw=1.2, label="30% (fix threshold)")
    ax.axhline(10, ls=":", color="orange", lw=1.2, label="10% (partial threshold)")
    ax.set_xlabel("Lead time T (s)"); ax.set_ylabel("Steady-turn success rate (%)")
    ax.set_title("Does Lead Pursuit Fix the Steady-Turn Failure?")
    ax.set_ylim(-3, max(35, max(ys) + 10)); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig3_turn_success_focus.pdf"))
    fig.savefig(os.path.join(OUTDIR, "fig3_turn_success_focus.png"))
    plt.close(fig)


def fig4_fov(results):
    fg = grid(results, "fov_retention_pct", LEAD_TIMES, "lead_time", np.mean)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for mode in MODES:
        ax.plot(LEAD_TIMES, fg[mode], "o-", lw=2, ms=5, color=MODE_COLORS[mode], label=MODE_SHORT[mode])
    ax.set_xlabel("Lead time T (s)"); ax.set_ylabel("FOV retention (%)")
    ax.set_title("FOV Retention vs Lead Time (does leading push the target off-frame?)")
    ax.set_ylim(0, 103); ax.legend(title="Maneuver mode")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig4_fov_retention.pdf"))
    fig.savefig(os.path.join(OUTDIR, "fig4_fov_retention.png"))
    plt.close(fig)


def fig5_closure(results):
    cg = grid(results, "closure_rate_mean", LEAD_TIMES, "lead_time", np.mean)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for mode in MODES:
        ax.plot(LEAD_TIMES, cg[mode], "o-", lw=2, ms=5, color=MODE_COLORS[mode], label=MODE_SHORT[mode])
    ax.axhline(0.0, ls=":", color="black", lw=1)
    ax.set_xlabel("Lead time T (s)"); ax.set_ylabel("Mean closure rate (m/s, negative = closing)")
    ax.set_title("Closure Rate vs Lead Time")
    ax.legend(title="Maneuver mode")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig5_closure_rate.pdf"))
    fig.savefig(os.path.join(OUTDIR, "fig5_closure_rate.png"))
    plt.close(fig)


def fig6_comparison(results):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    # Left: agility ablation success vs multiplier (if available)
    if os.path.exists(AGILITY_JSON):
        ag = json.load(open(AGILITY_JSON))
        mults = sorted(set(r["multiplier"] for r in ag))
        agsg = success_grid(ag, mults, "multiplier")
        for mode in MODES:
            axL.plot(mults, agsg[mode], "o-", lw=2, ms=5, color=MODE_COLORS[mode], label=MODE_SHORT[mode])
        axL.axvline(1.0, ls="--", color="gray", lw=1.2)
        axL.set_xlabel("Interceptor agility multiplier")
    else:
        axL.text(0.5, 0.5, "agility_ablation/raw_results.json not found", ha="center", transform=axL.transAxes)
    axL.set_ylabel("Success rate (%)"); axL.set_title("Agility ablation (guidance-limited)")
    axL.set_ylim(-3, 103); axL.legend(title="Maneuver mode", fontsize=8)

    # Right: lead pursuit success vs lead time
    sg = success_grid(results, LEAD_TIMES, "lead_time")
    for mode in MODES:
        axR.plot(LEAD_TIMES, sg[mode], "o-", lw=2, ms=5, color=MODE_COLORS[mode], label=MODE_SHORT[mode])
    axR.axvline(0.0, ls="--", color="gray", lw=1.2)
    axR.set_xlabel("Lead time T (s)"); axR.set_title("Lead pursuit (guidance fix attempt)")
    axR.set_ylim(-3, 103)
    fig.suptitle("Agility (guidance-limited) vs Lead Pursuit (guidance fix attempt)", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig6_comparison.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUTDIR, "fig6_comparison.png"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ #
# Summary + interpretation
# ------------------------------------------------------------------ #
def print_summary(results):
    sg = success_grid(results, LEAD_TIMES, "lead_time")
    turn = sg["steady_turn"]
    lines = ["=" * 60, "LEAD PURSUIT EVALUATION SUMMARY", "=" * 60,
             "Lead T | Cruise | Turn  | Weave | Break | Random | Mean",
             "-------|--------|-------|-------|-------|--------|------"]
    for j, T in enumerate(LEAD_TIMES):
        row = [sg[m][j] for m in MODES]
        tag = "  <- baseline" if T == 0.0 else ""
        lines.append(f"{T:>4.2f}s | {row[0]:5.0f}% | {row[1]:4.0f}% | {row[2]:4.0f}% | "
                     f"{row[3]:4.0f}% | {row[4]:5.0f}% | {np.mean(row):4.1f}%{tag}")
    lines.append("=" * 60)
    best_turn = max(turn)
    bj = int(np.argmax(turn))
    if best_turn > 0:
        ans = f"YES at T={LEAD_TIMES[bj]:g}s with {best_turn:.0f}%"
    else:
        ans = "NO at all tested values"
    if best_turn > 30:
        concl = "LEAD PURSUIT FIXES IT"
    elif best_turn >= 10:
        concl = "PARTIAL FIX"
    else:
        concl = "INSUFFICIENT"
    lines.append("KEY QUESTION: Does steady-turn success exceed 0% at any T?")
    lines.append(f"ANSWER: {ans}")
    lines.append(f"CONCLUSION: {concl}")
    lines.append("=" * 60)
    out = "\n".join(lines)
    print("\n" + out)
    return sg, concl


def write_interpretation(results, sg, concl, d_success):
    turn = sg["steady_turn"]
    mean_succ = [np.mean([sg[m][j] for m in MODES]) for j in range(len(LEAD_TIMES))]
    fg = grid(results, "fov_retention_pct", LEAD_TIMES, "lead_time", np.mean)
    cg = grid(results, "closure_rate_mean", LEAD_TIMES, "lead_time", np.mean)

    bj_turn = int(np.argmax(turn)); best_turn_T = LEAD_TIMES[bj_turn]; best_turn_v = turn[bj_turn]
    bj_mean = int(np.argmax(mean_succ)); best_mean_T = LEAD_TIMES[bj_mean]; best_mean_v = mean_succ[bj_mean]
    cruise0 = sg["cruise"][0]
    # a lead time that improves turn without hurting cruise (>= baseline-3pp)?
    no_harm = [LEAD_TIMES[j] for j in range(len(LEAD_TIMES))
               if turn[j] > turn[0] + 1e-9 and sg["cruise"][j] >= cruise0 - 3]
    # FOV degradation with longer lead?
    turn_fov = fg["steady_turn"]
    fov_degrades = turn_fov[-1] < turn_fov[0] - 3

    if best_turn_v > 30:
        nxt = ("Retrain the policy WITH the lead-pursuit observation: the zero-shot "
               "transform already lifts steady-turn success above 30%, so a policy "
               "trained to exploit the predicted error should consolidate the gain.")
    elif best_turn_v >= 10:
        nxt = ("Combine lead pursuit with a short retrain. The zero-shot transform "
               "partially recovers turning interception but the policy was trained on "
               "pure-pursuit observations and cannot fully use the shifted signal.")
    else:
        nxt = ("Proceed to a temporal-memory experiment (LSTM / stacked observation "
               "history). A single-step lead from the DKF velocity is insufficient; "
               "the policy likely needs to infer the target's turn from a history of "
               "observations rather than a one-step linear extrapolation.")

    L = []
    L.append("LEAD-PURSUIT GUIDANCE — INTERPRETATION")
    L.append("=" * 60)
    L.append("")
    L.append("Setup: HardNet-D (lambda=0.05) locked policy, the SAME checkpoint as the")
    L.append("agility ablation and the standard evader eval. 100 seeds (0-99) per mode,")
    L.append("identical across lead times, deterministic. Only the observation is")
    L.append("transformed (obs[0:2] -> lead-corrected); reward/dynamics/target unchanged.")
    L.append("Velocity used for the lead is the DKF FILTERED estimate (obs[2:4]),")
    L.append("converted to consistent units (position scaled by tan(half_fov),")
    L.append("velocity by MAX_DP=10.0; factors kx=14.28, ky=19.21 per second).")
    L.append("")
    L.append("Steady-turn success by lead time (%):")
    L.append("  " + "  ".join(f"{t:g}s={v:.0f}" for t, v in zip(LEAD_TIMES, turn)))
    L.append("")
    L.append(f"1) Best lead time for STEADY-TURN success: T={best_turn_T:g}s -> {best_turn_v:.0f}%")
    L.append(f"2) Best lead time for OVERALL MEAN success: T={best_mean_T:g}s -> {best_mean_v:.1f}%")
    L.append(f"   (baseline T=0 mean = {mean_succ[0]:.1f}%)")
    L.append(f"3) Recovers steady-turn above 30%? {'YES' if best_turn_v > 30 else 'NO'}; "
             f"above 10%? {'YES' if best_turn_v >= 10 else 'NO'}")
    L.append(f"4) Does FOV retention degrade with longer lead (steady-turn)? "
             f"{'YES' if fov_degrades else 'NO'}")
    L.append("   steady-turn FOV retention by T (%): "
             + ", ".join(f"{t:g}s={v:.0f}" for t, v in zip(LEAD_TIMES, turn_fov)))
    L.append(f"5) A lead time that improves turn WITHOUT hurting cruise: "
             + (", ".join(f"{t:g}s" for t in no_harm) if no_harm else "NONE"))
    L.append("")
    L.append(f"OVERALL CONCLUSION: {concl}.")
    L.append("")
    L.append("Supporting detail — does leading improve closure even when success does not?")
    L.append("  steady-turn mean closure rate (m/s, neg=closing): "
             + ", ".join(f"{t:g}s={v:.2f}" for t, v in zip(LEAD_TIMES, cg['steady_turn'])))
    L.append("")
    L.append(f"RECOMMENDED NEXT STEP: {nxt}")
    L.append("")
    L.append("-" * 60)
    L.append("CAVEATS / NOTES:")
    L.append("- Zero-shot transform: the policy was TRAINED on pure-pursuit observations,")
    L.append("  so feeding it a lead-shifted error is itself out-of-distribution. A drop")
    L.append("  in cruise success at larger T is expected for this reason and does not by")
    L.append("  itself prove lead pursuit is a bad idea --- only that this policy cannot")
    L.append("  exploit it without retraining.")
    L.append("- The lead uses the DKF filtered image velocity (obs[2:4]), not the raw")
    L.append("  noisy measurement; the DKF velocity is smoother but lags fast turns.")
    L.append("- Unit handling: position and velocity obs channels are normalized")
    L.append("  differently (tan(half_fov) vs MAX_DP=10.0); the wrapper converts to a")
    L.append("  common raw-p_bar unit before adding, so the lead is dimensionally correct.")
    L.append(f"- Success radius for reference: d_success = {d_success:g} m.")
    out = "\n".join(L)
    with open(os.path.join(OUTDIR, "interpretation.txt"), "w") as f:
        f.write(out + "\n")
    print(f"\nWrote {os.path.join(OUTDIR, 'interpretation.txt')}")


def main():
    results, d_success = run_all()
    fig1_heatmap(results)
    fig2_by_mode(results)
    fig3_turn_focus(results)
    fig4_fov(results)
    fig5_closure(results)
    fig6_comparison(results)
    print(f"\nFigures saved to {OUTDIR}")
    sg, concl = print_summary(results)
    write_interpretation(results, sg, concl, d_success)
    print(f"\nDONE. Conclusion: {concl}")


if __name__ == "__main__":
    main()
