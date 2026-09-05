"""
Figure Generation for the IBVS Safe-RL Report
=============================================
Produces every figure consumed by ``references/report.tex``.

Two classes of figure:

1. **Episode figures (A, B, C, L)** — generated from REAL rollouts of the
   locked stage policies through the exact noise + delay + DKF perception stack
   used at evaluation time. No per-step arrays are stored in the repository, so
   these are regenerated on demand by loading the committed checkpoints
   (``logs/stages/<stage>/models/*.zip``) and rolling out episodes here.

2. **Summary / ablation charts (D, E, F, G, H, I, J, K)** — bar/line charts of
   the committed evaluation matrices. Every numeric value is taken verbatim from
   the authoritative sources in the repository: ``report/generate_figures.py``
   (the already-published figure values) and ``docs/legacy/studies/*.md``.
   Where the task brief gave approximate or transposed numbers, the repository
   values are used.

Image-plane quantities are plotted in the simulator's NATIVE normalized image
coordinates ``p_bar = (x_c/z_c, y_c/z_c)`` (not pixels): the simulation has no
physical sensor resolution, so reporting normalized coordinates is the faithful
choice. This is one deliberate difference from the reference paper (which plots
pixels) and is stated in the relevant captions.

Run:  python scripts/legacy/reporting/generate_reference_report_figures.py
Out:  figures/{paper_comparison,ablations,evader}/*.{pdf,png}
"""

import os
import sys
import copy
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

FIGDIR = os.path.join(ROOT, "figures")
DIRS = {
    "paper": os.path.join(FIGDIR, "paper_comparison"),
    "abl": os.path.join(FIGDIR, "ablations"),
    "evader": os.path.join(FIGDIR, "evader"),
}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

# ------------------------------------------------------------------ #
# Style
# ------------------------------------------------------------------ #
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
    }
)

COLORS = {
    "interceptor": "#1f77b4",
    "target": "#ff7f0e",
    "dkf": "#ff7f0e",
    "origin": "#1f77b4",
    "roll": "#1f77b4",
    "pitch": "#ff7f0e",
    "yaw": "#2ca02c",
    "6dof": "#2ca02c",
    "kinematic": "#ff7f0e",
    "dr_trained": "#9467bd",
    "locked": "#7f7f7f",
    "cep_prev": "#d62728",
    "cep_prop": "#2ca02c",
    "gray": "#7f7f7f",
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "cyan": "#17becf",
}

# Paper benchmarks (Yang et al. 2024).
CEP_PREV = 0.457
CEP_PROP = 0.332

# Locked policies that match the current 16-D observation space.
POLICIES = {
    "3b": "logs/stages/stage3b_noisy_mild/models/ibvs_ppo_final.zip",
    "4a": "logs/stages/stage4a_hocbf_locked/models/ibvs_ppo_best.zip",
    "4b": "logs/stages/stage4b_dr_finetune/models/ibvs_ppo_final.zip",
}


def savefig(fig, subdir, name):
    base = os.path.join(DIRS[subdir], name)
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] {os.path.relpath(base, ROOT)}.{{pdf,png}}")


# ================================================================== #
# Rollout harness (real episodes)
# ================================================================== #
def _load_cfg():
    with open(os.path.join(ROOT, "configs", "legacy", "stage3_stage4.yaml")) as f:
        return yaml.safe_load(f)


def build_env(cfg, target_modes=None, target_vmax=None):
    """Interceptor + NoiseDelay + DKF — the canonical evaluation stack."""
    from envs.interception_env import InterceptionEnv
    from envs.wrappers.noise_delay_wrapper import NoiseDelayWrapper
    from envs.wrappers.dkf_wrapper import DKFWrapper

    cfg = copy.deepcopy(cfg)
    if target_modes is not None:
        cfg["target"]["maneuver_modes"] = target_modes
    if target_vmax is not None:
        cfg["target"]["v_max"] = target_vmax
    env = InterceptionEnv(config=cfg)
    nd, dkf = cfg["noise_delay"], cfg["dkf"]
    env = NoiseDelayWrapper(env, delay=nd["delay"], sigma_noise=nd["sigma_noise"])
    env = DKFWrapper(
        env,
        delay=nd["delay"],
        dt=cfg["interceptor"]["dt"],
        sigma_pos_process=dkf["sigma_pos_process"],
        sigma_vel_process=dkf["sigma_vel_process"],
        sigma_measurement=dkf["sigma_measurement"],
        use_imu=True,
    )
    return env, cfg


def rollout(model, env, cfg, seed, max_steps=800):
    """Run one deterministic episode, returning real per-step telemetry.

    Captured signals (all in native normalized image coords for the image plane):
      truth   — clean target projection p_bar from the base env
      meas    — noisy + delayed measurement fed to the DKF (recorded z)
      est     — DKF filtered current-time estimate (x_hat[0:2])
      vel_est — DKF filtered image-plane velocity (x_hat[2:4])
    Plus 3-D positions, distance, FOV flag, and attitude (roll/pitch/yaw).
    """
    base = env.unwrapped
    dkf_filter = env.dkf  # DKFWrapper is outermost; .dkf is the filter

    # Record the noisy/delayed measurement the DKF actually consumes.
    z_log = []
    orig_step = dkf_filter.step

    def _recording_step(*args, **kwargs):
        z = kwargs.get("z", args[0] if args else None)
        z_log.append(np.array(z, dtype=float) if z is not None else np.array([np.nan, np.nan]))
        return orig_step(*args, **kwargs)

    dkf_filter.step = _recording_step

    def _est():
        xh = getattr(dkf_filter, "x_hat", None)
        if xh is None:
            return np.array([np.nan] * 4)
        return np.asarray(xh, dtype=float).ravel()[:4]

    obs, info = env.reset(seed=seed)
    d = {k: [] for k in [
        "t", "int_pos", "tgt_pos", "truth", "meas", "est", "vel_est",
        "dist", "in_fov", "roll", "pitch", "yaw",
    ]}
    dt = cfg["interceptor"]["dt"]

    def snap(step_idx, z_meas):
        xh = _est()
        d["t"].append(step_idx * dt)
        d["int_pos"].append(info["interceptor_pos"].copy())
        d["tgt_pos"].append(info["target_pos"].copy())
        d["truth"].append(np.asarray(info["p_bar"], dtype=float).copy())
        d["meas"].append(z_meas)
        d["est"].append(xh[:2])
        d["vel_est"].append(xh[2:4])
        d["dist"].append(float(info["relative_distance"]))
        d["in_fov"].append(bool(info["in_fov"]))
        d["roll"].append(np.deg2rad(info["roll_deg"]))
        d["pitch"].append(np.deg2rad(info["pitch_deg"]))
        d["yaw"].append(float(getattr(base.interceptor, "yaw", np.nan)))

    snap(0, np.array([np.nan, np.nan]))
    step = 0
    while step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        step += 1
        z_meas = z_log[-1] if z_log else np.array([np.nan, np.nan])
        snap(step, z_meas)
        if term or trunc:
            break

    dkf_filter.step = orig_step  # restore
    for k in d:
        d[k] = np.array(d[k])
    d["outcome"] = info.get("episode_outcome", "unknown")
    d["final_dist"] = d["dist"][-1]
    # terminal relative position components (EFCS): interceptor - target
    rp = d["int_pos"][-1] - d["tgt_pos"][-1]
    d["e_x"], d["e_y"], d["e_z"] = float(rp[0]), float(rp[1]), float(rp[2])
    return d


def run_episodes(stage, n, seed0, target_modes=None, target_vmax=None):
    """Roll out n episodes; return list of telemetry dicts (or [] if unavailable)."""
    from stable_baselines3 import PPO

    path = os.path.join(ROOT, POLICIES[stage])
    if not os.path.exists(path):
        print(f"  [skip] policy missing: {POLICIES[stage]}")
        return []
    model = PPO.load(path)
    cfg = _load_cfg()
    eps = []
    for i in range(n):
        try:
            env, cfg_used = build_env(cfg, target_modes, target_vmax)
            eps.append(rollout(model, env, cfg_used, seed=seed0 + i))
            env.close()
        except Exception as e:  # keep going; one bad seed must not kill the run
            print(f"  [warn] {stage} ep seed {seed0+i} failed: {str(e)[:80]}")
    return eps


def pick_median_success(eps):
    succ = [e for e in eps if e["outcome"] == "success"]
    pool = succ if succ else eps
    if not pool:
        return None
    pool = sorted(pool, key=lambda e: e["final_dist"])
    return pool[len(pool) // 2]


# ================================================================== #
# Figure A — Statistical comparison (mirrors paper Fig. 10)
# ================================================================== #
def fig_A(stage, eps):
    if not eps:
        print(f"  [skip] fig_A_{stage}: no episodes")
        return
    med = pick_median_success(eps)
    succ_rate = 100.0 * np.mean([e["outcome"] == "success" for e in eps])

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.30)

    # (a) 3-D trajectory bundle
    axa = fig.add_subplot(gs[0, 0], projection="3d")
    for e in eps:
        ip, tp = e["int_pos"], e["tgt_pos"]
        axa.plot(ip[:, 0], ip[:, 1], -ip[:, 2], color=COLORS["interceptor"], lw=0.7, alpha=0.5)
        axa.plot(tp[:, 0], tp[:, 1], -tp[:, 2], color=COLORS["target"], lw=0.7, alpha=0.5)
    axa.plot([], [], color=COLORS["interceptor"], label="Interceptor")
    axa.plot([], [], color=COLORS["target"], label="Target")
    axa.set_xlabel("$x_e$ (m)"); axa.set_ylabel("$y_e$ (m)"); axa.set_zlabel("$z_e$ (m)")
    axa.set_title(f"(a) Trajectory bundle ($N={len(eps)}$)")
    axa.legend(loc="upper left", fontsize=8)

    # (b-left) boxplot of final distance with CEP reference lines
    axb = fig.add_subplot(gs[0, 1])
    finals = np.array([e["final_dist"] for e in eps])
    axb.boxplot(finals, widths=0.5, patch_artist=True,
                boxprops=dict(facecolor=COLORS["interceptor"], alpha=0.6),
                medianprops=dict(color="black"))
    axb.axhline(CEP_PREV, ls="--", color=COLORS["cep_prev"], lw=1.5, label=f"Paper: prev. IBVS ({CEP_PREV})")
    axb.axhline(CEP_PROP, ls="--", color=COLORS["cep_prop"], lw=1.5, label=f"Paper: IBVS+DKF ({CEP_PROP})")
    axb.set_ylabel(r"Final $\|p_r\|$ (m)")
    axb.set_xticks([1]); axb.set_xticklabels([f"Stage {stage}"])
    axb.set_title("(b) Final distance vs paper CEP")
    axb.legend(fontsize=7, loc="upper right")

    # (b-right) terminal error scatter with unit circle
    axc = fig.add_subplot(gs[0, 2])
    ex = np.array([e["e_x"] for e in eps]); ez = np.array([e["e_z"] for e in eps])
    axc.scatter(ex, ez, s=18, color=COLORS["interceptor"], alpha=0.7, edgecolor="k", lw=0.3)
    th = np.linspace(0, 2 * np.pi, 200)
    axc.plot(np.cos(th), np.sin(th), color="black", lw=1.2)
    axc.set_aspect("equal"); axc.set_xlabel("$e_x$ (m)"); axc.set_ylabel("$e_z$ (m)")
    axc.set_title("(c) Terminal error (unit circle = 1 m)")

    # (c) median single-episode 3-D trajectory
    axd = fig.add_subplot(gs[1, 0], projection="3d")
    ip, tp = med["int_pos"], med["tgt_pos"]
    axd.plot(ip[:, 0], ip[:, 1], -ip[:, 2], color=COLORS["interceptor"], lw=2, label="Interceptor")
    axd.plot(tp[:, 0], tp[:, 1], -tp[:, 2], color=COLORS["target"], lw=2, ls="--", label="Target")
    axd.set_xlabel("$x_e$ (m)"); axd.set_ylabel("$y_e$ (m)"); axd.set_zlabel("$z_e$ (m)")
    axd.set_title("(d) Median episode trajectory"); axd.legend(fontsize=8)

    # (d) image coordinates truth vs DKF
    axe = fig.add_subplot(gs[1, 1])
    t = med["t"]
    axe.plot(t, med["truth"][:, 0], color=COLORS["origin"], lw=1.4, label=r"truth $\bar p_x$")
    axe.plot(t, med["est"][:, 0], color=COLORS["dkf"], lw=1.4, ls="--", label=r"DKF $\hat p_x$")
    axe.plot(t, med["truth"][:, 1], color=COLORS["origin"], lw=1.0, alpha=0.5)
    axe.plot(t, med["est"][:, 1], color=COLORS["dkf"], lw=1.0, ls="--", alpha=0.5)
    axe.axvline(t[-1], color="gray", ls=":", lw=1)
    axe.set_xlabel("$t$ (s)"); axe.set_ylabel(r"image coord (norm.)")
    axe.set_title("(e) Image coords: truth vs DKF"); axe.legend(fontsize=8)

    # (e) attitude
    axf = fig.add_subplot(gs[1, 2])
    axf.plot(t, med["roll"], color=COLORS["roll"], lw=1.3, label="Roll")
    axf.plot(t, med["pitch"], color=COLORS["pitch"], lw=1.3, label="Pitch")
    axf.plot(t, med["yaw"], color=COLORS["yaw"], lw=1.3, label="Yaw")
    axf.set_xlabel("$t$ (s)"); axf.set_ylabel("angle (rad)")
    axf.set_title("(f) Interceptor attitude"); axf.legend(fontsize=8)

    fig.suptitle(
        f"Figure A — Statistical results, Stage {stage} "
        f"(success {succ_rate:.0f}%, $N={len(eps)}$). Mirror of paper Fig. 10.",
        fontsize=12, y=1.00,
    )
    savefig(fig, "paper", f"fig_A_statistical_{stage}")


# ================================================================== #
# Figures B / C — single-episode panels (mirror paper Figs. 11 / 12)
# ================================================================== #
def _episode_panels(ep, title, fname):
    if ep is None:
        print(f"  [skip] {fname}: no episode")
        return
    t = ep["t"]
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.33, wspace=0.32)

    # (b) 3-D trajectory
    axb = fig.add_subplot(gs[0, 0], projection="3d")
    ip, tp = ep["int_pos"], ep["tgt_pos"]
    axb.plot(ip[:, 0], ip[:, 1], -ip[:, 2], color=COLORS["interceptor"], lw=2, label="Interceptor")
    axb.plot(tp[:, 0], tp[:, 1], -tp[:, 2], color=COLORS["target"], lw=2, ls="--", label="Target")
    axb.set_xlabel("$x_e$ (m)"); axb.set_ylabel("$y_e$ (m)"); axb.set_zlabel("$z_e$ (m)")
    axb.set_title("(b) Trajectory"); axb.legend(fontsize=8)

    # (c) image coords u(t), v(t): truth vs DKF (+ noisy measurement)
    axc1 = fig.add_subplot(gs[0, 1])
    axc1.plot(t, ep["truth"][:, 0], color=COLORS["origin"], lw=1.4, label="truth")
    axc1.plot(t, ep["est"][:, 0], color=COLORS["dkf"], lw=1.4, ls="--", label="DKF")
    m = ep["meas"][:, 0]
    axc1.scatter(t[np.isfinite(m)], m[np.isfinite(m)], s=5, color="gray", alpha=0.35, label="noisy meas.")
    axc1.set_ylabel(r"$\bar p_x$ (norm.)"); axc1.set_title("(c) Image coordinates"); axc1.legend(fontsize=7)
    axc2 = fig.add_subplot(gs[1, 1])
    axc2.plot(t, ep["truth"][:, 1], color=COLORS["origin"], lw=1.4, label="truth")
    axc2.plot(t, ep["est"][:, 1], color=COLORS["dkf"], lw=1.4, ls="--", label="DKF")
    axc2.set_xlabel("$t$ (s)"); axc2.set_ylabel(r"$\bar p_y$ (norm.)")

    # (d) local zoom near interception
    axd = fig.add_subplot(gs[0, 2])
    n = len(t); lo = max(0, n - max(15, n // 5))
    axd.plot(t[lo:], ep["truth"][lo:, 0], color=COLORS["origin"], lw=1.4, label="truth")
    axd.plot(t[lo:], ep["est"][lo:, 0], color=COLORS["dkf"], lw=1.4, ls="--", label="DKF")
    mm = ep["meas"][lo:, 0]
    axd.scatter(t[lo:][np.isfinite(mm)], mm[np.isfinite(mm)], s=8, color="gray", alpha=0.4)
    axd.set_xlabel("$t$ (s)"); axd.set_ylabel(r"$\bar p_x$"); axd.set_title("(d) Local zoom near interception")
    axd.legend(fontsize=7)

    # (e) target path on image plane
    axe = fig.add_subplot(gs[1, 0])
    axe.plot(ep["truth"][:, 0], ep["truth"][:, 1], color=COLORS["interceptor"], lw=1.2)
    axe.scatter([0], [0], marker="+", s=90, color="black")  # image centre
    axe.set_xlabel(r"$\bar p_x$"); axe.set_ylabel(r"$\bar p_y$")
    axe.set_title("(e) Target on image plane"); axe.invert_yaxis()

    # (f) attitude
    axf = fig.add_subplot(gs[1, 2])
    axf.plot(t, ep["roll"], color=COLORS["roll"], lw=1.3, label="Roll")
    axf.plot(t, ep["pitch"], color=COLORS["pitch"], lw=1.3, label="Pitch")
    axf.plot(t, ep["yaw"], color=COLORS["yaw"], lw=1.3, label="Yaw")
    axf.set_xlabel("$t$ (s)"); axf.set_ylabel("angle (rad)")
    axf.set_title("(f) Interceptor attitude"); axf.legend(fontsize=8)

    fig.suptitle(title, fontsize=12, y=1.00)
    savefig(fig, "paper", fname)


# ================================================================== #
# Figure L — DKF tracking validation
# ================================================================== #
def fig_L(ep):
    if ep is None:
        print("  [skip] fig_L: no episode")
        return
    t = ep["t"]
    truth, est, meas = ep["truth"], ep["est"], ep["meas"]
    # velocity truth via finite difference of truth p_bar
    dt = t[1] - t[0] if len(t) > 1 else 0.02
    vtruth = np.gradient(truth, dt, axis=0)

    fin = np.isfinite(meas[:, 0])
    raw_rmse = np.sqrt(np.nanmean(np.sum((meas[fin] - truth[fin]) ** 2, axis=1)))
    dkf_rmse = np.sqrt(np.nanmean(np.sum((est - truth) ** 2, axis=1)))
    impr = 100.0 * (1 - dkf_rmse / raw_rmse) if raw_rmse > 0 else float("nan")
    vel_rmse = np.sqrt(np.nanmean(np.sum((ep["vel_est"] - vtruth) ** 2, axis=1)))

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.subplots_adjust(hspace=0.32, wspace=0.30)

    ax = axes[0, 0]
    ax.plot(t, truth[:, 0], color="black", lw=1.4, label="truth")
    ax.scatter(t[fin], meas[fin, 0], s=6, color="gray", alpha=0.4, label="noisy+delayed")
    ax.plot(t, est[:, 0], color=COLORS["dkf"], lw=1.3, ls="--", label="DKF")
    ax.set_title(r"(a) Image $\bar p_x$"); ax.set_ylabel("norm."); ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.plot(t, truth[:, 1], color="black", lw=1.4, label="truth")
    ax.scatter(t[fin], meas[fin, 1], s=6, color="gray", alpha=0.4)
    ax.plot(t, est[:, 1], color=COLORS["dkf"], lw=1.3, ls="--")
    ax.set_title(r"(b) Image $\bar p_y$")

    ax = axes[0, 2]
    ax.bar(["raw", "DKF"], [raw_rmse, dkf_rmse],
           color=[COLORS["gray"], COLORS["dkf"]], edgecolor="black", lw=0.6)
    ax.set_title(f"(c) Position RMSE (−{impr:.0f}%)"); ax.set_ylabel("RMSE (norm.)")
    for i, v in enumerate([raw_rmse, dkf_rmse]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1, 0]
    ax.plot(t, vtruth[:, 0], color="black", lw=1.4, label="truth")
    ax.plot(t, ep["vel_est"][:, 0], color=COLORS["dkf"], lw=1.3, ls="--", label="DKF")
    ax.set_title(r"(d) Image velocity $\dot{\bar p}_x$"); ax.set_xlabel("$t$ (s)")
    ax.set_ylabel("norm./s"); ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.plot(t, vtruth[:, 1], color="black", lw=1.4)
    ax.plot(t, ep["vel_est"][:, 1], color=COLORS["dkf"], lw=1.3, ls="--")
    ax.set_title(r"(e) Image velocity $\dot{\bar p}_y$"); ax.set_xlabel("$t$ (s)")

    ax = axes[1, 2]
    ax.plot(truth[:, 0], truth[:, 1], color="black", lw=1.4, label="truth path")
    ax.scatter(meas[fin, 0], meas[fin, 1], s=6, color="gray", alpha=0.35, label="noisy")
    ax.plot(est[:, 0], est[:, 1], color=COLORS["dkf"], lw=1.2, ls="--", label="DKF")
    ax.set_title("(f) Image-plane path"); ax.set_xlabel(r"$\bar p_x$"); ax.set_ylabel(r"$\bar p_y$")
    ax.invert_yaxis(); ax.legend(fontsize=7)

    fig.suptitle(
        f"Figure L — DKF tracking validation (pos RMSE −{impr:.0f}%, "
        f"vel RMSE {vel_rmse:.3f}, {len(t)} steps, outcome: {ep['outcome']}).",
        fontsize=12, y=0.99,
    )
    savefig(fig, "paper", "fig_L_dkf_tracking")
    return dict(raw_rmse=raw_rmse, dkf_rmse=dkf_rmse, impr=impr, vel_rmse=vel_rmse)


# ================================================================== #
# Summary / ablation charts (committed numbers)
# ================================================================== #
def _bar(ax, labels, vals, colors, ylim=(0, 105), fs=9):
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", lw=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + ylim[1] * 0.012,
                f"{v:.1f}", ha="center", va="bottom", fontsize=fs, fontweight="bold")
    ax.set_ylim(*ylim)


def fig_D():
    labels = ["1a\nkinematic", "2a\nvision", "2b\n+DKF", "3a-noisy\n+IMU-DKF",
              "3b-noisy\n6-DOF", "4a\n+HOCBF", "4b\nrealistic"]
    vals = [100.0, 70.0, 62.0, 66.0, 96.5, 89.0, 91.0]
    cols = [COLORS["gray"], COLORS["gray"], COLORS["gray"], COLORS["orange"],
            COLORS["green"], COLORS["blue"], COLORS["purple"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _bar(ax, labels, vals, cols)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Success Rate Across the Staged Curriculum")
    savefig(fig, "abl", "fig_D_curriculum_progression")


def fig_E():
    conds = ["clean", "mild noise\n($\\delta$=1,$\\sigma$=.015)", "full noise\n($\\delta$=3,$\\sigma$=.03)"]
    kin = [46.0, 66.0, 0.0]
    dof = [96.0, 96.5, 2.0]
    x = np.arange(len(conds)); w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, kin, w, label="Kinematic (3a)", color=COLORS["kinematic"], edgecolor="black", lw=0.6)
    ax.bar(x + w / 2, dof, w, label="6-DOF (3b)", color=COLORS["6dof"], edgecolor="black", lw=0.6)
    for i, (a, b) in enumerate(zip(kin, dof)):
        ax.text(i - w / 2, a + 1.5, f"{a:.0f}", ha="center", fontsize=9)
        ax.text(i + w / 2, b + 1.5, f"{b:.0f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(conds)
    ax.set_ylabel("Success rate (%)"); ax.set_ylim(0, 105)
    ax.set_title("Dynamics Fidelity vs Perception Noise"); ax.legend()
    savefig(fig, "abl", "fig_E_dynamics_fidelity")


def fig_F():
    labels = ["baseline\n(no CBF)", "bisection", "bisection\n+co-train", "HOCBF"]
    vals = [96.5, 84.0, 85.5, 89.0]
    cols = [COLORS["gray"], COLORS["red"], COLORS["orange"], COLORS["green"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _bar(ax, labels, vals, cols)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Stage 4a — Safety-Filter Method Comparison (success %)")
    savefig(fig, "abl", "fig_F_safety_methods")


def fig_G():
    labels = ["baseline\nHardNet", "C+B\n(entropy+curr.)", "Plan D\n(feas. loss)"]
    nom = [90.2, 85.2, 88.7]
    worst = [24.2, 21.2, 33.9]
    x = np.arange(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, nom, w, label="Nominal", color=COLORS["blue"], edgecolor="black", lw=0.6)
    ax.bar(x + w / 2, worst, w, label="Worst-case", color=COLORS["red"], edgecolor="black", lw=0.6)
    for i, (n, wv) in enumerate(zip(nom, worst)):
        ax.text(i - w / 2, n + 1, f"{n:.1f}", ha="center", fontsize=9)
        ax.text(i + w / 2, wv + 1, f"{wv:.1f}", ha="center", fontsize=9)
    ax.axhline(31.5, ls="--", color=COLORS["purple"], lw=1.4, label="ext. filter worst (31.5)")
    ax.axhline(24.2, ls=":", color=COLORS["gray"], lw=1.4, label="baseline no-D (24.2)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Success rate (%)"); ax.set_ylim(0, 100)
    ax.set_title("HardNet Robustness Study — Staged Interventions"); ax.legend(fontsize=8)
    savefig(fig, "abl", "fig_G_hardnet_robustness")


def fig_H():
    conds = ["nominal", "hard", "worst"]
    locked = [76.5, 51.0, 16.0]
    dr = [91.0, 74.0, 31.5]
    x = np.arange(len(conds)); w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, locked, w, label="3b locked (no DR)", color=COLORS["locked"], edgecolor="black", lw=0.6)
    ax.bar(x + w / 2, dr, w, label="4b DR-trained", color=COLORS["dr_trained"], edgecolor="black", lw=0.6)
    for i, (a, b) in enumerate(zip(locked, dr)):
        ax.text(i - w / 2, a + 1.2, f"{a:.0f}", ha="center", fontsize=9)
        ax.text(i + w / 2, b + 1.2, f"{b:.0f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(conds)
    ax.set_ylabel("Success rate (%)"); ax.set_ylim(0, 100)
    ax.set_title("Stage 4b — Domain Randomization vs Realistic Perception"); ax.legend()
    savefig(fig, "abl", "fig_H_domain_randomization")


def fig_I():
    modes = ["Cruise", "Turn", "Weave", "Break", "Random"]
    claim_a = [75, 0, 5, 12, 12]  # docs/legacy/studies/sixdof_target_claimA.md
    claim_b = [78, 0, 9, 5, 5]    # docs/legacy/studies/sixdof_target_claimB.md
    x = np.arange(len(modes)); w = 0.4
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, vals, title, col in [
        (axA, claim_a, "Claim A: zero-shot", COLORS["blue"]),
        (axB, claim_b, "Claim B: curriculum", COLORS["orange"]),
    ]:
        bars = ax.bar(x, vals, w, color=col, edgecolor="black", lw=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v}", ha="center", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(modes)
        ax.set_ylim(0, 100); ax.set_title(title); ax.set_xlabel("Maneuver mode")
    axA.set_ylabel("Interception success (%)")
    fig.suptitle("Zero-Shot vs Curriculum on Equal-Agility 6-DOF Evader "
                 "(FOV retention 91–98% across all modes)", fontsize=11, y=1.02)
    savefig(fig, "evader", "fig_I_equal_agility_evader")


def fig_J():
    lam = [0.02, 0.05, 0.1, 0.2]
    worst = [32.2, 33.9, 31.8, 36.7]
    worst_sd = [3.0, 4.3, 2.6, 4.4]
    nom = [90.0, 88.7, 89.2, 86.6]
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.errorbar(lam, worst, yerr=worst_sd, fmt="o-", color=COLORS["red"], lw=2, ms=7,
                 capsize=4, label="Worst-case (all-ckpt)")
    ax1.axhline(31.5, ls="--", color=COLORS["purple"], lw=1.3, label="ext. filter (31.5)")
    ax1.axhline(24.2, ls=":", color=COLORS["gray"], lw=1.3, label="baseline no-D (24.2)")
    ax1.set_xlabel(r"Feasibility-loss weight $\lambda$")
    ax1.set_ylabel("Worst-case success (%)", color=COLORS["red"])
    ax1.tick_params(axis="y", labelcolor=COLORS["red"])
    ax1.set_xscale("log"); ax1.set_xticks(lam); ax1.set_xticklabels([str(l) for l in lam])
    ax2 = ax1.twinx()
    ax2.plot(lam, nom, "s--", color=COLORS["blue"], lw=1.8, ms=6, label="Nominal")
    ax2.set_ylabel("Nominal success (%)", color=COLORS["blue"])
    ax2.tick_params(axis="y", labelcolor=COLORS["blue"]); ax2.set_ylim(80, 95); ax2.grid(False)
    ax1.set_title(r"Intervention D — $\lambda$ Ablation (Pareto frontier)")
    l1, lab1 = ax1.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, fontsize=8, loc="center right")
    savefig(fig, "abl", "fig_J_lambda_ablation")


def fig_K():
    lam = [0.02, 0.05, 0.1, 0.2]
    raw_safe = [0.149, 0.086, 0.059, 0.034]
    proj_active = [0.610, 0.501, 0.435, 0.345]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(lam, raw_safe, "o-", color=COLORS["brown"], lw=2, ms=7, label="raw–safe distance")
    ax.plot(lam, proj_active, "s-", color=COLORS["cyan"], lw=2, ms=7, label="projection active frac")
    ax.set_xscale("log"); ax.set_xticks(lam); ax.set_xticklabels([str(l) for l in lam])
    ax.set_xlabel(r"Feasibility-loss weight $\lambda$"); ax.set_ylabel("End-of-training value")
    ax.set_title("HardNet-D Instrumentation — Gradient-Conditioning Dose-Response")
    ax.legend()
    savefig(fig, "abl", "fig_K_hardnet_instrumentation")


# ================================================================== #
# Main
# ================================================================== #
def main():
    print("Summary / ablation charts (committed numbers):")
    fig_D(); fig_E(); fig_F(); fig_G(); fig_H(); fig_I(); fig_J(); fig_K()

    print("\nEpisode figures (real rollouts):")
    # Statistical: stage 3b and stage 4a (HOCBF)
    eps_3b = run_episodes("3b", n=30, seed0=4000)
    fig_A("3b", eps_3b)
    eps_4a = run_episodes("4a", n=30, seed0=4100)
    fig_A("4a", eps_4a)

    # Static target (near-zero velocity) — mirrors paper Fig. 11
    eps_static = run_episodes("3b", n=20, seed0=4200,
                              target_modes=["constant_velocity"], target_vmax=0.0)
    _episode_panels(
        pick_median_success(eps_static),
        "Figure B — Static-target interception (Stage 3b). Mirror of paper Fig. 11. "
        "Key similarity: target held near image centre; DKF tracks through noise. "
        "Key difference: normalized image coords (no physical sensor), simulation only.",
        "fig_B_static_target",
    )

    # Moving / maneuvering target (sinusoidal weave) — mirrors paper Fig. 12
    eps_moving = run_episodes("3b", n=20, seed0=4300, target_modes=["sinusoidal"])
    _episode_panels(
        pick_median_success(eps_moving),
        "Figure C — Maneuvering-target interception (Stage 3b, sinusoidal). Mirror of paper Fig. 12. "
        "Key similarity: pursuit closure under target motion with DKF smoothing. "
        "Key difference: scripted target, normalized image coords, simulation only.",
        "fig_C_moving_target",
    )

    # DKF tracking validation — use a moving-target episode (richer signal)
    dkf_ep = pick_median_success(eps_moving) or pick_median_success(eps_3b)
    stats = fig_L(dkf_ep)
    if stats:
        print(f"\n  DKF tracking: pos RMSE raw={stats['raw_rmse']:.4f} "
              f"DKF={stats['dkf_rmse']:.4f} (−{stats['impr']:.0f}%), "
              f"vel RMSE={stats['vel_rmse']:.4f}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
