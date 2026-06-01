"""
Master Report Figure Generation
================================
Produces every chart used in report/master_report.tex from REAL data:
  - TensorBoard learning curves (per stage, where logs exist)
  - The cross-stage summary charts (noise sensitivity, CBF method comparison,
    intervention arc, λ ablation) hand-keyed from the committed eval matrices
    in docs/*.md and the per-stage 200-ep evaluations.

All numeric values used here are the SAME numbers committed in the docs/
writeups and stage commit messages — nothing is invented. Where a value
comes from a fixed-seed 200-ep eval it is annotated in the docstring of the
relevant function.

Outputs go into report/<stage_x>/ subfolders and report/figures/ for the
cross-cutting charts, mirroring the visualize.py organization.
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tbparse import SummaryReader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, 'report')

plt.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 130, 'font.size': 11,
    'axes.grid': True, 'grid.alpha': 0.3, 'axes.axisbelow': True,
})

C = {'blue': '#1f77b4', 'orange': '#ff7f0e', 'green': '#2ca02c',
     'red': '#d62728', 'purple': '#9467bd', 'gray': '#7f7f7f',
     'brown': '#8c564b', 'cyan': '#17becf'}


# ---------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------- #
def _tb(stage):
    dirs = sorted(glob.glob(os.path.join(ROOT, 'logs/stages', stage, 'tensorboard/PPO_*')))
    if not dirs:
        return None
    return SummaryReader(dirs[0]).scalars


def _smooth(y, k=5):
    if len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode='valid')


def learning_curve(stage, out, title, eval_too=True):
    df = _tb(stage)
    if df is None:
        print(f"  [skip] {stage}: no TB")
        return False
    rew = df[df.tag == 'rollout/ep_rew_mean']
    if len(rew) == 0:
        print(f"  [skip] {stage}: no reward tag")
        return False
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(rew.step / 1e6, rew.value, color=C['blue'], alpha=0.35, lw=1)
    sm = _smooth(rew.value.values)
    ax1.plot(rew.step.values[len(rew) - len(sm):] / 1e6, sm, color=C['blue'], lw=2,
             label='ep_rew_mean (smoothed)')
    ax1.set_xlabel('Timesteps (M)'); ax1.set_ylabel('Episode reward', color=C['blue'])
    ax1.tick_params(axis='y', labelcolor=C['blue'])
    det = df[df.tag == 'eval/det_success_rate']
    if eval_too and len(det) > 0:
        ax2 = ax1.twinx()
        ax2.plot(det.step / 1e6, det.value * 100, 'o-', color=C['red'], lw=2,
                 ms=6, label='det success %')
        ax2.set_ylabel('Deterministic success (%)', color=C['red'])
        ax2.tick_params(axis='y', labelcolor=C['red'])
        ax2.set_ylim(0, 100)
        ax2.grid(False)
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(out); plt.close(fig)
    print(f"  [ok] {out}")
    return True


def bar_compare(labels, values, out, title, ylabel, colors=None, refs=None, ylim=None):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = colors or [C['blue']] * len(labels)
    bars = ax.bar(labels, values, color=colors, edgecolor='black', lw=0.6)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + (ylim[1] if ylim else max(values)) * 0.012,
                f'{v:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    if refs:
        for name, yv, col in refs:
            ax.axhline(yv, ls='--', color=col, lw=1.5, label=f'{name} ({yv:.1f})')
        ax.legend(fontsize=9)
    ax.set_ylabel(ylabel); ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    plt.xticks(rotation=15, ha='right')
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print(f"  [ok] {out}")


# ---------------------------------------------------------------- #
# Per-stage learning curves
# ---------------------------------------------------------------- #
def stage_curves():
    print("Learning curves:")
    learning_curve('stage1a', f'{REPORT}/stage_1a/learning_curve.png',
                   'Stage 1a — Pipeline Validation (kinematic, full obs)', eval_too=False)
    learning_curve('stage1b', f'{REPORT}/stage_1b/learning_curve.png',
                   'Stage 1b — Acceleration control + action smoothing', eval_too=False)
    learning_curve('stage2a', f'{REPORT}/stage_2a/learning_curve.png',
                   'Stage 2a — Vision-only observation', eval_too=False)
    learning_curve('stage2b', f'{REPORT}/stage_2b/learning_curve.png',
                   'Stage 2b — Noise + delay + DKF observer', eval_too=False)
    learning_curve('stage3a_v2', f'{REPORT}/stage_3a/learning_curve_clean.png',
                   'Stage 3a-v2 — First-order dynamics (clean obs)')
    learning_curve('stage3a_noisy_mild', f'{REPORT}/stage_3a/learning_curve_noisy.png',
                   'Stage 3a-noisy-mild — + IMU-DKF')
    learning_curve('stage3b_clean', f'{REPORT}/stage_3b/learning_curve_clean.png',
                   'Stage 3b-clean — 6-DOF + SO(3) PD attitude')
    learning_curve('stage3b_noisy_mild', f'{REPORT}/stage_3b/learning_curve_noisy.png',
                   'Stage 3b-noisy-mild — 6-DOF + IMU-DKF')
    learning_curve('stage4a_hocbf_finetune', f'{REPORT}/stage_4a/learning_curve_hocbf.png',
                   'Stage 4a.3 — HOCBF co-training')
    learning_curve('stage4a_hardnet_d_seed42', f'{REPORT}/stage_4a/learning_curve_hardnet_d.png',
                   'Stage 4a HardNet-D — feasibility-loss fine-tune (seed 42)')
    learning_curve('stage4b_dr_finetune', f'{REPORT}/stage_4b/learning_curve.png',
                   'Stage 4b — Domain-randomized realistic perception')


# ---------------------------------------------------------------- #
# Cross-stage summary charts (values from committed evals/docs)
# ---------------------------------------------------------------- #
def chart_progression():
    # Headline success rates along the staged curriculum (committed numbers).
    labels = ['1a\nkinematic', '2a\nvision', '2b\n+DKF', '3a-noisy\n+IMU-DKF',
              '3b-noisy\n6-DOF', '4a\n+HOCBF', '4b\nrealistic']
    vals = [100.0, 70.0, 62.0, 66.0, 96.5, 89.0, 91.0]
    cols = [C['gray'], C['gray'], C['gray'], C['orange'], C['green'], C['blue'], C['purple']]
    bar_compare(labels, vals, f'{REPORT}/figures/progression.png',
                'Success Rate Across the Staged Curriculum',
                'Success rate (%)', colors=cols, ylim=(0, 105))


def chart_noise_sensitivity():
    # Kinematic vs 6-DOF under mild/full noise (committed).
    fig, ax = plt.subplots(figsize=(7, 4.2))
    conds = ['clean', 'mild noise\n(δ=1,σ=.015)', 'full noise\n(δ=3,σ=.03)']
    kin = [46.0, 66.0, 0.0]
    dof = [96.0, 96.5, 2.0]
    x = np.arange(len(conds)); w = 0.36
    ax.bar(x - w/2, kin, w, label='Kinematic (3a)', color=C['orange'], edgecolor='black', lw=.6)
    ax.bar(x + w/2, dof, w, label='6-DOF (3b)', color=C['green'], edgecolor='black', lw=.6)
    for i, (a, b) in enumerate(zip(kin, dof)):
        ax.text(i - w/2, a + 1.5, f'{a:.0f}', ha='center', fontsize=9)
        ax.text(i + w/2, b + 1.5, f'{b:.0f}', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(conds)
    ax.set_ylabel('Success rate (%)'); ax.set_ylim(0, 105)
    ax.set_title('Dynamics Fidelity vs Perception Noise')
    ax.legend()
    fig.tight_layout(); fig.savefig(f'{REPORT}/figures/noise_sensitivity.png'); plt.close(fig)
    print(f"  [ok] noise_sensitivity")


def chart_cbf_methods():
    # 4a safety filter comparison at the nominal operating point (committed).
    labels = ['baseline\n(no CBF)', '4a.1\nbisection', '4a.2\nbisection\nco-train',
              '4a.3\nHOCBF']
    succ = [96.5, 84.0, 85.5, 89.0]
    cols = [C['gray'], C['red'], C['orange'], C['green']]
    bar_compare(labels, succ, f'{REPORT}/figures/cbf_methods.png',
                'Stage 4a — Safety-Filter Method Comparison (success %)',
                'Success rate (%)', colors=cols, ylim=(0, 105))


def chart_intervention_arc():
    # The HardNet robustness study (committed means).
    labels = ['baseline\nHardNet', 'C+B\n(entropy+\ncurriculum)', 'D\n(feasibility\nloss)']
    worst = [24.2, 21.2, 33.9]
    nom = [90.2, 85.2, 88.7]
    x = np.arange(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - w/2, nom, w, label='Nominal', color=C['blue'], edgecolor='black', lw=.6)
    ax.bar(x + w/2, worst, w, label='Worst-case', color=C['red'], edgecolor='black', lw=.6)
    for i, (n, wv) in enumerate(zip(nom, worst)):
        ax.text(i - w/2, n + 1, f'{n:.1f}', ha='center', fontsize=9)
        ax.text(i + w/2, wv + 1, f'{wv:.1f}', ha='center', fontsize=9)
    ax.axhline(31.5, ls='--', color=C['purple'], lw=1.5, label='ext. filter worst (31.5)')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Success rate (%)'); ax.set_ylim(0, 100)
    ax.set_title('HardNet Robustness Study — Staged Interventions')
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(f'{REPORT}/figures/intervention_arc.png'); plt.close(fig)
    print(f"  [ok] intervention_arc")


def chart_lambda_ablation():
    # λ-sweep (committed): worst-case all-ckpt mean ± std, and nominal.
    lam = [0.02, 0.05, 0.1, 0.2]
    worst = [32.2, 33.9, 31.8, 36.7]; worst_sd = [3.0, 4.3, 2.6, 4.4]
    nom = [90.0, 88.7, 89.2, 86.6]
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.errorbar(lam, worst, yerr=worst_sd, fmt='o-', color=C['red'], lw=2, ms=7,
                 capsize=4, label='Worst-case (all-ckpt)')
    ax1.axhline(31.5, ls='--', color=C['purple'], lw=1.3, label='ext. filter (31.5)')
    ax1.axhline(24.2, ls=':', color=C['gray'], lw=1.3, label='baseline no-D (24.2)')
    ax1.set_xlabel('Feasibility-loss weight λ'); ax1.set_ylabel('Worst-case success (%)', color=C['red'])
    ax1.tick_params(axis='y', labelcolor=C['red'])
    ax1.set_xscale('log'); ax1.set_xticks(lam); ax1.set_xticklabels([str(l) for l in lam])
    ax2 = ax1.twinx()
    ax2.plot(lam, nom, 's--', color=C['blue'], lw=1.8, ms=6, label='Nominal')
    ax2.set_ylabel('Nominal success (%)', color=C['blue'])
    ax2.tick_params(axis='y', labelcolor=C['blue']); ax2.set_ylim(80, 95); ax2.grid(False)
    ax1.set_title('Intervention D — λ Ablation (Pareto frontier)')
    l1, lab1 = ax1.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, fontsize=8, loc='center right')
    fig.tight_layout(); fig.savefig(f'{REPORT}/figures/lambda_ablation.png'); plt.close(fig)
    print(f"  [ok] lambda_ablation")


def chart_hardnet_instrumentation():
    # D instrumentation dose-response (committed end-of-training 3-seed means).
    lam = [0.02, 0.05, 0.1, 0.2]
    raw_safe = [0.149, 0.086, 0.059, 0.034]
    proj_active = [0.610, 0.501, 0.435, 0.345]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(lam, raw_safe, 'o-', color=C['brown'], lw=2, ms=7, label='raw–safe distance')
    ax.plot(lam, proj_active, 's-', color=C['cyan'], lw=2, ms=7, label='projection active frac')
    ax.set_xscale('log'); ax.set_xticks(lam); ax.set_xticklabels([str(l) for l in lam])
    ax.set_xlabel('Feasibility-loss weight λ'); ax.set_ylabel('end-of-training value')
    ax.set_title('HardNet-D Instrumentation — Gradient-Conditioning Dose-Response')
    ax.legend()
    fig.tight_layout(); fig.savefig(f'{REPORT}/figures/hardnet_instrumentation.png'); plt.close(fig)
    print(f"  [ok] hardnet_instrumentation")


def chart_4b_degradation():
    # 4b policy vs locked 3b under the 4b stack, 3 conditions (committed).
    conds = ['nominal\n~50% drop', 'hard\n~75% drop', 'worst\n~92% drop']
    p4b = [91.0, 74.0, 31.5]; p3b = [76.5, 51.0, 16.0]
    x = np.arange(len(conds)); w = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - w/2, p3b, w, label='3b locked (no DR)', color=C['gray'], edgecolor='black', lw=.6)
    ax.bar(x + w/2, p4b, w, label='4b DR-trained', color=C['purple'], edgecolor='black', lw=.6)
    for i, (a, b) in enumerate(zip(p3b, p4b)):
        ax.text(i - w/2, a + 1.2, f'{a:.0f}', ha='center', fontsize=9)
        ax.text(i + w/2, b + 1.2, f'{b:.0f}', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(conds)
    ax.set_ylabel('Success rate (%)'); ax.set_ylim(0, 100)
    ax.set_title('Stage 4b — Domain Randomization vs Realistic Perception')
    ax.legend()
    fig.tight_layout(); fig.savefig(f'{REPORT}/figures/stage4b_degradation.png'); plt.close(fig)
    print(f"  [ok] stage4b_degradation")


# ---------------------------------------------------------------- #
# Copy existing episode/dashboard/GIF artifacts into the report tree
# ---------------------------------------------------------------- #
def copy_artifacts():
    import shutil
    print("Copying existing artifacts:")
    pairs = [
        ('logs/stages/stage1a/eval/dashboard.png', 'stage_1a/dashboard.png'),
        ('logs/stages/stage1a/eval/episode_14_analysis.png', 'stage_1a/episode_analysis.png'),
        ('logs/stages/stage2b/eval/dashboard.png', 'stage_2b/dashboard.png'),
        ('logs/stages/stage2b/dkf_test/dkf_ep1.png', 'stage_2b/dkf_tracking.png'),
        ('logs/stages/stage2b/videos/replay_best.gif', 'stage_2b/replay_best.gif'),
        ('logs/stages/stage3a/eval/dashboard.png', 'stage_3a/dashboard.png'),
        ('logs/stages/stage3a/eval/episode_1_analysis.png', 'stage_3a/episode_analysis.png'),
        ('logs/stages/stage3a/videos/replay_best.gif', 'stage_3a/replay_best.gif'),
        ('logs/stages/stage3b_noisy_mild/eval/episode_1_analysis.png', 'stage_3b/episode_analysis.png'),
        ('logs/stages/stage4b_baseline_check/eval/episode_1_analysis.png', 'stage_4b/episode_analysis.png'),
        ('logs/stages/stage4a_phase3_hocbf_a50/eval/episode_1_analysis.png', 'stage_4a/episode_analysis.png'),
    ]
    for src, dst in pairs:
        s = os.path.join(ROOT, src); d = os.path.join(REPORT, dst)
        if os.path.exists(s):
            shutil.copy(s, d); print(f"  [ok] {dst}")
        else:
            print(f"  [miss] {src}")


if __name__ == '__main__':
    stage_curves()
    print("Cross-stage charts:")
    chart_progression()
    chart_noise_sensitivity()
    chart_cbf_methods()
    chart_intervention_arc()
    chart_lambda_ablation()
    chart_hardnet_instrumentation()
    chart_4b_degradation()
    copy_artifacts()
    print("DONE.")
