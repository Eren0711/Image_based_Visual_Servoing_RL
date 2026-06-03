# Research Notebook Website

`index.html` is a static, self-contained research-portfolio page (HTML/CSS/JS,
no build step). Open it directly in a browser. MathJax is loaded from a CDN for
equation rendering; all other media is local and referenced by relative path.

## What it is

A detailed engineering record of the project's evolution, structured to mirror
the **actual repository stages and commit history** (not a reconstructed
narrative). It is written to teach: a professor, researcher, or graduate student
should be able to understand the full system without opening the source.

Sections:
- **Concept primers** — IBVS, PPO/RL, Control Barrier Functions, HardNet, the
  6-DOF multicopter model, and the camera/observer — each with equations.
- **System architecture** — the closed loop, with observation/action/safety
  definitions taken from the code.
- **Evolution matrix** — how every subsystem (dynamics, camera, observation,
  action, reward, network/safety, observer, sensor/noise, DR, target) changed
  across stages.
- **Commit timeline** — reconstructed from `git log`.
- **Per-stage deep dives** (expandable) for Stages 1–5 and every sub-step, each
  following: problem → why the previous approach was insufficient → what changed
  → math/model → results → limitation → why the next step.
- **Conclusion** — two contributions and one honest capability boundary.

## Media (all referenced by relative path)

Stage demonstration videos (`../logs/stages/<stage>/videos/replay_best.mp4`):
1a (kinematic), 1b (acceleration), 2a (vision-only), 2b (DKF), 3a (reward
hacking), 3b (96.5% robust), 4a HOCBF, 4a HardNet-D, 4b (wind+dropout).

Final 6-DOF equal-agility evader clip: `media/sixdof_equal_agility_steady_turn.mp4`,
rendered by `render_sixdof_evader_video.py` using the same 6-DOF target stack as
`eval_evasion.py` (NOT the older point-mass `--evasive` target). Seed 3001,
steady-turn level, FOV retention 93.9%, final distance 18.13 m, outcome FOV
loss — the faithful clip of the Stage 5 finding.

Figures from `../report/figures/` and `../report/stage_2b/`: noise sensitivity,
CBF method comparison, intervention arc, λ ablation, HardNet instrumentation,
4b degradation, DKF tracking.

## Regenerate the final 6-DOF video

```bash
python presentation_html/render_sixdof_evader_video.py
```

## Integrity

All numeric claims trace to committed evaluation logs and the `docs/*.md`
writeups. The site explicitly states the two methodological confounds (speed;
stacked difficulty / std-runaway) that were caught and fixed in the Stage 5
study.
