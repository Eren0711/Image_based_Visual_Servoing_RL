#!/usr/bin/env python3
"""Render one faithful 6-DOF equal-agility evader replay for the HTML deck."""

import os
import sys
import yaml
import numpy as np
from stable_baselines3 import PPO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "legacy"))

import visualize  # noqa: E402
from eval_evasion import build_env  # noqa: E402
from envs.wrappers.cbf_context_wrapper import CBFContextWrapper  # noqa: E402
from safety.hardnet_policy import HardNetActorCriticPolicy  # noqa: F401,E402

OUT = os.path.join(ROOT, "presentation_html", "media",
                   "sixdof_equal_agility_steady_turn.mp4")
MODEL = os.path.join(ROOT, "logs", "stages", "stage4a_hardnet_d_locked",
                     "models", "ibvs_ppo_best.zip")


def make_env(cfg, level):
    env = build_env(cfg, level)
    env = CBFContextWrapper(env, alpha_fov=100.0, alpha_attitude=100.0,
                            attitude_safety_margin=0.10)
    return env


def main():
    with open(os.path.join(ROOT, "configs", "legacy", "stage3_stage4.yaml")) as f:
        cfg = yaml.safe_load(f)

    model = PPO.load(MODEL)
    if model.observation_space.shape[0] != 36:
        raise RuntimeError("Expected a HardNet 36-D observation model")

    candidates = []
    for seed in range(3000, 3006):
        env = make_env(cfg, "steady_turn")
        data = visualize.collect_episode(model, env, deterministic=True, seed=seed)
        fov_retention = float(np.mean(data["in_fov"]))
        final_distance = float(data["relative_distance"][-1])
        # Prefer a faithful failure case: target held in view, but no intercept.
        score = fov_retention - 0.002 * final_distance
        candidates.append((score, seed, data, fov_retention, final_distance,
                           data["outcome"]))

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, seed, data, fov_retention, final_distance, outcome = candidates[0]
    print(f"selected seed={seed} outcome={outcome} "
          f"fov_retention={100*fov_retention:.1f}% "
          f"final_distance={final_distance:.2f} m")
    visualize.create_animation(data, cfg, fps=20, save_path=OUT, skip=3)
    print(OUT)


if __name__ == "__main__":
    main()
