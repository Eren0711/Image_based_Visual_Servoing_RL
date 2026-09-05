"""Canonical paired-scenario rollout runner and immutable output writer."""

from __future__ import annotations

import json
import math
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import gymnasium as gym
import numpy as np

from evaluation.metrics import write_summary_from_jsonl
from evaluation.schema import EPISODE_SCHEMA_VERSION, EpisodeRecord
from evaluation.suites import EvaluationSuite, load_suite
from experiment_paths import validate_run_id
from runtime.environment import EnvironmentOptions, build_environment
from runtime.manifest import (
    file_sha256,
    finalize_manifest,
    initialize_run_manifest,
    semantic_sha256,
)


EPISODES_FILENAME = "episodes.jsonl"
SUMMARY_FILENAME = "summary.json"


class EvaluationProtocolError(RuntimeError):
    """Raised when a model, environment, or rollout violates the protocol."""


def _space_shape(owner: Any, attribute: str, owner_name: str) -> tuple[int, ...]:
    space = getattr(owner, attribute, None)
    shape = getattr(space, "shape", None)
    if shape is None:
        raise EvaluationProtocolError(
            f"{owner_name}.{attribute} must expose a fixed shape"
        )
    return tuple(int(value) for value in shape)


def validate_model_environment(model: Any, env: gym.Env, config: Mapping[str, Any]) -> None:
    """Fail before rollouts when policy, runtime, and config spaces disagree."""
    try:
        declared_observation = int(
            config["policy_contract"]["observation"]["dimension"]
        )
        declared_action = int(config["policy_contract"]["action"]["dimension"])
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationProtocolError(
            "config must declare policy_contract observation/action dimensions"
        ) from error

    env_observation = _space_shape(env, "observation_space", "environment")
    env_action = _space_shape(env, "action_space", "environment")
    model_observation = _space_shape(model, "observation_space", "model")
    model_action = _space_shape(model, "action_space", "model")
    expected_observation = (declared_observation,)
    expected_action = (declared_action,)
    mismatches = []
    if env_observation != expected_observation:
        mismatches.append(
            f"environment observation {env_observation} != {expected_observation}"
        )
    if env_action != expected_action:
        mismatches.append(f"environment action {env_action} != {expected_action}")
    if model_observation != env_observation:
        mismatches.append(
            f"model observation {model_observation} != environment {env_observation}"
        )
    if model_action != env_action:
        mismatches.append(f"model action {model_action} != environment {env_action}")
    if mismatches:
        raise EvaluationProtocolError("space contract mismatch: " + "; ".join(mismatches))


def _evaluation_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"evaluation-{timestamp}-{uuid.uuid4().hex[:8]}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _metric(info: Mapping[str, Any], key: str) -> float:
    try:
        value = float(info[key])
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationProtocolError(f"episode info is missing numeric {key!r}") from error
    if not math.isfinite(value):
        raise EvaluationProtocolError(f"episode info {key!r} is not finite")
    return value


def _set_scenario(env: gym.Env, target_mode: str) -> None:
    base = env.unwrapped
    setter = getattr(base, "set_target_modes", None)
    if not callable(setter):
        raise EvaluationProtocolError(
            "unwrapped environment must implement set_target_modes"
        )
    setter([target_mode])


def _predict_action(
    model: Any,
    observation: np.ndarray,
    *,
    deterministic: bool,
    env: gym.Env,
) -> np.ndarray:
    prediction = model.predict(observation, deterministic=deterministic)
    action = prediction[0] if isinstance(prediction, tuple) else prediction
    action_array = np.asarray(action)
    expected_shape = tuple(env.action_space.shape)
    if action_array.shape == (1, *expected_shape):
        action_array = action_array[0]
    if action_array.shape != expected_shape:
        raise EvaluationProtocolError(
            f"model returned action shape {action_array.shape}, expected {expected_shape}"
        )
    if not np.all(np.isfinite(action_array)):
        raise EvaluationProtocolError("model returned a non-finite action")
    low = np.asarray(env.action_space.low)
    high = np.asarray(env.action_space.high)
    if np.any(action_array < low - 1e-6) or np.any(action_array > high + 1e-6):
        raise EvaluationProtocolError("model returned an action outside its declared space")
    return action_array


def _run_scenario(
    *,
    model: Any,
    env: gym.Env,
    config: Mapping[str, Any],
    evaluation_id: str,
    episode_index: int,
    scenario: Any,
    deterministic: bool,
    experiment_id: str,
    method_id: str,
    model_id: str,
    condition_id: str,
    config_sha256: str,
    model_sha256: str | None,
    projection_expected: bool,
) -> EpisodeRecord:
    _set_scenario(env, scenario.target_mode)
    observation, reset_info = env.reset(seed=scenario.seed)
    actual_mode = str(reset_info.get("target_mode", ""))
    if actual_mode != scenario.target_mode:
        raise EvaluationProtocolError(
            f"scenario requested {scenario.target_mode!r}, reset selected {actual_mode!r}"
        )

    initial_distance = _metric(reset_info, "relative_distance")
    initial_image_error = _metric(reset_info, "image_error")
    distances = [initial_distance]
    image_errors = [initial_image_error]
    fov_margins = [_metric(reset_info, "fov_margin")]
    in_fov_samples: list[bool] = []
    pitch_samples: list[float] = []
    roll_samples: list[float] = []
    envelope_samples: list[bool] = []
    projection_active: list[bool] = []
    projection_norms: list[float] = []
    projection_feasible: list[bool] = []
    total_reward = 0.0
    consecutive_fov_loss = 0
    max_consecutive_fov_loss = 0
    terminal_info: Mapping[str, Any] | None = None
    final_terminated = False
    final_truncated = False
    max_steps = int(config["env"]["max_steps"])

    for step_index in range(1, max_steps + 1):
        action = _predict_action(
            model,
            observation,
            deterministic=deterministic,
            env=env,
        )
        observation, reward, terminated, truncated, info = env.step(action)
        reward_value = float(reward)
        if not math.isfinite(reward_value):
            raise EvaluationProtocolError("environment returned a non-finite reward")
        total_reward += reward_value
        distances.append(_metric(info, "relative_distance"))
        image_errors.append(_metric(info, "image_error"))
        fov_margins.append(_metric(info, "fov_margin"))
        visible = bool(info.get("in_fov", False))
        in_fov_samples.append(visible)
        if visible:
            consecutive_fov_loss = 0
        else:
            consecutive_fov_loss += 1
            max_consecutive_fov_loss = max(
                max_consecutive_fov_loss, consecutive_fov_loss
            )
        pitch_samples.append(abs(_metric(info, "pitch_deg")))
        roll_samples.append(abs(_metric(info, "roll_deg")))
        envelope_samples.append(bool(info.get("attitude_violation", False)))
        cbf_info = info.get("cbf")
        if isinstance(cbf_info, Mapping):
            projection_active.append(bool(cbf_info.get("corrected", False)))
            projection_norms.append(float(cbf_info.get("correction_norm", 0.0)))
            projection_feasible.append(bool(cbf_info.get("feasible", False)))

        final_terminated = bool(terminated)
        final_truncated = bool(truncated)
        if final_terminated or final_truncated:
            terminal_info = info
            steps = step_index
            break
    else:
        raise EvaluationProtocolError(
            f"environment did not terminate within configured max_steps={max_steps}"
        )

    if terminal_info is None:
        raise EvaluationProtocolError("terminal episode info was not captured")
    if final_terminated and final_truncated:
        raise EvaluationProtocolError("environment set terminated and truncated together")
    outcome = str(terminal_info.get("episode_outcome", ""))
    terminal_mode = str(terminal_info.get("target_mode", actual_mode))
    if terminal_mode != scenario.target_mode:
        raise EvaluationProtocolError(
            f"target mode changed during scenario: {terminal_mode!r}"
        )
    dt = float(config["interceptor"]["dt"])
    duration = steps * dt
    seed_bundle = _json_safe(reset_info.get("seed_bundle", {}))
    if not isinstance(seed_bundle, dict):
        raise EvaluationProtocolError("reset seed_bundle must be a mapping")
    seed_bundle.setdefault("scenario", scenario.seed)

    errors = np.asarray(image_errors, dtype=np.float64)
    minimum_fov_margin = float(min(fov_margins))
    if projection_active:
        projection_status = "partial"
        projection_active_fraction = float(
            sum(projection_active) / len(projection_active)
        )
        projection_intervention_mean = float(np.mean(projection_norms))
        projection_intervention_max = float(max(projection_norms))
        projection_infeasible_steps = int(
            sum(not feasible for feasible in projection_feasible)
        )
    else:
        projection_status = "unavailable" if projection_expected else "not_applicable"
        projection_active_fraction = None
        projection_intervention_mean = None
        projection_intervention_max = None
        projection_infeasible_steps = None
    record = EpisodeRecord(
        schema_version=EPISODE_SCHEMA_VERSION,
        evaluation_id=evaluation_id,
        experiment_id=experiment_id,
        method_id=method_id,
        model_id=model_id,
        condition_id=condition_id,
        config_sha256=config_sha256,
        model_sha256=model_sha256,
        episode_id=f"episode-{episode_index:04d}",
        scenario_id=scenario.scenario_id,
        suite_id=scenario.suite_id,
        seed=int(scenario.seed),
        seed_bundle=seed_bundle,
        target_mode=scenario.target_mode,
        deterministic=bool(deterministic),
        outcome=outcome,
        terminated=final_terminated,
        truncated=final_truncated,
        steps=steps,
        duration_s=float(duration),
        total_reward=float(total_reward),
        initial_distance_m=float(distances[0]),
        final_distance_m=float(distances[-1]),
        min_distance_m=float(min(distances)),
        terminal_closure_rate_mps=float(
            (distances[-2] - distances[-1]) / dt
        ),
        intercept_time_s=float(duration) if outcome == "success" else None,
        fov_retention_fraction=float(sum(in_fov_samples) / steps),
        fov_loss_steps=int(steps - sum(in_fov_samples)),
        fov_loss_duration_s=float((steps - sum(in_fov_samples)) * dt),
        max_consecutive_fov_loss_steps=max_consecutive_fov_loss,
        min_fov_margin=minimum_fov_margin,
        image_error_initial=float(errors[0]),
        image_error_final=float(errors[-1]),
        image_error_mean=float(np.mean(errors)),
        image_error_rms=float(np.sqrt(np.mean(np.square(errors)))),
        image_error_max=float(np.max(errors)),
        max_abs_pitch_deg=float(max(pitch_samples, default=0.0)),
        max_abs_roll_deg=float(max(roll_samples, default=0.0)),
        envelope_violation_steps=int(sum(envelope_samples)),
        envelope_violation_duration_s=float(sum(envelope_samples) * dt),
        projection_metrics_status=projection_status,
        projection_active_fraction=projection_active_fraction,
        projection_intervention_l2_mean=projection_intervention_mean,
        projection_intervention_l2_max=projection_intervention_max,
        projection_residual_max=None,
        projection_infeasible_steps=projection_infeasible_steps,
    )
    record.to_dict()
    return record


def run_evaluation(
    *,
    model: Any,
    config: Mapping[str, Any],
    suite: EvaluationSuite,
    output_dir: str | Path,
    deterministic: bool | None = None,
    environment_options: EnvironmentOptions | None = None,
    environment_factory: Callable[[], gym.Env] | None = None,
    evaluation_id: str | None = None,
    command: Sequence[str] | None = None,
    source_config: str | Path | None = None,
    source_model: str | Path | None = None,
    experiment_id: str = "clean_ppo_interception_mvp_v1",
    method_id: str = "M1_ppo",
    model_id: str = "in_memory_model",
    condition_id: str = "clean",
) -> dict[str, Any]:
    """Run the suite and return its JSONL-derived summary.

    ``output_dir`` must not exist. This prevents two evaluations from silently
    mixing records and makes the manifest an immutable run boundary.
    """
    resolved_options = environment_options or EnvironmentOptions()
    if load_suite(suite.source_path) != suite:
        raise EvaluationProtocolError(
            "suite object differs from its immutable source YAML"
        )
    env = (
        environment_factory()
        if environment_factory is not None
        else build_environment(dict(config), resolved_options)
    )
    manifest_path: Path | None = None

    try:
        validate_model_environment(model, env, config)
        run_id = validate_run_id(evaluation_id or _evaluation_id())
        resolved_deterministic = (
            suite.deterministic if deterministic is None else bool(deterministic)
        )
        suite_provenance = suite.to_manifest_dict()
        suite_provenance["source_sha256"] = file_sha256(suite.source_path)
        destination = Path(output_dir).expanduser().resolve()
        resolved_config_sha256 = semantic_sha256(config)
        resolved_model_path = (
            Path(source_model).expanduser() if source_model else None
        )
        if resolved_model_path and not resolved_model_path.is_file():
            zip_candidate = Path(f"{resolved_model_path}.zip")
            if zip_candidate.is_file():
                resolved_model_path = zip_candidate
        resolved_model_sha256 = (
            file_sha256(resolved_model_path)
            if resolved_model_path and resolved_model_path.is_file()
            else None
        )
        manifest_path = initialize_run_manifest(
            destination,
            run_id=run_id,
            run_kind="evaluation",
            config=config,
            seed=suite.seeds[0],
            command=list(command or [sys.executable, *sys.argv]),
            runtime_options={
                "protocol": "canonical_paired_interception_v1",
                "episode_schema_version": EPISODE_SCHEMA_VERSION,
                "deterministic": resolved_deterministic,
                "environment": resolved_options.to_dict(),
                "active_components": resolved_options.active_components(),
                "suite": suite_provenance,
                "identity": {
                    "experiment_id": experiment_id,
                    "method_id": method_id,
                    "model_id": model_id,
                    "condition_id": condition_id,
                },
            },
            source_config=source_config,
            source_model=source_model,
        )
        suite_snapshot = destination / "resolved_suite.yaml"
        suite_snapshot.write_bytes(suite.source_path.read_bytes())
        episodes_path = destination / EPISODES_FILENAME
        with episodes_path.open("x", encoding="utf-8") as stream:
            for episode_index, scenario in enumerate(suite.scenarios(), start=1):
                record = _run_scenario(
                    model=model,
                    env=env,
                    config=config,
                    evaluation_id=run_id,
                    episode_index=episode_index,
                    scenario=scenario,
                    deterministic=resolved_deterministic,
                    experiment_id=experiment_id,
                    method_id=method_id,
                    model_id=model_id,
                    condition_id=condition_id,
                    config_sha256=resolved_config_sha256,
                    model_sha256=resolved_model_sha256,
                    projection_expected=(
                        resolved_options.use_cbf
                        or resolved_options.use_cbf_context
                    ),
                )
                stream.write(
                    json.dumps(record.to_dict(), sort_keys=True, allow_nan=False)
                    + "\n"
                )
                stream.flush()

        summary_path = destination / SUMMARY_FILENAME
        summary = write_summary_from_jsonl(episodes_path, summary_path)
        finalize_manifest(
            manifest_path,
            status="complete",
            extra={
                "evaluation": {
                    "suite_snapshot": "resolved_suite.yaml",
                    "episodes_path": EPISODES_FILENAME,
                    "episodes_sha256": file_sha256(episodes_path),
                    "summary_path": SUMMARY_FILENAME,
                    "summary_sha256": file_sha256(summary_path),
                    "episode_count": suite.episode_count,
                }
            },
        )
        return summary
    except KeyboardInterrupt:
        if manifest_path is not None:
            finalize_manifest(manifest_path, status="interrupted")
        raise
    except Exception as error:
        if manifest_path is not None:
            finalize_manifest(
                manifest_path,
                status="failed",
                extra={
                    "evaluation": {
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                },
            )
        raise
    finally:
        env.close()
