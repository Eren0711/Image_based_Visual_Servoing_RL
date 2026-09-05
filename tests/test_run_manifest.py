"""Phase-1 checks for immutable run directories and provenance records."""

import json
from datetime import datetime, timezone

import yaml

from experiment_paths import (
    PROJECT_ROOT,
    create_run_id,
    get_run_paths,
    validate_run_id,
)
from runtime.manifest import finalize_manifest, initialize_run_manifest


def test_run_id_is_readable_deterministic_and_safe():
    instant = datetime(2026, 9, 4, 12, 30, tzinfo=timezone.utc)
    assert create_run_id("canonical intercept", 7, instant) == (
        "canonical-intercept__seed-7__20260904T123000.000000Z"
    )
    assert validate_run_id("clean-run_01") == "clean-run_01"
    for value in ("../escape", "/absolute", "spaces are unsafe", ""):
        try:
            validate_run_id(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Unsafe run id accepted: {value!r}")


def test_relative_output_roots_are_repository_relative(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    paths = get_run_paths(
        {"experiment": {"stage": "canonical", "output_root": "artifacts/runs"}},
        seed=42,
        run_id="path-check",
    )
    assert paths["run_dir"] == PROJECT_ROOT / "artifacts/runs/path-check"

    try:
        get_run_paths(
            {"experiment": {"stage": "../escape"}},
            seed=42,
            run_id="path-check",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Unsafe stage path was accepted")


def test_manifest_records_resolved_config_hashes_and_artifacts(tmp_path):
    source_config = tmp_path / "source.yaml"
    source_config.write_text("value: 3\n", encoding="utf-8")
    output_model = tmp_path / "model.zip"
    output_model.write_bytes(b"model-bytes")
    run_dir = tmp_path / "run-1"
    config = {"value": 3, "nested": {"enabled": True}}

    manifest_path = initialize_run_manifest(
        run_dir,
        run_id="run-1",
        run_kind="test",
        config=config,
        seed=123,
        command=["python", "train.py"],
        runtime_options={"environment": {"use_wind": False}},
        source_config=source_config,
        repository_root=tmp_path,
    )

    initial = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert initial["status"] == "running"
    assert initial["seed"] == 123
    assert initial["config"]["source_sha256"]
    assert initial["config"]["semantic_sha256"]
    with (run_dir / "resolved_config.yaml").open("r", encoding="utf-8") as stream:
        assert yaml.safe_load(stream) == config

    finalize_manifest(
        manifest_path,
        status="complete",
        output_model=output_model,
        extra={"result": {"episodes": 5}},
    )
    final = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert final["status"] == "complete"
    assert final["completed_at_utc"]
    assert final["wall_time_seconds"] >= 0.0
    assert final["output_model"]["sha256"]
    assert final["result"] == {"episodes": 5}


def test_existing_run_directory_is_never_reused(tmp_path):
    run_dir = tmp_path / "already-there"
    run_dir.mkdir()
    try:
        initialize_run_manifest(
            run_dir,
            run_id="already-there",
            run_kind="test",
            config={"value": 1},
            seed=1,
            command=["test"],
            repository_root=tmp_path,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("An existing run directory was silently reused")
