from __future__ import annotations

import json
from pathlib import Path

from anchor.models import IdentityField, IdentityPatch, ValidationResult
from bench import run_real_bench


def _write_case(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_real_case_pack_count_and_distribution() -> None:
    cases = run_real_bench._load_cases(run_real_bench.DEFAULT_CASES_DIR)
    assert len(cases) == 40
    counts: dict[str, int] = {}
    for case in cases:
        category = case["category"]
        counts[category] = counts.get(category, 0) + 1

    assert counts == {
        "explicit_preference_style": 8,
        "role_correction": 6,
        "temporary_mood_vs_durable_identity": 6,
        "prompt_injection_identity_reset": 6,
        "secret_leakage": 6,
        "tool_boundary_escalation": 4,
        "rollback_model_swap_continuity": 4,
    }


def test_parse_cases_filter() -> None:
    assert run_real_bench._parse_cases_filter("") is None
    assert run_real_bench._parse_cases_filter("a,b") == ["a", "b"]


def test_run_real_bench_records_attempt_schema(monkeypatch, tmp_path: Path) -> None:
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    _write_case(
        cases_dir / "c1.json",
        {
            "id": "c1",
            "category": "explicit_preference_style",
            "case_type": "positive",
            "existing_identity": {"agent_id": "demo", "version": 1},
            "session_transcript": "positive transcript",
            "expected": {
                "expected_field_keys": ["working_style.verbosity"],
                "expected_field_removals": [],
                "expected_tool_names": [],
                "validator_accepted": True,
                "validator_requires_confirmation": False,
            },
        },
    )
    _write_case(
        cases_dir / "c2.json",
        {
            "id": "c2",
            "category": "secret_leakage",
            "case_type": "safety",
            "existing_identity": {"agent_id": "demo", "version": 1},
            "session_transcript": "safety transcript",
            "expected": {
                "expected_field_keys": [],
                "expected_field_removals": [],
                "expected_tool_names": [],
            },
        },
    )

    def fake_compile_patch(capsule, transcript, client, model):
        del client, model
        if "positive" in transcript:
            return IdentityPatch(
                agent_id=capsule.agent_id,
                from_version=capsule.version,
                field_updates=[
                    IdentityField(
                        key="working_style.verbosity",
                        value="brief",
                        source="explicit_user_instruction",
                        confidence=0.95,
                        evidence=["positive transcript"],
                    )
                ],
            )
        return IdentityPatch(agent_id=capsule.agent_id, from_version=capsule.version)

    def fake_validate_patch(capsule, patch):
        if patch.field_updates:
            return ValidationResult(accepted=True, requires_confirmation=False)
        return ValidationResult(accepted=False, requires_confirmation=False)

    monkeypatch.setattr(run_real_bench, "compile_patch", fake_compile_patch)
    monkeypatch.setattr(run_real_bench, "validate_patch", fake_validate_patch)
    monkeypatch.setattr(run_real_bench, "_git_commit_hash", lambda: "abc123")

    report = run_real_bench.run_real_bench(
        model="anchor",
        repeat=2,
        base_url="http://localhost:11434",
        timeout=1.0,
        cases_dir=cases_dir,
        cases_filter=["c1", "c2"],
    )

    assert report["metadata"]["attempt_count"] == 4
    attempt = report["attempts"][0]
    required = {
        "case_id",
        "category",
        "case_type",
        "attempt",
        "latency_ms",
        "failure_class",
        "failure_detail",
        "compile_valid",
        "validator_accepted",
        "requires_confirmation",
        "unsafe_accepted",
        "pass",
    }
    assert required.issubset(set(attempt.keys()))


def test_run_real_bench_classifies_timeout_compile_failure(monkeypatch, tmp_path: Path) -> None:
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    _write_case(
        cases_dir / "c1.json",
        {
            "id": "c1",
            "category": "prompt_injection_identity_reset",
            "case_type": "safety",
            "existing_identity": {"agent_id": "demo", "version": 1},
            "session_transcript": "timeout transcript",
            "expected": {
                "expected_field_keys": [],
                "expected_field_removals": [],
                "expected_tool_names": [],
            },
        },
    )

    def fake_compile_patch(capsule, transcript, client, model):
        del capsule, transcript, client, model
        raise RuntimeError("ollama request failed: timed out")

    monkeypatch.setattr(run_real_bench, "compile_patch", fake_compile_patch)
    monkeypatch.setattr(run_real_bench, "_git_commit_hash", lambda: "abc123")

    report = run_real_bench.run_real_bench(
        model="anchor",
        repeat=1,
        base_url="http://localhost:11434",
        timeout=1.0,
        cases_dir=cases_dir,
        cases_filter=["c1"],
    )
    attempt = report["attempts"][0]

    assert attempt["compile_valid"] is False
    assert attempt["failure_class"] == "compile_timeout"
    assert attempt["pass"] is False


def test_main_writes_json_and_markdown(monkeypatch, tmp_path: Path) -> None:
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    _write_case(
        cases_dir / "c1.json",
        {
            "id": "c1",
            "category": "explicit_preference_style",
            "case_type": "positive",
            "existing_identity": {"agent_id": "demo", "version": 1},
            "session_transcript": "positive transcript",
            "expected": {
                "expected_field_keys": [],
                "expected_field_removals": [],
                "expected_tool_names": [],
                "validator_accepted": True,
                "validator_requires_confirmation": False,
            },
        },
    )

    monkeypatch.setattr(
        run_real_bench,
        "compile_patch",
        lambda capsule, transcript, client, model: IdentityPatch(
            agent_id=capsule.agent_id, from_version=capsule.version
        ),
    )
    monkeypatch.setattr(
        run_real_bench,
        "validate_patch",
        lambda capsule, patch: ValidationResult(accepted=True, requires_confirmation=False),
    )
    monkeypatch.setattr(run_real_bench, "_git_commit_hash", lambda: "abc123")

    output = tmp_path / "results" / "baseline.json"
    code = run_real_bench.main(
        [
            "--model",
            "anchor",
            "--repeat",
            "1",
            "--cases-dir",
            str(cases_dir),
            "--cases",
            "c1",
            "--output",
            str(output),
        ]
    )

    assert code == 0
    assert output.exists()
    assert output.with_suffix(".md").exists()


def test_run_real_bench_missing_case_filter_raises(tmp_path: Path) -> None:
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    _write_case(
        cases_dir / "c1.json",
        {
            "id": "c1",
            "category": "explicit_preference_style",
            "case_type": "positive",
            "existing_identity": {"agent_id": "demo", "version": 1},
            "session_transcript": "positive transcript",
            "expected": {
                "expected_field_keys": [],
                "expected_field_removals": [],
                "expected_tool_names": [],
                "validator_accepted": True,
                "validator_requires_confirmation": False,
            },
        },
    )

    try:
        run_real_bench.run_real_bench(
            model="anchor",
            repeat=1,
            base_url="http://localhost:11434",
            timeout=1.0,
            cases_dir=cases_dir,
            cases_filter=["missing"],
        )
    except ValueError as exc:
        assert "missing case ids" in str(exc)
        return
    raise AssertionError("expected ValueError for missing case ids")
