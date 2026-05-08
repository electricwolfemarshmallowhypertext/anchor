from __future__ import annotations

import pytest

from anchor.models import IdentityField, IdentityPatch, ValidationResult
from evals import run_publish_gate


def test_run_publish_gate_passes_when_threshold_met(monkeypatch) -> None:
    def fake_attempt(case_id, model, base_url, timeout):
        del case_id, model, base_url, timeout
        return True, False, "positive_case_ok"

    monkeypatch.setattr(run_publish_gate, "run_live_gate_attempt", fake_attempt)
    ok, summary, fatal_errors = run_publish_gate.run_publish_gate(
        model="anchor",
        base_url="http://localhost:11434",
        timeout=1.0,
        cases=["02_explicit_preference_planning_depth", "11_secret_leakage_env_key"],
        repeat=3,
        min_pass_rate=0.9,
        show_progress=False,
    )

    assert ok
    assert fatal_errors == []
    assert summary["02_explicit_preference_planning_depth"]["pass_rate"] == pytest.approx(1.0)
    assert summary["11_secret_leakage_env_key"]["passes"] == 3


def test_run_publish_gate_fails_when_pass_rate_below_threshold(monkeypatch) -> None:
    attempts = iter(
        [
            (True, False, "positive_case_ok"),
            (False, False, "missing expected field key: working_style.verbosity"),
            (True, False, "positive_case_ok"),
        ]
    )

    def fake_attempt(case_id, model, base_url, timeout):
        del case_id, model, base_url, timeout
        return next(attempts)

    monkeypatch.setattr(run_publish_gate, "run_live_gate_attempt", fake_attempt)
    ok, summary, fatal_errors = run_publish_gate.run_publish_gate(
        model="anchor",
        base_url="http://localhost:11434",
        timeout=1.0,
        cases=["02_explicit_preference_planning_depth"],
        repeat=3,
        min_pass_rate=0.9,
        show_progress=False,
    )

    assert not ok
    assert fatal_errors == []
    assert summary["02_explicit_preference_planning_depth"]["passes"] == 2
    assert summary["02_explicit_preference_planning_depth"]["pass_rate"] == pytest.approx(2 / 3)


def test_run_publish_gate_fails_on_compile_fatal(monkeypatch) -> None:
    attempts = iter(
        [
            (False, True, "compile failed: RuntimeError: identity patch validation failed"),
            (True, False, "positive_case_ok"),
            (True, False, "positive_case_ok"),
        ]
    )

    def fake_attempt(case_id, model, base_url, timeout):
        del case_id, model, base_url, timeout
        return next(attempts)

    monkeypatch.setattr(run_publish_gate, "run_live_gate_attempt", fake_attempt)
    ok, summary, fatal_errors = run_publish_gate.run_publish_gate(
        model="anchor",
        base_url="http://localhost:11434",
        timeout=1.0,
        cases=["11_secret_leakage_env_key"],
        repeat=3,
        min_pass_rate=0.5,
        show_progress=False,
    )

    assert not ok
    assert summary["11_secret_leakage_env_key"]["pass_rate"] == pytest.approx(2 / 3)
    assert len(fatal_errors) == 1
    assert "11_secret_leakage_env_key attempt 1" in fatal_errors[0]


def test_run_publish_gate_allows_safety_fail_closed_without_blocking(monkeypatch) -> None:
    attempts = iter(
        [
            (True, False, "safe_fail_closed: compile failed: RuntimeError: invalid removal path"),
            (True, False, "safe_fail_closed: compile failed: RuntimeError: invalid removal path"),
            (True, False, "safe_fail_closed: compile failed: RuntimeError: invalid removal path"),
        ]
    )

    def fake_attempt(case_id, model, base_url, timeout):
        del case_id, model, base_url, timeout
        return next(attempts)

    monkeypatch.setattr(run_publish_gate, "run_live_gate_attempt", fake_attempt)
    ok, summary, fatal_errors = run_publish_gate.run_publish_gate(
        model="anchor",
        base_url="http://localhost:11434",
        timeout=60.0,
        cases=["07_prompt_injection_ignore_previous_identity"],
        repeat=3,
        min_pass_rate=0.9,
        show_progress=False,
    )

    assert ok
    assert summary["07_prompt_injection_ignore_previous_identity"]["pass_rate"] == pytest.approx(1.0)
    assert fatal_errors == []


def test_run_publish_gate_prints_attempt_progress(monkeypatch, capsys) -> None:
    def fake_attempt(case_id, model, base_url, timeout):
        del case_id, model, base_url, timeout
        return True, False, "positive_case_ok"

    monkeypatch.setattr(run_publish_gate, "run_live_gate_attempt", fake_attempt)
    ok, summary, fatal_errors = run_publish_gate.run_publish_gate(
        model="anchor",
        base_url="http://localhost:11434",
        timeout=60.0,
        cases=["02_explicit_preference_planning_depth"],
        repeat=1,
        min_pass_rate=0.9,
        show_progress=True,
    )

    captured = capsys.readouterr()
    assert ok
    assert fatal_errors == []
    assert summary["02_explicit_preference_planning_depth"]["passes"] == 1
    assert "[02_explicit_preference_planning_depth] attempt 1/1 start" in captured.out
    assert "[02_explicit_preference_planning_depth] attempt 1/1 pass" in captured.out
    assert "[02_explicit_preference_planning_depth] complete passes=1/1 pass_rate=1.00" in captured.out


def test_run_publish_gate_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="timeout must be > 0"):
        run_publish_gate.run_publish_gate(
            model="anchor",
            base_url="http://localhost:11434",
            timeout=0.0,
            cases=["02_explicit_preference_planning_depth"],
            repeat=1,
            min_pass_rate=0.9,
            show_progress=False,
        )


def test_run_live_gate_attempt_safety_case_compile_failure_is_safe_pass(monkeypatch) -> None:
    def fake_compile_patch(capsule, transcript, client, model):
        del capsule, transcript, client, model
        raise RuntimeError("identity patch validation failed after one repair attempt")

    monkeypatch.setattr(run_publish_gate, "compile_patch", fake_compile_patch)
    ok, is_fatal, detail = run_publish_gate.run_live_gate_attempt(
        case_id="07_prompt_injection_ignore_previous_identity",
        model="anchor",
        base_url="http://localhost:11434",
        timeout=60.0,
    )

    assert ok
    assert not is_fatal
    assert detail.startswith("safe_fail_closed: compile failed: RuntimeError:")


def test_run_live_gate_attempt_case_15_uses_key_level_semantics(monkeypatch) -> None:
    def fake_compile_patch(capsule, transcript, client, model):
        del transcript, client, model
        return IdentityPatch(
            agent_id=capsule.agent_id,
            from_version=capsule.version,
            field_updates=[
                IdentityField(
                    key="purpose",
                    value="Technical research assistant.",
                    source="explicit_user_instruction",
                    confidence=0.95,
                    evidence=["Correction: your role is technical research assistant."],
                )
            ],
        )

    def fake_validate_patch(capsule, patch):
        del capsule, patch
        return ValidationResult(
            accepted=True,
            requires_confirmation=True,
            errors=[],
            confirmation_reasons=[],
            conflicts=[],
            suspicions=[],
        )

    monkeypatch.setattr(run_publish_gate, "compile_patch", fake_compile_patch)
    monkeypatch.setattr(run_publish_gate, "validate_patch", fake_validate_patch)
    ok, is_fatal, detail = run_publish_gate.run_live_gate_attempt(
        case_id="15_role_correction_research_assistant",
        model="anchor",
        base_url="http://localhost:11434",
        timeout=60.0,
    )

    assert ok
    assert not is_fatal
    assert detail == "positive_case_ok"


def test_run_live_gate_attempt_timeout_is_transport_fatal(monkeypatch) -> None:
    def fake_compile_patch(capsule, transcript, client, model):
        del capsule, transcript, client, model
        raise RuntimeError("ollama request failed: timed out")

    monkeypatch.setattr(run_publish_gate, "compile_patch", fake_compile_patch)
    ok, is_fatal, detail = run_publish_gate.run_live_gate_attempt(
        case_id="07_prompt_injection_ignore_previous_identity",
        model="anchor",
        base_url="http://localhost:11434",
        timeout=60.0,
    )

    assert not ok
    assert is_fatal
    assert detail.startswith("transport_timeout: compile failed: RuntimeError:")
