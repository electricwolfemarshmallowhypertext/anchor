from __future__ import annotations

from anchor.models import IdentityField, IdentityPatch
from evals import run_ollama_benchmark


def test_run_benchmark_passes_when_compiler_matches_expected(monkeypatch) -> None:
    def fake_compile_patch(capsule, transcript, client, model):
        return IdentityPatch(
            agent_id=capsule.agent_id,
            from_version=capsule.version,
            field_updates=[
                IdentityField(
                    key="working_style.verbosity",
                    value="detailed during planning, concise during execution",
                    source="explicit_user_instruction",
                    confidence=0.95,
                    evidence=["User: During planning, go deeper; otherwise stay concise."],
                )
            ],
        )

    monkeypatch.setattr(run_ollama_benchmark, "compile_patch", fake_compile_patch)
    ok, errors = run_ollama_benchmark.run_benchmark(
        case_id="02_explicit_preference_planning_depth",
        model="tionne/anchor",
        base_url="http://localhost:11434",
        timeout=1.0,
    )
    assert ok
    assert errors == []


def test_run_benchmark_fails_on_conditional_collapse(monkeypatch) -> None:
    def fake_compile_patch(capsule, transcript, client, model):
        return IdentityPatch(
            agent_id=capsule.agent_id,
            from_version=capsule.version,
            field_updates=[
                IdentityField(
                    key="working_style.verbosity",
                    value="deeper",
                    source="explicit_user_instruction",
                    confidence=0.95,
                    evidence=["User: During planning, go deeper; otherwise stay concise."],
                )
            ],
        )

    monkeypatch.setattr(run_ollama_benchmark, "compile_patch", fake_compile_patch)
    ok, errors = run_ollama_benchmark.run_benchmark(
        case_id="02_explicit_preference_planning_depth",
        model="tionne/anchor",
        base_url="http://localhost:11434",
        timeout=1.0,
    )
    assert not ok
    assert any("field mismatch for working_style.verbosity" in err for err in errors)


def test_run_benchmark_reports_compile_failure_instead_of_traceback(monkeypatch) -> None:
    def fake_compile_patch(capsule, transcript, client, model):
        del capsule, transcript, client, model
        raise ValueError("missing required patch fields")

    monkeypatch.setattr(run_ollama_benchmark, "compile_patch", fake_compile_patch)
    ok, errors = run_ollama_benchmark.run_benchmark(
        case_id="02_explicit_preference_planning_depth",
        model="anchor",
        base_url="http://localhost:11434",
        timeout=1.0,
    )
    assert not ok
    assert any("compile failed: ValueError: missing required patch fields" in err for err in errors)


def test_run_benchmark_passes_timeout_to_ollama_client(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, base_url: str, timeout: float):
            captured["base_url"] = base_url
            captured["timeout"] = timeout

    def fake_compile_patch(capsule, transcript, client, model):
        del transcript, model
        assert isinstance(client, FakeClient)
        return IdentityPatch(agent_id=capsule.agent_id, from_version=capsule.version)

    monkeypatch.setattr(run_ollama_benchmark, "OllamaClient", FakeClient)
    monkeypatch.setattr(run_ollama_benchmark, "compile_patch", fake_compile_patch)
    ok, errors = run_ollama_benchmark.run_benchmark(
        case_id="17_model_swap_continuity_no_change",
        model="anchor",
        base_url="http://localhost:11434",
        timeout=42.5,
    )

    assert ok
    assert errors == []
    assert captured == {"base_url": "http://localhost:11434", "timeout": 42.5}
