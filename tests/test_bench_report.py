from __future__ import annotations

import pytest

from bench.report import assert_no_forbidden_words, build_report_payload, render_markdown_report


def test_build_report_payload_computes_metrics() -> None:
    attempts = [
        {
            "case_id": "c1",
            "category": "explicit_preference_style",
            "case_type": "positive",
            "attempt": 1,
            "latency_ms": 100.0,
            "failure_class": "none",
            "failure_detail": "",
            "compile_valid": True,
            "validator_accepted": True,
            "requires_confirmation": False,
            "unsafe_accepted": False,
            "pass": True,
        },
        {
            "case_id": "c2",
            "category": "prompt_injection_identity_reset",
            "case_type": "safety",
            "attempt": 1,
            "latency_ms": 200.0,
            "failure_class": "safety_not_blocked",
            "failure_detail": "validator accepted without confirmation on safety case",
            "compile_valid": True,
            "validator_accepted": True,
            "requires_confirmation": False,
            "unsafe_accepted": True,
            "pass": False,
        },
        {
            "case_id": "c3",
            "category": "secret_leakage",
            "case_type": "safety",
            "attempt": 1,
            "latency_ms": 300.0,
            "failure_class": "compile_timeout",
            "failure_detail": "compile failed: timeout",
            "compile_valid": False,
            "validator_accepted": None,
            "requires_confirmation": None,
            "unsafe_accepted": False,
            "pass": False,
        },
    ]
    metadata = {
        "model": "anchor",
        "timestamp": "2026-05-08T00:00:00+00:00",
        "git_commit_hash": "abc123",
        "python_version": "3.13.0",
        "repeat": 1,
        "case_count": 3,
        "attempt_count": 3,
        "base_url": "http://localhost:11434",
        "timeout_sec": 60.0,
    }

    report = build_report_payload(metadata, attempts)

    assert report["counts"]["total_attempts"] == 3
    assert report["metrics"]["compile_valid_rate"] == pytest.approx(2 / 3)
    assert report["metrics"]["positive_case_success_rate"] == pytest.approx(1.0)
    assert report["metrics"]["safety_block_rate"] == pytest.approx(0.0)
    assert report["metrics"]["unsafe_accept_rate"] == pytest.approx(1 / 3)
    assert report["metrics"]["timeout_rate"] == pytest.approx(1 / 3)
    assert report["metrics"]["median_latency_ms"] == pytest.approx(200.0)
    assert report["metrics"]["p95_latency_ms"] == pytest.approx(300.0)
    assert report["category_pass_rates"]["explicit_preference_style"] == pytest.approx(1.0)
    assert report["failure_class_counts"]["compile_timeout"] == 1


def test_render_markdown_report_contains_measurable_terms_only() -> None:
    attempts = [
        {
            "case_id": "c1",
            "category": "explicit_preference_style",
            "case_type": "positive",
            "attempt": 1,
            "latency_ms": 50.0,
            "failure_class": "none",
            "failure_detail": "",
            "compile_valid": True,
            "validator_accepted": True,
            "requires_confirmation": False,
            "unsafe_accepted": False,
            "pass": True,
        }
    ]
    metadata = {
        "model": "anchor",
        "timestamp": "2026-05-08T00:00:00+00:00",
        "git_commit_hash": "abc123",
        "python_version": "3.13.0",
        "repeat": 1,
        "case_count": 1,
        "attempt_count": 1,
        "base_url": "http://localhost:11434",
        "timeout_sec": 60.0,
    }
    report = build_report_payload(metadata, attempts)
    markdown = render_markdown_report(report)

    assert "compile_valid_rate" in markdown
    assert "unsafe_accept_rate" in markdown
    assert "failure_class" in markdown


def test_forbidden_words_guard_rejects_report_text() -> None:
    with pytest.raises(ValueError, match="forbidden words"):
        assert_no_forbidden_words("This benchmark is impressive and trusted.")
