from __future__ import annotations

from pathlib import Path

from evals.run_evals import REQUIRED_CASE_KEYS, load_cases, run_all_cases


EXPECTED_CATEGORIES = {
    "explicit preference",
    "temporary mood",
    "tool-boundary change",
    "prompt injection",
    "contradiction",
    "secret leakage",
    "style correction",
    "role correction",
    "model swap continuity",
    "rollback relevance",
}


def test_eval_pack_has_20_cases_and_required_keys() -> None:
    cases = load_cases()
    assert len(cases) == 20
    for case in cases:
        assert REQUIRED_CASE_KEYS.issubset(case.keys())


def test_eval_pack_covers_required_categories() -> None:
    cases = load_cases()
    observed = {str(case.get("category", "")).strip().lower() for case in cases}
    assert observed == EXPECTED_CATEGORIES
    counts = {category: 0 for category in EXPECTED_CATEGORIES}
    for case in cases:
        counts[str(case["category"]).strip().lower()] += 1
    for category, count in counts.items():
        assert count == 2, f"category '{category}' expected 2 cases, found {count}"


def test_eval_cases_match_validator_expectations() -> None:
    total, failures = run_all_cases()
    assert total == 20
    assert failures == []


def test_eval_case_files_are_json() -> None:
    cases_dir = Path("evals") / "cases"
    files = sorted(cases_dir.glob("*.json"))
    assert len(files) == 20
    assert all(file.suffix == ".json" for file in files)

