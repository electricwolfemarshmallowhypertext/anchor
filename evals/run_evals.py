from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anchor.models import IdentityCapsule, IdentityPatch
from anchor.validator import validate_patch

CASES_DIR = Path(__file__).resolve().parent / "cases"
REQUIRED_CASE_KEYS = {
    "existing_identity",
    "session_transcript",
    "expected_patch",
    "expected_validator_result",
}


@dataclass
class EvalFailure:
    case_id: str
    issues: list[str]


def load_cases(cases_dir: Path | None = None) -> list[dict[str, Any]]:
    root = CASES_DIR if cases_dir is None else Path(cases_dir)
    case_files = sorted(root.glob("*.json"))
    cases: list[dict[str, Any]] = []
    for case_file in case_files:
        payload = json.loads(case_file.read_text(encoding="utf-8"))
        payload.setdefault("id", case_file.stem)
        payload.setdefault("category", "uncategorized")
        payload["__file__"] = str(case_file)
        cases.append(payload)
    return cases


def _check_required_keys(case: dict[str, Any]) -> list[str]:
    missing = REQUIRED_CASE_KEYS.difference(case.keys())
    if not missing:
        return []
    return [f"missing keys: {', '.join(sorted(missing))}"]


def evaluate_case(case: dict[str, Any]) -> list[str]:
    issues = _check_required_keys(case)
    if issues:
        return issues

    capsule = IdentityCapsule.model_validate(case["existing_identity"])
    patch = IdentityPatch.model_validate(case["expected_patch"])
    expected = case["expected_validator_result"]
    session_transcript = case["session_transcript"]

    if not isinstance(session_transcript, str) or not session_transcript.strip():
        issues.append("session_transcript must be a non-empty string")
    if patch.agent_id != capsule.agent_id:
        issues.append("expected_patch.agent_id must match existing_identity.agent_id")
    if patch.from_version != capsule.version:
        issues.append("expected_patch.from_version must match existing_identity.version")
    if not isinstance(expected, dict):
        issues.append("expected_validator_result must be an object")
        return issues

    result = validate_patch(capsule, patch)
    expected_accepted = bool(expected.get("accepted"))
    expected_requires_confirmation = bool(expected.get("requires_confirmation"))
    if result.accepted != expected_accepted:
        issues.append(f"accepted mismatch: expected={expected_accepted} actual={result.accepted}")
    if result.requires_confirmation != expected_requires_confirmation:
        issues.append(
            "requires_confirmation mismatch: "
            f"expected={expected_requires_confirmation} actual={result.requires_confirmation}"
        )

    expected_error_contains = expected.get("error_contains", [])
    if not isinstance(expected_error_contains, list):
        issues.append("expected_validator_result.error_contains must be a list")
    else:
        for fragment in expected_error_contains:
            if not any(fragment in err for err in result.errors):
                issues.append(f"missing error fragment: {fragment}")

    expected_confirmation_contains = expected.get("confirmation_contains", [])
    if not isinstance(expected_confirmation_contains, list):
        issues.append("expected_validator_result.confirmation_contains must be a list")
    else:
        for fragment in expected_confirmation_contains:
            if not any(fragment in reason for reason in result.confirmation_reasons):
                issues.append(f"missing confirmation fragment: {fragment}")

    return issues


def run_all_cases(cases_dir: Path | None = None) -> tuple[int, list[EvalFailure]]:
    failures: list[EvalFailure] = []
    for case in load_cases(cases_dir):
        case_id = str(case.get("id", "unknown"))
        issues = evaluate_case(case)
        if issues:
            failures.append(EvalFailure(case_id=case_id, issues=issues))
    total = len(load_cases(cases_dir))
    return total, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic Anchor eval cases.")
    parser.add_argument("--cases-dir", default=str(CASES_DIR))
    args = parser.parse_args(argv)

    total, failures = run_all_cases(Path(args.cases_dir))
    if failures:
        print(f"evals failed: {len(failures)}/{total}")
        for failure in failures:
            print(f"- {failure.case_id}")
            for issue in failure.issues:
                print(f"  - {issue}")
        return 1

    print(f"evals passed: {total}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

