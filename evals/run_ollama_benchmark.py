from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from anchor.compiler import compile_patch
from anchor.models import IdentityCapsule, IdentityPatch
from anchor.ollama_client import OllamaClient
from anchor.validator import validate_patch

CASES_DIR = Path(__file__).resolve().parent / "cases"


def _load_case(case_id: str) -> dict[str, Any]:
    case_path = CASES_DIR / f"{case_id}.json"
    if not case_path.exists():
        raise ValueError(f"case not found: {case_id}")
    return json.loads(case_path.read_text(encoding="utf-8"))


def _field_map(patch: IdentityPatch) -> dict[str, Any]:
    return {field.key: field.value for field in patch.field_updates}


def _check_expected_patch_subset(actual: IdentityPatch, expected: IdentityPatch) -> list[str]:
    errors: list[str] = []
    actual_fields = _field_map(actual)
    for field in expected.field_updates:
        if field.key not in actual_fields:
            errors.append(f"missing expected field update: {field.key}")
            continue
        if actual_fields[field.key] != field.value:
            errors.append(
                "field mismatch for "
                f"{field.key}: expected={field.value!r} actual={actual_fields[field.key]!r}"
            )

    for key in expected.field_removals:
        if key not in actual.field_removals:
            errors.append(f"missing expected field removal: {key}")

    if expected.tool_boundary_updates and actual.tool_boundary_updates != expected.tool_boundary_updates:
        errors.append("tool_boundary_updates mismatch")

    return errors


def run_benchmark(case_id: str, model: str, base_url: str, timeout: float) -> tuple[bool, list[str]]:
    case = _load_case(case_id)
    capsule = IdentityCapsule.model_validate(case["existing_identity"])
    expected_patch = IdentityPatch.model_validate(case["expected_patch"])
    expected_validator = case["expected_validator_result"]
    transcript = case["session_transcript"]

    client = OllamaClient(base_url=base_url, timeout=timeout)
    try:
        actual_patch = compile_patch(capsule=capsule, transcript=transcript, client=client, model=model)
    except Exception as exc:  # noqa: BLE001
        return False, [f"compile failed: {type(exc).__name__}: {exc}"]
    validator_result = validate_patch(capsule, actual_patch)

    errors = _check_expected_patch_subset(actual_patch, expected_patch)
    expected_accepted = bool(expected_validator.get("accepted"))
    expected_requires_confirmation = bool(expected_validator.get("requires_confirmation"))
    if validator_result.accepted != expected_accepted:
        errors.append(
            "validator accepted mismatch: "
            f"expected={expected_accepted} actual={validator_result.accepted}"
        )
    if validator_result.requires_confirmation != expected_requires_confirmation:
        errors.append(
            "validator requires_confirmation mismatch: "
            f"expected={expected_requires_confirmation} actual={validator_result.requires_confirmation}"
        )
    return len(errors) == 0, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run live Ollama benchmark via structured JSON output.")
    parser.add_argument("--case", default="02_explicit_preference_planning_depth")
    parser.add_argument("--model", default="tionne/anchor")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)

    ok, errors = run_benchmark(
        case_id=args.case,
        model=args.model,
        base_url=args.base_url,
        timeout=args.timeout,
    )
    if not ok:
        print("benchmark failed")
        for item in errors:
            print(f"- {item}")
        return 1
    print("benchmark passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
