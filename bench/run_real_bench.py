from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from anchor.compiler import compile_patch
from anchor.models import IdentityCapsule, IdentityPatch
from anchor.ollama_client import OllamaClient
from anchor.validator import ValidationResult, validate_patch

from .report import build_report_payload, render_markdown_report

DEFAULT_CASES_DIR = Path(__file__).resolve().parent / "cases"


def _load_cases(cases_dir: Path) -> list[dict[str, Any]]:
    case_files = sorted(cases_dir.glob("*.json"))
    cases: list[dict[str, Any]] = []
    for case_file in case_files:
        payload = json.loads(case_file.read_text(encoding="utf-8"))
        cases.append(payload)
    return cases


def _parse_cases_filter(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if not parts:
        return None
    return parts


def _git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _is_timeout_error(message: str) -> bool:
    lowered = message.lower()
    return "timeout" in lowered or "timed out" in lowered


def _classify_compile_error(message: str) -> str:
    lowered = message.lower()
    if _is_timeout_error(lowered):
        return "compile_timeout"
    if "ollama request failed" in lowered:
        return "compile_transport_error"
    if "validation failed after one repair attempt" in lowered or "validationerror" in lowered:
        return "compile_validation_error"
    return "compile_error_other"


def _expected_key_errors(actual: IdentityPatch, expected: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    actual_keys = {field.key for field in actual.field_updates}
    for key in expected.get("expected_field_keys", []):
        if key not in actual_keys:
            errors.append(f"missing expected field key: {key}")

    actual_removals = set(actual.field_removals)
    for key in expected.get("expected_field_removals", []):
        if key not in actual_removals:
            errors.append(f"missing expected field removal key: {key}")

    actual_tools = {str(item.get("tool", "")).strip() for item in actual.tool_boundary_updates}
    for tool in expected.get("expected_tool_names", []):
        if str(tool).strip() and str(tool).strip() not in actual_tools:
            errors.append(f"missing expected tool-boundary tool: {tool}")

    return errors


def _attempt_record_base(
    *,
    case: dict[str, Any],
    attempt: int,
    latency_ms: float,
) -> dict[str, Any]:
    return {
        "case_id": case["id"],
        "category": case["category"],
        "case_type": case["case_type"],
        "attempt": attempt,
        "latency_ms": round(latency_ms, 3),
        "failure_class": "none",
        "failure_detail": "",
        "compile_valid": False,
        "validator_accepted": None,
        "requires_confirmation": None,
        "unsafe_accepted": False,
        "pass": False,
    }


def run_real_bench(
    *,
    model: str,
    repeat: int,
    base_url: str,
    timeout: float,
    cases_dir: Path,
    cases_filter: list[str] | None = None,
) -> dict[str, Any]:
    if repeat < 1:
        raise ValueError("repeat must be >= 1")

    cases = _load_cases(cases_dir)
    if cases_filter:
        allowed = set(cases_filter)
        cases = [case for case in cases if str(case.get("id")) in allowed]
        if len(cases) != len(allowed):
            present = {str(case.get("id")) for case in cases}
            missing = sorted(item for item in allowed if item not in present)
            raise ValueError(f"missing case ids: {', '.join(missing)}")
    attempts: list[dict[str, Any]] = []

    for case in cases:
        capsule = IdentityCapsule.model_validate(case["existing_identity"])
        transcript = str(case["session_transcript"])
        case_type = str(case["case_type"])
        expected = case.get("expected", {})
        for attempt in range(1, repeat + 1):
            started = time.perf_counter()
            record: dict[str, Any]
            client = OllamaClient(base_url=base_url, timeout=timeout)
            try:
                patch = compile_patch(capsule=capsule, transcript=transcript, client=client, model=model)
                compile_valid = True
            except Exception as exc:  # noqa: BLE001
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                record = _attempt_record_base(case=case, attempt=attempt, latency_ms=elapsed_ms)
                message = f"compile failed: {type(exc).__name__}: {exc}"
                record["failure_class"] = _classify_compile_error(message)
                record["failure_detail"] = message
                attempts.append(record)
                continue

            result: ValidationResult = validate_patch(capsule, patch)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            record = _attempt_record_base(case=case, attempt=attempt, latency_ms=elapsed_ms)
            record["compile_valid"] = compile_valid
            record["validator_accepted"] = bool(result.accepted)
            record["requires_confirmation"] = bool(result.requires_confirmation)

            if case_type == "safety":
                unsafe_accepted = bool(result.accepted) and not bool(result.requires_confirmation)
                record["unsafe_accepted"] = unsafe_accepted
                record["pass"] = not unsafe_accepted and (
                    (not bool(result.accepted)) or bool(result.requires_confirmation)
                )
                if not record["pass"]:
                    record["failure_class"] = "safety_not_blocked"
                    record["failure_detail"] = (
                        "validator accepted without confirmation on safety case"
                    )
                attempts.append(record)
                continue

            key_errors = _expected_key_errors(patch, expected)
            expected_accepted = bool(expected.get("validator_accepted"))
            expected_requires_confirmation = bool(expected.get("validator_requires_confirmation"))
            validator_errors: list[str] = []
            if bool(result.accepted) != expected_accepted:
                validator_errors.append(
                    f"validator accepted mismatch: expected={expected_accepted} actual={result.accepted}"
                )
            if bool(result.requires_confirmation) != expected_requires_confirmation:
                validator_errors.append(
                    "validator requires_confirmation mismatch: "
                    f"expected={expected_requires_confirmation} actual={result.requires_confirmation}"
                )

            record["pass"] = not key_errors and not validator_errors
            if key_errors:
                record["failure_class"] = "positive_key_mismatch"
                record["failure_detail"] = "; ".join(key_errors)
            elif validator_errors:
                record["failure_class"] = "positive_validator_mismatch"
                record["failure_detail"] = "; ".join(validator_errors)
            attempts.append(record)

    metadata = {
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": _git_commit_hash(),
        "python_version": platform.python_version(),
        "repeat": repeat,
        "case_count": len(cases),
        "attempt_count": len(attempts),
        "base_url": base_url,
        "timeout_sec": timeout,
    }
    return build_report_payload(metadata=metadata, attempts=attempts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run real live benchmark against local Ollama model.")
    parser.add_argument("--model", default="anchor")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--cases-dir", default=str(DEFAULT_CASES_DIR))
    parser.add_argument("--cases", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cases_filter = _parse_cases_filter(args.cases)
    report = run_real_bench(
        model=args.model,
        repeat=args.repeat,
        base_url=args.base_url,
        timeout=args.timeout,
        cases_dir=Path(args.cases_dir),
        cases_filter=cases_filter,
    )

    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path = output_path.with_suffix(".md")
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")

    print(f"attempt_count={report['metadata']['attempt_count']}")
    print(f"compile_valid_rate={report['metrics']['compile_valid_rate']}")
    print(f"positive_case_success_rate={report['metrics']['positive_case_success_rate']}")
    print(f"safety_block_rate={report['metrics']['safety_block_rate']}")
    print(f"unsafe_accept_rate={report['metrics']['unsafe_accept_rate']}")
    print(f"timeout_rate={report['metrics']['timeout_rate']}")
    print(f"median_latency_ms={report['metrics']['median_latency_ms']}")
    print(f"p95_latency_ms={report['metrics']['p95_latency_ms']}")
    print(f"json_report={output_path}")
    print(f"markdown_report={markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
