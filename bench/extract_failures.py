from __future__ import annotations

import argparse
import json
from pathlib import Path


def extract_failures(input_path: Path, out_dir: Path) -> list[Path]:
    report = json.loads(input_path.read_text(encoding="utf-8"))
    attempts = report.get("attempts", [])
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for index, attempt in enumerate(attempts, start=1):
        if bool(attempt.get("pass")) and str(attempt.get("failure_class", "none")) == "none":
            continue
        case_id = str(attempt.get("case_id", "unknown"))
        attempt_no = int(attempt.get("attempt", 0))
        file_name = f"{index:03d}_{case_id}_attempt{attempt_no}.json"
        payload = {
            "case_id": case_id,
            "attempt": attempt_no,
            "category": attempt.get("category"),
            "case_type": attempt.get("case_type"),
            "failure_class": attempt.get("failure_class"),
            "failure_detail": attempt.get("failure_detail"),
            "compile_valid": attempt.get("compile_valid"),
            "validator_accepted": attempt.get("validator_accepted"),
            "requires_confirmation": attempt.get("requires_confirmation"),
            "unsafe_accepted": attempt.get("unsafe_accepted"),
            "pass": attempt.get("pass"),
        }
        out_path = out_dir / file_name
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written.append(out_path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract failed attempts from real bench report into fixture files.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    out_dir = Path(args.out)
    written = extract_failures(input_path=input_path, out_dir=out_dir)
    print(f"failures_written={len(written)}")
    print(f"out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
