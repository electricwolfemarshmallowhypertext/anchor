from __future__ import annotations

import json
from pathlib import Path

from bench.extract_failures import extract_failures


def test_extract_failures_writes_only_failed_attempts(tmp_path: Path) -> None:
    report = {
        "attempts": [
            {"case_id": "a", "attempt": 1, "pass": True, "failure_class": "none"},
            {
                "case_id": "b",
                "attempt": 1,
                "pass": False,
                "failure_class": "positive_key_mismatch",
                "failure_detail": "missing key",
                "category": "cat",
                "case_type": "positive",
            },
            {
                "case_id": "c",
                "attempt": 2,
                "pass": True,
                "failure_class": "compile_validation_error",
                "failure_detail": "compile failed",
                "category": "cat2",
                "case_type": "safety",
            },
        ]
    }
    input_path = tmp_path / "report.json"
    input_path.write_text(json.dumps(report), encoding="utf-8")
    out_dir = tmp_path / "failures"

    written = extract_failures(input_path=input_path, out_dir=out_dir)
    assert len(written) == 2
    names = sorted(path.name for path in written)
    assert names[0].endswith("_b_attempt1.json")
    assert names[1].endswith("_c_attempt2.json")

    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["case_id"] == "b"
    assert payload["failure_class"] == "positive_key_mismatch"
