from __future__ import annotations

import re
from statistics import median
from typing import Any

FORBIDDEN_WORDS = (
    "impressive",
    "robust",
    "reliable",
    "strong",
    "good",
    "bad",
    "buyer-grade",
    "production-ready",
    "trusted",
)

_FORBIDDEN_PATTERNS = [
    re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE) for word in FORBIDDEN_WORDS
]


def assert_no_forbidden_words(text: str) -> None:
    matches: list[str] = []
    for pattern, word in zip(_FORBIDDEN_PATTERNS, FORBIDDEN_WORDS):
        if pattern.search(text):
            matches.append(word)
    if matches:
        found = ", ".join(sorted(set(matches)))
        raise ValueError(f"report contains forbidden words: {found}")


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(round((len(sorted_values) - 1) * percentile))
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def compute_metrics(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    total_attempts = len(attempts)
    compile_valid_count = sum(1 for item in attempts if bool(item.get("compile_valid")))
    positive_attempts = [item for item in attempts if item.get("case_type") == "positive"]
    safety_attempts = [item for item in attempts if item.get("case_type") == "safety"]

    positive_success_count = sum(1 for item in positive_attempts if bool(item.get("pass")))
    safety_block_count = sum(1 for item in safety_attempts if bool(item.get("pass")))
    unsafe_accepted_count = sum(1 for item in attempts if bool(item.get("unsafe_accepted")))
    timeout_count = sum(1 for item in attempts if item.get("failure_class") == "compile_timeout")

    latencies = [float(item.get("latency_ms", 0.0)) for item in attempts]

    category_counts: dict[str, dict[str, int]] = {}
    for item in attempts:
        category = str(item.get("category", "unknown"))
        current = category_counts.setdefault(category, {"pass_count": 0, "attempt_count": 0})
        current["attempt_count"] += 1
        if bool(item.get("pass")):
            current["pass_count"] += 1

    category_pass_rates = {
        key: _rate(value["pass_count"], value["attempt_count"]) for key, value in category_counts.items()
    }

    failure_class_counts: dict[str, int] = {}
    for item in attempts:
        failure_class = str(item.get("failure_class", "unknown"))
        failure_class_counts[failure_class] = failure_class_counts.get(failure_class, 0) + 1

    return {
        "counts": {
            "total_attempts": total_attempts,
            "compile_valid_count": compile_valid_count,
            "positive_attempt_count": len(positive_attempts),
            "positive_success_count": positive_success_count,
            "safety_attempt_count": len(safety_attempts),
            "safety_block_count": safety_block_count,
            "unsafe_accepted_count": unsafe_accepted_count,
            "timeout_count": timeout_count,
        },
        "compile_valid_rate": _rate(compile_valid_count, total_attempts),
        "positive_case_success_rate": _rate(positive_success_count, len(positive_attempts)),
        "safety_block_rate": _rate(safety_block_count, len(safety_attempts)),
        "unsafe_accept_rate": _rate(unsafe_accepted_count, total_attempts),
        "timeout_rate": _rate(timeout_count, total_attempts),
        "median_latency_ms": median(latencies) if latencies else 0.0,
        "p95_latency_ms": _percentile(latencies, 0.95),
        "category_pass_rates": category_pass_rates,
        "failure_class_counts": failure_class_counts,
    }


def build_report_payload(metadata: dict[str, Any], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = compute_metrics(attempts)
    return {
        "metadata": metadata,
        "counts": metrics["counts"],
        "metrics": {
            "compile_valid_rate": metrics["compile_valid_rate"],
            "positive_case_success_rate": metrics["positive_case_success_rate"],
            "safety_block_rate": metrics["safety_block_rate"],
            "unsafe_accept_rate": metrics["unsafe_accept_rate"],
            "timeout_rate": metrics["timeout_rate"],
            "median_latency_ms": metrics["median_latency_ms"],
            "p95_latency_ms": metrics["p95_latency_ms"],
        },
        "category_pass_rates": metrics["category_pass_rates"],
        "failure_class_counts": metrics["failure_class_counts"],
        "attempts": attempts,
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    metrics = report["metrics"]
    category_pass_rates = report["category_pass_rates"]
    failure_class_counts = report["failure_class_counts"]

    lines: list[str] = []
    lines.append("# Real Bench Baseline")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append("| field | value |")
    lines.append("| --- | --- |")
    for key in (
        "model",
        "timestamp",
        "git_commit_hash",
        "python_version",
        "repeat",
        "case_count",
        "attempt_count",
        "base_url",
        "timeout_sec",
    ):
        lines.append(f"| {key} | {metadata.get(key)} |")
    lines.append("")
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    for key in (
        "compile_valid_rate",
        "positive_case_success_rate",
        "safety_block_rate",
        "unsafe_accept_rate",
        "timeout_rate",
        "median_latency_ms",
        "p95_latency_ms",
    ):
        lines.append(f"| {key} | {metrics.get(key)} |")
    lines.append("")
    lines.append("## Category Pass Rates")
    lines.append("")
    lines.append("| category | pass_rate |")
    lines.append("| --- | --- |")
    for category in sorted(category_pass_rates):
        lines.append(f"| {category} | {category_pass_rates[category]} |")
    lines.append("")
    lines.append("## Failure Class Counts")
    lines.append("")
    lines.append("| failure_class | count |")
    lines.append("| --- | --- |")
    for failure_class in sorted(failure_class_counts):
        lines.append(f"| {failure_class} | {failure_class_counts[failure_class]} |")

    markdown = "\n".join(lines) + "\n"
    assert_no_forbidden_words(markdown)
    return markdown
