from __future__ import annotations

import re
from pathlib import Path


def _benchmark_section(readme: str) -> str:
    pattern = re.compile(
        r"## Evals(?P<section>.*?)## Ollama Model Usage",
        re.DOTALL,
    )
    match = pattern.search(readme)
    assert match is not None
    return match.group("section")


def test_readme_contains_real_bench_command_and_metrics() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    section = _benchmark_section(readme)

    assert "python -m bench.run_real_bench --model anchor --repeat 5 --output results/baseline-anchor.json" in section
    assert "compile_valid_rate" in section
    assert "positive_case_success_rate" in section
    assert "safety_block_rate" in section
    assert "unsafe_accept_rate" in section
    assert "timeout_rate" in section
    assert "median_latency_ms" in section
    assert "p95_latency_ms" in section
    assert "per-category pass rates" in section
    assert "failure-class counts" in section


def test_readme_benchmark_section_excludes_forbidden_words() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    section = _benchmark_section(readme).lower()
    forbidden = [
        "impressive",
        "robust",
        "reliable",
        "strong",
        "good",
        "bad",
        "buyer-grade",
        "production-ready",
        "trusted",
    ]
    for word in forbidden:
        assert re.search(rf"\\b{re.escape(word)}\\b", section) is None
