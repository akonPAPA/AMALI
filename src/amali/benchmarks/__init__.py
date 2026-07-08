"""Benchmark harness and honest external-comparison protocol."""

from amali.benchmarks.harness import (
    BenchmarkItem,
    BenchmarkReport,
    ItemResult,
    run_benchmark,
    score_answer,
)
from amali.benchmarks.comparison import (
    ComparisonReport,
    compare_deterministic_baseline,
    external_comparison_report,
    load_external_baseline,
)

__all__ = [
    "BenchmarkItem",
    "ItemResult",
    "BenchmarkReport",
    "run_benchmark",
    "score_answer",
    "ComparisonReport",
    "compare_deterministic_baseline",
    "external_comparison_report",
    "load_external_baseline",
]
