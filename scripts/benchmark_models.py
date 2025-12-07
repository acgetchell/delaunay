#!/usr/bin/env python3
"""benchmark_models.py - Data models and utilities for benchmark processing.

This module contains data models, parsing functions, and formatting utilities
for benchmark data processing. It provides the core data structures used
throughout the benchmark infrastructure.
"""

import re
from dataclasses import dataclass


@dataclass
class BenchmarkData:
    """Represents benchmark data for a single test case."""

    points: int
    dimension: str
    time_low: float = 0.0
    time_mean: float = 0.0
    time_high: float = 0.0
    time_unit: str = ""
    throughput_low: float | None = None
    throughput_mean: float | None = None
    throughput_high: float | None = None
    throughput_unit: str | None = None

    def with_timing(self, low: float, mean: float, high: float, unit: str) -> "BenchmarkData":
        """Set timing data (fluent interface)."""
        self.time_low = low
        self.time_mean = mean
        self.time_high = high
        self.time_unit = unit
        return self

    def with_throughput(self, low: float, mean: float, high: float, unit: str) -> "BenchmarkData":
        """Set throughput data (fluent interface)."""
        self.throughput_low = low
        self.throughput_mean = mean
        self.throughput_high = high
        self.throughput_unit = unit
        return self

    def to_baseline_format(self) -> str:
        """Convert to baseline file format."""
        lines = [
            f"=== {self.points} Points ({self.dimension}) ===",
            f"Time: [{self.time_low}, {self.time_mean}, {self.time_high}] {self.time_unit}",
        ]

        if self.throughput_low is not None and self.throughput_mean is not None and self.throughput_high is not None and self.throughput_unit:
            lines.append(f"Throughput: [{self.throughput_low}, {self.throughput_mean}, {self.throughput_high}] {self.throughput_unit}")

        lines.append("")
        return "\n".join(lines)


@dataclass
class CircumspherePerformanceData:
    """Represents circumsphere containment performance data."""

    method: str  # insphere, insphere_distance, insphere_lifted
    time_ns: float
    relative_performance: float | None = None
    winner: bool = False


@dataclass
class CircumsphereTestCase:
    """Represents a circumsphere test case with multiple method results."""

    test_name: str
    dimension: str
    methods: dict[str, CircumspherePerformanceData]
    is_boundary_case: bool = False  # True for boundary/edge cases with early-exit optimizations

    def get_winner(self) -> str | None:
        """Get the method name with the best performance."""
        if not self.methods:
            return None
        return min(self.methods.keys(), key=lambda m: self.methods[m].time_ns)

    def get_relative_performance(self, method: str, baseline_method: str | None = None) -> float:
        """Calculate relative performance compared to baseline method."""
        if method not in self.methods:
            return 0.0

        if baseline_method is None:
            baseline_method = self.get_winner()

        if baseline_method is None or baseline_method not in self.methods:
            return 1.0

        baseline_time = self.methods[baseline_method].time_ns
        method_time = self.methods[method].time_ns

        if baseline_time <= 0:
            return 1.0

        return method_time / baseline_time


@dataclass
class VersionComparisonData:
    """Represents performance comparison between versions."""

    test_case: str
    method: str
    old_version: str
    new_version: str
    old_value: float
    new_value: float
    unit: str
    improvement_pct: float = 0.0

    def __post_init__(self) -> None:
        """Calculate improvement percentage."""
        if self.old_value > 0:
            self.improvement_pct = ((self.old_value - self.new_value) / self.old_value) * 100
        else:
            self.improvement_pct = 0.0


# Benchmark parsing functions


def parse_benchmark_header(line: str) -> BenchmarkData | None:
    """
    Parse benchmark header lines to extract test configuration.

    Args:
        line: Input line potentially containing benchmark header

    Returns:
        BenchmarkData object or None if no match
    """
    # Match pattern like "=== 1000 Points (2D) ==="
    match = re.match(r"^=== (\d+) Points \((.+)\) ===$", line.strip())
    if match:
        points = int(match.group(1))
        dimension = match.group(2)
        return BenchmarkData(points=points, dimension=dimension)
    return None


# NOTE: For hot-path parsing of large baseline files in CI, consider precompiling
# the regex patterns below using re.compile() for better performance.


def parse_time_data(benchmark: BenchmarkData, line: str) -> bool:
    """
    Parse time data lines to extract timing information.

    Args:
        benchmark: BenchmarkData object to update
        line: Input line potentially containing time data

    Returns:
        True if data was parsed successfully, False otherwise
    """
    # Match pattern like "Time: [100.0, 110.0, 120.0] µs"
    # Support scientific notation (1.2e3), negative values, and flexible whitespace
    match = re.match(r"^Time:\s*\[([0-9eE+.\-,\s]+)\]\s+(.+)$", line.strip())
    if match:
        try:
            # Parse the list of numbers
            values_str = match.group(1)
            unit = match.group(2)
            values = [float(x.strip()) for x in values_str.split(",")]

            if len(values) == 3:
                benchmark.time_low = values[0]
                benchmark.time_mean = values[1]
                benchmark.time_high = values[2]
                benchmark.time_unit = unit
                return True
        except ValueError:
            pass
    return False


def parse_throughput_data(benchmark: BenchmarkData, line: str) -> bool:
    """
    Parse throughput data lines to extract throughput information.

    Args:
        benchmark: BenchmarkData object to update
        line: Input line potentially containing throughput data

    Returns:
        True if data was parsed successfully, False otherwise
    """
    # Match pattern like "Throughput: [8000.0, 9090.9, 10000.0] Kelem/s"
    # Support scientific notation (1.2e3), negative values, and flexible whitespace
    match = re.match(r"^Throughput:\s*\[([0-9eE+.\-,\s]+)\]\s+(.+)$", line.strip())
    if match:
        try:
            # Parse the list of numbers
            values_str = match.group(1)
            unit = match.group(2)
            values = [float(x.strip()) for x in values_str.split(",")]

            if len(values) == 3:
                benchmark.throughput_low = values[0]
                benchmark.throughput_mean = values[1]
                benchmark.throughput_high = values[2]
                benchmark.throughput_unit = unit
                return True
        except ValueError:
            pass
    return False


def extract_benchmark_data(baseline_content: str) -> list[BenchmarkData]:
    """
    Extract benchmark data from baseline file content.

    Args:
        baseline_content: Content from baseline results file

    Returns:
        List of BenchmarkData objects parsed from content
    """
    benchmarks = []
    current_benchmark = None

    for line in baseline_content.split("\n"):
        # Try to parse as benchmark header
        benchmark = parse_benchmark_header(line)
        if benchmark:
            # Save previous benchmark if it exists
            if current_benchmark:
                benchmarks.append(current_benchmark)
            current_benchmark = benchmark
            continue

        if current_benchmark:
            # Try to parse time data
            if parse_time_data(current_benchmark, line):
                continue

            # Try to parse throughput data
            if parse_throughput_data(current_benchmark, line):
                continue

    # Don't forget the last benchmark
    if current_benchmark:
        benchmarks.append(current_benchmark)

    return benchmarks


# Benchmark formatting functions


def format_time_value(value: float, unit: str) -> str:
    """
    Format time values with appropriate precision and unit conversion.

    Args:
        value: Time value to format
        unit: Current unit of the value

    Returns:
        Formatted time string with appropriate unit, or "N/A" for invalid values
    """
    # Return N/A for zero or negative values (invalid measurements)
    if value <= 0:
        return "N/A"

    # Normalize microsecond aliases to standard µs
    unit = {"us": "µs", "μs": "µs"}.get((unit or "").strip(), (unit or "").strip())
    # Convert µs to ms if >= 1000 µs
    if unit == "µs" and value >= 1000:
        return f"{value / 1000:.3f} ms"
    # Convert ms to s if >= 1000 ms
    if unit == "ms" and value >= 1000:
        return f"{value / 1000:.4f} s"
    if unit == "µs":
        # Use 3 decimal places for values < 1, 2 decimal places otherwise
        if value < 1:
            return f"{value:.3f} µs"
        return f"{value:.2f} µs"
    return f"{value:.2f} {unit}"


def format_throughput_value(value: float | None, unit: str | None) -> str:
    """
    Format throughput values with appropriate precision.

    Args:
        value: Throughput value to format (can be None)
        unit: Unit of the value (can be None)

    Returns:
        Formatted throughput string
    """
    if value is None or unit is None:
        return "N/A"

    # Use 3 decimal places for values < 1 or with fractional parts needing precision
    if value < 1 or (value % 1) != 0:
        return f"{value:.3f} {unit}"
    return f"{value:.2f} {unit}"


def format_benchmark_tables(benchmarks: list[BenchmarkData]) -> list[str]:
    """
    Format benchmark data as markdown tables grouped by dimension.

    Args:
        benchmarks: List of BenchmarkData objects to format

    Returns:
        List of markdown lines containing formatted tables
    """
    lines = []

    # Group benchmarks by dimension
    by_dimension: dict[str, list[BenchmarkData]] = {}
    for bench in benchmarks:
        dim = bench.dimension
        if dim not in by_dimension:
            by_dimension[dim] = []
        by_dimension[dim].append(bench)

    # Sort dimensions numerically (2D, 3D, etc.) rather than lexically
    def _dim_key(d: str) -> tuple[int, str]:
        """Sort key for dimensions: numeric prefix first, then string fallback."""
        m = re.match(r"^\s*(\d+)\s*[dD]\b", d)
        return (int(m.group(1)) if m else 1_000_000, d)

    for dimension in sorted(by_dimension.keys(), key=_dim_key):
        dim_benchmarks = sorted(by_dimension[dimension], key=lambda b: b.points)

        lines.extend(
            [
                f"### {dimension} Triangulation Performance",
                "",
                "| Points | Time (mean) | Throughput (mean) | Scaling |",
                "|--------|-------------|-------------------|----------|",
            ],
        )

        # Calculate scaling relative to smallest benchmark
        first_nonzero = next((b for b in dim_benchmarks if b.time_mean and b.time_mean > 0), None)
        baseline_time = first_nonzero.time_mean if first_nonzero else None

        for bench in dim_benchmarks:
            # Format time and throughput
            time_str = format_time_value(bench.time_mean, bench.time_unit) if bench.time_unit else "N/A"
            throughput_str = (
                format_throughput_value(bench.throughput_mean, bench.throughput_unit)
                if bench.throughput_unit and bench.throughput_mean is not None
                else "N/A"
            )

            # Calculate scaling factor
            if bench.time_mean > 0 and baseline_time and baseline_time > 0:
                scaling = bench.time_mean / baseline_time
                scaling_str = f"{scaling:.1f}x"
            else:
                scaling_str = "N/A"

            lines.append(f"| {bench.points} | {time_str} | {throughput_str} | {scaling_str} |")

        lines.append("")  # Empty line between tables

    return lines
