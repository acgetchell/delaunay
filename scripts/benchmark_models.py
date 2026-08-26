#!/usr/bin/env python3
"""benchmark_models.py - Data models and utilities for benchmark processing.

This module contains data models, parsing functions, and formatting utilities
for benchmark data processing. It provides the core data structures used
throughout the benchmark infrastructure.
"""

import math
import re
from dataclasses import dataclass
from typing import cast

TIME_UNIT_TO_MICROSECONDS = {
    "ns": 1e-3,
    "µs": 1.0,
    "μs": 1.0,
    "us": 1.0,
    "ms": 1e3,
    "s": 1e6,
}
_CANONICAL_TIME_UNIT = "µs"
_BENCHMARK_HEADER_RE = re.compile(r"^=== (?:(\d+) Points|Unsized Workload) \((.+)\) ===$")
_TIME_LINE_RE = re.compile(r"^Time:\s*\[([^]]*)\]\s+(\S+)\s*$")
_THROUGHPUT_LINE_RE = re.compile(r"^Throughput:\s*\[([^]]*)\]\s+(\S+)\s*$")
_BENCHMARK_ID_RE = re.compile(r"^Benchmark ID:\s*(\S(?:.*\S)?)\s*$")


@dataclass
class BenchmarkData:
    """Represents benchmark data for a single test case."""

    points: int | None
    dimension: str
    time_low: float = 0.0
    time_mean: float = 0.0
    time_high: float = 0.0
    time_unit: str = ""
    throughput_low: float | None = None
    throughput_mean: float | None = None
    throughput_high: float | None = None
    throughput_unit: str | None = None
    benchmark_id: str = ""
    simplices: int | None = None

    @property
    def comparison_key(self) -> str:
        """Return the stable key used for baseline/regression matching."""
        if self.benchmark_id:
            return self.benchmark_id
        if self.points is None:
            msg = "Unsized benchmarks require benchmark_id for comparison matching"
            raise ValueError(msg)
        return f"{self.points}_{self.dimension}"

    @property
    def points_label(self) -> str:
        """Return a display label for the benchmark input size."""
        return str(self.points) if self.points is not None else "n/a"

    def header_line(self) -> str:
        """Return the baseline/comparison section header for this benchmark."""
        if self.points is None:
            return f"=== Unsized Workload ({self.dimension}) ==="
        return f"=== {self.points} Points ({self.dimension}) ==="

    def with_timing(self, low: float, mean: float, high: float, unit: str) -> BenchmarkData:
        """Set timing data (fluent interface)."""
        self.time_low = low
        self.time_mean = mean
        self.time_high = high
        self.time_unit = unit
        return self

    def with_throughput(self, low: float, mean: float, high: float, unit: str) -> BenchmarkData:
        """Set throughput data (fluent interface)."""
        self.throughput_low = low
        self.throughput_mean = mean
        self.throughput_high = high
        self.throughput_unit = unit
        return self

    def with_simplices(self, count: int | None) -> BenchmarkData:
        """Set the number of generated maximal simplices."""
        self.simplices = count
        return self

    def to_baseline_format(self) -> str:
        """Convert to baseline file format."""
        lines = [
            self.header_line(),
        ]
        if self.benchmark_id:
            lines.append(f"Benchmark ID: {self.benchmark_id}")
        lines.append(f"Time: [{self.time_low}, {self.time_mean}, {self.time_high}] {self.time_unit}")

        if self.throughput_low is not None and self.throughput_mean is not None and self.throughput_high is not None and self.throughput_unit:
            lines.append(f"Throughput: [{self.throughput_low}, {self.throughput_mean}, {self.throughput_high}] {self.throughput_unit}")

        lines.append("")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ParsedBenchmarkRecord:
    """One immutable, fully validated legacy benchmark section."""

    points: int | None
    dimension: str
    benchmark_id: str
    time_low_us: float
    time_mean_us: float
    time_high_us: float
    throughput_low: float | None = None
    throughput_mean: float | None = None
    throughput_high: float | None = None
    throughput_unit: str | None = None

    @property
    def comparison_key(self) -> str:
        """Return the section's single validated comparison identity."""
        if self.benchmark_id:
            return self.benchmark_id
        if self.points is None:
            msg = "Unsized benchmarks require exactly one Benchmark ID line"
            raise ValueError(msg)
        return f"{self.points}_{self.dimension}"

    def to_benchmark_data(self) -> BenchmarkData:
        """Convert the trusted parse record to the mutable reporting model."""
        benchmark = BenchmarkData(
            points=self.points,
            dimension=self.dimension,
            benchmark_id=self.benchmark_id,
        ).with_timing(
            self.time_low_us,
            self.time_mean_us,
            self.time_high_us,
            _CANONICAL_TIME_UNIT,
        )
        if self.throughput_low is not None and self.throughput_mean is not None and self.throughput_high is not None and self.throughput_unit is not None:
            benchmark.with_throughput(
                self.throughput_low,
                self.throughput_mean,
                self.throughput_high,
                self.throughput_unit,
            )
        return benchmark


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
    # Match patterns like "=== 1000 Points (2D) ===" or "=== Unsized Workload (4D) ==="
    match = _BENCHMARK_HEADER_RE.match(line.strip())
    if match:
        points = int(match.group(1)) if match.group(1) is not None else None
        if points is not None and points <= 0:
            return None
        dimension = match.group(2).strip()
        if not dimension:
            return None
        return BenchmarkData(points=points, dimension=dimension)
    return None


def _is_valid_positive_interval(values: tuple[float, ...] | list[float]) -> bool:
    """Return whether values form a finite, positive low/mean/high interval."""
    return len(values) == 3 and all(math.isfinite(value) and value > 0.0 for value in values) and values[0] <= values[1] <= values[2]


def _parse_positive_interval(values_text: str, *, label: str) -> tuple[float, float, float]:
    """Parse one finite positive low/mean/high interval without partial state."""
    try:
        values = tuple(float(value.strip()) for value in values_text.split(","))
    except ValueError as exc:
        msg = f"{label} interval contains a non-numeric value: {values_text!r}"
        raise ValueError(msg) from exc
    if not _is_valid_positive_interval(values):
        msg = f"{label} interval must contain three ordered positive finite values: {values_text!r}"
        raise ValueError(msg)
    return values[0], values[1], values[2]


def _parse_normalized_time_line(line: str) -> tuple[float, float, float]:
    """Parse a Time line and normalize its supported unit to microseconds."""
    match = _TIME_LINE_RE.match(line.strip())
    if match is None:
        msg = f"malformed Time line: {line.strip()!r}"
        raise ValueError(msg)
    low, mean, high = _parse_positive_interval(match.group(1), label="Time")
    unit = match.group(2)
    scale = TIME_UNIT_TO_MICROSECONDS.get(unit)
    if scale is None:
        supported = ", ".join(sorted(TIME_UNIT_TO_MICROSECONDS))
        msg = f"unsupported Time unit {unit!r}; expected one of: {supported}"
        raise ValueError(msg)
    normalized = (low * scale, mean * scale, high * scale)
    if not _is_valid_positive_interval(normalized):
        msg = f"Time interval is not finite after normalizing {unit!r} to {_CANONICAL_TIME_UNIT}"
        raise ValueError(msg)
    return normalized


def _parse_throughput_line(line: str) -> tuple[float, float, float, str]:
    """Parse one complete optional Throughput line."""
    match = _THROUGHPUT_LINE_RE.match(line.strip())
    if match is None:
        msg = f"malformed Throughput line: {line.strip()!r}"
        raise ValueError(msg)
    low, mean, high = _parse_positive_interval(match.group(1), label="Throughput")
    return low, mean, high, cast("str", match.group(2))


def parse_time_data(benchmark: BenchmarkData, line: str) -> bool:
    """
    Parse time data lines to extract timing information.

    Args:
        benchmark: BenchmarkData object to update
        line: Input line potentially containing time data

    Returns:
        True if data was parsed successfully, False otherwise
    """
    try:
        low, mean, high = _parse_normalized_time_line(line)
    except ValueError:
        return False
    benchmark.time_low = low
    benchmark.time_mean = mean
    benchmark.time_high = high
    benchmark.time_unit = _CANONICAL_TIME_UNIT
    return True


def parse_throughput_data(benchmark: BenchmarkData, line: str) -> bool:
    """
    Parse throughput data lines to extract throughput information.

    Args:
        benchmark: BenchmarkData object to update
        line: Input line potentially containing throughput data

    Returns:
        True if data was parsed successfully, False otherwise
    """
    try:
        low, mean, high, unit = _parse_throughput_line(line)
    except ValueError:
        return False
    benchmark.throughput_low = low
    benchmark.throughput_mean = mean
    benchmark.throughput_high = high
    benchmark.throughput_unit = unit
    return True


@dataclass(slots=True)
class _PendingBenchmarkSection:
    """Mutable parser state that is never exposed as trusted benchmark data."""

    points: int | None
    dimension: str
    benchmark_id: str | None = None
    timing_us: tuple[float, float, float] | None = None
    throughput: tuple[float, float, float, str] | None = None

    @property
    def label(self) -> str:
        """Return a stable diagnostic label before identity validation completes."""
        if self.benchmark_id is not None:
            return self.benchmark_id
        if self.points is None:
            return f"Unsized Workload ({self.dimension})"
        return f"{self.points} Points ({self.dimension})"

    def finish(self) -> ParsedBenchmarkRecord:
        """Validate the complete section and publish an immutable record."""
        if self.points is None and self.benchmark_id is None:
            msg = f"Malformed baseline section {self.label!r}: missing Benchmark ID line"
            raise ValueError(msg)
        if self.timing_us is None:
            msg = f"Malformed baseline section {self.label!r}: missing or invalid Time line"
            raise ValueError(msg)
        throughput_low: float | None = None
        throughput_mean: float | None = None
        throughput_high: float | None = None
        throughput_unit: str | None = None
        if self.throughput is not None:
            throughput_low, throughput_mean, throughput_high, throughput_unit = self.throughput
        return ParsedBenchmarkRecord(
            points=self.points,
            dimension=self.dimension,
            benchmark_id=self.benchmark_id or "",
            time_low_us=self.timing_us[0],
            time_mean_us=self.timing_us[1],
            time_high_us=self.timing_us[2],
            throughput_low=throughput_low,
            throughput_mean=throughput_mean,
            throughput_high=throughput_high,
            throughput_unit=throughput_unit,
        )


def _consume_section_line(
    section: _PendingBenchmarkSection,
    line: str,
    *,
    line_number: int,
) -> None:
    """Parse one recognized line into untrusted section state."""
    if line.startswith("Benchmark ID"):
        match = _BENCHMARK_ID_RE.match(line)
        if match is None:
            msg = f"Malformed baseline section {section.label!r} at line {line_number}: malformed Benchmark ID line"
            raise ValueError(msg)
        if section.benchmark_id is not None:
            msg = f"Malformed baseline section {section.label!r} at line {line_number}: duplicate Benchmark ID line"
            raise ValueError(msg)
        section.benchmark_id = match.group(1)
    elif line.startswith("Time"):
        if section.timing_us is not None:
            msg = f"Malformed baseline section {section.label!r} at line {line_number}: duplicate Time line"
            raise ValueError(msg)
        try:
            section.timing_us = _parse_normalized_time_line(line)
        except ValueError as exc:
            msg = f"Malformed baseline section {section.label!r} at line {line_number}: {exc}"
            raise ValueError(msg) from exc
    elif line.startswith("Throughput"):
        if section.throughput is not None:
            msg = f"Malformed baseline section {section.label!r} at line {line_number}: duplicate Throughput line"
            raise ValueError(msg)
        try:
            section.throughput = _parse_throughput_line(line)
        except ValueError as exc:
            msg = f"Malformed baseline section {section.label!r} at line {line_number}: {exc}"
            raise ValueError(msg) from exc


def extract_validated_benchmark_records(baseline_content: str) -> tuple[ParsedBenchmarkRecord, ...]:
    """Parse legacy baseline text into unique immutable benchmark records."""
    records: list[ParsedBenchmarkRecord] = []
    current: _PendingBenchmarkSection | None = None

    for line_number, line in enumerate(baseline_content.splitlines(), start=1):
        stripped = line.strip()
        header = parse_benchmark_header(line)
        if header is not None:
            if current is not None:
                records.append(current.finish())
            current = _PendingBenchmarkSection(points=header.points, dimension=header.dimension)
            continue
        if stripped.startswith("==="):
            msg = f"Malformed benchmark section header at line {line_number}: {stripped!r}"
            raise ValueError(msg)
        if current is None:
            continue
        _consume_section_line(current, stripped, line_number=line_number)

    if current is not None:
        records.append(current.finish())

    seen: set[str] = set()
    for record in records:
        if record.comparison_key in seen:
            msg = f"Duplicate benchmark comparison key in baseline: {record.comparison_key!r}"
            raise ValueError(msg)
        seen.add(record.comparison_key)
    return tuple(records)


def extract_benchmark_data(baseline_content: str) -> list[BenchmarkData]:
    """
    Extract benchmark data from baseline file content.

    Args:
        baseline_content: Content from baseline results file

    Returns:
        List of BenchmarkData objects parsed from content
    """
    return [record.to_benchmark_data() for record in extract_validated_benchmark_records(baseline_content)]


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


def format_count_value(value: int | None) -> str:
    """
    Format a count value for benchmark tables.

    Args:
        value: Count value to format (can be None)

    Returns:
        Formatted count string
    """
    if value is None:
        return "N/A"
    return f"{value:,}"


def format_benchmark_tables(
    benchmarks: list[BenchmarkData],
    *,
    input_label: str = "Points",
    include_simplices: bool = False,
) -> list[str]:
    """
    Format benchmark data as markdown tables grouped by dimension.

    Args:
        benchmarks: List of BenchmarkData objects to format
        input_label: Display label for the benchmark input-size column
        include_simplices: When true, replace the scaling column with generated
            maximal simplex counts.

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
        dim_benchmarks = sorted(
            by_dimension[dimension],
            key=lambda b: (b.points is None, b.points or 0, b.comparison_key),
        )
        include_benchmark_id = any(bench.benchmark_id for bench in dim_benchmarks)

        lines.extend([f"### {dimension} Triangulation Performance", ""])
        if include_benchmark_id:
            final_column_label = "Simplices Generated" if include_simplices else "Scaling"
            final_column_separator = "---------------------" if include_simplices else "----------"
            lines.extend(
                [
                    f"| Benchmark ID | {input_label} | Time (mean) | Throughput (mean) | {final_column_label} |",
                    f"|--------------|--------|-------------|-------------------|{final_column_separator}|",
                ],
            )
        else:
            final_column_label = "Simplices Generated" if include_simplices else "Scaling"
            final_column_separator = "---------------------" if include_simplices else "----------"
            lines.extend(
                [
                    f"| {input_label} | Time (mean) | Throughput (mean) | {final_column_label} |",
                    f"|--------|-------------|-------------------|{final_column_separator}|",
                ],
            )

        # Calculate scaling relative to the smallest numeric workload only for
        # legacy homogeneous tables. Expanded benchmark IDs mix different API
        # surfaces, so a single per-dimension scaling baseline is misleading.
        first_nonzero = None if include_benchmark_id else next((b for b in dim_benchmarks if b.time_mean and b.time_mean > 0), None)
        baseline_time = first_nonzero.time_mean if first_nonzero else None

        for bench in dim_benchmarks:
            # Format time and throughput
            time_str = format_time_value(bench.time_mean, bench.time_unit) if bench.time_unit else "N/A"
            throughput_str = (
                format_throughput_value(bench.throughput_mean, bench.throughput_unit) if bench.throughput_unit and bench.throughput_mean is not None else "N/A"
            )

            if include_simplices:
                final_column_str = format_count_value(bench.simplices)
            elif bench.time_mean > 0 and baseline_time and baseline_time > 0:
                scaling = bench.time_mean / baseline_time
                final_column_str = f"{scaling:.1f}x"
            else:
                final_column_str = "N/A"

            if include_benchmark_id:
                lines.append(
                    f"| `{bench.comparison_key}` | {bench.points_label} | {time_str} | {throughput_str} | {final_column_str} |",
                )
            else:
                lines.append(f"| {bench.points_label} | {time_str} | {throughput_str} | {final_column_str} |")

        lines.append("")  # Empty line between tables

    return lines
