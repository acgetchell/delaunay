#!/usr/bin/env python3
"""
benchmark_utils.py - Benchmark parsing, baseline generation, and performance comparison

This module provides functions for:
- Parsing Criterion benchmark output and JSON data
- Generating performance baselines
- Comparing current performance against baselines
- Detecting performance regressions

Replaces complex bash parsing logic with maintainable Python code.
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

try:
    # When executed as a script from scripts/
    from hardware_utils import HardwareComparator, HardwareInfo  # type: ignore[no-redef]
    from subprocess_utils import get_git_commit_hash, run_cargo_command  # type: ignore[no-redef]
except ModuleNotFoundError:
    # When imported as a module (e.g., scripts.benchmark_utils)
    from scripts.hardware_utils import HardwareComparator, HardwareInfo  # type: ignore[no-redef]
    from scripts.subprocess_utils import get_git_commit_hash, run_cargo_command  # type: ignore[no-redef]


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
        lines = [f"=== {self.points} Points ({self.dimension}) ===", f"Time: [{self.time_low}, {self.time_mean}, {self.time_high}] {self.time_unit}"]

        if self.throughput_mean is not None:
            lines.append(f"Throughput: [{self.throughput_low}, {self.throughput_mean}, {self.throughput_high}] {self.throughput_unit}")

        lines.append("")
        return "\n".join(lines)


class CriterionParser:
    """Parse Criterion benchmark output and JSON data."""

    @staticmethod
    def parse_estimates_json(estimates_path: Path, points: int, dimension: str) -> BenchmarkData | None:
        """
        Parse Criterion estimates.json file to extract benchmark data.

        Args:
            estimates_path: Path to estimates.json file
            points: Number of points in the benchmark
            dimension: Dimension string (e.g., "2D", "3D")

        Returns:
            BenchmarkData object or None if parsing fails
        """
        try:
            with estimates_path.open(encoding="utf-8") as f:
                data = json.load(f)

            # Extract timing data (nanoseconds from Criterion)
            mean_ns = data["mean"]["point_estimate"]
            low_ns = data["mean"]["confidence_interval"]["lower_bound"]
            high_ns = data["mean"]["confidence_interval"]["upper_bound"]

            if mean_ns <= 0:
                return None

            # Convert nanoseconds to microseconds
            mean_us = mean_ns / 1000
            low_us = low_ns / 1000
            high_us = high_ns / 1000

            # Calculate throughput in Kelem/s
            # Throughput = points / time_in_seconds
            # For time in microseconds: throughput = points * 1,000,000 / time_us
            # For Kelem/s: throughput_kelem = (points * 1,000,000 / time_us) / 1000 = points * 1000 / time_us
            thrpt_mean = points * 1000 / mean_us
            thrpt_low = points * 1000 / high_us  # Lower time = higher throughput
            thrpt_high = points * 1000 / low_us  # Higher time = lower throughput

            return (
                BenchmarkData(points, dimension)
                .with_timing(round(low_us, 2), round(mean_us, 2), round(high_us, 2), "µs")
                .with_throughput(round(thrpt_low, 3), round(thrpt_mean, 3), round(thrpt_high, 3), "Kelem/s")
            )

        except (FileNotFoundError, json.JSONDecodeError, KeyError, ZeroDivisionError, ValueError):
            return None

    @staticmethod
    def find_criterion_results(target_dir: Path) -> list[BenchmarkData]:
        """
        Find and parse all Criterion benchmark results.

        Args:
            target_dir: Path to target directory containing Criterion results

        Returns:
            List of BenchmarkData objects sorted by dimension and point count
        """
        results = []
        criterion_dir = target_dir / "criterion"

        if not criterion_dir.exists():
            return results

        # Look for benchmark results in tds_new_*d directories
        for dim in [2, 3, 4]:
            benchmark_dir = criterion_dir / f"tds_new_{dim}d" / "tds_new"

            if not benchmark_dir.exists():
                continue

            # Find point count directories
            for point_dir in benchmark_dir.iterdir():
                if not point_dir.is_dir():
                    continue

                try:
                    point_count = int(point_dir.name)
                except ValueError:
                    continue

                # Look for estimates.json (prefer new/ over base/)
                estimates_file = None
                if (point_dir / "new" / "estimates.json").exists():
                    estimates_file = point_dir / "new" / "estimates.json"
                elif (point_dir / "base" / "estimates.json").exists():
                    estimates_file = point_dir / "base" / "estimates.json"

                if estimates_file:
                    benchmark_data = CriterionParser.parse_estimates_json(estimates_file, point_count, f"{dim}D")
                    if benchmark_data:
                        results.append(benchmark_data)

        # Sort by dimension, then by point count
        results.sort(key=lambda x: (int(x.dimension.rstrip("D")), x.points))
        return results


class BaselineGenerator:
    """Generate performance baselines from benchmark data."""

    def __init__(self, project_root: Path, tag: str | None = None):
        self.project_root = project_root
        self.hardware = HardwareInfo()
        self.tag = tag

    def generate_baseline(self, dev_mode: bool = False, output_file: Path | None = None) -> bool:
        """
        Generate a performance baseline by running benchmarks and parsing results.

        Args:
            dev_mode: Use faster benchmark settings
            output_file: Output file path (default: benches/baseline_results.txt)

        Returns:
            True if successful, False otherwise
        """
        if output_file is None:
            output_file = self.project_root / "benches" / "baseline_results.txt"

        try:
            # Clean previous results only for full runs to keep dev mode fast
            if not dev_mode:
                run_cargo_command(["clean"], cwd=self.project_root)

            # Run fresh benchmark - using secure subprocess wrapper
            if dev_mode:
                run_cargo_command(
                    [
                        "bench",
                        "--bench",
                        "small_scale_triangulation",
                        "--",
                        "--sample-size",
                        "10",
                        "--measurement-time",
                        "2s",
                        "--warm-up-time",
                        "1s",
                        "--noplot",
                    ],
                    cwd=self.project_root,
                )
            else:
                run_cargo_command(["bench", "--bench", "small_scale_triangulation"], cwd=self.project_root)

            # Parse Criterion results
            target_dir = self.project_root / "target"
            benchmark_results = CriterionParser.find_criterion_results(target_dir)

            if not benchmark_results:
                return False

            # Generate baseline file
            self._write_baseline_file(benchmark_results, output_file)

            return True

        except Exception:
            return False

    def _write_baseline_file(self, benchmark_results: list[BenchmarkData], output_file: Path) -> None:
        """Write baseline results to file."""
        # Get current date, git commit, and hardware info
        # Get current date with timezone
        now = datetime.now(UTC).astimezone()
        current_date = now.strftime("%Y-%m-%d %H:%M:%S %Z")

        try:
            # Use secure subprocess wrapper for git command
            git_commit = get_git_commit_hash()
        except Exception:
            git_commit = "unknown"

        hardware_info = self.hardware.format_hardware_info()

        # Write baseline file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            f.write(f"Date: {current_date}\n")
            f.write(f"Git commit: {git_commit}\n")
            if self.tag:
                f.write(f"Tag: {self.tag}\n")
            f.write(hardware_info)

            for benchmark in benchmark_results:
                f.write(benchmark.to_baseline_format())


class PerformanceComparator:
    """Compare current performance against baseline."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.hardware = HardwareInfo()
        self.regression_threshold = 5.0  # 5% regression threshold

    def compare_with_baseline(self, baseline_file: Path, dev_mode: bool = False, output_file: Path | None = None) -> tuple[bool, bool]:
        """
        Compare current performance against baseline.

        Args:
            baseline_file: Path to baseline file
            dev_mode: Use faster benchmark settings
            output_file: Output file path (default: benches/compare_results.txt)

        Returns:
            Tuple of (success, regression_found)
        """
        if output_file is None:
            output_file = self.project_root / "benches" / "compare_results.txt"

        if not baseline_file.exists():
            return False, False

        try:
            # Run fresh benchmark - using secure subprocess wrapper
            if dev_mode:
                run_cargo_command(
                    [
                        "bench",
                        "--bench",
                        "small_scale_triangulation",
                        "--",
                        "--sample-size",
                        "10",
                        "--measurement-time",
                        "2s",
                        "--warm-up-time",
                        "1s",
                        "--noplot",
                    ],
                    cwd=self.project_root,
                )
            else:
                run_cargo_command(["bench", "--bench", "small_scale_triangulation"], cwd=self.project_root)

            # Parse current results
            target_dir = self.project_root / "target"
            current_results = CriterionParser.find_criterion_results(target_dir)

            if not current_results:
                return False, False

            # Parse baseline
            baseline_content = baseline_file.read_text()
            baseline_results = self._parse_baseline_file(baseline_content)

            # Generate comparison report
            regression_found = self._write_comparison_file(current_results, baseline_results, baseline_content, output_file)

            return True, regression_found

        except Exception:
            return False, False

    def _parse_baseline_file(self, baseline_content: str) -> dict[str, BenchmarkData]:
        """Parse baseline file content into benchmark data."""
        results = {}
        lines = baseline_content.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Look for benchmark sections
            match = re.match(r"=== (\d+) Points \((\d+)D\) ===", line)
            if match:
                points = int(match.group(1))
                dimension = f"{match.group(2)}D"

                # Parse time line
                if i + 1 < len(lines):
                    time_line = lines[i + 1].strip()
                    time_match = re.match(r"Time: \[([0-9.]+), ([0-9.]+), ([0-9.]+)\] (.+)", time_line)
                    if time_match:
                        time_low = float(time_match.group(1))
                        time_mean = float(time_match.group(2))
                        time_high = float(time_match.group(3))
                        time_unit = time_match.group(4)

                        # Parse throughput line if present
                        throughput_low = throughput_mean = throughput_high = None
                        throughput_unit = None

                        if i + 2 < len(lines):
                            thrpt_line = lines[i + 2].strip()
                            thrpt_match = re.match(r"Throughput: \[([0-9.]+), ([0-9.]+), ([0-9.]+)\] (.+)", thrpt_line)
                            if thrpt_match:
                                throughput_low = float(thrpt_match.group(1))
                                throughput_mean = float(thrpt_match.group(2))
                                throughput_high = float(thrpt_match.group(3))
                                throughput_unit = thrpt_match.group(4)

                        key = f"{points}_{dimension}"
                        benchmark = BenchmarkData(points, dimension).with_timing(time_low, time_mean, time_high, time_unit)
                        if throughput_mean is not None:
                            benchmark.with_throughput(throughput_low, throughput_mean, throughput_high, throughput_unit)
                        results[key] = benchmark

            i += 1

        return results

    def _write_comparison_file(
        self, current_results: list[BenchmarkData], baseline_results: dict[str, BenchmarkData], baseline_content: str, output_file: Path
    ) -> bool:
        """Write comparison results to file."""
        # Prepare metadata
        metadata = self._prepare_comparison_metadata(baseline_content)

        # Prepare hardware comparison
        hardware_report = self._prepare_hardware_comparison(baseline_content)

        # Write comparison file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            self._write_comparison_header(f, metadata, hardware_report)
            return self._write_performance_comparison(f, current_results, baseline_results)

    def _prepare_comparison_metadata(self, baseline_content: str) -> dict[str, str]:
        """Prepare metadata for comparison report."""
        # Get current date with timezone
        now = datetime.now(UTC).astimezone()
        current_date = now.strftime("%a %b %d %H:%M:%S %Z %Y")

        try:
            git_commit = get_git_commit_hash()
        except Exception:
            git_commit = "unknown"

        # Parse baseline metadata
        baseline_date = "Unknown"
        baseline_commit = "Unknown"

        for line in baseline_content.split("\n"):
            if line.startswith("Date: "):
                baseline_date = line[6:].strip()
            elif line.startswith("Git commit: "):
                baseline_commit = line[12:].strip()

        return {
            "current_date": current_date,
            "current_commit": git_commit,
            "baseline_date": baseline_date,
            "baseline_commit": baseline_commit,
        }

    def _prepare_hardware_comparison(self, baseline_content: str) -> str:
        """Prepare hardware comparison report."""
        current_hardware = self.hardware.get_hardware_info()
        baseline_hardware = HardwareComparator.parse_baseline_hardware(baseline_content)
        hardware_report, _ = HardwareComparator.compare_hardware(current_hardware, baseline_hardware)
        return hardware_report

    def _write_comparison_header(self, f, metadata: dict[str, str], hardware_report: str) -> None:
        """Write the header section of comparison file."""
        f.write("Comparison Results\n")
        f.write("==================\n")
        f.write(f"Current Date: {metadata['current_date']}\n")
        f.write(f"Current Git commit: {metadata['current_commit']}\n\n")
        f.write(f"Baseline Date: {metadata['baseline_date']}\n")
        f.write(f"Baseline Git commit: {metadata['baseline_commit']}\n\n")
        f.write(hardware_report)

    def _write_performance_comparison(self, f, current_results: list[BenchmarkData], baseline_results: dict[str, BenchmarkData]) -> bool:
        """Write performance comparison section and return whether regression was found."""
        regression_found = False

        for current_benchmark in current_results:
            key = f"{current_benchmark.points}_{current_benchmark.dimension}"
            baseline_benchmark = baseline_results.get(key)

            self._write_benchmark_header(f, current_benchmark)
            self._write_current_benchmark_data(f, current_benchmark)

            if baseline_benchmark:
                self._write_baseline_benchmark_data(f, baseline_benchmark)
                if self._write_time_comparison(f, current_benchmark, baseline_benchmark):
                    regression_found = True
                self._write_throughput_comparison(f, current_benchmark, baseline_benchmark)

            f.write("\n")

        return regression_found

    def _write_benchmark_header(self, f, benchmark: BenchmarkData) -> None:
        """Write benchmark section header."""
        f.write(f"=== {benchmark.points} Points ({benchmark.dimension}) ===\n")

    def _write_current_benchmark_data(self, f, benchmark: BenchmarkData) -> None:
        """Write current benchmark data."""
        f.write(f"Current Time: [{benchmark.time_low}, {benchmark.time_mean}, {benchmark.time_high}] {benchmark.time_unit}\n")
        if benchmark.throughput_mean is not None:
            f.write(
                f"Current Throughput: [{benchmark.throughput_low}, {benchmark.throughput_mean}, "
                f"{benchmark.throughput_high}] {benchmark.throughput_unit}\n"
            )

    def _write_baseline_benchmark_data(self, f, benchmark: BenchmarkData) -> None:
        """Write baseline benchmark data."""
        f.write(f"Baseline Time: [{benchmark.time_low}, {benchmark.time_mean}, {benchmark.time_high}] {benchmark.time_unit}\n")
        if benchmark.throughput_mean is not None:
            f.write(
                f"Baseline Throughput: [{benchmark.throughput_low}, {benchmark.throughput_mean}, "
                f"{benchmark.throughput_high}] {benchmark.throughput_unit}\n"
            )

    def _write_time_comparison(self, f, current: BenchmarkData, baseline: BenchmarkData) -> bool:
        """Write time comparison and return whether regression was found."""
        if baseline.time_mean <= 0:
            f.write("Time Change: N/A (baseline mean is 0)\n")
            return False

        time_change_pct = ((current.time_mean - baseline.time_mean) / baseline.time_mean) * 100
        f.write(f"Time Change (mean): {time_change_pct:.1f}%\n")

        if time_change_pct > self.regression_threshold:
            f.write(f"⚠️  REGRESSION: Time increased by {time_change_pct:.1f}% (slower performance)\n")
            return True
        if time_change_pct < -self.regression_threshold:
            f.write(f"✅ IMPROVEMENT: Time decreased by {abs(time_change_pct):.1f}% (faster performance)\n")
        else:
            f.write("✅ OK: Time change within acceptable range\n")

        return False

    def _write_throughput_comparison(self, f, current: BenchmarkData, baseline: BenchmarkData) -> None:
        """Write throughput comparison if data is available."""
        if current.throughput_mean is None or baseline.throughput_mean is None:
            return

        if baseline.throughput_mean <= 0:
            f.write("Throughput Change: N/A (baseline throughput is 0)\n")
        else:
            thrpt_change_pct = ((current.throughput_mean - baseline.throughput_mean) / baseline.throughput_mean) * 100
            f.write(f"Throughput Change (mean): {thrpt_change_pct:.1f}%\n")


def main():
    """Command-line interface for benchmark utilities."""
    parser = argparse.ArgumentParser(description="Benchmark utilities for baseline generation and comparison")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate baseline command
    gen_parser = subparsers.add_parser("generate-baseline", help="Generate performance baseline")
    gen_parser.add_argument("--dev", action="store_true", help="Use development mode with faster benchmark settings")
    gen_parser.add_argument("--output", type=Path, help="Output file path")
    gen_parser.add_argument("--tag", type=str, default=os.getenv("TAG_NAME"), help="Tag name for this baseline (from TAG_NAME env or --tag option)")

    # Compare benchmarks command
    cmp_parser = subparsers.add_parser("compare", help="Compare current performance against baseline")
    cmp_parser.add_argument("--baseline", type=Path, required=True, help="Path to baseline file")
    cmp_parser.add_argument("--dev", action="store_true", help="Use development mode with faster benchmark settings")
    cmp_parser.add_argument("--output", type=Path, help="Output file path")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Find project root
    current_dir = Path.cwd()
    project_root = current_dir
    while project_root != project_root.parent:
        if (project_root / "Cargo.toml").exists():
            break
        project_root = project_root.parent
    else:
        print("error: could not locate Cargo.toml to determine project root", file=sys.stderr)
        sys.exit(2)

    if args.command == "generate-baseline":
        generator = BaselineGenerator(project_root, tag=args.tag)
        success = generator.generate_baseline(dev_mode=args.dev, output_file=args.output)
        sys.exit(0 if success else 1)

    elif args.command == "compare":
        comparator = PerformanceComparator(project_root)
        success, regression_found = comparator.compare_with_baseline(args.baseline, dev_mode=args.dev, output_file=args.output)

        if not success:
            sys.exit(1)

        if regression_found:
            sys.exit(1)
        else:
            sys.exit(0)


if __name__ == "__main__":
    main()
