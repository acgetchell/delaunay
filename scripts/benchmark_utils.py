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
import logging
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

try:
    # When executed as a script from scripts/
    from hardware_utils import HardwareComparator, HardwareInfo  # type: ignore[no-redef]
    from subprocess_utils import get_git_commit_hash, run_cargo_command, run_git_command  # type: ignore[no-redef]
except ModuleNotFoundError:
    # When imported as a module (e.g., scripts.benchmark_utils)
    from scripts.hardware_utils import HardwareComparator, HardwareInfo  # type: ignore[no-redef]
    from scripts.subprocess_utils import get_git_commit_hash, run_cargo_command, run_git_command  # type: ignore[no-redef]

# Development mode arguments - centralized to keep baseline generation and comparison in sync
# Reduces samples for faster iteration during development (10x faster than full benchmarks)
DEV_MODE_BENCH_ARGS = [
    "--sample-size",
    "10",
    "--measurement-time",
    "2",
    "--warm-up-time",
    "1",
    "--noplot",
]


class ProjectRootNotFoundError(Exception):
    """Raised when project root directory cannot be located."""


# Use the shared secure wrapper from subprocess_utils


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
            # Guard against division by zero for very fast benchmarks
            eps = 1e-9  # µs - minimum time to prevent division by zero
            thrpt_mean = points * 1000 / max(mean_us, eps)
            thrpt_low = points * 1000 / max(high_us, eps)  # Lower time = higher throughput
            thrpt_high = points * 1000 / max(low_us, eps)  # Higher time = lower throughput

            return (
                BenchmarkData(points, dimension)
                # Baseline timing values are rounded to 2 decimal places for consistency
                # This standardizes storage format and avoids spurious precision differences
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

        # Look for benchmark results in *d directories (group names can change)
        for dim_dir in sorted(p for p in criterion_dir.iterdir() if p.is_dir() and p.name.endswith("d")):
            dim = dim_dir.name.removesuffix("d")
            if not dim.isdigit():
                # Fallback: extract trailing "<digits>d"
                m = re.search(r"(\d+)d$", dim_dir.name)
                if not m:
                    continue
                dim = m.group(1)
            # Criterion nests one directory per benchmark target under each *d group
            benchmark_dir = next((p for p in dim_dir.iterdir() if p.is_dir()), None)
            if not benchmark_dir or not benchmark_dir.exists():
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

    def generate_baseline(self, dev_mode: bool = False, output_file: Path | None = None, bench_timeout: int = 1800) -> bool:
        """
        Generate a performance baseline by running benchmarks and parsing results.

        Args:
            dev_mode: Use faster benchmark settings
            output_file: Output file path (default: baseline-artifact/baseline_results.txt)
            bench_timeout: Timeout for cargo bench commands in seconds

        Returns:
            True if successful, False otherwise
        """
        if output_file is None:
            output_file = self.project_root / "baseline-artifact" / "baseline_results.txt"

        try:
            # Clean previous results only for full runs to keep dev mode fast
            if not dev_mode:
                run_cargo_command(["clean"], cwd=self.project_root, timeout=bench_timeout)

            # Run fresh benchmark - using secure subprocess wrapper
            if dev_mode:
                run_cargo_command(
                    ["bench", "--bench", "ci_performance_suite", "--", *DEV_MODE_BENCH_ARGS],
                    cwd=self.project_root,
                    timeout=bench_timeout,
                )
            else:
                run_cargo_command(
                    ["bench", "--bench", "ci_performance_suite"],
                    cwd=self.project_root,
                    timeout=bench_timeout,
                )

            # Parse Criterion results
            target_dir = self.project_root / "target"
            benchmark_results = CriterionParser.find_criterion_results(target_dir)

            if not benchmark_results:
                return False

            # Generate baseline file
            self._write_baseline_file(benchmark_results, output_file)

            return True

        except subprocess.TimeoutExpired:
            print(f"❌ Benchmark execution timed out after {bench_timeout} seconds", file=sys.stderr)
            print("   Consider increasing --bench-timeout or using --dev mode for faster benchmarks", file=sys.stderr)
            return False
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

    def compare_with_baseline(
        self,
        baseline_file: Path,
        dev_mode: bool = False,
        output_file: Path | None = None,
        bench_timeout: int = 1800,
    ) -> tuple[bool, bool]:
        """
        Compare current performance against baseline.

        Args:
            baseline_file: Path to baseline file
            dev_mode: Use faster benchmark settings
            output_file: Output file path (default: benches/compare_results.txt)
            bench_timeout: Timeout for cargo bench commands in seconds

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
                    ["bench", "--bench", "ci_performance_suite", "--", *DEV_MODE_BENCH_ARGS],
                    cwd=self.project_root,
                    timeout=bench_timeout,
                )
            else:
                run_cargo_command(
                    ["bench", "--bench", "ci_performance_suite"],
                    cwd=self.project_root,
                    timeout=bench_timeout,
                )

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

        except subprocess.TimeoutExpired:
            print(f"❌ Benchmark execution timed out after {bench_timeout} seconds", file=sys.stderr)
            print("   Consider increasing --bench-timeout or using --dev mode for faster benchmarks", file=sys.stderr)
            return False, False
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
        """Write performance comparison section and return whether average regression exceeds threshold."""
        time_changes = []  # Track all time changes for average calculation
        individual_regressions = 0

        for current_benchmark in current_results:
            key = f"{current_benchmark.points}_{current_benchmark.dimension}"
            baseline_benchmark = baseline_results.get(key)

            self._write_benchmark_header(f, current_benchmark)
            self._write_current_benchmark_data(f, current_benchmark)

            if baseline_benchmark:
                self._write_baseline_benchmark_data(f, baseline_benchmark)
                time_change, is_individual_regression = self._write_time_comparison(f, current_benchmark, baseline_benchmark)
                if time_change is not None:
                    time_changes.append(time_change)
                    if is_individual_regression:
                        individual_regressions += 1
                self._write_throughput_comparison(f, current_benchmark, baseline_benchmark)
            else:
                f.write("Baseline: N/A (no matching entry)\n")

            f.write("\n")

        # Calculate and report average regression
        if time_changes:
            # Prefer geometric mean of ratios to reflect multiplicative changes across benchmarks
            ratios = [1.0 + (tc / 100.0) for tc in time_changes]
            # Guard against non-positive ratios (defensive; should not occur with sane data)
            positive_ratios = [r for r in ratios if r > 0]
            if not positive_ratios:
                average_change = 0.0
            else:
                avg_log = sum(math.log(r) for r in positive_ratios) / len(positive_ratios)
                avg_ratio = math.exp(avg_log)
                average_change = (avg_ratio - 1.0) * 100.0
            f.write("\n=== SUMMARY ===\n")
            f.write(f"Total benchmarks compared: {len(time_changes)}\n")
            f.write(f"Individual regressions (>{self.regression_threshold}%): {individual_regressions}\n")
            f.write(f"Average time change: {average_change:.1f}%\n")
            # Optional: top regressions
            top = sorted(time_changes, reverse=True)[:5]
            if top:
                f.write("Top regressions (by time change %): " + ", ".join(f"{t:.1f}%" for t in top) + "\n")

            average_regression_found = average_change > self.regression_threshold
            if average_regression_found:
                f.write(
                    f"🚨 OVERALL REGRESSION: Average performance decreased by {average_change:.1f}% "
                    f"(exceeds {self.regression_threshold}% threshold)\n"
                )
            elif average_change < -self.regression_threshold:
                f.write(
                    f"🎉 OVERALL IMPROVEMENT: Average performance improved by {abs(average_change):.1f}% "
                    f"(exceeds {self.regression_threshold}% threshold)\n"
                )
            else:
                f.write(f"✅ OVERALL OK: Average change within acceptable range (±{self.regression_threshold}%)\n")

            f.write("\n")
            return average_regression_found

        return False

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

    def _write_time_comparison(self, f, current: BenchmarkData, baseline: BenchmarkData) -> tuple[float | None, bool]:
        """Write time comparison and return time change percentage and whether individual regression was found."""
        if baseline.time_mean <= 0:
            f.write("Time Change: N/A (baseline mean is 0)\n")
            return None, False
        # Normalize to microseconds when units differ (supports ns, µs/μs/us, ms, s)
        # Note: Both µ (micro sign U+00B5) and μ (Greek mu U+03BC) symbols are supported
        unit_scale = {"ns": 1e-3, "µs": 1.0, "μs": 1.0, "us": 1.0, "ms": 1e3, "s": 1e6}
        cur_unit = current.time_unit or "µs"
        base_unit = baseline.time_unit or "µs"
        if cur_unit not in unit_scale or base_unit not in unit_scale:
            f.write(f"Time Change: N/A (unit mismatch: {cur_unit} vs {base_unit})\n")
            return None, False
        cur_mean_us = current.time_mean * unit_scale[cur_unit]
        base_mean_us = baseline.time_mean * unit_scale[base_unit]
        if base_mean_us <= 0:
            f.write("Time Change: N/A (baseline mean is 0)\n")
            return None, False

        time_change_pct = ((cur_mean_us - base_mean_us) / base_mean_us) * 100
        is_individual_regression = time_change_pct > self.regression_threshold
        if is_individual_regression:
            f.write(f"⚠️  REGRESSION: Time increased by {time_change_pct:.1f}% (slower performance)\n")
        elif time_change_pct < -self.regression_threshold:
            f.write(f"✅ IMPROVEMENT: Time decreased by {abs(time_change_pct):.1f}% (faster performance)\n")
        else:
            f.write(f"✅ OK: Time change {time_change_pct:+.1f}% within acceptable range\n")

        return time_change_pct, is_individual_regression

    def _write_throughput_comparison(self, f, current: BenchmarkData, baseline: BenchmarkData) -> None:
        """Write throughput comparison if data is available."""
        if current.throughput_mean is None or baseline.throughput_mean is None:
            return

        if baseline.throughput_mean <= 0:
            f.write("Throughput Change: N/A (baseline throughput is 0)\n")
        else:
            thrpt_change_pct = ((current.throughput_mean - baseline.throughput_mean) / baseline.throughput_mean) * 100
            f.write(f"Throughput Change (mean): {thrpt_change_pct:.1f}%\n")


class WorkflowHelper:
    """Helper functions for GitHub Actions workflow integration."""

    @staticmethod
    def determine_tag_name() -> str:
        """
        Determine tag name for baseline generation.

        Returns:
            Tag name based on GITHUB_REF or generated timestamp
        """
        github_ref = os.getenv("GITHUB_REF", "")

        if github_ref.startswith("refs/tags/"):
            tag_name = github_ref[len("refs/tags/") :]
            print(f"Using push tag: {tag_name}", file=sys.stderr)
        else:
            # Generate timestamp-based tag
            now = datetime.now(UTC)
            tag_name = f"manual-{now.strftime('%Y%m%d-%H%M%S')}"
            print(f"Using generated tag name: {tag_name}", file=sys.stderr)

        # Set GitHub Actions output if available
        github_output = os.getenv("GITHUB_OUTPUT")
        if github_output:
            with open(github_output, "a", encoding="utf-8") as f:
                f.write(f"tag_name={tag_name}\n")

        print(f"Final tag name: {tag_name}", file=sys.stderr)
        return tag_name

    @staticmethod
    def create_metadata(tag_name: str, output_dir: Path) -> bool:
        """
        Create metadata.json file for baseline artifact.

        Args:
            tag_name: Tag name for this baseline
            output_dir: Directory to write metadata.json

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get required environment variables
            commit_sha = os.getenv("GITHUB_SHA", os.getenv("SAFE_COMMIT_SHA", "unknown"))
            run_id = os.getenv("GITHUB_RUN_ID", os.getenv("SAFE_RUN_ID", "unknown"))
            runner_os = os.getenv("RUNNER_OS", "unknown")
            runner_arch = os.getenv("RUNNER_ARCH", "unknown")

            # Generate current timestamp
            now = datetime.now(UTC)
            generated_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")

            # Create metadata dictionary
            metadata = {
                "tag": tag_name,
                "commit": commit_sha,
                "workflow_run_id": run_id,
                "generated_at": generated_at,
                "runner_os": runner_os,
                "runner_arch": runner_arch,
            }

            # Write metadata file
            output_dir.mkdir(parents=True, exist_ok=True)
            metadata_file = output_dir / "metadata.json"

            with metadata_file.open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            print(f"📦 Created metadata file: {metadata_file}")
            return True

        except Exception as e:
            print(f"❌ Failed to create metadata: {e}", file=sys.stderr)
            return False

    @staticmethod
    def display_baseline_summary(baseline_file: Path) -> bool:
        """
        Display summary information about a baseline file.

        Args:
            baseline_file: Path to baseline file

        Returns:
            True if successful, False otherwise
        """
        try:
            if not baseline_file.exists():
                print(f"❌ Baseline file not found: {baseline_file}", file=sys.stderr)
                return False

            # Show first 10 lines
            print("📊 Baseline summary:")
            with baseline_file.open("r", encoding="utf-8") as f:
                lines = f.readlines()
                for _i, line in enumerate(lines[:10]):
                    print(line.rstrip())

            if len(lines) > 10:
                print("...")

            # Count benchmarks
            benchmark_count = sum(1 for line in lines if line.strip().startswith("==="))
            print(f"Total benchmarks: {benchmark_count}")

            return True

        except Exception as e:
            print(f"❌ Failed to display baseline summary: {e}", file=sys.stderr)
            return False

    @staticmethod
    def sanitize_artifact_name(tag_name: str) -> str:
        """
        Sanitize tag name for GitHub Actions artifact upload.

        Args:
            tag_name: Original tag name

        Returns:
            Sanitized artifact name
        """
        # Replace any non-alphanumeric characters (except . _ -) with underscore
        clean_name = re.sub(r"[^a-zA-Z0-9._-]", "_", tag_name)
        artifact_name = f"performance-baseline-{clean_name}"

        # Set GitHub Actions output if available
        github_output = os.getenv("GITHUB_OUTPUT")
        if github_output:
            with open(github_output, "a", encoding="utf-8") as f:
                f.write(f"artifact_name={artifact_name}\n")

        print(f"Using sanitized artifact name: {artifact_name}", file=sys.stderr)
        return artifact_name

    def generate_summary(self, output_path: Path | None = None, run_benchmarks: bool = False) -> bool:
        """
        Generate performance summary markdown file.

        Args:
            output_path: Output file path (defaults to benches/PERFORMANCE_RESULTS.md)
            run_benchmarks: Whether to run fresh circumsphere benchmarks

        Returns:
            True if successful, False otherwise
        """
        try:
            if output_path is None:
                output_path = self.project_root / "benches" / "PERFORMANCE_RESULTS.md"

            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Optionally run fresh benchmarks
            if run_benchmarks and not self._run_circumsphere_benchmarks():
                print("⚠️ Benchmark run failed, using existing/fallback data")

            # Generate markdown content
            content = self._generate_markdown_content()

            # Write to output file
            with output_path.open("w", encoding="utf-8") as f:
                f.write(content)

            print(f"📊 Generated performance summary: {output_path}")
            return True

        except Exception as e:
            print(f"❌ Failed to generate performance summary: {e}", file=sys.stderr)
            return False

    def _generate_markdown_content(self) -> str:
        """
        Generate the complete markdown content for performance results.

        Returns:
            Formatted markdown content as string
        """
        lines = [
            "# Delaunay Library Performance Results",
            "",
            "This file contains performance benchmarks and analysis for the delaunay library.",
            "The results are automatically generated and updated by the benchmark infrastructure.",
            "",
            f"**Last Updated**: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "**Generated By**: benchmark_utils.py",
        ]

        # Add git information
        try:
            commit_hash = get_git_commit_hash()
            if commit_hash and commit_hash != "unknown":
                lines.append(f"**Git Commit**: {commit_hash}")
        except Exception as e:
            logging.debug("Could not get git commit hash: %s", e)

        lines.extend(
            [
                "",
                "## Performance Results Summary",
                "",
            ]
        )

        # Add circumsphere performance results from actual benchmark data
        lines.extend(self._get_circumsphere_performance_results())

        # Add baseline results if available
        if self.baseline_file.exists():
            lines.extend(self._parse_baseline_results())

        # Add comparison results if available
        if self.comparison_file.exists():
            lines.extend(self._parse_comparison_results())

        # Add dynamic content sections based on performance data
        lines.extend(self._get_dynamic_analysis_sections())

        # Add performance data update instructions
        lines.extend(self._get_update_instructions())

        return "\n".join(lines)

        return "\n".join(lines)

    def _parse_baseline_results(self) -> list[str]:
        """
        Parse baseline results file and format as markdown.

        Returns:
            List of markdown lines
        """
        lines = []
        try:
            with self.baseline_file.open("r", encoding="utf-8") as f:
                content = f.read()

            # Extract metadata from baseline
            metadata_lines = []
            for line in content.split("\n")[:10]:  # Check first 10 lines for metadata
                if line.startswith(("Generated at:", "Git commit:", "Hardware:")):
                    metadata_lines.append(line)

            if metadata_lines:
                lines.extend(
                    [
                        "### Current Baseline Information",
                        "",
                    ]
                )
                for meta_line in metadata_lines:
                    lines.append(f"- **{meta_line}**")
                lines.append("")

            # Parse benchmark data into tables
            benchmark_data = self._extract_benchmark_data(content)
            if benchmark_data:
                lines.extend(self._format_benchmark_tables(benchmark_data))

        except Exception as e:
            lines.extend(
                [
                    "### Baseline Results",
                    "",
                    f"*Error parsing baseline results: {e}*",
                    "",
                ]
            )

        return lines

    def _parse_comparison_results(self) -> list[str]:
        """
        Parse comparison results file and format as markdown.

        Returns:
            List of markdown lines
        """
        lines = []
        try:
            with self.comparison_file.open("r", encoding="utf-8") as f:
                content = f.read()

            if "REGRESSION" in content:
                lines.extend(
                    [
                        "### ⚠️ Performance Regression Detected",
                        "",
                        "Recent benchmark comparison detected performance regressions.",
                        "See comparison details below:",
                        "",
                        "```",
                    ]
                )
                # Include relevant parts of comparison output
                for line in content.split("\n"):
                    if "REGRESSION" in line or "IMPROVEMENT" in line or "OK:" in line:
                        lines.append(line)
                lines.extend(["", "```", ""])
            else:
                lines.extend(
                    [
                        "### ✅ Performance Status: Good",
                        "",
                        "Recent benchmark comparison shows no significant performance regressions.",
                        "",
                    ]
                )

        except Exception as e:
            lines.extend(
                [
                    "### Comparison Results",
                    "",
                    f"*Error parsing comparison results: {e}*",
                    "",
                ]
            )

        return lines

    def _extract_benchmark_data(self, content: str) -> list[BenchmarkData]:
        """
        Extract benchmark data from baseline content.

        Args:
            content: Baseline file content

        Returns:
            List of BenchmarkData objects
        """
        benchmarks = []
        current_benchmark = None

        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Process different line types
            if line.startswith("===") and "Points" in line:
                current_benchmark = self._parse_benchmark_header(line)
            elif current_benchmark and line.startswith("Time:"):
                self._parse_time_data(current_benchmark, line)
            elif current_benchmark and line.startswith("Throughput:") and self._parse_throughput_data(current_benchmark, line):
                # Benchmark is complete, add to list
                benchmarks.append(current_benchmark)
                current_benchmark = None

        return benchmarks

    def _parse_benchmark_header(self, line: str) -> BenchmarkData | None:
        """Parse benchmark header line and return BenchmarkData object."""
        try:
            parts = line.replace("===", "").strip().split()
            if len(parts) >= 3:
                points = int(parts[0])
                dimension = parts[2].strip("()")
                return BenchmarkData(points, dimension)
        except (ValueError, IndexError):
            pass
        return None

    def _parse_time_data(self, benchmark: BenchmarkData, line: str) -> bool:
        """Parse time data line and update benchmark object."""
        try:
            time_part = line.split(":", 1)[1].strip()
            if "]" in time_part:
                values_part, unit = time_part.rsplit("]")
                values_str = values_part.strip("[")
                values = [float(v.strip()) for v in values_str.split(",")]
                unit = unit.strip()

                if len(values) >= 3:
                    benchmark.with_timing(values[0], values[1], values[2], unit)
                    return True
        except (ValueError, IndexError):
            pass
        return False

    def _get_current_version(self) -> str:
        """
        Get the current version from git tags.

        Returns:
            Current version string (e.g., "0.4.3") or "unknown" if not found
        """
        try:
            # Get the latest tag that matches version pattern
            result = run_git_command(["describe", "--tags", "--abbrev=0", "--match=v*"], cwd=self.project_root)
            if result and result.startswith("v"):
                return result[1:]  # Remove 'v' prefix
            return "unknown"
        except Exception:
            # Fallback: try to get any recent tag
            try:
                result = run_git_command(["tag", "-l", "--sort=-version:refname"], cwd=self.project_root)
                if result:
                    tags = result.strip().split("\n")
                    for tag in tags:
                        if tag.startswith("v") and len(tag) > 1:
                            return tag[1:]
                return "unknown"
            except Exception:
                return "unknown"

    def _get_version_date(self) -> str:
        """
        Get the date of the current version tag.

        Returns:
            Date string in YYYY-MM-DD format or current date if not found
        """
        try:
            # Get the date of the latest version tag
            tag_name = f"v{self.current_version}" if self.current_version != "unknown" else None
            if tag_name:
                result = run_git_command(["log", "-1", "--format=%cd", "--date=format:%Y-%m-%d", tag_name], cwd=self.project_root)
                if result:
                    return result.strip()

            # Fallback to current date
            return datetime.now(UTC).strftime("%Y-%m-%d")
        except Exception:
            # Fallback to current date if any error occurs
            return datetime.now(UTC).strftime("%Y-%m-%d")

    def _parse_throughput_data(self, benchmark: BenchmarkData, line: str) -> bool:
        """Parse throughput data line and update benchmark object."""
        try:
            throughput_part = line.split(":", 1)[1].strip()
            if "]" in throughput_part:
                values_part, unit = throughput_part.rsplit("]")
                values_str = values_part.strip("[")
                values = [float(v.strip()) for v in values_str.split(",")]
                unit = unit.strip()

                if len(values) >= 3:
                    benchmark.with_throughput(values[0], values[1], values[2], unit)
                    return True
        except (ValueError, IndexError):
            pass
        return False

    def _format_benchmark_tables(self, benchmarks: list[BenchmarkData]) -> list[str]:
        """
        Format benchmark data as markdown tables.

        Args:
            benchmarks: List of BenchmarkData objects

        Returns:
            List of markdown lines
        """
        lines = []
        if not benchmarks:
            return lines

        # Group by dimension
        by_dimension = {}
        for benchmark in benchmarks:
            dim = benchmark.dimension
            if dim not in by_dimension:
                by_dimension[dim] = []
            by_dimension[dim].append(benchmark)

        # Sort dimensions numerically
        sorted_dims = sorted(by_dimension.keys(), key=lambda x: int(x.rstrip("D")) if x.rstrip("D").isdigit() else float("inf"))

        for dim in sorted_dims:
            dim_benchmarks = sorted(by_dimension[dim], key=lambda x: x.points)

            lines.extend(
                [
                    f"### {dim} Triangulation Performance",
                    "",
                    "| Points | Time (mean) | Throughput (mean) | Scaling |",
                    "|--------|-------------|-------------------|----------|",
                ]
            )

            base_time = None
            for benchmark in dim_benchmarks:
                if base_time is None:
                    base_time = benchmark.time_mean
                    scaling = "1.0x"
                else:
                    scaling = f"{benchmark.time_mean / base_time:.1f}x" if base_time > 0 else "N/A"

                time_str = self._format_time_value(benchmark.time_mean, benchmark.time_unit)
                throughput_str = self._format_throughput_value(benchmark.throughput_mean, benchmark.throughput_unit)

                lines.append(f"| {benchmark.points} | {time_str} | {throughput_str} | {scaling} |")

            lines.append("")

        return lines

    def _format_time_value(self, value: float, unit: str) -> str:
        """
        Format time value with appropriate precision.

        Args:
            value: Time value
            unit: Time unit

        Returns:
            Formatted time string
        """
        if value < 1:
            return f"{value:.3f} {unit}"
        if value < 1000:
            return f"{value:.2f} {unit}"
        # Convert to next unit if too large
        if unit == "µs" and value >= 1000:
            return f"{value / 1000:.3f} ms"
        if unit == "ms" and value >= 1000:
            return f"{value / 1000:.4f} s"
        return f"{value:.1f} {unit}"

    def _format_throughput_value(self, value: float | None, unit: str | None) -> str:
        """
        Format throughput value with appropriate precision.

        Args:
            value: Throughput value (may be None)
            unit: Throughput unit (may be None)

        Returns:
            Formatted throughput string
        """
        if value is None or unit is None:
            return "N/A"

        if value < 1:
            return f"{value:.3f} {unit}"
        if value < 1000:
            return f"{value:.2f} {unit}"
        return f"{value:.3f} {unit}"

    def _get_static_content(self) -> list[str]:
        """
        Get static content sections for the performance results.

        Returns:
            List of markdown lines with static content
        """
        return [
            "## Key Findings",
            "",
            "### Performance Ranking",
            "",
            "1. **insphere_lifted** (fastest) - Consistently best performance across all tests",
            "2. **insphere** (middle) - ~25% slower than lifted, but good performance",
            "3. **insphere_distance** (slowest) - ~2x slower due to explicit circumcenter calculation",
            "",
            "## Recommendations",
            "",
            "### For Performance-Critical Applications",
            "",
            "- **Use `insphere_lifted`** for maximum performance",
            "- ~50% better performance compared to standard method",
            "- Best choice for batch processing and high-frequency queries",
            "",
            "### For Numerical Stability",
            "",
            "- **Use `insphere`** for most reliable results",
            "- Standard determinant-based approach with proven properties",
            "- Good balance of performance and reliability",
            "",
            "### For Educational/Research Purposes",
            "",
            "- **Use `insphere_distance`** to understand geometric intuition",
            "- Explicit circumcenter calculation makes algorithm transparent",
            "- Useful for debugging and validation despite slower performance",
            "",
            "## Performance Data Updates",
            "",
            "This file is automatically generated from benchmark results. To update:",
            "",
            "```bash",
            "# Generate new baseline results",
            "uv run benchmark-utils generate-baseline",
            "",
            "# Update performance summary",
            "uv run benchmark-utils generate-summary",
            "```",
            "",
            "For manual updates or custom analysis, modify the `PerformanceSummaryGenerator`",
            "class in `scripts/benchmark_utils.py`.",
            "",
        ]


class BenchmarkRegressionHelper:
    """Helper functions for performance regression testing workflow."""

    @staticmethod
    def prepare_baseline(baseline_dir: Path) -> bool:
        """
        Prepare baseline for comparison and set environment variables.

        Args:
            baseline_dir: Directory containing baseline artifacts

        Returns:
            True if baseline exists and is valid, False otherwise
        """
        baseline_file = baseline_dir / "baseline_results.txt"

        if baseline_file.exists():
            print("📦 Prepared baseline from artifact")

            # Set GitHub Actions environment variables
            github_env = os.getenv("GITHUB_ENV")
            if github_env:
                with open(github_env, "a", encoding="utf-8") as f:
                    f.write("BASELINE_EXISTS=true\n")
                    f.write("BASELINE_SOURCE=artifact\n")

            # Show baseline metadata
            print("=== Baseline Information (from artifact) ===")
            with baseline_file.open("r", encoding="utf-8") as f:
                for _i, line in enumerate(f.readlines()[:3]):
                    print(line.rstrip())

            return True
        print("❌ Downloaded artifact but no baseline_results.txt found")

        # Set GitHub Actions environment variables
        github_env = os.getenv("GITHUB_ENV")
        if github_env:
            with open(github_env, "a", encoding="utf-8") as f:
                f.write("BASELINE_EXISTS=false\n")
                f.write("BASELINE_SOURCE=missing\n")
                f.write("BASELINE_ORIGIN=unknown\n")

        return False

    @staticmethod
    def set_no_baseline_status() -> None:
        """Set environment variables when no baseline is found."""
        print("📈 No baseline artifact found for performance comparison")

        # Set GitHub Actions environment variables
        github_env = os.getenv("GITHUB_ENV")
        if github_env:
            with open(github_env, "a", encoding="utf-8") as f:
                f.write("BASELINE_EXISTS=false\n")
                f.write("BASELINE_SOURCE=none\n")
                f.write("BASELINE_ORIGIN=none\n")

    @staticmethod
    def extract_baseline_commit(baseline_dir: Path) -> str:
        """
        Extract the baseline commit SHA from baseline files.

        Args:
            baseline_dir: Directory containing baseline artifacts

        Returns:
            Commit SHA string, or "unknown" if not found
        """
        baseline_file = baseline_dir / "baseline_results.txt"
        metadata_file = baseline_dir / "metadata.json"

        commit_sha = "unknown"

        # Try to extract from baseline_results.txt first
        if baseline_file.exists():
            try:
                with baseline_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("Git commit:"):
                            parts = line.strip().split()
                            if len(parts) >= 3:
                                potential_sha = parts[2]
                                # Validate SHA format (7-40 hex characters)
                                if re.match(r"^[0-9A-Fa-f]{7,40}$", potential_sha):
                                    commit_sha = potential_sha
                                    break
            except (OSError, ValueError) as e:
                # Failed to read/parse baseline file - continue with metadata fallback
                logging.debug("Could not extract commit from baseline_results.txt: %s", e)

        # Fallback to metadata.json if needed
        if commit_sha == "unknown" and metadata_file.exists():
            try:
                with metadata_file.open("r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    potential_sha = metadata.get("commit", "")
                    if re.match(r"^[0-9A-Fa-f]{7,40}$", potential_sha):
                        commit_sha = potential_sha
            except (OSError, json.JSONDecodeError, KeyError) as e:
                # Failed to read/parse metadata file - will use "unknown" commit
                logging.debug("Could not extract commit from metadata.json: %s", e)

        # Set GitHub Actions environment variable
        github_env = os.getenv("GITHUB_ENV")
        if github_env:
            with open(github_env, "a", encoding="utf-8") as f:
                f.write(f"BASELINE_COMMIT={commit_sha}\n")

        return commit_sha

    @staticmethod
    def determine_benchmark_skip(baseline_commit: str, current_commit: str) -> tuple[bool, str]:
        """
        Determine if benchmarks should be skipped based on commits and changes.

        Args:
            baseline_commit: SHA of the baseline commit
            current_commit: SHA of the current commit

        Returns:
            Tuple of (should_skip, reason)
        """
        if baseline_commit == "unknown":
            return False, "unknown_baseline"

        if baseline_commit == current_commit:
            return True, "same_commit"

        try:
            # Check if baseline commit exists in git history
            # Validate baseline_commit is a proper SHA (security: prevent injection)
            if not re.match(r"^[0-9A-Fa-f]{6,40}$", baseline_commit):
                return False, "invalid_baseline_sha"

            commit_ref = f"{baseline_commit}^{{commit}}"
            run_git_command(["cat-file", "-e", commit_ref], timeout=60)

            # Check for relevant changes
            diff_range = f"{baseline_commit}..HEAD"
            result = run_git_command(["diff", "--name-only", diff_range], timeout=60)

            relevant_patterns = [r"^src/", r"^benches/", r"^Cargo\.toml$", r"^Cargo\.lock$"]
            changed_files = result.stdout.strip().split("\n") if result.stdout.strip() else []

            has_relevant_changes = any(re.match(pattern, file) for file in changed_files for pattern in relevant_patterns)

            # Return result based on whether changes were detected
            # Future improvement: Consider skipping when HEAD is a merge commit of the same baseline
            # (e.g., when baseline commit is one of the parents of HEAD merge commit)
            return (False, "changes_detected") if has_relevant_changes else (True, "no_relevant_changes")

        except subprocess.CalledProcessError:
            return False, "baseline_commit_not_found"
        except Exception:
            return False, "error_checking_changes"

    @staticmethod
    def display_skip_message(skip_reason: str, baseline_commit: str = "") -> None:
        """
        Display appropriate skip message based on reason.

        Args:
            skip_reason: Reason for skipping benchmarks
            baseline_commit: Baseline commit SHA (if applicable)
        """
        messages = {
            "same_commit": f"🔍 Current commit matches baseline ({baseline_commit}); skipping benchmarks.",
            "no_relevant_changes": f"🔍 No relevant code changes since {baseline_commit}; skipping benchmarks.",
        }

        print(messages.get(skip_reason, "🔍 Benchmarks skipped."))

    @staticmethod
    def display_no_baseline_message() -> None:
        """Display message when no baseline is available."""
        print("⚠️ No performance baseline available for comparison.")
        print("   - No baseline artifacts found in recent workflow runs")
        print("   - Performance regression testing requires a baseline")
        print("")
        print("💡 To enable performance regression testing:")
        print("   1. Create a release tag (e.g., v0.4.3), or")
        print("   2. Manually trigger the 'Generate Performance Baseline' workflow")
        print("   3. Future PRs and pushes will use that baseline for comparison")
        print("   4. Baselines use full benchmark settings for accurate comparisons")

    @staticmethod
    def run_regression_test(baseline_path: Path) -> bool:
        """
        Run performance regression test against baseline.

        Args:
            baseline_path: Path to baseline file

        Returns:
            True if test completed successfully (regardless of regressions), False on error
        """
        try:
            print("🚀 Running performance regression test...")
            print(f"   Using CI performance suite against baseline: {baseline_path}")

            # Use existing PerformanceComparator
            project_root = find_project_root()
            comparator = PerformanceComparator(project_root)
            success, regression_found = comparator.compare_with_baseline(baseline_path)

            if not success:
                print("❌ Performance regression test failed", file=sys.stderr)
                return False

            # Provide feedback about regression results
            if regression_found:
                print("⚠️ Performance regressions detected in benchmark comparison")
            else:
                print("✅ No significant performance regressions detected")

            return True

        except Exception as e:
            print(f"❌ Error running regression test: {e}", file=sys.stderr)
            return False

    @staticmethod
    def display_results(results_file: Path) -> None:
        """
        Display regression test results.

        Args:
            results_file: Path to results file
        """
        if results_file.exists():
            print("=== Performance Regression Test Results ===")
            with results_file.open("r", encoding="utf-8") as f:
                print(f.read())
        else:
            print("⚠️ No comparison results file found")

    @staticmethod
    def generate_summary() -> None:
        """
        Generate final summary of regression testing.
        """
        # Get environment variables
        baseline_source = os.getenv("BASELINE_SOURCE", "none")
        baseline_origin = os.getenv("BASELINE_ORIGIN", "unknown")
        baseline_tag = os.getenv("BASELINE_TAG", "n/a")
        baseline_exists = os.getenv("BASELINE_EXISTS", "false")
        skip_benchmarks = os.getenv("SKIP_BENCHMARKS", "unknown")
        skip_reason = os.getenv("SKIP_REASON", "n/a")

        print("📊 Performance Regression Testing Summary")
        print("===========================================")
        print(f"Baseline source: {baseline_source}")
        print(f"Baseline origin: {baseline_origin}")
        print(f"Baseline tag: {baseline_tag}")
        print(f"Baseline exists: {baseline_exists}")
        print(f"Skip benchmarks: {skip_benchmarks}")
        print(f"Skip reason: {skip_reason}")

        if baseline_exists == "true" and skip_benchmarks == "false":
            results_file = Path("benches/compare_results.txt")
            if results_file.exists():
                with results_file.open("r", encoding="utf-8") as f:
                    content = f.read()
                    if "REGRESSION" in content:
                        print("Result: ⚠️ Performance regressions detected")
                        # Set environment variable for machine consumption by CI systems
                        os.environ["BENCHMARK_REGRESSION_DETECTED"] = "true"
                        # Also export to GITHUB_ENV if available
                        github_env = os.getenv("GITHUB_ENV")
                        if github_env:
                            with open(github_env, "a", encoding="utf-8") as f:
                                f.write("BENCHMARK_REGRESSION_DETECTED=true\n")
                        print("   Exported BENCHMARK_REGRESSION_DETECTED=true for downstream CI steps")
                    else:
                        print("Result: ✅ No significant performance regressions")
            else:
                print("Result: ❓ Benchmark comparison completed but no results file found")
        elif skip_benchmarks == "true":
            skip_messages = {
                "same_commit": "Result: ⏭️ Benchmarks skipped (same commit as baseline)",
                "no_relevant_changes": "Result: ⏭️ Benchmarks skipped (no relevant code changes)",
                "baseline_commit_not_found": "Result: ⚠️ Baseline commit not found in history (force-push/shallow clone?)",
            }
            print(skip_messages.get(skip_reason, "Result: ⏭️ Benchmarks skipped"))
        else:
            print("Result: ⏭️ Benchmarks skipped (no baseline available)")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(description="Benchmark utilities for baseline generation and comparison")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate baseline command
    gen_parser = subparsers.add_parser("generate-baseline", help="Generate performance baseline")
    gen_parser.add_argument("--dev", action="store_true", help="Use development mode with faster benchmark settings")
    gen_parser.add_argument("--output", type=Path, help="Output file path")
    gen_parser.add_argument("--tag", type=str, default=os.getenv("TAG_NAME"), help="Tag name for this baseline (from TAG_NAME env or --tag option)")
    gen_parser.add_argument(
        "--bench-timeout",
        type=int,
        default=int(os.getenv("BENCHMARK_TIMEOUT", "1800")),
        help="Timeout for cargo bench commands in seconds (default: 1800, from BENCHMARK_TIMEOUT env)",
    )

    # Compare benchmarks command
    cmp_parser = subparsers.add_parser("compare", help="Compare current performance against baseline")
    cmp_parser.add_argument("--baseline", type=Path, required=True, help="Path to baseline file")
    cmp_parser.add_argument("--dev", action="store_true", help="Use development mode with faster benchmark settings")
    cmp_parser.add_argument("--output", type=Path, help="Output file path")
    cmp_parser.add_argument(
        "--bench-timeout",
        type=int,
        default=int(os.getenv("BENCHMARK_TIMEOUT", "1800")),
        help="Timeout for cargo bench commands in seconds (default: 1800, from BENCHMARK_TIMEOUT env)",
    )

    # Workflow helper commands
    subparsers.add_parser("determine-tag", help="Determine tag name for baseline generation")

    meta_parser = subparsers.add_parser("create-metadata", help="Create metadata.json file for baseline artifact")
    meta_parser.add_argument("--tag", type=str, required=True, help="Tag name for this baseline")
    meta_parser.add_argument("--output-dir", type=Path, default=Path("baseline-artifact"), help="Output directory for metadata.json")

    summary_parser = subparsers.add_parser("display-summary", help="Display baseline file summary")
    summary_parser.add_argument("--baseline", type=Path, required=True, help="Path to baseline file")

    artifact_parser = subparsers.add_parser("sanitize-artifact-name", help="Sanitize tag name for GitHub Actions artifact")
    artifact_parser.add_argument("--tag", type=str, required=True, help="Tag name to sanitize")

    # Regression testing helper commands
    prepare_parser = subparsers.add_parser("prepare-baseline", help="Prepare baseline for regression testing")
    prepare_parser.add_argument("--baseline-dir", type=Path, default=Path("baseline-artifact"), help="Baseline artifact directory")

    subparsers.add_parser("set-no-baseline", help="Set environment when no baseline found")

    extract_parser = subparsers.add_parser("extract-baseline-commit", help="Extract baseline commit SHA")
    extract_parser.add_argument("--baseline-dir", type=Path, default=Path("baseline-artifact"), help="Baseline artifact directory")

    skip_parser = subparsers.add_parser("determine-skip", help="Determine if benchmarks should be skipped")
    skip_parser.add_argument("--baseline-commit", type=str, required=True, help="Baseline commit SHA")
    skip_parser.add_argument("--current-commit", type=str, required=True, help="Current commit SHA")

    skip_msg_parser = subparsers.add_parser("display-skip-message", help="Display skip message")
    skip_msg_parser.add_argument("--reason", type=str, required=True, help="Skip reason")
    skip_msg_parser.add_argument("--baseline-commit", type=str, help="Baseline commit SHA")

    subparsers.add_parser("display-no-baseline", help="Display no baseline message")

    regress_parser = subparsers.add_parser("run-regression-test", help="Run performance regression test")
    regress_parser.add_argument("--baseline", type=Path, required=True, help="Path to baseline file")

    results_parser = subparsers.add_parser("display-results", help="Display regression test results")
    results_parser.add_argument("--results", type=Path, default=Path("benches/compare_results.txt"), help="Results file path")

    subparsers.add_parser("regression-summary", help="Generate regression testing summary")

    return parser


def find_project_root() -> Path:
    """Find the project root by looking for Cargo.toml.

    Returns:
        Path to project root directory

    Raises:
        ProjectRootNotFoundError: If Cargo.toml cannot be found in any parent directory
    """
    current_dir = Path.cwd()
    project_root = current_dir
    while project_root != project_root.parent:
        if (project_root / "Cargo.toml").exists():
            return project_root
        project_root = project_root.parent
    msg = "Could not locate Cargo.toml to determine project root"
    raise ProjectRootNotFoundError(msg)


def execute_baseline_commands(args: argparse.Namespace, project_root: Path) -> None:
    """Execute baseline generation and comparison commands."""
    if args.command == "generate-baseline":
        generator = BaselineGenerator(project_root, tag=args.tag)
        success = generator.generate_baseline(dev_mode=args.dev, output_file=args.output, bench_timeout=args.bench_timeout)
        sys.exit(0 if success else 1)

    elif args.command == "compare":
        comparator = PerformanceComparator(project_root)
        success, regression_found = comparator.compare_with_baseline(
            args.baseline, dev_mode=args.dev, output_file=args.output, bench_timeout=args.bench_timeout
        )

        if not success:
            sys.exit(1)

        sys.exit(1 if regression_found else 0)


def execute_workflow_commands(args: argparse.Namespace) -> None:
    """Execute workflow helper commands."""
    if args.command == "determine-tag":
        tag_name = WorkflowHelper.determine_tag_name()
        print(tag_name)  # Output tag name to stdout
        sys.exit(0)

    elif args.command == "create-metadata":
        success = WorkflowHelper.create_metadata(args.tag, args.output_dir)
        sys.exit(0 if success else 1)

    elif args.command == "display-summary":
        success = WorkflowHelper.display_baseline_summary(args.baseline)
        sys.exit(0 if success else 1)

    elif args.command == "sanitize-artifact-name":
        artifact_name = WorkflowHelper.sanitize_artifact_name(args.tag)
        print(artifact_name)  # Output sanitized name to stdout
        sys.exit(0)


def execute_regression_commands(args: argparse.Namespace) -> None:
    """Execute regression testing commands."""
    if args.command == "prepare-baseline":
        success = BenchmarkRegressionHelper.prepare_baseline(args.baseline_dir)
        sys.exit(0 if success else 1)

    elif args.command == "set-no-baseline":
        BenchmarkRegressionHelper.set_no_baseline_status()
        sys.exit(0)

    elif args.command == "extract-baseline-commit":
        commit_sha = BenchmarkRegressionHelper.extract_baseline_commit(args.baseline_dir)
        print(commit_sha)  # Output commit SHA to stdout
        sys.exit(0)

    elif args.command == "determine-skip":
        should_skip, reason = BenchmarkRegressionHelper.determine_benchmark_skip(args.baseline_commit, args.current_commit)

        # Set GitHub Actions environment variables
        github_env = os.getenv("GITHUB_ENV")
        if github_env:
            with open(github_env, "a", encoding="utf-8") as f:
                f.write(f"SKIP_BENCHMARKS={'true' if should_skip else 'false'}\n")
                f.write(f"SKIP_REASON={reason}\n")

        print(f"skip={should_skip}")
        print(f"reason={reason}")
        sys.exit(0)

    elif args.command == "display-skip-message":
        BenchmarkRegressionHelper.display_skip_message(args.reason, args.baseline_commit or "")
        sys.exit(0)

    elif args.command == "display-no-baseline":
        BenchmarkRegressionHelper.display_no_baseline_message()
        sys.exit(0)

    elif args.command == "run-regression-test":
        success = BenchmarkRegressionHelper.run_regression_test(args.baseline)
        sys.exit(0 if success else 1)

    elif args.command == "display-results":
        BenchmarkRegressionHelper.display_results(args.results)
        sys.exit(0)

    elif args.command == "regression-summary":
        BenchmarkRegressionHelper.generate_summary()
        sys.exit(0)


def execute_command(args: argparse.Namespace, project_root: Path) -> None:
    """Execute the selected command based on parsed arguments."""
    # Try baseline commands first
    if args.command in ("generate-baseline", "compare"):
        execute_baseline_commands(args, project_root)
        return

    # Try workflow commands
    if args.command in ("determine-tag", "create-metadata", "display-summary", "sanitize-artifact-name"):
        execute_workflow_commands(args)
        return

    # Try regression commands
    if args.command in (
        "prepare-baseline",
        "set-no-baseline",
        "extract-baseline-commit",
        "determine-skip",
        "display-skip-message",
        "display-no-baseline",
        "run-regression-test",
        "display-results",
        "regression-summary",
    ):
        execute_regression_commands(args)
        return


def main():
    """Command-line interface for benchmark utilities."""
    parser = create_argument_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        project_root = find_project_root()
    except ProjectRootNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)

    execute_command(args, project_root)


if __name__ == "__main__":
    main()
