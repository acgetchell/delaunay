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
from datetime import UTC, datetime
from pathlib import Path

# NOTE: Use copy2 (metadata-preserving) under the 'copyfile' alias for tests/patching convenience.
from shutil import copy2 as copyfile
from uuid import uuid4

from packaging.version import Version

try:
    # When executed as a script from scripts/
    from benchmark_models import (  # type: ignore[no-redef]
        BenchmarkData,
        CircumspherePerformanceData,
        CircumsphereTestCase,
        extract_benchmark_data,
        format_benchmark_tables,
    )
    from hardware_utils import HardwareComparator, HardwareInfo  # type: ignore[no-redef]
    from subprocess_utils import (  # type: ignore[no-redef]
        ProjectRootNotFoundError,
        find_project_root,
        get_git_commit_hash,
        run_cargo_command,
        run_git_command,
    )
except ModuleNotFoundError:
    # When imported as a module (e.g., scripts.benchmark_utils)
    from scripts.benchmark_models import (  # type: ignore[no-redef]
        BenchmarkData,
        CircumspherePerformanceData,
        CircumsphereTestCase,
        extract_benchmark_data,
        format_benchmark_tables,
    )
    from scripts.hardware_utils import HardwareComparator, HardwareInfo  # type: ignore[no-redef]
    from scripts.subprocess_utils import (  # type: ignore[no-redef]
        ProjectRootNotFoundError,
        find_project_root,
        get_git_commit_hash,
        run_cargo_command,
        run_git_command,
    )

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


# Use the shared secure wrapper from subprocess_utils
# ProjectRootNotFoundError and find_project_root are imported from subprocess_utils


# =============================================================================
# PERFORMANCE SUMMARY GENERATOR
# =============================================================================


class PerformanceSummaryGenerator:
    """Generate performance summary markdown from benchmark results."""

    def __init__(self, project_root: Path):
        """Initialize with project root directory."""
        self.project_root = project_root
        # Prefer CI artifact location; fall back to benches/ for local runs
        self.baseline_file = project_root / "baseline-artifact" / "baseline_results.txt"
        self._baseline_fallback = project_root / "benches" / "baseline_results.txt"
        self.comparison_file = project_root / "benches" / "compare_results.txt"

        # Path for storing circumsphere benchmark results
        self.circumsphere_results_dir = project_root / "target" / "criterion"

        # Storage for numerical accuracy data from benchmarks
        self.numerical_accuracy_data = None

        # Extract current version and date information
        self.current_version = self._get_current_version()
        self.current_date = self._get_version_date()

    def generate_summary(self, output_path: Path | None = None, run_benchmarks: bool = False, generator_name: str | None = None) -> bool:
        """
        Generate performance summary markdown file.

        Args:
            output_path: Output file path (defaults to benches/PERFORMANCE_RESULTS.md)
            run_benchmarks: Whether to run fresh circumsphere benchmarks
            generator_name: Name of the tool generating the summary (for attribution)

        Returns:
            True if successful, False otherwise
        """
        try:
            if output_path is None:
                output_path = self.project_root / "benches" / "PERFORMANCE_RESULTS.md"

            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Optionally run fresh benchmarks
            if run_benchmarks:
                success, accuracy_data = self._run_circumsphere_benchmarks()
                if success:
                    self.numerical_accuracy_data = accuracy_data
                else:
                    print("⚠️ Benchmark run failed, using existing/fallback data")

            # Generate markdown content
            content = self._generate_markdown_content(generator_name)

            # Write to output file
            with output_path.open("w", encoding="utf-8") as f:
                f.write(content)

            print(f"📊 Generated performance summary: {output_path}")
            return True

        except Exception as e:
            print(f"❌ Failed to generate performance summary: {e}", file=sys.stderr)
            return False

    def _generate_markdown_content(self, generator_name: str | None = None) -> str:
        """
        Generate the complete markdown content for performance results.

        Args:
            generator_name: Name of the tool generating the summary (for attribution)

        Returns:
            Formatted markdown content as string
        """
        # Determine the generator name for attribution
        if generator_name is None:
            generator_name = "benchmark_utils.py"

        lines = [
            "# Delaunay Library Performance Results",
            "",
            "This file contains performance benchmarks and analysis for the delaunay library.",
            "The results are automatically generated and updated by the benchmark infrastructure.",
            "",
            f"**Last Updated**: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Generated By**: {generator_name}",
        ]

        # Add git information
        try:
            commit_hash = get_git_commit_hash()
            if commit_hash and commit_hash != "unknown":
                lines.append(f"**Git Commit**: {commit_hash}")
        except Exception as e:
            logging.debug("Could not get git commit hash: %s", e)

        # Add hardware information
        try:
            hardware_info = HardwareInfo()
            hw_info = hardware_info.get_hardware_info()
            lines.extend(
                [
                    f"**Hardware**: {hw_info['CPU']} ({hw_info['CPU_CORES']} cores)",
                    f"**Memory**: {hw_info['MEMORY']}",
                    f"**OS**: {hw_info['OS']}",
                    f"**Rust**: {hw_info['RUST']}",
                ],
            )
        except Exception as e:
            logging.debug("Could not get hardware info: %s", e)
            lines.append("**Hardware**: Unknown")

        lines.extend(
            [
                "",
                "## Performance Results Summary",
                "",
            ],
        )

        # Add circumsphere performance results from actual benchmark data
        lines.extend(self._get_circumsphere_performance_results())

        # Add baseline results if available
        if self.baseline_file.exists() or self._baseline_fallback.exists():
            # Use fallback if primary is missing
            if not self.baseline_file.exists():
                self.baseline_file = self._baseline_fallback
            lines.extend(self._parse_baseline_results())

        # Add comparison results if available
        if self.comparison_file.exists():
            lines.extend(self._parse_comparison_results())

        # Add dynamic analysis sections based on performance data
        lines.extend(self._get_dynamic_analysis_sections())

        # Add static content sections (moved to end)
        lines.extend(self._get_static_sections())

        # Add performance data update instructions
        lines.extend(self._get_update_instructions())

        return "\n".join(lines)

    def _get_current_version(self) -> str:
        """
        Get the current version from git tags.

        Returns:
            Current version string (e.g., "0.4.3") or "unknown" if not found
        """
        try:
            # Get the latest tag that matches version pattern
            cp = run_git_command(["describe", "--tags", "--abbrev=0", "--match=v*"], cwd=self.project_root)
            result = cp.stdout.strip()
            if result.startswith("v"):
                return result[1:]  # Remove 'v' prefix
            return "unknown"
        except Exception:
            # Fallback: try to get any recent tag
            try:
                cp = run_git_command(["tag", "-l", "--sort=-version:refname"], cwd=self.project_root)
                out = cp.stdout.strip()
                if out:
                    tags = out.split("\n")
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
                cp = run_git_command(["log", "-1", "--format=%cd", "--date=format:%Y-%m-%d", tag_name], cwd=self.project_root)
                log_output = cp.stdout.strip()
                if log_output:
                    return log_output

            # Fallback to current date
            return datetime.now(UTC).strftime("%Y-%m-%d")
        except Exception:
            return datetime.now(UTC).strftime("%Y-%m-%d")

    def _run_circumsphere_benchmarks(self) -> tuple[bool, dict[str, str] | None]:
        """
        Run the circumsphere containment benchmarks to generate fresh data.

        Returns:
            Tuple of (success, numerical_accuracy_data)
        """
        try:
            print("🔄 Running circumsphere containment benchmarks...")

            # Run the circumsphere benchmark with reduced sample size for speed
            result = run_cargo_command(
                ["bench", "--bench", "circumsphere_containment", "--", *DEV_MODE_BENCH_ARGS],
                cwd=self.project_root,
                timeout=240,  # 4 minute timeout for quick benchmarks
                capture_output=True,
            )

            # Parse numerical accuracy data from stdout
            numerical_accuracy_data = self._parse_numerical_accuracy_output(result.stdout)

            print("✅ Circumsphere benchmarks completed successfully")
            return True, numerical_accuracy_data

        except Exception as e:
            print(f"❌ Error running circumsphere benchmarks: {e}")
            return False, None

    def _parse_numerical_accuracy_output(self, stdout: str) -> dict[str, str] | None:
        """
        Parse numerical accuracy data from circumsphere benchmark stdout.

        Args:
            stdout: The stdout output from the circumsphere benchmark

        Returns:
            Dictionary with accuracy percentages or None if parsing failed
        """
        try:
            lines = stdout.split("\n")
            accuracy_data = {}

            # Look for the Method Comparisons section
            for i, line in enumerate(lines):
                if "Method Comparisons" in line and "total tests" in line:
                    # Parse the following lines for accuracy percentages
                    # Expected format:
                    # "  insphere vs insphere_distance:  1000/1000 (100.00%)"
                    patterns = [
                        (r"insphere vs insphere_distance:\s+\d+/\d+\s+\(([\d.]+)%\)", "insphere_distance"),
                        (r"insphere vs insphere_lifted:\s+\d+/\d+\s+\(([\d.]+)%\)", "insphere_lifted"),
                        (r"insphere_distance vs insphere_lifted:\s+\d+/\d+\s+\(([\d.]+)%\)", "distance_lifted"),
                        (r"All three methods agree:\s+\d+/\d+\s+\(([\d.]+)%\)", "all_agree"),
                    ]

                    # Look at the next several lines for the percentages
                    for j in range(i + 1, min(i + 6, len(lines))):
                        check_line = lines[j]
                        for pattern, key in patterns:
                            match = re.search(pattern, check_line)
                            if match:
                                accuracy_data[key] = f"{float(match.group(1)):.1f}%"
                    break

            return accuracy_data if accuracy_data else None

        except Exception:
            return None

    def _get_numerical_accuracy_analysis(self) -> list[str]:
        """
        Generate numerical accuracy analysis section using dynamic data if available.

        Returns:
            List of markdown lines with numerical accuracy analysis
        """
        lines = [
            "",
            "### Numerical Accuracy Analysis",
            "",
            "Based on random test cases:",
            "",
        ]

        if self.numerical_accuracy_data:
            # Use actual dynamic data from benchmark runs
            insphere_distance = self.numerical_accuracy_data.get("insphere_distance", "unknown")
            insphere_lifted = self.numerical_accuracy_data.get("insphere_lifted", "unknown")
            distance_lifted = self.numerical_accuracy_data.get("distance_lifted", "unknown")
            all_agree = self.numerical_accuracy_data.get("all_agree", "unknown")

            lines.extend(
                [
                    f"- **insphere vs insphere_distance**: {insphere_distance} agreement",
                    f"- **insphere vs insphere_lifted**: {insphere_lifted} agreement (different algorithms)",
                    f"- **insphere_distance vs insphere_lifted**: {distance_lifted} agreement",
                    f"- **All three methods agree**: {all_agree} (expected due to different numerical approaches)",
                ],
            )
        else:
            # Use reference data when no fresh benchmark data is available
            lines.extend(
                [
                    "- **insphere vs insphere_distance**: ~82% agreement (reference data)",
                    "- **insphere vs insphere_lifted**: ~0% agreement (different algorithms, reference data)",
                    "- **insphere_distance vs insphere_lifted**: ~18% agreement (reference data)",
                    "- **All three methods agree**: ~0% (expected due to different numerical approaches, reference data)",
                    "",
                    "*Note: To get current numerical accuracy data, run with `--run-benchmarks` flag.*",
                ],
            )

        lines.append("")
        return lines

    def _parse_circumsphere_benchmark_results(self) -> list[CircumsphereTestCase]:
        """
        Parse circumsphere benchmark results from Criterion output.

        Returns:
            List of CircumsphereTestCase objects with parsed performance data
        """
        if not self.circumsphere_results_dir.exists():
            print(f"⚠️ No criterion results found at {self.circumsphere_results_dir}")
            return self._get_fallback_circumsphere_data()

        benchmark_mappings, edge_case_mappings, method_mappings, edge_method_mappings = self._get_benchmark_mappings()

        test_cases = []
        test_cases.extend(self._parse_regular_benchmarks(benchmark_mappings, method_mappings))
        test_cases.extend(self._parse_edge_case_benchmarks(edge_case_mappings, edge_method_mappings))

        # If no results were parsed, use fallback data
        if not test_cases:
            print("⚠️ No benchmark results parsed, using fallback data")
            return self._get_fallback_circumsphere_data()

        return test_cases

    def _get_benchmark_mappings(self) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[str, str]], dict[str, str], dict[str, str]]:
        """
        Get the mapping configurations for parsing benchmark results.

        Returns:
            Tuple of (benchmark_mappings, edge_case_mappings, method_mappings, edge_method_mappings)
        """
        benchmark_mappings = {
            "2d": ("Basic 2D", "2D"),
            "3d": ("Basic 3D", "3D"),
            "4d": ("Basic 4D", "4D"),
            "5d": ("Basic 5D", "5D"),
        }

        edge_case_mappings = {
            "edge_cases_2d_boundary_point": ("Boundary vertex", "2D"),
            "edge_cases_2d_far_point": ("Far vertex", "2D"),
            "edge_cases_3d_boundary_point": ("Boundary vertex", "3D"),
            "edge_cases_3d_far_point": ("Far vertex", "3D"),
            "edge_cases_4d_boundary_point": ("Boundary vertex", "4D"),
            "edge_cases_4d_far_point": ("Far vertex", "4D"),
            "edge_cases_5d_boundary_point": ("Boundary vertex", "5D"),
            "edge_cases_5d_far_point": ("Far vertex", "5D"),
        }

        method_mappings = {
            "insphere": "insphere",
            "insphere_distance": "insphere_distance",
            "insphere_lifted": "insphere_lifted",
        }

        edge_method_mappings = {
            "insphere": "insphere",
            "distance": "insphere_distance",
            "lifted": "insphere_lifted",
        }

        return benchmark_mappings, edge_case_mappings, method_mappings, edge_method_mappings

    def _parse_regular_benchmarks(
        self,
        benchmark_mappings: dict[str, tuple[str, str]],
        method_mappings: dict[str, str],
    ) -> list[CircumsphereTestCase]:
        """
        Parse regular benchmark results.

        Args:
            benchmark_mappings: Mapping of benchmark keys to (test_name, dimension)
            method_mappings: Mapping of method suffixes to method names

        Returns:
            List of parsed CircumsphereTestCase objects
        """
        test_cases = []

        for bench_key, (test_name, dimension) in benchmark_mappings.items():
            methods = self._parse_benchmark_methods(bench_key, method_mappings)

            if methods:
                test_case = CircumsphereTestCase(test_name=test_name, dimension=dimension, methods=methods)
                test_cases.append(test_case)

        return test_cases

    def _parse_edge_case_benchmarks(
        self,
        edge_case_mappings: dict[str, tuple[str, str]],
        edge_method_mappings: dict[str, str],
    ) -> list[CircumsphereTestCase]:
        """
        Parse edge case benchmark results.

        Args:
            edge_case_mappings: Mapping of edge case keys to (test_name, dimension)
            edge_method_mappings: Mapping of edge case method suffixes to method names

        Returns:
            List of parsed CircumsphereTestCase objects
        """
        test_cases = []

        for edge_key, (test_name, dimension) in edge_case_mappings.items():
            methods = self._parse_benchmark_methods(edge_key, edge_method_mappings)

            if methods:
                test_case = CircumsphereTestCase(test_name=test_name, dimension=dimension, methods=methods)
                test_cases.append(test_case)

        return test_cases

    def _parse_benchmark_methods(self, bench_key: str, method_mappings: dict[str, str]) -> dict[str, CircumspherePerformanceData]:
        """
        Parse methods for a single benchmark.

        Args:
            bench_key: The benchmark key (e.g., "2d" or "edge_cases_2d_boundary_point")
            method_mappings: Mapping of method suffixes to method names

        Returns:
            Dictionary mapping method names to CircumspherePerformanceData
        """
        methods = {}

        for method_suffix, method_name in method_mappings.items():
            criterion_path = self.circumsphere_results_dir / f"{bench_key}_{method_suffix}"
            performance_data = self._parse_single_method_result(criterion_path, method_name)

            if performance_data:
                methods[method_name] = performance_data

        return methods

    def _parse_single_method_result(self, criterion_path: Path, method_name: str) -> CircumspherePerformanceData | None:
        """
        Parse a single method result from Criterion output.

        Args:
            criterion_path: Path to the Criterion benchmark directory
            method_name: Name of the method being benchmarked

        Returns:
            CircumspherePerformanceData object or None if parsing failed
        """
        estimates_file = criterion_path / "base" / "estimates.json"
        if not estimates_file.exists():
            estimates_file = criterion_path / "new" / "estimates.json"

        if estimates_file.exists():
            try:
                with estimates_file.open(encoding="utf-8") as f:
                    estimates = json.load(f)

                # Extract mean time in nanoseconds
                mean_ns = estimates["mean"]["point_estimate"]
                return CircumspherePerformanceData(method=method_name, time_ns=mean_ns)

            except Exception as e:
                print(f"⚠️ Could not parse {estimates_file}: {e}")

        return None

    def _get_fallback_circumsphere_data(self) -> list[CircumsphereTestCase]:
        """
        Get fallback circumsphere performance data when live benchmarks aren't available.

        Returns:
            List of CircumsphereTestCase objects with known performance data
        """
        return [
            # 2D results
            CircumsphereTestCase(
                "Basic 2D",
                "2D",
                {
                    "insphere": CircumspherePerformanceData("insphere", 560),
                    "insphere_distance": CircumspherePerformanceData("insphere_distance", 644),
                    "insphere_lifted": CircumspherePerformanceData("insphere_lifted", 448),
                },
            ),
            CircumsphereTestCase(
                "Boundary vertex",
                "2D",
                {
                    "insphere": CircumspherePerformanceData("insphere", 570),
                    "insphere_distance": CircumspherePerformanceData("insphere_distance", 644),
                    "insphere_lifted": CircumspherePerformanceData("insphere_lifted", 451),
                },
            ),
            CircumsphereTestCase(
                "Far vertex",
                "2D",
                {
                    "insphere": CircumspherePerformanceData("insphere", 570),
                    "insphere_distance": CircumspherePerformanceData("insphere_distance", 641),
                    "insphere_lifted": CircumspherePerformanceData("insphere_lifted", 449),
                },
            ),
            # 3D results
            CircumsphereTestCase(
                "Basic 3D",
                "3D",
                {
                    "insphere": CircumspherePerformanceData("insphere", 805),
                    "insphere_distance": CircumspherePerformanceData("insphere_distance", 1463),
                    "insphere_lifted": CircumspherePerformanceData("insphere_lifted", 637),
                },
            ),
            CircumsphereTestCase(
                "Boundary vertex",
                "3D",
                {
                    "insphere": CircumspherePerformanceData("insphere", 811),
                    "insphere_distance": CircumspherePerformanceData("insphere_distance", 1497),
                    "insphere_lifted": CircumspherePerformanceData("insphere_lifted", 647),
                },
            ),
            CircumsphereTestCase(
                "Far vertex",
                "3D",
                {
                    "insphere": CircumspherePerformanceData("insphere", 808),
                    "insphere_distance": CircumspherePerformanceData("insphere_distance", 1493),
                    "insphere_lifted": CircumspherePerformanceData("insphere_lifted", 649),
                },
            ),
            # 4D results
            CircumsphereTestCase(
                "Basic 4D",
                "4D",
                {
                    "insphere": CircumspherePerformanceData("insphere", 1200),
                    "insphere_distance": CircumspherePerformanceData("insphere_distance", 1900),
                    "insphere_lifted": CircumspherePerformanceData("insphere_lifted", 979),
                },
            ),
            CircumsphereTestCase(
                "Boundary vertex",
                "4D",
                {
                    "insphere": CircumspherePerformanceData("insphere", 1300),
                    "insphere_distance": CircumspherePerformanceData("insphere_distance", 1900),
                    "insphere_lifted": CircumspherePerformanceData("insphere_lifted", 987),
                },
            ),
            CircumsphereTestCase(
                "Far vertex",
                "4D",
                {
                    "insphere": CircumspherePerformanceData("insphere", 1300),
                    "insphere_distance": CircumspherePerformanceData("insphere_distance", 1900),
                    "insphere_lifted": CircumspherePerformanceData("insphere_lifted", 975),
                },
            ),
            # 5D results
            CircumsphereTestCase(
                "Basic 5D",
                "5D",
                {
                    "insphere": CircumspherePerformanceData("insphere", 1800),
                    "insphere_distance": CircumspherePerformanceData("insphere_distance", 3000),
                    "insphere_lifted": CircumspherePerformanceData("insphere_lifted", 1500),
                },
            ),
            CircumsphereTestCase(
                "Boundary vertex",
                "5D",
                {
                    "insphere": CircumspherePerformanceData("insphere", 1800),
                    "insphere_distance": CircumspherePerformanceData("insphere_distance", 3100),
                    "insphere_lifted": CircumspherePerformanceData("insphere_lifted", 1500),
                },
            ),
            CircumsphereTestCase(
                "Far vertex",
                "5D",
                {
                    "insphere": CircumspherePerformanceData("insphere", 1800),
                    "insphere_distance": CircumspherePerformanceData("insphere_distance", 3000),
                    "insphere_lifted": CircumspherePerformanceData("insphere_lifted", 1500),
                },
            ),
        ]

    def _get_circumsphere_performance_results(self) -> list[str]:
        """
        Generate circumsphere containment performance results section with dynamic data.

        Returns:
            List of markdown lines with circumsphere performance data
        """
        # Parse actual benchmark results
        test_cases = self._parse_circumsphere_benchmark_results()

        if not test_cases:
            return [
                "### Circumsphere Performance Results",
                "",
                f"#### Version {self.current_version} Results ({self.current_date})",
                "",
                "⚠️ No benchmark results available. Run benchmarks first:",
                "```bash",
                "uv run benchmark-utils generate-summary --run-benchmarks",
                "```",
                "",
            ]

        lines = [
            "### Circumsphere Performance Results",
            "",
            f"#### Version {self.current_version} Results ({self.current_date})",
            "",
        ]

        # Group test cases by dimension for better organization
        cases_by_dimension = {}
        for test_case in test_cases:
            dim = test_case.dimension
            if dim not in cases_by_dimension:
                cases_by_dimension[dim] = []
            cases_by_dimension[dim].append(test_case)

        # Sort dimensions numerically (2D, 3D, 4D, etc.) to avoid misordering
        sorted_dims = sorted(
            cases_by_dimension.keys(),
            key=lambda d: (
                int(str(d).strip().removesuffix("D").removesuffix("d"))
                if str(d).strip().removesuffix("D").removesuffix("d").isdigit()
                else sys.maxsize
            ),
        )

        for dimension in sorted_dims:
            dim_cases = cases_by_dimension[dimension]

            lines.extend(
                [
                    f"#### Single Query Performance ({dimension})",
                    "",
                    "| Test Case | insphere | insphere_distance | insphere_lifted | Winner |",
                    "|-----------|----------|------------------|-----------------|---------|",
                ],
            )

            # Add single query performance data from parsed results
            for test_case in dim_cases:
                winner = test_case.get_winner()
                winner_text = f"**{winner}**" if winner else "N/A"

                # Convert nanoseconds to a more readable format
                methods_formatted = {}
                for method_name, perf_data in test_case.methods.items():
                    ns_time = perf_data.time_ns
                    if ns_time >= 1000:
                        # Convert to microseconds if >= 1000ns
                        methods_formatted[method_name] = f"{ns_time / 1000:.1f} µs"
                    else:
                        methods_formatted[method_name] = f"{ns_time:.0f} ns"

                insphere_time = methods_formatted.get("insphere", "N/A")
                distance_time = methods_formatted.get("insphere_distance", "N/A")
                lifted_time = methods_formatted.get("insphere_lifted", "N/A")

                lines.append(f"| {test_case.test_name} | {insphere_time} | {distance_time} | {lifted_time} | {winner_text} |")

            lines.append("")  # Add spacing between dimensions

        # Historical version comparison has been moved to static sections

        return lines

    def _parse_baseline_results(self) -> list[str]:
        """Parse baseline results and add to summary."""
        lines = [
            "## Triangulation Data Structure Performance",
            "",
        ]

        try:
            with self.baseline_file.open("r", encoding="utf-8") as f:
                content = f.read()

            # Extract metadata from baseline
            metadata_lines = []
            first_lines = content.split("\n")[:20]
            for line in first_lines:
                if line.startswith(("Generated at:", "Date:", "Git commit:", "Hardware:")):
                    metadata_lines.append(line)
            if not any(line.startswith("Hardware:") for line in metadata_lines) and "Hardware Information:" in content:
                hw = HardwareComparator.parse_baseline_hardware(content)
                cpu = hw.get("CPU", "")
                cores = hw.get("CPU_CORES", "")
                if cpu:
                    summary = f"{cpu} ({cores} cores)" if cores and cores != "Unknown" else cpu
                    metadata_lines.append(f"Hardware: {summary}")

            if metadata_lines:
                lines.extend(
                    [
                        "### Current Baseline Information",
                        "",
                    ],
                )
                for meta_line in metadata_lines:
                    lines.append(f"- **{meta_line}**")
                lines.append("")

            # Extract and format benchmark data
            benchmarks = extract_benchmark_data(content)
            if benchmarks:
                lines.extend(format_benchmark_tables(benchmarks))

        except Exception as e:
            lines.extend(
                [
                    "### Baseline Results",
                    "",
                    f"*Error parsing baseline results: {e}*",
                    "",
                ],
            )

        return lines

    def _parse_comparison_results(self) -> list[str]:
        """Parse comparison results and add status information."""
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
                        "See comparison details in the benchmark comparison output.",
                        "",
                    ],
                )

                # Extract and include specific regression details from content
                content_lines = content.split("\n")
                for line in content_lines:
                    if "REGRESSION:" in line or "IMPROVEMENT:" in line:
                        lines.append(f"- {line.strip()}")

                if any("REGRESSION:" in line or "IMPROVEMENT:" in line for line in content_lines):
                    lines.append("")
            else:
                lines.extend(
                    [
                        "### ✅ Performance Status: Good",
                        "",
                        "Recent benchmark comparison shows no significant performance regressions.",
                        "",
                    ],
                )

        except Exception:
            lines.extend(
                [
                    "### Comparison Results",
                    "",
                    "*No recent comparison data available*",
                    "",
                ],
            )

        return lines

    def _get_dynamic_analysis_sections(self) -> list[str]:
        """
        Generate dynamic analysis sections based on performance data.

        Returns:
            List of markdown lines with dynamic analysis
        """
        test_data = self._parse_circumsphere_benchmark_results()
        performance_ranking = self._analyze_performance_ranking(test_data)

        lines = [
            "## Key Findings",
            "",
            "### Performance Ranking",
            "",
        ]

        # Generate dynamic ranking based on data
        for i, (method, _avg_performance, description) in enumerate(performance_ranking, 1):
            lines.append(f"{i}. **{method}** - {description}")

        # Add numerical accuracy analysis with dynamic data if available
        lines.extend(self._get_numerical_accuracy_analysis())

        lines.extend(
            [
                "## Recommendations",
                "",
            ],
        )

        # Generate dynamic recommendations based on performance ranking
        lines.extend(self._generate_dynamic_recommendations(performance_ranking))

        # Add dynamic conclusion based on performance ranking
        if performance_ranking:
            fastest_method = performance_ranking[0][0]
            lines.extend(
                [
                    "",
                    "## Conclusion",
                    "",
                    f"The `{fastest_method}` method provides the best performance while maintaining reasonable numerical behavior.",
                    "For most applications requiring high-performance circumsphere containment tests, it should be the preferred choice.",
                    "",
                    "The standard `insphere` method remains the most numerically stable option when correctness is prioritized over performance.",
                    "",
                ],
            )

        return lines

    def _analyze_performance_ranking(self, test_data: list[CircumsphereTestCase]) -> list[tuple[str, float, str]]:
        """
        Analyze performance data to generate dynamic rankings.

        Args:
            test_data: List of CircumsphereTestCase objects

        Returns:
            List of tuples (method_name, average_performance, description)
        """
        method_totals = {"insphere": [], "insphere_distance": [], "insphere_lifted": []}

        # Collect performance data from all test cases
        for test_case in test_data:
            for method_name, perf_data in test_case.methods.items():
                method_totals[method_name].append(perf_data.time_ns)

        # Calculate averages and determine ranking
        method_averages = {}
        for method, times in method_totals.items():
            if times:
                method_averages[method] = sum(times) / len(times)
            else:
                method_averages[method] = float("inf")

        # Sort by performance (lowest time first)
        sorted_methods = sorted(method_averages.items(), key=lambda x: x[1])

        # Generate descriptions with relative performance
        rankings = []
        if sorted_methods:
            fastest_time = sorted_methods[0][1]
            rank_index = {m: i for i, (m, _) in enumerate(sorted_methods)}

            for method, avg_time in sorted_methods:
                idx = rank_index[method]
                slowdown = (avg_time / fastest_time) if fastest_time > 0 else 1
                if idx == 0:
                    desc = "(fastest) - Consistently best performance across all tests"
                elif idx == 1:
                    desc = f"(middle) - ~{slowdown:.1f}x slower than fastest"
                else:
                    desc = f"(slowest) - ~{slowdown:.1f}x slower than fastest"

                rankings.append((method, avg_time, desc))

        return rankings

    def _generate_dynamic_recommendations(self, performance_ranking: list[tuple[str, float, str]]) -> list[str]:
        """
        Generate dynamic recommendations based on performance ranking.

        Args:
            performance_ranking: List of performance ranking tuples

        Returns:
            List of markdown lines with recommendations
        """
        if not performance_ranking:
            return []

        fastest_method = performance_ranking[0][0]
        fastest_time = performance_ranking[0][1]

        # Calculate performance differences more accurately
        performance_diffs = []
        for method, avg_time, _ in performance_ranking[1:]:
            if fastest_time > 0:
                slowdown_factor = avg_time / fastest_time
                performance_diffs.append((method, slowdown_factor))

        lines = [
            "### For Performance-Critical Applications",
            "",
            f"- **Use `{fastest_method}`** for maximum performance",
        ]

        # Add specific performance comparison if we have data
        if performance_diffs:
            for method, slowdown in performance_diffs:
                if slowdown > 1.5:  # Significant difference
                    lines.append(f"- ~{slowdown:.1f}x faster than `{method}`")

        lines.extend(
            [
                "- Best choice for batch processing and high-frequency queries",
                "- Recommended for applications requiring millions of containment tests",
                "",
                "### For Numerical Stability",
                "",
                "- **Use `insphere`** for most reliable results",
                "- Standard determinant-based approach with proven mathematical properties",
                "- Good balance of performance and numerical stability",
                "- Recommended for applications where correctness is paramount",
                "",
                "### For Educational/Research Purposes",
                "",
                "- **Use `insphere_distance`** to understand geometric intuition",
                "- Explicit circumcenter calculation makes algorithm transparent",
                "- Excellent for debugging and algorithm validation",
                "- Useful for educational materials despite slower performance",
                "",
                "### Performance Summary",
                "",
            ],
        )

        # Add current benchmark-based summary
        if len(performance_ranking) >= 3:
            times = [f"{time / 1000:.1f} µs" if time >= 1000 else f"{time:.0f} ns" for _, time, _ in performance_ranking]

            lines.extend(
                [
                    "Based on current benchmarks:",
                    "",
                    f"- `{performance_ranking[0][0]}`: {times[0]} (fastest)",
                    f"- `{performance_ranking[1][0]}`: {times[1]} (balanced)",
                    f"- `{performance_ranking[2][0]}`: {times[2]} (transparent)",
                ],
            )

        return lines

    def _get_static_sections(self) -> list[str]:
        """
        Get static content sections (implementation notes, benchmark structure, etc.).

        Returns:
            List of markdown lines with static content
        """
        return [
            "## Historical Version Comparison",
            "",
            "*Based on archived performance measurements from previous releases:*",
            "",
            "### v0.3.0 → v0.3.1 Performance Improvements",
            "",
            "| Test Case | Method | v0.3.0 | v0.3.1 | Improvement |",
            "|-----------|--------|--------|--------|-------------|",
            "| Basic 3D | insphere | 808 ns | 805 ns | +0.4% |",
            "| Basic 3D | insphere_distance | 1,505 ns | 1,463 ns | +2.8% |",
            "| Basic 3D | insphere_lifted | 646 ns | 637 ns | +1.4% |",
            "| Random 1000 queries | insphere | 822 µs | 811 µs | +1.3% |",
            "| Random 1000 queries | insphere_distance | 1,535 µs | 1,494 µs | +2.7% |",
            "| Random 1000 queries | insphere_lifted | 661 µs | 650 µs | +1.7% |",
            "| 2D | insphere_lifted | 442 ns | 440 ns | +0.5% |",
            "| 4D | insphere_lifted | 962 ns | 955 ns | +0.7% |",
            "",
            "**Key Improvements**: Version 0.3.1 showed consistent performance gains across all methods,",
            "with `insphere_distance` seeing the largest improvement (+2.8%). The changes implemented",
            "improved numerical stability using `hypot` and `squared_norm` functions while providing",
            "measurable performance gains.",
            "",
            "## Implementation Notes",
            "",
            "### Performance Advantages of `insphere_lifted`",
            "",
            "1. More efficient matrix formulation using relative coordinates",
            "2. Avoids redundant circumcenter calculations",
            "3. Optimized determinant computation",
            "",
            "### Method Disagreements",
            "",
            "The disagreements between methods are expected due to:",
            "",
            "1. Different numerical approaches and tolerances",
            "2. Floating-point precision differences in multi-step calculations",
            "3. Varying sensitivity to degenerate cases",
            "",
            "## Benchmark Structure",
            "",
            "The `circumsphere_containment.rs` benchmark includes:",
            "",
            "- **Random queries**: Batch processing performance with 1000 random test points",
            "- **Dimensional tests**: Performance across 2D, 3D, 4D, and 5D simplices",
            "- **Edge cases**: Boundary vertices and far-away points",
            "- **Numerical consistency**: Agreement analysis between all methods",
            "",
        ]

    def _get_update_instructions(self) -> list[str]:
        """
        Generate performance data update instructions.

        Returns:
            List of markdown lines with update instructions
        """
        return [
            "## Performance Data Updates",
            "",
            "This file is automatically generated from benchmark results. To update:",
            "",
            "```bash",
            "# Generate performance summary with current data",
            "uv run benchmark-utils generate-summary",
            "",
            "# Run fresh benchmarks and generate summary (includes numerical accuracy)",
            "uv run benchmark-utils generate-summary --run-benchmarks",
            "",
            "# Generate baseline results for regression testing",
            "uv run benchmark-utils generate-baseline",
            "```",
            "",
            "### Customization",
            "",
            "For manual updates or custom analysis, modify the `PerformanceSummaryGenerator`",
            "class in `scripts/benchmark_utils.py`. This provides enhanced control over",
            "dynamic vs static content organization and supports parsing numerical accuracy",
            "data from live benchmark runs.",
            "",
        ]


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
    def _extract_dimension_from_dir(dim_dir: Path) -> str | None:
        """Extract dimension string from directory name (e.g., '2d' -> '2')."""
        dim = dim_dir.name.removesuffix("d")
        if dim.isdigit():
            return dim
        # Fallback: extract trailing "<digits>d" or "<digits>D"
        m = re.search(r"(\d+)[dD]$", dim_dir.name)
        return m.group(1) if m else None

    @staticmethod
    def _find_estimates_file(point_dir: Path) -> Path | None:
        """Find estimates.json file in point directory (prefer new/ over base/)."""
        new_file = point_dir / "new" / "estimates.json"
        if new_file.exists():
            return new_file
        base_file = point_dir / "base" / "estimates.json"
        return base_file if base_file.exists() else None

    @staticmethod
    def _process_point_directory(point_dir: Path, dim: str) -> BenchmarkData | None:
        """Process a single point count directory and extract benchmark data."""
        if not point_dir.is_dir():
            return None

        try:
            point_count = int(point_dir.name)
        except ValueError:
            return None

        estimates_file = CriterionParser._find_estimates_file(point_dir)
        if not estimates_file:
            return None

        return CriterionParser.parse_estimates_json(estimates_file, point_count, f"{dim}D")

    @staticmethod
    def _process_fallback_discovery(criterion_dir: Path) -> list[BenchmarkData]:
        """Recursively discover estimates.json files when structured search fails."""
        results = []
        seen: set[tuple[int, str]] = set()

        for estimates_file in criterion_dir.rglob("estimates.json"):
            parent_name = estimates_file.parent.name
            if parent_name not in {"base", "new"}:
                continue

            # Find nearest numeric points dir and nearest "<Nd>" or "<ND>" dir in ancestors
            points_dir = next((p for p in estimates_file.parents if p.name.isdigit()), None)
            dim_dir = next((p for p in estimates_file.parents if re.search(r"\d+[dD]$", p.name)), None)
            if not points_dir or not dim_dir:
                continue

            dim_match = re.search(r"(\d+)[dD]$", dim_dir.name)
            if not dim_match:
                continue

            points = int(points_dir.name)
            dimension = f"{dim_match.group(1)}D"
            key = (points, dimension)

            # Prefer "new" over "base" when duplicates exist
            if key in seen and parent_name == "base":
                continue

            bd = CriterionParser.parse_estimates_json(estimates_file, points, dimension)
            if bd:
                seen.add(key)
                results.append(bd)

        return results

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
        for dim_dir in sorted(p for p in criterion_dir.iterdir() if p.is_dir() and re.search(r"\d+[dD]$", p.name)):
            dim = CriterionParser._extract_dimension_from_dir(dim_dir)
            if not dim:
                continue

            # Iterate all nested benchmark targets under the <Nd> group
            for benchmark_dir in (p for p in dim_dir.iterdir() if p.is_dir()):
                # Find point count directories
                for point_dir in benchmark_dir.iterdir():
                    benchmark_data = CriterionParser._process_point_directory(point_dir, dim)
                    if benchmark_data:
                        results.append(benchmark_data)

        # Fallback: recursively discover estimates.json if nothing was found above
        if not results:
            results = CriterionParser._process_fallback_discovery(criterion_dir)

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
                    ["bench", "--bench", "ci_performance_suite", "--quiet", "--", *DEV_MODE_BENCH_ARGS],
                    cwd=self.project_root,
                    timeout=bench_timeout,
                    capture_output=True,
                )
            else:
                run_cargo_command(
                    ["bench", "--bench", "ci_performance_suite", "--quiet"],
                    cwd=self.project_root,
                    timeout=bench_timeout,
                    capture_output=True,
                )

            # Parse Criterion results
            target_dir = self.project_root / "target"
            benchmark_results = CriterionParser.find_criterion_results(target_dir)

            if not benchmark_results:
                return False

            # Generate baseline file
            self._write_baseline_file(benchmark_results, output_file)

            return True

        except subprocess.TimeoutExpired as e:
            print(f"❌ Benchmark execution timed out after {bench_timeout} seconds", file=sys.stderr)
            print("   Consider increasing --bench-timeout or using --dev mode for faster benchmarks", file=sys.stderr)
            logging.debug("TimeoutExpired: %s", e)
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
                    ["bench", "--bench", "ci_performance_suite", "--quiet", "--", *DEV_MODE_BENCH_ARGS],
                    cwd=self.project_root,
                    timeout=bench_timeout,
                    capture_output=True,
                )
            else:
                run_cargo_command(
                    ["bench", "--bench", "ci_performance_suite", "--quiet"],
                    cwd=self.project_root,
                    timeout=bench_timeout,
                    capture_output=True,
                )

            # Parse current results
            target_dir = self.project_root / "target"
            current_results = CriterionParser.find_criterion_results(target_dir)

            if not current_results:
                return False, False

            # Parse baseline
            baseline_content = baseline_file.read_text(encoding="utf-8")
            baseline_results = self._parse_baseline_file(baseline_content)

            # Generate comparison report
            regression_found = self._write_comparison_file(current_results, baseline_results, baseline_content, output_file)

            return True, regression_found

        except subprocess.TimeoutExpired as e:
            print(f"❌ Benchmark execution timed out after {bench_timeout} seconds", file=sys.stderr)
            print("   Consider increasing --bench-timeout or using --dev mode for faster benchmarks", file=sys.stderr)
            logging.debug("TimeoutExpired: %s", e)
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
        self,
        current_results: list[BenchmarkData],
        baseline_results: dict[str, BenchmarkData],
        baseline_content: str,
        output_file: Path,
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
                    f"(exceeds {self.regression_threshold}% threshold)\n",
                )
            elif average_change < -self.regression_threshold:
                f.write(
                    f"🎉 OVERALL IMPROVEMENT: Average performance improved by {abs(average_change):.1f}% "
                    f"(exceeds {self.regression_threshold}% threshold)\n",
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
                f"{benchmark.throughput_high}] {benchmark.throughput_unit}\n",
            )

    def _write_baseline_benchmark_data(self, f, benchmark: BenchmarkData) -> None:
        """Write baseline benchmark data."""
        f.write(f"Baseline Time: [{benchmark.time_low}, {benchmark.time_mean}, {benchmark.time_high}] {benchmark.time_unit}\n")
        if benchmark.throughput_mean is not None:
            f.write(
                f"Baseline Throughput: [{benchmark.throughput_low}, {benchmark.throughput_mean}, "
                f"{benchmark.throughput_high}] {benchmark.throughput_unit}\n",
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


class BenchmarkRegressionHelper:
    """Helper functions for performance regression testing workflow."""

    @staticmethod
    def write_github_env_vars(env_vars: dict[str, str]) -> None:
        """Helper to write multiple environment variables to GITHUB_ENV.
        Args:
            env_vars: Dictionary of environment variable names and values
        """
        github_env = os.getenv("GITHUB_ENV")
        if github_env:
            with open(github_env, "a", encoding="utf-8") as f:
                for key, value in env_vars.items():
                    val = "" if value is None else str(value)
                    # Normalize CR to avoid breaking heredoc boundaries
                    val = val.replace("\r", "")
                    if "\n" in val:
                        token = f"EOF_{uuid4().hex}"
                        f.write(f"{key}<<{token}\n{val}\n{token}\n")
                    else:
                        f.write(f"{key}={val}\n")
        # Make variables immediately available in this process as well
        for key, value in env_vars.items():
            os.environ[key] = value

    @staticmethod
    def prepare_baseline(baseline_dir: Path) -> bool:
        """
        Prepare baseline for comparison and set environment variables.

        Args:
            baseline_dir: Directory containing baseline artifacts

        Returns:
            True if baseline exists and is valid, False otherwise
        """
        # Look for baseline files using shared logic
        baseline_file = BenchmarkRegressionHelper._find_baseline_file(baseline_dir)

        # If a baseline file was found, copy it to baseline_results.txt for consistency
        if baseline_file and baseline_file.name != "baseline_results.txt":
            target_file = baseline_dir / "baseline_results.txt"
            try:
                copyfile(baseline_file, target_file)
                print(f"📦 Prepared baseline from artifact: {baseline_file.name} → baseline_results.txt")
            except OSError as e:
                print(f"❌ Failed to prepare baseline: {e}", file=sys.stderr)
                BenchmarkRegressionHelper.write_github_env_vars(
                    {"BASELINE_EXISTS": "false", "BASELINE_SOURCE": "artifact", "BASELINE_ORIGIN": "artifact"}
                )
                return False
        elif baseline_file:
            print("📦 Prepared baseline from artifact")
        else:
            print("❌ Downloaded artifact but no baseline*.txt files found", file=sys.stderr)
            BenchmarkRegressionHelper.write_github_env_vars({"BASELINE_EXISTS": "false", "BASELINE_SOURCE": "missing", "BASELINE_ORIGIN": "unknown"})
            return False

        # Set GitHub Actions environment variables
        BenchmarkRegressionHelper.write_github_env_vars(
            {"BASELINE_EXISTS": "true", "BASELINE_SOURCE": "artifact", "BASELINE_ORIGIN": "artifact", "BASELINE_SOURCE_FILE": baseline_file.name}
        )

        # Show baseline metadata
        print("=== Baseline Information (from artifact) ===")
        target_file = baseline_dir / "baseline_results.txt"  # Use the copied/standard file
        lines: list[str] = []
        try:
            with target_file.open("r", encoding="utf-8") as f:
                lines = f.readlines()
            for _i, line in enumerate(lines[:10]):
                print(line.rstrip())
        except OSError as e:
            print(f"⚠️ Failed to read baseline summary: {e}", file=sys.stderr)
            lines = []

        # Propagate tag (if present) to the workflow environment
        if lines:
            tag_line = next((ln for ln in lines if ln.startswith("Tag: ")), None)
            if tag_line:
                raw_tag = tag_line.split(":", 1)[1].strip()
                # Allow [A-Za-z0-9._-+]; replace others with underscore and cap length
                safe_tag = re.sub(r"[^A-Za-z0-9._\-+]", "_", raw_tag)[:64]
                BenchmarkRegressionHelper.write_github_env_vars({"BASELINE_TAG": safe_tag})

        return True

    @staticmethod
    def set_no_baseline_status() -> None:
        """Set environment variables when no baseline is found."""
        print("📈 No baseline artifact found for performance comparison")

        BenchmarkRegressionHelper.write_github_env_vars({"BASELINE_EXISTS": "false", "BASELINE_SOURCE": "none", "BASELINE_ORIGIN": "none"})

    @staticmethod
    def _find_baseline_file(baseline_dir: Path) -> Path | None:
        """Find the best available baseline file in the directory."""
        # Try standard name first
        baseline_file = baseline_dir / "baseline_results.txt"
        if baseline_file.exists():
            return baseline_file

        # Try tag-specific files (prefer highest semver if available)
        tag_files = list(baseline_dir.glob("baseline-v*.txt"))

        def _version_key(p: Path) -> tuple[int, Version | str, str]:
            # Parse semantic version from baseline filename (baseline-vX.Y.Z[-prerelease]?.txt)
            # Using packaging.version.Version for proper semantic version comparison
            m = re.match(r"baseline-v(.+)\.txt$", p.name)
            if m:
                version_str = m.group(1)
                try:
                    version = Version(version_str)
                    # Valid version: priority 1 (sorts first when reversed)
                    return (1, version, p.name)
                except Exception as e:
                    # Invalid version format, treat as non-semver
                    logging.debug("Invalid version format in %s: %s", p.name, e)
            # Fallback: put non-matching names last (priority 0, sorts after valid versions when reversed)
            return (0, p.name, "")

        if tag_files:
            # Sort by version (descending), with None (invalid versions) sorted last
            tag_files.sort(key=_version_key, reverse=True)
            # Return the highest valid version, or first file if no valid versions
            return tag_files[0]

        # Try any baseline*.txt files
        baseline_files = list(baseline_dir.glob("baseline*.txt"))
        if baseline_files:
            # Prefer most recent file when no semver match is available
            return max(baseline_files, key=lambda p: p.stat().st_mtime)

        return None

    @staticmethod
    def _extract_commit_from_baseline_file(baseline_file: Path) -> str | None:
        """Extract commit SHA from baseline text file."""
        try:
            with baseline_file.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("Git commit:"):
                        potential_sha = line.partition(":")[2].strip().split()[0]
                        if re.match(r"^[0-9A-Fa-f]{7,40}$", potential_sha):
                            return potential_sha
        except (OSError, ValueError) as e:
            logging.debug("Could not extract commit from %s: %s", baseline_file.name, e)
        return None

    @staticmethod
    def _extract_commit_from_metadata(metadata_file: Path) -> str | None:
        """Extract commit SHA from metadata.json file."""
        try:
            with metadata_file.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
                potential_sha = metadata.get("commit", "")
                if re.match(r"^[0-9A-Fa-f]{7,40}$", potential_sha):
                    return potential_sha
        except (OSError, json.JSONDecodeError, KeyError) as e:
            logging.debug("Could not extract commit from metadata.json: %s", e)
        return None

    @staticmethod
    def extract_baseline_commit(baseline_dir: Path) -> str:
        """
        Extract the baseline commit SHA from baseline files.

        Args:
            baseline_dir: Directory containing baseline artifacts

        Returns:
            Commit SHA string, or "unknown" if not found
        """
        commit_sha = "unknown"
        commit_source = "unknown"

        # Try to extract from baseline file first
        baseline_file = BenchmarkRegressionHelper._find_baseline_file(baseline_dir)
        if baseline_file:
            extracted_sha = BenchmarkRegressionHelper._extract_commit_from_baseline_file(baseline_file)
            if extracted_sha:
                commit_sha = extracted_sha
                commit_source = "baseline"

        # Fallback to metadata.json if needed
        if commit_sha == "unknown":
            metadata_file = baseline_dir / "metadata.json"
            if metadata_file.exists():
                extracted_sha = BenchmarkRegressionHelper._extract_commit_from_metadata(metadata_file)
                if extracted_sha:
                    commit_sha = extracted_sha
                    commit_source = "metadata"

        # Set GitHub Actions environment variables
        env_vars = {
            "BASELINE_COMMIT": commit_sha,
            "BASELINE_COMMIT_SOURCE": commit_source,
        }
        if baseline_file:
            env_vars["BASELINE_SOURCE_FILE"] = baseline_file.name
        BenchmarkRegressionHelper.write_github_env_vars(env_vars)

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
            if not re.match(r"^[0-9A-Fa-f]{7,40}$", baseline_commit):
                return False, "invalid_baseline_sha"

            commit_ref = f"{baseline_commit}^{{commit}}"
            root = find_project_root()
            run_git_command(["cat-file", "-e", commit_ref], cwd=root, timeout=60)

            # Check for relevant changes
            diff_range = f"{baseline_commit}..HEAD"
            result = run_git_command(["diff", "--name-only", diff_range], cwd=root, timeout=60)

            patterns = [re.compile(p) for p in (r"^src/", r"^benches/", r"^Cargo\.toml$", r"^Cargo\.lock$")]
            changed_files = result.stdout.strip().split("\n") if result.stdout.strip() else []
            has_relevant_changes = any(p.match(file) for file in changed_files for p in patterns)

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
            True if comparison ran and no regressions detected; False on regressions or error
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
                return False  # cause non-zero exit in CLI

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
                        # Also export to GITHUB_ENV using safe helper
                        BenchmarkRegressionHelper.write_github_env_vars({"BENCHMARK_REGRESSION_DETECTED": "true"})
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

    # Performance summary command
    perf_summary_parser = subparsers.add_parser("generate-summary", help="Generate performance summary markdown")
    perf_summary_parser.add_argument("--output", type=Path, help="Output file path (defaults to benches/PERFORMANCE_RESULTS.md)")
    perf_summary_parser.add_argument("--run-benchmarks", action="store_true", help="Run fresh circumsphere benchmarks before generating summary")

    return parser


def execute_baseline_commands(args: argparse.Namespace, project_root: Path) -> None:
    """Execute baseline generation and comparison commands."""
    if args.command == "generate-baseline":
        generator = BaselineGenerator(project_root, tag=args.tag)
        success = generator.generate_baseline(dev_mode=args.dev, output_file=args.output, bench_timeout=args.bench_timeout)
        sys.exit(0 if success else 1)

    elif args.command == "compare":
        comparator = PerformanceComparator(project_root)
        success, regression_found = comparator.compare_with_baseline(
            args.baseline,
            dev_mode=args.dev,
            output_file=args.output,
            bench_timeout=args.bench_timeout,
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
        BenchmarkRegressionHelper.write_github_env_vars(
            {
                "SKIP_BENCHMARKS": "true" if should_skip else "false",
                "SKIP_REASON": reason,
            }
        )

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

    # Try performance summary commands
    if args.command == "generate-summary":
        generator = PerformanceSummaryGenerator(project_root)
        success = generator.generate_summary(output_path=args.output, run_benchmarks=args.run_benchmarks, generator_name="benchmark_utils.py")
        sys.exit(0 if success else 1)

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
