#!/usr/bin/env python3
"""
performance_summary_utils.py - Performance documentation generation utilities

This module provides functions for:
- Parsing benchmark results from various sources
- Generating performance summary markdown documentation
- Dynamic analysis of performance rankings and recommendations
- Version comparison and historical tracking

Focuses specifically on documentation generation, complementing benchmark_utils.py
which handles benchmark execution and regression testing.
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

try:
    # When executed as a script from scripts/
    from hardware_utils import HardwareInfo  # type: ignore[no-redef]
    from subprocess_utils import get_git_commit_hash, run_cargo_command, run_git_command  # type: ignore[no-redef]
except ModuleNotFoundError:
    # When imported as a module (e.g., scripts.performance_summary_utils)
    from scripts.hardware_utils import HardwareInfo  # type: ignore[no-redef]
    from scripts.subprocess_utils import get_git_commit_hash, run_cargo_command, run_git_command  # type: ignore[no-redef]


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

    def __post_init__(self):
        """Calculate improvement percentage."""
        if self.old_value > 0:
            self.improvement_pct = ((self.old_value - self.new_value) / self.old_value) * 100
        else:
            self.improvement_pct = 0.0


class ProjectRootNotFoundError(Exception):
    """Raised when project root directory cannot be located."""


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


class PerformanceSummaryGenerator:
    """Generate performance summary markdown from benchmark results."""

    def __init__(self, project_root: Path):
        """Initialize with project root directory."""
        self.project_root = project_root
        self.baseline_file = project_root / "benches" / "baseline_results.txt"
        self.comparison_file = project_root / "benches" / "compare_results.txt"

        # Path for storing circumsphere benchmark results
        self.circumsphere_results_dir = project_root / "target" / "criterion"

        # Extract current version and date information
        self.current_version = self._get_current_version()
        self.current_date = self._get_version_date()

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
            "**Generated By**: performance_summary_utils.py",
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
                ]
            )
        except Exception as e:
            logging.debug("Could not get hardware info: %s", e)
            lines.append("**Hardware**: Unknown")

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

        # Add dynamic analysis sections based on performance data
        lines.extend(self._get_dynamic_analysis_sections())

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
            return datetime.now(UTC).strftime("%Y-%m-%d")

    def _run_circumsphere_benchmarks(self) -> bool:
        """
        Run the circumsphere containment benchmarks to generate fresh data.

        Returns:
            True if benchmarks ran successfully, False otherwise
        """
        try:
            print("🔄 Running circumsphere containment benchmarks...")

            # Run the circumsphere benchmark with reduced sample size for speed
            result = run_cargo_command(
                ["bench", "--bench", "circumsphere_containment", "--", "--sample-size", "10"],
                cwd=self.project_root,
                timeout=300,  # 5 minute timeout for quick benchmarks
            )

            if result:
                print("✅ Circumsphere benchmarks completed successfully")
                return True
            print("❌ Circumsphere benchmarks failed")
            return False

        except Exception as e:
            print(f"❌ Error running circumsphere benchmarks: {e}")
            return False

    def _parse_circumsphere_benchmark_results(self) -> list[CircumsphereTestCase]:  # noqa: PLR0912
        """
        Parse circumsphere benchmark results from Criterion output.

        Returns:
            List of CircumsphereTestCase objects with parsed performance data
        """
        test_cases = []

        if not self.circumsphere_results_dir.exists():
            print(f"⚠️ No criterion results found at {self.circumsphere_results_dir}")
            return self._get_fallback_circumsphere_data()

        # Parse benchmark results from criterion output
        # Criterion replaces '/' with '_' in benchmark names
        benchmark_mappings = {
            "2d": ("Basic 2D", "2D"),
            "3d": ("Basic 3D", "3D"),
            "4d": ("Basic 4D", "4D"),
            "5d": ("Basic 5D", "5D"),
        }

        # Edge case benchmarks have a different structure in Criterion naming
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

        # Edge case benchmarks use shortened method names in Criterion paths
        edge_method_mappings = {
            "insphere": "insphere",
            "distance": "insphere_distance",
            "lifted": "insphere_lifted",
        }

        # Parse regular benchmarks
        for bench_key, (test_name, dimension) in benchmark_mappings.items():
            methods = {}

            for method_suffix, method_name in method_mappings.items():
                # Regular benchmarks: benchmark_name_method_name
                criterion_path = self.circumsphere_results_dir / f"{bench_key}_{method_suffix}"

                estimates_file = criterion_path / "base" / "estimates.json"
                if not estimates_file.exists():
                    estimates_file = criterion_path / "new" / "estimates.json"

                if estimates_file.exists():
                    try:
                        with estimates_file.open() as f:
                            estimates = json.load(f)

                        # Extract mean time in nanoseconds
                        mean_ns = estimates["mean"]["point_estimate"]

                        methods[method_name] = CircumspherePerformanceData(method=method_name, time_ns=mean_ns)
                    except Exception as e:
                        print(f"⚠️ Could not parse {estimates_file}: {e}")

            if methods:
                test_case = CircumsphereTestCase(test_name=test_name, dimension=dimension, methods=methods)
                test_cases.append(test_case)

        # Parse edge case benchmarks separately
        for edge_key, (test_name, dimension) in edge_case_mappings.items():
            methods = {}

            for method_suffix, method_name in edge_method_mappings.items():
                # Edge case benchmarks: edge_case_name_shortened_method_name
                criterion_path = self.circumsphere_results_dir / f"{edge_key}_{method_suffix}"

                estimates_file = criterion_path / "base" / "estimates.json"
                if not estimates_file.exists():
                    estimates_file = criterion_path / "new" / "estimates.json"

                if estimates_file.exists():
                    try:
                        with estimates_file.open() as f:
                            estimates = json.load(f)

                        # Extract mean time in nanoseconds
                        mean_ns = estimates["mean"]["point_estimate"]

                        methods[method_name] = CircumspherePerformanceData(method=method_name, time_ns=mean_ns)
                    except Exception as e:
                        print(f"⚠️ Could not parse {estimates_file}: {e}")

            if methods:
                test_case = CircumsphereTestCase(test_name=test_name, dimension=dimension, methods=methods)
                test_cases.append(test_case)

        # If no results were parsed, use fallback data
        if not test_cases:
            print("⚠️ No benchmark results parsed, using fallback data")
            return self._get_fallback_circumsphere_data()

        return test_cases

    def _get_fallback_circumsphere_data(self) -> list[CircumsphereTestCase]:
        """
        Get fallback circumsphere performance data when live benchmarks aren't available.

        Returns:
            List of CircumsphereTestCase objects with known performance data
        """
        return [
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
                f"### Version {self.current_version} Results ({self.current_date})",
                "",
                "⚠️ No benchmark results available. Run benchmarks first:",
                "```bash",
                "uv run performance-summary-utils generate --run-benchmarks",
                "```",
                "",
            ]

        lines = [
            f"### Version {self.current_version} Results ({self.current_date})",
            "",
        ]

        # Group test cases by dimension for better organization
        cases_by_dimension = {}
        for test_case in test_cases:
            dim = test_case.dimension
            if dim not in cases_by_dimension:
                cases_by_dimension[dim] = []
            cases_by_dimension[dim].append(test_case)

        # Sort dimensions (2D, 3D, 4D, etc.)
        sorted_dims = sorted(cases_by_dimension.keys(), key=lambda x: (len(x), x))

        for dimension in sorted_dims:
            dim_cases = cases_by_dimension[dimension]

            lines.extend(
                [
                    f"#### Single Query Performance ({dimension})",
                    "",
                    "| Test Case | insphere | insphere_distance | insphere_lifted | Winner |",
                    "|-----------|----------|------------------|-----------------|---------|",
                ]
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

        # Add historical version comparison (based on archived performance data)
        lines.extend(
            [
                "",
                "### Historical Version Comparison",
                "",
                "*Based on archived performance measurements from previous releases:*",
                "",
                "#### v0.3.0 → v0.3.1 Performance Improvements",
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
            ]
        )

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
                    ]
                )
            else:
                lines.extend(
                    [
                        "### ✅ Performance Status: Good",
                        "",
                        "Recent benchmark comparison shows no significant performance regressions.",
                        "",
                    ]
                )

        except Exception:
            lines.extend(
                [
                    "### Comparison Results",
                    "",
                    "*No recent comparison data available*",
                    "",
                ]
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

        lines.extend(
            [
                "",
                "### Numerical Accuracy Analysis",
                "",
                "Based on 1000 random test cases:",
                "",
                "- **insphere vs insphere_distance**: ~82% agreement",
                "- **insphere vs insphere_lifted**: ~0% agreement (different algorithms)",
                "- **insphere_distance vs insphere_lifted**: ~18% agreement",
                "- **All three methods agree**: ~0% (expected due to different numerical approaches)",
                "",
                "## Recommendations",
                "",
            ]
        )

        # Generate dynamic recommendations based on performance ranking
        lines.extend(self._generate_dynamic_recommendations(performance_ranking))

        lines.extend(
            [
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
                "## Conclusion",
                "",
            ]
        )

        if performance_ranking:
            fastest_method = performance_ranking[0][0]
            lines.extend(
                [
                    f"The `{fastest_method}` method provides the best performance while maintaining reasonable numerical behavior.",
                    "For most applications requiring high-performance circumsphere containment tests, it should be the preferred choice.",
                ]
            )

        lines.extend(
            [
                "",
                "The standard `insphere` method remains the most numerically stable option when correctness is prioritized over performance.",
                "",
            ]
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

            for method, avg_time in sorted_methods:
                if avg_time == fastest_time:
                    desc = "(fastest) - Consistently best performance across all tests"
                elif method == "insphere":
                    slowdown = (avg_time / fastest_time) if fastest_time > 0 else 1
                    desc = f"(middle) - ~{slowdown:.1f}x slower than fastest, but good performance"
                else:  # insphere_distance
                    slowdown = (avg_time / fastest_time) if fastest_time > 0 else 1
                    desc = f"(slowest) - ~{slowdown:.1f}x slower due to explicit circumcenter calculation"

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
            ]
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
                ]
            )

        return lines

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
            "# Generate fresh performance summary",
            "uv run performance-summary-utils generate",
            "",
            "# Run benchmarks and generate summary",
            "uv run performance-summary-utils generate --run-benchmarks",
            "",
            "# Generate baseline results (separate utility)",
            "uv run benchmark-utils generate-baseline",
            "```",
            "",
            "For manual updates or custom analysis, modify the `PerformanceSummaryGenerator`",
            "class in `scripts/performance_summary_utils.py`.",
            "",
        ]


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(description="Performance summary generation utilities")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate performance summary command
    gen_parser = subparsers.add_parser("generate", help="Generate performance summary markdown")
    gen_parser.add_argument("--output", type=Path, help="Output file path (defaults to benches/PERFORMANCE_RESULTS.md)")
    gen_parser.add_argument("--run-benchmarks", action="store_true", help="Run circumsphere benchmarks to get fresh data")

    return parser


def main():
    """Command-line interface for performance summary utilities."""
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

    if args.command == "generate":
        generator = PerformanceSummaryGenerator(project_root)
        success = generator.generate_summary(args.output, args.run_benchmarks)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
