#!/usr/bin/env python3
"""compare_storage_backends.py - Compare SlotMap vs DenseSlotMap performance

This script runs Phase 4 benchmarks with both storage backends and generates
a detailed comparison report showing performance differences in:
- Construction time
- Iteration speed (vertices, cells, neighbors)
- Query performance
- Validation overhead

Note: The benchmarks measure memory usage (RSS) internally, but this script
currently only parses and reports timing metrics from Criterion output.

Usage:
    # Run comparison with default settings
    uv run compare-storage-backends

    # Quick comparison (development mode)
    uv run compare-storage-backends --dev

    # Custom output location
    uv run compare-storage-backends --output artifacts/storage_comparison.md

    # Specify benchmark to run
    uv run compare-storage-backends --bench large_scale_performance
"""

import argparse
import logging
import re
import sys
from datetime import UTC, datetime
from json import loads
from pathlib import Path

try:
    from subprocess_utils import find_project_root, run_cargo_command
except ModuleNotFoundError:
    from scripts.subprocess_utils import find_project_root, run_cargo_command

logger = logging.getLogger(__name__)


class StorageBackendComparator:
    """Compare performance between SlotMap and DenseSlotMap storage backends."""

    def __init__(self, project_root: Path):
        """Initialize with project root directory."""
        self.project_root = project_root
        self.criterion_dir = project_root / "target" / "criterion"

    def run_comparison(
        self,
        benchmark_name: str = "large_scale_performance",
        dev_mode: bool = False,
        output_path: Path | None = None,
        extra_args: list[str] | None = None,
    ) -> bool:
        """
        Run benchmarks with both storage backends and generate comparison report.

        Args:
            benchmark_name: Name of benchmark to run (default: large_scale_performance)
            dev_mode: Use reduced scale for faster iteration
            output_path: Output file path (defaults to artifacts/storage_comparison.md)
            extra_args: Additional arguments to pass to cargo bench

        Returns:
            True if successful, False otherwise
        """
        try:
            if output_path is None:
                output_path = self.project_root / "artifacts" / "storage_comparison.md"

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            print("🔍 Running storage backend comparison...")
            print(f"   Benchmark: {benchmark_name}")
            print(f"   Mode: {'Development (reduced scale)' if dev_mode else 'Production (full scale)'}")
            print()

            # Run benchmarks with SlotMap (default)
            print("📊 Running benchmarks with SlotMap backend...")
            logger.debug("Running SlotMap benchmarks with extra_args=%s", extra_args)
            slotmap_results = self._run_benchmark(benchmark_name, use_dense_slotmap=False, dev_mode=dev_mode, extra_args=extra_args)

            if slotmap_results is None:
                print("❌ SlotMap benchmark failed", file=sys.stderr)
                return False

            # Run benchmarks with DenseSlotMap
            print("\n📊 Running benchmarks with DenseSlotMap backend...")
            logger.debug("Running DenseSlotMap benchmarks with extra_args=%s", extra_args)
            denseslotmap_results = self._run_benchmark(benchmark_name, use_dense_slotmap=True, dev_mode=dev_mode, extra_args=extra_args)

            if denseslotmap_results is None:
                print("❌ DenseSlotMap benchmark failed", file=sys.stderr)
                return False

            # Generate comparison report
            print("\n📝 Generating comparison report...")
            logger.debug(
                "Generating comparison report with %d SlotMap and %d DenseSlotMap benchmarks",
                len(slotmap_results.get("benchmarks", [])),
                len(denseslotmap_results.get("benchmarks", [])),
            )
            report = self._generate_comparison_report(slotmap_results, denseslotmap_results, benchmark_name, dev_mode)

            # Write report to file
            with output_path.open("w", encoding="utf-8") as f:
                f.write(report)

            print(f"\n✅ Comparison report saved: {output_path}")
            return True

        except Exception as e:
            print(f"❌ Comparison failed: {e}", file=sys.stderr)
            logging.exception("Comparison failed")
            return False

    def _run_benchmark(self, benchmark_name: str, use_dense_slotmap: bool, dev_mode: bool, extra_args: list[str] | None = None) -> dict | None:
        """
        Run benchmark with specified storage backend.

        Args:
            benchmark_name: Name of benchmark to run
            use_dense_slotmap: Whether to use DenseSlotMap feature
            dev_mode: Use reduced scale for faster iteration
            extra_args: Additional arguments to pass to cargo bench

        Returns:
            Dictionary of benchmark results, or None if failed
        """
        try:
            # Build cargo bench command
            args = ["bench", "--bench", benchmark_name]

            if use_dense_slotmap:
                args.extend(["--features", "dense-slotmap"])

            # Add development mode arguments or extra args
            if dev_mode or extra_args:
                if "--" not in args:
                    args.append("--")

                if dev_mode:
                    args.extend(
                        [
                            "--sample-size",
                            "10",
                            "--measurement-time",
                            "2",
                            "--warm-up-time",
                            "1",
                            "--noplot",
                        ]
                    )

                if extra_args:
                    args.extend(extra_args)

            # Run benchmark
            result = run_cargo_command(
                args,
                cwd=self.project_root,
                check=False,
            )
            success = result.returncode == 0
            stdout = result.stdout or ""
            stderr = result.stderr or ""

            if not success:
                print(f"   ⚠️ Benchmark failed for {'DenseSlotMap' if use_dense_slotmap else 'SlotMap'}", file=sys.stderr)
                if stderr:
                    print(f"   Error: {stderr}", file=sys.stderr)
                return None

            # Parse Criterion output
            results = self._parse_criterion_output(stdout)

            # Add backend info to results
            results["backend"] = "DenseSlotMap" if use_dense_slotmap else "SlotMap"
            results["features"] = ["dense-slotmap"] if use_dense_slotmap else []

            return results

        except Exception:
            logging.exception("Benchmark execution failed")
            return None

    def _parse_criterion_output(self, output: str) -> dict:
        """
        Parse Criterion benchmark output.
        Prefers JSON from target/criterion/**/new/estimates.json for robustness,
        falls back to regex parsing stdout if JSON not available.

        Args:
            output: Raw stdout from cargo bench

        Returns:
            Dictionary of parsed benchmark results
        """
        results = {
            "benchmarks": [],
            "raw_output": output,
        }

        # Attempt JSON parsing from target/criterion (more robust)
        json_found = False
        try:
            criterion_path = self.criterion_dir
            if criterion_path.exists():
                # Use Path.rglob for recursive glob matching
                for path in criterion_path.rglob("new/estimates.json"):
                    try:
                        data = loads(path.read_text(encoding="utf-8"))
                        estimate = data["mean"]["point_estimate"]
                        # Infer name from parent directory
                        name = path.parent.parent.name
                        lower_bound = float(data["mean"]["confidence_interval"]["lower_bound"])
                        upper_bound = float(data["mean"]["confidence_interval"]["upper_bound"])

                        results["benchmarks"].append(
                            {
                                "name": name,
                                "estimate": float(estimate),
                                "unit": "ns",  # Criterion estimates.json uses nanoseconds
                                "lower": lower_bound,
                                "upper": upper_bound,
                            }
                        )
                        json_found = True
                        logger.debug("Parsed JSON for benchmark: %s", name)
                    except Exception as e:
                        logger.debug("Failed to parse JSON from %s: %s", path, e)
                        continue
        except Exception:
            logger.debug("JSON parsing failed, falling back to regex")

        # Fallback to stdout regex parsing if no JSON found
        if not json_found:
            logger.debug("Using regex fallback for Criterion output parsing")
            # Format: "benchmark_name          time:   [12.345 ms 12.456 ms 12.567 ms]"
            # Pattern allows hyphens, colons, slashes for scoped names (e.g., "construction/2D/1000v")
            pattern = r"([A-Za-z0-9_:\-./]+)\s+time:\s+\[([0-9.]+)\s+(\w+)\s+([0-9.]+)\s+(\w+)\s+([0-9.]+)\s+(\w+)\]"

            for match in re.finditer(pattern, output):
                name = match.group(1)
                lower_value = float(match.group(2))
                # lower_unit = match.group(3)  # Same as estimate_unit
                estimate = float(match.group(4))
                estimate_unit = match.group(5)
                upper_value = float(match.group(6))
                # upper_unit = match.group(7)  # Same as estimate_unit

                results["benchmarks"].append(
                    {
                        "name": name,
                        "estimate": estimate,
                        "unit": estimate_unit,
                        "lower": lower_value,
                        "upper": upper_value,
                    }
                )

        return results

    def _build_comparison_table(self, slotmap_by_name: dict, denseslotmap_by_name: dict, all_names: list) -> tuple[list[str], list[float]]:
        """Build comparison table rows and collect diffs."""
        lines = []
        diffs = []

        for name in all_names:
            slotmap_bench = slotmap_by_name.get(name)
            denseslotmap_bench = denseslotmap_by_name.get(name)

            if slotmap_bench and denseslotmap_bench:
                slotmap_time = slotmap_bench["estimate"]
                denseslotmap_time = denseslotmap_bench["estimate"]
                unit = slotmap_bench["unit"]
                diff_pct = ((denseslotmap_time - slotmap_time) / slotmap_time) * 100
                diffs.append(diff_pct)

                # Determine winner (green=faster, yellow=same)
                if abs(diff_pct) < 2.0:
                    winner, emoji = "~Same", "🟡"
                elif diff_pct < 0:
                    winner, emoji = "✅ DenseSlotMap", "🟢"
                else:
                    winner, emoji = "✅ SlotMap", "🟢"

                lines.append(f"| {name} | {slotmap_time:.2f} {unit} | {denseslotmap_time:.2f} {unit} | {diff_pct:+.1f}% {emoji} | {winner} |")
            elif slotmap_bench:
                lines.append(f"| {name} | {slotmap_bench['estimate']:.2f} {slotmap_bench['unit']} | N/A | - | - |")
            elif denseslotmap_bench:
                lines.append(f"| {name} | N/A | {denseslotmap_bench['estimate']:.2f} {denseslotmap_bench['unit']} | - | - |")

        return lines, diffs

    def _generate_comparison_report(
        self,
        slotmap_results: dict,
        denseslotmap_results: dict,
        benchmark_name: str,
        dev_mode: bool,
    ) -> str:
        """
        Generate markdown comparison report.

        Args:
            slotmap_results: Results from SlotMap backend
            denseslotmap_results: Results from DenseSlotMap backend
            benchmark_name: Name of benchmark
            dev_mode: Whether development mode was used

        Returns:
            Formatted markdown report
        """
        # Header
        lines = [
            "# Storage Backend Comparison Report",
            "",
            f"**Benchmark**: `{benchmark_name}`",
            f"**Mode**: {'Development (reduced scale)' if dev_mode else 'Production (full scale)'}",
            f"**Generated**: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "## Executive Summary",
            "",
            "This report compares the performance of SlotMap (default) vs DenseSlotMap storage backends",
            "for the Delaunay triangulation data structure.",
            "",
            "### Key Metrics",
            "",
            "- **Construction**: Time to build triangulation",
            "- **Iteration**: Speed of vertex/cell/neighbor traversals",
            "- **Queries**: Lookup and contains-key performance",
            "",
            "## Detailed Results",
            "",
            "### Performance Comparison",
            "",
            "| Benchmark | SlotMap | DenseSlotMap | Difference | Winner |",
            "|-----------|---------|--------------|------------|--------|",
        ]

        # Build comparison table
        slotmap_by_name = {b["name"]: b for b in slotmap_results["benchmarks"]}
        denseslotmap_by_name = {b["name"]: b for b in denseslotmap_results["benchmarks"]}
        all_names = sorted(set(slotmap_by_name.keys()) | set(denseslotmap_by_name.keys()))

        table_lines, diffs = self._build_comparison_table(slotmap_by_name, denseslotmap_by_name, all_names)
        lines.extend(table_lines)

        # Summary statistics
        lines.extend(["", "## Summary Statistics", ""])

        if diffs:
            avg_diff = sum(diffs) / len(diffs)
            lines.extend(
                [
                    f"- **Average Performance Difference**: {avg_diff:+.1f}%",
                    f"- **Best Case (DenseSlotMap)**: {min(diffs):+.1f}%",
                    f"- **Worst Case (DenseSlotMap)**: {max(diffs):+.1f}%",
                    f"- **Benchmarks Compared**: {len(diffs)}",
                    "",
                ]
            )

            if avg_diff < -5:
                lines.append("**Interpretation**: DenseSlotMap shows significant performance improvement (>5% faster)")
            elif avg_diff > 5:
                lines.append("**Interpretation**: SlotMap shows better performance (DenseSlotMap >5% slower)")
            else:
                lines.append("**Interpretation**: Performance is comparable between backends (within 5%)")

            # Recommendations
            lines.extend(["", "## Recommendations", ""])
            if avg_diff < -5:
                lines.extend(
                    [
                        "✅ **Recommend DenseSlotMap** for this workload:",
                        "- Significant iteration performance improvement",
                        "- Better cache locality for traversal patterns",
                        "- Use `--features dense-slotmap` to enable",
                    ]
                )
            elif avg_diff > 5:
                lines.extend(
                    [
                        "✅ **Recommend SlotMap (default)** for this workload:",
                        "- Better overall performance",
                        "- Default backend requires no feature flags",
                    ]
                )
            else:
                lines.extend(
                    [
                        "🟡 **Either backend is suitable** for this workload:",
                        "- Performance is comparable",
                        "- Choose based on use case:",
                        "  - SlotMap: Better for dynamic insertions/removals",
                        "  - DenseSlotMap: Better for iteration-heavy workloads",
                    ]
                )
        else:
            lines.append("*No matching benchmarks found for comparison*")

        # Reproduction instructions
        lines.extend(
            [
                "",
                "## Reproduction",
                "",
                "To reproduce these results:",
                "",
                "```bash",
                "# SlotMap (default)",
                f"cargo bench --bench {benchmark_name}",
                "",
                "# DenseSlotMap",
                f"cargo bench --bench {benchmark_name} --features dense-slotmap",
                "```",
                "",
                "---",
                "",
                "*Generated by compare_storage_backends.py*",
            ]
        )

        return "\n".join(lines)


def main():
    """Main entry point for storage backend comparison."""
    parser = argparse.ArgumentParser(
        description="Compare SlotMap vs DenseSlotMap storage backend performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--bench",
        default="large_scale_performance",
        help="Benchmark to run (default: large_scale_performance)",
    )

    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use development mode (reduced scale for faster iteration)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: artifacts/storage_comparison.md)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--filter",
        help="Pass filter to Criterion benchmarks (e.g., 'construction')",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        # Find project root
        project_root = find_project_root()

        # Create comparator and run comparison
        comparator = StorageBackendComparator(project_root)

        # Build extra args if filter provided
        extra_args = [args.filter] if args.filter else None

        success = comparator.run_comparison(
            benchmark_name=args.bench,
            dev_mode=args.dev,
            output_path=args.output,
            extra_args=extra_args,
        )

        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        logging.exception("Fatal error")
        sys.exit(1)


if __name__ == "__main__":
    main()
