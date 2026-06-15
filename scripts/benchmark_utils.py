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
import io
import json
import logging
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from shutil import copy2 as copyfile  # NOTE: Use copy2 (metadata-preserving) under the 'copyfile' alias for tests/patching convenience.
from typing import TYPE_CHECKING, Literal, NoReturn, TextIO, TypeIs
from urllib.parse import urlparse
from uuid import uuid4

from packaging.version import InvalidVersion, Version

logger = logging.getLogger(__name__)

DEFAULT_REGRESSION_THRESHOLD = 7.5
TIME_UNIT_TO_MICROSECONDS = {"ns": 1e-3, "µs": 1.0, "μs": 1.0, "us": 1.0, "ms": 1e3, "s": 1e6}
ComparisonFailurePolicy = Literal["strict", "total-time"]


class BaselineParseError(ValueError):
    """Raised when a benchmark baseline cannot be parsed without losing coverage."""


@dataclass(frozen=True)
class BaselineArtifactMetadata:
    """Metadata values written beside a generated baseline artifact."""

    commit_sha: str = "unknown"
    run_id: str = "unknown"
    runner_os: str = "unknown"
    runner_arch: str = "unknown"

    @classmethod
    def from_environment(cls) -> "BaselineArtifactMetadata":
        """Create artifact metadata from GitHub Actions-compatible environment variables."""
        return cls(
            commit_sha=os.getenv("GITHUB_SHA", os.getenv("SAFE_COMMIT_SHA", "unknown")),
            run_id=os.getenv("GITHUB_RUN_ID", os.getenv("SAFE_RUN_ID", "unknown")),
            runner_os=os.getenv("RUNNER_OS", "unknown"),
            runner_arch=os.getenv("RUNNER_ARCH", "unknown"),
        )


if TYPE_CHECKING:
    from benchmark_models import (
        BenchmarkData,
        CircumspherePerformanceData,
        CircumsphereTestCase,
        extract_benchmark_data,
        format_benchmark_tables,
    )
    from hardware_utils import HardwareComparator, HardwareInfo
    from subprocess_utils import (
        ExecutableNotFoundError,
        ProjectRootNotFoundError,
        find_project_root,
        get_git_commit_hash,
        get_git_remote_url,
        run_cargo_command,
        run_git_command,
        run_safe_command,
    )
else:
    try:
        # When executed as a script from scripts/
        from benchmark_models import (
            BenchmarkData,
            CircumspherePerformanceData,
            CircumsphereTestCase,
            extract_benchmark_data,
            format_benchmark_tables,
        )
        from hardware_utils import HardwareComparator, HardwareInfo
        from subprocess_utils import (
            ExecutableNotFoundError,
            ProjectRootNotFoundError,
            find_project_root,
            get_git_commit_hash,
            get_git_remote_url,
            run_cargo_command,
            run_git_command,
            run_safe_command,
        )
    except ModuleNotFoundError:
        # When imported as a module (e.g., scripts.benchmark_utils)
        from scripts.benchmark_models import (
            BenchmarkData,
            CircumspherePerformanceData,
            CircumsphereTestCase,
            extract_benchmark_data,
            format_benchmark_tables,
        )
        from scripts.hardware_utils import HardwareComparator, HardwareInfo
        from scripts.subprocess_utils import (
            ExecutableNotFoundError,
            ProjectRootNotFoundError,
            find_project_root,
            get_git_commit_hash,
            get_git_remote_url,
            run_cargo_command,
            run_git_command,
            run_safe_command,
        )

_RECOVERABLE_CLI_ERRORS: tuple[type[BaseException], ...] = (
    ExecutableNotFoundError,
    ProjectRootNotFoundError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    KeyError,
    subprocess.SubprocessError,
)

# Trusted benchmark commands use this Cargo profile so local, CI, and release
# numbers are generated with the same ThinLTO/codegen-units settings.
BENCHMARK_BUILD_FLAVOR = "perf"

CI_PERFORMANCE_SUITE_GROUPS = {
    "construction": (
        "Construction",
        "DelaunayTriangulation::new_with_options",
    ),
    "boundary_facets": (
        "Boundary facets",
        "DelaunayTriangulation::boundary_facets",
    ),
    "convex_hull": (
        "Convex hull",
        "ConvexHull::from_triangulation",
    ),
    "validation": (
        "Validation",
        "DelaunayTriangulation::validate",
    ),
    "incremental_insert": (
        "Incremental insert",
        "DelaunayTriangulation::insert",
    ),
    "bistellar_flips": (
        "Bistellar flips",
        "BistellarFlips",
    ),
}

CI_PERFORMANCE_SUITE_GROUP_ORDER = tuple(CI_PERFORMANCE_SUITE_GROUPS)
_CI_PERFORMANCE_SUITE_MANIFEST_IDS_FILE = "ci_performance_suite_manifest_ids.txt"
_CI_PERFORMANCE_SUITE_METRICS_FILE = "ci_performance_suite_metrics.json"
_CI_PERFORMANCE_SUITE_RUN_METADATA_FILE = "ci_performance_suite_run_metadata.json"
PERF_NO_REGRESSIONS_REQUIRED_BENCHMARK_ID = "tds_new_2d/tds_new/4000"
MAIN_VS_RELEASE_COMPARISON_RESULTS_FILE = "main_vs_release_compare_results.txt"
WORKTREE_VS_REF_COMPARISON_RESULTS_TEMPLATE = "worktree_vs_{ref}_compare_results.txt"
PERF_NO_REGRESSIONS_RELEVANT_PATHS = (
    "src",
    "benches",
    "Cargo.toml",
    "Cargo.lock",
    "scripts/benchmark_utils.py",
)


@dataclass(frozen=True)
class BenchmarkTimeChange:
    """Normalized timing comparison used by benchmark summary policies."""

    label: str
    current_mean_us: float
    baseline_mean_us: float
    time_change_pct: float


@dataclass(frozen=True)
class ComparisonFileRequest:
    """Context for writing a benchmark comparison report."""

    baseline_content: str
    output_file: Path
    dev_mode: bool
    failure_policy: ComparisonFailurePolicy


@dataclass(frozen=True)
class ComparisonSummaryStats:
    """Summary statistics for benchmark comparison failure policy decisions."""

    total_time_change: float
    geomean_change: float
    median_change: float
    individual_regressions: int
    compared_count: int
    failure_policy: ComparisonFailurePolicy


@dataclass(frozen=True)
class CiPerformanceMetric:
    """Validated construction metric emitted by ci_performance_suite."""

    vertices: int
    simplices: int

    def __post_init__(self) -> None:
        """Keep construction counts positive and integral."""
        _require_positive_int_field("vertices", self.vertices)
        _require_positive_int_field("simplices", self.simplices)


@dataclass(frozen=True)
class CriterionEstimate:
    """Validated Criterion timing estimate in nanoseconds."""

    mean_ns: float
    low_ns: float
    high_ns: float


def ci_suite_group_key(first_path_part: str) -> str | None:
    """Map a Criterion path prefix to a ci_performance_suite group key."""
    if first_path_part.startswith("tds_new_"):
        return "construction"
    if first_path_part.startswith("bistellar_flips"):
        return "bistellar_flips"
    if first_path_part in CI_PERFORMANCE_SUITE_GROUPS:
        return first_path_part
    return None


def ci_suite_dimension(benchmark_id: str) -> str:
    """Extract the dimension label from a ci_performance_suite benchmark ID."""
    match = re.search(r"(?:^|_|/)(\d+)d(?:_|/|$)", benchmark_id)
    if match:
        return f"{match.group(1)}D"
    return "n/a"


def _expand_ci_benchmark_id_pattern(pattern: str) -> set[str]:
    """Expand the simple brace patterns emitted by ci_performance_suite."""
    segments = []
    for segment in pattern.split("/"):
        if segment.startswith("{") and segment.endswith("}"):
            segments.append([option for option in segment[1:-1].split(",") if option])
        else:
            segments.append([segment])
    return {"/".join(parts) for parts in product(*segments)}


def _parse_ci_performance_manifest_ids(stdout: str) -> set[str]:
    """Parse benchmark IDs from ci_performance_suite manifest stdout lines."""
    manifest_ids: set[str] = set()
    for line in stdout.splitlines():
        if not line.startswith("api_benchmark "):
            continue
        fields = dict(token.split("=", 1) for token in line.split()[1:] if "=" in token)
        benchmark_ids = fields.get("benchmark_ids", "")
        for pattern in benchmark_ids.split(";"):
            if pattern:
                manifest_ids.update(_expand_ci_benchmark_id_pattern(pattern))
    return manifest_ids


def _parse_ci_performance_metrics(stdout: str) -> dict[str, dict[str, int]]:
    """Parse construction metrics emitted by ci_performance_suite."""
    metrics: dict[str, dict[str, int]] = {}
    for line in stdout.splitlines():
        if not line.startswith("api_benchmark_metric "):
            continue
        fields = dict(token.split("=", 1) for token in line.split()[1:] if "=" in token)
        benchmark_id = fields.get("benchmark_id")
        if not benchmark_id:
            continue
        try:
            vertices = int(fields["vertices"])
            simplices = int(fields["simplices"])
        except (KeyError, ValueError):
            logger.debug("Skipping malformed ci_performance_suite metric line: %s", line)
            continue
        if vertices <= 0 or simplices <= 0:
            logger.debug("Skipping non-positive ci_performance_suite metric line: %s", line)
            continue
        metrics[benchmark_id] = {
            "vertices": vertices,
            "simplices": simplices,
        }
    return metrics


def _ci_performance_manifest_ids_path(criterion_dir: Path) -> Path:
    """Return the sidecar manifest path used to filter ci_performance_suite results."""
    return criterion_dir / _CI_PERFORMANCE_SUITE_MANIFEST_IDS_FILE


def _ci_performance_metrics_path(criterion_dir: Path) -> Path:
    """Return the sidecar metrics path used to annotate ci_performance_suite results."""
    return criterion_dir / _CI_PERFORMANCE_SUITE_METRICS_FILE


def _ci_performance_run_metadata_path(criterion_dir: Path) -> Path:
    """Return the sidecar metadata path for the latest ci_performance_suite run."""
    return criterion_dir / _CI_PERFORMANCE_SUITE_RUN_METADATA_FILE


def _write_ci_performance_manifest_ids(project_root: Path, stdout: str) -> None:
    """Persist the runtime ci_performance_suite manifest beside Criterion results."""
    if not isinstance(stdout, str):
        msg = "ci_performance_suite completed but stdout was not text; cannot extract api_benchmark manifest"
        raise TypeError(msg)
    criterion_dir = project_root / "target" / "criterion"
    manifest_path = _ci_performance_manifest_ids_path(criterion_dir)
    manifest_ids = _parse_ci_performance_manifest_ids(stdout)
    if not manifest_ids:
        msg = f"ci_performance_suite completed but emitted no api_benchmark manifest in stdout: {stdout!r}"
        raise RuntimeError(msg)
    criterion_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "\n".join(sorted(manifest_ids)) + "\n",
        encoding="utf-8",
    )


def _write_ci_performance_metrics(project_root: Path, stdout: str, *, require_metrics: bool = False) -> None:
    """Persist ci_performance_suite construction metrics beside Criterion results."""
    criterion_dir = project_root / "target" / "criterion"
    metrics_path = _ci_performance_metrics_path(criterion_dir)
    criterion_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(stdout, str):
        metrics_path.write_text("{}\n", encoding="utf-8")
        if require_metrics:
            msg = "ci_performance_suite completed but stdout was not text; cleared stale construction metrics"
            raise TypeError(msg)
        return

    metrics = _parse_ci_performance_metrics(stdout)
    if not metrics:
        metrics_path.write_text("{}\n", encoding="utf-8")
        if require_metrics:
            msg = f"ci_performance_suite emitted no construction metrics; cleared stale metrics sidecar: {metrics_path}"
            raise RuntimeError(msg)
        return

    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_ci_performance_run_metadata(
    project_root: Path,
    *,
    completed_at: datetime,
    cargo_profile: str,
    use_dev_mode: bool,
) -> None:
    """Persist metadata for the latest successful ci_performance_suite run."""
    criterion_dir = project_root / "target" / "criterion"
    metadata_path = _ci_performance_run_metadata_path(criterion_dir)
    criterion_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "cargo_profile": cargo_profile,
        "completed_at": completed_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "sampling_mode": "dev" if use_dev_mode else "full",
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_ci_performance_manifest_ids(criterion_dir: Path) -> set[str] | None:
    """Load ci_performance_suite benchmark IDs when a runtime manifest exists."""
    manifest_path = _ci_performance_manifest_ids_path(criterion_dir)
    if not manifest_path.exists():
        return None
    try:
        manifest_ids = {line.strip() for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    except OSError:
        return None
    return manifest_ids or None


def _parse_ci_performance_metric(benchmark_id: str, values: Mapping[object, object], metrics_path: Path) -> CiPerformanceMetric | None:
    """Parse one metrics sidecar entry into a validated metric object."""
    vertices = values.get("vertices")
    simplices = values.get("simplices")
    if not isinstance(vertices, int) or isinstance(vertices, bool) or vertices <= 0:
        logger.debug("Skipping malformed ci_performance_suite metric entry %r from %s", benchmark_id, metrics_path)
        return None
    if not isinstance(simplices, int) or isinstance(simplices, bool) or simplices <= 0:
        logger.debug("Skipping malformed ci_performance_suite metric entry %r from %s", benchmark_id, metrics_path)
        return None
    return CiPerformanceMetric(vertices=vertices, simplices=simplices)


def _load_ci_performance_metrics(criterion_dir: Path) -> dict[str, CiPerformanceMetric]:
    """Load ci_performance_suite construction metrics when present."""
    metrics_path = _ci_performance_metrics_path(criterion_dir)
    if not metrics_path.exists():
        return {}
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except OSError as error:
        msg = f"failed to read ci_performance_suite metrics sidecar {metrics_path}: {error}"
        raise OSError(msg) from error
    except json.JSONDecodeError as error:
        msg = f"malformed ci_performance_suite metrics sidecar {metrics_path}: {error}"
        raise ValueError(msg) from error
    if not isinstance(data, dict):
        msg = f"malformed ci_performance_suite metrics sidecar {metrics_path}: expected JSON object"
        raise TypeError(msg)

    metrics: dict[str, CiPerformanceMetric] = {}
    for benchmark_id, values in data.items():
        if not isinstance(benchmark_id, str) or not isinstance(values, dict):
            logger.debug("Skipping malformed ci_performance_suite metric entry %r from %s", benchmark_id, metrics_path)
            continue
        metric = _parse_ci_performance_metric(benchmark_id, values, metrics_path)
        if metric is not None:
            metrics[benchmark_id] = metric
    return metrics


def _load_ci_performance_run_metadata(criterion_dir: Path) -> dict[str, str]:
    """Load metadata for the latest ci_performance_suite run when present."""
    metadata_path = _ci_performance_run_metadata_path(criterion_dir)
    if not metadata_path.exists():
        return {}
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {key: value for key, value in data.items() if isinstance(key, str) and isinstance(value, str)}


def _ci_performance_sidecar_timestamp(criterion_dir: Path) -> str | None:
    """Return a best-effort timestamp from ci_performance_suite sidecar mtimes."""
    sidecars = [
        _ci_performance_manifest_ids_path(criterion_dir),
        _ci_performance_metrics_path(criterion_dir),
    ]
    timestamps = [path.stat().st_mtime for path in sidecars if path.exists()]
    if not timestamps:
        return None
    return datetime.fromtimestamp(max(timestamps), UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def is_valid_criterion_estimate(mean_ns: float, low_ns: float, high_ns: float) -> bool:
    """Return whether Criterion estimate values are finite and ordered."""
    return all(math.isfinite(value) for value in (mean_ns, low_ns, high_ns)) and mean_ns > 0 and 0 <= low_ns <= mean_ns <= high_ns


def _is_object_mapping(value: object) -> TypeIs[Mapping[object, object]]:
    """Return whether a raw value can be treated as an object-keyed mapping."""
    return isinstance(value, Mapping)


def _require_positive_int_field(name: str, value: object) -> None:
    """Reject values that are not positive non-bool integers."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        msg = f"{name} must be a positive integer (got {value!r})"
        raise ValueError(msg)


def _criterion_float(value: object) -> float:
    """Convert a raw Criterion JSON scalar into a float."""
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        msg = f"expected numeric Criterion estimate value, got {value!r}"
        raise TypeError(msg)
    return float(value)


def _parse_criterion_estimate(data: object) -> CriterionEstimate | None:
    """Parse raw Criterion estimates.json data into a validated estimate."""
    if not _is_object_mapping(data):
        return None
    mean_data = data.get("mean", {})
    if not _is_object_mapping(mean_data):
        return None
    confidence_interval = mean_data.get("confidence_interval", {})
    if not _is_object_mapping(confidence_interval):
        return None

    try:
        mean_ns = _criterion_float(mean_data["point_estimate"])
        low_ns = _criterion_float(confidence_interval.get("lower_bound", mean_ns))
        high_ns = _criterion_float(confidence_interval.get("upper_bound", mean_ns))
    except (KeyError, TypeError, ValueError):
        return None

    if not is_valid_criterion_estimate(mean_ns, low_ns, high_ns):
        return None
    return CriterionEstimate(mean_ns=mean_ns, low_ns=low_ns, high_ns=high_ns)


def _load_criterion_estimate(estimates_path: Path) -> CriterionEstimate | None:
    """Load and validate a Criterion estimates.json file."""
    try:
        with estimates_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return _parse_criterion_estimate(data)


def _collect_ci_suite_estimates(criterion_dir: Path) -> list[tuple[tuple[str, ...], Path]]:
    """Collect deduplicated ci_performance_suite estimates, preferring new over base."""
    manifest_ids = _load_ci_performance_manifest_ids(criterion_dir)
    estimates_by_id: dict[tuple[str, ...], tuple[str, Path]] = {}

    for estimates_path in sorted(criterion_dir.glob("**/estimates.json")):
        if estimates_path.parent.name not in {"base", "new"}:
            continue

        try:
            path_parts = estimates_path.relative_to(criterion_dir).parts[:-2]
        except ValueError:
            continue

        if not path_parts or ci_suite_group_key(path_parts[0]) is None:
            continue

        benchmark_id = "/".join(path_parts)
        if manifest_ids is not None and benchmark_id not in manifest_ids:
            continue

        existing = estimates_by_id.get(path_parts)
        if existing is None or (existing[0] == "base" and estimates_path.parent.name == "new"):
            estimates_by_id[path_parts] = (estimates_path.parent.name, estimates_path)

    return [(path_parts, estimates_path) for path_parts, (_, estimates_path) in estimates_by_id.items()]


# Development mode arguments - centralized to keep baseline generation and comparison in sync
# Reduces samples for faster iteration during development (10x faster than full benchmarks)
#
# Note: These are Criterion CLI arguments. Some benchmarks can also be configured via
# environment variables documented in benches/README.md:
#   CRIT_SAMPLE_SIZE=10 CRIT_MEASUREMENT_MS=2000 CRIT_WARMUP_MS=1000
# The CLI arguments take precedence over env vars when both are present.
DEV_MODE_BENCH_ARGS = [
    "--sample-size",
    "10",
    "--measurement-time",
    "2",
    "--warm-up-time",
    "1",
    "--noplot",
]


@dataclass(frozen=True)
class CiPerformanceResult:
    """Parsed Criterion result for one ci_performance_suite benchmark ID."""

    group_key: str
    benchmark_id: str
    dimension: str
    input_size: str
    mean_ns: float
    low_ns: float
    high_ns: float

    @property
    def variant(self) -> str:
        """Return the geometry/input variant label for this benchmark."""
        if "adversarial" in self.benchmark_id:
            return "adversarial"
        return "well-conditioned"


def _criterion_arg_value(args: list[str], flag: str) -> str:
    """Return the Criterion value that follows flag in args."""
    try:
        index = args.index(flag)
    except ValueError:
        return "unknown"

    value_index = index + 1
    if value_index >= len(args):
        return "unknown"
    return args[value_index]


def _sampling_metadata(dev_mode: bool) -> dict[str, str]:
    """Return benchmark sampling metadata for baseline/compare validation."""
    if not dev_mode:
        return {
            "sampling_mode": "full",
            "cargo_profile": BENCHMARK_BUILD_FLAVOR,
            "criterion_args": "default",
            "criterion_sample_size": "criterion-default",
            "criterion_measurement_time": "criterion-default",
            "criterion_warm_up_time": "criterion-default",
        }

    return {
        "sampling_mode": "dev",
        "cargo_profile": BENCHMARK_BUILD_FLAVOR,
        "criterion_args": " ".join(DEV_MODE_BENCH_ARGS),
        "criterion_sample_size": _criterion_arg_value(DEV_MODE_BENCH_ARGS, "--sample-size"),
        "criterion_measurement_time": _criterion_arg_value(DEV_MODE_BENCH_ARGS, "--measurement-time"),
        "criterion_warm_up_time": _criterion_arg_value(DEV_MODE_BENCH_ARGS, "--warm-up-time"),
    }


# Use the shared secure wrapper from subprocess_utils
# ProjectRootNotFoundError and find_project_root are imported from subprocess_utils


# =============================================================================
# PERFORMANCE SUMMARY GENERATOR
# =============================================================================


class PerformanceSummaryGenerator:
    """Generate performance summary markdown from benchmark results."""

    def __init__(self, project_root: Path) -> None:
        """Initialize with project root directory."""
        self.project_root = project_root
        # Prefer CI artifact location; fall back to benches/ for local runs
        self.baseline_file = project_root / "baseline-artifact" / "baseline_results.txt"
        self._baseline_fallback = project_root / "benches" / "baseline_results.txt"
        self.comparison_file = release_comparison_results_path(project_root)

        # Path for storing Criterion benchmark results
        self.circumsphere_results_dir = project_root / "target" / "criterion"

        # Storage for numerical accuracy data from benchmarks
        self.numerical_accuracy_data: dict[str, str] | None = None

        # Extract current version and date information
        self.current_version = self._get_current_version()
        self.current_date = self._get_version_date()

    def generate_summary(
        self,
        output_path: Path | None = None,
        run_benchmarks: bool = False,
        generator_name: str | None = None,
        cargo_profile: str | None = None,
        strict: bool = False,
    ) -> bool:
        """
        Generate performance summary markdown file.

        Args:
            output_path: Output file path (defaults to benches/PERFORMANCE_RESULTS.md)
            run_benchmarks: Whether to run fresh public API and circumsphere benchmarks
            generator_name: Name of the tool generating the summary (for attribution)
            cargo_profile: Optional Cargo profile for fresh benchmark runs.  When
                ``run_benchmarks`` is True and no profile is specified, defaults
                to :data:`BENCHMARK_BUILD_FLAVOR` so fresh runs match baseline
                and comparison measurements.
            strict: Fail instead of rendering from existing or fallback data
                when fresh benchmark execution is requested and any benchmark
                command fails.

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
                # Default fresh runs to the trusted perf profile so numbers are
                # comparable with baseline/compare output.
                if cargo_profile is None:
                    cargo_profile = BENCHMARK_BUILD_FLAVOR
                ci_success = self._run_ci_performance_suite(cargo_profile=cargo_profile)
                circumsphere_success, accuracy_data = self._run_circumsphere_benchmarks(cargo_profile=cargo_profile)
                if circumsphere_success:
                    self.numerical_accuracy_data = accuracy_data
                if not ci_success or not circumsphere_success:
                    if strict:
                        print("❌ Benchmark run failed; strict summary mode refuses fallback data", file=sys.stderr)
                        return False
                    print("⚠️ Benchmark run failed, using existing/fallback data")

            # Generate markdown content
            content = self._generate_markdown_content(generator_name)
            if strict and self._contains_fallback_summary_data(content):
                print("❌ Strict summary mode detected fallback benchmark data", file=sys.stderr)
                return False

            # Write to output file
            with output_path.open("w", encoding="utf-8") as f:
                f.write(content)

            print(f"📊 Generated performance summary: {output_path}")
            return True

        except _RECOVERABLE_CLI_ERRORS as e:
            print(f"❌ Failed to generate performance summary: {e}", file=sys.stderr)
            return False

    @staticmethod
    def _contains_fallback_summary_data(content: str) -> bool:
        """Return whether generated summary content used fallback/reference data."""
        fallback_markers = (
            "reference data",
            "No `ci_performance_suite` Criterion results available",
            "No benchmark results available. Run benchmarks first",
            "To get current numerical accuracy data",
        )
        return any(marker in content for marker in fallback_markers)

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
            f"- **Last Updated**: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"- **Generated By**: {generator_name}",
        ]

        # Add git information
        try:
            commit_hash = get_git_commit_hash(cwd=self.project_root)
            if commit_hash and commit_hash != "unknown":
                lines.append(f"- **Git Commit**: {commit_hash}")
        except _RECOVERABLE_CLI_ERRORS as e:
            logger.debug("Could not get git commit hash: %s", e)

        # Add hardware information
        try:
            hardware_info = HardwareInfo()
            hw_info = hardware_info.get_hardware_info(cwd=self.project_root)
            lines.extend(
                [
                    f"- **Hardware**: {hw_info['CPU']} ({hw_info['CPU_CORES']} cores)",
                    f"- **Memory**: {hw_info['MEMORY']}",
                    f"- **OS**: {hw_info['OS']}",
                    f"- **Rust**: {hw_info['RUST']}",
                ],
            )
        except _RECOVERABLE_CLI_ERRORS as e:
            logger.debug("Could not get hardware info: %s", e)
            lines.append("- **Hardware**: Unknown")

        # Lead with the focused construction/TDS section so the Criterion run
        # metadata and user-facing construction results are immediately visible.
        tds_results = self._get_triangulation_data_structure_results()
        if tds_results:
            lines.append("")
            lines.extend(tds_results)

        if lines[-1] != "":
            lines.append("")
        lines.extend(
            [
                "## Performance Results Summary",
                "",
            ],
        )

        # Add public API performance results from the CI suite next. This is
        # the versioned benchmark contract used by baseline/comparison tooling.
        lines.extend(self._get_ci_performance_suite_results())

        # Add circumsphere predicate results as a focused subsection. These
        # remain important because they exercise la-stack-backed predicates.
        lines.extend(self._get_circumsphere_performance_results())

        # Add circumsphere-specific implementation notes next to the data they
        # explain.
        lines.extend(self._get_implementation_notes())

        # Add comparison results if available
        if self.comparison_file.exists():
            lines.extend(self._parse_comparison_results())

        # Add static content sections (moved to end)
        lines.extend(self._get_static_sections())

        # Add performance data update instructions
        lines.extend(self._get_update_instructions())

        return "\n".join(lines)

    def _get_current_version(self) -> str:
        """
        Get the current crate version.

        Returns:
            Current version string (e.g., "0.4.3") or "unknown" if not found
        """
        package_version = self._get_package_version()
        if package_version:
            return package_version

        try:
            # Get the latest tag that matches version pattern
            cp = run_git_command(["describe", "--tags", "--abbrev=0", "--match=v*"], cwd=self.project_root)
            result = cp.stdout.strip()
            if result.startswith("v"):
                return result[1:]  # Remove 'v' prefix
            return "unknown"
        except _RECOVERABLE_CLI_ERRORS:
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
            except _RECOVERABLE_CLI_ERRORS:
                return "unknown"

    def _get_package_version(self) -> str | None:
        """Return the root Cargo package version when Cargo.toml is available."""
        cargo_toml = self.project_root / "Cargo.toml"
        try:
            with cargo_toml.open("rb") as f:
                manifest = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError):
            return None

        package = manifest.get("package")
        if not isinstance(package, dict):
            return None

        version = package.get("version")
        if isinstance(version, str) and version.strip():
            return version.strip()
        return None

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
        except _RECOVERABLE_CLI_ERRORS:
            return datetime.now(UTC).strftime("%Y-%m-%d")

    def _run_circumsphere_benchmarks(self, cargo_profile: str | None = None) -> tuple[bool, dict[str, str] | None]:
        """
        Run the circumsphere containment benchmarks to generate fresh data.

        Args:
            cargo_profile: Cargo profile for the fresh run.  Defaults to
                :data:`BENCHMARK_BUILD_FLAVOR` so every fresh benchmark run
                goes through the same ThinLTO/codegen-units settings used
                by baseline generation and comparison.

        Returns:
            Tuple of (success, numerical_accuracy_data)
        """
        try:
            print("🔄 Running circumsphere containment benchmarks...")

            profile = cargo_profile if cargo_profile is not None else BENCHMARK_BUILD_FLAVOR
            cargo_args = ["bench", "--profile", profile, "--bench", "circumsphere_containment", "--", *DEV_MODE_BENCH_ARGS]

            # Run the circumsphere benchmark with reduced sample size for speed
            result = run_cargo_command(
                cargo_args,
                cwd=self.project_root,
                timeout=240,  # 4 minute timeout for quick benchmarks
                capture_output=True,
            )

            # Parse numerical accuracy data from stdout
            numerical_accuracy_data = self._parse_numerical_accuracy_output(result.stdout)

            print("✅ Circumsphere benchmarks completed successfully")
            return True, numerical_accuracy_data

        except _RECOVERABLE_CLI_ERRORS as e:
            print(f"❌ Error running circumsphere benchmarks: {e}")
            return False, None

    def _run_ci_performance_suite(self, cargo_profile: str | None = None, *, use_dev_mode: bool = False) -> bool:
        """
        Run the public API CI performance suite to generate fresh Criterion data.

        Args:
            cargo_profile: Cargo profile for the fresh run. Defaults to
                :data:`BENCHMARK_BUILD_FLAVOR` so summary, baseline, and
                comparison measurements use the same optimized profile.
            use_dev_mode: When true, pass reduced Criterion sampling arguments
                for local development feedback. Full sampling is used by
                default.

        Returns:
            True if the benchmark completed successfully, False otherwise.
        """
        try:
            print("🔄 Running ci_performance_suite benchmarks...")

            profile = cargo_profile if cargo_profile is not None else BENCHMARK_BUILD_FLAVOR
            cargo_args = ["bench", "--profile", profile, "--bench", "ci_performance_suite"]
            if use_dev_mode:
                cargo_args.extend(["--", *DEV_MODE_BENCH_ARGS])

            result = run_cargo_command(
                cargo_args,
                cwd=self.project_root,
                timeout=900,
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                print(f"❌ Error running ci_performance_suite benchmarks: cargo exited with status {result.returncode}")
                return False

            completed_at = datetime.now(UTC)
            _write_ci_performance_manifest_ids(self.project_root, result.stdout)
            _write_ci_performance_metrics(self.project_root, result.stdout, require_metrics=True)
            _write_ci_performance_run_metadata(
                self.project_root,
                completed_at=completed_at,
                cargo_profile=profile,
                use_dev_mode=use_dev_mode,
            )
            print("✅ ci_performance_suite benchmarks completed successfully")
            return True

        except ExecutableNotFoundError as e:
            print(f"❌ Error running ci_performance_suite benchmarks: {e}")
            return False
        except subprocess.TimeoutExpired as e:
            print(f"❌ Error running ci_performance_suite benchmarks: {e}")
            return False
        except OSError as e:
            print(f"❌ Error running ci_performance_suite benchmarks: {e}")
            return False

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

            return accuracy_data or None

        except (IndexError, TypeError, ValueError):
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
                # Mark boundary cases: "Boundary vertex" tests have early-exit optimizations
                is_boundary = "boundary" in edge_key.lower()
                test_case = CircumsphereTestCase(test_name=test_name, dimension=dimension, methods=methods, is_boundary_case=is_boundary)
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
            estimate = _load_criterion_estimate(estimates_file)
            if estimate is not None:
                return CircumspherePerformanceData(method=method_name, time_ns=estimate.mean_ns)

        return None

    def _get_fallback_circumsphere_data(self) -> list[CircumsphereTestCase]:
        """
        Get fallback circumsphere performance data when live benchmarks aren't available.

        Returns:
            List of CircumsphereTestCase objects with known performance data
        """
        fallback_rows = (
            ("Basic 2D", "2D", False, 560, 644, 448),
            ("Boundary vertex", "2D", True, 570, 644, 451),
            ("Far vertex", "2D", False, 570, 641, 449),
            ("Basic 3D", "3D", False, 805, 1463, 637),
            ("Boundary vertex", "3D", True, 811, 1497, 647),
            ("Far vertex", "3D", False, 808, 1493, 649),
            ("Basic 4D", "4D", False, 1200, 1900, 979),
            ("Boundary vertex", "4D", True, 1300, 1900, 987),
            ("Far vertex", "4D", False, 1300, 1900, 975),
            ("Basic 5D", "5D", False, 1800, 3000, 1500),
            ("Boundary vertex", "5D", True, 1800, 3100, 1500),
            ("Far vertex", "5D", False, 1800, 3000, 1500),
        )
        return [
            CircumsphereTestCase(
                name,
                dimension,
                {
                    "insphere": CircumspherePerformanceData("insphere", insphere_ns),
                    "insphere_distance": CircumspherePerformanceData("insphere_distance", distance_ns),
                    "insphere_lifted": CircumspherePerformanceData("insphere_lifted", lifted_ns),
                },
                is_boundary_case=is_boundary_case,
            )
            for name, dimension, is_boundary_case, insphere_ns, distance_ns, lifted_ns in fallback_rows
        ]

    @staticmethod
    def _format_duration_ns(time_ns: float) -> str:
        """Format nanosecond Criterion timings with readable units."""
        if time_ns >= 1_000_000_000:
            return f"{time_ns / 1_000_000_000:.3f} s"
        if time_ns >= 1_000_000:
            return f"{time_ns / 1_000_000:.3f} ms"
        if time_ns >= 1_000:
            return f"{time_ns / 1_000:.1f} µs"
        return f"{time_ns:.0f} ns"

    @staticmethod
    def _ci_suite_input_size(path_parts: tuple[str, ...]) -> str:
        """Extract a human-readable input size from Criterion benchmark path parts."""
        if path_parts and path_parts[-1].isdigit():
            return path_parts[-1]
        return "roundtrip"

    @staticmethod
    def _load_criterion_estimate(estimates_path: Path) -> tuple[float, float, float] | None:
        """Load mean and confidence interval values from a Criterion estimates file."""
        estimate = _load_criterion_estimate(estimates_path)
        if estimate is None:
            return None
        return estimate.mean_ns, estimate.low_ns, estimate.high_ns

    def _parse_ci_performance_suite_results(self) -> list[CiPerformanceResult]:
        """
        Parse Criterion data for the versioned ci_performance_suite benchmark IDs.

        Criterion stores each benchmark under a path derived from its group and
        benchmark ID. This parser keeps those IDs intact so the generated
        summary can compare API surfaces side-by-side as the suite grows.
        """
        criterion_dir = self.circumsphere_results_dir
        if not criterion_dir.exists():
            return []

        results = []
        for path_parts, estimates_path in _collect_ci_suite_estimates(criterion_dir):
            estimates = self._load_criterion_estimate(estimates_path)
            if estimates is None:
                continue

            benchmark_id = "/".join(path_parts)
            group_key = ci_suite_group_key(path_parts[0])
            if group_key is None:
                continue

            mean_ns, low_ns, high_ns = estimates
            results.append(
                CiPerformanceResult(
                    group_key=group_key,
                    benchmark_id=benchmark_id,
                    dimension=ci_suite_dimension(benchmark_id),
                    input_size=self._ci_suite_input_size(path_parts),
                    mean_ns=mean_ns,
                    low_ns=low_ns,
                    high_ns=high_ns,
                ),
            )

        group_order = {group: index for index, group in enumerate(CI_PERFORMANCE_SUITE_GROUP_ORDER)}
        results.sort(
            key=lambda result: (
                group_order.get(result.group_key, sys.maxsize),
                int(result.dimension.removesuffix("D")) if result.dimension.removesuffix("D").isdigit() else sys.maxsize,
                int(result.input_size) if result.input_size.isdigit() else sys.maxsize,
                result.benchmark_id,
            ),
        )
        return results

    def _get_ci_performance_suite_results(self) -> list[str]:
        """
        Generate the public API performance summary from ci_performance_suite data.

        Returns:
            List of markdown lines with ci_performance_suite benchmark data.
        """
        results = self._parse_ci_performance_suite_results()

        lines = [
            "### Public API Performance Contract (`ci_performance_suite`)",
            "",
            "This suite is the versioned benchmark contract for public Delaunay workflows.",
            "It covers construction, hull extraction, validation, incremental insertion,",
            "boundary traversal, and explicit bistellar flip roundtrips.",
            "",
        ]

        if not results:
            lines.extend(
                [
                    "⚠️ No `ci_performance_suite` Criterion results available. Run:",
                    "```bash",
                    f"cargo bench --profile {BENCHMARK_BUILD_FLAVOR} --bench ci_performance_suite",
                    "```",
                    "",
                ],
            )
            return lines

        results_by_group: dict[str, list[CiPerformanceResult]] = {}
        for result in results:
            results_by_group.setdefault(result.group_key, []).append(result)

        for group_key in CI_PERFORMANCE_SUITE_GROUP_ORDER:
            group_results = results_by_group.get(group_key)
            if not group_results:
                continue

            group_label, public_api = CI_PERFORMANCE_SUITE_GROUPS[group_key]
            lines.extend(
                [
                    f"#### {group_label}",
                    "",
                    f"Public API: `{public_api}`",
                    "",
                    "| Benchmark ID | Dimension | Input | Variant | Mean | 95% CI |",
                    "|--------------|-----------|-------|---------|------|--------|",
                ],
            )

            for result in group_results:
                confidence_interval = f"{self._format_duration_ns(result.low_ns)} - {self._format_duration_ns(result.high_ns)}"
                lines.append(
                    f"| `{result.benchmark_id}` | {result.dimension} | {result.input_size} | {result.variant} | "
                    f"{self._format_duration_ns(result.mean_ns)} | {confidence_interval} |",
                )

            lines.append("")

        return lines

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
                "### Circumsphere Predicate Performance",
                "",
                f"#### Version {self.current_version} Results ({self.current_date})",
                "",
                "⚠️ No benchmark results available. Run benchmarks first:",
                "```bash",
                f"uv run benchmark-utils generate-summary --run-benchmarks --profile {BENCHMARK_BUILD_FLAVOR}",
                "```",
                "",
            ]

        lines = [
            "### Circumsphere Predicate Performance",
            "",
            "This focused predicate suite tracks `la-stack`-backed circumsphere and",
            "insphere query performance independently from full triangulation workflows.",
            "",
            f"#### Version {self.current_version} Results ({self.current_date})",
            "",
        ]

        # Group test cases by dimension for better organization
        cases_by_dimension: dict[str, list[CircumsphereTestCase]] = {}
        for test_case in test_cases:
            dim = test_case.dimension
            if dim not in cases_by_dimension:
                cases_by_dimension[dim] = []
            cases_by_dimension[dim].append(test_case)

        # Sort dimensions numerically (2D, 3D, 4D, etc.) to avoid misordering
        sorted_dims = sorted(
            cases_by_dimension.keys(),
            key=lambda d: (
                int(str(d).strip().removesuffix("D").removesuffix("d")) if str(d).strip().removesuffix("D").removesuffix("d").isdigit() else sys.maxsize
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
            first_lines = content.split("\n")[:20]
            metadata_lines = [line for line in first_lines if line.startswith(("Generated at:", "Date:", "Git commit:", "Hardware:"))]
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
                        "### Baseline Artifact Information",
                        "",
                    ],
                )
                lines.extend(f"- **{meta_line}**" for meta_line in metadata_lines)
                lines.append("")

            # Extract and format benchmark data
            benchmarks = extract_benchmark_data(content)
            if benchmarks:
                lines.extend(format_benchmark_tables(benchmarks, input_label="Vertices"))

        except (OSError, TypeError, ValueError, KeyError) as e:
            lines.extend(
                [
                    "### Baseline Results",
                    "",
                    f"*Error parsing baseline results: {e}*",
                    "",
                ],
            )

        return lines

    def _current_tds_benchmarks(self) -> list[BenchmarkData]:
        """Return current construction/TDS Criterion results from ci_performance_suite."""
        target_dir = self.project_root / "target"
        benchmarks = CriterionParser.find_criterion_results(target_dir)
        return [
            benchmark
            for benchmark in benchmarks
            if benchmark.benchmark_id and ci_suite_group_key(benchmark.benchmark_id.split("/", maxsplit=1)[0]) == "construction"
        ]

    def _get_triangulation_data_structure_results(self) -> list[str]:
        """Generate the triangulation data-structure section from current data when possible."""
        current_benchmarks = self._current_tds_benchmarks()
        if current_benchmarks:
            criterion_dir = self.project_root / "target" / "criterion"
            run_metadata = _load_ci_performance_run_metadata(criterion_dir)
            run_date = run_metadata.get("completed_at") or None
            if run_date is None:
                run_date = _ci_performance_sidecar_timestamp(criterion_dir)
                if run_date is not None:
                    run_date = f"{run_date} (sidecar timestamp)"

            lines = [
                "## Triangulation Data Structure Performance",
                "",
                "### Current Criterion Run Information",
                "",
            ]
            if run_date is not None:
                lines.append(f"- **Date: {run_date}**")
            else:
                lines.append("- **Date: unavailable**")
            try:
                commit_hash = get_git_commit_hash(cwd=self.project_root)
                if commit_hash and commit_hash != "unknown":
                    lines.append(f"- **Git commit: {commit_hash}**")
            except _RECOVERABLE_CLI_ERRORS as e:
                logger.debug("Could not get git commit hash for TDS section: %s", e)

            lines.extend(
                [
                    "- **Source: current `target/criterion` construction results**",
                    "",
                ],
            )
            lines.extend(
                format_benchmark_tables(
                    current_benchmarks,
                    input_label="Vertices",
                    include_simplices=True,
                ),
            )
            return lines

        if self.baseline_file.exists() or self._baseline_fallback.exists():
            if not self.baseline_file.exists():
                self.baseline_file = self._baseline_fallback
            return self._parse_baseline_results()

        return []

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
                lines.extend(f"- {line.strip()}" for line in content_lines if "REGRESSION:" in line or "IMPROVEMENT:" in line)

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

        except OSError:
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
            "## Circumsphere Predicate Analysis",
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
                "### Recommendations",
                "",
            ],
        )

        # Generate dynamic recommendations based on performance ranking
        lines.extend(self._generate_dynamic_recommendations(performance_ranking))

        # Add dynamic conclusion based on performance ranking
        if performance_ranking:
            lines.extend(
                [
                    "",
                    "### Conclusion",
                    "",
                    "All three methods are mathematically correct and produce valid results. Performance characteristics vary by dimension:",
                    "",
                ],
            )

            # Add dimension-specific winners
            for method, _, desc in performance_ranking:
                if "best in" in desc:
                    lines.append(f"- `{method}` {desc}")

            lines.extend(
                [
                    "",
                    "For general-purpose applications, choose based on your primary use case:",
                    "",
                    "- **Performance-critical**: Use the method that performs best in your target dimension",
                    "- **Numerical stability**: Use `insphere` for its proven mathematical properties",
                    "- **Educational/debugging**: Use `insphere_distance` for its transparent algorithm",
                    "",
                ],
            )

        return lines

    @staticmethod
    def _collect_method_performance(test_data: list[CircumsphereTestCase]) -> tuple[dict[str, list[float]], dict[str, list[str]]]:
        """Collect per-method timings and dimension wins, excluding trivial boundary cases."""
        method_totals: dict[str, list[float]] = {"insphere": [], "insphere_distance": [], "insphere_lifted": []}
        method_wins: dict[str, list[str]] = {"insphere": [], "insphere_distance": [], "insphere_lifted": []}

        for test_case in test_data:
            if test_case.is_boundary_case:
                continue

            winner = test_case.get_winner()
            if winner:
                method_wins[winner].append(test_case.dimension)

            for method_name, perf_data in test_case.methods.items():
                method_totals[method_name].append(perf_data.time_ns)

        return method_totals, method_wins

    @staticmethod
    def _ranking_description(method: str, avg_time: float, fastest_time: float, method_wins: dict[str, list[str]]) -> str:
        """Describe relative method performance for the dynamic ranking table."""
        if avg_time == float("inf"):
            return "No benchmark data available"

        slowdown = (avg_time / fastest_time) if fastest_time > 0 and fastest_time != float("inf") else 1
        wins = method_wins.get(method, [])
        if not wins:
            return f"~{slowdown:.1f}x slower than fastest on average"

        dims_text = ", ".join(sorted(set(wins)))
        if slowdown > 1.01:
            return f"(best in {dims_text}) - ~{slowdown:.1f}x average vs fastest"
        return f"(best in {dims_text}) - Best average performance"

    def _analyze_performance_ranking(self, test_data: list[CircumsphereTestCase]) -> list[tuple[str, float, str]]:
        """
        Analyze performance data to generate dynamic rankings.

        Args:
            test_data: List of CircumsphereTestCase objects

        Returns:
            List of tuples (method_name, average_performance, description)
        """
        method_totals, method_wins = self._collect_method_performance(test_data)

        # Calculate averages and determine ranking
        method_averages = {}
        for method, times in method_totals.items():
            if times:
                method_averages[method] = sum(times) / len(times)
            else:
                method_averages[method] = float("inf")

        # Sort by performance (lowest time first)
        sorted_methods = sorted(method_averages.items(), key=lambda x: x[1])

        rankings = []
        if sorted_methods:
            fastest_time = sorted_methods[0][1]

            for method, avg_time in sorted_methods:
                rankings.append((method, avg_time, self._ranking_description(method, avg_time, fastest_time, method_wins)))

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

        lines = [
            "#### Method Selection Guide",
            "",
            "**All three methods are mathematically correct** (they produce valid insphere test results).",
            "Choose based on your specific requirements:",
            "",
        ]

        # Add dimension-specific performance recommendations
        lines.append("##### Performance Optimization by Dimension")
        lines.append("")

        for method, _avg_time, desc in performance_ranking:
            if "best in" in desc:
                # Extract dimension info from description
                lines.append(f"- **`{method}`**: {desc}")

        lines.extend(
            [
                "",
                "##### General Recommendations",
                "",
                "**For maximum performance**: Choose the method that performs best in your target dimension (see above)",
                "",
                "**For general-purpose use**: `insphere` provides consistent performance across all dimensions",
                "and uses the standard determinant-based approach with well-understood numerical properties",
                "",
                "**For algorithm transparency**: `insphere_distance` explicitly calculates the circumcenter,",
                "making it excellent for educational purposes, debugging, and algorithm validation",
                "",
                "##### Performance Comparison",
                "",
                "Average performance across all non-boundary test cases:",
                "",
            ],
        )

        # Add current benchmark-based summary with data-driven labels
        if len(performance_ranking) >= 3:
            # Format times, handling inf gracefully
            times = []
            for _, time, _ in performance_ranking:
                if time == float("inf"):
                    times.append("N/A")
                elif time >= 1000:
                    times.append(f"{time / 1000:.1f} µs")
                else:
                    times.append(f"{time:.0f} ns")

            # Extract brief labels from descriptions or use position-based defaults
            def brief_label(desc: str, position: int) -> str:
                """Extract label from description or use position-based default."""
                if "best in" in desc:
                    # Extract just the dimension info without outer parens;
                    # the caller's f-string wraps the result in (...) already.
                    # Use removeprefix/removesuffix (not strip) to avoid
                    # accidentally removing internal parentheses.
                    return desc.split(" - ", maxsplit=1)[0].removeprefix("(").removesuffix(")")
                defaults = ["fastest average", "second fastest", "third fastest"]
                return defaults[position] if position < len(defaults) else "slower"

            lines.extend(
                [
                    f"- `{performance_ranking[0][0]}`: {times[0]} ({brief_label(performance_ranking[0][2], 0)})",
                    f"- `{performance_ranking[1][0]}`: {times[1]} ({brief_label(performance_ranking[1][2], 1)})",
                    f"- `{performance_ranking[2][0]}`: {times[2]} ({brief_label(performance_ranking[2][2], 2)})",
                ],
            )

        return lines

    @staticmethod
    def _get_implementation_notes() -> list[str]:
        """
        Get circumsphere-specific implementation notes.

        Returns:
            List of markdown lines with implementation notes
        """
        return [
            "## Implementation Notes",
            "",
            "### Dimension-Dependent InSphere Predicate Performance",
            "",
            "The tables above are the source of truth for predicate timing. `insphere_lifted`",
            "shows advantages in lower dimensions such as 2D/3D, while `insphere_distance`",
            "often wins in 4D/5D; boundary cases may favor `insphere` because of early exits.",
            "",
        ]

    def _get_static_sections(self) -> list[str]:
        """
        Get static content sections (benchmark structure, etc.).

        Returns:
            List of markdown lines with static content
        """
        return [
            "## Benchmark Structure",
            "",
            "The `ci_performance_suite.rs` benchmark is the primary regression and",
            "release-summary suite. It emits a versioned `api_benchmark_manifest` and",
            "covers public construction, hull, validation, insertion, boundary, and",
            "bistellar-flip workflows across supported dimensions.",
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
            "This file is automatically generated from benchmark results. For release-facing updates:",
            "",
            "```bash",
            "just bench-perf-summary",
            "```",
            "",
            "For manual diagnostics without the release recipe, use the underlying CLI:",
            "",
            "```bash",
            "# Re-render from currently available Criterion data",
            "uv run benchmark-utils generate-summary",
            "",
            "# Run fresh perf-profile public API and circumsphere benchmarks",
            f"uv run benchmark-utils generate-summary --run-benchmarks --profile {BENCHMARK_BUILD_FLAVOR}",
            "",
            "# Package existing ci_performance_suite Criterion results for release-asset comparisons",
            "uv run benchmark-utils write-baseline --ref vX.Y.Z --output baseline_results.txt",
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
    def parse_estimates_json(estimates_path: Path, points: int | None, dimension: str) -> BenchmarkData | None:
        """
        Parse Criterion estimates.json file to extract benchmark data.

        Args:
            estimates_path: Path to estimates.json file
            points: Number of points in the benchmark
            dimension: Dimension string (e.g., "2D", "3D")

        Returns:
            BenchmarkData object or None if parsing fails
        """
        estimate = _load_criterion_estimate(estimates_path)
        if estimate is None:
            return None

        # Convert nanoseconds to microseconds
        mean_us = estimate.mean_ns / 1000
        low_us = estimate.low_ns / 1000
        high_us = estimate.high_ns / 1000

        benchmark = BenchmarkData(points, dimension).with_timing(round(low_us, 2), round(mean_us, 2), round(high_us, 2), "µs")

        if points is not None:
            # Calculate throughput in Kelem/s
            # Throughput = points / time_in_seconds
            # For time in microseconds: throughput = points * 1,000,000 / time_us
            # For Kelem/s: throughput_kelem = (points * 1,000,000 / time_us) / 1000 = points * 1000 / time_us
            # Guard against division by zero for very fast benchmarks
            eps = 1e-9  # µs - minimum time to prevent division by zero
            thrpt_mean = points * 1000 / max(mean_us, eps)
            thrpt_low = points * 1000 / max(high_us, eps)  # Lower time = higher throughput
            thrpt_high = points * 1000 / max(low_us, eps)  # Higher time = lower throughput
            benchmark.with_throughput(round(thrpt_low, 3), round(thrpt_mean, 3), round(thrpt_high, 3), "Kelem/s")

        return benchmark

    @staticmethod
    def _ci_suite_input_points(path_parts: tuple[str, ...]) -> int | None:
        """Extract the numeric input size when the Criterion ID has one."""
        if path_parts and path_parts[-1].isdigit():
            return int(path_parts[-1])
        return None

    @staticmethod
    def _ci_suite_metric_simplices(
        metric: CiPerformanceMetric | None,
        *,
        benchmark_id: str,
        path_parts: tuple[str, ...],
        points: int | None,
        dimension: str,
    ) -> int | None:
        """Return sidecar simplex counts only when they match the Criterion result."""
        if metric is None:
            return None

        expected_dimension = ci_suite_dimension(benchmark_id)
        expected_points = CriterionParser._ci_suite_input_points(path_parts)
        if expected_dimension != dimension or expected_points != points:
            logger.debug("Skipping stale ci_performance_suite metric for %s", benchmark_id)
            return None

        if points is None or metric.vertices != points:
            logger.debug(
                "Skipping stale ci_performance_suite metric for %s: vertices=%s, Criterion input=%s",
                benchmark_id,
                metric.vertices,
                points,
            )
            return None

        return metric.simplices

    @staticmethod
    def _process_ci_performance_suite_results(criterion_dir: Path) -> list[BenchmarkData]:
        """Discover ci_performance_suite Criterion results with expanded benchmark IDs."""
        results: list[BenchmarkData] = []
        metrics = _load_ci_performance_metrics(criterion_dir)
        for path_parts, estimates_path in _collect_ci_suite_estimates(criterion_dir):
            benchmark_id = "/".join(path_parts)
            dimension = ci_suite_dimension(benchmark_id)
            if dimension == "n/a":
                continue

            points = CriterionParser._ci_suite_input_points(path_parts)
            benchmark_data = CriterionParser.parse_estimates_json(estimates_path, points, dimension)
            if benchmark_data is None:
                continue

            benchmark_data.benchmark_id = benchmark_id
            metric_simplices = CriterionParser._ci_suite_metric_simplices(
                metrics.get(benchmark_id),
                benchmark_id=benchmark_id,
                path_parts=path_parts,
                points=points,
                dimension=dimension,
            )
            if metric_simplices is not None:
                benchmark_data.simplices = metric_simplices
            results.append(benchmark_data)

        group_order = {group: index for index, group in enumerate(CI_PERFORMANCE_SUITE_GROUP_ORDER)}
        results.sort(
            key=lambda result: (
                group_order.get(ci_suite_group_key(result.benchmark_id.split("/", 1)[0]) or "", sys.maxsize),
                int(result.dimension.removesuffix("D")) if result.dimension.removesuffix("D").isdigit() else sys.maxsize,
                result.points is None,
                result.points or 0,
                result.benchmark_id,
            ),
        )
        return results

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
        results_by_key: dict[str, tuple[str, BenchmarkData]] = {}

        for estimates_file in sorted(criterion_dir.rglob("estimates.json")):
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
            key = f"{points}_{dimension}"

            bd = CriterionParser.parse_estimates_json(estimates_file, points, dimension)
            if bd:
                existing = results_by_key.get(key)
                if existing is None or (existing[0] == "base" and parent_name == "new"):
                    results_by_key[key] = (parent_name, bd)

        results = [benchmark for _, benchmark in results_by_key.values()]
        results.sort(key=lambda result: (int(result.dimension.rstrip("D")), result.points is None, result.points or 0))
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
        results: list[BenchmarkData] = []
        criterion_dir = target_dir / "criterion"

        if not criterion_dir.exists():
            return results

        results = CriterionParser._process_ci_performance_suite_results(criterion_dir)
        if results:
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

        # Sort by dimension, then by point count. Unsized benchmarks sort after
        # numeric workloads within the same dimension.
        results.sort(key=lambda x: (int(x.dimension.rstrip("D")), x.points is None, x.points or 0))
        return results


def _is_semver_tag_ref(ref_name: str) -> bool:
    """Return whether a git ref name is a release-style semver tag."""
    return re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?", ref_name) is not None


DISALLOWED_BASELINE_REF_PREFIXES = (
    "refs/pull/",
    "refs/merge-requests/",
    "refs/changes/",
    "pull/",
)
TRUSTED_BASELINE_BRANCH_RE = re.compile(
    r"(?:(?:codex|copilot|cursor)/)?"
    r"(?:main|(?:fix|feat|feature|perf|doc|docs|test|refactor|ci|build|chore|style|release)/[A-Za-z0-9][A-Za-z0-9._/-]*)"
)


def _normalize_baseline_ref_name(ref_name: str) -> str:
    """Normalize trusted fully qualified branch/tag refs to checkout-safe names."""
    if ref_name.startswith("refs/heads/"):
        return ref_name.removeprefix("refs/heads/")
    if ref_name.startswith("refs/tags/"):
        return ref_name.removeprefix("refs/tags/")
    return ref_name


def _validate_baseline_ref_name(ref_name: str) -> str:
    """Validate the workflow checkout ref and return the normalized ref name."""
    stripped = ref_name.strip()
    if not stripped:
        msg = "Baseline ref is empty after resolution"
        raise ValueError(msg)
    if stripped != ref_name or any(ch in stripped for ch in "\r\n"):
        msg = f"Disallowed baseline ref {ref_name!r}: refs may not contain surrounding whitespace or newlines"
        raise ValueError(msg)
    if any(stripped.startswith(prefix) for prefix in DISALLOWED_BASELINE_REF_PREFIXES):
        msg = f"Disallowed baseline ref {stripped!r}: untrusted ref namespace"
        raise ValueError(msg)

    normalized = _normalize_baseline_ref_name(stripped)
    if stripped.startswith("refs/") and normalized == stripped:
        msg = f"Disallowed baseline ref {stripped!r}: unsupported ref namespace"
        raise ValueError(msg)
    if _is_semver_tag_ref(normalized) or TRUSTED_BASELINE_BRANCH_RE.fullmatch(normalized):
        return normalized

    msg = f"Disallowed baseline ref {stripped!r} (resolved as {normalized!r}); allowed refs are main, semver release tags, and trusted branch prefixes"
    raise ValueError(msg)


class BaselineGenerator:
    """Generate performance baselines from benchmark data."""

    def __init__(self, project_root: Path, ref_name: str | None = None) -> None:
        """Initialize baseline generation for a project root and optional git ref."""
        self.project_root = project_root
        self.hardware = HardwareInfo()
        self.ref_name = ref_name

    def generate_baseline(self, dev_mode: bool = False, output_file: Path | None = None, bench_timeout: int = 1800) -> bool:
        """
        Generate a performance baseline by running benchmarks and parsing results.

        Args:
            dev_mode: Use faster Criterion settings with the trusted Cargo profile
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
                result = run_cargo_command(
                    [
                        "bench",
                        "--profile",
                        BENCHMARK_BUILD_FLAVOR,
                        "--bench",
                        "ci_performance_suite",
                        "--",
                        *DEV_MODE_BENCH_ARGS,
                    ],
                    cwd=self.project_root,
                    timeout=bench_timeout,
                    capture_output=True,
                )
            else:
                result = run_cargo_command(
                    ["bench", "--profile", BENCHMARK_BUILD_FLAVOR, "--bench", "ci_performance_suite"],
                    cwd=self.project_root,
                    timeout=bench_timeout,
                    capture_output=True,
                )
            _write_ci_performance_manifest_ids(self.project_root, result.stdout)

            # Parse Criterion results
            target_dir = self.project_root / "target"
            benchmark_results = CriterionParser.find_criterion_results(target_dir)

            if not benchmark_results:
                return False

            # Generate baseline file
            self._write_baseline_file(benchmark_results, output_file, dev_mode=dev_mode)

            return True

        except subprocess.TimeoutExpired as e:
            print(f"❌ Benchmark execution timed out after {bench_timeout} seconds", file=sys.stderr)
            print("   Consider increasing --bench-timeout or using --dev mode for faster benchmarks", file=sys.stderr)
            logger.debug("TimeoutExpired: %s", e)
            return False
        except subprocess.CalledProcessError as e:
            # Print captured stderr/stdout from cargo bench failure
            print("❌ Cargo bench failed with exit code:", e.returncode, file=sys.stderr)
            if e.stderr:
                print("\n=== cargo bench stderr ===", file=sys.stderr)
                print(e.stderr, file=sys.stderr)
                print("=== end stderr ===\n", file=sys.stderr)
            if e.stdout:
                print("\n=== cargo bench stdout ===", file=sys.stderr)
                print(e.stdout, file=sys.stderr)
                print("=== end stdout ===\n", file=sys.stderr)
            logger.exception("Error in generate_baseline")
            return False
        except _RECOVERABLE_CLI_ERRORS:
            logger.exception("Error in generate_baseline")
            return False

    def write_baseline_from_existing_results(self, output_file: Path, *, dev_mode: bool = False) -> bool:
        """
        Write a baseline file from existing Criterion results.

        This is intended for workflows that already ran `ci_performance_suite`
        through another command, such as the release performance summary. It
        avoids a duplicate benchmark run while preserving the baseline file
        format used by comparison tooling.
        """
        try:
            target_dir = self.project_root / "target"
            benchmark_results = CriterionParser.find_criterion_results(target_dir)

            if not benchmark_results:
                print(f"❌ No Criterion results found under {target_dir / 'criterion'}", file=sys.stderr)
                return False

            benchmark_results = [
                result for result in benchmark_results if result.benchmark_id and ci_suite_group_key(result.benchmark_id.split("/", maxsplit=1)[0]) is not None
            ]
            if not benchmark_results:
                print(f"❌ No ci_performance_suite Criterion results found under {target_dir / 'criterion'}", file=sys.stderr)
                return False

            self._write_baseline_file(benchmark_results, output_file, dev_mode=dev_mode)
            return True
        except _RECOVERABLE_CLI_ERRORS:
            logger.exception("Error in write_baseline_from_existing_results")
            return False

    def _write_baseline_file(self, benchmark_results: list[BenchmarkData], output_file: Path, *, dev_mode: bool = False) -> None:
        """Write baseline results to file."""
        # Get current date, git commit, and hardware info
        # Get current date with timezone
        now = datetime.now(UTC).astimezone()
        current_date = now.strftime("%Y-%m-%d %H:%M:%S %Z")

        try:
            # Use secure subprocess wrapper for git command
            git_commit = get_git_commit_hash(cwd=self.project_root)
        except _RECOVERABLE_CLI_ERRORS:
            git_commit = "unknown"

        hardware_info = self.hardware.format_hardware_info(cwd=self.project_root)

        # Write baseline file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            f.write(f"Date: {current_date}\n")
            f.write(f"Git commit: {git_commit}\n")
            if self.ref_name:
                f.write(f"Ref: {self.ref_name}\n")
            if self.ref_name and _is_semver_tag_ref(self.ref_name):
                f.write(f"Tag: {self.ref_name}\n")
            sampling = _sampling_metadata(dev_mode)
            f.write(f"Sampling mode: {sampling['sampling_mode']}\n")
            f.write(f"Cargo profile: {sampling['cargo_profile']}\n")
            f.write(f"Criterion args: {sampling['criterion_args']}\n")
            f.write(f"Criterion sample size: {sampling['criterion_sample_size']}\n")
            f.write(f"Criterion measurement time: {sampling['criterion_measurement_time']}\n")
            f.write(f"Criterion warm-up time: {sampling['criterion_warm_up_time']}\n")
            f.write(hardware_info)

            for benchmark in benchmark_results:
                f.write(benchmark.to_baseline_format())


class LocalRefBaselineGenerator:
    """Generate a same-machine performance baseline for a git ref."""

    def __init__(self, project_root: Path, *, remote: str = "origin") -> None:
        """Initialize local ref baseline generation from a project repository."""
        self.project_root = project_root
        self.remote = remote

    def generate_for_ref(
        self,
        *,
        ref_name: str,
        out_dir: Path,
        dev_mode: bool = False,
        bench_timeout: int = 1800,
    ) -> Path:
        """Generate a baseline for ref_name in a temporary checkout.

        The temporary checkout is always removed when this method returns or
        raises. Only the final baseline artifact files are written to out_dir.
        """
        remote_url = get_git_remote_url(remote=self.remote, cwd=self.project_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_file = out_dir / "baseline_results.txt"
        tmp_output_file = out_dir / "baseline_results.txt.tmp"
        tmp_output_file.unlink(missing_ok=True)

        with tempfile.TemporaryDirectory(prefix="delaunay-baseline-") as temp_dir:
            checkout_dir = Path(temp_dir) / "checkout"
            print(f"📥 Checking out {ref_name} from {self.remote} into a temporary directory...", file=sys.stderr)
            run_git_command(
                ["clone", "--no-checkout", "--filter=blob:none", remote_url, str(checkout_dir)],
                cwd=Path(temp_dir),
                timeout=300,
            )
            run_git_command(["fetch", "--depth", "1", "origin", ref_name], cwd=checkout_dir, timeout=300)
            run_git_command(["checkout", "--detach", "FETCH_HEAD"], cwd=checkout_dir, timeout=120)

            baseline_commit = get_git_commit_hash(cwd=checkout_dir)
            print(f"🚀 Generating local baseline for {ref_name} at {baseline_commit}...", file=sys.stderr)
            generator = BaselineGenerator(checkout_dir, ref_name=ref_name)
            success = generator.generate_baseline(dev_mode=dev_mode, output_file=tmp_output_file, bench_timeout=bench_timeout)

        if not success:
            tmp_output_file.unlink(missing_ok=True)
            msg = f"Failed to generate baseline for ref {ref_name}"
            raise RuntimeError(msg)

        tmp_output_file.replace(output_file)
        metadata_success = WorkflowHelper.create_metadata(
            ref_name,
            out_dir,
            BaselineArtifactMetadata(
                commit_sha=baseline_commit,
                run_id="local",
                runner_os=platform.system() or "unknown",
                runner_arch=platform.machine() or "unknown",
            ),
        )
        if not metadata_success:
            msg = f"Failed to write metadata for baseline ref {ref_name}"
            raise RuntimeError(msg)

        print(f"✅ Local baseline ready: {output_file}", file=sys.stderr)
        return output_file


@dataclass(frozen=True)
class LocalRefBaselineCacheOptions:
    """Options for a cached same-machine baseline generated from a git ref."""

    ref_name: str = "main"
    remote: str = "origin"
    cache_root: Path | None = None
    dev_mode: bool = False
    bench_timeout: int = 1800
    required_benchmark_id: str = PERF_NO_REGRESSIONS_REQUIRED_BENCHMARK_ID


@dataclass(frozen=True)
class LocalRefBaselineCacheResult:
    """Result of ensuring a cached same-machine ref baseline exists."""

    baseline_path: Path
    resolved_commit: str | None
    reused: bool


def _sanitize_cache_component(value: str, *, fallback: str) -> str:
    """Return a stable filesystem-safe cache component."""
    sanitized = _sanitize_ref_name(value.strip())
    return sanitized or fallback


def release_comparison_results_path(project_root: Path) -> Path:
    """Return the release-baseline comparison report path."""
    return project_root / "benches" / MAIN_VS_RELEASE_COMPARISON_RESULTS_FILE


def ref_comparison_results_path(project_root: Path, ref_name: str) -> Path:
    """Return the worktree-vs-ref comparison report path for a git ref."""
    ref_key = _sanitize_cache_component(ref_name, fallback="ref")
    return project_root / "benches" / WORKTREE_VS_REF_COMPARISON_RESULTS_TEMPLATE.format(ref=ref_key)


def _first_ls_remote_commit(stdout: str) -> str | None:
    """Extract the first object id from git ls-remote output."""
    for line in stdout.splitlines():
        parts = line.split()
        if parts and re.fullmatch(r"[0-9a-fA-F]+", parts[0]):
            return parts[0]
    return None


def _remote_ref_candidates(ref_name: str) -> list[str]:
    """Return deterministic ls-remote candidates for a branch, tag, or full ref."""
    if ref_name.startswith("refs/"):
        return [ref_name]
    return [
        f"refs/heads/{ref_name}",
        f"refs/tags/{ref_name}^{{}}",
        f"refs/tags/{ref_name}",
        ref_name,
    ]


def _local_tracking_ref_candidates(remote: str, ref_name: str) -> list[str]:
    """Return local remote-tracking refs that can stand in when offline."""
    if ref_name.startswith("refs/heads/"):
        branch = ref_name.removeprefix("refs/heads/")
    elif ref_name.startswith("refs/"):
        return []
    else:
        branch = ref_name
    return [f"refs/remotes/{remote}/{branch}"]


def resolve_ref_commit(project_root: Path, *, ref_name: str, remote: str = "origin") -> str | None:
    """Resolve a remote git ref to a commit-ish object id, falling back to local tracking refs."""
    for candidate in _remote_ref_candidates(ref_name):
        result = run_git_command(
            ["ls-remote", remote, candidate],
            cwd=project_root,
            check=False,
            timeout=120,
        )
        if result.returncode == 0:
            commit = _first_ls_remote_commit(result.stdout)
            if commit is not None:
                return commit
        else:
            logger.debug("git ls-remote failed for %s/%s: %s", remote, candidate, (result.stderr or result.stdout or "").strip())
            break

    for candidate in _local_tracking_ref_candidates(remote, ref_name):
        result = run_git_command(
            ["rev-parse", "--verify", "--quiet", candidate],
            cwd=project_root,
            check=False,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()

    return None


def _local_rustc_version(project_root: Path) -> str:
    """Return the local rustc version used to key same-machine benchmark caches."""
    try:
        result = run_safe_command("rustc", ["-V"], cwd=project_root, check=False, timeout=30)
    except (ExecutableNotFoundError, OSError, subprocess.SubprocessError):
        return "unknown-rustc"
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return "unknown-rustc"


def _default_local_ref_baseline_cache_root(project_root: Path) -> Path:
    """Default cache root for local same-machine ref baselines."""
    if env_cache_root := os.getenv("DELAUNAY_PERF_BASELINE_CACHE"):
        cache_root = Path(env_cache_root)
        return cache_root if cache_root.is_absolute() else project_root / cache_root
    return project_root / "baseline-artifacts" / "perf-no-regressions"


def _local_ref_baseline_cache_dir(
    project_root: Path,
    options: LocalRefBaselineCacheOptions,
    *,
    resolved_commit: str | None,
) -> Path:
    """Return the deterministic cache directory for a local ref baseline."""
    cache_root = options.cache_root or _default_local_ref_baseline_cache_root(project_root)
    if not cache_root.is_absolute():
        cache_root = project_root / cache_root

    ref_key = _sanitize_cache_component(options.ref_name, fallback="ref")
    commit_key = _sanitize_cache_component(resolved_commit or options.ref_name, fallback="unresolved")
    mode_key = "dev" if options.dev_mode else "full"
    toolchain_key = _sanitize_cache_component(_local_rustc_version(project_root), fallback="unknown-rustc")
    return cache_root / ref_key / commit_key / mode_key / toolchain_key


def _local_ref_baseline_candidates(
    project_root: Path,
    options: LocalRefBaselineCacheOptions,
    *,
    resolved_commit: str | None,
) -> list[Path]:
    """Return primary and commit-alias cache candidates for a local ref baseline."""
    primary = _local_ref_baseline_cache_dir(project_root, options, resolved_commit=resolved_commit) / "baseline_results.txt"
    if resolved_commit is None:
        return [primary]

    cache_root = options.cache_root or _default_local_ref_baseline_cache_root(project_root)
    if not cache_root.is_absolute():
        cache_root = project_root / cache_root

    commit_key = _sanitize_cache_component(resolved_commit, fallback="unresolved")
    mode_key = "dev" if options.dev_mode else "full"
    toolchain_key = _sanitize_cache_component(_local_rustc_version(project_root), fallback="unknown-rustc")
    alias_pattern = f"*/{commit_key}/{mode_key}/{toolchain_key}/baseline_results.txt"
    aliases = sorted(cache_root.glob(alias_pattern)) if cache_root.exists() else []

    candidates = [primary]
    candidates.extend(alias for alias in aliases if alias != primary)
    return candidates


def _cached_baseline_valid(
    project_root: Path,
    baseline_path: Path,
    *,
    expected_commit: str | None,
    required_benchmark_id: str,
) -> tuple[bool, str]:
    """Validate cached baseline metadata and parseability before reuse."""
    if not baseline_path.exists():
        return False, f"missing baseline file: {baseline_path}"

    try:
        baseline_content = baseline_path.read_text(encoding="utf-8")
    except OSError as exc:
        return False, f"unable to read baseline file {baseline_path}: {exc}"

    metadata = _parse_baseline_metadata(baseline_content)
    if expected_commit is not None and metadata["commit"] != expected_commit:
        return False, f"cached commit {metadata['commit']} does not match expected {expected_commit}"

    try:
        baseline_results = PerformanceComparator(project_root).parse_baseline_file(baseline_content)
    except BaselineParseError as exc:
        return False, f"malformed baseline: {exc}"

    if not any(benchmark.benchmark_id == required_benchmark_id for benchmark in baseline_results.values()):
        return False, f"missing required benchmark id {required_benchmark_id}"

    return True, "valid"


def ensure_cached_ref_baseline(
    project_root: Path,
    options: LocalRefBaselineCacheOptions,
    *,
    resolved_commit: str | None,
) -> LocalRefBaselineCacheResult:
    """Ensure a cached same-machine baseline exists for a resolved git ref."""
    baseline_dir = _local_ref_baseline_cache_dir(project_root, options, resolved_commit=resolved_commit)
    reason = "no cache candidates checked"
    for baseline_path in _local_ref_baseline_candidates(project_root, options, resolved_commit=resolved_commit):
        valid, reason = _cached_baseline_valid(
            project_root,
            baseline_path,
            expected_commit=resolved_commit,
            required_benchmark_id=options.required_benchmark_id,
        )
        if not valid:
            continue

        print(f"📦 Reusing cached {options.ref_name} baseline: {baseline_path}", file=sys.stderr)
        return LocalRefBaselineCacheResult(baseline_path=baseline_path, resolved_commit=resolved_commit, reused=True)

    print(f"🚀 Refreshing cached {options.ref_name} baseline ({reason})...", file=sys.stderr)
    generator = LocalRefBaselineGenerator(project_root, remote=options.remote)
    generated_path = generator.generate_for_ref(
        ref_name=options.ref_name,
        out_dir=baseline_dir,
        dev_mode=options.dev_mode,
        bench_timeout=options.bench_timeout,
    )

    valid, reason = _cached_baseline_valid(
        project_root,
        generated_path,
        expected_commit=resolved_commit,
        required_benchmark_id=options.required_benchmark_id,
    )
    if not valid:
        msg = f"Generated baseline for {options.ref_name} is not reusable: {reason}"
        raise RuntimeError(msg)

    return LocalRefBaselineCacheResult(baseline_path=generated_path, resolved_commit=resolved_commit, reused=False)


def ensure_cached_ref_baseline_for_ref(project_root: Path, options: LocalRefBaselineCacheOptions) -> LocalRefBaselineCacheResult:
    """Resolve a ref and ensure its cached same-machine baseline exists."""
    resolved_commit = resolve_ref_commit(project_root, ref_name=options.ref_name, remote=options.remote)
    if resolved_commit is None:
        print(f"⚠️ Could not resolve {options.remote}/{options.ref_name}; cache freshness cannot be verified.", file=sys.stderr)
    return ensure_cached_ref_baseline(project_root, options, resolved_commit=resolved_commit)


def relevant_perf_worktree_dirty(project_root: Path, paths: tuple[str, ...] = PERF_NO_REGRESSIONS_RELEVANT_PATHS) -> bool:
    """Return whether performance-relevant tracked or untracked paths changed."""
    diff_args = ["diff", "--quiet", "--", *paths]
    for label, args in (
        ("unstaged diff", diff_args),
        ("staged diff", ["diff", "--cached", "--quiet", "--", *paths]),
    ):
        result = run_git_command(args, cwd=project_root, check=False, timeout=60)
        if result.returncode == 1:
            return True
        if result.returncode != 0:
            details = (result.stderr or result.stdout or "").strip()
            msg = f"git {label} failed with exit code {result.returncode}: {details}"
            raise RuntimeError(msg)

    result = run_git_command(
        ["ls-files", "--others", "--exclude-standard", "--", *paths],
        cwd=project_root,
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        msg = f"git ls-files for untracked perf paths failed with exit code {result.returncode}: {details}"
        raise RuntimeError(msg)
    return bool(result.stdout.strip())


def compare_with_cached_ref_baseline(
    project_root: Path,
    options: LocalRefBaselineCacheOptions,
    *,
    threshold: float,
    output_file: Path | None = None,
) -> int:
    """Compare the current worktree against a cached same-machine ref baseline."""
    current_commit = get_git_commit_hash(cwd=project_root)
    dirty = relevant_perf_worktree_dirty(project_root)
    resolved_commit = resolve_ref_commit(project_root, ref_name=options.ref_name, remote=options.remote)

    if resolved_commit == current_commit and not dirty:
        print(f"🔍 {options.remote}/{options.ref_name} matches HEAD ({current_commit}); no relevant worktree changes to compare.")
        print("   Skipping before generating a same-commit baseline.")
        return 0

    cache_result = ensure_cached_ref_baseline(project_root, options, resolved_commit=resolved_commit)
    baseline_content = cache_result.baseline_path.read_text(encoding="utf-8")
    baseline_commit = _parse_baseline_metadata(baseline_content)["commit"]

    if baseline_commit == current_commit:
        if not dirty:
            print(f"🔍 Current commit matches the {options.ref_name} baseline ({baseline_commit}); no relevant worktree changes to compare.")
            print("   Skipping because a same-commit baseline would mask regressions.")
            return 0
        print(f"⚠️ {options.ref_name} baseline commit matches HEAD, but relevant uncommitted changes exist; comparing the worktree against HEAD.")

    if output_file is None:
        output_file = ref_comparison_results_path(project_root, options.ref_name)

    comparator = PerformanceComparator(project_root)
    comparator.regression_threshold = threshold
    success, regression_found = comparator.compare_with_baseline(
        cache_result.baseline_path,
        dev_mode=options.dev_mode,
        output_file=output_file,
        failure_policy="total-time",
        bench_timeout=options.bench_timeout,
    )
    _display_comparison_result(output_file, success=success, regression_found=regression_found)
    if not success:
        return 1
    return 1 if regression_found else 0


def _display_comparison_result(output_file: Path, *, success: bool, regression_found: bool) -> None:
    """Print the comparison outcome and report path for command-line users."""
    if not success:
        print(f"❌ Benchmark comparison failed; see {output_file}", file=sys.stderr)
        return

    if regression_found:
        print(f"⚠️ Performance regressions detected; see {output_file}", file=sys.stderr)
        return

    try:
        report_text = output_file.read_text(encoding="utf-8")
    except OSError:
        report_text = ""
    if "INDIVIDUAL REGRESSION WARNING" in report_text:
        print(f"✅ Net performance OK; individual regression warnings in report: {output_file}")
        return

    print(f"✅ No significant performance regressions detected; report: {output_file}")


class PerformanceComparator:
    """Compare current performance against baseline."""

    def __init__(self, project_root: Path) -> None:
        """Initialize comparison state for benchmark results under a project root."""
        self.project_root = project_root
        self.hardware = HardwareInfo()
        self.regression_threshold = DEFAULT_REGRESSION_THRESHOLD  # default threshold for proactive regression detection in CI
        logger.debug(
            "PerformanceComparator initialized with regression_threshold=%s for project_root=%s",
            self.regression_threshold,
            project_root,
        )

    def compare_with_baseline(
        self,
        baseline_file: Path,
        dev_mode: bool = False,
        output_file: Path | None = None,
        bench_timeout: int = 1800,
        failure_policy: ComparisonFailurePolicy = "strict",
    ) -> tuple[bool, bool]:
        """
        Compare current performance against baseline.

        Args:
            baseline_file: Path to baseline file
            dev_mode: Use faster Criterion settings with the trusted Cargo profile
            output_file: Output file path (default: benches/main_vs_release_compare_results.txt)
            bench_timeout: Timeout for cargo bench commands in seconds
            failure_policy: Regression policy for deciding the command exit status

        Returns:
            Tuple of (success, regression_found)
        """
        if output_file is None:
            output_file = release_comparison_results_path(self.project_root)

        if not baseline_file.exists():
            self._write_error_file(output_file, "Baseline file not found", baseline_file)
            return False, False

        try:
            # Run fresh benchmark - using secure subprocess wrapper
            if dev_mode:
                result = run_cargo_command(
                    [
                        "bench",
                        "--profile",
                        BENCHMARK_BUILD_FLAVOR,
                        "--bench",
                        "ci_performance_suite",
                        "--",
                        *DEV_MODE_BENCH_ARGS,
                    ],
                    cwd=self.project_root,
                    timeout=bench_timeout,
                    capture_output=True,
                )
            else:
                result = run_cargo_command(
                    ["bench", "--profile", BENCHMARK_BUILD_FLAVOR, "--bench", "ci_performance_suite"],
                    cwd=self.project_root,
                    timeout=bench_timeout,
                    capture_output=True,
                )
            _write_ci_performance_manifest_ids(self.project_root, result.stdout)

            # Parse current results
            target_dir = self.project_root / "target"
            current_results = CriterionParser.find_criterion_results(target_dir)

            if not current_results:
                self._write_error_file(output_file, "No benchmark results found", target_dir / "criterion")
                return False, False

            # Parse baseline
            baseline_content = baseline_file.read_text(encoding="utf-8")
            baseline_results = self._parse_baseline_file(baseline_content)

            # Generate comparison report
            regression_found = self._write_comparison_file(
                current_results,
                baseline_results,
                ComparisonFileRequest(
                    baseline_content=baseline_content,
                    output_file=output_file,
                    dev_mode=dev_mode,
                    failure_policy=failure_policy,
                ),
            )

            return True, regression_found

        except subprocess.TimeoutExpired as e:
            print(f"❌ Benchmark execution timed out after {bench_timeout} seconds", file=sys.stderr)
            print("   Consider increasing --bench-timeout or using --dev mode for faster benchmarks", file=sys.stderr)
            logger.debug("TimeoutExpired: %s", e)
            self._write_error_file(output_file, "Benchmark execution timeout", f"{e} (timeout after {bench_timeout} seconds)")
            return False, False
        except subprocess.CalledProcessError as e:
            # Print captured stderr/stdout from cargo bench failure
            print("❌ Cargo bench failed with exit code:", e.returncode, file=sys.stderr)
            if e.stderr:
                print("\n=== cargo bench stderr ===", file=sys.stderr)
                print(e.stderr, file=sys.stderr)
                print("=== end stderr ===\n", file=sys.stderr)
            if e.stdout:
                print("\n=== cargo bench stdout ===", file=sys.stderr)
                print(e.stdout, file=sys.stderr)
                print("=== end stdout ===\n", file=sys.stderr)
            self._write_error_file(output_file, "Benchmark execution error", str(e))
            logger.exception("Error in compare_with_baseline")
            return False, False
        except _RECOVERABLE_CLI_ERRORS as e:
            self._write_error_file(output_file, "Benchmark execution error", str(e))
            logger.exception("Error in compare_with_baseline")
            return False, False

    def _parse_baseline_file(self, baseline_content: str) -> dict[str, BenchmarkData]:
        """Parse baseline file content into benchmark data."""
        results = {}
        for benchmark in extract_benchmark_data(baseline_content):
            if not benchmark.time_unit:
                section_label = benchmark.benchmark_id or benchmark.header_line()
                msg = f"Malformed baseline section {section_label!r}: missing or invalid Time line"
                raise BaselineParseError(msg)
            try:
                results[benchmark.comparison_key] = benchmark
            except ValueError as exc:
                section_label = benchmark.benchmark_id or benchmark.header_line()
                msg = f"Malformed baseline section {section_label!r}: {exc}"
                raise BaselineParseError(msg) from exc
        return results

    def parse_baseline_file(self, baseline_content: str) -> dict[str, BenchmarkData]:
        """Public wrapper for parsing a baseline file."""
        return self._parse_baseline_file(baseline_content)

    def write_performance_comparison(self, f: TextIO, current_results: list[BenchmarkData], baseline_results: dict[str, BenchmarkData]) -> bool:
        """Public wrapper for writing the performance comparison section.

        Returns:
            True if the selected failure policy detects a regression exceeding the
            regression threshold.
        """
        return self._write_performance_comparison(f, current_results, baseline_results)

    def _write_comparison_file(
        self,
        current_results: list[BenchmarkData],
        baseline_results: dict[str, BenchmarkData],
        request: ComparisonFileRequest,
    ) -> bool:
        """Write comparison results to file."""
        logger.debug(
            "Writing performance comparison: threshold=%.2f current_results=%s baseline_entries=%s",
            self.regression_threshold,
            len(current_results),
            len(baseline_results),
        )
        # Prepare metadata
        metadata = self._prepare_comparison_metadata(request.baseline_content)

        # Prepare hardware comparison
        hardware_report = self._prepare_hardware_comparison(request.baseline_content)
        sampling_warning = self._sampling_warning(request.baseline_content, dev_mode=request.dev_mode)

        # Write comparison file
        request.output_file.parent.mkdir(parents=True, exist_ok=True)
        with request.output_file.open("w", encoding="utf-8") as f:
            self._write_comparison_header(f, metadata, hardware_report, sampling_warning=sampling_warning)
            return self._write_performance_comparison(
                f,
                current_results,
                baseline_results,
                failure_policy=request.failure_policy,
            )

    def _prepare_comparison_metadata(self, baseline_content: str) -> dict[str, str]:
        """Prepare metadata for comparison report."""
        # Get current date with timezone
        now = datetime.now(UTC).astimezone()
        current_date = now.strftime("%a %b %d %H:%M:%S %Z %Y")

        try:
            git_commit = get_git_commit_hash(cwd=self.project_root)
        except _RECOVERABLE_CLI_ERRORS:
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
        current_hardware = self.hardware.get_hardware_info(cwd=self.project_root)
        baseline_hardware = HardwareComparator.parse_baseline_hardware(baseline_content)
        hardware_report, _ = HardwareComparator.compare_hardware(current_hardware, baseline_hardware)
        return hardware_report

    @staticmethod
    def _parse_sampling_metadata(baseline_content: str) -> dict[str, str]:
        """Extract benchmark sampling metadata from a baseline file."""
        fields = {
            "sampling_mode": "Unknown",
            "cargo_profile": "Unknown",
            "criterion_sample_size": "Unknown",
            "criterion_measurement_time": "Unknown",
            "criterion_warm_up_time": "Unknown",
        }
        line_map = {
            "Sampling mode: ": "sampling_mode",
            "Cargo profile: ": "cargo_profile",
            "Criterion sample size: ": "criterion_sample_size",
            "Criterion measurement time: ": "criterion_measurement_time",
            "Criterion warm-up time: ": "criterion_warm_up_time",
        }

        for line in baseline_content.splitlines():
            for prefix, field in line_map.items():
                if line.startswith(prefix):
                    fields[field] = line.removeprefix(prefix).strip()
                    break

        return fields

    def _sampling_warning(self, baseline_content: str, *, dev_mode: bool) -> str:
        """Return a warning when current benchmark sampling differs from baseline."""
        baseline = self._parse_sampling_metadata(baseline_content)
        current = _sampling_metadata(dev_mode)
        checks = [
            ("sampling mode", "sampling_mode"),
            ("Cargo profile", "cargo_profile"),
            ("Criterion sample size", "criterion_sample_size"),
            ("Criterion measurement time", "criterion_measurement_time"),
            ("Criterion warm-up time", "criterion_warm_up_time"),
        ]

        mismatches = []
        for label, field in checks:
            baseline_value = baseline[field]
            if baseline_value == "Unknown" or baseline_value != current[field]:
                mismatches.append(f"{label}: baseline={baseline_value}, current={current[field]}")

        if not mismatches:
            return ""

        return "⚠️ Sampling configuration differs from baseline: " + "; ".join(mismatches)

    def _write_comparison_header(self, f, metadata: dict[str, str], hardware_report: str, *, sampling_warning: str = "") -> None:
        """Write the header section of comparison file."""
        f.write("Comparison Results\n")
        f.write("==================\n")
        f.write(f"Current Date: {metadata['current_date']}\n")
        f.write(f"Current Git commit: {metadata['current_commit']}\n\n")
        f.write(f"Baseline Date: {metadata['baseline_date']}\n")
        f.write(f"Baseline Git commit: {metadata['baseline_commit']}\n\n")
        if sampling_warning:
            f.write(f"{sampling_warning}\n\n")
        f.write(hardware_report)

    @staticmethod
    def _matching_baseline(current: BenchmarkData, baseline_results: dict[str, BenchmarkData]) -> BenchmarkData | None:
        """Return the matching baseline entry, using legacy keys only for legacy current IDs."""
        baseline_benchmark = baseline_results.get(current.comparison_key)
        if baseline_benchmark is not None or current.benchmark_id:
            return baseline_benchmark
        if current.points is None:
            return None
        return baseline_results.get(f"{current.points}_{current.dimension}")

    def _write_performance_comparison(
        self,
        f: TextIO,
        current_results: list[BenchmarkData],
        baseline_results: dict[str, BenchmarkData],
        *,
        failure_policy: ComparisonFailurePolicy = "strict",
    ) -> bool:
        """Write performance comparison section and return whether any regression exceeds threshold."""
        time_changes: list[BenchmarkTimeChange] = []
        individual_regressions = 0
        individual_improvements = 0

        for current_benchmark in current_results:
            baseline_benchmark = self._matching_baseline(current_benchmark, baseline_results)

            self._write_benchmark_header(f, current_benchmark)
            self._write_current_benchmark_data(f, current_benchmark)

            if baseline_benchmark:
                self._write_baseline_benchmark_data(f, baseline_benchmark)
                time_change, is_individual_regression = self._write_time_comparison(f, current_benchmark, baseline_benchmark)
                if time_change is not None:
                    mean_times = self._mean_times_us(current_benchmark, baseline_benchmark)
                    if mean_times is not None:
                        current_mean_us, baseline_mean_us = mean_times
                        time_changes.append(
                            BenchmarkTimeChange(
                                label=self._comparison_label(current_benchmark),
                                current_mean_us=current_mean_us,
                                baseline_mean_us=baseline_mean_us,
                                time_change_pct=time_change,
                            ),
                        )
                    if is_individual_regression:
                        individual_regressions += 1
                    elif time_change < -self.regression_threshold:
                        individual_improvements += 1
                self._write_throughput_comparison(f, current_benchmark, baseline_benchmark)
            else:
                f.write("Baseline: N/A (no matching entry)\n")

            f.write("\n")

        if time_changes:
            total_current_us = sum(change.current_mean_us for change in time_changes)
            total_baseline_us = sum(change.baseline_mean_us for change in time_changes)
            total_time_change = ((total_current_us - total_baseline_us) / total_baseline_us) * 100.0
            geomean_change = self._geomean_time_change(time_changes)
            median_change = self._median_time_change(time_changes)

            f.write("\n=== SUMMARY ===\n")
            f.write(f"Total benchmarks compared: {len(time_changes)}\n")
            f.write(f"Individual regressions (>{self.regression_threshold}%): {individual_regressions}\n")
            f.write(f"Individual improvements (>{self.regression_threshold}%): {individual_improvements}\n")
            f.write(f"Total baseline matched mean time: {total_baseline_us:.3f} µs\n")
            f.write(f"Total current matched mean time: {total_current_us:.3f} µs\n")
            f.write(f"Total time change: {total_time_change:+.1f}%\n")
            f.write(f"Geomean time change: {geomean_change:+.1f}%\n")
            f.write(f"Median time change: {median_change:+.1f}%\n")
            self._write_top_time_changes(f, "Top regressions", self._top_regressions(time_changes))
            self._write_top_time_changes(f, "Top improvements", self._top_improvements(time_changes))

            regression_found = self._write_summary_status(
                f,
                ComparisonSummaryStats(
                    total_time_change=total_time_change,
                    geomean_change=geomean_change,
                    median_change=median_change,
                    individual_regressions=individual_regressions,
                    compared_count=len(time_changes),
                    failure_policy=failure_policy,
                ),
            )

            logger.debug(
                "Performance comparison summary: policy=%s total_change=%.2f%% geomean_change=%.2f%% median_change=%.2f%% individual_regressions=%s",
                failure_policy,
                total_time_change,
                geomean_change,
                median_change,
                individual_regressions,
            )

            f.write("\n")
            return regression_found

        return False

    @staticmethod
    def _geomean_time_change(time_changes: list[BenchmarkTimeChange]) -> float:
        """Return the geometric mean time change across matched benchmarks."""
        ratios = [1.0 + (change.time_change_pct / 100.0) for change in time_changes]
        positive_ratios = [ratio for ratio in ratios if ratio > 0.0]
        if not positive_ratios:
            return 0.0
        avg_log = sum(math.log(ratio) for ratio in positive_ratios) / len(positive_ratios)
        return (math.exp(avg_log) - 1.0) * 100.0

    @staticmethod
    def _median_time_change(time_changes: list[BenchmarkTimeChange]) -> float:
        """Return the median time change across matched benchmarks."""
        sorted_changes = sorted(change.time_change_pct for change in time_changes)
        midpoint = len(sorted_changes) // 2
        if len(sorted_changes) % 2 == 1:
            return sorted_changes[midpoint]
        return (sorted_changes[midpoint - 1] + sorted_changes[midpoint]) / 2.0

    def _top_regressions(self, time_changes: list[BenchmarkTimeChange]) -> list[BenchmarkTimeChange]:
        """Return the largest individual slowdowns beyond the regression threshold."""
        regressions = [change for change in time_changes if change.time_change_pct > self.regression_threshold]
        return sorted(regressions, key=lambda change: change.time_change_pct, reverse=True)[:5]

    def _top_improvements(self, time_changes: list[BenchmarkTimeChange]) -> list[BenchmarkTimeChange]:
        """Return the largest individual speedups beyond the improvement threshold."""
        improvements = [change for change in time_changes if change.time_change_pct < -self.regression_threshold]
        return sorted(improvements, key=lambda change: change.time_change_pct)[:5]

    @staticmethod
    def _write_top_time_changes(f: TextIO, title: str, changes: list[BenchmarkTimeChange]) -> None:
        """Write a compact top-N timing change list."""
        if not changes:
            return
        f.write(f"{title}:\n")
        f.writelines(f"- {change.label}: {change.time_change_pct:+.1f}%\n" for change in changes)

    def _write_summary_status(self, f: TextIO, summary: ComparisonSummaryStats) -> bool:
        """Write the summary status line and return whether the comparison failed."""
        total_regression_found = summary.total_time_change > self.regression_threshold
        if total_regression_found:
            f.write(
                f"🚨 OVERALL REGRESSION: Total matched benchmark time increased by {summary.total_time_change:.1f}% "
                f"(exceeds {self.regression_threshold}% threshold)\n",
            )
            logger.warning(
                "Total-time regression detected: total_time_change=%.2f%% threshold=%.2f%% benchmarks=%s geomean=%.2f%% median=%.2f%%",
                summary.total_time_change,
                self.regression_threshold,
                summary.compared_count,
                summary.geomean_change,
                summary.median_change,
            )
            return True

        if summary.individual_regressions > 0:
            if summary.failure_policy == "total-time":
                f.write(
                    f"⚠️ INDIVIDUAL REGRESSION WARNING: {summary.individual_regressions} benchmark(s) exceeded "
                    f"{self.regression_threshold}% threshold while total matched time changed by {summary.total_time_change:.1f}%\n",
                )
                logger.warning(
                    "Individual regressions warning under total-time policy: individual_regressions=%s total_time_change=%.2f%% threshold=%.2f%% benchmarks=%s",
                    summary.individual_regressions,
                    summary.total_time_change,
                    self.regression_threshold,
                    summary.compared_count,
                )
                return False

            f.write(
                f"⚠️ INDIVIDUAL REGRESSION: {summary.individual_regressions} benchmark(s) exceeded "
                f"{self.regression_threshold}% threshold while total matched time changed by {summary.total_time_change:.1f}%\n",
            )
            logger.warning(
                "Individual regression detected: individual_regressions=%s total_time_change=%.2f%% threshold=%.2f%% benchmarks=%s",
                summary.individual_regressions,
                summary.total_time_change,
                self.regression_threshold,
                summary.compared_count,
            )
            return True

        if summary.total_time_change < -self.regression_threshold:
            f.write(
                f"🎉 OVERALL IMPROVEMENT: Total matched benchmark time improved by {abs(summary.total_time_change):.1f}% "
                f"(exceeds {self.regression_threshold}% threshold)\n",
            )
            logger.info(
                "Total-time improvement detected: total_time_change=%.2f%% threshold=%.2f%% benchmarks=%s",
                summary.total_time_change,
                self.regression_threshold,
                summary.compared_count,
            )
            return False

        f.write(f"✅ OVERALL OK: Total matched time change within acceptable range (±{self.regression_threshold}%)\n")
        logger.debug(
            "Total-time change within threshold: total_time_change=%.2f%% threshold=%.2f%% benchmarks=%s",
            summary.total_time_change,
            self.regression_threshold,
            summary.compared_count,
        )
        return False

    @staticmethod
    def _comparison_label(benchmark: BenchmarkData) -> str:
        """Return a stable label for summary timing change lists."""
        return benchmark.benchmark_id or f"{benchmark.points}_{benchmark.dimension}"

    @staticmethod
    def _mean_time_us(benchmark: BenchmarkData) -> float | None:
        """Return the benchmark mean time in microseconds when its unit is supported."""
        unit = benchmark.time_unit or "µs"
        scale = TIME_UNIT_TO_MICROSECONDS.get(unit)
        if scale is None:
            return None
        return benchmark.time_mean * scale

    def _mean_times_us(self, current: BenchmarkData, baseline: BenchmarkData) -> tuple[float, float] | None:
        """Return normalized current and baseline mean times for a valid comparison."""
        if baseline.time_mean <= 0:
            return None
        cur_mean_us = self._mean_time_us(current)
        base_mean_us = self._mean_time_us(baseline)
        if cur_mean_us is None or base_mean_us is None or base_mean_us <= 0:
            return None
        return cur_mean_us, base_mean_us

    def _write_benchmark_header(self, f, benchmark: BenchmarkData) -> None:
        """Write benchmark section header."""
        f.write(f"{benchmark.header_line()}\n")
        if benchmark.benchmark_id:
            f.write(f"Benchmark ID: {benchmark.benchmark_id}\n")

    def _write_current_benchmark_data(self, f, benchmark: BenchmarkData) -> None:
        """Write current benchmark data."""
        f.write(f"Current Time: [{benchmark.time_low}, {benchmark.time_mean}, {benchmark.time_high}] {benchmark.time_unit}\n")
        if benchmark.throughput_mean is not None:
            f.write(
                f"Current Throughput: [{benchmark.throughput_low}, {benchmark.throughput_mean}, {benchmark.throughput_high}] {benchmark.throughput_unit}\n",
            )

    def _write_baseline_benchmark_data(self, f, benchmark: BenchmarkData) -> None:
        """Write baseline benchmark data."""
        f.write(f"Baseline Time: [{benchmark.time_low}, {benchmark.time_mean}, {benchmark.time_high}] {benchmark.time_unit}\n")
        if benchmark.throughput_mean is not None:
            f.write(
                f"Baseline Throughput: [{benchmark.throughput_low}, {benchmark.throughput_mean}, {benchmark.throughput_high}] {benchmark.throughput_unit}\n",
            )

    def _write_time_comparison(self, f, current: BenchmarkData, baseline: BenchmarkData) -> tuple[float | None, bool]:
        """Write time comparison and return time change percentage and whether individual regression was found."""
        if baseline.time_mean <= 0:
            f.write("Time Change: N/A (baseline mean is 0)\n")
            return None, False
        cur_unit = current.time_unit or "µs"
        base_unit = baseline.time_unit or "µs"
        if cur_unit not in TIME_UNIT_TO_MICROSECONDS or base_unit not in TIME_UNIT_TO_MICROSECONDS:
            f.write(f"Time Change: N/A (unit mismatch: {cur_unit} vs {base_unit})\n")
            return None, False
        mean_times = self._mean_times_us(current, baseline)
        if mean_times is None:
            f.write("Time Change: N/A (baseline mean is 0)\n")
            return None, False
        cur_mean_us, base_mean_us = mean_times

        time_change_pct = ((cur_mean_us - base_mean_us) / base_mean_us) * 100
        is_individual_regression = time_change_pct > self.regression_threshold

        logger.debug(
            "Benchmark %s_%s comparison: current_mean=%.3fµs baseline_mean=%.3fµs change=%.2f%% threshold=%.2f%%",
            current.points,
            current.dimension,
            cur_mean_us,
            base_mean_us,
            time_change_pct,
            self.regression_threshold,
        )

        if is_individual_regression:
            f.write(f"⚠️  REGRESSION: Time increased by {time_change_pct:.1f}% (slower performance)\n")
            logger.warning(
                "Individual regression detected for %s_%s: change=%.2f%% exceeds threshold=%.2f%%",
                current.points,
                current.dimension,
                time_change_pct,
                self.regression_threshold,
            )
        elif time_change_pct < -self.regression_threshold:
            f.write(f"✅ IMPROVEMENT: Time decreased by {abs(time_change_pct):.1f}% (faster performance)\n")
            logger.info(
                "Individual improvement detected for %s_%s: change=%.2f%% beyond threshold=%.2f%%",
                current.points,
                current.dimension,
                time_change_pct,
                self.regression_threshold,
            )
        else:
            f.write(f"✅ OK: Time change {time_change_pct:+.1f}% within acceptable range\n")
            logger.debug(
                "Benchmark %s_%s within acceptable range: change=%.2f%% threshold=%.2f%%",
                current.points,
                current.dimension,
                time_change_pct,
                self.regression_threshold,
            )

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

    def _write_error_file(self, output_file: Path, error_title: str, error_detail: str | Path) -> None:
        """Write an error message to the comparison results file."""
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with output_file.open("w", encoding="utf-8") as f:
                f.write("Comparison Results\n")
                f.write("==================\n\n")
                f.write(f"❌ Error: {error_title}\n\n")
                f.write(f"Details: {error_detail}\n\n")
                f.write("This error prevented the benchmark comparison from completing successfully.\n")
                f.write("Please check the CI logs for more information.\n")
        except OSError:
            logger.exception("Failed to write error file")


class WorkflowHelper:
    """Helper functions for GitHub Actions workflow integration."""

    @staticmethod
    def determine_ref_name() -> str:
        """
        Determine the git ref to benchmark in the baseline workflow.

        Returns:
            Ref name based on BASELINE_REF, workflow input, or GITHUB_REF.
        """
        explicit_ref = os.getenv("BASELINE_REF") or os.getenv("INPUT_REF")
        github_ref = os.getenv("GITHUB_REF", "")
        github_ref_name = os.getenv("GITHUB_REF_NAME", "")

        if explicit_ref:
            ref_name = explicit_ref
            print(f"Using input ref: {ref_name}", file=sys.stderr)
        elif github_ref_name:
            ref_name = github_ref_name
            print(f"Using GitHub ref name: {ref_name}", file=sys.stderr)
        elif github_ref.startswith("refs/tags/"):
            ref_name = github_ref[len("refs/tags/") :]
            print(f"Using push tag ref: {ref_name}", file=sys.stderr)
        elif github_ref.startswith("refs/heads/"):
            ref_name = github_ref[len("refs/heads/") :]
            print(f"Using branch ref: {ref_name}", file=sys.stderr)
        elif github_ref:
            ref_name = github_ref
            print(f"Using GitHub ref: {ref_name}", file=sys.stderr)
        else:
            ref_name = "main"
            print("Using default baseline ref: main", file=sys.stderr)

        try:
            ref_name = _validate_baseline_ref_name(ref_name)
        except ValueError as error:
            print(f"❌ {error}", file=sys.stderr)
            raise SystemExit(1) from error

        github_output = os.getenv("GITHUB_OUTPUT")
        if github_output:
            with open(github_output, "a", encoding="utf-8") as f:
                f.write(f"ref_name={ref_name}\n")

        print(f"Final baseline ref: {ref_name}", file=sys.stderr)
        return ref_name

    @staticmethod
    def create_metadata(
        ref_name: str,
        output_dir: Path,
        artifact_metadata: BaselineArtifactMetadata | None = None,
    ) -> bool:
        """
        Create metadata.json file for baseline artifact.

        Args:
            ref_name: Git ref name for this baseline
            output_dir: Directory to write metadata.json

        Returns:
            True if successful, False otherwise
        """
        try:
            artifact_metadata = artifact_metadata or BaselineArtifactMetadata.from_environment()

            # Generate current timestamp
            now = datetime.now(UTC)
            generated_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")

            # Create metadata dictionary
            metadata = {
                "ref": ref_name,
                "commit": artifact_metadata.commit_sha,
                "workflow_run_id": artifact_metadata.run_id,
                "generated_at": generated_at,
                "runner_os": artifact_metadata.runner_os,
                "runner_arch": artifact_metadata.runner_arch,
            }
            if _is_semver_tag_ref(ref_name):
                metadata["tag"] = ref_name

            # Write metadata file
            output_dir.mkdir(parents=True, exist_ok=True)
            metadata_file = output_dir / "metadata.json"

            with metadata_file.open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            print(f"📦 Created metadata file: {metadata_file}", file=sys.stderr)
            return True

        except (OSError, TypeError, ValueError) as e:
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

        except OSError as e:
            print(f"❌ Failed to display baseline summary: {e}", file=sys.stderr)
            return False

    @staticmethod
    def sanitize_artifact_name(ref_name: str) -> str:
        """
        Sanitize a git ref name for GitHub Actions artifact upload.

        Args:
            ref_name: Original git ref name

        Returns:
            Sanitized artifact name
        """
        # Replace any non-alphanumeric characters (except . _ -) with underscore.
        clean_name = re.sub(r"[^a-zA-Z0-9._-]", "_", ref_name)

        # Avoid dots in artifact names.
        #
        # Some tooling (including common unzip behavior on macOS) treats dot-separated segments
        # as file extensions and can truncate extracted directory names for artifacts like
        # "performance-baseline-v0.6.2".
        clean_name = clean_name.replace(".", "_")

        artifact_name = f"performance-baseline-{clean_name}"

        # Set GitHub Actions output if available
        github_output = os.getenv("GITHUB_OUTPUT")
        if github_output:
            safe = artifact_name.replace("\r", "").replace("\n", "")
            with open(github_output, "a", encoding="utf-8") as f:
                f.write(f"artifact_name={safe}\n")

        print(f"Using sanitized artifact name: {artifact_name}", file=sys.stderr)
        return artifact_name


class BenchmarkRegressionHelper:
    """Helper functions for performance regression testing workflow."""

    @staticmethod
    def write_github_env_vars(env_vars: Mapping[str, str | None]) -> None:
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
            val = "" if value is None else str(value)
            val = val.replace("\r", "")
            os.environ[key] = val

    @staticmethod
    def _export_baseline_identity(lines: list[str]) -> None:
        """Export sanitized baseline ref/tag metadata from baseline file lines."""
        ref_line = next((ln for ln in lines if ln.startswith("Ref: ")), None)
        tag_line = next((ln for ln in lines if ln.startswith("Tag: ")), None)
        raw_ref = None
        if ref_line:
            raw_ref = ref_line.split(":", 1)[1].strip()
        elif tag_line:
            raw_ref = tag_line.split(":", 1)[1].strip()

        if raw_ref:
            safe_ref = re.sub(r"[^A-Za-z0-9._/\-+]", "_", raw_ref)[:128]
            BenchmarkRegressionHelper.write_github_env_vars({"BASELINE_REF": safe_ref})

        if tag_line:
            raw_tag = tag_line.split(":", 1)[1].strip()
            # Allow [A-Za-z0-9._-+]; replace others with underscore and cap length
            safe_tag = re.sub(r"[^A-Za-z0-9._\-+]", "_", raw_tag)[:64]
            BenchmarkRegressionHelper.write_github_env_vars({"BASELINE_TAG": safe_tag})

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
        if baseline_file is None:
            print("❌ Downloaded artifact but no baseline*.txt files found", file=sys.stderr)
            BenchmarkRegressionHelper.write_github_env_vars(
                {
                    "BASELINE_EXISTS": "false",
                    "BASELINE_SOURCE": "missing",
                    "BASELINE_ORIGIN": "unknown",
                }
            )
            return False

        # If a baseline file was found, copy it to baseline_results.txt for consistency
        if baseline_file.name != "baseline_results.txt":
            target_file = baseline_dir / "baseline_results.txt"
            try:
                copyfile(baseline_file, target_file)
                print(f"📦 Prepared baseline from artifact: {baseline_file.name} → baseline_results.txt")
            except OSError as e:
                print(f"❌ Failed to prepare baseline: {e}", file=sys.stderr)
                BenchmarkRegressionHelper.write_github_env_vars(
                    {
                        "BASELINE_EXISTS": "false",
                        "BASELINE_SOURCE": "artifact",
                        "BASELINE_ORIGIN": "artifact",
                    }
                )
                return False
        else:
            print("📦 Prepared baseline from artifact")

        # Set GitHub Actions environment variables
        BenchmarkRegressionHelper.write_github_env_vars(
            {
                "BASELINE_EXISTS": "true",
                "BASELINE_SOURCE": "artifact",
                "BASELINE_ORIGIN": "artifact",
                "BASELINE_SOURCE_FILE": baseline_file.name,
            }
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

        if lines:
            BenchmarkRegressionHelper._export_baseline_identity(lines)

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
                except InvalidVersion as e:
                    # Invalid version format, treat as non-semver
                    logger.debug("Invalid version format in %s: %s", p.name, e)
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
            logger.debug("Could not extract commit from %s: %s", baseline_file.name, e)
        return None

    @staticmethod
    def _extract_commit_from_metadata(metadata_file: Path) -> str | None:
        """Extract commit SHA from metadata.json file."""
        try:
            with metadata_file.open("r", encoding="utf-8") as f:
                data: object = json.load(f)

            if not _is_object_mapping(data):
                return None

            potential_sha = data.get("commit")
            if isinstance(potential_sha, str) and re.match(r"^[0-9A-Fa-f]{7,40}$", potential_sha):
                return potential_sha
        except (OSError, json.JSONDecodeError, KeyError) as e:
            logger.debug("Could not extract commit from metadata.json: %s", e)
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
        except _RECOVERABLE_CLI_ERRORS:
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
        print("   - No GitHub Release benchmark baseline asset was found")
        print("   - Performance regression testing compares against the latest released baseline")
        print()
        print("💡 To enable performance regression testing:")
        print("   1. Publish a GitHub Release")
        print("   2. Wait for release-benchmarks.yml to attach the baseline asset")
        print("   3. Future PRs and pushes will compare against that release baseline")
        print("   4. Baselines use full perf-profile benchmark settings for accurate comparisons")

    @staticmethod
    def run_regression_test(baseline_path: Path, bench_timeout: int = 1800, dev_mode: bool = False) -> bool:
        """
        Run performance regression test against baseline.

        Args:
            baseline_path: Path to baseline file
            bench_timeout: Timeout for cargo bench commands in seconds (default: 1800)
            dev_mode: Use development mode with faster benchmark settings (default: False)

        Returns:
            True if comparison ran and no regressions detected; False on regressions or error
        """
        try:
            mode_str = "dev mode (10x faster)" if dev_mode else "full mode"
            print(f"🚀 Running performance regression test ({mode_str})...")
            print(f"   Using CI performance suite against baseline: {baseline_path}")

            # Use existing PerformanceComparator
            project_root = find_project_root()
            comparator = PerformanceComparator(project_root)
            success, regression_found = comparator.compare_with_baseline(baseline_path, dev_mode=dev_mode, bench_timeout=bench_timeout)

            if not success:
                print("❌ Performance regression test failed", file=sys.stderr)
                return False

            # Provide feedback about regression results
            if regression_found:
                print("⚠️ Performance regressions detected in benchmark comparison")
                return False  # cause non-zero exit in CLI

            print("✅ No significant performance regressions detected")
            return True

        except _RECOVERABLE_CLI_ERRORS as e:
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
        baseline_ref = os.getenv("BASELINE_REF", "n/a")
        baseline_tag = os.getenv("BASELINE_TAG", "n/a")
        baseline_exists = os.getenv("BASELINE_EXISTS", "false")
        skip_benchmarks = os.getenv("SKIP_BENCHMARKS", "unknown")
        skip_reason = os.getenv("SKIP_REASON", "n/a")

        print("📊 Performance Regression Testing Summary")
        print("===========================================")
        print(f"Baseline source: {baseline_source}")
        print(f"Baseline origin: {baseline_origin}")
        print(f"Baseline ref: {baseline_ref}")
        print(f"Baseline tag: {baseline_tag}")
        print(f"Baseline exists: {baseline_exists}")
        print(f"Skip benchmarks: {skip_benchmarks}")
        print(f"Skip reason: {skip_reason}")

        if baseline_exists == "true" and skip_benchmarks == "false":
            results_file = Path("benches") / MAIN_VS_RELEASE_COMPARISON_RESULTS_FILE
            if results_file.exists():
                with results_file.open("r", encoding="utf-8") as f:
                    content = f.read()
                    if "❌ Error:" in content:
                        print(f"Result: ❌ Benchmark comparison failed (see {results_file} for details)")
                    elif "REGRESSION" in content:
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


def get_default_bench_timeout() -> int:
    """
    Get the default benchmark timeout from environment or fallback.

    Returns:
        Timeout in seconds (from BENCHMARK_TIMEOUT env var or 1800 default)
    """
    try:
        timeout = int(os.getenv("BENCHMARK_TIMEOUT", "1800"))
    except (ValueError, TypeError):
        return 1800
    return timeout if timeout > 0 else 1800


# =============================================================================
# LOCAL BASELINE FETCH/COMPARE HELPERS
# =============================================================================


def _sanitize_ref_name(ref_name: str) -> str:
    """Sanitize a git ref name for use in local cache directories."""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", ref_name)


def _sanitize_ref_name_for_artifact(ref_name: str) -> str:
    """Sanitize a git ref name for GitHub Actions artifact names.

    We avoid dots because some tools treat dot-separated segments as file extensions
    and can truncate extracted directory names (e.g., v0.6.2 → v0).
    """
    return _sanitize_ref_name(ref_name).replace(".", "_")


def _default_baseline_cache_dir(project_root: Path, ref_name: str) -> Path:
    """Default on-disk cache location for downloaded baseline artifacts."""
    return project_root / "baseline-artifacts" / _sanitize_ref_name(ref_name)


def _parse_github_owner_repo(remote_url: str) -> tuple[str, str] | None:
    """Parse a GitHub owner/repo from a git remote URL."""
    url = remote_url.strip()
    url = url.removesuffix(".git")

    # https://github.com/OWNER/REPO
    if url.startswith(("https://", "http://")):
        parsed = urlparse(url)
        if parsed.netloc.lower() in {"github.com", "www.github.com"}:
            parts = parsed.path.strip("/").split("/")
            if len(parts) >= 2:
                return parts[0], parts[1]
        return None

    # git@github.com:OWNER/REPO
    match = re.match(r"^git@github\.com:(?P<owner>[^/]+)/(?P<repo>.+)$", url)
    if match:
        return match.group("owner"), match.group("repo")

    # ssh://git@github.com/OWNER/REPO
    if url.startswith("ssh://"):
        parsed = urlparse(url)
        if (parsed.hostname or "").lower() == "github.com":
            parts = (parsed.path or "").strip("/").split("/")
            if len(parts) >= 2:
                return parts[0], parts[1]

    return None


def _resolve_github_repo(project_root: Path, repo: str | None, remote: str) -> str:
    """Resolve the GitHub repo in OWNER/REPO form."""
    if repo is not None:
        return repo

    remote_url = get_git_remote_url(remote=remote, cwd=project_root)
    parsed = _parse_github_owner_repo(remote_url)
    if parsed is None:
        msg = f"Unable to determine GitHub repo from remote '{remote}': {remote_url}"
        raise ValueError(msg)

    owner, repo_name = parsed
    return f"{owner}/{repo_name}"


def _parse_baseline_metadata(baseline_content: str) -> dict[str, str]:
    """Parse basic metadata fields from a baseline file."""
    metadata = {
        "date": "Unknown",
        "commit": "Unknown",
        "ref": "Unknown",
        "tag": "Unknown",
    }

    for line in baseline_content.splitlines():
        if line.startswith("Date: "):
            metadata["date"] = line[6:].strip()
        elif line.startswith("Git commit: "):
            metadata["commit"] = line[12:].strip()
        elif line.startswith("Ref: "):
            metadata["ref"] = line[5:].strip()
        elif line.startswith("Tag: "):
            metadata["tag"] = line[5:].strip()
        elif line.strip() == "Hardware Information:":
            break

    if metadata["ref"] == "Unknown" and metadata["tag"] != "Unknown":
        metadata["ref"] = metadata["tag"]

    return metadata


def _sorted_benchmark_list(results: Mapping[str, "BenchmarkData"]) -> list["BenchmarkData"]:
    """Return benchmarks sorted by (dimension, point count) for stable output."""
    return sorted(results.values(), key=lambda b: (int(b.dimension.rstrip("D")), b.points is None, b.points or 0))


def _find_downloaded_baseline_file(download_dir: Path) -> Path:
    """Find baseline_results.txt in a downloaded artifact directory."""
    direct = download_dir / "baseline_results.txt"
    if direct.exists():
        return direct

    nested = download_dir / "baseline-artifact" / "baseline_results.txt"
    if nested.exists():
        return nested

    matches = list(download_dir.rglob("baseline_results.txt"))
    if len(matches) == 1:
        return matches[0]

    if matches:
        msg = f"Multiple baseline_results.txt files found under: {download_dir}"
        raise FileNotFoundError(msg)

    msg = f"baseline_results.txt not found under: {download_dir}"
    raise FileNotFoundError(msg)


def render_baseline_comparison(project_root: Path, old_baseline: Path, new_baseline: Path) -> tuple[str, bool]:
    """Render a baseline-vs-baseline comparison report.

    Returns:
        (report_text, regression_found)
    """
    old_content = old_baseline.read_text(encoding="utf-8")
    new_content = new_baseline.read_text(encoding="utf-8")

    old_meta = _parse_baseline_metadata(old_content)
    new_meta = _parse_baseline_metadata(new_content)

    # Treat "new" as the "current" side for the hardware comparator.
    new_hw = HardwareComparator.parse_baseline_hardware(new_content)
    old_hw = HardwareComparator.parse_baseline_hardware(old_content)
    hardware_report, _ = HardwareComparator.compare_hardware(new_hw, old_hw)

    comparator = PerformanceComparator(project_root)
    old_results = comparator.parse_baseline_file(old_content)
    new_results = comparator.parse_baseline_file(new_content)

    buf = io.StringIO()
    buf.write("Baseline Comparison Results\n")
    buf.write("==========================\n")
    buf.write(f"New baseline file: {new_baseline}\n")
    buf.write(f"  Date: {new_meta['date']}\n")
    buf.write(f"  Ref: {new_meta['ref']}\n")
    buf.write(f"  Tag: {new_meta['tag']}\n")
    buf.write(f"  Git commit: {new_meta['commit']}\n")
    buf.write(f"Old baseline file: {old_baseline}\n")
    buf.write(f"  Date: {old_meta['date']}\n")
    buf.write(f"  Ref: {old_meta['ref']}\n")
    buf.write(f"  Tag: {old_meta['tag']}\n")
    buf.write(f"  Git commit: {old_meta['commit']}\n\n")

    buf.write(hardware_report)
    buf.write("\n")

    current_results = _sorted_benchmark_list(new_results)
    regression_found = comparator.write_performance_comparison(buf, current_results, old_results)

    return buf.getvalue(), regression_found


@dataclass(frozen=True)
class BaselineFetchOptions:
    """Options controlling how missing performance baselines are fetched."""

    regenerate_missing: bool = False
    workflow_ref: str = "main"
    wait_seconds: int = 3600
    poll_seconds: int = 30

    def __post_init__(self) -> None:
        """Reject invalid wait/poll durations before workflow dispatch."""
        _require_positive_int_field("wait_seconds", self.wait_seconds)
        _require_positive_int_field("poll_seconds", self.poll_seconds)


class GitHubBaselineFetcher:
    """Fetch git-ref baselines from GitHub Actions artifacts using the GitHub CLI."""

    def __init__(self, project_root: Path, *, repo: str | None = None, remote: str = "origin") -> None:
        """Initialize artifact fetching for a project repository."""
        self.project_root = project_root
        self.repo = _resolve_github_repo(project_root, repo=repo, remote=remote)

    def _artifact_name_for_ref(self, ref_name: str) -> str:
        return f"performance-baseline-{_sanitize_ref_name_for_artifact(ref_name)}"

    def _legacy_artifact_name_for_ref(self, ref_name: str) -> str:
        # Legacy naming kept dots from the tag (e.g., v0.6.2).
        return f"performance-baseline-{_sanitize_ref_name(ref_name)}"

    def _try_download_artifact(self, *, artifact_name: str, out_dir: Path) -> bool:
        out_dir.mkdir(parents=True, exist_ok=True)

        result = run_safe_command(
            "gh",
            [
                "run",
                "download",
                "-R",
                self.repo,
                "-n",
                artifact_name,
                "-D",
                str(out_dir),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return True

        logger.debug("gh run download failed (artifact=%s rc=%s stderr=%s)", artifact_name, result.returncode, (result.stderr or "").strip())
        return False

    def _dispatch_generate_baseline(self, *, ref_name: str, workflow_ref: str) -> None:
        result = run_safe_command(
            "gh",
            [
                "workflow",
                "run",
                "generate-baseline.yml",
                "-R",
                self.repo,
                "--ref",
                workflow_ref,
                "-f",
                f"ref={ref_name}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            details = (result.stderr or result.stdout or "").strip()
            msg = f"Failed to dispatch generate-baseline.yml for ref {ref_name} on workflow ref {workflow_ref}: {details}"
            raise RuntimeError(msg)

    def fetch_baseline(self, *, ref_name: str, out_dir: Path, options: BaselineFetchOptions) -> Path:
        """Fetch a baseline for a git ref.

        If options.regenerate_missing is True, this will trigger a workflow_dispatch run
        when the artifact is missing/expired, and poll until it becomes available.

        Returns:
            Path to the downloaded baseline_results.txt
        """
        artifact_name = self._artifact_name_for_ref(ref_name)
        legacy_artifact_name = self._legacy_artifact_name_for_ref(ref_name)

        # Try the current artifact name first, then fall back to the legacy dotful name.
        candidates = list(dict.fromkeys([artifact_name, legacy_artifact_name]))

        def _try_download_any() -> bool:
            return any(self._try_download_artifact(artifact_name=candidate, out_dir=out_dir) for candidate in candidates)

        try:
            if _try_download_any():
                return _find_downloaded_baseline_file(out_dir)

            if not options.regenerate_missing:
                expected = ", ".join(candidates)
                msg = f"Baseline artifact not found for ref {ref_name} (expected artifact name(s): {expected})"
                raise FileNotFoundError(msg)

            print(f"🔁 Baseline artifact not found for {ref_name}; dispatching generate-baseline.yml and waiting...")
            self._dispatch_generate_baseline(ref_name=ref_name, workflow_ref=options.workflow_ref)

            deadline = time.monotonic() + options.wait_seconds
            attempt = 0
            while time.monotonic() < deadline:
                attempt += 1
                time.sleep(options.poll_seconds)

                if _try_download_any():
                    return _find_downloaded_baseline_file(out_dir)

                if attempt % 5 == 0:
                    remaining = int(max(0.0, deadline - time.monotonic()))
                    print(f"⏳ Waiting for baseline artifact {artifact_name}... ({remaining}s remaining)")

            expected = ", ".join(candidates)
            msg = f"Timed out waiting for baseline artifact(s) {expected} (ref {ref_name})"
            raise TimeoutError(msg)

        except ExecutableNotFoundError as e:
            msg = f"Missing dependency: {e} (install the GitHub CLI: gh)"
            raise RuntimeError(msg) from e


def _positive_int_arg(value: str) -> int:
    """Parse a positive integer CLI argument."""
    try:
        parsed = int(value)
    except ValueError as error:
        msg = f"expected a positive integer, got {value!r}"
        raise argparse.ArgumentTypeError(msg) from error
    if parsed <= 0:
        msg = f"expected a positive integer, got {parsed}"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _non_negative_float_arg(value: str) -> float:
    """Parse a non-negative finite float CLI argument."""
    try:
        parsed = float(value)
    except ValueError as error:
        msg = f"expected a non-negative number, got {value!r}"
        raise argparse.ArgumentTypeError(msg) from error
    if not math.isfinite(parsed) or parsed < 0:
        msg = f"expected a non-negative finite number, got {value!r}"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _add_dev_arg(parser: argparse.ArgumentParser, *, help_text: str | None = None) -> None:
    parser.add_argument(
        "--dev",
        action="store_true",
        help=help_text or f"Use faster Criterion settings while retaining the {BENCHMARK_BUILD_FLAVOR} Cargo profile",
    )


def _add_project_root_arg(
    parser: argparse.ArgumentParser, *, help_text: str = "Project root containing the git repo (directory containing Cargo.toml)"
) -> None:
    parser.add_argument("--project-root", type=Path, help=help_text)


def _add_bench_timeout_arg(parser: argparse.ArgumentParser, *, help_text: str | None = None) -> None:
    parser.add_argument(
        "--bench-timeout",
        type=_positive_int_arg,
        default=get_default_bench_timeout(),
        help=help_text or "Timeout for cargo bench in seconds (from BENCHMARK_TIMEOUT env, default: 1800)",
    )


def _add_fetch_wait_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wait-seconds", type=_positive_int_arg, default=3600, help="Max seconds to wait when regenerating (default: 3600)")
    parser.add_argument("--poll-seconds", type=_positive_int_arg, default=30, help="Polling interval seconds when waiting (default: 30)")


def _add_remote_arg(parser: argparse.ArgumentParser, *, help_text: str) -> None:
    parser.add_argument("--remote", type=str, default="origin", help=help_text)


def _add_benchmark_subcommands(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    """Add benchmark-running subcommands."""
    gen_parser = subparsers.add_parser("generate-baseline", help="Generate performance baseline")
    _add_dev_arg(gen_parser)
    gen_parser.add_argument("--output", type=Path, help="Output file path")
    _add_project_root_arg(gen_parser, help_text="Project root to benchmark (directory containing Cargo.toml)")
    gen_parser.add_argument(
        "--ref",
        dest="ref_name",
        type=str,
        default=os.getenv("BASELINE_REF") or os.getenv("REF_NAME"),
        help="Git ref name for this baseline (from BASELINE_REF/REF_NAME env or --ref option)",
    )
    _add_bench_timeout_arg(gen_parser)

    write_parser = subparsers.add_parser("write-baseline", help="Write a baseline from existing Criterion results")
    write_parser.add_argument("--output", type=Path, required=True, help="Output baseline_results.txt path")
    write_parser.add_argument("--project-root", type=Path, help="Project root containing existing target/criterion results")
    write_parser.add_argument(
        "--ref",
        dest="ref_name",
        type=str,
        default=os.getenv("BASELINE_REF") or os.getenv("REF_NAME"),
        help="Git ref name for this baseline (from BASELINE_REF/REF_NAME env or --ref option)",
    )
    write_parser.add_argument("--dev", action="store_true", help="Mark the baseline sampling metadata as dev mode")

    ref_parser = subparsers.add_parser("generate-ref-baseline", help="Generate a local baseline for a git ref")
    ref_parser.add_argument("--ref", dest="ref_name", type=str, default="main", help="Git ref to benchmark (default: main)")
    ref_parser.add_argument("--out", dest="out_dir", type=Path, default=Path("baseline-artifact"), help="Output artifact directory")
    _add_remote_arg(ref_parser, help_text="Git remote to fetch the ref from (default: origin)")
    _add_dev_arg(ref_parser)
    _add_bench_timeout_arg(ref_parser)
    _add_project_root_arg(ref_parser)

    ensure_ref_parser = subparsers.add_parser("ensure-ref-baseline", help="Ensure a cached same-machine baseline exists for a git ref")
    ensure_ref_parser.add_argument("--ref", dest="ref_name", type=str, default="main", help="Git ref to benchmark/cache (default: main)")
    _add_remote_arg(ensure_ref_parser, help_text="Git remote used to resolve/fetch the ref (default: origin)")
    ensure_ref_parser.add_argument(
        "--cache-root",
        type=Path,
        help="Cache root for local same-machine baselines (default: baseline-artifacts/perf-no-regressions)",
    )
    ensure_ref_parser.add_argument(
        "--required-benchmark-id",
        default=PERF_NO_REGRESSIONS_REQUIRED_BENCHMARK_ID,
        help=f"Benchmark ID required before reusing a cache entry (default: {PERF_NO_REGRESSIONS_REQUIRED_BENCHMARK_ID})",
    )
    _add_dev_arg(ensure_ref_parser)
    _add_bench_timeout_arg(
        ensure_ref_parser,
        help_text="Timeout for cargo bench in seconds when refreshing the cache (from BENCHMARK_TIMEOUT env, default: 1800)",
    )
    _add_project_root_arg(ensure_ref_parser)

    cmp_parser = subparsers.add_parser("compare", help="Compare current performance against baseline")
    cmp_parser.add_argument("--baseline", type=Path, required=True, help="Path to baseline file")
    cmp_parser.add_argument(
        "--threshold",
        type=_non_negative_float_arg,
        default=DEFAULT_REGRESSION_THRESHOLD,
        help=f"Regression threshold percentage for marking regressions (default: {DEFAULT_REGRESSION_THRESHOLD})",
    )
    _add_dev_arg(cmp_parser)
    cmp_parser.add_argument(
        "--output",
        type=Path,
        help=f"Output file path (default: benches/{MAIN_VS_RELEASE_COMPARISON_RESULTS_FILE})",
    )
    _add_project_root_arg(cmp_parser, help_text="Project root to benchmark (directory containing Cargo.toml)")
    _add_bench_timeout_arg(cmp_parser)

    cmp_ref_parser = subparsers.add_parser("compare-ref", help="Compare current performance against a cached same-machine git-ref baseline")
    cmp_ref_parser.add_argument("--ref", dest="ref_name", type=str, default="main", help="Git ref to benchmark/cache (default: main)")
    _add_remote_arg(cmp_ref_parser, help_text="Git remote used to resolve/fetch the ref (default: origin)")
    cmp_ref_parser.add_argument(
        "--cache-root",
        type=Path,
        help="Cache root for local same-machine baselines (default: baseline-artifacts/perf-no-regressions)",
    )
    cmp_ref_parser.add_argument(
        "--required-benchmark-id",
        default=PERF_NO_REGRESSIONS_REQUIRED_BENCHMARK_ID,
        help=f"Benchmark ID required before reusing a cache entry (default: {PERF_NO_REGRESSIONS_REQUIRED_BENCHMARK_ID})",
    )
    cmp_ref_parser.add_argument(
        "--threshold",
        type=_non_negative_float_arg,
        default=DEFAULT_REGRESSION_THRESHOLD,
        help=f"Regression threshold percentage for marking regressions (default: {DEFAULT_REGRESSION_THRESHOLD})",
    )
    _add_dev_arg(cmp_ref_parser)
    _add_bench_timeout_arg(cmp_ref_parser)
    cmp_ref_parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: benches/worktree_vs_<ref>_compare_results.txt)",
    )
    _add_project_root_arg(cmp_ref_parser)


def _add_local_baseline_subcommands(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    """Add subcommands that operate on existing baseline artifacts/files."""
    bb_parser = subparsers.add_parser("compare-baselines", help="Compare two baseline files (no benchmarks)")
    bb_parser.add_argument("--old", dest="old_baseline", type=Path, required=True, help="Path to the older baseline file")
    bb_parser.add_argument("--new", dest="new_baseline", type=Path, required=True, help="Path to the newer baseline file")
    bb_parser.add_argument("--output", type=Path, help="Optional path to write the comparison report")
    bb_parser.add_argument("--project-root", type=Path, help="Project root (only used for repo context; optional)")

    fetch_parser = subparsers.add_parser("fetch-baseline", help="Fetch a git-ref baseline artifact from GitHub Actions")
    fetch_parser.add_argument("--ref", dest="ref_name", type=str, help="Git ref to fetch (e.g., main or v0.6.2)")
    fetch_parser.add_argument("--out", dest="out_dir", type=Path, help="Output directory for downloaded artifact contents")
    fetch_parser.add_argument("--repo", type=str, help="GitHub repo in OWNER/REPO form (defaults to parsing the git remote)")
    _add_remote_arg(fetch_parser, help_text="Git remote name used to infer repo when --repo is not set")
    fetch_parser.add_argument("--regenerate-missing", action="store_true", help="If missing, dispatch generate-baseline.yml and wait for artifact")
    fetch_parser.add_argument(
        "--workflow-ref",
        type=str,
        default="main",
        help="Git ref to run generate-baseline.yml from when regenerating (default: main)",
    )
    _add_fetch_wait_args(fetch_parser)
    _add_project_root_arg(fetch_parser)

    tags_parser = subparsers.add_parser("compare-tags", help="Compare two tags by fetching their baselines and comparing locally")
    tags_parser.add_argument("--old-tag", dest="old_tag", type=str, required=True, help="Older tag (e.g., v0.6.1)")
    tags_parser.add_argument("--new-tag", dest="new_tag", type=str, required=True, help="Newer tag (e.g., v0.6.2)")
    tags_parser.add_argument("--output", type=Path, help="Optional path to write the comparison report")
    tags_parser.add_argument("--repo", type=str, help="GitHub repo in OWNER/REPO form (defaults to parsing the git remote)")
    _add_remote_arg(tags_parser, help_text="Git remote name used to infer repo when --repo is not set")
    tags_parser.add_argument("--regenerate-missing", action="store_true", help="If missing, dispatch generate-baseline.yml and wait for artifacts")
    tags_parser.add_argument(
        "--workflow-ref",
        type=str,
        default="main",
        help="Git ref to run generate-baseline.yml from when regenerating (default: main)",
    )
    _add_fetch_wait_args(tags_parser)
    _add_project_root_arg(tags_parser)


def _add_workflow_helper_subcommands(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    """Add subcommands used by GitHub Actions workflows."""
    subparsers.add_parser("determine-ref", help="Determine git ref name for baseline generation")

    meta_parser = subparsers.add_parser("create-metadata", help="Create metadata.json file for baseline artifact")
    meta_parser.add_argument("--ref", dest="ref_name", type=str, help="Git ref name for this baseline")
    meta_parser.add_argument("--output-dir", type=Path, default=Path("baseline-artifact"), help="Output directory for metadata.json")

    summary_parser = subparsers.add_parser("display-summary", help="Display baseline file summary")
    summary_parser.add_argument("--baseline", type=Path, required=True, help="Path to baseline file")

    artifact_parser = subparsers.add_parser("sanitize-artifact-name", help="Sanitize git ref name for GitHub Actions artifact")
    artifact_parser.add_argument("--ref", dest="ref_name", type=str, help="Git ref name to sanitize")


def _add_regression_subcommands(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    """Add regression-testing helper subcommands."""
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
    _add_dev_arg(regress_parser)
    _add_bench_timeout_arg(regress_parser)

    results_parser = subparsers.add_parser("display-results", help="Display regression test results")
    results_parser.add_argument(
        "--results",
        type=Path,
        default=Path("benches") / MAIN_VS_RELEASE_COMPARISON_RESULTS_FILE,
        help="Results file path",
    )

    subparsers.add_parser("regression-summary", help="Generate regression testing summary")


def _add_performance_summary_subcommands(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    """Add performance summary generation subcommands."""
    perf_summary_parser = subparsers.add_parser("generate-summary", help="Generate performance summary markdown")
    perf_summary_parser.add_argument("--output", type=Path, help="Output file path (defaults to benches/PERFORMANCE_RESULTS.md)")
    perf_summary_parser.add_argument(
        "--run-benchmarks",
        action="store_true",
        help="Run fresh ci_performance_suite and circumsphere benchmarks before generating summary",
    )
    perf_summary_parser.add_argument(
        "--profile",
        default=BENCHMARK_BUILD_FLAVOR,
        help=f"Cargo profile to use when --run-benchmarks is set (default: {BENCHMARK_BUILD_FLAVOR})",
    )
    perf_summary_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of rendering from existing or fallback data when fresh benchmark execution fails",
    )


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(description="Benchmark utilities for baseline generation and comparison")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    _add_benchmark_subcommands(subparsers)
    _add_local_baseline_subcommands(subparsers)
    _add_workflow_helper_subcommands(subparsers)
    _add_regression_subcommands(subparsers)
    _add_performance_summary_subcommands(subparsers)

    return parser


def configure_logging(*, verbose: bool) -> None:
    """Configure CLI logging before command execution."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def _exit_called_process_error(error: subprocess.CalledProcessError) -> NoReturn:
    print(f"❌ Git command failed with exit code {error.returncode}: {error.cmd}", file=sys.stderr)
    if error.stderr:
        print(error.stderr, file=sys.stderr)
    if error.stdout:
        print(error.stdout, file=sys.stderr)
    sys.exit(1)


def _local_ref_cache_options_from_args(args: argparse.Namespace) -> LocalRefBaselineCacheOptions:
    return LocalRefBaselineCacheOptions(
        ref_name=args.ref_name,
        remote=args.remote,
        cache_root=args.cache_root,
        dev_mode=args.dev,
        bench_timeout=args.bench_timeout,
        required_benchmark_id=args.required_benchmark_id,
    )


def _cmd_generate_baseline(args: argparse.Namespace, project_root: Path) -> None:
    generator = BaselineGenerator(project_root, ref_name=args.ref_name)
    success = generator.generate_baseline(dev_mode=args.dev, output_file=args.output, bench_timeout=args.bench_timeout)
    sys.exit(0 if success else 1)


def _cmd_write_baseline(args: argparse.Namespace, project_root: Path) -> None:
    output_file = args.output if args.output.is_absolute() else project_root / args.output
    generator = BaselineGenerator(project_root, ref_name=args.ref_name)
    success = generator.write_baseline_from_existing_results(output_file, dev_mode=args.dev)
    sys.exit(0 if success else 1)


def _cmd_generate_ref_baseline(args: argparse.Namespace, project_root: Path) -> None:
    out_dir = args.out_dir if args.out_dir.is_absolute() else project_root / args.out_dir
    try:
        generator = LocalRefBaselineGenerator(project_root, remote=args.remote)
        baseline_path = generator.generate_for_ref(
            ref_name=args.ref_name,
            out_dir=out_dir,
            dev_mode=args.dev,
            bench_timeout=args.bench_timeout,
        )
    except subprocess.CalledProcessError as e:
        _exit_called_process_error(e)
    except _RECOVERABLE_CLI_ERRORS as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)

    print(baseline_path)
    sys.exit(0)


def _cmd_ensure_ref_baseline(args: argparse.Namespace, project_root: Path) -> None:
    options = _local_ref_cache_options_from_args(args)
    try:
        cache_result = ensure_cached_ref_baseline_for_ref(project_root, options)
    except subprocess.CalledProcessError as e:
        _exit_called_process_error(e)
    except _RECOVERABLE_CLI_ERRORS as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)

    print(cache_result.baseline_path)
    sys.exit(0)


def _cmd_compare(args: argparse.Namespace, project_root: Path) -> None:
    comparator = PerformanceComparator(project_root)
    comparator.regression_threshold = args.threshold
    output_file = args.output or release_comparison_results_path(project_root)
    success, regression_found = comparator.compare_with_baseline(
        args.baseline,
        dev_mode=args.dev,
        output_file=output_file,
        bench_timeout=args.bench_timeout,
    )
    _display_comparison_result(output_file, success=success, regression_found=regression_found)

    if not success:
        sys.exit(1)

    sys.exit(1 if regression_found else 0)


def _cmd_compare_ref(args: argparse.Namespace, project_root: Path) -> None:
    options = _local_ref_cache_options_from_args(args)
    try:
        exit_code = compare_with_cached_ref_baseline(
            project_root,
            options,
            threshold=args.threshold,
            output_file=args.output,
        )
    except subprocess.CalledProcessError as e:
        _exit_called_process_error(e)
    except _RECOVERABLE_CLI_ERRORS as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)

    sys.exit(exit_code)


def execute_baseline_commands(args: argparse.Namespace, project_root: Path) -> None:
    """Execute baseline generation and comparison commands."""
    handlers = {
        "generate-baseline": _cmd_generate_baseline,
        "write-baseline": _cmd_write_baseline,
        "generate-ref-baseline": _cmd_generate_ref_baseline,
        "ensure-ref-baseline": _cmd_ensure_ref_baseline,
        "compare": _cmd_compare,
        "compare-ref": _cmd_compare_ref,
    }
    handler = handlers.get(args.command)
    if handler is None:
        msg = f"Unknown baseline command: {args.command}"
        raise ValueError(msg)
    handler(args, project_root)


def _write_optional_report(output_path: Path | None, report_text: str) -> None:
    if output_path is None:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")


def _baseline_fetch_options_from_args(args: argparse.Namespace) -> BaselineFetchOptions:
    return BaselineFetchOptions(
        regenerate_missing=args.regenerate_missing,
        workflow_ref=args.workflow_ref,
        wait_seconds=args.wait_seconds,
        poll_seconds=args.poll_seconds,
    )


def _cmd_compare_baselines(args: argparse.Namespace, project_root: Path) -> None:
    if not args.old_baseline.exists():
        print(f"❌ Baseline file not found: {args.old_baseline}", file=sys.stderr)
        sys.exit(3)
    if not args.new_baseline.exists():
        print(f"❌ Baseline file not found: {args.new_baseline}", file=sys.stderr)
        sys.exit(3)

    try:
        report_text, regression_found = render_baseline_comparison(project_root, args.old_baseline, args.new_baseline)
    except FileNotFoundError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(3)
    except BaselineParseError as e:
        print(f"❌ Failed to parse baseline file: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"❌ Failed to compare baseline files: {e}", file=sys.stderr)
        sys.exit(1)

    print(report_text, end="" if report_text.endswith("\n") else "\n")
    _write_optional_report(args.output, report_text)
    sys.exit(1 if regression_found else 0)


def _cmd_fetch_baseline(args: argparse.Namespace, project_root: Path) -> None:
    if not args.ref_name:
        print("❌ Missing required --ref argument", file=sys.stderr)
        sys.exit(2)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = _default_baseline_cache_dir(project_root, args.ref_name)

    try:
        fetcher = GitHubBaselineFetcher(project_root, repo=args.repo, remote=args.remote)
        options = _baseline_fetch_options_from_args(args)
        baseline_path = fetcher.fetch_baseline(ref_name=args.ref_name, out_dir=out_dir, options=options)
    except FileNotFoundError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(3)
    except TimeoutError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(2 if str(e).startswith("Missing dependency:") else 1)

    print(baseline_path)
    sys.exit(0)


def _cmd_compare_tags(args: argparse.Namespace, project_root: Path) -> None:
    try:
        fetcher = GitHubBaselineFetcher(project_root, repo=args.repo, remote=args.remote)
        options = _baseline_fetch_options_from_args(args)

        old_dir = _default_baseline_cache_dir(project_root, args.old_tag)
        new_dir = _default_baseline_cache_dir(project_root, args.new_tag)

        old_baseline = fetcher.fetch_baseline(ref_name=args.old_tag, out_dir=old_dir, options=options)
        new_baseline = fetcher.fetch_baseline(ref_name=args.new_tag, out_dir=new_dir, options=options)

        report_text, regression_found = render_baseline_comparison(project_root, old_baseline, new_baseline)
    except FileNotFoundError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(3)
    except BaselineParseError as e:
        print(f"❌ Failed to parse baseline file: {e}", file=sys.stderr)
        sys.exit(1)
    except TimeoutError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(2 if str(e).startswith("Missing dependency:") else 1)

    print(report_text, end="" if report_text.endswith("\n") else "\n")
    _write_optional_report(args.output, report_text)
    sys.exit(1 if regression_found else 0)


def execute_local_baseline_commands(args: argparse.Namespace, project_root: Path) -> None:
    """Execute local (non-benchmark) baseline fetch/compare commands."""
    handlers = {
        "compare-baselines": _cmd_compare_baselines,
        "fetch-baseline": _cmd_fetch_baseline,
        "compare-tags": _cmd_compare_tags,
    }

    handler = handlers.get(args.command)
    if handler is None:
        msg = f"Unknown local baseline command: {args.command}"
        raise ValueError(msg)

    handler(args, project_root)


def _cmd_determine_ref(_args: argparse.Namespace) -> None:
    ref_name = WorkflowHelper.determine_ref_name()
    print(ref_name)
    sys.exit(0)


def _cmd_create_metadata(args: argparse.Namespace) -> None:
    if not args.ref_name:
        print("❌ Missing required --ref argument", file=sys.stderr)
        sys.exit(2)
    success = WorkflowHelper.create_metadata(args.ref_name, args.output_dir)
    sys.exit(0 if success else 1)


def _cmd_display_summary(args: argparse.Namespace) -> None:
    success = WorkflowHelper.display_baseline_summary(args.baseline)
    sys.exit(0 if success else 1)


def _cmd_sanitize_artifact_name(args: argparse.Namespace) -> None:
    if not args.ref_name:
        print("❌ Missing required --ref argument", file=sys.stderr)
        sys.exit(2)
    artifact_name = WorkflowHelper.sanitize_artifact_name(args.ref_name)
    print(artifact_name)
    sys.exit(0)


def execute_workflow_commands(args: argparse.Namespace) -> None:
    """Execute workflow helper commands."""
    handlers = {
        "determine-ref": _cmd_determine_ref,
        "create-metadata": _cmd_create_metadata,
        "display-summary": _cmd_display_summary,
        "sanitize-artifact-name": _cmd_sanitize_artifact_name,
    }
    handler = handlers.get(args.command)
    if handler is None:
        msg = f"Unknown workflow command: {args.command}"
        raise ValueError(msg)
    handler(args)


def _cmd_prepare_baseline(args: argparse.Namespace) -> None:
    success = BenchmarkRegressionHelper.prepare_baseline(args.baseline_dir)
    sys.exit(0 if success else 1)


def _cmd_set_no_baseline(_args: argparse.Namespace) -> None:
    BenchmarkRegressionHelper.set_no_baseline_status()
    sys.exit(0)


def _cmd_extract_baseline_commit(args: argparse.Namespace) -> None:
    commit_sha = BenchmarkRegressionHelper.extract_baseline_commit(args.baseline_dir)
    print(commit_sha)
    sys.exit(0)


def _cmd_determine_skip(args: argparse.Namespace) -> None:
    should_skip, reason = BenchmarkRegressionHelper.determine_benchmark_skip(args.baseline_commit, args.current_commit)

    BenchmarkRegressionHelper.write_github_env_vars(
        {
            "SKIP_BENCHMARKS": "true" if should_skip else "false",
            "SKIP_REASON": reason,
        }
    )

    print(f"skip={should_skip}")
    print(f"reason={reason}")
    sys.exit(0)


def _cmd_display_skip_message(args: argparse.Namespace) -> None:
    BenchmarkRegressionHelper.display_skip_message(args.reason, args.baseline_commit or "")
    sys.exit(0)


def _cmd_display_no_baseline(_args: argparse.Namespace) -> None:
    BenchmarkRegressionHelper.display_no_baseline_message()
    sys.exit(0)


def _cmd_run_regression_test(args: argparse.Namespace) -> None:
    success = BenchmarkRegressionHelper.run_regression_test(args.baseline, bench_timeout=args.bench_timeout, dev_mode=args.dev)
    sys.exit(0 if success else 1)


def _cmd_display_results(args: argparse.Namespace) -> None:
    BenchmarkRegressionHelper.display_results(args.results)
    sys.exit(0)


def _cmd_regression_summary(_args: argparse.Namespace) -> None:
    BenchmarkRegressionHelper.generate_summary()
    sys.exit(0)


def execute_regression_commands(args: argparse.Namespace) -> None:
    """Execute regression testing commands."""
    handlers = {
        "prepare-baseline": _cmd_prepare_baseline,
        "set-no-baseline": _cmd_set_no_baseline,
        "extract-baseline-commit": _cmd_extract_baseline_commit,
        "determine-skip": _cmd_determine_skip,
        "display-skip-message": _cmd_display_skip_message,
        "display-no-baseline": _cmd_display_no_baseline,
        "run-regression-test": _cmd_run_regression_test,
        "display-results": _cmd_display_results,
        "regression-summary": _cmd_regression_summary,
    }
    handler = handlers.get(args.command)
    if handler is None:
        msg = f"Unknown regression command: {args.command}"
        raise ValueError(msg)
    handler(args)


def _cmd_generate_summary(args: argparse.Namespace, project_root: Path) -> None:
    generator = PerformanceSummaryGenerator(project_root)
    success = generator.generate_summary(
        output_path=args.output,
        run_benchmarks=args.run_benchmarks,
        generator_name="benchmark_utils.py",
        cargo_profile=args.profile,
        strict=args.strict,
    )
    sys.exit(0 if success else 1)


def execute_performance_summary_commands(args: argparse.Namespace, project_root: Path) -> None:
    """Execute performance summary commands."""
    handlers = {
        "generate-summary": _cmd_generate_summary,
    }
    handler = handlers.get(args.command)
    if handler is None:
        msg = f"Unknown performance summary command: {args.command}"
        raise ValueError(msg)
    handler(args, project_root)


def _execute_workflow_commands_with_root(args: argparse.Namespace, _project_root: Path) -> None:
    execute_workflow_commands(args)


def _execute_regression_commands_with_root(args: argparse.Namespace, _project_root: Path) -> None:
    execute_regression_commands(args)


def execute_command(args: argparse.Namespace, project_root: Path) -> None:
    """Execute the selected command based on parsed arguments."""
    handlers = {
        "generate-baseline": execute_baseline_commands,
        "write-baseline": execute_baseline_commands,
        "generate-ref-baseline": execute_baseline_commands,
        "ensure-ref-baseline": execute_baseline_commands,
        "compare": execute_baseline_commands,
        "compare-ref": execute_baseline_commands,
        "compare-baselines": execute_local_baseline_commands,
        "fetch-baseline": execute_local_baseline_commands,
        "compare-tags": execute_local_baseline_commands,
        "determine-ref": _execute_workflow_commands_with_root,
        "create-metadata": _execute_workflow_commands_with_root,
        "display-summary": _execute_workflow_commands_with_root,
        "sanitize-artifact-name": _execute_workflow_commands_with_root,
        "generate-summary": execute_performance_summary_commands,
        "prepare-baseline": _execute_regression_commands_with_root,
        "set-no-baseline": _execute_regression_commands_with_root,
        "extract-baseline-commit": _execute_regression_commands_with_root,
        "determine-skip": _execute_regression_commands_with_root,
        "display-skip-message": _execute_regression_commands_with_root,
        "display-no-baseline": _execute_regression_commands_with_root,
        "run-regression-test": _execute_regression_commands_with_root,
        "display-results": _execute_regression_commands_with_root,
        "regression-summary": _execute_regression_commands_with_root,
    }
    handler = handlers.get(args.command)
    if handler is None:
        msg = f"Unknown command: {args.command}"
        raise ValueError(msg)
    handler(args, project_root)


def main() -> None:
    """Command-line interface for benchmark utilities."""
    parser = create_argument_parser()
    args = parser.parse_args()
    configure_logging(verbose=args.verbose)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        project_root: Path
        if hasattr(args, "project_root") and args.project_root is not None:
            project_root = args.project_root.resolve()
            if not (project_root / "Cargo.toml").exists():
                parser.error(f"--project-root must contain Cargo.toml (got: {project_root})")
        else:
            project_root = find_project_root()
    except ProjectRootNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)

    execute_command(args, project_root)


if __name__ == "__main__":
    main()
