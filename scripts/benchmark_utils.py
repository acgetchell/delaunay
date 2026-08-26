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
import hashlib
import io
import json
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from shutil import copy2 as copyfile  # NOTE: Use copy2 (metadata-preserving) under the 'copyfile' alias for tests/patching convenience.
from typing import TYPE_CHECKING, Literal, NoReturn, TextIO, TypeIs, cast
from urllib.parse import urlparse
from uuid import uuid4

from packaging.version import InvalidVersion, Version

logger = logging.getLogger(__name__)

DEFAULT_REGRESSION_THRESHOLD = 7.5
TIME_UNIT_TO_MICROSECONDS = {"ns": 1e-3, "µs": 1.0, "μs": 1.0, "us": 1.0, "ms": 1e3, "s": 1e6}
type ComparisonFailurePolicy = Literal["strict", "total-time"]


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
    def from_environment(cls) -> BaselineArtifactMetadata:
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
    from performance_artifacts import (
        ArtifactContext,
        ArtifactPaths,
        HostIdentity,
        MeasurementArtifact,
        PerformanceBundle,
        PerformanceRow,
        ReleasePair,
        SourceState,
        TimingEstimate,
        ToolchainState,
        ensure_distinct_paths,
        load_bundle,
        publish_bundle,
        serialize_bundle,
    )
    from subprocess_utils import (
        ExceptionFamily,
        ExecutableNotFoundError,
        ProjectRootNotFoundError,
        find_project_root,
        get_git_commit_hash,
        get_git_remote_url,
        run_cargo_command,
        run_git_command,
        run_git_command_with_input,
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
        from performance_artifacts import (
            ArtifactContext,
            ArtifactPaths,
            HostIdentity,
            MeasurementArtifact,
            PerformanceBundle,
            PerformanceRow,
            ReleasePair,
            SourceState,
            TimingEstimate,
            ToolchainState,
            ensure_distinct_paths,
            load_bundle,
            publish_bundle,
            serialize_bundle,
        )
        from subprocess_utils import (
            ExceptionFamily,
            ExecutableNotFoundError,
            ProjectRootNotFoundError,
            find_project_root,
            get_git_commit_hash,
            get_git_remote_url,
            run_cargo_command,
            run_git_command,
            run_git_command_with_input,
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
        from scripts.performance_artifacts import (
            ArtifactContext,
            ArtifactPaths,
            HostIdentity,
            MeasurementArtifact,
            PerformanceBundle,
            PerformanceRow,
            ReleasePair,
            SourceState,
            TimingEstimate,
            ToolchainState,
            ensure_distinct_paths,
            load_bundle,
            publish_bundle,
            serialize_bundle,
        )
        from scripts.subprocess_utils import (
            ExceptionFamily,
            ExecutableNotFoundError,
            ProjectRootNotFoundError,
            find_project_root,
            get_git_commit_hash,
            get_git_remote_url,
            run_cargo_command,
            run_git_command,
            run_git_command_with_input,
            run_safe_command,
        )

_RECOVERABLE_CLI_ERRORS: ExceptionFamily = (
    ExecutableNotFoundError,
    ProjectRootNotFoundError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    KeyError,
    subprocess.SubprocessError,
)
_CI_PERFORMANCE_METRIC_PARSE_ERRORS: ExceptionFamily = (KeyError, ValueError)
_CI_PERFORMANCE_SIDECAR_LOAD_ERRORS: ExceptionFamily = (OSError, json.JSONDecodeError)
_CRITERION_ESTIMATE_PARSE_ERRORS: ExceptionFamily = (KeyError, TypeError, ValueError)
_NUMERICAL_ACCURACY_PARSE_ERRORS: ExceptionFamily = (IndexError, TypeError, ValueError)
_CARGO_MANIFEST_LOAD_ERRORS: ExceptionFamily = (OSError, tomllib.TOMLDecodeError)
_LOCAL_RUSTC_VERSION_ERRORS: ExceptionFamily = (ExecutableNotFoundError, OSError, subprocess.SubprocessError)
_BENCHMARK_TIMEOUT_PARSE_ERRORS: ExceptionFamily = (ValueError, TypeError)

# Trusted benchmark commands use this Cargo profile so local, CI, and release
# numbers are generated with the same ThinLTO/codegen-units settings.
BENCHMARK_BUILD_FLAVOR = "perf"


@dataclass(frozen=True, slots=True)
class BenchmarkTargetMeasurement:
    """One exact benchmark target invocation in a retained measurement plan."""

    target: str
    report_section: str = ""
    required_group_prefixes: tuple[str, ...] = ()
    sampling_mode: Literal["full", "reduced"] = "full"
    criterion_arguments: tuple[str, ...] = ()

    @property
    def command(self) -> tuple[str, ...]:
        """Return the exact Cargo argument vector for this measurement."""
        command = ("cargo", "bench", "--profile", BENCHMARK_BUILD_FLAVOR, "--bench", self.target)
        if self.criterion_arguments:
            return (*command, "--", *self.criterion_arguments)
        return command


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
RELEASE_SIGNAL_MEASUREMENT_PLAN = (
    BenchmarkTargetMeasurement(
        "ci_performance_suite",
        "Public API performance",
        (
            "tds_new_",
            "boundary_facets",
            "convex_hull",
            "convex_hull_queries",
            "validation",
            "incremental_insert",
            "explicit_import",
            "proof_boundaries",
            "bistellar_flips_",
        ),
    ),
    BenchmarkTargetMeasurement(
        "circumsphere_containment",
        "Circumsphere predicates",
        ("random", "2d", "3d", "4d", "5d", "edge_cases_", "circumcenter"),
    ),
    BenchmarkTargetMeasurement(
        "cold_path_predicates",
        "Predicate hot and cold paths",
        ("predicates",),
    ),
    BenchmarkTargetMeasurement(
        "locate",
        "Point location",
        ("locate",),
    ),
    BenchmarkTargetMeasurement(
        "realization_validation",
        "Realization validation",
        ("realization_",),
    ),
)
RELEASE_SIGNAL_BENCH_TARGETS = tuple(measurement.target for measurement in RELEASE_SIGNAL_MEASUREMENT_PLAN)
RELEASE_SIGNAL_GROUP_PREFIXES = tuple(
    dict.fromkeys(prefix for measurement in RELEASE_SIGNAL_MEASUREMENT_PLAN for prefix in measurement.required_group_prefixes)
)
RELEASE_ASSET_METADATA_SCHEMA_VERSION = 2
RELEASE_ASSET_MEASUREMENT_COMMANDS = tuple(measurement.command for measurement in RELEASE_SIGNAL_MEASUREMENT_PLAN)
BENCH_TARGET_SUITES = {
    "release-signal": RELEASE_SIGNAL_BENCH_TARGETS,
    "ci": ("ci_performance_suite",),
    "query": ("circumsphere_containment", "locate"),
    "predicates": ("circumsphere_containment", "cold_path_predicates"),
    "topology": ("topology_guarantee_construction",),
}
BENCH_COMPARE_GROUP_PREFIXES_BY_SUITE = {
    "release-signal": RELEASE_SIGNAL_GROUP_PREFIXES,
    "ci": (
        "tds_new_",
        "boundary_facets",
        "convex_hull",
        "convex_hull_queries",
        "validation",
        "incremental_insert",
        "explicit_import",
        "bistellar_flips_",
    ),
    "query": (
        "random",
        "2d",
        "3d",
        "4d",
        "5d",
        "edge_cases_",
        "locate",
    ),
    "predicates": (
        "random",
        "2d",
        "3d",
        "4d",
        "5d",
        "edge_cases_",
        "predicates",
    ),
    "topology": ("topology_guarantee_construction",),
}
BENCH_COMPARE_SUITE_CHOICES = tuple(BENCH_TARGET_SUITES)
PERFORMANCE_REPORT_SOURCE = Path("target") / "bench-reports" / "performance.md"
GITHUB_ASSETS_PERFORMANCE_REPORT = Path("target") / "bench-reports" / "github-assets-performance.md"
DOCS_PERFORMANCE_REPORT = Path("docs") / "PERFORMANCE.md"
PERFORMANCE_ARCHIVE_DIR = Path("docs") / "archive" / "performance"
RELEASE_BENCH_TIMEOUT_SECONDS = 7200
RELEASE_COMMAND_TIMEOUT_SECONDS = 600
DELAUNAY_REPORT_VERSION_RE = re.compile(r"^\*\*delaunay\*\* v(?P<version>[^\s`]+)", re.MULTILINE)
DELAUNAY_REPORT_BASELINE_RE = re.compile(r"^Comparison against baseline \*\*(?P<baseline>[^*]+)\*\*:", re.MULTILINE)
SEMVER_IDENTIFIER_RE = r"(?:0|[1-9][0-9]*|[0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*)"
SEMVER_TAG_RE = re.compile(
    rf"^v?(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    rf"(?:-{SEMVER_IDENTIFIER_RE}(?:\.{SEMVER_IDENTIFIER_RE})*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
STABLE_SEMVER_TAG_RE = re.compile(r"^v?(?P<major>0|[1-9][0-9]*)\.(?P<minor>0|[1-9][0-9]*)\.(?P<patch>0|[1-9][0-9]*)$")
HOW_TO_UPDATE_RE = re.compile(r"(?ms)^## How to Update\n.*\Z")


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
class CriterionComparison:
    """A comparison between current Criterion output and a named saved baseline."""

    benchmark_id: str
    baseline: TimingEstimate
    current: TimingEstimate

    @property
    def baseline_ns(self) -> float:
        """Return the baseline point estimate for compatibility with report math."""
        return self.baseline.median_ns

    @property
    def current_ns(self) -> float:
        """Return the current point estimate for compatibility with report math."""
        return self.current.median_ns

    @property
    def percent_change(self) -> float:
        """Return signed current-vs-baseline timing change."""
        if self.baseline_ns <= 0:
            return 0.0
        return ((self.current_ns - self.baseline_ns) / self.baseline_ns) * 100.0

    @property
    def speedup(self) -> float:
        """Return baseline/current speedup, where values above 1 mean faster."""
        if self.current_ns <= 0:
            return float("inf")
        return self.baseline_ns / self.current_ns


@dataclass(frozen=True)
class CriterionReportSettings:
    """Settings rendered into a Criterion-baseline comparison report."""

    baseline_name: str
    stat: str
    suite: str
    scope: str


@dataclass(frozen=True)
class CriterionReportRequest:
    """CLI request to write a Criterion saved-baseline comparison report."""

    baseline_name: str
    output: Path
    stat: str = "median"
    suite: str = "release-signal"
    scope: str = "release-signal"
    criterion_dir: Path | None = None


@dataclass(frozen=True)
class PerformanceReportId:
    """Release-pair identity parsed from a benchmark report."""

    current_tag: str
    baseline_tag: str

    @property
    def archive_name(self) -> str:
        """Return the canonical archive filename for this release pair."""
        return f"{self.current_tag}-vs-{self.baseline_tag}.md"


@dataclass(frozen=True)
class PerformancePromotionDestinations:
    """Explicit repository boundary and tracked performance destinations."""

    project_root: Path
    current: Path
    archive_dir: Path


@dataclass(frozen=True)
class PerformancePromotionPlan:
    """Validated file payloads and destinations for one report promotion."""

    report_id: PerformanceReportId
    source_text: str
    current_text: str | None
    archive_path: Path | None
    durable_artifacts: ArtifactPaths
    source_csv: bytes
    source_provenance: bytes
    mutation_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class PublishedRelease:
    """Stable GitHub release metadata used to infer release pairs."""

    tag: str
    published_at: datetime

    def __post_init__(self) -> None:
        """Require normalized stable semver identity and an aware timestamp."""
        if self.tag != normalize_release_tag(self.tag):
            msg = f"published release tag must be normalized: {self.tag!r}"
            raise ValueError(msg)
        _stable_semver_sort_key(self.tag)
        if not isinstance(self.published_at, datetime):
            msg = f"published release timestamp must be a datetime: {self.published_at!r}"
            raise TypeError(msg)
        if self.published_at.tzinfo is None or self.published_at.utcoffset() is None:
            msg = f"published release timestamp must include a timezone: {self.published_at!r}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class GitHubRelease:
    """Strict immutable DTO for one ``gh release list`` JSON object."""

    tag: str
    is_draft: bool
    is_prerelease: bool
    published_at: datetime | None

    @classmethod
    def from_raw(cls, raw: object, *, index: int) -> GitHubRelease:
        """Parse exactly the requested GitHub fields without truthy coercions."""
        if not isinstance(raw, Mapping):
            msg = f"GitHub release at index {index} must be a JSON object"
            raise TypeError(msg)
        expected_fields = {"tagName", "isDraft", "isPrerelease", "publishedAt"}
        actual_fields = set(raw)
        if actual_fields != expected_fields:
            missing = sorted(expected_fields - actual_fields)
            extra = sorted(str(field) for field in actual_fields - expected_fields)
            msg = f"GitHub release at index {index} has unexpected fields: missing={missing}, extra={extra}"
            raise ValueError(msg)

        is_draft = raw["isDraft"]
        is_prerelease = raw["isPrerelease"]
        if type(is_draft) is not bool or type(is_prerelease) is not bool:
            msg = f"GitHub release at index {index} requires exact boolean isDraft/isPrerelease fields"
            raise TypeError(msg)

        tag_name = raw["tagName"]
        if not isinstance(tag_name, str):
            msg = f"GitHub release at index {index} tagName must be a string"
            raise TypeError(msg)
        try:
            tag = normalize_release_tag(tag_name)
        except ValueError as exc:
            msg = f"GitHub release at index {index} has invalid semantic-version tag {tag_name!r}"
            raise ValueError(msg) from exc

        published_raw = raw["publishedAt"]
        published_at: datetime | None
        if published_raw is None and is_draft:
            published_at = None
        elif isinstance(published_raw, str) and published_raw:
            try:
                published_at = datetime.fromisoformat(published_raw)
            except ValueError as exc:
                msg = f"GitHub release at index {index} has invalid publishedAt timestamp {published_raw!r}"
                raise ValueError(msg) from exc
            if published_at.tzinfo is None or published_at.utcoffset() is None:
                msg = f"GitHub release at index {index} publishedAt must include a timezone"
                raise ValueError(msg)
            published_at = published_at.astimezone(UTC)
        else:
            msg = f"GitHub release at index {index} requires a publishedAt timestamp unless it is a draft"
            raise TypeError(msg)
        return cls(
            tag=tag,
            is_draft=is_draft,
            is_prerelease=is_prerelease,
            published_at=published_at,
        )


type BaselineSource = Literal["local", "github-assets"]


@dataclass(frozen=True)
class ReleaseReportConfig:
    """Configuration for generating a release-comparison report."""

    repo_root: Path
    current_tag: str
    baseline_tag: str
    worktree_ref: str
    suite: str = "release-signal"
    scope: str = "release-signal"
    stat: str = "median"
    apply_current_diff: bool = True
    baseline_source: BaselineSource = "local"

    def __post_init__(self) -> None:
        """Reject unsupported artifact settings before any workflow effects."""
        if self.suite not in BENCH_COMPARE_SUITE_CHOICES:
            msg = f"unsupported benchmark suite: {self.suite!r}"
            raise ValueError(msg)
        if self.scope not in ("release-signal", "all-benches"):
            msg = f"unsupported benchmark scope: {self.scope!r}"
            raise ValueError(msg)
        if self.stat != "median":
            msg = f"release artifact workflows require the median statistic, got {self.stat!r}"
            raise ValueError(msg)
        if self.baseline_source not in ("local", "github-assets"):
            msg = f"unsupported release baseline source: {self.baseline_source!r}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ToolRunOptions:
    """Execution controls for one repository support command."""

    timeout: int = RELEASE_COMMAND_TIMEOUT_SECONDS
    env: dict[str, str] | None = None
    stream_output: bool = False


@dataclass(frozen=True)
class RevisionEvidence:
    """Source, toolchain, and command evidence for one measured revision."""

    source: SourceState
    toolchain: ToolchainState
    commands: tuple[tuple[str, ...], ...]
    completed_targets: tuple[str, ...]


@dataclass(frozen=True)
class RevisionMeasurement:
    """The suite, commands, and shared target plan measured for one revision."""

    suite: str
    commands: tuple[tuple[str, ...], ...]
    comparison_targets: tuple[str, ...] | None = None


@dataclass(frozen=True)
class CriterionSample:
    """One Criterion sample path with identity recovered from benchmark.json."""

    benchmark_id: str
    group: str
    benchmark: str
    estimates: Path


@dataclass(frozen=True)
class DownloadedReleaseAsset:
    """A downloaded release archive and the exact acquisition command."""

    archive: Path
    command: tuple[str, ...]


@dataclass(frozen=True)
class ReleaseAssetEvidence:
    """Validated measurement evidence extracted from one release archive."""

    revision: RevisionEvidence
    measurement_host: HostIdentity
    artifact: MeasurementArtifact
    acquisition_commands: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class ReleaseAssetLoadRequest:
    """Trusted local paths and requested identity for one release archive."""

    requested_tag: str
    expected_commit: str
    extracted_root: Path
    archive: Path
    acquisition_command: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedPerformanceRequest:
    """Release pair and checkout ref resolved from CLI arguments."""

    current_tag: str
    baseline_tag: str
    worktree_ref: str
    tags_to_fetch: tuple[str, ...] = ()


@dataclass(frozen=True)
class PerformanceRequestOptions:
    """CLI options used to resolve performance comparison tags."""

    current_tag: str | None
    baseline_tag: str | None
    published_latest: bool
    infer_release: bool
    current_vs_latest: bool
    worktree_ref: str
    repo_root: Path


@dataclass(frozen=True)
class ComparisonSummaryStats:
    """Summary statistics for benchmark comparison failure policy decisions."""

    total_time_change: float
    geomean_change: float
    median_change: float
    individual_regressions: int
    compared_count: int
    failure_policy: ComparisonFailurePolicy


@dataclass(frozen=True, slots=True)
class BenchmarkComparisonCoverage:
    """Exact keyset evidence required before aggregate timings are comparable."""

    current_keys: frozenset[str]
    baseline_keys: frozenset[str]
    duplicate_current_keys: tuple[str, ...] = ()
    invalid_current_timing_keys: tuple[str, ...] = ()
    invalid_baseline_timing_keys: tuple[str, ...] = ()

    @property
    def missing_from_baseline(self) -> tuple[str, ...]:
        """Return current benchmark keys absent from the baseline."""
        return tuple(sorted(self.current_keys - self.baseline_keys))

    @property
    def missing_from_current(self) -> tuple[str, ...]:
        """Return baseline benchmark keys absent from the current run."""
        return tuple(sorted(self.baseline_keys - self.current_keys))

    @property
    def is_comparable(self) -> bool:
        """Return whether coverage is nonempty, unique, and exactly symmetric."""
        return (
            bool(self.current_keys)
            and not self.duplicate_current_keys
            and not self.invalid_current_timing_keys
            and not self.invalid_baseline_timing_keys
            and self.current_keys == self.baseline_keys
        )


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
        except _CI_PERFORMANCE_METRIC_PARSE_ERRORS:
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
    except _CI_PERFORMANCE_SIDECAR_LOAD_ERRORS:
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
    return all(math.isfinite(value) and value > 0 for value in (mean_ns, low_ns, high_ns)) and low_ns <= high_ns


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
        low_ns = _criterion_float(confidence_interval["lower_bound"])
        high_ns = _criterion_float(confidence_interval["upper_bound"])
    except _CRITERION_ESTIMATE_PARSE_ERRORS:
        return None

    if not is_valid_criterion_estimate(mean_ns, low_ns, high_ns):
        return None
    return CriterionEstimate(mean_ns=mean_ns, low_ns=low_ns, high_ns=high_ns)


def _load_criterion_estimate(estimates_path: Path) -> CriterionEstimate | None:
    """Load and validate a Criterion estimates.json file."""
    try:
        with estimates_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except _CI_PERFORMANCE_SIDECAR_LOAD_ERRORS:
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


@dataclass(frozen=True)
class CiPerformanceSummaryEvidence:
    """Validated public-API benchmark rows and their completeness basis."""

    results: tuple[CiPerformanceResult, ...]
    missing_result_ids: tuple[str, ...]
    completeness_basis: Literal["runtime-manifest", "group-contract"]

    @property
    def is_complete(self) -> bool:
        """Return whether every structurally expected result was parsed."""
        return not self.missing_result_ids


@dataclass(frozen=True)
class CircumsphereSummaryEvidence:
    """Circumsphere rows with explicit measured-or-fallback provenance."""

    test_cases: tuple[CircumsphereTestCase, ...]
    provenance: Literal["criterion", "reference-fallback"]
    missing_result_ids: tuple[str, ...]

    @property
    def is_complete(self) -> bool:
        """Return whether all expected Criterion rows were parsed without fallback."""
        return self.provenance == "criterion" and not self.missing_result_ids


@dataclass(frozen=True)
class BenchmarkPlanSectionEvidence:
    """Criterion coverage for one report section owned by the release plan."""

    target: str
    report_section: str
    result_ids: tuple[str, ...]
    missing_group_prefixes: tuple[str, ...]

    @property
    def is_complete(self) -> bool:
        """Return whether every required Criterion group produced a valid estimate."""
        return bool(self.result_ids) and not self.missing_group_prefixes


def _criterion_result_ids(criterion_dir: Path) -> tuple[str, ...]:
    """Return valid Criterion result IDs, preferring ``new`` over ``base``."""
    results: dict[str, str] = {}
    for estimates_path in sorted(criterion_dir.glob("**/estimates.json")):
        sample = estimates_path.parent.name
        if sample not in {"base", "new"} or _load_criterion_estimate(estimates_path) is None:
            continue
        path_parts = estimates_path.relative_to(criterion_dir).parts[:-2]
        if not path_parts:
            continue
        result_id = "/".join(path_parts)
        previous = results.get(result_id)
        if previous is None or (previous == "base" and sample == "new"):
            results[result_id] = sample
    return tuple(sorted(results))


def _criterion_group_matches(result_id: str, group_prefix: str) -> bool:
    """Return whether a Criterion result belongs to one planned top-level group."""
    group = result_id.split("/", maxsplit=1)[0]
    return group.startswith(group_prefix) if group_prefix.endswith("_") else group == group_prefix


@dataclass(frozen=True)
class PerformanceSummaryEvidence:
    """Complete structured evidence consumed by one summary render."""

    ci_performance: CiPerformanceSummaryEvidence
    circumsphere: CircumsphereSummaryEvidence
    release_signal_sections: tuple[BenchmarkPlanSectionEvidence, ...]

    def validation_errors(self) -> tuple[str, ...]:
        """Return actionable reasons this evidence is not release-complete."""
        errors = []
        if not self.ci_performance.is_complete:
            missing = ", ".join(self.ci_performance.missing_result_ids)
            errors.append(f"ci_performance_suite is incomplete ({self.ci_performance.completeness_basis}; missing: {missing})")
        if self.circumsphere.provenance != "criterion":
            errors.append("circumsphere evidence uses reference fallback timings")
        if self.circumsphere.missing_result_ids:
            missing = ", ".join(self.circumsphere.missing_result_ids)
            errors.append(f"circumsphere Criterion evidence is incomplete (missing: {missing})")
        planned_targets = RELEASE_SIGNAL_BENCH_TARGETS
        evidence_targets = tuple(section.target for section in self.release_signal_sections)
        if evidence_targets != planned_targets:
            errors.append(
                "release-signal report sections do not match the measurement plan "
                f"(expected: {', '.join(planned_targets)}; found: {', '.join(evidence_targets)})",
            )
        for section in self.release_signal_sections:
            if section.is_complete:
                continue
            missing = ", ".join(section.missing_group_prefixes) or "all results"
            errors.append(f"{section.target} report section {section.report_section!r} is incomplete (missing groups: {missing})")
        return tuple(errors)


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


def run_release_signal_measurement_plan(
    project_root: Path,
    *,
    cargo_profile: str = BENCHMARK_BUILD_FLAVOR,
    bench_timeout: int = 1800,
    save_baseline: str | None = None,
) -> dict[str, str]:
    """Execute the maintained release-signal plan and return stdout by target."""
    _require_positive_int_field("bench_timeout", bench_timeout)
    outputs: dict[str, str] = {}
    for measurement in RELEASE_SIGNAL_MEASUREMENT_PLAN:
        cargo_args = ["bench", "--profile", cargo_profile, "--bench", measurement.target]
        criterion_arguments = list(measurement.criterion_arguments)
        if save_baseline is not None:
            criterion_arguments.extend(["--save-baseline", save_baseline])
        if criterion_arguments:
            cargo_args.extend(["--", *criterion_arguments])

        print(f"🔄 Running release-signal target {measurement.target}...")
        result = run_cargo_command(
            cargo_args,
            cwd=project_root,
            timeout=bench_timeout,
            capture_output=True,
            check=False,
        )
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
        if result.returncode != 0:
            msg = f"release-signal target {measurement.target} exited with status {result.returncode}"
            raise RuntimeError(msg)
        outputs[measurement.target] = result.stdout

        if measurement.target == "ci_performance_suite":
            completed_at = datetime.now(UTC)
            _write_ci_performance_manifest_ids(project_root, result.stdout)
            _write_ci_performance_metrics(project_root, result.stdout, require_metrics=True)
            _write_ci_performance_run_metadata(
                project_root,
                completed_at=completed_at,
                cargo_profile=cargo_profile,
                use_dev_mode=False,
            )

    return outputs


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
        cargo_profile: str | None = None,
        bench_timeout: int = 1800,
        strict: bool = False,
    ) -> bool:
        """
        Generate performance summary markdown file.

        Args:
            output_path: Output file path (defaults to benches/PERFORMANCE_RESULTS.md)
            run_benchmarks: Whether to run the fresh release-signal measurement plan
            cargo_profile: Optional Cargo profile for fresh benchmark runs.  When
                ``run_benchmarks`` is True and no profile is specified, defaults
                to :data:`BENCHMARK_BUILD_FLAVOR` so fresh runs match baseline
                and comparison measurements.
            bench_timeout: Per-target timeout for the release-signal plan in seconds.
            strict: Reject fallback or incomplete benchmark evidence. Fresh
                benchmark requests enforce the same completeness contract.

        Returns:
            True if successful, False otherwise
        """
        _require_positive_int_field("bench_timeout", bench_timeout)
        try:
            if output_path is None:
                output_path = self.project_root / "benches" / "PERFORMANCE_RESULTS.md"

            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Optionally run fresh benchmarks
            if run_benchmarks:
                if cargo_profile is None:
                    cargo_profile = BENCHMARK_BUILD_FLAVOR
                try:
                    outputs = run_release_signal_measurement_plan(
                        self.project_root,
                        cargo_profile=cargo_profile,
                        bench_timeout=bench_timeout,
                    )
                    self.numerical_accuracy_data = self._parse_numerical_accuracy_output(
                        outputs["circumsphere_containment"],
                    )
                except _RECOVERABLE_CLI_ERRORS as error:
                    logger.debug("Fresh release-signal benchmark plan failed: %s", error)
                    print("❌ Fresh benchmark run failed; summary publication was not attempted", file=sys.stderr)
                    return False

            evidence = self._collect_summary_evidence()
            validation_errors = evidence.validation_errors()
            if (strict or run_benchmarks) and validation_errors:
                print(
                    "❌ Strict/fresh summary requires complete measured evidence: " + "; ".join(validation_errors),
                    file=sys.stderr,
                )
                return False

            # Render from the evidence that was validated above, then publish only
            # after the complete content has been durably written beside the target.
            content = self._generate_markdown_content(evidence=evidence)
            _write_text_atomic(output_path, content)

            print(f"📊 Generated performance summary: {output_path}")
            return True

        except _RECOVERABLE_CLI_ERRORS as e:
            print(f"❌ Failed to generate performance summary: {e}", file=sys.stderr)
            return False

    def _collect_ci_performance_summary_evidence(self) -> CiPerformanceSummaryEvidence:
        """Collect parsed CI-suite rows and structural completeness evidence."""
        results = tuple(self._parse_ci_performance_suite_results())
        manifest_ids = _load_ci_performance_manifest_ids(self.circumsphere_results_dir)
        if manifest_ids is not None:
            parsed_ids = {result.benchmark_id for result in results}
            missing_result_ids = tuple(sorted(manifest_ids - parsed_ids))
            completeness_basis: Literal["runtime-manifest", "group-contract"] = "runtime-manifest"
        else:
            parsed_groups = {result.group_key for result in results}
            missing_result_ids = tuple(f"group:{group}" for group in CI_PERFORMANCE_SUITE_GROUP_ORDER if group not in parsed_groups)
            completeness_basis = "group-contract"
        return CiPerformanceSummaryEvidence(
            results=results,
            missing_result_ids=missing_result_ids,
            completeness_basis=completeness_basis,
        )

    def _circumsphere_expected_results(self) -> dict[tuple[str, str, str], str]:
        """Return the fixed circumsphere case/method contract keyed by rendered identity."""
        benchmark_mappings, edge_case_mappings, method_mappings, edge_method_mappings = self._get_benchmark_mappings()
        expected = {
            (test_name, dimension, method_name): f"{bench_key}_{method_suffix}"
            for bench_key, (test_name, dimension) in benchmark_mappings.items()
            for method_suffix, method_name in method_mappings.items()
        }
        expected.update(
            {
                (test_name, dimension, method_name): f"{bench_key}_{method_suffix}"
                for bench_key, (test_name, dimension) in edge_case_mappings.items()
                for method_suffix, method_name in edge_method_mappings.items()
            },
        )
        return expected

    def _collect_circumsphere_summary_evidence(self) -> CircumsphereSummaryEvidence:
        """Collect measured circumsphere rows or an explicitly identified fallback."""
        parsed_cases = tuple(self._parse_circumsphere_benchmark_results())
        expected = self._circumsphere_expected_results()
        parsed_keys = {(test_case.test_name, test_case.dimension, method_name) for test_case in parsed_cases for method_name in test_case.methods}
        missing_result_ids = tuple(sorted(result_id for result_key, result_id in expected.items() if result_key not in parsed_keys))
        if parsed_cases:
            return CircumsphereSummaryEvidence(
                test_cases=parsed_cases,
                provenance="criterion",
                missing_result_ids=missing_result_ids,
            )
        return CircumsphereSummaryEvidence(
            test_cases=tuple(self._get_fallback_circumsphere_data()),
            provenance="reference-fallback",
            missing_result_ids=missing_result_ids,
        )

    def _collect_release_signal_section_evidence(self) -> tuple[BenchmarkPlanSectionEvidence, ...]:
        """Collect report coverage directly from the executable release plan."""
        result_ids = _criterion_result_ids(self.circumsphere_results_dir)
        sections = []
        for measurement in RELEASE_SIGNAL_MEASUREMENT_PLAN:
            matching_ids = tuple(
                result_id for result_id in result_ids if any(_criterion_group_matches(result_id, prefix) for prefix in measurement.required_group_prefixes)
            )
            missing_prefixes = tuple(
                prefix for prefix in measurement.required_group_prefixes if not any(_criterion_group_matches(result_id, prefix) for result_id in matching_ids)
            )
            sections.append(
                BenchmarkPlanSectionEvidence(
                    target=measurement.target,
                    report_section=measurement.report_section,
                    result_ids=matching_ids,
                    missing_group_prefixes=missing_prefixes,
                ),
            )
        return tuple(sections)

    def _collect_summary_evidence(self) -> PerformanceSummaryEvidence:
        """Collect every dynamic result exactly once for validation and rendering."""
        return PerformanceSummaryEvidence(
            ci_performance=self._collect_ci_performance_summary_evidence(),
            circumsphere=self._collect_circumsphere_summary_evidence(),
            release_signal_sections=self._collect_release_signal_section_evidence(),
        )

    @staticmethod
    def _release_signal_coverage_section(sections: tuple[BenchmarkPlanSectionEvidence, ...]) -> list[str]:
        """Render exact report-section coverage from the executable measurement plan."""
        lines = [
            "## Release Signal Measurement Coverage",
            "",
            "| Benchmark target | Report section | Valid results | Status |",
            "|------------------|----------------|--------------:|--------|",
        ]
        for section in sections:
            if section.is_complete:
                status = "complete"
            else:
                missing = ", ".join(section.missing_group_prefixes) or "all results"
                status = f"incomplete: missing {missing}"
            lines.append(f"| `{section.target}` | {section.report_section} | {len(section.result_ids)} | {status} |")
        lines.extend(
            [
                "",
                "The benchmark target, report section, and required Criterion groups are owned by the",
                "same executable release-signal plan. Strict generation rejects every incomplete row.",
                "",
            ],
        )
        return lines

    def _generate_markdown_content(
        self,
        generator_name: str | None = None,
        *,
        evidence: PerformanceSummaryEvidence | None = None,
    ) -> str:
        """
        Generate the complete markdown content for performance results.

        Args:
            generator_name: Name of the tool generating the summary (for attribution)

        Returns:
            Formatted markdown content as string
        """
        if evidence is None:
            evidence = self._collect_summary_evidence()

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

        lines.extend(self._release_signal_coverage_section(evidence.release_signal_sections))

        # Add public API performance results from the CI suite next. This is
        # the versioned benchmark contract used by baseline/comparison tooling.
        lines.extend(self._get_ci_performance_suite_results(evidence.ci_performance))

        # Add circumsphere predicate results as a focused subsection. These
        # remain important because they exercise la-stack-backed predicates.
        lines.extend(self._get_circumsphere_performance_results(evidence.circumsphere))

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
        except _CARGO_MANIFEST_LOAD_ERRORS:
            return None

        package = manifest.get("package")
        if not isinstance(package, dict):
            return None

        version = package.get("version")
        if isinstance(version, str) and version.strip():
            return cast("str", version).strip()
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

    def _run_ci_performance_suite(
        self,
        cargo_profile: str | None = None,
        *,
        use_dev_mode: bool = False,
        bench_timeout: int = 1800,
    ) -> bool:
        """
        Run the public API CI performance suite to generate fresh Criterion data.

        Args:
            cargo_profile: Cargo profile for the fresh run. Defaults to
                :data:`BENCHMARK_BUILD_FLAVOR` so summary, baseline, and
                comparison measurements use the same optimized profile.
            use_dev_mode: When true, pass reduced Criterion sampling arguments
                for local development feedback. Full sampling is used by
                default.
            bench_timeout: Maximum runtime for the Cargo benchmark command in seconds.

        Returns:
            True if the benchmark completed successfully, False otherwise.
        """
        _require_positive_int_field("bench_timeout", bench_timeout)
        try:
            print("🔄 Running ci_performance_suite benchmarks...")

            profile = cargo_profile if cargo_profile is not None else BENCHMARK_BUILD_FLAVOR
            cargo_args = ["bench", "--profile", profile, "--bench", "ci_performance_suite"]
            if use_dev_mode:
                cargo_args.extend(["--", *DEV_MODE_BENCH_ARGS])

            result = run_cargo_command(
                cargo_args,
                cwd=self.project_root,
                timeout=bench_timeout,
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

        except _NUMERICAL_ACCURACY_PARSE_ERRORS:
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
            return []

        benchmark_mappings, edge_case_mappings, method_mappings, edge_method_mappings = self._get_benchmark_mappings()

        test_cases = []
        test_cases.extend(self._parse_regular_benchmarks(benchmark_mappings, method_mappings))
        test_cases.extend(self._parse_edge_case_benchmarks(edge_case_mappings, edge_method_mappings))

        if not test_cases:
            print("⚠️ No circumsphere benchmark results parsed")

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
        estimates_file = criterion_path / "new" / "estimates.json"
        if not estimates_file.exists():
            estimates_file = criterion_path / "base" / "estimates.json"

        if estimates_file.exists():
            estimate = _load_criterion_estimate(estimates_file)
            if estimate is not None:
                return CircumspherePerformanceData(method=method_name, time_ns=estimate.mean_ns)

        return None

    def _get_fallback_circumsphere_data(self) -> list[CircumsphereTestCase]:
        """
        Get explicitly labeled reference fallback data for permissive reports.

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

    def _get_ci_performance_suite_results(
        self,
        evidence: CiPerformanceSummaryEvidence | None = None,
    ) -> list[str]:
        """
        Generate the public API performance summary from ci_performance_suite data.

        Returns:
            List of markdown lines with ci_performance_suite benchmark data.
        """
        if evidence is None:
            evidence = self._collect_ci_performance_summary_evidence()
        results = evidence.results

        lines = [
            "### Public API Performance Contract (`ci_performance_suite`)",
            "",
            "This suite is the versioned benchmark contract for public Delaunay workflows.",
            "It covers construction, hull extraction, validation, incremental insertion,",
            "boundary traversal, and explicit bistellar flip roundtrips.",
            "",
        ]

        if evidence.missing_result_ids and results:
            lines.extend(
                [
                    "⚠️ This section is incomplete; some expected Criterion estimates were missing or malformed.",
                    "",
                ],
            )

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

    def _get_circumsphere_performance_results(
        self,
        evidence: CircumsphereSummaryEvidence | None = None,
    ) -> list[str]:
        """
        Generate circumsphere containment performance results section with dynamic data.

        Returns:
            List of markdown lines with circumsphere performance data
        """
        if evidence is None:
            evidence = self._collect_circumsphere_summary_evidence()
        test_cases = evidence.test_cases

        if not test_cases:
            return [
                "### Circumsphere Predicate Performance",
                "",
                f"#### Version {self.current_version} Results ({self.current_date})",
                "",
                "⚠️ No benchmark results available. Run benchmarks first:",
                "```bash",
                f"uv run --locked benchmark-utils generate-summary --run-benchmarks --profile {BENCHMARK_BUILD_FLAVOR}",
                "```",
                "",
            ]

        lines = [
            "### Circumsphere Predicate Performance",
            "",
            "This focused predicate suite tracks `la-stack`-backed circumsphere and",
            "insphere query performance independently from full triangulation workflows.",
            "",
        ]
        lines.extend(self._circumsphere_evidence_header(evidence))

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

    def _circumsphere_evidence_header(self, evidence: CircumsphereSummaryEvidence) -> list[str]:
        """Render an honest heading for measured, partial, or fallback evidence."""
        if evidence.provenance == "reference-fallback":
            return [
                "⚠️ Reference fallback timings are shown below; they are not measurements for the current version.",
                "",
                "#### Reference Fallback Timings",
                "",
            ]

        lines = []
        if evidence.missing_result_ids:
            lines.extend(
                [
                    "⚠️ This section is incomplete; some expected Criterion estimates were missing or malformed.",
                    "",
                ],
            )
        lines.extend(
            [
                f"#### Version {self.current_version} Results ({self.current_date})",
                "",
            ],
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
            "uv run --locked benchmark-utils generate-summary",
            "",
            "# Run the fresh perf-profile release-signal plan",
            f"uv run --locked benchmark-utils generate-summary --run-benchmarks --profile {BENCHMARK_BUILD_FLAVOR}",
            "",
            "# Package existing ci_performance_suite Criterion results for release-asset comparisons",
            "uv run --locked benchmark-utils write-baseline --ref vX.Y.Z --output baseline_results.txt",
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
        return cast("str", m.group(1)) if m else None

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


def normalize_release_tag(tag: str) -> str:
    """Return a semver release tag with a leading ``v``."""
    normalized = tag.strip()
    if not normalized:
        msg = "tag must not be empty"
        raise ValueError(msg)
    if not normalized.startswith("v"):
        normalized = f"v{normalized}"
    if SEMVER_TAG_RE.fullmatch(normalized) is None:
        msg = f"expected a semver tag like v0.8.0, got {tag!r}"
        raise ValueError(msg)
    return normalized


def _stable_semver_sort_key(tag: str) -> tuple[int, int, int]:
    """Return a sortable key for stable semver tags."""
    match = STABLE_SEMVER_TAG_RE.fullmatch(normalize_release_tag(tag))
    if match is None:
        msg = f"expected a stable semver tag like v0.8.0, got {tag!r}"
        raise ValueError(msg)
    return (int(match.group("major")), int(match.group("minor")), int(match.group("patch")))


def _read_text(path: Path) -> str:
    """Read UTF-8 text."""
    return path.read_text(encoding="utf-8")


def _write_text_atomic(path: Path, text: str) -> None:
    """Write UTF-8 text through a same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(text)
            tmp.flush()
            os.fsync(tmp.fileno())
        tmp_path.replace(path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Write bytes through a durable same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(payload)
            tmp.flush()
            os.fsync(tmp.fileno())
        tmp_path.replace(path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _restore_file_snapshot(path: Path, payload: bytes | None) -> None:
    """Restore one prior file payload or prior absence."""
    if payload is None:
        path.unlink(missing_ok=True)
    else:
        _write_bytes_atomic(path, payload)


def _read_cargo_package_version(repo_root: Path) -> str:
    """Return the package version from Cargo.toml."""
    cargo_toml = repo_root / "Cargo.toml"
    data = tomllib.loads(_read_text(cargo_toml))
    package = data.get("package")
    if not isinstance(package, dict):
        msg = f"could not find [package] in {cargo_toml}"
        raise TypeError(msg)
    version = package.get("version")
    if not isinstance(version, str):
        msg = f"could not find package.version in {cargo_toml}"
        raise TypeError(msg)
    return version


def _current_package_tag(repo_root: Path) -> str:
    """Return the current Cargo package version as a release tag."""
    return normalize_release_tag(_read_cargo_package_version(repo_root))


def _get_git_info(repo_root: Path) -> tuple[str, str]:
    """Return short commit hash and branch name for report headers."""
    short_hash = "unknown"
    branch = "unknown"
    try:
        result = run_git_command(["--no-pager", "rev-parse", "--short", "HEAD"], cwd=repo_root, timeout=30)
        short_hash = result.stdout.strip() or short_hash
    except _RECOVERABLE_CLI_ERRORS:
        logger.debug("Unable to read short git hash for benchmark report", exc_info=True)
    try:
        result = run_git_command(["--no-pager", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root, timeout=30)
        branch = result.stdout.strip() or branch
    except _RECOVERABLE_CLI_ERRORS:
        logger.debug("Unable to read branch name for benchmark report", exc_info=True)
    return short_hash, branch


def _benchmark_report_environment_lines(repo_root: Path) -> list[str]:
    """Return compact environment metadata for benchmark report reproducibility."""
    lines = [
        "## Environment",
        "",
        f"- **Cargo profile**: `{BENCHMARK_BUILD_FLAVOR}`",
        "- **Raw Criterion data**: `target/criterion/`",
    ]
    try:
        hardware = HardwareInfo().get_hardware_info(cwd=repo_root)
    except _RECOVERABLE_CLI_ERRORS:
        logger.debug("Unable to collect hardware metadata for benchmark report", exc_info=True)
        lines.append("- **Hardware**: Unknown")
    else:
        lines.extend(
            [
                f"- **OS**: {hardware['OS']}",
                f"- **CPU**: {hardware['CPU']} ({hardware['CPU_CORES']} cores, {hardware['CPU_THREADS']} threads)",
                f"- **Memory**: {hardware['MEMORY']}",
                f"- **Rust**: {hardware['RUST']}",
                f"- **Target**: {hardware['TARGET']}",
            ]
        )
    return lines


def _format_ns(ns: float) -> str:
    """Format nanoseconds as a compact human-readable duration."""
    if ns < 1_000:
        return f"{ns:.1f} ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.2f} µs"
    if ns < 1_000_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    return f"{ns / 1_000_000_000:.2f} s"


def _format_pct_change(percent: float) -> str:
    """Format signed timing change, bolding material improvements."""
    if percent < -1.0:
        return f"**{percent:+.1f}%**"
    return f"{percent:+.1f}%"


def _criterion_numeric_field(obj: Mapping[str, object], field: str, estimates_json: Path, stat: str) -> float:
    """Read one numeric field from Criterion estimates JSON."""
    value = obj.get(field)
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        msg = f"field {field!r} for stat {stat!r} in {estimates_json} is not numeric: {value!r}"
        raise TypeError(msg)
    try:
        numeric = float(value)
    except ValueError as exc:
        msg = f"field {field!r} for stat {stat!r} in {estimates_json} is not numeric: {value!r}"
        raise ValueError(msg) from exc
    if not math.isfinite(numeric) or numeric <= 0.0:
        msg = f"field {field!r} for stat {stat!r} in {estimates_json} must be a positive finite number: {value!r}"
        raise ValueError(msg)
    return numeric


def _read_criterion_timing_estimate(estimates_json: Path, stat: str) -> TimingEstimate:
    """Read a Criterion point estimate and confidence interval in nanoseconds."""
    try:
        data = json.loads(_read_text(estimates_json))
    except json.JSONDecodeError as exc:
        msg = f"malformed Criterion estimates JSON in {estimates_json}: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(data, dict):
        msg = f"expected JSON object in {estimates_json}"
        raise TypeError(msg)
    stat_obj = data.get(stat)
    if not isinstance(stat_obj, Mapping):
        msg = f"stat {stat!r} not found in {estimates_json}"
        raise KeyError(msg)
    confidence_interval = stat_obj.get("confidence_interval")
    if not isinstance(confidence_interval, Mapping):
        msg = f"confidence_interval for stat {stat!r} in {estimates_json} is not an object"
        raise TypeError(msg)
    return TimingEstimate(
        median_ns=_criterion_numeric_field(stat_obj, "point_estimate", estimates_json, stat),
        ci_lower_ns=_criterion_numeric_field(confidence_interval, "lower_bound", estimates_json, stat),
        ci_upper_ns=_criterion_numeric_field(confidence_interval, "upper_bound", estimates_json, stat),
        confidence_level=_criterion_numeric_field(confidence_interval, "confidence_level", estimates_json, stat),
    )


def _read_criterion_point_estimate(estimates_json: Path, stat: str) -> float:
    """Read a Criterion point estimate in nanoseconds."""
    return _read_criterion_timing_estimate(estimates_json, stat).median_ns


def _criterion_sample(estimates_json: Path, criterion_dir: Path) -> CriterionSample | None:
    """Recover a Criterion benchmark identity from its sample metadata."""
    if estimates_json.name != "estimates.json":
        return None
    sample_dir = estimates_json.parent
    benchmark_dir = sample_dir.parent
    try:
        benchmark_dir.relative_to(criterion_dir)
    except ValueError:
        return None
    metadata_path = sample_dir / "benchmark.json"
    try:
        metadata = json.loads(_read_text(metadata_path))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"could not load Criterion benchmark metadata {metadata_path}: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(metadata, Mapping):
        msg = f"Criterion benchmark metadata must be an object: {metadata_path}"
        raise TypeError(msg)
    full_id = metadata.get("full_id")
    group_id = metadata.get("group_id")
    if not isinstance(full_id, str) or not full_id.strip() or not isinstance(group_id, str) or not group_id.strip():
        msg = f"Criterion benchmark metadata requires non-empty full_id and group_id: {metadata_path}"
        raise ValueError(msg)
    prefix = f"{group_id}/"
    if full_id.startswith(prefix):
        group = group_id
        benchmark = full_id.removeprefix(prefix)
    elif "/" in full_id:
        group, benchmark = full_id.split("/", maxsplit=1)
    else:
        msg = f"Criterion full_id must contain a group and benchmark: {full_id!r} in {metadata_path}"
        raise ValueError(msg)
    return CriterionSample(benchmark_id=full_id, group=group, benchmark=benchmark, estimates=estimates_json)


def _criterion_estimates_by_id(criterion_dir: Path, sample: str) -> dict[str, CriterionSample]:
    """Map Criterion benchmark IDs to estimates files for one sample name."""
    results: dict[str, CriterionSample] = {}
    if not criterion_dir.is_dir():
        return results
    for estimates_json in sorted(criterion_dir.rglob("estimates.json")):
        if estimates_json.parent.name != sample:
            continue
        criterion_sample = _criterion_sample(estimates_json, criterion_dir)
        if criterion_sample is not None:
            prior = results.get(criterion_sample.benchmark_id)
            if prior is not None and prior.estimates != criterion_sample.estimates:
                msg = f"duplicate Criterion full_id {criterion_sample.benchmark_id!r} under {criterion_dir}"
                raise ValueError(msg)
            results[criterion_sample.benchmark_id] = criterion_sample
    return results


def _criterion_scope_prefix(benchmark_id: str) -> str:
    """Return the first Criterion group component for suite filtering."""
    return benchmark_id.split("/", maxsplit=1)[0]


def _benchmark_in_compare_scope(benchmark_id: str, suite: str, scope: str) -> bool:
    """Return whether a Criterion benchmark belongs in the requested comparison."""
    if scope == "all-benches":
        return True
    prefixes = BENCH_COMPARE_GROUP_PREFIXES_BY_SUITE.get(suite, RELEASE_SIGNAL_GROUP_PREFIXES)
    group = _criterion_scope_prefix(benchmark_id)
    return any(group.startswith(prefix) for prefix in prefixes)


def collect_criterion_comparisons(
    criterion_dir: Path,
    baseline_name: str,
    *,
    stat: str = "median",
    suite: str = "release-signal",
    scope: str = "release-signal",
) -> list[CriterionComparison]:
    """Collect Criterion comparisons between ``new`` and a named saved baseline."""
    current = _criterion_estimates_by_id(criterion_dir, "new")
    baseline = _criterion_estimates_by_id(criterion_dir, baseline_name)
    comparisons: list[CriterionComparison] = []

    for benchmark_id, current_path in current.items():
        if not _benchmark_in_compare_scope(benchmark_id, suite, scope):
            continue
        baseline_path = baseline.get(benchmark_id)
        if baseline_path is None:
            continue
        comparisons.append(
            CriterionComparison(
                benchmark_id=benchmark_id,
                baseline=_read_criterion_timing_estimate(baseline_path.estimates, stat),
                current=_read_criterion_timing_estimate(current_path.estimates, stat),
            )
        )

    return comparisons


def collect_performance_rows(
    criterion_dir: Path,
    baseline_name: str,
    *,
    suite: str = "release-signal",
    scope: str = "release-signal",
    comparison_note: str = "",
) -> tuple[PerformanceRow, ...]:
    """Collect comparable and one-sided Criterion rows for retained artifacts."""
    current = _criterion_estimates_by_id(criterion_dir, "new")
    baseline = _criterion_estimates_by_id(criterion_dir, baseline_name)
    rows: list[PerformanceRow] = []
    for benchmark_id in sorted(set(current) | set(baseline)):
        if not _benchmark_in_compare_scope(benchmark_id, suite, scope):
            continue
        identity = current.get(benchmark_id) or baseline[benchmark_id]
        group = identity.group
        benchmark = identity.benchmark
        current_estimate = _read_criterion_timing_estimate(current[benchmark_id].estimates, "median") if benchmark_id in current else None
        baseline_estimate = _read_criterion_timing_estimate(baseline[benchmark_id].estimates, "median") if benchmark_id in baseline else None
        if current_estimate is not None and baseline_estimate is not None:
            if comparison_note:
                coverage_status = "not-comparable"
                coverage_note = comparison_note
            elif current_estimate.confidence_level != baseline_estimate.confidence_level:
                coverage_status = "not-comparable"
                coverage_note = "Criterion confidence levels differ between revisions."
            else:
                coverage_status = "comparable"
                coverage_note = ""
        elif current_estimate is not None:
            coverage_status = "current-only"
            coverage_note = "No matching baseline sample was present."
        else:
            coverage_status = "baseline-only"
            coverage_note = "No matching current sample was present."
        rows.append(
            PerformanceRow(
                suite=suite,
                scope=scope,
                benchmark_id=benchmark_id,
                group=group,
                benchmark=benchmark,
                coverage_status=coverage_status,
                coverage_note=coverage_note,
                baseline=baseline_estimate,
                current=current_estimate,
            )
        )
    if not rows:
        msg = f"no Criterion rows found for suite {suite!r}, scope {scope!r}, and baseline {baseline_name!r}"
        raise ValueError(msg)
    return tuple(rows)


def _criterion_comparison_table(comparisons: list[CriterionComparison], baseline_name: str) -> str:
    """Render Criterion comparisons as grouped Markdown tables."""
    sections: list[str] = []
    by_group: dict[str, list[CriterionComparison]] = {}
    for comparison in comparisons:
        by_group.setdefault(_criterion_scope_prefix(comparison.benchmark_id), []).append(comparison)

    for group in sorted(by_group):
        lines = [
            f"### {group}",
            "",
            f"| Benchmark | {baseline_name} | Latest | Change | Speedup |",
            "|-----------|-------:|-------:|-------:|--------:|",
        ]
        for comparison in sorted(by_group[group], key=lambda item: item.benchmark_id):
            label = comparison.benchmark_id.removeprefix(f"{group}/")
            lines.append(
                "| "
                + " | ".join(
                    [
                        label,
                        _format_ns(comparison.baseline_ns),
                        _format_ns(comparison.current_ns),
                        _format_pct_change(comparison.percent_change),
                        f"{comparison.speedup:.2f}x",
                    ]
                )
                + " |"
            )
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def _how_to_update_section() -> str:
    """Return the standard release-performance workflow footer."""
    return """## How to Update

Local performance reports are generated in isolated temporary worktrees:

```bash
# Local development: compare the current tree with the latest release
just performance-local

# Release PR: measure, retain, validate, and promote documentation
just performance-release

# Rebuild and promote documentation from retained CSV/JSON only
just performance-doc

# GitHub Release benchmark assets
just performance-github-assets

# Explicit repair
just performance-release <current-tag> <previous-tag>
```

`just performance-local` writes `performance.md` plus retained `performance.csv` and
`performance.provenance.json` under `target/bench-reports/` without promoting documentation.
`just performance-github-assets` writes a `github-assets-performance.*` bundle without local
Cargo benchmark runs. New release archives must contain the supported versioned measurement
metadata. Existing legacy archives remain loadable as provenance-limited absolute timing
evidence, but they cannot be promoted. GitHub-asset ratios are always suppressed because the
archives were measured in separate sessions. Local-worktree ratios require compatible hosts,
toolchains, harnesses, normalized measurement plans, completed targets, and confidence levels.
`just performance-doc` consumes the retained canonical CSV/JSON pair without Cargo or
measurement worktrees and rejects incomplete, invalid, stale, same-version, or scientifically
non-comparable inputs. `just performance-release` retains and reload-validates the same bundle,
copies the exact CSV/provenance bytes to `docs/archive/performance/data/`, and promotes the
documentation with per-file atomic replacement plus rollback for caught failures. After a hard
interruption, inspect the destinations and rerun the command.

CSV is the canonical tabular artifact because these small audit records are diffable and usable
without a dataframe runtime. Notebooks may derive Parquet caches for analysis, but Parquet is not
an accepted promotion input and must be regenerated from the validated CSV.

Release-comparison commands are release evidence, not routine pre-`just ci` checks.
Older curated reports and the exact evidence for new promotions are archived in
`docs/archive/performance/`.

See `benches/README.md` for the full Delaunay benchmark workflow.
"""


def _normalize_how_to_update(text: str) -> str:
    """Replace or append the standard release-performance workflow footer."""
    section = _how_to_update_section()
    if HOW_TO_UPDATE_RE.search(text):
        return HOW_TO_UPDATE_RE.sub(section, text)
    return f"{text.rstrip()}\n\n{section}"


def render_criterion_comparison_report(
    repo_root: Path,
    comparisons: list[CriterionComparison],
    settings: CriterionReportSettings,
) -> str:
    """Render a Markdown report for Criterion saved-baseline comparisons."""
    version = _read_cargo_package_version(repo_root)
    short_hash, branch = _get_git_info(repo_root)
    now = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    table = _criterion_comparison_table(comparisons, settings.baseline_name)

    lines = [
        "# Benchmark Performance",
        "",
        f"**delaunay** v{version} · `{short_hash}` ({branch}) · {now}",
        f"**Statistic**: {settings.stat}",
        f"**Suite**: {settings.suite}",
        f"**Scope**: {settings.scope}",
        "",
        *_benchmark_report_environment_lines(repo_root),
        "",
        "## Benchmark Results",
        "",
        f"Comparison against baseline **{settings.baseline_name}**:",
        "",
        "Negative change = faster. Speedup > 1.00x = improvement.",
        "",
        table,
        "",
        _how_to_update_section().rstrip(),
        "",
    ]
    return "\n".join(lines)


def write_criterion_comparison_report(repo_root: Path, request: CriterionReportRequest) -> bool:
    """Write a Markdown report comparing current Criterion output with a saved baseline."""
    criterion_dir = request.criterion_dir
    if criterion_dir is None:
        criterion_dir = repo_root / "target" / "criterion"
    elif not criterion_dir.is_absolute():
        criterion_dir = repo_root / criterion_dir

    if not criterion_dir.is_dir():
        print(
            f"No Criterion results found at {criterion_dir}.\nRun benchmarks first:\n  just bench-latest\n",
            file=sys.stderr,
        )
        return False

    comparisons = collect_criterion_comparisons(
        criterion_dir,
        request.baseline_name,
        stat=request.stat,
        suite=request.suite,
        scope=request.scope,
    )
    if not comparisons:
        print(
            f"No comparison data found for baseline {request.baseline_name!r}.\n"
            f"Save a baseline first:\n  just bench-save-baseline {request.baseline_name}\n"
            "Then run benchmarks:\n  just bench-latest\n",
            file=sys.stderr,
        )
        return False

    output_path = request.output if request.output.is_absolute() else repo_root / request.output
    report = render_criterion_comparison_report(
        repo_root,
        comparisons,
        CriterionReportSettings(
            baseline_name=request.baseline_name,
            stat=request.stat,
            suite=request.suite,
            scope=request.scope,
        ),
    )
    _write_text_atomic(output_path, report)
    print(f"📊 Wrote {output_path}")
    return True


def _format_artifact_estimate(estimate: TimingEstimate | None) -> str:
    """Format one retained timing estimate and confidence interval."""
    if estimate is None:
        return "—"
    confidence = estimate.confidence_level * 100.0
    return f"{_format_ns(estimate.median_ns)} [{_format_ns(estimate.ci_lower_ns)}, {_format_ns(estimate.ci_upper_ns)}] ({confidence:g}% CI)"


def _artifact_comparison_table(bundle: PerformanceBundle) -> str:
    """Render retained rows as grouped Markdown tables."""
    sections: list[str] = []
    by_group: dict[str, list[PerformanceRow]] = {}
    for row in bundle.sorted_rows:
        by_group.setdefault(row.group, []).append(row)

    baseline_name = bundle.context.release.baseline
    for group in sorted(by_group):
        lines = [
            f"### {group}",
            "",
            f"| Benchmark | {baseline_name} (median + CI) | Current (median + CI) | Change | Speedup | Coverage |",
            "|-----------|----------------------------:|-----------------------:|-------:|--------:|----------|",
        ]
        for row in by_group[group]:
            if row.coverage_status == "comparable" and row.baseline is not None and row.current is not None:
                percent_change = ((row.current.median_ns - row.baseline.median_ns) / row.baseline.median_ns) * 100.0
                change = _format_pct_change(percent_change)
                speedup = f"{row.baseline.median_ns / row.current.median_ns:.2f}x"
                coverage = "Comparable"
            else:
                change = "—"
                speedup = "—"
                coverage = row.coverage_note
            lines.append(
                "| "
                + " | ".join(
                    (
                        row.benchmark,
                        _format_artifact_estimate(row.baseline),
                        _format_artifact_estimate(row.current),
                        change,
                        speedup,
                        coverage,
                    )
                )
                + " |"
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _artifact_host_lines(bundle: PerformanceBundle) -> list[str]:
    """Render measurement and publication host provenance."""
    context = bundle.context
    publication = context.publication_host
    measurement_lines: list[str] = []
    for label, measurement in (
        ("Current measurement", context.current_measurement_host),
        ("Baseline measurement", context.baseline_measurement_host),
    ):
        if measurement.status == "recorded":
            measurement_lines.extend(
                (
                    f"- **{label} CPU**: {measurement.cpu}",
                    f"- **{label} OS**: {measurement.operating_system}",
                    f"- **{label} architecture**: {measurement.architecture}",
                )
            )
        else:
            measurement_lines.append(f"- **{label} host**: unavailable — {measurement.reason}")
            if measurement.operating_system:
                measurement_lines.append(f"- **{label} recorded OS**: {measurement.operating_system}")
            if measurement.architecture:
                measurement_lines.append(f"- **{label} recorded architecture**: {measurement.architecture}")
    return [
        *measurement_lines,
        f"- **Publication CPU**: {publication.cpu}",
        f"- **Publication OS**: {publication.operating_system}",
        f"- **Publication architecture**: {publication.architecture}",
    ]


def _artifact_revision_lines(label: str, context: ArtifactContext, side: Literal["current", "baseline"]) -> list[str]:
    """Render one revision's source, command, and toolchain provenance."""
    evidence = context.current_source if side == "current" else context.baseline_source
    toolchain = context.current_toolchain if side == "current" else context.baseline_toolchain
    commands = context.current_commands if side == "current" else context.baseline_commands
    completed_targets = context.current_completed_targets if side == "current" else context.baseline_completed_targets
    acquisition_commands = context.current_acquisition_commands if side == "current" else context.baseline_acquisition_commands
    artifact = context.current_artifact if side == "current" else context.baseline_artifact
    lines = [
        f"**{label} revision**:",
        "",
        f"- Version/ref: `{evidence.version}` / `{evidence.ref}`",
        f"- Commit: `{evidence.commit}`",
        f"- Revision timestamp: `{evidence.revision_timestamp}`",
        f"- Cargo profile: `{toolchain.cargo_profile}`",
        f"- Criterion artifact origin: `{artifact.origin}`",
        f"- Criterion content SHA-256: `{artifact.content_sha256}`",
        f"- Criterion sample: `{artifact.sample_name}`",
    ]
    if evidence.limitation:
        lines.append(f"- Source evidence limitation: {evidence.limitation}")
    else:
        lines.extend(
            (
                f"- Git clean: `{str(evidence.git_clean).lower()}`",
                f"- Source-state SHA-256: `{evidence.source_state_sha256}`",
            )
        )
    if toolchain.limitation:
        lines.append(f"- Toolchain evidence limitation: {toolchain.limitation}")
    else:
        lines.extend(
            (
                f"- rustc: `{toolchain.rustc}`",
                f"- Criterion: `{toolchain.criterion_version}`",
                f"- Cargo.lock SHA-256: `{toolchain.cargo_lock_sha256}`",
                f"- Harness SHA-256: `{toolchain.harness_sha256}`",
                f"- Configuration SHA-256: `{toolchain.configuration_sha256}`",
                f"- Measurement-plan SHA-256: `{toolchain.measurement_plan_sha256}`",
            )
        )
    if artifact.archive_sha256 is not None:
        lines.append(f"- Release archive SHA-256: `{artifact.archive_sha256}`")
    if completed_targets:
        lines.append(f"- Completed benchmark targets: `{', '.join(completed_targets)}`")
    else:
        lines.append("- Completed benchmark targets: unavailable")
    if commands:
        lines.extend(f"- Measurement command: `{' '.join(command)}`" for command in commands)
    else:
        lines.append("- Measurement commands: unavailable")
    lines.extend(f"- Acquisition command: `{' '.join(command)}`" for command in acquisition_commands)
    return lines


def _artifact_evidence_path(path: Path) -> str:
    """Return a Markdown-safe artifact path for a report notice."""
    rendered = path.as_posix()
    if not rendered or "|" in rendered or "`" in rendered or any(ord(char) < 32 or ord(char) == 127 for char in rendered):
        msg = f"artifact evidence path must be single-line Markdown-safe text: {path}"
        raise ValueError(msg)
    return rendered


def render_performance_bundle(
    bundle: PerformanceBundle,
    *,
    evidence_paths: ArtifactPaths,
    evidence_state: Literal["scratch", "promoted"],
) -> str:
    """Render a report exclusively from one validated retained bundle."""
    context = bundle.context
    current = context.current_source
    csv_payload, _ = serialize_bundle(bundle)
    if evidence_state not in ("scratch", "promoted"):
        msg = f"unsupported evidence state: {evidence_state!r}"
        raise ValueError(msg)
    evidence_label = "Retained scratch evidence" if evidence_state == "scratch" else "Promoted evidence"
    evidence_csv = _artifact_evidence_path(evidence_paths.csv)
    evidence_provenance = _artifact_evidence_path(evidence_paths.provenance)
    has_comparisons = any(row.coverage_status == "comparable" for row in bundle.rows)
    lines = [
        "# Benchmark Performance",
        "",
        "> [!IMPORTANT]",
        "> Generated by `benchmark-utils` from a validated CSV/provenance pair; do not edit this report directly.",
        f"> {evidence_label}: `{evidence_csv}` and `{evidence_provenance}` (CSV SHA-256 `{hashlib.sha256(csv_payload).hexdigest()}`).",
        "> Edit workflow guidance in `benches/README.md` or `docs/dev/commands.md`, then rerun the named performance workflow.",
        "",
        f"**delaunay** v{context.release.current.removeprefix('v')} · `{current.commit}` ({current.ref}) · {current.revision_timestamp}",
        f"**Statistic**: {context.statistic}",
        f"**Suite**: {context.suite}",
        f"**Scope**: {context.scope}",
        "",
        "## Environment and Provenance",
        "",
        f"- **Measurement mode**: `{context.measurement_mode}`",
        *_artifact_host_lines(bundle),
        "",
        *_artifact_revision_lines("Current", context, "current"),
        "",
        *_artifact_revision_lines("Baseline", context, "baseline"),
        "",
        "## Benchmark Results",
        "",
        (
            f"Comparison against baseline **{context.release.baseline}**:"
            if has_comparisons
            else f"Measurements for current and baseline **{context.release.baseline}**:"
        ),
        "",
        (
            "Negative change = faster. Speedup > 1.00x = improvement. Each confidence interval includes its retained Criterion confidence level."
            if has_comparisons
            else "Ratios are suppressed because the retained provenance does not establish scientifically comparable measurements."
        ),
        "",
        _artifact_comparison_table(bundle),
        "",
        _how_to_update_section().rstrip(),
        "",
    ]
    return "\n".join(lines)


def render_performance_artifacts(paths: ArtifactPaths) -> str:
    """Reload, validate, and render a retained artifact pair."""
    return render_performance_bundle(load_bundle(paths), evidence_paths=paths, evidence_state="scratch")


def parse_performance_report_id(text: str) -> PerformanceReportId:
    """Parse current and baseline release tags from a benchmark report."""
    version_match = DELAUNAY_REPORT_VERSION_RE.search(text)
    if version_match is None:
        msg = "could not find delaunay version line in benchmark report"
        raise ValueError(msg)
    baseline_match = DELAUNAY_REPORT_BASELINE_RE.search(text)
    if baseline_match is None:
        msg = "could not find comparison baseline line in benchmark report"
        raise ValueError(msg)
    return PerformanceReportId(
        current_tag=normalize_release_tag(version_match.group("version")),
        baseline_tag=normalize_release_tag(baseline_match.group("baseline")),
    )


def _archive_index_text(archive_dir: Path) -> str:
    """Return the archive README text for curated performance reports."""
    reports = sorted(path.name for path in archive_dir.glob("*.md") if path.name != "README.md")
    lines = [
        "# Archived Performance Reports",
        "",
        "Older release-to-release benchmark comparisons are archived here.",
        "`docs/PERFORMANCE.md` contains the latest curated comparison.",
        "",
    ]
    if reports:
        lines.extend(f"- [{name.removesuffix('.md')}]({name})" for name in reports)
    else:
        lines.append("- No archived performance reports yet.")
    return "\n".join(lines) + "\n"


def update_performance_archive_index(archive_dir: Path) -> None:
    """Write the sorted performance archive index."""
    _write_text_atomic(archive_dir / "README.md", _archive_index_text(archive_dir))


def _durable_performance_artifact_paths(archive_dir: Path, report_id: PerformanceReportId) -> ArtifactPaths:
    """Return tracked evidence paths for one promoted release pair."""
    stem = f"{report_id.current_tag}-vs-{report_id.baseline_tag}"
    data_dir = archive_dir / "data"
    return ArtifactPaths(csv=data_dir / f"{stem}.csv", provenance=data_dir / f"{stem}.provenance.json")


def _repository_relative_path(project_root: Path, path: Path, *, label: str) -> Path:
    """Resolve *path* and return its location relative to the repository root."""
    resolved_root = project_root.resolve(strict=False)
    resolved_path = path.resolve(strict=False)
    try:
        return resolved_path.relative_to(resolved_root)
    except ValueError as error:
        msg = f"{label} must be contained by repository root {resolved_root}, got {resolved_path}"
        raise ValueError(msg) from error


def _promoted_evidence_paths(durable: ArtifactPaths, *, project_root: Path) -> ArtifactPaths:
    """Return validated repository-relative evidence paths for tracked reports."""
    return ArtifactPaths(
        csv=_repository_relative_path(project_root, durable.csv, label="promoted performance CSV"),
        provenance=_repository_relative_path(
            project_root,
            durable.provenance,
            label="promoted performance provenance",
        ),
    )


def _validated_promotion_source(
    source: Path,
    artifacts: ArtifactPaths,
    expected: PerformanceReportId,
    durable_artifacts: ArtifactPaths,
    project_root: Path,
) -> tuple[PerformanceBundle, str, PerformanceReportId]:
    """Validate one canonical report and its independently expected identity."""
    bundle = load_bundle(artifacts)
    bundle.require_promotable()
    source_text = _normalize_how_to_update(_read_text(source))
    rendered_text = _normalize_how_to_update(
        render_performance_bundle(
            bundle,
            evidence_paths=_promoted_evidence_paths(durable_artifacts, project_root=project_root),
            evidence_state="promoted",
        )
    )
    if source_text != rendered_text:
        msg = "benchmark report is not the canonical rendering of its retained artifact pair"
        raise ValueError(msg)
    source_id = parse_performance_report_id(source_text)
    normalized_expected = PerformanceReportId(
        current_tag=normalize_release_tag(expected.current_tag),
        baseline_tag=normalize_release_tag(expected.baseline_tag),
    )
    if source_id != normalized_expected:
        msg = (
            "benchmark report does not match requested release pair: "
            f"found {source_id.current_tag} vs {source_id.baseline_tag}, "
            f"expected {normalized_expected.current_tag} vs {normalized_expected.baseline_tag}"
        )
        raise ValueError(msg)
    if source_id.current_tag == source_id.baseline_tag:
        msg = "cannot promote a same-version local performance comparison"
        raise ValueError(msg)
    bundle_id = PerformanceReportId(
        current_tag=bundle.context.release.current,
        baseline_tag=bundle.context.release.baseline,
    )
    if bundle_id != source_id:
        msg = f"benchmark report identity {source_id} does not match retained artifact identity {bundle_id}"
        raise ValueError(msg)
    return bundle, source_text, source_id


def _promotion_archive_destination(
    current: Path,
    archive_dir: Path,
    source_id: PerformanceReportId,
) -> tuple[str | None, Path | None]:
    """Return the current report payload and any required archive destination."""
    if not current.exists():
        return None, None
    current_text = _normalize_how_to_update(_read_text(current))
    current_id = parse_performance_report_id(current_text)
    archive_path = archive_dir / current_id.archive_name if current_id != source_id else None
    return current_text, archive_path


def _reject_conflicting_payload(path: Path, payload: bytes, *, description: str) -> None:
    """Reject a pre-existing destination whose exact bytes differ."""
    if path.exists() and path.read_bytes() != payload:
        msg = f"existing {description} conflicts with retained evidence: {path}"
        raise ValueError(msg)


def _plan_performance_promotion(
    *,
    source: Path,
    artifacts: ArtifactPaths,
    destinations: PerformancePromotionDestinations,
    expected: PerformanceReportId,
) -> PerformancePromotionPlan:
    """Validate all promotion inputs and conflicts before the first mutation."""
    current = destinations.current
    archive_dir = destinations.archive_dir
    project_root = destinations.project_root
    normalized_expected = PerformanceReportId(
        current_tag=normalize_release_tag(expected.current_tag),
        baseline_tag=normalize_release_tag(expected.baseline_tag),
    )
    durable_artifacts = _durable_performance_artifact_paths(archive_dir, normalized_expected)
    _, source_text, source_id = _validated_promotion_source(
        source,
        artifacts,
        expected,
        durable_artifacts,
        project_root,
    )
    current_text, archive_path = _promotion_archive_destination(current, archive_dir, source_id)

    index_path = archive_dir / "README.md"
    if source_id != normalized_expected:
        msg = "validated report identity changed while planning promotion"
        raise ValueError(msg)
    paths = {
        "source report": source,
        "source CSV": artifacts.csv,
        "source provenance": artifacts.provenance,
        "current report": current,
        "archive index": index_path,
        "durable CSV": durable_artifacts.csv,
        "durable provenance": durable_artifacts.provenance,
    }
    if archive_path is not None:
        paths["archive report"] = archive_path
    ensure_distinct_paths(paths)

    if archive_path is not None and archive_path.exists() and current_text is not None:
        existing_archive = _normalize_how_to_update(_read_text(archive_path))
        if existing_archive != current_text:
            msg = f"existing performance archive conflicts with the current report: {archive_path}"
            raise ValueError(msg)
    source_csv = artifacts.csv.read_bytes()
    source_provenance = artifacts.provenance.read_bytes()
    _reject_conflicting_payload(durable_artifacts.csv, source_csv, description="durable performance CSV")
    _reject_conflicting_payload(
        durable_artifacts.provenance,
        source_provenance,
        description="durable performance provenance",
    )

    mutation_paths = (
        current,
        index_path,
        durable_artifacts.csv,
        durable_artifacts.provenance,
        *(() if archive_path is None else (archive_path,)),
    )
    for label, path in (
        ("current performance report", current),
        ("performance archive index", index_path),
        ("durable performance CSV", durable_artifacts.csv),
        ("durable performance provenance", durable_artifacts.provenance),
        *(() if archive_path is None else (("archived performance report", archive_path),)),
    ):
        _repository_relative_path(project_root, path, label=label)
    return PerformancePromotionPlan(
        report_id=source_id,
        source_text=source_text,
        current_text=current_text,
        archive_path=archive_path,
        durable_artifacts=durable_artifacts,
        source_csv=source_csv,
        source_provenance=source_provenance,
        mutation_paths=mutation_paths,
    )


def _apply_performance_promotion(plan: PerformancePromotionPlan, *, current: Path, archive_dir: Path) -> None:
    """Apply one validated plan and roll back caught failures."""
    snapshots = tuple((path, path.read_bytes() if path.exists() else None) for path in plan.mutation_paths)
    try:
        if plan.archive_path is not None and not plan.archive_path.exists() and plan.current_text is not None:
            _write_text_atomic(plan.archive_path, plan.current_text)
        if not plan.durable_artifacts.csv.exists():
            _write_bytes_atomic(plan.durable_artifacts.csv, plan.source_csv)
        if not plan.durable_artifacts.provenance.exists():
            _write_bytes_atomic(plan.durable_artifacts.provenance, plan.source_provenance)
        _write_text_atomic(current, plan.source_text)
        update_performance_archive_index(archive_dir)
    except BaseException as exc:
        restore_errors: list[BaseException] = []
        for path, payload in reversed(snapshots):
            try:
                _restore_file_snapshot(path, payload)
            except OSError as restore_exc:
                restore_errors.append(restore_exc)
        if restore_errors:
            msg = "performance report promotion and rollback both failed"
            raise BaseExceptionGroup(msg, [exc, *restore_errors]) from exc
        raise


def promote_performance_report(
    *,
    source: Path,
    artifacts: ArtifactPaths,
    destinations: PerformancePromotionDestinations,
    expected: PerformanceReportId,
) -> PerformanceReportId:
    """Archive the old report and durably promote its exact evidence pair."""
    plan = _plan_performance_promotion(
        source=source,
        artifacts=artifacts,
        destinations=destinations,
        expected=expected,
    )
    _apply_performance_promotion(
        plan,
        current=destinations.current,
        archive_dir=destinations.archive_dir,
    )
    return plan.report_id


def _format_command_failure(command: list[str], exc: subprocess.CalledProcessError) -> str:
    """Return a readable command failure with captured output."""
    parts = [f"command failed ({exc.returncode}): {' '.join(command)}"]
    if exc.stdout:
        parts.append(f"stdout:\n{exc.stdout.strip()}")
    if exc.stderr:
        parts.append(f"stderr:\n{exc.stderr.strip()}")
    return "\n".join(parts)


def _run_tool(command: str, args: list[str], *, cwd: Path, options: ToolRunOptions | None = None) -> None:
    """Run a support command and translate subprocess failures."""
    resolved_options = options or ToolRunOptions()
    try:
        run_safe_command(
            command,
            args,
            cwd=cwd,
            timeout=resolved_options.timeout,
            env=resolved_options.env,
            capture_output=not resolved_options.stream_output,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(_format_command_failure([command, *args], exc)) from exc


def _progress(message: str) -> None:
    """Write one immediately visible performance-workflow phase marker."""
    print(f"[performance] {message}", file=sys.stderr, flush=True)


def _run_git(args: list[str], *, cwd: Path, timeout: int = RELEASE_COMMAND_TIMEOUT_SECONDS) -> None:
    """Run a git command and translate subprocess failures."""
    try:
        run_git_command(args, cwd=cwd, timeout=timeout)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(_format_command_failure(["git", *args], exc)) from exc


def _github_release_list(repo_root: Path) -> object:
    """Return GitHub release JSON from ``gh release list``."""
    command = [
        "release",
        "list",
        "--json",
        "tagName,isDraft,isPrerelease,publishedAt",
        "--limit",
        "100",
    ]
    try:
        result = run_safe_command("gh", command, cwd=repo_root, timeout=RELEASE_COMMAND_TIMEOUT_SECONDS)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(_format_command_failure(["gh", *command], exc)) from exc
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        msg = "could not parse GitHub release list JSON"
        raise RuntimeError(msg) from exc


def _stable_published_releases(releases: object) -> list[PublishedRelease]:
    """Parse stable published semver releases from GitHub release JSON."""
    if not isinstance(releases, list):
        msg = "expected GitHub release list to be a JSON array"
        raise TypeError(msg)

    parsed_releases: list[GitHubRelease] = []
    seen_tags: set[str] = set()
    for index, raw_release in enumerate(releases):
        release = GitHubRelease.from_raw(raw_release, index=index)
        if release.tag in seen_tags:
            msg = f"duplicate GitHub release tag after normalization: {release.tag!r}"
            raise ValueError(msg)
        seen_tags.add(release.tag)
        parsed_releases.append(release)

    stable_releases: list[PublishedRelease] = []
    for release in parsed_releases:
        if release.is_draft or release.is_prerelease:
            continue
        try:
            _stable_semver_sort_key(release.tag)
        except ValueError:
            continue
        if release.published_at is None:
            msg = f"published GitHub release {release.tag!r} is missing publishedAt"
            raise ValueError(msg)
        stable_releases.append(PublishedRelease(tag=release.tag, published_at=release.published_at))

    return stable_releases


def _published_stable_releases(repo_root: Path) -> list[PublishedRelease]:
    """Return stable published releases for the current GitHub repository."""
    return _stable_published_releases(_github_release_list(repo_root))


def published_stable_release_tags(repo_root: Path) -> list[str]:
    """Return stable tag names from published, non-draft GitHub releases."""
    return [release.tag for release in _published_stable_releases(repo_root)]


def _latest_published_release(repo_root: Path) -> PublishedRelease:
    """Return the latest published stable release by publish timestamp."""
    stable_releases = _published_stable_releases(repo_root)
    if not stable_releases:
        msg = "expected at least one published stable semver release"
        raise RuntimeError(msg)
    return max(stable_releases, key=lambda release: release.published_at)


def _previous_release_from_list(stable_releases: list[PublishedRelease], current_tag: str) -> PublishedRelease:
    """Return the previous stable semver release before current_tag."""
    current_key = _stable_semver_sort_key(current_tag)
    previous = sorted(
        (release for release in stable_releases if _stable_semver_sort_key(release.tag) < current_key),
        key=lambda release: _stable_semver_sort_key(release.tag),
    )
    if not previous:
        msg = f"could not find a previous stable semver release before {current_tag}"
        raise RuntimeError(msg)
    return previous[-1]


def _previous_published_release(repo_root: Path, current_tag: str) -> PublishedRelease:
    """Return the previous published stable semver release."""
    return _previous_release_from_list(_published_stable_releases(repo_root), current_tag)


def _published_release_pair(repo_root: Path) -> PerformanceReportId:
    """Return the latest published stable release pair."""
    stable_releases = _published_stable_releases(repo_root)
    if len(stable_releases) < 2:
        msg = "expected at least two published stable semver releases"
        raise RuntimeError(msg)
    current = max(stable_releases, key=lambda release: release.published_at)
    previous = _previous_release_from_list(stable_releases, current.tag)
    return PerformanceReportId(current_tag=current.tag, baseline_tag=previous.tag)


def _normalize_worktree_ref_for_tag(worktree_ref: str, current_tag: str) -> str:
    """Use the normalized current tag when a bare matching tag was requested."""
    try:
        normalized_ref = normalize_release_tag(worktree_ref)
    except ValueError:
        return worktree_ref
    return current_tag if normalized_ref == current_tag else worktree_ref


def resolve_performance_request(options: PerformanceRequestOptions) -> ResolvedPerformanceRequest:
    """Resolve explicit, package-inferred, or latest-published release arguments."""
    requested_modes = sum((options.published_latest, options.infer_release, options.current_vs_latest))
    if requested_modes > 1:
        msg = "choose only one of --published-latest, --infer-release, or --current-vs-latest"
        raise ValueError(msg)

    if options.published_latest:
        if options.current_tag is not None or options.baseline_tag is not None:
            msg = "do not pass current_tag or baseline_tag with --published-latest"
            raise ValueError(msg)
        published_pair = _published_release_pair(options.repo_root)
        worktree_ref = published_pair.current_tag if options.worktree_ref == "HEAD" else options.worktree_ref
        return ResolvedPerformanceRequest(
            current_tag=published_pair.current_tag,
            baseline_tag=published_pair.baseline_tag,
            worktree_ref=worktree_ref,
            tags_to_fetch=(published_pair.current_tag, published_pair.baseline_tag),
        )

    if options.infer_release:
        if options.current_tag is not None or options.baseline_tag is not None:
            msg = "do not pass current_tag or baseline_tag with --infer-release"
            raise ValueError(msg)
        current_tag = _current_package_tag(options.repo_root)
        baseline_tag = _previous_published_release(options.repo_root, current_tag).tag
        return ResolvedPerformanceRequest(current_tag=current_tag, baseline_tag=baseline_tag, worktree_ref=options.worktree_ref, tags_to_fetch=(baseline_tag,))

    if options.current_vs_latest:
        if options.current_tag is not None or options.baseline_tag is not None:
            msg = "do not pass current_tag or baseline_tag with --current-vs-latest"
            raise ValueError(msg)
        current_tag = _current_package_tag(options.repo_root)
        latest = _latest_published_release(options.repo_root).tag
        if current_tag == latest:
            msg = (
                f"current package tag and latest published release are both {latest}; "
                "use a named local Criterion baseline for same-version experiments, "
                "or rerun after updating the package version"
            )
            raise ValueError(msg)
        return ResolvedPerformanceRequest(current_tag=current_tag, baseline_tag=latest, worktree_ref=options.worktree_ref, tags_to_fetch=(latest,))

    if options.current_tag is None or options.baseline_tag is None:
        msg = "current_tag and baseline_tag are required unless an inference mode is used"
        raise ValueError(msg)
    current_tag = normalize_release_tag(options.current_tag)
    baseline_tag = normalize_release_tag(options.baseline_tag)
    return ResolvedPerformanceRequest(
        current_tag=current_tag,
        baseline_tag=baseline_tag,
        worktree_ref=_normalize_worktree_ref_for_tag(options.worktree_ref, current_tag),
        tags_to_fetch=(baseline_tag,),
    )


def _fetch_release_tags(*, repo_root: Path, tags: tuple[str, ...], include_current: str | None = None) -> None:
    """Fetch the release tags required before adding detached worktrees."""
    tags_to_fetch = tags
    if include_current is not None and include_current not in tags_to_fetch:
        tags_to_fetch = (*tags_to_fetch, include_current)
    if not tags_to_fetch:
        return
    refspecs = [f"refs/tags/{tag}:refs/tags/{tag}" for tag in dict.fromkeys(tags_to_fetch)]
    _run_git(["fetch", "origin", *refspecs], cwd=repo_root)


def _current_rust_toolchain(checkout: Path) -> str | None:
    """Return the rust-toolchain channel for benchmark temp worktrees."""
    rust_toolchain = checkout / "rust-toolchain.toml"
    if not rust_toolchain.exists():
        return None
    data = tomllib.loads(_read_text(rust_toolchain))
    toolchain = data.get("toolchain")
    if not isinstance(toolchain, dict):
        return None
    channel = toolchain.get("channel")
    return channel if isinstance(channel, str) else None


def _benchmark_env(checkout: Path) -> dict[str, str] | None:
    """Set RUSTUP_TOOLCHAIN from rust-toolchain.toml unless the user already did."""
    if "RUSTUP_TOOLCHAIN" in os.environ:
        return None
    toolchain = _current_rust_toolchain(checkout)
    if toolchain is None:
        return None
    env = os.environ.copy()
    env["RUSTUP_TOOLCHAIN"] = toolchain
    return env


def _run_tool_output(
    command: str,
    args: list[str],
    *,
    cwd: Path,
    timeout: int = RELEASE_COMMAND_TIMEOUT_SECONDS,
    env: dict[str, str] | None = None,
) -> str:
    """Run a support command and return non-empty stripped stdout."""
    try:
        result = run_safe_command(command, args, cwd=cwd, timeout=timeout, env=env)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(_format_command_failure([command, *args], exc)) from exc
    output = result.stdout.strip()
    if not output:
        msg = f"command produced empty stdout: {command} {' '.join(args)}"
        raise RuntimeError(msg)
    return output


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one required file."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        msg = f"could not hash required performance input {path}: {exc}"
        raise OSError(msg) from exc


def _directory_digest(directory: Path) -> str:
    """Hash every relative file path and payload below a required directory."""
    if not directory.is_dir():
        msg = f"could not hash required performance directory {directory}"
        raise FileNotFoundError(msg)
    digest = hashlib.sha256()
    files = tuple(path for path in sorted(directory.rglob("*")) if path.is_file())
    if not files:
        msg = f"performance directory contains no files: {directory}"
        raise ValueError(msg)
    for path in files:
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _criterion_sample_digest(criterion_dir: Path, sample_name: str) -> str:
    """Hash one named sample across a Criterion tree."""
    digest = hashlib.sha256()
    files: list[Path] = []
    for sample_dir in sorted(path for path in criterion_dir.rglob(sample_name) if path.is_dir() and path.name == sample_name):
        files.extend(path for path in sorted(sample_dir.rglob("*")) if path.is_file())
    if not files:
        msg = f"Criterion sample {sample_name!r} contains no files under {criterion_dir}"
        raise FileNotFoundError(msg)
    for path in files:
        digest.update(path.relative_to(criterion_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _criterion_dependency_version(checkout: Path) -> str:
    """Return the Criterion version selected in Cargo.lock."""
    cargo_lock = checkout / "Cargo.lock"
    data = tomllib.loads(_read_text(cargo_lock))
    packages = data.get("package")
    if not isinstance(packages, list):
        msg = f"could not find package entries in {cargo_lock}"
        raise TypeError(msg)
    versions = {
        package.get("version")
        for package in packages
        if isinstance(package, dict) and package.get("name") == "criterion" and isinstance(package.get("version"), str)
    }
    if len(versions) != 1:
        msg = f"expected exactly one Criterion version in {cargo_lock}, found {sorted(versions)}"
        raise ValueError(msg)
    return cast("str", versions.pop())


def _comparison_targets_for_suite(suite: str, targets: tuple[str, ...] | None) -> tuple[str, ...]:
    """Validate and return an ordered target plan for provenance comparison."""
    requested = BENCH_TARGET_SUITES.get(suite)
    if requested is None:
        msg = f"unsupported benchmark suite: {suite}"
        raise ValueError(msg)
    if targets is None:
        return requested
    ordered = tuple(target for target in requested if target in set(targets))
    if not ordered or targets != ordered:
        msg = f"comparison targets must be a non-empty canonical subset of suite {suite!r}: {targets!r}"
        raise ValueError(msg)
    return targets


def _benchmark_harness_files(
    checkout: Path,
    suite: str,
    comparison_targets: tuple[str, ...] | None = None,
) -> tuple[Path, ...]:
    """Return deterministic benchmark harness inputs for one suite."""
    targets = _comparison_targets_for_suite(suite, comparison_targets)
    relative_paths = [Path("benches") / f"{target}.rs" for target in targets]
    common_dir = checkout / "benches" / "common"
    if common_dir.is_dir():
        relative_paths.extend(path.relative_to(checkout) for path in sorted(common_dir.rglob("*.rs")))
    files = tuple(checkout / relative for relative in relative_paths if (checkout / relative).is_file())
    if not files:
        msg = f"no benchmark harness files found in {checkout}"
        raise FileNotFoundError(msg)
    return files


def _path_content_digest(checkout: Path, paths: tuple[Path, ...]) -> bytes:
    """Hash relative paths and contents into a deterministic identity."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(checkout).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.digest()


def _benchmark_harness_digest(
    checkout: Path,
    suite: str,
    comparison_targets: tuple[str, ...] | None = None,
) -> str:
    """Hash benchmark paths and contents into a deterministic harness identity."""
    return _path_content_digest(checkout, _benchmark_harness_files(checkout, suite, comparison_targets)).hex()


def _benchmark_configuration_digest(checkout: Path) -> str:
    """Hash orchestration files and benchmark-affecting environment settings."""
    relative_paths = (
        Path(".cargo") / "config.toml",
        Path("justfile"),
        Path("rust-toolchain.toml"),
        Path("scripts") / "benchmark_utils.py",
    )
    paths = tuple(checkout / relative for relative in relative_paths if (checkout / relative).is_file())
    digest = hashlib.sha256(_path_content_digest(checkout, paths))
    cargo_manifest = tomllib.loads(_read_text(checkout / "Cargo.toml"))
    package = cargo_manifest.get("package")
    if isinstance(package, dict):
        package = dict(package)
        package.pop("version", None)
        cargo_manifest = {**cargo_manifest, "package": package}
    digest.update(json.dumps(cargo_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\0")
    cargo_lock = tomllib.loads(_read_text(checkout / "Cargo.lock"))
    packages = cargo_lock.get("package")
    if isinstance(packages, list):
        normalized_packages = []
        for package_entry in packages:
            if isinstance(package_entry, dict) and package_entry.get("name") == "delaunay":
                package_entry = dict(package_entry)
                package_entry.pop("version", None)
            normalized_packages.append(package_entry)
        cargo_lock = {**cargo_lock, "package": normalized_packages}
    digest.update(json.dumps(cargo_lock, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\0")
    relevant_environment = sorted(
        (key, value)
        for key, value in os.environ.items()
        if key in {"CARGO_ENCODED_RUSTFLAGS", "RUSTFLAGS", "RUSTUP_TOOLCHAIN"} or key.startswith(("BENCH_", "CRIT_", "DELAUNAY_BENCH_"))
    )
    for key, value in relevant_environment:
        digest.update(f"env:{key}".encode())
        digest.update(b"\0")
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _measurement_plan_digest(
    checkout: Path,
    suite: str,
    comparison_targets: tuple[str, ...] | None = None,
) -> str:
    """Hash the normalized benchmark method independently of measured source."""
    targets = _comparison_targets_for_suite(suite, comparison_targets)
    if suite == "release-signal":
        target_set = set(targets)
        measurements = tuple(measurement for measurement in RELEASE_SIGNAL_MEASUREMENT_PLAN if measurement.target in target_set)
    else:
        present = set(_bench_targets_for_suite(checkout, suite))
        measurements = tuple(BenchmarkTargetMeasurement(target) for target in targets if target in present)
    payload: dict[str, object] = {
        "cargo_features": [],
        "cargo_profile": BENCHMARK_BUILD_FLAVOR,
        "statistic": "median",
        "suite": suite,
        "targets": [
            {
                "command": list(measurement.command),
                "criterion_arguments": list(measurement.criterion_arguments),
                "sampling_mode": measurement.sampling_mode,
                "target": measurement.target,
            }
            for measurement in measurements
        ],
    }
    relevant_environment = sorted(
        (key, value)
        for key, value in os.environ.items()
        if key in {"CARGO_ENCODED_RUSTFLAGS", "RUSTFLAGS", "RUSTUP_TOOLCHAIN"} or key.startswith(("BENCH_", "CRIT_", "DELAUNAY_BENCH_"))
    )
    payload["environment"] = relevant_environment
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_state(checkout: Path, *, version: str, ref: str) -> SourceState:
    """Capture one checkout's commit and tracked working-tree state."""
    commit = run_git_command(["rev-parse", "HEAD"], cwd=checkout, timeout=RELEASE_COMMAND_TIMEOUT_SECONDS).stdout.strip()
    revision_timestamp = run_git_command(
        ["show", "-s", "--format=%cI", "HEAD"],
        cwd=checkout,
        timeout=RELEASE_COMMAND_TIMEOUT_SECONDS,
    ).stdout.strip()
    diff = run_git_command(["diff", "--binary", "HEAD"], cwd=checkout, timeout=RELEASE_COMMAND_TIMEOUT_SECONDS).stdout
    status = run_git_command(
        ["status", "--short", "--untracked-files=no"],
        cwd=checkout,
        timeout=RELEASE_COMMAND_TIMEOUT_SECONDS,
    ).stdout
    state_digest = hashlib.sha256(f"commit {commit}\n".encode() + diff.encode("utf-8")).hexdigest()
    return SourceState(
        version=normalize_release_tag(version),
        commit=commit,
        ref=ref,
        revision_timestamp=revision_timestamp,
        git_clean=not status.strip(),
        source_state_sha256=state_digest,
    )


def _toolchain_state(
    checkout: Path,
    suite: str,
    comparison_targets: tuple[str, ...] | None = None,
) -> ToolchainState:
    """Capture the Rust, Criterion, lockfile, and harness configuration."""
    return ToolchainState(
        rustc=_run_tool_output("rustc", ["--version"], cwd=checkout, env=_benchmark_env(checkout)),
        criterion_version=_criterion_dependency_version(checkout),
        cargo_profile=BENCHMARK_BUILD_FLAVOR,
        cargo_lock_sha256=_sha256_file(checkout / "Cargo.lock"),
        harness_sha256=_benchmark_harness_digest(checkout, suite, comparison_targets),
        configuration_sha256=_benchmark_configuration_digest(checkout),
        measurement_plan_sha256=_measurement_plan_digest(checkout, suite, comparison_targets),
    )


def _revision_evidence(
    checkout: Path,
    *,
    version: str,
    ref: str,
    measurement: RevisionMeasurement,
) -> RevisionEvidence:
    """Capture complete evidence for one measured revision."""
    expected_version = normalize_release_tag(version)
    observed_version = _current_package_tag(checkout)
    if observed_version != expected_version:
        msg = f"measured checkout package version {observed_version} does not match requested release {expected_version}: {checkout}"
        raise ValueError(msg)
    return RevisionEvidence(
        source=_source_state(checkout, version=observed_version, ref=ref),
        toolchain=_toolchain_state(checkout, measurement.suite, measurement.comparison_targets),
        commands=measurement.commands,
        completed_targets=_bench_targets_for_suite(checkout, measurement.suite),
    )


def _recorded_host_identity(repo_root: Path) -> HostIdentity:
    """Capture the current host for local measurement or publication."""
    hardware = HardwareInfo().get_hardware_info(cwd=repo_root)
    return HostIdentity(
        status="recorded",
        cpu=hardware["CPU"],
        operating_system=hardware["OS"],
        architecture=platform.machine() or hardware["TARGET"],
    )


def _apply_current_diff_to_worktree(*, repo_root: Path, worktree: Path) -> None:
    """Apply the current tracked diff to a temporary worktree."""
    diff = run_git_command(["diff", "--binary", "HEAD"], cwd=repo_root, timeout=RELEASE_COMMAND_TIMEOUT_SECONDS).stdout
    if not diff.strip():
        return
    try:
        run_git_command_with_input(["apply", "--index", "--binary"], diff, cwd=worktree, timeout=RELEASE_COMMAND_TIMEOUT_SECONDS)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(_format_command_failure(["git", "apply", "--index", "--binary"], exc)) from exc


def _cargo_manifest_bench_targets(worktree: Path) -> set[str]:
    """Return benchmark target names declared by Cargo.toml."""
    cargo_toml = worktree / "Cargo.toml"
    if not cargo_toml.exists():
        return set()
    data = tomllib.loads(_read_text(cargo_toml))
    benches = data.get("bench")
    if not isinstance(benches, list):
        return set()
    names: set[str] = set()
    for bench in benches:
        if isinstance(bench, dict):
            name = bench.get("name")
            if isinstance(name, str):
                names.add(cast("str", name))
    return names


def _bench_targets_for_suite(worktree: Path, suite: str) -> tuple[str, ...]:
    """Return present Cargo benchmark targets for a Delaunay release suite."""
    requested = BENCH_TARGET_SUITES.get(suite)
    if requested is None:
        msg = f"unsupported benchmark suite: {suite}"
        raise ValueError(msg)
    present = _cargo_manifest_bench_targets(worktree)
    if not present:
        return requested
    return tuple(target for target in requested if target in present)


def _shared_bench_targets(current_worktree: Path, baseline_worktree: Path, suite: str) -> tuple[str, ...]:
    """Return the canonical target plan supported by both measured revisions."""
    current = set(_bench_targets_for_suite(current_worktree, suite))
    baseline = set(_bench_targets_for_suite(baseline_worktree, suite))
    shared = tuple(target for target in BENCH_TARGET_SUITES[suite] if target in current and target in baseline)
    if not shared:
        msg = f"current and baseline revisions share no benchmark targets for suite {suite!r}"
        raise RuntimeError(msg)
    return shared


def _run_saved_baseline_for_suite(
    *,
    worktree: Path,
    baseline_tag: str,
    suite: str,
    env: dict[str, str] | None,
) -> tuple[tuple[str, ...], ...]:
    """Run present release-signal benchmarks and save a named Criterion baseline."""
    targets = _bench_targets_for_suite(worktree, suite)
    if not targets:
        msg = f"no benchmark targets found for suite {suite!r} in {worktree}"
        raise RuntimeError(msg)
    commands: list[tuple[str, ...]] = []
    for target in targets:
        command = (
            "cargo",
            "bench",
            "--profile",
            BENCHMARK_BUILD_FLAVOR,
            "--bench",
            target,
            "--",
            "--save-baseline",
            baseline_tag,
        )
        _progress(f"running baseline benchmark {target} for {baseline_tag}")
        _run_tool(
            command[0],
            list(command[1:]),
            cwd=worktree,
            options=ToolRunOptions(
                timeout=RELEASE_BENCH_TIMEOUT_SECONDS,
                env=env,
                stream_output=True,
            ),
        )
        _progress(f"completed baseline benchmark {target} for {baseline_tag}")
        commands.append(command)
    return tuple(commands)


def _run_latest_for_suite(*, worktree: Path, suite: str, env: dict[str, str] | None) -> tuple[tuple[str, ...], ...]:
    """Run current benchmarks for a suite."""
    if suite == "release-signal" and (worktree / "justfile").exists():
        _progress("running current release-signal benchmarks")
        _run_tool(
            "just",
            ["bench-latest"],
            cwd=worktree,
            options=ToolRunOptions(
                timeout=RELEASE_BENCH_TIMEOUT_SECONDS,
                env=env,
                stream_output=True,
            ),
        )
        _progress("completed current release-signal benchmarks")
        return (("just", "bench-latest"),)
    commands: list[tuple[str, ...]] = []
    for target in _bench_targets_for_suite(worktree, suite):
        command = ("cargo", "bench", "--profile", BENCHMARK_BUILD_FLAVOR, "--bench", target)
        _progress(f"running current benchmark {target}")
        _run_tool(
            command[0],
            list(command[1:]),
            cwd=worktree,
            options=ToolRunOptions(
                timeout=RELEASE_BENCH_TIMEOUT_SECONDS,
                env=env,
                stream_output=True,
            ),
        )
        _progress(f"completed current benchmark {target}")
        commands.append(command)
    return tuple(commands)


def _generate_local_baseline_into_worktree(
    *,
    config: ReleaseReportConfig,
    target_worktree: Path,
    tmp_dir: Path,
) -> RevisionEvidence:
    """Generate a local baseline and return its complete revision evidence."""
    baseline_worktree = tmp_dir / "baseline-worktree"
    _progress(f"preparing baseline worktree for {config.baseline_tag}")
    _run_git(["worktree", "add", "--detach", str(baseline_worktree), config.baseline_tag], cwd=config.repo_root)
    try:
        observed_baseline_tag = _current_package_tag(baseline_worktree)
        if observed_baseline_tag != config.baseline_tag:
            msg = f"prepared baseline checkout version {observed_baseline_tag} does not match requested release {config.baseline_tag}"
            raise ValueError(msg)
        comparison_targets = _shared_bench_targets(target_worktree, baseline_worktree, config.suite)
        commands = _run_saved_baseline_for_suite(
            worktree=baseline_worktree,
            baseline_tag=config.baseline_tag,
            suite=config.suite,
            env=_benchmark_env(baseline_worktree),
        )
        baseline_criterion = baseline_worktree / "target" / "criterion"
        if not baseline_criterion.is_dir():
            msg = f"generated baseline Criterion results were not found: {baseline_criterion}"
            raise FileNotFoundError(msg)
        target_criterion = target_worktree / "target" / "criterion"
        target_criterion.parent.mkdir(parents=True, exist_ok=True)
        copied = _copy_criterion_sample(
            source_criterion=baseline_criterion,
            target_criterion=target_criterion,
            source_sample=config.baseline_tag,
            target_sample=config.baseline_tag,
        )
        if copied == 0:
            msg = f"generated baseline contains no saved sample named {config.baseline_tag!r}"
            raise FileNotFoundError(msg)
        return _revision_evidence(
            baseline_worktree,
            version=config.baseline_tag,
            ref=config.baseline_tag,
            measurement=RevisionMeasurement(
                suite=config.suite,
                commands=commands,
                comparison_targets=comparison_targets,
            ),
        )
    finally:
        try:
            _run_git(["worktree", "remove", "--force", str(baseline_worktree)], cwd=config.repo_root)
        except RuntimeError as exc:
            print(f"benchmark-utils: failed to remove baseline worktree: {exc}", file=sys.stderr)


def _safe_extract_tar(archive: Path, target_dir: Path) -> None:
    """Extract a tar.gz archive without allowing path traversal."""
    target_dir.mkdir(parents=True, exist_ok=True)
    target_root = target_dir.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = (target_dir / member.name).resolve()
            if not member_path.is_relative_to(target_root):
                msg = f"refusing to extract unsafe archive member {member.name!r}"
                raise ValueError(msg)
        tar.extractall(target_dir, filter="data")


def _source_state_payload(source: SourceState) -> dict[str, object]:
    """Return the release-archive JSON shape for source evidence."""
    return {
        "version": source.version,
        "commit": source.commit,
        "ref": source.ref,
        "revision_timestamp": source.revision_timestamp,
        "git_clean": source.git_clean,
        "source_state_sha256": source.source_state_sha256,
        "limitation": source.limitation,
    }


def _toolchain_state_payload(toolchain: ToolchainState) -> dict[str, str | None]:
    """Return the release-archive JSON shape for toolchain evidence."""
    return {
        "rustc": toolchain.rustc,
        "criterion_version": toolchain.criterion_version,
        "cargo_profile": toolchain.cargo_profile,
        "cargo_lock_sha256": toolchain.cargo_lock_sha256,
        "harness_sha256": toolchain.harness_sha256,
        "configuration_sha256": toolchain.configuration_sha256,
        "measurement_plan_sha256": toolchain.measurement_plan_sha256,
        "limitation": toolchain.limitation,
    }


def _host_identity_payload(host: HostIdentity) -> dict[str, str]:
    """Return the release-archive JSON shape for host evidence."""
    return {
        "status": host.status,
        "cpu": host.cpu,
        "operating_system": host.operating_system,
        "architecture": host.architecture,
        "reason": host.reason,
    }


def write_release_benchmark_metadata(*, repo_root: Path, tag: str, criterion_dir: Path, output: Path) -> None:
    """Write versioned measurement provenance inside a release benchmark archive."""
    resolved_criterion = criterion_dir.resolve(strict=False)
    resolved_output = output.resolve(strict=False)
    if resolved_output == resolved_criterion or resolved_output.is_relative_to(resolved_criterion):
        msg = "release metadata output must be outside the Criterion directory it binds"
        raise ValueError(msg)
    normalized_tag = normalize_release_tag(tag)
    observed_tag = _current_package_tag(repo_root)
    if normalized_tag != observed_tag:
        msg = f"release tag {normalized_tag} does not match package version {observed_tag}"
        raise ValueError(msg)
    source = _source_state(repo_root, version=observed_tag, ref=normalized_tag)
    expected_commit = _expected_tag_commit(repo_root, normalized_tag)
    if source.commit != expected_commit:
        msg = f"release benchmark checkout commit {source.commit} does not match tag {normalized_tag} commit {expected_commit}"
        raise ValueError(msg)
    if source.git_clean is not True:
        msg = f"release benchmark checkout for {normalized_tag} must be clean before metadata is written"
        raise ValueError(msg)
    clean_source_digest = hashlib.sha256(f"commit {expected_commit}\n".encode()).hexdigest()
    if source.source_state_sha256 != clean_source_digest:
        msg = f"release benchmark checkout source-state digest is inconsistent with clean tag {normalized_tag}"
        raise ValueError(msg)
    toolchain = _toolchain_state(repo_root, "release-signal")
    host = _recorded_host_identity(repo_root)
    _criterion_sample_digest(criterion_dir, "new")
    payload = {
        "schema_version": RELEASE_ASSET_METADATA_SCHEMA_VERSION,
        "source": _source_state_payload(source),
        "measurement_commands": [list(command) for command in RELEASE_ASSET_MEASUREMENT_COMMANDS],
        "completed_targets": list(RELEASE_SIGNAL_BENCH_TARGETS),
        "toolchain": _toolchain_state_payload(toolchain),
        "measurement_host": _host_identity_payload(host),
        "criterion": {"content_sha256": _directory_digest(criterion_dir), "sample_name": "new"},
    }
    _write_text_atomic(output, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _metadata_object(data: Mapping[str, object], field: str, *, source: Path) -> Mapping[str, object]:
    """Return one required object from release metadata."""
    value = data.get(field)
    if not isinstance(value, Mapping):
        msg = f"{source}: {field} must be an object"
        raise TypeError(msg)
    return cast("Mapping[str, object]", value)


def _metadata_string(data: Mapping[str, object], field: str, *, source: Path) -> str:
    """Return one required non-empty string from release metadata."""
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        msg = f"{source}: {field} must be a non-empty string"
        raise ValueError(msg)
    return value


def _require_metadata_keys(data: Mapping[str, object], expected: set[str], *, source: Path) -> None:
    """Reject missing or unknown release metadata fields."""
    if set(data) != expected:
        msg = f"{source}: fields do not match release benchmark metadata schema"
        raise ValueError(msg)


def _release_metadata_commands(value: object, *, source: Path) -> tuple[tuple[str, ...], ...]:
    """Parse measurement command argument vectors from release metadata."""
    if not isinstance(value, list) or not value:
        msg = f"{source}: measurement_commands must be a non-empty array"
        raise ValueError(msg)
    commands: list[tuple[str, ...]] = []
    for command in value:
        if not isinstance(command, list) or not command or not all(isinstance(part, str) and part for part in command):
            msg = f"{source}: measurement_commands must contain non-empty string arrays"
            raise ValueError(msg)
        commands.append(tuple(cast("list[str]", command)))
    return tuple(commands)


def _expected_tag_commit(repo_root: Path, tag: str) -> str:
    """Resolve one requested release tag to its peeled commit object."""
    normalized_tag = normalize_release_tag(tag)
    commit = run_git_command(
        ["rev-parse", f"{normalized_tag}^{{commit}}"],
        cwd=repo_root,
        timeout=RELEASE_COMMAND_TIMEOUT_SECONDS,
    ).stdout.strip()
    if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit) is None:
        msg = f"could not resolve release tag {normalized_tag} to a full commit ID"
        raise ValueError(msg)
    return commit


def _legacy_release_asset_evidence(
    *,
    data: Mapping[str, object],
    request: ReleaseAssetLoadRequest,
) -> ReleaseAssetEvidence:
    """Load an existing unversioned archive without inventing missing facts."""
    metadata_path = request.extracted_root / "metadata.json"
    criterion_dir = request.extracted_root / "criterion"
    _require_metadata_keys(
        data,
        {
            "tag",
            "commit",
            "run_id",
            "generated_at",
            "cargo_profile",
            "sampling_mode",
            "runner_os",
            "runner_arch",
            "summary",
            "criterion_dir",
        },
        source=metadata_path,
    )
    normalized_tag = normalize_release_tag(request.requested_tag)
    metadata_tag = normalize_release_tag(_metadata_string(data, "tag", source=metadata_path))
    if metadata_tag != normalized_tag:
        msg = f"release benchmark metadata identifies {metadata_tag}, expected {normalized_tag}"
        raise ValueError(msg)
    commit = _metadata_string(data, "commit", source=metadata_path)
    if commit != request.expected_commit:
        msg = f"release benchmark metadata commit {commit!r} does not match tag {normalized_tag} commit {request.expected_commit}"
        raise ValueError(msg)
    if _metadata_string(data, "cargo_profile", source=metadata_path) != BENCHMARK_BUILD_FLAVOR:
        msg = f"{metadata_path}: legacy release benchmark cargo_profile must be {BENCHMARK_BUILD_FLAVOR!r}"
        raise ValueError(msg)
    if _metadata_string(data, "sampling_mode", source=metadata_path) != "full":
        msg = f"{metadata_path}: legacy release benchmark sampling_mode must be 'full'"
        raise ValueError(msg)
    if _metadata_string(data, "summary", source=metadata_path) != "PERFORMANCE_RESULTS.md":
        msg = f"{metadata_path}: legacy release benchmark summary path is unsupported"
        raise ValueError(msg)
    if _metadata_string(data, "criterion_dir", source=metadata_path) != "criterion":
        msg = f"{metadata_path}: legacy release benchmark Criterion path is unsupported"
        raise ValueError(msg)
    run_id = _metadata_string(data, "run_id", source=metadata_path)
    runner_os = _metadata_string(data, "runner_os", source=metadata_path)
    runner_arch = _metadata_string(data, "runner_arch", source=metadata_path)
    generated_at = _metadata_string(data, "generated_at", source=metadata_path)
    observed_content_sha256 = _directory_digest(criterion_dir)
    _criterion_sample_digest(criterion_dir, "new")
    return ReleaseAssetEvidence(
        revision=RevisionEvidence(
            source=SourceState(
                version=normalized_tag,
                commit=commit,
                ref=normalized_tag,
                revision_timestamp=generated_at,
                git_clean=None,
                source_state_sha256=None,
                limitation="Legacy archive did not record clean source state or its digest.",
            ),
            toolchain=ToolchainState(
                rustc=None,
                criterion_version=None,
                cargo_profile=BENCHMARK_BUILD_FLAVOR,
                cargo_lock_sha256=None,
                harness_sha256=None,
                configuration_sha256=None,
                measurement_plan_sha256=None,
                limitation=("Legacy archive did not record Rust, Criterion, lock, harness, configuration, or measurement-plan identity."),
            ),
            commands=(),
            completed_targets=(),
        ),
        measurement_host=HostIdentity(
            status="unavailable",
            cpu="",
            operating_system=runner_os,
            architecture=runner_arch,
            reason=f"Legacy archive run {run_id} did not record CPU identity or a controlled paired measurement host.",
        ),
        artifact=MeasurementArtifact(
            origin="release-archive",
            content_sha256=observed_content_sha256,
            sample_name="new",
            archive_sha256=_sha256_file(request.archive),
        ),
        acquisition_commands=(request.acquisition_command,),
    )


def _versioned_release_source(
    data: Mapping[str, object],
    *,
    metadata_path: Path,
    normalized_tag: str,
    expected_commit: str,
) -> SourceState:
    """Parse and bind complete source evidence to the requested clean tag."""
    source_data = _metadata_object(data, "source", source=metadata_path)
    _require_metadata_keys(
        source_data,
        {"version", "commit", "ref", "revision_timestamp", "git_clean", "source_state_sha256", "limitation"},
        source=metadata_path,
    )
    if source_data.get("limitation") != "":
        msg = f"{metadata_path}: versioned release source evidence must be complete"
        raise ValueError(msg)
    git_clean = source_data.get("git_clean")
    if not isinstance(git_clean, bool):
        msg = f"{metadata_path}: source.git_clean must be a boolean"
        raise TypeError(msg)
    source = SourceState(
        version=_metadata_string(source_data, "version", source=metadata_path),
        commit=_metadata_string(source_data, "commit", source=metadata_path),
        ref=_metadata_string(source_data, "ref", source=metadata_path),
        revision_timestamp=_metadata_string(source_data, "revision_timestamp", source=metadata_path),
        git_clean=git_clean,
        source_state_sha256=_metadata_string(source_data, "source_state_sha256", source=metadata_path),
    )
    if source.version != normalized_tag or normalize_release_tag(source.ref) != normalized_tag:
        msg = f"release benchmark metadata identifies {source.version}/{source.ref}, expected {normalized_tag}"
        raise ValueError(msg)
    if source.commit != expected_commit:
        msg = f"release benchmark metadata commit {source.commit} does not match tag {normalized_tag} commit {expected_commit}"
        raise ValueError(msg)
    if not source.git_clean:
        msg = f"release benchmark metadata for {normalized_tag} does not identify a clean checkout"
        raise ValueError(msg)
    clean_source_digest = hashlib.sha256(f"commit {expected_commit}\n".encode()).hexdigest()
    if source.source_state_sha256 != clean_source_digest:
        msg = f"release benchmark source-state digest is inconsistent with clean tag {normalized_tag}"
        raise ValueError(msg)
    return source


def _versioned_release_toolchain(data: Mapping[str, object], *, metadata_path: Path) -> ToolchainState:
    """Parse complete versioned release toolchain evidence."""
    toolchain_data = _metadata_object(data, "toolchain", source=metadata_path)
    _require_metadata_keys(
        toolchain_data,
        {
            "rustc",
            "criterion_version",
            "cargo_profile",
            "cargo_lock_sha256",
            "harness_sha256",
            "configuration_sha256",
            "measurement_plan_sha256",
            "limitation",
        },
        source=metadata_path,
    )
    if toolchain_data.get("limitation") != "":
        msg = f"{metadata_path}: versioned release toolchain evidence must be complete"
        raise ValueError(msg)
    return ToolchainState(
        rustc=_metadata_string(toolchain_data, "rustc", source=metadata_path),
        criterion_version=_metadata_string(toolchain_data, "criterion_version", source=metadata_path),
        cargo_profile=_metadata_string(toolchain_data, "cargo_profile", source=metadata_path),
        cargo_lock_sha256=_metadata_string(toolchain_data, "cargo_lock_sha256", source=metadata_path),
        harness_sha256=_metadata_string(toolchain_data, "harness_sha256", source=metadata_path),
        configuration_sha256=_metadata_string(toolchain_data, "configuration_sha256", source=metadata_path),
        measurement_plan_sha256=_metadata_string(toolchain_data, "measurement_plan_sha256", source=metadata_path),
    )


def _versioned_release_commands_and_targets(data: Mapping[str, object], *, metadata_path: Path) -> tuple[tuple[tuple[str, ...], ...], tuple[str, ...]]:
    """Validate the exact versioned release producer command and target contract."""
    commands = _release_metadata_commands(data.get("measurement_commands"), source=metadata_path)
    if commands != RELEASE_ASSET_MEASUREMENT_COMMANDS:
        msg = f"{metadata_path}: release benchmark measurement commands do not match the producer contract"
        raise ValueError(msg)
    completed_targets = data.get("completed_targets")
    if not isinstance(completed_targets, list) or not all(isinstance(target, str) for target in completed_targets):
        msg = f"{metadata_path}: completed_targets must be a string array"
        raise TypeError(msg)
    parsed_targets = tuple(cast("list[str]", completed_targets))
    if parsed_targets != RELEASE_SIGNAL_BENCH_TARGETS:
        msg = f"{metadata_path}: completed_targets do not match the release-signal producer contract"
        raise ValueError(msg)
    return commands, parsed_targets


def _versioned_release_host(data: Mapping[str, object], *, metadata_path: Path) -> HostIdentity:
    """Parse a non-placeholder recorded host from versioned metadata."""
    host_data = _metadata_object(data, "measurement_host", source=metadata_path)
    _require_metadata_keys(host_data, {"status", "cpu", "operating_system", "architecture", "reason"}, source=metadata_path)
    for field in ("cpu", "operating_system", "architecture", "reason"):
        if not isinstance(host_data.get(field), str):
            msg = f"{metadata_path}: measurement_host.{field} must be a string"
            raise TypeError(msg)
    status = _metadata_string(host_data, "status", source=metadata_path)
    if status != "recorded":
        msg = f"{metadata_path}: versioned release metadata requires a recorded measurement host"
        raise ValueError(msg)
    return HostIdentity(
        status="recorded",
        cpu=cast("str", host_data["cpu"]),
        operating_system=cast("str", host_data["operating_system"]),
        architecture=cast("str", host_data["architecture"]),
        reason=cast("str", host_data["reason"]),
    )


def _load_release_asset_evidence(
    *,
    requested_tag: str,
    expected_commit: str,
    extracted_root: Path,
    archive: Path,
    acquisition_command: tuple[str, ...],
) -> ReleaseAssetEvidence:
    """Load and verify measurement provenance from one extracted release archive."""
    metadata_path = extracted_root / "metadata.json"
    try:
        raw = json.loads(_read_text(metadata_path))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"release benchmark asset lacks valid versioned metadata: {metadata_path}: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(raw, Mapping):
        msg = f"release benchmark metadata must be an object: {metadata_path}"
        raise TypeError(msg)
    data = cast("Mapping[str, object]", raw)
    request = ReleaseAssetLoadRequest(
        requested_tag=requested_tag,
        expected_commit=expected_commit,
        extracted_root=extracted_root,
        archive=archive,
        acquisition_command=acquisition_command,
    )
    criterion_dir = extracted_root / "criterion"
    if "schema_version" not in data:
        return _legacy_release_asset_evidence(data=data, request=request)
    _require_metadata_keys(
        data,
        {
            "schema_version",
            "source",
            "measurement_commands",
            "completed_targets",
            "toolchain",
            "measurement_host",
            "criterion",
        },
        source=metadata_path,
    )
    schema_version = data.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != RELEASE_ASSET_METADATA_SCHEMA_VERSION:
        msg = f"unsupported release benchmark metadata schema: {schema_version!r}"
        raise ValueError(msg)

    normalized_tag = normalize_release_tag(requested_tag)
    source = _versioned_release_source(
        data,
        metadata_path=metadata_path,
        normalized_tag=normalized_tag,
        expected_commit=expected_commit,
    )
    toolchain = _versioned_release_toolchain(data, metadata_path=metadata_path)
    commands, parsed_targets = _versioned_release_commands_and_targets(data, metadata_path=metadata_path)
    host = _versioned_release_host(data, metadata_path=metadata_path)
    criterion_data = _metadata_object(data, "criterion", source=metadata_path)
    _require_metadata_keys(criterion_data, {"content_sha256", "sample_name"}, source=metadata_path)
    expected_content_sha256 = _metadata_string(criterion_data, "content_sha256", source=metadata_path)
    observed_content_sha256 = _directory_digest(criterion_dir)
    if observed_content_sha256 != expected_content_sha256:
        msg = f"release benchmark Criterion digest mismatch for {normalized_tag}"
        raise ValueError(msg)
    sample_name = _metadata_string(criterion_data, "sample_name", source=metadata_path)
    _criterion_sample_digest(criterion_dir, sample_name)
    return ReleaseAssetEvidence(
        revision=RevisionEvidence(
            source=source,
            toolchain=toolchain,
            commands=commands,
            completed_targets=parsed_targets,
        ),
        measurement_host=host,
        artifact=MeasurementArtifact(
            origin="release-archive",
            content_sha256=observed_content_sha256,
            sample_name=sample_name,
            archive_sha256=_sha256_file(archive),
        ),
        acquisition_commands=(acquisition_command,),
    )


def _download_release_baseline(*, tag: str, download_dir: Path, repo_root: Path) -> DownloadedReleaseAsset:
    """Download a Delaunay release benchmark asset."""
    artifact = download_dir / f"delaunay-{tag}-criterion-baseline.tar.gz"
    command = ("gh", "release", "download", tag, "--pattern", artifact.name, "--dir", str(download_dir))
    _run_tool(command[0], list(command[1:]), cwd=repo_root)
    if not artifact.exists():
        msg = f"release baseline asset was not downloaded: {artifact}"
        raise FileNotFoundError(msg)
    return DownloadedReleaseAsset(archive=artifact, command=command)


def _copy_criterion_sample(*, source_criterion: Path, target_criterion: Path, source_sample: str, target_sample: str) -> int:
    """Copy one Criterion sample name into a target Criterion tree."""
    copied = 0
    for estimates_json in sorted(source_criterion.rglob("estimates.json")):
        if estimates_json.parent.name != source_sample:
            continue
        criterion_sample = _criterion_sample(estimates_json, source_criterion)
        if criterion_sample is None:
            continue
        source_dir = estimates_json.parent
        relative_benchmark_dir = source_dir.parent.relative_to(source_criterion)
        target_dir = target_criterion / relative_benchmark_dir / target_sample
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)
        copied += 1
    return copied


def _copy_first_available_sample(*, source_criterion: Path, target_criterion: Path, candidate_samples: tuple[str, ...], target_sample: str) -> None:
    """Copy the first Criterion sample name that exists in an extracted asset."""
    for sample in candidate_samples:
        if _copy_criterion_sample(source_criterion=source_criterion, target_criterion=target_criterion, source_sample=sample, target_sample=target_sample):
            return
    msg = f"could not find Criterion sample {candidate_samples!r} under {source_criterion}"
    raise FileNotFoundError(msg)


def _prepare_github_release_assets(
    *,
    config: ReleaseReportConfig,
    target_worktree: Path,
    tmp_dir: Path,
) -> tuple[ReleaseAssetEvidence, ReleaseAssetEvidence]:
    """Prepare release-asset samples and return current/baseline measurement evidence."""
    baseline_download = _download_release_baseline(tag=config.baseline_tag, download_dir=tmp_dir, repo_root=config.repo_root)
    current_download = _download_release_baseline(tag=config.current_tag, download_dir=tmp_dir, repo_root=config.repo_root)
    baseline_extract = tmp_dir / "baseline-asset"
    current_extract = tmp_dir / "current-asset"
    _safe_extract_tar(baseline_download.archive, baseline_extract)
    _safe_extract_tar(current_download.archive, current_extract)
    baseline_commit = _expected_tag_commit(config.repo_root, config.baseline_tag)
    current_commit = _expected_tag_commit(config.repo_root, config.current_tag)

    baseline_evidence = _load_release_asset_evidence(
        requested_tag=config.baseline_tag,
        expected_commit=baseline_commit,
        extracted_root=baseline_extract,
        archive=baseline_download.archive,
        acquisition_command=baseline_download.command,
    )
    current_evidence = _load_release_asset_evidence(
        requested_tag=config.current_tag,
        expected_commit=current_commit,
        extracted_root=current_extract,
        archive=current_download.archive,
        acquisition_command=current_download.command,
    )

    baseline_criterion = baseline_extract / "criterion"
    current_criterion = current_extract / "criterion"
    if not baseline_criterion.is_dir() or not current_criterion.is_dir():
        msg = "release benchmark asset does not contain criterion/ data"
        raise FileNotFoundError(msg)

    target_criterion = target_worktree / "target" / "criterion"
    target_criterion.mkdir(parents=True, exist_ok=True)
    _copy_first_available_sample(
        source_criterion=current_criterion,
        target_criterion=target_criterion,
        candidate_samples=(current_evidence.artifact.sample_name,),
        target_sample="new",
    )
    _copy_first_available_sample(
        source_criterion=baseline_criterion,
        target_criterion=target_criterion,
        candidate_samples=(baseline_evidence.artifact.sample_name,),
        target_sample=config.baseline_tag,
    )
    return current_evidence, baseline_evidence


def _build_performance_bundle_in_temp_worktree(*, config: ReleaseReportConfig) -> PerformanceBundle:
    """Measure or load a comparison in temporary worktrees and return trusted data."""
    with tempfile.TemporaryDirectory(prefix="delaunay-performance-") as tmp:
        tmp_dir = Path(tmp)
        worktree = tmp_dir / "worktree"

        _progress(f"preparing current worktree for {config.current_tag}")
        _run_git(["worktree", "add", "--detach", str(worktree), config.worktree_ref], cwd=config.repo_root)
        try:
            if config.apply_current_diff:
                _apply_current_diff_to_worktree(repo_root=config.repo_root, worktree=worktree)
            observed_current_tag = _current_package_tag(worktree)
            if observed_current_tag != config.current_tag:
                msg = f"prepared current checkout version {observed_current_tag} does not match requested release {config.current_tag}"
                raise ValueError(msg)
            if config.baseline_source == "github-assets":
                current_asset_evidence, baseline_asset_evidence = _prepare_github_release_assets(
                    config=config,
                    target_worktree=worktree,
                    tmp_dir=tmp_dir,
                )
                current_evidence = current_asset_evidence.revision
                baseline_evidence = baseline_asset_evidence.revision
                current_host = current_asset_evidence.measurement_host
                baseline_host = baseline_asset_evidence.measurement_host
                current_artifact = current_asset_evidence.artifact
                baseline_artifact = baseline_asset_evidence.artifact
                current_acquisition_commands = current_asset_evidence.acquisition_commands
                baseline_acquisition_commands = baseline_asset_evidence.acquisition_commands
            else:
                baseline_evidence = _generate_local_baseline_into_worktree(config=config, target_worktree=worktree, tmp_dir=tmp_dir)
                current_commands = _run_latest_for_suite(worktree=worktree, suite=config.suite, env=_benchmark_env(worktree))
                current_targets = _bench_targets_for_suite(worktree, config.suite)
                baseline_targets = set(baseline_evidence.completed_targets)
                comparison_targets = tuple(target for target in current_targets if target in baseline_targets)
                current_evidence = _revision_evidence(
                    worktree,
                    version=config.current_tag,
                    ref=config.worktree_ref,
                    measurement=RevisionMeasurement(
                        suite=config.suite,
                        commands=current_commands,
                        comparison_targets=comparison_targets,
                    ),
                )
                current_host = _recorded_host_identity(config.repo_root)
                baseline_host = current_host
                criterion_dir = worktree / "target" / "criterion"
                current_artifact = MeasurementArtifact(
                    origin="local-run",
                    content_sha256=_criterion_sample_digest(criterion_dir, "new"),
                    sample_name="new",
                )
                baseline_artifact = MeasurementArtifact(
                    origin="local-run",
                    content_sha256=_criterion_sample_digest(criterion_dir, config.baseline_tag),
                    sample_name=config.baseline_tag,
                )
                current_acquisition_commands = ()
                baseline_acquisition_commands = ()
            _progress("collecting and validating performance artifacts")
            context = ArtifactContext(
                release=ReleasePair(current=config.current_tag, baseline=config.baseline_tag),
                statistic="median",
                suite=config.suite,
                scope=config.scope,
                measurement_mode="github-assets" if config.baseline_source == "github-assets" else "local-worktrees",
                current_source=current_evidence.source,
                baseline_source=baseline_evidence.source,
                current_commands=current_evidence.commands,
                baseline_commands=baseline_evidence.commands,
                current_completed_targets=current_evidence.completed_targets,
                baseline_completed_targets=baseline_evidence.completed_targets,
                current_acquisition_commands=current_acquisition_commands,
                baseline_acquisition_commands=baseline_acquisition_commands,
                current_toolchain=current_evidence.toolchain,
                baseline_toolchain=baseline_evidence.toolchain,
                current_measurement_host=current_host,
                baseline_measurement_host=baseline_host,
                current_artifact=current_artifact,
                baseline_artifact=baseline_artifact,
                publication_host=_recorded_host_identity(config.repo_root),
            )
            comparison_note = "; ".join(context.comparison_blockers)
            rows = collect_performance_rows(
                worktree / "target" / "criterion",
                config.baseline_tag,
                suite=config.suite,
                scope=config.scope,
                comparison_note=comparison_note,
            )
            return PerformanceBundle(context=context, rows=rows)
        finally:
            try:
                _run_git(["worktree", "remove", "--force", str(worktree)], cwd=config.repo_root)
            except RuntimeError as exc:
                print(f"benchmark-utils: failed to remove temporary worktree: {exc}", file=sys.stderr)


def _artifact_paths_for_output(output: Path) -> ArtifactPaths:
    """Return the adjacent canonical artifact paths for one Markdown output."""
    return ArtifactPaths(csv=output.with_suffix(".csv"), provenance=output.with_suffix(".provenance.json"))


def _preflight_performance_destinations(
    *,
    output: Path,
    report_id: PerformanceReportId,
    current: Path | None = None,
    archive_dir: Path | None = None,
    project_root: Path | None = None,
) -> None:
    """Reject deterministic output aliases before fetches or measurements."""
    artifacts = _artifact_paths_for_output(output)
    paths = {"Markdown output": output, "artifact CSV": artifacts.csv, "artifact provenance": artifacts.provenance}
    if current is not None:
        if archive_dir is None:
            msg = "archive_dir is required when preflighting a promotion"
            raise ValueError(msg)
        if project_root is None:
            msg = "project_root is required when preflighting a promotion"
            raise ValueError(msg)
        paths.update(
            {
                "current documentation": current,
                "archive index": archive_dir / "README.md",
                "promoted report archive": archive_dir / report_id.archive_name,
            }
        )
        durable = _durable_performance_artifact_paths(archive_dir, report_id)
        paths["durable CSV"] = durable.csv
        paths["durable provenance"] = durable.provenance
        tracked_destinations = {
            current,
            archive_dir / "README.md",
            archive_dir / report_id.archive_name,
            durable.csv,
            durable.provenance,
        }
        for label, path in paths.items():
            if path in tracked_destinations:
                _repository_relative_path(project_root, path, label=label)
        if current.exists():
            current_id = parse_performance_report_id(_normalize_how_to_update(_read_text(current)))
            if current_id != report_id:
                prior_archive = archive_dir / current_id.archive_name
                _repository_relative_path(project_root, prior_archive, label="prior report archive")
                paths["prior report archive"] = prior_archive
    ensure_distinct_paths(paths)


def _publish_performance_bundle(
    *,
    bundle: PerformanceBundle,
    output: Path,
    current: Path | None = None,
    archive_dir: Path | None = None,
    project_root: Path | None = None,
) -> PerformanceReportId:
    """Publish artifacts, reload-render Markdown, and optionally promote docs."""
    artifacts = _artifact_paths_for_output(output)
    paths = {"Markdown output": output, "artifact CSV": artifacts.csv, "artifact provenance": artifacts.provenance}
    if current is not None:
        paths["current documentation"] = current
    ensure_distinct_paths(paths)
    prior_output = output.read_bytes() if output.exists() else None
    try:
        with publish_bundle(artifacts, bundle):
            report_id = PerformanceReportId(
                current_tag=bundle.context.release.current,
                baseline_tag=bundle.context.release.baseline,
            )
            if current is None:
                rendered = render_performance_artifacts(artifacts)
            else:
                if archive_dir is None:
                    msg = "archive_dir is required when promoting performance documentation"
                    raise ValueError(msg)
                if project_root is None:
                    msg = "project_root is required when promoting performance documentation"
                    raise ValueError(msg)
                durable_artifacts = _durable_performance_artifact_paths(archive_dir, report_id)
                rendered = render_performance_bundle(
                    load_bundle(artifacts),
                    evidence_paths=_promoted_evidence_paths(durable_artifacts, project_root=project_root),
                    evidence_state="promoted",
                )
            _write_text_atomic(output, rendered)
            if current is not None:
                if archive_dir is None:
                    msg = "archive_dir is required when promoting performance documentation"
                    raise ValueError(msg)
                if project_root is None:
                    msg = "project_root is required when promoting performance documentation"
                    raise ValueError(msg)
                report_id = promote_performance_report(
                    source=output,
                    artifacts=artifacts,
                    destinations=PerformancePromotionDestinations(
                        project_root=project_root,
                        current=current,
                        archive_dir=archive_dir,
                    ),
                    expected=report_id,
                )
            return report_id
    except BaseException:
        _restore_file_snapshot(output, prior_output)
        raise


def generate_performance_worktree_report(*, output: Path, config: ReleaseReportConfig) -> PerformanceReportId:
    """Generate and retain a validated non-promoting comparison bundle."""
    current_tag = normalize_release_tag(config.current_tag)
    baseline_tag = normalize_release_tag(config.baseline_tag)
    normalized = ReleaseReportConfig(
        repo_root=config.repo_root,
        current_tag=current_tag,
        baseline_tag=baseline_tag,
        worktree_ref=config.worktree_ref,
        suite=config.suite,
        scope=config.scope,
        stat=config.stat,
        apply_current_diff=config.apply_current_diff,
        baseline_source=config.baseline_source,
    )
    _preflight_performance_destinations(
        output=output,
        report_id=PerformanceReportId(current_tag=current_tag, baseline_tag=baseline_tag),
    )
    bundle = _build_performance_bundle_in_temp_worktree(config=normalized)
    return _publish_performance_bundle(bundle=bundle, output=output)


def generate_and_promote_performance_report(
    *,
    output: Path,
    current: Path,
    archive_dir: Path,
    config: ReleaseReportConfig,
) -> PerformanceReportId:
    """Generate, retain, reload-render, and promote one comparison bundle."""
    current_tag = normalize_release_tag(config.current_tag)
    baseline_tag = normalize_release_tag(config.baseline_tag)
    if current_tag == baseline_tag:
        msg = "performance-release requires distinct current and baseline tags"
        raise ValueError(msg)
    _preflight_performance_destinations(
        output=output,
        report_id=PerformanceReportId(current_tag=current_tag, baseline_tag=baseline_tag),
        current=current,
        archive_dir=archive_dir,
        project_root=config.repo_root,
    )
    bundle = _build_performance_bundle_in_temp_worktree(
        config=ReleaseReportConfig(
            repo_root=config.repo_root,
            current_tag=current_tag,
            baseline_tag=baseline_tag,
            worktree_ref=config.worktree_ref,
            suite=config.suite,
            scope=config.scope,
            stat=config.stat,
            apply_current_diff=config.apply_current_diff,
            baseline_source=config.baseline_source,
        )
    )
    return _publish_performance_bundle(
        bundle=bundle,
        output=output,
        current=current,
        archive_dir=archive_dir,
        project_root=config.repo_root,
    )


def render_and_promote_performance_artifacts(
    *,
    output: Path,
    artifacts: ArtifactPaths,
    destinations: PerformancePromotionDestinations,
    expected_current_tag: str,
) -> PerformanceReportId:
    """Render and promote retained artifacts without Cargo or worktrees."""
    current = destinations.current
    archive_dir = destinations.archive_dir
    project_root = destinations.project_root
    ensure_distinct_paths(
        {
            "Markdown output": output,
            "artifact CSV": artifacts.csv,
            "artifact provenance": artifacts.provenance,
            "current documentation": current,
        }
    )
    bundle = load_bundle(artifacts)
    normalized_expected_current = normalize_release_tag(expected_current_tag)
    if bundle.context.release.current != normalized_expected_current:
        msg = f"retained current release {bundle.context.release.current} does not match independently expected release {normalized_expected_current}"
        raise ValueError(msg)
    if bundle.context.release.current == bundle.context.release.baseline:
        msg = "performance-doc cannot promote a same-version local performance comparison"
        raise ValueError(msg)
    bundle.require_promotable()
    _preflight_performance_destinations(
        output=output,
        report_id=PerformanceReportId(
            current_tag=bundle.context.release.current,
            baseline_tag=bundle.context.release.baseline,
        ),
        current=current,
        archive_dir=archive_dir,
        project_root=project_root,
    )
    prior_output = output.read_bytes() if output.exists() else None
    try:
        report_id = PerformanceReportId(
            current_tag=bundle.context.release.current,
            baseline_tag=bundle.context.release.baseline,
        )
        durable_artifacts = _durable_performance_artifact_paths(archive_dir, report_id)
        _write_text_atomic(
            output,
            render_performance_bundle(
                bundle,
                evidence_paths=_promoted_evidence_paths(durable_artifacts, project_root=project_root),
                evidence_state="promoted",
            ),
        )
        return promote_performance_report(
            source=output,
            artifacts=artifacts,
            destinations=destinations,
            expected=report_id,
        )
    except BaseException:
        _restore_file_snapshot(output, prior_output)
        raise


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

    @staticmethod
    def _restore_artifact_pair(
        published_files: tuple[tuple[bool, Path], ...],
        backups: tuple[tuple[bool, Path, Path], ...],
    ) -> list[OSError]:
        """Restore the prior artifact pair and return any rollback failures."""
        rollback_errors: list[OSError] = []
        for published, live_file in published_files:
            if published:
                try:
                    live_file.unlink(missing_ok=True)
                except OSError as rollback_error:
                    rollback_errors.append(rollback_error)

        for backed_up, backup_file, live_file in backups:
            if backed_up:
                try:
                    backup_file.replace(live_file)
                except OSError as rollback_error:
                    rollback_errors.append(rollback_error)

        return rollback_errors

    @staticmethod
    def _publish_artifact_pair(
        staged_results: Path,
        staged_metadata: Path,
        output_file: Path,
        metadata_file: Path,
    ) -> None:
        """Publish a staged baseline/metadata pair, restoring the prior pair on failure."""
        backup_dir = Path(tempfile.mkdtemp(prefix=".delaunay-baseline-backup-", dir=output_file.parent))
        backup_results = backup_dir / output_file.name
        backup_metadata = backup_dir / metadata_file.name
        backed_up_results = False
        backed_up_metadata = False
        published_results = False
        published_metadata = False

        try:
            if output_file.exists():
                output_file.replace(backup_results)
                backed_up_results = True
            if metadata_file.exists():
                metadata_file.replace(backup_metadata)
                backed_up_metadata = True

            staged_results.replace(output_file)
            published_results = True
            staged_metadata.replace(metadata_file)
            published_metadata = True
        except OSError as publish_error:
            rollback_errors = LocalRefBaselineGenerator._restore_artifact_pair(
                (
                    (published_results, output_file),
                    (published_metadata, metadata_file),
                ),
                (
                    (backed_up_results, backup_results, output_file),
                    (backed_up_metadata, backup_metadata, metadata_file),
                ),
            )

            if rollback_errors:
                details = "; ".join(str(error) for error in rollback_errors)
                msg = f"Failed to publish baseline artifacts and restore the prior pair: {details}"
                raise RuntimeError(msg) from publish_error

            shutil.rmtree(backup_dir, ignore_errors=True)
            raise

        shutil.rmtree(backup_dir, ignore_errors=True)

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
        metadata_file = out_dir / "metadata.json"

        with tempfile.TemporaryDirectory(
            prefix=f".{out_dir.name}-staging-",
            dir=out_dir.parent,
        ) as staging_dir_name:
            staging_dir = Path(staging_dir_name)
            staged_output_file = staging_dir / output_file.name

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
                success = generator.generate_baseline(
                    dev_mode=dev_mode,
                    output_file=staged_output_file,
                    bench_timeout=bench_timeout,
                )

            if not success:
                msg = f"Failed to generate baseline for ref {ref_name}"
                raise RuntimeError(msg)

            metadata_success = WorkflowHelper.create_metadata(
                ref_name,
                staging_dir,
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

            self._publish_artifact_pair(
                staged_output_file,
                staging_dir / metadata_file.name,
                output_file,
                metadata_file,
            )

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
    except _LOCAL_RUSTC_VERSION_ERRORS:
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
        try:
            benchmarks = extract_benchmark_data(baseline_content)
        except ValueError as exc:
            raise BaselineParseError(str(exc)) from exc
        results: dict[str, BenchmarkData] = {}
        for benchmark in benchmarks:
            results[benchmark.comparison_key] = benchmark
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
        coverage = self._comparison_coverage(current_results, baseline_results)
        if not coverage.is_comparable:
            self._write_non_comparable_coverage(f, coverage)

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

        if not coverage.is_comparable:
            self._write_failed_aggregate_summary(
                f,
                reason="benchmark coverage differs or is empty",
                requirement="complete identical benchmark coverage is required",
            )
            return True

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

        self._write_failed_aggregate_summary(
            f,
            reason="no valid timing pairs",
            requirement="every covered benchmark requires a valid timing pair",
        )
        return True

    @staticmethod
    def _write_failed_aggregate_summary(
        f: TextIO,
        *,
        reason: str,
        requirement: str,
    ) -> None:
        """Render a non-comparable enforcing-policy result."""
        f.write("\n=== SUMMARY ===\n")
        f.write(f"Aggregate timing comparison: NOT COMPARABLE ({reason})\n")
        f.write(f"🚨 PERFORMANCE GUARD FAILED: {requirement}\n\n")

    @staticmethod
    def _comparison_coverage(
        current_results: list[BenchmarkData],
        baseline_results: Mapping[str, BenchmarkData],
    ) -> BenchmarkComparisonCoverage:
        """Build exact comparison-key evidence without silently collapsing duplicates."""
        current_keys = [benchmark.comparison_key for benchmark in current_results]
        counts: dict[str, int] = {}
        for key in current_keys:
            counts[key] = counts.get(key, 0) + 1
        duplicates = tuple(sorted(key for key, count in counts.items() if count > 1))
        invalid_current = tuple(sorted(benchmark.comparison_key for benchmark in current_results if not PerformanceComparator._has_valid_timing(benchmark)))
        invalid_baseline = tuple(sorted(key for key, benchmark in baseline_results.items() if not PerformanceComparator._has_valid_timing(benchmark)))
        return BenchmarkComparisonCoverage(
            current_keys=frozenset(current_keys),
            baseline_keys=frozenset(baseline_results),
            duplicate_current_keys=duplicates,
            invalid_current_timing_keys=invalid_current,
            invalid_baseline_timing_keys=invalid_baseline,
        )

    @staticmethod
    def _has_valid_timing(benchmark: BenchmarkData) -> bool:
        """Return whether a reporting record has one supported physical timing."""
        values = (benchmark.time_low, benchmark.time_mean, benchmark.time_high)
        return (
            benchmark.time_unit in TIME_UNIT_TO_MICROSECONDS
            and all(math.isfinite(value) and value > 0.0 for value in values)
            and benchmark.time_low <= benchmark.time_mean <= benchmark.time_high
        )

    @staticmethod
    def _write_non_comparable_coverage(f: TextIO, coverage: BenchmarkComparisonCoverage) -> None:
        """Render complete diagnostics for a non-comparable benchmark keyset."""
        f.write("=== COVERAGE ===\n")
        f.write("Comparison coverage: NON-COMPARABLE\n")
        if not coverage.current_keys and not coverage.baseline_keys:
            f.write("Reason: current and baseline benchmark keysets are empty\n")
        if coverage.duplicate_current_keys:
            f.write(f"Duplicate current benchmark keys: {', '.join(coverage.duplicate_current_keys)}\n")
        if coverage.invalid_current_timing_keys:
            f.write(f"Invalid current timings: {', '.join(coverage.invalid_current_timing_keys)}\n")
        if coverage.invalid_baseline_timing_keys:
            f.write(f"Invalid baseline timings: {', '.join(coverage.invalid_baseline_timing_keys)}\n")
        if coverage.missing_from_baseline:
            f.write(f"Missing from baseline: {', '.join(coverage.missing_from_baseline)}\n")
        if coverage.missing_from_current:
            f.write(f"Missing from current run: {', '.join(coverage.missing_from_current)}\n")
        f.write("\n")

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

            # Write metadata through a same-directory temporary so an interrupted
            # write cannot truncate a previously published metadata file.
            output_dir.mkdir(parents=True, exist_ok=True)
            metadata_file = output_dir / "metadata.json"
            tmp_metadata_file = output_dir / "metadata.json.tmp"
            tmp_metadata_file.unlink(missing_ok=True)

            with tmp_metadata_file.open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            tmp_metadata_file.replace(metadata_file)

            print(f"📦 Created metadata file: {metadata_file}", file=sys.stderr)
            return True

        except (OSError, TypeError, ValueError) as e:
            tmp_metadata_file = output_dir / "metadata.json.tmp"
            with suppress(OSError):
                tmp_metadata_file.unlink(missing_ok=True)
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
    except _BENCHMARK_TIMEOUT_PARSE_ERRORS:
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
        return cast("str", match.group("owner")), cast("str", match.group("repo"))

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


def _sorted_benchmark_list(results: Mapping[str, BenchmarkData]) -> list[BenchmarkData]:
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


def _add_bench_compare_subcommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the Criterion saved-baseline comparison subcommand."""
    bench_compare_parser = subparsers.add_parser("bench-compare", help="Compare Criterion new results against a saved baseline")
    bench_compare_parser.add_argument("baseline", nargs="?", default="last", help="Saved Criterion baseline name (default: last)")
    bench_compare_parser.add_argument("--stat", default="median", choices=["mean", "median"], help="Criterion statistic to compare (default: median)")
    bench_compare_parser.add_argument(
        "--suite",
        default="release-signal",
        choices=BENCH_COMPARE_SUITE_CHOICES,
        help="Benchmark suite to compare (default: release-signal)",
    )
    bench_compare_parser.add_argument(
        "--scope",
        default="release-signal",
        choices=("release-signal", "all-benches"),
        help="Comparison scope (default: release-signal)",
    )
    bench_compare_parser.add_argument("--criterion-dir", type=Path, default=Path("target") / "criterion", help="Criterion output directory")
    bench_compare_parser.add_argument("--output", type=Path, default=PERFORMANCE_REPORT_SOURCE, help="Output Markdown report path")
    _add_project_root_arg(bench_compare_parser)


def _add_benchmark_subcommands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add benchmark-running subcommands."""
    _add_bench_compare_subcommand(subparsers)

    release_signal_parser = subparsers.add_parser(
        "run-release-signal",
        help="Run the maintained release-signal benchmark measurement plan",
    )
    release_signal_parser.add_argument(
        "--profile",
        default=BENCHMARK_BUILD_FLAVOR,
        help=f"Cargo profile for every planned target (default: {BENCHMARK_BUILD_FLAVOR})",
    )
    release_signal_parser.add_argument(
        "--save-baseline",
        help="Also save every planned Criterion result under this baseline name",
    )
    _add_bench_timeout_arg(release_signal_parser)
    _add_project_root_arg(release_signal_parser)

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


def _add_local_baseline_subcommands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
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


def _add_workflow_helper_subcommands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add subcommands used by GitHub Actions workflows."""
    subparsers.add_parser("determine-ref", help="Determine git ref name for baseline generation")

    meta_parser = subparsers.add_parser("create-metadata", help="Create metadata.json file for baseline artifact")
    meta_parser.add_argument("--ref", dest="ref_name", type=str, help="Git ref name for this baseline")
    meta_parser.add_argument("--output-dir", type=Path, default=Path("baseline-artifact"), help="Output directory for metadata.json")

    summary_parser = subparsers.add_parser("display-summary", help="Display baseline file summary")
    summary_parser.add_argument("--baseline", type=Path, required=True, help="Path to baseline file")

    artifact_parser = subparsers.add_parser("sanitize-artifact-name", help="Sanitize git ref name for GitHub Actions artifact")
    artifact_parser.add_argument("--ref", dest="ref_name", type=str, help="Git ref name to sanitize")


def _add_regression_subcommands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
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


def _add_performance_summary_subcommands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add performance summary generation subcommands."""
    perf_summary_parser = subparsers.add_parser("generate-summary", help="Generate performance summary markdown")
    perf_summary_parser.add_argument("--output", type=Path, help="Output file path (defaults to benches/PERFORMANCE_RESULTS.md)")
    perf_summary_parser.add_argument(
        "--run-benchmarks",
        action="store_true",
        help="Run the maintained release-signal measurement plan before generating summary",
    )
    perf_summary_parser.add_argument(
        "--profile",
        default=BENCHMARK_BUILD_FLAVOR,
        help=f"Cargo profile to use when --run-benchmarks is set (default: {BENCHMARK_BUILD_FLAVOR})",
    )
    perf_summary_parser.add_argument(
        "--strict",
        action="store_true",
        help="Reject fallback or incomplete benchmark evidence before publishing the summary",
    )
    _add_bench_timeout_arg(perf_summary_parser)


def _add_release_performance_subcommands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add release performance report subcommands."""
    metadata_parser = subparsers.add_parser(
        "create-release-benchmark-metadata",
        help="Write versioned measurement provenance for a release Criterion archive",
    )
    metadata_parser.add_argument("--tag", required=True, help="Release tag measured by the archive")
    metadata_parser.add_argument("--criterion-dir", type=Path, required=True, help="Criterion directory copied into the archive")
    metadata_parser.add_argument("--output", type=Path, required=True, help="Metadata JSON path inside the archive staging directory")
    _add_project_root_arg(metadata_parser)

    local_parser = subparsers.add_parser("performance-local", help="Compare the current tree against the latest stable release locally")
    local_parser.add_argument("--output", type=Path, default=PERFORMANCE_REPORT_SOURCE, help="Output Markdown report path")
    local_parser.add_argument("--worktree-ref", default="HEAD", help="Git ref for the current temp worktree (default: HEAD)")
    local_parser.add_argument("--no-apply-current-diff", action="store_true", help="Do not apply the current checkout diff to the temp worktree")
    _add_project_root_arg(local_parser)

    assets_parser = subparsers.add_parser("performance-github-assets", help="Compare stored GitHub Release benchmark assets")
    assets_parser.add_argument("current_tag", nargs="?", help="Current release tag")
    assets_parser.add_argument("baseline_tag", nargs="?", help="Baseline release tag")
    assets_parser.add_argument("--output", type=Path, default=GITHUB_ASSETS_PERFORMANCE_REPORT, help="Output Markdown report path")
    assets_parser.add_argument("--worktree-ref", default="HEAD", help="Git ref used to render the report (default: current tag)")
    _add_project_root_arg(assets_parser)

    release_parser = subparsers.add_parser("performance-release", help="Promote a curated release-to-release performance report")
    release_parser.add_argument("current_tag", nargs="?", help="Current release tag")
    release_parser.add_argument("baseline_tag", nargs="?", help="Baseline release tag")
    release_parser.add_argument("--output", type=Path, default=PERFORMANCE_REPORT_SOURCE, help="Retained scratch Markdown report path")
    release_parser.add_argument("--current", type=Path, default=DOCS_PERFORMANCE_REPORT, help="Committed performance report path")
    release_parser.add_argument("--archive-dir", type=Path, default=PERFORMANCE_ARCHIVE_DIR, help="Archive directory for older reports")
    release_parser.add_argument("--worktree-ref", default="HEAD", help="Git ref for the current temp worktree (default: HEAD)")
    release_parser.add_argument("--no-apply-current-diff", action="store_true", help="Do not apply the current checkout diff to the temp worktree")
    _add_project_root_arg(release_parser)

    doc_parser = subparsers.add_parser("performance-doc", help="Promote performance docs from retained CSV and provenance inputs")
    doc_parser.add_argument("--output", type=Path, default=PERFORMANCE_REPORT_SOURCE, help="Scratch Markdown report path")
    doc_parser.add_argument("--artifact-csv", type=Path, default=PERFORMANCE_REPORT_SOURCE.with_suffix(".csv"), help="Retained performance CSV path")
    doc_parser.add_argument(
        "--artifact-provenance",
        type=Path,
        default=PERFORMANCE_REPORT_SOURCE.with_suffix(".provenance.json"),
        help="Retained performance provenance JSON path",
    )
    doc_parser.add_argument("--current", type=Path, default=DOCS_PERFORMANCE_REPORT, help="Committed performance report path")
    doc_parser.add_argument("--archive-dir", type=Path, default=PERFORMANCE_ARCHIVE_DIR, help="Archive directory for older reports")
    _add_project_root_arg(doc_parser)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Benchmark utilities for baseline generation and comparison",
        suggest_on_error=True,
        color=False,
    )
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
    _add_release_performance_subcommands(subparsers)

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


def _cmd_bench_compare(args: argparse.Namespace, project_root: Path) -> None:
    output = args.output if args.output.is_absolute() else project_root / args.output
    success = write_criterion_comparison_report(
        project_root,
        CriterionReportRequest(
            baseline_name=args.baseline,
            output=output,
            stat=args.stat,
            suite=args.suite,
            scope=args.scope,
            criterion_dir=args.criterion_dir,
        ),
    )
    sys.exit(0 if success else 2)


def execute_baseline_commands(args: argparse.Namespace, project_root: Path) -> None:
    """Execute baseline generation and comparison commands."""
    handlers = {
        "bench-compare": _cmd_bench_compare,
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

    _write_text_atomic(output_path, report_text)


def _baseline_fetch_options_from_args(args: argparse.Namespace) -> BaselineFetchOptions:
    return BaselineFetchOptions(
        regenerate_missing=args.regenerate_missing,
        workflow_ref=args.workflow_ref,
        wait_seconds=args.wait_seconds,
        poll_seconds=args.poll_seconds,
    )


def _cmd_compare_baselines(args: argparse.Namespace, project_root: Path) -> None:
    command = "benchmark-utils compare-baselines"
    for option, baseline_path in (("--old", args.old_baseline), ("--new", args.new_baseline)):
        if not baseline_path.is_file():
            print(f"{command}: error: {option} must name a regular file: {baseline_path}", file=sys.stderr)
            sys.exit(3)

    try:
        report_text, regression_found = render_baseline_comparison(project_root, args.old_baseline, args.new_baseline)
    except FileNotFoundError as e:
        print(f"{command}: error: baseline input disappeared: {e}", file=sys.stderr)
        sys.exit(3)
    except UnicodeDecodeError as e:
        print(f"{command}: error: baseline input is not valid UTF-8: {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"{command}: error: could not read baseline input: {e}", file=sys.stderr)
        sys.exit(1)
    except BaselineParseError as e:
        print(f"{command}: error: failed to parse baseline file: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"{command}: error: failed to compare baseline files: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        _write_optional_report(args.output, report_text)
    except OSError as e:
        print(f"{command}: error: could not publish --output {args.output}: {e}", file=sys.stderr)
        sys.exit(1)

    print(report_text, end="" if report_text.endswith("\n") else "\n")
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
        cargo_profile=args.profile,
        bench_timeout=args.bench_timeout,
        strict=args.strict,
    )
    sys.exit(0 if success else 1)


def _cmd_run_release_signal(args: argparse.Namespace, project_root: Path) -> None:
    """Execute the release-signal plan from its single Python owner."""
    try:
        run_release_signal_measurement_plan(
            project_root,
            cargo_profile=args.profile,
            bench_timeout=args.bench_timeout,
            save_baseline=args.save_baseline,
        )
    except _RECOVERABLE_CLI_ERRORS as error:
        print(f"run-release-signal: {error}", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)


def _path_from_root(project_root: Path, path: Path) -> Path:
    """Resolve a CLI path relative to the project root."""
    return path if path.is_absolute() else project_root / path


def _release_config_from_args(
    args: argparse.Namespace,
    project_root: Path,
    request: ResolvedPerformanceRequest,
    *,
    baseline_source: BaselineSource,
    apply_current_diff: bool,
) -> ReleaseReportConfig:
    """Build release report generation config from parsed arguments."""
    worktree_ref = request.worktree_ref
    if baseline_source == "github-assets" and worktree_ref == "HEAD":
        worktree_ref = request.current_tag
    return ReleaseReportConfig(
        repo_root=project_root,
        current_tag=request.current_tag,
        baseline_tag=request.baseline_tag,
        worktree_ref=worktree_ref,
        suite=getattr(args, "suite", "release-signal"),
        scope=getattr(args, "scope", "release-signal"),
        stat=getattr(args, "stat", "median"),
        apply_current_diff=apply_current_diff,
        baseline_source=baseline_source,
    )


def _performance_request_options(
    *,
    args: argparse.Namespace,
    project_root: Path,
    published_latest: bool = False,
    infer_release: bool = False,
    current_vs_latest: bool = False,
) -> PerformanceRequestOptions:
    """Construct tag-resolution options from a performance subcommand."""
    return PerformanceRequestOptions(
        current_tag=getattr(args, "current_tag", None),
        baseline_tag=getattr(args, "baseline_tag", None),
        published_latest=published_latest,
        infer_release=infer_release,
        current_vs_latest=current_vs_latest,
        worktree_ref=args.worktree_ref,
        repo_root=project_root,
    )


def _fetch_for_performance_request(*, project_root: Path, request: ResolvedPerformanceRequest, include_current: bool) -> None:
    """Fetch tags required before release performance worktree generation."""
    current = request.current_tag if include_current else None
    if not include_current and request.worktree_ref == request.current_tag:
        current = request.current_tag
    _fetch_release_tags(repo_root=project_root, tags=request.tags_to_fetch, include_current=current)


def _cmd_create_release_benchmark_metadata(args: argparse.Namespace, project_root: Path) -> None:
    """Write the measurement sidecar consumed by GitHub-asset comparisons."""
    try:
        write_release_benchmark_metadata(
            repo_root=project_root,
            tag=args.tag,
            criterion_dir=_path_from_root(project_root, args.criterion_dir),
            output=_path_from_root(project_root, args.output),
        )
    except _RECOVERABLE_CLI_ERRORS as exc:
        print(f"create-release-benchmark-metadata: {exc}", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)


def _cmd_performance_local(args: argparse.Namespace, project_root: Path) -> None:
    try:
        request = resolve_performance_request(_performance_request_options(args=args, project_root=project_root, current_vs_latest=True))
        output = _path_from_root(project_root, args.output)
        _preflight_performance_destinations(
            output=output,
            report_id=PerformanceReportId(current_tag=request.current_tag, baseline_tag=request.baseline_tag),
        )
        _fetch_for_performance_request(project_root=project_root, request=request, include_current=False)
        config = _release_config_from_args(
            args,
            project_root,
            request,
            baseline_source="local",
            apply_current_diff=not args.no_apply_current_diff,
        )
        report_id = generate_performance_worktree_report(output=output, config=config)
    except _RECOVERABLE_CLI_ERRORS as exc:
        print(f"performance-local: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Generated benchmark report in a temporary worktree and wrote it to {output}")
    print(f"Retained artifact bundle: {_artifact_paths_for_output(output).csv} and {_artifact_paths_for_output(output).provenance}")
    print(f"Current performance report: {report_id.current_tag} vs {report_id.baseline_tag}")
    sys.exit(0)


def _cmd_performance_github_assets(args: argparse.Namespace, project_root: Path) -> None:
    explicit_pair = args.current_tag is not None or args.baseline_tag is not None
    try:
        request = resolve_performance_request(_performance_request_options(args=args, project_root=project_root, published_latest=not explicit_pair))
        if request.current_tag == request.baseline_tag:
            msg = "performance-github-assets requires distinct current and baseline tags"
            raise ValueError(msg)
        output = _path_from_root(project_root, args.output)
        _preflight_performance_destinations(
            output=output,
            report_id=PerformanceReportId(current_tag=request.current_tag, baseline_tag=request.baseline_tag),
        )
        _fetch_for_performance_request(project_root=project_root, request=request, include_current=True)
        config = _release_config_from_args(
            args,
            project_root,
            request,
            baseline_source="github-assets",
            apply_current_diff=False,
        )
        report_id = generate_performance_worktree_report(output=output, config=config)
    except _RECOVERABLE_CLI_ERRORS as exc:
        print(f"performance-github-assets: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Generated benchmark report from GitHub Release assets and wrote it to {output}")
    print(f"Retained artifact bundle: {_artifact_paths_for_output(output).csv} and {_artifact_paths_for_output(output).provenance}")
    print(f"Current performance report: {report_id.current_tag} vs {report_id.baseline_tag}")
    sys.exit(0)


def _cmd_performance_release(args: argparse.Namespace, project_root: Path) -> None:
    explicit_pair = args.current_tag is not None or args.baseline_tag is not None
    try:
        request = resolve_performance_request(_performance_request_options(args=args, project_root=project_root, infer_release=not explicit_pair))
        if request.current_tag == request.baseline_tag:
            msg = "performance-release requires distinct current and baseline tags"
            raise ValueError(msg)
        if explicit_pair and request.worktree_ref == "HEAD" and request.current_tag != _current_package_tag(project_root):
            msg = f"explicit current tag {request.current_tag} does not match the HEAD package version {_current_package_tag(project_root)}"
            raise ValueError(msg)
        current = _path_from_root(project_root, args.current)
        archive_dir = _path_from_root(project_root, args.archive_dir)
        output = _path_from_root(project_root, args.output)
        _preflight_performance_destinations(
            output=output,
            report_id=PerformanceReportId(current_tag=request.current_tag, baseline_tag=request.baseline_tag),
            current=current,
            archive_dir=archive_dir,
            project_root=project_root,
        )
        _fetch_for_performance_request(project_root=project_root, request=request, include_current=False)
        config = _release_config_from_args(
            args,
            project_root,
            request,
            baseline_source="local",
            apply_current_diff=not args.no_apply_current_diff,
        )
        report_id = generate_and_promote_performance_report(output=output, current=current, archive_dir=archive_dir, config=config)
    except _RECOVERABLE_CLI_ERRORS as exc:
        print(f"performance-release: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Generated benchmark report in a temporary worktree and promoted it to {current}")
    print(f"Retained artifact bundle: {_artifact_paths_for_output(output).csv} and {_artifact_paths_for_output(output).provenance}")
    print(f"Current performance report: {report_id.current_tag} vs {report_id.baseline_tag}")
    print(f"Archive directory: {archive_dir}")
    sys.exit(0)


def _cmd_performance_doc(args: argparse.Namespace, project_root: Path) -> None:
    """Render and promote docs from retained artifacts only."""
    try:
        output = _path_from_root(project_root, args.output)
        artifacts = ArtifactPaths(
            csv=_path_from_root(project_root, args.artifact_csv),
            provenance=_path_from_root(project_root, args.artifact_provenance),
        )
        current = _path_from_root(project_root, args.current)
        archive_dir = _path_from_root(project_root, args.archive_dir)
        report_id = render_and_promote_performance_artifacts(
            output=output,
            artifacts=artifacts,
            destinations=PerformancePromotionDestinations(
                project_root=project_root,
                current=current,
                archive_dir=archive_dir,
            ),
            expected_current_tag=_current_package_tag(project_root),
        )
    except _RECOVERABLE_CLI_ERRORS as exc:
        print(f"performance-doc: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Rendered retained artifacts and promoted the report to {current}")
    print(f"Current performance report: {report_id.current_tag} vs {report_id.baseline_tag}")
    print(f"Archive directory: {archive_dir}")
    sys.exit(0)


def execute_release_performance_commands(args: argparse.Namespace, project_root: Path) -> None:
    """Execute release performance report commands."""
    handlers = {
        "create-release-benchmark-metadata": _cmd_create_release_benchmark_metadata,
        "performance-local": _cmd_performance_local,
        "performance-github-assets": _cmd_performance_github_assets,
        "performance-release": _cmd_performance_release,
        "performance-doc": _cmd_performance_doc,
    }
    handler = handlers.get(args.command)
    if handler is None:
        msg = f"Unknown release performance command: {args.command}"
        raise ValueError(msg)
    handler(args, project_root)


def execute_performance_summary_commands(args: argparse.Namespace, project_root: Path) -> None:
    """Execute performance summary commands."""
    handlers = {
        "generate-summary": _cmd_generate_summary,
        "run-release-signal": _cmd_run_release_signal,
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
        "bench-compare": execute_baseline_commands,
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
        "run-release-signal": execute_performance_summary_commands,
        "create-release-benchmark-metadata": execute_release_performance_commands,
        "prepare-baseline": _execute_regression_commands_with_root,
        "set-no-baseline": _execute_regression_commands_with_root,
        "extract-baseline-commit": _execute_regression_commands_with_root,
        "determine-skip": _execute_regression_commands_with_root,
        "display-skip-message": _execute_regression_commands_with_root,
        "display-no-baseline": _execute_regression_commands_with_root,
        "run-regression-test": _execute_regression_commands_with_root,
        "display-results": _execute_regression_commands_with_root,
        "regression-summary": _execute_regression_commands_with_root,
        "performance-local": execute_release_performance_commands,
        "performance-github-assets": execute_release_performance_commands,
        "performance-release": execute_release_performance_commands,
        "performance-doc": execute_release_performance_commands,
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
            project_root = cast("Path", args.project_root).resolve()
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
