#!/usr/bin/env python3
"""
benchmark_utils.py - Delaunay workload policy and shared performance evidence

This module provides functions for:
- Parsing Criterion benchmark output and JSON data
- Selecting and preflighting scientific benchmark workloads
- Comparing fresh measurements with compatible recorded provenance
- Retaining and publishing shared JSON evidence

Generic parsing, snapshots, processes, and publication use research-repo-tools.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NoReturn, TypeIs, cast

from research_repo_tools.archives import extract_archive as _safe_extract_tar
from research_repo_tools.criterion import Comparison, Estimate, Sample, compare_samples, read_estimate
from research_repo_tools.files import replace_many
from research_repo_tools.host_metadata import capture_host
from research_repo_tools.process import ExecutableNotFoundError, format_exception_diagnostics, run_command as run_safe_command, run_command_live
from research_repo_tools.publication import plan_outputs, publish_publication
from research_repo_tools.release_assets import download_release_asset
from research_repo_tools.release_pairs import PairMode, resolve_pair
from research_repo_tools.releases import published_releases
from research_repo_tools.worktrees import apply_snapshot, capture_snapshot, temporary_worktree

from benchmark_models import (
    CircumspherePerformanceData,
    CircumsphereTestCase,
)
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
    load_bundle_bytes,
    serialize_bundle,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

TIME_UNIT_TO_MICROSECONDS = {"ns": 1e-3, "µs": 1.0, "μs": 1.0, "us": 1.0, "ms": 1e3, "s": 1e6}


type ExceptionFamily = tuple[type[BaseException], ...]

run_cargo_command = partial(run_safe_command, "cargo")
run_cargo_live = partial(run_command_live, "cargo")
run_git_command = partial(run_safe_command, "git")


class ProjectRootNotFoundError(Exception):
    """The benchmark command was invoked outside a Cargo repository."""


def find_project_root() -> Path:
    """Find the enclosing Cargo repository for benchmark commands."""
    current = Path.cwd()
    for candidate in (current, *current.parents):
        if (candidate / "Cargo.toml").is_file():
            return candidate
    message = "Could not locate Cargo.toml to determine project root"
    raise ProjectRootNotFoundError(message)


def get_git_commit_hash(cwd: Path | None = None) -> str:
    """Read the measured checkout's source identity."""
    return run_git_command(["rev-parse", "HEAD"], cwd=cwd).stdout.strip()


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
        "DelaunayTriangulationBuilder::build",
    ),
    "boundary_facets": (
        "Boundary facets",
        "DelaunayTriangulation::boundary_facets",
    ),
    "convex_hull": (
        "Convex hull",
        "ConvexHull::try_from_triangulation",
    ),
    "convex_hull_queries": (
        "Convex hull queries",
        "ConvexHull::{is_point_outside,find_visible_facets,find_nearest_visible_facet}",
    ),
    "validation": (
        "Validation",
        "DelaunayTriangulation::validate",
    ),
    "incremental_insert": (
        "Incremental insert",
        "DelaunayTriangulation::insert_vertex",
    ),
    "explicit_import": (
        "Explicit pseudomanifold import",
        "DelaunayTriangulationBuilder::try_from_vertices_and_simplices",
    ),
    "proof_boundaries": (
        "Proof boundaries",
        "TriangulationBuilder::build;DelaunayRefinementBuilder::build",
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
DOCS_PERFORMANCE_REPORT = Path("docs") / "performance.md"
PERFORMANCE_ARCHIVE_DIR = Path("docs") / "archive" / "performance"
RELEASE_BENCH_TIMEOUT_SECONDS = 7200
RELEASE_PREFLIGHT_TIMEOUT_SECONDS = 600
RELEASE_COMMAND_TIMEOUT_SECONDS = 600
RELEASE_SIGNAL_TIMEOUT_SECONDS = (RELEASE_BENCH_TIMEOUT_SECONDS + RELEASE_PREFLIGHT_TIMEOUT_SECONDS) * len(RELEASE_SIGNAL_MEASUREMENT_PLAN)
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
    bundle: PerformanceBundle
    source_payload: bytes
    source_text: str
    current_text: str | None
    archive_path: Path | None
    durable_artifacts: ArtifactPaths
    source_evidence: bytes
    source_provenance: bytes
    mutation_paths: tuple[Path, ...]


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

    def __post_init__(self) -> None:
        """Reject contract crossings before any tag fetch or benchmark worktree."""


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


def _write_ci_performance_metrics(project_root: Path, stdout: object, *, require_metrics: bool = False) -> None:
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


def _is_object_mapping(value: object) -> TypeIs[Mapping[object, object]]:
    """Return whether a raw value can be treated as an object-keyed mapping."""
    return isinstance(value, Mapping)


def _require_positive_int_field(name: str, value: object) -> None:
    """Reject values that are not positive non-bool integers."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        msg = f"{name} must be a positive integer (got {value!r})"
        raise ValueError(msg)


def _load_criterion_estimate(estimates_path: Path) -> CriterionEstimate | None:
    """Load shared Criterion data with the summary's required interval bounds."""
    try:
        estimate = read_estimate(estimates_path, statistic="mean")
    except OSError, ValueError:
        return None
    if estimate.lower is None or estimate.upper is None:
        return None
    return CriterionEstimate(mean_ns=estimate.point, low_ns=estimate.lower, high_ns=estimate.upper)


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
    # Directory names escape slashes in group/function IDs. Reuse the canonical
    # metadata reader so report coverage and release comparisons agree.
    # Stale samples without usable identity metadata cannot contribute coverage.
    samples = _criterion_estimates_by_id(criterion_dir, "base", skip_invalid_metadata=True)
    samples.update(_criterion_estimates_by_id(criterion_dir, "new", skip_invalid_metadata=True))
    return tuple(sorted(result_id for result_id, sample in samples.items() if _load_criterion_estimate(sample.estimates) is not None))


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


def preflight_release_signal(project_root: Path, *, cargo_profile: str, bench_timeout: int) -> None:
    """Execute every fixture once; do not publish timing or measurement metadata."""
    for measurement in RELEASE_SIGNAL_MEASUREMENT_PLAN:
        print(f"🔎 Preflight release-signal target {measurement.target}...", flush=True)
        run_cargo_live(
            ["bench", "--profile", cargo_profile, "--bench", measurement.target, "--", "--test"],
            cwd=project_root,
            timeout=min(bench_timeout, RELEASE_PREFLIGHT_TIMEOUT_SECONDS),
        )


def run_release_signal_measurement_plan(
    project_root: Path,
    *,
    cargo_profile: str = BENCHMARK_BUILD_FLAVOR,
    bench_timeout: int = 1800,
    save_baseline: str | None = None,
    preflight_only: bool = False,
) -> dict[str, str]:
    """Preflight every curated fixture before sampling; smoke output is never evidence."""
    _require_positive_int_field("bench_timeout", bench_timeout)
    if preflight_only and save_baseline is not None:
        msg = "preflight cannot save a measurement baseline"
        raise ValueError(msg)
    preflight_release_signal(project_root, cargo_profile=cargo_profile, bench_timeout=bench_timeout)
    if preflight_only:
        return {}
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


# =============================================================================
# PERFORMANCE SUMMARY GENERATOR
# =============================================================================


class PerformanceSummaryGenerator:
    """Generate performance summary markdown from benchmark results."""

    def __init__(self, project_root: Path) -> None:
        """Initialize with project root directory."""
        self.project_root = project_root

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
            f"- **Package Version**: {self.current_version}",
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
            host = capture_host(self.project_root, probes=(("rustc", ("rustc", "--version")),))
            lines.extend(
                [
                    f"- **Hardware**: {host.cpu} ({host.physical_cores} cores)",
                    f"- **Memory bytes**: {host.memory_bytes}",
                    f"- **OS**: {host.os}",
                    f"- **Rust**: {dict(host.tools).get('rustc')}",
                ],
            )
        except _RECOVERABLE_CLI_ERRORS as e:
            logger.debug("Could not get hardware info: %s", e)
            lines.append("- **Hardware**: Unknown")

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
        return path_parts[-1] if len(path_parts) > 2 else "fixed fixture"

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
            "Combined input counts use `v` for vertices and `s` for simplices (for example, `120v/226s`).",
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
                input_size = re.sub(r"^vertices_(\d+)_simplices_(\d+)$", r"\1v/\2s", result.input_size)
                lines.append(
                    f"| `{result.benchmark_id}` | {result.dimension} | {input_size} | {result.variant} | "
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
            "# Measure both revisions with one current harness and retain shared JSON evidence",
            "just performance-local",
            "```",
            "",
            "### Customization",
            "",
            "For manual updates or custom analysis, modify the `PerformanceSummaryGenerator`",
            "class in `tooling/python/benchmark_utils.py`. This provides enhanced control over",
            "dynamic vs static content organization and supports parsing numerical accuracy",
            "data from live benchmark runs.",
            "",
        ]


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


def _read_text(path: Path) -> str:
    """Read UTF-8 text."""
    return path.read_text(encoding="utf-8")


def _write_text_atomic(path: Path, text: str) -> None:
    """Publish UTF-8 report bytes through the shared transaction."""
    replace_many({path: text.encode("utf-8")})


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Publish retained bytes through the shared transaction."""
    replace_many({path: payload})


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
        host = capture_host(repo_root, probes=(("rustc", ("rustc", "--version")),))
    except _RECOVERABLE_CLI_ERRORS:
        logger.debug("Unable to collect hardware metadata for benchmark report", exc_info=True)
        lines.append("- **Hardware**: Unknown")
    else:
        lines.extend(
            [
                f"- **OS**: {host.os}",
                f"- **CPU**: {host.cpu} ({host.physical_cores} cores, {host.logical_threads} threads)",
                f"- **Memory bytes**: {host.memory_bytes}",
                f"- **Rust**: {dict(host.tools).get('rustc')}",
                f"- **Architecture**: {host.architecture}",
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


def _read_criterion_timing_estimate(estimates_json: Path, stat: str) -> TimingEstimate:
    """Read shared Criterion data and require the consumer's complete interval."""
    if stat not in {"mean", "median"}:
        raise ValueError(f"unsupported Criterion statistic: {stat!r}")
    estimate = read_estimate(estimates_json, statistic=stat)
    if estimate.lower is None or estimate.upper is None or estimate.confidence_level is None:
        raise ValueError(f"Criterion estimate requires confidence bounds and confidence_level: {estimates_json}")
    return TimingEstimate(estimate.point, estimate.lower, estimate.upper, estimate.confidence_level)


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


def _criterion_estimates_by_id(criterion_dir: Path, sample: str, *, skip_invalid_metadata: bool = False) -> dict[str, CriterionSample]:
    """Map sample IDs to estimates, optionally skipping unusable identity metadata."""
    results: dict[str, CriterionSample] = {}
    if not criterion_dir.is_dir():
        return results
    for estimates_json in sorted(criterion_dir.rglob("estimates.json")):
        if estimates_json.parent.name != sample:
            continue
        try:
            criterion_sample = _criterion_sample(estimates_json, criterion_dir)
        except TypeError, ValueError:
            if not skip_invalid_metadata:
                raise
            continue
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
) -> list[Comparison]:
    """Collect Criterion comparisons between ``new`` and a named saved baseline."""
    current = _criterion_estimates_by_id(criterion_dir, "new")
    baseline = _criterion_estimates_by_id(criterion_dir, baseline_name)
    if stat not in {"mean", "median"}:
        raise ValueError(f"unsupported Criterion statistic: {stat!r}")

    def selected_sample(inventory: dict[str, CriterionSample]) -> Sample:
        """Apply scientific scope while the shared package owns pairing and ratios."""
        estimates = []
        for benchmark_id, identity in inventory.items():
            if not _benchmark_in_compare_scope(benchmark_id, suite, scope):
                continue
            # Historical tables require complete intervals on both sides.
            estimate = _read_criterion_timing_estimate(identity.estimates, stat)
            estimates.append((benchmark_id, Estimate(estimate.median_ns, estimate.ci_lower_ns, estimate.ci_upper_ns, estimate.confidence_level)))
        return Sample(tuple(estimates), statistic=stat, unit="ns")

    return list(compare_samples(selected_sample(baseline), selected_sample(current)).comparisons)


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


def _criterion_comparison_table(comparisons: list[Comparison], baseline_name: str) -> str:
    """Render Criterion comparisons as grouped Markdown tables."""
    sections: list[str] = []
    by_group: dict[str, list[Comparison]] = {}
    for comparison in comparisons:
        by_group.setdefault(_criterion_scope_prefix(comparison.benchmark), []).append(comparison)

    for group in sorted(by_group):
        lines = [
            f"### {group}",
            "",
            f"| Benchmark | {baseline_name} | Latest | Change | Speedup |",
            "|-----------|-------:|-------:|-------:|--------:|",
        ]
        for comparison in sorted(by_group[group], key=lambda item: item.benchmark):
            label = comparison.benchmark.removeprefix(f"{group}/")
            lines.append(
                "| "
                + " | ".join(
                    [
                        label,
                        _format_ns(comparison.baseline.point),
                        _format_ns(comparison.current.point),
                        _format_pct_change(-comparison.percent_reduction),
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

# Rebuild and promote documentation from retained shared JSON only
just performance-doc

# GitHub Release benchmark assets
just performance-github-assets

# Explicit repair
just performance-release <current-tag> <previous-tag>
```

`just performance-local` writes `performance.md` plus retained `performance.comparison.json` and
`performance.evidence.json` under `target/bench-reports/` without promoting documentation.
`just performance-github-assets` writes a `github-assets-performance.*` bundle without local
Cargo benchmark runs. New release archives must contain the supported versioned measurement
metadata. Existing legacy archives remain loadable as provenance-limited absolute timing
evidence, but they cannot be promoted. GitHub-asset ratios are always suppressed because the
archives were measured in separate sessions. Local-worktree ratios require compatible hosts,
toolchains, harnesses, normalized measurement plans, completed targets, and confidence levels.
`just performance-doc` consumes the retained canonical shared JSON pair without Cargo or
measurement worktrees and rejects incomplete, invalid, stale, same-version, or scientifically
non-comparable inputs. `just performance-release` retains and reload-validates the same bundle,
copies the exact comparison JSON/evidence bytes to `docs/archive/performance/data/`, and promotes the
documentation with per-file atomic replacement plus rollback for caught failures. After a hard
interruption, inspect the destinations and rerun the command.

The shared Criterion comparison JSON and digest-bound evidence envelope are canonical.
The envelope retains Delaunay's workload and comparability policy. Timing changes are descriptive;
the marginal timing confidence intervals are not confidence intervals for ratios or significance tests.

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
    comparisons: list[Comparison],
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
    payload, _ = serialize_bundle(bundle)
    if evidence_state not in ("scratch", "promoted"):
        msg = f"unsupported evidence state: {evidence_state!r}"
        raise ValueError(msg)
    evidence_label = "Retained scratch evidence" if evidence_state == "scratch" else "Promoted evidence"
    evidence_payload = _artifact_evidence_path(evidence_paths.payload)
    evidence_provenance = _artifact_evidence_path(evidence_paths.provenance)
    has_comparisons = any(row.coverage_status == "comparable" for row in bundle.rows)
    lines = [
        "# Benchmark Performance",
        "",
        "> [!IMPORTANT]",
        "> Generated by `benchmark-utils` from validated shared JSON evidence; do not edit this report directly.",
        f"> {evidence_label}: `{evidence_payload}` and `{evidence_provenance}` (Evidence SHA-256 `{hashlib.sha256(payload).hexdigest()}`).",
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


def _archive_index_text(archive_dir: Path, additional: tuple[str, ...] = ()) -> str:
    """Render Delaunay's existing archive navigation before publication."""
    reports = sorted({path.name for path in archive_dir.glob("*.md") if path.name != "README.md"} | set(additional))
    lines = [
        "# Archived Performance Reports",
        "",
        "Older release-to-release benchmark comparisons are archived here.",
        "`docs/performance.md` contains the latest curated comparison.",
        "",
    ]
    if reports:
        lines.extend(f"- [{name.removesuffix('.md')}]({name})" for name in reports)
    else:
        lines.append("- No archived performance reports yet.")
    return "\n".join(lines) + "\n"


def _durable_performance_artifact_paths(archive_dir: Path, report_id: PerformanceReportId) -> ArtifactPaths:
    """Return tracked evidence paths for one promoted release pair."""
    stem = f"{report_id.current_tag}-vs-{report_id.baseline_tag}"
    data_dir = archive_dir / "data"
    return ArtifactPaths(payload=data_dir / f"{stem}.comparison.json", provenance=data_dir / f"{stem}.evidence.json")


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
        payload=_repository_relative_path(project_root, durable.payload, label="promoted performance payload"),
        provenance=_repository_relative_path(
            project_root,
            durable.provenance,
            label="promoted performance provenance",
        ),
    )


def _validated_promotion_source(
    source: bytes,
    bundle: PerformanceBundle,
    expected: PerformanceReportId,
    durable_artifacts: ArtifactPaths,
    project_root: Path,
) -> tuple[PerformanceBundle, str, PerformanceReportId]:
    """Validate one canonical report and its independently expected identity."""
    bundle.require_promotable()
    source_text = _normalize_how_to_update(source.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n"))
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
    source_payload = source.read_bytes()
    source_evidence = artifacts.payload.read_bytes()
    source_provenance = artifacts.provenance.read_bytes()
    bundle = load_bundle_bytes(source_evidence, source_provenance, source=str(artifacts.payload.parent))
    _, source_text, source_id = _validated_promotion_source(
        source_payload,
        bundle,
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
        "source payload": artifacts.payload,
        "source provenance": artifacts.provenance,
        "current report": current,
        "archive index": index_path,
        "durable payload": durable_artifacts.payload,
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
    _reject_conflicting_payload(durable_artifacts.payload, source_evidence, description="durable performance payload")
    _reject_conflicting_payload(
        durable_artifacts.provenance,
        source_provenance,
        description="durable performance provenance",
    )

    mutation_paths = (
        current,
        index_path,
        durable_artifacts.payload,
        durable_artifacts.provenance,
        *(() if archive_path is None else (archive_path,)),
    )
    for label, path in (
        ("current performance report", current),
        ("performance archive index", index_path),
        ("durable performance payload", durable_artifacts.payload),
        ("durable performance provenance", durable_artifacts.provenance),
        *(() if archive_path is None else (("archived performance report", archive_path),)),
    ):
        _repository_relative_path(project_root, path, label=label)
    return PerformancePromotionPlan(
        report_id=source_id,
        bundle=bundle,
        source_payload=source_payload,
        source_text=source_text,
        current_text=current_text,
        archive_path=archive_path,
        durable_artifacts=durable_artifacts,
        source_evidence=source_evidence,
        source_provenance=source_provenance,
        mutation_paths=mutation_paths,
    )


def _promotion_outputs(  # noqa: PLR0913 - evidence and publication destinations are independent inputs
    *,
    bundle: PerformanceBundle,
    rendered: str,
    current: Path,
    archive_dir: Path,
    project_root: Path,
    payload: bytes,
    provenance_payload: bytes,
) -> tuple[dict[Path, bytes], tuple[Path, ...]]:
    """Select scientific report candidates and preserve the original legacy evidence."""
    bundle.require_promotable()
    report_id = PerformanceReportId(current_tag=bundle.context.release.current, baseline_tag=bundle.context.release.baseline)
    durable = _durable_performance_artifact_paths(archive_dir, report_id)
    prior_text, archive_path = _promotion_archive_destination(current, archive_dir, report_id)
    _reject_conflicting_payload(durable.payload, payload, description="durable performance payload")
    _reject_conflicting_payload(durable.provenance, provenance_payload, description="durable performance provenance")
    outputs = {current: rendered.encode("utf-8"), durable.payload: payload, durable.provenance: provenance_payload}
    immutable = [durable.payload, durable.provenance]
    if archive_path is not None:
        if archive_path.exists():
            if _normalize_how_to_update(_read_text(archive_path)) != prior_text:
                msg = f"existing performance archive conflicts with the current report: {archive_path}"
                raise ValueError(msg)
            archived = archive_path.read_bytes()
        else:
            archived = current.read_bytes()
        outputs[archive_path] = archived
        immutable.append(archive_path)
    outputs[archive_dir / "README.md"] = _archive_index_text(
        archive_dir,
        () if archive_path is None else (archive_path.name,),
    ).encode("utf-8")
    for path in outputs:
        _repository_relative_path(project_root, path, label="performance publication")
    return outputs, tuple(immutable)


def _publish_performance_outputs(
    root: Path,
    outputs: dict[Path, bytes],
    inputs: dict[Path, bytes],
    immutable: tuple[Path, ...] = (),
) -> None:
    """Compose validated Delaunay candidates into the supported shared transaction."""
    root = root.absolute()
    publish_publication(
        plan_outputs(
            root,
            {path.absolute().relative_to(root).as_posix(): data for path, data in outputs.items()},
            inputs={path.absolute().relative_to(root).as_posix(): data for path, data in inputs.items()},
            immutable=tuple(path.absolute().relative_to(root).as_posix() for path in immutable),
        )
    )


def promote_performance_report(
    *,
    source: Path,
    artifacts: ArtifactPaths,
    destinations: PerformancePromotionDestinations,
    expected: PerformanceReportId,
) -> PerformanceReportId:
    """Validate the shared report and publish its archive, evidence and index together."""
    plan = _plan_performance_promotion(source=source, artifacts=artifacts, destinations=destinations, expected=expected)
    outputs, immutable = _promotion_outputs(
        bundle=plan.bundle,
        rendered=plan.source_text,
        current=destinations.current,
        archive_dir=destinations.archive_dir,
        project_root=destinations.project_root,
        payload=plan.source_evidence,
        provenance_payload=plan.source_provenance,
    )
    _publish_performance_outputs(
        destinations.project_root,
        outputs,
        {source: plan.source_payload, artifacts.payload: plan.source_evidence, artifacts.provenance: plan.source_provenance},
        immutable,
    )
    return plan.report_id


def published_stable_release_tags(repo_root: Path) -> list[str]:
    """Return shared-discovered stable releases for Delaunay artifact selection."""
    return [release.tag for release in published_releases(repo_root)]


def _run_tool(command: str, args: list[str], *, cwd: Path, options: ToolRunOptions | None = None) -> None:
    """Run a support command through shared bounded process execution."""
    resolved_options = options or ToolRunOptions()
    try:
        runner = run_command_live if resolved_options.stream_output else run_safe_command
        runner(command, args, cwd=cwd, timeout=resolved_options.timeout, env=resolved_options.env)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(format_exception_diagnostics(exc)) from exc


def _progress(message: str) -> None:
    """Write one immediately visible performance-workflow phase marker."""
    print(f"[performance] {message}", file=sys.stderr, flush=True)


def _run_git(args: list[str], *, cwd: Path, timeout: int = RELEASE_COMMAND_TIMEOUT_SECONDS) -> None:
    """Run the explicitly invoked performance recipe's Git operations."""
    try:
        run_git_command(args, cwd=cwd, timeout=timeout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(format_exception_diagnostics(exc)) from exc


def _normalize_worktree_ref_for_tag(worktree_ref: str, current_tag: str) -> str:
    """Use the normalized current tag when a bare matching tag was requested."""
    try:
        normalized_ref = normalize_release_tag(worktree_ref)
    except ValueError:
        return worktree_ref
    return current_tag if normalized_ref == current_tag else worktree_ref


def resolve_performance_request(options: PerformanceRequestOptions) -> ResolvedPerformanceRequest:
    """Apply shared pair selection, retaining the shared publication restriction."""
    requested_modes = sum((options.published_latest, options.infer_release, options.current_vs_latest))
    if requested_modes > 1:
        msg = "choose only one of --published-latest, --infer-release, or --current-vs-latest"
        raise ValueError(msg)
    mode: PairMode = (
        "published-latest"
        if options.published_latest
        else "infer-release"
        if options.infer_release
        else "current-vs-latest"
        if options.current_vs_latest
        else "explicit"
    )
    pair = resolve_pair(
        mode,
        package_tag=_current_package_tag(options.repo_root) if requested_modes else options.current_tag or "v0.0.0",
        releases=published_releases(options.repo_root) if requested_modes else (),
        order="version" if options.infer_release else "published",
        current=options.current_tag,
        baseline=options.baseline_tag,
    )
    worktree_ref = pair.current if options.published_latest and options.worktree_ref == "HEAD" else options.worktree_ref
    return ResolvedPerformanceRequest(
        current_tag=pair.current,
        baseline_tag=pair.baseline,
        worktree_ref=_normalize_worktree_ref_for_tag(worktree_ref, pair.current),
        tags_to_fetch=(pair.current, pair.baseline) if options.published_latest else (pair.baseline,),
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
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(format_exception_diagnostics(exc)) from exc
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
        Path("tooling") / "python" / "benchmark_utils.py",
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
    """Bind source identity to the shared snapshot, including new unignored files."""
    snapshot = capture_snapshot(checkout)
    revision_timestamp = run_git_command(
        ["show", "-s", "--format=%cI", snapshot.revision],
        cwd=checkout,
        timeout=RELEASE_COMMAND_TIMEOUT_SECONDS,
    ).stdout.strip()
    # Preserve the clean-tag identity used by release archives; bind any new files too.
    digest = hashlib.sha256(f"commit {snapshot.revision}\n".encode("ascii") + snapshot.patch)
    if snapshot.untracked:
        digest.update(json.dumps([(name, payload.hex(), mode) for name, payload, mode in snapshot.untracked], separators=(",", ":")).encode("utf-8"))
    return SourceState(
        version=normalize_release_tag(version),
        commit=snapshot.revision,
        ref=ref,
        revision_timestamp=revision_timestamp,
        git_clean=not (snapshot.patch or snapshot.untracked),
        source_state_sha256=digest.hexdigest(),
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
    host = capture_host(repo_root)
    return HostIdentity(
        status="recorded",
        cpu=host.cpu or "",
        operating_system=host.os or "",
        architecture=host.architecture or "",
    )


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
        command = ("just", "bench-latest", str(RELEASE_BENCH_TIMEOUT_SECONDS))
        _progress("running current release-signal benchmarks")
        _run_tool(
            command[0],
            list(command[1:]),
            cwd=worktree,
            options=ToolRunOptions(
                timeout=RELEASE_SIGNAL_TIMEOUT_SECONDS,
                env=env,
                stream_output=True,
            ),
        )
        _progress("completed current release-signal benchmarks")
        return (command,)
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
    revision = _resolve_worktree_revision(config.repo_root, config.baseline_tag)
    with temporary_worktree(config.repo_root, baseline_worktree, revision, allow_git_mutations=True):
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


def _resolve_worktree_revision(repo_root: Path, ref: str) -> str:
    """Resolve a caller-selected ref to the shared worktree API's commit identity."""
    commit = run_git_command(
        ["rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"],
        cwd=repo_root,
        timeout=RELEASE_COMMAND_TIMEOUT_SECONDS,
    ).stdout.strip()
    if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit) is None:
        msg = f"could not resolve Git ref {ref!r} to a full commit ID"
        raise ValueError(msg)
    return commit


def _expected_tag_commit(repo_root: Path, tag: str) -> str:
    """Resolve one requested release tag to its peeled commit object."""
    return _resolve_worktree_revision(repo_root, normalize_release_tag(tag))


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
    """Acquire bounded exact Delaunay asset bytes through the shared API."""
    artifact = download_dir / f"delaunay-{tag}-criterion-baseline.tar.gz"
    repository = _run_tool_output("gh", ["repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"], cwd=repo_root)
    download_release_asset(repo_root, repository, tag, artifact.name, artifact)
    command = (
        "python-api",
        "research_repo_tools.release_assets.download_release_asset",
        repository,
        tag,
        artifact.name,
        str(artifact),
    )
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
    tmp_dir.mkdir(parents=True, exist_ok=True)
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


@contextmanager
def _performance_workspace() -> Iterator[Path]:
    """Retain the containing workspace when Git cannot remove either checkout."""
    directory = Path(tempfile.mkdtemp(prefix="delaunay-performance-")).resolve()
    try:
        yield directory
    finally:
        if any((directory / name).exists() for name in ("worktree", "baseline-worktree")):
            _progress(f"retained performance workspace for recovery: {directory}")
        else:
            shutil.rmtree(directory)


def _build_performance_bundle_in_temp_worktree(*, config: ReleaseReportConfig) -> PerformanceBundle:
    """Measure or load a comparison in temporary worktrees and return trusted data."""
    with _performance_workspace() as tmp_dir:
        worktree = tmp_dir / "worktree"

        _progress(f"preparing current worktree for {config.current_tag}")
        revision = _resolve_worktree_revision(config.repo_root, config.worktree_ref)
        with temporary_worktree(config.repo_root, worktree, revision, allow_git_mutations=True):
            if config.apply_current_diff:
                apply_snapshot(worktree, capture_snapshot(config.repo_root))
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


def _artifact_paths_for_output(output: Path) -> ArtifactPaths:
    """Return the adjacent canonical artifact paths for one Markdown output."""
    return ArtifactPaths(payload=output.with_suffix(".comparison.json"), provenance=output.with_suffix(".evidence.json"))


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
    paths = {"Markdown output": output, "artifact payload": artifacts.payload, "artifact provenance": artifacts.provenance}
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
        paths["durable payload"] = durable.payload
        paths["durable provenance"] = durable.provenance
        tracked_destinations = {
            current,
            archive_dir / "README.md",
            archive_dir / report_id.archive_name,
            durable.payload,
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
    """Validate serialized evidence and publish every candidate in one shared transaction."""
    artifacts = _artifact_paths_for_output(output)
    payload, provenance_payload = serialize_bundle(bundle)
    validated = load_bundle_bytes(payload, provenance_payload, source="new measurement")
    report_id = PerformanceReportId(current_tag=bundle.context.release.current, baseline_tag=bundle.context.release.baseline)
    root = project_root if project_root is not None else output.parent
    root.mkdir(parents=True, exist_ok=True)
    immutable: tuple[Path, ...] = ()
    if current is None:
        rendered = render_performance_bundle(validated, evidence_paths=artifacts, evidence_state="scratch")
        outputs: dict[Path, bytes] = {}
    else:
        if archive_dir is None or project_root is None:
            msg = "archive_dir and project_root are required when promoting performance documentation"
            raise ValueError(msg)
        durable = _durable_performance_artifact_paths(archive_dir, report_id)
        rendered = render_performance_bundle(
            validated,
            evidence_paths=_promoted_evidence_paths(durable, project_root=project_root),
            evidence_state="promoted",
        )
        outputs, immutable = _promotion_outputs(
            bundle=validated,
            rendered=rendered,
            current=current,
            archive_dir=archive_dir,
            project_root=project_root,
            payload=payload,
            provenance_payload=provenance_payload,
        )
    outputs.update({output: rendered.encode("utf-8"), artifacts.payload: payload, artifacts.provenance: provenance_payload})
    ensure_distinct_paths({"report": output, "payload": artifacts.payload, "provenance": artifacts.provenance, **({"current": current} if current else {})})
    # The legacy adapter validates original bytes before the planner snapshots them.
    # These temporary inputs are never promoted or mistaken for a shared complete run.
    with tempfile.TemporaryDirectory(prefix=".performance-inputs-", dir=root) as directory:
        retained = ArtifactPaths(payload=Path(directory) / "input.comparison.json", provenance=Path(directory) / "input.json")
        retained.payload.write_bytes(payload)
        retained.provenance.write_bytes(provenance_payload)
        _publish_performance_outputs(root, outputs, {retained.payload: payload, retained.provenance: provenance_payload}, immutable)
    return report_id


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
    """Render retained shared data and promote all candidates without measurement."""
    payload, provenance_payload = artifacts.payload.read_bytes(), artifacts.provenance.read_bytes()
    bundle = load_bundle_bytes(payload, provenance_payload, source=str(artifacts.payload.parent))
    normalized_expected_current = normalize_release_tag(expected_current_tag)
    if bundle.context.release.current != normalized_expected_current:
        msg = f"retained current release {bundle.context.release.current} does not match independently expected release {normalized_expected_current}"
        raise ValueError(msg)
    if bundle.context.release.current == bundle.context.release.baseline:
        msg = "performance-doc cannot promote a same-version local performance comparison"
        raise ValueError(msg)
    bundle.require_promotable()
    report_id = PerformanceReportId(current_tag=bundle.context.release.current, baseline_tag=bundle.context.release.baseline)
    _preflight_performance_destinations(
        output=output,
        report_id=report_id,
        current=destinations.current,
        archive_dir=destinations.archive_dir,
        project_root=destinations.project_root,
    )
    durable = _durable_performance_artifact_paths(destinations.archive_dir, report_id)
    rendered = render_performance_bundle(
        bundle,
        evidence_paths=_promoted_evidence_paths(durable, project_root=destinations.project_root),
        evidence_state="promoted",
    )
    outputs, immutable = _promotion_outputs(
        bundle=bundle,
        rendered=rendered,
        current=destinations.current,
        archive_dir=destinations.archive_dir,
        project_root=destinations.project_root,
        payload=payload,
        provenance_payload=provenance_payload,
    )
    outputs[output] = rendered.encode("utf-8")
    _publish_performance_outputs(
        destinations.project_root,
        outputs,
        {artifacts.payload: payload, artifacts.provenance: provenance_payload},
        immutable,
    )
    return report_id


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


def _add_release_signal_subcommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Expose measurement and preflight modes of the shared curated plan."""
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
    release_signal_parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Execute every fixture and one operation per case without Criterion sampling or evidence writes",
    )
    _add_bench_timeout_arg(release_signal_parser)
    _add_project_root_arg(release_signal_parser)


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

    release_parser = subparsers.add_parser(
        "performance-release",
        help="Measure a release comparison with verified workload provenance",
    )
    release_parser.add_argument("current_tag", nargs="?", help="Current release tag")
    release_parser.add_argument("baseline_tag", nargs="?", help="Baseline release tag")
    release_parser.add_argument("--output", type=Path, default=PERFORMANCE_REPORT_SOURCE, help="Retained scratch Markdown report path")
    release_parser.add_argument("--current", type=Path, default=DOCS_PERFORMANCE_REPORT, help="Committed performance report path")
    release_parser.add_argument("--archive-dir", type=Path, default=PERFORMANCE_ARCHIVE_DIR, help="Archive directory for older reports")
    release_parser.add_argument("--worktree-ref", default="HEAD", help="Git ref for the current temp worktree (default: HEAD)")
    release_parser.add_argument("--no-apply-current-diff", action="store_true", help="Do not apply the current checkout diff to the temp worktree")
    _add_project_root_arg(release_parser)

    doc_parser = subparsers.add_parser("performance-doc", help="Promote performance docs from retained shared JSON evidence")
    doc_parser.add_argument("--output", type=Path, default=PERFORMANCE_REPORT_SOURCE, help="Scratch Markdown report path")
    doc_parser.add_argument(
        "--artifact-payload", type=Path, default=PERFORMANCE_REPORT_SOURCE.with_suffix(".comparison.json"), help="Retained performance payload path"
    )
    doc_parser.add_argument(
        "--artifact-provenance",
        type=Path,
        default=PERFORMANCE_REPORT_SOURCE.with_suffix(".evidence.json"),
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

    _add_bench_compare_subcommand(subparsers)
    _add_release_signal_subcommand(subparsers)
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
    print(format_exception_diagnostics(error), file=sys.stderr)
    sys.exit(1)


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
            preflight_only=args.preflight_only,
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
    print(f"Retained artifact bundle: {_artifact_paths_for_output(output).payload} and {_artifact_paths_for_output(output).provenance}")
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
    print(f"Retained artifact bundle: {_artifact_paths_for_output(output).payload} and {_artifact_paths_for_output(output).provenance}")
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
    print(f"Retained artifact bundle: {_artifact_paths_for_output(output).payload} and {_artifact_paths_for_output(output).provenance}")
    print(f"Current performance report: {report_id.current_tag} vs {report_id.baseline_tag}")
    print(f"Archive directory: {archive_dir}")
    sys.exit(0)


def _cmd_performance_doc(args: argparse.Namespace, project_root: Path) -> None:
    """Render and promote docs from retained artifacts only."""
    try:
        output = _path_from_root(project_root, args.output)
        artifacts = ArtifactPaths(
            payload=_path_from_root(project_root, args.artifact_payload),
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


def execute_command(args: argparse.Namespace, project_root: Path) -> None:
    """Execute the selected command based on parsed arguments."""
    handlers = {
        "bench-compare": _cmd_bench_compare,
        "generate-summary": execute_performance_summary_commands,
        "run-release-signal": execute_performance_summary_commands,
        "create-release-benchmark-metadata": execute_release_performance_commands,
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

    try:
        execute_command(args, project_root)
    except ExceptionGroup as error:
        print(f"benchmark-utils: {format_exception_diagnostics(error)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
