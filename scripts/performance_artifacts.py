"""Schema-versioned performance comparison CSV and provenance artifacts."""

import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

SCHEMA_VERSION = 3
SUITES = ("release-signal", "ci", "query", "predicates", "topology")
SCOPES = ("release-signal", "all-benches")
COVERAGE_STATES = ("comparable", "not-comparable", "current-only", "baseline-only")
MEASUREMENT_MODES = ("local-worktrees", "github-assets")
RELEASE_SIGNAL_TARGETS = (
    "ci_performance_suite",
    "circumsphere_containment",
    "cold_path_predicates",
    "locate",
    "realization_validation",
)

type CoverageState = Literal["comparable", "not-comparable", "current-only", "baseline-only"]
type MeasurementMode = Literal["local-worktrees", "github-assets"]
type HostStatus = Literal["recorded", "unavailable"]

CSV_COLUMNS = (
    "schema_version",
    "suite",
    "scope",
    "benchmark_id",
    "group",
    "benchmark",
    "coverage_status",
    "coverage_note",
    "baseline_median_ns",
    "baseline_ci_lower_ns",
    "baseline_ci_upper_ns",
    "baseline_confidence_level",
    "current_median_ns",
    "current_ci_lower_ns",
    "current_ci_upper_ns",
    "current_confidence_level",
)

_SEMVER_TAG_RE = re.compile(
    r"^v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
_GIT_OBJECT_ID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_GIT_REF_RE = re.compile(r"^(?:HEAD|[A-Za-z0-9][A-Za-z0-9._/+-]*)$")
_RUSTC_RE = re.compile(r"^rustc\s+[0-9]+\.[0-9]+\.[0-9]+(?:[-+][^\s]+)?(?:\s+.*)?$")
_VERSION_RE = re.compile(r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:[-+][0-9A-Za-z.-]+)?$")


def _require_non_empty(field: str, value: str) -> None:
    """Reject an empty invariant-bearing string."""
    if not value.strip():
        msg = f"{field} must not be empty"
        raise ValueError(msg)


def _require_markdown_text(field: str, value: str) -> None:
    """Reject text that can escape one Markdown heading or table cell."""
    _require_non_empty(field, value)
    if "|" in value or "`" in value or any(ord(char) < 32 or ord(char) == 127 for char in value):
        msg = f"{field} must be single-line Markdown-safe text"
        raise ValueError(msg)


def _require_release_tag(field: str, value: str) -> None:
    """Require one normalized semver release tag."""
    if _SEMVER_TAG_RE.fullmatch(value) is None:
        msg = f"{field} must be a normalized semver tag beginning with 'v': {value!r}"
        raise ValueError(msg)


def _require_timestamp(field: str, value: str) -> None:
    """Require a timezone-bearing ISO-8601 timestamp."""
    _require_markdown_text(field, value)
    if len(value) < 11 or value[10] != "T":
        msg = f"{field} must use the canonical 'T' date/time separator: {value!r}"
        raise ValueError(msg)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        msg = f"{field} must be an ISO-8601 timestamp: {value!r}"
        raise ValueError(msg) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        msg = f"{field} must include a timezone: {value!r}"
        raise ValueError(msg)


def _require_sha256(field: str, value: str) -> None:
    """Reject a value that is not a lowercase SHA-256 digest."""
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        msg = f"{field} must be a lowercase SHA-256 digest"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class TimingEstimate:
    """A finite positive median and its complete confidence interval."""

    median_ns: float
    ci_lower_ns: float
    ci_upper_ns: float
    confidence_level: float

    def __post_init__(self) -> None:
        """Reject non-finite, non-positive, or inconsistent timings."""
        for field, value in (
            ("median_ns", self.median_ns),
            ("ci_lower_ns", self.ci_lower_ns),
            ("ci_upper_ns", self.ci_upper_ns),
        ):
            if not math.isfinite(value) or value <= 0:
                msg = f"{field} must be finite and positive: {value!r}"
                raise ValueError(msg)
        if not self.ci_lower_ns <= self.median_ns <= self.ci_upper_ns:
            msg = f"confidence interval must contain the median: {self.ci_lower_ns} <= {self.median_ns} <= {self.ci_upper_ns}"
            raise ValueError(msg)
        if not math.isfinite(self.confidence_level) or not 0.0 < self.confidence_level < 1.0:
            msg = f"confidence_level must be finite and strictly between zero and one: {self.confidence_level!r}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class PerformanceRow:
    """One validated row in a performance comparison dataset."""

    suite: str
    scope: str
    benchmark_id: str
    group: str
    benchmark: str
    coverage_status: CoverageState
    coverage_note: str
    baseline: TimingEstimate | None
    current: TimingEstimate | None

    def __post_init__(self) -> None:
        """Preserve benchmark identity and coverage-state invariants."""
        if self.suite not in SUITES:
            msg = f"unsupported suite: {self.suite!r}"
            raise ValueError(msg)
        if self.scope not in SCOPES:
            msg = f"unsupported scope: {self.scope!r}"
            raise ValueError(msg)
        for field, value in (
            ("benchmark_id", self.benchmark_id),
            ("group", self.group),
            ("benchmark", self.benchmark),
        ):
            _require_markdown_text(field, value)
        expected_id = f"{self.group}/{self.benchmark}"
        if self.benchmark_id != expected_id:
            msg = f"benchmark_id must be {expected_id!r}, got {self.benchmark_id!r}"
            raise ValueError(msg)
        _validate_row_coverage(self.coverage_status, self.coverage_note, self.baseline, self.current)


def _validate_row_coverage(
    coverage_status: CoverageState,
    coverage_note: str,
    baseline: TimingEstimate | None,
    current: TimingEstimate | None,
) -> None:
    """Validate timing presence, notes, and confidence-level compatibility."""
    if coverage_status not in COVERAGE_STATES:
        msg = f"unsupported coverage status: {coverage_status!r}"
        raise ValueError(msg)
    expected_presence = {
        "comparable": (True, True),
        "not-comparable": (True, True),
        "current-only": (False, True),
        "baseline-only": (True, False),
    }[coverage_status]
    observed_presence = (baseline is not None, current is not None)
    if observed_presence != expected_presence:
        msg = f"coverage status {coverage_status!r} requires baseline/current presence {expected_presence}, got {observed_presence}"
        raise ValueError(msg)
    if coverage_status == "comparable" and coverage_note:
        msg = "comparable rows must not contain a coverage note"
        raise ValueError(msg)
    if coverage_status != "comparable" and not coverage_note.strip():
        msg = f"{coverage_status} rows require a coverage note"
        raise ValueError(msg)
    if coverage_note:
        _require_markdown_text("coverage_note", coverage_note)
    if coverage_status == "comparable" and baseline is not None and current is not None and baseline.confidence_level != current.confidence_level:
        msg = "comparable rows require equal confidence levels"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ReleasePair:
    """The current and baseline package identifiers represented by a report."""

    current: str
    baseline: str

    def __post_init__(self) -> None:
        """Require non-empty identifiers while allowing local same-version runs."""
        _require_release_tag("current release", self.current)
        _require_release_tag("baseline release", self.baseline)


@dataclass(frozen=True, slots=True)
class SourceState:
    """One measured source revision and its complete tracked-file state."""

    version: str
    commit: str
    ref: str
    revision_timestamp: str
    git_clean: bool | None
    source_state_sha256: str | None
    limitation: str = ""

    def __post_init__(self) -> None:
        """Reject incomplete or weakly identified source state."""
        _require_release_tag("source version", self.version)
        if _GIT_OBJECT_ID_RE.fullmatch(self.commit) is None:
            msg = f"source commit must be a full lowercase Git object ID: {self.commit!r}"
            raise ValueError(msg)
        if _GIT_REF_RE.fullmatch(self.ref) is None or ".." in self.ref or "@{" in self.ref or self.ref.endswith(("/", ".")):
            msg = f"source ref is not a supported Git ref: {self.ref!r}"
            raise ValueError(msg)
        _require_timestamp("source revision_timestamp", self.revision_timestamp)
        if self.limitation:
            _require_markdown_text("source limitation", self.limitation)
            if self.git_clean is not None or self.source_state_sha256 is not None:
                msg = "limited source evidence must leave clean state and source-state digest unavailable"
                raise ValueError(msg)
        else:
            if not isinstance(self.git_clean, bool) or self.source_state_sha256 is None:
                msg = "complete source evidence requires clean state and source-state digest"
                raise ValueError(msg)
            _require_sha256("source_state_sha256", self.source_state_sha256)


@dataclass(frozen=True, slots=True)
class ToolchainState:
    """Toolchain and harness identity for one measured source state."""

    rustc: str | None
    criterion_version: str | None
    cargo_profile: str
    cargo_lock_sha256: str | None
    harness_sha256: str | None
    configuration_sha256: str | None
    measurement_plan_sha256: str | None
    limitation: str = ""

    def __post_init__(self) -> None:
        """Reject incomplete toolchain or configuration identity."""
        if self.cargo_profile != "perf":
            msg = f"toolchain cargo_profile must be 'perf': {self.cargo_profile!r}"
            raise ValueError(msg)
        if self.limitation:
            _require_markdown_text("toolchain limitation", self.limitation)
            if any(
                value is not None
                for value in (
                    self.rustc,
                    self.criterion_version,
                    self.cargo_lock_sha256,
                    self.harness_sha256,
                    self.configuration_sha256,
                    self.measurement_plan_sha256,
                )
            ):
                msg = "limited toolchain evidence must leave unavailable fields null"
                raise ValueError(msg)
            return
        if self.rustc is None or _RUSTC_RE.fullmatch(self.rustc) is None:
            msg = f"toolchain rustc is not a supported rustc version string: {self.rustc!r}"
            raise ValueError(msg)
        _require_markdown_text("toolchain rustc", self.rustc)
        if self.criterion_version is None or _VERSION_RE.fullmatch(self.criterion_version) is None:
            msg = f"toolchain criterion_version is not a semantic version: {self.criterion_version!r}"
            raise ValueError(msg)
        for field, value in (
            ("cargo_lock_sha256", self.cargo_lock_sha256),
            ("harness_sha256", self.harness_sha256),
            ("configuration_sha256", self.configuration_sha256),
            ("measurement_plan_sha256", self.measurement_plan_sha256),
        ):
            if value is None:
                msg = f"complete toolchain evidence requires {field}"
                raise ValueError(msg)
            _require_sha256(field, value)


@dataclass(frozen=True, slots=True)
class HostIdentity:
    """Recorded host identity or an explicit historical-data limitation."""

    status: HostStatus
    cpu: str
    operating_system: str
    architecture: str
    reason: str = ""

    def __post_init__(self) -> None:
        """Require complete recorded identity or an explicit unavailable reason."""
        if self.status == "recorded":
            for field, value in (
                ("cpu", self.cpu),
                ("operating_system", self.operating_system),
                ("architecture", self.architecture),
            ):
                _require_markdown_text(f"host {field}", value)
                if value.strip().lower() in {"unknown", "unavailable", "n/a"}:
                    msg = f"recorded host {field} must not use a placeholder value"
                    raise ValueError(msg)
            if self.reason:
                msg = "recorded host identity must not contain an unavailable reason"
                raise ValueError(msg)
        elif self.status == "unavailable":
            _require_markdown_text("unavailable host reason", self.reason)
            for field, value in (
                ("cpu", self.cpu),
                ("operating_system", self.operating_system),
                ("architecture", self.architecture),
            ):
                if value:
                    _require_markdown_text(f"unavailable host {field}", value)
        else:
            msg = f"unsupported host status: {self.status!r}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class MeasurementArtifact:
    """Identity of the Criterion sample tree that supplied one revision."""

    origin: Literal["local-run", "release-archive"]
    content_sha256: str
    sample_name: str
    archive_sha256: str | None = None

    def __post_init__(self) -> None:
        """Require a content-bound sample and archive digest when applicable."""
        if self.origin not in ("local-run", "release-archive"):
            msg = f"unsupported measurement artifact origin: {self.origin!r}"
            raise ValueError(msg)
        _require_sha256("measurement artifact content_sha256", self.content_sha256)
        _require_markdown_text("measurement artifact sample_name", self.sample_name)
        if self.origin == "release-archive":
            if self.archive_sha256 is None:
                msg = "release-archive measurements require archive_sha256"
                raise ValueError(msg)
            _require_sha256("measurement artifact archive_sha256", self.archive_sha256)
        elif self.archive_sha256 is not None:
            msg = "local-run measurements must not contain archive_sha256"
            raise ValueError(msg)


def _validate_commands(label: str, commands: tuple[tuple[str, ...], ...]) -> None:
    """Require at least one non-empty argument-vector command."""
    if not commands:
        msg = f"{label} commands must not be empty"
        raise ValueError(msg)
    for command in commands:
        if not command or any(not part for part in command):
            msg = f"{label} commands must contain non-empty argument vectors"
            raise ValueError(msg)
        for part in command:
            _require_markdown_text(f"{label} command argument", part)


def _validate_optional_commands(label: str, commands: tuple[tuple[str, ...], ...]) -> None:
    """Validate zero or more argument-vector commands."""
    for command in commands:
        if not command or any(not part for part in command):
            msg = f"{label} commands must contain non-empty argument vectors"
            raise ValueError(msg)
        for part in command:
            _require_markdown_text(f"{label} command argument", part)


@dataclass(frozen=True, slots=True)
class ArtifactContext:
    """Validated non-tabular metadata for a performance dataset."""

    release: ReleasePair
    statistic: Literal["median"]
    suite: str
    scope: str
    measurement_mode: MeasurementMode
    current_source: SourceState
    baseline_source: SourceState
    current_commands: tuple[tuple[str, ...], ...]
    baseline_commands: tuple[tuple[str, ...], ...]
    current_completed_targets: tuple[str, ...]
    baseline_completed_targets: tuple[str, ...]
    current_acquisition_commands: tuple[tuple[str, ...], ...]
    baseline_acquisition_commands: tuple[tuple[str, ...], ...]
    current_toolchain: ToolchainState
    baseline_toolchain: ToolchainState
    current_measurement_host: HostIdentity
    baseline_measurement_host: HostIdentity
    current_artifact: MeasurementArtifact
    baseline_artifact: MeasurementArtifact
    publication_host: HostIdentity

    def __post_init__(self) -> None:
        """Bind selection, sources, commands, toolchains, and hosts together."""
        _validate_context_selection(self.statistic, self.suite, self.scope, self.measurement_mode)
        if self.current_source.version != self.release.current:
            msg = "current source version does not match release.current"
            raise ValueError(msg)
        if self.baseline_source.version != self.release.baseline:
            msg = "baseline source version does not match release.baseline"
            raise ValueError(msg)
        _validate_optional_commands("current", self.current_commands)
        _validate_optional_commands("baseline", self.baseline_commands)
        for label, targets in (
            ("current", self.current_completed_targets),
            ("baseline", self.baseline_completed_targets),
        ):
            if len(set(targets)) != len(targets):
                msg = f"{label} completed benchmark targets must be unique"
                raise ValueError(msg)
            for target in targets:
                _require_markdown_text(f"{label} completed benchmark target", target)
        _validate_optional_commands("current acquisition", self.current_acquisition_commands)
        _validate_optional_commands("baseline acquisition", self.baseline_acquisition_commands)
        _validate_context_mode(self)
        if self.publication_host.status != "recorded":
            msg = "artifact publication requires a recorded publication host"
            raise ValueError(msg)

    @property
    def comparison_blockers(self) -> tuple[str, ...]:
        """Return provenance differences that prevent before/after ratios."""
        blockers: list[str] = []
        if self.current_source.limitation or self.baseline_source.limitation:
            blockers.append("complete source-state evidence is unavailable")
        if self.current_toolchain.limitation or self.baseline_toolchain.limitation:
            blockers.append("complete toolchain evidence is unavailable")
        if not self.current_commands or not self.baseline_commands:
            blockers.append("measurement command evidence is unavailable")
        if self.measurement_mode == "github-assets":
            blockers.append("release archives were measured in separate sessions")
        if self.current_measurement_host.status != "recorded" or self.baseline_measurement_host.status != "recorded":
            blockers.append("measurement host identity is unavailable")
        elif self.current_measurement_host != self.baseline_measurement_host:
            blockers.append("measurement hosts differ")
        toolchain_fields = (
            ("rustc", self.current_toolchain.rustc, self.baseline_toolchain.rustc),
            ("Criterion version", self.current_toolchain.criterion_version, self.baseline_toolchain.criterion_version),
            ("Cargo profile", self.current_toolchain.cargo_profile, self.baseline_toolchain.cargo_profile),
            ("benchmark harness", self.current_toolchain.harness_sha256, self.baseline_toolchain.harness_sha256),
            ("measurement plan", self.current_toolchain.measurement_plan_sha256, self.baseline_toolchain.measurement_plan_sha256),
        )
        blockers.extend(f"{label} differs" for label, current, baseline in toolchain_fields if current != baseline)
        blockers.extend(self.target_transition_blockers)
        return tuple(blockers)

    @property
    def shared_completed_targets(self) -> tuple[str, ...]:
        """Return the canonical release targets completed by both revisions."""
        current = set(self.current_completed_targets)
        baseline = set(self.baseline_completed_targets)
        return tuple(target for target in RELEASE_SIGNAL_TARGETS if target in current and target in baseline)

    @property
    def target_transition_blockers(self) -> tuple[str, ...]:
        """Reject target drift unless it is a complete, ordered release-plan transition."""
        current = self.current_completed_targets
        baseline = self.baseline_completed_targets
        if current == baseline:
            return ()
        if self.suite != "release-signal":
            return ("completed benchmark targets differ",)

        canonical = set(RELEASE_SIGNAL_TARGETS)
        blockers: list[str] = []
        unknown = sorted((set(current) | set(baseline)) - canonical)
        if unknown:
            blockers.append("completed benchmark targets contain unsupported targets: " + ", ".join(unknown))
        for label, targets in (("current", current), ("baseline", baseline)):
            ordered = tuple(target for target in RELEASE_SIGNAL_TARGETS if target in set(targets))
            if targets != ordered:
                blockers.append(f"{label} completed benchmark targets are not in canonical release-plan order")
        if not self.shared_completed_targets:
            blockers.append("release target transition has no shared completed targets")
        if set(current) | set(baseline) != canonical:
            blockers.append("release target transition does not cover the canonical release-signal plan")
        return tuple(blockers)


def _validate_context_selection(statistic: str, suite: str, scope: str, measurement_mode: str) -> None:
    """Validate the finite supported report-selection vocabulary."""
    if statistic != "median":
        msg = f"unsupported performance statistic: {statistic!r}"
        raise ValueError(msg)
    if suite not in SUITES:
        msg = f"unsupported suite: {suite!r}"
        raise ValueError(msg)
    if scope not in SCOPES:
        msg = f"unsupported scope: {scope!r}"
        raise ValueError(msg)
    if measurement_mode not in MEASUREMENT_MODES:
        msg = f"unsupported measurement mode: {measurement_mode!r}"
        raise ValueError(msg)


def _validate_context_mode(context: ArtifactContext) -> None:
    """Validate mode-specific host, acquisition, and artifact contracts."""
    if context.measurement_mode == "local-worktrees":
        if context.current_measurement_host.status != "recorded" or context.baseline_measurement_host.status != "recorded":
            msg = "local-worktree measurements require recorded hosts for both revisions"
            raise ValueError(msg)
        if context.current_acquisition_commands or context.baseline_acquisition_commands:
            msg = "local-worktree measurements must not contain acquisition commands"
            raise ValueError(msg)
        if context.current_artifact.origin != "local-run" or context.baseline_artifact.origin != "local-run":
            msg = "local-worktree measurements require local-run artifacts"
            raise ValueError(msg)
        if context.current_source.limitation or context.baseline_source.limitation:
            msg = "local-worktree measurements require complete source evidence"
            raise ValueError(msg)
        if context.current_toolchain.limitation or context.baseline_toolchain.limitation:
            msg = "local-worktree measurements require complete toolchain evidence"
            raise ValueError(msg)
        _validate_commands("current", context.current_commands)
        _validate_commands("baseline", context.baseline_commands)
    else:
        if not context.current_acquisition_commands or not context.baseline_acquisition_commands:
            msg = "GitHub asset measurements require acquisition commands for both revisions"
            raise ValueError(msg)
        if context.current_artifact.origin != "release-archive" or context.baseline_artifact.origin != "release-archive":
            msg = "GitHub asset measurements require release-archive artifacts"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class PerformanceBundle:
    """A complete validated performance dataset and its provenance."""

    context: ArtifactContext
    rows: tuple[PerformanceRow, ...]

    def __post_init__(self) -> None:
        """Require non-empty unique rows bound to one selection."""
        if not self.rows:
            msg = "performance dataset must contain at least one row"
            raise ValueError(msg)
        keys: set[str] = set()
        for row in self.rows:
            if row.suite != self.context.suite:
                msg = f"row suite {row.suite!r} does not match report suite {self.context.suite!r}"
                raise ValueError(msg)
            if row.scope != self.context.scope:
                msg = f"row scope {row.scope!r} does not match report scope {self.context.scope!r}"
                raise ValueError(msg)
            if row.benchmark_id in keys:
                msg = f"duplicate benchmark_id: {row.benchmark_id!r}"
                raise ValueError(msg)
            keys.add(row.benchmark_id)
            if row.coverage_status == "comparable" and self.context.comparison_blockers:
                msg = "comparable row is not supported by compatible measurement provenance"
                raise ValueError(msg)

    @property
    def sorted_rows(self) -> tuple[PerformanceRow, ...]:
        """Return rows in deterministic group and benchmark order."""
        return tuple(sorted(self.rows, key=lambda row: (row.group, row.benchmark)))

    @property
    def promotion_blockers(self) -> tuple[str, ...]:
        """Return reasons this bundle cannot become curated release evidence."""
        blockers = list(self.context.comparison_blockers)
        comparable = tuple(row for row in self.rows if row.coverage_status == "comparable")
        if not comparable:
            blockers.append("no scientifically comparable rows")
        target_transition = self.context.current_completed_targets != self.context.baseline_completed_targets
        incomplete = tuple(
            row.benchmark_id for row in self.rows if row.coverage_status == "not-comparable" or (not target_transition and row.coverage_status != "comparable")
        )
        if incomplete:
            blockers.append("release-signal coverage is incomplete or non-comparable: " + ", ".join(sorted(incomplete)))
        if self.context.suite != "release-signal" or self.context.scope != "release-signal":
            blockers.append("curated promotion requires the release-signal suite and scope")
        if not target_transition:
            missing_current = sorted(set(RELEASE_SIGNAL_TARGETS) - set(self.context.current_completed_targets))
            missing_baseline = sorted(set(RELEASE_SIGNAL_TARGETS) - set(self.context.baseline_completed_targets))
            if missing_current:
                blockers.append("current measurement did not complete required targets: " + ", ".join(missing_current))
            if missing_baseline:
                blockers.append("baseline measurement did not complete required targets: " + ", ".join(missing_baseline))
        if self.context.release.current == self.context.release.baseline:
            blockers.append("current and baseline releases are identical")
        return tuple(dict.fromkeys(blockers))

    def require_promotable(self) -> None:
        """Reject bundles that cannot support a curated release comparison."""
        blockers = self.promotion_blockers
        if blockers:
            msg = "performance bundle is not promotable: " + "; ".join(blockers)
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    """Adjacent CSV and JSON provenance destinations."""

    csv: Path
    provenance: Path

    def __post_init__(self) -> None:
        """Require distinct adjacent artifact paths."""
        ensure_distinct_paths({"CSV": self.csv, "provenance": self.provenance})
        if self.csv.resolve(strict=False).parent != self.provenance.resolve(strict=False).parent:
            msg = "CSV and provenance sidecar must be adjacent"
            raise ValueError(msg)


def _paths_alias(first: Path, second: Path) -> bool:
    """Return whether two paths resolve to the same filesystem target."""
    if first.exists() and second.exists() and first.samefile(second):
        return True
    return first.resolve(strict=False) == second.resolve(strict=False)


def ensure_distinct_paths(paths: Mapping[str, Path]) -> None:
    """Reject lexical, symlink, or existing-file aliases among named paths."""
    items = tuple(paths.items())
    for index, (first_name, first_path) in enumerate(items):
        for second_name, second_path in items[index + 1 :]:
            if _paths_alias(first_path, second_path):
                msg = f"{first_name} and {second_name} must use distinct paths: {first_path}"
                raise ValueError(msg)


def _timing_fields(prefix: str, estimate: TimingEstimate | None) -> dict[str, str]:
    """Serialize optional timing values into one CSV row fragment."""
    if estimate is None:
        return {
            f"{prefix}_median_ns": "",
            f"{prefix}_ci_lower_ns": "",
            f"{prefix}_ci_upper_ns": "",
            f"{prefix}_confidence_level": "",
        }
    return {
        f"{prefix}_median_ns": format(estimate.median_ns, ".17g"),
        f"{prefix}_ci_lower_ns": format(estimate.ci_lower_ns, ".17g"),
        f"{prefix}_ci_upper_ns": format(estimate.ci_upper_ns, ".17g"),
        f"{prefix}_confidence_level": format(estimate.confidence_level, ".17g"),
    }


def _row_to_csv(row: PerformanceRow) -> dict[str, str]:
    """Serialize one trusted row into the versioned CSV shape."""
    return {
        "schema_version": str(SCHEMA_VERSION),
        "suite": row.suite,
        "scope": row.scope,
        "benchmark_id": row.benchmark_id,
        "group": row.group,
        "benchmark": row.benchmark,
        "coverage_status": row.coverage_status,
        "coverage_note": row.coverage_note,
        **_timing_fields("baseline", row.baseline),
        **_timing_fields("current", row.current),
    }


def _serialize_csv(bundle: PerformanceBundle) -> bytes:
    """Serialize deterministic RFC 4180-style UTF-8 CSV bytes."""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(_row_to_csv(row) for row in bundle.sorted_rows)
    return output.getvalue().encode("utf-8")


def _source_payload(source: SourceState) -> dict[str, object]:
    """Convert trusted source metadata to its JSON transport shape."""
    return {
        "version": source.version,
        "commit": source.commit,
        "ref": source.ref,
        "revision_timestamp": source.revision_timestamp,
        "git_clean": source.git_clean,
        "source_state_sha256": source.source_state_sha256,
        "limitation": source.limitation,
    }


def _toolchain_payload(toolchain: ToolchainState) -> dict[str, str | None]:
    """Convert trusted toolchain metadata to its JSON transport shape."""
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


def _host_payload(host: HostIdentity) -> dict[str, str]:
    """Convert trusted host metadata to its JSON transport shape."""
    return {
        "status": host.status,
        "cpu": host.cpu,
        "operating_system": host.operating_system,
        "architecture": host.architecture,
        "reason": host.reason,
    }


def _artifact_payload(artifact: MeasurementArtifact) -> dict[str, str | None]:
    """Convert a trusted measurement-artifact identity to JSON."""
    return {
        "origin": artifact.origin,
        "content_sha256": artifact.content_sha256,
        "sample_name": artifact.sample_name,
        "archive_sha256": artifact.archive_sha256,
    }


def _side_payload(context: ArtifactContext, side: Literal["current", "baseline"]) -> dict[str, object]:
    """Convert one complete revision's evidence to JSON."""
    source = context.current_source if side == "current" else context.baseline_source
    measurement_commands = context.current_commands if side == "current" else context.baseline_commands
    completed_targets = context.current_completed_targets if side == "current" else context.baseline_completed_targets
    acquisition_commands = context.current_acquisition_commands if side == "current" else context.baseline_acquisition_commands
    toolchain = context.current_toolchain if side == "current" else context.baseline_toolchain
    measurement_host = context.current_measurement_host if side == "current" else context.baseline_measurement_host
    artifact = context.current_artifact if side == "current" else context.baseline_artifact
    return {
        "source": _source_payload(source),
        "measurement_commands": [list(command) for command in measurement_commands],
        "completed_targets": list(completed_targets),
        "acquisition_commands": [list(command) for command in acquisition_commands],
        "toolchain": _toolchain_payload(toolchain),
        "measurement_host": _host_payload(measurement_host),
        "artifact": _artifact_payload(artifact),
    }


def _serialize_provenance(bundle: PerformanceBundle, csv_payload: bytes) -> bytes:
    """Serialize deterministic provenance bound to the exact CSV payload."""
    context = bundle.context
    payload = {
        "schema_version": SCHEMA_VERSION,
        "csv_sha256": hashlib.sha256(csv_payload).hexdigest(),
        "csv_row_count": len(bundle.rows),
        "csv_columns": list(CSV_COLUMNS),
        "release": {"current": context.release.current, "baseline": context.release.baseline},
        "selection": {"statistic": context.statistic, "suite": context.suite, "scope": context.scope},
        "measurement_mode": context.measurement_mode,
        "current": _side_payload(context, "current"),
        "baseline": _side_payload(context, "baseline"),
        "publication_host": _host_payload(context.publication_host),
    }
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def serialize_bundle(bundle: PerformanceBundle) -> tuple[bytes, bytes]:
    """Return deterministic CSV and provenance bytes for a trusted bundle."""
    csv_payload = _serialize_csv(bundle)
    return csv_payload, _serialize_provenance(bundle, csv_payload)


def _require_exact_keys(data: Mapping[str, object], expected: frozenset[str], *, source: str) -> None:
    """Reject missing or unsupported object fields."""
    observed = frozenset(data)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        msg = f"{source} fields do not match schema; missing={missing}, unknown={unknown}"
        raise ValueError(msg)


def _required_object(data: Mapping[str, object], field: str, *, source: str) -> dict[str, object]:
    """Parse a required JSON object field."""
    value = data.get(field)
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        msg = f"{source}.{field} must be an object"
        raise TypeError(msg)
    return cast("dict[str, object]", value)


def _required_string(data: Mapping[str, object], field: str, *, source: str) -> str:
    """Parse a required non-empty JSON string field."""
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        msg = f"{source}.{field} must be a non-empty string"
        raise ValueError(msg)
    return value


def _parse_source(data: Mapping[str, object], *, source: str) -> SourceState:
    """Parse one raw source-state object into a trusted value."""
    _require_exact_keys(
        data,
        frozenset({"version", "commit", "ref", "revision_timestamp", "git_clean", "source_state_sha256", "limitation"}),
        source=source,
    )
    git_clean = data.get("git_clean")
    if git_clean is not None and not isinstance(git_clean, bool):
        msg = f"{source}.git_clean must be a boolean or null"
        raise TypeError(msg)
    source_state_sha256 = data.get("source_state_sha256")
    if source_state_sha256 is not None and not isinstance(source_state_sha256, str):
        msg = f"{source}.source_state_sha256 must be a string or null"
        raise TypeError(msg)
    return SourceState(
        version=_required_string(data, "version", source=source),
        commit=_required_string(data, "commit", source=source),
        ref=_required_string(data, "ref", source=source),
        revision_timestamp=_required_string(data, "revision_timestamp", source=source),
        git_clean=git_clean,
        source_state_sha256=source_state_sha256,
        limitation=_required_string(data, "limitation", source=source) if data.get("limitation") else "",
    )


def _parse_toolchain(data: Mapping[str, object], *, source: str) -> ToolchainState:
    """Parse one raw toolchain object into a trusted value."""
    _require_exact_keys(
        data,
        frozenset(
            {
                "rustc",
                "criterion_version",
                "cargo_profile",
                "cargo_lock_sha256",
                "harness_sha256",
                "configuration_sha256",
                "measurement_plan_sha256",
                "limitation",
            }
        ),
        source=source,
    )
    optional_fields: dict[str, str | None] = {}
    for field in (
        "rustc",
        "criterion_version",
        "cargo_lock_sha256",
        "harness_sha256",
        "configuration_sha256",
        "measurement_plan_sha256",
    ):
        value = data.get(field)
        if value is not None and not isinstance(value, str):
            msg = f"{source}.{field} must be a string or null"
            raise TypeError(msg)
        optional_fields[field] = value
    return ToolchainState(
        rustc=optional_fields["rustc"],
        criterion_version=optional_fields["criterion_version"],
        cargo_profile=_required_string(data, "cargo_profile", source=source),
        cargo_lock_sha256=optional_fields["cargo_lock_sha256"],
        harness_sha256=optional_fields["harness_sha256"],
        configuration_sha256=optional_fields["configuration_sha256"],
        measurement_plan_sha256=optional_fields["measurement_plan_sha256"],
        limitation=_required_string(data, "limitation", source=source) if data.get("limitation") else "",
    )


def _parse_host(data: Mapping[str, object], *, source: str) -> HostIdentity:
    """Parse one raw host object into a trusted value."""
    _require_exact_keys(data, frozenset({"status", "cpu", "operating_system", "architecture", "reason"}), source=source)
    status = _required_string(data, "status", source=source)
    if status not in ("recorded", "unavailable"):
        msg = f"{source}.status has unsupported value {status!r}"
        raise ValueError(msg)
    for field in ("cpu", "operating_system", "architecture", "reason"):
        if not isinstance(data.get(field), str):
            msg = f"{source}.{field} must be a string"
            raise TypeError(msg)
    return HostIdentity(
        status=status,
        cpu=cast("str", data["cpu"]),
        operating_system=cast("str", data["operating_system"]),
        architecture=cast("str", data["architecture"]),
        reason=cast("str", data["reason"]),
    )


def _parse_artifact(data: Mapping[str, object], *, source: str) -> MeasurementArtifact:
    """Parse one measurement-artifact identity."""
    _require_exact_keys(data, frozenset({"origin", "content_sha256", "sample_name", "archive_sha256"}), source=source)
    origin = _required_string(data, "origin", source=source)
    if origin not in ("local-run", "release-archive"):
        msg = f"{source}.origin has unsupported value {origin!r}"
        raise ValueError(msg)
    archive_sha256 = data.get("archive_sha256")
    if archive_sha256 is not None and not isinstance(archive_sha256, str):
        msg = f"{source}.archive_sha256 must be a string or null"
        raise TypeError(msg)
    return MeasurementArtifact(
        origin=origin,
        content_sha256=_required_string(data, "content_sha256", source=source),
        sample_name=_required_string(data, "sample_name", source=source),
        archive_sha256=archive_sha256,
    )


def _parse_commands(value: object, *, source: str, allow_empty: bool = False) -> tuple[tuple[str, ...], ...]:
    """Parse a JSON array of argument-vector commands."""
    if not isinstance(value, list):
        msg = f"{source} must be an array"
        raise TypeError(msg)
    commands: list[tuple[str, ...]] = []
    for index, command in enumerate(value):
        if not isinstance(command, list) or not command or not all(isinstance(part, str) and part for part in command):
            msg = f"{source}[{index}] must be a non-empty string array"
            raise ValueError(msg)
        commands.append(tuple(cast("list[str]", command)))
    result = tuple(commands)
    if allow_empty:
        _validate_optional_commands(source, result)
    else:
        _validate_commands(source, result)
    return result


def _parse_side(
    data: Mapping[str, object], *, source: str
) -> tuple[
    SourceState,
    tuple[tuple[str, ...], ...],
    tuple[str, ...],
    tuple[tuple[str, ...], ...],
    ToolchainState,
    HostIdentity,
    MeasurementArtifact,
]:
    """Parse one current/baseline provenance side."""
    _require_exact_keys(
        data,
        frozenset(
            {
                "source",
                "measurement_commands",
                "completed_targets",
                "acquisition_commands",
                "toolchain",
                "measurement_host",
                "artifact",
            }
        ),
        source=source,
    )
    targets = data.get("completed_targets")
    if not isinstance(targets, list) or not all(isinstance(target, str) for target in targets):
        msg = f"{source}.completed_targets must be a string array"
        raise TypeError(msg)
    return (
        _parse_source(_required_object(data, "source", source=source), source=f"{source}.source"),
        _parse_commands(data.get("measurement_commands"), source=f"{source}.measurement_commands", allow_empty=True),
        tuple(cast("list[str]", targets)),
        _parse_commands(data.get("acquisition_commands"), source=f"{source}.acquisition_commands", allow_empty=True),
        _parse_toolchain(_required_object(data, "toolchain", source=source), source=f"{source}.toolchain"),
        _parse_host(_required_object(data, "measurement_host", source=source), source=f"{source}.measurement_host"),
        _parse_artifact(_required_object(data, "artifact", source=source), source=f"{source}.artifact"),
    )


def _parse_context(data: Mapping[str, object], *, source: str) -> tuple[ArtifactContext, str, int, tuple[str, ...]]:
    """Parse provenance JSON into trusted context plus CSV binding metadata."""
    expected = frozenset(
        {
            "schema_version",
            "csv_sha256",
            "csv_row_count",
            "csv_columns",
            "release",
            "selection",
            "measurement_mode",
            "current",
            "baseline",
            "publication_host",
        }
    )
    _require_exact_keys(data, expected, source=source)
    schema_version = data.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version != SCHEMA_VERSION:
        msg = f"unsupported provenance schema version: {data.get('schema_version')!r}"
        raise ValueError(msg)
    csv_sha256 = _required_string(data, "csv_sha256", source=source)
    _require_sha256("csv_sha256", csv_sha256)
    row_count = data.get("csv_row_count")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
        msg = f"{source}.csv_row_count must be a positive integer"
        raise ValueError(msg)
    columns = data.get("csv_columns")
    if not isinstance(columns, list) or not all(isinstance(column, str) for column in columns):
        msg = f"{source}.csv_columns must be a string array"
        raise TypeError(msg)

    release_data = _required_object(data, "release", source=source)
    _require_exact_keys(release_data, frozenset({"current", "baseline"}), source=f"{source}.release")
    selection = _required_object(data, "selection", source=source)
    _require_exact_keys(selection, frozenset({"statistic", "suite", "scope"}), source=f"{source}.selection")
    statistic = _required_string(selection, "statistic", source=f"{source}.selection")
    if statistic != "median":
        msg = f"unsupported performance statistic: {statistic!r}"
        raise ValueError(msg)
    mode = _required_string(data, "measurement_mode", source=source)
    if mode not in MEASUREMENT_MODES:
        msg = f"unsupported measurement mode: {mode!r}"
        raise ValueError(msg)

    current_source, current_commands, current_targets, current_acquisition_commands, current_toolchain, current_host, current_artifact = _parse_side(
        _required_object(data, "current", source=source),
        source=f"{source}.current",
    )
    baseline_source, baseline_commands, baseline_targets, baseline_acquisition_commands, baseline_toolchain, baseline_host, baseline_artifact = _parse_side(
        _required_object(data, "baseline", source=source),
        source=f"{source}.baseline",
    )
    context = ArtifactContext(
        release=ReleasePair(
            current=_required_string(release_data, "current", source=f"{source}.release"),
            baseline=_required_string(release_data, "baseline", source=f"{source}.release"),
        ),
        statistic="median",
        suite=_required_string(selection, "suite", source=f"{source}.selection"),
        scope=_required_string(selection, "scope", source=f"{source}.selection"),
        measurement_mode=mode,
        current_source=current_source,
        baseline_source=baseline_source,
        current_commands=current_commands,
        baseline_commands=baseline_commands,
        current_completed_targets=current_targets,
        baseline_completed_targets=baseline_targets,
        current_acquisition_commands=current_acquisition_commands,
        baseline_acquisition_commands=baseline_acquisition_commands,
        current_toolchain=current_toolchain,
        baseline_toolchain=baseline_toolchain,
        current_measurement_host=current_host,
        baseline_measurement_host=baseline_host,
        current_artifact=current_artifact,
        baseline_artifact=baseline_artifact,
        publication_host=_parse_host(_required_object(data, "publication_host", source=source), source=f"{source}.publication_host"),
    )
    return context, csv_sha256, row_count, tuple(cast("list[str]", columns))


def _parse_float(value: str, field: str, *, row_number: int, source: str) -> float:
    """Parse one required finite timing number from CSV."""
    if not value:
        msg = f"missing {field} at {source} row {row_number}"
        raise ValueError(msg)
    try:
        return float(value)
    except ValueError as exc:
        msg = f"invalid {field} at {source} row {row_number}: {value!r}"
        raise ValueError(msg) from exc


def _parse_timing(row: Mapping[str, str], prefix: str, *, row_number: int, source: str) -> TimingEstimate | None:
    """Parse an all-present or all-absent timing triple."""
    fields = (f"{prefix}_median_ns", f"{prefix}_ci_lower_ns", f"{prefix}_ci_upper_ns", f"{prefix}_confidence_level")
    values = tuple(row[field] for field in fields)
    if not any(values):
        return None
    if not all(values):
        msg = f"partial {prefix} timing at {source} row {row_number}"
        raise ValueError(msg)
    return TimingEstimate(
        median_ns=_parse_float(values[0], fields[0], row_number=row_number, source=source),
        ci_lower_ns=_parse_float(values[1], fields[1], row_number=row_number, source=source),
        ci_upper_ns=_parse_float(values[2], fields[2], row_number=row_number, source=source),
        confidence_level=_parse_float(values[3], fields[3], row_number=row_number, source=source),
    )


def _parse_rows(csv_payload: bytes, *, source: str) -> tuple[PerformanceRow, ...]:
    """Parse exact-schema CSV bytes into trusted rows."""
    try:
        text = csv_payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        msg = f"{source} CSV is not UTF-8"
        raise ValueError(msg) from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
        msg = f"unsupported CSV columns in {source}: {reader.fieldnames!r}"
        raise ValueError(msg)
    rows: list[PerformanceRow] = []
    for row_number, raw in enumerate(reader, start=2):
        if None in raw or any(value is None for value in raw.values()):
            msg = f"malformed CSV field count at {source} row {row_number}"
            raise ValueError(msg)
        row = cast("dict[str, str]", raw)
        if row["schema_version"] != str(SCHEMA_VERSION):
            msg = f"unsupported CSV schema version at {source} row {row_number}: {row['schema_version']!r}"
            raise ValueError(msg)
        coverage = row["coverage_status"]
        if coverage not in COVERAGE_STATES:
            msg = f"unsupported coverage status at {source} row {row_number}: {coverage!r}"
            raise ValueError(msg)
        rows.append(
            PerformanceRow(
                suite=row["suite"],
                scope=row["scope"],
                benchmark_id=row["benchmark_id"],
                group=row["group"],
                benchmark=row["benchmark"],
                coverage_status=coverage,
                coverage_note=row["coverage_note"],
                baseline=_parse_timing(row, "baseline", row_number=row_number, source=source),
                current=_parse_timing(row, "current", row_number=row_number, source=source),
            )
        )
    return tuple(rows)


def load_bundle_bytes(csv_payload: bytes, provenance_payload: bytes, *, source: str) -> PerformanceBundle:
    """Parse and cross-validate one in-memory CSV/provenance pair."""
    try:
        raw = json.loads(provenance_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        msg = f"malformed provenance JSON in {source}: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        msg = f"provenance in {source} must be a JSON object"
        raise TypeError(msg)
    context, expected_sha256, expected_count, expected_columns = _parse_context(cast("dict[str, object]", raw), source=source)
    if expected_columns != CSV_COLUMNS:
        msg = f"provenance CSV columns do not match schema in {source}"
        raise ValueError(msg)
    observed_sha256 = hashlib.sha256(csv_payload).hexdigest()
    if observed_sha256 != expected_sha256:
        msg = f"CSV SHA-256 does not match provenance in {source}"
        raise ValueError(msg)
    rows = _parse_rows(csv_payload, source=source)
    if len(rows) != expected_count:
        msg = f"CSV row count does not match provenance in {source}: {len(rows)} != {expected_count}"
        raise ValueError(msg)
    bundle = PerformanceBundle(context=context, rows=rows)
    canonical_csv, canonical_provenance = serialize_bundle(bundle)
    if csv_payload != canonical_csv:
        msg = f"CSV payload is not in canonical serialized form in {source}"
        raise ValueError(msg)
    if provenance_payload != canonical_provenance:
        msg = f"provenance payload is not in canonical serialized form in {source}"
        raise ValueError(msg)
    return bundle


def load_bundle(paths: ArtifactPaths) -> PerformanceBundle:
    """Load and validate an adjacent artifact pair from disk."""
    try:
        csv_payload = paths.csv.read_bytes()
        provenance_payload = paths.provenance.read_bytes()
    except OSError as exc:
        msg = f"could not read performance artifacts {paths.csv} and {paths.provenance}: {exc}"
        raise OSError(msg) from exc
    return load_bundle_bytes(csv_payload, provenance_payload, source=str(paths.csv.parent))


def _stage_payload(path: Path, payload: bytes) -> Path:
    """Write one durable same-directory temporary payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        return Path(handle.name)


def _restore_path(path: Path, payload: bytes | None) -> None:
    """Restore one exact prior payload or prior absence."""
    if payload is None:
        path.unlink(missing_ok=True)
        return
    staged = _stage_payload(path, payload)
    try:
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


@contextmanager
def publish_bundle(paths: ArtifactPaths, bundle: PerformanceBundle) -> Iterator[None]:
    """Publish, reload-validate, and roll back a bundle if its consumer fails."""
    csv_payload, provenance_payload = serialize_bundle(bundle)
    prior_csv = paths.csv.read_bytes() if paths.csv.exists() else None
    prior_provenance = paths.provenance.read_bytes() if paths.provenance.exists() else None
    staged_csv = _stage_payload(paths.csv, csv_payload)
    staged_provenance = _stage_payload(paths.provenance, provenance_payload)
    try:
        staged_csv.replace(paths.csv)
        staged_provenance.replace(paths.provenance)
        if load_bundle(paths) != PerformanceBundle(context=bundle.context, rows=bundle.sorted_rows):
            msg = "reloaded performance bundle does not match published bundle"
            raise ValueError(msg)
        try:
            yield
        except BaseException:
            _restore_path(paths.csv, prior_csv)
            _restore_path(paths.provenance, prior_provenance)
            raise
    except BaseException:
        observed_csv = paths.csv.read_bytes() if paths.csv.exists() else None
        observed_provenance = paths.provenance.read_bytes() if paths.provenance.exists() else None
        if observed_csv != prior_csv:
            _restore_path(paths.csv, prior_csv)
        if observed_provenance != prior_provenance:
            _restore_path(paths.provenance, prior_provenance)
        raise
    finally:
        staged_csv.unlink(missing_ok=True)
        staged_provenance.unlink(missing_ok=True)


def write_bundle(paths: ArtifactPaths, bundle: PerformanceBundle) -> None:
    """Publish one validated bundle transactionally."""
    with publish_bundle(paths, bundle):
        pass
