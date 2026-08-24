"""Tests for retained performance CSV and provenance artifacts."""

import csv
import hashlib
import io
import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from performance_artifacts import (
    CSV_COLUMNS,
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
    load_bundle,
    load_bundle_bytes,
    publish_bundle,
    serialize_bundle,
    write_bundle,
)

if TYPE_CHECKING:
    from pathlib import Path

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
RELEASE_TARGETS = (
    "ci_performance_suite",
    "circumsphere_containment",
    "cold_path_predicates",
    "locate",
    "realization_validation",
)


def source_state(version: str, *, current: bool) -> SourceState:
    return SourceState(
        version=version,
        commit=("a" if current else "b") * 40,
        ref="HEAD" if current else version,
        revision_timestamp="2026-08-23T12:00:00-07:00",
        git_clean=not current,
        source_state_sha256=SHA_A if current else SHA_B,
    )


def toolchain_state() -> ToolchainState:
    return ToolchainState(
        rustc="rustc 1.98.0",
        criterion_version="0.7.0",
        cargo_profile="perf",
        cargo_lock_sha256=SHA_B,
        harness_sha256=SHA_C,
        configuration_sha256=SHA_A,
        measurement_plan_sha256=SHA_B,
    )


def context(*, current: str = "v0.8.0", baseline: str = "v0.7.8") -> ArtifactContext:
    host = HostIdentity(status="recorded", cpu="Test CPU", operating_system="Test OS", architecture="test-arch")
    return ArtifactContext(
        release=ReleasePair(current=current, baseline=baseline),
        statistic="median",
        suite="release-signal",
        scope="release-signal",
        measurement_mode="local-worktrees",
        current_source=source_state(current, current=True),
        baseline_source=source_state(baseline, current=False),
        current_commands=(("just", "bench-latest"),),
        baseline_commands=(("cargo", "bench", "--save-baseline", baseline),),
        current_completed_targets=RELEASE_TARGETS,
        baseline_completed_targets=RELEASE_TARGETS,
        current_acquisition_commands=(),
        baseline_acquisition_commands=(),
        current_toolchain=toolchain_state(),
        baseline_toolchain=toolchain_state(),
        current_measurement_host=host,
        baseline_measurement_host=host,
        current_artifact=MeasurementArtifact(origin="local-run", content_sha256=SHA_B, sample_name="new"),
        baseline_artifact=MeasurementArtifact(origin="local-run", content_sha256=SHA_C, sample_name=baseline),
        publication_host=host,
    )


def estimate(value: float) -> TimingEstimate:
    return TimingEstimate(median_ns=value, ci_lower_ns=value * 0.9, ci_upper_ns=value * 1.1, confidence_level=0.95)


def bundle(*, current: str = "v0.8.0", baseline: str = "v0.7.8") -> PerformanceBundle:
    return PerformanceBundle(
        context=context(current=current, baseline=baseline),
        rows=(
            PerformanceRow(
                suite="release-signal",
                scope="release-signal",
                benchmark_id="validation/validate_3d/750",
                group="validation",
                benchmark="validate_3d/750",
                coverage_status="comparable",
                coverage_note="",
                baseline=estimate(2_000_000.0),
                current=estimate(1_000_000.0),
            ),
            PerformanceRow(
                suite="release-signal",
                scope="release-signal",
                benchmark_id="validation/new_case/750",
                group="validation",
                benchmark="new_case/750",
                coverage_status="current-only",
                coverage_note="No matching baseline sample was present.",
                baseline=None,
                current=estimate(500_000.0),
            ),
        ),
    )


def replace_csv_rows(csv_payload: bytes, transform) -> bytes:
    reader = csv.DictReader(io.StringIO(csv_payload.decode("utf-8"), newline=""))
    rows = [dict(row) for row in reader]
    transform(rows)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def bind_csv(provenance_payload: bytes, csv_payload: bytes, *, row_count: int | None = None) -> bytes:
    data = json.loads(provenance_payload)
    data["csv_sha256"] = hashlib.sha256(csv_payload).hexdigest()
    if row_count is not None:
        data["csv_row_count"] = row_count
    return (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8")


def test_artifact_round_trip_preserves_comparable_and_one_sided_rows() -> None:
    original = bundle()
    csv_payload, provenance_payload = serialize_bundle(original)

    parsed = load_bundle_bytes(csv_payload, provenance_payload, source="round-trip fixture")

    assert parsed == PerformanceBundle(context=original.context, rows=original.sorted_rows)
    assert csv_payload.startswith(",".join(CSV_COLUMNS).encode())
    assert csv_payload.endswith(b"\n")
    assert provenance_payload.endswith(b"\n")


def test_artifact_round_trip_allows_same_version_local_comparison() -> None:
    original = bundle(current="v0.8.0", baseline="v0.8.0")
    csv_payload, provenance_payload = serialize_bundle(original)

    parsed = load_bundle_bytes(csv_payload, provenance_payload, source="same-version fixture")

    assert parsed.context.release == ReleasePair(current="v0.8.0", baseline="v0.8.0")


def test_artifact_serialization_sorts_rows_deterministically() -> None:
    original = bundle()
    reordered = PerformanceBundle(context=original.context, rows=tuple(reversed(original.rows)))

    assert serialize_bundle(original) == serialize_bundle(reordered)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_timing_estimate_rejects_non_positive_or_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        TimingEstimate(median_ns=value, ci_lower_ns=1.0, ci_upper_ns=2.0, confidence_level=0.95)


def test_timing_estimate_accepts_ordered_interval_that_excludes_point_estimate() -> None:
    estimate = TimingEstimate(median_ns=3.0, ci_lower_ns=1.0, ci_upper_ns=2.0, confidence_level=0.95)

    assert estimate.median_ns == 3.0


def test_timing_estimate_rejects_reversed_interval() -> None:
    with pytest.raises(ValueError, match="confidence interval must be ordered"):
        TimingEstimate(median_ns=2.0, ci_lower_ns=3.0, ci_upper_ns=1.0, confidence_level=0.95)


def test_performance_row_rejects_coverage_presence_mismatch() -> None:
    with pytest.raises(ValueError, match="requires baseline/current presence"):
        PerformanceRow(
            suite="release-signal",
            scope="release-signal",
            benchmark_id="group/bench",
            group="group",
            benchmark="bench",
            coverage_status="comparable",
            coverage_note="",
            baseline=None,
            current=estimate(1.0),
        )


def test_loader_rejects_unsupported_provenance_schema() -> None:
    csv_payload, provenance_payload = serialize_bundle(bundle())
    data = json.loads(provenance_payload)
    data["schema_version"] = 4

    with pytest.raises(ValueError, match="unsupported provenance schema version"):
        load_bundle_bytes(csv_payload, json.dumps(data).encode(), source="future schema")


@pytest.mark.parametrize("schema_version", [True, 2.0, "2"])
def test_loader_rejects_non_integer_provenance_schema(schema_version: object) -> None:
    csv_payload, provenance_payload = serialize_bundle(bundle())
    data = json.loads(provenance_payload)
    data["schema_version"] = schema_version

    with pytest.raises(ValueError, match="unsupported provenance schema version"):
        load_bundle_bytes(csv_payload, json.dumps(data).encode(), source="invalid schema type")


def test_loader_rejects_missing_provenance_field() -> None:
    csv_payload, provenance_payload = serialize_bundle(bundle())
    data = json.loads(provenance_payload)
    del data["baseline"]["toolchain"]["harness_sha256"]

    with pytest.raises(ValueError, match="fields do not match schema"):
        load_bundle_bytes(csv_payload, json.dumps(data).encode(), source="incomplete provenance")


def test_loader_rejects_csv_digest_mismatch() -> None:
    csv_payload, provenance_payload = serialize_bundle(bundle())

    with pytest.raises(ValueError, match="CSV SHA-256 does not match"):
        load_bundle_bytes(csv_payload + b"\n", provenance_payload, source="digest mismatch")


def test_loader_rejects_unknown_coverage_state() -> None:
    csv_payload, provenance_payload = serialize_bundle(bundle())
    changed = replace_csv_rows(csv_payload, lambda rows: rows[0].__setitem__("coverage_status", "unknown"))

    with pytest.raises(ValueError, match="unsupported coverage status"):
        load_bundle_bytes(changed, bind_csv(provenance_payload, changed), source="unknown coverage")


def test_loader_rejects_partial_timing_triple() -> None:
    csv_payload, provenance_payload = serialize_bundle(bundle())

    def remove_comparable_bound(rows: list[dict[str, str]]) -> None:
        comparable = next(row for row in rows if row["coverage_status"] == "comparable")
        comparable["baseline_ci_upper_ns"] = ""

    changed = replace_csv_rows(csv_payload, remove_comparable_bound)

    with pytest.raises(ValueError, match="partial baseline timing"):
        load_bundle_bytes(changed, bind_csv(provenance_payload, changed), source="partial timing")


def test_loader_rejects_duplicate_benchmark_ids() -> None:
    csv_payload, provenance_payload = serialize_bundle(bundle())

    def duplicate_first(rows: list[dict[str, str]]) -> None:
        rows.append(dict(rows[0]))

    changed = replace_csv_rows(csv_payload, duplicate_first)
    rebound = bind_csv(provenance_payload, changed, row_count=3)

    with pytest.raises(ValueError, match="duplicate benchmark_id"):
        load_bundle_bytes(changed, rebound, source="duplicate rows")


def test_write_bundle_publishes_pair_and_validates_reload(tmp_path: Path) -> None:
    paths = ArtifactPaths(csv=tmp_path / "performance.csv", provenance=tmp_path / "performance.provenance.json")

    write_bundle(paths, bundle())

    assert load_bundle(paths) == PerformanceBundle(context=bundle().context, rows=bundle().sorted_rows)
    assert not list(tmp_path.glob(".performance.*.tmp"))


def test_loader_rejects_semantically_equivalent_noncanonical_bytes() -> None:
    csv_payload, provenance_payload = serialize_bundle(bundle())
    compact_provenance = json.dumps(json.loads(provenance_payload), sort_keys=True).encode()

    with pytest.raises(ValueError, match="provenance payload is not in canonical serialized form"):
        load_bundle_bytes(csv_payload, compact_provenance, source="compact provenance")

    crlf_csv = csv_payload.replace(b"\n", b"\r\n")
    rebound_provenance = bind_csv(provenance_payload, crlf_csv, row_count=len(bundle().rows))

    with pytest.raises(ValueError, match="CSV payload is not in canonical serialized form"):
        load_bundle_bytes(crlf_csv, rebound_provenance, source="CRLF CSV")


def test_publish_bundle_rolls_back_pair_when_consumer_fails(tmp_path: Path) -> None:
    paths = ArtifactPaths(csv=tmp_path / "performance.csv", provenance=tmp_path / "performance.provenance.json")
    old_bundle = bundle(current="v0.7.8", baseline="v0.7.7")
    write_bundle(paths, old_bundle)
    old_csv = paths.csv.read_bytes()
    old_provenance = paths.provenance.read_bytes()

    msg = "consumer failed"

    def fail_after_observing_new_bundle() -> None:
        with publish_bundle(paths, bundle()):
            assert load_bundle(paths) == PerformanceBundle(context=bundle().context, rows=bundle().sorted_rows)
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match=msg):
        fail_after_observing_new_bundle()

    assert paths.csv.read_bytes() == old_csv
    assert paths.provenance.read_bytes() == old_provenance
    assert load_bundle(paths) == PerformanceBundle(context=old_bundle.context, rows=old_bundle.sorted_rows)
    assert not list(tmp_path.glob(".performance.*.tmp"))


def test_publish_bundle_restores_prior_absence_when_consumer_fails(tmp_path: Path) -> None:
    paths = ArtifactPaths(csv=tmp_path / "performance.csv", provenance=tmp_path / "performance.provenance.json")
    msg = "consumer failed"

    def fail_after_observing_new_bundle() -> None:
        with publish_bundle(paths, bundle()):
            assert load_bundle(paths).context.release.current == "v0.8.0"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match=msg):
        fail_after_observing_new_bundle()

    assert not paths.csv.exists()
    assert not paths.provenance.exists()
    assert not list(tmp_path.glob(".performance.*.tmp"))


@pytest.mark.parametrize(
    ("field", "value"),
    [("group", "bad|group"), ("benchmark", "bad\nbenchmark"), ("coverage_note", "bad\rnote")],
)
def test_performance_row_rejects_markdown_structure(field: str, value: str) -> None:
    original = bundle().rows[1]

    with pytest.raises(ValueError, match="Markdown-safe"):
        replace(original, **{field: value})


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("release", "current"), "0.8.0", "normalized semver"),
        (("current", "source", "commit"), "abc123", "full lowercase Git object ID"),
        (("current", "source", "ref"), "refs/../bad", "supported Git ref"),
        (("current", "source", "revision_timestamp"), "2026-08-23T12:00:00", "include a timezone"),
        (("current", "source", "revision_timestamp"), "2026-08-23\n12:00:00+00:00", "Markdown-safe"),
        (("current", "source", "revision_timestamp"), "2026-08-23`12:00:00+00:00", "Markdown-safe"),
        (("current", "source", "revision_timestamp"), "2026-08-23 12:00:00+00:00", "canonical 'T'"),
        (("current", "toolchain", "cargo_profile"), "release", "must be 'perf'"),
    ],
)
def test_loader_rejects_malformed_invariant_provenance(path: tuple[str, ...], value: object, message: str) -> None:
    csv_payload, provenance_payload = serialize_bundle(bundle())
    data = json.loads(provenance_payload)
    target = data
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        load_bundle_bytes(csv_payload, json.dumps(data).encode(), source="malformed provenance")


def test_bundle_requires_comparable_complete_release_signal_coverage_for_promotion() -> None:
    original = bundle()
    rows = tuple(
        replace(row, coverage_status="not-comparable", coverage_note="benchmark harness differs") if row.coverage_status == "comparable" else row
        for row in original.rows
    )
    unverified = PerformanceBundle(context=original.context, rows=rows)

    with pytest.raises(ValueError, match="no scientifically comparable rows"):
        unverified.require_promotable()


def test_configuration_digest_is_provenance_not_a_comparison_blocker() -> None:
    original = context()
    changed = replace(
        original,
        baseline_toolchain=replace(original.baseline_toolchain, configuration_sha256=SHA_C),
    )

    assert changed.comparison_blockers == ()


def test_measurement_plan_difference_blocks_comparable_rows() -> None:
    original = bundle()
    changed = replace(
        original.context,
        baseline_toolchain=replace(original.context.baseline_toolchain, measurement_plan_sha256=SHA_C),
    )

    assert "measurement plan differs" in changed.comparison_blockers
    with pytest.raises(ValueError, match="compatible measurement provenance"):
        PerformanceBundle(context=changed, rows=(original.rows[0],))


def test_promotion_rejects_symmetric_missing_release_targets() -> None:
    original = bundle()
    incomplete_targets = RELEASE_TARGETS[:-1]
    incomplete = PerformanceBundle(
        context=replace(
            original.context,
            current_completed_targets=incomplete_targets,
            baseline_completed_targets=incomplete_targets,
        ),
        rows=(original.rows[0],),
    )

    with pytest.raises(ValueError, match=r"current measurement did not complete required targets.*baseline measurement did not complete"):
        incomplete.require_promotable()


def test_promotion_allows_supported_release_target_transition_with_one_sided_rows() -> None:
    """A newly added target must not invalidate shared historical measurements."""
    original = bundle()
    transitioning = PerformanceBundle(
        context=replace(
            original.context,
            baseline_completed_targets=RELEASE_TARGETS[:-1],
        ),
        rows=original.rows,
    )

    assert transitioning.context.shared_completed_targets == RELEASE_TARGETS[:-1]
    assert transitioning.context.target_transition_blockers == ()
    transitioning.require_promotable()


def test_target_transition_requires_the_union_to_cover_the_release_plan() -> None:
    """Asymmetric target loss is not a valid versioned plan transition."""
    original = context()
    invalid = replace(
        original,
        current_completed_targets=RELEASE_TARGETS[:-1],
        baseline_completed_targets=RELEASE_TARGETS[:-2],
    )

    assert "release target transition does not cover the canonical release-signal plan" in invalid.comparison_blockers


@pytest.mark.parametrize("value", ["bad\nvalue", "bad`value", "unknown"])
def test_recorded_host_rejects_unsafe_or_placeholder_identity(value: str) -> None:
    with pytest.raises(ValueError, match=r"Markdown-safe|placeholder"):
        HostIdentity(status="recorded", cpu=value, operating_system="Test OS", architecture="test")


def test_context_rejects_markdown_unsafe_command_arguments() -> None:
    with pytest.raises(ValueError, match="Markdown-safe"):
        replace(context(), current_commands=(("cargo", "bad\nargument"),))


def test_github_assets_are_always_separate_measurement_sessions() -> None:
    original = context()
    archive_sha = "f" * 64
    github_context = replace(
        original,
        measurement_mode="github-assets",
        current_acquisition_commands=(("gh", "release", "download", original.release.current),),
        baseline_acquisition_commands=(("gh", "release", "download", original.release.baseline),),
        current_artifact=MeasurementArtifact(
            origin="release-archive",
            content_sha256=SHA_B,
            sample_name="new",
            archive_sha256=archive_sha,
        ),
        baseline_artifact=MeasurementArtifact(
            origin="release-archive",
            content_sha256=SHA_C,
            sample_name="new",
            archive_sha256=archive_sha,
        ),
    )

    assert "release archives were measured in separate sessions" in github_context.comparison_blockers


def test_artifact_paths_reject_aliases(tmp_path: Path) -> None:
    target = tmp_path / "performance.csv"

    with pytest.raises(ValueError, match="must use distinct paths"):
        ArtifactPaths(csv=target, provenance=target)
