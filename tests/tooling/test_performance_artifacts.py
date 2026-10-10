"""Tests for retained performance CSV and provenance artifacts."""

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from research_repo_tools.criterion import COMPARISON_SCHEMA, parse_comparison
from research_repo_tools.evidence import parse_evidence, sha256

from performance_artifacts import (
    POLICY_CONTEXT,
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


def test_artifact_round_trip_preserves_comparable_and_one_sided_rows() -> None:
    """The published parser must accept new consumer artifacts without legacy CSV."""
    original = bundle()
    payload, manifest = serialize_bundle(original)
    evidence = parse_evidence(payload, manifest)
    comparison = parse_comparison(evidence.payload)

    assert evidence.payload_schema == COMPARISON_SCHEMA
    assert comparison.missing_baseline == ("validation/new_case/750",)
    assert dict(comparison.baseline.estimates)["validation/validate_3d/750"].point == 2_000_000
    assert load_bundle_bytes(payload, manifest, source="round-trip fixture") == PerformanceBundle(context=original.context, rows=original.sorted_rows)
    assert b'"speedup"' not in payload
    assert b'"percent_reduction"' not in payload


def test_artifact_round_trip_allows_same_version_local_comparison() -> None:
    original = bundle(current="v0.8.0", baseline="v0.8.0")
    payload, provenance_payload = serialize_bundle(original)

    parsed = load_bundle_bytes(payload, provenance_payload, source="same-version fixture")

    assert parsed.context.release == ReleasePair(current="v0.8.0", baseline="v0.8.0")


def test_artifact_serialization_sorts_rows_deterministically() -> None:
    original = bundle()
    reordered = PerformanceBundle(context=original.context, rows=tuple(reversed(original.rows)))

    assert serialize_bundle(original) == serialize_bundle(reordered)


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


def test_write_bundle_publishes_pair_and_validates_reload(tmp_path: Path) -> None:
    paths = ArtifactPaths(payload=tmp_path / "performance.comparison.json", provenance=tmp_path / "performance.evidence.json")

    write_bundle(paths, bundle())

    assert load_bundle(paths) == PerformanceBundle(context=bundle().context, rows=bundle().sorted_rows)
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
        (("release", "current"), "v0.8.2-01", "normalized semver"),
        (("release", "baseline"), "v0.8.2-rc.01", "normalized semver"),
        (("current", "source", "commit"), "abc123", "full lowercase Git object ID"),
        (("current", "source", "ref"), "refs/../bad", "supported Git ref"),
        (("current", "source", "revision_timestamp"), "2026-08-23T12:00:00", "include a timezone"),
        (("current", "source", "revision_timestamp"), "2026-08-23\n12:00:00+00:00", "Markdown-safe"),
        (("current", "source", "revision_timestamp"), "2026-08-23 12:00:00+00:00", "canonical 'T'"),
        (("current", "toolchain", "cargo_profile"), "release", "must be 'perf'"),
    ],
)
def test_loader_rejects_malformed_invariant_provenance(path: tuple[str, ...], value: object, message: str) -> None:
    payload, manifest = serialize_bundle(bundle())
    data = json.loads(manifest)
    policy = json.loads(data["sources"]["current"]["context"][POLICY_CONTEXT])
    target = policy["context"]
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    data["sources"]["current"]["context"][POLICY_CONTEXT] = json.dumps(policy, separators=(",", ":"))

    with pytest.raises(ValueError, match=message):
        load_bundle_bytes(payload, json.dumps(data).encode(), source="malformed consumer policy")


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


@pytest.mark.parametrize("field", ["current", "baseline"])
@pytest.mark.parametrize("tag", ["v0.8.2-01", "v0.8.2-rc.01", "v0.8.2-1.00+build.01"])
def test_release_pair_rejects_numeric_prerelease_identifiers_with_leading_zeroes(field: str, tag: str) -> None:
    """Reject malformed tags during construction, before precedence evaluation."""
    tags = {"current": "v0.8.2", "baseline": "v0.8.2"}
    tags[field] = tag

    with pytest.raises(ValueError, match=f"{field} release must be a normalized semver"):
        ReleasePair(**tags)


@pytest.mark.parametrize(
    "tag",
    ["v0.8.2-0", "v0.8.2-1", "v0.8.2-10", "v0.8.2-rc.0", "v0.8.2-01alpha", "v0.8.2-01-alpha", "v0.8.2-1+build.01", "v0.8.2+001"],
)
def test_bundle_roundtrip_preserves_valid_prerelease_and_build_identifiers(tag: str) -> None:
    """Keep valid numeric, alphanumeric, and build identifiers intact in artifacts."""
    payload, provenance_payload = serialize_bundle(bundle(current=tag, baseline=tag))

    loaded = load_bundle_bytes(payload, provenance_payload, source="valid semver")

    assert loaded.context.release.current == tag
    assert loaded.context.release.baseline == tag
    assert loaded.context.comparison_blockers == ()


@pytest.mark.parametrize(
    ("current", "baseline", "blocked"),
    [
        ("v0.8.2", "v0.8.1", True),
        ("v0.10.0", "v0.8.1", True),
        ("v0.8.1", "v0.8.2", True),
        ("v0.8.2+build.1", "v0.8.1", True),
        ("v0.8.2", "v0.8.2-rc.1", True),
        ("v0.8.2-rc.1", "v0.8.2", True),
        ("v0.8.2+build.1", "v0.8.2-rc.1+build.2", True),
        ("v0.8.2", "v0.8.2-0", True),
        ("v0.8.3-rc.1", "v0.8.2-rc.1", True),
        ("v0.8.3", "v0.8.2", False),
        ("v0.8.1", "v0.8.0", False),
        ("v0.8.2-rc.1", "v0.8.1", False),
        ("v0.8.2-rc.10", "v0.8.2-rc.2", False),
        ("v0.8.3-rc.1", "v0.8.2", False),
        ("v0.8.2+build-with-hyphens", "v0.8.2", False),
    ],
)
def test_fresh_evidence_uses_harness_identity_instead_of_historical_release_boundary(current: str, baseline: str, blocked: bool) -> None:
    """Fresh matching workloads can cross the old boundary; archive policy still records it."""
    measured = context(current=current, baseline=baseline)
    assert bool(measured.release.benchmark_contract_blockers) is blocked
    assert measured.comparison_blockers == ()
    PerformanceBundle(context=measured, rows=(bundle().rows[0],)).require_promotable()


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
    target = tmp_path / "performance.comparison.json"

    with pytest.raises(ValueError, match="must use distinct paths"):
        ArtifactPaths(payload=target, provenance=target)


def test_shared_payload_integrity_is_checked_before_consumer_policy() -> None:
    """A retained hash mismatch must fail before timing interpretation."""
    payload, manifest = serialize_bundle(bundle())
    with pytest.raises(ValueError, match="recorded SHA-256"):
        load_bundle_bytes(payload + b" ", manifest, source="changed timing payload")


def test_loader_requires_recorded_confidence_intervals() -> None:
    """Shared optional intervals cannot weaken Delaunay's recorded-interval gate."""
    payload, manifest = serialize_bundle(bundle())
    timings = json.loads(payload)
    row = timings["baseline"][0]
    row["lower"] = row["upper"] = row["confidence_level"] = None
    changed = json.dumps(timings).encode()
    metadata = json.loads(manifest)
    metadata["payload_sha256"] = sha256(changed)
    with pytest.raises(ValueError, match="complete confidence intervals"):
        load_bundle_bytes(changed, json.dumps(metadata).encode(), source="missing marginal interval")


def test_loader_requires_policy_inventory_to_match_both_samples() -> None:
    """The policy may not hide an added or removed measurement."""
    payload, manifest = serialize_bundle(bundle())
    data = json.loads(manifest)
    policy = json.loads(data["sources"]["current"]["context"][POLICY_CONTEXT])
    policy["rows"].pop()
    data["sources"]["current"]["context"][POLICY_CONTEXT] = json.dumps(policy, separators=(",", ":"))
    with pytest.raises(ValueError, match="complete shared benchmark inventory"):
        load_bundle_bytes(payload, json.dumps(data).encode(), source="hidden measurement")


def test_loader_binds_shared_revision_to_consumer_policy() -> None:
    """A valid but different envelope revision cannot replace the measured source."""
    payload, manifest = serialize_bundle(bundle())
    data = json.loads(manifest)
    data["sources"]["current"]["revision"] = "c" * 40
    with pytest.raises(ValueError, match="does not match Delaunay source identity"):
        load_bundle_bytes(payload, json.dumps(data).encode(), source="changed source identity")


def test_separate_sessions_remain_non_comparable_after_shared_round_trip() -> None:
    """Shared serialization must not turn archive measurements into paired timings."""
    original = bundle()
    host = original.context.publication_host
    archive = replace(
        original.context,
        measurement_mode="github-assets",
        current_acquisition_commands=(("gh", "release", "download", "v0.8.0"),),
        baseline_acquisition_commands=(("gh", "release", "download", "v0.7.8"),),
        current_artifact=MeasurementArtifact(origin="release-archive", content_sha256=SHA_B, sample_name="new", archive_sha256=SHA_A),
        baseline_artifact=MeasurementArtifact(origin="release-archive", content_sha256=SHA_C, sample_name="new", archive_sha256=SHA_A),
        current_measurement_host=host,
        baseline_measurement_host=host,
    )
    row = replace(original.rows[0], coverage_status="not-comparable", coverage_note="release archives were measured in separate sessions")
    payload, manifest = serialize_bundle(PerformanceBundle(archive, (row,)))
    loaded = load_bundle_bytes(payload, manifest, source="separate archive sessions")
    assert loaded.rows[0].coverage_status == "not-comparable"
    assert "release archives were measured in separate sessions" in loaded.context.comparison_blockers
    with pytest.raises(ValueError, match="not promotable"):
        loaded.require_promotable()
