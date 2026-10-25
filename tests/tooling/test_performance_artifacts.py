"""Tests for retained shared performance JSON and provenance artifacts."""

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from research_repo_tools.complete_runs import CompletePolicy, parse_policy, serialize_policy
from research_repo_tools.criterion import COMPARISON_SCHEMA, SAMPLE_SCHEMA, Estimate, Sample, parse_comparison, serialize_sample
from research_repo_tools.criterion_inventory import CriterionInventory, parse_inventory, serialize_inventory
from research_repo_tools.evidence import Evidence, Provenance, parse_evidence, serialize_evidence, sha256
from research_repo_tools.host_metadata import HostMetadata, serialize_host

import benchmark_utils
import publish_readme_performance
from performance_artifacts import (
    MEASUREMENTS_CONTEXT,
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
    "locate",
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


def retained_bundle() -> PerformanceBundle:
    """Retain a promotable comparison with independent shared measurement records."""
    original = bundle()
    groups = ("validation", "circumcenter", "locate")
    original = replace(
        original,
        rows=tuple(replace(original.rows[0], group=group, benchmark_id=f"{group}/{original.rows[0].benchmark}") for group in groups),
    )
    inventory = CriterionInventory(tuple((target, (row.benchmark_id,)) for target, row in zip(RELEASE_TARGETS, original.rows, strict=True)))
    return retain_measurements(original, inventory)


def retain_measurements(original: PerformanceBundle, inventory: CriterionInventory) -> PerformanceBundle:
    """Attach shared samples matching the rows and the independently declared inventory."""
    measurements = []
    for side in ("baseline", "current"):
        targets = original.context.current_completed_targets if side == "current" else original.context.baseline_completed_targets
        measured_inventory = replace(inventory, suites=tuple((target, names) for target, names in inventory.suites if target in targets))
        state = original.context.current_source if side == "current" else original.context.baseline_source
        toolchain = original.context.current_toolchain if side == "current" else original.context.baseline_toolchain
        host = original.context.current_measurement_host if side == "current" else original.context.baseline_measurement_host
        shared_host = HostMetadata(host.operating_system, host.architecture, host.cpu, 4, 8, 4096)
        artifact = original.context.current_artifact if side == "current" else original.context.baseline_artifact
        assert toolchain.criterion_version is not None
        estimates = []
        for row in original.rows:
            timing = row.current if side == "current" else row.baseline
            if timing is not None:
                estimates.append((row.benchmark_id, Estimate(timing.median_ns, timing.ci_lower_ns, timing.ci_upper_ns, timing.confidence_level)))
        evidence = Evidence(
            serialize_sample(Sample(tuple(estimates))),
            SAMPLE_SCHEMA,
            (
                (
                    "measurement",
                    Provenance(
                        state.commit,
                        "1" * 64,
                        "2" * 64,
                        (
                            ("release", state.version),
                            ("tool.rustc", json.dumps(toolchain.rustc)),
                            ("dependency.criterion", toolchain.criterion_version),
                            ("host", json.dumps(json.loads(serialize_host(shared_host)))),
                            ("delaunay.inventory", json.dumps(json.loads(serialize_inventory(measured_inventory)))),
                            ("delaunay.sampling-policy", serialize_policy(CompletePolicy(measured_inventory.expected)).decode()),
                            ("delaunay.scientific-preflight", "passed"),
                            ("delaunay.sample", artifact.sample_name),
                            ("delaunay.raw-sample-sha256", artifact.content_sha256),
                        ),
                    ),
                ),
            ),
        )
        measurements.append((side, parse_evidence(*serialize_evidence(evidence))))
    return replace(original, measurements=tuple(measurements))


def serialize_changed_measurement(original: PerformanceBundle, side: str, evidence: Evidence) -> tuple[bytes, bytes]:
    """Change one retained record while preserving the shared envelope integrity."""
    payload, manifest = serialize_bundle(original)
    metadata = json.loads(manifest)
    context = metadata["sources"]["current"]["context"]
    measurements = json.loads(context[MEASUREMENTS_CONTEXT])
    measurements[side] = [part.decode() for part in serialize_evidence(evidence)]
    context[MEASUREMENTS_CONTEXT] = json.dumps(measurements)
    return payload, json.dumps(metadata).encode()


def measurement_with_policy(evidence: Evidence, policy: CompletePolicy) -> Evidence:
    """Change the declared policy without altering the recorded timing estimates."""
    source = evidence.sources[0][1]
    metadata = dict(source.context)
    metadata["delaunay.sampling-policy"] = serialize_policy(policy).decode()
    return replace(evidence, sources=(("measurement", replace(source, context=tuple(metadata.items()))),))


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


@pytest.mark.parametrize("current_first", [False, True])
def test_retained_measurements_round_trip_in_either_input_order(tmp_path: Path, current_first: bool) -> None:
    original = retained_bundle()
    original.require_promotable()
    reordered = replace(original, measurements=tuple(reversed(original.measurements)) if current_first else original.measurements)
    paths = ArtifactPaths(payload=tmp_path / "performance.comparison.json", provenance=tmp_path / "performance.evidence.json")

    assert tuple(side for side, _ in reordered.measurements) == ("baseline", "current")
    assert serialize_bundle(reordered) == serialize_bundle(original)
    expected = replace(original, rows=original.sorted_rows)
    assert load_bundle_bytes(*serialize_bundle(reordered), source="retained measurement order") == expected
    write_bundle(paths, reordered)
    assert load_bundle(paths) == expected
    load_bundle(paths).require_promotable()


@pytest.mark.parametrize(
    ("suite", "suites"),
    [
        ("ci", (("ci_performance_suite", ("validation/validate_3d/750", "proof_boundaries/validation/3d")),)),
        (
            "query",
            (("circumsphere_containment", ("random/insphere", "circumcenter/3d")), ("locate", ("locate/no_hint/3d/case",))),
        ),
        (
            "predicates",
            (("circumsphere_containment", ("random/insphere", "circumcenter/3d")), ("cold_path_predicates", ("predicates/hot/insphere/3d",))),
        ),
    ],
)
def test_smaller_suites_retain_complete_measurement_inventory(tmp_path: Path, suite: str, suites: tuple[tuple[str, tuple[str, ...]], ...]) -> None:
    """Collected report rows must retain every selected target case through serialization."""
    inventory = CriterionInventory(suites)
    for full_id in inventory.expected:
        for sample_name in ("new", "v0.7.8"):
            directory = tmp_path / full_id.replace("/", "_") / sample_name
            directory.mkdir(parents=True)
            benchmark = {"full_id": full_id, "group_id": full_id.split("/", maxsplit=1)[0]}
            timing = {"point_estimate": 100, "confidence_interval": {"lower_bound": 90, "upper_bound": 110, "confidence_level": 0.95}}
            (directory / "benchmark.json").write_text(json.dumps(benchmark), encoding="utf-8")
            (directory / "estimates.json").write_text(json.dumps({"median": timing}), encoding="utf-8")
            (directory / "sample.json").write_text(json.dumps({"iters": [1, 1], "times": [90, 110]}), encoding="utf-8")
    targets = tuple(target for target, _ in inventory.suites)
    measured_context = replace(context(), suite=suite, current_completed_targets=targets, baseline_completed_targets=targets)
    rows = benchmark_utils.collect_performance_rows(tmp_path, "v0.7.8", suite=suite)
    retained = retain_measurements(PerformanceBundle(measured_context, rows), inventory)

    loaded = load_bundle_bytes(*serialize_bundle(retained), source="complete smaller suite")

    assert {row.benchmark_id for row in loaded.rows} == set(inventory.expected)
    assert loaded == replace(retained, rows=retained.sorted_rows)


@pytest.mark.parametrize("side", ["baseline", "current"])
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tool.rustc", json.dumps("rustc 1.97.0"), "rustc does not match report toolchain"),
        ("tool.rustc", None, "rustc does not match report toolchain"),
        ("tool.rustc", "not-json", "rustc must be a JSON-encoded compiler identity"),
        ("dependency.criterion", "0.6.0", "Criterion version does not match report toolchain"),
        ("dependency.criterion", None, "Criterion version does not match report toolchain"),
        ("delaunay.sample", "unrelated-sample", "sample name does not match artifact"),
        ("delaunay.sample", None, "sample name does not match artifact"),
    ],
)
def test_loader_binds_retained_measurement_identity(side: str, field: str, value: str | None, message: str) -> None:
    original = retained_bundle()
    original.require_promotable()
    evidence = dict(original.measurements)[side]
    source = evidence.sources[0][1]
    metadata = dict(source.context)
    if value is None:
        metadata.pop(field)
    else:
        metadata[field] = value
    changed = replace(evidence, sources=(("measurement", replace(source, context=tuple(metadata.items()))),))

    with pytest.raises(ValueError, match=message):
        load_bundle_bytes(*serialize_changed_measurement(original, side, changed), source="changed measurement identity")


@pytest.mark.parametrize("side", ["baseline", "current"])
@pytest.mark.parametrize("field", ["host", "delaunay.sampling-policy", "delaunay.inventory"])
def test_missing_retained_metadata_has_actionable_reader_and_cli_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    side: str,
    field: str,
) -> None:
    """Missing consumer fields reject publication without a traceback or file changes."""
    original = retained_bundle()
    evidence = dict(original.measurements)[side]
    source = evidence.sources[0][1]
    metadata = dict(source.context)
    metadata.pop(field)
    changed = replace(evidence, sources=(("measurement", replace(source, context=tuple(metadata.items()))),))
    payload, manifest = serialize_changed_measurement(original, side, changed)
    message = f"retained {side} measurement.{field} must be a non-empty string"
    with pytest.raises(ValueError, match=f"retained {side} measurement") as raised:
        load_bundle_bytes(payload, manifest, source="missing measurement metadata")
    assert str(raised.value) == message

    (tmp_path / "comparison.json").write_bytes(payload)
    (tmp_path / "evidence.json").write_bytes(manifest)
    (tmp_path / "Cargo.toml").write_text('[package]\nversion = "0.8.0"\n', encoding="utf-8")
    (tmp_path / "README.md").write_bytes(b"previous README\n")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    result = publish_readme_performance.main(["--root", str(tmp_path), "--artifact-payload", "comparison.json", "--artifact-provenance", "evidence.json"])

    assert result == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert message in captured.err
    assert "Traceback" not in captured.err
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("side", ["baseline", "current"])
@pytest.mark.parametrize("change", ["rename", "remove", "add"])
def test_loader_binds_retained_inventory_targets_to_completed_targets(side: str, change: str) -> None:
    original = retained_bundle()
    original.require_promotable()
    evidence = dict(original.measurements)[side]
    source = evidence.sources[0][1]
    metadata = dict(source.context)
    inventory = parse_inventory(metadata["delaunay.inventory"].encode())
    suites = list(inventory.suites)
    excluded = inventory.excluded
    if change == "rename":
        suites[0] = ("unmeasured_target", suites[0][1])
    elif change == "remove":
        suites[1] = (suites[1][0], (*suites[1][1], *suites[0][1]))
        suites.pop(0)
    else:
        suites.append(("unmeasured_target", ("unselected/case",)))
        excluded = (*excluded, "unselected/case")
    changed_inventory = replace(inventory, suites=tuple(suites), excluded=excluded)
    assert changed_inventory.expected == inventory.expected
    metadata["delaunay.inventory"] = json.dumps(json.loads(serialize_inventory(changed_inventory)))
    changed = replace(evidence, sources=(("measurement", replace(source, context=tuple(metadata.items()))),))

    with pytest.raises(ValueError, match="inventory targets do not match completed targets"):
        load_bundle_bytes(*serialize_changed_measurement(original, side, changed), source="changed measurement target inventory")


@pytest.mark.parametrize("side", ["baseline", "current"])
@pytest.mark.parametrize(
    ("field", "value"),
    [("unit", "us"), ("statistics", ["mean"]), ("confidence_level", 0.9)],
)
def test_loader_binds_sampling_policy_to_estimates(side: str, field: str, value: str | list[str] | float) -> None:
    original = retained_bundle()
    original.require_promotable()
    evidence = dict(original.measurements)[side]
    source = evidence.sources[0][1]
    metadata = dict(source.context)
    policy = json.loads(metadata["delaunay.sampling-policy"])
    policy[field] = value
    metadata["delaunay.sampling-policy"] = json.dumps(policy)
    changed = replace(evidence, sources=(("measurement", replace(source, context=tuple(metadata.items()))),))

    with pytest.raises(ValueError, match="estimates differ from their sampling policy"):
        load_bundle_bytes(*serialize_changed_measurement(original, side, changed), source="changed sampling policy")


@pytest.mark.parametrize("side", ["baseline", "current"])
@pytest.mark.parametrize("per_case", [False, True])
def test_loader_rejects_count_policies_that_contradict_matching_plans(side: str, per_case: bool) -> None:
    original = retained_bundle()
    original.require_promotable()
    evidence = dict(original.measurements)[side]
    policy = parse_policy(dict(evidence.sources[0][1].context)["delaunay.sampling-policy"].encode())
    changed_policy = (
        replace(policy, sample_count=None, sample_counts=tuple((name, 2 if index == 0 else 100) for index, name in enumerate(policy.expected)))
        if per_case
        else replace(policy, sample_count=2)
    )
    changed = measurement_with_policy(evidence, changed_policy)

    with pytest.raises(ValueError, match="sampling policies contradict matching measurement plans"):
        load_bundle_bytes(*serialize_changed_measurement(original, side, changed), source="contradictory sample counts")


def test_loader_accepts_equivalent_uniform_and_per_case_counts() -> None:
    original = retained_bundle()
    evidence = dict(original.measurements)["current"]
    policy = parse_policy(dict(evidence.sources[0][1].context)["delaunay.sampling-policy"].encode())
    changed = measurement_with_policy(evidence, replace(policy, sample_count=None, sample_counts=tuple((name, 100) for name in policy.expected)))

    loaded = load_bundle_bytes(*serialize_changed_measurement(original, "current", changed), source="equivalent sample counts")

    loaded.require_promotable()
    assert loaded.rows == original.sorted_rows


def test_loader_retains_different_counts_when_plans_are_non_comparable() -> None:
    original = retained_bundle()
    original = replace(
        original,
        context=replace(original.context, current_toolchain=replace(original.context.current_toolchain, measurement_plan_sha256="f" * 64)),
        rows=tuple(replace(row, coverage_status="not-comparable", coverage_note="measurement plan differs") for row in original.rows),
    )
    evidence = dict(original.measurements)["current"]
    policy = parse_policy(dict(evidence.sources[0][1].context)["delaunay.sampling-policy"].encode())
    changed = measurement_with_policy(evidence, replace(policy, sample_count=2))

    loaded = load_bundle_bytes(*serialize_changed_measurement(original, "current", changed), source="different measurement plans")

    assert "measurement plan differs" in loaded.promotion_blockers
    assert all(row.coverage_status == "not-comparable" for row in loaded.rows)
    with pytest.raises(ValueError, match="measurement plan differs"):
        loaded.require_promotable()


@pytest.mark.parametrize("missing_side", ["baseline", "current"])
def test_retained_policy_comparison_projects_onto_shared_targets(missing_side: str) -> None:
    original = retained_bundle()
    inventory = parse_inventory(dict(original.measurements[0][1].sources[0][1].context)["delaunay.inventory"].encode())
    changed_context = (
        replace(original.context, baseline_completed_targets=RELEASE_TARGETS[:-1])
        if missing_side == "baseline"
        else replace(original.context, current_completed_targets=RELEASE_TARGETS[:-1])
    )
    extra = original.rows[-1]
    extra = (
        replace(extra, baseline=None, coverage_status="current-only", coverage_note="added target")
        if missing_side == "baseline"
        else replace(extra, current=None, coverage_status="baseline-only", coverage_note="removed target")
    )
    transitioning = retain_measurements(replace(original, context=changed_context, rows=(*original.rows[:-1], extra), measurements=()), inventory)
    full_side = "current" if missing_side == "baseline" else "baseline"
    evidence = dict(transitioning.measurements)[full_side]
    policy = parse_policy(dict(evidence.sources[0][1].context)["delaunay.sampling-policy"].encode())
    changed = measurement_with_policy(
        evidence,
        replace(policy, sample_count=None, sample_counts=tuple((name, 2 if name == extra.benchmark_id else 100) for name in policy.expected)),
    )

    loaded = load_bundle_bytes(*serialize_changed_measurement(transitioning, full_side, changed), source="supported target transition")

    if missing_side == "baseline":
        loaded.require_promotable()
    else:
        with pytest.raises(ValueError, match="active public API benchmark plan"):
            loaded.require_promotable()
    assert loaded.rows == transitioning.sorted_rows


def test_retained_measurements_reject_changed_report_timings() -> None:
    original = retained_bundle()
    changed_row = replace(original.rows[0], current=estimate(3_000_000.0))

    with pytest.raises(ValueError, match="report timing differs from retained measurement"):
        replace(original, rows=(changed_row, *original.rows[1:]))


def test_retained_measurements_reject_omitted_report_rows() -> None:
    original = retained_bundle()

    with pytest.raises(ValueError, match="report omits retained measurement rows"):
        replace(original, rows=original.rows[1:])


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
        rows=(original.rows[0], replace(original.rows[1], group="locate", benchmark_id="locate/new_case/750")),
    )

    assert transitioning.context.shared_completed_targets == RELEASE_TARGETS[:-1]
    assert transitioning.context.target_transition_blockers == ()
    transitioning.require_promotable()


@pytest.mark.parametrize("side", ["current-only", "baseline-only"])
@pytest.mark.parametrize("group", ["validation", "unknown"])
def test_unrelated_target_transition_cannot_hide_missing_cases(side: str, group: str) -> None:
    """A target added elsewhere does not excuse missing shared or unknown rows."""
    original = bundle()
    missing = replace(
        original.rows[1],
        group=group,
        benchmark_id=f"{group}/new_case/750",
        coverage_status=side,
        baseline=estimate(500_000.0) if side == "baseline-only" else None,
        current=estimate(500_000.0) if side == "current-only" else None,
    )
    transitioning = PerformanceBundle(
        context=replace(original.context, baseline_completed_targets=RELEASE_TARGETS[:-1]),
        rows=(original.rows[0], missing),
    )
    with pytest.raises(ValueError, match="coverage is incomplete"):
        transitioning.require_promotable()


def test_promotion_rejects_removing_a_target_still_in_the_active_plan() -> None:
    """One-sided rows cannot excuse omitting a required current public API workload."""
    original = bundle()
    removed = replace(
        original.rows[1],
        group="locate",
        benchmark_id="locate/new_case/750",
        coverage_status="baseline-only",
        baseline=estimate(500_000.0),
        current=None,
    )
    incomplete = PerformanceBundle(
        context=replace(original.context, current_completed_targets=RELEASE_TARGETS[:-1]),
        rows=(original.rows[0], removed),
    )
    with pytest.raises(ValueError, match="active public API benchmark plan"):
        incomplete.require_promotable()


def test_promotion_rejects_a_complete_superseded_plan() -> None:
    """Historical diagnostics remain readable but cannot be promoted as the new contract."""
    original = bundle()
    old_targets = ("ci_performance_suite", "circumsphere_containment", "cold_path_predicates", "locate", "realization_validation")
    historical = replace(original, context=replace(original.context, current_completed_targets=old_targets, baseline_completed_targets=old_targets))
    payload, manifest = serialize_bundle(historical)
    loaded = load_bundle_bytes(payload, manifest, source="historical benchmark plan")
    with pytest.raises(ValueError, match="active public API benchmark plan"):
        loaded.require_promotable()


@pytest.mark.parametrize("record", ["source", "toolchain"])
@pytest.mark.parametrize("value", [None, False, 0, [], {}])
def test_loader_rejects_non_string_limitation(record: str, value: object) -> None:
    payload, manifest = serialize_bundle(bundle())
    data = json.loads(manifest)
    policy = json.loads(data["sources"]["current"]["context"][POLICY_CONTEXT])
    policy["context"]["current"][record]["limitation"] = value
    data["sources"]["current"]["context"][POLICY_CONTEXT] = json.dumps(policy, separators=(",", ":"))
    with pytest.raises(TypeError, match="limitation must be a string"):
        load_bundle_bytes(payload, json.dumps(data).encode(), source="malformed limitation")


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
