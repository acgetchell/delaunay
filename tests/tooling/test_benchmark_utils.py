#!/usr/bin/env python3
"""
Test suite for benchmark_utils.py module.

Tests benchmark parsing, baseline generation, and performance comparison functionality,
with special focus on benchmark regression policy and summary calculations.

Note: This test file accesses private methods (prefixed with _) which is expected
and necessary for comprehensive unit testing of internal functionality.
"""

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import Mock, patch

import pytest
from research_repo_tools import measurement as shared_measurement
from research_repo_tools.complete_runs import serialize_policy
from research_repo_tools.criterion import SAMPLE_SCHEMA, Sample, read_estimate, serialize_sample
from research_repo_tools.criterion_inventory import CriterionInventory, serialize_inventory
from research_repo_tools.evidence import Evidence, Provenance, parse_evidence, serialize_evidence
from research_repo_tools.host_metadata import HostMetadata, serialize_host
from research_repo_tools.process import run_command as run_safe_command
from research_repo_tools.release_discovery import PublishedRelease
from research_repo_tools.worktrees import TreeSnapshot

import benchmark_utils
import performance_artifacts
from benchmark_models import (
    CircumspherePerformanceData,
    CircumsphereTestCase,
)
from benchmark_utils import (
    _CI_PERFORMANCE_SUITE_MANIFEST_IDS_FILE,
    _CI_PERFORMANCE_SUITE_METRICS_FILE,
    BENCHMARK_BUILD_FLAVOR,
    DEV_MODE_BENCH_ARGS,
    CiPerformanceMetric,
    CiPerformanceResult,
    CriterionReportSettings,
    PerformanceRequestOptions,
    PerformanceSummaryGenerator,
    ProjectRootNotFoundError,
    ReleaseReportConfig,
    _expand_ci_benchmark_id_pattern,
    _load_ci_performance_metrics,
    _parse_ci_performance_metrics,
    _write_ci_performance_metrics,
    collect_criterion_comparisons,
    collect_performance_rows,
    configure_logging,
    create_argument_parser,
    execute_command,
    find_project_root,
    generate_performance_worktree_report,
    normalize_release_tag,
    parse_performance_report_id,
    promote_performance_report,
    render_criterion_comparison_report,
    resolve_performance_request,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager
    from unittest.mock import MagicMock


CI_MANIFEST_STDOUT = (
    "api_benchmark group=boundary_facets public_api=DelaunayTriangulation::boundary_facets "
    "dimensions=3 benchmark_ids=boundary_facets/boundary_facets_3d/50 note=test\n"
    "api_benchmark_metric benchmark_id=tds_new_2d/tds_new/10 vertices=10 simplices=17\n"
)
PUBLIC_API_TITLE = "### Public API Performance Contract (`ci_performance_suite`)"
CIRCUMSPHERE_TITLE = "### Circumsphere Predicate Performance"
TDS_TITLE = "## Triangulation Data Structure Performance"
PERFORMANCE_UPDATES_TITLE = "## Performance Data Updates"
UTF8 = "utf-8"
DUPLICATE_BASELINE_KEY_SECTIONS = (
    "=== 10 Points (2D) ===\nBenchmark ID: duplicate/key\nTime: [1, 2, 3] µs\n=== 20 Points (2D) ===\nBenchmark ID: duplicate/key\nTime: [1, 2, 3] µs\n"
)


def completed_process(
    stdout: str = "",
    *,
    returncode: int = 0,
    stderr: str = "",
    args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Return a typed subprocess result for command-wrapper mocks."""
    return subprocess.CompletedProcess(args=args or [], returncode=returncode, stdout=stdout, stderr=stderr)


def complete_ci_performance_results() -> list[CiPerformanceResult]:
    """Return one valid parsed result for every public-API summary group."""
    return [
        CiPerformanceResult(
            group_key=group_key,
            benchmark_id=f"{group_key}/fixture/10",
            dimension="2D",
            input_size="10",
            mean_ns=1_000.0,
            low_ns=900.0,
            high_ns=1_100.0,
            confidence_level=0.95,
        )
        for group_key in benchmark_utils.CI_PERFORMANCE_SUITE_GROUP_ORDER
    ]


def complete_circumsphere_results(generator: PerformanceSummaryGenerator) -> list[CircumsphereTestCase]:
    """Return one valid result for every case/method in the summary contract."""
    methods_by_case: dict[tuple[str, str], dict[str, CircumspherePerformanceData]] = {}
    for test_name, dimension, method_name in generator._circumsphere_expected_results():
        methods = methods_by_case.setdefault((test_name, dimension), {})
        methods[method_name] = CircumspherePerformanceData(method_name, 1_000.0 + len(methods) * 100.0)
    return [
        CircumsphereTestCase(
            test_name,
            dimension,
            methods,
            is_boundary_case=test_name == "Boundary vertex",
        )
        for (test_name, dimension), methods in methods_by_case.items()
    ]


def retained_performance_bundle(*, current: str = "v0.8.0", baseline: str = "v0.7.8") -> performance_artifacts.PerformanceBundle:
    """Return a complete retained performance bundle fixture."""
    host = performance_artifacts.HostIdentity(status="recorded", cpu="Test CPU", operating_system="Test OS", architecture="test")
    toolchain = performance_artifacts.ToolchainState(
        rustc="rustc 1.98.0",
        criterion_version="0.7.0",
        cargo_profile="perf",
        cargo_lock_sha256="a" * 64,
        harness_sha256="b" * 64,
        configuration_sha256="c" * 64,
        measurement_plan_sha256="d" * 64,
    )
    return performance_artifacts.PerformanceBundle(
        context=performance_artifacts.ArtifactContext(
            release=performance_artifacts.ReleasePair(current=current, baseline=baseline),
            statistic="median",
            suite="release-signal",
            scope="release-signal",
            measurement_mode="local-worktrees",
            current_source=performance_artifacts.SourceState(
                version=current,
                commit="a" * 40,
                ref="HEAD",
                revision_timestamp="2026-08-23T12:00:00-07:00",
                git_clean=False,
                source_state_sha256="c" * 64,
            ),
            baseline_source=performance_artifacts.SourceState(
                version=baseline,
                commit="b" * 40,
                ref=baseline,
                revision_timestamp="2026-08-01T12:00:00-07:00",
                git_clean=True,
                source_state_sha256="d" * 64,
            ),
            current_commands=(("just", "bench-latest"),),
            baseline_commands=(("cargo", "bench", "--save-baseline", baseline),),
            current_completed_targets=benchmark_utils.RELEASE_SIGNAL_BENCH_TARGETS,
            baseline_completed_targets=benchmark_utils.RELEASE_SIGNAL_BENCH_TARGETS,
            current_acquisition_commands=(),
            baseline_acquisition_commands=(),
            current_toolchain=toolchain,
            baseline_toolchain=toolchain,
            current_measurement_host=host,
            baseline_measurement_host=host,
            current_artifact=performance_artifacts.MeasurementArtifact(
                origin="local-run",
                content_sha256="e" * 64,
                sample_name="new",
            ),
            baseline_artifact=performance_artifacts.MeasurementArtifact(
                origin="local-run",
                content_sha256="f" * 64,
                sample_name=baseline,
            ),
            publication_host=host,
        ),
        rows=(
            performance_artifacts.PerformanceRow(
                suite="release-signal",
                scope="release-signal",
                benchmark_id="validation/validate_3d/750",
                group="validation",
                benchmark="validate_3d/750",
                coverage_status="comparable",
                coverage_note="",
                baseline=performance_artifacts.TimingEstimate(2_000_000.0, 1_800_000.0, 2_200_000.0, 0.95),
                current=performance_artifacts.TimingEstimate(1_000_000.0, 900_000.0, 1_100_000.0, 0.95),
            ),
        ),
    )


def write_estimate(target_dir: Path, path_parts: tuple[str, ...], mean_ns: float, *, confidence_level: float | None = 0.95) -> None:
    """Write a minimal Criterion estimates.json fixture."""
    estimates_dir = target_dir / "criterion" / Path(*path_parts) / "base"
    estimates_dir.mkdir(parents=True)
    estimates = {
        "mean": {
            "point_estimate": mean_ns,
            "confidence_interval": {
                "confidence_level": confidence_level,
                "lower_bound": mean_ns * 0.9,
                "upper_bound": mean_ns * 1.1,
            },
        },
    }
    (estimates_dir / "estimates.json").write_text(json.dumps(estimates), encoding="utf-8")
    (estimates_dir / "sample.json").write_text('{"iters": [1, 2], "times": [1000, 2000]}', encoding="utf-8")
    full_id = "/".join(path_parts)
    (estimates_dir / "benchmark.json").write_text(
        json.dumps({"full_id": full_id, "group_id": path_parts[0]}),
        encoding="utf-8",
    )


def write_named_estimate(  # noqa: PLR0913
    target_dir: Path,
    path_parts: tuple[str, ...],
    sample: str,
    point_ns: float,
    stat: str = "median",
    *,
    full_id: str | None = None,
    group_id: str | None = None,
    confidence_level: float = 0.95,
) -> None:
    """Write a minimal named Criterion sample estimates.json fixture."""
    estimates_dir = target_dir / "criterion" / Path(*path_parts) / sample
    estimates_dir.mkdir(parents=True)
    estimates = {
        stat: {
            "point_estimate": point_ns,
            "confidence_interval": {
                "confidence_level": confidence_level,
                "lower_bound": point_ns * 0.9,
                "upper_bound": point_ns * 1.1,
            },
        },
    }
    (estimates_dir / "estimates.json").write_text(json.dumps(estimates), encoding="utf-8")
    (estimates_dir / "sample.json").write_text('{"iters": [1, 2], "times": [1000, 2000]}', encoding="utf-8")
    benchmark_id = full_id or "/".join(path_parts)
    (estimates_dir / "benchmark.json").write_text(
        json.dumps({"full_id": benchmark_id, "group_id": group_id or benchmark_id.split("/", maxsplit=1)[0]}),
        encoding="utf-8",
    )


def write_complete_release_signal_coverage(project_root: Path) -> None:
    """Write one valid Criterion estimate for every planned report group."""
    for measurement in benchmark_utils.RELEASE_SIGNAL_MEASUREMENT_PLAN:
        for prefix in measurement.required_group_prefixes:
            group = f"{prefix}fixture" if prefix.endswith("_") else prefix
            write_estimate(project_root / "target", (group, "fixture"), 1_000.0)


def write_versioned_release_asset_metadata(root: Path, *, tag: str, commit: str) -> dict[str, object]:
    """Write and return complete release-asset metadata for loader tests."""
    write_named_estimate(root, ("validation", "validate_3d", "750"), "new", 1_000_000.0)
    metadata: dict[str, object] = {
        "schema_version": benchmark_utils.RELEASE_ASSET_METADATA_SCHEMA_VERSION,
        "source": {
            "version": tag,
            "commit": commit,
            "ref": tag,
            "revision_timestamp": "2026-08-01T00:00:00Z",
            "git_clean": True,
            "source_state_sha256": hashlib.sha256(f"commit {commit}\n".encode()).hexdigest(),
            "limitation": "",
        },
        "measurement_commands": [list(command) for command in benchmark_utils.RELEASE_ASSET_MEASUREMENT_COMMANDS],
        "completed_targets": list(benchmark_utils.RELEASE_SIGNAL_BENCH_TARGETS),
        "toolchain": {
            "rustc": "rustc 1.98.0",
            "criterion_version": "0.7.0",
            "cargo_profile": "perf",
            "cargo_lock_sha256": "c" * 64,
            "harness_sha256": "d" * 64,
            "configuration_sha256": "e" * 64,
            "measurement_plan_sha256": "f" * 64,
            "limitation": "",
        },
        "measurement_host": {
            "status": "recorded",
            "cpu": "GitHub test CPU",
            "operating_system": "Linux",
            "architecture": "x86_64",
            "reason": "",
        },
        "criterion": {
            "content_sha256": benchmark_utils._directory_digest(root / "criterion"),
            "sample_name": "new",
        },
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding=UTF8)
    return metadata


def delaunay_report(version: str, baseline: str) -> str:
    """Return a minimal Delaunay benchmark performance report."""
    return (
        "# Benchmark Performance\n\n"
        f"**delaunay** v{version} · `abc1234` (release/test) · 2026-07-07 12:00:00 UTC\n"
        "**Statistic**: median\n"
        "**Suite**: release-signal\n"
        "**Scope**: release-signal\n\n"
        "## Benchmark Results\n\n"
        f"Comparison against baseline **{baseline}**:\n\n"
        "Negative change = faster. Speedup > 1.00x = improvement.\n\n"
        "### validation\n\n"
        "| Benchmark | Baseline | Latest | Change | Speedup |\n"
        "|-----------|---------:|-------:|-------:|--------:|\n"
        "| validate_3d/750 | 2.0 ms | 1.0 ms | -50.0% | 2.00x |\n"
    )


def normalized_delaunay_report(version: str, baseline: str) -> str:
    """Return a report after standard workflow footer normalization."""
    return benchmark_utils._normalize_how_to_update(delaunay_report(version, baseline))


def write_ci_performance_manifest(target_dir: Path, benchmark_ids: list[str]) -> None:
    """Write the ci_performance_suite runtime manifest sidecar."""
    criterion_dir = target_dir / "criterion"
    criterion_dir.mkdir(parents=True, exist_ok=True)
    (criterion_dir / _CI_PERFORMANCE_SUITE_MANIFEST_IDS_FILE).write_text(
        "\n".join(benchmark_ids) + "\n",
        encoding="utf-8",
    )


def write_ci_performance_metrics(target_dir: Path, metrics: dict[str, dict[str, int]]) -> None:
    """Write the ci_performance_suite metrics sidecar."""
    criterion_dir = target_dir / "criterion"
    criterion_dir.mkdir(parents=True, exist_ok=True)
    (criterion_dir / _CI_PERFORMANCE_SUITE_METRICS_FILE).write_text(
        json.dumps(metrics),
        encoding="utf-8",
    )


def test_collect_criterion_comparisons_filters_release_signal_scope(tmp_path: Path) -> None:
    """Saved Criterion baseline comparison should include curated release-signal groups."""
    write_named_estimate(tmp_path, ("validation", "validate_3d", "750"), "new", 1_000_000.0)
    write_named_estimate(tmp_path, ("validation", "validate_3d", "750"), "last", 2_000_000.0)
    write_named_estimate(tmp_path, ("profiling_manual", "large_scale"), "new", 10_000_000.0)
    write_named_estimate(tmp_path, ("profiling_manual", "large_scale"), "last", 20_000_000.0)

    comparisons = collect_criterion_comparisons(tmp_path / "criterion", "last")

    assert [comparison.benchmark for comparison in comparisons] == ["validation/validate_3d/750"]
    comparison = comparisons[0]
    assert (-comparison.percent_reduction) == pytest.approx(-50.0)
    assert comparison.speedup == pytest.approx(2.0)


def test_collect_performance_rows_preserves_all_coverage_and_single_directory_ids(tmp_path: Path) -> None:
    """Retained rows should use Criterion metadata and keep both one-sided classes."""
    write_named_estimate(tmp_path, ("validation", "matched", "750"), "new", 1_000_000.0)
    write_named_estimate(tmp_path, ("validation", "matched", "750"), "last", 2_000_000.0)
    write_named_estimate(tmp_path, ("validation_current_only",), "new", 3_000_000.0, full_id="validation/current_only")
    write_named_estimate(tmp_path, ("validation_baseline_only",), "last", 4_000_000.0, full_id="validation/baseline_only")
    write_named_estimate(tmp_path, ("profiling_manual", "excluded"), "new", 5_000_000.0)

    rows = collect_performance_rows(tmp_path / "criterion", "last")

    assert [row.benchmark_id for row in rows] == ["validation/baseline_only", "validation/current_only", "validation/matched/750"]
    assert [row.coverage_status for row in rows] == ["baseline-only", "current-only", "comparable"]
    assert rows[0].baseline is not None
    assert rows[0].current is None
    assert rows[1].baseline is None
    assert rows[1].current is not None
    assert rows[2].baseline is not None
    assert rows[2].current is not None
    assert rows[2].baseline.confidence_level == pytest.approx(0.95)


def test_collect_performance_rows_marks_name_matches_unverified_when_provenance_differs(tmp_path: Path) -> None:
    write_named_estimate(tmp_path, ("validation", "matched", "750"), "new", 1_000_000.0)
    write_named_estimate(tmp_path, ("validation", "matched", "750"), "last", 2_000_000.0)

    rows = collect_performance_rows(tmp_path / "criterion", "last", comparison_note="measurement hosts differ")

    assert rows[0].coverage_status == "not-comparable"
    assert rows[0].coverage_note == "measurement hosts differ"


def test_collect_performance_rows_marks_confidence_level_mismatch_unverified(tmp_path: Path) -> None:
    write_named_estimate(tmp_path, ("validation", "matched", "750"), "new", 1_000_000.0, confidence_level=0.99)
    write_named_estimate(tmp_path, ("validation", "matched", "750"), "last", 2_000_000.0, confidence_level=0.95)

    rows = collect_performance_rows(tmp_path / "criterion", "last")

    assert rows[0].coverage_status == "not-comparable"
    assert "confidence levels differ" in rows[0].coverage_note


def test_release_signal_excludes_manual_topology_benchmark() -> None:
    """Release comparisons should not run the explicitly selected topology suite."""
    assert "topology_guarantee_construction" not in benchmark_utils.RELEASE_SIGNAL_BENCH_TARGETS
    assert "topology_guarantee_construction" not in benchmark_utils.RELEASE_SIGNAL_GROUP_PREFIXES
    assert benchmark_utils.BENCH_TARGET_SUITES["topology"] == ("topology_guarantee_construction",)
    assert benchmark_utils.BENCH_COMPARE_GROUP_PREFIXES_BY_SUITE["topology"] == ("topology_guarantee_construction",)


def test_release_signal_includes_realization_validation_benchmark() -> None:
    """Release comparisons should retain the focused Level 4 regression signal."""
    assert "realization_validation" in benchmark_utils.RELEASE_SIGNAL_BENCH_TARGETS
    assert "realization_" in benchmark_utils.RELEASE_SIGNAL_GROUP_PREFIXES


def test_release_measurement_plan_matches_workflow_and_just_recipe() -> None:
    """Workflow and Just should delegate execution to the immutable Python plan."""
    project_root = Path(__file__).resolve().parents[2]
    workflow = (project_root / ".github" / "workflows" / "release-benchmarks.yml").read_text(encoding=UTF8)
    workflow_step = workflow.split("- name: Generate release benchmark summary", maxsplit=1)[1].split(
        "- name: Package release Criterion baseline",
        maxsplit=1,
    )[0]
    justfile = (project_root / "justfile").read_text(encoding=UTF8)
    just_recipe_match = re.search(r"^bench-latest\b.*?(?=^# )", justfile, flags=re.MULTILINE | re.DOTALL)
    assert just_recipe_match is not None
    just_recipe = just_recipe_match.group()

    assert "just bench-latest" in workflow_step
    assert "cargo bench --profile perf --bench" not in workflow_step
    assert "benchmark-utils run-release-signal" in just_recipe
    assert "cargo bench --profile perf --bench" not in just_recipe
    assert tuple(measurement.command for measurement in benchmark_utils.RELEASE_SIGNAL_MEASUREMENT_PLAN) == benchmark_utils.RELEASE_ASSET_MEASUREMENT_COMMANDS
    assert len({measurement.target for measurement in benchmark_utils.RELEASE_SIGNAL_MEASUREMENT_PLAN}) == len(
        benchmark_utils.RELEASE_SIGNAL_MEASUREMENT_PLAN,
    )
    assert len({measurement.report_section for measurement in benchmark_utils.RELEASE_SIGNAL_MEASUREMENT_PLAN}) == len(
        benchmark_utils.RELEASE_SIGNAL_MEASUREMENT_PLAN,
    )
    assert all(measurement.report_section for measurement in benchmark_utils.RELEASE_SIGNAL_MEASUREMENT_PLAN)
    assert all(measurement.required_group_prefixes for measurement in benchmark_utils.RELEASE_SIGNAL_MEASUREMENT_PLAN)
    assert all(measurement.sampling_mode == "full" for measurement in benchmark_utils.RELEASE_SIGNAL_MEASUREMENT_PLAN)
    assert all(not measurement.criterion_arguments for measurement in benchmark_utils.RELEASE_SIGNAL_MEASUREMENT_PLAN)
    assert "--run-benchmarks" not in workflow_step

    expected_group_contracts = {
        "ci_performance_suite": (
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
        "circumsphere_containment": ("random", "2d", "3d", "4d", "5d", "edge_cases_", "circumcenter"),
        "cold_path_predicates": ("predicates",),
        "locate": ("locate",),
        "realization_validation": ("realization_",),
    }
    assert {measurement.target: measurement.required_group_prefixes for measurement in benchmark_utils.RELEASE_SIGNAL_MEASUREMENT_PLAN} == (
        expected_group_contracts
    )


@patch("benchmark_utils.run_cargo_command")
@patch("benchmark_utils.run_cargo_live")
def test_release_preflight_runs_every_target_without_writing_evidence(mock_live: MagicMock, mock_cargo: MagicMock, tmp_path: Path) -> None:
    mock_live.side_effect = mock_cargo
    mock_cargo.return_value = completed_process(stdout=CI_MANIFEST_STDOUT)
    assert benchmark_utils.run_release_signal_measurement_plan(tmp_path, preflight_only=True) == {}
    assert [call.args[0] for call in mock_cargo.call_args_list] == [
        [*measurement.command[1:], "--", "--test"] for measurement in benchmark_utils.RELEASE_SIGNAL_MEASUREMENT_PLAN
    ]
    assert list(tmp_path.iterdir()) == []


@patch("benchmark_utils.run_cargo_command")
def test_preflight_cannot_be_saved_as_baseline(mock_cargo: MagicMock, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="preflight cannot save"):
        benchmark_utils.run_release_signal_measurement_plan(tmp_path, preflight_only=True, save_baseline="last")
    mock_cargo.assert_not_called()


@pytest.mark.parametrize("logging_enabled", [False, True])
def test_benchmark_abort_prints_underlying_error_without_subscriber(tmp_path: Path, logging_enabled: bool) -> None:
    """Run the actual shared fatal adapter, including with tracing filtered off."""
    adapter = Path(__file__).resolve().parents[2] / "benches" / "common" / "bench_utils.rs"
    source = tmp_path / "abort.rs"
    source.write_text(
        f'#[path = "{adapter.as_posix()}"] mod bench_utils;\n'
        "use bench_utils::OrAbort;\n"
        "fn main() {\n"
        '    let result: Result<(), std::io::Error> = Err(std::io::Error::other("fixture contract failed"));\n'
        "    result.or_abort();\n"
        "}\n",
        encoding="utf-8",
    )
    binary = tmp_path / ("abort.exe" if os.name == "nt" else "abort")
    args = [str(source), "--edition", "2024", "-o", str(binary)]
    if logging_enabled:
        args.extend(["--cfg", 'feature="bench-logging"'])
    run_safe_command("rustc", args, timeout=60)
    result = run_safe_command(str(binary), [], env={**os.environ, "RUST_LOG": "off"}, check=False, timeout=10)
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "benchmark failed: fixture contract failed\n"


def test_release_measurement_plan_digest_binds_per_target_sampling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    full_sampling_digest = benchmark_utils._measurement_plan_digest(tmp_path, "release-signal")
    first, *remaining = benchmark_utils.RELEASE_SIGNAL_MEASUREMENT_PLAN
    monkeypatch.setattr(
        benchmark_utils,
        "RELEASE_SIGNAL_MEASUREMENT_PLAN",
        (
            replace(
                first,
                sampling_mode="reduced",
                criterion_arguments=("--sample-size", "10"),
            ),
            *remaining,
        ),
    )

    assert benchmark_utils._measurement_plan_digest(tmp_path, "release-signal") != full_sampling_digest


def test_release_transition_digests_only_the_targets_shared_by_both_revisions(tmp_path: Path) -> None:
    """A newly added harness must not alter provenance for the historical shared plan."""
    current = tmp_path / "current"
    baseline = tmp_path / "baseline"
    for checkout in (current, baseline):
        benches = checkout / "benches"
        benches.mkdir(parents=True)
        (benches / "ci_performance_suite.rs").write_text("fn shared() {}\n", encoding=UTF8)
    (current / "benches" / "realization_validation.rs").write_text("fn added() {}\n", encoding=UTF8)
    shared_targets = ("ci_performance_suite",)
    for checkout in (current, baseline):
        suites = (("ci_performance_suite", ("validation/case",)),)
        if checkout == current:
            suites += (("realization_validation", ("realization_validation/case",)),)
        inventory = CriterionInventory(suites)
        criterion = checkout / "target/criterion"
        criterion.mkdir(parents=True)
        (criterion / benchmark_utils.MEASUREMENT_INVENTORY_FILE).write_bytes(serialize_inventory(inventory))
        (criterion / benchmark_utils.MEASUREMENT_POLICY_FILE).write_bytes(serialize_policy(benchmark_utils._sampling_policy(inventory.expected)))

    assert benchmark_utils._benchmark_harness_digest(
        current,
        "release-signal",
        shared_targets,
    ) == benchmark_utils._benchmark_harness_digest(
        baseline,
        "release-signal",
        shared_targets,
    )
    assert benchmark_utils._measurement_plan_digest(
        current,
        "release-signal",
        shared_targets,
    ) == benchmark_utils._measurement_plan_digest(
        baseline,
        "release-signal",
        shared_targets,
    )


def test_measurement_child_routes_cargo_and_scientific_diagnostics_to_private_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared runner's child must not overwrite the last successful generation."""
    criterion = tmp_path / "target/private-output"
    run = Mock(return_value=completed_process(CI_MANIFEST_STDOUT))
    monkeypatch.setattr(benchmark_utils, "run_cargo_command", run)
    args = create_argument_parser().parse_args(
        ["sample-suite", "--criterion-dir", str(criterion), "--targets", "ci_performance_suite", "--save-baseline", "last"]
    )
    execute_command(args, tmp_path)
    assert run.call_args.kwargs["env"]["CRITERION_HOME"] == str(criterion)
    assert run.call_args.args[0][-2:] == ["--", "--save-baseline=last"]
    assert json.loads((criterion / benchmark_utils.MEASUREMENT_OUTPUT_FILE).read_bytes()) == {"ci_performance_suite": CI_MANIFEST_STDOUT}
    assert benchmark_utils._load_ci_performance_metrics(criterion)
    assert benchmark_utils._load_ci_performance_manifest_ids(criterion)
    assert benchmark_utils._load_ci_performance_run_metadata(criterion)["sampling_mode"] == "full"
    assert not (tmp_path / "target/criterion").exists()


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param((1, CI_MANIFEST_STDOUT, "exited with status 1"), id="cargo-failure"),
        pytest.param((0, CI_MANIFEST_STDOUT.splitlines(keepends=True)[1], "no api_benchmark manifest"), id="missing-manifest"),
        pytest.param((0, CI_MANIFEST_STDOUT.splitlines(keepends=True)[0], "no construction metrics"), id="missing-metrics"),
    ],
)
def test_measurement_child_rejects_failed_or_incomplete_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: tuple[int, str, str],
) -> None:
    """An invalid first target must preserve diagnostics and stop the measurement."""
    returncode, stdout, message = failure
    criterion = tmp_path / "target/private-output"
    run = Mock(return_value=completed_process(stdout, returncode=returncode, stderr="child diagnostic\n"))
    monkeypatch.setattr(benchmark_utils, "run_cargo_command", run)
    args = create_argument_parser().parse_args(["sample-suite", "--criterion-dir", str(criterion), "--targets", "ci_performance_suite", "cold_path_predicates"])

    with pytest.raises(RuntimeError, match=message):
        execute_command(args, tmp_path)

    run.assert_called_once()
    assert run.call_args.args[0][-1] == "ci_performance_suite"
    captured = capsys.readouterr()
    assert stdout in captured.out
    assert "child diagnostic\n" in captured.err
    assert not (criterion / benchmark_utils.MEASUREMENT_OUTPUT_FILE).exists()


@pytest.mark.parametrize("as_bytes", [False, True], ids=["text", "bytes"])
def test_measurement_child_preserves_timeout_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    as_bytes: bool,
) -> None:
    """Captured output from timed-out Cargo runs remains visible to the parent."""
    criterion = tmp_path / "target/private-output"
    stdout = "completed fixture diagnostic\n"
    stderr = "incomplete fixture diagnostic\n"
    error = subprocess.TimeoutExpired(
        ["cargo", "bench"],
        30,
        output=stdout.encode(UTF8) if as_bytes else stdout,
        stderr=stderr.encode(UTF8) if as_bytes else stderr,
    )
    run = Mock(side_effect=error)
    monkeypatch.setattr(benchmark_utils, "run_cargo_command", run)
    args = create_argument_parser().parse_args(["sample-suite", "--criterion-dir", str(criterion), "--targets", "ci_performance_suite", "cold_path_predicates"])

    with pytest.raises(subprocess.TimeoutExpired) as raised:
        execute_command(args, tmp_path)

    assert raised.value is error
    run.assert_called_once()
    assert run.call_args.args[0][-1] == "ci_performance_suite"
    captured = capsys.readouterr()
    assert stdout in captured.err
    assert stderr in captured.err
    assert not (criterion / benchmark_utils.MEASUREMENT_OUTPUT_FILE).exists()


def test_realization_validation_stable_ids_survive_saved_baseline_comparison(tmp_path: Path) -> None:
    """Level 4 comparisons should match on dimension and fixed input size."""
    benchmark_ids = (
        ("realization_validation", "2d", "500v"),
        ("realization_validation", "3d", "20v"),
        ("realization_validation", "4d", "10v"),
        ("realization_validation", "5d", "8v"),
    )
    for benchmark_id in benchmark_ids:
        write_named_estimate(tmp_path, benchmark_id, "new", 1_000_000.0)
        write_named_estimate(tmp_path, benchmark_id, "last", 2_000_000.0)

    comparisons = collect_criterion_comparisons(tmp_path / "criterion", "last")

    assert {comparison.benchmark for comparison in comparisons} == {"/".join(benchmark_id) for benchmark_id in benchmark_ids}


@pytest.mark.parametrize("point_estimate", [float("nan"), float("inf"), 0.0, -1.0])
def test_bench_compare_cli_rejects_invalid_point_estimates(tmp_path: Path, capsys: pytest.CaptureFixture[str], point_estimate: float) -> None:
    """Non-physical timings fail at the CLI without replacing the previous report."""
    write_named_estimate(tmp_path, ("validation", "validate_3d", "750"), "new", point_estimate)
    write_named_estimate(tmp_path, ("validation", "validate_3d", "750"), "last", 2_000_000.0)
    output = tmp_path / "report.md"
    output.write_text("previous report\n", encoding=UTF8)
    args = create_argument_parser().parse_args(["bench-compare", "last", "--criterion-dir", str(tmp_path / "criterion"), "--output", str(output)])

    with pytest.raises(SystemExit) as failure:
        execute_command(args, tmp_path)

    assert failure.value.code == 1
    assert output.read_text(encoding=UTF8) == "previous report\n"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "bench-compare:" in captured.err
    assert re.search(r"finite|positive|JSON", captured.err)
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("stage", ["sample", "publication"])
def test_bench_compare_cli_reports_file_failure_without_replacing_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], stage: str
) -> None:
    """Incomplete inputs and an unwritable output produce contextual stderr."""
    for sample in ("new", "last"):
        write_named_estimate(tmp_path, ("validation", "validate_3d", "750"), sample, 1_000_000.0)
    output = tmp_path / "report.md"
    output.write_text("previous report\n", encoding=UTF8)
    publisher = Mock(side_effect=PermissionError(13, "publication denied", str(output)))
    renderer = Mock(return_value="new report\n")
    monkeypatch.setattr(benchmark_utils, "_write_text_atomic", publisher)
    monkeypatch.setattr(benchmark_utils, "render_criterion_comparison_report", renderer)
    missing = tmp_path / "criterion" / "validation" / "validate_3d" / "750" / "new" / "sample.json"
    if stage == "sample":
        missing.unlink()
    args = create_argument_parser().parse_args(["bench-compare", "last", "--criterion-dir", str(tmp_path / "criterion"), "--output", str(output)])

    with pytest.raises(SystemExit) as failure:
        execute_command(args, tmp_path)

    assert failure.value.code == 1
    assert output.read_text(encoding=UTF8) == "previous report\n"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "bench-compare:" in captured.err
    assert repr(str(missing if stage == "sample" else output)) in captured.err
    assert "Traceback" not in captured.err
    if stage == "sample":
        renderer.assert_not_called()
        publisher.assert_not_called()
    else:
        renderer.assert_called_once()
        publisher.assert_called_once_with(output, "new report\n")
        assert "publication denied" in captured.err


def test_render_criterion_comparison_report_includes_release_workflow_footer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Markdown report should advertise the la-stack-compatible command surface."""
    (tmp_path / "Cargo.toml").write_text('[package]\nversion = "0.8.0"\n', encoding=UTF8)
    monkeypatch.setattr(benchmark_utils, "_get_git_info", lambda _root: ("abc1234", "main"))
    monkeypatch.setattr(
        benchmark_utils,
        "_benchmark_report_environment_lines",
        lambda _root: [
            "## Environment",
            "",
            "- **Cargo profile**: `perf`",
            "- **Raw Criterion data**: `target/criterion/`",
            "- **OS**: test-os",
            "- **CPU**: test-cpu (8 cores, 8 threads)",
            "- **Memory**: test-memory",
            "- **Rust**: rustc test",
            "- **Target**: test-target",
        ],
    )
    write_named_estimate(tmp_path, ("validation", "validate_3d", "750"), "new", 1_000_000.0)
    write_named_estimate(tmp_path, ("validation", "validate_3d", "750"), "last", 2_000_000.0)
    comparisons = collect_criterion_comparisons(tmp_path / "criterion", "last")
    assert [comparison.benchmark for comparison in comparisons] == ["validation/validate_3d/750"]

    report = render_criterion_comparison_report(
        tmp_path,
        comparisons,
        CriterionReportSettings(
            baseline_name="last",
            stat="median",
            suite="release-signal",
            scope="release-signal",
        ),
    )

    assert "**delaunay** v0.8.0" in report
    assert "- **Raw Criterion data**: `target/criterion/`" in report
    assert "- **Rust**: rustc test" in report
    assert "| validate_3d/750 | 2.00 ms | 1.00 ms | **-50.0%** | 2.00x |" in report
    assert "just performance-local" in report
    assert "just performance-doc" in report
    assert "just performance-github-assets" in report
    assert "just performance-release <current-tag> <previous-tag>" in report
    assert "Existing legacy archives remain loadable as provenance-limited absolute timing" in report
    assert "New release archives must contain the supported versioned measurement" in report
    assert "Each release archive must contain" not in report
    assert "GitHub-asset ratios are always suppressed" in report
    assert "legacy archives without it are rejected" not in report
    assert "Ratios are emitted only when" not in report


def test_bench_compare_cli_reports_no_baseline(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Missing saved Criterion baselines should fail with an actionable hint."""
    (tmp_path / "Cargo.toml").write_text('[package]\nversion = "0.8.0"\n', encoding=UTF8)
    write_named_estimate(tmp_path, ("validation", "validate_3d", "750"), "new", 1_000_000.0)
    output = tmp_path / "target" / "bench-reports" / "performance.md"

    args = create_argument_parser().parse_args(["bench-compare", "last", "--criterion-dir", str(tmp_path / "criterion"), "--output", str(output)])
    with pytest.raises(SystemExit) as failure:
        execute_command(args, tmp_path)

    assert failure.value.code == 2
    assert not output.exists()
    captured = capsys.readouterr()
    assert "No comparison data found for baseline 'last'" in captured.err
    assert "just bench-save-baseline last" in captured.err


def test_normalize_release_tag_accepts_bare_and_prefixed_semver() -> None:
    assert normalize_release_tag("0.8.0") == "v0.8.0"
    assert normalize_release_tag("v0.8.0") == "v0.8.0"
    assert normalize_release_tag("v1.2.3-rc.1+build.7") == "v1.2.3-rc.1+build.7"


def test_normalize_release_tag_rejects_saved_baseline_names() -> None:
    with pytest.raises(ValueError, match="semver tag"):
        normalize_release_tag("last")


def test_parse_performance_report_id_reads_current_and_baseline_tags() -> None:
    report_id = parse_performance_report_id(delaunay_report("0.8.0", "v0.7.8"))

    assert report_id.current_tag == "v0.8.0"
    assert report_id.baseline_tag == "v0.7.8"
    assert report_id.archive_name == "v0.8.0-vs-v0.7.8.md"


def test_resolve_performance_request_current_vs_latest_uses_package_version_and_latest_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """performance-local should compare the package version against latest published stable release."""
    (tmp_path / "Cargo.toml").write_text('[package]\nversion = "0.8.0"\n', encoding=UTF8)

    monkeypatch.setattr(
        benchmark_utils,
        "published_releases",
        lambda _root: (PublishedRelease("v0.7.8", datetime(2026, 2, 1, tzinfo=UTC)),),
    )

    request = resolve_performance_request(
        PerformanceRequestOptions(
            current_tag=None,
            baseline_tag=None,
            published_latest=False,
            infer_release=False,
            current_vs_latest=True,
            worktree_ref="HEAD",
            repo_root=tmp_path,
        )
    )

    assert request.current_tag == "v0.8.0"
    assert request.baseline_tag == "v0.7.8"
    assert request.tags_to_fetch == ("v0.7.8",)


def test_performance_local_accepts_same_version_scratch_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Commits identify scratch measurements even when both retain the same package version."""
    (tmp_path / "Cargo.toml").write_text('[package]\nversion = "0.8.0"\n', encoding=UTF8)
    monkeypatch.setattr(
        benchmark_utils,
        "published_releases",
        lambda _root: (
            PublishedRelease(
                tag="v0.8.0",
                published_at=datetime(2026, 8, 1, tzinfo=UTC),
            ),
        ),
    )
    request = resolve_performance_request(
        PerformanceRequestOptions(
            current_tag=None,
            baseline_tag=None,
            published_latest=False,
            infer_release=False,
            current_vs_latest=True,
            worktree_ref="HEAD",
            repo_root=tmp_path,
        )
    )
    assert request.current_tag == request.baseline_tag == "v0.8.0"
    assert request.worktree_ref == "HEAD"


def test_promote_performance_report_archives_previous_and_updates_index(tmp_path: Path) -> None:
    source = tmp_path / "target" / "bench-reports" / "performance.md"
    current = tmp_path / "docs" / "performance.md"
    archive_dir = tmp_path / "docs" / "archive" / "performance"

    source.parent.mkdir(parents=True)
    current.parent.mkdir(parents=True)
    archive_dir.mkdir(parents=True)
    source.write_text(delaunay_report("0.8.0", "v0.7.8"), encoding=UTF8)
    current.write_text(delaunay_report("0.7.8", "v0.7.7"), encoding=UTF8)
    (archive_dir / "v0.7.6-vs-v0.7.5.md").write_text(delaunay_report("0.7.6", "v0.7.5"), encoding=UTF8)
    artifacts = performance_artifacts.ArtifactPaths(payload=source.with_suffix(".comparison.json"), provenance=source.with_suffix(".evidence.json"))
    performance_artifacts.write_bundle(artifacts, retained_performance_bundle())
    durable = performance_artifacts.ArtifactPaths(
        payload=archive_dir / "data" / "v0.8.0-vs-v0.7.8.comparison.json",
        provenance=archive_dir / "data" / "v0.8.0-vs-v0.7.8.evidence.json",
    )
    promoted_report = benchmark_utils.render_performance_bundle(
        performance_artifacts.load_bundle(artifacts),
        evidence_paths=benchmark_utils._promoted_evidence_paths(durable, project_root=tmp_path),
        evidence_state="promoted",
    )
    source.write_text(promoted_report, encoding=UTF8)

    promoted = promote_performance_report(
        source=source,
        artifacts=artifacts,
        destinations=benchmark_utils.PerformancePromotionDestinations(
            project_root=tmp_path,
            current=current,
            archive_dir=archive_dir,
        ),
        expected=benchmark_utils.PerformanceReportId(current_tag="v0.8.0", baseline_tag="v0.7.8"),
    )

    assert promoted.archive_name == "v0.8.0-vs-v0.7.8.md"
    assert current.read_text(encoding=UTF8) == benchmark_utils._normalize_how_to_update(promoted_report)
    assert (archive_dir / "v0.7.8-vs-v0.7.7.md").read_text(encoding=UTF8) == delaunay_report("0.7.8", "v0.7.7")
    assert (archive_dir / "data" / "v0.8.0-vs-v0.7.8.comparison.json").read_bytes() == artifacts.payload.read_bytes()
    assert (archive_dir / "data" / "v0.8.0-vs-v0.7.8.evidence.json").read_bytes() == artifacts.provenance.read_bytes()
    assert (archive_dir / "README.md").read_text(encoding=UTF8) == (
        "# Archived Performance Reports\n\n"
        "Older release-to-release benchmark comparisons are archived here.\n"
        "`docs/performance.md` contains the latest curated comparison.\n\n"
        "- [v0.7.6-vs-v0.7.5](v0.7.6-vs-v0.7.5.md)\n"
        "- [v0.7.8-vs-v0.7.7](v0.7.8-vs-v0.7.7.md)\n"
    )


def test_promote_performance_report_rolls_back_every_destination_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed archive-index update must leave all promoted files unchanged."""
    source = tmp_path / "performance.md"
    current = tmp_path / "docs" / "performance.md"
    archive_dir = tmp_path / "docs" / "archive" / "performance"
    index = archive_dir / "README.md"
    archived = archive_dir / "v0.7.8-vs-v0.7.7.md"
    current.parent.mkdir(parents=True)
    archive_dir.mkdir(parents=True)
    source.write_text(delaunay_report("0.8.0", "v0.7.8"), encoding=UTF8)
    current.write_text(delaunay_report("0.7.8", "v0.7.7"), encoding=UTF8)
    index.write_text("old index\n", encoding=UTF8)
    artifacts = performance_artifacts.ArtifactPaths(payload=source.with_suffix(".comparison.json"), provenance=source.with_suffix(".evidence.json"))
    performance_artifacts.write_bundle(artifacts, retained_performance_bundle())
    durable = performance_artifacts.ArtifactPaths(
        payload=archive_dir / "data" / "v0.8.0-vs-v0.7.8.comparison.json",
        provenance=archive_dir / "data" / "v0.8.0-vs-v0.7.8.evidence.json",
    )
    source.write_text(
        benchmark_utils.render_performance_bundle(
            performance_artifacts.load_bundle(artifacts),
            evidence_paths=benchmark_utils._promoted_evidence_paths(durable, project_root=tmp_path),
            evidence_state="promoted",
        ),
        encoding=UTF8,
    )

    def fail_index_update(_archive_dir: Path, _additional: tuple[str, ...] = ()) -> str:
        msg = "index write failed"
        raise OSError(msg)

    monkeypatch.setattr(benchmark_utils, "_archive_index_text", fail_index_update)

    with pytest.raises(OSError, match="index write failed"):
        promote_performance_report(
            source=source,
            artifacts=artifacts,
            destinations=benchmark_utils.PerformancePromotionDestinations(
                project_root=tmp_path,
                current=current,
                archive_dir=archive_dir,
            ),
            expected=benchmark_utils.PerformanceReportId(current_tag="v0.8.0", baseline_tag="v0.7.8"),
        )

    assert current.read_text(encoding=UTF8) == delaunay_report("0.7.8", "v0.7.7")
    assert index.read_text(encoding=UTF8) == "old index\n"
    assert not archived.exists()
    assert not durable.payload.exists()
    assert not durable.provenance.exists()


def test_promote_performance_report_rejects_conflicting_existing_archive_before_mutation(tmp_path: Path) -> None:
    source = tmp_path / "performance.md"
    current = tmp_path / "docs" / "performance.md"
    archive_dir = tmp_path / "docs" / "archive" / "performance"
    current.parent.mkdir(parents=True)
    archive_dir.mkdir(parents=True)
    current.write_text(delaunay_report("0.7.8", "v0.7.7"), encoding=UTF8)
    conflicting = archive_dir / "v0.7.8-vs-v0.7.7.md"
    conflicting.write_text("conflicting archive\n", encoding=UTF8)
    artifacts = performance_artifacts.ArtifactPaths(payload=source.with_suffix(".comparison.json"), provenance=source.with_suffix(".evidence.json"))
    performance_artifacts.write_bundle(artifacts, retained_performance_bundle())
    durable = performance_artifacts.ArtifactPaths(
        payload=archive_dir / "data" / "v0.8.0-vs-v0.7.8.comparison.json",
        provenance=archive_dir / "data" / "v0.8.0-vs-v0.7.8.evidence.json",
    )
    source.write_text(
        benchmark_utils.render_performance_bundle(
            performance_artifacts.load_bundle(artifacts),
            evidence_paths=benchmark_utils._promoted_evidence_paths(durable, project_root=tmp_path),
            evidence_state="promoted",
        ),
        encoding=UTF8,
    )
    prior_current = current.read_bytes()

    with pytest.raises(ValueError, match="existing performance archive conflicts"):
        promote_performance_report(
            source=source,
            artifacts=artifacts,
            destinations=benchmark_utils.PerformancePromotionDestinations(
                project_root=tmp_path,
                current=current,
                archive_dir=archive_dir,
            ),
            expected=benchmark_utils.PerformanceReportId(current_tag="v0.8.0", baseline_tag="v0.7.8"),
        )

    assert current.read_bytes() == prior_current
    assert conflicting.read_text(encoding=UTF8) == "conflicting archive\n"
    assert not (archive_dir / "data").exists()


def test_release_generation_preflights_output_aliases_before_measurement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    destination = tmp_path / "docs" / "performance.md"

    def unexpected_measurement(*_args: object, **_kwargs: object) -> None:
        msg = "measurement must not start"
        raise AssertionError(msg)

    monkeypatch.setattr(benchmark_utils, "_build_performance_bundle_in_temp_worktree", unexpected_measurement)

    with pytest.raises(ValueError, match="must use distinct paths"):
        benchmark_utils.generate_and_promote_performance_report(
            output=destination,
            current=destination,
            archive_dir=tmp_path / "docs" / "archive" / "performance",
            config=ReleaseReportConfig(
                repo_root=tmp_path,
                current_tag="v0.8.0",
                baseline_tag="v0.7.8",
                worktree_ref="HEAD",
            ),
        )


def test_release_generation_rejects_external_scratch_before_measurement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    measurement = Mock()
    monkeypatch.setattr(benchmark_utils, "_build_performance_bundle_in_temp_worktree", measurement)
    with pytest.raises(ValueError, match="Markdown output must be contained"):
        benchmark_utils.generate_and_promote_performance_report(
            output=tmp_path / "outside.md",
            current=project_root / "docs/performance.md",
            archive_dir=project_root / "docs/archive/performance",
            config=ReleaseReportConfig(repo_root=project_root, current_tag="v0.8.0", baseline_tag="v0.7.8", worktree_ref="HEAD"),
        )
    measurement.assert_not_called()
    assert not (tmp_path / "outside.md").exists()


def test_performance_doc_rejects_external_inputs_before_reading(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    with pytest.raises(ValueError, match="artifact payload input must be contained"):
        benchmark_utils.render_and_promote_performance_artifacts(
            output=project_root / "target/performance.md",
            artifacts=performance_artifacts.ArtifactPaths(payload=tmp_path / "missing.comparison.json", provenance=tmp_path / "missing.evidence.json"),
            destinations=benchmark_utils.PerformancePromotionDestinations(
                project_root=project_root, current=project_root / "docs/performance.md", archive_dir=project_root / "docs/archive/performance"
            ),
            expected_current_tag="v0.8.0",
        )
    assert list(project_root.iterdir()) == []


@pytest.mark.parametrize("escape", ["traversal", "symlink"])
def test_release_generation_rejects_archive_escape_before_measurement_or_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    escape: str,
) -> None:
    """Tracked performance destinations must remain below the explicit root."""
    project_root = tmp_path / "repo"
    project_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    if escape == "traversal":
        archive_dir = project_root / "docs" / "archive" / ".." / ".." / ".." / "outside"
    else:
        docs = project_root / "docs"
        docs.mkdir()
        (docs / "archive").symlink_to(outside, target_is_directory=True)
        archive_dir = docs / "archive" / "performance"

    measurement = Mock()
    publication = Mock()
    monkeypatch.setattr(benchmark_utils, "_build_performance_bundle_in_temp_worktree", measurement)
    monkeypatch.setattr(benchmark_utils, "_publish_performance_bundle", publication)

    with pytest.raises(ValueError, match="contained by repository root"):
        benchmark_utils.generate_and_promote_performance_report(
            output=project_root / "target" / "bench-reports" / "performance.md",
            current=project_root / "docs" / "performance.md",
            archive_dir=archive_dir,
            config=ReleaseReportConfig(
                repo_root=project_root,
                current_tag="v0.8.0",
                baseline_tag="v0.7.8",
                worktree_ref="HEAD",
            ),
        )

    measurement.assert_not_called()
    publication.assert_not_called()


def test_performance_doc_promotes_retained_bundle_without_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Retained artifacts should be sufficient to rebuild and promote docs."""
    output = tmp_path / "target" / "bench-reports" / "performance.md"
    artifacts = performance_artifacts.ArtifactPaths(
        payload=output.with_suffix(".comparison.json"),
        provenance=output.with_suffix(".evidence.json"),
    )
    current = tmp_path / "docs" / "performance.md"
    archive_dir = tmp_path / "docs" / "archive" / "performance"
    current.parent.mkdir(parents=True)
    current.write_text(delaunay_report("0.7.8", "v0.7.7"), encoding=UTF8)
    performance_artifacts.write_bundle(artifacts, retained_performance_bundle())

    def unexpected_command(*_args: object, **_kwargs: object) -> None:
        msg = "performance-doc must not run commands"
        raise AssertionError(msg)

    monkeypatch.setattr(benchmark_utils, "run_safe_command", unexpected_command)
    monkeypatch.setattr(benchmark_utils, "run_git_command", unexpected_command)

    report_id = benchmark_utils.render_and_promote_performance_artifacts(
        output=output,
        artifacts=artifacts,
        destinations=benchmark_utils.PerformancePromotionDestinations(
            project_root=tmp_path,
            current=current,
            archive_dir=archive_dir,
        ),
        expected_current_tag="v0.8.0",
    )

    assert report_id.archive_name == "v0.8.0-vs-v0.7.8.md"
    assert output.read_text(encoding=UTF8) == current.read_text(encoding=UTF8)
    assert (archive_dir / "v0.7.8-vs-v0.7.7.md").is_file()


def test_performance_doc_rejects_same_version_without_changing_outputs(tmp_path: Path) -> None:
    """Local same-version evidence is retainable but cannot be promoted."""
    output = tmp_path / "performance.md"
    output.write_text("prior output\n", encoding=UTF8)
    artifacts = performance_artifacts.ArtifactPaths(
        payload=tmp_path / "performance.comparison.json",
        provenance=tmp_path / "performance.evidence.json",
    )
    performance_artifacts.write_bundle(artifacts, retained_performance_bundle(current="v0.8.0", baseline="v0.8.0"))
    current = tmp_path / "docs" / "performance.md"
    current.parent.mkdir()
    current.write_text("prior docs\n", encoding=UTF8)

    with pytest.raises(ValueError, match="same-version"):
        benchmark_utils.render_and_promote_performance_artifacts(
            output=output,
            artifacts=artifacts,
            destinations=benchmark_utils.PerformancePromotionDestinations(
                project_root=tmp_path,
                current=current,
                archive_dir=tmp_path / "docs" / "archive" / "performance",
            ),
            expected_current_tag="v0.8.0",
        )

    assert output.read_text(encoding=UTF8) == "prior output\n"
    assert current.read_text(encoding=UTF8) == "prior docs\n"


def test_performance_doc_rejects_malformed_pair_without_changing_outputs(tmp_path: Path) -> None:
    """Malformed retained inputs must fail before scratch or tracked reports change."""
    output = tmp_path / "performance.md"
    output.write_text("prior output\n", encoding=UTF8)
    artifacts = performance_artifacts.ArtifactPaths(
        payload=tmp_path / "performance.comparison.json",
        provenance=tmp_path / "performance.evidence.json",
    )
    performance_artifacts.write_bundle(artifacts, retained_performance_bundle())
    artifacts.payload.write_text("not,the,canonical,schema\n", encoding=UTF8)
    current = tmp_path / "docs" / "performance.md"
    current.parent.mkdir()
    current.write_text("prior docs\n", encoding=UTF8)

    with pytest.raises(ValueError, match="SHA-256"):
        benchmark_utils.render_and_promote_performance_artifacts(
            output=output,
            artifacts=artifacts,
            destinations=benchmark_utils.PerformancePromotionDestinations(
                project_root=tmp_path,
                current=current,
                archive_dir=tmp_path / "docs" / "archive" / "performance",
            ),
            expected_current_tag="v0.8.0",
        )

    assert output.read_text(encoding=UTF8) == "prior output\n"
    assert current.read_text(encoding=UTF8) == "prior docs\n"


def test_performance_doc_rejects_stale_current_release_before_writing(tmp_path: Path) -> None:
    output = tmp_path / "performance.md"
    artifacts = performance_artifacts.ArtifactPaths(payload=tmp_path / "performance.comparison.json", provenance=tmp_path / "performance.evidence.json")
    performance_artifacts.write_bundle(artifacts, retained_performance_bundle(current="v0.7.9", baseline="v0.7.8"))
    current = tmp_path / "docs" / "performance.md"
    current.parent.mkdir()
    current.write_text("prior docs\n", encoding=UTF8)

    with pytest.raises(ValueError, match=r"independently expected release v0\.8\.0"):
        benchmark_utils.render_and_promote_performance_artifacts(
            output=output,
            artifacts=artifacts,
            destinations=benchmark_utils.PerformancePromotionDestinations(
                project_root=tmp_path,
                current=current,
                archive_dir=tmp_path / "docs" / "archive" / "performance",
            ),
            expected_current_tag="v0.8.0",
        )

    assert not output.exists()
    assert current.read_text(encoding=UTF8) == "prior docs\n"


def test_performance_doc_rejects_zero_comparable_rows_before_writing(tmp_path: Path) -> None:
    output = tmp_path / "performance.md"
    original = retained_performance_bundle()
    unverified = performance_artifacts.PerformanceBundle(
        context=original.context,
        rows=(replace(original.rows[0], coverage_status="not-comparable", coverage_note="measurement hosts differ"),),
    )
    artifacts = performance_artifacts.ArtifactPaths(payload=tmp_path / "performance.comparison.json", provenance=tmp_path / "performance.evidence.json")
    performance_artifacts.write_bundle(artifacts, unverified)

    with pytest.raises(ValueError, match="no scientifically comparable rows"):
        benchmark_utils.render_and_promote_performance_artifacts(
            output=output,
            artifacts=artifacts,
            destinations=benchmark_utils.PerformancePromotionDestinations(
                project_root=tmp_path,
                current=tmp_path / "docs" / "performance.md",
                archive_dir=tmp_path / "docs" / "archive" / "performance",
            ),
            expected_current_tag="v0.8.0",
        )

    assert not output.exists()


def test_renderer_suppresses_ratios_for_incompatible_measurement_hosts(tmp_path: Path) -> None:
    original = retained_performance_bundle()
    incompatible_context = replace(
        original.context,
        baseline_measurement_host=performance_artifacts.HostIdentity(
            status="recorded",
            cpu="Different CPU",
            operating_system="Test OS",
            architecture="test",
        ),
    )
    unverified = performance_artifacts.PerformanceBundle(
        context=incompatible_context,
        rows=(replace(original.rows[0], coverage_status="not-comparable", coverage_note="measurement hosts differ"),),
    )

    report = benchmark_utils.render_performance_bundle(
        unverified,
        evidence_paths=performance_artifacts.ArtifactPaths(
            payload=tmp_path / "performance.comparison.json",
            provenance=tmp_path / "performance.evidence.json",
        ),
        evidence_state="scratch",
    )

    assert "Ratios are suppressed" in report
    assert "2.00x" not in report
    assert "measurement hosts differ" in report


def test_renderer_distinguishes_scratch_and_promoted_evidence_paths(tmp_path: Path) -> None:
    scratch = performance_artifacts.ArtifactPaths(
        payload=tmp_path / "scratch" / "performance.comparison.json",
        provenance=tmp_path / "scratch" / "performance.evidence.json",
    )
    performance_artifacts.write_bundle(scratch, retained_performance_bundle())
    durable = performance_artifacts.ArtifactPaths(
        payload=tmp_path / "docs" / "archive" / "performance" / "data" / "v0.8.0-vs-v0.7.8.comparison.json",
        provenance=tmp_path / "docs" / "archive" / "performance" / "data" / "v0.8.0-vs-v0.7.8.evidence.json",
    )
    current = tmp_path / "docs" / "performance.md"

    scratch_report = benchmark_utils.render_performance_artifacts(scratch)
    promoted_report = benchmark_utils.render_performance_bundle(
        performance_artifacts.load_bundle(scratch),
        evidence_paths=benchmark_utils._promoted_evidence_paths(durable, project_root=tmp_path),
        evidence_state="promoted",
    )

    assert "Retained scratch evidence" in scratch_report
    assert scratch.payload.as_posix() in scratch_report
    assert "Promoted evidence" in promoted_report
    assert durable.payload.relative_to(tmp_path).as_posix() in promoted_report
    assert durable.payload.as_posix() not in promoted_report
    assert scratch.payload.as_posix() not in promoted_report


def test_renderer_hashes_exact_retained_bytes_including_json_whitespace(tmp_path: Path) -> None:
    payload, manifest = performance_artifacts.serialize_bundle(retained_performance_bundle())
    compact = json.dumps(json.loads(payload), separators=(",", ":")).encode()
    assert compact != payload
    evidence = replace(parse_evidence(payload, manifest), payload=compact)
    _, compact_manifest = serialize_evidence(evidence)
    loaded = performance_artifacts.load_bundle_bytes(compact, compact_manifest, source="compact fixture")
    report = benchmark_utils.render_performance_bundle(
        loaded,
        evidence_paths=performance_artifacts.ArtifactPaths(payload=tmp_path / "performance.comparison.json", provenance=tmp_path / "performance.evidence.json"),
        evidence_state="scratch",
    )
    assert f"Evidence SHA-256 `{hashlib.sha256(compact).hexdigest()}`" in report
    assert hashlib.sha256(payload).hexdigest() not in report
    assert replace(loaded).payload_sha256 is None


def test_prepare_github_release_assets_copies_current_and_baseline_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GitHub Release asset reports should compare extracted raw Criterion samples."""
    archives: dict[str, benchmark_utils.DownloadedReleaseAsset] = {}

    def write_asset(tag: str, point_ns: float) -> None:
        root = tmp_path / f"{tag}-asset-root"
        write_named_estimate(root, ("validation", "validate_3d", "750"), "new", point_ns)
        commit = ("a" if tag == "v0.8.0" else "b") * 40
        metadata = {
            "schema_version": benchmark_utils.RELEASE_ASSET_METADATA_SCHEMA_VERSION,
            "source": {
                "version": tag,
                "commit": commit,
                "ref": tag,
                "revision_timestamp": "2026-08-01T00:00:00Z",
                "git_clean": True,
                "source_state_sha256": hashlib.sha256(f"commit {commit}\n".encode()).hexdigest(),
                "limitation": "",
            },
            "measurement_commands": [list(command) for command in benchmark_utils.RELEASE_ASSET_MEASUREMENT_COMMANDS],
            "completed_targets": list(benchmark_utils.RELEASE_SIGNAL_BENCH_TARGETS),
            "toolchain": {
                "rustc": "rustc 1.98.0",
                "criterion_version": "0.7.0",
                "cargo_profile": "perf",
                "cargo_lock_sha256": "e" * 64,
                "harness_sha256": "f" * 64,
                "configuration_sha256": "a" * 64,
                "measurement_plan_sha256": "b" * 64,
                "limitation": "",
            },
            "measurement_host": {
                "status": "recorded",
                "cpu": "GitHub test CPU",
                "operating_system": "Linux",
                "architecture": "x86_64",
                "reason": "",
            },
            "criterion": {
                "content_sha256": benchmark_utils._directory_digest(root / "criterion"),
                "sample_name": "new",
            },
        }
        (root / "metadata.json").write_text(json.dumps(metadata), encoding=UTF8)
        archive = tmp_path / f"delaunay-{tag}-criterion-baseline.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(root / "criterion", arcname="criterion")
            tar.add(root / "metadata.json", arcname="metadata.json")
        command = (
            "gh",
            "release",
            "download",
            tag,
            "--pattern",
            archive.name,
            "--dir",
            str(tmp_path / "scratch"),
        )
        archives[tag] = benchmark_utils.DownloadedReleaseAsset(archive=archive, command=command)

    write_asset("v0.8.0", 1_000_000.0)
    write_asset("v0.7.8", 2_000_000.0)

    def fake_download_release_baseline(*, tag: str, download_dir: Path, repo_root: Path) -> benchmark_utils.DownloadedReleaseAsset:
        del download_dir, repo_root
        return archives[tag]

    monkeypatch.setattr(benchmark_utils, "_download_release_baseline", fake_download_release_baseline)
    monkeypatch.setattr(
        benchmark_utils,
        "_expected_tag_commit",
        lambda _repo_root, tag: ("a" if tag == "v0.8.0" else "b") * 40,
    )

    target_worktree = tmp_path / "worktree"
    scratch = tmp_path / "scratch"
    current_evidence, baseline_evidence = benchmark_utils._prepare_github_release_assets(
        config=ReleaseReportConfig(
            repo_root=tmp_path,
            current_tag="v0.8.0",
            baseline_tag="v0.7.8",
            worktree_ref="v0.8.0",
            apply_current_diff=False,
            baseline_source="github-assets",
        ),
        target_worktree=target_worktree,
        tmp_dir=scratch,
    )

    comparisons = collect_criterion_comparisons(target_worktree / "target" / "criterion", "v0.7.8")

    assert [comparison.benchmark for comparison in comparisons] == ["validation/validate_3d/750"]
    assert comparisons[0].current.point == pytest.approx(1_000_000.0)
    assert comparisons[0].baseline.point == pytest.approx(2_000_000.0)
    assert current_evidence.revision.source.version == "v0.8.0"
    assert baseline_evidence.revision.source.version == "v0.7.8"
    assert current_evidence.revision.commands == benchmark_utils.RELEASE_ASSET_MEASUREMENT_COMMANDS
    assert current_evidence.measurement_host.cpu == "GitHub test CPU"
    assert current_evidence.artifact.content_sha256 == benchmark_utils._directory_digest(tmp_path / "v0.8.0-asset-root" / "criterion")
    assert current_evidence.acquisition_commands[0] == archives["v0.8.0"].command
    assert "--pattern" in current_evidence.acquisition_commands[0]
    assert "--dir" in current_evidence.acquisition_commands[0]


def test_release_asset_evidence_loads_legacy_archive_as_limited_noncomparable_evidence(tmp_path: Path) -> None:
    extracted = tmp_path / "legacy"
    write_named_estimate(extracted, ("validation", "validate_3d", "750"), "new", 1_000_000.0)
    commit = "a" * 40
    (extracted / "metadata.json").write_text(
        json.dumps(
            {
                "tag": "v0.8.0",
                "commit": commit,
                "run_id": "12345",
                "generated_at": "2026-08-01T00:00:00Z",
                "cargo_profile": "perf",
                "sampling_mode": "full",
                "runner_os": "Linux",
                "runner_arch": "X64",
                "summary": "PERFORMANCE_RESULTS.md",
                "criterion_dir": "criterion",
            }
        ),
        encoding=UTF8,
    )
    archive = tmp_path / "legacy.tar.gz"
    archive.write_bytes(b"legacy archive placeholder")

    evidence = benchmark_utils._load_release_asset_evidence(
        requested_tag="v0.8.0",
        expected_commit=commit,
        extracted_root=extracted,
        archive=archive,
        acquisition_command=("gh", "release", "download", "v0.8.0"),
    )

    assert evidence.revision.source.limitation
    assert evidence.revision.toolchain.limitation
    assert evidence.revision.commands == ()
    assert evidence.revision.completed_targets == ()
    assert evidence.measurement_host.status == "unavailable"
    assert evidence.measurement_host.operating_system == "Linux"
    assert evidence.measurement_host.architecture == "X64"
    assert "12345" in evidence.measurement_host.reason
    assert evidence.artifact.content_sha256 == benchmark_utils._directory_digest(extracted / "criterion")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("commit", "b" * 40, "does not match tag"),
        ("git_clean", False, "does not identify a clean checkout"),
        ("source_state_sha256", "0" * 64, "digest is inconsistent with clean tag"),
    ],
)
def test_versioned_release_asset_binds_clean_source_to_requested_tag(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    extracted = tmp_path / "versioned"
    commit = "a" * 40
    metadata = write_versioned_release_asset_metadata(extracted, tag="v0.8.0", commit=commit)
    source = metadata["source"]
    assert isinstance(source, dict)
    source[field] = value
    (extracted / "metadata.json").write_text(json.dumps(metadata), encoding=UTF8)
    archive = tmp_path / "versioned.tar.gz"
    archive.write_bytes(b"versioned archive placeholder")

    with pytest.raises(ValueError, match=message):
        benchmark_utils._load_release_asset_evidence(
            requested_tag="v0.8.0",
            expected_commit=commit,
            extracted_root=extracted,
            archive=archive,
            acquisition_command=("gh", "release", "download", "v0.8.0"),
        )


def test_versioned_release_asset_rejects_changed_measurement_commands(tmp_path: Path) -> None:
    extracted = tmp_path / "versioned"
    commit = "a" * 40
    metadata = write_versioned_release_asset_metadata(extracted, tag="v0.8.0", commit=commit)
    metadata["measurement_commands"] = [["cargo", "bench", "--profile", "perf"]]
    (extracted / "metadata.json").write_text(json.dumps(metadata), encoding=UTF8)
    archive = tmp_path / "versioned.tar.gz"
    archive.write_bytes(b"versioned archive placeholder")

    with pytest.raises(ValueError, match="measurement commands do not match"):
        benchmark_utils._load_release_asset_evidence(
            requested_tag="v0.8.0",
            expected_commit=commit,
            extracted_root=extracted,
            archive=archive,
            acquisition_command=("gh", "release", "download", "v0.8.0"),
        )


def test_versioned_release_asset_rejects_placeholder_recorded_host(tmp_path: Path) -> None:
    extracted = tmp_path / "versioned"
    commit = "a" * 40
    metadata = write_versioned_release_asset_metadata(extracted, tag="v0.8.0", commit=commit)
    host = metadata["measurement_host"]
    assert isinstance(host, dict)
    host["cpu"] = "unknown"
    (extracted / "metadata.json").write_text(json.dumps(metadata), encoding=UTF8)
    archive = tmp_path / "versioned.tar.gz"
    archive.write_bytes(b"versioned archive placeholder")

    with pytest.raises(ValueError, match="placeholder"):
        benchmark_utils._load_release_asset_evidence(
            requested_tag="v0.8.0",
            expected_commit=commit,
            extracted_root=extracted,
            archive=archive,
            acquisition_command=("gh", "release", "download", "v0.8.0"),
        )


def write_complete_release_samples(tmp_path: Path) -> tuple[Path, CriterionInventory]:
    """Write raw cases with independent counts for every release-signal target."""
    ids = ("validation/validate_3d/750", "random/insphere", "predicates/hot/case", "locate/no_hint/3d/case", "realization_validation/case")
    for name in ids:
        write_named_estimate(tmp_path, tuple(name.split("/")), "new", 1_000_000.0)
    criterion = tmp_path / "criterion"
    inventory = CriterionInventory(tuple((target, (name,)) for target, name in zip(benchmark_utils.RELEASE_SIGNAL_BENCH_TARGETS, ids, strict=True)))
    policy = benchmark_utils._sampling_policy(inventory.expected)
    for name, count in zip(ids, (15, 100, 100, 10, 100), strict=True):
        directory = criterion / name / "new"
        (directory / "sample.json").write_text(json.dumps({"iters": [1] * count, "times": [100] * count}), encoding=UTF8)
        estimates = json.loads((directory / "estimates.json").read_bytes())
        estimates["mean"] = estimates["median"]
        (directory / "estimates.json").write_text(json.dumps(estimates), encoding=UTF8)
    (criterion / benchmark_utils.MEASUREMENT_INVENTORY_FILE).write_bytes(serialize_inventory(inventory))
    (criterion / benchmark_utils.MEASUREMENT_POLICY_FILE).write_bytes(serialize_policy(policy))
    return criterion, inventory


@pytest.mark.parametrize("tag", ["v0.8.0", "v0.8.0+build.7"])
@pytest.mark.parametrize("changed_input", [None, "src/lib.rs", "benches/ci_performance_suite.rs", "src/added.rs", "benches/common/added.rs"])
@pytest.mark.parametrize("execution_case", [("native", "perf"), ("windows", "perf"), ("native", "release"), ("windows", "release")])
def test_write_release_benchmark_metadata_binds_measurement_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tag: str,
    changed_input: str | None,
    execution_case: tuple[str, str],
) -> None:
    path_ordering, cargo_profile = execution_case
    if path_ordering == "windows":
        monkeypatch.setattr(type(tmp_path), "__lt__", lambda first, second: PureWindowsPath(first.as_posix()) < PureWindowsPath(second.as_posix()))
    (tmp_path / "Cargo.toml").write_text(f'[package]\nversion = "{tag.removeprefix("v")}"\n', encoding=UTF8)
    source_paths = tuple(Path(name) for name in ("Cargo.lock", "Cargo.toml", "rust-toolchain.toml", "src/lib.rs"))
    harness_paths = (*(Path(f"benches/{target}.rs") for target in benchmark_utils.RELEASE_SIGNAL_BENCH_TARGETS), *source_paths[:3])
    for path in set(source_paths + harness_paths) - {Path("Cargo.toml")}:
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text("// measured input\n", encoding=UTF8)
    criterion, inventory = write_complete_release_samples(tmp_path)
    policy = benchmark_utils._sampling_policy(inventory.expected)
    commit = "a" * 40
    source = replace(
        retained_performance_bundle().context.current_source,
        version=tag,
        ref=tag,
        git_clean=True,
        source_state_sha256=hashlib.sha256(f"commit {commit}\n".encode()).hexdigest(),
    )
    toolchain = retained_performance_bundle().context.current_toolchain
    monkeypatch.setattr(benchmark_utils, "_source_state", lambda *_args, **_kwargs: source)
    monkeypatch.setattr(benchmark_utils, "_expected_tag_commit", lambda *_args, **_kwargs: commit)
    monkeypatch.setattr(benchmark_utils, "_toolchain_state", lambda *_args, **_kwargs: toolchain)
    sample = Sample(tuple((name, read_estimate(criterion / name / "new/estimates.json")) for name in inventory.expected))
    assert toolchain.criterion_version is not None
    (tmp_path / "Cargo.lock").write_text(f'[[package]]\nname = "criterion"\nversion = "{toolchain.criterion_version}"\n', encoding=UTF8)
    monkeypatch.setattr(shared_measurement, "resolve_revision", lambda *_args: commit)
    monkeypatch.setattr(
        shared_measurement,
        "capture_host",
        lambda *_args, **_kwargs: HostMetadata("Test OS", "test", "Test CPU", 4, 8, 4096, (("rustc", toolchain.rustc),)),
    )
    captured = shared_measurement.capture_provenance(
        tmp_path,
        shared_measurement.MeasurementConfig(
            command=(
                sys.executable,
                str(Path(benchmark_utils.__file__).resolve()),
                "sample-suite",
                "--profile",
                cargo_profile,
                "--bench-timeout",
                "7200",
                "--criterion-dir",
                str(criterion),
                "--targets",
                *benchmark_utils.RELEASE_SIGNAL_BENCH_TARGETS,
            ),
            sources=tuple(path.as_posix() for path in source_paths),
            harness=tuple(path.as_posix() for path in harness_paths),
            dependencies=(("criterion", "Cargo.lock"),),
            context=(
                ("delaunay.inventory", json.dumps(json.loads(serialize_inventory(inventory)))),
                ("delaunay.sampling-policy", serialize_policy(policy).decode()),
                ("delaunay.scientific-preflight", "passed"),
                ("delaunay.sample", "new"),
                ("delaunay.raw-sample-sha256", benchmark_utils._criterion_sample_digest(criterion, "new")),
            ),
        ),
        tag,
        mode="release",
    )
    measurement = Evidence(
        serialize_sample(sample),
        SAMPLE_SCHEMA,
        (("measurement", captured),),
    )
    sample_payload, sample_manifest = serialize_evidence(measurement)
    (criterion / "delaunay-sample.json").write_bytes(sample_payload)
    (criterion / "delaunay-evidence.json").write_bytes(sample_manifest)
    output = tmp_path / "metadata.json"

    if cargo_profile != "perf":
        output.write_bytes(b"previous release metadata\n")
        with pytest.raises(ValueError, match="captured perf sample-suite command"):
            benchmark_utils.write_release_benchmark_metadata(repo_root=tmp_path, tag=tag, criterion_dir=criterion, output=output)
        assert output.read_bytes() == b"previous release metadata\n"
        return

    if changed_input is not None:
        changed_path = tmp_path / changed_input
        changed_path.parent.mkdir(parents=True, exist_ok=True)
        changed_path.write_text("// restored clean tag differs from measured bytes\n", encoding=UTF8)
        with pytest.raises(ValueError, match="source or harness differs"):
            benchmark_utils.write_release_benchmark_metadata(repo_root=tmp_path, tag=tag, criterion_dir=criterion, output=output)
        assert not output.exists()
        return

    benchmark_utils.write_release_benchmark_metadata(
        repo_root=tmp_path,
        tag=tag,
        criterion_dir=tmp_path / "criterion",
        output=output,
    )

    metadata = json.loads(output.read_text(encoding=UTF8))
    assert metadata["schema_version"] == benchmark_utils.RELEASE_ASSET_METADATA_SCHEMA_VERSION
    assert metadata["source"]["version"] == tag
    assert metadata["source"]["ref"] == tag
    assert metadata["toolchain"]["cargo_profile"] == "perf"
    assert metadata["measurement_host"]["cpu"] == "Test CPU"
    assert metadata["measurement_commands"] == [list(command) for command in benchmark_utils.RELEASE_ASSET_MEASUREMENT_COMMANDS]
    assert metadata["completed_targets"] == list(benchmark_utils.RELEASE_SIGNAL_BENCH_TARGETS)
    assert metadata["criterion"] == {
        "content_sha256": benchmark_utils._directory_digest(tmp_path / "criterion"),
        "sample_name": "new",
    }


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            replace(
                retained_performance_bundle().context.current_source,
                ref="v0.8.0",
                commit="b" * 40,
                git_clean=True,
                source_state_sha256=hashlib.sha256(("commit " + "b" * 40 + "\n").encode()).hexdigest(),
            ),
            "does not match tag",
        ),
        (
            replace(
                retained_performance_bundle().context.current_source,
                ref="v0.8.0",
                git_clean=False,
                source_state_sha256=hashlib.sha256(("commit " + "a" * 40 + "\n").encode()).hexdigest(),
            ),
            "must be clean",
        ),
        (
            replace(
                retained_performance_bundle().context.current_source,
                ref="v0.8.0",
                git_clean=True,
                source_state_sha256="c" * 64,
            ),
            "source-state digest",
        ),
    ],
)
def test_write_release_benchmark_metadata_rejects_non_tag_source_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: performance_artifacts.SourceState,
    message: str,
) -> None:
    (tmp_path / "Cargo.toml").write_text('[package]\nversion = "0.8.0"\n', encoding=UTF8)
    write_named_estimate(tmp_path, ("validation", "validate_3d", "750"), "new", 1_000_000.0)
    monkeypatch.setattr(benchmark_utils, "_source_state", lambda *_args, **_kwargs: source)
    monkeypatch.setattr(benchmark_utils, "_expected_tag_commit", lambda *_args, **_kwargs: "a" * 40)
    output = tmp_path / "metadata.json"

    with pytest.raises(ValueError, match=message):
        benchmark_utils.write_release_benchmark_metadata(
            repo_root=tmp_path,
            tag="v0.8.0",
            criterion_dir=tmp_path / "criterion",
            output=output,
        )

    assert not output.exists()


def test_write_release_benchmark_metadata_rejects_output_inside_criterion(tmp_path: Path) -> None:
    criterion_dir = tmp_path / "criterion"
    criterion_dir.mkdir()

    with pytest.raises(ValueError, match="outside the Criterion directory"):
        benchmark_utils.write_release_benchmark_metadata(
            repo_root=tmp_path,
            tag="v0.8.0",
            criterion_dir=criterion_dir,
            output=criterion_dir / "metadata.json",
        )


@pytest.mark.parametrize("diff", [b"line\r\nnext\n", b"non-UTF-8: \xff\r\n"])
def test_source_state_binds_exact_shared_snapshot_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, diff: bytes) -> None:
    """Archive source identity preserves the exact binary snapshot patch."""
    commit = "a" * 40
    snapshot = TreeSnapshot(commit, diff, ())
    monkeypatch.setattr(benchmark_utils, "capture_snapshot", lambda _root: snapshot)
    monkeypatch.setattr(benchmark_utils, "run_git_command", lambda *_args, **_kwargs: completed_process("2026-08-01T00:00:00Z"))
    source = benchmark_utils._source_state(tmp_path, version="v0.8.2", ref="HEAD")
    assert source.commit == commit
    assert source.source_state_sha256 == hashlib.sha256(b"commit " + commit.encode("ascii") + b"\n" + diff).hexdigest()
    assert source.git_clean is False


def test_untracked_workload_bytes_and_modes_affect_source_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """New scientific inputs cannot be measured under a falsely clean source identity."""
    monkeypatch.setattr(benchmark_utils, "run_git_command", lambda *_args, **_kwargs: completed_process("2026-08-01T00:00:00Z"))
    states = []
    for payload, mode in ((b"first workload", 0o644), (b"second workload", 0o644), (b"first workload", 0o755)):
        snapshot = TreeSnapshot("a" * 40, b"", (("benches/new.rs", payload, mode),))
        monkeypatch.setattr(benchmark_utils, "capture_snapshot", lambda _root, captured=snapshot: captured)
        states.append(benchmark_utils._source_state(tmp_path, version="v0.8.2", ref="HEAD"))
    assert all(state.git_clean is False for state in states)
    assert len({state.source_state_sha256 for state in states}) == 3


@pytest.mark.parametrize("fail_cleanup", [False, True])
def test_performance_worktree_failure_preserves_recovery_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_cleanup: bool) -> None:
    """Exercise the installed worktree lifecycle without mutating any Git repository."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    monkeypatch.setattr(benchmark_utils, "run_git_command", lambda *_args, **_kwargs: completed_process("a" * 40))
    observed: list[Path] = []

    def git_boundary(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        assert args[:3] == ["--no-pager", "--no-replace-objects", "worktree"]
        destination = Path(args[-2] if args[3] == "add" else args[-1])
        if args[3] == "add":
            assert args[-1] == "a" * 40
            destination.mkdir()
            (destination / "Cargo.toml").write_text('[package]\nversion = "0.9.0"\n', encoding=UTF8)
            observed.append(destination)
        elif fail_cleanup:
            raise subprocess.CalledProcessError(1, args, stderr=b"locked checkout")
        else:
            shutil.rmtree(destination)
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr("research_repo_tools.worktrees.run_git_bytes", git_boundary)
    config = ReleaseReportConfig(repo_root=tmp_path, current_tag="v0.8.3", baseline_tag="v0.8.2", worktree_ref="a" * 40, apply_current_diff=False)
    if fail_cleanup:
        with pytest.raises(ExceptionGroup) as failure:
            benchmark_utils._build_performance_bundle_in_temp_worktree(config=config)
        assert "does not match requested release" in str(failure.value.exceptions[0])
        assert "retained checkout" in str(failure.value.exceptions[1])
        assert (observed[0] / "Cargo.toml").is_file()
    else:
        with pytest.raises(ValueError, match="does not match requested release"):
            benchmark_utils._build_performance_bundle_in_temp_worktree(config=config)
        assert observed
        assert not list(scratch.iterdir())


@pytest.mark.parametrize("output", ["", "HEAD", "a" * 39, "a" * 40 + "\n" + "b" * 40])
def test_worktree_revision_rejects_empty_or_malformed_git_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output: str) -> None:
    monkeypatch.setattr(benchmark_utils, "run_git_command", lambda *_args, **_kwargs: completed_process(output))
    with pytest.raises(ValueError, match="full commit ID"):
        benchmark_utils._resolve_worktree_revision(tmp_path, "HEAD")


@pytest.mark.parametrize("host_drift", [False, True])
def test_generate_performance_worktree_report_uses_temp_worktrees_and_saved_baseline(  # noqa: C901, PLR0915
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    host_drift: bool,
) -> None:
    """Local release report generation should isolate baseline and current benchmark runs."""
    output = tmp_path / "target" / "bench-reports" / "performance.md"
    calls: list[tuple[str, tuple[str, ...], Path | None]] = []

    def write_manifest(worktree: Path, version: str) -> None:
        worktree.mkdir(parents=True, exist_ok=True)
        (worktree / "Cargo.toml").write_text(
            f'[package]\nversion = "{version}"\n\n[[bench]]\nname = "ci_performance_suite"\n',
            encoding=UTF8,
        )

    def fake_run_git(args: list[str], cwd: Path | None = None, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(("git", tuple(args), cwd))
        if args[0] == "rev-parse":
            return completed_process(("b" if args[-1] == "v0.7.8^{commit}" else "a") * 40)
        if args[:3] == ["worktree", "add", "--detach"]:
            version = "0.7.8" if args[4] == "b" * 40 else "0.8.0"
            write_manifest(Path(args[3]), version)
        if args[:3] == ["worktree", "remove", "--force"]:
            shutil.rmtree(args[3])
        if args == ["diff", "--binary", "HEAD"]:
            return completed_process("")
        return completed_process()

    def fake_run_safe(command: str, args: list[str], cwd: Path | None = None, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, tuple(args), cwd))
        if command == "cargo" and "--save-baseline=v0.7.8" in args:
            assert cwd is not None
            write_named_estimate(cwd / "target", ("validation", "validate_3d", "750"), "v0.7.8", 2_000_000.0)
            write_named_estimate(cwd / "target", ("validation", "second_case", "750"), "v0.7.8", 3_000_000.0)
        if command == "cargo" and "--save-baseline=v0.7.8" not in args:
            assert cwd is not None
            write_named_estimate(cwd / "target", ("validation", "validate_3d", "750"), "new", 1_000_000.0)
            write_named_estimate(cwd / "target", ("validation", "second_case", "750"), "new", 1_500_000.0)
        if command == "uv":
            report = Path(args[args.index("--output") + 1])
            report.write_text(delaunay_report("0.8.0", "v0.7.8"), encoding=UTF8)
        return completed_process()

    def fake_worktree_git(args: list[str], cwd: Path | None = None, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        assert args[:2] == ["--no-pager", "--no-replace-objects"]
        fake_run_git(args[2:], cwd=cwd)
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr("research_repo_tools.worktrees.run_git_bytes", fake_worktree_git)
    monkeypatch.setattr(benchmark_utils, "run_command_live", fake_run_safe)
    monkeypatch.setattr(benchmark_utils, "run_git_command", fake_run_git)
    snapshot = TreeSnapshot("a" * 40, b"binary patch", (("benches/new.rs", b"new workload", 0o644),))
    monkeypatch.setattr(benchmark_utils, "capture_snapshot", lambda _root: snapshot)
    apply = Mock()
    monkeypatch.setattr(benchmark_utils, "apply_snapshot", apply)
    monkeypatch.setattr(benchmark_utils, "run_safe_command", fake_run_safe)

    def fake_measure_suite(root: Path, targets: tuple[str, ...], *, save_baseline: str | None = None) -> dict[str, str]:
        for target in targets:
            args = ["bench", "--profile", "perf", "--bench", target]
            if save_baseline:
                args.extend(["--", f"--save-baseline={save_baseline}"])
            fake_run_safe("cargo", args, cwd=root)
        return dict.fromkeys(targets, "")

    monkeypatch.setattr(benchmark_utils, "_measure_suite", fake_measure_suite)

    def fake_revision_evidence(
        checkout: Path,
        *,
        version: str,
        ref: str,
        measurement: benchmark_utils.RevisionMeasurement,
    ) -> benchmark_utils.RevisionEvidence:
        sample_name = "new" if version == "v0.8.0" else version
        samples = benchmark_utils._criterion_estimates_by_id(checkout / "target/criterion", sample_name)
        inventory = CriterionInventory((("ci_performance_suite", tuple(samples)),))
        shared = Evidence(
            serialize_sample(Sample(tuple((name, read_estimate(case.estimates)) for name, case in samples.items()))),
            SAMPLE_SCHEMA,
            (
                (
                    "measurement",
                    Provenance(
                        ("a" if version == "v0.8.0" else "b") * 40,
                        "1" * 64,
                        "2" * 64,
                        (
                            ("release", version),
                            ("tool.rustc", json.dumps("rustc 1.98.0")),
                            ("dependency.criterion", "0.7.0"),
                            ("delaunay.sample", sample_name),
                            (
                                "host",
                                json.dumps(
                                    json.loads(
                                        serialize_host(
                                            HostMetadata("Test OS", "test", "Changed CPU" if host_drift and version != "v0.8.0" else "Test CPU", 4, 8, 4096)
                                        )
                                    )
                                ),
                            ),
                            ("delaunay.inventory", json.dumps(json.loads(serialize_inventory(inventory)))),
                            ("delaunay.sampling-policy", serialize_policy(benchmark_utils._sampling_policy(inventory.expected)).decode()),
                            ("delaunay.scientific-preflight", "passed"),
                            ("delaunay.raw-sample-sha256", benchmark_utils._criterion_sample_digest(checkout / "target/criterion", sample_name)),
                        ),
                    ),
                ),
            ),
        )
        return benchmark_utils.RevisionEvidence(
            source=benchmark_utils.SourceState(
                version=version,
                commit=("a" if version == "v0.8.0" else "b") * 40,
                ref=ref,
                revision_timestamp="2026-08-01T00:00:00Z",
                git_clean=version != "v0.8.0",
                source_state_sha256=("a" if version == "v0.8.0" else "b") * 64,
            ),
            toolchain=benchmark_utils.ToolchainState(
                rustc="rustc 1.98.0",
                criterion_version="0.7.0",
                cargo_profile="perf",
                cargo_lock_sha256="c" * 64,
                harness_sha256="d" * 64,
                configuration_sha256="e" * 64,
                measurement_plan_sha256="f" * 64,
            ),
            commands=measurement.commands,
            completed_targets=("ci_performance_suite",),
            measurement=shared,
        )

    host = benchmark_utils.HostIdentity(status="recorded", cpu="Test CPU", operating_system="Test OS", architecture="test")
    monkeypatch.setattr(benchmark_utils, "_revision_evidence", fake_revision_evidence)
    monkeypatch.setattr(benchmark_utils, "_recorded_host_identity", lambda _root: host)

    report_id = generate_performance_worktree_report(
        output=output,
        config=ReleaseReportConfig(
            repo_root=tmp_path,
            current_tag="v0.8.0",
            baseline_tag="v0.7.8",
            worktree_ref="HEAD",
            apply_current_diff=True,
        ),
    )

    assert report_id.archive_name == "v0.8.0-vs-v0.7.8.md"
    assert apply.call_count == 1
    assert apply.call_args.args[0].name == "worktree"
    assert apply.call_args.args[1] == snapshot
    report = output.read_text(encoding=UTF8)
    assert "**delaunay** v0.8.0" in report
    assert ("Measurements for current and baseline **v0.7.8**" if host_drift else "Comparison against baseline **v0.7.8**") in report
    assert "| validate_3d/750 | 2.00 ms" in report
    assert output.with_suffix(".comparison.json").is_file()
    assert output.with_suffix(".evidence.json").is_file()
    retained = performance_artifacts.load_bundle(
        performance_artifacts.ArtifactPaths(payload=output.with_suffix(".comparison.json"), provenance=output.with_suffix(".evidence.json"))
    )
    assert [(row.benchmark_id, row.coverage_status) for row in retained.sorted_rows] == [
        ("validation/second_case/750", "not-comparable" if host_drift else "comparable"),
        ("validation/validate_3d/750", "not-comparable" if host_drift else "comparable"),
    ]
    assert retained.context.current_source.commit == "a" * 40
    assert retained.context.baseline_source.commit == "b" * 40
    assert retained.context.current_commands == (("cargo", "bench", "--profile", "perf", "--bench", "ci_performance_suite"),)
    assert retained.context.baseline_commands == (("cargo", "bench", "--profile", "perf", "--bench", "ci_performance_suite", "--", "--save-baseline=v0.7.8"),)
    assert retained.context.current_measurement_host == host
    assert retained.context.baseline_measurement_host == (replace(host, cpu="Changed CPU") if host_drift else host)
    assert len(retained.measurements) == 2
    changed_row = replace(retained.rows[-1], current=performance_artifacts.TimingEstimate(2_000_000, 1_900_000, 2_100_000, 0.95))
    with pytest.raises(ValueError, match="timing differs from retained measurement"):
        replace(retained, rows=(*retained.rows[:-1], changed_row))
    with pytest.raises(ValueError, match="omits retained measurement rows"):
        replace(retained, rows=(retained.rows[-1],))
    assert "measurement hosts differ" in retained.context.comparison_blockers if host_drift else not retained.context.comparison_blockers
    assert retained.context.current_artifact.sample_name == "new"
    assert retained.context.baseline_artifact.sample_name == "v0.7.8"
    assert any(kind == "git" and args[:3] == ("worktree", "add", "--detach") and args[4] == "a" * 40 for kind, args, _ in calls)
    assert any(kind == "git" and args[:3] == ("worktree", "add", "--detach") and args[4] == "b" * 40 for kind, args, _ in calls)
    assert any(kind == "cargo" and "--save-baseline=v0.7.8" in args for kind, args, _ in calls)
    assert any(kind == "cargo" and "--save-baseline=v0.7.8" not in args for kind, args, _ in calls)
    assert not any(kind == "uv" for kind, _, _ in calls)


@pytest.fixture
def sample_estimates_data() -> dict[str, object]:
    """Fixture for common estimates.json test data."""
    return {
        "mean": {
            "point_estimate": 110000.0,  # 110 microseconds in nanoseconds
            "confidence_interval": {"lower_bound": 100000.0, "upper_bound": 120000.0},
        },
    }


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("operation", ["fsync", "replace"])
def test_atomic_write_failure_preserves_original_and_cleans_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binary: bool, operation: str
) -> None:
    """Cleanup must remain valid both before and after a temporary path exists."""
    destination = tmp_path / "artifact.txt"
    destination.write_bytes(b"original")

    def fail(*_args: object, **_kwargs: object) -> None:
        message = "simulated atomic write failure"
        raise OSError(message)

    if operation == "fsync":
        monkeypatch.setattr(benchmark_utils.os, "fsync", fail)
    else:
        monkeypatch.setattr(Path, "replace", fail)

    if binary:
        with pytest.raises(OSError, match="simulated atomic write failure"):
            benchmark_utils._write_bytes_atomic(destination, b"replacement")
    else:
        with pytest.raises(OSError, match="simulated atomic write failure"):
            benchmark_utils._write_text_atomic(destination, "replacement")

    assert destination.read_bytes() == b"original"
    assert list(tmp_path.glob(".artifact.txt.*.tmp")) == []


class TestDelaunayBenchmarkPolicy:
    """Keep scientific counts, canonical IDs, and strict summary admission covered."""

    def test_ci_benchmark_id_pattern_expands_braced_segments(self) -> None:
        """Test ci_performance_suite manifest brace patterns expand to concrete IDs."""
        result = _expand_ci_benchmark_id_pattern("tds_new_2d/{tds_new,tds_new_adversarial}/{10,25}")

        assert result == {
            "tds_new_2d/tds_new/10",
            "tds_new_2d/tds_new/25",
            "tds_new_2d/tds_new_adversarial/10",
            "tds_new_2d/tds_new_adversarial/25",
        }

    def test_ci_performance_metrics_parse_construction_counts(self) -> None:
        """Test parsing generated construction cell counts from benchmark stdout."""
        stdout = """
api_benchmark_metric benchmark_id=tds_new_2d/tds_new/2000 vertices=2000 simplices=3995
api_benchmark_metric benchmark_id=tds_new_3d/tds_new_adversarial/700 vertices=700 simplices=4211
api_benchmark_metric benchmark_id=tds_new_2d/tds_new/0 vertices=0 simplices=0
malformed api_benchmark_metric benchmark_id=ignored vertices=x simplices=y
"""

        metrics = _parse_ci_performance_metrics(stdout)

        assert metrics == {
            "tds_new_2d/tds_new/2000": {"vertices": 2000, "simplices": 3995},
            "tds_new_3d/tds_new_adversarial/700": {"vertices": 700, "simplices": 4211},
        }

    def test_write_ci_performance_metrics_clears_stale_file_when_required_metrics_missing(self) -> None:
        """Test missing fresh metrics clears stale simplex data and fails loudly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            metrics_path = project_root / "target" / "criterion" / _CI_PERFORMANCE_SUITE_METRICS_FILE
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(
                json.dumps({"tds_new_2d/tds_new/10": {"vertices": 10, "simplices": 17}}),
                encoding="utf-8",
            )

            with pytest.raises(RuntimeError, match="emitted no construction metrics"):
                _write_ci_performance_metrics(project_root, CI_MANIFEST_STDOUT.splitlines()[0], require_metrics=True)

            assert metrics_path.read_text(encoding="utf-8") == "{}\n"

    @pytest.mark.parametrize("require_metrics", [False, True])
    def test_nontext_metrics_clear_stale_file(self, tmp_path: Path, require_metrics: bool) -> None:
        """Raw subprocess output is admitted before parsing construction metrics."""
        metrics_path = tmp_path / "target" / "criterion" / _CI_PERFORMANCE_SUITE_METRICS_FILE
        metrics_path.parent.mkdir(parents=True)
        metrics_path.write_text('{"stale": true}', encoding="utf-8")

        if require_metrics:
            with pytest.raises(TypeError, match="stdout was not text"):
                _write_ci_performance_metrics(tmp_path, None, require_metrics=True)
        else:
            _write_ci_performance_metrics(tmp_path, None)

        assert metrics_path.read_text(encoding="utf-8") == "{}\n"

    def test_load_ci_performance_metrics_rejects_malformed_json_with_path(self) -> None:
        """Test corrupt metrics sidecars fail with the offending path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            criterion_dir = Path(temp_dir) / "target" / "criterion"
            criterion_dir.mkdir(parents=True)
            metrics_path = criterion_dir / _CI_PERFORMANCE_SUITE_METRICS_FILE
            metrics_path.write_text("{ invalid json", encoding=UTF8)

            with pytest.raises(ValueError, match=re.escape(str(metrics_path))):
                _load_ci_performance_metrics(criterion_dir)

    def test_load_ci_performance_metrics_returns_validated_metrics(self) -> None:
        """Loaded ci_performance_suite metrics carry typed validated counts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            criterion_dir = Path(temp_dir) / "target" / "criterion"
            criterion_dir.mkdir(parents=True)
            metrics_path = criterion_dir / _CI_PERFORMANCE_SUITE_METRICS_FILE
            metrics_path.write_text(
                json.dumps(
                    {
                        "tds_new_2d/tds_new/10": {"vertices": 10, "simplices": 17},
                        "tds_new_2d/tds_new/25": {"vertices": True, "simplices": 45},
                        "tds_new_2d/tds_new/50": {"vertices": 50, "simplices": -1},
                        "tds_new_2d/tds_new/0": {"vertices": 0, "simplices": 0},
                    }
                ),
                encoding=UTF8,
            )

            metrics = _load_ci_performance_metrics(criterion_dir)

            assert metrics == {"tds_new_2d/tds_new/10": CiPerformanceMetric(vertices=10, simplices=17)}

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"vertices": 0, "simplices": 17}, "vertices must be a positive integer"),
            ({"vertices": True, "simplices": 17}, "vertices must be a positive integer"),
            ({"vertices": 10, "simplices": 0}, "simplices must be a positive integer"),
            ({"vertices": 10, "simplices": True}, "simplices must be a positive integer"),
        ],
    )
    def test_ci_performance_metric_rejects_invalid_counts(self, kwargs: dict[str, int], message: str) -> None:
        """CiPerformanceMetric rejects non-positive and bool counts."""
        with pytest.raises(ValueError, match=message):
            CiPerformanceMetric(**kwargs)

    def test_summary_retains_exact_import_proof_and_query_manifest_ids(self, tmp_path: Path) -> None:
        """Dynamic simplex-count parameters must survive strict summary validation."""
        target_dir = tmp_path / "target"
        benchmark_ids = (
            "explicit_import/import_pseudomanifold_4d/vertices_16_simplices_68",
            "proof_boundaries/promote_pseudomanifold_strict_4d/simplices_68",
            "proof_boundaries/promote_pseudomanifold_canonicalizing_4d/simplices_68",
            "proof_boundaries/certify_pl_manifold_4d/simplices_68",
            "convex_hull_queries/is_point_outside_3d/100",
        )
        for benchmark_id in benchmark_ids:
            write_estimate(target_dir, tuple(benchmark_id.split("/")), 10_000.0)
        benchmark_utils._write_ci_performance_manifest_ids(
            tmp_path,
            "\n".join(f"api_benchmark benchmark_ids={benchmark_id}" for benchmark_id in benchmark_ids),
        )
        generator = PerformanceSummaryGenerator(tmp_path)
        evidence = generator._collect_ci_performance_summary_evidence()
        assert evidence.missing_result_ids == ()
        assert {result.benchmark_id for result in evidence.results} == set(benchmark_ids)
        assert {result.benchmark_id: result.input_size for result in evidence.results} == {
            benchmark_id: benchmark_id.rsplit("/", maxsplit=1)[1] for benchmark_id in benchmark_ids
        }


class TestProjectRootHandling:
    """Test cases for find_project_root functionality."""

    def test_find_project_root_success(self, temp_chdir: Callable[[os.PathLike[str] | str], AbstractContextManager[None]]) -> None:
        """Test finding project root when Cargo.toml exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create Cargo.toml in temp directory
            cargo_toml = temp_path / "Cargo.toml"
            cargo_toml.write_text('[package]\nname = "test"\n', encoding=UTF8)

            # Create subdirectory and change to it
            sub_dir = temp_path / "subdir"
            sub_dir.mkdir()

            with temp_chdir(sub_dir):
                result = find_project_root()
                # Resolve both paths to handle symlinks (macOS /var -> /private/var)
                assert result.resolve() == temp_path.resolve()

    def test_find_project_root_not_found(self, temp_chdir: Callable[[os.PathLike[str] | str], AbstractContextManager[None]]) -> None:
        """Test finding project root when Cargo.toml doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with temp_chdir(temp_path), pytest.raises(ProjectRootNotFoundError, match=r"Could not locate Cargo\.toml"):
                find_project_root()


class TestTimeoutHandling:
    """Test cases for benchmark timeout functionality."""

    @pytest.mark.parametrize(
        "case",
        [
            ("generate-summary", ("--run-benchmarks",), "run_release_signal_measurement_plan", "Fresh benchmark run failed"),
            ("run-release-signal", (), "run_release_signal_measurement_plan", "run-release-signal:"),
            (
                "create-release-benchmark-metadata",
                ("--tag", "v0.8.0", "--criterion-dir", "target/criterion", "--output", "metadata.json"),
                "write_release_benchmark_metadata",
                "create-release-benchmark-metadata:",
            ),
            ("performance-local", (), "published_releases", "performance-local:"),
            ("performance-github-assets", (), "published_releases", "performance-github-assets:"),
            ("performance-release", (), "published_releases", "performance-release:"),
            ("performance-doc", (), "render_and_promote_performance_artifacts", "performance-doc:"),
        ],
    )
    @pytest.mark.parametrize(
        "error",
        [
            subprocess.CalledProcessError(7, ["gh", "release", "list"], output="release lookup started\n", stderr="release lookup denied\n"),
            subprocess.TimeoutExpired("gh release list", 30, output=b"release lookup started\n", stderr=b"release lookup timed out\n"),
        ],
        ids=["failed-command-text", "timeout-bytes"],
    )
    def test_benchmark_cli_keeps_captured_failure_diagnostics(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        case: tuple[str, tuple[str, ...], str, str],
        error: subprocess.SubprocessError,
    ) -> None:
        """Expected failures retain command, captured streams, and existing reports."""
        command, arguments, operation, context = case
        (tmp_path / "Cargo.toml").write_text('[package]\nversion = "0.8.0"\n', encoding=UTF8)
        reports = (tmp_path / "benches" / "PERFORMANCE_RESULTS.md", tmp_path / "docs" / "performance.md")
        for report in reports:
            report.parent.mkdir(parents=True)
            report.write_text("previous report\n", encoding=UTF8)
        failed_operation = Mock(side_effect=error)
        monkeypatch.setattr(benchmark_utils, operation, failed_operation)
        monkeypatch.setattr(benchmark_utils, "find_project_root", lambda: tmp_path)
        monkeypatch.setattr(sys, "argv", ["benchmark-utils", command, *arguments])

        with pytest.raises(SystemExit) as failure:
            benchmark_utils.main()

        assert failure.value.code == 1
        failed_operation.assert_called_once()
        assert all(report.read_text(encoding=UTF8) == "previous report\n" for report in reports)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert context in captured.err
        assert "gh release list" in captured.err
        assert "stdout:\nrelease lookup started\n" in captured.err
        assert "stderr:\nrelease lookup " in captured.err
        assert ("release lookup denied" if isinstance(error, subprocess.CalledProcessError) else "release lookup timed out") in captured.err
        assert ("exit status 7" if isinstance(error, subprocess.CalledProcessError) else "30") in captured.err
        assert "Traceback" not in captured.err

    def test_parser_accepts_verbose_flag(self) -> None:
        """Test that the CLI parser accepts the shared verbose logging flag."""
        parser = create_argument_parser()
        args = parser.parse_args(["--verbose", "generate-summary"])

        assert args.verbose
        assert args.command == "generate-summary"

    def test_parser_suggests_close_subcommand_name(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Python 3.14 argparse suggestions should help recover from CLI typos."""
        parser = create_argument_parser()

        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["generate-summry"])

        captured = capsys.readouterr()
        assert exc_info.value.code == 2
        assert captured.out == ""
        assert "maybe you meant 'generate-summary'?" in captured.err
        assert "\x1b[" not in captured.err

    def test_parser_accepts_strict_summary_generation(self) -> None:
        """Test that release workflows can fail summary generation on fallback."""
        parser = create_argument_parser()
        args = parser.parse_args(
            [
                "generate-summary",
                "--run-benchmarks",
                "--strict",
                "--bench-timeout",
                "3600",
            ],
        )

        assert args.command == "generate-summary"
        assert args.run_benchmarks
        assert args.strict
        assert args.bench_timeout == 3600

    def test_parser_accepts_release_performance_commands(self) -> None:
        """Test that release-performance commands expose the documented options."""
        parser = create_argument_parser()

        bench_args = parser.parse_args(
            [
                "bench-compare",
                "v0.7.8",
                "--suite",
                "query",
                "--scope",
                "all-benches",
                "--stat",
                "mean",
                "--output",
                "target/bench-reports/query.md",
            ],
        )
        local_args = parser.parse_args(
            [
                "performance-local",
                "--output",
                "target/bench-reports/local.md",
                "--worktree-ref",
                "feature/ref",
                "--no-apply-current-diff",
            ],
        )
        assets_args = parser.parse_args(
            [
                "performance-github-assets",
                "v0.8.0",
                "v0.7.8",
                "--output",
                "target/bench-reports/assets.md",
                "--worktree-ref",
                "v0.8.0",
            ],
        )
        release_args = parser.parse_args(
            [
                "performance-release",
                "v0.8.0",
                "v0.7.8",
                "--current",
                "docs/performance.md",
                "--archive-dir",
                "docs/archive/performance",
                "--no-apply-current-diff",
            ],
        )
        doc_args = parser.parse_args(
            [
                "performance-doc",
                "--output",
                "target/bench-reports/performance.md",
                "--artifact-payload",
                "target/bench-reports/performance.comparison.json",
                "--artifact-provenance",
                "target/bench-reports/performance.evidence.json",
            ],
        )

        assert bench_args.command == "bench-compare"
        assert bench_args.baseline == "v0.7.8"
        assert bench_args.suite == "query"
        assert bench_args.scope == "all-benches"
        assert bench_args.stat == "mean"
        assert bench_args.output == Path("target/bench-reports/query.md")
        assert local_args.command == "performance-local"
        assert local_args.output == Path("target/bench-reports/local.md")
        assert local_args.worktree_ref == "feature/ref"
        assert local_args.no_apply_current_diff
        assert assets_args.command == "performance-github-assets"
        assert assets_args.current_tag == "v0.8.0"
        assert assets_args.baseline_tag == "v0.7.8"
        assert assets_args.output == Path("target/bench-reports/assets.md")
        assert assets_args.worktree_ref == "v0.8.0"
        assert release_args.command == "performance-release"
        assert release_args.current_tag == "v0.8.0"
        assert release_args.baseline_tag == "v0.7.8"
        assert release_args.current == Path("docs/performance.md")
        assert release_args.archive_dir == Path("docs/archive/performance")
        assert release_args.no_apply_current_diff
        assert doc_args.command == "performance-doc"
        assert doc_args.output == Path("target/bench-reports/performance.md")
        assert doc_args.artifact_payload == Path("target/bench-reports/performance.comparison.json")
        assert doc_args.artifact_provenance == Path("target/bench-reports/performance.evidence.json")

    def test_execute_command_dispatches_performance_doc_without_measurement_commands(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The public dispatcher should pass exact retained paths to performance-doc."""
        parser = create_argument_parser()
        args = parser.parse_args(
            [
                "performance-doc",
                "--output",
                "target/bench-reports/performance.md",
                "--artifact-payload",
                "target/bench-reports/performance.comparison.json",
                "--artifact-provenance",
                "target/bench-reports/performance.evidence.json",
                "--current",
                "docs/performance.md",
                "--archive-dir",
                "docs/archive/performance",
            ]
        )
        renderer = Mock(return_value=benchmark_utils.PerformanceReportId(current_tag="v0.8.0", baseline_tag="v0.7.8"))
        monkeypatch.setattr(benchmark_utils, "render_and_promote_performance_artifacts", renderer)
        monkeypatch.setattr(benchmark_utils, "_current_package_tag", lambda _root: "v0.8.0")

        with pytest.raises(SystemExit) as exit_info:
            execute_command(args, tmp_path)

        assert exit_info.value.code == 0
        renderer.assert_called_once_with(
            output=tmp_path / "target" / "bench-reports" / "performance.md",
            artifacts=performance_artifacts.ArtifactPaths(
                payload=tmp_path / "target" / "bench-reports" / "performance.comparison.json",
                provenance=tmp_path / "target" / "bench-reports" / "performance.evidence.json",
            ),
            destinations=benchmark_utils.PerformancePromotionDestinations(
                project_root=tmp_path,
                current=tmp_path / "docs" / "performance.md",
                archive_dir=tmp_path / "docs" / "archive" / "performance",
            ),
            expected_current_tag="v0.8.0",
        )

    @patch("benchmark_utils.PerformanceSummaryGenerator")
    def test_execute_command_passes_strict_summary_generation(self, mock_generator_class: MagicMock) -> None:
        """Test that CLI dispatch passes strict mode and exits nonzero on summary failure."""
        parser = create_argument_parser()
        args = parser.parse_args(
            [
                "generate-summary",
                "--run-benchmarks",
                "--strict",
                "--bench-timeout",
                "3600",
                "--profile",
                "release",
                "--output",
                "summary.md",
            ],
        )
        mock_generator = mock_generator_class.return_value
        mock_generator.generate_summary.return_value = False

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)

            with pytest.raises(SystemExit) as exc_info:
                execute_command(args, project_root)

        assert exc_info.value.code == 1
        mock_generator_class.assert_called_once_with(project_root)
        mock_generator.generate_summary.assert_called_once_with(
            output_path=Path("summary.md"),
            run_benchmarks=True,
            cargo_profile="release",
            bench_timeout=3600,
            strict=True,
        )

    def test_configure_logging_uses_debug_when_verbose(self) -> None:
        """Test that verbose mode configures debug-level CLI logging."""
        with patch("benchmark_utils.logging.basicConfig") as mock_basic_config:
            configure_logging(verbose=True)

        mock_basic_config.assert_called_once_with(
            level=logging.DEBUG,
            format="%(levelname)s: %(message)s",
        )

    def test_configure_logging_defaults_to_info(self) -> None:
        """Test that non-verbose mode configures info-level CLI logging."""
        with patch("benchmark_utils.logging.basicConfig") as mock_basic_config:
            configure_logging(verbose=False)

        mock_basic_config.assert_called_once_with(
            level=logging.INFO,
            format="%(levelname)s: %(message)s",
        )


class TestPerformanceSummaryGenerator:
    """Test cases for PerformanceSummaryGenerator class."""

    def test_init(self) -> None:
        """Test PerformanceSummaryGenerator initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            assert generator.project_root == project_root
            assert generator.circumsphere_results_dir == project_root / "target" / "criterion"
            assert isinstance(generator.current_version, str)
            assert isinstance(generator.current_date, str)

    def test_parse_single_method_result_accepts_valid_estimate(self, tmp_path: Path) -> None:
        """Test circumsphere method parsing accepts finite ordered Criterion estimates."""
        criterion_path = tmp_path / "target" / "criterion" / "basic_2d_insphere"
        estimates_dir = criterion_path / "base"
        estimates_dir.mkdir(parents=True)
        (estimates_dir / "estimates.json").write_text(
            json.dumps(
                {
                    "mean": {
                        "point_estimate": 100000.0,
                        "confidence_interval": {
                            "lower_bound": 90000.0,
                            "upper_bound": 110000.0,
                        },
                    },
                },
            ),
            encoding="utf-8",
        )
        generator = PerformanceSummaryGenerator(tmp_path)

        result = generator._parse_single_method_result(criterion_path, "insphere")

        assert result is not None
        assert result.method == "insphere"
        assert result.time_ns == pytest.approx(100000.0)

    @pytest.mark.parametrize(
        "estimates_data",
        [
            [],
            {"mean": []},
            {"mean": {"point_estimate": 100000.0, "confidence_interval": []}},
            {"mean": {"point_estimate": float("nan"), "confidence_interval": {"lower_bound": 90000.0, "upper_bound": 110000.0}}},
            {"mean": {"point_estimate": 100000.0, "confidence_interval": {"lower_bound": 120000.0, "upper_bound": 110000.0}}},
        ],
    )
    def test_parse_single_method_result_rejects_invalid_estimates(self, tmp_path: Path, estimates_data: object) -> None:
        """Test circumsphere method parsing rejects malformed or invalid Criterion estimates."""
        criterion_path = tmp_path / "target" / "criterion" / "basic_2d_insphere"
        estimates_dir = criterion_path / "base"
        estimates_dir.mkdir(parents=True)
        (estimates_dir / "estimates.json").write_text(json.dumps(estimates_data), encoding="utf-8")
        generator = PerformanceSummaryGenerator(tmp_path)

        assert generator._parse_single_method_result(criterion_path, "insphere") is None

    def test_generate_summary_parser_defaults_to_trusted_profile(self) -> None:
        """Test that fresh summary benchmarks default to the trusted Cargo profile."""
        parser = create_argument_parser()
        args = parser.parse_args(["generate-summary", "--run-benchmarks"])

        assert args.profile == BENCHMARK_BUILD_FLAVOR
        assert args.bench_timeout == 1800

    @patch("benchmark_utils.run_git_command")
    def test_get_current_version_prefers_cargo_package_version(self, mock_git_command: MagicMock) -> None:
        """Test getting current version from Cargo.toml before falling back to tags."""
        mock_git_command.side_effect = RuntimeError("git unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            (project_root / "Cargo.toml").write_text(
                '[package]\nname = "delaunay"\nversion = "1.2.3"\n',
                encoding="utf-8",
            )

            generator = PerformanceSummaryGenerator(project_root)

            assert generator.current_version == "1.2.3"

    @patch("benchmark_utils.run_git_command")
    def test_get_current_version_with_tag(self, mock_git_command: MagicMock) -> None:
        """Test getting current version from git tags."""
        mock_git_command.return_value = completed_process("v1.2.3\n")

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            version = generator._get_current_version()
            assert version == "1.2.3"  # v prefix should be removed
            mock_git_command.assert_called_with(["describe", "--tags", "--abbrev=0", "--match=v*"], cwd=project_root)

    @patch("benchmark_utils.run_git_command")
    def test_get_current_version_fallback(self, mock_git_command: MagicMock) -> None:
        """Test fallback version detection when describe fails."""
        # First call (describe) fails, second call (tag -l) succeeds
        mock_result = completed_process("v0.1.0\nv0.2.0")

        # The second call is made within the exception handler
        def side_effect(args: list[str], _cwd: Path | None = None, **_kwargs: object) -> subprocess.CompletedProcess[str]:
            if "describe" in args:
                raise subprocess.CalledProcessError(1, "git describe", "describe failed")
            return mock_result

        mock_git_command.side_effect = side_effect

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            version = generator._get_current_version()
            assert version == "0.1.0"

    @patch("benchmark_utils.run_git_command")
    def test_get_current_version_no_tags(self, mock_git_command: MagicMock) -> None:
        """Test version detection when no tags are found."""
        mock_git_command.side_effect = RuntimeError("No tags found")

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            version = generator._get_current_version()
            assert version == "unknown"

    @patch("benchmark_utils.run_git_command")
    @patch("benchmark_utils.datetime")
    def test_get_version_date_with_tag(self, mock_datetime: MagicMock, mock_git_command: MagicMock) -> None:  # noqa: ARG002
        """Test getting version date from git tag."""
        mock_git_command.return_value = completed_process("2024-01-15\n")

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)
            generator.current_version = "1.2.3"

            date = generator._get_version_date()
            assert date == "2024-01-15"
            mock_git_command.assert_called_with(["log", "-1", "--format=%cd", "--date=format:%Y-%m-%d", "v1.2.3"], cwd=project_root)

    @patch("benchmark_utils.run_git_command")
    @patch("benchmark_utils.datetime")
    def test_get_version_date_fallback(self, mock_datetime: MagicMock, mock_git_command: MagicMock) -> None:
        """Test version date fallback to current date."""
        mock_git_command.side_effect = RuntimeError("Git command failed")
        mock_now = Mock()
        mock_now.strftime.return_value = "2024-01-15"
        mock_datetime.now.return_value = mock_now
        mock_datetime.UTC = Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            date = generator._get_version_date()
            assert date == "2024-01-15"
            mock_now.strftime.assert_called_with("%Y-%m-%d")

    @patch("benchmark_utils.get_git_commit_hash")
    @patch("benchmark_utils.run_git_command")
    @patch("benchmark_utils.datetime")
    def test_generate_markdown_content(self, mock_datetime: MagicMock, mock_run_git: MagicMock, mock_git_commit: MagicMock) -> None:
        """Test generating complete markdown content."""
        # Avoid calling actual git in __init__ helpers
        mock_run_git.side_effect = RuntimeError("git unavailable in test")
        mock_git_commit.return_value = "abc123def456"
        mock_now = Mock()
        mock_now.strftime.return_value = "2024-01-15 10:30:00 UTC"
        mock_datetime.now.return_value = mock_now
        mock_datetime.UTC = Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            content = generator._generate_markdown_content()

            # Check basic structure
            assert "# Delaunay Library Performance Results" in content
            assert "- **Last Updated**: 2024-01-15 10:30:00 UTC" in content
            assert "- **Generated By**: benchmark_utils.py" in content
            assert "- **Git Commit**: abc123def456" in content
            assert "## Performance Results Summary" in content

            # Check static content sections
            assert PUBLIC_API_TITLE in content
            assert CIRCUMSPHERE_TITLE in content
            assert "## Circumsphere Predicate Analysis" not in content
            assert "### Performance Ranking" not in content
            assert "### Recommendations" not in content
            assert PERFORMANCE_UPDATES_TITLE in content

            # Retained evidence identifies the measured bytes, even after HEAD changes.
            generator.circumsphere_results_dir.mkdir(parents=True)
            (generator.circumsphere_results_dir / "delaunay-evidence.json").touch()
            measured = Provenance("a" * 40, "b" * 64, "c" * 64, (("delaunay.cargo-profile", "perf"),))
            retained = Evidence(b"{}", SAMPLE_SCHEMA, (("measurement", measured),))
            with patch("benchmark_utils._load_fresh_measurement", return_value=retained) as load_measurement:
                content = generator._generate_markdown_content()
            load_measurement.assert_called_once_with(generator.circumsphere_results_dir)
            assert f"- **Measured base commit**: {'a' * 40}" in content
            assert f"- **Measured source SHA-256**: {'b' * 64}" in content
            assert f"- **Measured harness SHA-256**: {'c' * 64}" in content
            assert "- **Measured Cargo profile**: perf" in content
            assert "- **Scientific preflight**: passed; complete raw Criterion evidence verified" in content
            assert "- **Measurement provenance**: unavailable" not in content
            assert "including uncommitted changes" in content

    @pytest.mark.parametrize(("confidence_level", "label"), [(0.95, "95%"), (0.99, "99%"), (0.975, "97.5%"), (None, "unreported")])
    def test_get_ci_performance_suite_results(self, tmp_path: Path, confidence_level: float | None, label: str) -> None:
        """Keep each interval's confidence level through parsing, evidence, and rendering."""
        write_estimate(tmp_path / "target", ("tds_new_2d", "tds_new", "10"), 120_000.0)
        write_estimate(tmp_path / "target", ("boundary_facets", "boundary_facets_3d", "50"), 8_000.0)
        write_estimate(tmp_path / "target", ("boundary_facets", "boundary_facets_3d_adversarial", "50"), 7_500.0, confidence_level=confidence_level)
        write_estimate(tmp_path / "target", ("bistellar_flips_4d", "k2_roundtrip"), 950.0)

        generator = PerformanceSummaryGenerator(tmp_path)
        evidence = generator._collect_ci_performance_summary_evidence()
        lines = generator._get_ci_performance_suite_results(evidence)
        content = "\n".join(lines)

        assert PUBLIC_API_TITLE in content
        assert "#### Construction" in content
        assert "Public API: `DelaunayTriangulationBuilder::build`" in content
        assert "| `tds_new_2d/tds_new/10` | 2D | 10 | well-conditioned | 120.0 µs | 108.0 µs - 132.0 µs | 95% |" in lines
        assert "#### Boundary facets" in content
        assert "| Benchmark ID | Dimension | Input | Variant | Mean | CI | Confidence |" in lines
        assert "95% CI" not in content
        assert "| `boundary_facets/boundary_facets_3d/50` | 3D | 50 | well-conditioned | 8.0 µs | 7.2 µs - 8.8 µs | 95% |" in lines
        assert f"| `boundary_facets/boundary_facets_3d_adversarial/50` | 3D | 50 | adversarial | 7.5 µs | 6.8 µs - 8.2 µs | {label} |" in lines
        assert "#### Bistellar flips" in content
        assert "| `bistellar_flips_4d/k2_roundtrip` | 4D | fixed fixture | well-conditioned | 950 ns | 855 ns - 1.0 µs | 95% |" in lines
        result = next(row for row in evidence.results if row.benchmark_id == "boundary_facets/boundary_facets_3d_adversarial/50")
        if confidence_level is None:
            assert result.confidence_level is None
        else:
            assert result.confidence_level == pytest.approx(confidence_level)

    def test_ci_summary_compacts_input_counts_without_changing_benchmark_ids(self, tmp_path: Path) -> None:
        """Import and proof rows keep exact IDs and timings within the line limit."""
        sizes = ((2, 120, 226, None, "unreported"), (3, 30, 115, 0.975, "97.5%"), (4, 16, 68, 0.99, "99%"), (5, 10, 20, 0.95, "95%"))
        for dimension, vertices, simplices, confidence_level, _label in sizes:
            benchmark_id = ("explicit_import", f"import_pseudomanifold_{dimension}d", f"vertices_{vertices}_simplices_{simplices}")
            write_estimate(tmp_path / "target", benchmark_id, 100_000_000.0, confidence_level=confidence_level)
            proof_id = ("proof_boundaries", f"promote_pseudomanifold_canonicalizing_{dimension}d", f"simplices_{simplices}")
            write_estimate(tmp_path / "target", proof_id, 177_176_000.0, confidence_level=confidence_level)
        generator = PerformanceSummaryGenerator(tmp_path)

        lines = generator._get_ci_performance_suite_results()

        for dimension, vertices, simplices, _confidence_level, label in sizes:
            expected_id = f"explicit_import/import_pseudomanifold_{dimension}d/vertices_{vertices}_simplices_{simplices}"
            row = next(line for line in lines if line.startswith(f"| `{expected_id}` |"))
            assert f"| {dimension}D | {vertices}v/{simplices}s |" in row
            assert row.endswith(f"| 100.000 ms | 90.000 ms - 110.000 ms | {label} |")
            proof_id = f"proof_boundaries/promote_pseudomanifold_canonicalizing_{dimension}d/simplices_{simplices}"
            proof_row = next(line for line in lines if line.startswith(f"| `{proof_id}` |"))
            assert f"| {dimension}D | {simplices}s |" in proof_row
            assert proof_row.endswith(f"| 177.176 ms | 159.458 ms - 194.894 ms | {label} |")
        assert all(len(line) <= 160 for line in lines)
        assert any("`v` for vertices and `s` for simplices" in line for line in lines)

    def test_get_circumsphere_performance_results(self) -> None:
        """Test getting circumsphere performance results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            lines = generator._get_circumsphere_performance_results()
            content = "\n".join(lines)

            assert "### Circumsphere Predicate Performance" in content
            assert "Reference fallback timings are shown below" in content
            assert "#### Reference Fallback Timings" in content
            assert "Version unknown Results" not in content

    def test_get_update_instructions(self) -> None:
        """Test getting performance data update instructions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            lines = generator._get_update_instructions()
            content = "\n".join(lines)

            assert PERFORMANCE_UPDATES_TITLE in content
            assert "just performance-local" in content
            assert "just bench-perf-summary" in content
            assert "PerformanceSummaryGenerator" in content

    @patch("benchmark_utils.run_git_command")
    def test_generate_summary_success(self, mock_git: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        """Test successful generation of performance summary."""
        mock_git.side_effect = RuntimeError("git unavailable in test")
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            output_file = Path(temp_dir) / "test_summary.md"

            success = generator.generate_summary(output_path=output_file)

            assert success is True
            assert output_file.exists()

            # Check file contains expected content
            content = output_file.read_text(encoding=UTF8)
            assert "# Delaunay Library Performance Results" in content
            assert "## Performance Results Summary" in content

            # Check success message was printed
            captured = capsys.readouterr()
            assert "Generated performance summary" in captured.out

    @pytest.mark.parametrize("bench_timeout", [0, -1])
    def test_generate_summary_rejects_non_positive_timeout(self, bench_timeout: int) -> None:
        """Direct summary calls must reject invalid benchmark budgets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = PerformanceSummaryGenerator(Path(temp_dir))

            with pytest.raises(ValueError, match="bench_timeout must be a positive integer"):
                generator.generate_summary(bench_timeout=bench_timeout)

    @pytest.mark.parametrize("mode", ["permissive", "strict", "fresh"])
    @pytest.mark.parametrize("stale_state", ["changed", "missing"])
    def test_generate_summary_stale_measurement_policy(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        caplog: pytest.LogCaptureFixture,
        mode: Literal["permissive", "strict", "fresh"],
        stale_state: str,
    ) -> None:
        """Permissive reports omit stale provenance; required evidence protects prior output."""
        strict = mode == "strict"
        run_benchmarks = mode == "fresh"
        generator = PerformanceSummaryGenerator(tmp_path)
        write_complete_release_signal_coverage(tmp_path)
        measured = Provenance("a" * 40, "b" * 64, "c" * 64, (("delaunay.cargo-profile", "perf"),))
        payload, manifest = serialize_evidence(Evidence(b"{}", SAMPLE_SCHEMA, (("measurement", measured),)))
        criterion = generator.circumsphere_results_dir
        (criterion / "delaunay-evidence.json").write_bytes(manifest)
        if stale_state == "changed":
            (criterion / "delaunay-sample.json").write_bytes(payload + b"\n")
        with pytest.raises(ValueError if stale_state == "changed" else FileNotFoundError) as failure:
            benchmark_utils._load_fresh_measurement(criterion)

        output = tmp_path / "summary.md"
        output.write_text("previous report\n", encoding=UTF8)
        with (
            patch("benchmark_utils.run_release_signal_measurement_plan", return_value={}) as run_plan,
            patch.object(generator, "_parse_ci_performance_suite_results", return_value=complete_ci_performance_results()),
            patch.object(generator, "_parse_circumsphere_benchmark_results", return_value=complete_circumsphere_results(generator)),
        ):
            success = generator.generate_summary(output_path=output, strict=strict, run_benchmarks=run_benchmarks)

        assert run_plan.call_count == int(run_benchmarks)
        assert success is not (strict or run_benchmarks)
        content = output.read_text(encoding=UTF8)
        captured = capsys.readouterr()
        if strict or run_benchmarks:
            assert content == "previous report\n"
            assert str(failure.value) in captured.err
            assert "Generated performance summary" not in captured.out
        else:
            assert "- **Measurement provenance**: unavailable; retained evidence is stale or unreadable." in content
            assert "- **Measured " not in content
            assert "- **Scientific preflight**:" not in content
            assert "including uncommitted changes" not in content
            assert PUBLIC_API_TITLE in content
            assert CIRCUMSPHERE_TITLE in content
            assert str(failure.value) in caplog.text
            assert "Generated performance summary" in captured.out
        assert list(tmp_path.glob(".summary.md.*.tmp")) == []

    @patch("benchmark_utils.run_release_signal_measurement_plan")
    def test_generate_summary_with_benchmarks(self, mock_run_plan: MagicMock) -> None:
        """Test generating summary with fresh benchmark run."""
        mock_run_plan.return_value = {measurement.target: "" for measurement in benchmark_utils.RELEASE_SIGNAL_MEASUREMENT_PLAN}

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)
            write_complete_release_signal_coverage(project_root)

            output_file = Path(temp_dir) / "test_summary.md"

            with (
                patch.object(generator, "_parse_ci_performance_suite_results", return_value=complete_ci_performance_results()),
                patch.object(generator, "_parse_circumsphere_benchmark_results", return_value=complete_circumsphere_results(generator)),
            ):
                success = generator.generate_summary(output_path=output_file, run_benchmarks=True)

            assert success is True
            mock_run_plan.assert_called_once_with(
                project_root,
                cargo_profile=BENCHMARK_BUILD_FLAVOR,
                bench_timeout=1800,
            )
            assert output_file.exists()

    @patch("benchmark_utils.run_release_signal_measurement_plan")
    def test_generate_summary_passes_cargo_profile_to_benchmarks(self, mock_run_plan: MagicMock) -> None:
        """Test generating a summary with fresh benchmarks under a specific Cargo profile."""
        mock_run_plan.return_value = {measurement.target: "" for measurement in benchmark_utils.RELEASE_SIGNAL_MEASUREMENT_PLAN}

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)
            write_complete_release_signal_coverage(project_root)

            output_file = Path(temp_dir) / "test_summary.md"

            requested_profile = "release"
            with (
                patch.object(generator, "_parse_ci_performance_suite_results", return_value=complete_ci_performance_results()),
                patch.object(generator, "_parse_circumsphere_benchmark_results", return_value=complete_circumsphere_results(generator)),
            ):
                success = generator.generate_summary(output_path=output_file, run_benchmarks=True, cargo_profile=requested_profile)

            assert success is True
            mock_run_plan.assert_called_once_with(
                project_root,
                cargo_profile=requested_profile,
                bench_timeout=1800,
            )
            assert output_file.exists()

    @pytest.mark.parametrize("strict", [False, True])
    def test_generate_summary_fresh_benchmark_failure_preserves_previous_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], strict: bool
    ) -> None:
        """The ordinary CLI retains the real saved-sample repair hint and prior output."""
        output = tmp_path / "summary.md"
        if not strict:
            output.write_text("previous report\n", encoding=UTF8)
        saved_sample = tmp_path / "target" / "criterion" / "fixture" / "last"
        saved_sample.mkdir(parents=True)
        estimates = saved_sample / "estimates.json"
        estimates.write_text("{ invalid json", encoding=UTF8)
        measure = Mock(side_effect=AssertionError("invalid saved samples must stop native measurement"))
        monkeypatch.setattr(benchmark_utils, "_measure_suite_into", measure)
        args = create_argument_parser().parse_args(
            ["generate-summary", "--run-benchmarks", "--output", str(output), *(["--strict"] if strict else [])],
        )
        assert not args.verbose

        with pytest.raises(SystemExit) as failure:
            execute_command(args, tmp_path)

        assert failure.value.code == 1
        measure.assert_not_called()
        assert estimates.read_text(encoding=UTF8) == "{ invalid json"
        if strict:
            assert not output.exists()
        else:
            assert output.read_text(encoding=UTF8) == "previous report\n"
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "Fresh benchmark run failed" in captured.err
        assert f"Remove or repair saved sample {saved_sample.resolve()} before measuring" in captured.err
        assert "Traceback" not in captured.err

    def test_generate_summary_strict_rejects_structural_fallback_provenance(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Strict generation rejects fallback based on provenance, not rendered prose."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)
            output_file = Path(temp_dir) / "test_summary.md"

            success = generator.generate_summary(output_path=output_file, strict=True)

            assert success is False
            assert not output_file.exists()
            captured = capsys.readouterr()
            assert "circumsphere evidence uses reference fallback timings" in captured.err

    def test_generate_summary_strict_accepts_existing_release_workflow_results(self) -> None:
        """Complete prior Criterion evidence is sufficient; accuracy stdout is optional."""

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)
            write_complete_release_signal_coverage(project_root)
            output_file = Path(temp_dir) / "test_summary.md"
            circumsphere_results = complete_circumsphere_results(generator)
            with (
                patch.object(generator, "_parse_ci_performance_suite_results", return_value=complete_ci_performance_results()),
                patch.object(generator, "_parse_circumsphere_benchmark_results", return_value=circumsphere_results),
            ):
                success = generator.generate_summary(
                    output_path=output_file,
                    run_benchmarks=False,
                    strict=True,
                )

            assert success is True
            content = output_file.read_text(encoding=UTF8)
            assert "Reference fallback timings" not in content
            assert "section is incomplete" not in content

    def test_release_signal_report_coverage_is_derived_from_every_planned_section(self, tmp_path: Path) -> None:
        """Every executable target should own exactly one visible report-coverage row."""
        generator = PerformanceSummaryGenerator(tmp_path)
        write_complete_release_signal_coverage(tmp_path)

        sections = generator._collect_release_signal_section_evidence()
        rendered = "\n".join(generator._release_signal_coverage_section(sections))

        assert tuple((section.target, section.report_section) for section in sections) == tuple(
            (measurement.target, measurement.report_section) for measurement in benchmark_utils.RELEASE_SIGNAL_MEASUREMENT_PLAN
        )
        assert all(section.is_complete for section in sections)
        for measurement in benchmark_utils.RELEASE_SIGNAL_MEASUREMENT_PLAN:
            assert f"| `{measurement.target}` | {measurement.report_section} |" in rendered

    @pytest.mark.parametrize(
        ("target", "path_parts", "full_id", "group_id"),
        [
            ("circumsphere_containment", ("random_insphere_1000_queries",), "random/insphere_1000_queries", "random/insphere_1000_queries"),
            ("circumsphere_containment", ("3d_insphere",), "3d/insphere", "3d/insphere"),
            ("circumsphere_containment", ("circumcenter_solve_path", "regular_lu_3d"), "circumcenter/solve_path/regular_lu_3d", "circumcenter/solve_path"),
            ("cold_path_predicates", ("predicates_hot", "insphere_3d", "10000"), "predicates/hot/insphere_3d/10000", "predicates/hot"),
            (
                "locate",
                ("locate_no_hint_2d", "locate", "vertices_500_simplices_983"),
                "locate/no_hint/2d/locate/vertices_500_simplices_983",
                "locate/no_hint/2d",
            ),
        ],
    )
    def test_release_signal_coverage_uses_canonical_metadata_ids(
        self,
        tmp_path: Path,
        target: str,
        path_parts: tuple[str, ...],
        full_id: str,
        group_id: str,
    ) -> None:
        """Criterion's escaped disk paths must not hide measured release groups."""
        write_named_estimate(tmp_path / "target", path_parts, "new", 1_000.0, stat="mean", full_id=full_id, group_id=group_id)
        sections = PerformanceSummaryGenerator(tmp_path)._collect_release_signal_section_evidence()
        section = next(section for section in sections if section.target == target)

        assert section.result_ids == (full_id,)
        assert full_id.split("/", maxsplit=1)[0] not in section.missing_group_prefixes

    def test_release_signal_coverage_requires_identity_metadata(self, tmp_path: Path) -> None:
        """A plausible directory name cannot substitute for missing benchmark identity."""
        write_estimate(tmp_path / "target", ("validation", "validate_2d"), 1_000.0)
        (tmp_path / "target/criterion/validation/validate_2d/base/benchmark.json").unlink()

        sections = PerformanceSummaryGenerator(tmp_path)._collect_release_signal_section_evidence()
        section = next(section for section in sections if section.target == "ci_performance_suite")
        assert section.result_ids == ()
        assert "validation" in section.missing_group_prefixes
        assert not section.is_complete

    @pytest.mark.parametrize("sample", ["base", "new"])
    @pytest.mark.parametrize("metadata", [None, "{", "[]", '{"full_id":"","group_id":"obsolete"}', '{"full_id":"standalone","group_id":"standalone"}'])
    def test_release_signal_coverage_skips_unusable_metadata(self, tmp_path: Path, sample: str, metadata: str | None) -> None:
        """Unusable cached samples must not hide valid measured report groups."""
        write_complete_release_signal_coverage(tmp_path)
        write_named_estimate(tmp_path / "target", ("obsolete", "fixture"), sample, 1_000.0, stat="mean")
        criterion_dir = tmp_path / "target/criterion"
        metadata_path = criterion_dir / "obsolete/fixture" / sample / "benchmark.json"
        if metadata is None:
            metadata_path.unlink()
        else:
            metadata_path.write_text(metadata, encoding=UTF8)

        sections = PerformanceSummaryGenerator(tmp_path)._collect_release_signal_section_evidence()
        assert all(section.is_complete for section in sections)
        assert "obsolete/fixture" not in benchmark_utils._criterion_result_ids(criterion_dir)
        # Comparison and artifact consumers retain the strict default.
        with pytest.raises((OSError, TypeError, ValueError)):
            benchmark_utils._criterion_estimates_by_id(criterion_dir, sample)

    def test_release_signal_result_ids_preserve_samples_and_sorting(self, tmp_path: Path) -> None:
        """Collect valid base/new IDs once, with new estimates taking precedence."""
        write_named_estimate(tmp_path, ("validation", "z_base"), "base", 1_000.0, stat="mean")
        write_named_estimate(tmp_path, ("validation", "a_new"), "new", 1_000.0, stat="mean")
        write_named_estimate(tmp_path, ("validation", "shared"), "base", 1_000.0, stat="mean")
        write_named_estimate(tmp_path, ("validation", "shared"), "new", 2_000.0, stat="mean")
        write_named_estimate(tmp_path, ("validation", "invalid_new"), "base", 1_000.0, stat="mean")
        write_named_estimate(tmp_path, ("validation", "invalid_new"), "new", -1.0, stat="mean")

        assert benchmark_utils._criterion_result_ids(tmp_path / "criterion") == (
            "validation/a_new",
            "validation/shared",
            "validation/z_base",
        )

    def test_release_signal_coverage_still_rejects_duplicate_ids(self, tmp_path: Path) -> None:
        """Skipping unreadable metadata must not hide ambiguous valid identities."""
        for directory in ("first", "second"):
            write_named_estimate(tmp_path, (directory,), "new", 1_000.0, stat="mean", full_id="validation/duplicate", group_id="validation")

        with pytest.raises(ValueError, match="duplicate Criterion full_id"):
            benchmark_utils._criterion_result_ids(tmp_path / "criterion")

    def test_generate_summary_strict_rejects_missing_planned_report_section(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """A planned target without Criterion groups must block strict publication."""
        generator = PerformanceSummaryGenerator(tmp_path)
        for measurement in benchmark_utils.RELEASE_SIGNAL_MEASUREMENT_PLAN[:-1]:
            for prefix in measurement.required_group_prefixes:
                group = f"{prefix}fixture" if prefix.endswith("_") else prefix
                write_estimate(tmp_path / "target", (group, "fixture"), 1_000.0)

        with (
            patch.object(generator, "_parse_ci_performance_suite_results", return_value=complete_ci_performance_results()),
            patch.object(generator, "_parse_circumsphere_benchmark_results", return_value=complete_circumsphere_results(generator)),
        ):
            success = generator.generate_summary(output_path=tmp_path / "summary.md", strict=True)

        assert success is False
        assert not (tmp_path / "summary.md").exists()
        captured = capsys.readouterr()
        assert "realization_validation report section 'Realization validation' is incomplete" in captured.err

    def test_generate_summary_strict_rejects_missing_circumsphere_results(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that strict summary generation fails when circumsphere results are absent."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)
            output_file = Path(temp_dir) / "test_summary.md"

            with (
                patch.object(generator, "_parse_ci_performance_suite_results", return_value=complete_ci_performance_results()),
                patch.object(generator, "_parse_circumsphere_benchmark_results", return_value=[]),
            ):
                success = generator.generate_summary(output_path=output_file, strict=True)

            assert success is False
            assert not output_file.exists()
            captured = capsys.readouterr()
            assert "circumsphere evidence uses reference fallback timings" in captured.err

    def test_generate_summary_strict_rejects_partial_circumsphere_estimates(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """One missing method estimate makes the fixed circumsphere contract incomplete."""
        generator = PerformanceSummaryGenerator(tmp_path)
        circumsphere_results = complete_circumsphere_results(generator)
        first_case = circumsphere_results[0]
        first_case.methods.pop("insphere")
        output_file = tmp_path / "summary.md"
        output_file.write_text("previous report\n", encoding=UTF8)

        with (
            patch.object(generator, "_parse_ci_performance_suite_results", return_value=complete_ci_performance_results()),
            patch.object(generator, "_parse_circumsphere_benchmark_results", return_value=circumsphere_results),
        ):
            success = generator.generate_summary(output_path=output_file, strict=True)

        assert success is False
        assert output_file.read_text(encoding=UTF8) == "previous report\n"
        captured = capsys.readouterr()
        assert "circumsphere Criterion evidence is incomplete" in captured.err
        assert "2d_insphere" in captured.err

    def test_generate_summary_strict_rejects_manifest_result_with_malformed_estimate(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """The runtime manifest makes a skipped malformed CI estimate an explicit gap."""
        generator = PerformanceSummaryGenerator(tmp_path)
        write_complete_release_signal_coverage(tmp_path)
        estimates_dir = generator.circumsphere_results_dir / "validation" / "malformed" / "20" / "base"
        estimates_dir.mkdir(parents=True)
        (estimates_dir / "estimates.json").write_text("{ invalid json", encoding=UTF8)
        manifest_path = tmp_path / "target" / "criterion" / _CI_PERFORMANCE_SUITE_MANIFEST_IDS_FILE
        manifest_path.write_text("validation/malformed/20\n", encoding=UTF8)
        output_file = tmp_path / "summary.md"

        with patch.object(generator, "_parse_circumsphere_benchmark_results", return_value=complete_circumsphere_results(generator)):
            success = generator.generate_summary(output_path=output_file, strict=True)

        assert success is False
        assert not output_file.exists()
        captured = capsys.readouterr()
        assert "runtime-manifest" in captured.err
        assert "validation/malformed/20" in captured.err

    def test_generate_summary_atomic_replace_failure_preserves_previous_report(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """A final replacement failure leaves the previous tracked report byte-identical."""
        generator = PerformanceSummaryGenerator(tmp_path)
        write_complete_release_signal_coverage(tmp_path)
        output_file = tmp_path / "summary.md"
        output_file.write_text("previous report\n", encoding=UTF8)

        with (
            patch.object(generator, "_parse_ci_performance_suite_results", return_value=complete_ci_performance_results()),
            patch.object(generator, "_parse_circumsphere_benchmark_results", return_value=complete_circumsphere_results(generator)),
            patch.object(Path, "replace", side_effect=OSError("injected replace failure")),
        ):
            success = generator.generate_summary(output_path=output_file, strict=True)

        assert success is False
        assert output_file.read_text(encoding=UTF8) == "previous report\n"
        assert list(tmp_path.glob(".summary.md.*.tmp")) == []
        captured = capsys.readouterr()
        assert "injected replace failure" in captured.err

    def test_generate_summary_exception_handling(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test exception handling in generate_summary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            output_file = Path(temp_dir) / "readonly" / "summary.md"
            with patch("benchmark_utils._write_text_atomic", side_effect=OSError("permission denied")):
                success = generator.generate_summary(output_path=output_file)

            assert success is False

            # Check error was printed (looking for the error message)
            captured = capsys.readouterr()
            assert "Failed to generate performance summary" in captured.err

    def test_get_static_content(self) -> None:
        """Test getting static content sections."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            lines = generator._get_static_sections()
            content = "\n".join(lines)

            assert "## Historical Version Comparison" not in content
            assert "## Implementation Notes" not in content
            assert "### Method Disagreements" not in content
            assert "## Benchmark Structure" in content

    @pytest.mark.parametrize("fastest", ["insphere", "insphere_distance", "insphere_lifted"])
    def test_get_implementation_notes(self, tmp_path: Path, fastest: str) -> None:
        """Conclusions follow the same evidence as the tables, including ties."""
        generator = PerformanceSummaryGenerator(tmp_path)
        cases = complete_circumsphere_results(generator)
        for case in cases:
            for name, performance in case.methods.items():
                performance.time_ns = 10.0 if name == fastest else 20.0
        # A tie must credit both methods rather than depend on dictionary order.
        tied = next(name for name in cases[0].methods if name != fastest)
        cases[0].methods[tied].time_ns = 10.0
        evidence = benchmark_utils.CircumsphereSummaryEvidence(tuple(cases), "criterion", ())
        notes = "\n".join(generator._get_implementation_notes(evidence))
        tables = "\n".join(generator._get_circumsphere_performance_results(evidence))

        for dimension in ("2D", "3D", "4D", "5D"):
            counts = {name: 3 if name == fastest else 0 for name in cases[0].methods}
            if dimension == cases[0].dimension:
                counts[tied] += 1
            assert f"| {dimension} | 3 | {counts['insphere']} | {counts['insphere_distance']} | {counts['insphere_lifted']} |" in notes
        assert "Lowest mean" in tables
        assert "**" + ", ".join(sorted((fastest, tied))) + "**" in tables
        assert "statistical significance" in notes
        assert "tolerance" in notes
        assert "filtered-exact" in notes
        assert all(len(line) <= 160 for line in notes.splitlines())

    @pytest.mark.parametrize(("provenance", "missing"), [("reference-fallback", ()), ("criterion", ("missing/result",))])
    def test_implementation_notes_withhold_incomplete_conclusions(
        self, tmp_path: Path, provenance: Literal["criterion", "reference-fallback"], missing: tuple[str, ...]
    ) -> None:
        """Fallback and partial results cannot support measured conclusions."""
        generator = PerformanceSummaryGenerator(tmp_path)
        evidence = benchmark_utils.CircumsphereSummaryEvidence(tuple(complete_circumsphere_results(generator)), provenance, missing)
        content = "\n".join(PerformanceSummaryGenerator._get_implementation_notes(evidence))
        assert "Complete measured predicate evidence is unavailable" in content
        assert "| 2D |" not in content

    def test_empty_benchmark_results_edge_case(self) -> None:
        """Test handling of empty benchmark results (edge case)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            # Parsing stays honest; fallback is added only by the evidence collector.
            results = generator._parse_circumsphere_benchmark_results()
            assert results == []
            evidence = generator._collect_circumsphere_summary_evidence()
            assert evidence.provenance == "reference-fallback"
            assert evidence.test_cases

    def test_malformed_estimates_json_edge_case(self) -> None:
        """Test handling of malformed estimates.json files (edge case)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)

            # Create malformed estimates.json
            criterion_dir = project_root / "target" / "criterion" / "basic-insphere" / "base"
            criterion_dir.mkdir(parents=True)

            estimates_file = criterion_dir / "estimates.json"
            estimates_file.write_text("{ invalid json", encoding=UTF8)

            generator = PerformanceSummaryGenerator(project_root)

            # Malformed raw evidence must not be returned as measured rows.
            results = generator._parse_circumsphere_benchmark_results()
            assert results == []
            assert generator._collect_circumsphere_summary_evidence().provenance == "reference-fallback"

    def test_missing_git_info_edge_case(self) -> None:
        """Test handling when git information is not available (edge case)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            output_file = Path(temp_dir) / "test_output.md"

            with (
                patch("benchmark_utils.run_git_command") as mock_git,
                patch("benchmark_utils.get_git_commit_hash") as mock_commit,
            ):
                mock_git.side_effect = RuntimeError("Git not available")
                mock_commit.side_effect = RuntimeError("Git not available")

                generator = PerformanceSummaryGenerator(project_root)
                success = generator.generate_summary(output_file)

                # Should still succeed
                assert success

                content = output_file.read_text(encoding=UTF8)
                assert "Reference fallback timings are shown below" in content

                # Note: The baseline file parsing extracts metadata, not performance data
                # Performance data "1000 Points (3D)" would come from benchmark parsing,
                # not baseline parsing. The important test is that the fallback file is read.

    def test_dimension_sorting_numeric_order(self) -> None:
        """Test that dimensions are sorted numerically, not lexically."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            generator = PerformanceSummaryGenerator(project_root)

            # Create test cases with dimensions that would sort wrong lexically
            test_cases = [
                CircumsphereTestCase("Test10", "10D", {"insphere": CircumspherePerformanceData("insphere", 1000)}),
                CircumsphereTestCase("Test2", "2D", {"insphere": CircumspherePerformanceData("insphere", 1000)}),
                CircumsphereTestCase("Test3", "3D", {"insphere": CircumspherePerformanceData("insphere", 1000)}),
                CircumsphereTestCase("Test1", "1D", {"insphere": CircumspherePerformanceData("insphere", 1000)}),
                CircumsphereTestCase("Test9", "9D", {"insphere": CircumspherePerformanceData("insphere", 1000)}),
            ]

            # Patch the generator to use our test cases instead of parsing from files
            with patch.object(generator, "_parse_circumsphere_benchmark_results", return_value=test_cases):
                # Generate the circumsphere performance results section
                result_lines = generator._get_circumsphere_performance_results()
                content = "\n".join(result_lines)

                # Find the order of dimension headers in the generated content

                dimension_headers = re.findall(r"#### Single Query Performance \((\d+D)\)", content)

                # Verify that dimensions appear in numeric order: 1D, 2D, 3D, 9D, 10D
                expected_order = ["1D", "2D", "3D", "9D", "10D"]
                assert dimension_headers == expected_order, f"Expected {expected_order}, got {dimension_headers}"

                # Also verify that each dimension's test case appears in the content
                assert "Test1" in content  # 1D test case
                assert "Test2" in content  # 2D test case
                assert "Test3" in content  # 3D test case
                assert "Test9" in content  # 9D test case
                assert "Test10" in content  # 10D test case

    def test_dev_mode_args_consistency(self) -> None:
        """Test that DEV_MODE_BENCH_ARGS is used consistently."""
        # Verify the constant exists and has expected structure
        assert isinstance(DEV_MODE_BENCH_ARGS, list)
        assert "--sample-size" in DEV_MODE_BENCH_ARGS
        assert "--measurement-time" in DEV_MODE_BENCH_ARGS
        assert "--warm-up-time" in DEV_MODE_BENCH_ARGS

        # The specific values may change, but the structure should be consistent
        # with pairs of argument name and value
        assert len(DEV_MODE_BENCH_ARGS) >= 6  # At least 3 arg-value pairs
