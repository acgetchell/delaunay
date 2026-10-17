"""Delaunay sampling policy integrated with the installed shared measurement APIs."""

import json
import shutil
import subprocess
from itertools import product
from pathlib import Path

import pytest
from research_repo_tools import criterion_inventory, measurement
from research_repo_tools.complete_runs import collect_complete_sample, parse_policy
from research_repo_tools.evidence import load_evidence
from research_repo_tools.host_metadata import HostMetadata

import benchmark_utils

CASES = {
    "validation/validate_3d/750": 15,
    "predicates/sos_degenerate/insphere_5d": 10,
    "predicates/hot/insphere/3d": 100,
}
TARGETS = ("ci_performance_suite", "cold_path_predicates")


def write_case(criterion: Path, full_id: str, count: int, *, saved: bool = True) -> None:
    """Write independent raw fixtures whose semantic IDs differ from storage paths."""
    directory = criterion / full_id.replace("/", "_") / "new"
    directory.mkdir(parents=True)
    group_id = "predicates/hot" if full_id.startswith("predicates/hot/") else full_id.split("/", maxsplit=1)[0]
    (directory / "benchmark.json").write_text(json.dumps({"full_id": full_id, "group_id": group_id}), encoding="utf-8")
    estimate = {"point_estimate": 100, "confidence_interval": {"lower_bound": 90, "upper_bound": 110, "confidence_level": 0.95}}
    (directory / "estimates.json").write_text(json.dumps({"mean": estimate, "median": estimate}), encoding="utf-8")
    (directory / "sample.json").write_text(json.dumps({"iters": [1] * count, "times": [100] * count}), encoding="utf-8")
    (directory / "tukey.json").write_bytes(b"[1,2,3,4]\r\n")
    if saved:
        shutil.copytree(directory, directory.with_name("last"))


@pytest.fixture
def native_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Model Cargo/Git observations while exercising installed shared APIs and files."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src/lib.rs").write_text("// scientific source\n", encoding="utf-8")
    (tmp_path / "benches").mkdir()
    for target in TARGETS:
        (tmp_path / "benches" / f"{target}.rs").write_text("// workload\n", encoding="utf-8")
    (tmp_path / "Cargo.toml").write_text('[package]\nversion = "0.8.2"\n', encoding="utf-8")
    (tmp_path / "Cargo.lock").write_text('[[package]]\nname = "criterion"\nversion = "0.7.0"\n', encoding="utf-8")
    (tmp_path / "rust-toolchain.toml").write_text('[toolchain]\nchannel = "1.99.0"\n', encoding="utf-8")
    events: list[str] = []
    monkeypatch.setattr(measurement, "resolve_revision", lambda *_args: "a" * 40)
    host = HostMetadata("Test OS", "test", "Test CPU", 4, 8, 4096, (("rustc", "rustc 1.99.0"),))
    monkeypatch.setattr(measurement, "capture_host", lambda *_args, **_kwargs: host)

    def discover(_command: str, args: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        target = args[args.index("--bench") + 1]
        events.append(f"discover:{target}")
        assert kwargs["cwd"] == tmp_path
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert Path(environment["CRITERION_HOME"]) != tmp_path / "target/criterion"
        ids = [name for name in CASES if name.startswith("validation/") == (target == "ci_performance_suite")]
        return subprocess.CompletedProcess(args, 0, stdout="build noise\r\n" + "".join(f"{name}: benchmark\r\n" for name in ids))

    def preflight(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        assert args[-1] == "--test"
        events.append(f"gate:{args[args.index('--bench') + 1]}")
        return subprocess.CompletedProcess(args, 0)

    def sample(_command: str, args: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        events.append("measure")
        assert "sample-suite" in args
        saved = "--save-baseline" in args
        criterion = Path(args[args.index("--criterion-dir") + 1])
        assert criterion != tmp_path / "target/criterion"
        for full_id, count in CASES.items():
            write_case(criterion, full_id, count, saved=saved)
        output = criterion / benchmark_utils.MEASUREMENT_OUTPUT_FILE
        output.write_text(json.dumps(dict.fromkeys(TARGETS, "scientific diagnostics")), encoding="utf-8")
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(criterion_inventory, "run_command", discover)
    monkeypatch.setattr(benchmark_utils, "run_cargo_live", preflight)
    monkeypatch.setattr(measurement, "run_command_live", sample)
    return events


def test_mixed_sampling_retains_canonical_ids_policy_and_provenance(tmp_path: Path, native_boundary: list[str]) -> None:
    """No raw observations are padded or dropped to fit a uniform policy."""
    outputs = benchmark_utils._measure_suite(tmp_path, TARGETS, save_baseline="last")
    assert outputs == dict.fromkeys(TARGETS, "scientific diagnostics")
    assert native_boundary == [*(f"discover:{target}" for target in TARGETS), *(f"gate:{target}" for target in TARGETS), "measure"]
    criterion = tmp_path / "target/criterion"
    policy = parse_policy((criterion / benchmark_utils.MEASUREMENT_POLICY_FILE).read_bytes())
    assert dict(policy.sample_counts) == CASES
    assert policy.sample_count is None
    sample = collect_complete_sample(criterion, "last", policy)
    assert {case.full_id: case.sample_count for case in sample.cases} == CASES
    assert all(case.storage_path == case.full_id.replace("/", "_") for case in sample.cases)
    displayed = benchmark_utils._criterion_estimates_by_id(criterion, "last")["predicates/hot/insphere/3d"]
    assert (displayed.group, displayed.benchmark) == ("predicates/hot", "insphere/3d")
    evidence = load_evidence(criterion / "delaunay-sample.json", criterion / "delaunay-evidence.json")
    source = dict(evidence.sources)["measurement"]
    assert source.source_sha256
    assert source.harness_sha256
    assert dict(source.context)["delaunay.scientific-preflight"] == "passed"
    assert parse_policy(dict(source.context)["delaunay.sampling-policy"].encode()) == policy
    assert benchmark_utils._load_fresh_measurement(criterion) == evidence


@pytest.mark.parametrize("mutation", ["raw", "policy", "gate"])
def test_retained_evidence_cannot_certify_changed_release_inputs(
    tmp_path: Path,
    native_boundary: list[str],
    mutation: str,
) -> None:
    """Packaging cannot reuse preflight evidence after observations or policy change."""
    benchmark_utils._measure_suite(tmp_path, TARGETS, save_baseline="last")
    criterion = tmp_path / "target/criterion"
    if mutation == "raw":
        (criterion / "validation_validate_3d_750/last/tukey.json").write_bytes(b"[1,2,3,4]\n")
    elif mutation == "policy":
        policy_path = criterion / benchmark_utils.MEASUREMENT_POLICY_FILE
        data = json.loads(policy_path.read_bytes())
        data["sample_counts"]["validation/validate_3d/750"] = 20
        policy_path.write_text(json.dumps(data), encoding="utf-8")
    else:
        manifest = criterion / "delaunay-evidence.json"
        manifest.write_bytes(manifest.read_bytes().replace(b'"passed"', b'"failed"'))
    with pytest.raises(ValueError, match=r"raw Criterion|Delaunay workload"):
        benchmark_utils._load_fresh_measurement(criterion)


@pytest.mark.parametrize("failure", ["missing", "extra", "count", "saved", "source", "gate"])
def test_successful_process_is_insufficient_measurement_evidence(
    tmp_path: Path,
    native_boundary: list[str],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Shared completeness rejects bad native output before retaining evidence."""
    original = measurement.run_command_live
    prior = tmp_path / "target/criterion/previous-evidence"
    prior.parent.mkdir(parents=True)
    prior.write_bytes(b"retain on failure\r\n")

    def damaged(command: str, args: tuple[str, ...], *, cwd: Path, timeout: int) -> subprocess.CompletedProcess[bytes]:
        result = original(command, args, cwd=cwd, timeout=timeout)
        criterion = Path(args[args.index("--criterion-dir") + 1])
        directory = criterion / "validation_validate_3d_750"
        if failure == "missing":
            (directory / "new/sample.json").unlink()
        elif failure == "extra":
            write_case(criterion, "validation/unexpected", 15)
        elif failure == "count":
            (directory / "new/sample.json").write_text('{"iters":[1,1],"times":[100,100]}', encoding="utf-8")
        elif failure == "saved":
            (directory / "last/tukey.json").write_bytes(b"[1,2,3,4]\n")
        elif failure == "source":
            (tmp_path / "src/lib.rs").write_text("// changed during measurement\n", encoding="utf-8")
        return result

    monkeypatch.setattr(measurement, "run_command_live", damaged)
    if failure == "gate":

        def fail_gate(*_args: object, **_kwargs: object) -> None:
            raise subprocess.CalledProcessError(101, ["cargo", "bench", "--test"])

        monkeypatch.setattr(benchmark_utils, "run_cargo_live", fail_gate)
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        benchmark_utils._measure_suite(tmp_path, TARGETS, save_baseline="last")
    assert not (tmp_path / "target/criterion/delaunay-evidence.json").exists()
    assert prior.read_bytes() == b"retain on failure\r\n"
    if failure == "gate":
        assert "measure" not in native_boundary
    else:
        assert "measure" in native_boundary


def test_latest_measurement_retains_saved_baseline_and_replaces_stale_new_rows(tmp_path: Path, native_boundary: list[str]) -> None:
    """The save-last then measure-latest workflow keeps named raw bytes and coverage."""
    criterion = tmp_path / "target/criterion"
    benchmark_utils._measure_suite(tmp_path, TARGETS, save_baseline="last")
    write_case(criterion, "validation/removed", 15)
    saved = {path.relative_to(criterion): path.read_bytes() for path in criterion.glob("*/last/*")}
    benchmark_utils._measure_suite(tmp_path, TARGETS)
    assert {path: (criterion / path).read_bytes() for path in saved} == saved
    assert not (criterion / "validation_removed/new").exists()
    assert benchmark_utils._load_fresh_measurement(criterion).payload


@pytest.mark.parametrize(
    ("filename", "contents"),
    [
        ("benchmark.json", b"{broken"),
        ("benchmark.json", b'{"full_id":"ungrouped"}'),
        ("benchmark.json", None),
        ("sample.json", None),
        ("estimates.json", b"[]"),
    ],
)
def test_invalid_preserved_sample_fails_before_native_work(
    tmp_path: Path,
    native_boundary: list[str],
    filename: str,
    contents: bytes | None,
) -> None:
    """Every case in a retained baseline must be usable before native work starts."""
    criterion = tmp_path / "target/criterion"
    write_case(criterion, "validation/a", 15)
    write_case(criterion, "validation/z", 15)
    sample = criterion / "validation_z/last"
    if contents is None:
        (sample / filename).unlink()
    else:
        (sample / filename).write_bytes(contents)
    prior = {path.relative_to(criterion): path.read_bytes() for path in criterion.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="Remove or repair saved sample") as raised:
        benchmark_utils._measure_suite(tmp_path, TARGETS)

    assert str(sample) in str(raised.value)
    assert native_boundary == []
    assert {path.relative_to(criterion): path.read_bytes() for path in criterion.rglob("*") if path.is_file()} == prior


def test_invalid_replaced_samples_do_not_block_measurement(tmp_path: Path, native_boundary: list[str]) -> None:
    """Fresh samples may replace invalid new and explicitly selected baseline bytes."""
    criterion = tmp_path / "target/criterion"
    write_case(criterion, "validation/old", 15)
    for sample in ("new", "last"):
        (criterion / "validation_old" / sample / "benchmark.json").write_bytes(b"{broken")

    benchmark_utils._measure_suite(tmp_path, TARGETS, save_baseline="last")

    assert "measure" in native_boundary
    assert not (criterion / "validation_old").exists()
    assert benchmark_utils._load_fresh_measurement(criterion).payload


@pytest.mark.parametrize("old_name", ["Last", "New", "Base", "Change"])
def test_saved_baseline_preservation_respects_filesystem_aliases(tmp_path: Path, native_boundary: list[str], old_name: str) -> None:
    """A case alias cannot restore old bytes over a freshly validated sample."""
    criterion = tmp_path / "target/criterion"
    benchmark_utils._measure_suite(tmp_path, TARGETS, save_baseline="last")
    old_bytes = b"[2,3,4,5]\n"
    preserved = []
    for sample in criterion.glob("*/last"):
        destination = sample.with_name(old_name)
        if destination.exists() and not destination.samefile(sample):
            shutil.rmtree(destination)
        sample.rename(destination)
        (destination / "tukey.json").write_bytes(old_bytes)
        preserved.append(destination)
    sibling = preserved[0].with_name(old_name.lower())
    aliases_reserved_name = sibling.exists() and sibling.samefile(preserved[0])
    native_boundary.clear()

    if not aliases_reserved_name:
        prior = {path.relative_to(criterion): path.read_bytes() for path in criterion.rglob("*") if path.is_file()}
        with pytest.raises(ValueError, match="Rename saved baseline"):
            benchmark_utils._measure_suite(tmp_path, TARGETS, save_baseline="last")
        assert native_boundary == []
        assert {path.relative_to(criterion): path.read_bytes() for path in criterion.rglob("*") if path.is_file()} == prior
        return

    benchmark_utils._measure_suite(tmp_path, TARGETS, save_baseline="last")

    for directory in criterion.glob("*/last"):
        assert (directory / "tukey.json").read_bytes() == b"[1,2,3,4]\r\n"
    assert benchmark_utils._load_fresh_measurement(criterion).payload


@pytest.mark.parametrize("names", [("Release", "release"), ("café",), ("café",)])
def test_unpublishable_saved_names_fail_before_sampling_without_changing_prior_bytes(
    tmp_path: Path,
    native_boundary: list[str],
    names: tuple[str, ...],
) -> None:
    """Logical saved names stay portable even when stored under different cases."""
    criterion = tmp_path / "target/criterion"
    benchmark_utils._measure_suite(tmp_path, TARGETS, save_baseline="last")
    for index, sample in enumerate(sorted(criterion.glob("*/last"))):
        sample.rename(sample.with_name(names[index % len(names)]))
    prior = {path.relative_to(criterion): path.read_bytes() for path in criterion.rglob("*") if path.is_file()}
    native_boundary.clear()

    with pytest.raises(ValueError, match="Rename saved baseline"):
        benchmark_utils._measure_suite(tmp_path, TARGETS, save_baseline="last")

    assert native_boundary == []
    assert {path.relative_to(criterion): path.read_bytes() for path in criterion.rglob("*") if path.is_file()} == prior


@pytest.mark.parametrize("scenario", tuple(product(("run-release-signal", "generate-summary"), ("exit", "timeout"), (False, True))))
def test_discovery_failures_keep_captured_diagnostics_and_prior_output(
    tmp_path: Path,
    native_boundary: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    scenario: tuple[str, str, bool],
) -> None:
    """Both CLI routes retain captured discovery diagnostics before any sampling."""
    command, failure, as_bytes = scenario
    for target in benchmark_utils.RELEASE_SIGNAL_BENCH_TARGETS:
        (tmp_path / "benches" / f"{target}.rs").write_text("// discovery fixture\n", encoding="utf-8")
    prior = tmp_path / "target/criterion/prior"
    prior.parent.mkdir(parents=True)
    prior.write_bytes(b"previous measurement")
    report = tmp_path / "summary.md"
    report.write_bytes(b"previous summary")
    stdout = "fixture setup diagnostic\n"
    stderr = "compiler error diagnostic\n"
    output = stdout.encode() if as_bytes else stdout
    diagnostic = stderr.encode() if as_bytes else stderr

    def fail_discovery(*_args: object, **_kwargs: object) -> None:
        native_boundary.append("discovery-failure")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(["cargo", "bench", "--list"], 30, output=output, stderr=diagnostic)
        raise subprocess.CalledProcessError(101, ["cargo", "bench", "--list"], output=output, stderr=diagnostic)

    monkeypatch.setattr(criterion_inventory, "run_command", fail_discovery)
    monkeypatch.setattr(benchmark_utils.PerformanceSummaryGenerator, "_get_version_date", lambda _self: "2026-10-17")
    options = ["--suite", "ci"] if command == "run-release-signal" else ["--run-benchmarks", "--output", str(report)]
    args = benchmark_utils.create_argument_parser().parse_args([command, *options])

    with pytest.raises(SystemExit) as raised:
        benchmark_utils.execute_command(args, tmp_path)

    assert raised.value.code == 1
    captured = capsys.readouterr()
    assert stdout in captured.err
    assert stderr in captured.err
    assert native_boundary == ["discovery-failure"]
    assert prior.read_bytes() == b"previous measurement"
    assert report.read_bytes() == b"previous summary"
    assert not (prior.parent / "delaunay-evidence.json").exists()


def test_delaunay_policy_keeps_each_declared_group_count() -> None:
    """The independent oracle covers every nondefault release sampling group."""
    expected = {
        "tds_new_2d/tds_new/10": 10,
        "bistellar_flips_5d/flip": 10,
        "explicit_import/case": 10,
        "proof_boundaries/case": 10,
        "locate/no_hint/3d/case": 10,
        "boundary_facets/case": 25,
        "convex_hull/case": 20,
        "convex_hull_queries/case": 20,
        "incremental_insert/case": 15,
        "realization_validation/case": 100,
        "random/insphere_1000_queries": 100,
        **CASES,
    }
    assert dict(benchmark_utils._sampling_policy(tuple(expected)).sample_counts) == expected


@pytest.mark.parametrize("stage", ["preflight", "sampling"])
def test_controller_changes_reject_measurement(tmp_path: Path, native_boundary: list[str], monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    """The controller lies outside the measured checkout's Rust file inventory."""
    controller = tmp_path / "controller.py"
    controller.write_text("# original controller\n", encoding="utf-8")
    monkeypatch.setattr(benchmark_utils, "__file__", str(controller))
    owner = benchmark_utils if stage == "preflight" else measurement
    name = "run_cargo_live" if stage == "preflight" else "run_command_live"
    original = getattr(owner, name)

    def change_controller(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)
        controller.write_text("# edited controller\n", encoding="utf-8")
        return result

    monkeypatch.setattr(owner, name, change_controller)
    prior = tmp_path / "target/criterion/prior"
    prior.parent.mkdir(parents=True)
    prior.write_bytes(b"previous successful generation")
    with pytest.raises(ValueError, match="changed during"):
        benchmark_utils._measure_suite(tmp_path, TARGETS)
    assert prior.read_bytes() == b"previous successful generation"
    assert not (prior.parent / "delaunay-evidence.json").exists()
    assert ("measure" in native_boundary) == (stage == "sampling")
