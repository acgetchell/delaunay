"""Consumer policy over the installed complete-acquisition contract.

Tiny native Python fixtures exercise workflow correctness, never performance.
Git revision resolution is modeled; these tests do not mutate any Git state.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest
from research_repo_tools import (
    candidate_reuse,
    common_measurement,
    dependency_provenance,
    measurement,
    performance_release,
    publication as publication_api,
    release_interfaces,
    source_exports,
)
from research_repo_tools.common_measurement import CompleteMeasurementPlan, DiscoveryPolicy, measure_prepared_pair, parse_plan, validate_run_policy
from research_repo_tools.complete_runs import ratio_incompatibilities, render_run
from research_repo_tools.evidence import deterministic_json, load_evidence, publish_evidence
from research_repo_tools.files import replace_many
from research_repo_tools.host_metadata import HostMetadata
from research_repo_tools.measurement import load_measurement
from research_repo_tools.performance_progress import ProgressReporter
from research_repo_tools.performance_release import load_performance_release, load_readme_plan, prepare_performance
from research_repo_tools.performance_reports import load_report_plan
from research_repo_tools.process import run_command
from research_repo_tools.publication import publish_publication
from research_repo_tools.release_assets import GitHubRelease, ReleaseTarget
from research_repo_tools.release_pairs import ReleasePair
from research_repo_tools.run_reports import load_latest_run
from research_repo_tools.scheduled_runs import ScheduledRun, select_trial
from research_repo_tools.suite_measurement import SuiteMeasurementPlan
from research_repo_tools.suite_runs import comparison_incompatibilities, parse_suite_run
from research_repo_tools.trial_schedule import SequentialOrder, TrialSchedule

import benchmark_utils
import release_candidate

ROOT = Path(__file__).resolve().parents[2]
COUNTS = (10, 12, 15, 20, 25, 100)
NATIVE = """import json, os, shutil, sys
from pathlib import Path
MARKER = {marker!r}
CASES = {cases!r}
mode = sys.argv[1]
with Path(os.environ["EVENTS"]).open("a") as output:
    output.write(MARKER + ":" + mode + "\\n")
if mode == "list":
    for case in CASES:
        print(case + ": benchmark")
elif mode == "gate":
    assert Path("benches/native.py").read_text().find(repr(MARKER)) >= 0
    if MARKER == "baseline":
        assert Path("benches/baseline_only.py").read_bytes() == b"# original baseline-only harness\\n"
else:
    home = Path(os.environ["CRITERION_HOME"])
    estimate = {{"point_estimate": 100, "confidence_interval": {{"lower_bound": 90, "upper_bound": 110, "confidence_level": 0.95}}}}
    for index, (case, count) in enumerate(CASES.items()):
        destination = home / ("sanitized_" + str(index)) / "new"
        destination.mkdir(parents=True)
        (destination / "benchmark.json").write_text(json.dumps({{"full_id": case, "group_id": "workload"}}))
        (destination / "estimates.json").write_text(json.dumps({{"mean": estimate, "median": estimate}}))
        (destination / "sample.json").write_text(json.dumps({{"iters": [1] * count, "times": [100] * count}}))
        (destination / "tukey.json").write_bytes(b"[1,2,3,4]\\r\\n")
        shutil.copytree(destination, destination.with_name("retained"))
"""


def fixture_plan(root: Path, monkeypatch: pytest.MonkeyPatch) -> CompleteMeasurementPlan:
    """Load the real schema with native discovery, gates and heterogeneous counts."""
    declaration = tomllib.loads((ROOT / "tooling/measurement-release-signal.toml").read_text(encoding="utf-8"))
    declaration = declaration["suites"][0]["measurement"]
    declaration.update(
        sources=["src/*.rs"],
        harness=["benches/*.py"],
        workloads=["benches/*.py"],
        probes={"python": [sys.executable, "--version"]},
        dependencies={},
        backend="worktree",
        compatible=[],
    )
    declaration["harness-mode"] = "independent"
    for phase in ("baseline", "current"):
        checkout = root / phase
        (checkout / "src").mkdir(parents=True)
        (checkout / "src/lib.rs").write_text(f"// {phase} scientific source\n", encoding="utf-8")
        (checkout / "benches").mkdir()
        cases = {f"workload / count {count}": count for count in COUNTS}
        cases[f"workload / {phase} only"] = 10
        (checkout / "benches/native.py").write_text(NATIVE.format(marker=phase, cases=cases), encoding="utf-8")
        if phase == "baseline":
            (checkout / "benches/baseline_only.py").write_bytes(b"# original baseline-only harness\n")
        declaration["phases"][phase] = {
            "command": [sys.executable, "benches/native.py", "measure"],
            "gate": [sys.executable, "benches/native.py", "gate"],
            "sample-counts": cases,
            "environment": {"EVENTS": str(root / "events.log"), "CRITERION_HOME": "target/criterion"},
            "discovery": {"required": list(cases), "suites": [{"name": "native", "command": [sys.executable, "benches/native.py", "list"]}]},
        }
    # The installed parser is the same loader used by the actual consumer CLI.
    monkeypatch.setattr(measurement, "resolve_revision", lambda checkout, _reference: ("b" if checkout.name == "baseline" else "a") * 40)
    return parse_plan(declaration)


def test_retained_report_migration_preserves_frozen_history() -> None:
    """Verify all historical bytes without inventing a schedule for old evidence."""
    before = {path: path.read_bytes() for path in (ROOT / "docs/performance-runs").rglob("*") if path.is_file()}
    historical = load_latest_run(ROOT, "docs/performance-runs")
    assert historical.payload_schema == "research-repo-tools/complete-run/v3"
    # A historic acquisition remains readable, but cannot meet the new policy.
    with pytest.raises(ValueError, match="scheduled run"):
        load_report_plan(ROOT, "tooling/performance-report.toml")
    assert {path: path.read_bytes() for path in before} == before


def test_actual_declarations_preserve_scientific_counts_and_release_membership() -> None:
    plan = load_measurement(ROOT, "tooling/measurement-release-signal.toml")
    assert isinstance(plan, SuiteMeasurementPlan)
    assert plan.schedule == TrialSchedule(1, SequentialOrder())
    assert plan.outputs == ("README.md", "docs/assets/bench/release-performance.csv")
    assert len(plan.suites) == 1
    plan = plan.suites[0].measurement
    assert isinstance(plan, CompleteMeasurementPlan)
    assert (plan.harness_mode, plan.backend, plan.saved_sample, plan.raw_files) == ("common", "export", "retained", ("tukey.json",))
    for phase in (plan.baseline, plan.current):
        assert isinstance(phase.policy, DiscoveryPolicy)
        counts = dict(phase.policy.sample_counts)
        assert len(counts) == 173
        assert set(counts.values()) == {10, 15, 20, 25, 100}
        assert dict(benchmark_utils._sampling_policy(tuple(counts)).sample_counts) == counts
        assert tuple(suite.name for suite in phase.policy.discovery.suites) == benchmark_utils.RELEASE_SIGNAL_BENCH_TARGETS
        assert set(phase.policy.discovery.required) == set(counts)
        assert all(target not in phase.command for target in ("memory_footprint", "traversal", "validation_components", "hull_queries", "construction_scaling"))
    topology = load_measurement(ROOT, "tooling/measurement-topology.toml")
    assert isinstance(topology, CompleteMeasurementPlan)
    counts = dict(topology.current.policy.sample_counts)
    assert {count for name, count in counts.items() if "/4d/" in name} == {12}
    assert dict(benchmark_utils._sampling_policy(tuple(counts)).sample_counts) == counts
    historical = load_measurement(ROOT, "tooling/measurement-release-signal-v082.toml")
    assert isinstance(historical, CompleteMeasurementPlan)
    baseline = dict(historical.baseline.policy.sample_counts)
    current = dict(historical.current.policy.sample_counts)
    assert len(baseline) == len(current) == 173
    assert len(baseline.keys() - current.keys()) == 20
    assert len(current.keys() - baseline.keys()) == 20
    assert all(name.startswith("locate/") for name in baseline.keys() ^ current.keys())
    assert dict(benchmark_utils._sampling_policy(tuple(baseline)).sample_counts) == baseline
    probe = dict(plan.measurement.probes)["build-policy"]
    assert dict(topology.measurement.probes)["build-policy"] == probe
    assert dict(historical.measurement.probes)["build-policy"] == probe
    observed = run_command(
        probe[0],
        probe[1:],
        cwd=ROOT,
        env={**os.environ, "RUSTFLAGS": "-Cdebug-assertions=off", "CARGO_PROFILE_PERF_LTO": "thin", "CARGO_REGISTRY_TOKEN": "fixture-credential"},
    ).stdout
    environment = json.loads(observed)["environment"]
    assert environment["RUSTFLAGS"] == "-Cdebug-assertions=off"
    assert environment["CARGO_PROFILE_PERF_LTO"] == "thin"
    assert "CARGO_REGISTRY_TOKEN" not in environment
    assert "fixture-credential" not in observed


@pytest.mark.parametrize("suite", ["release-signal", "release-signal-v082", "topology"])
def test_complete_measurement_commands_do_not_add_positional_filters(suite: str) -> None:
    """Original native harnesses interpret bare option values as case filters."""
    plan = load_measurement(ROOT, f"tooling/measurement-{suite}.toml")
    if isinstance(plan, SuiteMeasurementPlan):
        plan = plan.suites[0].measurement
    assert isinstance(plan, CompleteMeasurementPlan)
    for phase in (plan.baseline, plan.current):
        criterion_arguments = phase.command[phase.command.index("--") + 1 :]
        assert all(argument.startswith("-") for argument in criterion_arguments)


def test_release_harness_unifies_changed_fixtures_before_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changed harness bytes become comparable without rewriting library source."""
    release_plan = load_measurement(ROOT, "tooling/measurement-release-signal.toml")
    assert isinstance(release_plan, SuiteMeasurementPlan)
    release_plan = release_plan.suites[0].measurement
    assert isinstance(release_plan, CompleteMeasurementPlan)
    fixture = fixture_plan(tmp_path, monkeypatch)
    plan = replace(fixture, harness_mode=release_plan.harness_mode, baseline=replace(fixture.baseline, policy=fixture.current.policy))
    original_sources = {phase: (tmp_path / phase / "src/lib.rs").read_bytes() for phase in ("baseline", "current")}
    current_harness = (tmp_path / "current/benches/native.py").read_bytes()
    retained = measure_prepared_pair(tmp_path / "baseline", tmp_path / "current", plan, ReleasePair("v0.8.4", "v0.8.3"))
    run = validate_run_policy(retained, plan, release=True)
    assert not ratio_incompatibilities(retained)
    for phase, sample in run.phases:
        assert (tmp_path / phase / "src/lib.rs").read_bytes() == original_sources[phase]
        assert (tmp_path / phase / "benches/native.py").read_bytes() == current_harness
        assert {case.full_id for case in sample.cases} == set(dict(fixture.current.policy.sample_counts))
    assert not (tmp_path / "baseline/benches/baseline_only.py").exists()
    assert (tmp_path / "events.log").read_text(encoding="utf-8").splitlines() == [
        "current:list",
        "current:list",
        "current:gate",
        "current:gate",
        "current:measure",
        "current:measure",
    ]


@pytest.mark.parametrize("same_release", [False, True])
def test_installed_independent_pair_preserves_native_harnesses_and_union_after_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    same_release: bool,
) -> None:
    plan = fixture_plan(tmp_path, monkeypatch)
    harness = {phase: (tmp_path / phase / "benches/native.py").read_bytes() for phase in ("baseline", "current")}
    retained = measure_prepared_pair(
        tmp_path / "baseline",
        tmp_path / "current",
        plan,
        ReleasePair("v0.8.3", "v0.8.3" if same_release else "v0.8.2"),
        working_tree=same_release,
    )
    run = validate_run_policy(retained, plan)
    for phase, sample in run.phases:
        assert (tmp_path / phase / "benches/native.py").read_bytes() == harness[phase]
        for case in sample.cases:
            assert len(json.loads(case.sample)["times"]) == dict(sample.policy.sample_counts)[case.full_id]
            assert dict(case.raw_files)["tukey.json"] == b"[1,2,3,4]\r\n"
    events = (tmp_path / "events.log").read_text(encoding="utf-8").splitlines()
    assert events == ["baseline:list", "current:list", "baseline:gate", "current:gate", "baseline:measure", "current:measure"]
    assert (tmp_path / "baseline/benches/baseline_only.py").read_bytes() == b"# original baseline-only harness\n"
    publish_evidence(retained, tmp_path / "run.json", tmp_path / "evidence.json")
    for phase in ("baseline", "current"):
        shutil.rmtree(tmp_path / phase)
    offline = load_evidence(tmp_path / "run.json", tmp_path / "evidence.json")
    assert validate_run_policy(offline, plan) == run
    report = render_run(offline).decode()
    assert "baseline only" in report
    assert "current only" in report
    assert "count 100" in report
    assert "suppressed" in report
    assert any("harness_sha256" in reason for reason in ratio_incompatibilities(offline))
    with pytest.raises(ValueError, match="ineligible"):
        validate_run_policy(offline, plan, release=True)
    # Exercise installed CLI entry points outside the source tree after cleanup.
    run_command(
        sys.executable,
        ["-m", "research_repo_tools", "--root", str(tmp_path), "performance", "verify", "run.json", "evidence.json"],
        cwd=tmp_path,
    )
    run_command(
        sys.executable,
        ["-m", "research_repo_tools", "--root", str(tmp_path), "performance", "render", "run.json", "evidence.json", "--output", "offline.md"],
        cwd=tmp_path,
    )
    assert "baseline only" in (tmp_path / "offline.md").read_text(encoding="utf-8")
    configuration = tmp_path / "report.toml"
    configuration.write_text('schema = 2\ncurrent = "report.md"\narchive = "history"\ntitle = "Consumer fixture"\nrelease = false\n', encoding="utf-8")
    publication = load_report_plan(tmp_path, "report.toml", payload="run.json", manifest="evidence.json")
    publish_publication(publication)
    (tmp_path / "run.json").unlink()
    (tmp_path / "evidence.json").unlink()
    rerender = load_report_plan(tmp_path, "report.toml")
    assert not rerender.changed_paths
    prior = {path: path.read_bytes() for path in (tmp_path / "history").rglob("*") if path.is_file()}
    prior[tmp_path / "report.md"] = (tmp_path / "report.md").read_bytes()
    configuration.write_text(configuration.read_text(encoding="utf-8") + "# changed after planning\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        publish_publication(rerender)
    assert {path: path.read_bytes() for path in prior} == prior


@pytest.mark.parametrize("damage", ["gate", "count", "saved", "missing", "unexpected", "changed-input"])
def test_invalid_acquisition_never_replaces_previous_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str) -> None:
    plan = fixture_plan(tmp_path, monkeypatch)
    script = tmp_path / "current/benches/native.py"
    source = script.read_text(encoding="utf-8")
    if damage == "gate":
        source = source.replace('elif mode == "gate":', 'elif mode == "gate":\n    raise RuntimeError("scientific gate failed")')
    elif damage == "count":
        source = source.replace('home = Path(os.environ["CRITERION_HOME"])', 'CASES[next(iter(CASES))] -= 1\n    home = Path(os.environ["CRITERION_HOME"])')
    elif damage == "saved":
        source = source.replace(
            'shutil.copytree(destination, destination.with_name("retained"))',
            'shutil.copytree(destination, destination.with_name("retained"))\n'
            '        (destination.with_name("retained") / "tukey.json").write_bytes(b"[5,6,7,8]")',
        )
    elif damage == "unexpected":
        source = source.replace("for index, (case, count)", 'CASES["unexpected"] = 10\n    for index, (case, count)')
    elif damage == "missing":
        source = source.replace("mode = sys.argv[1]", "CASES.pop(next(iter(CASES)))\nmode = sys.argv[1]")
    else:
        source = source.replace(
            'home = Path(os.environ["CRITERION_HOME"])',
            'Path("src/lib.rs").write_text("changed during execution")\n    home = Path(os.environ["CRITERION_HOME"])',
        )
    script.write_text(source, encoding="utf-8")
    prior = {tmp_path / "run.json": b"previous payload", tmp_path / "evidence.json": b"previous evidence"}
    replace_many(prior)
    with pytest.raises((ValueError, RuntimeError, subprocess.SubprocessError)):
        measure_prepared_pair(tmp_path / "baseline", tmp_path / "current", plan, ReleasePair("v0.8.3", "v0.8.2"))
    assert {path: path.read_bytes() for path in prior} == prior


def test_changed_acquisition_policy_rejects_retained_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = fixture_plan(tmp_path, monkeypatch)
    retained = measure_prepared_pair(tmp_path / "baseline", tmp_path / "current", plan, ReleasePair("v0.8.3", "v0.8.2"))
    with pytest.raises(ValueError, match="mode/schema"):
        validate_run_policy(retained, replace(plan, harness_mode="common"))


def test_same_label_pair_requires_distinct_scientific_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = fixture_plan(tmp_path, monkeypatch)
    (tmp_path / "current/src/lib.rs").write_bytes((tmp_path / "baseline/src/lib.rs").read_bytes())
    with pytest.raises(ValueError, match="distinct source"):
        measure_prepared_pair(tmp_path / "baseline", tmp_path / "current", plan, ReleasePair("v0.8.3", "v0.8.3"), working_tree=True)


def publication_fixture(tmp_path: Path) -> SuiteMeasurementPlan:
    """Keep real selections/counts while substituting a tiny native executable."""
    declaration = tomllib.loads((ROOT / "tooling/measurement-release-signal.toml").read_text(encoding="utf-8"))
    cases = declaration["suites"][0]["measurement"]["phases"]["current"]["sample-counts"]
    (tmp_path / "tooling").mkdir()
    for name in ("performance-report.toml", "performance-publication.toml", "performance-release.toml", "release-evidence.toml"):
        shutil.copyfile(ROOT / "tooling" / name, tmp_path / "tooling" / name)
    # The native executable, input paths, host and export boundary are modeled.
    # The counts, semantic IDs and publication selections are the consumer's.
    sections = [
        'schema = 4\nharness-mode = "common"\nbackend = "export"\n'
        'sources = ["src/*.rs"]\nharness = ["benches/*.py"]\nworkloads = ["benches/*.py"]\n'
        'saved-sample = "retained"\nraw-files = ["tukey.json"]\n'
        '[context]\nrepository = "acgetchell/delaunay"\nacquisition-owner = "local"\n'
        "[probes]\npython = " + json.dumps([sys.executable, "--version"]) + "\n",
    ]
    for phase in ("baseline", "current"):
        checkout = tmp_path / phase
        (checkout / "src").mkdir(parents=True)
        (checkout / "src/lib.rs").write_text(f"// {phase} scientific source\n", encoding="utf-8")
        (checkout / "benches").mkdir()
        (checkout / "benches/native.py").write_text(NATIVE.format(marker="baseline", cases=cases), encoding="utf-8")
        (checkout / "benches/baseline_only.py").write_bytes(b"# original baseline-only harness\n")
        sections.append(
            f"[phases.{phase}]\ncommand = {json.dumps([sys.executable, 'benches/native.py', 'measure'])}\n"
            f"gate = {json.dumps([sys.executable, 'benches/native.py', 'gate'])}\n"
            f"environment = {{ EVENTS = {json.dumps(str(tmp_path / 'events.log'))}, CRITERION_HOME = 'target/criterion' }}\n"
            f"[phases.{phase}.sample-counts]\n"
            + "\n".join(f"{json.dumps(name)} = {count}" for name, count in cases.items())
            + f"\n[phases.{phase}.discovery]\nrequired = {json.dumps(list(cases))}\n"
            f"[[phases.{phase}.discovery.suites]]\nname = 'native'\n"
            f"command = {json.dumps([sys.executable, 'benches/native.py', 'list'])}\n",
        )
    sections.extend(f'[[series]]\nname = "{phase}"\nphase = "{phase}"\nselect = {{ include = ["*"] }}\n' for phase in ("baseline", "current"))
    leaf = "\n".join(sections)
    leaf = "\n".join(line.replace("[", "[suites.measurement.", 1) if line.startswith("[") else line for line in leaf.splitlines())
    leaf = leaf.replace("[suites.measurement.[", "[[suites.measurement.")
    wrapper = (
        'schema = 6\noutputs = ["README.md", "docs/assets/bench/release-performance.csv"]\n'
        '[schedule]\norder = "sequential"\ntrials = 1\n'
        '[[suites]]\nname = "release-signal"\ncontract = "delaunay-public-api-v1"\n[suites.measurement]\n'
    )
    (tmp_path / "tooling/measurement-release-signal.toml").write_text(wrapper + leaf, encoding="utf-8", newline="\n")
    plan = load_measurement(tmp_path, "tooling/measurement-release-signal.toml")
    assert isinstance(plan, SuiteMeasurementPlan)
    return plan


def model_source_reads(root: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, bytes]]:
    """Model immutable Git blobs, retaining actual export and native fixture execution."""
    current = {
        path.relative_to(root).as_posix(): path.read_bytes() for name in ("src", "benches", "tooling") for path in (root / name).rglob("*") if path.is_file()
    }
    current.update({name: (root / name).read_bytes() for name in ("Cargo.toml", "README.md")})
    baseline = {**current, "src/lib.rs": b"// baseline scientific source\n"}
    revisions = {"b" * 40: baseline, "a" * 40: current}

    def oid(data: bytes) -> bytes:
        return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data, usedforsecurity=False).hexdigest().encode()

    def git(_directory: Path, *args: str, input: bytes | None = None) -> bytes:  # noqa: A002 - Match the subprocess byte-input interface.
        if args == ("rev-parse", "--show-prefix"):
            return b""
        if args[0] == "rev-parse":
            ref = args[-1].removeprefix("refs/tags/").removesuffix("^{commit}")
            return ((ref if ref in revisions else "b" * 40 if ref == "v0.8.2" else next(reversed(revisions))) + "\n").encode()
        if args[0] == "ls-tree":
            return b"".join(b"100644 blob " + oid(data) + b"\t" + name.encode() + b"\0" for name, data in sorted(revisions[args[-1]].items()))
        assert args == ("cat-file", "--batch"), args
        assert input is not None
        blobs = {oid(data): data for files in revisions.values() for data in files.values()}
        return b"".join(key + b" blob " + str(len(blobs[key])).encode() + b"\n" + blobs[key] + b"\n" for key in input.splitlines())

    monkeypatch.setattr(source_exports, "_git", git)
    monkeypatch.setattr(source_exports, "select_files", lambda *_args, **_kwargs: tuple(sorted(next(reversed(revisions.values())))))
    return revisions


def acquire_owned_fixture(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Exercise the actual before-promotion hook and immutable retention over native fixtures."""
    owner = root.parent / (root.name + "-owner")
    monkeypatch.setattr(release_candidate, "selection_directory", lambda _root: root.parent / (root.name + "-selection"))
    monkeypatch.setattr(release_candidate, "check_release_dependencies", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(dependency_provenance, "check_release_dependencies", lambda *_args, **_kwargs: ())
    release_candidate.acquire(root, owner, "target/checkpoint", resume=False, progress=ProgressReporter(mode="none"))
    assert release_candidate.selected_owner(root) == owner
    assert (owner / "publication.sha256").read_text(encoding="utf-8").strip() == hashlib.sha256((owner / "publication.json").read_bytes()).hexdigest()
    events = (root / "events.log").read_bytes()
    assert events.count(b":measure\n") == 2
    monkeypatch.setattr(release_candidate, "measure_release", lambda *_args, **_kwargs: pytest.fail("publication retries must not acquire"))
    release_candidate.capture(root, owner)
    settings = load_performance_release(root, "tooling/performance-release.toml")
    publish_publication(load_report_plan(root, settings.report, payload=settings.payload, manifest=settings.manifest))
    assert (root / "events.log").read_bytes() == events
    return owner


def assert_final_candidate_binding(root: Path, owner: Path, revisions: dict[str, dict[str, bytes]], monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the installed whole-tree and archive verifier with only native CI modeled."""
    final = {name: (root / name).read_bytes() for name in revisions["a" * 40]}
    final.update({path.relative_to(root).as_posix(): path.read_bytes() for path in (root / "docs").rglob("*") if path.is_file()})
    revisions["c" * 40] = final
    monkeypatch.setattr(release_candidate, "resolve_revision", lambda *_args: "c" * 40)
    monkeypatch.setattr(candidate_reuse, "require_release_checkout", lambda *_args: None)
    monkeypatch.setattr(release_interfaces, "require_release_checkout", lambda *_args: None)
    monkeypatch.setattr(release_interfaces, "resolve_revision", lambda *_args: "c" * 40)
    native = deterministic_json(
        {
            "policy": {
                "guarantee": "cargo-archive",
                "repository": "acgetchell/delaunay",
                "event": "push",
                "ref": "refs/heads/main",
                "signing_job": "sign-package",
            },
            "package": {"name": "delaunay", "version": "0.8.3"},
            "run": {"id": 123, "run_attempt": 1, "head_sha": "c" * 40, "path": ".github/workflows/ci.yml"},
            "artifact": {"id": 456, "digest": "sha256:" + "d" * 64},
        }
    )

    def native_evidence(*_args: object, expected: bytes | None = None, **_kwargs: object) -> bytes:
        if expected is not None:
            assert expected == native
        return native

    monkeypatch.setattr(candidate_reuse, "verify_release_evidence", native_evidence)
    monkeypatch.setattr(common_measurement, "run_command_live", lambda *_args, **_kwargs: pytest.fail("Step 2 must not acquire, preflight or build"))
    original_events = (root / "events.log").read_bytes()
    accepted = release_candidate.verify(root, owner)
    selected = accepted.read_bytes()
    assert (accepted.parent / "native.json").read_bytes() == native
    assert release_candidate.verify_existing(root, owner, "reviewed").read_bytes() == selected
    artifact = release_candidate.archive(owner, "c" * 40)
    policy = candidate_reuse.parse_candidate_policy((accepted.parent / "policy.json").read_bytes())
    retained = candidate_reuse.verify_candidate_archive(artifact, policy, selected)
    assert retained.payload == (owner / "run.json").read_bytes()
    original = artifact.read_bytes()
    assert release_candidate.archive(owner, "c" * 40).read_bytes() == original
    assert_published_candidate_reader(root, owner, accepted, artifact, monkeypatch)
    assert (root / "events.log").read_bytes() == original_events
    before = final["src/lib.rs"]
    final["src/lib.rs"] = b"// changed measured source\n"
    (root / "src/lib.rs").write_bytes(final["src/lib.rs"])
    with pytest.raises(RuntimeError, match="candidate verification failed"):
        release_candidate.verify_existing(root, owner, "reviewed")
    final["src/lib.rs"] = before
    (root / "src/lib.rs").write_bytes(before)
    assert accepted.read_bytes() == selected
    assert artifact.read_bytes() == original


def assert_published_candidate_reader(root: Path, owner: Path, accepted: Path, artifact: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Read the real derived archive through an immutable release, preserving measured source."""
    target = ReleaseTarget(release_candidate.REPOSITORY, "v0.8.3", 42, "c" * 40, "v0.8.3")
    sources = {
        "policy": accepted.parent / "policy.json",
        "publication": owner / "publication.json",
        "acceptance": accepted,
        "archive": artifact,
    }
    assets = {
        role: release_candidate.AssetAssertion(f"{role}.bin", index, hashlib.sha256(path.read_bytes()).hexdigest())
        for index, (role, path) in enumerate(sources.items(), 1)
    }
    original = release_candidate.DraftTransfer(
        target,
        (owner / "acquisition.sha256").read_text(encoding="utf-8").strip(),
        (owner / "publication.sha256").read_text(encoding="utf-8").strip(),
        release_candidate.OriginalAssets(assets["policy"], assets["publication"], assets["acceptance"], assets["archive"]),
    )
    transfer = release_candidate.PublicationTransfer(
        original, release_candidate.AssetAssertion(artifact.name, 5, hashlib.sha256(artifact.read_bytes()).hexdigest())
    )
    payloads = {assets[role].name: path.read_bytes() for role, path in sources.items()}
    payloads[f"delaunay-{target.tag}-candidate.json"] = release_candidate.serialize_transfer(transfer)
    release = GitHubRelease(
        target.repository,
        target.tag,
        target.release_id,
        draft=False,
        immutable=True,
        prerelease=False,
        assets=tuple((asset.name, asset.identifier, 1, asset.sha256) for _role, asset in transfer.role_assertions()),
        title=target.tag,
    )

    def download(_root: Path, _repo: str, _tag: str, name: str, destination: Path, *, expected_sha256: str | None = None) -> None:
        payload = payloads[name]
        if expected_sha256 is not None:
            assert hashlib.sha256(payload).hexdigest() == expected_sha256
        destination.write_bytes(payload)

    with monkeypatch.context() as patch:
        patch.setattr(release_candidate, "lookup_release", lambda *_args: release)
        patch.setattr(release_candidate, "download_release_asset", download)
        directory = owner / "historical-reader"
        directory.mkdir()
        evidence = release_candidate.read_published_archive(root, target.tag, target.commit, artifact, directory)
        assert evidence.payload == (owner / "run.json").read_bytes()
        assert dict(evidence.sources)["current"].revision == "a" * 40
        release = replace(release, assets=tuple((name, identifier + 100, size, digest) for name, identifier, size, digest in release.assets))
        with pytest.raises(ValueError, match="asset identity differs"):
            release_candidate.read_published_archive(root, target.tag, target.commit, artifact, directory)


@pytest.fixture
def publication_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Model existing-baseline and absent-prospective tag reads only."""

    def git_read(arguments: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[bytes]:
        assert cwd == tmp_path
        if arguments == ["--no-pager", "show-ref", "--verify", "--quiet", "refs/tags/v0.8.3"]:
            return subprocess.CompletedProcess(arguments, 1, b"", b"")
        if arguments == ["--no-pager", "--no-replace-objects", "rev-parse", "--show-prefix"]:
            return subprocess.CompletedProcess(arguments, 0, b"", b"")
        if arguments == ["--no-pager", "--no-replace-objects", "check-ref-format", "refs/tags/v0.8.2"]:
            return subprocess.CompletedProcess(arguments, 0, b"", b"")
        assert arguments == ["--no-pager", "--no-replace-objects", "rev-parse", "--verify", "refs/tags/v0.8.2^{commit}"]
        return subprocess.CompletedProcess(arguments, 0, b"b" * 40 + b"\n", b"")

    monkeypatch.setattr(publication_api, "run_git_bytes", git_read)


@pytest.mark.usefixtures("publication_git")
@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_consumer_preparation_strict_promotion_and_readme_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, newline: str) -> None:
    """Execute the real report/README declarations over synthetic native observations."""
    plan = publication_fixture(tmp_path)
    pair = ReleasePair("v0.8.3", "v0.8.2")
    # Model a complete, stable host before acquisition so the success path is
    # portable to restricted hosts; these synthetic timings make no speed claim.
    monkeypatch.setattr(
        measurement,
        "capture_host",
        lambda *_args, **_kwargs: HostMetadata("fixture OS", "fixture arch", "fixture CPU", 4, 8, 1024, (("python", "fixture Python"),)),
    )
    monkeypatch.setattr(measurement, "resolve_revision", lambda checkout, _reference: ("b" if checkout.name == "baseline" else "a") * 40)
    # Publication checks the original measured current inputs, never a later host.
    for name in ("src", "benches"):
        shutil.copytree(tmp_path / "current" / name, tmp_path / name)
    (tmp_path / "Cargo.toml").write_bytes('[package]\nname = "delaunay"\nversion = "0.8.3"\n'.replace("\n", newline).encode("utf-8"))
    (tmp_path / "README.md").write_bytes(
        "prefix\n<!-- PERFORMANCE_RELEASE_TABLE:BEGIN -->\nold snapshot\n<!-- PERFORMANCE_RELEASE_TABLE:END -->\nsuffix\n".replace("\n", newline).encode()
    )
    revisions = model_source_reads(tmp_path, monkeypatch)
    settings = load_performance_release(tmp_path, "tooling/performance-release.toml")
    # Read-only Git/ref/provider boundaries are modeled; no repository is created.
    monkeypatch.setattr(performance_release, "select_measurement_pair", lambda *_args, **_kwargs: (pair, True))
    monkeypatch.setattr(performance_release, "resolve_revision", lambda _root, ref: ("b" if ref == pair.baseline else "a") * 40)
    monkeypatch.setattr(performance_release, "run_git_bytes", lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, b"", b""))
    record = prepare_performance(tmp_path, settings)
    assert record.identities == {"baseline": ("v0.8.2", "b" * 40), "current": ("v0.8.3", "a" * 40)}
    owner = acquire_owned_fixture(tmp_path, monkeypatch)
    retained = load_evidence(owner / "run.json", owner / "evidence.json")
    assert isinstance(validate_run_policy(retained, plan, release=True), ScheduledRun)
    trial = select_trial(retained, 1)
    assert not comparison_incompatibilities(trial, ("baseline", "tds_new_2d/tds_new/500"), ("current", "tds_new_2d/tds_new/500"))
    assert all(len(sample.cases) == 173 for _suite, _phase, sample, _source in parse_suite_run(trial).observations)
    shutil.rmtree(tmp_path / "baseline")
    shutil.rmtree(tmp_path / "current")
    (tmp_path / settings.payload).unlink()
    (tmp_path / settings.manifest).unlink()
    assert not (tmp_path / "docs/assets/bench/release-performance.csv").exists()
    publication = load_readme_plan(tmp_path, settings)
    publish_publication(publication)
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("prefix\n")
    assert readme.endswith("\nsuffix\n")
    assert "Baseline v0.8.2" in readme
    assert "Current v0.8.3" in readme
    assert "5D construction (12 vertices)" in readme
    assert "Baseline / current" in readme
    assert "Selection coverage:\n\n- Baseline" in readme
    assert (tmp_path / "docs/assets/bench/release-performance.csv").is_file()
    run_command(
        "uv",
        [
            "run",
            "--locked",
            "--group",
            "dev",
            "research-repo-tools",
            "toolchain",
            "run",
            "--",
            "rumdl",
            "check",
            "--config",
            str(ROOT / "pyproject.toml"),
            "--no-cache",
            str(tmp_path / "README.md"),
        ],
        cwd=ROOT,
    )
    assert not load_readme_plan(tmp_path, settings).changed_paths
    if newline == "\n":
        assert_final_candidate_binding(tmp_path, owner, revisions, monkeypatch)
    stale = load_readme_plan(tmp_path, settings)
    previous = (tmp_path / "README.md").read_bytes()
    (tmp_path / "src/lib.rs").write_text("// changed after measurement\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from measured inputs"):
        load_readme_plan(tmp_path, settings)
    with pytest.raises(ValueError, match="changed after planning"):
        publish_publication(stale)
    assert (tmp_path / "README.md").read_bytes() == previous
