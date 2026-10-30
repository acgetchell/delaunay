"""Consumer policy over the installed independent-acquisition contract.

Tiny native Python fixtures exercise workflow correctness, never performance.
Git revision resolution is modeled; these tests do not mutate any Git state.
"""

import json
import os
import shutil
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest
from research_repo_tools import measurement, performance_release
from research_repo_tools.common_measurement import CompleteMeasurementPlan, DiscoveryPolicy, measure_prepared_pair, parse_plan, validate_run_policy
from research_repo_tools.complete_runs import ratio_incompatibilities, render_run
from research_repo_tools.evidence import load_evidence, publish_evidence
from research_repo_tools.files import replace_many
from research_repo_tools.host_metadata import HostMetadata
from research_repo_tools.measurement import load_measurement
from research_repo_tools.performance_release import load_performance_release, load_readme_plan, prepare_performance
from research_repo_tools.performance_reports import load_report_plan
from research_repo_tools.process import run_command
from research_repo_tools.publication import publish_publication
from research_repo_tools.release_pairs import ReleasePair

import benchmark_utils

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
    declaration.update(
        sources=["src/*.rs"],
        harness=["benches/*.py"],
        workloads=["benches/*.py"],
        probes={"python": [sys.executable, "--version"]},
        dependencies={},
        backend="worktree",
        compatible=[],
    )
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


def test_actual_declarations_preserve_scientific_counts_and_release_membership() -> None:
    plan = load_measurement(ROOT, "tooling/measurement-release-signal.toml")
    assert isinstance(plan, CompleteMeasurementPlan)
    assert (plan.harness_mode, plan.backend, plan.saved_sample, plan.raw_files) == ("independent", "export", "retained", ("tukey.json",))
    for phase in (plan.baseline, plan.current):
        assert isinstance(phase.policy, DiscoveryPolicy)
        counts = dict(phase.policy.sample_counts)
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
            assert len(json.loads(case.sample)["times"]) == sample.policy.count_for(case.full_id)
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


def publication_fixture(tmp_path: Path) -> CompleteMeasurementPlan:
    """Keep real selections/counts while substituting a tiny native executable."""
    declaration = tomllib.loads((ROOT / "tooling/measurement-release-signal.toml").read_text(encoding="utf-8"))
    cases = declaration["phases"]["current"]["sample-counts"]
    (tmp_path / "tooling").mkdir()
    for name in ("performance-report.toml", "performance-publication.toml", "performance-release.toml"):
        shutil.copyfile(ROOT / "tooling" / name, tmp_path / "tooling" / name)
    # The native executable, input paths, host and export boundary are modeled.
    # The counts, semantic IDs and publication selections are the consumer's.
    sections = [
        'schema = 4\nharness-mode = "independent"\nbackend = "worktree"\n'
        'sources = ["src/*.rs"]\nharness = ["benches/*.py"]\nworkloads = ["benches/*.py"]\n'
        'saved-sample = "retained"\nraw-files = ["tukey.json"]\n'
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
    (tmp_path / "tooling/measurement-release-signal.toml").write_text("\n".join(sections), encoding="utf-8")
    plan = load_measurement(tmp_path, "tooling/measurement-release-signal.toml")
    assert isinstance(plan, CompleteMeasurementPlan)
    return plan


def test_consumer_preparation_strict_promotion_and_readme_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    retained = measure_prepared_pair(tmp_path / "baseline", tmp_path / "current", plan, pair)
    assert not ratio_incompatibilities(retained)
    assert len(dict(validate_run_policy(retained, plan, release=True).phases)["current"].cases) == 173
    # Publication checks the original measured current inputs, never a later host.
    for name in ("src", "benches"):
        shutil.copytree(tmp_path / "current" / name, tmp_path / name)
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "consumer"\nversion = "0.8.3"\n', encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "prefix\n<!-- PERFORMANCE_RELEASE_TABLE:BEGIN -->\nold snapshot\n<!-- PERFORMANCE_RELEASE_TABLE:END -->\nsuffix\n",
        encoding="utf-8",
    )
    settings = load_performance_release(tmp_path, "tooling/performance-release.toml")
    # Read-only Git/ref/provider boundaries are modeled; no repository is created.
    monkeypatch.setattr(performance_release, "select_measurement_pair", lambda *_args, **_kwargs: (pair, False))
    monkeypatch.setattr(performance_release, "resolve_revision", lambda _root, ref: ("b" if ref == pair.baseline else "a") * 40)
    monkeypatch.setattr(performance_release, "run_git_bytes", lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, b"", b""))
    record = prepare_performance(tmp_path, settings)
    assert record.identities == {"baseline": ("v0.8.2", "b" * 40), "current": ("v0.8.3", "a" * 40)}
    publish_evidence(retained, tmp_path / settings.payload, tmp_path / settings.manifest)
    publish_publication(load_report_plan(tmp_path, settings.report, payload=settings.payload, manifest=settings.manifest))
    shutil.rmtree(tmp_path / "baseline")
    shutil.rmtree(tmp_path / "current")
    (tmp_path / settings.payload).unlink()
    (tmp_path / settings.manifest).unlink()
    publication = load_readme_plan(tmp_path, settings)
    publish_publication(publication)
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("prefix\n")
    assert readme.endswith("\nsuffix\n")
    assert "Baseline v0.8.2" in readme
    assert "Current v0.8.3" in readme
    assert "5D construction (12 vertices)" in readme
    assert "Baseline / current" in readme
    assert (tmp_path / "docs/assets/bench/release-performance.csv").is_file()
    assert not load_readme_plan(tmp_path, settings).changed_paths
    stale = load_readme_plan(tmp_path, settings)
    previous = (tmp_path / "README.md").read_bytes()
    (tmp_path / "src/lib.rs").write_text("// changed after measurement\n", encoding="utf-8")
    with pytest.raises(ValueError, match="measured source"):
        load_readme_plan(tmp_path, settings)
    with pytest.raises(ValueError, match="changed after planning"):
        publish_publication(stale)
    assert (tmp_path / "README.md").read_bytes() == previous
