"""Remote focused recipes cover the complete local CI contract without repeated lint."""

import json
import os
import shlex
import shutil
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from research_repo_tools.just_inspect import dry_run, inspect_justfile
from research_repo_tools.process import run_command

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parents[2]
NATIVE_STAGES = {
    "python-docs": {"python-check", "notebook-check", "python-fixture-lint", "_validation-doc-figures-check-if-canonical", "doc-package-check", "test-python"},
    "rust-builds": {"cargo-lock-check", "unused-deps", "clippy", "doc-check", "bench-compile", "examples"},
    "rust-integrations": {"test-integration"},
    "rust-tests": {"test-unit", "test-cli", "test-doc", "test-unit-features", "test-doc-features", "test-allocation"},
}


def test_remote_subsets_preserve_every_local_validator() -> None:
    """Production Semgrep is owned by its independently required SARIF workflow."""
    recipes = inspect_justfile(ROOT).recipes

    def validators(name: str) -> set[str]:
        dependencies = [item["recipe"] for item in recipes[name]["dependencies"] if not item["recipe"].startswith("_ensure-")]
        if not dependencies:
            return {name}
        result: set[str] = set()
        for child in dependencies:
            result.update(validators(child))
        return result

    portable, native = validators("ci-portable"), validators("ci-native")
    assert not portable & native
    assert validators("ci") == portable | native | {"semgrep-scan", "_ci-setup"}
    stages = {name: validators(f"ci-{name}") for name in NATIVE_STAGES}
    assert stages == NATIVE_STAGES
    assert set().union(*stages.values()) == native
    for first, second in combinations(stages.values(), 2):
        assert first.isdisjoint(second)
    assert {
        "python-check",
        "python-fixture-lint",
        "notebook-check",
        "test-python",
        "clippy",
        "doc-check",
        "doc-package-check",
        "test-unit",
        "test-unit-features",
        "test-allocation",
        "bench-compile",
        "examples",
    } <= native
    assert {"semgrep-test", "fmt-check", "yaml-fmt-check", "markdown-check"} <= portable


def test_native_tooling_failures_surface_before_scientific_test_buckets() -> None:
    """Exercise the aggregate expansion so harness failures do not wait for Rust."""
    commands = dry_run(ROOT, "test").stderr
    assert commands.index("uv run --locked --no-sync pytest") < commands.index("cargo nextest run")


@pytest.mark.parametrize("failure", [False, True])
def test_release_benchmark_dispatch_forwards_one_literal_tag(
    tmp_path: Path, native_console_scripts: Callable[[Path, str, dict[str, str]], Path], *, failure: bool
) -> None:
    """Exercise actual recipe forwarding into the isolated adapter without shell interpolation."""
    tag = "v0.9.0 $(touch SHOULD_NOT_EXIST); 'literal'"
    arguments = tmp_path / "arguments"
    outside = native_console_scripts(
        tmp_path,
        "import json, os, sys\nfrom pathlib import Path\n"
        "def main():\n"
        '    Path(os.environ["ARGUMENTS"]).write_text(json.dumps(sys.argv), encoding="utf-8")\n'
        '    return int(os.environ["DISPATCH_EXIT"])\n',
        {"uvx": "main"},
    )
    (tmp_path / "justfile").write_text(f"import '{(ROOT / 'justfile').as_posix()}'\n", encoding="utf-8")
    result = run_command(
        "just",
        ["release-benchmarks", tag],
        cwd=tmp_path,
        env=os.environ | {"PATH": f"{outside}{os.pathsep}{os.environ['PATH']}", "ARGUMENTS": str(arguments), "DISPATCH_EXIT": "17" if failure else "0"},
        check=False,
    )
    assert result.returncode == (17 if failure else 0), result.stderr
    assert json.loads(arguments.read_bytes())[-2:] == ["benchmarks", tag]
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()


def test_dispatch_requires_the_tag_argument() -> None:
    result = run_command("just", ["--dry-run", "release-benchmarks"], cwd=ROOT, check=False)
    assert result.returncode != 0
    assert "argument" in result.stderr


@pytest.mark.parametrize("owner", ["portable", "prepare", "native"])
@pytest.mark.parametrize("result", ["success", "failure", "skipped", "cancelled", ""])
def test_required_builds_reject_every_incomplete_dependency(owner: str, result: str) -> None:
    """Missing, cancelled and skipped work must fail the retained required checks."""
    job = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_bytes())["jobs"]["build"]
    assert job["needs"] == ["portable", "prepare", "native"]
    assert job["if"] == "${{ always() }}"
    assert job["runs-on"] == "ubuntu-latest"
    guard = job["steps"][0]
    assert guard["name"] == "Require complete native validation"
    assert guard["env"] == {f"{name.upper()}_RESULT": "${{ needs." + name + ".result }}" for name in job["needs"]}
    environment = {f"{name.upper()}_RESULT": "success" for name in job["needs"]}
    environment[f"{owner.upper()}_RESULT"] = result
    observed = run_command("bash", ["-euc", guard["run"]], env=os.environ | environment, check=False)
    assert (observed.returncode == 0) is (result == "success")


def test_native_matrix_dispatches_every_stage_after_cache_preparation() -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_bytes())["jobs"]
    assert jobs["prepare"]["needs"] == "portable"
    assert jobs["native"]["needs"] == "prepare"
    assert jobs["native"]["name"] == "native-${{ matrix.stage }} (${{ matrix.os }})"
    assert set(jobs["native"]["strategy"]["matrix"]["stage"]) == set(NATIVE_STAGES)
    platforms = {"ubuntu-latest", "macos-latest", "windows-latest"}
    for name in ("prepare", "native", "build"):
        assert set(jobs[name]["strategy"]["matrix"]["os"]) == platforms
        assert jobs[name]["strategy"]["fail-fast"] is False
    steps = jobs["native"]["steps"]
    dispatch = next(step for step in steps if step.get("name") == "Run native validation")
    assert dispatch["env"] == {"CI_STAGE": "${{ matrix.stage }}"}
    assert dispatch["run"] == 'just "ci-$CI_STAGE"'
    assert dispatch["shell"] == "bash"


def test_cache_owners_keep_tool_preparation_and_native_stage_outputs_separate() -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_bytes())["jobs"]
    prepare = next(step for step in jobs["prepare"]["steps"] if step.get("uses") == "$/.github/actions/setup-just")
    native = next(step for step in jobs["native"]["steps"] if step.get("uses") == "$/.github/actions/setup-just")
    assert prepare["with"]["consumer-cache"] == "false"
    assert native["with"]["cache-scope"] == "${{ matrix.stage }}"
    setup = yaml.safe_load((ROOT / ".github/actions/setup-just/action.yml").read_bytes())
    caches = {step["name"]: step for step in setup["runs"]["steps"] if step.get("uses", "").startswith("actions/cache@")}
    managed = caches["Cache managed Cargo compilation"]
    assert managed["if"] == "inputs.cache == 'true'"
    assert "rrt-build" in managed["with"]["path"]
    assert "restore-keys" not in managed["with"]
    for required in ("runner.os", "runner.arch", "pyproject.toml", "rust-toolchain.toml", "uv.lock", "Cargo.lock"):
        assert required in managed["with"]["key"]
    consumer = caches["Cache consumer Rust builds"]
    assert consumer["if"] == "inputs.cache == 'true' && inputs.consumer-cache == 'true'"
    assert set(consumer["with"]["path"].split()) == {"target/debug", "target/release", "target/perf"}
    for field in ("key", "restore-keys"):
        for required in ("inputs.cache-scope", "runner.os", "runner.arch", "rust-toolchain.toml", "Cargo.lock"):
            assert required in consumer["with"][field]
    assert "github.sha" in consumer["with"]["key"]
    assert "github.sha" not in consumer["with"]["restore-keys"]
    package = next(step for step in jobs["package"]["steps"] if step.get("uses") == "$/.github/actions/setup-just")
    assert package["with"]["cache"] == "false"


def reachable(recipes: dict[str, dict[str, Any]], name: str) -> set[str]:
    """Follow real Just dependencies without parsing recipe text."""
    return {name}.union(*(reachable(recipes, item["recipe"]) for item in recipes[name]["dependencies"]))


@pytest.fixture
def ci_graph(tmp_path: Path) -> tuple[Path, set[str]]:
    """Replace command bodies, retaining the real prerequisites and parallel node."""
    recipes = inspect_justfile(ROOT).recipes
    assert recipes["_ci-execution"]["attributes"] == ["parallel", "private"]
    selected = {name for name in reachable(recipes, "ci") if recipes[name]["body"]}
    script = tmp_path / "observe.py"
    script.write_text(
        "import os, pathlib, socket, sys\n"
        "name = sys.argv[1]\nroot = pathlib.Path.cwd()\n"
        "with (root / (name + '.started')).open('x', encoding='utf-8'): pass\n"
        "if name in {'test-unit', 'test-python'}:\n"
        "    for prerequisite in ('_ci-setup', 'check', 'python-fixture-lint', '_validation-doc-figures-check-if-canonical', 'doc-package-check'):\n"
        "        assert (root / (prerequisite + '.done')).exists(), prerequisite + ' has not finished'\n"
        "    if os.environ.get('CI_GRAPH_PORT'):\n"
        "        with socket.create_connection(('127.0.0.1', int(os.environ['CI_GRAPH_PORT'])), timeout=30) as peer:\n"
        "            assert peer.recv(1) == b'G', 'independent lane never started'\n"
        "if os.environ['CI_GRAPH_FAILURE'] == name:\n"
        "    print('injected failure: ' + name, file=sys.stderr)\n    sys.exit(19)\n"
        "(root / (name + '.done')).touch()\n",
        encoding="utf-8",
        newline="\n",
    )
    overrides = []
    for name in sorted(selected):
        recipe = recipes[name]
        dependencies = " ".join(
            "(" + item["recipe"] + " " + " ".join(json.dumps(arg) for arg in item["arguments"]) + ")" if item["arguments"] else item["recipe"]
            for item in recipe["dependencies"]
        )
        overrides.append(f"{name} *args: {dependencies}\n    {shlex.quote(Path(sys.executable).as_posix())} {shlex.quote(script.as_posix())} {name}\n")
    shutil.copytree(ROOT / "just", tmp_path / "just")
    source = (ROOT / "justfile").read_text(encoding="utf-8")
    (tmp_path / "justfile").write_text("set allow-duplicate-recipes\n" + source + "\n" + "\n".join(overrides), encoding="utf-8", newline="\n")
    return tmp_path, selected


def release_both_lanes(listener: socket.socket) -> None:
    """Neither child can continue until both independent lanes have connected."""
    first, _ = listener.accept()
    with first:
        second, _ = listener.accept()
        with second:
            first.sendall(b"G")
            second.sendall(b"G")


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
@pytest.mark.parametrize("failure", ["none", "_ci-setup", "python-check", "test-unit", "test-python"])
def test_local_graph_orders_prerequisites_runs_once_and_propagates_failure(ci_graph: tuple[Path, set[str]], parallel: bool, failure: str) -> None:
    root, selected = ci_graph
    environment = os.environ | {"CI_GRAPH_FAILURE": failure, "CI_GRAPH_PORT": ""}
    args = ["--jobs", "2" if parallel else "1", "ci"]
    if parallel and failure not in {"_ci-setup", "python-check"}:
        with socket.socket() as listener, ThreadPoolExecutor(max_workers=1) as pool:
            listener.bind(("127.0.0.1", 0))
            listener.listen(2)
            listener.settimeout(30)
            environment["CI_GRAPH_PORT"] = str(listener.getsockname()[1])
            barrier = pool.submit(release_both_lanes, listener)
            result = run_command("just", args, cwd=root, env=environment, check=False, timeout=60)
            barrier.result()
    else:
        result = run_command("just", args, cwd=root, env=environment, check=False, timeout=60)
    completed = {path.name.removesuffix(".done") for path in root.glob("*.done")}
    if failure == "none":
        assert result.returncode == 0, result.stderr
        assert completed == selected
    else:
        assert result.returncode != 0
        assert f"injected failure: {failure}" in result.stderr
        assert failure not in completed
        if failure in {"_ci-setup", "python-check"}:
            assert not (root / "test-unit.started").exists()
            assert not (root / "test-python.started").exists()


def test_parallel_lanes_borrow_the_prepared_environment() -> None:
    setup = dry_run(ROOT, "_ci-setup").stderr
    assert "uv sync --locked --managed-python --group dev --group notebooks" in setup
    commands = dry_run(ROOT, "_ci-execution").stderr
    assert "uv sync" not in commands
    for line in commands.splitlines():
        if "uv run --locked" in line:
            assert "uv run --locked --no-sync" in line
