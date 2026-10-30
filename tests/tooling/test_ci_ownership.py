"""Remote focused recipes cover the complete local CI contract without repeated lint."""

import os
from pathlib import Path

import pytest
import yaml
from research_repo_tools.just_inspect import dry_run, inspect_justfile
from research_repo_tools.process import run_command

ROOT = Path(__file__).resolve().parents[2]


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
    assert validators("ci") == portable | native | {"semgrep-scan"}
    assert {
        "python-check",
        "python-fixture-lint",
        "notebook-check",
        "test-python",
        "clippy",
        "doc-check",
        "test-unit",
        "test-unit-features",
        "bench-compile",
        "examples",
    } <= native
    assert {"semgrep-test", "fmt-check", "yaml-fmt-check", "markdown-check"} <= portable


def test_native_tooling_failures_surface_before_scientific_test_buckets() -> None:
    """Exercise the aggregate expansion so harness failures do not wait for Rust."""
    commands = dry_run(ROOT, "test").stderr
    assert commands.index("uv run --locked pytest") < commands.index("cargo nextest run")


@pytest.mark.parametrize("failure", [False, True])
def test_release_benchmark_dispatch_forwards_one_literal_tag(tmp_path: Path, *, failure: bool) -> None:
    """Exercise actual recipe expansion with a fake gh; dispatch failure survives."""
    tag = "v0.9.0 $(touch SHOULD_NOT_EXIST); 'literal'"
    arguments = tmp_path / "arguments"
    program = tmp_path / "gh"
    program.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$ARGUMENTS"\nexit "$DISPATCH_EXIT"\n', encoding="utf-8", newline="\n")
    program.chmod(0o755)
    script = dry_run(ROOT, "release-benchmarks", [tag]).stderr
    runner = "uv() { return 0; }\n" + script
    result = run_command(
        "bash",
        ["--noprofile", "--norc", "-euc", runner],
        cwd=tmp_path,
        env=os.environ | {"PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}", "ARGUMENTS": str(arguments), "DISPATCH_EXIT": "17" if failure else "0"},
        check=False,
    )
    assert result.returncode == (17 if failure else 0), result.stderr
    assert arguments.read_text(encoding="utf-8").splitlines() == [
        "workflow",
        "run",
        "release-benchmarks.yml",
        "--repo",
        "acgetchell/delaunay",
        "--ref",
        tag,
        "-f",
        f"tag={tag}",
    ]
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()


def test_dispatch_requires_the_tag_argument() -> None:
    result = run_command("just", ["--dry-run", "release-benchmarks"], cwd=ROOT, check=False)
    assert result.returncode != 0
    assert "argument" in result.stderr


@pytest.mark.parametrize("portable_result", ["success", "failure", "skipped", "cancelled"])
def test_required_builds_fail_when_portable_validation_does_not_succeed(portable_result: str) -> None:
    """A failed dependency must produce a failing required check instead of a skipped job."""
    job = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_bytes())["jobs"]["build"]
    assert job["needs"] == "portable"
    assert job["if"] == "${{ !cancelled() }}"
    guard = job["steps"][0]
    assert guard["name"] == "Require portable checks"
    result = run_command("bash", ["-euc", guard["run"]], env=os.environ | {"PORTABLE_RESULT": portable_result}, check=False)
    assert (result.returncode == 0) is (portable_result == "success")
