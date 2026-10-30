"""Exercise workflow shell behavior without running benchmarks or uploading artifacts."""

import os
from pathlib import Path

import pytest
import yaml
from research_repo_tools.process import run_command

ROOT = Path(__file__).resolve().parents[2]
PROFILING = ROOT / ".github/workflows/profiling-benchmarks.yml"


@pytest.mark.parametrize(
    ("pipeline", "failed_tool"),
    [
        *(
            (("profiling-benchmarks.yml", "comprehensive-profiling", name, log), tool)
            for name, log in (("Measure selected diagnostic suite once", "profiling-results/profiling_output.log"),)
            for tool in ("", "cargo")
        ),
        *(
            (("rust-clippy.yml", "clippy-sarif", "Run clippy with SARIF output", "rust-clippy-results.sarif"), tool)
            for tool in ("", "cargo", "clippy-sarif", "sarif-fmt")
        ),
    ],
)
def test_analysis_pipeline_status(
    tmp_path: Path,
    pipeline: tuple[str, str, str, str],
    failed_tool: str,
) -> None:
    """A successful tee or diagnostics upload must not hide an upstream failure."""
    workflow_file, job_id, step_name, log_name = pipeline
    exit_code = 23 if failed_tool else 0
    workflow = yaml.safe_load((ROOT / ".github/workflows" / workflow_file).read_text(encoding="utf-8"))
    job = workflow["jobs"][job_id]
    step = next(step for step in job["steps"] if step.get("name") == step_name)
    shell = step.get("shell", job.get("defaults", workflow.get("defaults", {})).get("run", {}).get("shell"))
    assert shell in (None, "bash")
    # Match the documented runner distinction: implicit bash lacks pipefail.
    shell_args = ["--noprofile", "--norc", "-eo", "pipefail"] if shell == "bash" else ["-e"]
    executables = tmp_path / "bin"
    executables.mkdir()
    for tool in ("cargo", "clippy-sarif", "sarif-fmt"):
        program = executables / tool
        body = "printf 'analysis output\\n'\n" if tool == "cargo" else "cat\n"
        program.write_text(f"#!/usr/bin/env bash\n{body}exit {exit_code if tool == failed_tool else 0}\n", encoding="utf-8", newline="\n")
        program.chmod(0o755)
    (tmp_path / "profiling-results").mkdir()
    result = run_command(
        "bash",
        [*shell_args, "-c", step["run"]],
        cwd=tmp_path,
        env=os.environ | {"PATH": f"{executables}{os.pathsep}{os.environ['PATH']}", "BENCH_FILTER_VALUE": "", "INPUT_SUITE": "timing"},
        check=False,
        timeout=30,
    )
    assert result.returncode == exit_code, result.stderr
    assert (tmp_path / log_name).read_text(encoding="utf-8") == "analysis output\n"
    assert not step.get("continue-on-error", False)
    assert not job.get("continue-on-error", False)
    if job_id == "clippy-sarif":
        upload = next(step for step in job["steps"] if step.get("name") == "Upload SARIF results")
        assert "always()" in upload["if"]
        assert upload["with"]["sarif_file"] == log_name


def test_diagnostic_profiling_is_manual_with_one_measurement_owner() -> None:
    """Diagnostic timings must never start on a PR, push, schedule, or release tag."""
    workflow = yaml.safe_load(PROFILING.read_text(encoding="utf-8"))
    assert set(workflow[True]) == {"workflow_dispatch"}
    assert set(workflow["jobs"]) == {"comprehensive-profiling"}
    steps = workflow["jobs"]["comprehensive-profiling"]["steps"]
    measurements = [step for step in steps if "cargo_args=(bench" in step.get("run", "")]
    assert len(measurements) == 1
    uploads = [step for step in steps if step.get("uses", "").startswith("actions/upload-artifact@")]
    assert len(uploads) == 1
    assert set(uploads[0]["with"]["path"].splitlines()) == {"profiling-results/", "target/criterion/"}


def test_public_api_comparisons_are_not_repeated_on_main_push() -> None:
    """PR comparisons and exact-tag release measurements have distinct owners."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/benchmarks.yml").read_text(encoding="utf-8"))
    assert set(workflow[True]) == {"pull_request", "workflow_dispatch"}
    steps = workflow["jobs"]["performance-regression"]["steps"]
    assert sum(step.get("run") == "just performance-local" for step in steps) == 1


@pytest.mark.parametrize("mode", ["development", "production"])
@pytest.mark.parametrize(
    ("suite", "requested_filter", "selected_filter"),
    [
        ("timing", "", ""),
        ("timing", "construction/3D/1000v/construct", "construction/3D/1000v/construct"),
        ("allocations", "", "memory_profiling"),
        ("allocations", "memory_profiling/memory_usage_2d/1000", "memory_profiling/memory_usage_2d/1000"),
        ("traversal", "queries/neighbors/2D", "queries/neighbors/2D"),
        ("validation_components", "validation_components_2d", "validation_components_2d"),
        ("circumsphere_queries", "", ""),
        ("hull_queries", "boundary_facets", "boundary_facets"),
        ("memory_footprint", "memory/2D", "memory/2D"),
    ],
)
def test_profiling_executes_selected_suite_once(
    tmp_path: Path,
    mode: str,
    suite: str,
    requested_filter: str,
    selected_filter: str,
) -> None:
    """Execute real workflow shell with a fake Cargo, preserving argv and sample ownership."""
    workflow = yaml.safe_load(PROFILING.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["comprehensive-profiling"]["steps"]
    select = next(step for step in steps if step.get("name") == "Select diagnostic workload")
    measure = next(step for step in steps if step.get("name") == "Measure selected diagnostic suite once")
    output = tmp_path / "environment"
    environment = os.environ | {"INPUT_MODE": mode, "INPUT_SUITE": suite, "INPUT_FILTER": requested_filter, "GITHUB_ENV": str(output)}
    run_command("bash", ["-euo", "pipefail", "-c", select["run"]], cwd=tmp_path, env=environment, timeout=30)
    settings = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())
    assert settings["BENCH_FILTER_VALUE"] == selected_filter
    assert ("BENCH_LARGE_SCALE" in settings) is (mode == "production")
    assert ("PROFILING_DEV_MODE" in settings) is (mode == "development")
    arguments = tmp_path / "arguments"
    cargo = tmp_path / "cargo"
    cargo.write_text('#!/usr/bin/env bash\nprintf "%s\\n" CALL "$@" >> "$ARGUMENTS"\n', encoding="utf-8", newline="\n")
    cargo.chmod(0o755)
    environment.update(settings)
    environment.update({"PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}", "ARGUMENTS": str(arguments)})
    run_command("bash", ["-euo", "pipefail", "-c", measure["run"]], cwd=tmp_path, env=environment, timeout=30)
    argv = arguments.read_text(encoding="utf-8").splitlines()
    assert argv.count("CALL") == 1
    assert argv[:6] == ["CALL", "bench", "--locked", "--profile", "perf", "--bench"]
    assert argv[6] == ("memory_footprint" if suite == "allocations" else "construction_scaling" if suite == "timing" else suite)
    assert ("--features" in argv) is (suite == "allocations")
    assert ("count-allocations" in argv) is (suite == "allocations")
    assert "--skip" not in argv
    if selected_filter == "memory_profiling":
        assert argv[-1] == "^memory_profiling/"
    elif selected_filter:
        assert argv[-1] == selected_filter
        assert ("--exact" in argv) is (suite == "allocations")


@pytest.mark.parametrize(
    ("suite", "requested_filter"),
    [
        ("allocations", "construction"),
        ("timing", "memory_profiling"),
        ("timing", "--test"),
        ("timing", "construction\nINJECTED=value"),
        ("allocations", "memory_profiling/1000\rINJECTED=value"),
    ],
)
def test_profiling_rejects_incompatible_or_multiline_filters(tmp_path: Path, suite: str, requested_filter: str) -> None:
    """Reject ambiguous selections and env-file injection before publishing settings."""
    workflow = yaml.safe_load(PROFILING.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["comprehensive-profiling"]["steps"]
    select = next(step for step in steps if step.get("name") == "Select diagnostic workload")
    output = tmp_path / "environment"
    result = run_command(
        "bash",
        ["-euo", "pipefail", "-c", select["run"]],
        cwd=tmp_path,
        check=False,
        timeout=30,
        env=os.environ | {"INPUT_MODE": "development", "INPUT_SUITE": suite, "INPUT_FILTER": requested_filter, "GITHUB_ENV": str(output)},
    )
    assert result.returncode == 2
    assert not output.exists()
