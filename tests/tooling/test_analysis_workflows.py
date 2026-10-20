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
            for name, log in (
                ("Run comprehensive profiling benchmarks", "profiling-results/profiling_output.log"),
                ("Run profiling with memory allocation tracking (if not already done)", "profiling-results/memory_profiling_detailed.log"),
            )
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
        program.write_text(f"#!/usr/bin/env bash\n{body}exit {exit_code if tool == failed_tool else 0}\n", encoding="utf-8")
        program.chmod(0o755)
    (tmp_path / "profiling-results").mkdir()
    result = run_command(
        "bash",
        [*shell_args, "-c", step["run"]],
        cwd=tmp_path,
        env=os.environ | {"PATH": f"{executables}{os.pathsep}{os.environ['PATH']}", "BENCH_FILTER_VALUE": ""},
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


@pytest.mark.parametrize(
    ("event", "day", "expected"),
    [
        ("schedule", "01", "true"),
        ("schedule", "07", "true"),
        ("schedule", "08", "false"),
        ("schedule", "31", "false"),
        ("push", "31", "true"),
        ("workflow_dispatch", "31", "true"),
    ],
)
def test_monthly_profiling_gate(tmp_path: Path, event: str, day: str, expected: str) -> None:
    """Only first Sundays launch both expensive jobs; explicit triggers bypass the gate."""
    workflow = yaml.safe_load(PROFILING.read_text(encoding="utf-8"))
    # PyYAML's YAML 1.1 loader interprets the workflow's `on` key as True.
    assert workflow[True]["schedule"] == [{"cron": "0 2 * * 0"}]
    gate = workflow["jobs"]["schedule-gate"]
    step = gate["steps"][0]
    assert gate["outputs"]["run"] == "${{ steps." + step["id"] + ".outputs.run }}"
    for job_id in ("comprehensive-profiling", "memory-stress-test"):
        job = workflow["jobs"][job_id]
        assert job["needs"] == "schedule-gate"
        assert job["if"] == "needs.schedule-gate.outputs.run == 'true'"
    date = tmp_path / "date"
    date.write_text('#!/usr/bin/env bash\n[[ "$*" == "-u +%d" ]] || exit 2\nprintf "%s\\n" "$TEST_DAY"\n', encoding="utf-8")
    date.chmod(0o755)
    output = tmp_path / "output"
    run_command(
        "bash",
        ["--noprofile", "--norc", "-eo", "pipefail", "-c", step["run"]],
        cwd=tmp_path,
        env=os.environ | {"PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}", "TEST_DAY": day, "GITHUB_EVENT_NAME": event, "GITHUB_OUTPUT": str(output)},
        timeout=30,
    )
    assert output.read_text(encoding="utf-8") == f"run={expected}\n"


@pytest.mark.parametrize(
    ("event", "mode", "production"),
    [
        ("schedule", "development", True),
        ("workflow_dispatch", "development", False),
        ("workflow_dispatch", "production", True),
        ("push", "development", False),
        ("push", "production", False),
    ],
)
def test_profiling_mode(tmp_path: Path, event: str, mode: str, *, production: bool) -> None:
    """Schedules select full workloads despite the manual input's development default."""
    workflow = yaml.safe_load(PROFILING.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["comprehensive-profiling"]["steps"]
    step = next(step for step in steps if step.get("name") == "Set profiling mode")
    output = tmp_path / "environment"
    run_command(
        "bash",
        ["--noprofile", "--norc", "-eo", "pipefail", "-c", step["run"]],
        cwd=tmp_path,
        env=os.environ | {"INPUT_MODE": mode, "GITHUB_EVENT_NAME": event, "GITHUB_ENV": str(output)},
        timeout=30,
    )
    settings = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())
    assert settings == (
        {"BENCH_SAMPLE_SIZE": "100", "BENCH_WARMUP_SECS": "5", "BENCH_MEASUREMENT_TIME": "10", "BENCH_LARGE_SCALE": "1"}
        if production
        else {"PROFILING_DEV_MODE": "1", "BENCH_SAMPLE_SIZE": "30", "BENCH_WARMUP_SECS": "2", "BENCH_MEASUREMENT_TIME": "3"}
    )


def test_profiling_artifact_contains_raw_criterion_for_every_trigger() -> None:
    """Manual and monthly results retain the samples alongside logs and provenance."""
    workflow = yaml.safe_load(PROFILING.read_text(encoding="utf-8"))
    for job_id in ("comprehensive-profiling", "memory-stress-test"):
        uploads = [step for step in workflow["jobs"][job_id]["steps"] if step.get("uses", "").startswith("actions/upload-artifact@") and "if" not in step]
        assert len(uploads) == 1
        assert set(uploads[0]["with"]["path"].splitlines()) == {"profiling-results/", "target/criterion/"}
