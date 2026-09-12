"""Keep the declared Rust targets aligned with native CI execution."""

import os
from pathlib import Path

import pytest
import yaml

from subprocess_utils import run_safe_command

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"
NATIVE_TARGETS = {
    "ubuntu-latest": "x86_64-unknown-linux-gnu",
    "macos-latest": "aarch64-apple-darwin",
    "windows-latest": "x86_64-pc-windows-msvc",
}


@pytest.fixture(scope="module")
def host_check_script() -> str:
    """Check matrix wiring and return the actual production host guard."""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    job = workflow["jobs"]["build"]
    matrix = job["strategy"]["matrix"]
    assert set(matrix["os"]) == set(NATIVE_TARGETS)
    assert {entry["os"]: entry["target"] for entry in matrix["include"]} == NATIVE_TARGETS
    steps = job["steps"]
    guard_index = next(index for index, step in enumerate(steps) if step.get("id") == "rust-host")
    ci_index = next(index for index, step in enumerate(steps) if step.get("run") == "just ci")
    assert guard_index < ci_index
    guard = steps[guard_index]
    assert guard["shell"] == "bash"
    assert guard["env"]["EXPECTED_RUST_HOST"] == "${{ matrix.target }}"
    script = guard["run"]
    assert isinstance(script, str)
    return script


@pytest.mark.parametrize("host", NATIVE_TARGETS.values())
def test_native_host_guard_accepts_the_expected_target(tmp_path: Path, host_check_script: str, host: str) -> None:
    result = run_host_guard(tmp_path, host_check_script, host, f"host: {host}\n")
    assert result == 0


@pytest.mark.parametrize("compiler", ["host: x86_64-apple-darwin\n", "rustc without a host field\n"])
def test_native_host_guard_rejects_missing_or_mismatched_target(tmp_path: Path, host_check_script: str, compiler: str) -> None:
    result = run_host_guard(tmp_path, host_check_script, NATIVE_TARGETS["macos-latest"], compiler)
    assert result != 0


def run_host_guard(tmp_path: Path, script: str, expected: str, compiler: str) -> int:
    """Exercise shell behavior without installing compilers or running remote CI."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    rustc = fake_bin / "rustc"
    rustc.write_text("#!/usr/bin/env bash\nprintf '%s' \"$COMPILER_INFO\"\n", encoding="utf-8")
    rustc.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "EXPECTED_RUST_HOST": expected,
        "COMPILER_INFO": compiler,
    }
    result = run_safe_command("bash", ["--noprofile", "--norc", "-eo", "pipefail", "-c", script], cwd=tmp_path, env=env, check=False, timeout=10)
    if result.returncode:
        assert "differs from matrix target" in result.stdout
    return result.returncode
