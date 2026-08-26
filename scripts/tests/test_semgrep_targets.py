#!/usr/bin/env python3
"""Tests for shared local and hosted Semgrep target enumeration."""

import subprocess
from typing import TYPE_CHECKING

import pytest

import semgrep_targets

if TYPE_CHECKING:
    from pathlib import Path


def _git_result(stdout: str) -> subprocess.CompletedProcess[str]:
    """Return a typed successful git result for target-listing tests."""
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def test_targets_include_tracked_python_and_rust_tests_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The shared target set covers both ignored test trees without duplicates."""
    monkeypatch.setattr(
        semgrep_targets,
        "run_git_command",
        lambda _args, **_kwargs: _git_result("scripts/tests/test_one.py\0tests/cli.rs\0tests/cli.rs\0"),
    )

    assert semgrep_targets.tracked_semgrep_targets(tmp_path) == (".", "scripts/tests/test_one.py", "tests/cli.rs")


def test_targets_fail_closed_if_deliberate_fixtures_escape_exclusion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A pathspec regression cannot feed annotated violations to a real scan."""
    monkeypatch.setattr(
        semgrep_targets,
        "run_git_command",
        lambda _args, **_kwargs: _git_result("tests/semgrep/src/project_rules/rust_style.rs\0"),
    )

    with pytest.raises(RuntimeError, match="included deliberate fixtures"):
        semgrep_targets.tracked_semgrep_targets(tmp_path)


def test_null_output_round_trips_paths_without_line_splitting(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """The shell interface uses NUL delimiters for arbitrary tracked filenames."""
    monkeypatch.setattr(semgrep_targets, "tracked_semgrep_targets", lambda _repo_root: (".", "tests/name with spaces.rs"))

    assert semgrep_targets.main(["--null"]) == 0
    assert capsys.readouterr().out == ".\0tests/name with spaces.rs\0"
