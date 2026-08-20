"""Tests for shared notebook boundary helpers."""

import subprocess
from pathlib import Path

import pytest

import notebook_utils


def test_find_repo_root_walks_from_nested_directory(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    nested = root / "notebooks" / "scratch"
    nested.mkdir(parents=True)
    (root / "Cargo.toml").touch()
    (root / "pyproject.toml").touch()

    assert notebook_utils.find_repo_root(nested) == root


@pytest.mark.parametrize("raw_value", ["0", "-1", "not-an-integer"])
def test_positive_int_env_rejects_invalid_values(monkeypatch: pytest.MonkeyPatch, raw_value: str) -> None:
    monkeypatch.setenv("NOTEBOOK_COUNT", raw_value)

    with pytest.raises(ValueError, match="NOTEBOOK_COUNT"):
        notebook_utils.positive_int_env("NOTEBOOK_COUNT", 3)


def test_tracked_figure_path_accepts_only_exact_repo_relative_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expected_relative = Path("docs/assets/readme/hero.png")
    monkeypatch.setenv("NOTEBOOK_FIGURE", expected_relative.as_posix())

    assert (
        notebook_utils.tracked_figure_path_from_env("NOTEBOOK_FIGURE", tmp_path, tmp_path / "target/hero.png", expected_relative)
        == (tmp_path / expected_relative).resolve()
    )

    monkeypatch.setenv("NOTEBOOK_FIGURE", "docs/assets/readme/other.png")
    with pytest.raises(ValueError, match="repo-relative path"):
        notebook_utils.tracked_figure_path_from_env("NOTEBOOK_FIGURE", tmp_path, tmp_path / "target/hero.png", expected_relative)


def test_run_command_rejects_nonzero_exit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    completed = subprocess.CompletedProcess(args=["tool"], returncode=7, stdout="out", stderr="err")

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return completed

    monkeypatch.setattr(notebook_utils, "run_safe_command", fake_run)

    with pytest.raises(RuntimeError, match="exit code 7"):
        notebook_utils.run_command(["tool"], cwd=tmp_path, timeout=1)
