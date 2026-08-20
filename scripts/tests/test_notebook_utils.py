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


def test_delaunay_command_prefix_uses_configured_relative_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = tmp_path / "tools" / "delaunay"
    binary.parent.mkdir()
    binary.touch()
    monkeypatch.setenv("DELAUNAY_BINARY", "tools/delaunay")

    assert notebook_utils.delaunay_command_prefix(tmp_path) == [str(binary)]


def test_delaunay_command_prefix_rejects_empty_configured_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DELAUNAY_BINARY", "")

    with pytest.raises(ValueError, match="must not be empty"):
        notebook_utils.delaunay_command_prefix(tmp_path)


def test_delaunay_command_prefix_rejects_missing_configured_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DELAUNAY_BINARY", "tools/missing")

    with pytest.raises(FileNotFoundError, match="does not point to a file"):
        notebook_utils.delaunay_command_prefix(tmp_path)


def test_delaunay_command_prefix_uses_required_built_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DELAUNAY_BINARY", raising=False)
    binary_name = "delaunay.exe" if notebook_utils.os.name == "nt" else "delaunay"
    binary = tmp_path / "target" / "perf" / binary_name
    binary.parent.mkdir(parents=True)
    binary.touch()

    assert notebook_utils.delaunay_command_prefix(tmp_path, require_built_binary=True) == [str(binary)]


def test_delaunay_command_prefix_rejects_missing_required_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DELAUNAY_BINARY", raising=False)

    with pytest.raises(FileNotFoundError, match="build the notebook CLI first"):
        notebook_utils.delaunay_command_prefix(tmp_path, require_built_binary=True)


def test_delaunay_command_prefix_falls_back_to_cargo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DELAUNAY_BINARY", raising=False)
    monkeypatch.setattr(notebook_utils.shutil, "which", lambda executable: "/tools/cargo" if executable == "cargo" else None)

    assert notebook_utils.delaunay_command_prefix(tmp_path) == [
        "/tools/cargo",
        "run",
        "--profile",
        "perf",
        "--features",
        "cli",
        "--bin",
        "delaunay",
        "--",
    ]


def test_delaunay_command_prefix_rejects_missing_cargo_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DELAUNAY_BINARY", raising=False)
    monkeypatch.setattr(notebook_utils.shutil, "which", lambda _executable: None)

    with pytest.raises(RuntimeError, match="cargo executable was not found"):
        notebook_utils.delaunay_command_prefix(tmp_path)


def test_run_command_rejects_nonzero_exit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    completed = subprocess.CompletedProcess(args=["tool"], returncode=7, stdout="out", stderr="err")

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return completed

    monkeypatch.setattr(notebook_utils, "run_safe_command", fake_run)

    with pytest.raises(RuntimeError, match="exit code 7"):
        notebook_utils.run_command(["tool"], cwd=tmp_path, timeout=1)
