"""Tests for shared notebook boundary helpers."""

import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest
from research_repo_tools import process

import notebook_utils

REPO_ROOT = Path(__file__).resolve().parents[2]


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


@pytest.mark.parametrize(("raw_value", "expected"), [(None, 0.28), ("0", 0.0), ("0.5", 0.5), ("1", 1.0)])
def test_unit_interval_float_env_accepts_defaults_and_inclusive_bounds(monkeypatch: pytest.MonkeyPatch, raw_value: str | None, expected: float) -> None:
    monkeypatch.delenv("NOTEBOOK_ALPHA", raising=False)
    if raw_value is not None:
        monkeypatch.setenv("NOTEBOOK_ALPHA", raw_value)

    assert notebook_utils.unit_interval_float_env("NOTEBOOK_ALPHA", 0.28) == expected


@pytest.mark.parametrize(
    ("raw_value", "default"),
    [("-0.01", 0.28), ("1.01", 0.28), ("nan", 0.28), ("inf", 0.28), ("-inf", 0.28), ("not-a-float", 0.28), (None, float("nan")), (None, 2.0)],
)
def test_unit_interval_float_env_rejects_invalid_overrides_and_defaults(monkeypatch: pytest.MonkeyPatch, raw_value: str | None, default: float) -> None:
    monkeypatch.delenv("NOTEBOOK_ALPHA", raising=False)
    if raw_value is not None:
        monkeypatch.setenv("NOTEBOOK_ALPHA", raw_value)

    with pytest.raises(ValueError, match=r"NOTEBOOK_ALPHA.*finite.*\[0, 1\]"):
        notebook_utils.unit_interval_float_env("NOTEBOOK_ALPHA", default)


@pytest.mark.parametrize("setting", ["EDGE_ALPHA", "POINT_ALPHA", "HULL_ALPHA"])
def test_quickstart_rejects_invalid_opacity_before_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setting: str) -> None:
    """Exercise the real configuration cell without launching the Rust CLI."""
    notebook_path = REPO_ROOT / "notebooks/00_quickstart.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cell = next(cell for cell in notebook["cells"] if cell["id"] == "generate-3d-code")
    for name in tuple(os.environ):
        if name.startswith("DELAUNAY_QUICKSTART_"):
            monkeypatch.delenv(name)
    name = f"DELAUNAY_QUICKSTART_{setting}"
    monkeypatch.setenv(name, "2")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Cargo.toml").touch()
    (tmp_path / "pyproject.toml").touch()
    runner = Mock()
    monkeypatch.setattr(process, "run_command_live", runner)
    monkeypatch.setattr(notebook_utils, "delaunay_command_prefix", lambda _root: ["test-delaunay"])

    with pytest.raises(ValueError, match=rf"{name}.*\[0, 1\]"):
        # Execute only repository-owned source with the process boundary stubbed.
        exec(compile("".join(cell["source"]), str(notebook_path), "exec"), {})  # noqa: S102

    runner.assert_not_called()
    assert not (tmp_path / "target").exists()


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
        "--locked",
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
