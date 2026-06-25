"""Tests for notebook_check.py notebook linting diagnostics."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from notebook_check import (
    LintOptions,
    NotebookDocument,
    code_cell_diagnostics,
    discover_notebooks,
    execute_notebooks,
    external_tool_diagnostics,
    extract_code,
    lint,
    load_notebook,
    main,
)


def write_notebook(path: Path, cells: list[dict[str, Any]]) -> None:
    """Write a minimal nbformat v4 notebook fixture."""
    notebook = {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook), encoding="utf-8")


def load_notebook_fixture(tmp_path: Path, name: str, cells: list[dict[str, Any]]) -> NotebookDocument:
    """Write and load a minimal notebook through the real parser."""
    path = tmp_path / name
    write_notebook(path, cells)
    return load_notebook(path)


def code_cell(source: Any, *, outputs: Any = None, execution_count: Any = None) -> dict[str, Any]:
    """Return a minimal code cell fixture."""
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": [] if outputs is None else outputs,
        "source": source,
    }


def test_discover_notebooks_excludes_checkpoints(tmp_path: Path) -> None:
    """Notebook discovery should stay under notebooks/ and skip checkpoint files."""
    notebooks = tmp_path / "notebooks"
    checkpoints = notebooks / ".ipynb_checkpoints"
    nested = notebooks / "nested"
    checkpoints.mkdir(parents=True)
    nested.mkdir()
    first = notebooks / "a.ipynb"
    second = nested / "b.ipynb"
    checkpoint = checkpoints / "a-checkpoint.ipynb"
    elsewhere = tmp_path / "outside.ipynb"
    for path in [first, second, checkpoint, elsewhere]:
        path.write_text("{}", encoding="utf-8")

    assert discover_notebooks(tmp_path) == [first, second]


def test_discover_notebooks_returns_empty_list_without_notebook_directory(tmp_path: Path) -> None:
    """Repositories without notebooks should still pass notebook validation."""
    assert discover_notebooks(tmp_path) == []


def test_load_notebook_rejects_missing_cells(tmp_path: Path) -> None:
    """Notebook JSON must contain a cells list."""
    notebook = tmp_path / "missing-cells.ipynb"
    notebook.write_text('{"nbformat": 4, "nbformat_minor": 5, "metadata": {}}', encoding="utf-8")

    with pytest.raises(TypeError, match="expected notebook cells to be a list"):
        load_notebook(notebook)


def test_load_notebook_rejects_non_object_json(tmp_path: Path) -> None:
    """Notebook root JSON must be an object."""
    notebook = tmp_path / "list.ipynb"
    notebook.write_text("[]", encoding="utf-8")

    with pytest.raises(TypeError, match="expected notebook JSON to be an object"):
        load_notebook(notebook)


def test_load_notebook_rejects_wrong_nbformat(tmp_path: Path) -> None:
    """Only nbformat v4 notebooks are supported."""
    notebook = tmp_path / "old-format.ipynb"
    notebook.write_text('{"cells": [], "nbformat": 3, "nbformat_minor": 0, "metadata": {}}', encoding="utf-8")

    with pytest.raises(ValueError, match="expected nbformat 4"):
        load_notebook(notebook)


def test_load_notebook_rejects_bad_cell_source(tmp_path: Path) -> None:
    """Cell source lists must contain only strings."""
    notebook = tmp_path / "bad-source.ipynb"
    write_notebook(notebook, [code_cell(["valid", 5])])

    with pytest.raises(TypeError, match="source list must contain only strings"):
        load_notebook(notebook)


def test_load_notebook_rejects_unknown_cell_type(tmp_path: Path) -> None:
    """Only standard nbformat cell types are accepted."""
    notebook = tmp_path / "bad-cell-type.ipynb"
    write_notebook(notebook, [{**code_cell("x = 1"), "cell_type": "python"}])

    with pytest.raises(ValueError, match="expected cell_type to be one of"):
        load_notebook(notebook)


def test_load_notebook_rejects_bad_execution_count(tmp_path: Path) -> None:
    """Code cell execution counts must be integers or null."""
    notebook = tmp_path / "bad-execution-count.ipynb"
    write_notebook(notebook, [code_cell("x = 1", execution_count="1")])

    with pytest.raises(TypeError, match="execution_count must be an integer or null"):
        load_notebook(notebook)


def test_load_notebook_rejects_bad_outputs_shape(tmp_path: Path) -> None:
    """Code cell outputs must be a list."""
    notebook = tmp_path / "bad-outputs.ipynb"
    write_notebook(notebook, [code_cell("x = 1", outputs={"output_type": "stream"})])

    with pytest.raises(TypeError, match="outputs must be a list"):
        load_notebook(notebook)


def test_code_cell_diagnostics_report_dirty_outputs_and_syntax(tmp_path: Path) -> None:
    """Dirty outputs and syntax errors should be reported before commit."""
    notebook_path = tmp_path / "dirty.ipynb"
    write_notebook(
        notebook_path,
        [
            code_cell("x = 1", outputs=[{"output_type": "stream", "name": "stdout", "text": "1"}], execution_count=7),
            code_cell("def broken(:\n    pass"),
        ],
    )
    notebook = load_notebook(notebook_path)

    diagnostics = code_cell_diagnostics(notebook_path, notebook, LintOptions(run_ruff=False, run_format=False, run_ty=False))

    messages = [diagnostic.message for diagnostic in diagnostics]
    assert "has 1 output block(s); clear outputs before committing" in messages
    assert "execution_count=7; clear execution counts" in messages
    assert any(message.startswith("syntax error:") for message in messages)


def test_lint_allow_outputs_accepts_rendered_notebook(tmp_path: Path) -> None:
    """Rendered notebooks can be allowed explicitly."""
    notebook = tmp_path / "rendered.ipynb"
    write_notebook(notebook, [code_cell("x = 1", outputs=[{"output_type": "stream", "name": "stdout", "text": "1"}], execution_count=1)])

    result = lint(notebook, LintOptions(allow_outputs=True, run_ruff=False, run_format=False, run_ty=False))

    assert result == 0


def test_lint_strict_fails_on_notebook_warnings(tmp_path: Path) -> None:
    """Strict mode should promote advisory notebook warnings to failures."""
    notebook = tmp_path / "warnings.ipynb"
    write_notebook(notebook, [code_cell("import pandas\n\ndef helper(value):\n    return value\n")])

    result = lint(notebook, LintOptions(strict=True, run_ruff=False, run_format=False, run_ty=False))

    assert result == 1


def test_lint_without_external_tools_passes_clean_notebook(tmp_path: Path) -> None:
    """A clean notebook should pass when external tool checks are disabled."""
    notebook = tmp_path / "clean.ipynb"
    write_notebook(notebook, [code_cell("def helper(value: int) -> int:\n    return value + 1\n")])

    result = lint(notebook, LintOptions(run_ruff=False, run_format=False, run_ty=False))

    assert result == 0


def test_external_tool_diagnostics_report_missing_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing Ruff or ty should be explicit notebook diagnostics."""
    monkeypatch.setattr("notebook_check.shutil.which", lambda _command: None)
    notebook = NotebookDocument(path=Path("missing-tools.ipynb"), nbformat=4, nbformat_minor=5, cells=())

    diagnostics = external_tool_diagnostics(Path("missing-tools.ipynb"), notebook, LintOptions())

    messages = [diagnostic.message for diagnostic in diagnostics]
    assert "ruff is required for notebook linting; run through `uv run` or install Ruff" in messages
    assert "ty is required for notebook linting; run through `uv run` or install ty" in messages


def test_main_lint_reports_errors_to_stderr(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI lint errors should be concise and actionable."""
    notebook = tmp_path / "dirty.ipynb"
    write_notebook(notebook, [code_cell("x = 1", outputs=[{"output_type": "stream", "name": "stdout", "text": "1"}], execution_count=3)])

    result = main(["lint", "--no-ruff", "--no-format", "--no-ty", str(notebook)])

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert f"{notebook}: cell 1: error: has 1 output block(s); clear outputs before committing" in captured.err
    assert f"{notebook}: cell 1: error: execution_count=3; clear execution counts" in captured.err


def test_main_reports_missing_notebook_without_traceback(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Missing notebook paths should not print a traceback."""
    missing_notebook = tmp_path / "missing.ipynb"

    result = main(["summary", str(missing_notebook)])

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert "error: notebook does not exist or is not a file:" in captured.err
    assert "Traceback" not in captured.err


def test_main_rejects_non_positive_timeout(capsys: pytest.CaptureFixture[str]) -> None:
    """Execution timeouts must be positive."""
    with pytest.raises(SystemExit) as exc_info:
        main(["execute", "--timeout", "0", "notebook.ipynb"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "expected a positive integer" in captured.err


def test_main_returns_success_when_no_notebooks_are_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Notebook validation should be a clean no-op before notebooks are added."""
    monkeypatch.chdir(tmp_path)

    assert main(["lint"]) == 0

    assert "No notebooks found." in capsys.readouterr().out


def test_extract_code_maps_lines_to_source_cells(tmp_path: Path) -> None:
    """Extracted Python should retain enough mapping for diagnostics."""
    notebook = load_notebook_fixture(
        tmp_path,
        "line-map.ipynb",
        [
            {"cell_type": "markdown", "metadata": {}, "source": "context"},
            code_cell("first = 1\nsecond = 2\n"),
            code_cell("third = 3"),
        ],
    )

    snapshot = extract_code(notebook)

    assert "# %% notebook cell 2" in snapshot.source
    assert "# %% notebook cell 3" in snapshot.source
    assert snapshot.line_to_cell[2] == 2
    assert snapshot.line_to_cell[5] == 3


def test_execute_notebooks_uses_headless_kernel_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Notebook execution should run headlessly from the repository root."""
    repo_root = tmp_path
    notebook = tmp_path / "notebooks" / "valid.ipynb"
    notebook.parent.mkdir()
    notebook.write_text("{}", encoding="utf-8")
    calls: dict[str, Any] = {}

    fake_nbformat = SimpleNamespace(read=lambda handle, as_version: {"handle": handle.name, "as_version": as_version})

    class FakeNotebookClient:
        """Capture nbclient construction without starting a real kernel."""

        def __init__(self, notebook_value: dict[str, Any], **kwargs: Any) -> None:
            calls["notebook"] = notebook_value
            calls["kwargs"] = kwargs

        def execute(self) -> None:
            """Record notebook execution."""
            calls["executed"] = True

    fake_nbclient = SimpleNamespace(NotebookClient=FakeNotebookClient)

    def fake_import(name: str) -> SimpleNamespace:
        if name == "nbformat":
            return fake_nbformat
        if name == "nbclient":
            return fake_nbclient
        msg = f"unexpected import {name}"
        raise AssertionError(msg)

    monkeypatch.delenv("MPLBACKEND", raising=False)
    monkeypatch.setattr("notebook_check.import_module", fake_import)

    execute_notebooks([notebook], repo_root, timeout=123)

    assert calls["notebook"]["as_version"] == 4
    assert calls["kwargs"]["timeout"] == 123
    assert calls["kwargs"]["kernel_name"] == "python3"
    assert calls["kwargs"]["resources"] == {"metadata": {"path": str(repo_root)}}
    assert calls["executed"] is True
    assert os.environ["MPLBACKEND"] == "Agg"
    assert f"OK executed {notebook}" in capsys.readouterr().out
