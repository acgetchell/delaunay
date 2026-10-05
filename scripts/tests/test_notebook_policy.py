"""Repository notebook policy, independent of shared lint mechanics."""

import json
import re
import sys
from pathlib import Path

import pytest
from research_repo_tools.process import run_safe_command
from research_repo_tools.selection import select_files

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = select_files(REPO_ROOT, include=("notebooks/*.ipynb",), exclude=("**/.ipynb_checkpoints/**",))


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_cell_ids_use_lowercase_kebab_case(name: str) -> None:
    """Check every current source cell without executing or modifying the notebook."""
    notebook = json.loads((REPO_ROOT / name).read_bytes())
    for index, cell in enumerate(notebook["cells"], start=1):
        assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", cell["id"]), f"{name}: cell {index}: {cell['id']!r} must use lowercase kebab-case"


@pytest.mark.parametrize(
    ("source", "prohibited"),
    [
        ('import subprocess\n\nsubprocess.Popen(["/absolute/program"])', True),
        ('import subprocess as sp\n\nsp.Popen(["/absolute/program"])', True),
        ('from subprocess import Popen\n\nPopen(["/absolute/program"])', True),
        ('import subprocess\n\nsubprocess.run(["/absolute/program"], check=True, timeout=10)', False),
    ],
)
def test_native_notebook_lint_rejects_popen(tmp_path: Path, source: str, *, prohibited: bool) -> None:
    """Keep the former Popen prohibition in consumer Ruff policy, including aliases."""
    notebook = tmp_path / "process-policy.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "cells": [{"cell_type": "code", "execution_count": None, "id": "process-policy", "metadata": {}, "outputs": [], "source": source}],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            },
        ),
        encoding="utf-8",
    )
    result = run_safe_command(
        sys.executable,
        ["-m", "ruff", "check", "--config", str(REPO_ROOT / "pyproject.toml"), "--output-format", "json", str(notebook)],
        check=False,
    )

    diagnostics = json.loads(result.stdout)
    assert result.returncode == int(prohibited), result.stderr
    assert result.stderr == ""
    assert [item["code"] for item in diagnostics] == (["TID251"] if prohibited else [])
    assert all(item["cell"] == 1 for item in diagnostics)
