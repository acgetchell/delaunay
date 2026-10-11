"""Repository notebook policy, independent of shared lint mechanics."""

import json
import sys
from pathlib import Path

import pytest
from research_repo_tools.config import load
from research_repo_tools.process import run_safe_command

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_notebook_spelling_is_declared_to_shared_lint() -> None:
    assert load(root=REPO_ROOT).notebooks.id_pattern == r"[a-z0-9]+(?:-[a-z0-9]+)*"


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
