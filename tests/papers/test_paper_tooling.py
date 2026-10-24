"""Paper policy and explicit command boundaries using the root environment."""

import json
import os
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml
from research_repo_tools.cli import main
from research_repo_tools.config import load
from research_repo_tools.paper_dates import read_source_date
from research_repo_tools.process import run_command
from research_repo_tools.toolchain_config import load as load_toolchain

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "papers/tooling.toml"


def test_optional_toolchain_reuses_root_python_and_rust() -> None:
    settings = load(CONFIG, root=ROOT)
    plan = load_toolchain(settings)
    base = load_toolchain(load(root=ROOT))
    assert plan.root == ROOT
    assert plan.python_request == base.python_request
    assert plan.rust == base.rust
    assert {tool.package for tool in plan.cargo} == {"tectonic", "tex-fmt"}


def test_retained_validation_paper_satisfies_consumer_policy() -> None:
    assert read_source_date(ROOT / "papers/validation.tex").raw
    assert (
        main(
            [
                "--root",
                str(ROOT),
                "--config",
                str(CONFIG),
                "papers",
                "check",
                "papers/validation.pdf",
                "--require-text",
                "Validation Architecture in delaunay",
                "--require-text",
                "REFERENCES",
                "--forbid-text",
                r"\today",
                "--forbid-text",
                "Manuscript submitted to ACM",
            ]
        )
        == 0
    )


def test_paper_workflow_is_manual_and_checks_shared_figures() -> None:
    path = ROOT / ".github/workflows/papers.yml"
    workflow: Any = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)  # noqa: S506 - BaseLoader constructs data only.
    assert set(workflow["on"]) == {"workflow_dispatch"}
    commands = "\n".join(step.get("run", "") for step in workflow["jobs"]["papers"]["steps"])
    assert "just validation-doc-figures-check" in commands
    assert "just paper-artifact-check" in commands
    assert "just papers\n" not in commands


def test_paper_test_command_uses_root_environment_from_subdirectory(tmp_path: Path) -> None:
    """Execute the imported command without synchronizing or creating an environment."""
    for name in ("just", "papers"):
        shutil.copytree(ROOT / name, tmp_path / name)
    (tmp_path / "justfile").write_bytes((ROOT / "justfile").read_bytes())
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    script = binary_dir / "record.py"
    script.write_text(
        "import json, os, sys\nfrom pathlib import Path\nPath('calls.json').write_text(json.dumps([os.getcwd(), sys.argv[1:]]))\n", encoding="utf-8"
    )
    stub = binary_dir / "uv"
    stub.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(script))} "$@"\n', encoding="utf-8")
    stub.chmod(0o755)
    if os.name == "nt":
        (binary_dir / "uv.cmd").write_text(f'@echo off\n"{sys.executable}" "{script}" %*\n', encoding="utf-8")
    result = run_command(
        "just",
        ["paper-test"],
        cwd=tmp_path / "papers",
        env={**os.environ, "PATH": str(binary_dir) + os.pathsep + os.environ["PATH"]},
    )
    assert result.returncode == 0
    directory, arguments = json.loads((tmp_path / "calls.json").read_text(encoding="utf-8"))
    assert Path(directory) == tmp_path
    assert arguments == ["run", "--locked", "--group", "dev", "--group", "papers", "pytest", "tests/papers"]
    assert not (tmp_path / "papers/.venv").exists()
