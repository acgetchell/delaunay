"""Published shared scanner selection preserves the consumer's fixture gates."""

import json
import os
import sys
import sysconfig
from importlib.metadata import version
from pathlib import Path

import pytest
from research_repo_tools import semgrep_tool
from research_repo_tools.cli import main
from research_repo_tools.process import run_command

ROOT = Path(__file__).resolve().parents[2]


def test_scanner_setup_uses_standard_environment(capsys: pytest.CaptureFixture[str]) -> None:
    """The shared setup path cannot acquire a scanner in another environment."""
    previous_setup, previous_version, previous_path = semgrep_tool.setup, semgrep_tool.VERSION, os.environ.get("PATH")
    assert sys.version_info[:2] == (3, 15)
    assert main(["--root", str(ROOT), "semgrep", "setup"]) == 0
    expected = Path(sysconfig.get_path("scripts")) / ("semgrep.exe" if os.name == "nt" else "semgrep")
    output = capsys.readouterr().out
    assert f"Semgrep {version('semgrep')}: {expected}" in output
    assert semgrep_tool.setup is previous_setup
    assert previous_version == semgrep_tool.VERSION
    assert os.environ.get("PATH") == previous_path


@pytest.mark.parametrize(
    ("source", "diagnostic"),
    [("print('blocked')", ""), ("print('allowed')", "expected finding not reported"), ("print('blocked')\nprint('blocked')", "unexpected finding")],
)
def test_real_fixture_checker_keeps_positive_negative_gates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, diagnostic: str) -> None:
    """Actual native scans accept a match and reject missing or extra findings."""
    (tmp_path / "pyproject.toml").write_text(
        '[tool.research-repo-tools.semgrep]\nconfig = "rules.yaml"\nfixtures = "fixture.py"\nnamespace = "delaunay."\ncwd = "."\n',
        encoding="utf-8",
    )
    (tmp_path / "rules.yaml").write_text(
        json.dumps(
            {"rules": [{"id": "delaunay.fixture", "languages": ["python"], "message": "fixture violation", "severity": "ERROR", "pattern": "print('blocked')"}]}
        ),
        encoding="utf-8",
    )
    (tmp_path / "fixture.py").write_text(f"# ruleid: delaunay.fixture\n{source}\n# ok: delaunay.fixture\nprint('allowed')\n", encoding="utf-8")
    # A broken ambient scanner must not be selected by either native launcher.
    ambient = tmp_path / "ambient"
    ambient.mkdir()
    for name in ("semgrep", "pysemgrep"):
        binary = ambient / name
        binary.write_text("#!/bin/sh\nexit 97\n", encoding="utf-8", newline="\n")
        binary.chmod(0o755)
        if os.name == "nt":
            (ambient / f"{name}.cmd").write_text("@echo off\nexit /b 97\n", encoding="utf-8")
    monkeypatch.setenv("PATH", os.pathsep.join([str(ambient), os.environ["PATH"]]))
    previous_path = os.environ["PATH"]
    # Exercise the real consumer entry point: uv activates the locked scripts
    # directory before Semgrep's launcher resolves its pysemgrep helper.
    result = run_command(
        "uv",
        ["run", "--locked", "--group", "dev", "research-repo-tools", "--root", str(tmp_path), "semgrep", "check-fixtures"],
        cwd=ROOT,
        env=dict(os.environ),
        timeout=120,
        check=False,
    )
    assert result.returncode == (1 if diagnostic else 0)
    errors = result.stderr
    if diagnostic:
        assert "delaunay.fixture" in errors
        assert diagnostic in errors
    else:
        assert not errors
    assert os.environ["PATH"] == previous_path
