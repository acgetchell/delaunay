"""Temporary scanner selection preserves the installed fixture checker's gates."""

import json
import os
import sys
import sysconfig
from importlib.metadata import version
from pathlib import Path

import pytest
from research_repo_tools import semgrep, semgrep_scan, semgrep_tool

import shared_tooling as adapter

ROOT = Path(__file__).resolve().parents[2]


def test_scanner_setup_uses_standard_environment_and_restores_override(capsys: pytest.CaptureFixture[str]) -> None:
    """The shared setup path cannot acquire a scanner in another environment."""
    previous_setup, previous_version, previous_path = semgrep_tool.setup, semgrep_tool.VERSION, os.environ.get("PATH")
    assert sys.version_info[:2] == (3, 15)
    assert adapter.main(["--root", str(ROOT), "semgrep", "setup"]) == 0
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
def test_real_fixture_checker_keeps_positive_negative_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, diagnostic: str, capsys: pytest.CaptureFixture[str]
) -> None:
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
    assert adapter.main(["--root", str(tmp_path), "semgrep", "check-fixtures"]) == (1 if diagnostic else 0)
    errors = capsys.readouterr().err
    if diagnostic:
        assert "delaunay.fixture" in errors
        assert diagnostic in errors
    else:
        assert not errors
    assert os.environ["PATH"] == previous_path


def test_override_requires_review_when_shared_package_advances(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not silently keep the temporary internal override after its fixed release."""
    monkeypatch.setattr(adapter, "version", lambda _name: "0.2.5")
    with pytest.raises(RuntimeError, match="delaunay#673"):
        adapter.main(["semgrep", "setup"])


@pytest.mark.parametrize("with_path", [False, True])
def test_override_restores_process_state_after_failure(monkeypatch: pytest.MonkeyPatch, *, with_path: bool) -> None:
    """Failure cannot leave the caller's scanner or PATH overridden."""
    if not with_path:
        monkeypatch.delenv("PATH", raising=False)
    previous_setup, previous_version, previous_path = semgrep_tool.setup, semgrep_tool.VERSION, os.environ.get("PATH")
    hooks = [(semgrep_scan, "_scan_groups"), (semgrep_scan, "scan"), (semgrep_scan, "publish_directory"), (semgrep, "_documentation_cases")]
    previous_hooks = [getattr(module, name) for module, name in hooks]

    def fail(argv: list[str] | None = None) -> int:
        semgrep_tool.setup(ROOT)
        message = "scanner probe failed"
        raise OSError(message)

    monkeypatch.setattr(adapter, "tooling_main", fail)
    with pytest.raises(OSError, match="scanner probe failed"):
        adapter.main(["semgrep", "setup"])
    assert semgrep_tool.setup is previous_setup
    assert previous_version == semgrep_tool.VERSION
    assert os.environ.get("PATH") == previous_path
    assert [getattr(module, name) for module, name in hooks] == previous_hooks
