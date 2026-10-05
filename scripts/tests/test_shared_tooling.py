"""Consumer integration checks for the pinned public maintenance CLI and recipes."""

import json
import os
import re
import shlex
import shutil
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from research_repo_tools.cli import main

from subprocess_utils import run_safe_command

if TYPE_CHECKING:
    import subprocess

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
UPDATE_PREFIX = ["run", "--locked", "--only-group", "tooling", "--inexact", "research-repo-tools"]
FIXTURE_MANIFEST = "tests/fixtures/checkpoint_no_float_roundtrip/Cargo.toml"


def write_stub(directory: Path, name: str, source: str) -> None:
    """Make a portable executable backed by this test environment's interpreter."""
    program = directory / f"{name}_stub.py"
    program.write_text(source, encoding="utf-8")
    executable = directory / name
    content = f'#!/bin/sh\nexec {shlex.quote(Path(sys.executable).as_posix())} {shlex.quote(program.as_posix())} "$@"\n'
    executable.write_text(content, encoding="utf-8")
    executable.chmod(0o755)
    if os.name == "nt":
        (directory / f"{name}.cmd").write_text(f'@echo off\n"{sys.executable}" "{program}" %*\n', encoding="utf-8")


def test_published_pin_configuration_and_inventory() -> None:
    groups = MANIFEST["dependency-groups"]
    assert groups["tooling"] == ["research-repo-tools==0.1.7"]
    assert {"include-group": "tooling"} in groups["dev"]
    assert version("research-repo-tools") == "0.1.7"
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    package = next(item for item in lock["package"] if item["name"] == "research-repo-tools")
    assert package["version"] == "0.1.7"
    assert package["source"] == {"registry": "https://pypi.org/simple"}
    assert MANIFEST["tool"]["uv"]["required-version"].startswith("==")
    tools = MANIFEST["tool"]["research-repo-tools"]["toolchain"]["cargo"]
    assert all(re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", pin) for pin in tools.values())
    assert {"cargo-edit", "cargo-audit", "cargo-nextest", "samply", "tectonic", "tex-fmt", "clippy-sarif", "sarif-fmt"} <= tools.keys()
    assert "just" not in tools
    assert "cargo-update" not in tools


@pytest.fixture
def recording_recipes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Execute the merged Justfile with local recording processes only."""
    (tmp_path / "justfile").write_bytes((ROOT / "justfile").read_bytes())
    shutil.copytree(ROOT / "just", tmp_path / "just")
    (tmp_path / "scripts").mkdir()
    # Native dependency provisioning is checked separately by managed setup.
    (tmp_path / "scripts" / "tectonic_native_dependencies.sh").write_text("# Fixture native dependencies are available.\n", encoding="utf-8")
    for name in ("pyproject.toml", "uv.lock", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".python-version"):
        (tmp_path / name).write_bytes((ROOT / name).read_bytes())
    fixture = tmp_path / FIXTURE_MANIFEST
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes((ROOT / FIXTURE_MANIFEST).read_bytes())
    fixture.with_name("Cargo.lock").write_bytes((ROOT / FIXTURE_MANIFEST).with_name("Cargo.lock").read_bytes())
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_stub(
        bin_dir,
        "uv",
        "import json, os, sys\nfrom pathlib import Path\n"
        "log = Path('calls.jsonl')\n"
        "with log.open('a', encoding='utf-8') as stream: stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "sys.exit(23 if len(log.read_text(encoding='utf-8').splitlines()) == int(os.environ.get('FAIL_STEP', '0')) else 0)\n",
    )
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    monkeypatch.delenv("FAIL_STEP", raising=False)
    return tmp_path


def invoke_recipe(root: Path, recipe: str, *args: str) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    """Run real Just composition and return the recorded public CLI boundaries."""
    result = run_safe_command("just", ["--justfile", str(root / "justfile"), recipe, *args], cwd=root, check=False)
    log = root / "calls.jsonl"
    calls: list[list[str]] = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    return result, calls


def update_commands(recipe: str) -> list[list[str]]:
    """Describe the consumer's update contract, including its isolated Cargo root."""
    tools = [
        ["run", "--no-config", "--no-sync", "--no-python-downloads", "research-repo-tools", "deps", "update-uv"],
        [*UPDATE_PREFIX, "toolchain", "upgrade"],
        ["run", "--locked", "--managed-python", "--only-group", "tooling", "research-repo-tools", "setup"],
    ]
    cargo = [
        [*UPDATE_PREFIX, "toolchain", "run", "--", "cargo", *args]
        for args in (
            ["upgrade", "--incompatible", "allow"],
            ["upgrade", "--manifest-path", FIXTURE_MANIFEST, "--incompatible", "allow"],
            ["update"],
            ["update", "--manifest-path", FIXTURE_MANIFEST],
        )
    ]
    python = [
        [*UPDATE_PREFIX, "deps", "update-python"],
        ["lock", "--upgrade"],
        [
            "run",
            "--locked",
            "--no-sync",
            "--no-python-downloads",
            "research-repo-tools",
            "toolchain",
            "run",
            "--",
            "uv",
            "sync",
            "--locked",
            "--managed-python",
            "--group",
            "dev",
        ],
    ]
    return {
        "update": tools + cargo + python,
        "update-tools": tools,
        "update-cargo-tools": tools[1:2],
        "update-dependencies": cargo + python,
        "update-cargo-dependencies": cargo,
        "update-python-dependencies": python,
        "update-python-deps": python,
    }[recipe]


@pytest.mark.parametrize(
    "recipe",
    ["update", "update-tools", "update-cargo-tools", "update-dependencies", "update-cargo-dependencies", "update-python-dependencies", "update-python-deps"],
)
def test_update_composition_and_boundaries(recording_recipes: Path, recipe: str) -> None:
    files = [recording_recipes / name for name in ("pyproject.toml", "uv.lock", "Cargo.toml", "Cargo.lock", FIXTURE_MANIFEST)]
    before = {path: path.read_bytes() for path in files}
    result, calls = invoke_recipe(recording_recipes, recipe)
    assert result.returncode == 0, result.stderr
    assert calls == update_commands(recipe)
    assert {path: path.read_bytes() for path in files} == before


@pytest.mark.parametrize("step", range(1, 11))
def test_update_failure_stops_later_steps(recording_recipes: Path, monkeypatch: pytest.MonkeyPatch, step: int) -> None:
    monkeypatch.setenv("FAIL_STEP", str(step))
    result, calls = invoke_recipe(recording_recipes, "update")
    assert result.returncode == 23, result.stderr
    assert calls == update_commands("update")[:step]


@pytest.mark.parametrize(
    ("recipe", "arguments", "command"),
    [
        ("review", [], ["review", "branch", "--base=origin/main"]),
        ("review", ["topic/$(touch SHOULD_NOT_EXIST);literal"], ["review", "branch", "--base=topic/$(touch SHOULD_NOT_EXIST);literal"]),
        ("review-uncommitted", [], ["review", "uncommitted"]),
        (
            "changelog-unreleased",
            ["v0.9.0", "2026-10-04"],
            ["toolchain", "run", "--", "research-repo-tools", "changelog", "generate", "--tag", "v0.9.0", "--date", "2026-10-04"],
        ),
        ("release-notes", ["v0.7.5"], ["changelog", "notes", "v0.7.5"]),
    ],
)
def test_public_recipe_arguments(recording_recipes: Path, recipe: str, arguments: list[str], command: list[str]) -> None:
    result, calls = invoke_recipe(recording_recipes, recipe, *arguments)
    assert result.returncode == 0, result.stderr
    assert calls == [["run", "--locked", "--group", "dev", "research-repo-tools", *command]]
    assert not (recording_recipes / "SHOULD_NOT_EXIST").exists()


@pytest.mark.parametrize(("scope", "base"), [("branch", "origin/main"), ("branch", "topic/local"), ("uncommitted", None)])
@pytest.mark.parametrize("status", [0, 23])
def test_installed_review_discovers_consumer_instructions_and_propagates_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scope: str, base: str | None, status: int
) -> None:
    """Exercise the installed shared implementation without GitHub or CodeRabbit."""
    git_log = tmp_path / "git.jsonl"
    monkeypatch.setenv("GIT_LOG", str(git_log))
    write_stub(
        tmp_path,
        "git",
        "import json, os, sys\nfrom pathlib import Path\n"
        "with Path(os.environ['GIT_LOG']).open('a') as stream: stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "print('a' * 40 if 'ls-remote' not in sys.argv else 'a' * 40 + '\\trefs/heads/main')\n",
    )
    log = tmp_path / "review.json"
    monkeypatch.setenv("REVIEW_LOG", str(log))
    write_stub(
        tmp_path,
        "coderabbit",
        f"import json, os, sys\nfrom pathlib import Path\nPath(os.environ['REVIEW_LOG']).write_text(json.dumps(sys.argv[1:]))\nsys.exit({status})\n",
    )
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    options = ["--base", base] if base is not None else []
    assert main(["--root", str(ROOT), "review", scope, *options]) == status
    arguments = json.loads(log.read_text(encoding="utf-8"))
    assert arguments[:3] == ["review", "--agent", "--include-untracked"]
    assert arguments[-3:] == ["--config", str(ROOT / "AGENTS.md"), str(ROOT / ".coderabbit.yml")]
    if base is not None:
        assert f"--base={base}" in arguments
    git_calls = [json.loads(line) for line in git_log.read_text(encoding="utf-8").splitlines()] if git_log.exists() else []
    assert any("ls-remote" in call for call in git_calls) == (base == "origin/main")
    assert bool(git_calls) == (scope == "branch")
    assert ("--uncommitted" in arguments) == (scope == "uncommitted")


def test_real_release_configuration_is_valid() -> None:
    assert main(["--root", str(ROOT), "release", "check"]) == 0


def test_active_document_rules_exclude_migrated_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The same stale prose blocks active docs and remains valid in release history."""
    active = tmp_path / "docs/history_probe.md"
    archived = tmp_path / "docs/archives/changelog/0.7.md"
    archived.parent.mkdir(parents=True)
    content = (
        "# Historical notes\n\n`Point<f64, 3>` was the old API.\n\n"
        "The old example called `build().unwrap()`.\n\n"
        "This release introduced a 4-level validation hierarchy.\n\n"
        "Validate compact 3D toroidal quotients.\n"
    )
    for path in (active, archived):
        path.write_text(content, encoding="utf-8")
    monkeypatch.setenv("SEMGREP_SETTINGS_FILE", str(tmp_path / "semgrep-settings.yml"))
    result = run_safe_command(
        "semgrep",
        [
            "scan",
            "--config",
            str(ROOT / "semgrep.yaml"),
            "--json",
            "--error",
            "--metrics=off",
            "--disable-version-check",
            "--no-git-ignore",
            str(active),
            str(archived),
        ],
        cwd=tmp_path,
        check=False,
    )
    assert result.returncode == 1, result.stderr
    findings = json.loads(result.stdout)["results"]
    assert len(findings) == 4
    assert {Path(finding["path"]).resolve() for finding in findings} == {active.resolve()}


def test_archive_notes_are_available_from_installed_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--root", str(ROOT), "changelog", "notes", "v0.7.5"]) == 0
    assert "0.7.5" in capsys.readouterr().out


def test_installed_review_rejects_stale_default_before_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """The actual consumer fails closed before its review stub can run."""
    write_stub(tmp_path, "git", "import sys\nprint('a' * 40 if 'ls-remote' not in sys.argv else 'b' * 40 + '\\trefs/heads/main')\n")
    marker = tmp_path / "started"
    monkeypatch.setenv("REVIEW_MARKER", str(marker))
    write_stub(tmp_path, "coderabbit", "import os\nfrom pathlib import Path\nPath(os.environ['REVIEW_MARKER']).touch()\n")
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    assert main(["--root", str(ROOT), "review", "branch"]) == 1
    assert "git fetch origin" in capsys.readouterr().err
    assert not marker.exists()


def test_installed_prospective_changelog_preserves_date_and_rust_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Preview and publication preserve declared dates and literal Rust generics."""
    manifest = (ROOT / "pyproject.toml").read_text(encoding="utf-8").replace('formatter = "docs/templates/changelog_format.toml"\n', "")
    (tmp_path / "pyproject.toml").write_text(manifest, encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## [0.8.2] - 2026-09-15\n\n- Previous release.\n", encoding="utf-8")
    generated = (
        "# Changelog\n\n## [0.9.0] - 1999-01-01\n\n### Changed\n\n"
        "- Preserve `Tds<f64, (), (), 3>`.\n\n```rust\nlet x: Vec<Option<f64>> = Vec::new();\n```\n\n"
        "## [0.8.2] - 1999-01-01\n\n- Previous release.\n"
    )
    write_stub(tmp_path, "git-cliff", f"print({generated!r})\n")
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    original = (tmp_path / "CHANGELOG.md").read_bytes()
    arguments = ["--root", str(tmp_path), "changelog", "generate", "--tag", "v0.9.0", "--date", "2026-10-04"]
    assert main([*arguments, "--dry-run"]) == 0
    preview = capsys.readouterr().out
    assert (tmp_path / "CHANGELOG.md").read_bytes() == original
    assert not (tmp_path / "docs").exists()
    assert "## [0.9.0] - 2026-10-04" in preview
    assert "`Tds<f64, (), (), 3>`" in preview
    assert "let x: Vec<Option<f64>> = Vec::new();" in preview
    assert main(arguments) == 0
    assert (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8") == preview
    assert "## [0.8.2] - 2026-09-15" in (tmp_path / "docs/archives/changelog/0.8.md").read_text(encoding="utf-8")
