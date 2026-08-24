"""Tests for resolver-backed Python development-tool pin updates."""

import subprocess
from typing import TYPE_CHECKING

import pytest

import update_python_dev_pins

if TYPE_CHECKING:
    from pathlib import Path


def project_text(*requirements: str) -> str:
    """Return a minimal project with a mixed development requirement group."""
    rendered = "\n".join(f'    "{requirement}",' for requirement in requirements)
    return f"""[build-system]
requires = ["setuptools>=83"]

[project]
name = "fixture"
version = "0.1.0"
requires-python = ">=3.14"
dependencies = ["packaging>=26"]

[dependency-groups]
dev = [
{rendered}
]
"""


def test_parse_project_selects_only_exact_simple_dev_pins() -> None:
    python_version, pins = update_python_dev_pins.parse_project(
        project_text("pytest>=9.1", "ruff==0.16.2", "semgrep==1.172.0", "ty~=0.0.66"),
    )

    assert python_version == "3.14"
    assert pins == [
        update_python_dev_pins.DevPin("ruff", "0.16.2"),
        update_python_dev_pins.DevPin("semgrep", "1.172.0"),
    ]


def test_parse_project_leaves_compound_and_wildcard_requirements_unmanaged() -> None:
    python_version, pins = update_python_dev_pins.parse_project(
        project_text("ruff==0.16.2,!=0.16.3", "semgrep==1.172.*", "ty==0.0.66; python_version >= '3.14'"),
    )

    assert python_version == "3.14"
    assert pins == []


def test_parse_project_accepts_group_without_exact_pins() -> None:
    python_version, pins = update_python_dev_pins.parse_project(project_text("pytest>=9.1", "ruff~=0.16"))

    assert python_version == "3.14"
    assert pins == []


def test_parse_resolution_preserves_direct_order_and_ignores_transitives() -> None:
    pins = [
        update_python_dev_pins.DevPin("ruff", "0.16.2"),
        update_python_dev_pins.DevPin("semgrep", "1.172.0"),
    ]
    output = "packaging==26.3\nsemgrep==1.174.0\nruff==0.16.4\nmcp==1.29.0\n"

    assert update_python_dev_pins.parse_resolution(output, pins) == [
        update_python_dev_pins.DevPin("ruff", "0.16.4"),
        update_python_dev_pins.DevPin("semgrep", "1.174.0"),
    ]


def test_parse_resolution_rejects_missing_direct_tool() -> None:
    pins = [update_python_dev_pins.DevPin("semgrep", "1.172.0")]

    with pytest.raises(ValueError, match="uv resolver output omitted direct development tool: semgrep"):
        update_python_dev_pins.parse_resolution("mcp==1.29.0\n", pins)


def test_update_dev_pins_resolves_then_applies_one_exact_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    original = project_text("pytest>=9.1", "ruff==0.16.2", "semgrep==1.172.0", "ty~=0.0.66")
    pyproject.write_text(original, encoding="utf-8")
    uv_lock = tmp_path / "uv.lock"
    uv_lock.write_text("version = 1\nruff = 0.16.2\nsemgrep = 1.172.0\n", encoding="utf-8")
    calls: list[tuple[str, list[str], dict[str, object]]] = []

    def fake_run(command: str, args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, args, kwargs))
        if args[:2] == ["pip", "compile"]:
            output = "ruff==0.16.4\nsemgrep==1.174.0\nmcp==1.29.0\n"
        else:
            pyproject.write_text(
                pyproject.read_text(encoding="utf-8").replace("ruff==0.16.2", "ruff==0.16.4").replace("semgrep==1.172.0", "semgrep==1.174.0"),
                encoding="utf-8",
            )
            uv_lock.write_text("version = 1\nruff = 0.16.4\nsemgrep = 1.174.0\n", encoding="utf-8")
            output = ""
        return subprocess.CompletedProcess([command, *args], 0, stdout=output, stderr="")

    monkeypatch.setattr(update_python_dev_pins, "run_safe_command", fake_run)

    changes = update_python_dev_pins.update_dev_pins(pyproject)

    assert changes == {
        "ruff": ("0.16.2", "0.16.4"),
        "semgrep": ("1.172.0", "1.174.0"),
    }
    assert calls[0] == (
        "uv",
        [
            "pip",
            "compile",
            "-",
            "--universal",
            "--no-header",
            "--no-annotate",
            "--python-version",
            "3.14",
        ],
        {
            "cwd": tmp_path,
            "input": "ruff\nsemgrep\n",
            "timeout": update_python_dev_pins.UV_PIP_COMPILE_TIMEOUT_SECONDS,
        },
    )
    assert calls[1] == (
        "uv",
        ["add", "--dev", "--no-sync", "ruff==0.16.4", "semgrep==1.174.0"],
        {"cwd": tmp_path, "timeout": update_python_dev_pins.UV_ADD_TIMEOUT_SECONDS},
    )
    assert uv_lock.read_text(encoding="utf-8") == "version = 1\nruff = 0.16.4\nsemgrep = 1.174.0\n"


def test_update_dev_pins_rolls_back_collateral_manifest_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    original = project_text("pytest>=9.1", "ruff==0.16.2")
    pyproject.write_text(original, encoding="utf-8")

    def mutate_unmanaged_requirement(command: str, args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[:2] == ["pip", "compile"]:
            return subprocess.CompletedProcess([command, *args], 0, stdout="ruff==0.16.4\n", stderr="")
        pyproject.write_text(original.replace("ruff==0.16.2", "ruff==0.16.4").replace("pytest>=9.1", "pytest>=9.2"), encoding="utf-8")
        return subprocess.CompletedProcess([command, *args], 0, stdout="", stderr="")

    monkeypatch.setattr(update_python_dev_pins, "run_safe_command", mutate_unmanaged_requirement)

    with pytest.raises(ValueError, match="changed non-target manifest content"):
        update_python_dev_pins.update_dev_pins(pyproject)

    assert pyproject.read_text(encoding="utf-8") == original


def test_resolver_failure_leaves_manifest_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    original = project_text("pytest>=9.1", "ruff==0.16.2")
    pyproject.write_text(original, encoding="utf-8")

    def failed_uv(_command: str, args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, ["uv", *args], stderr="resolver conflict")

    monkeypatch.setattr(update_python_dev_pins, "run_safe_command", failed_uv)

    with pytest.raises(subprocess.CalledProcessError):
        update_python_dev_pins.update_dev_pins(pyproject)

    assert pyproject.read_text(encoding="utf-8") == original


def test_main_skips_uv_when_group_has_no_exact_pins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(project_text("pytest>=9.1", "ruff~=0.16"), encoding="utf-8")

    def unexpected_uv(*_args: object, **_kwargs: object) -> None:
        msg = "uv must not run without exact pins"
        raise AssertionError(msg)

    monkeypatch.setattr(update_python_dev_pins, "run_safe_command", unexpected_uv)

    assert update_python_dev_pins.main(["--pyproject", str(pyproject)]) == 0
    assert capsys.readouterr().out == "No exact direct Python development-tool pins to update.\n"


def test_main_reports_uv_diagnostics_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(project_text("semgrep==1.172.0"), encoding="utf-8")

    def failed_uv(_command: str, args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, ["uv", *args], stderr="resolver conflict")

    monkeypatch.setattr(update_python_dev_pins, "run_safe_command", failed_uv)

    assert update_python_dev_pins.main(["--pyproject", str(pyproject)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "failed to update Python development-tool pins: resolver conflict\n"


def test_main_uses_stdout_when_uv_stderr_is_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(project_text("semgrep==1.172.0"), encoding="utf-8")

    def failed_uv(_command: str, args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, ["uv", *args], output="actionable resolver conflict", stderr="")

    monkeypatch.setattr(update_python_dev_pins, "run_safe_command", failed_uv)

    assert update_python_dev_pins.main(["--pyproject", str(pyproject)]) == 1
    assert capsys.readouterr().err == "failed to update Python development-tool pins: actionable resolver conflict\n"


def test_update_dev_pins_rejects_nonstandard_manifest_before_uv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "release-tools.toml"
    original = project_text("ruff==0.16.2")
    manifest.write_text(original, encoding="utf-8")

    def unexpected_uv(*_args: object, **_kwargs: object) -> None:
        msg = "uv must not discover a sibling project for a nonstandard manifest"
        raise AssertionError(msg)

    monkeypatch.setattr(update_python_dev_pins, "run_safe_command", unexpected_uv)

    with pytest.raises(ValueError, match=r"conventional pyproject\.toml"):
        update_python_dev_pins.update_dev_pins(manifest)

    assert manifest.read_text(encoding="utf-8") == original


def test_update_dev_pins_rejects_uv_workspace_member_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_manifest = tmp_path / "pyproject.toml"
    root_manifest.write_text('[tool.uv.workspace]\nmembers = ["member"]\n', encoding="utf-8")
    root_lock = tmp_path / "uv.lock"
    root_lock.write_bytes(b"workspace lock\n")
    member = tmp_path / "member"
    member.mkdir()
    member_manifest = member / "pyproject.toml"
    member_manifest.write_text(project_text("ruff==0.16.2"), encoding="utf-8")

    def unexpected_uv(*_args: object, **_kwargs: object) -> None:
        msg = "uv must not run for a workspace member"
        raise AssertionError(msg)

    monkeypatch.setattr(update_python_dev_pins, "run_safe_command", unexpected_uv)

    with pytest.raises(TypeError, match="must not select a uv workspace member"):
        update_python_dev_pins.update_dev_pins(member_manifest)

    assert root_lock.read_bytes() == b"workspace lock\n"
    assert member_manifest.read_text(encoding="utf-8") == project_text("ruff==0.16.2")


def test_main_reports_resolver_timeout_and_leaves_project_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    original = project_text("ruff==0.16.2")
    pyproject.write_text(original, encoding="utf-8")

    def timed_out(_command: str, args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        timeout = kwargs.get("timeout")
        if not isinstance(timeout, int | float):
            msg = "expected a finite subprocess timeout"
            raise TypeError(msg)
        raise subprocess.TimeoutExpired(["uv", *args], timeout)

    monkeypatch.setattr(update_python_dev_pins, "run_safe_command", timed_out)

    assert update_python_dev_pins.main(["--pyproject", str(pyproject)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "timed out after 300 seconds" in captured.err
    assert pyproject.read_text(encoding="utf-8") == original


def test_main_rolls_back_manifest_and_lock_after_uv_add_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    original_manifest = project_text("ruff==0.16.2")
    pyproject.write_text(original_manifest, encoding="utf-8")
    uv_lock = tmp_path / "uv.lock"
    original_lock = b"version = 1\n"
    uv_lock.write_bytes(original_lock)

    def time_out_after_mutation(command: str, args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[:2] == ["pip", "compile"]:
            return subprocess.CompletedProcess([command, *args], 0, stdout="ruff==0.16.4\n", stderr="")
        pyproject.write_text(original_manifest.replace("ruff==0.16.2", "ruff==0.16.4"), encoding="utf-8")
        uv_lock.write_text("partially updated\n", encoding="utf-8")
        timeout = kwargs.get("timeout")
        if not isinstance(timeout, int | float):
            msg = "expected a finite subprocess timeout"
            raise TypeError(msg)
        raise subprocess.TimeoutExpired(["uv", *args], timeout)

    monkeypatch.setattr(update_python_dev_pins, "run_safe_command", time_out_after_mutation)

    assert update_python_dev_pins.main(["--pyproject", str(pyproject)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "timed out after 300 seconds" in captured.err
    assert pyproject.read_text(encoding="utf-8") == original_manifest
    assert uv_lock.read_bytes() == original_lock


def test_main_reports_primary_and_rollback_failures_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(project_text("ruff==0.16.2"), encoding="utf-8")

    def fail_update(command: str, args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[:2] == ["pip", "compile"]:
            return subprocess.CompletedProcess([command, *args], 0, stdout="ruff==0.16.4\n", stderr="")
        msg = "primary update failure"
        raise OSError(msg)

    def fail_restore(_snapshots: object) -> None:
        msg = "rollback failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(update_python_dev_pins, "run_safe_command", fail_update)
    monkeypatch.setattr(update_python_dev_pins, "_restore_snapshots", fail_restore)

    assert update_python_dev_pins.main(["--pyproject", str(pyproject)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "primary update failure" in captured.err
    assert "rollback failure" in captured.err
