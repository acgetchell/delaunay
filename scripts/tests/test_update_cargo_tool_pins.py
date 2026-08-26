"""Tests for atomic repository tool-pin reconciliation."""

import subprocess
from typing import TYPE_CHECKING, Never

import pytest

import update_cargo_tool_pins
from subprocess_utils import ExecutableNotFoundError

if TYPE_CHECKING:
    from pathlib import Path


def installed_output(*, override: tuple[str, str] | None = None) -> str:
    """Return representative ``cargo install --list`` output for every managed tool."""
    versions = dict.fromkeys(update_cargo_tool_pins.PIN_TO_PACKAGE.values(), "1.2.3")
    if override is not None:
        versions[override[0]] = override[1]
    return "".join(f"{package} v{version}:\n    {package}\n" for package, version in versions.items())


def justfile_text(version: str = "1.2.3") -> str:
    """Return one assignment for every managed Just pin."""
    return "".join(f'{pin} := "{version}"\n' for pin in update_cargo_tool_pins.PIN_TO_TOOL)


def test_reconcile_pins_updates_one_changed_version_atomically(tmp_path: Path) -> None:
    justfile = tmp_path / "justfile"
    justfile.write_text(justfile_text(), encoding="utf-8")

    changes = update_cargo_tool_pins.reconcile_pins(
        justfile,
        installed_output(override=("rumdl", "2.0.0")),
        "uv 1.2.3",
    )

    assert changes == {"rumdl_version": ("1.2.3", "2.0.0")}
    assert 'rumdl_version := "2.0.0"' in justfile.read_text(encoding="utf-8")
    assert list(tmp_path.glob(".justfile.*")) == []


def test_reconcile_pins_updates_uv_version_atomically(tmp_path: Path) -> None:
    justfile = tmp_path / "justfile"
    justfile.write_text(justfile_text(), encoding="utf-8")

    changes = update_cargo_tool_pins.reconcile_pins(justfile, installed_output(), "uv 2.0.0")

    assert changes == {"uv_version": ("1.2.3", "2.0.0")}
    assert 'uv_version := "2.0.0"' in justfile.read_text(encoding="utf-8")
    assert list(tmp_path.glob(".justfile.*")) == []


def test_reconcile_pins_rejects_missing_package_without_writing(tmp_path: Path) -> None:
    justfile = tmp_path / "justfile"
    original = justfile_text()
    justfile.write_text(original, encoding="utf-8")
    incomplete = installed_output().replace("rumdl v1.2.3:\n    rumdl\n", "")

    with pytest.raises(ValueError, match="managed tool is not installed: rumdl"):
        update_cargo_tool_pins.reconcile_pins(justfile, incomplete, "uv 1.2.3")

    assert justfile.read_text(encoding="utf-8") == original


def test_update_pin_text_rejects_duplicate_assignment() -> None:
    duplicated = justfile_text() + 'rumdl_version := "1.2.3"\n'
    installed = update_cargo_tool_pins.parse_installed_packages(installed_output())
    installed["uv"] = "1.2.3"

    with pytest.raises(ValueError, match="expected exactly one rumdl_version assignment, found 2"):
        update_cargo_tool_pins.update_pin_text(duplicated, installed)


def test_parse_installed_packages_accepts_prerelease_with_build_metadata() -> None:
    version = "1.2.3-rc.1+build.5"

    installed = update_cargo_tool_pins.parse_installed_packages(installed_output(override=("rumdl", version)))

    assert installed["rumdl"] == version


def test_reconcile_pins_preserves_prerelease_with_build_metadata(tmp_path: Path) -> None:
    version = "1.2.3-rc.1+build.5"
    justfile = tmp_path / "justfile"
    justfile.write_text(justfile_text(), encoding="utf-8")

    changes = update_cargo_tool_pins.reconcile_pins(
        justfile,
        installed_output(override=("rumdl", version)),
        "uv 1.2.3",
    )

    assert changes == {"rumdl_version": ("1.2.3", version)}
    assert f'rumdl_version := "{version}"' in justfile.read_text(encoding="utf-8").splitlines()


@pytest.mark.parametrize("output", ["uv 1.2.3-rc.1", "uv 1.2.3.4", "uv release-1.2.3"])
def test_parse_tool_version_rejects_nonstable_or_embedded_versions(output: str) -> None:
    with pytest.raises(ValueError, match="expected exactly one uv version"):
        update_cargo_tool_pins.parse_tool_version(output, "uv")


def test_main_reports_missing_cargo_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def missing_cargo(_args: list[str], **_kwargs: object) -> Never:
        msg = "Required executable 'cargo' not found in PATH"
        raise ExecutableNotFoundError(msg)

    monkeypatch.setattr(update_cargo_tool_pins, "run_cargo_command", missing_cargo)

    assert update_cargo_tool_pins.main([]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "failed to update tool pins: Required executable 'cargo' not found in PATH\n"


def test_main_reports_missing_uv_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        update_cargo_tool_pins,
        "run_cargo_command",
        lambda _args, **_kwargs: subprocess.CompletedProcess([], 0, stdout=installed_output(), stderr=""),
    )

    def missing_uv(_command: str, _args: list[str], **_kwargs: object) -> Never:
        msg = "Required executable 'uv' not found in PATH"
        raise ExecutableNotFoundError(msg)

    monkeypatch.setattr(update_cargo_tool_pins, "run_safe_command", missing_uv)

    assert update_cargo_tool_pins.main([]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "failed to update tool pins: Required executable 'uv' not found in PATH\n"
