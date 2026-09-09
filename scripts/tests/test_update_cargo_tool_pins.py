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


@pytest.mark.parametrize(
    ("cargo_output", "message"),
    [
        (installed_output() + "rumdl v2.0.0:\n    rumdl\n", "duplicate installed Cargo package: rumdl"),
        (installed_output(override=("rumdl", "2.0")), "invalid installed version for rumdl: 2.0"),
    ],
)
def test_reconcile_pins_rejects_invalid_cargo_inventory_without_writing(
    cargo_output: str,
    message: str,
    tmp_path: Path,
) -> None:
    justfile = tmp_path / "justfile"
    original = justfile_text()
    justfile.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        update_cargo_tool_pins.reconcile_pins(justfile, cargo_output, "uv 2.0.0")

    assert justfile.read_text(encoding="utf-8") == original
    assert list(tmp_path.iterdir()) == [justfile]


def test_reconcile_pins_preserves_file_and_cleans_up_failed_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    justfile = tmp_path / "justfile"
    original = justfile_text()
    justfile.write_text(original, encoding="utf-8")
    original_mode = justfile.stat().st_mode

    def failed_replace(temporary: Path, destination: Path) -> Never:
        assert temporary.parent == tmp_path
        assert temporary != justfile
        assert destination == justfile
        msg = "replacement denied"
        raise PermissionError(msg)

    with monkeypatch.context() as patch:
        patch.setattr(type(justfile), "replace", failed_replace)
        with pytest.raises(PermissionError, match="replacement denied"):
            update_cargo_tool_pins.reconcile_pins(justfile, installed_output(), "uv 2.0.0")

    assert justfile.read_text(encoding="utf-8") == original
    assert justfile.stat().st_mode == original_mode
    assert list(tmp_path.iterdir()) == [justfile]

    changes = update_cargo_tool_pins.reconcile_pins(justfile, installed_output(), "uv 2.0.0")

    assert changes == {"uv_version": ("1.2.3", "2.0.0")}
    assert justfile.read_text(encoding="utf-8") == original.replace('uv_version := "1.2.3"', 'uv_version := "2.0.0"')
    assert justfile.stat().st_mode == original_mode
    assert list(tmp_path.iterdir()) == [justfile]


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


@pytest.mark.parametrize(
    "output",
    ["", "uv unknown", "uv 1.2", "uv 1.2.3-rc.1", "uv 1.2.3+build.1", "uv 1.2.3.4", "uv release-1.2.3", "uv 1.2.3 2.0.0", "uv 1.2.3\nuv 1.2.3"],
)
def test_uv_preflight_and_reconciler_reject_invalid_versions_without_writing(
    output: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    justfile = tmp_path / "justfile"
    original = justfile_text()
    justfile.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        update_cargo_tool_pins,
        "run_safe_command",
        lambda _command, _args, **_kwargs: subprocess.CompletedProcess([], 0, stdout=output, stderr=""),
    )

    def unexpected_cargo(_args: list[str], **_kwargs: object) -> Never:
        pytest.fail("uv preflight must not inspect Cargo packages")

    monkeypatch.setattr(update_cargo_tool_pins, "run_cargo_command", unexpected_cargo)

    assert update_cargo_tool_pins.main(["--check-uv", "--justfile", str(justfile)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "failed uv preflight (requires stable X.Y.Z): expected exactly one uv version" in captured.err
    assert justfile.read_text(encoding="utf-8") == original

    with pytest.raises(ValueError, match="expected exactly one uv version"):
        update_cargo_tool_pins.reconcile_pins(justfile, installed_output(), output)

    assert justfile.read_text(encoding="utf-8") == original
    assert list(tmp_path.iterdir()) == [justfile]


@pytest.mark.parametrize("output", ["uv 99.0.0", "uv v99.0.0", "uv 99.0.0 (Homebrew 2026-09-04 aarch64-apple-darwin)\n"])
def test_uv_preflight_accepts_stable_versions_without_reading_pins_or_cargo(
    output: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_justfile = tmp_path / "justfile"

    def uv_version(command: str, args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command == "uv"
        assert args == ["--version"]
        assert kwargs == {"timeout": 30}
        return subprocess.CompletedProcess([], 0, stdout=output, stderr="")

    def unexpected_cargo(_args: list[str], **_kwargs: object) -> Never:
        pytest.fail("uv preflight must not inspect Cargo packages")

    monkeypatch.setattr(update_cargo_tool_pins, "run_safe_command", uv_version)
    monkeypatch.setattr(update_cargo_tool_pins, "run_cargo_command", unexpected_cargo)

    assert update_cargo_tool_pins.main(["--check-uv", "--justfile", str(missing_justfile)]) == 0
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert not missing_justfile.exists()


@pytest.mark.parametrize(
    "error",
    [
        ExecutableNotFoundError("uv missing"),
        OSError("uv unavailable"),
        subprocess.CalledProcessError(2, ["uv", "--version"]),
        subprocess.TimeoutExpired("uv", 30),
    ],
)
def test_uv_preflight_reports_command_failure_without_traceback(
    error: Exception,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def failed_uv(_command: str, _args: list[str], **_kwargs: object) -> Never:
        raise error

    monkeypatch.setattr(update_cargo_tool_pins, "run_safe_command", failed_uv)

    assert update_cargo_tool_pins.main(["--check-uv"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"failed uv preflight (requires stable X.Y.Z): {error}\n"


def test_main_reports_updated_pins_and_leaves_matching_file_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    justfile = tmp_path / "justfile"
    original = justfile_text()
    justfile.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        update_cargo_tool_pins,
        "run_cargo_command",
        lambda _args, **_kwargs: subprocess.CompletedProcess([], 0, stdout=installed_output(), stderr=""),
    )
    monkeypatch.setattr(
        update_cargo_tool_pins,
        "run_safe_command",
        lambda _command, _args, **_kwargs: subprocess.CompletedProcess([], 0, stdout="uv 2.0.0", stderr=""),
    )

    assert update_cargo_tool_pins.main(["--justfile", str(justfile)]) == 0
    captured = capsys.readouterr()
    assert captured.out == "Updated uv_version: 1.2.3 -> 2.0.0\n"
    assert captured.err == ""
    expected = original.replace('uv_version := "1.2.3"', 'uv_version := "2.0.0"')
    assert justfile.read_text(encoding="utf-8") == expected
    updated_stat = justfile.stat()

    assert update_cargo_tool_pins.main(["--justfile", str(justfile)]) == 0
    captured = capsys.readouterr()
    assert captured.out == "Tool pins already match installed repository tools.\n"
    assert captured.err == ""
    assert justfile.read_text(encoding="utf-8") == expected
    assert justfile.stat().st_ino == updated_stat.st_ino
    assert justfile.stat().st_mtime_ns == updated_stat.st_mtime_ns
    assert list(tmp_path.iterdir()) == [justfile]


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
