"""Reconcile repository tool pins with installed package versions."""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from subprocess_utils import ExecutableNotFoundError, run_cargo_command, run_safe_command

PIN_TO_PACKAGE = {
    "cargo_edit_version": "cargo-edit",
    "cargo_llvm_cov_version": "cargo-llvm-cov",
    "cargo_machete_version": "cargo-machete",
    "dprint_version": "dprint",
    "git_cliff_version": "git-cliff",
    "just_version": "just",
    "nextest_version": "cargo-nextest",
    "rumdl_version": "rumdl",
    "samply_version": "samply",
    "taplo_version": "taplo-cli",
    "tectonic_version": "tectonic",
    "tex_fmt_version": "tex-fmt",
    "typos_version": "typos-cli",
    "zizmor_version": "zizmor",
}
PIN_TO_TOOL = {**PIN_TO_PACKAGE, "uv_version": "uv"}
PACKAGE_HEADER = re.compile(r"^(?P<package>[A-Za-z0-9_-]+) v(?P<version>[^\s:]+):$", re.MULTILINE)
VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")
STABLE_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
TOOL_VERSION = re.compile(r"(?<![0-9A-Za-z_.+-])v?(?P<version>[0-9]+\.[0-9]+\.[0-9]+)(?![0-9A-Za-z_.+-])")


def parse_installed_packages(output: str) -> dict[str, str]:
    """Parse package versions from ``cargo install --list`` output."""
    packages: dict[str, str] = {}
    for match in PACKAGE_HEADER.finditer(output):
        package = match.group("package")
        version = match.group("version")
        if package in packages:
            msg = f"duplicate installed Cargo package: {package}"
            raise ValueError(msg)
        if VERSION.fullmatch(version) is None:
            msg = f"invalid installed version for {package}: {version}"
            raise ValueError(msg)
        packages[package] = version
    return packages


def parse_tool_version(output: str, tool: str) -> str:
    """Extract one stable semantic version from a tool's version output."""
    versions = [match.group("version") for match in TOOL_VERSION.finditer(output)]
    if len(versions) != 1:
        msg = f"expected exactly one {tool} version, found {len(versions)}"
        raise ValueError(msg)
    version = versions[0]
    if STABLE_VERSION.fullmatch(version) is None:
        msg = f"invalid installed version for {tool}: {version}; expected stable X.Y.Z"
        raise ValueError(msg)
    return str(version)


def update_pin_text(text: str, installed: dict[str, str]) -> tuple[str, dict[str, tuple[str, str]]]:
    """Return Just source with every managed pin reconciled exactly once."""
    updated = text
    changes: dict[str, tuple[str, str]] = {}
    for pin, tool in PIN_TO_TOOL.items():
        version = installed.get(tool)
        if version is None:
            msg = f"managed tool is not installed: {tool}"
            raise ValueError(msg)
        assignment = re.compile(rf'^(?P<prefix>{re.escape(pin)}\s*:=\s*")(?P<version>[^"]+)(?P<suffix>"\s*)$', re.MULTILINE)
        matches = list(assignment.finditer(updated))
        if len(matches) != 1:
            msg = f"expected exactly one {pin} assignment, found {len(matches)}"
            raise ValueError(msg)
        old_version = matches[0].group("version")
        if old_version == version:
            continue
        updated = assignment.sub(rf"\g<prefix>{version}\g<suffix>", updated, count=1)
        changes[pin] = (old_version, version)
    return updated, changes


def reconcile_pins(justfile: Path, installed_output: str, uv_output: str) -> dict[str, tuple[str, str]]:
    """Atomically reconcile ``justfile`` and return changed pin versions."""
    original = justfile.read_text(encoding="utf-8")
    installed = parse_installed_packages(installed_output)
    installed["uv"] = parse_tool_version(uv_output, "uv")
    updated, changes = update_pin_text(original, installed)
    if not changes:
        return changes

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{justfile.name}.", dir=justfile.parent, text=True)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(updated)
        temporary.chmod(justfile.stat().st_mode)
        temporary.replace(justfile)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return changes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--justfile", type=Path, default=Path("justfile"), help="Just source containing repository tool pins")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Reconcile managed pins from the active Cargo and uv installations."""
    args = parse_args(argv)
    try:
        cargo = run_cargo_command(["install", "--list"], timeout=30)
        uv = run_safe_command("uv", ["--version"], timeout=30)
        changes = reconcile_pins(args.justfile, cargo.stdout, uv.stdout)
    except (ExecutableNotFoundError, OSError, subprocess.SubprocessError, ValueError) as error:
        print(f"failed to update tool pins: {error}", file=sys.stderr)
        return 1

    if not changes:
        print("Tool pins already match installed repository tools.")
        return 0
    for pin, (old_version, new_version) in changes.items():
        print(f"Updated {pin}: {old_version} -> {new_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
