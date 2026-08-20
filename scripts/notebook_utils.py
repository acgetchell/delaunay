"""Typed boundary helpers shared by the repository's tutorial notebooks."""

import math
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from subprocess_utils import run_safe_command

if TYPE_CHECKING:
    import subprocess


def find_repo_root(start: Path) -> Path:
    """Return the nearest parent containing the repository marker files."""
    for candidate in (start, *start.parents):
        if (candidate / "Cargo.toml").is_file() and (candidate / "pyproject.toml").is_file():
            return candidate
    msg = "run this notebook from inside the delaunay repository"
    raise RuntimeError(msg)


def positive_int_env(name: str, default: int) -> int:
    """Parse a positive integer environment override."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer, got {raw_value!r}") from error
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def bounded_int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    """Parse an integer environment override inside an inclusive range."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        value = default
    else:
        try:
            value = int(raw_value)
        except ValueError as error:
            raise ValueError(f"{name} must be an integer, got {raw_value!r}") from error
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got {value}")
    return value


def uint64_env(name: str, default: int) -> int:
    """Parse an unsigned 64-bit integer environment override."""
    return bounded_int_env(name, default, minimum=0, maximum=2**64 - 1)


def nonnegative_float_env(name: str, default: float) -> float:
    """Parse a finite non-negative floating-point environment override."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be a finite non-negative float, got {raw_value!r}") from error
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value}")
    return value


def string_choice_env(name: str, default: str, choices: set[str]) -> str:
    """Parse a lower-case environment choice from a finite set."""
    value = os.environ.get(name, default).strip().lower()
    if value not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}, got {value!r}")
    return value


def bool_env(name: str, *, default: bool) -> bool:
    """Parse an explicit boolean environment override."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    value = raw_value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {raw_value!r}")


def repo_output_path_env(name: str, root: Path, default: Path) -> Path:
    """Parse an optional output path, resolving relative values from the repository root."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    if not raw_value.strip():
        raise ValueError(f"{name} must not be empty")
    path = Path(raw_value).expanduser()
    return path if path.is_absolute() else root / path


def _tracked_figure_target_from_env[MissingTrackedTarget](
    name: str,
    root: Path,
    expected_relative: Path,
    absent: MissingTrackedTarget,
) -> Path | MissingTrackedTarget:
    """Validate an optional exact repo-relative tracked-figure target."""
    configured = os.environ.get(name)
    if configured is None:
        return absent
    if not configured.strip():
        raise ValueError(f"{name} must not be empty")
    path = Path(configured)
    expected = (root / expected_relative).resolve()
    if path.is_absolute() or (root / path).resolve() != expected:
        raise ValueError(f"{name} must be the repo-relative path {expected_relative.as_posix()!r}")
    return expected


def tracked_figure_path_from_env(name: str, root: Path, default: Path, expected_relative: Path) -> Path:
    """Return scratch output unless the exact tracked repository path is enabled."""
    return _tracked_figure_target_from_env(name, root, expected_relative, default)


def tracked_figure_dir_from_env(name: str, root: Path, expected_relative: Path) -> Path | None:
    """Return an exact tracked figure directory only when explicitly enabled."""
    return _tracked_figure_target_from_env(name, root, expected_relative, None)


def delaunay_command_prefix(root: Path, *, require_built_binary: bool = False) -> list[str]:
    """Return the configured, built, or Cargo-backed local CLI prefix."""
    configured = os.environ.get("DELAUNAY_BINARY")
    if configured is not None:
        if not configured.strip():
            message = "DELAUNAY_BINARY must not be empty"
            raise ValueError(message)
        binary = Path(configured).expanduser()
        binary = binary if binary.is_absolute() else root / binary
        if not binary.is_file():
            raise FileNotFoundError(f"DELAUNAY_BINARY does not point to a file: {binary}")
        return [str(binary)]

    binary_name = "delaunay.exe" if os.name == "nt" else "delaunay"
    local_binary = root / "target" / "perf" / binary_name
    if local_binary.is_file():
        return [str(local_binary)]
    if require_built_binary:
        raise FileNotFoundError(f"build the notebook CLI first: {local_binary}")

    cargo = shutil.which("cargo")
    if cargo is None:
        message = "cargo executable was not found on PATH; set DELAUNAY_BINARY to a built binary"
        raise RuntimeError(message)
    return [cargo, "run", "--profile", "perf", "--features", "cli", "--bin", "delaunay", "--"]


def run_command(command: list[str], *, cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    """Run one argv command with captured output, a timeout, and actionable failure context."""
    if not command:
        message = "command must contain an executable"
        raise ValueError(message)
    print("$", " ".join(command))
    result = run_safe_command(
        command[0],
        command[1:],
        cwd=cwd,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {' '.join(command)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result
