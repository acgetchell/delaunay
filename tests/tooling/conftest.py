"""
Shared pytest fixtures and utilities for test modules.

Provides common testing utilities that can be reused across multiple test files.
"""

import base64
import csv
import hashlib
import io
import os
import subprocess
import sys
import venv
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from research_repo_tools.process import resolve_executable, run_command

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from contextlib import AbstractContextManager

# Ensure the scientific module source is on sys.path for test imports.
# This must be done before importing any local modules
_modules = Path(__file__).resolve().parents[2] / "tooling" / "python"
if str(_modules) not in sys.path:
    sys.path.insert(0, str(_modules))


@pytest.fixture
def temp_chdir() -> Callable[[os.PathLike[str] | str], AbstractContextManager[None]]:
    """
    Pytest fixture for temporarily changing working directory.

    Returns a context manager that can be used to change directories
    and automatically restore the original directory.

    Usage:
        def test_something(temp_chdir):
            with temp_chdir(some_path):
                # Code that runs in some_path
                pass
            # Back to original directory
    """

    @contextmanager
    def _temp_chdir_context(path: os.PathLike[str] | str) -> Iterator[None]:
        """Context manager for temporarily changing working directory."""
        original_cwd = Path.cwd()
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(target)
        os.chdir(target)
        try:
            yield
        finally:
            os.chdir(original_cwd)

    return _temp_chdir_context


@pytest.fixture
def mock_git_command_result() -> Callable[[str], subprocess.CompletedProcess[str]]:
    """
    Pytest fixture for creating mock CompletedProcess objects for git commands.

    Returns a function that creates a CompletedProcess with the specified stdout output.
    This standardizes git command mocking across all test files.

    Usage:
        def test_something(mock_git_command_result):
            mock_result = mock_git_command_result("v0.4.2")
            # mock_result.stdout.strip() will return "v0.4.2"
    """

    def _create_mock_result(output: str) -> subprocess.CompletedProcess[str]:
        """Create a typed CompletedProcess object for git commands."""
        return subprocess.CompletedProcess(args=["git"], returncode=0, stdout=output, stderr="")

    return _create_mock_result


@pytest.fixture
def native_console_scripts() -> Callable[[Path, str, dict[str, str]], Path]:
    """Install offline fixture entry points with native Windows/Unix launchers.

    Extensionless shell scripts on PATH cannot intercept native Windows callers.
    Batch files also introduce shell parsing of provider queries and arguments.
    A small wheel lets uv supply the same native launchers as installed tools.
    """

    def install(root: Path, source: str, entry_points: dict[str, str]) -> Path:
        """Return the isolated script directory after verifying PATH resolution."""
        runtime = root / "fixture-runtime"
        venv.EnvBuilder(with_pip=False).create(runtime)
        scripts = runtime / ("Scripts" if os.name == "nt" else "bin")
        python = scripts / ("python.exe" if os.name == "nt" else "python")
        metadata = "fixture_tools-0.0.0.dist-info"
        files = {
            "fixture_tools.py": source.encode(),
            f"{metadata}/METADATA": b"Metadata-Version: 2.1\nName: fixture-tools\nVersion: 0.0.0\n",
            f"{metadata}/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            f"{metadata}/entry_points.txt": (
                "[console_scripts]\n" + "".join(f"{name} = fixture_tools:{function}\n" for name, function in entry_points.items())
            ).encode(),
        }
        record = io.StringIO(newline="")
        writer = csv.writer(record)
        for name, content in files.items():
            digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
            writer.writerow([name, f"sha256={digest}", len(content)])
        writer.writerow([f"{metadata}/RECORD", "", ""])
        files[f"{metadata}/RECORD"] = record.getvalue().encode()
        wheel = root / "fixture_tools-0.0.0-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            for name, content in files.items():
                archive.writestr(name, content)
        run_command(
            "uv",
            ["pip", "install", "--python", str(python), "--no-index", "--no-deps", str(wheel)],
            env=os.environ | {"UV_OFFLINE": "1", "UV_CACHE_DIR": str(root / "uv-cache")},
            timeout=60,
        )
        environment = os.environ | {"PATH": f"{scripts}{os.pathsep}{os.environ['PATH']}"}
        for name in entry_points:
            executable = resolve_executable(name, cwd=root, env=environment)
            if executable.parent != scripts or (os.name == "nt" and executable.suffix.lower() != ".exe"):
                pytest.fail(f"Fixture {name} did not resolve to its isolated native launcher: {executable}")
        return scripts

    return install
