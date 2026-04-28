"""
Shared pytest fixtures and utilities for test modules.

Provides common testing utilities that can be reused across multiple test files.
"""

import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

import pytest

# Ensure `scripts/` is on sys.path for test imports
# This must be done before importing any local modules
_scripts = Path(__file__).resolve().parents[1]
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))


@pytest.fixture
def temp_chdir() -> Callable[[os.PathLike | str], AbstractContextManager[None]]:
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
    def _temp_chdir_context(path: os.PathLike | str) -> Iterator[None]:
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
