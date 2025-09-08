#!/usr/bin/env python3
"""
subprocess_utils.py - Secure subprocess utilities for all Python scripts

This module provides secure subprocess wrappers that:
- Use full executable paths instead of command names
- Validate executables exist before running
- Provide consistent error handling
- Mitigate security vulnerabilities flagged by Bandit

All scripts should use these functions instead of calling subprocess directly.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Any


class ExecutableNotFoundError(Exception):
    """Raised when a required executable is not found in PATH."""


def get_safe_executable(command: str) -> str:
    """
    Get the full path to an executable, validating it exists.

    Args:
        command: Command name to find (e.g., "git", "cargo")

    Returns:
        Full path to the executable

    Raises:
        ExecutableNotFoundError: If executable is not found in PATH
    """
    full_path = shutil.which(command)
    if full_path is None:
        raise ExecutableNotFoundError(f"Required executable '{command}' not found in PATH")
    return full_path


def run_git_command(args: list[str], cwd: Path | None = None, **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """
    Run a git command securely using full executable path.

    Args:
        args: Git command arguments (without 'git' prefix)
        cwd: Working directory for the command
        **kwargs: Additional arguments passed to subprocess.run
                  (e.g., capture_output=True, text=True, check=True, timeout=60)

    Returns:
        CompletedProcess result

    Raises:
        ExecutableNotFoundError: If git is not found
        subprocess.CalledProcessError: If command fails and check=True
        subprocess.TimeoutExpired: If command times out

    Note:
        When text=True (default), output uses locale encoding. For deterministic UTF-8
        in CI environments, consider passing encoding="utf-8" via kwargs.
    """
    git_path = get_safe_executable("git")
    # Set secure defaults for subprocess.run
    run_kwargs = {
        "capture_output": True,
        "text": True,
        "check": True,  # Secure default
        **kwargs,  # Allow overriding defaults
    }
    return subprocess.run(  # noqa: S603,PLW1510  # Uses validated full executable path, no shell=True, check is in run_kwargs
        [git_path, *args], cwd=cwd, **run_kwargs
    )


def run_cargo_command(
    args: list[str],
    cwd: Path | None = None,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    """
    Run a cargo command securely using full executable path.

    Args:
        args: Cargo command arguments (without 'cargo' prefix)
        cwd: Working directory for the command
        **kwargs: Additional arguments passed to subprocess.run
                 (e.g., capture_output=True, text=True, check=True, timeout=60)

    Returns:
        CompletedProcess result

    Raises:
        ExecutableNotFoundError: If cargo is not found
        subprocess.CalledProcessError: If command fails and check=True
        subprocess.TimeoutExpired: If command times out
    """
    cargo_path = get_safe_executable("cargo")
    # Set secure defaults for subprocess.run
    run_kwargs = {
        "capture_output": True,
        "text": True,
        "check": True,  # Secure default
        **kwargs,  # Allow overriding defaults
    }
    return subprocess.run(  # noqa: S603,PLW1510  # Uses validated full executable path, no shell=True, check is in run_kwargs
        [cargo_path, *args], cwd=cwd, **run_kwargs
    )


def run_safe_command(command: str, args: list[str], cwd: Path | None = None, **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """
    Run any command securely using full executable path.

    Args:
        command: Command name to run (e.g., "rustc", "nproc")
        args: Command arguments
        cwd: Working directory for the command
        **kwargs: Additional arguments passed to subprocess.run
                  (e.g., capture_output=True, text=True, check=True)

    Returns:
        CompletedProcess result

    Raises:
        ExecutableNotFoundError: If command is not found
        subprocess.CalledProcessError: If command fails and check=True
    """
    command_path = get_safe_executable(command)
    # Set secure defaults for subprocess.run
    run_kwargs = {
        "capture_output": True,
        "text": True,
        "check": True,  # Secure default
        **kwargs,  # Allow overriding defaults
    }
    return subprocess.run(  # noqa: S603,PLW1510  # Uses validated full executable path, no shell=True, check is in run_kwargs
        [command_path, *args], cwd=cwd, **run_kwargs
    )


# Convenience functions for commonly used git commands
def get_git_commit_hash() -> str:
    """
    Get the current git commit hash.

    Returns:
        Current commit hash

    Raises:
        ExecutableNotFoundError: If git is not found
        subprocess.CalledProcessError: If git command fails
    """
    result = run_git_command(["rev-parse", "HEAD"])
    return result.stdout.strip()


def get_git_remote_url(remote: str = "origin") -> str:
    """
    Get the URL of a git remote.

    Args:
        remote: Remote name (default: "origin")

    Returns:
        Remote URL

    Raises:
        ExecutableNotFoundError: If git is not found
        subprocess.CalledProcessError: If git command fails
    """
    result = run_git_command(["remote", "get-url", remote])
    return result.stdout.strip()


def check_git_repo() -> bool:
    """
    Check if current directory is in a git repository.

    Returns:
        True if in a git repository, False otherwise
    """
    try:
        run_git_command(["rev-parse", "--git-dir"])
        return True
    except (ExecutableNotFoundError, subprocess.CalledProcessError):
        return False


def check_git_history() -> bool:
    """
    Check if git repository has commit history.

    Returns:
        True if git history exists, False otherwise
    """
    try:
        run_git_command(["log", "--oneline", "-n", "1"])
        return True
    except (ExecutableNotFoundError, subprocess.CalledProcessError):
        return False


def run_git_command_with_input(
    args: list[str],
    input_data: str,
    cwd: Path | None = None,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    """
    Run a git command securely with stdin input using full executable path.

    Args:
        args: Git command arguments (without 'git' prefix)
        input_data: Data to send to stdin
        cwd: Working directory for the command
        **kwargs: Additional arguments passed to subprocess.run
                 (e.g., text=True, check=True, timeout=60)

    Returns:
        CompletedProcess result

    Raises:
        ExecutableNotFoundError: If git is not found
        subprocess.CalledProcessError: If command fails and check=True
        subprocess.TimeoutExpired: If command times out

    Note:
        When text=True (default), output uses locale encoding. For deterministic UTF-8
        in CI environments, consider passing encoding="utf-8" via kwargs.
    """
    git_path = get_safe_executable("git")
    # Set secure defaults for subprocess.run
    run_kwargs = {
        "capture_output": True,
        "text": True,
        "check": True,  # Secure default
        **kwargs,  # Allow overriding defaults
    }
    return subprocess.run(  # noqa: S603,PLW1510  # Uses validated full executable path, no shell=True, check is in run_kwargs
        [git_path, *args], cwd=cwd, input=input_data, **run_kwargs
    )
