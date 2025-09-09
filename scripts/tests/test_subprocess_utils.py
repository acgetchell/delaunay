#!/usr/bin/env python3
"""
Comprehensive tests for subprocess_utils.py

These tests ensure the security utilities function correctly and maintain
their secure-by-default behavior while providing flexibility through kwargs.
"""

import subprocess
import sys
from pathlib import Path

import pytest

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from subprocess_utils import (
    ExecutableNotFoundError,
    check_git_history,
    check_git_repo,
    get_git_commit_hash,
    get_git_remote_url,
    get_safe_executable,
    run_cargo_command,
    run_git_command,
    run_safe_command,
)


class TestGetSafeExecutable:
    """Test get_safe_executable function."""

    def test_finds_existing_executable(self):
        """Test that it finds common executables."""
        # Test with common unix command
        result = get_safe_executable("echo")
        assert isinstance(result, str)
        assert len(result) > 0
        assert result.endswith("echo") or "echo" in result

    def test_finds_git_executable(self):
        """Test that it finds git (required for project)."""
        git_path = get_safe_executable("git")
        assert isinstance(git_path, str)
        assert len(git_path) > 0
        # Should be an absolute path
        assert git_path.startswith("/")

    def test_raises_on_nonexistent_executable(self):
        """Test that it raises ExecutableNotFoundError for nonexistent commands."""
        with pytest.raises(ExecutableNotFoundError, match="not found in PATH"):
            get_safe_executable("definitely-nonexistent-command-xyz")

    def test_error_message_contains_command_name(self):
        """Test that error message contains the command name."""
        fake_command = "fake-command-for-testing"
        with pytest.raises(ExecutableNotFoundError) as exc_info:
            get_safe_executable(fake_command)
        assert fake_command in str(exc_info.value)


class TestRunGitCommand:
    """Test run_git_command function."""

    def test_git_version(self):
        """Test basic git command execution."""
        result = run_git_command(["--version"])
        assert result.returncode == 0
        assert "git version" in result.stdout.lower()
        assert isinstance(result.stdout, str)

    def test_git_command_with_custom_params(self):
        """Test git command with custom parameters."""
        result = run_git_command(["status", "--porcelain"], check=False)
        # Should not raise even if there are changes (check=False)
        assert isinstance(result.returncode, int)
        assert isinstance(result.stdout, str)

    def test_git_command_failure_handling(self):
        """Test that failed git commands raise CalledProcessError when check=True."""
        with pytest.raises(subprocess.CalledProcessError):
            run_git_command(["invalid-git-subcommand-xyz"], check=True)

    def test_git_command_no_failure_with_check_false(self):
        """Test that failed git commands don't raise when check=False."""
        result = run_git_command(["invalid-git-subcommand-xyz"], check=False)
        assert result.returncode != 0
        assert isinstance(result.stdout, str)


class TestRunCargoCommand:
    """Test run_cargo_command function."""

    def test_cargo_version(self):
        """Test basic cargo command execution."""
        result = run_cargo_command(["--version"])
        assert result.returncode == 0
        assert "cargo" in result.stdout.lower()
        assert isinstance(result.stdout, str)

    def test_cargo_command_with_custom_params(self):
        """Test cargo command with custom parameters."""
        result = run_cargo_command(["check", "--dry-run"], check=False)
        assert isinstance(result.returncode, int)
        assert isinstance(result.stdout, str)


class TestRunSafeCommand:
    """Test run_safe_command function with various scenarios."""

    def test_basic_command_execution(self):
        """Test basic command execution with default parameters."""
        result = run_safe_command("echo", ["hello world"])
        assert result.returncode == 0
        assert result.stdout.strip() == "hello world"
        assert isinstance(result.stdout, str)

    def test_secure_defaults_are_applied(self):
        """Test that secure defaults are applied."""
        result = run_safe_command("echo", ["test"])
        # Should use secure defaults:
        # - capture_output=True (stdout captured)
        # - text=True (string output)
        # - check=True (would raise on failure)
        assert isinstance(result.stdout, str)
        assert result.stdout.strip() == "test"

    def test_text_parameter_enforced(self):
        """Test that text parameter is enforced for security/stability."""
        # run_safe_command enforces text=True for stable CompletedProcess[str] typing
        result = run_safe_command("echo", ["test output"], text=False)  # text=False is ignored
        assert isinstance(result.stdout, str)  # Should still be string
        assert "test output" in result.stdout

    def test_custom_check_parameter(self):
        """Test overriding check parameter."""
        # Command that will fail
        result = run_safe_command("ls", ["/definitely-nonexistent-directory"], check=False)
        assert result.returncode != 0
        # Should not raise because check=False

    def test_custom_capture_output_parameter(self):
        """Test overriding capture_output parameter."""
        result = run_safe_command("echo", ["no capture"], capture_output=False)
        # When capture_output=False, stdout should be None
        assert result.stdout is None

    def test_multiple_custom_parameters(self):
        """Test multiple custom parameters at once (text is enforced)."""
        result = run_safe_command("echo", ["multi param test"], text=False, check=False, capture_output=True)
        assert isinstance(result.stdout, str)  # text=False is ignored, still returns string
        assert result.returncode == 0
        assert "multi param test" in result.stdout

    def test_nonexistent_command_raises_error(self):
        """Test that nonexistent commands raise ExecutableNotFoundError."""
        with pytest.raises(ExecutableNotFoundError):
            run_safe_command("definitely-nonexistent-command", ["arg"])

    def test_additional_kwargs_passed_through(self):
        """Test that additional kwargs are passed through to subprocess.run."""
        # Test with timeout (a subprocess.run parameter not explicitly handled)
        result = run_safe_command("echo", ["timeout test"], timeout=10)
        assert result.returncode == 0
        assert "timeout test" in result.stdout


class TestGitRepositoryFunctions:
    """Test git repository detection functions."""

    def test_check_git_repo_in_git_repo(self):
        """Test check_git_repo returns True when in a git repository."""
        # Assuming tests are run from within the git repository
        assert check_git_repo() is True

    def test_check_git_history_with_history(self):
        """Test check_git_history returns True when git history exists."""
        # Assuming the repository has commit history
        assert check_git_history() is True

    def test_get_git_commit_hash_returns_hash(self):
        """Test that get_git_commit_hash returns a valid commit hash."""
        commit_hash = get_git_commit_hash()
        assert isinstance(commit_hash, str)
        assert len(commit_hash) >= 7  # At least short hash length
        # Should be hexadecimal
        assert all(c in "0123456789abcdef" for c in commit_hash.lower())

    def test_get_git_remote_url_returns_url(self):
        """Test that get_git_remote_url returns a valid URL."""
        remote_url = get_git_remote_url("origin")
        assert isinstance(remote_url, str)
        assert len(remote_url) > 0
        # Should be a git URL (https or git@)
        assert any(remote_url.startswith(prefix) for prefix in ["https://", "git@", "ssh://"])


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_executable_not_found_error_attributes(self):
        """Test ExecutableNotFoundError has proper attributes."""
        error = ExecutableNotFoundError("test message")
        assert str(error) == "test message"
        assert isinstance(error, Exception)

    def test_git_functions_handle_missing_git(self, monkeypatch):
        """Test git functions handle missing git executable gracefully."""

        # Mock get_safe_executable to raise ExecutableNotFoundError for git
        def mock_get_safe_executable(command):
            if command == "git":
                raise ExecutableNotFoundError(f"Required executable '{command}' not found in PATH")
            return "/bin/echo"  # Return echo for other commands

        monkeypatch.setattr("subprocess_utils.get_safe_executable", mock_get_safe_executable)

        # These should return False when git is not available
        assert check_git_repo() is False
        assert check_git_history() is False

        # These should raise ExecutableNotFoundError
        with pytest.raises(ExecutableNotFoundError):
            get_git_commit_hash()

        with pytest.raises(ExecutableNotFoundError):
            get_git_remote_url()


class TestSecurityFeatures:
    """Test security-related features of the utilities."""

    def test_uses_full_executable_paths(self):
        """Test that commands use full executable paths."""
        # This is implicitly tested by get_safe_executable tests,
        # but let's verify the behavior
        git_path = get_safe_executable("git")
        assert git_path.startswith("/")  # Should be absolute path
        assert "git" in git_path

    def test_no_shell_execution(self):
        """Test that commands don't use shell=True."""
        # The functions should not use shell=True, which would be a security risk
        # We can't directly test this, but the implementation uses subprocess.run
        # with a list of arguments, which is secure
        result = run_safe_command("echo", ["$HOME"])  # Should not expand $HOME
        # If shell=True was used, this would expand the environment variable
        assert result.stdout.strip() == "$HOME"

    def test_check_parameter_security_default(self):
        """Test that check=True is the default for security."""
        # Command that will fail should raise by default
        with pytest.raises(subprocess.CalledProcessError):
            run_safe_command("ls", ["/definitely-nonexistent-directory"])

    def test_run_git_command_rejects_executable_override(self):
        """Test that run_git_command raises ValueError when executable is overridden."""
        with pytest.raises(ValueError, match="Overriding 'executable' is not allowed"):
            run_git_command(["status"], executable="/malicious/fake/git")

    def test_run_cargo_command_rejects_executable_override(self):
        """Test that run_cargo_command raises ValueError when executable is overridden."""
        with pytest.raises(ValueError, match="Overriding 'executable' is not allowed"):
            run_cargo_command(["--version"], executable="/malicious/fake/cargo")

    def test_run_safe_command_rejects_executable_override(self):
        """Test that run_safe_command raises ValueError when executable is overridden."""
        with pytest.raises(ValueError, match="Overriding 'executable' is not allowed"):
            run_safe_command("echo", ["test"], executable="/malicious/fake/command")

    def test_run_git_command_with_input_rejects_executable_override(self):
        """Test that run_git_command_with_input raises ValueError when executable is overridden."""
        from subprocess_utils import run_git_command_with_input

        with pytest.raises(ValueError, match="Overriding 'executable' is not allowed"):
            run_git_command_with_input(["hash-object", "--stdin"], "test content", executable="/malicious/fake/git")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
