"""
Comprehensive tests for changelog_utils.py functionality.

Tests include temporary git tag operations, markdown escaping,
error handling, and changelog generation workflows.
"""

import json
import shutil
import subprocess

# Import the module under test
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest

from changelog_utils import ChangelogError, ChangelogUtils, GitRepoError, VersionError


class TestChangelogUtils:
    """Test suite for ChangelogUtils class."""

    def setup_method(self, method):  # noqa: ARG002
        """Set up test fixtures."""
        self.temp_dir = None

    def teardown_method(self, method):  # noqa: ARG002
        """Clean up after tests."""
        if self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_escape_markdown(self):
        """Test markdown character escaping."""
        test_cases = [
            # Basic escaping
            ("Simple text", "Simple text"),
            ("Text with *bold*", "Text with \\*bold\\*"),
            ("Text with _italic_", "Text with \\_italic\\_"),
            ("Text with `code`", "Text with \\`code\\`"),
            ("Text with [link]", "Text with \\[link\\]"),
            ("Text with \\backslash", "Text with \\\\backslash"),
            # Multiple characters
            ("*Bold* and _italic_ and `code`", "\\*Bold\\* and \\_italic\\_ and \\`code\\`"),
            ("[Link](url) with *emphasis*", "\\[Link\\](url) with \\*emphasis\\*"),
            # Edge cases
            ("", ""),
            ("No special chars", "No special chars"),
            ("***", "\\*\\*\\*"),
            ("___", "\\_\\_\\_"),
            ("```", "\\`\\`\\`"),
            ("[[[]]]", "\\[\\[\\[\\]\\]\\]"),
        ]

        for input_text, expected in test_cases:
            result = ChangelogUtils.escape_markdown(input_text)
            assert result == expected, f"Failed for input: {input_text!r}"

    def test_validate_semver(self):
        """Test semantic version validation."""
        # Valid versions
        valid_versions = [
            "v1.0.0",
            "v0.1.0",
            "v10.20.30",
            "v1.0.0-alpha",
            "v1.0.0-beta.1",
            "v2.0.0-rc.1+build.123",
        ]

        for version in valid_versions:
            # Should not raise any exception
            ChangelogUtils.validate_semver(version)

        # Invalid versions
        invalid_versions = [
            "1.0.0",  # Missing 'v' prefix
            "v1.0",  # Missing patch version
            "v1",  # Missing minor and patch
            "vx.y.z",  # Non-numeric components
            "v1.0.0.0",  # Too many components
            "",  # Empty string
            "random-text",  # Not a version at all
        ]

        for version in invalid_versions:
            with pytest.raises(VersionError):
                ChangelogUtils.validate_semver(version)

    @patch("changelog_utils._check_git_repo")
    def test_validate_git_repo_success(self, mock_check_git_repo):
        """Test successful git repository validation."""
        mock_check_git_repo.return_value = True

        # Should not raise any exception
        ChangelogUtils.validate_git_repo()
        mock_check_git_repo.assert_called_once()

    @patch("changelog_utils._check_git_repo")
    def test_validate_git_repo_failure(self, mock_check_git_repo):
        """Test git repository validation failure."""
        mock_check_git_repo.return_value = False

        with pytest.raises(GitRepoError) as cm:
            ChangelogUtils.validate_git_repo()

        assert "not in a git repository" in str(cm.value).lower()

    @patch("changelog_utils._check_git_history")
    def test_check_git_history_success(self, mock_check_git_history):
        """Test successful git history check."""
        mock_check_git_history.return_value = True

        # Should not raise any exception
        ChangelogUtils.check_git_history()
        mock_check_git_history.assert_called_once()

    @patch("changelog_utils._check_git_history")
    def test_check_git_history_failure(self, mock_check_git_history):
        """Test git history check failure."""
        mock_check_git_history.return_value = False

        with pytest.raises(GitRepoError) as cm:
            ChangelogUtils.check_git_history()

        assert "no git history" in str(cm.value).lower()

    def test_get_markdown_line_limit_with_config(self):
        """Test markdown line limit extraction from config file."""
        config_content = json.dumps({"MD013": {"line_length": 120}})

        # Mock the Path class and its methods
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.open.return_value.__enter__.return_value.read.return_value = config_content

        with patch("changelog_utils.Path", return_value=mock_path), patch("json.load", return_value={"MD013": {"line_length": 120}}):
            limit = ChangelogUtils.get_markdown_line_limit()
            assert limit == 120

    @patch("pathlib.Path.exists", return_value=False)
    def test_get_markdown_line_limit_no_config(self, mock_exists):  # noqa: ARG002
        """Test markdown line limit default when no config file."""
        limit = ChangelogUtils.get_markdown_line_limit()
        assert limit == 160  # Default value

    @patch("builtins.open", new_callable=mock_open)
    def test_get_markdown_line_limit_invalid_config(self, mock_file):
        """Test markdown line limit with invalid JSON config."""
        mock_file.return_value.read.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

        with patch("pathlib.Path.exists", return_value=True):
            limit = ChangelogUtils.get_markdown_line_limit()
            assert limit == 160  # Should fall back to default

    def test_wrap_markdown_line(self):
        """Test markdown line wrapping functionality."""
        # Short line - no wrapping needed
        result = ChangelogUtils.wrap_markdown_line("Short text", 80, "  ")
        assert result == ["  Short text"]

        # Long line - needs wrapping
        long_text = "This is a very long line that needs to be wrapped because it exceeds the maximum length"
        result = ChangelogUtils.wrap_markdown_line(long_text, 50, "  ")

        # Should return multiple lines
        assert len(result) > 1
        # Each line should start with indent
        for line in result:
            assert line.startswith("  ")
        # No line should exceed max length
        for line in result:
            assert len(line) <= 50

    def test_escape_version_for_regex(self):
        """Test version string escaping for regex use."""
        test_cases = [
            ("v1.0.0", "v1\\.0\\.0"),
            ("v2.1.0-beta.1", "v2\\.1\\.0\\-beta\\.1"),
            ("v1.0.0+build.123", "v1\\.0\\.0\\+build\\.123"),
        ]

        for version, expected in test_cases:
            result = ChangelogUtils.escape_version_for_regex(version)
            assert result == expected, f"Failed for version: {version!r}"

    @patch("pathlib.Path.cwd")
    def test_get_project_root_current_dir(self, mock_cwd):
        """Test project root detection when in project root."""
        mock_dir = MagicMock()
        mock_cwd.return_value = mock_dir

        # Mock CHANGELOG.md exists in current directory
        changelog_path = MagicMock()
        changelog_path.exists.return_value = True
        mock_dir.__truediv__.return_value = changelog_path

        result = ChangelogUtils.get_project_root()
        assert result == str(mock_dir)

    @patch("pathlib.Path.cwd")
    def test_get_project_root_parent_dir(self, mock_cwd):
        """Test project root detection when in subdirectory."""
        mock_dir = MagicMock()
        mock_cwd.return_value = mock_dir

        # Mock CHANGELOG.md doesn't exist in current dir but exists in parent
        changelog_current = MagicMock()
        changelog_current.exists.return_value = False
        changelog_parent = MagicMock()
        changelog_parent.exists.return_value = True

        mock_dir.__truediv__.return_value = changelog_current
        mock_dir.parent.__truediv__.return_value = changelog_parent

        result = ChangelogUtils.get_project_root()
        assert result == str(mock_dir.parent)

    @patch("pathlib.Path.cwd")
    def test_get_project_root_not_found(self, mock_cwd):
        """Test project root detection failure."""
        mock_dir = MagicMock()
        mock_cwd.return_value = mock_dir

        # Mock CHANGELOG.md doesn't exist anywhere
        changelog_mock = MagicMock()
        changelog_mock.exists.return_value = False
        mock_dir.__truediv__.return_value = changelog_mock
        mock_dir.parent.__truediv__.return_value = changelog_mock

        with pytest.raises(ChangelogError) as cm:
            ChangelogUtils.get_project_root()

        assert "Cannot determine project root" in str(cm.value)


class TestChangelogUtilsWithGitOperations:
    """Test suite for changelog utils that require git operations."""

    def setup_method(self, method):  # noqa: ARG002
        """Set up temporary git repository for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = Path.cwd()

        # Change to temp directory and initialize git repo
        import os

        os.chdir(self.temp_dir)

        # Initialize git repo with initial commit
        subprocess.run(["git", "init"], check=True, capture_output=True)  # noqa: S607
        subprocess.run(["git", "config", "user.name", "Test User"], check=True)  # noqa: S607
        subprocess.run(["git", "config", "user.email", "test@example.com"], check=True)  # noqa: S607

        # Create initial commit
        Path("README.md").write_text("# Test Repo\n")
        subprocess.run(["git", "add", "README.md"], check=True)  # noqa: S607
        subprocess.run(["git", "commit", "-m", "Initial commit"], check=True)  # noqa: S607

    def teardown_method(self, method):  # noqa: ARG002
        """Clean up temporary git repository."""
        import os

        os.chdir(self.original_cwd)
        if self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_temporary_git_tag_operations(self):
        """Test git tag creation and cleanup with temporary tags."""
        test_tag = "v0.1.0-test"

        try:
            # Verify tag doesn't exist initially
            result = subprocess.run(["git", "tag", "-l", test_tag], check=False, capture_output=True, text=True)  # noqa: S603, S607
            assert result.stdout.strip() == ""

            # Create temporary tag
            subprocess.run(["git", "tag", "-a", test_tag, "-m", "Test tag"], check=True)  # noqa: S603, S607

            # Verify tag exists
            result = subprocess.run(["git", "tag", "-l", test_tag], check=False, capture_output=True, text=True)  # noqa: S603, S607
            assert result.stdout.strip() == test_tag

            # Test tag validation would work
            ChangelogUtils.validate_semver(test_tag)

        finally:
            # Clean up - delete the temporary tag
            subprocess.run(["git", "tag", "-d", test_tag], check=True, capture_output=True)  # noqa: S603, S607

            # Verify tag is deleted
            result = subprocess.run(["git", "tag", "-l", test_tag], check=False, capture_output=True, text=True)  # noqa: S603, S607
            assert result.stdout.strip() == ""

    def test_git_repository_url_normalization(self):
        """Test repository URL normalization from various formats."""
        test_cases = [
            ("git@github.com:owner/repo.git", "https://github.com/owner/repo"),
            ("https://github.com/owner/repo.git", "https://github.com/owner/repo"),
            ("https://github.com/owner/repo", "https://github.com/owner/repo"),
            ("ssh://git@github.com/owner/repo.git", "https://github.com/owner/repo"),
        ]

        for input_url, expected_url in test_cases:
            # Add a test remote
            subprocess.run(["git", "remote", "add", "test-origin", input_url], check=True, capture_output=True)  # noqa: S603, S607

            try:
                # Mock the get_git_remote_url to return our test URL
                with patch("changelog_utils.get_git_remote_url", return_value=input_url):
                    result = ChangelogUtils.get_repository_url()
                    assert result == expected_url, f"Failed for URL: {input_url!r}"
            finally:
                # Clean up remote
                subprocess.run(["git", "remote", "remove", "test-origin"], check=True, capture_output=True)  # noqa: S607

    def test_commit_processing_with_test_commits(self):
        """Test commit processing with specially crafted test commits."""
        # Create a commit with special characters in the title
        test_content = "Test content with *bold* and _italic_ and `code`"
        test_file = Path("test.txt")
        test_file.write_text(test_content)

        subprocess.run(["git", "add", "test.txt"], check=True)  # noqa: S607
        commit_msg = "feat: Add *special* formatting and _emphasis_ with `code`"
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)  # noqa: S603, S607

        # Get the commit SHA to verify git operations work
        subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)  # noqa: S607

        # Test that markdown escaping would work on this commit
        escaped_title = ChangelogUtils.escape_markdown(commit_msg)

        # Verify special characters are escaped
        assert "\\*special\\*" in escaped_title
        assert "\\_emphasis\\_" in escaped_title
        assert "\\`code\\`" in escaped_title


class TestChangelogUtilsErrorHandling:
    """Test suite for changelog utils error handling."""

    def test_version_error_inheritance(self):
        """Test that VersionError inherits from ChangelogError."""
        error = VersionError("Test version error")
        assert isinstance(error, ChangelogError)
        assert str(error) == "Test version error"

    def test_git_repo_error_inheritance(self):
        """Test that GitRepoError inherits from ChangelogError."""
        error = GitRepoError("Test git error")
        assert isinstance(error, ChangelogError)
        assert str(error) == "Test git error"

    def test_changelog_error_basic(self):
        """Test basic ChangelogError functionality."""
        error = ChangelogError("Test changelog error")
        assert str(error) == "Test changelog error"

    @patch("changelog_utils.get_git_remote_url")
    def test_repository_url_error_handling(self, mock_get_git_remote):
        """Test repository URL extraction error handling."""
        mock_get_git_remote.side_effect = Exception("Git command failed")

        with pytest.raises(GitRepoError) as cm:
            ChangelogUtils.get_repository_url()

        assert "Could not detect git remote origin URL" in str(cm.value)

    def test_invalid_repository_url_format(self):
        """Test handling of invalid repository URL formats."""
        invalid_urls = [
            "not-a-url",
            "ftp://invalid.com/repo",
            "https://notgithub.com/owner/repo",
        ]

        for invalid_url in invalid_urls:
            with patch("changelog_utils.get_git_remote_url", return_value=invalid_url):
                with pytest.raises(GitRepoError) as cm:
                    ChangelogUtils.get_repository_url()

                assert "Unsupported git remote URL" in str(cm.value), f"Failed for URL: {invalid_url!r}"

    def test_empty_repository_url(self):
        """Test handling of empty repository URL."""
        with patch("changelog_utils.get_git_remote_url", return_value=""):
            with pytest.raises(GitRepoError) as cm:
                ChangelogUtils.get_repository_url()

            assert "Git remote origin URL is empty" in str(cm.value)


if __name__ == "__main__":
    # Run the tests with pytest
    pytest.main([__file__, "-v"])
