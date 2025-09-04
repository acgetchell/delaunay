"""
Comprehensive tests for changelog_utils.py functionality.

Tests include temporary git tag operations, markdown escaping,
error handling, and changelog generation workflows.
"""

import json
import os
import shutil

# Import the module under test
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest

from changelog_utils import ChangelogError, ChangelogUtils, GitRepoError, VersionError
from subprocess_utils import run_git_command


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

    @patch("changelog_utils.ChangelogUtils.find_changelog_path")
    def test_get_project_root_current_dir(self, mock_find_changelog):
        """Test project root detection when in project root."""
        # Mock find_changelog_path to return a path in current directory
        mock_changelog_path = "/test/project/CHANGELOG.md"
        mock_find_changelog.return_value = mock_changelog_path

        result = ChangelogUtils.get_project_root()
        assert result == "/test/project"

    @patch("changelog_utils.ChangelogUtils.find_changelog_path")
    def test_get_project_root_parent_dir(self, mock_find_changelog):
        """Test project root detection when in subdirectory."""
        # Mock find_changelog_path to return a path in parent directory
        mock_changelog_path = "/test/project/CHANGELOG.md"
        mock_find_changelog.return_value = mock_changelog_path

        result = ChangelogUtils.get_project_root()
        assert result == "/test/project"

    @patch("changelog_utils.ChangelogUtils.find_changelog_path")
    def test_get_project_root_not_found(self, mock_find_changelog):
        """Test project root detection failure."""
        # Mock find_changelog_path to raise ChangelogNotFoundError
        from changelog_utils import ChangelogNotFoundError

        mock_find_changelog.side_effect = ChangelogNotFoundError("CHANGELOG.md not found")

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
        os.chdir(self.temp_dir)

        # Initialize git repo with initial commit
        run_git_command(["init"])
        run_git_command(["config", "user.name", "Test User"])
        run_git_command(["config", "user.email", "test@example.com"])

        # Create initial commit
        Path("README.md").write_text("# Test Repo\n")
        run_git_command(["add", "README.md"])
        run_git_command(["commit", "-m", "Initial commit"])

    def teardown_method(self, method):  # noqa: ARG002
        """Clean up temporary git repository."""
        os.chdir(self.original_cwd)
        if self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_temporary_git_tag_operations(self):
        """Test git tag creation and cleanup with temporary tags."""
        test_tag = "v0.1.0-test"

        try:
            # Verify tag doesn't exist initially
            result = run_git_command(["tag", "-l", test_tag], check=False)
            assert result.stdout.strip() == ""

            # Create temporary tag
            run_git_command(["tag", "-a", test_tag, "-m", "Test tag"])

            # Verify tag exists
            result = run_git_command(["tag", "-l", test_tag], check=False)
            assert result.stdout.strip() == test_tag

            # Test tag validation would work
            ChangelogUtils.validate_semver(test_tag)

        finally:
            # Clean up - delete the temporary tag
            run_git_command(["tag", "-d", test_tag])

            # Verify tag is deleted
            result = run_git_command(["tag", "-l", test_tag], check=False)
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
            run_git_command(["remote", "add", "test-origin", input_url])

            try:
                # Mock the get_git_remote_url to return our test URL
                with patch("changelog_utils.get_git_remote_url", return_value=input_url):
                    result = ChangelogUtils.get_repository_url()
                    assert result == expected_url, f"Failed for URL: {input_url!r}"
            finally:
                # Clean up remote
                run_git_command(["remote", "remove", "test-origin"])

    def test_commit_processing_with_test_commits(self):
        """Test commit processing with specially crafted test commits."""
        # Create a commit with special characters in the title
        test_content = "Test content with *bold* and _italic_ and `code`"
        test_file = Path("test.txt")
        test_file.write_text(test_content)

        run_git_command(["add", "test.txt"])
        commit_msg = "feat: Add *special* formatting and _emphasis_ with `code`"
        run_git_command(["commit", "-m", commit_msg])

        # Get the commit SHA to verify git operations work
        run_git_command(["rev-parse", "HEAD"])

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


class TestChangelogTitleFormatting:
    """Test suite for changelog title formatting functionality."""

    def test_format_entry_title_short_title(self):
        """Test formatting of short titles that fit in one line."""
        title = "Add new feature"
        commit_sha = "abc123f"
        repo_url = "https://github.com/owner/repo"
        max_line_length = 160

        result = ChangelogUtils._format_entry_title(title, commit_sha, repo_url, max_line_length)

        # Should return a single line
        assert len(result) == 1
        expected = f"- **{title}** [`{commit_sha}`]({repo_url}/commit/{commit_sha})"
        assert result[0] == expected
        # Verify it doesn't exceed line length
        assert len(result[0]) <= max_line_length

    def test_format_entry_title_long_title_short_limit(self):
        """Test formatting of long titles with short line limit."""
        title = "Add extremely long feature that does many things and has a very long descriptive title"
        commit_sha = "abc123f"
        repo_url = "https://github.com/owner/repo"
        max_line_length = 50

        result = ChangelogUtils._format_entry_title(title, commit_sha, repo_url, max_line_length)

        # Should wrap the title into multiple lines
        assert len(result) > 1

        # All lines should respect the length limit
        for line in result:
            assert len(line) <= max_line_length, f"Line too long: {line!r} (length: {len(line)})"

        # First line should start with "- **" and end with "**"
        assert result[0].startswith("- **")
        assert result[0].endswith("**")

        # Continuation lines should start with "  **" and end with "**"
        for line in result[1:-1]:  # Exclude last line (commit link)
            if line.startswith("  ["):  # Skip commit link line
                continue
            assert line.startswith("  **")
            assert line.endswith("**")

        # Last line(s) should be the commit link (may be split for very short limits)
        commit_link_full = f"  [`{commit_sha}`]({repo_url}/commit/{commit_sha})"
        if len(commit_link_full) <= max_line_length:
            # Single commit link line
            assert result[-1] == commit_link_full
        else:
            # Split commit link
            assert result[-2] == f"  [`{commit_sha}`]"
            assert result[-1] == f"  ({repo_url}/commit/{commit_sha})"

    def test_format_entry_title_markdown_escaping(self):
        """Test that markdown characters are properly escaped in wrapped titles."""
        title = "Fix *bold* _italic_ `code` [link] \\ characters that need escaping in a very long title"
        commit_sha = "def456a"
        repo_url = "https://github.com/owner/repo"
        max_line_length = 60

        result = ChangelogUtils._format_entry_title(title, commit_sha, repo_url, max_line_length)

        # Should wrap the title
        assert len(result) > 1

        # All lines should respect length limit
        for line in result:
            assert len(line) <= max_line_length

        # Check that markdown characters are escaped in the wrapped content
        title_content = "".join(result[:-1])  # Exclude commit link
        assert "\\*bold\\*" in title_content
        assert "\\_italic\\_" in title_content
        assert "\\`code\\`" in title_content
        assert "\\[link\\]" in title_content
        assert "\\\\" in title_content  # Backslash should be escaped

    def test_format_entry_title_edge_cases(self):
        """Test edge cases for title formatting."""
        commit_sha = "xyz789b"
        repo_url = "https://github.com/owner/repo"

        # Test reasonably short line limit (shorter than normal but not extreme)
        title = "Short title"
        result = ChangelogUtils._format_entry_title(title, commit_sha, repo_url, 60)

        # Should still produce valid output
        assert len(result) >= 1
        for line in result:
            assert len(line) <= 60, f"Line too long ({len(line)} > 60): {line!r}"

        # Test empty title
        title = ""
        result = ChangelogUtils._format_entry_title(title, commit_sha, repo_url, 160)
        assert len(result) >= 1

        # Test title with only spaces
        title = "   "
        result = ChangelogUtils._format_entry_title(title, commit_sha, repo_url, 160)
        assert len(result) >= 1

    def test_format_entry_title_regression_long_line(self):
        """Regression test for the specific long line issue found in CHANGELOG.md."""
        # This is the actual long title that caused the markdown lint failure
        title = "Moves the test_alloc_api.rs and test_circumsphere.rs examples to the tests/ directory and renames them to allocation_api.rs and circumsphere_debug_tools.rs, respectively, to reflect their role as debugging and testing utilities."
        commit_sha = "f10aba3"
        repo_url = "https://github.com/acgetchell/delaunay"
        max_line_length = 160  # From .markdownlint.json

        result = ChangelogUtils._format_entry_title(title, commit_sha, repo_url, max_line_length)

        # Should produce multiple lines due to length
        assert len(result) > 1

        # Critical: ALL lines must respect the markdown line length limit
        for i, line in enumerate(result):
            assert len(line) <= max_line_length, f"Line {i} too long ({len(line)} > {max_line_length}): {line!r}"

        # Should preserve markdown formatting structure
        assert result[0].startswith("- **")
        # Last line should be commit-related (may be SHA or URL depending on wrapping)
        assert (
            result[-1].startswith("  [")  # Full commit link
            or result[-1].startswith("  (")
        )  # URL part of split commit link

        # Verify the title content is preserved across lines (minus escaping)
        title_lines_content = []
        for line in result[:-1]:  # Exclude commit link
            if line.startswith("- **"):
                title_lines_content.append(line[4:-2])  # Remove "- **" and "**"
            elif line.startswith("  **"):
                title_lines_content.append(line[4:-2])  # Remove "  **" and "**"

        reconstructed_title = "".join(title_lines_content)
        # Should contain the key parts of the original title (allowing for escaping)
        assert "test_alloc_api.rs" in reconstructed_title or "test\\_alloc\\_api.rs" in reconstructed_title
        assert "circumsphere" in reconstructed_title
        assert "allocation_api.rs" in reconstructed_title or "allocation\\_api.rs" in reconstructed_title

    def test_format_entry_title_typical_github_length(self):
        """Test with typical GitHub commit title lengths."""
        # Test various realistic title lengths
        test_cases = [
            ("feat: Add new API endpoint", 160, 1),  # Short - single line
            ("fix: Resolve issue with long database query timeout handling", 160, 1),  # Medium - single line
            (
                "refactor: Restructure the authentication middleware to support multiple providers and improve error handling",
                160,
                2,
            ),  # Long - should split
        ]

        commit_sha = "abc123"
        repo_url = "https://github.com/test/repo"

        for title, max_length, expected_min_lines in test_cases:
            result = ChangelogUtils._format_entry_title(title, commit_sha, repo_url, max_length)

            # Should have at least expected minimum lines
            assert len(result) >= expected_min_lines, f"Title '{title}' should produce at least {expected_min_lines} lines"

            # All lines should respect length limit
            for line in result:
                assert len(line) <= max_length, f"Line too long for title '{title}': {line!r}"


if __name__ == "__main__":
    # Run the tests with pytest
    pytest.main([__file__, "-v"])
