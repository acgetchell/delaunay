"""
Comprehensive tests for changelog_utils.py functionality.

Tests include temporary git tag operations, markdown escaping,
error handling, and changelog generation workflows.
"""

import json

# Import the module under test
import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest

from changelog_utils import ChangelogError, ChangelogNotFoundError, ChangelogProcessor, ChangelogUtils, GitRepoError, VersionError
from subprocess_utils import run_git_command


class TestChangelogUtils:
    """Test suite for ChangelogUtils class."""

    @pytest.mark.parametrize(
        ("input_text", "expected"),
        [
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
            ("Text with (parens)", "Text with (parens)"),
            ("Curly {braces}", "Curly {braces}"),
        ],
    )
    def test_escape_markdown(self, input_text, expected):
        """Test markdown character escaping."""
        result = ChangelogUtils.escape_markdown(input_text)
        assert result == expected

    @pytest.mark.parametrize(
        "version",
        [
            "v1.0.0",
            "v0.1.0",
            "v10.20.30",
            "v1.0.0-alpha",
            "v1.0.0-beta.1",
            "v2.0.0-rc.1+build.123",
        ],
    )
    def test_validate_semver_valid(self, version):
        """Test semantic version validation with valid versions."""
        # Should not raise and should return True
        assert ChangelogUtils.validate_semver(version) is True

    @pytest.mark.parametrize(
        "version",
        [
            "1.0.0",  # Missing 'v' prefix
            "v1.0",  # Missing patch version
            "v1",  # Missing minor and patch
            "vx.y.z",  # Non-numeric components
            "v1.0.0.0",  # Too many components
            "v01.2.3",  # Leading zero in MAJOR
            "v1.02.3",  # Leading zero in MINOR
            "v1.2.03",  # Leading zero in PATCH
            "v1.2.3-01",  # Leading zero in pre-release numeric id
            "v1.2.3-rc.01",  # Leading zero in dotted pre-release numeric id
            "",  # Empty string
            "random-text",  # Not a version at all
        ],
    )
    def test_validate_semver_invalid(self, version):
        """Test semantic version validation with invalid versions."""
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

    @patch("changelog_utils.json.load", side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
    def test_get_markdown_line_limit_invalid_config(self, _mock_json_load):  # noqa: PT019
        """Test markdown line limit with invalid JSON config."""
        with patch("changelog_utils.Path.exists", return_value=True), patch("changelog_utils.Path.open", mock_open(read_data="{}")):
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

    @pytest.mark.parametrize(
        ("version", "expected"),
        [
            ("v1.0.0", "v1\\.0\\.0"),
            ("v2.1.0-beta.1", "v2\\.1\\.0\\-beta\\.1"),
            ("v1.0.0+build.123", "v1\\.0\\.0\\+build\\.123"),
        ],
    )
    def test_escape_version_for_regex(self, version, expected):
        """Test version string escaping for regex use."""
        result = ChangelogUtils.escape_version_for_regex(version)
        assert result == expected

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
    @patch("changelog_utils._run_git_command")
    def test_get_project_root_not_found(self, mock_run_git_command, mock_find_changelog):
        """Test project root detection failure when both CHANGELOG.md and git repo are unavailable."""
        # Mock find_changelog_path to raise ChangelogNotFoundError
        mock_find_changelog.side_effect = ChangelogNotFoundError("CHANGELOG.md not found")
        # Mock git command to also fail
        mock_run_git_command.side_effect = Exception("Not a git repository")

        with pytest.raises(ChangelogError) as cm:
            ChangelogUtils.get_project_root()

        assert "Cannot determine project root" in str(cm.value)


@pytest.fixture
def git_repo_fixture(tmp_path: Path, monkeypatch):
    """Fixture for temporary git repository setup (isolated cwd)."""
    monkeypatch.chdir(tmp_path)
    run_git_command(["init"])
    run_git_command(["config", "user.name", "Test User"])
    run_git_command(["config", "user.email", "test@example.com"])
    (tmp_path / "README.md").write_text("# Test Repo\n", encoding="utf-8")
    run_git_command(["add", "README.md"])
    run_git_command(["commit", "-m", "Initial commit"])
    return str(tmp_path)


class TestChangelogUtilsWithGitOperations:
    """Test suite for changelog utils that require git operations."""

    def test_temporary_git_tag_operations(self, git_repo_fixture):
        """Test git tag creation and cleanup with temporary tags."""
        # The fixture sets up a git repo in a temp directory and changes to it
        # We can verify we're in the right place by checking the temp dir
        assert str(git_repo_fixture) in str(Path.cwd())

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

    @pytest.mark.parametrize(
        ("input_url", "expected_url"),
        [
            ("git@github.com:owner/repo.git", "https://github.com/owner/repo"),
            ("https://github.com/owner/repo.git", "https://github.com/owner/repo"),
            ("https://github.com/owner/repo", "https://github.com/owner/repo"),
            ("ssh://git@github.com/owner/repo.git", "https://github.com/owner/repo"),
        ],
    )
    def test_git_repository_url_normalization(self, git_repo_fixture, input_url, expected_url):
        """Test repository URL normalization from various formats."""
        # Verify we're in the git repository set up by the fixture
        assert str(git_repo_fixture) in str(Path.cwd())

        # Mock the get_git_remote_url to return our test URL
        with patch("changelog_utils.get_git_remote_url", return_value=input_url):
            result = ChangelogUtils.get_repository_url()
            assert result == expected_url

    def test_commit_processing_with_test_commits(self, git_repo_fixture):
        """Test commit processing with specially crafted test commits."""
        # Verify we're in the git repository set up by the fixture
        assert str(git_repo_fixture) in str(Path.cwd())

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

    @pytest.mark.parametrize(
        "invalid_url",
        [
            "not-a-url",
            "ftp://invalid.com/repo",
            "https://notgithub.com/owner/repo",
            "ssh://git@gitlab.com/owner/repo.git",  # non-GitHub host
            "git://notgithub.com/owner/repo",  # git protocol to non-GitHub
            "git@github.com/owner/repo",  # scp-like form missing ':'
        ],
    )
    def test_invalid_repository_url_format(self, invalid_url):
        """Test handling of invalid repository URL formats."""
        with patch("changelog_utils.get_git_remote_url", return_value=invalid_url):
            with pytest.raises(GitRepoError) as cm:
                ChangelogUtils.get_repository_url()

            assert "Unsupported git remote URL" in str(cm.value)

    def test_empty_repository_url(self):
        """Test handling of empty repository URL."""
        with patch("changelog_utils.get_git_remote_url", return_value=""):
            with pytest.raises(GitRepoError) as cm:
                ChangelogUtils.get_repository_url()

            assert "Git remote origin URL is empty" in str(cm.value)


class TestURLWrapping:
    """Test suite for URL wrapping behavior in various contexts."""

    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            # Bare URLs should be wrapped
            ("Visit https://example.com for more", "Visit <https://example.com> for more"),
            ("Check http://test.org today", "Check <http://test.org> today"),
            # Markdown links should not be wrapped
            ("[text](https://example.com)", "[text](https://example.com)"),
            ("See [link](http://test.org) here", "See [link](http://test.org) here"),
            # Already wrapped URLs should not be double-wrapped
            ("Visit <https://example.com> today", "Visit <https://example.com> today"),
            # Inline code should preserve URLs
            ("`curl https://example.com`", "`curl https://example.com`"),
            ("Use `wget http://test.org` command", "Use `wget http://test.org` command"),
            ("Run `git clone https://github.com/repo.git`", "Run `git clone https://github.com/repo.git`"),
            # Multiple inline code spans
            ("`code https://one.com` and https://two.com", "`code https://one.com` and <https://two.com>"),
            ("https://one.com and `code https://two.com`", "<https://one.com> and `code https://two.com`"),
            # Code fence lines should be skipped
            ("```", "```"),
            ("```python", "```python"),
            # Line that is entirely a single inline code span
            ("`https://example.com`", "`https://example.com`"),
        ],
    )
    def test_wrap_bare_urls(self, line, expected):
        """Test URL wrapping in various contexts."""
        result = ChangelogUtils.wrap_bare_urls(line)
        assert result == expected

    def test_wrap_bare_urls_indented_code_block(self):
        """Test that indented code blocks are handled properly in _process_body_line."""
        # Indented code blocks (4+ spaces) should not have URLs wrapped
        code_line = "    curl https://example.com/api"
        result = ChangelogUtils._process_body_line(code_line)  # noqa: SLF001
        # Should preserve the line without wrapping the URL
        assert "https://example.com/api" in result
        assert "<https://example.com/api>" not in result

    def test_changelog_processor_fenced_code_block(self, tmp_path):
        """Test that ChangelogProcessor preserves URLs in fenced code blocks."""
        input_file = tmp_path / "input.md"
        output_file = tmp_path / "output.md"

        content = """# Changelog

## v1.0.0

Some text with https://example.com which should be wrapped.

```bash
curl https://api.example.com/data
wget http://files.example.com/file.txt
```

More text with https://another.com to wrap.
"""
        input_file.write_text(content, encoding="utf-8")

        processor = ChangelogProcessor("https://github.com/owner/repo")
        processor.process_file(input_file, output_file)

        result = output_file.read_text(encoding="utf-8")

        # URLs outside code blocks should be wrapped
        assert "<https://example.com>" in result
        assert "<https://another.com>" in result

        # URLs inside code blocks should NOT be wrapped
        assert "curl https://api.example.com/data" in result
        assert "wget http://files.example.com/file.txt" in result
        assert "<https://api.example.com/data>" not in result
        assert "<http://files.example.com/file.txt>" not in result

    def test_changelog_processor_indented_code_block(self, tmp_path):
        """Test that ChangelogProcessor preserves URLs in indented code blocks."""
        input_file = tmp_path / "input.md"
        output_file = tmp_path / "output.md"

        content = """# Changelog

## v1.0.0

Some text with https://example.com which should be wrapped.

    curl https://api.example.com/data
    wget http://files.example.com/file.txt

More text.
"""
        input_file.write_text(content, encoding="utf-8")

        processor = ChangelogProcessor("https://github.com/owner/repo")
        processor.process_file(input_file, output_file)

        result = output_file.read_text(encoding="utf-8")

        # URL outside code block should be wrapped
        assert "<https://example.com>" in result

        # URLs in indented code blocks should NOT be wrapped
        assert "curl https://api.example.com/data" in result
        assert "wget http://files.example.com/file.txt" in result
        assert "<https://api.example.com/data>" not in result
        assert "<http://files.example.com/file.txt>" not in result


class TestGitHubAnchorExtraction:
    """Test suite for GitHub anchor extraction from changelog headings."""

    @pytest.mark.parametrize(
        ("heading_line", "version", "expected_anchor"),
        [
            # Standard format with link
            ("## [v0.6.0](https://github.com/owner/repo/releases/tag/v0.6.0) - 2025-11-25", "0.6.0", "v060---2025-11-25"),
            # Without link
            ("## v0.6.0 - 2025-11-25", "0.6.0", "v060---2025-11-25"),
            # Version without 'v' prefix in link text
            ("## [0.6.0](https://github.com/owner/repo/releases/tag/v0.6.0) - 2025-11-25", "0.6.0", "060---2025-11-25"),
            # Pre-release with link
            ("## [v1.2.3-rc.1](https://github.com/owner/repo/releases/tag/v1.2.3-rc.1) - 2025-11-25", "1.2.3-rc.1", "v123-rc1---2025-11-25"),
            # Pre-release without link
            ("## v1.2.3-beta.2 - 2025-12-01", "1.2.3-beta.2", "v123-beta2---2025-12-01"),
            # Version with build metadata
            ("## [v2.0.0+build.123](url) - 2025-01-01", "2.0.0+build.123", "v200+build123---2025-01-01"),
            # Simple version without date
            ("## v1.0.0", "1.0.0", "v100"),
            # With angle brackets (edge case)
            ("## <v0.5.0> - 2025-10-15", "0.5.0", "v050---2025-10-15"),
        ],
    )
    def test_extract_github_anchor_from_heading(self, tmp_path, heading_line, version, expected_anchor):
        """Test GitHub anchor extraction from various changelog heading formats."""
        # Create a temporary changelog with the heading
        changelog_path = tmp_path / "CHANGELOG.md"
        changelog_content = f"""# Changelog

{heading_line}

Some release notes here.

## [v0.5.0](url) - 2025-10-01

Older release.
"""
        changelog_path.write_text(changelog_content, encoding="utf-8")

        result = ChangelogUtils._extract_github_anchor(str(changelog_path), version)  # noqa: SLF001
        assert result == expected_anchor

    def test_extract_github_anchor_fallback(self, tmp_path):
        """Test fallback behavior when heading is not found."""
        # Create a changelog without the target version
        changelog_path = tmp_path / "CHANGELOG.md"
        changelog_content = """# Changelog

## [v0.5.0](url) - 2025-10-01

Older release.
"""
        changelog_path.write_text(changelog_content, encoding="utf-8")

        # Request a version that doesn't exist
        result = ChangelogUtils._extract_github_anchor(str(changelog_path), "0.6.0")  # noqa: SLF001
        # Should fall back to version without dots
        assert result == "v060"

    def test_extract_github_anchor_missing_file(self, tmp_path):
        """Test fallback behavior when changelog file doesn't exist."""
        nonexistent_path = tmp_path / "NONEXISTENT.md"

        result = ChangelogUtils._extract_github_anchor(str(nonexistent_path), "1.2.3")  # noqa: SLF001
        # Should fall back to version without dots
        assert result == "v123"

    def test_extract_github_anchor_body_text_no_match(self, tmp_path):
        """Test that version strings in body text don't match (regression test)."""
        changelog_path = tmp_path / "CHANGELOG.md"
        # Version appears in body text but not as heading
        changelog_content = """# Changelog

## [v0.6.0](url) - 2025-11-25

Release notes that mention v0.5.0 in the body text.
Also references [0.5.0] in brackets.

## [v0.4.0](url) - 2025-10-01

Older release.
"""
        changelog_path.write_text(changelog_content, encoding="utf-8")

        # Should find the heading for 0.5.0, not match body text
        # Since 0.5.0 is only in body text, it should fall back
        result = ChangelogUtils._extract_github_anchor(str(changelog_path), "0.5.0")  # noqa: SLF001
        assert result == "v050"  # Fallback behavior

        # Should find 0.6.0 heading correctly
        result = ChangelogUtils._extract_github_anchor(str(changelog_path), "0.6.0")  # noqa: SLF001
        assert result == "v060---2025-11-25"


class TestChangelogTitleFormatting:
    """Test suite for changelog title formatting functionality."""

    def test_format_entry_title_short_title(self):
        """Test formatting of short titles that fit in one line."""
        title = "Add new feature"
        commit_sha = "abc123f"
        repo_url = "https://github.com/owner/repo"
        max_line_length = 160

        result = ChangelogUtils._format_entry_title(title, commit_sha, repo_url, max_line_length)  # noqa: SLF001

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

        result = ChangelogUtils._format_entry_title(title, commit_sha, repo_url, max_line_length)  # noqa: SLF001

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

        result = ChangelogUtils._format_entry_title(title, commit_sha, repo_url, max_line_length)  # noqa: SLF001

        # Should wrap the title
        assert len(result) > 1

        # All lines should respect length limit
        for line in result:
            assert len(line) <= max_line_length

        # Check that markdown characters are escaped in the wrapped content
        # Exclude any commit-link lines (whether single-line or split)
        title_only_lines = [line for line in result if not line.startswith(("  [", "  ("))]
        title_content = "".join(title_only_lines)
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
        result = ChangelogUtils._format_entry_title(title, commit_sha, repo_url, 60)  # noqa: SLF001

        # Should still produce valid output
        assert len(result) >= 1
        for line in result:
            assert len(line) <= 60, f"Line too long ({len(line)} > 60): {line!r}"

        # Test empty title
        title = ""
        result = ChangelogUtils._format_entry_title(title, commit_sha, repo_url, 160)  # noqa: SLF001
        assert len(result) >= 1
        assert result[0].startswith("- ")
        # Commit SHA should appear somewhere in the output (either same line or separate line)
        assert any(commit_sha in line for line in result)

        # Test title with only spaces
        title = "   "
        result = ChangelogUtils._format_entry_title(title, commit_sha, repo_url, 160)  # noqa: SLF001
        assert len(result) >= 1
        assert result[0].startswith("- ")
        # Commit SHA should appear somewhere in the output (either same line or separate line)
        assert any(commit_sha in line for line in result)

    def test_format_entry_title_tiny_limit_drops_bold_and_splits_link(self):
        """Force tiny limits: no bold on wrapped lines; commit link must split."""
        title = "A longish title to wrap"
        commit_sha = "abc123f"
        repo_url = "https://github.com/owner/repo"
        max_line_length = 12  # tiny; exercises no-bold path and link split

        result = ChangelogUtils._format_entry_title(title, commit_sha, repo_url, max_line_length)  # noqa: SLF001

        # First line should start with "- " (no bold) when width is too small
        assert result[0].startswith("- ")
        assert not result[0].startswith("- **")
        assert not result[0].endswith("**")
        # Continuations also without bold
        for line in result[1:]:
            if line.startswith(("  [", "  (")):
                break
            assert line.startswith("  ")
            assert not line.startswith("  **")
            assert not line.endswith("**")
        # Commit link split across two lines under tiny limit
        assert any(line == f"  [`{commit_sha}`]" for line in result)
        assert any(line == f"  ({repo_url}/commit/{commit_sha})" for line in result)

    def test_format_entry_title_title_fits_but_link_wraps(self):
        """Title-only fits; commit link must move to next line (and may split)."""
        title = "Compact title"
        commit_sha = "abc123f"
        repo_url = "https://github.com/owner/repo"
        max_line_length = 30  # fits "- **Compact title**", not the full line with link

        result = ChangelogUtils._format_entry_title(title, commit_sha, repo_url, max_line_length)  # noqa: SLF001

        assert result[0] == f"- **{title}**"
        # Commit link appears on following line(s)
        assert result[1].startswith("  [")
        # For very short limits, link may split further; accept either 1 or 2 lines for the link.
        assert len(result) in (2, 3)

    def test_format_entry_title_regression_long_line(self):
        """Regression test for the specific long line issue found in CHANGELOG.md."""
        # This is the actual long title that caused the markdown lint failure
        title = (
            "Moves the test_alloc_api.rs and test_circumsphere.rs examples to the tests/ directory and renames "
            "them to allocation_api.rs and circumsphere_debug_tools.rs, respectively, to reflect their role as "
            "debugging and testing utilities."
        )
        commit_sha = "f10aba3"
        repo_url = "https://github.com/acgetchell/delaunay"
        max_line_length = 160  # From .markdownlint.json

        result = ChangelogUtils._format_entry_title(title, commit_sha, repo_url, max_line_length)  # noqa: SLF001

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
        title_lines_content: list[str] = []
        for line in result[:-1]:  # Exclude commit link
            if line.startswith(("- ", "  ")):
                core = line[2:]
            else:
                continue
            if core.startswith("**") and core.endswith("**"):
                core = core[2:-2]
            title_lines_content.append(core)

        reconstructed_title = "".join(title_lines_content)
        # Should contain the key parts of the original title (allowing for escaping)
        assert "test_alloc_api.rs" in reconstructed_title or "test\\_alloc\\_api.rs" in reconstructed_title
        assert "circumsphere" in reconstructed_title
        assert "allocation_api.rs" in reconstructed_title or "allocation\\_api.rs" in reconstructed_title

    @pytest.mark.parametrize(
        ("title", "max_length", "expected_min_lines"),
        [
            ("feat: Add new API endpoint", 160, 1),  # Short - single line
            ("fix: Resolve issue with long database query timeout handling", 160, 1),  # Medium - single line
            (
                "refactor: Restructure the authentication middleware to support multiple providers and improve error handling",
                160,
                2,
            ),  # Long - should split
            ("abcdefghijklmno", 50, 2),  # Short title, reasonable limit - should wrap
            ("feat: long commit message that exceeds limit", 60, 2),  # Forces wrapping with reasonable limit
            ("X" * 60, 80, 2),  # Single long token: break_long_words=True path
        ],
    )
    def test_format_entry_title_typical_github_length(self, title, max_length, expected_min_lines):
        """Test with typical GitHub commit title lengths."""
        commit_sha = "abc123"
        repo_url = "https://github.com/test/repo"
        result = ChangelogUtils._format_entry_title(title, commit_sha, repo_url, max_length)  # noqa: SLF001
        assert len(result) >= expected_min_lines
        assert all(len(line) <= max_length for line in result)
