"""Tests for changelog_utils.py git tag size limit handling.

Tests the 125KB GitHub tag annotation limit detection and annotated tag creation with
CHANGELOG.md references.

Note: these tests use real CHANGELOG.md sections as a base, but may synthetically
inflate content to ensure we exercise the "oversized tag" code path even if the
repository's changelog is later compacted.
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from changelog_utils import ChangelogUtils


class TestTagSizeLimitHandling:
    """Test suite for git tag size limit handling (125KB GitHub limit).

    For large changelogs, creates annotated tags with a short message referencing CHANGELOG.md.
    """

    def test_oversized_changelog_triggers_reference_message(self):
        """Oversized changelog content should be replaced with a short CHANGELOG.md reference."""
        # Use a real section as a base to avoid hard-coding huge fixtures.
        changelog_path = ChangelogUtils.find_changelog_path()
        base_content = ChangelogUtils.extract_changelog_section(changelog_path, "0.5.4")

        # Force the content over GitHub's 125KB annotated-tag message limit.
        max_tag_size = 125000
        base_size = len(base_content.encode("utf-8"))
        assert base_size > 0

        repeats = (max_tag_size // base_size) + 2
        oversized_content = "\n\n".join([base_content] * repeats)
        assert len(oversized_content.encode("utf-8")) > max_tag_size

        # Patch extraction so _get_changelog_content sees our oversized payload.
        with patch.object(ChangelogUtils, "extract_changelog_section", return_value=oversized_content):
            tag_message, is_truncated = ChangelogUtils._get_changelog_content("v0.5.4")  # noqa: SLF001

        assert is_truncated is True, "Large changelog should be truncated"
        assert "See full changelog" in tag_message, "Should contain CHANGELOG.md reference"
        assert "github.com/acgetchell/delaunay" in tag_message, "Should contain GitHub link"
        assert len(tag_message) < 1000, "Reference message should be short"

    def test_v0_5_3_changelog_within_limit(self):
        """Test that v0.5.3 changelog content is within 125KB limit."""
        # Previous version should be smaller
        changelog_path = ChangelogUtils.find_changelog_path()
        content = ChangelogUtils.extract_changelog_section(changelog_path, "0.5.3")

        content_size = len(content.encode("utf-8"))

        # If this is also large, just verify the function works correctly
        tag_message, is_truncated = ChangelogUtils._get_changelog_content("v0.5.3")  # noqa: SLF001

        if content_size <= 125000:
            assert is_truncated is False, "Small changelog should not be truncated"
            assert tag_message == content, "Should return full content when under limit"
        else:
            # v0.5.3 is also large, so it should be truncated with reference
            assert is_truncated is True
            assert "See full changelog" in tag_message

    @patch("changelog_utils.run_git_command_with_input")
    def test_create_tag_with_message_truncated(self, mock_run_git_with_input):
        """Test annotated tag with reference message for oversized changelogs."""
        ref_message = "Version 0.5.4\n\nSee full changelog in CHANGELOG.md"
        ChangelogUtils._create_tag_with_message("v0.5.4", ref_message, is_truncated=True)  # noqa: SLF001

        # Should still create annotated tag with reference message
        mock_run_git_with_input.assert_called_once_with(["tag", "-a", "v0.5.4", "-F", "-"], input_data=ref_message)

    @patch("changelog_utils.run_git_command_with_input")
    def test_create_tag_with_message_normal(self, mock_run_git_with_input):
        """Test annotated tag creation with full message for normal-sized changelogs."""
        tag_message = "Version 1.0.0\n\n- Feature 1\n- Feature 2"

        ChangelogUtils._create_tag_with_message("v1.0.0", tag_message, is_truncated=False)  # noqa: SLF001

        # Should call git tag with -a flag and full message from stdin
        mock_run_git_with_input.assert_called_once_with(["tag", "-a", "v1.0.0", "-F", "-"], input_data=tag_message)

    @patch("builtins.print")
    def test_show_success_message_truncated_still_uses_notes_from_tag(self, mock_print):
        """Test success message for truncated changelog still uses --notes-from-tag."""
        ChangelogUtils._show_success_message("v0.5.4", is_truncated=True)  # noqa: SLF001

        # Collect all print calls
        print_calls = [str(call_args[0][0]) if call_args[0] else "" for call_args in mock_print.call_args_list]
        all_output = "\n".join(print_calls)

        # Should mention the tag was created
        assert "Successfully created tag" in all_output

        # Should still use --notes-from-tag (works with reference message)
        assert "--notes-from-tag" in all_output

        # Should note that it references CHANGELOG.md
        assert "references CHANGELOG.md" in all_output

    @patch("builtins.print")
    def test_show_success_message_normal_uses_notes_from_tag(self, mock_print):
        """Test success message for normal changelog uses --notes-from-tag flag."""
        ChangelogUtils._show_success_message("v1.0.0", is_truncated=False)  # noqa: SLF001

        # Collect all print calls
        print_calls = [str(call_args[0][0]) if call_args[0] else "" for call_args in mock_print.call_args_list]
        all_output = "\n".join(print_calls)

        # Should mention the tag was created
        assert "Successfully created tag" in all_output

        # Should use --notes-from-tag
        assert "--notes-from-tag" in all_output

        # Should NOT have truncation warning
        assert "references CHANGELOG.md" not in all_output

    @patch("changelog_utils.ChangelogUtils.validate_git_repo")
    @patch("changelog_utils.ChangelogUtils.validate_semver")
    @patch("changelog_utils.ChangelogUtils._handle_existing_tag")
    @patch("changelog_utils.ChangelogUtils._get_changelog_content")
    @patch("changelog_utils.ChangelogUtils._check_git_config")
    @patch("changelog_utils.ChangelogUtils._create_tag_with_message")
    @patch("changelog_utils.ChangelogUtils._show_success_message")
    def test_create_git_tag_full_workflow_large_changelog(  # noqa: PLR0913
        self,
        mock_show_success,
        mock_create_tag,
        mock_check_git_config,
        mock_get_changelog,
        mock_handle_existing,
        mock_validate_semver,
        mock_validate_git_repo,
    ):
        """Test full workflow for creating tag with large changelog (simulating v0.5.4)."""
        # Mock large changelog that exceeds limit (returns reference message)
        ref_message = "Version 0.5.4\n\nSee full changelog in CHANGELOG.md"
        mock_get_changelog.return_value = (ref_message, True)  # Reference message, truncated=True

        ChangelogUtils.create_git_tag("v0.5.4", force_recreate=False)

        # Verify workflow steps
        mock_validate_git_repo.assert_called_once()
        mock_validate_semver.assert_called_once_with("v0.5.4")
        mock_handle_existing.assert_called_once_with("v0.5.4", force_recreate=False)
        mock_get_changelog.assert_called_once_with("v0.5.4")

        # Should still check git config (for annotated tag)
        mock_check_git_config.assert_called_once()

        # Should create annotated tag with reference message
        mock_create_tag.assert_called_once_with("v0.5.4", ref_message, is_truncated=True)
        mock_show_success.assert_called_once_with("v0.5.4", is_truncated=True)
