#!/usr/bin/env python3
"""
Test suite for enhance_commits.py module.

Tests changelog processing, commit categorization, regex patterns,
and Keep a Changelog format output functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from enhance_commits import (
    _add_section_with_entries,
    _categorize_entry,
    _collect_commit_entry,
    _extract_title_text,
    _get_regex_patterns,
    _process_changelog_lines,
    _process_section_header,
    main,
    process_and_output_categorized_entries,
)


class TestRegexPatterns:
    """Test cases for regex pattern functionality."""

    def test_get_regex_patterns_structure(self):
        """Test that regex patterns have expected structure."""
        patterns = _get_regex_patterns()

        expected_categories = ["added", "removed", "fixed", "changed", "deprecated", "security"]
        assert set(patterns.keys()) == set(expected_categories)

        # Verify each category has patterns
        for category in expected_categories:
            assert isinstance(patterns[category], list)
            assert len(patterns[category]) > 0
            # Verify patterns are valid regex strings
            for pattern in patterns[category]:
                assert isinstance(pattern, str)
                assert len(pattern) > 0

    @pytest.mark.parametrize(
        "text",
        [
            "add new feature",
            "adds support for",
            "added functionality",
            "create new module",
            "enable advanced mode",
            "implement algorithm",
            "introduce new api",
            "new feature for users",
            "feat: add benchmarking",
            "feat: implement caching",
        ],
    )
    def test_added_patterns(self, text):
        """Test patterns for 'Added' category."""
        patterns = _get_regex_patterns()["added"]
        assert any(_match_pattern(pattern, text) for pattern in patterns), f"'{text}' should match 'added' patterns"

    @pytest.mark.parametrize(
        "text",
        [
            "remove deprecated function",
            "delete unused code",
            "drop legacy support",
            "eliminate dead code",
        ],
    )
    def test_removed_patterns(self, text):
        """Test patterns for 'Removed' category."""
        patterns = _get_regex_patterns()["removed"]
        assert any(_match_pattern(pattern, text) for pattern in patterns), f"'{text}' should match 'removed' patterns"

    @pytest.mark.parametrize(
        "text",
        [
            "fix memory leak",
            "resolve compilation error",
            "patch security vulnerability",
            "correct calculation bug",
            "address error handling",
            "improve robustness",
            "enhance stability",
            "fix degenerate case",
            "improve numerical precision",
            "add fallback mechanism",
        ],
    )
    def test_fixed_patterns(self, text):
        """Test patterns for 'Fixed' category."""
        patterns = _get_regex_patterns()["fixed"]
        assert any(_match_pattern(pattern, text) for pattern in patterns), f"'{text}' should match 'fixed' patterns"

    @pytest.mark.parametrize(
        "text",
        [
            "update dependencies",
            "refactor module structure",
            "improve performance",
            "optimize algorithm",
            "enhance benchmarks",
            "perf: speed up processing",
            "update msrv to 1.70",
            "bump version",
        ],
    )
    def test_changed_patterns(self, text):
        """Test patterns for 'Changed' category."""
        patterns = _get_regex_patterns()["changed"]
        assert any(_match_pattern(pattern, text) for pattern in patterns), f"'{text}' should match 'changed' patterns"

    @pytest.mark.parametrize(
        "text",
        [
            "deprecate old api",
            "mark function as deprecated",
        ],
    )
    def test_deprecated_patterns(self, text):
        """Test patterns for 'Deprecated' category."""
        patterns = _get_regex_patterns()["deprecated"]
        assert any(_match_pattern(pattern, text) for pattern in patterns), f"'{text}' should match 'deprecated' patterns"

    @pytest.mark.parametrize(
        "text",
        [
            "fix security vulnerability",
            "patch cve-2023-12345",
            "dependabot update",
            "address security exploit",
        ],
    )
    def test_security_patterns(self, text):
        """Test patterns for 'Security' category."""
        patterns = _get_regex_patterns()["security"]
        assert any(_match_pattern(pattern, text) for pattern in patterns), f"'{text}' should match 'security' patterns"


def _match_pattern(pattern, text):
    """Helper function to test if a pattern matches text."""
    import re

    try:
        return bool(re.search(pattern, text.lower()))
    except re.error:
        return False


class TestTitleExtraction:
    """Test cases for title text extraction."""

    def test_extract_title_with_markdown_bold(self):
        """Test extracting title from markdown bold format."""
        entry = "- **Add new feature** (#123) [`abc1234`](link)"
        result = _extract_title_text(entry)
        assert result == "add new feature"

    def test_extract_title_fallback_parsing(self):
        """Test fallback parsing when no markdown bold present."""
        entry = "- Fix memory leak (#456) [`def5678`](link)"
        result = _extract_title_text(entry)
        assert result == "fix memory leak"

    def test_extract_title_with_complex_format(self):
        """Test title extraction with complex commit format."""
        entry = "- Update dependencies and improve performance (#789)\n  Additional details here"
        result = _extract_title_text(entry)
        assert result == "update dependencies and improve performance"

    def test_extract_title_empty_input(self):
        """Test title extraction with empty or invalid input."""
        assert _extract_title_text("") == ""
        assert _extract_title_text("- ") == ""
        assert _extract_title_text("no match pattern") == ""


class TestCategorization:
    """Test cases for entry categorization logic."""

    def test_categorize_added_entries(self):
        """Test categorization of 'Added' entries."""
        patterns = _get_regex_patterns()

        test_cases = [
            "add new benchmarking system",
            "implement delaunay triangulation",
            "create performance metrics",
            "enable multithreading support",
        ]

        for title in test_cases:
            result = _categorize_entry(title, patterns)
            assert result == "added", f"'{title}' should be categorized as 'added'"

    def test_categorize_fixed_entries(self):
        """Test categorization of 'Fixed' entries."""
        patterns = _get_regex_patterns()

        test_cases = [
            "fix compilation error on windows",
            "resolve memory leak in allocator",
            "patch security vulnerability in parser",
            "correct numerical precision issues",
        ]

        for title in test_cases:
            result = _categorize_entry(title, patterns)
            assert result == "fixed", f"'{title}' should be categorized as 'fixed'"

    def test_categorize_changed_entries(self):
        """Test categorization of 'Changed' entries."""
        patterns = _get_regex_patterns()

        test_cases = ["improve performance by 15%", "update rust version to 1.70", "refactor core algorithms", "optimize memory usage"]

        for title in test_cases:
            result = _categorize_entry(title, patterns)
            assert result == "changed", f"'{title}' should be categorized as 'changed'"

    def test_categorize_default_fallback(self):
        """Test that uncategorized entries fall back to 'changed'."""
        patterns = _get_regex_patterns()

        # Ambiguous entries that don't match specific patterns
        test_cases = ["misc updates to documentation", "various tweaks and adjustments", "general maintenance"]

        for title in test_cases:
            result = _categorize_entry(title, patterns)
            assert result == "changed", f"'{title}' should default to 'changed'"

    def test_categorize_priority_order(self):
        """Test that categorization follows priority order (added > removed > fixed > changed)."""
        patterns = _get_regex_patterns()

        # Entry that could match multiple categories - should pick highest priority
        title = "add fix for memory leak"  # Could be 'added' or 'fixed'
        result = _categorize_entry(title, patterns)
        assert result == "added", "Should prioritize 'added' over 'fixed'"


class TestSectionHandling:
    """Test cases for section header processing."""

    def test_process_section_header_changes(self):
        """Test processing Changes section header."""
        test_cases = ["### Changes", "### Changed"]

        for header in test_cases:
            result = _process_section_header(header)
            assert result == ("changes", True, False, False)

    def test_process_section_header_fixed(self):
        """Test processing Fixed section header."""
        test_cases = ["### Fixed", "### Fixed Issues"]

        for header in test_cases:
            result = _process_section_header(header)
            assert result == ("fixed", False, True, False)

    def test_process_section_header_other_categories(self):
        """Test processing other category headers."""
        test_cases = {
            "### Added": ("added", True, False, False),
            "### Removed": ("removed", True, False, False),
            "### Deprecated": ("deprecated", True, False, False),
            "### Security": ("security", True, False, False),
            "### Merged Pull Requests": ("merged_prs", False, False, True),
        }

        for header, expected in test_cases.items():
            result = _process_section_header(header)
            assert result == expected, f"Header '{header}' should return {expected}"

    def test_process_section_header_no_match(self):
        """Test processing non-section headers."""
        test_cases = ["## Release v1.0.0", "### Some Other Section", "Normal text line", ""]

        for header in test_cases:
            result = _process_section_header(header)
            assert result is None, f"'{header}' should not match any section pattern"


class TestCommitEntryCollection:
    """Test cases for commit entry collection."""

    def test_collect_simple_commit_entry(self):
        """Test collecting simple single-line commit entry."""
        lines = ["- **Add new feature** (#123)", "- **Fix bug** (#456)", "### End Section"]

        entry, next_index = _collect_commit_entry(lines, 0)
        assert entry == "- **Add new feature** (#123)"
        assert next_index == 1

    def test_collect_commit_entry_with_body(self):
        """Test collecting commit entry with indented body content."""
        lines = [
            "- **Add performance monitoring** (#789)",
            "  This adds comprehensive performance tracking",
            "  with detailed metrics and reporting.",
            "",
            "  Includes benchmarking suite integration.",
            "- **Next entry** (#999)",
        ]

        entry, next_index = _collect_commit_entry(lines, 0)
        expected_entry = (
            "- **Add performance monitoring** (#789)\n"
            "  This adds comprehensive performance tracking\n"
            "  with detailed metrics and reporting.\n"
            "\n"
            "  Includes benchmarking suite integration."
        )
        assert entry == expected_entry
        assert next_index == 5

    def test_collect_commit_entry_at_end(self):
        """Test collecting commit entry at end of list."""
        lines = ["- **Final entry** (#999)", "  With some body content"]

        entry, next_index = _collect_commit_entry(lines, 0)
        expected_entry = "- **Final entry** (#999)\n  With some body content"
        assert entry == expected_entry
        assert next_index == 2


class TestOutputGeneration:
    """Test cases for section output generation."""

    def test_add_section_with_entries_first_section(self):
        """Test adding first section (no blank line before)."""
        output_lines = []
        entries = ["- **Add feature A**", "- **Add feature B**"]

        result = _add_section_with_entries(output_lines, "Added", entries, any_sections_output=False)

        assert result is True
        expected = ["### Added", "", "- **Add feature A**", "- **Add feature B**"]
        assert output_lines == expected

    def test_add_section_with_entries_subsequent_section(self):
        """Test adding subsequent section (with blank line before)."""
        output_lines = ["### Added", "", "- **Add something**"]
        entries = ["- **Fix bug A**", "- **Fix bug B**"]

        result = _add_section_with_entries(output_lines, "Fixed", entries, any_sections_output=True)

        assert result is True
        expected = [
            "### Added",
            "",
            "- **Add something**",
            "",  # Blank line before new section
            "### Fixed",
            "",
            "- **Fix bug A**",
            "- **Fix bug B**",
        ]
        assert output_lines == expected

    def test_add_section_with_no_entries(self):
        """Test adding section with no entries."""
        output_lines = []
        entries = []

        result = _add_section_with_entries(output_lines, "Empty", entries, any_sections_output=False)

        assert result is False
        assert output_lines == []

    def test_process_and_output_categorized_entries_complete(self):
        """Test complete categorized entry processing."""
        entries = [
            "- **Add new triangulation algorithm** (#123)",
            "- **Fix memory leak in allocator** (#456)",
            "- **Update dependencies** (#789)",
            "- **Remove deprecated API** (#101)",
        ]
        output_lines = []

        process_and_output_categorized_entries(entries, output_lines)

        # Verify sections appear in Keep a Changelog order
        content = "\n".join(output_lines)

        # Should have Added, Changed, Removed, Fixed sections in that order
        assert "### Added" in content
        assert "### Changed" in content
        assert "### Removed" in content
        assert "### Fixed" in content

        # Verify entries are in correct sections
        assert content.index("### Added") < content.index("### Changed")
        assert content.index("### Changed") < content.index("### Removed")
        assert content.index("### Removed") < content.index("### Fixed")

    def test_process_and_output_categorized_entries_empty(self):
        """Test processing empty entry list."""
        output_lines = []
        process_and_output_categorized_entries([], output_lines)
        assert output_lines == []


class TestChangelogProcessing:
    """Test cases for complete changelog processing."""

    def test_process_changelog_lines_simple(self):
        """Test processing simple changelog with one release."""
        input_lines = [
            "# Changelog",
            "",
            "## [1.0.0] - 2024-01-15",
            "",
            "### Changes",
            "",
            "- **Add new feature** (#123)",
            "- **Fix critical bug** (#456)",
            "- **Update dependencies** (#789)",
        ]

        result = _process_changelog_lines(input_lines)

        content = "\n".join(result)
        assert "### Added" in content
        assert "### Fixed" in content
        assert "### Changed" in content
        assert "Add new feature" in content
        assert "Fix critical bug" in content
        assert "Update dependencies" in content

    def test_process_changelog_lines_multiple_releases(self):
        """Test processing changelog with multiple releases."""
        input_lines = [
            "# Changelog",
            "",
            "## [1.1.0] - 2024-02-01",
            "",
            "### Changes",
            "",
            "- **Add performance monitoring** (#999)",
            "",
            "## [1.0.0] - 2024-01-15",
            "",
            "### Changes",
            "",
            "- **Initial release** (#001)",
        ]

        result = _process_changelog_lines(input_lines)

        content = "\n".join(result)
        assert "[1.1.0]" in content
        assert "[1.0.0]" in content
        assert "Add performance monitoring" in content
        assert "Initial release" in content

    def test_process_changelog_lines_with_merged_prs(self):
        """Test processing changelog that includes Merged Pull Requests section."""
        input_lines = [
            "## [1.0.0] - 2024-01-15",
            "",
            "### Changes",
            "",
            "- **Add feature** (#123)",
            "",
            "### Merged Pull Requests",
            "",
            "- Add feature by @user (#123)",
            "  This PR adds a new feature",
            "  with detailed implementation.",
        ]

        result = _process_changelog_lines(input_lines)

        content = "\n".join(result)
        assert "### Merged Pull Requests" in content
        assert "Add feature by @user" in content
        # PR descriptions should be filtered out
        assert "This PR adds a new feature" not in content

    def test_process_changelog_lines_preserves_structure(self):
        """Test that changelog processing preserves overall structure."""
        input_lines = [
            "# Changelog",
            "",
            "All notable changes to this project will be documented in this file.",
            "",
            "## [Unreleased]",
            "",
            "### Changes",
            "",
            "- **Work in progress** (#WIP)",
            "",
            "## [1.0.0] - 2024-01-15",
        ]

        result = _process_changelog_lines(input_lines)

        content = "\n".join(result)
        assert "# Changelog" in content
        assert "All notable changes" in content
        assert "[Unreleased]" in content
        assert "[1.0.0]" in content


class TestMainFunction:
    """Test cases for main function and CLI interface."""

    def test_main_with_valid_files(self, temp_chdir):
        """Test main function with valid input and output files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test input file
            input_file = temp_path / "input.md"
            input_content = """# Changelog

## [1.0.0] - 2024-01-15

### Changes

- **Add new triangulation algorithm** (#123)
- **Fix memory allocation bug** (#456)
- **Update performance benchmarks** (#789)
"""
            input_file.write_text(input_content, encoding="utf-8")

            output_file = temp_path / "output.md"

            with temp_chdir(temp_path):
                # Mock sys.argv
                with patch("sys.argv", ["enhance_commits.py", str(input_file), str(output_file)]):
                    main()

                # Verify output file was created
                assert output_file.exists()

                output_content = output_file.read_text(encoding="utf-8")
                assert "### Added" in output_content
                assert "### Fixed" in output_content
                assert "### Changed" in output_content
                assert "triangulation algorithm" in output_content

    def test_main_with_invalid_args(self):
        """Test main function with invalid arguments."""
        with patch("sys.argv", ["enhance_commits.py", "only_one_arg"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_main_with_nonexistent_input(self, temp_chdir):
        """Test main function with nonexistent input file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            input_file = temp_path / "nonexistent.md"
            output_file = temp_path / "output.md"

            with (
                temp_chdir(temp_path),
                patch("sys.argv", ["enhance_commits.py", str(input_file), str(output_file)]),
                pytest.raises(FileNotFoundError),
            ):
                main()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_changelog_processing(self):
        """Test processing empty changelog."""
        result = _process_changelog_lines([])
        assert result == []

    def test_changelog_with_no_commit_entries(self):
        """Test processing changelog with no commit entries."""
        input_lines = ["# Changelog", "", "## [1.0.0] - 2024-01-15", "", "### Changes", "", "No changes in this release."]

        result = _process_changelog_lines(input_lines)
        content = "\n".join(result)

        assert "# Changelog" in content
        assert "[1.0.0]" in content
        assert "No changes in this release." in content

    def test_malformed_commit_entries(self):
        """Test processing malformed commit entries."""
        entries = ["Not a proper commit entry", "- Missing markdown formatting", "- **Proper entry** (#123)"]
        output_lines = []

        process_and_output_categorized_entries(entries, output_lines)

        # Should still process and categorize what it can
        content = "\n".join(output_lines)
        assert "Proper entry" in content

    def test_very_long_changelog_processing(self):
        """Test processing changelog with many entries."""
        # Generate large changelog
        input_lines = ["# Changelog", "", "## [1.0.0] - 2024-01-15", "", "### Changes", ""]

        # Add 100 commit entries
        for i in range(100):
            input_lines.append(f"- **Add feature {i:03d}** (#{i:03d})")

        result = _process_changelog_lines(input_lines)

        # Should process without errors
        assert len(result) > len(input_lines)
        content = "\n".join(result)
        assert "### Added" in content
        assert "feature 099" in content
