#!/usr/bin/env python3
"""
Test suite for enhance_commits.py module.

Tests changelog processing, commit categorization, regex patterns,
and Keep a Changelog format output functionality.
"""

import io
import tempfile
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import pytest

from enhance_commits import (
    CATEGORY_PATTERNS,
    COMMIT_BULLET_RE,
    TITLE_FALLBACK_RE,
    _add_section_with_entries,
    _categorize_entry,
    _collect_commit_entry,
    _extract_title_text,
    _process_changelog_lines,
    _process_section_header,
    main,
    process_and_output_categorized_entries,
)


@pytest.fixture
def regex_patterns():
    """Fixture to provide regex patterns for categorization tests."""
    return CATEGORY_PATTERNS


class TestRegexPatterns:
    """Test cases for regex pattern functionality."""

    def test_get_regex_patterns_structure(self):
        """Test that regex patterns have expected structure."""
        patterns = CATEGORY_PATTERNS

        expected_categories = ["added", "removed", "fixed", "changed", "deprecated", "security"]
        assert set(patterns.keys()) == set(expected_categories)

        # Verify each category has patterns
        for category in expected_categories:
            assert isinstance(patterns[category], list)
            assert len(patterns[category]) > 0
            # Verify patterns are compiled regex objects
            for pattern in patterns[category]:
                assert hasattr(pattern, "search")  # Check it's a compiled regex
                assert hasattr(pattern, "pattern")  # Has pattern attribute

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
        patterns = CATEGORY_PATTERNS["added"]
        assert any(pattern.search(text.lower()) for pattern in patterns), f"'{text}' should match 'added' patterns"

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
        patterns = CATEGORY_PATTERNS["removed"]
        assert any(pattern.search(text.lower()) for pattern in patterns), f"'{text}' should match 'removed' patterns"

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
        patterns = CATEGORY_PATTERNS["fixed"]
        assert any(pattern.search(text.lower()) for pattern in patterns), f"'{text}' should match 'fixed' patterns"

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
        patterns = CATEGORY_PATTERNS["changed"]
        assert any(pattern.search(text.lower()) for pattern in patterns), f"'{text}' should match 'changed' patterns"

    @pytest.mark.parametrize(
        "text",
        [
            "deprecate old api",
            "mark function as deprecated",
        ],
    )
    def test_deprecated_patterns(self, text):
        """Test patterns for 'Deprecated' category."""
        patterns = CATEGORY_PATTERNS["deprecated"]
        assert any(pattern.search(text.lower()) for pattern in patterns), f"'{text}' should match 'deprecated' patterns"

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
        patterns = CATEGORY_PATTERNS["security"]
        assert any(pattern.search(text.lower()) for pattern in patterns), f"'{text}' should match 'security' patterns"


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

    @pytest.mark.parametrize(
        ("case", "expected"),
        [
            ("- Simple commit (#456)", "simple commit"),
            ("- Another fix [`def5678`](https://example.com/commit/def5678)", "another fix"),
            ("- Fix bug in parser", "fix bug in parser"),  # No PR number or commit hash
        ],
    )
    def test_extract_title_edge_cases(self, case, expected):
        """Test title extraction edge cases that use fallback regex."""
        title = _extract_title_text(case)
        assert title == expected


class TestCategorization:
    """Test cases for entry categorization logic."""

    @pytest.mark.parametrize(
        "title",
        [
            "add new benchmarking system",
            "implement delaunay triangulation",
            "create performance metrics",
            "enable multithreading support",
        ],
    )
    def test_categorize_added_entries(self, title, regex_patterns):
        """Test categorization of 'Added' entries."""
        result = _categorize_entry(title, regex_patterns)
        assert result == "added"

    @pytest.mark.parametrize(
        "title",
        [
            "fix compilation error on windows",
            "resolve memory leak in allocator",
            "patch security vulnerability in parser",
            "correct numerical precision issues",
        ],
    )
    def test_categorize_fixed_entries(self, title, regex_patterns):
        """Test categorization of 'Fixed' entries."""
        result = _categorize_entry(title, regex_patterns)
        assert result == "fixed"

    @pytest.mark.parametrize(
        "title",
        [
            "improve performance by 15%",
            "update rust version to 1.70",
            "refactor core algorithms",
            "optimize memory usage",
        ],
    )
    def test_categorize_changed_entries(self, title, regex_patterns):
        """Test categorization of 'Changed' entries."""
        result = _categorize_entry(title, regex_patterns)
        assert result == "changed"

    @pytest.mark.parametrize(
        "title",
        [
            "misc updates to documentation",
            "various tweaks and adjustments",
            "general maintenance",
        ],
    )
    def test_categorize_default_fallback(self, title, regex_patterns):
        """Test that uncategorized entries fall back to 'changed'."""
        result = _categorize_entry(title, regex_patterns)
        assert result == "changed"

    def test_categorize_priority_order(self, regex_patterns):
        """Test that categorization follows priority order (added > removed > fixed > changed)."""
        # Entry that could match multiple categories - should pick highest priority
        title = "add fix for memory leak"  # Could be 'added' or 'fixed'
        result = _categorize_entry(title, regex_patterns)
        assert result == "added", "Should prioritize 'added' over 'fixed'"

    def test_categorize_explicit_prefix_takes_precedence(self, regex_patterns):
        """Test that explicit category prefixes take precedence over keyword matching.

        Regression test for GitHub issue where "Fixed: Correctly count removed cells"
        was incorrectly categorized as "removed" instead of "fixed" because the word
        "removed" appeared in the commit message.
        """
        # This commit has "Fixed:" prefix but also contains "removed" keyword
        title = "fixed: correctly count removed cells in bowyer-watson"
        result = _categorize_entry(title, regex_patterns)
        assert result == "fixed", "Should categorize as 'fixed' due to explicit 'Fixed:' prefix"

        # Similar test cases
        assert _categorize_entry("added: remove deprecated functionality", regex_patterns) == "added"
        assert _categorize_entry("removed: fix for legacy code", regex_patterns) == "removed"
        assert _categorize_entry("changed: add new feature", regex_patterns) == "changed"

    def test_categorize_explicit_prefix_all_forms(self, regex_patterns):
        """Test that all forms of explicit prefixes work correctly.

        Verifies short forms (fix:, add:), past tense (fixed:, added:),
        and variations with spacing.
        """
        # Short forms (present tense)
        assert _categorize_entry("fix: memory leak in parser", regex_patterns) == "fixed"
        assert _categorize_entry("add: new triangulation algorithm", regex_patterns) == "added"
        assert _categorize_entry("remove: deprecated api endpoint", regex_patterns) == "removed"
        assert _categorize_entry("change: update dependencies", regex_patterns) == "changed"
        assert _categorize_entry("deprecate: old interface", regex_patterns) == "deprecated"

        # Past tense forms
        assert _categorize_entry("fixed: memory leak in parser", regex_patterns) == "fixed"
        assert _categorize_entry("added: new triangulation algorithm", regex_patterns) == "added"
        assert _categorize_entry("removed: deprecated api endpoint", regex_patterns) == "removed"
        assert _categorize_entry("changed: update dependencies", regex_patterns) == "changed"
        assert _categorize_entry("deprecated: old interface", regex_patterns) == "deprecated"

        # With spacing variations
        assert _categorize_entry("fix : memory leak in parser", regex_patterns) == "fixed"
        assert _categorize_entry("fixed : memory leak in parser", regex_patterns) == "fixed"
        assert _categorize_entry("add : new feature", regex_patterns) == "added"
        assert _categorize_entry("added : new feature", regex_patterns) == "added"


class TestSectionHandling:
    """Test cases for section header processing."""

    @pytest.mark.parametrize(
        "header",
        ["### Changes", "### Changed"],
    )
    def test_process_section_header_changes(self, header):
        """Test processing Changes section header."""
        result = _process_section_header(header)
        assert result == ("changes", True, False, False)

    @pytest.mark.parametrize(
        "header",
        ["### Fixed", "### Fixed Issues"],
    )
    def test_process_section_header_fixed(self, header):
        """Test processing Fixed section header."""
        result = _process_section_header(header)
        assert result == ("fixed", False, True, False)

    @pytest.mark.parametrize(
        ("header", "expected"),
        [
            ("### Added", ("added", True, False, False)),
            ("### Removed", ("removed", True, False, False)),
            ("### Deprecated", ("deprecated", True, False, False)),
            ("### Security", ("security", True, False, False)),
            ("### Merged Pull Requests", ("merged_prs", False, False, True)),
        ],
    )
    def test_process_section_header_other_categories(self, header, expected):
        """Test processing other category headers."""
        result = _process_section_header(header)
        assert result == expected

    @pytest.mark.parametrize(
        "header",
        ["## Release v1.0.0", "### Some Other Section", "Normal text line", ""],
    )
    def test_process_section_header_no_match(self, header):
        """Test processing non-section headers."""
        result = _process_section_header(header)
        assert result is None


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
        expected = ["### Added", "", "- **Add feature A**", "", "- **Add feature B**"]
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
            "",  # Blank line between entries
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

    def test_process_changelog_lines_proper_mpr_ordering(self):
        """Test that categorized entries appear before Merged Pull Requests section."""
        input_lines = [
            "## [1.0.0] - 2024-01-15",
            "",
            "### Changes",
            "",
            "- **Add new feature** (#123)",
            "- **Fix critical bug** (#456)",
            "- **Update dependencies** (#789)",
            "",
            "### Merged Pull Requests",
            "",
            "- Add feature by @user (#123)",
            "  This PR adds a new feature",
            "  with detailed implementation.",
            "- Fix bug by @contributor (#456)",
            "  Critical bug fix for production.",
        ]

        result = _process_changelog_lines(input_lines)
        content = "\n".join(result)

        # Verify that categorized sections appear in the output
        assert "### Added" in content
        assert "### Fixed" in content
        assert "### Changed" in content
        assert "### Merged Pull Requests" in content

        # Verify proper ordering: categorized sections should come before MPRs
        added_pos = content.find("### Added")
        fixed_pos = content.find("### Fixed")
        changed_pos = content.find("### Changed")
        mpr_pos = content.find("### Merged Pull Requests")

        assert added_pos < mpr_pos, "Added section should come before Merged Pull Requests"
        assert fixed_pos < mpr_pos, "Fixed section should come before Merged Pull Requests"
        assert changed_pos < mpr_pos, "Changed section should come before Merged Pull Requests"

        # Verify entries are correctly categorized
        assert "Add new feature" in content
        assert "Fix critical bug" in content
        assert "Update dependencies" in content

        # Verify MPR section is preserved but descriptions are filtered out
        assert "Add feature by @user" in content
        assert "Fix bug by @contributor" in content
        assert "This PR adds a new feature" not in content  # Should be filtered out
        assert "Critical bug fix for production." not in content  # Should be filtered out


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
                pytest.raises(SystemExit) as exc_info,
            ):
                main()
            assert exc_info.value.code == 1


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

    def test_compiled_regex_performance(self):
        """Test that the compiled fallback regex works correctly and efficiently."""
        # Test the compiled regex directly
        test_line = "- fix: improve performance (#123) [`abc1234`](https://example.com)"
        match = TITLE_FALLBACK_RE.match(test_line)
        assert match is not None
        assert match.group(1).strip() == "fix: improve performance"

        # Test through the extraction function (prefers markdown extraction)
        entry = "- **fix: improve performance** (#123) [`abc1234`](https://example.com)"
        title = _extract_title_text(entry)
        assert title == "fix: improve performance"

        # Test edge cases that go through fallback regex - moved to parametrized test
        # See test_extract_title_edge_cases for parametrized version

        # Verify the compiled regex is faster than recompiling each time
        # (This mainly tests that TITLE_FALLBACK_RE is available and compiled)
        assert TITLE_FALLBACK_RE.pattern is not None
        assert hasattr(TITLE_FALLBACK_RE, "match")

    def test_blank_line_guard_prevents_double_blanks(self):
        """Test that blank line guard prevents adding extra blank lines before releases."""
        input_lines = [
            "# Changelog",
            "",
            "## [1.0.0] - 2023-12-01",
            "",
            "### Changed",
            "",
            "- **Update something** [`abc123`](https://example.com)",
            "",  # Already has a blank line
            "## [0.9.0] - 2023-11-01",  # This should not get additional blanks
        ]

        output_lines = _process_changelog_lines(input_lines)

        # Find the second release header and verify correct blank line handling
        for i, line in enumerate(output_lines):
            if line.startswith("## [0.9.0]"):
                # There should be exactly one blank line before this release
                # (from the input), not two (which would happen without the guard)
                assert i >= 1, "Release header should not be at the very beginning"
                prev_line = output_lines[i - 1]
                assert prev_line == "", f"Expected blank line before release header, got: {prev_line!r}"

                # Check that line before the blank line is not blank (no double blanks)
                if i >= 2:
                    prev_prev_line = output_lines[i - 2]
                    assert prev_prev_line != "", f"Found double blank lines before release header: {prev_prev_line!r} + {prev_line!r} + {line!r}"

                # Success - found the pattern we expected
                break
        else:
            pytest.fail("Could not find the ## [0.9.0] release header in output")

    def test_no_redundant_clear_at_end(self):
        """Test that the function works correctly without redundant clear() call."""
        input_lines = [
            "# Changelog",
            "",
            "## [1.0.0] - 2023-12-01",
            "",
            "### Changed",
            "",
            "- **Update something** [`abc123`](https://example.com)",
        ]

        # This should work without issues despite removing the redundant clear()
        output_lines = _process_changelog_lines(input_lines)

        # Verify basic functionality still works
        assert len(output_lines) > 0
        output_text = "\n".join(output_lines)
        assert "### Changed" in output_text

        # Check that entries are still properly categorized
        # The "Update something" should be categorized as "Changed"
        assert "Update something" in output_text


class TestImprovements:
    """Test cases for the specific improvements made to the script."""

    def test_indented_commit_bullet_regex(self):
        """Test that COMMIT_BULLET_RE matches bullets with leading whitespace."""
        # Test standard format with dash bullets (no indentation)
        assert COMMIT_BULLET_RE.match("- **Fix: some issue**")

        # Test standard format with asterisk bullets (no indentation)
        assert COMMIT_BULLET_RE.match("* **Fix: some issue**")

        # Test with leading spaces - dash bullets
        assert COMMIT_BULLET_RE.match("  - **Fix: some issue**")
        assert COMMIT_BULLET_RE.match("    -   **Add: new feature**")

        # Test with leading spaces - asterisk bullets
        assert COMMIT_BULLET_RE.match("  * **Fix: some issue**")
        assert COMMIT_BULLET_RE.match("    *   **Add: new feature**")

        # Test with tabs - dash bullets
        assert COMMIT_BULLET_RE.match("\t- **Change: update deps**")
        assert COMMIT_BULLET_RE.match("\t\t-\t**Remove: old code**")

        # Test with tabs - asterisk bullets
        assert COMMIT_BULLET_RE.match("\t* **Change: update deps**")
        assert COMMIT_BULLET_RE.match("\t\t*\t**Remove: old code**")

        # Test mixed whitespace - both bullet types
        assert COMMIT_BULLET_RE.match(" \t - \t**Security: patch CVE**")
        assert COMMIT_BULLET_RE.match(" \t * \t**Security: patch CVE**")

        # Test non-matching patterns - the improved regex is more permissive
        # It only checks for bullet format, not bold formatting
        # These should match since they have bullet format:
        assert COMMIT_BULLET_RE.match("- Not bold text")
        assert COMMIT_BULLET_RE.match("* Not bold text")

        # Test non-bullet characters
        assert not COMMIT_BULLET_RE.match("+ **Not supported bullet**")
        assert not COMMIT_BULLET_RE.match("1. **Not a bullet list**")

    def test_optimized_first_line_extraction(self):
        """Test that first line extraction doesn't build unnecessary lists."""

        # Test with multiline entry - should only process first line
        multiline_entry = "- **Fix: issue with parser**\n  Additional details\n  More info"
        result = _extract_title_text(multiline_entry)
        assert result == "fix: issue with parser"

        # Test single line entry
        single_line = "- **Add: new feature**"
        result = _extract_title_text(single_line)
        assert result == "add: new feature"

        # Test empty entry
        result = _extract_title_text("")
        assert result == ""

    def test_generalized_indentation_matching(self):
        """Test that body content collection works with various indentation."""

        # Test with 2+ spaces (minimum indentation)
        lines = [
            "- **Fix: memory leak**",
            "  This fixes a critical issue",
            "   with additional details",
            "    and more info",
            "next non-indented line",
        ]
        entry, next_index = _collect_commit_entry(lines, 0)
        expected = "- **Fix: memory leak**\n  This fixes a critical issue\n   with additional details\n    and more info"
        assert entry == expected
        assert next_index == 4  # Should stop at the non-indented line

        # Test with tabs and mixed whitespace
        lines_tabs = ["- **Add: new feature**", "\t\tTab-indented content", " \t Mixed whitespace", "    Four spaces", "no indent"]
        entry, next_index = _collect_commit_entry(lines_tabs, 0)
        expected_tabs = "- **Add: new feature**\n\t\tTab-indented content\n \t Mixed whitespace\n    Four spaces"
        assert entry == expected_tabs
        assert next_index == 4

        # Test that single space doesn't count as indentation
        lines_single = ["- **Change: update**", " single space should not match", "normal line"]
        entry, next_index = _collect_commit_entry(lines_single, 0)
        assert entry == "- **Change: update**"  # Should not include single-space line
        assert next_index == 1

    def test_title_fallback_regex_handles_indentation(self):
        """Test that TITLE_FALLBACK_RE regex handles indented bullets robustly."""
        # Test cases without indentation (should still work)
        test_cases_no_indent = [
            "- Fix performance regression",
            "- Fix performance regression (#123)",
            "- Fix performance regression [`abc1234`]",
            "- Fix performance regression (#123) [`abc1234`](https://example.com)",
        ]

        for case in test_cases_no_indent:
            match = TITLE_FALLBACK_RE.match(case)
            assert match is not None, f"Should match non-indented case: {case}"
            assert match.group(1) == "Fix performance regression", f"Should extract correct title from: {case}"

        # Test cases with indentation (the improvement)
        test_cases_indented = [
            "  - Fix performance regression",  # 2 spaces
            "    - Fix performance regression",  # 4 spaces
            "\t- Fix performance regression",  # Tab
            " \t - Fix performance regression",  # Mixed whitespace
            "  - Fix performance regression (#123)",
            "    - Fix performance regression [`abc1234`]",
            "\t- Fix performance regression (#123) [`abc1234`](https://example.com)",
        ]

        for case in test_cases_indented:
            match = TITLE_FALLBACK_RE.match(case)
            assert match is not None, f"Should match indented case: {case!r}"
            assert match.group(1) == "Fix performance regression", f"Should extract correct title from: {case!r}"

        # Test through the extraction function for integration
        indented_entries = [
            "  - Fix memory leak (#456) [`def5678`](link)",
            "\t- Update dependencies and improve performance (#789)",
            "    - Remove deprecated functionality",
        ]

        expected_titles = [
            "fix memory leak",
            "update dependencies and improve performance",
            "remove deprecated functionality",
        ]

        for entry, expected in zip(indented_entries, expected_titles, strict=False):
            result = _extract_title_text(entry)
            assert result == expected, f"Should extract '{expected}' from '{entry}', got '{result}'"

        # Test cases that should NOT match (validation)
        non_matching_cases = [
            "not a bullet point",
            "+ Different bullet type",
            "1. Numbered list",
            "- ",  # Empty title
            "",  # Empty string
        ]

        for case in non_matching_cases:
            match = TITLE_FALLBACK_RE.match(case)
            if case in ["", "- "]:
                # These cases are handled by the extraction function's empty checks
                result = _extract_title_text(case)
                assert result == "", f"Empty case should return empty string: {case!r}"
            else:
                assert match is None, f"Should not match non-bullet case: {case!r}"

    def test_match_group_compatibility(self):
        """Test that title extraction uses Match.group(1) for Python compatibility."""

        # Test entry with markdown bold formatting
        test_entry = "- **Fix memory leak in allocator** (#456) [`def5678`](link)"

        # This should work correctly with Match.group(1) approach
        result = _extract_title_text(test_entry)
        assert result == "fix memory leak in allocator"

        # Test with complex formatting
        complex_entry = "- **Add comprehensive performance monitoring system** (#999) [`abc1234`](https://example.com)\n  Additional details here"
        result = _extract_title_text(complex_entry)
        assert result == "add comprehensive performance monitoring system"

        # Test edge case with empty bold section (should fall back to regex)
        edge_case = "- Fix simple bug (#123)"
        result = _extract_title_text(edge_case)
        assert result == "fix simple bug"

    def test_helpful_usage_message(self):
        """Test that the script provides helpful usage message on bad args."""

        # Test with no arguments
        stderr_capture = io.StringIO()
        with (
            patch("sys.argv", ["enhance_commits.py"]),
            redirect_stderr(stderr_capture),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1
        stderr_output = stderr_capture.getvalue()
        assert "Usage:" in stderr_output
        assert "enhance_commits.py" in stderr_output
        assert "<input_changelog>" in stderr_output
        assert "<output_changelog>" in stderr_output

        # Test with wrong number of arguments (single arg)
        stderr_capture = io.StringIO()
        with (
            patch("sys.argv", ["enhance_commits.py", "single_arg"]),
            redirect_stderr(stderr_capture),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1
        stderr_output = stderr_capture.getvalue()
        assert "Usage:" in stderr_output
