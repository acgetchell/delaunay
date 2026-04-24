"""Tests for postprocess_changelog.py — trailing blanks, reflow, code blocks, summaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from postprocess_changelog import (
    _compact_entry,
    _fix_typos,
    _inject_summary_sections,
    _max_pr_number,
    _normalize_indented_heading,
    _reflow_line,
    postprocess,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestStripTrailingBlanks:
    def test_strips_trailing_blank_lines(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        f.write_text("# Changelog\n\n- Item\n\n\n\n", encoding="utf-8")

        postprocess(f)

        assert f.read_text(encoding="utf-8") == "# Changelog\n\n- Item\n"

    def test_preserves_single_trailing_newline(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        f.write_text("# Changelog\n\n- Item\n", encoding="utf-8")

        postprocess(f)

        assert f.read_text(encoding="utf-8") == "# Changelog\n\n- Item\n"

    def test_adds_trailing_newline_if_missing(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        f.write_text("# Changelog\n\n- Item", encoding="utf-8")

        postprocess(f)

        assert f.read_text(encoding="utf-8") == "# Changelog\n\n- Item\n"

    def test_preserves_internal_blank_lines(self, tmp_path: Path) -> None:
        content = "# Changelog\n\n## [1.0.0]\n\n### Added\n\n- Item\n\n\n\n"
        f = tmp_path / "CHANGELOG.md"
        f.write_text(content, encoding="utf-8")

        postprocess(f)

        result = f.read_text(encoding="utf-8")
        assert result == "# Changelog\n\n## [1.0.0]\n\n### Added\n\n- Item\n"

    def test_single_newline_file(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        f.write_text("\n", encoding="utf-8")

        postprocess(f)

        assert f.read_text(encoding="utf-8") == "\n"

    def test_empty_file(self, tmp_path: Path) -> None:
        """
        Verifies that processing an empty changelog file results in a file containing exactly one newline.

        Creates an empty CHANGELOG.md at the provided temporary path, runs postprocess on it, and asserts the file's contents are "\n".
        """
        f = tmp_path / "CHANGELOG.md"
        f.write_text("", encoding="utf-8")

        postprocess(f)

        assert f.read_text(encoding="utf-8") == "\n"


class TestFixTypos:
    def test_fixes_deniest(self) -> None:
        assert _fix_typos("Also deniest warnings") == "Also denies warnings"

    def test_fixes_varous(self) -> None:
        assert _fix_typos("Fix varous issues") == "Fix various issues"

    def test_fixes_runtim(self) -> None:
        assert _fix_typos("Increase max runtim") == "Increase max runtime"

    def test_no_partial_match(self) -> None:
        assert _fix_typos("runtime is fine") == "runtime is fine"

    def test_no_change_when_clean(self) -> None:
        text = "All various things at runtime work"
        assert _fix_typos(text) == text

    def test_multiple_occurrences(self) -> None:
        assert _fix_typos("varous varous") == "various various"


class TestReflowLine:
    """Unit tests for the _reflow_line helper."""

    def test_short_line_unchanged(self) -> None:
        line = "- Short line `abc1234`"
        assert _reflow_line(line, max_width=160) == line

    def test_wraps_plain_text(self) -> None:
        line = "  " + "word " * 40
        result = _reflow_line(line.rstrip(), max_width=80)
        for part in result.split("\n"):
            assert len(part) <= 80

    def test_preserves_markdown_link(self) -> None:
        link = "[#235](https://github.com/acgetchell/delaunay/pull/235)"
        line = f"- Description text here {link}"
        result = _reflow_line(line, max_width=40)
        # The link must appear intact in one of the output lines.
        assert any(link in part for part in result.split("\n"))

    def test_preserves_code_span(self) -> None:
        span = "`orientation_from_matrix()`"
        line = f"  Use {span} for exact sign classification on finite inputs and more text padding"
        result = _reflow_line(line, max_width=60)
        assert any(span in part for part in result.split("\n"))

    def test_list_item_continuation_indent(self) -> None:
        line = "- " + "word " * 40
        result = _reflow_line(line.rstrip(), max_width=80)
        parts = result.split("\n")
        assert parts[0].startswith("- ")
        for cont in parts[1:]:
            assert cont.startswith("  ")

    def test_star_list_item(self) -> None:
        line = "* " + "word " * 40
        result = _reflow_line(line.rstrip(), max_width=80)
        parts = result.split("\n")
        assert parts[0].startswith("* ")
        for cont in parts[1:]:
            assert cont.startswith("  ")

    def test_indented_body_text(self) -> None:
        line = "  " + "word " * 40
        result = _reflow_line(line.rstrip(), max_width=80)
        parts = result.split("\n")
        for part in parts:
            assert part.startswith("  ")

    def test_single_long_token_kept(self) -> None:
        url = "https://github.com/acgetchell/delaunay/commit/" + "a" * 40
        line = f"- See [{url}]({url})"
        result = _reflow_line(line, max_width=80)
        # Cannot break inside the link; line may exceed max_width.
        assert url in result

    def test_commit_link_with_backticks(self) -> None:
        link = "[`a62437f`](https://github.com/acgetchell/delaunay/commit/a62437f25c27259f145d3c193ce149ee14b421c7)"
        pr1 = "[#235](https://github.com/acgetchell/delaunay/pull/235)"
        pr2 = "[#236](https://github.com/acgetchell/delaunay/pull/236)"
        line = f"- Use exact arithmetic for orientation predicates {pr1} {pr2} {link}"
        result = _reflow_line(line, max_width=160)
        parts = result.split("\n")
        # Every continuation line should be indented.
        for cont in parts[1:]:
            assert cont.startswith("  ")
        # All links must be intact.
        assert link in result


# ---------------------------------------------------------------------------
# Summary-section helpers
# ---------------------------------------------------------------------------

_OWNER_REPO = "acgetchell/delaunay"
_PR_URL = f"https://github.com/{_OWNER_REPO}/pull"
_COMMIT_URL = f"https://github.com/{_OWNER_REPO}/commit"


def _pr(n: int) -> str:
    """
    Return a Markdown-formatted pull request link for a given pull request number.

    Parameters:
        n (int): Pull request number.

    Returns:
        str: Markdown link in the form "[#<n>](<PR_URL>/<n>)".
    """
    return f"[#{n}]({_PR_URL}/{n})"


def _commit(short: str = "abc1234", full: str = "abc1234deadbeef0123456789") -> str:
    """
    Format a markdown link that references a commit using a short hash as link text and the full hash in the URL.

    Parameters:
        short (str): Short commit identifier used as the link text (rendered in backticks).
        full (str): Full commit hash used to construct the target URL.

    Returns:
        commit_link (str): Markdown link of the form [`<short>`](<commit_url>/<full>).
    """
    return f"[`{short}`]({_COMMIT_URL}/{full})"


class TestCompactEntry:
    def test_strips_commit_hash_link(self) -> None:
        line = f"- Some feature {_commit()}"
        assert _compact_entry(line) == "- Some feature"

    def test_strips_breaking_prefix(self) -> None:
        line = f"- [**breaking**] Some change {_commit()}"
        assert _compact_entry(line, strip_breaking=True) == "- Some change"

    def test_preserves_pr_links(self) -> None:
        line = f"- Feature {_pr(42)} {_commit()}"
        assert _compact_entry(line) == f"- Feature {_pr(42)}"

    def test_keeps_breaking_when_not_stripped(self) -> None:
        line = f"- [**breaking**] Change {_commit()}"
        assert _compact_entry(line) == "- [**breaking**] Change"


class TestMaxPrNumber:
    def test_single_pr(self) -> None:
        assert _max_pr_number(f"- Feature {_pr(42)}") == 42

    def test_multiple_prs(self) -> None:
        assert _max_pr_number(f"- Feature {_pr(10)} {_pr(99)}") == 99

    def test_no_prs(self) -> None:
        assert _max_pr_number("- Plain entry") == 0


class TestSummarySections:
    @staticmethod
    def _changelog(entries: str) -> str:
        """
        Create a sample changelog file containing a header, a 1.0.0 release section dated
        2026-01-01, and an "Added" subsection populated with the provided entries.

        Parameters:
            entries (str): Markdown content to place under the "Added" subsection (should include any list markers or paragraphs).

        Returns:
            str: The full changelog content as a string.
        """
        return f"# Changelog\n\n## [1.0.0] - 2026-01-01\n\n### Added\n\n{entries}\n"

    def test_injects_pr_summary(self) -> None:
        content = self._changelog(f"- Feature A {_pr(10)} {_commit()}\n- Plain commit {_commit('def5678', 'def5678deadbeef')}")
        result = _inject_summary_sections(content)
        assert "### Merged Pull Requests" in result
        # PR entry in summary (without commit hash).
        assert f"- Feature A {_pr(10)}" in result
        # Plain entry only appears once (in Added, not in summary).
        plain_lines = [ln for ln in result.split("\n") if ln.startswith("- Plain commit")]
        assert len(plain_lines) == 1

    def test_injects_breaking_summary(self) -> None:
        content = self._changelog(f"- [**breaking**] Big change {_pr(5)} {_commit()}")
        result = _inject_summary_sections(content)
        assert "### ⚠️ Breaking Changes" in result
        assert "### Merged Pull Requests" in result
        # Breaking section appears before Merged PRs.
        assert result.index("### ⚠️ Breaking Changes") < result.index("### Merged Pull Requests")

    def test_pr_sorted_descending(self) -> None:
        """
        Verifies that PRs in the injected "Merged Pull Requests" summary are sorted in descending order by PR number.

        Constructs a changelog with three entries containing PR links and confirms the summary lists them in order: highest PR number first.
        """
        content = self._changelog(
            f"- First {_pr(5)} {_commit('aaa1111', 'aaa1111deadbeef')}\n"
            f"- Second {_pr(20)} {_commit('bbb2222', 'bbb2222deadbeef')}\n"
            f"- Third {_pr(10)} {_commit('ccc3333', 'ccc3333deadbeef')}"
        )
        result = _inject_summary_sections(content)
        lines = result.split("\n")
        pr_idx = next(i for i, ln in enumerate(lines) if "### Merged Pull Requests" in ln)
        pr_lines = [ln for ln in lines[pr_idx + 1 :] if ln.startswith("- ")][:3]
        assert "#20" in pr_lines[0]
        assert "#10" in pr_lines[1]
        assert "#5" in pr_lines[2]

    def test_no_summary_without_prs(self) -> None:
        content = self._changelog(f"- Plain commit {_commit()}")
        result = _inject_summary_sections(content)
        assert "### Merged Pull Requests" not in result

    def test_idempotent(self) -> None:
        content = self._changelog(f"- Feature {_pr(10)} {_commit()}")
        first = _inject_summary_sections(content)
        second = _inject_summary_sections(first)
        assert first == second

    def test_idempotent_breaking_only(self) -> None:
        """Breaking-only sections (no PR links) must not be double-injected."""
        content = self._changelog(f"- [**breaking**] Remove old API {_commit()}")
        first = _inject_summary_sections(content)
        assert "### ⚠️ Breaking Changes" in first
        # "Merged Pull Requests" should NOT appear (no PR link).
        assert "### Merged Pull Requests" not in first
        second = _inject_summary_sections(first)
        assert first == second

    def test_ignores_indented_sub_items(self) -> None:
        content = self._changelog(f"- Feature {_commit()}\n  - Sub-item {_pr(99)}")
        result = _inject_summary_sections(content)
        assert "### Merged Pull Requests" not in result

    def test_multiple_pr_links_preserved(self) -> None:
        content = self._changelog(f"- Feature {_pr(10)} {_pr(20)} {_commit()}")
        result = _inject_summary_sections(content)
        assert f"- Feature {_pr(10)} {_pr(20)}" in result


class TestListMarkerNormalization:
    """MD004: consistent ``-`` list markers."""

    def test_star_to_dash_at_column_zero(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        f.write_text("* item one\n* item two\n", encoding="utf-8")
        postprocess(f)
        result = f.read_text(encoding="utf-8")
        assert "* item" not in result
        assert "- item one" in result
        assert "- item two" in result

    def test_star_to_dash_indented(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        f.write_text("- parent item\n  * sub-item\n", encoding="utf-8")
        postprocess(f)
        assert "  - sub-item" in f.read_text(encoding="utf-8")

    def test_star_in_bold_not_changed(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        f.write_text("Some **bold** text\n", encoding="utf-8")
        postprocess(f)
        assert "**bold**" in f.read_text(encoding="utf-8")

    def test_star_inside_code_block_not_changed(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        f.write_text("```text\n* keep me\n```\n", encoding="utf-8")
        postprocess(f)
        assert "* keep me" in f.read_text(encoding="utf-8")


class TestBlankLineBeforeList:
    """MD032: blank lines around lists."""

    def test_inserts_blank_before_list_after_prose(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        f.write_text("Some prose.\n- list item\n", encoding="utf-8")
        postprocess(f)
        assert f.read_text(encoding="utf-8") == "Some prose.\n\n- list item\n"

    def test_no_double_blank(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        f.write_text("Some prose.\n\n- list item\n", encoding="utf-8")
        postprocess(f)
        assert f.read_text(encoding="utf-8") == "Some prose.\n\n- list item\n"

    def test_no_blank_between_consecutive_items(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        f.write_text("- one\n- two\n", encoding="utf-8")
        postprocess(f)
        assert f.read_text(encoding="utf-8") == "- one\n- two\n"

    def test_no_blank_after_heading(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        f.write_text("### Added\n- item\n", encoding="utf-8")
        postprocess(f)
        # Heading directly followed by list is fine per MD032.
        assert "\n\n- item" not in f.read_text(encoding="utf-8")


class TestIndentedHeadingNormalization:
    """MD023: commit-body headings are rendered as prose, not nested headings."""

    def test_indented_atx_heading_becomes_bold_prose(self) -> None:
        assert _normalize_indented_heading("  ## Correctness Fixes") == "  **Correctness Fixes**"

    def test_indented_atx_closing_sequence_becomes_bold_prose(self) -> None:
        assert _normalize_indented_heading("  ### API Design ###") == "  **API Design**"

    def test_column_zero_changelog_heading_is_preserved(self) -> None:
        assert _normalize_indented_heading("### Added") == "### Added"

    def test_normalized_heading_is_idempotent(self) -> None:
        assert _normalize_indented_heading("  **Title**") == "  **Title**"

        once = _normalize_indented_heading("  ## Correctness Fixes")
        assert _normalize_indented_heading(once) == once

    def test_full_pipeline_normalizes_commit_body_headings(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        f.write_text(
            "# Changelog\n\n"
            "## [1.0.0]\n\n"
            "### Performance\n\n"
            "- perf: improve Hilbert curve correctness\n\n"
            "  ## Correctness Fixes\n\n"
            "  - Add debug_assert guards\n\n"
            "  ## API Design\n\n"
            "  - Add HilbertError enum\n",
            encoding="utf-8",
        )

        postprocess(f)

        result = f.read_text(encoding="utf-8")
        assert "  ## Correctness Fixes" not in result
        assert "  ## API Design" not in result
        assert "  **Correctness Fixes**" in result
        assert "  **API Design**" in result
        assert "### Performance" in result


class TestCodeBlockLanguage:
    def test_adds_language_to_bare_fence(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        f.write_text("  ```\n  let x = 1;\n  ```\n", encoding="utf-8")

        postprocess(f)

        result = f.read_text(encoding="utf-8")
        assert "```text" in result

    def test_preserves_existing_language(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        f.write_text("```rust\nlet x = 1;\n```\n", encoding="utf-8")

        postprocess(f)

        result = f.read_text(encoding="utf-8")
        assert "```rust" in result
        assert "```text" not in result

    def test_no_reflow_inside_code_block(self, tmp_path: Path) -> None:
        long_code = "  let very_long = " + "a" * 200 + ";"
        f = tmp_path / "CHANGELOG.md"
        f.write_text(f"```rust\n{long_code}\n```\n", encoding="utf-8")

        postprocess(f)

        result = f.read_text(encoding="utf-8")
        assert long_code in result


class TestIntegration:
    def test_full_changelog_reflow(self, tmp_path: Path) -> None:
        """Simulate a realistic changelog snippet with long lines."""
        long_entry = (
            "- Use exact arithmetic [#235](https://github.com/acgetchell/delaunay/pull/235) "
            "[#236](https://github.com/acgetchell/delaunay/pull/236) "
            "[`a62437f`](https://github.com/acgetchell/delaunay/commit/a62437f25c27259f145d3c193ce149ee14b421c7)"
        )
        long_body = "  " + "word " * 40
        content = f"# Changelog\n\n## [0.7.2]\n\n### Added\n\n{long_entry}\n\n{long_body.rstrip()}\n\n"

        f = tmp_path / "CHANGELOG.md"
        f.write_text(content, encoding="utf-8")

        postprocess(f)

        result = f.read_text(encoding="utf-8")
        for line in result.split("\n"):
            if line.strip():
                assert len(line) <= 160 or "[" in line, f"Line too long ({len(line)}): {line[:80]}..."

    def test_summary_sections_in_full_pipeline(self, tmp_path: Path) -> None:
        """Summary sections are injected and survive reflow."""
        entry = f"- Feature {_pr(42)} {_commit()}"
        content = f"# Changelog\n\n## [1.0.0] - 2026-01-01\n\n### Added\n\n{entry}\n"
        f = tmp_path / "CHANGELOG.md"
        f.write_text(content, encoding="utf-8")

        postprocess(f)

        result = f.read_text(encoding="utf-8")
        assert "### Merged Pull Requests" in result
        assert "### Added" in result
        # Summary appears before categorised sections.
        assert result.index("### Merged Pull Requests") < result.index("### Added")
