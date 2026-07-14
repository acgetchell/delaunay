"""Tests for postprocess_changelog.py — trailing blanks, reflow, code blocks, summaries."""

import stat
from pathlib import Path

import pytest

from postprocess_changelog import (
    _CodeFence,
    _compact_entry,
    _inject_summary_sections,
    _is_duplicate_squash_heading,
    _is_isolated_body_heading,
    _max_pr_number,
    _normalize_email_autolinks,
    _normalize_entry_heading,
    _normalize_indented_heading,
    _normalize_squash_heading,
    _plain_summary,
    _process_code_fence,
    _reflow_line,
    _squash_heading_parts,
    _strip_dependabot_metadata,
    postprocess,
    postprocess_text,
)


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


class TestDependabotMetadata:
    def test_strips_yaml_footer_from_commit_body(self) -> None:
        content = (
            "- Update setuptools [`abcdef0`](https://example.com/commit/abcdef0)\n\n"
            "  Updates setuptools to 83.0.0\n\n"
            "---\n\n"
            "  updated-dependencies:\n\n"
            "- dependency-name: setuptools\n"
            "  dependency-version: 83.0.0\n"
            "  dependency-type: direct:development\n"
            "  ...\n"
            "- Next entry\n"
        )

        assert _strip_dependabot_metadata(content) == (
            "- Update setuptools [`abcdef0`](https://example.com/commit/abcdef0)\n\n  Updates setuptools to 83.0.0\n\n- Next entry\n"
        )

    def test_preserves_unrelated_thematic_break(self) -> None:
        content = "- Entry\n\n---\n\nAdditional context\n"

        assert _strip_dependabot_metadata(content) == content

    def test_postprocess_text_strips_dependabot_footer(self) -> None:
        content = "- Update setuptools\n\n---\n\nupdated-dependencies:\n- dependency-name: setuptools\n  dependency-version: 83.0.0\n...\n\n- Next entry\n"

        result = postprocess_text(content)

        assert "updated-dependencies:" not in result
        assert result == "- Update setuptools\n- Next entry\n"

    def test_postprocess_text_preserves_dependabot_markers_inside_yaml_fence(self) -> None:
        fenced_example = "```yaml\n---\nupdated-dependencies:\n- dependency-name: setuptools\n...\n```"

        result = postprocess_text(f"Example metadata:\n\n{fenced_example}\n")

        assert fenced_example in result


class TestRumdlCompatibility:
    def test_removes_blank_between_peer_list_items(self) -> None:
        content = (
            "# Changelog\n\n"
            "## [1.0.0] - 2026-01-01\n\n"
            "### Added\n\n"
            "- First item\n"
            "  [`1111111`](https://github.com/acgetchell/delaunay/commit/1111111111111111111111111111111111111111)\n\n"
            "- Second item\n"
        )

        result = postprocess_text(content)

        assert "\n\n- Second item\n" not in result
        assert "\n- Second item\n" in result

    def test_removes_blank_after_nested_body_before_top_level_peer(self) -> None:
        content = "# Changelog\n\n## [1.0.0] - 2026-01-01\n\n### Added\n\n- First item\n\n  - Body point\n  - Body point\n\n- Second item\n"

        result = postprocess_text(content)

        assert "\n\n- Second item\n" not in result

    def test_preserves_blank_before_first_nested_item_after_parent_item(self) -> None:
        content = "# Changelog\n\n## [1.0.0] - 2026-01-01\n\n### Added\n\n- Parent item\n\n  - First nested detail\n"

        result = postprocess_text(content)

        assert "- Parent item\n\n  - First nested detail\n" in result

    def test_does_not_add_blank_before_peer_item_with_body(self) -> None:
        content = (
            "# Changelog\n\n"
            "## [1.0.0] - 2026-01-01\n\n"
            "### Added\n\n"
            "- Simple item [`1111111`](https://github.com/acgetchell/delaunay/commit/1111111111111111111111111111111111111111)\n"
            "- Item with body [`2222222`](https://github.com/acgetchell/delaunay/commit/2222222222222222222222222222222222222222)\n\n"
            "  Body paragraph.\n"
        )

        result = postprocess_text(content)

        assert "\n- Simple item" in result
        assert "\n- Item with body" in result
        assert "\n\n- Item with body" not in result

    def test_normalizes_git_cliff_escaped_email_autolink(self) -> None:
        line = "  Co-Authored-By: Oz &lt;oz-agent@warp.dev&gt;"

        result = _normalize_email_autolinks(line)

        assert result == "  Co-Authored-By: Oz <oz-agent@warp.dev>"


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

    @pytest.mark.parametrize("token", ["[API](https://example.com),", "`is_valid()`,"])
    def test_preserves_punctuation_attached_to_atomic_markdown(self, token: str) -> None:
        line = f"- Read {token} then continue with enough text to force the line to wrap"

        result = _reflow_line(line, max_width=40)

        assert token in result
        assert token.replace(",", " ,") not in result

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

    def test_preserves_link_with_balanced_destination_parentheses(self) -> None:
        link = "[API](https://example.com/search(function(arg(nested))))"
        line = f"- Read the detailed publication API notes before continuing with the release process {link}"

        result = _reflow_line(line, max_width=60)

        assert link in result

    def test_preserves_multi_backtick_code_span(self) -> None:
        span = "``call(`inner`, value)``"
        line = f"- Use {span} when documenting the generated command and all of its arguments"

        result = _reflow_line(line, max_width=45)

        assert span in result


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


def _merged_pr_summary_block(text: str) -> str:
    """Return the injected merged-PR summary block."""
    start = text.index("### Merged Pull Requests")
    end = text.find("\n### ", start + len("### Merged Pull Requests"))
    return text[start:] if end == -1 else text[start:end]


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
        summary_block = _merged_pr_summary_block(result)
        assert "### Merged Pull Requests" in summary_block
        # PR entry in summary (without commit hash).
        assert f"- Feature A {_pr(10)}" in summary_block
        assert "- Plain commit" not in summary_block
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

    def test_injects_breaking_summary_from_marker_variants(self) -> None:
        content = self._changelog(f"- **BREAKING** Big change {_pr(5)} {_commit()}")
        result = _inject_summary_sections(content)

        assert "### ⚠️ Breaking Changes" in result
        assert f"- Big change {_pr(5)}" in result

    def test_injects_summary_from_star_bullets(self) -> None:
        content = self._changelog(f"* [**breaking**] Star change {_pr(7)} {_commit()}")
        result = _inject_summary_sections(content)

        assert "### ⚠️ Breaking Changes" in result
        assert "### Merged Pull Requests" in result
        assert f"* Star change {_pr(7)}" in result

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
        summary_block = _merged_pr_summary_block(result)
        assert f"- Feature {_pr(10)} {_pr(20)}" in summary_block

    def test_duplicate_summary_entries_are_collapsed(self) -> None:
        entry = f"- Feature A {_pr(10)} {_commit()}"
        content = self._changelog(f"{entry}\n{entry}")
        result = _inject_summary_sections(content)
        summary_block = _merged_pr_summary_block(result)
        assert summary_block.count(f"- Feature A {_pr(10)}") == 1


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
        assert _normalize_indented_heading("  ## Correctness Fixes") == "#### Correctness Fixes"

    def test_indented_atx_closing_sequence_becomes_bold_prose(self) -> None:
        assert _normalize_indented_heading("  ### API Design ###") == "#### API Design"

    def test_column_zero_changelog_heading_is_preserved(self) -> None:
        assert _normalize_indented_heading("### Added") == "### Added"

    def test_column_zero_entry_heading_becomes_level_four(self) -> None:
        assert _normalize_entry_heading("## Duplicate Vertex Handling", "parent") == ("#### Duplicate Vertex Handling")

    def test_column_zero_category_heading_is_preserved(self) -> None:
        assert _normalize_entry_heading("### Fixed", "parent") == "### Fixed"

    def test_padded_column_zero_category_heading_is_preserved(self) -> None:
        assert _normalize_entry_heading("### Fixed ###", "parent") == "### Fixed ###"

    def test_archives_heading_is_preserved(self) -> None:
        assert _normalize_entry_heading("## Archives", "parent") == "## Archives"

    def test_bracketed_entry_heading_becomes_level_four(self) -> None:
        assert _normalize_entry_heading("## [Notes]", "parent") == "#### [Notes]"

    def test_bracketed_version_like_entry_heading_becomes_level_four(self) -> None:
        assert _normalize_entry_heading("## [1.2.3 Notes]", "parent") == "#### [1.2.3 Notes]"

    def test_release_heading_is_preserved(self) -> None:
        assert _normalize_entry_heading("## [v1.2.3] - 2026-05-22", "parent") == "## [v1.2.3] - 2026-05-22"

    def test_prerelease_heading_is_preserved(self) -> None:
        assert _normalize_entry_heading("## [1.2.3-rc.1+build.7]", "parent") == "## [1.2.3-rc.1+build.7]"

    def test_contextual_category_heading_becomes_level_four(self) -> None:
        assert _normalize_entry_heading("### Fixed: Add rollback", "parent") == "#### Fixed: Add rollback"

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
        assert "#### Correctness Fixes" in result
        assert "#### API Design" in result
        assert "### Performance" in result

    def test_full_pipeline_normalizes_unindented_commit_body_headings(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        f.write_text(
            "# Changelog\n\n"
            "## [1.0.0]\n\n"
            "### Fixed\n\n"
            "- Handle degenerate configurations [#116](https://github.com/acgetchell/delaunay/pull/116)\n"
            f"  {_commit()}\n\n"
            "## Duplicate Vertex Handling\n\n"
            "  - Add duplicate coordinate detection\n\n"
            "### Fixed: Add rollback on cell creation failure\n\n"
            "  Add rollback mechanisms when cell creation fails.\n",
            encoding="utf-8",
        )

        postprocess(f)

        result = f.read_text(encoding="utf-8")
        assert "\n## Duplicate Vertex Handling" not in result
        assert "\n### Fixed: Add rollback on cell creation failure" not in result
        assert "#### Duplicate Vertex Handling" in result
        assert "#### Fixed: Add rollback on cell creation failure" in result
        assert "### Fixed" in result

    def test_full_pipeline_code_spans_wildcard_identifiers_in_headings(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        f.write_text(
            "# Changelog\n\n"
            "## [1.0.0]\n\n"
            "### Documentation\n\n"
            "- Rename bounding-box helpers\n\n"
            "### Documentation: Rename saturating_* helpers to bbox_* and clarify float semantics\n",
            encoding="utf-8",
        )

        postprocess(f)

        result = f.read_text(encoding="utf-8")
        assert "#### Documentation: Rename `saturating_*` helpers to `bbox_*` and clarify float semantics" in result

    def test_full_pipeline_normalizes_indented_bold_entry_headings(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        f.write_text(
            "# Changelog\n\n"
            "## [1.0.0]\n\n"
            "### Performance\n\n"
            "- Improve duplicate handling\n\n"
            "  **Performance Optimization**\n\n"
            "  - First set of changes\n\n"
            "  **Performance Optimization**\n\n"
            "  - Additional changes\n",
            encoding="utf-8",
        )

        postprocess(f)

        result = f.read_text(encoding="utf-8")
        assert "  **Performance Optimization**" not in result
        assert "#### Performance Optimization\n" in result
        assert "#### Performance Improvements" in result


class TestSquashHeadingNormalization:
    """GitHub squash-body pseudo-commit headings are rendered as prose."""

    def test_plain_summary_removes_links_and_conventional_prefix(self) -> None:
        line = f"- fix: Improve benchmark output {_pr(42)} {_commit()}"
        assert _plain_summary(line) == "improve benchmark output"

    def test_plain_summary_removes_breaking_marker(self) -> None:
        line = f"- [**breaking**] feat!: Remove old API {_pr(42)} {_commit()}"
        assert _plain_summary(line) == "remove old api"

    def test_squash_heading_parts_maps_kind_to_changelog_label(self) -> None:
        assert _squash_heading_parts("  - perf(core): speed up predicates") == (
            "  ",
            "Performance",
            "Speed up predicates",
        )

    def test_squash_heading_parts_ignores_commit_entries(self) -> None:
        assert _squash_heading_parts(f"- fix: actual commit {_commit()}") is None

    def test_conventional_squash_heading_becomes_bold_prose(self) -> None:
        assert _normalize_squash_heading("- fix: close the 4D retry collapse") == "#### Fixed: Close the 4D retry collapse"

    def test_nested_squash_heading_is_indented(self) -> None:
        assert _normalize_squash_heading("- Changed: harden flip diagnostics", nested=True) == "#### Changed: Harden flip diagnostics"

    def test_commit_entry_is_preserved(self) -> None:
        line = f"- fix: actual commit {_commit()}"
        assert _normalize_squash_heading(line) == line

    def test_duplicate_squash_heading_matches_parent_summary(self) -> None:
        parent = _plain_summary("- Instrument large-scale 4D debugging")
        assert _is_duplicate_squash_heading("- feat: instrument large-scale 4D debugging", parent)

    def test_duplicate_squash_heading_rejects_distinct_heading(self) -> None:
        parent = _plain_summary("- Instrument large-scale 4D debugging")
        assert not _is_duplicate_squash_heading("- fix: close the 4D retry collapse", parent)

    def test_isolated_body_heading_requires_blank_neighbors(self) -> None:
        lines = ["- Parent entry", "", "- fix: child heading", "", "  - detail"]
        assert _is_isolated_body_heading(lines, 2)
        assert not _is_isolated_body_heading(lines, 4)

    def test_full_pipeline_drops_duplicate_squash_heading(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        f.write_text(
            "# Changelog\n\n"
            "## [1.0.0]\n\n"
            "### Added\n\n"
            f"- Instrument large-scale 4D debugging {_commit('3af976e', '3af976ec2f7c33d49803b24ab8f1a7da598fea0b')}\n\n"
            "* feat: instrument large-scale 4D debugging\n\n"
            "  - Thread cavity-touched cells through insertion.\n\n"
            "* fix: close the 4D bulk repair retry collapse\n\n"
            "  - Raise the D>=4 per-insertion repair budget.\n",
            encoding="utf-8",
        )

        postprocess(f)

        result = f.read_text(encoding="utf-8")
        assert "feat: instrument large-scale 4D debugging" not in result
        assert "#### Fixed: Close the 4D bulk repair retry collapse" in result
        assert "  - Thread cavity-touched cells through insertion." in result

    def test_full_pipeline_resets_parent_summary_at_version_heading(self, tmp_path: Path) -> None:
        """Version headings reset duplicate-squash tracking between releases."""
        f = tmp_path / "CHANGELOG.md"
        f.write_text(
            "# Changelog\n\n"
            "## [1.0.0]\n\n"
            "### Added\n\n"
            f"- Repeatable summary {_commit()}\n\n"
            "## [0.9.0]\n\n"
            "- fixed: repeatable summary\n\n"
            "  - Preserve this historical squash-body heading.\n",
            encoding="utf-8",
        )

        postprocess(f)

        result = f.read_text(encoding="utf-8")
        assert "#### Fixed: Repeatable summary" in result
        assert "  - Preserve this historical squash-body heading." in result

    def test_full_pipeline_preserves_non_isolated_conventional_bullets(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        f.write_text(
            "# Changelog\n\n"
            "## [1.0.0]\n\n"
            "### Documentation\n\n"
            f"- Update workflow docs {_commit()}\n\n"
            "  - Added: `just help-workflows` references throughout\n"
            "  - Expanded: Testing commands\n",
            encoding="utf-8",
        )

        postprocess(f)

        result = f.read_text(encoding="utf-8")
        assert "  - Added: `just help-workflows` references throughout" in result
        assert "#### Added: `just help-workflows`" not in result

    def test_full_pipeline_deindents_children_after_squash_heading(self) -> None:
        content = (
            "# Changelog\n\n"
            "## [1.0.0]\n\n"
            "### Added\n\n"
            f"- Identity-based SoS perturbation {_pr(272)} {_commit('a125d98', 'a125d98deadbeef')}\n\n"
            f"  - feat: Canonical vertex ordering details {_pr(266)}\n\n"
            "    - Add canonical_points module with sorted_cell_points helpers\n"
        )

        result = postprocess_text(content)

        assert f"#### Added: Canonical vertex ordering details {_pr(266)}" in result
        assert "\n  - Add canonical_points module with sorted_cell_points helpers\n" in result
        assert "\n    - Add canonical_points module" not in result

    def test_full_pipeline_mirrors_squash_body_heading_into_matching_section(self) -> None:
        content = (
            "# Changelog\n\n"
            "## [1.0.0]\n\n"
            "### Maintenance\n\n"
            f"- Replace Node markdown tooling with rumdl {_pr(394)} {_commit('5654d14', '5654d14deadbeef')}\n\n"
            "  * fix(tooling): align markdown lint policy\n\n"
            "    - Scope Codacy markdownlint to active Markdown docs.\n\n"
            "  * docs: refresh release guidance\n\n"
            "    - Document the changelog workflow.\n"
        )

        result = postprocess_text(content)

        assert f"- Align markdown lint policy {_commit('5654d14', '5654d14deadbeef')}" not in result
        assert "  - Scope Codacy markdownlint to active Markdown docs." in result
        assert f"- Refresh release guidance {_commit('5654d14', '5654d14deadbeef')}" not in result
        assert "- Replace Node markdown tooling with rumdl" in result
        assert "#### Fixed: Align markdown lint policy" in result
        assert "#### Documentation: Refresh release guidance" in result

    def test_full_pipeline_mirrors_squash_body_from_non_maintenance_section(self) -> None:
        content = (
            "# Changelog\n\n"
            "## [1.0.0]\n\n"
            "### Changed\n\n"
            f"- Refine insertion behavior {_commit('a6ec3fa', 'a6ec3fadeadbeef')}\n\n"
            "  * fix: add rollback on cell creation failure\n\n"
            "    - Keep failed insertions atomic.\n"
        )

        result = postprocess_text(content)

        assert f"- Add rollback on cell creation failure {_commit('a6ec3fa', 'a6ec3fadeadbeef')}" not in result
        assert "#### Fixed: Add rollback on cell creation failure" in result
        assert "  - Keep failed insertions atomic." in result

    def test_full_pipeline_stops_squash_children_at_star_top_level_entries(self) -> None:
        content = (
            "# Changelog\n\n"
            "## [1.0.0]\n\n"
            "### Maintenance\n\n"
            f"- Refresh tooling {_commit('1111111', '1111111deadbeef')}\n\n"
            "  * fix: align markdown lint policy\n\n"
            "    - Scope linting to active docs.\n\n"
            f"* Bump helper tools {_commit('2222222', '2222222deadbeef')}\n\n"
            "  - Keep dependency updates as a separate generated entry.\n"
        )

        result = postprocess_text(content)

        assert result.count("Bump helper tools") == 1
        assert "#### Fixed: Align markdown lint policy" in result
        assert result.index("#### Fixed: Align markdown lint policy") < result.index("- Bump helper tools")

    def test_full_pipeline_stops_squash_children_at_unicode_top_level_entries(self) -> None:
        content = (
            "# Changelog\n\n"
            "## [1.0.0]\n\n"
            "### Maintenance\n\n"
            f"- Refresh tooling {_commit('1111111', '1111111deadbeef')}\n\n"
            "  * docs: refresh release guidance\n\n"
            "    - Document the changelog workflow.\n\n"
            f"• Update helper docs {_commit('3333333', '3333333deadbeef')}\n\n"
            "  - Keep the unicode-list entry separate.\n"
        )

        result = postprocess_text(content)

        assert result.count("Update helper docs") == 1
        assert "#### Documentation: Refresh release guidance" in result
        assert result.index("#### Documentation: Refresh release guidance") < result.index("- Update helper docs")

    def test_full_pipeline_keeps_dependencies_heading_as_category_boundary(self) -> None:
        content = (
            "# Changelog\n\n"
            "## [1.0.0]\n\n"
            "### Fixed\n\n"
            f"- Fix parser boundaries {_commit('4444444', '4444444deadbeef')}\n\n"
            "### Dependencies\n\n"
            f"- Bump helper crate {_commit('5555555', '5555555deadbeef')}\n"
        )

        result = postprocess_text(content)

        assert "\n### Dependencies\n" in result
        assert "\n#### Dependencies\n" not in result

    def test_full_pipeline_does_not_mirror_duplicate_parent_heading(self) -> None:
        content = (
            "# Changelog\n\n"
            "## [1.0.0]\n\n"
            "### Added\n\n"
            f"- Instrument large-scale debugging {_commit('3af976e', '3af976edeadbeef')}\n\n"
            "  * feat: instrument large-scale debugging\n\n"
            "    - Keep this detail under the parent entry.\n"
        )

        result = postprocess_text(content)

        assert result.count("- Instrument large-scale debugging") == 1
        assert "#### Added: Instrument large-scale debugging" not in result
        assert "  - Keep this detail under the parent entry." in result

    def test_full_pipeline_canonicalizes_feature_names_before_deduplication(self) -> None:
        content = (
            "# Changelog\n\n"
            "## [1.0.0]\n\n"
            "### Fixed\n\n"
            f"- Gate diagnostics behind test-debug {_commit('5d8a9ff', '5d8a9ffdeadbeef')}\n\n"
            "  - Replace profile-based public diagnostic exports with the documented test-debug feature.\n"
        )

        result = postprocess_text(content)

        assert "test-debug" not in result
        assert "Gate public diagnostic exports behind diagnostics" in result
        assert "documented diagnostics feature" in result

    def test_full_pipeline_prefers_contextual_duplicate_squash_body_entry(self) -> None:
        content = (
            "# Changelog\n\n"
            "## [1.0.0]\n\n"
            "### Changed\n\n"
            f"- Avoid panic in stack-matrix dispatch {_commit('3ac8b11', '3ac8b11deadbeef')}\n\n"
            "  - Add StackMatrixDispatchError.\n\n"
            "### Performance\n\n"
            f"- Migrate geometry predicates to la-stack {_commit('3ac8b11', '3ac8b11deadbeef')}\n\n"
            "#### Fixed: Avoid panic in stack-matrix dispatch\n\n"
            "- Add StackMatrixDispatchError.\n"
        )

        result = postprocess_text(content)

        assert result.count("Avoid panic in stack-matrix dispatch") == 1
        assert "#### Fixed: Avoid panic in stack-matrix dispatch" in result
        assert "### Changed\n\n### Performance" not in result

    def test_full_pipeline_keeps_distinct_entries_with_same_title(self) -> None:
        content = (
            "# Changelog\n\n"
            "## [1.0.0]\n\n"
            "### Fixed\n\n"
            f"- Clarify diagnostics {_commit('aaaaaaa', 'aaaaaaadeadbeef')}\n\n"
            "  - Keep the public docs wording.\n\n"
            "### Maintenance\n\n"
            f"- Refresh release notes {_commit('aaaaaaa', 'aaaaaaadeadbeef')}\n\n"
            "#### Fixed: Clarify diagnostics\n\n"
            "- Keep the internal test comments aligned.\n"
        )

        result = postprocess_text(content)

        assert result.count("Clarify diagnostics") == 2
        assert "Keep the public docs wording." in result
        assert "Keep the internal test comments aligned." in result

    def test_full_pipeline_strips_followup_suffix_after_duplicate_removal(self) -> None:
        content = (
            "# Changelog\n\n"
            "## [1.0.0]\n\n"
            "### Changed\n\n"
            f"- Improve insertion errors {_commit('a6ec3fa', 'a6ec3fadeadbeef')}\n\n"
            "#### Early Exit for Empty Input\n\n"
            "- Return a clear empty-handle error.\n\n"
            "### Fixed\n\n"
            f"- Handle degenerate insertion {_commit('a6ec3fa', 'a6ec3fadeadbeef')}\n\n"
            "#### Changed: Improve insertion errors\n\n"
            "#### Early Exit for Empty Input - Follow-up\n\n"
            "- Return a clear empty-handle error.\n"
        )

        result = postprocess_text(content)

        assert "- Improve insertion errors" not in result
        assert "#### Early Exit for Empty Input - Follow-up" not in result
        assert "#### Early Exit for Empty Input" in result

    def test_full_pipeline_preserves_distinct_followup_heading_without_duplicate_removal(self) -> None:
        content = (
            "# Changelog\n\n"
            "## [1.0.0]\n\n"
            "### Documentation\n\n"
            f"- Refresh release notes {_commit('bbbbbbb', 'bbbbbbbdeadbeef')}\n\n"
            "#### Migration Guide - Follow-up\n\n"
            "- Clarify the manual release checklist.\n"
        )

        result = postprocess_text(content)

        assert "#### Migration Guide - Follow-up" in result

    def test_full_pipeline_deduplicates_real_archive_contextual_excerpt(self) -> None:
        content = (
            "# Changelog\n\n"
            "## [0.6.0]\n\n"
            "### Changed\n\n"
            f"- Reduce duplication and clarify tolerance/degeneracy docs {_commit('3ac8b11', '3ac8b11deadbeef')}\n\n"
            "  - Compute adaptive tolerance before consuming matrices for determinant evaluation.\n"
            "  - Clarify CoordinateConversionError::ConversionFailed coordinate_index semantics.\n\n"
            "### Fixed\n\n"
            f"- Avoid panic in stack-matrix dispatch {_commit('3ac8b11', '3ac8b11deadbeef')}\n\n"
            "  - Add StackMatrixDispatchError (UnsupportedDim/La) and try_with_la_stack_matrix! helper.\n"
            "  - Switch predicate/utility call sites to fallible dispatch.\n\n"
            "### Performance\n\n"
            f"- Migrate geometry predicates to la-stack {_commit('3ac8b11', '3ac8b11deadbeef')}\n\n"
            "  - Replace nalgebra-backed dynamic matrices with la-stack fixed-size matrices.\n\n"
            "#### Fixed: Avoid panic in stack-matrix dispatch\n\n"
            "- Add StackMatrixDispatchError (UnsupportedDim/La) and try_with_la_stack_matrix! helper.\n"
            "- Switch predicate/utility call sites to fallible dispatch.\n\n"
            "#### Changed: Reduce duplication and clarify tolerance/degeneracy docs\n\n"
            "- Compute adaptive tolerance before consuming matrices for determinant evaluation.\n"
            "- Clarify CoordinateConversionError::ConversionFailed coordinate_index semantics.\n"
        )

        result = postprocess_text(content)

        assert result.count("Avoid panic in stack-matrix dispatch") == 1
        assert result.count("Reduce duplication and clarify tolerance/degeneracy docs") == 1
        assert "#### Fixed: Avoid panic in stack-matrix dispatch" in result
        assert "#### Changed: Reduce duplication and clarify tolerance/degeneracy docs" in result
        assert "\n### Changed\n" not in result
        assert "\n### Fixed\n" not in result


class TestCodeBlockLanguage:
    def test_process_code_fence_opens_and_tags_bare_fence(self) -> None:
        result: list[str] = []

        handled, active_fence = _process_code_fence("```", result, active_fence=None, next_line="let x = 1;")

        assert handled
        assert active_fence == _CodeFence(delimiter="`", length=3)
        assert result == ["```text"]

    def test_process_code_fence_closes_existing_block(self) -> None:
        result: list[str] = []

        handled, active_fence = _process_code_fence(
            "```",
            result,
            active_fence=_CodeFence(delimiter="`", length=3),
            next_line=None,
        )

        assert handled
        assert active_fence is None
        assert result == ["```"]

    def test_process_code_fence_adds_blank_after_closing_fence(self) -> None:
        result: list[str] = []

        handled, active_fence = _process_code_fence(
            "```",
            result,
            active_fence=_CodeFence(delimiter="`", length=3),
            next_line="following prose",
        )

        assert handled
        assert active_fence is None
        assert result == ["```", ""]

    def test_process_code_fence_ignores_regular_line(self) -> None:
        result: list[str] = []

        handled, active_fence = _process_code_fence("regular text", result, active_fence=None, next_line=None)

        assert not handled
        assert active_fence is None
        assert result == []

    def test_tilde_fence_is_supported_and_tagged(self) -> None:
        result: list[str] = []

        handled, active_fence = _process_code_fence("~~~~", result, active_fence=None, next_line="code")

        assert handled
        assert active_fence == _CodeFence(delimiter="~", length=4)
        assert result == ["~~~~text"]

    def test_shorter_or_different_delimiter_does_not_close_fence(self) -> None:
        active = _CodeFence(delimiter="`", length=4)

        handled_short, still_active = _process_code_fence("```", [], active_fence=active, next_line=None)
        handled_tilde, still_active = _process_code_fence("~~~~", [], active_fence=still_active, next_line=None)
        handled_close, closed = _process_code_fence("`````", [], active_fence=still_active, next_line=None)

        assert not handled_short
        assert not handled_tilde
        assert still_active == active
        assert handled_close
        assert closed is None

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

    def test_no_reflow_or_heading_rewrite_inside_tilde_block(self, tmp_path: Path) -> None:
        long_code = "## " + "code " * 50
        f = tmp_path / "CHANGELOG.md"
        f.write_text(
            f"# Changelog\n\n## [1.0.0]\n\n### Fixed\n\n- Parent entry\n\n~~~markdown\n{long_code.rstrip()}\n~~~\n## Outside\n",
            encoding="utf-8",
        )

        postprocess(f)

        result = f.read_text(encoding="utf-8")
        assert long_code.rstrip() in result
        assert "~~~markdown" in result
        assert "#### Outside" in result

    def test_adds_blank_after_code_block_before_prose(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        f.write_text("```text\ncode\n```\nfollowing prose\n", encoding="utf-8")

        postprocess(f)

        assert f.read_text(encoding="utf-8") == "```text\ncode\n```\n\nfollowing prose\n"


class TestIntegration:
    def test_atomic_replace_preserves_file_mode(self, tmp_path: Path) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text("# Changelog\n\n* Original entry\n", encoding="utf-8")
        changelog.chmod(0o640)

        postprocess(changelog)

        assert stat.S_IMODE(changelog.stat().st_mode) == 0o640

    def test_failed_atomic_replace_preserves_original(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        original = "# Changelog\n\n* Original entry\n"
        changelog.write_text(original, encoding="utf-8")

        def fail_replace(_source: Path, _target: Path) -> Path:
            message = "simulated replacement failure"
            raise OSError(message)

        monkeypatch.setattr(Path, "replace", fail_replace)

        with pytest.raises(OSError, match="simulated replacement failure"):
            postprocess(changelog)

        assert changelog.read_text(encoding="utf-8") == original
        assert list(tmp_path.glob(".CHANGELOG.md.*.tmp")) == []

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
                assert len(line) <= 160 or "](" in line or "http" in line, f"Line too long ({len(line)}): {line[:80]}..."

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
