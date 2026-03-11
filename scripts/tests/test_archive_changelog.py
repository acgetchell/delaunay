"""Tests for archive_changelog.py — parsing, grouping, split/archive, and idempotency."""

from __future__ import annotations

from typing import TYPE_CHECKING

from archive_changelog import (
    _extract_link_defs,
    _format_link_defs,
    _minor_key,
    _version_sort_key,
    archive_changelog,
    build_root,
    group_by_minor,
    parse_changelog,
    write_archive,
)
from tag_release import _github_anchor, extract_changelog_section

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_PREAMBLE = "# Changelog\n\nAll notable changes to this project will be documented in this file.\n\n"

_UNRELEASED = "## [Unreleased]\n\n### Added\n\n- Something new\n\n"

_V072 = "## [0.7.2] - 2026-03-10\n\n### Fixed\n\n- Bug fix in 0.7.2\n\n"

_V071 = "## [0.7.1] - 2026-02-20\n\n### Changed\n\n- Change in 0.7.1\n\n"

_V062 = "## [0.6.2] - 2026-01-01\n\n### Maintenance\n\n- Bump dep in 0.6.2\n\n"

_V061 = "## [0.6.1] - 2025-12-17\n\n### Added\n\n- Feature in 0.6.1\n\n"

_V020 = "## [0.2.0] - 2024-09-13\n\n### Added\n\n- Initial release\n"

_LINK_DEFS = (
    "\n"
    "[unreleased]: https://github.com/acgetchell/delaunay/compare/v0.7.2..HEAD\n"
    "[0.7.2]: https://github.com/acgetchell/delaunay/compare/v0.7.1..v0.7.2\n"
    "[0.7.1]: https://github.com/acgetchell/delaunay/compare/v0.7.0..v0.7.1\n"
    "[0.6.2]: https://github.com/acgetchell/delaunay/compare/v0.6.1..v0.6.2\n"
    "[0.6.1]: https://github.com/acgetchell/delaunay/compare/v0.6.0..v0.6.1\n"
    "[0.2.0]: https://github.com/acgetchell/delaunay/tree/v0.2.0\n"
)


def _full_changelog() -> str:
    return _PREAMBLE + _UNRELEASED + _V072 + _V071 + _V062 + _V061 + _V020


def _full_changelog_with_links() -> str:
    return _full_changelog() + _LINK_DEFS


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestMinorKey:
    def test_simple(self) -> None:
        assert _minor_key("0.7.2") == "0.7"

    def test_prerelease(self) -> None:
        assert _minor_key("1.2.3-rc.1") == "1.2"

    def test_major(self) -> None:
        assert _minor_key("2.0.0") == "2.0"


class TestVersionSortKey:
    def test_numeric_ordering(self) -> None:
        labels = ["0.2.0", "0.10.0", "0.9.0", "0.7.2"]
        assert sorted(labels, key=_version_sort_key) == [
            "0.2.0",
            "0.7.2",
            "0.9.0",
            "0.10.0",
        ]

    def test_minor_keys(self) -> None:
        minors = ["0.2", "0.10", "0.9", "0.7"]
        assert sorted(minors, key=_version_sort_key, reverse=True) == [
            "0.10",
            "0.9",
            "0.7",
            "0.2",
        ]

    def test_unreleased_sorts_last(self) -> None:
        labels = ["0.7.2", "unreleased", "0.6.1"]
        assert sorted(labels, key=_version_sort_key) == [
            "0.6.1",
            "0.7.2",
            "unreleased",
        ]

    def test_reverse_unreleased_first(self) -> None:
        labels = ["0.7.2", "unreleased", "0.6.1"]
        assert sorted(labels, key=_version_sort_key, reverse=True) == [
            "unreleased",
            "0.7.2",
            "0.6.1",
        ]


class TestParseChangelog:
    def test_splits_preamble_unreleased_versions(self) -> None:
        preamble, unreleased, blocks = parse_changelog(_full_changelog())
        assert "# Changelog" in preamble
        assert "Unreleased" in unreleased
        assert len(blocks) == 5
        assert blocks[0][0] == "0.7.2"
        assert blocks[-1][0] == "0.2.0"

    def test_no_headings(self) -> None:
        preamble, unreleased, blocks = parse_changelog("Just some text\n")
        assert preamble == "Just some text\n"
        assert unreleased == ""
        assert blocks == []

    def test_no_unreleased(self) -> None:
        text = _PREAMBLE + _V072 + _V071
        _, unreleased, blocks = parse_changelog(text)
        assert unreleased == ""
        assert len(blocks) == 2


class TestGroupByMinor:
    def test_groups_correctly(self) -> None:
        _, _, blocks = parse_changelog(_full_changelog())
        groups = group_by_minor(blocks)
        assert list(groups.keys()) == ["0.7", "0.6", "0.2"]
        assert len(groups["0.7"]) == 2
        assert len(groups["0.6"]) == 2
        assert len(groups["0.2"]) == 1


class TestExtractLinkDefs:
    def test_extracts_trailing_defs(self) -> None:
        text = _full_changelog_with_links()
        cleaned, link_defs = _extract_link_defs(text)
        assert "unreleased" in link_defs
        assert "0.7.2" in link_defs
        assert "0.2.0" in link_defs
        assert len(link_defs) == 6
        # Cleaned text should not contain any link defs.
        assert "[unreleased]:" not in cleaned
        assert "[0.7.2]:" not in cleaned

    def test_no_link_defs(self) -> None:
        cleaned, link_defs = _extract_link_defs(_full_changelog())
        assert link_defs == {}
        assert cleaned == _full_changelog()

    def test_preserves_content_before_defs(self) -> None:
        text = _full_changelog_with_links()
        cleaned, _ = _extract_link_defs(text)
        # All version headings should still be present.
        assert "## [0.7.2]" in cleaned
        assert "## [0.2.0]" in cleaned
        assert "## [Unreleased]" in cleaned


class TestWriteArchive:
    def test_writes_archive_file(self, tmp_path: Path) -> None:
        _, _, blocks = parse_changelog(_full_changelog())
        groups = group_by_minor(blocks)
        path = write_archive(tmp_path, "0.6", groups["0.6"])
        assert path.name == "0.6.md"
        content = path.read_text(encoding="utf-8")
        assert content.startswith("# Changelog - 0.6.x\n")
        assert "## [0.6.2]" in content
        assert "## [0.6.1]" in content
        assert content.endswith("\n")

    def test_creates_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b"
        write_archive(nested, "0.2", [("0.2.0", _V020)])
        assert (nested / "0.2.md").is_file()

    def test_includes_relevant_link_defs(self, tmp_path: Path) -> None:
        _, link_defs = _extract_link_defs(_full_changelog_with_links())
        _, _, blocks = parse_changelog(_full_changelog())
        groups = group_by_minor(blocks)
        path = write_archive(tmp_path, "0.6", groups["0.6"], link_defs)
        content = path.read_text(encoding="utf-8")
        # Should include only 0.6.x link defs.
        assert "[0.6.2]:" in content
        assert "[0.6.1]:" in content
        # Should NOT include defs for other versions.
        assert "[0.7.2]:" not in content
        assert "[unreleased]:" not in content
        assert "[0.2.0]:" not in content


class TestBuildRoot:
    def test_includes_active_and_archives(self) -> None:
        preamble, unreleased, blocks = parse_changelog(_full_changelog())
        groups = group_by_minor(blocks)
        root = build_root(
            preamble,
            unreleased,
            groups["0.7"],
            sorted(["0.6", "0.2"], reverse=True),
            "docs/archive/changelog",
        )
        assert "## [Unreleased]" in root
        assert "## [0.7.2]" in root
        assert "## [0.7.1]" in root
        assert "## [0.6.2]" not in root
        assert "## Archives" in root
        assert "[0.6.x](docs/archive/changelog/0.6.md)" in root
        assert "[0.2.x](docs/archive/changelog/0.2.md)" in root

    def test_no_archives_when_empty(self) -> None:
        root = build_root("# H\n", "", [("1.0.0", _V072)], [], "archive")
        assert "## Archives" not in root

    def test_no_link_defs_by_default(self) -> None:
        """build_root does not emit link defs (handled by orchestrator)."""
        preamble, unreleased, blocks = parse_changelog(_full_changelog())
        groups = group_by_minor(blocks)
        root = build_root(
            preamble,
            unreleased,
            groups["0.7"],
            ["0.6", "0.2"],
            "docs/archive/changelog",
        )
        assert "[unreleased]:" not in root
        assert "[0.7.2]:" not in root


# ---------------------------------------------------------------------------
# Integration / workflow tests
# ---------------------------------------------------------------------------


class TestArchiveChangelog:
    def test_splits_and_archives(self, tmp_path: Path) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(_full_changelog(), encoding="utf-8")
        archive_dir = tmp_path / "docs" / "archive" / "changelog"

        archive_changelog(changelog, archive_dir)

        # Root should only contain 0.7.x + Archives.
        root = changelog.read_text(encoding="utf-8")
        assert "## [0.7.2]" in root
        assert "## [0.7.1]" in root
        assert "## [0.6.2]" not in root
        assert "## Archives" in root

        # Archive files should exist.
        assert (archive_dir / "0.6.md").is_file()
        assert (archive_dir / "0.2.md").is_file()

        # Archive content should be verbatim.
        a06 = (archive_dir / "0.6.md").read_text(encoding="utf-8")
        assert "## [0.6.2]" in a06
        assert "## [0.6.1]" in a06

    def test_idempotent(self, tmp_path: Path) -> None:
        """Running archive twice produces the same output."""
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(_full_changelog(), encoding="utf-8")
        archive_dir = tmp_path / "docs" / "archive" / "changelog"

        archive_changelog(changelog, archive_dir)
        first_root = changelog.read_text(encoding="utf-8")
        first_a06 = (archive_dir / "0.6.md").read_text(encoding="utf-8")

        archive_changelog(changelog, archive_dir)
        second_root = changelog.read_text(encoding="utf-8")
        second_a06 = (archive_dir / "0.6.md").read_text(encoding="utf-8")

        assert first_root == second_root
        assert first_a06 == second_a06

    def test_single_minor_no_op(self, tmp_path: Path) -> None:
        """When only one minor series exists, nothing is archived."""
        changelog = tmp_path / "CHANGELOG.md"
        text = _PREAMBLE + _UNRELEASED + _V072 + _V071
        changelog.write_text(text, encoding="utf-8")

        archive_changelog(changelog, tmp_path / "archive")
        # File should be unchanged.
        assert changelog.read_text(encoding="utf-8") == text
        assert not (tmp_path / "archive").exists()

    def test_no_versions_no_op(self, tmp_path: Path) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text("# Changelog\n\nNo versions yet.\n", encoding="utf-8")
        archive_changelog(changelog, tmp_path / "archive")
        assert changelog.read_text(encoding="utf-8") == "# Changelog\n\nNo versions yet.\n"

    def test_distributes_link_defs(self, tmp_path: Path) -> None:
        """Reference-style link definitions are distributed to the correct files."""
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(_full_changelog_with_links(), encoding="utf-8")
        archive_dir = tmp_path / "docs" / "archive" / "changelog"

        archive_changelog(changelog, archive_dir)

        root = changelog.read_text(encoding="utf-8")
        # Root should have active + unreleased link defs.
        assert "[unreleased]:" in root
        assert "[0.7.2]:" in root
        assert "[0.7.1]:" in root
        # Root should NOT have archived version defs.
        assert "[0.6.2]:" not in root
        assert "[0.2.0]:" not in root

        # Archive files should only contain their own version defs.
        a06 = (archive_dir / "0.6.md").read_text(encoding="utf-8")
        assert "[0.6.2]:" in a06
        assert "[0.6.1]:" in a06
        assert "[0.7.2]:" not in a06
        assert "[unreleased]:" not in a06

        a02 = (archive_dir / "0.2.md").read_text(encoding="utf-8")
        assert "[0.2.0]:" in a02
        assert "[0.7.2]:" not in a02

    def test_idempotent_with_link_defs(self, tmp_path: Path) -> None:
        """Idempotency holds when link definitions are present."""
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(_full_changelog_with_links(), encoding="utf-8")
        archive_dir = tmp_path / "docs" / "archive" / "changelog"

        archive_changelog(changelog, archive_dir)
        first_root = changelog.read_text(encoding="utf-8")
        first_a06 = (archive_dir / "0.6.md").read_text(encoding="utf-8")

        archive_changelog(changelog, archive_dir)
        assert changelog.read_text(encoding="utf-8") == first_root
        assert (archive_dir / "0.6.md").read_text(encoding="utf-8") == first_a06


# ---------------------------------------------------------------------------
# tag_release archive fallback
# ---------------------------------------------------------------------------


class TestFormatLinkDefs:
    def test_semver_ordering_with_double_digit_minor(self) -> None:
        """Versions like 0.10.x sort after 0.9.x, not before."""
        link_defs = {
            "0.10.0": "[0.10.0]: https://example.com/compare/v0.9.0..v0.10.0",
            "0.9.0": "[0.9.0]: https://example.com/compare/v0.8.0..v0.9.0",
            "0.7.10": "[0.7.10]: https://example.com/compare/v0.7.9..v0.7.10",
            "0.7.2": "[0.7.2]: https://example.com/compare/v0.7.1..v0.7.2",
        }
        labels = {"0.10.0", "0.9.0", "0.7.10", "0.7.2"}
        result = _format_link_defs(link_defs, labels)
        lines = result.split("\n")
        # Newest (0.10.0) first, then 0.9.0, 0.7.10, 0.7.2.
        assert lines[0].startswith("[0.10.0]:")
        assert lines[1].startswith("[0.9.0]:")
        assert lines[2].startswith("[0.7.10]:")
        assert lines[3].startswith("[0.7.2]:")

    def test_unreleased_sorts_first_in_reverse(self) -> None:
        link_defs = {
            "unreleased": "[unreleased]: https://example.com/compare/v0.7.2..HEAD",
            "0.7.2": "[0.7.2]: https://example.com/compare/v0.7.1..v0.7.2",
        }
        result = _format_link_defs(link_defs, {"unreleased", "0.7.2"})
        lines = result.split("\n")
        assert lines[0].startswith("[unreleased]:")
        assert lines[1].startswith("[0.7.2]:")


class TestTagReleaseArchiveFallback:
    def test_extract_from_archive(self, tmp_path: Path) -> None:
        """extract_changelog_section falls back to archive when version not in root."""
        # Write a root changelog without 0.6.x.
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(_PREAMBLE + _V072, encoding="utf-8")

        # Write an archive with 0.6.x.
        archive_dir = tmp_path / "docs" / "archive" / "changelog"
        archive_dir.mkdir(parents=True)
        (archive_dir / "0.6.md").write_text(
            "# Changelog - 0.6.x\n\n" + _V062 + _V061,
            encoding="utf-8",
        )

        body, source = extract_changelog_section(changelog, "0.6.2")
        assert "Bump dep in 0.6.2" in body
        assert source == archive_dir / "0.6.md"

    def test_extract_from_root_returns_root_source(self, tmp_path: Path) -> None:
        """extract_changelog_section returns the root changelog as source when found there."""
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(_PREAMBLE + _V072, encoding="utf-8")

        body, source = extract_changelog_section(changelog, "0.7.2")
        assert "Bug fix in 0.7.2" in body
        assert source == changelog

    def test_anchor_from_archive(self, tmp_path: Path) -> None:
        """_github_anchor falls back to archive for archived versions."""
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(_PREAMBLE + _V072, encoding="utf-8")

        archive_dir = tmp_path / "docs" / "archive" / "changelog"
        archive_dir.mkdir(parents=True)
        (archive_dir / "0.6.md").write_text(
            "# Changelog - 0.6.x\n\n" + _V062,
            encoding="utf-8",
        )

        anchor = _github_anchor(changelog, "0.6.2")
        assert "062" in anchor
