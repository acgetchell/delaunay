"""Tests for archive_changelog.py — parsing, grouping, split/archive, and idempotency."""

import logging
import os
import stat
import sys
from typing import TYPE_CHECKING

import pytest

import archive_changelog as archive_changelog_module
from archive_changelog import (
    _extract_link_defs,
    _format_link_defs,
    _minor_key,
    _version_sort_key,
    _write_text_atomic,
    archive_changelog,
    build_root,
    group_by_minor,
    main,
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

    def test_malformed_single_component(self) -> None:
        with pytest.raises(ValueError, match="at least two components"):
            _minor_key("1")

    def test_malformed_empty_string(self) -> None:
        with pytest.raises(ValueError, match="at least two components"):
            _minor_key("")


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

    def test_prerelease_labels_stay_semantic(self) -> None:
        labels = ["1.2.3", "1.2.3-rc.10", "1.2.3-rc.2", "1.2.3-alpha.1", "unreleased"]
        assert sorted(labels, key=_version_sort_key) == [
            "1.2.3-alpha.1",
            "1.2.3-rc.2",
            "1.2.3-rc.10",
            "1.2.3",
            "unreleased",
        ]

    def test_build_metadata_is_ignored_for_sorting(self) -> None:
        labels = [
            "1.2.3-rc.1+build.7",
            "1.2.3+build.7",
            "1.2.3-alpha.1+build.7",
        ]
        assert sorted(labels, key=_version_sort_key) == [
            "1.2.3-alpha.1+build.7",
            "1.2.3-rc.1+build.7",
            "1.2.3+build.7",
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
        parsed = parse_changelog(_full_changelog())
        assert "# Changelog" in parsed.preamble
        assert parsed.unreleased is not None
        assert "Unreleased" in parsed.unreleased
        assert len(parsed.version_blocks) == 5
        assert parsed.version_blocks[0][0] == "0.7.2"
        assert parsed.version_blocks[-1][0] == "0.2.0"

    def test_no_headings(self) -> None:
        parsed = parse_changelog("Just some text\n")
        assert parsed.preamble == "Just some text\n"
        assert parsed.unreleased is None
        assert parsed.version_blocks == ()

    def test_no_unreleased(self) -> None:
        text = _PREAMBLE + _V072 + _V071
        parsed = parse_changelog(text)
        assert parsed.unreleased is None
        assert len(parsed.version_blocks) == 2

    def test_accepts_semver_prerelease_and_build_metadata(self) -> None:
        text = _PREAMBLE + "## [1.2.3-rc.2+build.7] - 2026-03-10\n\n- Candidate\n"
        parsed = parse_changelog(text)
        assert parsed.version_blocks[0][0] == "1.2.3-rc.2+build.7"

    def test_rejects_unknown_bracketed_heading(self) -> None:
        text = _PREAMBLE + _V072 + "## [CustomLabel]\n\n- Something\n\n" + _V071
        with pytest.raises(ValueError, match=r"Unrecognized changelog heading at line \d+: '## \[CustomLabel\]'"):
            parse_changelog(text)

    @pytest.mark.parametrize(
        "heading",
        [
            "## [01.2.3] - 2026-03-10",
            "## [1.2.3-01] - 2026-03-10",
            "## [1.2.3oops] - 2026-03-10",
            "## [1.2] - 2026-03-10",
        ],
    )
    def test_rejects_invalid_semver_heading(self, heading: str) -> None:
        with pytest.raises(ValueError, match="Unrecognized changelog heading"):
            parse_changelog(f"{_PREAMBLE}{heading}\n\n- Invalid release\n")

    def test_rejects_invalid_release_date(self) -> None:
        text = _PREAMBLE + "## [0.7.2] - 2026-02-30\n\n- Invalid date\n"
        with pytest.raises(ValueError, match=r"Invalid release date.*'2026-02-30'"):
            parse_changelog(text)

    def test_rejects_duplicate_unreleased_heading(self) -> None:
        with pytest.raises(ValueError, match="Duplicate Unreleased heading"):
            parse_changelog(_PREAMBLE + _UNRELEASED + _UNRELEASED + _V072)

    def test_rejects_unreleased_heading_after_release(self) -> None:
        with pytest.raises(ValueError, match="Unreleased heading must be the first"):
            parse_changelog(_PREAMBLE + _V072 + _UNRELEASED)

    def test_rejects_duplicate_release_heading(self) -> None:
        with pytest.raises(ValueError, match=r"Duplicate release heading '0\.7\.2'"):
            parse_changelog(_PREAMBLE + _V072 + _V072)

    def test_rejects_release_headings_out_of_order(self) -> None:
        with pytest.raises(ValueError, match=r"'0\.7\.2' must be older than preceding '0\.7\.1'"):
            parse_changelog(_PREAMBLE + _V071 + _V072)


class TestGroupByMinor:
    def test_groups_correctly(self) -> None:
        parsed = parse_changelog(_full_changelog())
        groups = group_by_minor(parsed.version_blocks)
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
        parsed = parse_changelog(_full_changelog())
        groups = group_by_minor(parsed.version_blocks)
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
        parsed = parse_changelog(_full_changelog())
        groups = group_by_minor(parsed.version_blocks)
        path = write_archive(tmp_path, "0.6", groups["0.6"], link_defs)
        content = path.read_text(encoding="utf-8")
        # Should include only 0.6.x link defs.
        assert "[0.6.2]:" in content
        assert "[0.6.1]:" in content
        # Should NOT include defs for other versions.
        assert "[0.7.2]:" not in content
        assert "[unreleased]:" not in content
        assert "[0.2.0]:" not in content

    def test_postprocesses_archived_blocks(self, tmp_path: Path) -> None:
        block = (
            "## [0.5.0] - 2025-01-01\n\n"
            "### Fixed\n\n"
            "- Fix remove_vertex topology consistency [#124](https://github.com/acgetchell/delaunay/pull/124)\n"
            "  [`da473c8`](https://github.com/acgetchell/delaunay/commit/da473c8deadbeef)\n\n"
            "  This commit addresses three critical issues:\n\n"
            "  1. **Fix remove_vertex to maintain topology consistency**\n\n"
            "    - Added logic to clear dangling neighbor references\n"
        )

        path = write_archive(tmp_path, "0.5", [("0.5.0", block)])
        content = path.read_text(encoding="utf-8")

        assert "\n  - Added logic to clear dangling neighbor references\n" in content
        assert "\n    - Added logic to clear dangling neighbor references\n" not in content


class TestBuildRoot:
    def test_includes_active_and_archives(self) -> None:
        parsed = parse_changelog(_full_changelog())
        groups = group_by_minor(parsed.version_blocks)
        root = build_root(
            parsed.preamble,
            parsed.unreleased,
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
        root = build_root("# H\n", None, [("1.0.0", _V072)], [], "archive")
        assert "## Archives" not in root

    def test_no_link_defs_by_default(self) -> None:
        """build_root does not emit link defs (handled by orchestrator)."""
        parsed = parse_changelog(_full_changelog())
        groups = group_by_minor(parsed.version_blocks)
        root = build_root(
            parsed.preamble,
            parsed.unreleased,
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

    def test_archive_dir_outside_changelog_tree_uses_relative_link(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        changelog_dir = tmp_path / "repo"
        changelog_dir.mkdir()
        changelog = changelog_dir / "CHANGELOG.md"
        changelog.write_text(_full_changelog(), encoding="utf-8")
        archive_dir = tmp_path / "outside" / "archive"

        with caplog.at_level(logging.WARNING, logger="archive_changelog"):
            archive_changelog(changelog, archive_dir)

        root = changelog.read_text(encoding="utf-8")
        assert "- [0.6.x](../outside/archive/0.6.md)" in root
        assert str(archive_dir) in caplog.text
        assert str(changelog_dir) in caplog.text

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

    def test_existing_archives_are_postprocessed(self, tmp_path: Path) -> None:
        """Older archive files are normalized even when they are not regenerated."""
        changelog = tmp_path / "CHANGELOG.md"
        text = _PREAMBLE + _UNRELEASED + _V072 + _V071
        changelog.write_text(text, encoding="utf-8")
        archive_dir = tmp_path / "docs" / "archive" / "changelog"
        archive_dir.mkdir(parents=True)
        archive = archive_dir / "0.5.md"
        archive.write_text(
            "# Changelog - 0.5.x\n\n"
            "## [0.5.3] - 2025-10-31\n\n"
            "### Fixed\n\n"
            "- Handle degenerate configurations [#116](https://github.com/acgetchell/delaunay/pull/116)\n"
            "  [`a6ec3fa`](https://github.com/acgetchell/delaunay/commit/a6ec3fadeadbeef)\n\n"
            "## Duplicate Vertex Handling\n\n"
            "- Add duplicate coordinate detection\n",
            encoding="utf-8",
        )

        archive_changelog(changelog, archive_dir)

        content = archive.read_text(encoding="utf-8")
        assert "\n## Duplicate Vertex Handling" not in content
        assert "#### Duplicate Vertex Handling" in content

    def test_atomic_replace_failure_preserves_original_and_cleans_temporary_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_path / "CHANGELOG.md"
        original = "# Changelog\n\nOriginal content.\n"
        path.write_text(original, encoding="utf-8")

        def reject_replace(_source: Path, _destination: Path) -> None:
            msg = "simulated publication failure"
            raise OSError(msg)

        monkeypatch.setattr(archive_changelog_module, "_replace_path", reject_replace)

        with pytest.raises(OSError, match="simulated publication failure"):
            _write_text_atomic(path, "# Changelog\n\nReplacement content.\n")

        assert path.read_text(encoding="utf-8") == original
        assert list(tmp_path.glob(".CHANGELOG.md.*.tmp")) == []

    def test_backup_staging_failure_preserves_original_and_cleans_staged_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_path / "CHANGELOG.md"
        original = "# Changelog\n\nOriginal content.\n"
        path.write_text(original, encoding="utf-8")

        def reject_backup(_path: Path) -> Path:
            msg = "simulated backup failure"
            raise OSError(msg)

        monkeypatch.setattr(archive_changelog_module, "_stage_backup", reject_backup)

        with pytest.raises(OSError, match="simulated backup failure"):
            _write_text_atomic(path, "# Changelog\n\nReplacement content.\n")

        assert path.read_text(encoding="utf-8") == original
        assert list(tmp_path.glob(".CHANGELOG.md.*.tmp")) == []
        assert list(tmp_path.glob(".CHANGELOG.md.*.bak")) == []

    @pytest.mark.skipif(
        os.name == "nt",
        reason="Windows does not expose POSIX owner/group/other permission bits",
    )
    @pytest.mark.parametrize("requested_umask", [0o000, 0o077])
    def test_new_file_creation_is_owner_only_regardless_of_umask(
        self,
        tmp_path: Path,
        requested_umask: int,
    ) -> None:
        path = tmp_path / "new.md"
        previous_umask = os.umask(requested_umask)
        try:
            _write_text_atomic(path, "Private until the caller decides otherwise.\n")
        finally:
            os.umask(previous_umask)

        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    @pytest.mark.parametrize("failure_position", [1, 2, 3])
    def test_publication_failure_rolls_back_every_output_and_allows_retry(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        failure_position: int,
    ) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        original_root = _full_changelog()
        changelog.write_text(original_root, encoding="utf-8")
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        existing_archive = archive_dir / "0.6.md"
        original_archive = "# Existing 0.6 archive\n"
        existing_archive.write_text(original_archive, encoding="utf-8")
        new_archive = archive_dir / "0.2.md"

        real_replace = archive_changelog_module._replace_path
        replacement_count = 0

        def fail_one_replacement(source: Path, destination: Path) -> None:
            nonlocal replacement_count
            replacement_count += 1
            if replacement_count == failure_position:
                msg = f"simulated publication failure at position {failure_position}"
                raise OSError(msg)
            real_replace(source, destination)

        with monkeypatch.context() as context:
            context.setattr(archive_changelog_module, "_replace_path", fail_one_replacement)
            with pytest.raises(OSError, match=f"failure at position {failure_position}"):
                archive_changelog(changelog, archive_dir)

        assert changelog.read_text(encoding="utf-8") == original_root
        assert existing_archive.read_text(encoding="utf-8") == original_archive
        assert not new_archive.exists()
        assert list(tmp_path.rglob("*.tmp")) == []
        assert list(tmp_path.rglob("*.bak")) == []

        archive_changelog(changelog, archive_dir)
        assert "## [0.6.2]" not in changelog.read_text(encoding="utf-8")
        assert "## [0.6.2]" in existing_archive.read_text(encoding="utf-8")
        assert "## [0.2.0]" in new_archive.read_text(encoding="utf-8")

    @pytest.mark.parametrize(
        ("invalid_text", "error_match"),
        [
            (
                _PREAMBLE + _UNRELEASED + _V072 + "## [CustomLabel]\n\n- Must survive\n\n" + _V062,
                "Unrecognized changelog heading",
            ),
            (
                _PREAMBLE + _UNRELEASED + _UNRELEASED + _V072 + _V062,
                "Duplicate Unreleased heading",
            ),
            (
                _PREAMBLE + _UNRELEASED + _V072 + _V072 + _V062,
                "Duplicate release heading",
            ),
            (
                _PREAMBLE + _UNRELEASED + _V071 + _V072 + _V062,
                "Release heading out of order",
            ),
            (
                _PREAMBLE + _UNRELEASED + "## [0.7.2] - 2026-02-30\n\n- Invalid date\n\n" + _V062,
                "Invalid release date",
            ),
            (
                _PREAMBLE + _V072 + _UNRELEASED + _V062,
                "Unreleased heading must be the first",
            ),
        ],
        ids=[
            "unknown-heading",
            "duplicate-unreleased",
            "duplicate-release",
            "out-of-order",
            "invalid-date",
            "misplaced-unreleased",
        ],
    )
    def test_invalid_changelog_is_rejected_before_any_file_changes(
        self,
        tmp_path: Path,
        invalid_text: str,
        error_match: str,
    ) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(invalid_text, encoding="utf-8")
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        existing_archive = archive_dir / "0.5.md"
        existing_content = "# Existing archive\n\n## Heading that would otherwise be normalized\n"
        existing_archive.write_text(existing_content, encoding="utf-8")

        with pytest.raises(ValueError, match=error_match):
            archive_changelog(changelog, archive_dir)

        assert changelog.read_text(encoding="utf-8") == invalid_text
        assert existing_archive.read_text(encoding="utf-8") == existing_content
        assert list(archive_dir.iterdir()) == [existing_archive]

    def test_cli_reports_invalid_changelog_without_traceback(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        invalid_text = _PREAMBLE + "## [CustomLabel]\n\n- Invalid release\n"
        changelog.write_text(invalid_text, encoding="utf-8")
        archive_dir = tmp_path / "archive"
        monkeypatch.setattr(
            sys,
            "argv",
            ["archive-changelog", str(changelog), "--archive-dir", str(archive_dir)],
        )

        with pytest.raises(SystemExit) as exit_info:
            main()

        captured = capsys.readouterr()
        assert exit_info.value.code == 1
        assert captured.out == ""
        assert captured.err.startswith(f"Error: {changelog}: Unrecognized changelog heading")
        assert "Traceback" not in captured.err
        assert changelog.read_text(encoding="utf-8") == invalid_text
        assert not archive_dir.exists()

    def test_cli_reports_rollback_exception_group_without_traceback(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(_full_changelog(), encoding="utf-8")
        archive_dir = tmp_path / "archive"
        rollback_failure = BaseExceptionGroup(
            "Changelog publication failed and rollback was incomplete",
            [OSError("publication failed"), OSError("rollback failed")],
        )

        def reject_archive(_changelog: Path, _archive_dir: Path | None) -> None:
            raise rollback_failure

        monkeypatch.setattr(archive_changelog_module, "archive_changelog", reject_archive)
        monkeypatch.setattr(
            sys,
            "argv",
            ["archive-changelog", str(changelog), "--archive-dir", str(archive_dir)],
        )

        with pytest.raises(SystemExit) as exit_info:
            main()

        captured = capsys.readouterr()
        assert exit_info.value.code == 1
        assert captured.out == ""
        assert captured.err.startswith(f"Error: {changelog}: Changelog publication failed")
        assert "Traceback" not in captured.err

    def test_cli_reraises_unhandled_exception_group_members(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(_full_changelog(), encoding="utf-8")
        archive_dir = tmp_path / "archive"
        mixed_failure = BaseExceptionGroup(
            "Changelog publication interrupted during rollback",
            [OSError("rollback failed"), KeyboardInterrupt()],
        )

        def interrupt_archive(_changelog: Path, _archive_dir: Path | None) -> None:
            raise mixed_failure

        monkeypatch.setattr(archive_changelog_module, "archive_changelog", interrupt_archive)
        monkeypatch.setattr(
            sys,
            "argv",
            ["archive-changelog", str(changelog), "--archive-dir", str(archive_dir)],
        )

        with pytest.raises(BaseExceptionGroup) as error_info:
            main()

        captured = capsys.readouterr()
        assert error_info.value.subgroup(KeyboardInterrupt) is not None
        assert error_info.value.subgroup(OSError) is None
        assert captured.err.startswith(f"Error: {changelog}: Changelog publication interrupted")

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

    def test_archive_dir_relpath_value_error_uses_absolute_fallback(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Archive splitting survives Windows-style relpath failures across drives."""
        changelog_dir = tmp_path / "repo"
        changelog_dir.mkdir()
        changelog = changelog_dir / "CHANGELOG.md"
        changelog.write_text(_full_changelog(), encoding="utf-8")
        archive_dir = tmp_path / "outside" / "archive"

        def raise_cross_drive_value_error(_path: Path, _start: Path) -> str:
            msg = "path is on mount 'D:', start on mount 'C:'"
            raise ValueError(msg)

        monkeypatch.setattr("archive_changelog.os.path.relpath", raise_cross_drive_value_error)

        with caplog.at_level(logging.WARNING, logger="archive_changelog"):
            archive_changelog(changelog, archive_dir)

        root = changelog.read_text(encoding="utf-8")
        assert f"- [0.6.x]({archive_dir.as_posix()}/0.6.md)" in root
        assert "path is on mount 'D:', start on mount 'C:'" in caplog.text
        assert str(archive_dir) in caplog.text
        assert str(changelog_dir) in caplog.text


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
