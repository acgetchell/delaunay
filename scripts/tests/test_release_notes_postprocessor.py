"""Tests for changelog_utils.py release-notes post-processing.

These focus on the heuristics in _ReleaseNotesPostProcessor so that small refactors
or changelog format tweaks don't silently change output semantics.
"""

from __future__ import annotations

import re

import changelog_utils


def _section_body(lines: list[str], header: str) -> list[str]:
    """Return lines between a section header and the next section/release header."""
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == header)
    except StopIteration as exc:  # pragma: no cover
        msg = f"Section header not found: {header!r}"
        raise AssertionError(msg) from exc

    body: list[str] = []
    i = start + 1
    while i < len(lines) and not lines[i].startswith("### ") and not lines[i].startswith("## "):
        body.append(lines[i])
        i += 1

    return body


class TestReleaseNotesPostProcessor:
    def test_breaking_changes_moved_out_of_changed_section(self):
        content = """# Changelog

## [v1.0.0] - 2025-01-01

### Changed

- **Bump MSRV to 1.92.0** [`abcdef123`](https://github.com/acgetchell/delaunay/commit/abcdef123)
  Details that should not appear in the breaking-changes summary.

- **Regular change** [`123456789`](https://github.com/acgetchell/delaunay/commit/123456789)

### Added

- **Some new feature** [`999999999`](https://github.com/acgetchell/delaunay/commit/999999999)
"""

        processed = changelog_utils._ReleaseNotesPostProcessor.process(content)  # noqa: SLF001
        lines = processed.splitlines()

        assert "### ⚠️ Breaking Changes" in processed

        breaking_start = lines.index("### ⚠️ Breaking Changes")
        changed_start = lines.index("### Changed")
        assert breaking_start < changed_start

        breaking_body = _section_body(lines, "### ⚠️ Breaking Changes")
        assert any("MSRV" in line for line in breaking_body)

        changed_body = _section_body(lines, "### Changed")
        assert not any("MSRV" in line for line in changed_body)

        # The breaking section should not include verbose body details.
        assert not any("Details that should not appear" in line for line in breaking_body)

    def test_commit_links_attached_to_top_level_only(self):
        top_sha = "1234567890123456789012345678901234567890"  # 40 chars
        nested_sha = "9876543210987654321098765432109876543210"  # 40 chars

        content = f"""# Changelog

## v1.0.0

### Added

- **Top level item**
  [`{top_sha}`](https://github.com/acgetchell/delaunay/commit/{top_sha})

  - Nested item
    [`{nested_sha}`](https://github.com/acgetchell/delaunay/commit/{nested_sha})
"""

        processed = changelog_utils._ReleaseNotesPostProcessor.process(content)  # noqa: SLF001

        # Top-level bullet should keep an attached (shortened) commit link.
        assert "`1234567`" in processed

        # Nested bullets should not include commit links (they're dropped).
        assert "`9876543`" not in processed

        # No standalone commit-link-only lines should remain.
        commit_only_re = re.compile(r"^\s*\[`[a-f0-9]{7,40}`\]\([^)]*\)\s*$")
        assert not any(commit_only_re.match(line) for line in processed.splitlines())

    def test_added_section_consolidates_many_entries_from_same_commit(self):
        sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"  # 40 chars
        url = f"https://github.com/acgetchell/delaunay/commit/{sha}"

        content = f"""# Changelog

## v1.0.0

### Added

- **feat: Introduce new API surface** [`{sha}`]({url})
- **Add helper function** [`{sha}`]({url})
- **Add another helper** [`{sha}`]({url})
- **Add tests for new API** [`{sha}`]({url})
"""

        processed = changelog_utils._ReleaseNotesPostProcessor.process(content)  # noqa: SLF001
        lines = processed.splitlines()

        added_body = _section_body(lines, "### Added")
        top_level_bullets = [line for line in added_body if line.startswith("- **")]
        assert len(top_level_bullets) == 1

        # Ensure we preserved the details as nested items.
        assert "Add helper function" in processed
        assert "Add another helper" in processed
        assert "Add tests for new API" in processed

        # Bucket structure should be present.
        assert "  - Tests" in processed
        assert "  - Other" in processed

    def test_wording_normalization_replacements(self):
        content = """# Changelog

## v1.0.0

### Changed

- **Fix retriable dets edge case** [`beefbeef0`](https://github.com/acgetchell/delaunay/commit/beefbeef0)
"""

        processed = changelog_utils._ReleaseNotesPostProcessor.process(content)  # noqa: SLF001

        assert "retryable" in processed
        assert "determinants" in processed
        assert "retriable" not in processed
        assert re.search(r"\bdets\b", processed, flags=re.IGNORECASE) is None
