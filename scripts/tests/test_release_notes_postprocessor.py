"""Tests for changelog_utils.py release-notes post-processing.

These focus on the heuristics in _ReleaseNotesPostProcessor so that small refactors
or changelog format tweaks don't silently change output semantics.
"""

from __future__ import annotations

import re

import pytest

from changelog_utils import _ReleaseNotesPostProcessor


def _require(condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


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

        processed = _ReleaseNotesPostProcessor.process(content)
        lines = processed.splitlines()

        _require(
            "### ⚠️ Breaking Changes" in processed,
            "Expected breaking changes section to be present",
        )

        breaking_start = lines.index("### ⚠️ Breaking Changes")
        changed_start = lines.index("### Changed")
        _require(
            breaking_start < changed_start,
            "Expected breaking changes section to appear before the Changed section",
        )

        breaking_body = _section_body(lines, "### ⚠️ Breaking Changes")
        _require(
            any("MSRV" in line for line in breaking_body),
            "Expected MSRV entry to be present in breaking changes section",
        )

        changed_body = _section_body(lines, "### Changed")
        _require(
            not any("MSRV" in line for line in changed_body),
            "Expected MSRV entry to be removed from Changed section",
        )

        # The breaking section should not include verbose body details.
        _require(
            not any("Details that should not appear" in line for line in breaking_body),
            "Expected verbose breaking-change body details to be trimmed",
        )

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

        processed = _ReleaseNotesPostProcessor.process(content)

        # Top-level bullet should keep an attached (shortened) commit link.
        _require(
            "`1234567`" in processed,
            "Expected top-level bullet to keep an attached (shortened) commit link",
        )

        # Nested bullets should not include commit links (they're dropped).
        _require(
            "`9876543`" not in processed,
            "Expected nested bullet commit links to be dropped",
        )

        # No standalone commit-link-only lines should remain.
        commit_only_re = re.compile(r"^\s*\[`[a-f0-9]{7,40}`\]\([^)]*\)\s*$")
        _require(
            not any(commit_only_re.match(line) for line in processed.splitlines()),
            "Expected no standalone commit-link-only lines to remain",
        )

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

        processed = _ReleaseNotesPostProcessor.process(content)
        lines = processed.splitlines()

        added_body = _section_body(lines, "### Added")
        top_level_bullets = [line for line in added_body if line.startswith("- **")]
        _require(
            len(top_level_bullets) == 1,
            "Expected Added section to consolidate into a single top-level bullet",
        )

        # Ensure we preserved the details as nested items.
        _require("Add helper function" in processed, "Expected helper entry to be preserved")
        _require("Add another helper" in processed, "Expected helper entry to be preserved")
        _require("Add tests for new API" in processed, "Expected tests entry to be preserved")

        # Bucket structure should be present.
        _require("  - Tests" in processed, "Expected consolidated output to include a Tests bucket")
        _require("  - Other" in processed, "Expected consolidated output to include an Other bucket")

    def test_wording_normalization_replacements(self):
        content = """# Changelog

## v1.0.0

### Changed

- **Fix retriable dets edge case** [`beefbeef0`](https://github.com/acgetchell/delaunay/commit/beefbeef0)
"""

        processed = _ReleaseNotesPostProcessor.process(content)

        _require("retryable" in processed, "Expected 'retriable' to be normalized to 'retryable'")
        _require("determinants" in processed, "Expected 'dets' to be normalized to 'determinants'")
        _require("retriable" not in processed, "Expected original word 'retriable' to be removed")
        _require(
            re.search(r"\bdets\b", processed, flags=re.IGNORECASE) is None,
            "Expected original token 'dets' to be removed",
        )
