#!/usr/bin/env -S uv run
"""Post-process a git-cliff generated CHANGELOG.md.

Applies lightweight markdown hygiene that is difficult to express in
Tera templates:

  1. Inject summary sections (Breaking Changes, Merged Pull Requests).
  2. Reflow long lines at word boundaries, preserving markdown links
     and code spans as atomic tokens (MD013).
  3. Tag bare fenced code blocks with a language (MD040).
  4. Normalize indented commit-body headings (MD023).
  5. Strip trailing blank lines (MD012).

Usage:
    postprocess-changelog                     # default: CHANGELOG.md
    postprocess-changelog path/to/CHANGELOG.md
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Markdown line-length limit used by this project.
MAX_LINE_WIDTH = 160

# Tokenise a line into atomic markdown units that must not be split.
# Order matters: longer patterns first.
_TOKEN_RE = re.compile(
    r"""
    \[[^\]]*\]\([^)]*\)   # markdown link:  [text](url)
    | `[^`]+`              # code span:      `code`
    | \S+                  # regular word
    """,
    re.VERBOSE,
)


# Version section heading: ## [X.Y.Z] or ## [Unreleased]
_VERSION_RE = re.compile(r"^## \[")

# Generated changelog category headings that delimit entries.
_CATEGORY_HEADING_RE = re.compile(
    r"^### (?:"
    r"Merged Pull Requests|Breaking Changes|Added|Changed|Deprecated|"
    r"Documentation|Fixed|Maintenance|Performance|Removed|Security"
    r")$"
)

# PR link: [#123](https://github.com/.../pull/123)
_PR_LINK_RE = re.compile(r"\[#(\d+)\]\(https://github\.com/[^)]+/pull/\d+\)")

# Commit-hash link to strip from summary lines.
_COMMIT_LINK_RE = re.compile(r"\s*\[`[a-f0-9]{7}`\]\(https://github\.com/[^)]+/commit/[a-f0-9]+\)")

# Leading git-cliff breaking marker to strip from normalized comparison keys.
_BREAKING_MARKER_RE = re.compile(r"^\s*(?:[-*]\s+)?\[?\*\*breaking\*\*\]?\s*", re.IGNORECASE)

# Leading ``* `` list marker to normalise to ``- `` (MD004).
_STAR_LIST_RE = re.compile(r"^(\s*)\* ")

# Unicode bullets from historical changelog entries: ``•  item`` → ``- item``.
_BULLET_SYMBOL_RE = re.compile(r"^(\s*)•\s+")

# Extra spaces after list marker: ``-   `` → ``- `` (MD030).
_LIST_MARKER_SPACE_RE = re.compile(r"^(\s*-)\s{2,}")

# Indented ATX headings from commit bodies: ``  ## Title`` → ``#### Title``.
_INDENTED_ATX_HEADING_RE = re.compile(r"^(?P<indent>\s+)#{1,6}\s+(?P<title>.*?)(?:\s+#+\s*)?$")

# Isolated indented bold headings from historical commit bodies.
_INDENTED_BOLD_HEADING_RE = re.compile(r"^\s+\*\*(?P<title>[^*].*?)\*\*\s*$")

# Historical squash bodies can also contain unindented ATX headings inside a
# generated entry. Demote those without touching release/category headings.
_ENTRY_ATX_HEADING_RE = re.compile(r"^#{2,3}\s+(?P<title>.*?)(?:\s+#+\s*)?$")

# ATX headings after normalization. Used for final heading-specific cleanup.
_ATX_HEADING_RE = re.compile(r"^(?P<level>#{1,6})\s+(?P<title>.*?)(?:\s+#+\s*)?$")

# Bare glob-like identifiers in headings can be parsed as emphasis by linters.
_WILDCARD_IDENTIFIER_RE = re.compile(r"(?<![`A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*\*+[A-Za-z0-9_*]*)(?![`A-Za-z0-9_])")

# Squash-merge commit bodies often contain inner conventional-commit
# headings from the PR branch: ``* fix: thing``. After MD004 normalization
# those become ordinary list items, which makes them look like separate
# generated commits. Treat them as prose headings inside the parent entry.
_SQUASH_HEADING_RE = re.compile(r"^(?P<indent>\s*)-\s+(?P<prefix>[A-Za-z]+(?:\([^)]+\))?!?):\s+(?P<title>.+?)\s*$")

# This label set is intentionally broad, including release labels such as
# "added", "fixed", "changed", "removed", and "deprecated". Rewriting is
# only allowed when _is_isolated_body_heading accepts the line; do not relax
# that guard because tests rely on it to preserve handcrafted sub-bullets.
_SQUASH_HEADING_LABELS: dict[str, str] = {
    "feat": "Added",
    "fix": "Fixed",
    "perf": "Performance",
    "refactor": "Changed",
    "test": "Changed",
    "style": "Changed",
    "build": "Maintenance",
    "chore": "Maintenance",
    "ci": "Maintenance",
    "doc": "Documentation",
    "docs": "Documentation",
    "added": "Added",
    "fixed": "Fixed",
    "changed": "Changed",
    "performance": "Performance",
    "documentation": "Documentation",
    "maintenance": "Maintenance",
    "deprecated": "Deprecated",
    "removed": "Removed",
}


def _plain_summary(text: str) -> str:
    """Return a normalized comparison key for changelog entry text."""
    text = _BREAKING_MARKER_RE.sub("", text)
    text = _COMMIT_LINK_RE.sub("", text)
    text = _PR_LINK_RE.sub("", text)
    text = re.sub(r"^\s*[-*]\s+", "", text)
    text = re.sub(r"^[A-Za-z]+(?:\([^)]+\))?!?:\s+", "", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def _squash_heading_parts(line: str) -> tuple[str, str, str] | None:
    """Return ``(indent, label, title)`` for a squash-body pseudo-heading."""
    if _COMMIT_LINK_RE.search(line):
        return None

    match = _SQUASH_HEADING_RE.match(line)
    if match is None:
        return None

    raw_prefix = match.group("prefix")
    kind = re.sub(r"\([^)]+\)", "", raw_prefix).rstrip("!").casefold()
    label = _SQUASH_HEADING_LABELS.get(kind)
    if label is None:
        return None

    title = match.group("title").strip()
    if not title:
        return None

    return match.group("indent"), label, title[0].upper() + title[1:]


def _normalize_squash_heading(line: str, *, nested: bool = False) -> str:
    """
    Convert squash-merge pseudo-commit bullets into level-4 headings.

    This keeps release-note subsections from PR squash bodies readable while
    avoiding fake top-level changelog entries.
    """
    parts = _squash_heading_parts(line)
    if parts is None:
        return line

    _indent, label, title = parts
    return f"#### {label}: {title}"


def _is_duplicate_squash_heading(line: str, parent_summary: str | None) -> bool:
    """Return true when a squash-body heading repeats its parent entry."""
    parts = _squash_heading_parts(line)
    if parts is None or parent_summary is None:
        return False

    _, _, title = parts
    return _plain_summary(title) == parent_summary


def _is_isolated_body_heading(lines: list[str], idx: int) -> bool:
    """Return true when a body line is separated like a squash heading."""
    prev_is_blank = idx > 0 and not lines[idx - 1].strip()
    next_is_blank = idx + 1 < len(lines) and not lines[idx + 1].strip()
    return prev_is_blank and next_is_blank


def _is_squash_heading_candidate(lines: list[str], idx: int) -> bool:
    """Return true when an original body line will become bold prose."""
    return _squash_heading_parts(lines[idx]) is not None and _is_isolated_body_heading(lines, idx)


def _max_pr_number(entry: str) -> int:
    """
    Get the largest pull request number referenced in the given changelog entry.

    Returns:
        highest_pr (int): The largest PR number found, or 0 if no PR links are present.
    """
    numbers = [int(m) for m in _PR_LINK_RE.findall(entry)]
    return max(numbers) if numbers else 0


def _compact_entry(line: str, *, strip_breaking: bool = False) -> str:
    """
    Produce a compact summary of a changelog list item.

    Removes a trailing commit-hash link from the given line. If `strip_breaking` is True,
    also removes a single leading breaking marker.

    Parameters:
        line (str): The changelog list item to compact.
        strip_breaking (bool): If True, strip a single leading breaking marker.

    Returns:
        str: The compacted changelog entry with the commit-hash link (and optional breaking prefix) removed.
    """
    result = _COMMIT_LINK_RE.sub("", line).rstrip()
    if strip_breaking:
        bullet = result[:2] if result.startswith(("- ", "* ")) else ""
        body = result[2:] if bullet else result
        result = bullet + _BREAKING_MARKER_RE.sub("", body, count=1)
    return result


def _append_unique(entries: list[str], entry: str) -> None:
    """Append *entry* to *entries* only once, preserving first-seen order."""
    if entry not in entries:
        entries.append(entry)


def _extract_section_summaries(
    section: list[str],
) -> tuple[list[str], list[str]]:
    """
    Extract summary lines for merged pull requests and breaking changes from a version section.

    Processes only top-level list items in the provided `section` (lines starting with "- " or
    "* "), detects PR-linked entries and entries containing breaking markers. Each matching line
    is compacted (trailing commit-hash links removed; the breaking marker is stripped when
    requested) before inclusion.

    Parameters:
        section (list[str]): Lines belonging to a single version section from a changelog.

    Returns:
        tuple[list[str], list[str]]: `pr_entries` — compacted lines that contain PR links;
            `breaking_entries` — compacted lines marked as breaking changes.
    """
    pr_entries: list[str] = []
    breaking_entries: list[str] = []

    for sline in section:
        # Only top-level list items (no leading whitespace).
        if not sline.startswith(("- ", "* ")):
            continue

        is_breaking = bool(_BREAKING_MARKER_RE.search(sline))
        has_pr = bool(_PR_LINK_RE.search(sline))

        if is_breaking:
            _append_unique(breaking_entries, _compact_entry(sline, strip_breaking=True))
        if has_pr:
            _append_unique(pr_entries, _compact_entry(sline, strip_breaking=True))

    return pr_entries, breaking_entries


def _inject_summary_sections(text: str) -> str:
    """
    Insert "Merged Pull Requests" and "Breaking Changes" summary sections into a changelog text.

    Scans each version section for PR-linked list items and entries marked as breaking,
    builds compact summary lists (sorted by PR number), and injects a summary block
    immediately after the version heading when relevant.

    Returns:
        processed_text (str): The input text with summary sections inserted; unchanged if
        no version sections or no summary entries are found.
    """
    lines = text.split("\n")

    # Locate version-section boundaries.
    boundaries: list[int] = []
    for i, line in enumerate(lines):
        if _VERSION_RE.match(line):
            boundaries.append(i)

    if not boundaries:
        return text

    # Walk sections in reverse so insertions don't shift later indices.
    for sec_idx in reversed(range(len(boundaries))):
        start = boundaries[sec_idx]
        end = boundaries[sec_idx + 1] if sec_idx + 1 < len(boundaries) else len(lines)
        section = lines[start:end]

        # Guard against double-injection.
        if any("### Merged Pull Requests" in s or "### ⚠️ Breaking Changes" in s for s in section):
            continue

        pr_entries, breaking_entries = _extract_section_summaries(section)

        if not pr_entries and not breaking_entries:
            continue

        # Sort PRs by highest PR number, descending (newest first).
        pr_entries.sort(key=_max_pr_number, reverse=True)

        # Insertion point: first non-blank line after the heading.
        insert_at = start + 1
        while insert_at < end and lines[insert_at].strip() == "":
            insert_at += 1

        block: list[str] = []
        if breaking_entries:
            block.append("### ⚠️ Breaking Changes")
            block.append("")
            block.extend(breaking_entries)
            block.append("")
        if pr_entries:
            block.append("### Merged Pull Requests")
            block.append("")
            block.extend(pr_entries)
            block.append("")

        lines[insert_at:insert_at] = block

    return "\n".join(lines)


def _reflow_line(line: str, max_width: int = MAX_LINE_WIDTH) -> str:
    """
    Reflow a single markdown line to fit within max_width while preserving atomic markdown tokens.

    Preserves a leading list marker ("- " or "* ") on the first line and indents continuation
    lines to maintain list nesting. Tokens such as links and code spans are kept intact and not
    split across lines.

    Parameters:
        line (str): The original line to reflow.
        max_width (int): Maximum allowed line width; lines longer than this will be wrapped.

    Returns:
        str: The reflowed line, potentially containing newline characters so that no output line exceeds max_width.
    """
    if len(line) <= max_width:
        return line

    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]

    # Determine first-line prefix vs continuation indent.
    if stripped.startswith(("- ", "* ")):
        first_prefix = indent + stripped[:2]
        content = stripped[2:]
        cont_indent = indent + "  "
    else:
        first_prefix = indent
        content = stripped
        cont_indent = indent

    tokens = _TOKEN_RE.findall(content)
    if not tokens:
        return line

    lines: list[str] = []
    current = first_prefix + tokens[0]

    for token in tokens[1:]:
        candidate = current + " " + token
        if len(candidate) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = cont_indent + token

    lines.append(current)
    return "\n".join(lines)


def _deindent_orphan(line: str, lines: list[str], idx: int) -> str:
    """
    Normalize indentation for sub-bullet list items produced by git-cliff.

    Cliff's ``indent(prefix="  ")`` filter can compound with pre-existing
    indentation in commit bodies, producing non-standard nesting depths.
    This function scans backward through the original *lines* to find the
    nearest list ancestor and normalizes the indent to ``parent + 2``
    spaces (MD007).
    """
    stripped = line.lstrip()
    if not (line.startswith("  ") and stripped.startswith("- ")):
        return line

    our_indent = len(line) - len(stripped)
    nearest_parent_indent: int | None = None

    for j in range(idx - 1, -1, -1):
        prev = lines[j]
        if not prev.strip():
            continue  # skip blanks
        if prev.startswith(" "):
            prev_stripped = prev.lstrip()
            if prev_stripped.startswith(("- ", "* ")):
                if _is_squash_heading_candidate(lines, j):
                    continue
                parent_indent = len(prev) - len(prev_stripped)
                if our_indent > parent_indent and nearest_parent_indent is None:
                    nearest_parent_indent = parent_indent
            continue  # skip cliff-indented content
        # Column-0 non-blank line — determines final result.
        is_list_parent = prev.startswith(("- ", "* "))
        if is_list_parent:
            base = nearest_parent_indent + 2 if nearest_parent_indent is not None else 2
            return " " * base + stripped
        # Column-0 non-list — orphan.
        return line[2:] if nearest_parent_indent is not None else stripped
    # Reached top of document — orphan.
    return line[2:] if nearest_parent_indent is not None else stripped


def _normalize_continuation_indent(line: str, lines: list[str], idx: int) -> str:
    """Normalize over-indented list continuation lines for rumdl MD077."""
    stripped = line.lstrip()
    if not line.startswith(" ") or not stripped or stripped.startswith("- "):
        return line

    for j in range(idx - 1, -1, -1):
        prev = lines[j]
        if not prev.strip():
            continue

        prev_stripped = prev.lstrip()
        if prev_stripped.startswith(("- ", "* ", "• ")):
            parent_indent = len(prev) - len(prev_stripped)
            expected = parent_indent + 2
            actual = len(line) - len(stripped)
            if actual > expected:
                return " " * expected + stripped
        break

    return line


def _needs_blank_before(stripped: str, result: list[str]) -> bool:
    """
    Determine whether a blank line is required before a list item to satisfy Markdown rule MD032.

    Parameters:
        stripped (str): The current line with leading whitespace removed.
        result (list[str]): The lines already emitted immediately before the current line.

    Returns:
        bool: `True` if a blank line should be inserted before the list item, `False` otherwise.
    """
    if not stripped.startswith("- ") or not result or not result[-1].strip():
        return False
    prev = result[-1].lstrip()
    return not prev.startswith(("-", "#"))


def _normalize_indented_heading(line: str) -> str:
    """
    Convert indented commit-body headings into level-4 headings.

    git-cliff indents commit bodies under each changelog entry. If a historical
    commit body contains an ATX heading such as ``## Correctness Fixes``, the
    rendered changelog contains ``  ## Correctness Fixes``. Rumdl treats
    emphasis-only headings as MD036 violations, so internal commit-body headings
    become real level-4 headings below the release category heading.
    """
    match = _INDENTED_ATX_HEADING_RE.match(line)
    if match is None:
        return line

    title = match.group("title").strip()
    if not title:
        return line

    return f"#### {title}"


def _normalize_indented_bold_heading(line: str, current_entry_summary: str | None, is_isolated_body_heading: bool) -> str:
    """Convert isolated indented bold commit-body headings into level-4 headings."""
    if current_entry_summary is None or not is_isolated_body_heading:
        return line

    match = _INDENTED_BOLD_HEADING_RE.match(line)
    if match is None:
        return line

    title = match.group("title").strip()
    if not title:
        return line

    return f"#### {title}"


def _is_changelog_boundary_heading(line: str) -> bool:
    """Return true for root, version, or category headings that end an entry."""
    return line in {"# Changelog", "## Archives"} or bool(_VERSION_RE.match(line) or _CATEGORY_HEADING_RE.match(line))


def _normalize_entry_heading(line: str, current_entry_summary: str | None) -> str:
    """Demote column-zero headings that belong to the active changelog entry."""
    if current_entry_summary is None or _is_changelog_boundary_heading(line):
        return line

    match = _ENTRY_ATX_HEADING_RE.match(line)
    if match is None:
        return line

    title = match.group("title").strip()
    if not title:
        return line

    return f"#### {title}"


def _code_span_heading_wildcards(line: str) -> str:
    """Wrap bare wildcard identifiers in heading titles with code spans."""
    match = _ATX_HEADING_RE.match(line)
    if match is None:
        return line

    def code_span(match: re.Match[str]) -> str:
        return f"`{match.group(1)}`"

    title_parts = re.split(r"(`[^`]+`)", match.group("title"))
    title = "".join(part if part.startswith("`") and part.endswith("`") else _WILDCARD_IDENTIFIER_RE.sub(code_span, part) for part in title_parts)
    return f"{match.group('level')} {title}"


def _normalize_entry_heading_text(line: str) -> str:
    """Apply final cleanup for generated entry-local headings."""
    return _code_span_heading_wildcards(line)


def normalize_entry_headings_text(text: str) -> str:
    """Demote entry-local headings without applying broader changelog cleanup."""
    result: list[str] = []
    current_entry_summary: str | None = None

    for line in text.split("\n"):
        current_entry_summary = _update_entry_summary(line, current_entry_summary)
        line = _normalize_entry_heading(line, current_entry_summary)
        result.append(_normalize_entry_heading_text(line))

    return "\n".join(result)


def _normalize_horizontal_rule(line: str, result: list[str]) -> str:
    """Normalize indented horizontal rules and ensure they have surrounding blanks."""
    if line.strip() != "---":
        return line

    if result and result[-1].strip():
        result.append("")
    return "---"


def _process_code_fence(line: str, result: list[str], in_code_block: bool, next_line: str | None) -> tuple[bool, bool]:
    """Handle fenced-code transitions and append the line when consumed."""
    stripped = line.lstrip()
    if not stripped.startswith("```"):
        return False, in_code_block

    if not in_code_block:
        in_code_block = True
        # MD031: blank line before fenced code block.
        if result and result[-1].strip():
            result.append("")
        # MD040: add language tag if missing.
        if stripped == "```":
            line = line.replace("```", "```text", 1)
    else:
        in_code_block = False

    result.append(line)
    if not in_code_block and next_line is not None and next_line.strip():
        result.append("")
    return True, in_code_block


def _update_entry_summary(line: str, current_entry_summary: str | None) -> str | None:
    """Track the active changelog entry summary for squash-body cleanup."""
    if _squash_heading_parts(line) is not None:
        return current_entry_summary
    if line.startswith("- "):
        return _plain_summary(line)
    if _is_changelog_boundary_heading(line):
        return None
    return current_entry_summary


def _should_skip_duplicate_heading(
    line: str,
    result: list[str],
    current_entry_summary: str | None,
    is_isolated_body_heading: bool,
) -> tuple[bool, bool]:
    """Return whether to skip a duplicate squash heading and the following blank."""
    if is_isolated_body_heading and _is_duplicate_squash_heading(line, current_entry_summary):
        return True, bool(result and not result[-1].strip())
    return False, False


def _normalize_body_line(
    line: str,
    result: list[str],
    current_entry_summary: str | None,
    is_isolated_body_heading: bool,
) -> str:
    """Apply markdown hygiene transforms to a non-code line."""
    line = _normalize_indented_heading(line)
    line = _normalize_indented_bold_heading(line, current_entry_summary, is_isolated_body_heading)
    line = _normalize_entry_heading(line, current_entry_summary)
    horizontal_rule = _normalize_horizontal_rule(line, result)
    line = horizontal_rule

    if is_isolated_body_heading:
        line = _normalize_squash_heading(line, nested=current_entry_summary is not None)

    line = _normalize_entry_heading_text(line)

    if _needs_blank_before(line.lstrip(), result):
        result.append("")

    return _reflow_line(line) if len(line) > MAX_LINE_WIDTH else line


def postprocess_text(text: str) -> str:
    """Apply changelog markdown hygiene transforms to *text*."""
    # Inject PR / breaking-change summary sections before reflow.
    text = _inject_summary_sections(text)

    lines = text.split("\n")
    result: list[str] = []
    in_code_block = False
    current_entry_summary: str | None = None
    drop_next_blank = False

    for idx, line in enumerate(lines):
        # --- fenced code-block tracking ---
        next_line = lines[idx + 1] if idx + 1 < len(lines) else None
        handled, in_code_block = _process_code_fence(line, result, in_code_block, next_line)
        if handled:
            continue

        # Never reflow inside code blocks.
        if in_code_block:
            result.append(line)
            continue

        # --- MD004: normalise historical and ``* `` list markers to ``- `` ---
        line = _BULLET_SYMBOL_RE.sub(r"\1- ", line)
        line = _STAR_LIST_RE.sub(r"\1- ", line)

        # --- MD030: normalise spaces after list marker ---
        line = _LIST_MARKER_SPACE_RE.sub(r"\1 ", line)

        current_entry_summary = _update_entry_summary(line, current_entry_summary)
        is_isolated_body_heading = _is_isolated_body_heading(lines, idx)

        # --- GitHub squash bodies: collapse duplicate pseudo-headings ---
        should_skip, next_drop_blank = _should_skip_duplicate_heading(
            line,
            result,
            current_entry_summary,
            is_isolated_body_heading,
        )
        if should_skip:
            drop_next_blank = next_drop_blank
            continue
        if drop_next_blank and not line.strip():
            drop_next_blank = False
            continue
        drop_next_blank = False

        line = _deindent_orphan(line, lines, idx)
        line = _normalize_continuation_indent(line, lines, idx)
        normalized = _normalize_body_line(line, result, current_entry_summary, is_isolated_body_heading)
        result.append(normalized)
        if normalized == "---" and next_line is not None and next_line.strip():
            result.append("")

    # 1. Reassemble and strip trailing blank lines.
    text = "\n".join(result)
    return text.rstrip("\n") + "\n"


def postprocess(path: Path) -> None:
    """Read *path*, apply hygiene fixes, and write it back."""
    text = path.read_text(encoding="utf-8")
    text = postprocess_text(text)

    path.write_text(text, encoding="utf-8")


def main() -> None:
    """CLI entry point for ``postprocess-changelog``."""
    parser = argparse.ArgumentParser(
        prog="postprocess-changelog",
        description="Apply markdown hygiene to a git-cliff generated CHANGELOG.md.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="CHANGELOG.md",
        help="Path to CHANGELOG.md (default: CHANGELOG.md)",
    )
    args = parser.parse_args()

    changelog = Path(args.path)
    if not changelog.is_file():
        print(f"Error: {changelog} not found", file=sys.stderr)
        sys.exit(1)

    postprocess(changelog)


if __name__ == "__main__":
    main()
