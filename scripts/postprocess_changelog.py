#!/usr/bin/env python3
"""Post-process a git-cliff generated CHANGELOG.md.

Applies lightweight markdown hygiene that is difficult to express in
Tera templates:

  1. Inject summary sections (Breaking Changes, Merged Pull Requests).
  2. Reflow long lines at word boundaries, preserving markdown links
     and code spans as atomic tokens (MD013).
  3. Tag bare fenced code blocks with a language (MD040).
  4. Strip trailing blank lines (MD012).

Usage:
    postprocess-changelog                     # default: CHANGELOG.md
    postprocess-changelog path/to/CHANGELOG.md
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# markdownlint MD013 line-length limit used by this project.
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

# PR link: [#123](https://github.com/.../pull/123)
_PR_LINK_RE = re.compile(r"\[#(\d+)\]\(https://github\.com/[^)]+/pull/\d+\)")

# Commit-hash link to strip from summary lines.
_COMMIT_LINK_RE = re.compile(r"\s*\[`[a-f0-9]{7}`\]\(https://github\.com/[^)]+/commit/[a-f0-9]+\)")

# Leading ``* `` list marker to normalise to ``- `` (MD004).
_STAR_LIST_RE = re.compile(r"^(\s*)\* ")

# Extra spaces after list marker: ``-   `` → ``- `` (MD030).
_LIST_MARKER_SPACE_RE = re.compile(r"^(\s*-)\s{2,}")


def _max_pr_number(entry: str) -> int:
    """Return the highest PR number in *entry*, for descending sort."""
    numbers = [int(m) for m in _PR_LINK_RE.findall(entry)]
    return max(numbers) if numbers else 0


def _compact_entry(line: str, *, strip_breaking: bool = False) -> str:
    """Return a compact summary of a changelog entry line.

    Strips the trailing commit-hash link and, optionally, the
    ``[**breaking**]`` prefix.
    """
    result = _COMMIT_LINK_RE.sub("", line).rstrip()
    if strip_breaking:
        result = result.replace("[**breaking**] ", "", 1)
    return result


def _extract_section_summaries(
    section: list[str],
) -> tuple[list[str], list[str]]:
    """Return ``(pr_entries, breaking_entries)`` from *section* lines."""
    pr_entries: list[str] = []
    breaking_entries: list[str] = []

    for sline in section:
        # Only top-level list items (no leading whitespace).
        if not sline.startswith("- "):
            continue

        is_breaking = "[**breaking**]" in sline
        has_pr = bool(_PR_LINK_RE.search(sline))

        if is_breaking:
            breaking_entries.append(_compact_entry(sline, strip_breaking=True))
        if has_pr:
            pr_entries.append(_compact_entry(sline, strip_breaking=True))

    return pr_entries, breaking_entries


def _inject_summary_sections(text: str) -> str:
    """Insert *Merged Pull Requests* and *Breaking Changes* summaries.

    Scans each version section for PR-linked and breaking-change entries,
    builds compact summary lists, and inserts them immediately after the
    version heading (before the first categorised ``### …`` group).
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
        if any("### Merged Pull Requests" in s for s in section):
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
    """Wrap *line* at word boundaries, keeping markdown atoms intact.

    Returns the (possibly multi-line) replacement string.
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
    """Strip cliff's 2-space indent from an orphaned body list item.

    Git-cliff's ``indent(prefix="  ")`` filter adds 2 spaces to every
    body line.  When a ``  - `` (or ``    - ``, etc.) list item has no
    ``- `` ancestor at column 0, the indent is artificial and triggers
    markdownlint MD007.  The amount stripped depends on context:

    * If an indented ``- ``/``* `` ancestor exists in the original
      lines (the item is a sub-item of another cliff-indented
      unordered list), strip exactly 2 spaces (the cliff prefix) to
      preserve relative nesting.
    * Otherwise strip **all** leading whitespace — the item lives
      under ordered-list or prose context where MD007 expects column 0.

    The scan uses the **original** *lines* (not the already-processed
    result list) so that earlier de-indented siblings cannot create
    false parents.
    """
    stripped = line.lstrip()
    if not (line.startswith("  ") and stripped.startswith("- ")):
        return line

    our_indent = len(line) - len(stripped)
    has_unordered_ancestor = False

    for j in range(idx - 1, -1, -1):
        prev = lines[j]
        if not prev.strip():
            continue  # skip blanks
        if prev.startswith(" "):
            prev_stripped = prev.lstrip()
            if prev_stripped.startswith(("- ", "* ")):
                parent_indent = len(prev) - len(prev_stripped)
                if our_indent > parent_indent:
                    has_unordered_ancestor = True
            continue  # skip cliff-indented content
        # Column-0 non-blank line: real parent or prose.
        if prev.startswith(("- ", "* ")):
            return line  # column-0 list parent found
        # Column-0 non-list — orphan.
        if has_unordered_ancestor:
            return line[2:]  # strip cliff indent, keep relative nesting
        return stripped  # strip all — no unordered context
    # Reached top of document — orphan.
    if has_unordered_ancestor:
        return line[2:]
    return stripped


def _needs_blank_before(stripped: str, result: list[str]) -> bool:
    """Return *True* when a blank line is needed before a list item (MD032)."""
    if not stripped.startswith("- ") or not result or not result[-1].strip():
        return False
    prev = result[-1].lstrip()
    return not prev.startswith(("-", "#"))


def postprocess(path: Path) -> None:
    """Read *path*, apply hygiene fixes, and write it back."""
    text = path.read_text(encoding="utf-8")

    # Inject PR / breaking-change summary sections before reflow.
    text = _inject_summary_sections(text)

    lines = text.split("\n")
    result: list[str] = []
    in_code_block = False

    for idx, line in enumerate(lines):
        stripped = line.lstrip()

        # --- fenced code-block tracking ---
        if stripped.startswith("```"):
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
            continue

        # Never reflow inside code blocks.
        if in_code_block:
            result.append(line)
            continue

        # --- MD004: normalise ``* `` list markers to ``- `` ---
        line = _STAR_LIST_RE.sub(r"\1- ", line)

        # --- MD030: normalise spaces after list marker ---
        line = _LIST_MARKER_SPACE_RE.sub(r"\1 ", line)

        # --- MD007: de-indent orphaned body list items ---
        line = _deindent_orphan(line, lines, idx)
        stripped = line.lstrip()

        # --- MD032: blank line before a list item that follows prose ---
        if _needs_blank_before(stripped, result):
            result.append("")

        # --- reflow long lines ---
        if len(line) > MAX_LINE_WIDTH:
            result.append(_reflow_line(line))
        else:
            result.append(line)

    # 1. Reassemble and strip trailing blank lines.
    text = "\n".join(result)
    text = text.rstrip("\n") + "\n"

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
