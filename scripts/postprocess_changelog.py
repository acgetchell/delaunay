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

# Common misspellings found in historical commit messages.
# Keys are whole-word patterns; values are their replacements.
# Applied as word-boundary replacements so partial matches are avoided.
_TYPO_MAP: dict[str, str] = {
    "varous": "various",
    "runtim": "runtime",
}

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
    also removes a single leading "[**breaking**] " prefix.

    Parameters:
        line (str): The changelog list item to compact.
        strip_breaking (bool): If True, strip a single leading "[**breaking**] " prefix.

    Returns:
        str: The compacted changelog entry with the commit-hash link (and optional breaking prefix) removed.
    """
    result = _COMMIT_LINK_RE.sub("", line).rstrip()
    if strip_breaking:
        result = result.replace("[**breaking**] ", "", 1)
    return result


def _extract_section_summaries(
    section: list[str],
) -> tuple[list[str], list[str]]:
    """
    Extract summary lines for merged pull requests and breaking changes from a version section.

    Processes only top-level list items in the provided `section` (lines starting with "- "),
    detects PR-linked entries and entries containing "[**breaking**]". Each matching line is
    compacted (trailing commit-hash links removed; the "[**breaking**]" prefix is stripped when
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
    Adjust indentation for orphaned unordered list items produced by git-cliff's two-space prefix.

    If the line is a list item prefixed by two spaces from cliff's indentation, this function either
    strips exactly two leading spaces (preserving relative nesting) when an unordered-list ancestor
    exists in the original lines, or strips all leading whitespace when no unordered context is found.
    The scan inspects the provided original lines (not any already-processed result) to determine
    context and avoid false parents.
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


def _fix_typos(text: str) -> str:
    """Fix known misspellings from historical commit messages.

    Uses word-boundary matching so partial words are not affected.
    """
    for typo, correction in _TYPO_MAP.items():
        text = re.sub(rf"\b{re.escape(typo)}\b", correction, text)
    return text


def postprocess(path: Path) -> None:
    """Read *path*, apply hygiene fixes, and write it back."""
    text = path.read_text(encoding="utf-8")

    # Fix known typos from historical commit messages.
    text = _fix_typos(text)

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
