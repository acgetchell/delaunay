#!/usr/bin/env python3
"""Enhance AI-generated commit messages in changelog with Keep a Changelog categorization.

This script processes changelog entries and categorizes them according to
Keep a Changelog format (Added/Changed/Fixed/Removed/Deprecated/Security).
"""

import re
import sys
from collections.abc import Mapping, MutableSequence, Sequence
from pathlib import Path
from re import Pattern
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from changelog_utils import ChangelogUtils
else:
    try:
        # When executed as a script from scripts/
        from changelog_utils import ChangelogUtils
    except ModuleNotFoundError:
        # When imported as a module (e.g., scripts.enhance_commits)
        from scripts.changelog_utils import ChangelogUtils

# Precompiled regex patterns for performance
COMMIT_BULLET_RE = re.compile(r"^\s*[-*]\s+")
TITLE_FALLBACK_RE = re.compile(
    r"^\s*-\s+([^[(]+?)(?:\s+\(#\d+\))?\s*(?:\[`[a-f0-9]{7,40}`\].*)?$",
    re.IGNORECASE,
)


# Category patterns for Keep a Changelog classification
_CATEGORY_PATTERN_STRINGS = {
    "added": [
        r"\badd\b(?!itional)",  # 'add' but not 'additional'
        r"\badds\b",
        r"\badded\b",
        r"\badding\b",
        r"\baddition\b(?!al)",  # 'addition' but not 'additional'
        r"\bcreate\b",
        r"\bcreates\b",
        r"\bcreating\b",
        r"\bcreated\b",
        r"\benable\b",
        r"\benables\b",
        r"\benabling\b",
        r"\benabled\b",
        r"\badd(?:s|ed|ing)?\s+support\b",
        r"\bintroduc(?:e|es|ed|ing)\s+support\b",
        r"\bimplement\b",
        r"\bimplements\b",
        r"\bimplementing\b",
        r"\bintroduce\b",
        r"\bintroduces\b",
        r"\bintroducing\b",
        r"\bintroduced\b",
        r"^new\b",
        r"\bnew feature\b",
        r"\bnew functionality\b",
        r"^feat:\s*add\b",
        r"^feat:\s*implement\b",
    ],
    "removed": [
        r"\bremove\b",
        r"\bremoves\b",
        r"\bremoving\b",
        r"\bremoved\b",
        r"\bdelete\b",
        r"\bdeletes\b",
        r"\bdeleting\b",
        r"\bdeleted\b",
        r"\bdrop\b",
        r"\bdrops\b",
        r"\bdropping\b",
        r"\bdropped\b",
        r"\beliminate\b",
        r"\beliminates\b",
        r"\beliminating\b",
        r"\beliminated\b",
        r"\bremov(?:e|es|ed|ing)\s+support\b",
        r"\bdrop(?:s|ped|ping)?\s+support\b",
        r"\bdelet(?:e|es|ed|ing)\s+support\b",
    ],
    "fixed": [
        r"\bfix\b",
        r"\bfixes\b",
        r"\bfixing\b",
        r"\bfixed\b",
        r"\bbug\b",
        r"\bbugs\b",
        r"\bpatch\b",
        r"\bresolve\b",
        r"\bresolves\b",
        r"\bresolved\b",
        r"\bcorrect\b",
        r"\bcorrects\b",
        r"\bcorrecting\b",
        r"\bcorrected\b",
        r"\baddress\b.*\b(error|issue|problem)\b",
        r"\brobustness\b",
        r"\bstability\b",
        r"\bdegenerate\b",
        r"\bprecision\b",
        r"\bnumerical\b",
        r"\bfallback\b",
        r"\berror handling\b",
        r"\bconsistency check\b",
        r"\bfalse positives?\b",
        r"\bfalse negatives?\b",
        r"\bimproves?.*\b(error|stability|robustness|numerical|precision|fallback|consistency)\b",
        r"\benhances?.*\b(error|stability|robustness|numerical|precision|fallback|consistency)\b",
    ],
    "changed": [
        r"\bupdate\b",
        r"\bupdates\b",
        r"\bupdating\b",
        r"\brefactor\b",
        r"\brefactors\b",
        r"\brefactoring\b",
        r"\bchange\b",
        r"\bchanges\b",
        r"\bchanging\b",
        r"\bbump\b",
        r"\bmodify\b",
        r"\bmodifies\b",
        r"\bmodifying\b",
        r"\bimprove\b",
        r"\bimproves\b",
        r"\bimproving\b",
        r"\benhance\b",
        r"\benhances\b",
        r"\benhancing\b",
        r"\boptimize\b",
        r"\boptimizes\b",
        r"\boptimizing\b",
        r"\bperformance\b",
        r"^perf:\b",
        r"\bperf\b",
        r"\bperformance improvement\b",
        r"\bperformance regression\b",
        r"\bbenchmark\b",
        r"\bbenchmarks\b",
        r"\bbenchmarking\b",
        r"\bbaseline\b",
        r"\bthroughput\b",
        r"\bspeed\b",
        r"\bspeedup\b",
        r"\bspeedups\b",
        r"\bspeed-up\b",
        r"\bspeed-ups\b",
        r"\bspeeds up\b",
        r"\bslows down\b",
        r"\bfaster\b",
        r"\bslower\b",
        r"\blatency\b",
        r"\bruntime\b",
        r"\bci_performance_suite\b",
        r"\boverall (improvement|regression)\b",
        r"\boverall (ok|acceptable)\b",
        r"\bmsrv\b",
        r"\bminimum supported rust version\b",
    ],
    "deprecated": [
        r"\bdeprecate\b",
        r"\bdeprecates\b",
        r"\bdeprecating\b",
        r"\bdeprecated\b",
    ],
    "security": [
        r"\bsecurity\b",
        r"\bvulnerability\b",
        r"\bvulnerabilities\b",
        r"\bexploit\b",
        r"\bexploits\b",
        r"\bcve-\d{4}-\d{4,7}\b",  # CVE identifiers
        r"\bdependabot\b",
    ],
}
CATEGORY_PATTERNS = {category: [re.compile(pattern) for pattern in patterns] for category, patterns in _CATEGORY_PATTERN_STRINGS.items()}


def _extract_title_text(entry: str) -> str:
    """Extract commit title from entry for pattern matching."""
    if not entry or not entry.strip():
        return ""

    # Extract just the commit title (between first ** and second **)
    if title_match := re.search(r"\*\*(.*?)\*\*", entry):
        return title_match.group(1).lower().strip()

    # Fallback: parse from the first line
    first = entry.split("\n", 1)[0]
    match = TITLE_FALLBACK_RE.match(first)
    return match.group(1).lower().strip() if match else ""


def _categorize_entry(
    title_text: str,
    patterns: Mapping[str, Sequence[Pattern[str]]],
) -> str:
    """Categorize entry based on title text and patterns.

    Checks for explicit category prefixes first (e.g., "Fixed:", "Added:"),
    then falls back to keyword matching with priority ordering. This prevents
    misclassification when a commit contains keywords from multiple categories.

    Examples:
        - "Fixed: Correctly count removed cells" → "fixed" (explicit prefix)
        - "Remove deprecated API" → "removed" (action verb takes precedence)
        - "patch security vulnerability" → "fixed" (matches both, but test expects "fixed")
    """
    # Check for explicit category prefixes first (highest priority)
    # Match both short forms (fix:, add:) and past tense (fixed:, added:)
    # Allow optional whitespace before colon
    explicit_prefix_patterns = [
        (r"^(?:fix|fixed)\s*:", "fixed"),
        (r"^(?:add|added)\s*:", "added"),
        (r"^(?:remove|removed)\s*:", "removed"),
        (r"^(?:change|changed)\s*:", "changed"),
        (r"^(?:deprecate|deprecated)\s*:", "deprecated"),
        (r"^security\s*:", "security"),
    ]

    for pattern_str, category in explicit_prefix_patterns:
        if re.match(pattern_str, title_text, re.IGNORECASE):
            return category

    # Fall back to keyword-based categorization
    # Order prioritizes action verbs (add/remove/fix) over descriptive terms (security/deprecated)
    # This ensures "remove deprecated API" → "removed" not "deprecated"
    # and "patch security vulnerability" → "fixed" not "security"
    return next(
        (
            category
            for category in [
                "added",
                "removed",
                "fixed",
                "deprecated",
                "security",
                "changed",  # Most generic (catch-all)
            ]
            if any(pattern.search(title_text) for pattern in patterns.get(category, []))
        ),
        "changed",
    )


def _add_section_with_entries(
    output_lines: MutableSequence[str],
    section_name: str,
    entries: Sequence[str],
    any_sections_output: bool,
) -> bool:
    """Add a section with entries to output lines."""
    if not entries:
        return any_sections_output

    if any_sections_output:
        output_lines.append("")  # Blank line before section if not first
    output_lines.append(f"### {section_name}")
    output_lines.append("")  # Blank line after heading
    for i, entry in enumerate(entries):
        # Wrap bare URLs to satisfy MD034
        output_lines.append(ChangelogUtils.wrap_bare_urls(entry))
        # Add blank line after each entry except the last one in the section
        if i < len(entries) - 1:
            output_lines.append("")
    return True


def process_and_output_categorized_entries(
    entries: Sequence[str],
    output_lines: MutableSequence[str],
) -> None:
    """Categorize entries and output them in Keep a Changelog format."""
    if not entries:
        return

    # Categorize all entries
    categorized: dict[str, list[str]] = {
        "added": [],
        "changed": [],
        "removed": [],
        "fixed": [],
        "deprecated": [],
        "security": [],
    }

    for entry in entries:
        title_text = _extract_title_text(entry)
        category = _categorize_entry(title_text, CATEGORY_PATTERNS)
        categorized[category].append(entry)

    # Output entries in Keep a Changelog order
    # Check if output_lines already has content to determine if we need blank lines
    any_sections_output = bool(output_lines and output_lines[-1].strip())
    section_order = ["added", "changed", "deprecated", "removed", "fixed", "security"]
    section_names = {
        "added": "Added",
        "changed": "Changed",
        "deprecated": "Deprecated",
        "removed": "Removed",
        "fixed": "Fixed",
        "security": "Security",
    }

    for section in section_order:
        any_sections_output = _add_section_with_entries(
            output_lines,
            section_names[section],
            categorized[section],
            any_sections_output,
        )


def _process_section_header(line: str) -> tuple[str, bool, bool, bool] | None:
    """Process section headers and return section flags."""
    section_patterns = {
        r"^### *(Changes|Changed)$": ("changes", True, False, False),
        r"^### *(Fixed|Fixed Issues)$": ("fixed", False, True, False),
        r"^### *Added$": ("added", True, False, False),
        r"^### *Removed$": ("removed", True, False, False),
        r"^### *Deprecated$": ("deprecated", True, False, False),
        r"^### *Security$": ("security", True, False, False),
        r"^### *Merged Pull Requests$": ("merged_prs", False, False, True),
    }

    for pattern, flags in section_patterns.items():
        if re.match(pattern, line):
            return flags  # (section_name, in_changes, in_fixed, in_merged_prs)

    return None


def _collect_commit_entry(lines: Sequence[str], line_index: int) -> tuple[str, int]:
    """Collect a commit entry with its body content."""
    current_entry = [lines[line_index]]

    # Look ahead to collect any indented body content
    next_line_index = line_index + 1
    while next_line_index < len(lines) and (
        lines[next_line_index].strip() == ""  # Empty line
        or re.match(r"^\s{2,}", lines[next_line_index])  # Indented body content
    ):
        current_entry.append(lines[next_line_index].rstrip())
        next_line_index += 1

    return "\n".join(current_entry), next_line_index


def _handle_section_header_processing(
    section_flags: tuple[str, bool, bool, bool],
    section_state: dict[str, bool],
    categorize_entries_list: list[str],
    output_lines: list[str],
    line: str,
) -> None:
    """Handle processing of recognized section headers."""
    # Check if we're transitioning out of Changes or Fixed Issues sections
    was_in_changes_or_fixed = section_state["in_changes_section"] or section_state["in_fixed_issues"]
    will_be_in_changes_or_fixed = section_flags[1] or section_flags[2]

    # Flush categorized entries if transitioning out of Changes/Fixed sections
    if was_in_changes_or_fixed and not will_be_in_changes_or_fixed and categorize_entries_list:
        process_and_output_categorized_entries(categorize_entries_list, output_lines)
        categorize_entries_list.clear()

    section_state.update(
        {
            "in_changes_section": section_flags[1],
            "in_fixed_issues": section_flags[2],
            "in_merged_prs_section": section_flags[3],
        },
    )
    if section_flags[0] == "merged_prs":
        # Keep Merged Pull Requests header (entries already flushed above if needed)
        if output_lines and output_lines[-1] != "":
            output_lines.append("")
        output_lines.append(line)  # Keep Merged Pull Requests header


def _handle_unrecognized_header(
    categorize_entries_list: list[str],
    output_lines: list[str],
    section_state: dict[str, bool],
    line: str,
) -> None:
    """Handle unrecognized ### headers by flushing pending entries."""
    if categorize_entries_list:
        process_and_output_categorized_entries(categorize_entries_list, output_lines)
        categorize_entries_list.clear()
    section_state.update(
        {
            "in_changes_section": False,
            "in_fixed_issues": False,
            "in_merged_prs_section": False,
        },
    )
    output_lines.append(line)


def _handle_release_end(
    categorize_entries_list: list[str],
    output_lines: list[str],
    section_state: dict[str, bool],
    line: str,
) -> bool:
    """Handle end of release section. Returns True if line was processed."""
    # Process any pending entries only if we have them
    if categorize_entries_list:
        process_and_output_categorized_entries(categorize_entries_list, output_lines)
        categorize_entries_list.clear()

    # Reset section state
    section_state.update(
        {
            "in_changes_section": False,
            "in_fixed_issues": False,
            "in_merged_prs_section": False,
        },
    )

    # Add the release header to output if it's a new release
    if re.match(r"^## ", line):
        if output_lines and output_lines[-1] != "":
            output_lines.append("")  # Avoid double blank lines
        output_lines.append(line)
        return True
    return False


def _process_changelog_lines(lines: Sequence[str]) -> list[str]:
    """Process changelog lines and return categorized output."""
    output_lines: list[str] = []
    section_state = {
        "in_changes_section": False,
        "in_fixed_issues": False,
        "in_merged_prs_section": False,
    }
    categorize_entries_list: list[str] = []

    line_index = 0
    while line_index < len(lines):
        line = lines[line_index].rstrip()

        # Process section headers
        section_flags = _process_section_header(line)
        if section_flags:
            _handle_section_header_processing(section_flags, section_state, categorize_entries_list, output_lines, line)
            line_index += 1
            continue

        # Handle unrecognized ### headers - flush any pending entries
        if line.startswith("### ") and any(section_state.values()):
            _handle_unrecognized_header(categorize_entries_list, output_lines, section_state, line)
            line_index += 1
            continue

        # Process commit lines in Changes or Fixed Issues sections FIRST
        if (section_state["in_changes_section"] or section_state["in_fixed_issues"]) and COMMIT_BULLET_RE.match(line):
            entry, next_index = _collect_commit_entry(lines, line_index)
            categorize_entries_list.append(entry)
            line_index = next_index
            continue

        # Check if we're at a release end or file end AFTER processing entries
        in_section = any(section_state.values())
        is_release_end = re.match(r"^## ", line) and in_section
        is_file_end = line_index == len(lines) - 1

        if (is_release_end or (is_file_end and categorize_entries_list)) and _handle_release_end(
            categorize_entries_list, output_lines, section_state, line
        ):
            line_index += 1
            continue

        # Skip PR description content in Merged Pull Requests section
        if section_state["in_merged_prs_section"] and re.match(r"^  ", line):
            line_index += 1
            continue

        # Print all other lines normally (wrap bare URLs)
        output_lines.append(ChangelogUtils.wrap_bare_urls(line))
        line_index += 1

    # Process any remaining entries at the end of the file
    if categorize_entries_list:
        process_and_output_categorized_entries(categorize_entries_list, output_lines)

    return output_lines


def main() -> None:
    """Main function to process changelog entries."""
    if len(sys.argv) != 3:
        print(
            f"Usage: {Path(sys.argv[0]).name} <input_changelog> <output_changelog>",
            file=sys.stderr,
        )
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        # Read the input file
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"Error: Input file '{input_file}' not found", file=sys.stderr)
            sys.exit(1)

        with input_path.open(encoding="utf-8") as file:
            lines = file.readlines()

        # Process the changelog
        output_lines = _process_changelog_lines(lines)

        # Write the output file
        output_path = Path(output_file)
        with output_path.open("w", encoding="utf-8") as file:
            for line in output_lines:
                file.write(line + "\n")
    except OSError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
