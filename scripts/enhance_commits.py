#!/usr/bin/env python3
"""
Enhance AI-generated commit messages in changelog with Keep a Changelog categorization.

This script processes changelog entries and categorizes them according to
Keep a Changelog format (Added/Changed/Fixed/Removed/Deprecated/Security).
"""

import re
import sys
from pathlib import Path

# Add the script directory to Python path for shared utilities
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# Shared utilities available but not yet used in this script
# Can be imported when needed: from changelog_utils import ChangelogUtils


def _get_regex_patterns():
    """Get categorization regex patterns."""
    return {
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
            r"\bsupport\b",
            r"\bsupports\b",
            r"\badding support\b",
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


def _extract_title_text(entry):
    """Extract commit title from entry for pattern matching."""
    # Extract just the commit title (between first ** and second **)
    if title_match := re.search(r"\*\*(.*?)\*\*", entry):
        return title_match[1].lower().strip()

    # Fallback: parse from the first line
    first = entry.splitlines()[0]
    pattern = r"-\s+([^[(]+?)(?:\s+\(#\d+\))?\s*(?:\[`[a-f0-9]{7,40}`\].*)?$"
    match = re.match(pattern, first, re.I)
    return match[1].lower().strip() if match else ""


def _categorize_entry(title_text, patterns):
    """Categorize entry based on title text and patterns."""
    return next(
        (
            category
            for category in [
                "added",
                "removed",
                "fixed",
                "changed",
                "deprecated",
                "security",
            ]
            if any(re.search(pattern, title_text) for pattern in patterns.get(category, []))
        ),
        "changed",
    )


def _add_section_with_entries(output_lines, section_name, entries, any_sections_output):
    """Add a section with entries to output lines."""
    if not entries:
        return any_sections_output

    if any_sections_output:
        output_lines.append("")  # Blank line before section if not first
    output_lines.append(f"### {section_name}")
    output_lines.append("")
    for entry in entries:
        output_lines.append(entry)
    return True


def process_and_output_categorized_entries(entries, output_lines):
    """Categorize entries and output them in Keep a Changelog format."""
    if not entries:
        return

    # Get regex patterns
    patterns = _get_regex_patterns()

    # Categorize all entries
    categorized = {
        "added": [],
        "changed": [],
        "removed": [],
        "fixed": [],
        "deprecated": [],
        "security": [],
    }

    for entry in entries:
        title_text = _extract_title_text(entry)
        category = _categorize_entry(title_text, patterns)
        categorized[category].append(entry)

    # Output entries in Keep a Changelog order
    any_sections_output = False
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


def _process_section_header(line):
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


def _collect_commit_entry(lines, line_index):
    """Collect a commit entry with its body content."""
    current_entry = [lines[line_index]]

    # Look ahead to collect any indented body content
    next_line_index = line_index + 1
    while next_line_index < len(lines) and (
        lines[next_line_index].strip() == ""  # Empty line
        or lines[next_line_index].startswith("  ")
    ):  # Indented body content
        current_entry.append(lines[next_line_index].rstrip())
        next_line_index += 1

    return "\n".join(current_entry), next_line_index


def _process_changelog_lines(lines):
    """Process changelog lines and return categorized output."""
    output_lines = []
    section_state = {
        "in_changes_section": False,
        "in_fixed_issues": False,
        "in_merged_prs_section": False,
    }
    categorize_entries_list = []

    line_index = 0
    while line_index < len(lines):
        line = lines[line_index].rstrip()

        # Process section headers
        section_flags = _process_section_header(line)
        if section_flags:
            section_state.update(
                {
                    "in_changes_section": section_flags[1],
                    "in_fixed_issues": section_flags[2],
                    "in_merged_prs_section": section_flags[3],
                }
            )
            if section_flags[0] == "merged_prs":
                output_lines.append(line)  # Keep Merged Pull Requests header
            line_index += 1
            continue

        # Check if we're at a release end or file end
        in_section = any(section_state.values())
        is_release_end = re.match(r"^## ", line) and in_section
        is_file_end = line_index == len(lines) - 1

        if is_release_end or is_file_end:
            # Process any pending entries
            if categorize_entries_list:
                process_and_output_categorized_entries(categorize_entries_list, output_lines)
                categorize_entries_list.clear()

            # Reset section state
            section_state.update(
                {
                    "in_changes_section": False,
                    "in_fixed_issues": False,
                    "in_merged_prs_section": False,
                }
            )

            # Add the release header to output if it's a new release
            if re.match(r"^## ", line):
                output_lines.append("")  # Add blank line before new release
                output_lines.append(line)
            else:
                # Not a release header, add normally
                output_lines.append(line)
            line_index += 1
            continue

        # Process commit lines in Changes or Fixed Issues sections
        if (section_state["in_changes_section"] or section_state["in_fixed_issues"]) and re.match(r"^- \*\*", line):
            entry, next_index = _collect_commit_entry(lines, line_index)
            categorize_entries_list.append(entry)
            line_index = next_index
            continue

        # Skip PR description content in Merged Pull Requests section
        if section_state["in_merged_prs_section"] and re.match(r"^  ", line):
            line_index += 1
            continue

        # Print all other lines normally
        output_lines.append(line)
        line_index += 1

    return output_lines


def main():
    """Main function to process changelog entries."""
    if len(sys.argv) != 3:
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Read the input file
    input_path = Path(input_file)
    with input_path.open(encoding="utf-8") as file:
        lines = file.readlines()

    # Process the changelog
    output_lines = _process_changelog_lines(lines)

    # Write the output file
    output_path = Path(output_file)
    with output_path.open("w", encoding="utf-8") as file:
        for line in output_lines:
            file.write(line + "\n")


if __name__ == "__main__":
    main()
