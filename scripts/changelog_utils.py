#!/usr/bin/env python3
"""
Shared utilities for changelog operations.

This module provides common functionality used by multiple scripts
for changelog generation, parsing, and git tag management.

Requires Python 3.11+ for PEP 604 union types and datetime.UTC.
"""

import argparse
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from subprocess_utils import (
    ExecutableNotFoundError,
    check_git_history as _check_git_history,
    check_git_repo as _check_git_repo,
    get_git_remote_url,
    run_git_command as _run_git_command,
    run_git_command_with_input,
    run_safe_command,
)

# ANSI color codes for terminal output
COLOR_GREEN = "\033[0;32m"
COLOR_BLUE = "\033[0;34m"
COLOR_YELLOW = "\033[1;33m"
COLOR_RESET = "\033[0m"

# GitHub's maximum size for git tag annotations (bytes)
_GITHUB_TAG_ANNOTATION_LIMIT = 125_000


class ChangelogError(Exception):
    """Base exception for changelog operations."""


class ChangelogNotFoundError(ChangelogError):
    """Raised when CHANGELOG.md cannot be found."""


class GitRepoError(ChangelogError):
    """Raised when git repository operations fail."""


class VersionError(ChangelogError):
    """Raised when version validation fails."""


class ChangelogUtils:
    """Utility class for changelog operations."""

    @staticmethod
    def find_changelog_path() -> str:
        """
        Find CHANGELOG.md in current directory or parent directory.

        Returns:
            Absolute path to CHANGELOG.md

        Raises:
            ChangelogNotFoundError: If CHANGELOG.md is not found
        """
        current_dir = Path.cwd()

        # Check current directory
        changelog_path = current_dir / "CHANGELOG.md"
        if changelog_path.exists():
            return str(changelog_path)

        # Check parent directory
        parent_changelog = current_dir.parent / "CHANGELOG.md"
        if parent_changelog.exists():
            return str(parent_changelog)

        msg = "CHANGELOG.md not found in current directory or parent directory. Please run this script from the project root or scripts/ directory."
        raise ChangelogNotFoundError(msg)

    @staticmethod
    def validate_git_repo() -> bool:
        """
        Check if current directory is in a git repository.

        Returns:
            True if in a git repository

        Raises:
            GitRepoError: If not in a git repository or git is not available
        """
        try:
            if _check_git_repo():
                return True
            msg = "Not in a git repository"
            raise GitRepoError(msg)
        except ExecutableNotFoundError as exc:
            msg = "git command not found"
            raise GitRepoError(msg) from exc

    @staticmethod
    def check_git_history() -> bool:
        """
        Check if git repository has commit history.

        Returns:
            True if git history exists

        Raises:
            GitRepoError: If no git history found
        """
        try:
            if _check_git_history():
                return True
            msg = "No git history found. Cannot generate changelog."
            raise GitRepoError(msg)
        except ExecutableNotFoundError as exc:
            msg = "git command not found"
            raise GitRepoError(msg) from exc

    @staticmethod
    def parse_version(tag_version: str) -> str:
        """
        Parse version string, removing 'v' prefix if present.

        Args:
            tag_version: Version string (e.g., 'v0.4.1' or '0.4.1')

        Returns:
            Version number without 'v' prefix (e.g., '0.4.1')
        """
        return tag_version[1:] if tag_version.startswith("v") else tag_version

    @staticmethod
    def validate_semver(tag_version: str) -> bool:
        """
        Validate that tag follows SemVer format.

        Args:
            tag_version: Version string to validate

        Returns:
            True if valid SemVer format

        Raises:
            VersionError: If invalid SemVer format
        """
        # SemVer 2.0.0 strict: vMAJOR.MINOR.PATCH with optional -PRERELEASE and optional +BUILD
        # No leading zeros in numeric identifiers (MAJOR, MINOR, PATCH, pre-release numeric parts)
        semver_pattern = (
            r"^v"  # Required 'v' prefix
            r"(0|[1-9]\d*)\."  # MAJOR version (no leading zeros)
            r"(0|[1-9]\d*)\."  # MINOR version (no leading zeros)
            r"(0|[1-9]\d*)"  # PATCH version (no leading zeros)
            r"(?:-(?:"  # Optional prerelease: -PRERELEASE
            r"(?:0|[1-9]\d*)"  #   Numeric identifier (no leading zeros)
            r"|(?:[A-Za-z-][0-9A-Za-z-]*)"  #   or alphanumeric identifier
            r")(?:\.(?:0|[1-9]\d*|[A-Za-z-][0-9A-Za-z-]*))*"  #   Additional dot-separated identifiers
            r")?"  # End optional prerelease
            r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"  # Optional build metadata: +BUILD
        )

        if not re.match(semver_pattern, tag_version):
            msg = f"Tag version should follow SemVer format 'vX.Y.Z' (e.g., v0.3.5, v1.2.3-rc.1, v1.2.3+build.5). Got: {tag_version}"
            raise VersionError(msg)
        return True

    @staticmethod
    def extract_changelog_section(changelog_path: str, version: str) -> str:
        """
        Extract changelog content for a specific version.

        Args:
            changelog_path: Path to CHANGELOG.md file
            version: Version number (without 'v' prefix)

        Returns:
            Changelog content for the specified version

        Raises:
            ChangelogError: If version section not found or file cannot be read
        """
        try:
            changelog_path_obj = Path(changelog_path)
            with changelog_path_obj.open(encoding="utf-8") as file:
                content = file.read()
        except OSError as e:
            msg = f"Cannot read changelog file: {e}"
            raise ChangelogError(msg) from e

        # Escape regex metacharacters in version number
        escaped_version = re.escape(version)

        # Pattern to match version headers with various formats:
        # ## [vX.Y.Z] or ## [X.Y.Z] or ## vX.Y.Z or ## X.Y.Z
        # Also supports hyperlinks: ## [X.Y.Z](https://...)
        header_pattern = rf"^##\s*\[?v?{escaped_version}\]?(?:$|\s|\()"

        lines = content.split("\n")
        section_lines = []
        found = False
        collecting = False

        for line in lines:
            # Check for version header
            if re.match(r"^##\s", line):
                if collecting:
                    # Hit next section, stop collecting
                    break
                if re.match(header_pattern, line):
                    found = True
                    collecting = True
                    continue  # Skip the header itself
            elif collecting:
                section_lines.append(line)

        if not found:
            msg = (
                f"No changelog content found for version {version}. "
                f"Searched for version patterns:\n"
                f"  - ## [{version}] - <date> ...\n"
                f"  - ## v{version} ...\n"
                f"  - ## {version} ..."
            )
            raise ChangelogError(msg)

        # Clean up leading and trailing empty lines
        while section_lines and not section_lines[0].strip():
            section_lines.pop(0)
        while section_lines and not section_lines[-1].strip():
            section_lines.pop()

        result = "\n".join(section_lines)
        if not result.strip():
            msg = f"Changelog section for version {version} is empty"
            raise ChangelogError(msg)

        return result

    @staticmethod
    def get_repository_url() -> str:
        """
        Get the repository URL from git remote origin.

        Returns:
            Repository URL in HTTPS format

        Raises:
            GitRepoError: If remote origin URL cannot be determined
        """
        try:
            repo_url = get_git_remote_url("origin")
        except (ExecutableNotFoundError, Exception) as exc:
            msg = "Could not detect git remote origin URL"
            raise GitRepoError(msg) from exc

        if not repo_url:
            msg = "Git remote origin URL is empty"
            raise GitRepoError(msg)

        # Normalize to https://github.com/owner/repo
        patterns = [
            r"^git@github\.com:(?P<slug>[^/]+/[^/]+?)(?:\.git)?/?$",
            r"^ssh://git@github\.com[:/](?P<slug>[^/]+/[^/]+?)(?:\.git)?/?$",
            r"^ssh://github\.com[:/](?P<slug>[^/]+/[^/]+?)(?:\.git)?/?$",
            r"^git\+ssh://git@github\.com[:/](?P<slug>[^/]+/[^/]+?)(?:\.git)?/?$",
            r"^https://github\.com/(?P<slug>[^/]+/[^/]+?)(?:\.git)?/?$",
            r"^http://github\.com/(?P<slug>[^/]+/[^/]+?)(?:\.git)?/?$",
            r"^git://github\.com/(?P<slug>[^/]+/[^/]+?)(?:\.git)?/?$",
        ]
        normalized = None
        for pat in patterns:
            m = re.match(pat, repo_url)
            if m:
                normalized = f"https://github.com/{m.group('slug')}".rstrip("/")
                break
        if not normalized:
            raise GitRepoError(f"Unsupported git remote URL: {repo_url}")
        return normalized

    @staticmethod
    def get_project_root() -> str:
        """
        Find the project root directory (where CHANGELOG.md is located or should be created).

        Returns:
            Absolute path to project root

        Raises:
            ChangelogError: If project root cannot be determined
        """
        try:
            # First try to find existing CHANGELOG.md
            return str(Path(ChangelogUtils.find_changelog_path()).parent)
        except ChangelogNotFoundError:
            # If CHANGELOG.md doesn't exist, use git repository root
            try:
                result = _run_git_command(["rev-parse", "--show-toplevel"])
                git_root = Path(result.stdout.strip())
                if git_root.exists():
                    return str(git_root)
                msg = "Git root directory does not exist"
                raise ChangelogError(msg)
            except Exception as e:
                msg = "Cannot determine project root. Not in a git repository and no CHANGELOG.md found."
                raise ChangelogError(msg) from e

    @staticmethod
    def escape_version_for_regex(version: str) -> str:
        """
        Escape version string for use in regex patterns.

        Args:
            version: Version string to escape

        Returns:
            Escaped version string safe for regex use
        """
        return re.escape(version)

    @staticmethod
    def escape_markdown(text: str) -> str:
        """
        Escape Markdown formatting characters to prevent output corruption.

        Args:
            text: Text that may contain Markdown formatting characters

        Returns:
            Text with Markdown characters escaped
        """
        # Escape minimal set we use in formatting
        return re.sub(r"([\\*_`\[\]])", r"\\\1", text)

    @staticmethod
    def get_markdown_line_limit() -> int:
        """
        Get the markdown line length limit from .markdownlint.json configuration.

        Returns:
            Line length limit (defaults to 160 if not found or invalid)
        """
        config_file = Path(".markdownlint.json")
        default_limit = 160

        try:
            if config_file.exists():
                with config_file.open(encoding="utf-8") as file:
                    config = json.load(file)
                    # Extract MD013.line_length from markdownlint config
                    md013_config = config.get("MD013", {})
                    if isinstance(md013_config, dict):
                        limit = md013_config.get("line_length", default_limit)
                        if isinstance(limit, int) and limit > 0:
                            return limit
            return default_limit
        except (OSError, json.JSONDecodeError):
            return default_limit

    @staticmethod
    def wrap_markdown_line(text: str, max_length: int, indent: str = "  ") -> list[str]:
        """
        Wrap text to fit within markdown line length limits.

        Args:
            text: Text to wrap
            max_length: Maximum line length including indentation
            indent: Indentation string for wrapped lines

        Returns:
            List of wrapped lines
        """
        if not text.strip():
            return []

        # Calculate available space for content (subtract indent length)
        available_length = max(1, max_length - len(indent))

        # If the line is short enough, return as-is
        if len(indent + text) <= max_length:
            return [indent + text]

        # Use textwrap for proper line wrapping with long word breaking
        wrapped = textwrap.wrap(
            text,
            width=available_length,
            break_long_words=True,
            break_on_hyphens=False,
        )

        return [indent + line for line in wrapped]

    @staticmethod
    def process_squashed_commit(commit_sha: str, repo_url: str) -> str:
        """
        Process a squashed PR commit and format it with proper line wrapping.

        Args:
            commit_sha: Git commit SHA
            repo_url: Repository URL for links

        Returns:
            Formatted commit entries with proper line wrapping

        Raises:
            GitRepoError: If git commands fail
        """
        commit_msg = ChangelogUtils._get_commit_message(commit_sha)
        content_lines = ChangelogUtils._extract_content_lines(commit_msg)

        if not content_lines:
            return ""

        entries = ChangelogUtils._parse_commit_entries(content_lines)
        return ChangelogUtils._format_entries(entries, commit_sha, repo_url)

    @staticmethod
    def format_commit_body(commit_sha: str) -> list[str]:
        """Format a commit message body for inclusion under an existing bullet.

        This is used for non-PR (non-squash-merge) commits so that their commit
        message bodies are included in the changelog while staying markdownlint
        compliant (line wrapping, list indentation, no fenced code blocks).
        """
        commit_msg = ChangelogUtils._get_commit_message(commit_sha)
        body_lines = ChangelogUtils._extract_content_lines(commit_msg)
        if not body_lines:
            return []
        max_line_length = ChangelogUtils.get_markdown_line_limit()
        return ChangelogUtils._format_entry_body(body_lines, max_line_length)

    @staticmethod
    def _get_commit_message(commit_sha: str) -> str:
        """
        Get the full commit message for a given SHA.

        Args:
            commit_sha: Git commit SHA

        Returns:
            Full commit message

        Raises:
            GitRepoError: If git command fails
        """
        try:
            result = _run_git_command(["--no-pager", "show", commit_sha, "--format=%B", "--no-patch"])
            return result.stdout
        except subprocess.CalledProcessError as e:
            msg = f"Failed to get commit message for {commit_sha}: {e}"
            raise GitRepoError(msg) from e

    @staticmethod
    def _extract_content_lines(commit_msg: str) -> list[str]:
        """
        Extract content lines from commit message, skipping title and empty lines.

        Args:
            commit_msg: Full commit message

        Returns:
            List of content lines
        """
        lines = commit_msg.strip().split("\n")
        content_lines: list[str] = []

        # Skip the first line (PR title) and empty lines at start.
        trailer_re = re.compile(
            r"^\s*(Co-authored-by|Signed-off-by|Change-Id|Reviewed-on|Reviewed-by|See-Also):",
            re.I,
        )
        refs_re = re.compile(r"^\s*Refs:\s*(?P<refs>.*)$", re.I)

        for line in lines[1:]:
            if trailer_re.match(line):
                continue

            # Keep issue-style references ("Refs: #123"), but drop branch-like refs
            # ("Refs: feature/foo") which are not stable release notes.
            if (m := refs_re.match(line)) and "#" not in m.group("refs"):
                continue

            if line.strip() or content_lines:
                content_lines.append(line)

        # Remove trailing empty lines
        while content_lines and not content_lines[-1].strip():
            content_lines.pop()

        return content_lines

    @staticmethod
    def _parse_commit_entries(content_lines: list[str]) -> list[dict[str, Any]]:
        """
        Parse content lines into structured entries.

        Args:
            content_lines: Lines of content to parse

        Returns:
            List of entry dictionaries with 'title' and 'body_lines' keys
        """
        entries: list[dict[str, Any]] = []
        current_entry: dict[str, Any] | None = None

        for line in content_lines:
            # Detect bullet points: "* ", "- ", or "\d+. "
            bullet_match = re.match(r"^(\s*)([*-]|\d+\.)\s+(.*)$", line)
            if bullet_match:
                if current_entry:
                    entries.append(current_entry)
                current_entry = {
                    "title": bullet_match.group(3).strip(),
                    "body_lines": [],
                }
            elif current_entry:
                # This line belongs to the current entry's body
                current_entry["body_lines"].append(line)
            elif not current_entry and line.strip():
                # No current entry and this is a non-empty line - treat as standalone entry
                current_entry = {"title": line.strip(), "body_lines": []}

        if current_entry:
            entries.append(current_entry)

        return entries

    @staticmethod
    def _format_entries(entries: list[dict[str, Any]], commit_sha: str, repo_url: str) -> str:
        """
        Format parsed entries into markdown output.

        Args:
            entries: List of entry dictionaries
            commit_sha: Git commit SHA for links
            repo_url: Repository URL for links

        Returns:
            Formatted markdown string
        """
        max_line_length = ChangelogUtils.get_markdown_line_limit()
        output_lines = []

        for i, entry in enumerate(entries):
            if i > 0:
                output_lines.append("")  # Blank line between entries

            # Format title with commit link
            title_lines = ChangelogUtils._format_entry_title(entry["title"], commit_sha, repo_url, max_line_length)
            output_lines.extend(title_lines)

            # Format body content
            body_lines = ChangelogUtils._format_entry_body(entry["body_lines"], max_line_length)
            output_lines.extend(body_lines)

        return "\n".join(output_lines)

    @staticmethod
    def _format_entry_title(title: str, commit_sha: str, repo_url: str, max_line_length: int) -> list[str]:
        """Format an entry title with commit link."""
        escaped_title = ChangelogUtils.escape_markdown(title)
        title_line = f"- **{escaped_title}** [`{commit_sha}`]({repo_url}/commit/{commit_sha})"

        if len(title_line) <= max_line_length:
            return [title_line]

        title_only = f"- **{escaped_title}**"
        if len(title_only) > max_line_length:
            return ChangelogUtils._format_long_title(escaped_title, commit_sha, repo_url, max_line_length)

        return ChangelogUtils._format_split_title(escaped_title, commit_sha, repo_url, max_line_length)

    @staticmethod
    def _format_long_title(escaped_title: str, commit_sha: str, repo_url: str, max_line_length: int) -> list[str]:
        """Handle titles that need wrapping even without the commit link."""
        first_prefix, cont_prefix, bold_suffix = "- **", "  **", "**"
        avail_first = max(1, max_line_length - len(first_prefix) - len(bold_suffix))
        avail_cont = max(1, max_line_length - len(cont_prefix) - len(bold_suffix))
        wrap_width = min(avail_first, avail_cont)

        use_bold = wrap_width >= 8
        if not use_bold:
            first_prefix, cont_prefix, bold_suffix = "- ", "  ", ""
            avail_first = max(1, max_line_length - len(first_prefix) - len(bold_suffix))
            avail_cont = max(1, max_line_length - len(cont_prefix) - len(bold_suffix))
            wrap_width = min(avail_first, avail_cont)

        wrapped_title_lines = textwrap.wrap(
            escaped_title,
            width=wrap_width,
            break_long_words=True,
            break_on_hyphens=True,
        ) or [escaped_title[:wrap_width]]

        result_lines: list[str] = []
        for i, line in enumerate(wrapped_title_lines):
            prefix = first_prefix if i == 0 else cont_prefix
            result_lines.append(f"{prefix}{line}{bold_suffix}" if use_bold else f"{prefix}{line}")

        commit_link = f"  [`{commit_sha}`]({repo_url}/commit/{commit_sha})"
        if len(commit_link) <= max_line_length:
            result_lines.append(commit_link)
        else:
            result_lines.append(f"  [`{commit_sha}`]")
            result_lines.append(f"  ({repo_url}/commit/{commit_sha})")

        return result_lines

    @staticmethod
    def _format_split_title(escaped_title: str, commit_sha: str, repo_url: str, max_line_length: int) -> list[str]:
        """Handle titles that fit but need commit link on separate line."""
        lines = [f"- **{escaped_title}**"]
        commit_link = f"  [`{commit_sha}`]({repo_url}/commit/{commit_sha})"

        if len(commit_link) <= max_line_length:
            lines.append(commit_link)
        else:
            lines.append(f"  [`{commit_sha}`]")
            lines.append(f"  ({repo_url}/commit/{commit_sha})")

        return lines

    @staticmethod
    def _protect_cron_expressions(line: str) -> str:
        """
        Protect cron expressions and asterisk patterns from markdown interpretation.

        Wraps cron-like patterns (e.g., '0 2 * * 0') in backticks to prevent markdown
        from treating asterisks as emphasis markers.

        Args:
            line: Line that may contain cron expressions

        Returns:
            Line with cron expressions wrapped in backticks
        """

        # Match cron expressions: sequences with asterisks and numbers/spaces
        # Pattern: quoted strings containing digits, spaces, asterisks, and hyphens
        def protect_match(match: re.Match[str]) -> str:
            content = match.group(0)
            # If already in backticks, leave it alone
            if content.startswith("`") and content.endswith("`"):
                return content
            # Check if this looks like a cron expression (has * and digits/spaces)
            if "*" in content and any(c.isdigit() or c.isspace() for c in content):
                # Remove quotes and wrap in backticks
                inner = content.strip("'\"")
                return f"`{inner}`"
            return content

        # Match quoted strings that might be cron expressions
        return re.sub(r"'[0-9 *-]+'|\"[0-9 *-]+\"", protect_match, line)

    @staticmethod
    def _convert_setext_to_atx(body_lines: list[str]) -> list[str]:
        """
        Convert setext-style headings (underlined with === or ---) to ATX style (####).

        Setext-style headings use underlines:
        - Level 1: Text\n====
        - Level 2: Text\n----

        These are converted to #### to avoid conflicts with changelog structure
        and to satisfy markdownlint requirements.

        Args:
            body_lines: List of body lines that may contain setext headings

        Returns:
            List of lines with setext headings converted to ATX style
        """
        result: list[str] = []
        i = 0
        while i < len(body_lines):
            # Check if the next line is a setext underline
            if i + 1 < len(body_lines):
                # Never treat indented code blocks as headings.
                if body_lines[i].startswith("    ") or body_lines[i + 1].startswith("    "):
                    result.append(body_lines[i])
                    i += 1
                    continue

                current_line = body_lines[i].strip()
                next_line = body_lines[i + 1].strip()

                # Check for setext level 1 (=== underline) or level 2 (--- underline)
                if current_line and next_line:
                    if re.match(r"^=+$", next_line):
                        # Level 1 heading - convert to #### with blank line after
                        result.append(f"#### {current_line}")
                        result.append("")  # Add blank line after heading (MD022)
                        i += 2  # Skip both the heading and underline
                        continue
                    if re.match(r"^-+$", next_line) and not re.match(r"^-\s", current_line):
                        # Level 2 heading - convert to #### with blank line after
                        # (but not if current line starts with "- " which is a list item)
                        result.append(f"#### {current_line}")
                        result.append("")  # Add blank line after heading (MD022)
                        i += 2  # Skip both the heading and underline
                        continue

            # Not a setext heading, keep the line as-is
            result.append(body_lines[i])
            i += 1

        return result

    @staticmethod
    def _downgrade_headers(line: str) -> str:
        """
        Downgrade markdown headers to #### for commit message bodies.

        Changelog structure uses:
        - ## for release versions
        - ### for sections (Added, Changed, etc.)

        All headers in commit message bodies are converted to #### to:
        1. Avoid conflicts with changelog structure
        2. Maintain consistent hierarchy (### Section > #### Commit detail)
        3. Satisfy markdownlint MD001 (no heading level jumps)

        Args:
            line: Line that may contain markdown headers

        Returns:
            Line with headers converted to ####
        """
        # Match any markdown header: # through ######
        header_match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if header_match:
            content = header_match.group(2)
            # Convert all headers to #### for consistency and ensure a blank line follows (handled later)
            return f"#### {content}"
        return line

    @staticmethod
    def wrap_bare_urls(line: str) -> str:
        """Wrap bare URLs in angle brackets to satisfy markdownlint MD034.

        This avoids altering:
        - Markdown links of the form [text](url)
        - URLs already wrapped in <angle brackets>
        - URLs in code fences (```)
        - URLs in inline code (`...`)
        """
        # Skip URL wrapping in code contexts to preserve copy-paste behavior
        stripped = line.lstrip()
        if stripped.startswith("```") or re.match(r"^`[^`]+`$", stripped):
            return line

        def repl(match: re.Match[str]) -> str:
            url = match.group(0)
            start = match.start()
            # Skip if already inside <...>
            if start > 0 and line[start - 1] == "<":
                return url
            # Skip if immediately preceded by '(' which likely means [text](url)
            if start > 0 and line[start - 1] == "(":
                return url
            # Skip if inside an inline code span (odd number of backticks before)
            before = line[:start]
            if before.count("`") % 2 == 1:
                return url
            return f"<{url}>"

        return re.sub(r"https?://[^\s<>()]+", repl, line)

    @classmethod
    def _strip_heading_like_emphasis(cls, line: str) -> str:
        """Strip full-line emphasis used as pseudo-headings.

        Markdownlint MD036 flags lines that are *only* emphasis (e.g. '*Title*')
        because they are often used as headings. Commit message bodies sometimes
        contain these, so we normalize to plain text.
        """
        stripped = line.strip()
        for marker in ("*", "_"):
            for n in (3, 2, 1):
                token = marker * n
                if not (stripped.startswith(token) and stripped.endswith(token)):
                    continue
                if len(stripped) <= 2 * n:
                    continue
                inner = stripped[n:-n].strip()
                if not inner:
                    continue
                # Avoid stripping when the line contains multiple emphasis segments.
                if token in inner:
                    continue
                return inner
        return line

    @classmethod
    def _process_body_line(cls, line: str) -> str:
        """Process a single body line: protect crons, downgrade headers, wrap URLs.

        Preserves leading whitespace for code blocks (4+ spaces).
        """
        # Check if this is an indented code block (4+ spaces) before processing
        is_code_block = line.startswith("    ")

        # For code blocks, preserve indentation and don't wrap URLs
        if is_code_block:
            # Just protect crons, don't downgrade headers or wrap URLs in code
            return cls._protect_cron_expressions(line)

        # Normal text: strip, then process
        processed = cls._protect_cron_expressions(line.strip())
        processed = cls._downgrade_headers(processed)
        processed = cls.wrap_bare_urls(processed)
        return cls._strip_heading_like_emphasis(processed)

    @classmethod
    def _convert_fenced_code_blocks_to_indented(cls, body_lines: list[str]) -> list[str]:
        """Convert fenced code blocks (```...```) to indented code blocks.

        This repository's markdownlint config expects indented code blocks (MD046).
        Commit message bodies occasionally include fenced blocks; normalize them so
        changelog generation stays lint-clean.

        We drop fence marker lines and indent contents by 4 spaces.
        """
        out: list[str] = []
        in_fence = False

        for line in body_lines:
            stripped = line.lstrip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                continue

            if in_fence:
                # Avoid emitting whitespace-only lines (MD009). A blank line will split
                # the indented block, which is acceptable in changelog rendering.
                if not line.strip():
                    out.append("")
                else:
                    out.append("    " + line)
                continue

            out.append(line)

        if in_fence:
            logging.debug("Unclosed fenced code block detected in commit body; output may be misformatted")

        return out

    @classmethod
    def _indented_block_looks_like_code(cls, content_lines: list[str]) -> bool:
        """Heuristically decide whether an indented block is actually code.

        Some commit bodies indent wrapped prose by 4 spaces, which Markdown would
        interpret as a code block. We keep indentation only when the block looks
        code-like.
        """
        code_prefixes = (
            "$ ",
            "cargo ",
            "just ",
            "uv ",
            "git ",
            "python ",
            "pytest ",
            "ruff ",
            "taplo ",
            "jq ",
            "curl ",
            "wget ",
            "npm ",
            "npx ",
            "make ",
        )
        code_markers = (
            "::",
            "->",
            "=>",
            "#[",
            "//",
            ";",
            "{",
            "}",
        )

        for line in content_lines:
            stripped = line.strip()
            if not stripped:
                continue

            lowered = stripped.lower()
            if lowered.startswith(code_prefixes):
                return True

            if any(marker in stripped for marker in code_markers):
                return True

            # Treat assignment/config lines as code, but avoid classifying any line that
            # merely *contains* '=' somewhere in the middle of prose.
            if re.match(r"^[A-Za-z_][A-Za-z0-9_.-]*\s*=\s*\S", stripped):
                return True

            # Single-token lines inside indented blocks are more likely to be code/output
            # (hashes, identifiers, paths) than wrapped prose.
            if " " not in stripped and len(stripped) >= 16:
                return True

        return False

    @classmethod
    def _normalize_indented_blocks(cls, body_lines: list[str]) -> list[str]:
        """Deindent 4-space blocks that are likely wrapped prose (not code)."""
        out: list[str] = []
        block: list[str] = []

        def flush() -> None:
            nonlocal block
            if not block:
                return

            content = [line[4:] for line in block]
            if cls._indented_block_looks_like_code(content):
                out.extend(block)
            else:
                for line in content:
                    out.append(line if line.strip() else "")
            block = []

        for line in body_lines:
            if line.startswith("    ") and line.strip():
                block.append(line)
                continue

            flush()
            out.append(line if line.strip() else "")

        flush()
        return out

    @classmethod
    def _build_body_content(cls, body_lines: list[str]) -> list[str]:
        """Build body content from raw lines with processing."""
        body_lines = cls._convert_fenced_code_blocks_to_indented(body_lines)
        body_lines = cls._normalize_indented_blocks(body_lines)
        body_lines = cls._convert_setext_to_atx(body_lines)
        body_content: list[str] = []
        for line in body_lines:
            if line.strip():
                body_content.append(cls._process_body_line(line))
            elif body_content and body_content[-1]:
                body_content.append("")  # Preserve paragraph breaks
        # Remove trailing empty lines
        while body_content and not body_content[-1]:
            body_content.pop()
        return body_content

    @classmethod
    def _format_indented_code_block_line(cls, line: str, max_line_length: int) -> list[str]:
        """Format an indented code block line, enforcing line-length limits.

        We avoid emitting whitespace-only lines (MD009) by collapsing them to a
        blank line.
        """
        code_content = line[4:]
        if not code_content.strip():
            return [""]

        prefix = "  "
        code_prefix = prefix + "    "

        candidate = code_prefix + code_content
        if len(candidate) <= max_line_length:
            return [candidate]

        available = max(1, max_line_length - len(code_prefix))
        chunks = [code_content[i : i + available] for i in range(0, len(code_content), available)]

        wrapped: list[str] = []
        for chunk in chunks:
            trimmed = chunk.rstrip()
            if not trimmed:
                wrapped.append("")
            else:
                wrapped.append(code_prefix + trimmed)

        return wrapped

    @classmethod
    def _format_body_line(cls, line: str, max_line_length: int) -> list[str]:
        """Format a single line for output (wrap or preserve as-is)."""
        if line.startswith("    "):
            return cls._format_indented_code_block_line(line, max_line_length)
        if re.search(r"\[.*\]\(.*\)", line):
            return [f"  {line}"]  # Markdown links - preserve as-is
        return cls.wrap_markdown_line(line, max_line_length, "  ")  # Wrap regular text

    @classmethod
    def _add_heading_spacing(cls, line: str, output_lines: list[str]) -> bool:
        """Add blank lines around headings as needed.

        Returns:
            True if line is a header, False otherwise
        """
        is_header = line.startswith("#### ")
        # Blank line before header
        if is_header and output_lines and output_lines[-1] != "":
            output_lines.append("")
        return is_header

    @classmethod
    def _format_entry_body(cls, body_lines: list[str], max_line_length: int) -> list[str]:
        """Format entry body content with proper wrapping."""
        if not body_lines:
            return []

        body_content = cls._build_body_content(body_lines)
        if not body_content:
            return []

        output_lines: list[str] = [""]  # Blank line before body

        for line in body_content:
            if not line:  # Empty line - preserve as paragraph break
                if output_lines and output_lines[-1] != "":
                    output_lines.append("")
                continue

            is_header = cls._add_heading_spacing(line, output_lines)
            output_lines.extend(cls._format_body_line(line, max_line_length))
            # Blank line after header
            if is_header and output_lines and output_lines[-1] != "":
                output_lines.append("")

        # Collapse multiple blank lines (URLs already wrapped in _process_body_line)
        collapsed: list[str] = []
        for output_line in output_lines:
            if output_line == "" and collapsed and collapsed[-1] == "":
                continue
            collapsed.append(output_line)

        return collapsed

    @staticmethod
    def run_git_command(args: list[str], check: bool = True) -> tuple[str, int]:
        """
        Run a git command and return output and exit code using secure subprocess wrapper.

        Args:
            args: Git command arguments (without 'git')
            check: Whether to raise exception on non-zero exit code

        Returns:
            Tuple of (stdout, exit_code)

        Raises:
            GitRepoError: If command fails and check=True
        """
        try:
            result = _run_git_command(args, check=check)
            return result.stdout.strip(), result.returncode
        except subprocess.CalledProcessError as e:
            if check:
                err = (e.stderr or e.stdout or str(e)).strip()
                msg = f"Git command failed: git {' '.join(args)}: {err}"
                raise GitRepoError(msg) from e
            return e.stdout.strip() if hasattr(e, "stdout") and e.stdout else "", e.returncode
        except Exception as exc:
            if check:
                msg = f"Git command failed: git {' '.join(args)}: {exc}"
                raise GitRepoError(msg) from exc
            return "", 1

    @staticmethod
    def create_git_tag(tag_version: str, force_recreate: bool = False) -> None:
        """
        Create or recreate a git tag with changelog content as the tag message.
        For large changelogs (>125KB), creates a lightweight tag instead.

        Args:
            tag_version: The version tag to create (e.g., 'v0.3.5')
            force_recreate: Force recreate if tag already exists

        Raises:
            VersionError: If tag version format is invalid
            ChangelogError: If changelog content cannot be extracted
            GitRepoError: If git operations fail
        """
        # Validate prerequisites
        ChangelogUtils.validate_git_repo()
        ChangelogUtils.validate_semver(tag_version)

        # Handle existing tag
        ChangelogUtils._handle_existing_tag(tag_version, force_recreate=force_recreate)

        # Get changelog content (may be truncated if too large)
        tag_message, is_truncated = ChangelogUtils._get_changelog_content(tag_version)

        # Check git configuration
        ChangelogUtils._check_git_config()

        # Create the tag (always annotated, but with reference if truncated)
        ChangelogUtils._create_tag_with_message(tag_version, tag_message, is_truncated=is_truncated)

        # Show success message
        ChangelogUtils._show_success_message(tag_version, is_truncated=is_truncated)

    @staticmethod
    def _handle_existing_tag(tag_version: str, force_recreate: bool) -> None:
        """
        Handle existing tag detection and deletion if needed.

        Args:
            tag_version: The version tag to check
            force_recreate: Whether to force recreate existing tag

        Raises:
            ChangelogError: If tag exists and force_recreate is False
            GitRepoError: If git operations fail
        """
        try:
            _, result_code = ChangelogUtils.run_git_command(["rev-parse", "-q", "--verify", f"refs/tags/{tag_version}"], check=False)
            if result_code == 0:
                if not force_recreate:
                    print(
                        f"{COLOR_YELLOW}Tag '{tag_version}' already exists.{COLOR_RESET}",
                        file=sys.stderr,
                    )
                    print(
                        "Use --force to recreate it, or delete it first with:",
                        file=sys.stderr,
                    )
                    print(f"  git tag -d {tag_version}", file=sys.stderr)
                    raise ChangelogError(f"Tag '{tag_version}' already exists")
                print(f"{COLOR_BLUE}Deleting existing tag '{tag_version}'...{COLOR_RESET}")
                ChangelogUtils.run_git_command(["tag", "-d", tag_version])
        except subprocess.CalledProcessError as e:
            msg = f"Failed to check for existing tag: {e}"
            raise GitRepoError(msg) from e

    @staticmethod
    def _get_changelog_content(tag_version: str) -> tuple[str, bool]:
        """
        Get and preview changelog content for the tag.

        Args:
            tag_version: The version tag

        Returns:
            Tuple of (tag_message, is_truncated)
            - tag_message: Content for git tag annotation
            - is_truncated: True if content was truncated due to size limit
        """
        # GitHub's git tag annotation limit
        MAX_TAG_SIZE = _GITHUB_TAG_ANNOTATION_LIMIT

        changelog_path = ChangelogUtils.find_changelog_path()
        version = ChangelogUtils.parse_version(tag_version)
        full_content = ChangelogUtils.extract_changelog_section(changelog_path, version)

        # Check if content exceeds GitHub's limit
        content_size = len(full_content.encode("utf-8"))

        if content_size > MAX_TAG_SIZE:
            print(f"{COLOR_YELLOW}⚠ Changelog content ({content_size:,} bytes) exceeds GitHub's tag limit ({MAX_TAG_SIZE:,} bytes){COLOR_RESET}")
            print(f"{COLOR_BLUE}→ Creating annotated tag with CHANGELOG.md reference{COLOR_RESET}")

            # Create short message referencing CHANGELOG.md
            # Extract date from changelog heading to build proper GitHub anchor
            anchor = ChangelogUtils._extract_github_anchor(changelog_path, version)

            try:
                repo_url = ChangelogUtils.get_repository_url()
            except GitRepoError:
                # Fallback: keep a stable link even when running in a minimal test environment
                # without a configured `origin` remote. Override via CHANGELOG_FALLBACK_URL.
                repo_url = os.environ.get("CHANGELOG_FALLBACK_URL", "https://github.com/acgetchell/delaunay")

            short_message = f"""Version {version}

This release contains extensive changes. See full changelog:
<{repo_url}/blob/{tag_version}/CHANGELOG.md#{anchor}>

For detailed release notes, refer to CHANGELOG.md in the repository.
"""
            return short_message, True
        # Show preview
        print(f"{COLOR_BLUE}Tag message preview ({content_size:,} bytes):{COLOR_RESET}")
        print("----------------------------------------")
        preview_lines = full_content.split("\n")[:20]
        print("\n".join(preview_lines))
        if len(full_content.split("\n")) > 20:
            print("... (truncated for preview)")
        print("----------------------------------------")

        return full_content, False

    @staticmethod
    def _extract_github_anchor(changelog_path: str, version: str) -> str:
        """
        Extract GitHub-compatible anchor from changelog heading.

        GitHub generates anchors by:
        1. Removing markdown link syntax and angle brackets
        2. Converting to lowercase
        3. Replacing spaces with hyphens
        4. Removing dots from version numbers

        For heading: ## [v0.6.0](url) - 2025-11-25
        GitHub generates: #v060---2025-11-25

        Args:
            changelog_path: Path to CHANGELOG.md
            version: Version number (without 'v' prefix)

        Returns:
            GitHub-compatible anchor string
        """
        try:
            with Path(changelog_path).open(encoding="utf-8") as f:
                for line in f:
                    # Find the version heading line (must be a ## heading)
                    if line.startswith("## ") and (f"v{version}" in line or f"[{version}]" in line):
                        # Remove markdown link syntax: [text](url) -> text
                        heading = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
                        # Remove leading ##
                        heading = heading[2:].strip()
                        # Remove angle brackets if present
                        heading = heading.replace("<", "").replace(">", "")
                        # Convert to lowercase
                        heading = heading.lower()
                        # Replace spaces with hyphens
                        heading = heading.replace(" ", "-")
                        # Remove dots and return
                        return heading.replace(".", "")
        except OSError:
            pass

        # Fallback: just use version without dots
        return f"v{version.replace('.', '')}"

    @staticmethod
    def _check_git_config() -> None:
        """
        Check git user configuration and warn if not set.
        """

        try:
            ChangelogUtils.run_git_command(["config", "--get", "user.name"])
            ChangelogUtils.run_git_command(["config", "--get", "user.email"])
        except Exception:
            print(
                f"{COLOR_YELLOW}Warning: git user.name/email not configured; tag creation may fail.{COLOR_RESET}",
                file=sys.stderr,
            )

    @staticmethod
    def _create_tag_with_message(tag_version: str, tag_message: str, is_truncated: bool = False) -> None:
        """
        Create the git tag with the provided message.

        Args:
            tag_version: The version tag to create
            tag_message: The tag message content
            is_truncated: Whether the changelog was truncated (still creates annotated tag)

        Raises:
            GitRepoError: If tag creation fails
        """
        try:
            if is_truncated:
                print(f"{COLOR_BLUE}Creating annotated tag '{tag_version}' with CHANGELOG.md reference...{COLOR_RESET}")
            else:
                print(f"{COLOR_BLUE}Creating annotated tag '{tag_version}' with full changelog content...{COLOR_RESET}")

            # Always create annotated tag
            run_git_command_with_input(["tag", "-a", tag_version, "-F", "-"], input_data=tag_message)

        except Exception as e:
            msg = f"Error creating tag: {e}"
            raise GitRepoError(msg) from e

    @staticmethod
    def _show_success_message(tag_version: str, is_truncated: bool = False) -> None:
        """
        Show success message and next steps.

        Args:
            tag_version: The created tag version
            is_truncated: Whether the changelog was truncated
        """

        print(f"{COLOR_GREEN}✓ Successfully created tag '{tag_version}'{COLOR_RESET}")
        print("")
        print("Next steps:")
        print(f"  1. Push the tag: {COLOR_BLUE}git push origin {tag_version}{COLOR_RESET}")
        print(f"  2. Create GitHub release: {COLOR_BLUE}gh release create {tag_version} --notes-from-tag{COLOR_RESET}")

        if is_truncated:
            print(f"\n{COLOR_YELLOW}Note: Tag annotation references CHANGELOG.md due to size (>125KB).{COLOR_RESET}")
            print(f"{COLOR_YELLOW}The --notes-from-tag will use the reference message. Full details in CHANGELOG.md.{COLOR_RESET}")


def main() -> None:
    """
    Main entry point for changelog-utils CLI.

    This provides a Python replacement for generate_changelog.sh with the same
    functionality but better error handling and cross-platform support.
    """

    def signal_handler(signum, frame):
        """Handle interrupt signals gracefully."""
        sys.exit(1)

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Check for tag subcommand first
    if _is_tag_command():
        _handle_tag_command()
        return

    # Handle generate command or legacy mode
    args = _parse_generate_args()

    if args.help:
        _show_help()
        return

    if args.version:
        _show_version()
        return

    # Execute changelog generation workflow
    _execute_changelog_generation(args.debug)


def _is_tag_command() -> bool:
    """Check if the command is a tag operation."""
    return len(sys.argv) > 1 and sys.argv[1] == "tag"


def _handle_tag_command() -> None:
    """Handle tag subcommand."""
    if len(sys.argv) < 3:
        print("Error: tag command requires a version argument", file=sys.stderr)
        print("Usage: changelog-utils tag <version> [--force]", file=sys.stderr)
        sys.exit(1)

    tag_version = sys.argv[2]
    force_recreate = len(sys.argv) > 3 and sys.argv[3] == "--force"

    try:
        ChangelogUtils.create_git_tag(tag_version, force_recreate)
    except (ChangelogError, GitRepoError, VersionError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _parse_generate_args():
    """Parse command line arguments for generate command."""
    parser = argparse.ArgumentParser(
        description="Generate enhanced changelog with AI commit processing",
        add_help=False,  # We'll handle help ourselves
    )

    # Handle 'generate' subcommand by removing it from args if present
    args_to_parse = sys.argv[1:]
    if args_to_parse and args_to_parse[0] == "generate":
        args_to_parse = args_to_parse[1:]

    parser.add_argument("--debug", action="store_true", help="Preserve intermediate files for debugging")
    parser.add_argument("--help", "-h", action="store_true", help="Show help message")
    parser.add_argument("--version", action="store_true", help="Show version information")

    try:
        return parser.parse_args(args_to_parse)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        sys.exit(code)


def _show_help() -> None:
    """Show help information."""
    print("""
Usage: changelog-utils [COMMAND] [OPTIONS]

A comprehensive changelog management tool with AI commit processing and Keep a Changelog categorization.

Commands:
  generate        Generate an enhanced changelog (default)
  tag <version>   Create a git tag with changelog content as message

Options:
  --debug         Preserve intermediate files for debugging (generate only)
  --force         Force recreate existing tag (tag only)
  --help, -h      Show this help message
  --version       Show version information

Examples:
  changelog-utils                         # Generate changelog, clean up intermediate files
  changelog-utils generate --debug        # Generate changelog, keep intermediate files
  changelog-utils tag v0.4.2             # Create git tag with changelog content
  changelog-utils tag v0.4.2 --force     # Force recreate existing tag

Intermediate files (when using --debug with generate):
  - CHANGELOG.md.tmp (initial git-cliff output)
  - CHANGELOG.md.processed (after date processing)
  - CHANGELOG.md.processed.expanded (after PR expansion)
  - CHANGELOG.md.tmp2 (after AI enhancement)

This tool replaces both generate_changelog.sh and tag-from-changelog.sh.
""")


def _show_version() -> None:
    """Show version information."""
    print("changelog-utils v0.4.1 (Python implementation)")
    print("Part of delaunay-scripts package")


def _execute_changelog_generation(debug_mode: bool) -> None:
    """Execute the main changelog generation workflow."""
    file_paths: dict[str, Path] | None = None

    try:
        # Setup and validation
        project_root, original_cwd = _setup_project_environment()

        # Use absolute paths rooted at the project directory so that error cleanup/restoration
        # still works even after we restore the original working directory.
        file_paths = _initialize_file_paths()

        try:
            # Core workflow steps
            _validate_prerequisites()
            repo_url = _get_repository_url()
            _backup_existing_changelog(file_paths)

            # Execute processing pipeline
            _run_git_cliff(file_paths, project_root)
            _post_process_dates(file_paths)
            _expand_squashed_commits(file_paths, repo_url)
            _enhance_with_ai(file_paths, project_root)
            _post_process_release_notes(file_paths)
            _cleanup_final_output(file_paths)

            # Cleanup and success
            _cleanup_temp_files(file_paths, debug_mode)
            _show_success_message(file_paths)

        finally:
            os.chdir(original_cwd)

    except (ChangelogError, GitRepoError, VersionError) as exc:
        if file_paths is None:
            raise SystemExit(1) from exc
        _restore_backup_and_exit(file_paths)
    except KeyboardInterrupt as exc:
        if file_paths is None:
            raise SystemExit(1) from exc
        _restore_backup_and_exit(file_paths)
    except Exception as exc:  # restore backup and exit on any unexpected error
        if file_paths is None:
            raise SystemExit(1) from exc
        _restore_backup_and_exit(file_paths)


def _initialize_file_paths() -> dict[str, Path]:
    """Initialize file paths for changelog processing."""
    changelog_file = Path("CHANGELOG.md").resolve()
    return {
        "changelog": changelog_file,
        "temp": changelog_file.with_suffix(".md.tmp"),
        "processed": changelog_file.with_suffix(".md.processed"),
        "expanded": changelog_file.with_suffix(".md.processed.expanded"),
        "enhanced": changelog_file.with_suffix(".md.tmp2"),
        "backup": changelog_file.with_suffix(".md.backup"),
    }


def _setup_project_environment() -> tuple[Path, Path]:
    """Setup project environment and return project root and original cwd."""
    project_root = Path(ChangelogUtils.get_project_root())
    original_cwd = Path.cwd()
    os.chdir(project_root)

    return project_root, original_cwd


def _validate_prerequisites() -> None:
    """Validate git repository and required tools."""
    ChangelogUtils.validate_git_repo()
    ChangelogUtils.check_git_history()

    if not shutil.which("git-cliff"):
        print(
            "Error: git-cliff not found. Install via Homebrew (brew install git-cliff) or Cargo (cargo install git-cliff).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Verify git-cliff is runnable.
    try:
        run_safe_command("git-cliff", ["--version"])
    except Exception:
        print(
            "Error: git-cliff failed to run. Verify your installation and try again.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Verify configuration file
    config_path = Path("cliff.toml")
    if not config_path.exists():
        print(f"Error: git-cliff config not found at project root: {config_path}", file=sys.stderr)
        sys.exit(1)


def _get_repository_url() -> str:
    """Get repository URL with fallback."""
    try:
        return ChangelogUtils.get_repository_url()
    except GitRepoError:
        default_fallback = "https://github.com/acgetchell/delaunay"
        fallback = os.environ.get("CHANGELOG_FALLBACK_URL", default_fallback)
        print(
            f"Warning: Could not detect repository URL, using fallback: {fallback} (set CHANGELOG_FALLBACK_URL to override)",
            file=sys.stderr,
        )
        return fallback


def _backup_existing_changelog(file_paths: dict[str, Path]) -> None:
    """Backup existing changelog if it exists."""
    if file_paths["changelog"].exists():
        shutil.copy2(file_paths["changelog"], file_paths["backup"])


def _run_git_cliff(file_paths: dict[str, Path], project_root: Path) -> None:
    """Run git-cliff to generate the initial changelog."""
    config_path = Path("cliff.toml")
    try:
        result = run_safe_command("git-cliff", ["--config", str(config_path)], cwd=project_root)
        file_paths["temp"].write_text(result.stdout, encoding="utf-8")
    except subprocess.CalledProcessError as e:
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        print("Error: git-cliff failed.", file=sys.stderr)

        details = (e.stderr or e.stdout or str(e) or "").strip()
        msg = f"_run_git_cliff failed: git-cliff exited with status {e.returncode}"
        if details:
            msg = f"{msg}\n{details}"
        raise ChangelogError(msg) from e


def _post_process_dates(file_paths: dict[str, Path]) -> None:
    """Post-process dates from ISO format to YYYY-MM-DD."""
    content = file_paths["temp"].read_text(encoding="utf-8")
    processed_content = re.sub(
        r"T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})",
        "",
        content,
    )
    file_paths["processed"].write_text(processed_content, encoding="utf-8")


def _expand_squashed_commits(file_paths: dict[str, Path], repo_url: str) -> None:
    """Expand squashed PR commits."""
    ChangelogGenerator.expand_squashed_prs(file_paths["processed"], file_paths["expanded"], repo_url)


def _enhance_with_ai(file_paths: dict[str, Path], project_root: Path) -> None:
    """Enhance commits with AI categorization."""
    script_dir = Path(__file__).parent
    enhance_script = script_dir / "enhance_commits.py"

    if not enhance_script.exists():
        sys.exit(1)

    try:
        python_exe = sys.executable or "python"
        run_safe_command(python_exe, [str(enhance_script), str(file_paths["expanded"]), str(file_paths["enhanced"])], cwd=project_root)
    except Exception:
        print("Error: enhance_commits.py failed. Verify your Python environment and try again.", file=sys.stderr)
        sys.exit(1)


def _cleanup_final_output(file_paths: dict[str, Path]) -> None:
    """Clean up excessive blank lines in final output."""
    content = file_paths["enhanced"].read_text(encoding="utf-8")
    lines = content.split("\n")
    cleaned_lines = []
    empty_count = 0

    for line in lines:
        if line.strip() == "":
            empty_count += 1
            if empty_count <= 1:
                cleaned_lines.append(line)
        else:
            empty_count = 0
            cleaned_lines.append(line)

    final_content = "\n".join(cleaned_lines)
    file_paths["changelog"].write_text(final_content, encoding="utf-8")


@dataclass(frozen=True)
class _ReleaseNoteEntry:
    lines: tuple[str, ...]
    title: str
    sha: str | None
    url: str | None


# Release-notes post-processing helpers (internal).
#
# The original implementation lived entirely in `_ReleaseNotesPostProcessor` and was
# functional, but grew to include multiple distinct concerns:
# - breaking-change detection + promotion
# - entry consolidation (Added section) by commit
# - commit-link formatting (top-level bullets only)
# - wording normalization
#
# These are split into focused helpers for readability and testability, while keeping
# `_ReleaseNotesPostProcessor.process()` as the stable entry-point.

_RN_TEST_METRICS_RE = re.compile(r"^\s*All tests pass\s*:\s*", re.IGNORECASE)
_RN_PERF_RESULTS_RE = re.compile(r"refresh performance results", re.IGNORECASE)

# Reverse of `ChangelogUtils.escape_markdown()` for heuristic matching only.
_RN_MD_ESCAPES_RE = re.compile(r"\\([\\*_`\[\]])")

_RN_COMMIT_LINK_RE = re.compile(r"\[`(?P<sha>[a-f0-9]{7,40})`\]\((?P<url>[^)]+)\)")
_RN_TITLE_RE = re.compile(r"^\s*-\s+\*\*(?P<title>.*?)\*\*")
_RN_META_TITLE_RE = re.compile(
    r"^(feat|fix|perf|docs|refactor|chore|style|build|ci)(\([^)]*\))?:\s*",
    re.IGNORECASE,
)

# Project-specific breaking-change heuristics (overridable for reuse).
#
# These are literal tokens (not regex) matched with word boundaries. If you want to
# reuse this module in another project, override them with:
#   CHANGELOG_BREAKING_API_TOKENS="token1,token2"
_BREAKING_API_TOKENS_ENV = "CHANGELOG_BREAKING_API_TOKENS"
_DEFAULT_BREAKING_API_TOKENS = (
    "insert_with_statistics",
    "insert_transactional",
)
_BREAKING_API_TOKENS: tuple[str, ...] = tuple(
    token.strip() for token in os.environ.get(_BREAKING_API_TOKENS_ENV, ",".join(_DEFAULT_BREAKING_API_TOKENS)).split(",") if token.strip()
)


def _rn_skip_blank_lines(lines: list[str], start: int) -> int:
    i = start
    while i < len(lines) and lines[i] == "":
        i += 1
    return i


def _rn_commit_link_fullmatch(line: str) -> re.Match[str] | None:
    return _RN_COMMIT_LINK_RE.fullmatch(line.strip())


def _rn_shorten_commit_url(url: str, short_sha: str) -> str:
    if "/commit/" not in url:
        return url
    prefix, _sep, _rest = url.partition("/commit/")
    return f"{prefix}/commit/{short_sha}"


def _rn_strip_commit_link_inline(line: str) -> str:
    return _RN_COMMIT_LINK_RE.sub("", line).rstrip()


def _rn_commit_markup(short_sha: str, short_url: str) -> str:
    return f"[`{short_sha}`]({short_url})"


def _rn_collect_top_level_entry(lines: list[str], start: int) -> tuple[list[str], int]:
    entry: list[str] = [lines[start]]
    i = start + 1
    while i < len(lines):
        nxt = lines[i]
        if nxt.startswith(("## ", "### ", "- ")):
            break
        entry.append(nxt)
        i += 1

    # Trim trailing empty lines inside the entry
    while entry and not entry[-1].strip():
        entry.pop()
    return entry, i


def _rn_extract_entry_title(entry: list[str]) -> str:
    if not entry:
        return ""
    first = entry[0]
    if m := _RN_TITLE_RE.match(first):
        return m.group("title").strip()
    # Fallback: strip "- " prefix
    return first.lstrip("- ").strip()


def _rn_extract_commit_sha(entry: list[str]) -> str | None:
    joined = "\n".join(entry)
    if m := _RN_COMMIT_LINK_RE.search(joined):
        return m.group("sha")
    return None


def _rn_extract_commit_url(entry: list[str]) -> str | None:
    joined = "\n".join(entry)
    if m := _RN_COMMIT_LINK_RE.search(joined):
        return m.group("url")
    return None


def _rn_remove_empty_section(lines: list[str], section_headers: str | tuple[str, ...]) -> list[str]:
    headers = {section_headers} if isinstance(section_headers, str) else set(section_headers)

    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() not in headers:
            out.append(line)
            i += 1
            continue

        # Find section end (next header)
        j = i + 1
        while j < len(lines) and not lines[j].startswith("### ") and not lines[j].startswith("## "):
            j += 1

        section_body = lines[i + 1 : j]
        has_entry = any(section_line.startswith("- ") for section_line in section_body)
        if has_entry:
            out.extend(lines[i:j])
        # else: drop empty section completely
        i = j

    return out


def _rn_wrap_list_item(text: str, prefix: str, max_len: int) -> list[str]:
    """Wrap a list item to respect markdown line-length limits."""
    if not text:
        return [prefix.rstrip()]

    available = max(1, max_len - len(prefix))
    wrapped = textwrap.wrap(text, width=available, break_long_words=True, break_on_hyphens=False) or [text]
    cont_prefix = " " * len(prefix)
    return [prefix + wrapped[0], *[cont_prefix + w for w in wrapped[1:]]]


def _rn_normalize_title(title: str) -> str:
    lowered = title.lower().replace("`", "")
    normalized = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", normalized).strip()


class _BreakingChangeDetector:
    """Detect and promote breaking changes within a release block."""

    # Titles that should be treated as breaking changes when found inside a release-note entry.
    #
    # This list is heuristics-based and intentionally conservative: false positives will
    # promote entries into the "⚠️ Breaking Changes" section, so prefer patterns that
    # strongly imply downstream-facing breakage.
    #
    # NOTE: The fragments below are intentionally included to match common breaking-change wording.
    _BREAKING_TITLE_PATTERNS: tuple[re.Pattern[str], ...] = (
        # MSRV bumps are breaking for downstream users.
        re.compile(r"\bmsrv\b", re.IGNORECASE),
        re.compile(r"\bminimum supported rust version\b", re.IGNORECASE),
        re.compile(r"\brust-version\b", re.IGNORECASE),
        # Explicit breaking / compatibility language.
        re.compile(r"\bbreaking\b", re.IGNORECASE),
        re.compile(r"\bbackward(?:s)?[- ]?incompatib(?:le|ility)\b", re.IGNORECASE),
        re.compile(r"\bincompatib(?:le|ility)\b", re.IGNORECASE),
        re.compile(r"\bnot\b.*\bcompatib(?:le|ility)\b", re.IGNORECASE),
        # Return-type / signature changes (typically breaking API changes).
        re.compile(r"\bsignature changed\b", re.IGNORECASE),
        re.compile(r"\breturn type\b", re.IGNORECASE),
        re.compile(r"\bnow returns\b", re.IGNORECASE),
        re.compile(r"\breturns?\b.*\b(?:instead of|rather than)\b", re.IGNORECASE),
        # API removals / deprecations. We scope these to public-facing language to avoid
        # classifying internal cleanups as breaking changes.
        re.compile(
            r"\b(?:remove|removed|drop|dropped|delete|deleted)\b.*\b(?:public|api|interface|re-export|export|method|function|trait|struct|enum|variant)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bdeprecat(?:e|ed|ion)\b.*\b(?:api|public|interface|re-export|export)\b",
            re.IGNORECASE,
        ),
        # Project-specific known breaking API touchpoints.
        # Controlled via CHANGELOG_BREAKING_API_TOKENS (comma-separated literal tokens).
        *(re.compile(rf"\b{re.escape(token)}\b", re.IGNORECASE) for token in _BREAKING_API_TOKENS),
    )

    @classmethod
    def extract_breaking_changes(cls, lines: list[str]) -> tuple[list[str], list[list[str]]]:
        out: list[str] = []
        breaking_entries: list[list[str]] = []
        i = 0
        in_changed = False
        changed_headers = {"### Changed", "### Changes"}

        while i < len(lines):
            line = lines[i]

            # Track section boundaries
            if line.startswith("### "):
                in_changed = line.strip() in changed_headers
                out.append(line)
                i += 1
                continue

            if in_changed and line.startswith("- "):
                entry, next_i = _rn_collect_top_level_entry(lines, i)
                title = _rn_extract_entry_title(entry)
                if cls._is_breaking_title(title):
                    breaking_entries.append(cls._trim_entry_to_title_and_link(entry))
                else:
                    out.extend(entry)
                    out.append("")
                i = next_i
                continue

            out.append(line)
            i += 1

        out = _rn_remove_empty_section(out, ("### Changed", "### Changes"))
        # Trim trailing separator blanks added by entry writes
        while out and out[-1] == "":
            out.pop()
        out.append("")
        return out, breaking_entries

    @staticmethod
    def insert_breaking_changes_section(lines: list[str], breaking_entries: list[list[str]]) -> list[str]:
        # Insert before the first section header (typically "### Merged Pull Requests").
        insert_at = next((i for i, line in enumerate(lines) if line.startswith("### ")), len(lines))

        section_lines: list[str] = []
        # Ensure we start with a blank line after the header
        if insert_at == 0 or lines[0] != "":
            section_lines.append("")

        section_lines.extend(["### ⚠️ Breaking Changes", ""])

        for idx, entry in enumerate(breaking_entries):
            if idx > 0:
                section_lines.append("")
            section_lines.extend(entry)

        section_lines.append("")

        return [*lines[:insert_at], *section_lines, *lines[insert_at:]]

    @staticmethod
    def _trim_entry_to_title_and_link(entry: list[str]) -> list[str]:
        """Keep only the entry title and its commit link (drop verbose body content)."""
        if not entry:
            return []

        first = entry[0]
        # If the commit link is already inline on the title line, keep only that.
        if _RN_COMMIT_LINK_RE.search(first):
            return [first]

        link_line = next((line for line in entry[1:] if _RN_COMMIT_LINK_RE.search(line)), None)
        if link_line:
            return [first, "", link_line]
        return [first]

    @classmethod
    def _is_breaking_title(cls, title: str) -> bool:
        match_text = cls._unescape_markdown_escapes(title)
        return any(p.search(match_text) for p in cls._BREAKING_TITLE_PATTERNS)

    @staticmethod
    def _unescape_markdown_escapes(text: str) -> str:
        """Reverse markdown escaping for heuristic matching only.

        This intentionally mirrors `ChangelogUtils.escape_markdown()` so that patterns
        like `insert_transactional` still match titles that appear as `insert\\_transactional`
        in the changelog.
        """
        return _RN_MD_ESCAPES_RE.sub(r"\1", text)


class _EntryConsolidator:
    """Group and consolidate entries in a release block (currently: ### Added)."""

    _CONSOLIDATE_ADDED_MIN_ENTRIES = 4

    @classmethod
    def consolidate_added_section(cls, lines: list[str]) -> list[str]:
        out: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.strip() != "### Added":
                out.append(line)
                i += 1
                continue

            # Copy section header and following blank line(s)
            out.append(line)
            i += 1
            while i < len(lines) and lines[i] == "":
                out.append(lines[i])
                i += 1

            # Collect entries until next section/release header
            entries: list[list[str]] = []
            while i < len(lines) and not lines[i].startswith("### ") and not lines[i].startswith("## "):
                if not lines[i].strip():
                    i += 1
                    continue
                if lines[i].startswith("- "):
                    entry, next_i = _rn_collect_top_level_entry(lines, i)
                    entries.append(entry)
                    i = next_i
                    continue
                # Unexpected free text in Added section; preserve
                out.append(lines[i])
                i += 1

            out.extend(cls._consolidate_entries_by_commit(entries))
            # Ensure a blank line before the next section header
            if out and out[-1] != "":
                out.append("")

        return out

    @classmethod
    def _consolidate_entries_by_commit(cls, entries: list[list[str]]) -> list[str]:
        if not entries:
            return []

        groups = cls._group_entries_by_commit(entries)
        max_len = ChangelogUtils.get_markdown_line_limit()

        out: list[str] = []
        for group in groups:
            if cls._should_consolidate_group(group):
                out.extend(cls._render_group_consolidated(group, max_len=max_len))
            else:
                out.extend(cls._render_group_as_is(group))
            out.append("")

        # Trim trailing blanks
        while out and out[-1] == "":
            out.pop()
        return out

    @classmethod
    def _group_entries_by_commit(cls, entries: list[list[str]]) -> list[list[_ReleaseNoteEntry]]:
        parsed = [cls._parse_entry_info(entry) for entry in entries]

        order: list[str] = []
        groups: dict[str, list[_ReleaseNoteEntry]] = {}
        for idx, info in enumerate(parsed):
            key = info.sha if info.sha else f"__no_sha_{idx}"
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(info)

        return [groups[key] for key in order]

    @staticmethod
    def _parse_entry_info(entry: list[str]) -> _ReleaseNoteEntry:
        return _ReleaseNoteEntry(
            lines=tuple(entry),
            title=_rn_extract_entry_title(entry),
            sha=_rn_extract_commit_sha(entry),
            url=_rn_extract_commit_url(entry),
        )

    @classmethod
    def _should_consolidate_group(cls, group: list[_ReleaseNoteEntry]) -> bool:
        sha = group[0].sha
        return bool(sha) and len(group) >= cls._CONSOLIDATE_ADDED_MIN_ENTRIES

    @staticmethod
    def _render_group_as_is(group: list[_ReleaseNoteEntry]) -> list[str]:
        out: list[str] = []
        for info in group:
            out.extend(info.lines)
            out.append("")
        while out and out[-1] == "":
            out.pop()
        return out

    @classmethod
    def _render_group_consolidated(cls, group: list[_ReleaseNoteEntry], max_len: int) -> list[str]:
        sha = group[0].sha or ""

        parent = cls._choose_group_parent(group)
        parent_title = parent.title
        parent_url = parent.url or ""

        subtitles = cls._collect_subtitles(group, parent_title)

        out: list[str] = [f"- **{parent_title}**", "", f"  [`{sha}`]({parent_url})"]

        buckets = cls._bucket_subtitles(subtitles)
        for bucket_name in ["API", "Behavior", "Tests", "Other"]:
            items = buckets.get(bucket_name, [])
            if not items:
                continue
            out.append(f"  - {bucket_name}")
            for item in items:
                out.extend(_rn_wrap_list_item(item, prefix="    - ", max_len=max_len))

        return out

    @classmethod
    def _collect_subtitles(cls, group: list[_ReleaseNoteEntry], parent_title: str) -> list[str]:
        subtitles: list[str] = []
        seen: set[str] = set()

        for info in group:
            title = info.title
            if not title or title == parent_title:
                continue
            if _RN_META_TITLE_RE.match(title):
                continue
            norm = _rn_normalize_title(title)
            if norm in seen:
                continue
            seen.add(norm)
            subtitles.append(title)

        return subtitles

    @staticmethod
    def _bucket_subtitles(subtitles: list[str]) -> dict[str, list[str]]:
        buckets: dict[str, list[str]] = {"API": [], "Behavior": [], "Tests": [], "Other": []}
        for title in subtitles:
            t = title.lower()
            if "test" in t or "tests/" in t or "doctest" in t:
                buckets["Tests"].append(title)
            elif any(k in t for k in ["debug", "logging", "observability", "trace", "instrument"]):
                buckets["Behavior"].append(title)
            elif any(k in t for k in ["enum", "method", "api", "trait", "type", "variant", "re-export"]):
                buckets["API"].append(title)
            else:
                buckets["Other"].append(title)
        return buckets

    @staticmethod
    def _choose_group_parent(group: list[_ReleaseNoteEntry]) -> _ReleaseNoteEntry:
        candidates = [g for g in group if _RN_META_TITLE_RE.match(g.title)]
        if candidates:
            return max(candidates, key=lambda g: len(g.title))
        return max(group, key=lambda g: len(g.title))


class _CommitLinkFormatter:
    """Format commit links for release notes (top-level bullets only)."""

    @classmethod
    def format_commit_links(cls, lines: list[str]) -> list[str]:
        max_len = ChangelogUtils.get_markdown_line_limit()

        out: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]

            bullet_match = re.match(r"^(?P<indent>\s*)-\s+", line)
            if not bullet_match:
                # Drop standalone commit-link-only lines (they should be attached to a bullet).
                if _rn_commit_link_fullmatch(line) is not None:
                    i += 1
                    continue
                out.append(line)
                i += 1
                continue

            indent = bullet_match.group("indent")
            is_top_level = indent == ""

            title_line = _rn_strip_commit_link_inline(line)

            j = _rn_skip_blank_lines(lines, i + 1)
            next_commit = _rn_commit_link_fullmatch(lines[j]) if j < len(lines) else None
            inline_commit = _RN_COMMIT_LINK_RE.search(line)
            commit_match = inline_commit or next_commit

            if not is_top_level:
                out.append(title_line)
                # Skip any trailing commit link line attached to this nested bullet.
                i = j + 1 if next_commit is not None else i + 1
                continue

            if commit_match is None:
                out.append(line)
                i += 1
                continue

            out.extend(cls._format_top_level_bullet_commit_link(title_line, commit_match, max_len))

            # If we used a following commit-link-only line, skip it (and any preceding blank lines).
            i = j + 1 if next_commit is not None else i + 1

        return out

    @staticmethod
    def _format_top_level_bullet_commit_link(
        title_line: str,
        commit_match: re.Match[str],
        max_len: int,
    ) -> list[str]:
        short_sha = commit_match.group("sha")[:7]
        short_url = _rn_shorten_commit_url(commit_match.group("url"), short_sha)
        markup = _rn_commit_markup(short_sha, short_url)

        candidate = title_line + " " + markup
        if len(candidate) <= max_len:
            return [candidate]
        return [title_line, "  " + markup]


class _WordingNormalizer:
    """Normalize wording (readability + spellcheck) within a release block."""

    @classmethod
    def normalize(cls, lines: list[str]) -> list[str]:
        word_boundary = r"\b"
        replacements: tuple[tuple[re.Pattern[str], str], ...] = (
            (re.compile(word_boundary + "dets" + word_boundary, re.IGNORECASE), "determinants"),
            (re.compile(word_boundary + "retriable" + word_boundary, re.IGNORECASE), "retryable"),
        )

        out: list[str] = []
        for line in lines:
            new_line = line
            for pattern, replacement in replacements:
                new_line = pattern.sub(replacement, new_line)
            out.append(new_line)
        return out


class _ReleaseNotesPostProcessor:
    """Post-process the generated changelog to improve release-note readability.

    This is intentionally conservative (heuristics-based) and aims to:
    - Promote breaking changes into a dedicated section near the top of each release.
    - Remove operational/test-metrics lines that are not user-facing release notes.
    - Reduce redundancy in large "Added" sections by consolidating many entries
      from the same commit into a single entry with sub-bullets.
    """

    @classmethod
    def process(cls, content: str) -> str:
        lines = content.split("\n")
        release_starts = [i for i, line in enumerate(lines) if line.startswith("## ")]
        if not release_starts:
            return content

        out: list[str] = []
        # Preserve preamble before first release header
        out.extend(lines[: release_starts[0]])

        for idx, start in enumerate(release_starts):
            end = release_starts[idx + 1] if idx + 1 < len(release_starts) else len(lines)
            release_block = lines[start:end]
            processed_block = cls._process_release_block(release_block)
            # Avoid double blank lines across concatenated blocks
            if out and processed_block and out[-1] == "" and processed_block[0] == "":
                processed_block = processed_block[1:]
            out.extend(processed_block)

        return "\n".join(out)

    @classmethod
    def _process_release_block(cls, block_lines: list[str]) -> list[str]:
        if not block_lines:
            return []

        # First line is the release header (## ...)
        header = block_lines[0]
        body = block_lines[1:]

        # Drop non-user-facing operational lines.
        body = [line for line in body if not _RN_TEST_METRICS_RE.match(line)]

        # Add small navigational hint for performance-results refresh entries.
        body = [cls._annotate_perf_results_line(line) for line in body]

        # Extract breaking-change entries from "### Changed".
        body, breaking_entries = _BreakingChangeDetector.extract_breaking_changes(body)

        # Consolidate very large "### Added" groups (same commit) into a single entry.
        body = _EntryConsolidator.consolidate_added_section(body)

        # Normalize wording (readability + spellcheck).
        body = _WordingNormalizer.normalize(body)

        # If we found breaking changes, insert a dedicated section near the top.
        if breaking_entries:
            body = _BreakingChangeDetector.insert_breaking_changes_section(body, breaking_entries)

        # Contributor-first formatting: keep commit links only on top-level bullets and
        # reduce vertical bloat by inlining short SHAs when possible.
        body = _CommitLinkFormatter.format_commit_links(body)

        return [header, *body]

    @classmethod
    def _annotate_perf_results_line(cls, line: str) -> str:
        if not _RN_PERF_RESULTS_RE.search(line):
            return line
        if "benches/PERFORMANCE_RESULTS.md" in line:
            return line

        # Try to inject the reference inside the bold title when present.
        if m := _RN_TITLE_RE.match(line):
            title = m.group("title")
            new_title = f"{title} (see benches/PERFORMANCE_RESULTS.md)"
            return line.replace(f"**{title}**", f"**{new_title}**", 1)

        return line + " (see benches/PERFORMANCE_RESULTS.md)"


def _post_process_release_notes(file_paths: dict[str, Path]) -> None:
    """Post-process the generated changelog for readability (breaking changes, de-duping, etc.)."""
    content = file_paths["enhanced"].read_text(encoding="utf-8")
    processed = _ReleaseNotesPostProcessor.process(content)
    file_paths["enhanced"].write_text(processed, encoding="utf-8")


def _cleanup_temp_files(file_paths: dict[str, Path], debug_mode: bool) -> None:
    """Clean up temporary files unless in debug mode."""
    temp_files = ["temp", "processed", "expanded", "enhanced"]

    if not debug_mode:
        for key in temp_files:
            if file_paths[key].exists():
                file_paths[key].unlink()

    # Remove backup on success
    if file_paths["backup"].exists():
        file_paths["backup"].unlink()


def _show_success_message(file_paths: dict[str, Path]) -> None:
    """Show success message with release count."""
    try:
        content = file_paths["changelog"].read_text(encoding="utf-8")
        release_count = len([line for line in content.split("\n") if line.startswith("## ")])
        print("✅ Changelog generation completed successfully!")
        print(f"   Processed {release_count} releases with enhanced commit categorization")
    except Exception:
        print("✅ Changelog generation completed successfully!")


def _restore_backup_and_exit(file_paths: dict[str, Path]) -> None:
    """Restore backup file and exit with error."""
    if file_paths["backup"].exists():
        shutil.copy2(file_paths["backup"], file_paths["changelog"])
        file_paths["backup"].unlink()
    sys.exit(1)


class ChangelogGenerator:
    """Main changelog generation functionality."""

    @staticmethod
    def expand_squashed_prs(input_file: Path, output_file: Path, repo_url: str) -> None:
        """
        Expand squashed PR commits in changelog with enhanced body parsing.

        Args:
            input_file: Path to input changelog file
            output_file: Path to output file for expanded content
            repo_url: Repository URL for generating commit links
        """
        processor = ChangelogProcessor(repo_url)
        processor.process_file(input_file, output_file)


class ChangelogProcessor:
    """Handles the processing of changelog files with squashed PR expansion."""

    def __init__(self, repo_url: str) -> None:
        """Initialize the processor with repository URL."""
        self.repo_url = repo_url
        self.pending_expanded_commits: list[str] = []
        self.expanded_commit_shas: set[str] = set()  # Track SHAs we've expanded
        self.in_merged_prs_section = False
        self.current_release_has_changes_section = False
        self.changes_section_index = -1
        self._in_code_block = False  # Track fenced code block state

    def process_file(self, input_file: Path, output_file: Path) -> None:
        """Process changelog file and write expanded content."""
        content = input_file.read_text(encoding="utf-8")
        lines = content.split("\n")
        output_lines: list[str] = []

        for line in lines:
            processed_line = self._process_line(line, output_lines)
            if processed_line is not None:
                output_lines.append(self._wrap_bare_urls(processed_line))

        # Handle any remaining expanded commits at end of file
        self._finalize_pending_commits(output_lines)

        # Write processed content
        output_content = "\n".join(output_lines)
        output_file.write_text(output_content, encoding="utf-8")

    def _wrap_bare_urls(self, line: str) -> str:
        """Wrap bare URLs in angle brackets, skipping fenced and indented code."""
        stripped = line.lstrip()
        # Toggle fenced code-block state on ``` lines; never rewrite them
        if stripped.startswith("```"):
            self._in_code_block = not self._in_code_block
            return line
        # Skip contents of fenced or indented code blocks
        if self._in_code_block or line.startswith("    "):
            return line
        return ChangelogUtils.wrap_bare_urls(line)

    def _process_line(self, line: str, output_lines: list[str]) -> str | None:
        """Process a single line and return the line to append or None to skip."""
        # Handle section headers first
        if self._handle_section_headers(line, output_lines):
            return line

        # Handle new release sections
        if self._handle_new_release(line, output_lines):
            return line

        # Handle PR entries in merged PRs section
        if self._handle_pr_entry(line):
            return line

        # Handle commit lines with SHA patterns
        return self._handle_commit_line(line, output_lines)

    def _handle_section_headers(self, line: str, output_lines: list[str]) -> bool:
        """Handle section header lines and update state accordingly."""
        if re.match(r"^### *Merged Pull Requests$", line):
            self.in_merged_prs_section = True
            return True

        if re.match(r"^### *(Changes|Changed)$", line):
            self.current_release_has_changes_section = True
            self.changes_section_index = len(output_lines)
            self.in_merged_prs_section = False  # Reset merged PRs state when entering Changes section
            return True

        # Reset section tracking when we hit other sections
        if re.match(r"^### ", line) and not re.match(r"^### *Merged Pull Requests$", line):
            self.in_merged_prs_section = False

        return False

    def _handle_new_release(self, line: str, output_lines: list[str]) -> bool:
        """Handle new release sections and manage pending commits."""
        if re.match(r"^## ", line) and output_lines:
            self._insert_pending_commits(output_lines)
            self._reset_release_state()
            return True
        return False

    def _handle_pr_entry(self, line: str) -> bool:
        """Handle PR entries in Merged Pull Requests sections."""
        if not self.in_merged_prs_section:
            return False

        pr_match = re.search(r"^- .*?\[`#(?P<pr>\d+)`\]\(.*\)\s*$", line)
        if not pr_match:
            return False

        pr_number = pr_match.group("pr")
        self._process_pr_squashed_commit(pr_number)
        return True

    def _handle_commit_line(self, line: str, output_lines: list[str]) -> str | None:
        """Handle commit lines with SHA patterns."""
        commit_match = re.search(r"- \*\*.*?\*\*.*?\[`([a-f0-9]{7,40})`\]", line) or re.search(r"- .*?\(#[0-9]+\) \[`([a-f0-9]{7,40})`\]", line)

        if not commit_match:
            return line

        commit_sha = commit_match.group(1)

        # Skip commits that were already expanded from PRs
        if commit_sha in self.expanded_commit_shas:
            return None  # Skip this line to avoid duplication

        return self._process_commit_sha(commit_sha, line, output_lines)

    def _process_pr_squashed_commit(self, pr_number: str) -> None:
        """Process a squashed commit for the given PR number."""
        try:
            grep_pattern = "(#" + pr_number + ")$"
            sha_output, _ = ChangelogUtils.run_git_command(
                [
                    "--no-pager",
                    "log",
                    "--format=%H",
                    "--grep",
                    grep_pattern,
                    "-n",
                    "1",
                ],
                check=False,
            )
            commit_sha = sha_output.strip().splitlines()[0] if sha_output.strip() else ""
            if commit_sha:
                self._expand_squashed_commit(commit_sha)
                # Track this SHA to avoid duplicating it later
                self.expanded_commit_shas.add(commit_sha)
                self.expanded_commit_shas.add(commit_sha[:7])  # Also track short SHA
        except Exception as e:
            logging.debug("Failed to process PR squashed commit for PR #%s: %s", pr_number, e)

    def _process_commit_sha(self, commit_sha: str, original_line: str, output_lines: list[str]) -> str | None:
        """Process a commit SHA and return the appropriate line (or append and skip)."""
        try:
            result_output, _ = ChangelogUtils.run_git_command(
                ["--no-pager", "show", commit_sha, "--format=%s", "--no-patch"],
            )
            commit_subject = result_output.strip()

            # Fallback: if a PR squash commit appears here, expand it inline.
            if re.search(r"\(#[0-9]+\)$", commit_subject):
                processed_commit = ChangelogUtils.process_squashed_commit(commit_sha, self.repo_url)
                if processed_commit.strip():
                    for expanded_line in processed_commit.split("\n"):
                        output_lines.append(self._wrap_bare_urls(expanded_line))
                    self.expanded_commit_shas.add(commit_sha)
                    self.expanded_commit_shas.add(commit_sha[:7])
                    return None

        except Exception as e:
            logging.debug("Failed to process commit SHA %s: %s", commit_sha, e)
            return original_line

        # Non-PR commit: append its body under the existing bullet when present.
        try:
            body_lines = ChangelogUtils.format_commit_body(commit_sha)
        except Exception as e:
            logging.debug("Failed to format commit body for %s: %s", commit_sha, e)
            return original_line

        if not body_lines:
            return original_line

        output_lines.append(self._wrap_bare_urls(original_line))
        for body_line in body_lines:
            output_lines.append(self._wrap_bare_urls(body_line))
        return None

    def _expand_squashed_commit(self, commit_sha: str) -> None:
        """Expand a squashed commit and add to pending commits."""
        try:
            processed_commit = ChangelogUtils.process_squashed_commit(commit_sha, self.repo_url)
            if processed_commit.strip():
                self.pending_expanded_commits.extend(processed_commit.split("\n"))
        except Exception as e:
            logging.debug("Failed to expand squashed commit %s: %s", commit_sha, e)

    def _expand_squashed_commit_inline(self, commit_sha: str, original_line: str) -> str:
        """Expand a squashed commit inline or return original line."""
        try:
            processed_commit = ChangelogUtils.process_squashed_commit(commit_sha, self.repo_url)
            if processed_commit.strip():
                return processed_commit
        except Exception as e:
            logging.debug("Failed to expand squashed commit inline %s: %s", commit_sha, e)
        return original_line

    def _insert_pending_commits(self, output_lines: list[str]) -> None:
        """Insert pending expanded commits into the output."""
        if not self.pending_expanded_commits:
            return

        if self.current_release_has_changes_section and self.changes_section_index >= 0:
            # Add to existing Changes section - use slice insertion to preserve order and avoid O(n²)
            insert_index = self.changes_section_index + 1
            output_lines[insert_index:insert_index] = ["", *self.pending_expanded_commits]
            # Mark that we've used this index so it won't be reused
            self.changes_section_index = -1
        else:
            # Create new Changes section before this release
            output_lines.extend(["", "### Changes", ""])
            output_lines.extend(self.pending_expanded_commits)

        self.pending_expanded_commits.clear()

    def _reset_release_state(self) -> None:
        """Reset state variables for a new release section."""
        self.in_merged_prs_section = False
        self.current_release_has_changes_section = False
        self.changes_section_index = -1
        self.expanded_commit_shas.clear()  # Clear for next release

    def _finalize_pending_commits(self, output_lines: list[str]) -> None:
        """Handle any remaining expanded commits at the end of the file."""
        if not self.pending_expanded_commits:
            return

        if self.current_release_has_changes_section and self.changes_section_index >= 0:
            # Add to existing Changes section - use slice insertion to preserve order and avoid O(n²)
            insert_index = self.changes_section_index + 1
            output_lines[insert_index:insert_index] = ["", *self.pending_expanded_commits]
        else:
            # Add Changes section at the end
            output_lines.extend(["", "### Changes", ""])
            output_lines.extend(self.pending_expanded_commits)

        # Clear pending commits after insertion
        self.pending_expanded_commits.clear()


if __name__ == "__main__":
    main()
