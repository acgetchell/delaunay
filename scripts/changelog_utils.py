#!/usr/bin/env python3
"""
Shared utilities for changelog operations.

This module provides common functionality used by multiple scripts
for changelog generation, parsing, and git tag management.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Any


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

        raise ChangelogNotFoundError(
            "CHANGELOG.md not found in current directory or parent directory. Please run this script from the project root or scripts/ directory."
        )

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
            subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True,
                check=True,
                text=True,
            )
            return True
        except subprocess.CalledProcessError as exc:
            raise GitRepoError("Not in a git repository") from exc
        except FileNotFoundError as exc:
            raise GitRepoError("git command not found") from exc

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
            subprocess.run(
                ["git", "log", "--oneline", "-n", "1"],
                capture_output=True,
                check=True,
                text=True,
            )
            return True
        except subprocess.CalledProcessError as exc:
            raise GitRepoError("No git history found. Cannot generate changelog.") from exc

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
        # SemVer: vMAJOR.MINOR.PATCH with optional -PRERELEASE and optional +BUILD
        semver_pattern = r"^v[0-9]+(\.[0-9]+){2}(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$"

        if not re.match(semver_pattern, tag_version):
            raise VersionError(f"Tag version should follow SemVer format 'vX.Y.Z' (e.g., v0.3.5, v1.2.3-rc.1, v1.2.3+build.5). Got: {tag_version}")
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
            with open(changelog_path, encoding="utf-8") as file:
                content = file.read()
        except OSError as e:
            raise ChangelogError(f"Cannot read changelog file: {e}") from e

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
            raise ChangelogError(
                f"No changelog content found for version {version}. "
                f"Searched for version patterns:\n"
                f"  - ## [{version}] - <date> ...\n"
                f"  - ## v{version} ...\n"
                f"  - ## {version} ..."
            )

        # Clean up leading and trailing empty lines
        while section_lines and not section_lines[0].strip():
            section_lines.pop(0)
        while section_lines and not section_lines[-1].strip():
            section_lines.pop()

        result = "\n".join(section_lines)
        if not result.strip():
            raise ChangelogError(f"Changelog section for version {version} is empty")

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
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True,
                check=True,
                text=True,
            )
            repo_url = result.stdout.strip()
        except subprocess.CalledProcessError as exc:
            raise GitRepoError("Could not detect git remote origin URL") from exc

        if not repo_url:
            raise GitRepoError("Git remote origin URL is empty")

        # Convert SSH URLs to HTTPS format and clean up
        if re.match(r"^git@github\.com:(.+)\.git$", repo_url):
            match = re.match(r"^git@github\.com:(.+)\.git$", repo_url)
            repo_url = f"https://github.com/{match.group(1)}"
        elif re.match(r"^https://github\.com/(.+)\.git$", repo_url):
            match = re.match(r"^https://github\.com/(.+)\.git$", repo_url)
            repo_url = f"https://github.com/{match.group(1)}"
        elif re.match(r"^https://github\.com/(.+)$", repo_url):
            # Already in correct format, just remove trailing slash if present
            repo_url = repo_url.rstrip("/")

        return repo_url

    @staticmethod
    def get_project_root() -> str:
        """
        Find the project root directory (where CHANGELOG.md is located).

        Returns:
            Absolute path to project root

        Raises:
            ChangelogError: If project root cannot be determined
        """
        current_dir = Path.cwd()

        # Check if we're already in project root
        if (current_dir / "CHANGELOG.md").exists():
            return str(current_dir)

        # Check if we're in scripts/ subdirectory
        if (current_dir.parent / "CHANGELOG.md").exists():
            return str(current_dir.parent)

        raise ChangelogError("Cannot determine project root. CHANGELOG.md not found in current or parent directory.")

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
                with open(config_file, encoding="utf-8") as file:
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
        available_length = max_length - len(indent)

        # If the line is short enough, return as-is
        if len(indent + text) <= max_length:
            return [indent + text]

        # Split into words for wrapping
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            # Check if adding this word would exceed the limit
            test_line = current_line + (" " if current_line else "") + word
            if len(test_line) <= available_length:
                current_line = test_line
            # Current line is full, start a new line
            elif current_line:
                lines.append(indent + current_line)
                current_line = word
            else:
                # Single word is too long, force break
                lines.append(indent + word)

        # Add the last line if it has content
        if current_line:
            lines.append(indent + current_line)

        return lines

    @staticmethod
    def process_squashed_commit(commit_sha: str, repo_url: str) -> str:  # pylint: disable=too-many-locals,too-many-branches,too-many-statements,too-many-nested-blocks,unsubscriptable-object
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
        try:
            # Get the full commit message
            result = subprocess.run(
                ["git", "--no-pager", "show", commit_sha, "--format=%B", "--no-patch"],
                capture_output=True,
                check=True,
                text=True,
            )
            commit_msg = result.stdout
        except subprocess.CalledProcessError as e:
            raise GitRepoError(f"Failed to get commit message for {commit_sha}: {e}") from e

        # Get markdown line limit
        max_line_length = ChangelogUtils.get_markdown_line_limit()

        lines = commit_msg.strip().split("\n")
        content_lines = []

        # Skip the first line (PR title) and empty lines at start
        for line in lines[1:]:
            if line.strip() or content_lines:
                content_lines.append(line)

        # Remove trailing empty lines
        while content_lines and not content_lines[-1].strip():
            content_lines.pop()

        if not content_lines:
            return ""

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
                current_entry["body_lines"].append(line)  # pylint: disable=unsubscriptable-object
            elif not current_entry and line.strip():
                # No current entry and this is a non-empty line - treat as standalone entry
                current_entry = {"title": line.strip(), "body_lines": []}

        if current_entry:
            entries.append(current_entry)

        # Format entries
        output_lines = []
        for i, entry in enumerate(entries):
            if i > 0:
                output_lines.append("")  # Blank line between entries

            title = entry["title"]
            title_line = f"- **{title}** [`{commit_sha}`]({repo_url}/commit/{commit_sha})"

            # Handle title line wrapping
            if len(title_line) <= max_line_length:
                output_lines.append(title_line)
            else:
                # Title line is too long, try to wrap intelligently
                prefix = "- **"
                suffix = f"** [`{commit_sha}`]({repo_url}/commit/{commit_sha})"

                # Calculate space available for the title text
                available_for_title = max_line_length - len(prefix) - len(suffix)

                if available_for_title < 20:  # Very little space for title
                    # Split title and put commit link on next line
                    output_lines.append(f"- **{title}**")
                    output_lines.append(f"  [`{commit_sha}`]({repo_url}/commit/{commit_sha})")
                # Try to fit title on first line, wrap if necessary
                elif len(title) <= available_for_title:
                    # Title fits on one line with prefix and suffix
                    output_lines.append(title_line)
                else:
                    # Title needs wrapping - split at word boundaries
                    words = title.split()
                    first_line_words = []
                    remaining_words = words[:]

                    # Try to fit as many words as possible on the first line
                    while remaining_words:
                        test_title = " ".join(first_line_words + [remaining_words[0]])
                        if len(test_title) <= available_for_title:
                            first_line_words.append(remaining_words.pop(0))
                        else:
                            break

                    if not first_line_words:
                        # Not even one word fits, force break
                        output_lines.append(f"- **{title}**")
                        output_lines.append(f"  [`{commit_sha}`]({repo_url}/commit/{commit_sha})")
                    else:
                        # Output first line with prefix and some title words
                        first_line_title = " ".join(first_line_words)
                        if remaining_words:
                            # More words to wrap
                            output_lines.append(f"{prefix}{first_line_title}")
                            remaining_title = " ".join(remaining_words)
                            output_lines.append(f"  {remaining_title}{suffix}")
                        else:
                            # All words fit on first line
                            output_lines.append(f"{prefix}{first_line_title}{suffix}")

            # Process body content
            if entry["body_lines"]:
                body_content = []
                for line in entry["body_lines"]:
                    if line.strip():
                        body_content.append(line.strip())
                    elif body_content and body_content[-1]:
                        body_content.append("")  # Preserve paragraph breaks

                while body_content and not body_content[-1]:
                    body_content.pop()

                if body_content:
                    output_lines.append("")  # Blank line before body

                    for line in body_content:
                        if not line:  # Empty line - preserve as paragraph break
                            output_lines.append("")
                        elif line.startswith("    ") or "```" in line or re.search(r"\[.*\]\(.*\)|https?://\S+", line):
                            # Code blocks, links, or structured content - preserve as-is
                            output_lines.append(f"  {line}")
                        else:
                            # Regular text - wrap it
                            wrapped_lines = ChangelogUtils.wrap_markdown_line(line, max_line_length, "  ")
                            output_lines.extend(wrapped_lines)

        return "\n".join(output_lines)

    @staticmethod
    def run_git_command(args: list[str], check: bool = True) -> tuple[str, int]:
        """
        Run a git command and return output and exit code.

        Args:
            args: Git command arguments (without 'git')
            check: Whether to raise exception on non-zero exit code

        Returns:
            Tuple of (stdout, exit_code)

        Raises:
            GitRepoError: If command fails and check=True
        """
        try:
            result = subprocess.run(["git"] + args, capture_output=True, check=check, text=True)
            return result.stdout.strip(), result.returncode
        except subprocess.CalledProcessError as e:
            if check:
                raise GitRepoError(f"Git command failed: git {' '.join(args)}: {e.stderr}") from e
            return e.stdout.strip(), e.returncode
        except FileNotFoundError as exc:
            raise GitRepoError("git command not found") from exc
