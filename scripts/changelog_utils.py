#!/usr/bin/env python3
"""
Shared utilities for changelog operations.

This module provides common functionality used by multiple scripts
for changelog generation, parsing, and git tag management.

Requires Python 3.13+ for modern typing features and datetime.UTC.
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import textwrap
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
            r"^v"
            r"(0|[1-9]\d*)\."
            r"(0|[1-9]\d*)\."
            r"(0|[1-9]\d*)"
            r"(?:-(?:"
            r"(?:0|[1-9]\d*)"
            r"|(?:[A-Za-z-][0-9A-Za-z-]*)"
            r")(?:\.(?:0|[1-9]\d*|[A-Za-z-][0-9A-Za-z-]*))*"
            r")?"
            r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
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
        Find the project root directory (where CHANGELOG.md is located).

        Returns:
            Absolute path to project root

        Raises:
            ChangelogError: If project root cannot be determined
        """
        try:
            return str(Path(ChangelogUtils.find_changelog_path()).parent)
        except ChangelogNotFoundError as e:
            msg = "Cannot determine project root. CHANGELOG.md not found in current or parent directory."
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
        content_lines = []

        # Skip the first line (PR title) and empty lines at start
        trailer_re = re.compile(r"^\s*(Co-authored-by|Signed-off-by|Change-Id|Reviewed-on|Reviewed-by|Refs|See-Also):", re.I)
        for line in lines[1:]:
            if trailer_re.match(line):
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
        """
        Format an entry title with commit link.

        Args:
            title: Entry title
            commit_sha: Git commit SHA
            repo_url: Repository URL
            max_line_length: Maximum line length

        Returns:
            List of formatted title lines
        """
        escaped_title = ChangelogUtils.escape_markdown(title)
        title_line = f"- **{escaped_title}** [`{commit_sha}`]({repo_url}/commit/{commit_sha})"

        # Keep bolded title intact; place commit link on its own line if too long
        if len(title_line) <= max_line_length:
            return [title_line]

        # If the title alone (with bullet and bold formatting) is still too long, wrap it
        title_only = f"- **{escaped_title}**"
        if len(title_only) > max_line_length:
            # Compute exact overhead; drop bold if the line would be too cramped.
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

            # Add commit link on a separate line; split only if necessary.
            commit_link = f"  [`{commit_sha}`]({repo_url}/commit/{commit_sha})"
            if len(commit_link) <= max_line_length:
                result_lines.append(commit_link)
            else:
                result_lines.append(f"  [`{commit_sha}`]")
                result_lines.append(f"  ({repo_url}/commit/{commit_sha})")
            return result_lines

        # Title fits but full line doesn't - split normally
        commit_link = f"  [`{commit_sha}`]({repo_url}/commit/{commit_sha})"
        if len(commit_link) <= max_line_length:
            return [f"- **{escaped_title}**", commit_link]
        # Very short limit - split commit link too
        return [f"- **{escaped_title}**", f"  [`{commit_sha}`]", f"  ({repo_url}/commit/{commit_sha})"]

    @staticmethod
    def _format_entry_body(body_lines: list[str], max_line_length: int) -> list[str]:
        """
        Format entry body content with proper wrapping.

        Args:
            body_lines: Raw body lines
            max_line_length: Maximum line length

        Returns:
            List of formatted body lines
        """
        if not body_lines:
            return []

        body_content = []
        for line in body_lines:
            if line.strip():
                body_content.append(line.strip())
            elif body_content and body_content[-1]:
                body_content.append("")  # Preserve paragraph breaks

        while body_content and not body_content[-1]:
            body_content.pop()

        if not body_content:
            return []

        output_lines = [""]  # Blank line before body

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

        return output_lines

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
        This enables GitHub releases to use the changelog content via --notes-from-tag.

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
        ChangelogUtils._handle_existing_tag(tag_version, force_recreate)

        # Get changelog content
        tag_message = ChangelogUtils._get_changelog_content(tag_version)

        # Check git configuration
        ChangelogUtils._check_git_config()

        # Create the tag
        ChangelogUtils._create_tag_with_message(tag_version, tag_message)

        # Show success message
        ChangelogUtils._show_success_message(tag_version)

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
        YELLOW = "\033[1;33m"
        BLUE = "\033[0;34m"
        NC = "\033[0m"

        try:
            _, result_code = ChangelogUtils.run_git_command(["rev-parse", "-q", "--verify", f"refs/tags/{tag_version}"], check=False)
            if result_code == 0:
                if not force_recreate:
                    print(f"{YELLOW}Tag '{tag_version}' already exists.{NC}", file=sys.stderr)
                    print("Use --force to recreate it, or delete it first with:", file=sys.stderr)
                    print(f"  git tag -d {tag_version}", file=sys.stderr)
                    raise ChangelogError(f"Tag '{tag_version}' already exists")
                print(f"{BLUE}Deleting existing tag '{tag_version}'...{NC}")
                ChangelogUtils.run_git_command(["tag", "-d", tag_version])
        except subprocess.CalledProcessError as e:
            msg = f"Failed to check for existing tag: {e}"
            raise GitRepoError(msg) from e

    @staticmethod
    def _get_changelog_content(tag_version: str) -> str:
        """
        Get and preview changelog content for the tag.

        Args:
            tag_version: The version tag

        Returns:
            Changelog content for the tag message
        """
        BLUE = "\033[0;34m"
        NC = "\033[0m"

        changelog_path = ChangelogUtils.find_changelog_path()
        version = ChangelogUtils.parse_version(tag_version)
        tag_message = ChangelogUtils.extract_changelog_section(changelog_path, version)

        # Show preview
        print(f"{BLUE}Tag message preview:{NC}")
        print("----------------------------------------")
        print(tag_message)
        print("----------------------------------------")

        return tag_message

    @staticmethod
    def _check_git_config() -> None:
        """
        Check git user configuration and warn if not set.
        """
        YELLOW = "\033[1;33m"
        NC = "\033[0m"

        try:
            ChangelogUtils.run_git_command(["config", "--get", "user.name"])
            ChangelogUtils.run_git_command(["config", "--get", "user.email"])
        except Exception:
            print(f"{YELLOW}Warning: git user.name/email not configured; tag creation may fail.{NC}", file=sys.stderr)

    @staticmethod
    def _create_tag_with_message(tag_version: str, tag_message: str) -> None:
        """
        Create the git tag with the provided message.

        Args:
            tag_version: The version tag to create
            tag_message: The tag message content

        Raises:
            GitRepoError: If tag creation fails
        """
        BLUE = "\033[0;34m"
        NC = "\033[0m"

        print(f"{BLUE}Creating tag '{tag_version}' with changelog content...{NC}")

        try:
            # Tag format already validated by validate_semver(); no second check needed

            # Use secure wrapper for git command with stdin input
            run_git_command_with_input(["tag", "-a", tag_version, "-F", "-"], input_data=tag_message)

        except Exception as e:
            msg = f"Error creating tag: {e}"
            raise GitRepoError(msg) from e

    @staticmethod
    def _show_success_message(tag_version: str) -> None:
        """
        Show success message and next steps.

        Args:
            tag_version: The created tag version
        """
        GREEN = "\033[0;32m"
        BLUE = "\033[0;34m"
        NC = "\033[0m"

        print(f"{GREEN}✓ Successfully created tag '{tag_version}'{NC}")
        print("")
        print("Next steps:")
        print(f"  1. Push the tag: {BLUE}git push origin {tag_version}{NC}")
        print(f"  2. Create GitHub release: {BLUE}gh release create {tag_version} --notes-from-tag{NC}")


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
    except SystemExit:
        sys.exit(0)


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
  - CHANGELOG.md.tmp (initial auto-changelog output)
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
    # Initialize file paths
    file_paths = _initialize_file_paths()

    try:
        # Setup and validation
        project_root, original_cwd = _setup_project_environment()

        try:
            # Core workflow steps
            _validate_prerequisites()
            repo_url = _get_repository_url()
            _backup_existing_changelog(file_paths)

            # Execute processing pipeline
            _run_auto_changelog(file_paths, project_root)
            _post_process_dates(file_paths)
            _expand_squashed_commits(file_paths, repo_url)
            _enhance_with_ai(file_paths, project_root)
            _cleanup_final_output(file_paths)

            # Cleanup and success
            _cleanup_temp_files(file_paths, debug_mode)
            _show_success_message(file_paths)

        finally:
            os.chdir(original_cwd)

    except (ChangelogError, GitRepoError, VersionError):
        _restore_backup_and_exit(file_paths)
    except KeyboardInterrupt:
        _restore_backup_and_exit(file_paths)
    except Exception:
        _restore_backup_and_exit(file_paths)


def _initialize_file_paths() -> dict[str, Path]:
    """Initialize file paths for changelog processing."""
    changelog_file = Path("CHANGELOG.md")
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

    if not shutil.which("npx"):
        print("Error: npx not found. Install Node.js (which provides npx). See https://nodejs.org/", file=sys.stderr)
        sys.exit(1)

    # Verify auto-changelog availability
    try:
        run_safe_command("npx", ["--yes", "-p", "auto-changelog", "auto-changelog", "--version"])
    except Exception:
        print("Error: auto-changelog is not available via npx. Verify network access and try again.", file=sys.stderr)
        sys.exit(1)

    # Verify configuration files
    if not Path(".auto-changelog").exists():
        print("Error: .auto-changelog config not found at project root.", file=sys.stderr)
        sys.exit(1)

    template_path = Path("docs/templates/changelog.hbs")
    if not template_path.exists():
        print(f"Error: changelog template missing: {template_path}", file=sys.stderr)
        sys.exit(1)


def _get_repository_url() -> str:
    """Get repository URL with fallback."""
    try:
        return ChangelogUtils.get_repository_url()
    except GitRepoError:
        return "https://github.com/acgetchell/delaunay"  # Default fallback


def _backup_existing_changelog(file_paths: dict[str, Path]) -> None:
    """Backup existing changelog if it exists."""
    if file_paths["changelog"].exists():
        shutil.copy2(file_paths["changelog"], file_paths["backup"])


def _run_auto_changelog(file_paths: dict[str, Path], project_root: Path) -> None:
    """Run auto-changelog to generate initial changelog."""
    try:
        result = run_safe_command("npx", ["--yes", "-p", "auto-changelog", "auto-changelog", "--stdout"], cwd=project_root)
        file_paths["temp"].write_text(result.stdout, encoding="utf-8")
    except subprocess.CalledProcessError as e:
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        print("Error: auto-changelog failed.", file=sys.stderr)
        sys.exit(1)


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

        content = input_file.read_text(encoding="utf-8")
        lines = content.split("\n")
        output_lines = []

        for line in lines:
            # Check for commit lines with SHA patterns
            commit_match = re.search(r"- \*\*.*?\*\*.*?\[`([a-f0-9]{7,40})`\]", line) or re.search(r"- .*?\(#[0-9]+\) \[`([a-f0-9]{7,40})`\]", line)

            if commit_match:
                commit_sha = commit_match.group(1)

                # Check if this is a squashed PR commit
                try:
                    result_output, _ = ChangelogUtils.run_git_command(["--no-pager", "show", commit_sha, "--format=%s", "--no-patch"])
                    commit_subject = result_output.strip()

                    if re.search(r"\(#[0-9]+\)$", commit_subject):
                        # Use Python utility to process the commit
                        try:
                            processed_commit = ChangelogUtils.process_squashed_commit(commit_sha, repo_url)
                            if processed_commit.strip():
                                output_lines.append(processed_commit)
                            else:
                                # Fallback to original line if processing failed
                                output_lines.append(line)
                        except Exception:
                            # Fallback to original line if processing failed
                            output_lines.append(line)
                    else:
                        # Not a squashed PR, keep original line
                        output_lines.append(line)

                except Exception:
                    # Git command failed, keep original line
                    output_lines.append(line)
            else:
                # Not a commit line, keep as is
                output_lines.append(line)

        # Write processed content
        output_content = "\n".join(output_lines)
        output_file.write_text(output_content, encoding="utf-8")


if __name__ == "__main__":
    main()
