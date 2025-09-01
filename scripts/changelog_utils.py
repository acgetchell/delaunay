#!/usr/bin/env python3
"""
Shared utilities for changelog operations.

This module provides common functionality used by multiple scripts
for changelog generation, parsing, and git tag management.
"""

import builtins
import contextlib
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
            subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True,
                check=True,
                text=True,
            )
            return True
        except subprocess.CalledProcessError as exc:
            msg = "Not in a git repository"
            raise GitRepoError(msg) from exc
        except FileNotFoundError as exc:
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
            subprocess.run(
                ["git", "log", "--oneline", "-n", "1"],
                capture_output=True,
                check=True,
                text=True,
            )
            return True
        except subprocess.CalledProcessError as exc:
            msg = "No git history found. Cannot generate changelog."
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
        # SemVer: vMAJOR.MINOR.PATCH with optional -PRERELEASE and optional +BUILD
        semver_pattern = r"^v[0-9]+(\.[0-9]+){2}(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$"

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
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True,
                check=True,
                text=True,
            )
            repo_url = result.stdout.strip()
        except subprocess.CalledProcessError as exc:
            msg = "Could not detect git remote origin URL"
            raise GitRepoError(msg) from exc

        if not repo_url:
            msg = "Git remote origin URL is empty"
            raise GitRepoError(msg)

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

        msg = "Cannot determine project root. CHANGELOG.md not found in current or parent directory."
        raise ChangelogError(msg)

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
            msg = f"Failed to get commit message for {commit_sha}: {e}"
            raise GitRepoError(msg) from e

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
                        test_title = " ".join([*first_line_words, remaining_words[0]])
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
            result = subprocess.run(["git", *args], capture_output=True, check=check, text=True)
            return result.stdout.strip(), result.returncode
        except subprocess.CalledProcessError as e:
            if check:
                msg = f"Git command failed: git {' '.join(args)}: {e.stderr}"
                raise GitRepoError(msg) from e
            return e.stdout.strip(), e.returncode
        except FileNotFoundError as exc:
            msg = "git command not found"
            raise GitRepoError(msg) from exc

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
        import sys

        # Colors for output
        RED = "\033[0;31m"
        GREEN = "\033[0;32m"
        YELLOW = "\033[1;33m"
        BLUE = "\033[0;34m"
        NC = "\033[0m"  # No Color

        # Validate git repo and prerequisites
        ChangelogUtils.validate_git_repo()
        ChangelogUtils.validate_semver(tag_version)

        # Check if tag already exists
        try:
            result = subprocess.run(
                ["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag_version}"],
                capture_output=True,
                check=False,
                text=True,
            )
            if result.returncode == 0:
                if not force_recreate:
                    print(f"{YELLOW}Tag '{tag_version}' already exists.{NC}", file=sys.stderr)
                    print("Use --force to recreate it, or delete it first with:", file=sys.stderr)
                    print(f"  git tag -d {tag_version}", file=sys.stderr)
                    raise ChangelogError(f"Tag '{tag_version}' already exists")
                print(f"{BLUE}Deleting existing tag '{tag_version}'...{NC}")
                subprocess.run(["git", "tag", "-d", tag_version], check=True)
        except subprocess.CalledProcessError as e:
            msg = f"Failed to check for existing tag: {e}"
            raise GitRepoError(msg) from e

        # Find changelog and extract content
        changelog_path = ChangelogUtils.find_changelog_path()
        version = ChangelogUtils.parse_version(tag_version)
        tag_message = ChangelogUtils.extract_changelog_section(changelog_path, version)

        # Show preview of tag message
        print(f"{BLUE}Tag message preview:{NC}")
        print("----------------------------------------")
        print(tag_message)
        print("----------------------------------------")

        # Check git user configuration
        try:
            subprocess.run(["git", "config", "--get", "user.name"], capture_output=True, check=True)
            subprocess.run(["git", "config", "--get", "user.email"], capture_output=True, check=True)
        except subprocess.CalledProcessError:
            print(f"{YELLOW}Warning: git user.name/email not configured; tag creation may fail.{NC}", file=sys.stderr)

        # Create the tag
        print(f"{BLUE}Creating tag '{tag_version}' with changelog content...{NC}")
        try:
            process = subprocess.Popen(
                ["git", "tag", "-a", tag_version, "-F", "-"],
                stdin=subprocess.PIPE,
                text=True,
            )
            process.communicate(input=tag_message)

            if process.returncode != 0:
                msg = f"Failed to create tag '{tag_version}'"
                raise GitRepoError(msg)

            print(f"{GREEN}✓ Successfully created tag '{tag_version}'{NC}")
            print("")
            print("Next steps:")
            print(f"  1. Push the tag: {BLUE}git push origin {tag_version}{NC}")
            print(f"  2. Create GitHub release: {BLUE}gh release create {tag_version} --notes-from-tag{NC}")

        except Exception as e:
            msg = f"Error creating tag: {e}"
            raise GitRepoError(msg) from e


def main() -> None:
    """
    Main entry point for changelog-utils CLI.

    This provides a Python replacement for generate_changelog.sh with the same
    functionality but better error handling and cross-platform support.
    """
    import argparse
    import shutil
    import signal
    import sys
    from pathlib import Path

    def show_help():
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

    def signal_handler(signum, frame):
        """Handle interrupt signals gracefully."""
        sys.exit(1)

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Parse command line arguments - handle both subcommand and legacy modes
    if len(sys.argv) > 1 and sys.argv[1] == "tag":
        # Tag subcommand mode
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
        return

    # Generate subcommand or legacy mode
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
        args = parser.parse_args(args_to_parse)
    except SystemExit:
        return

    if args.help:
        show_help()
        return

    if args.version:
        print("changelog-utils v0.4.1 (Python implementation)")
        print("Part of delaunay-scripts package")
        return

    # File paths
    changelog_file = Path("CHANGELOG.md")
    temp_changelog = changelog_file.with_suffix(".md.tmp")
    processed_changelog = changelog_file.with_suffix(".md.processed")
    expanded_changelog = processed_changelog.with_suffix(".processed.expanded")
    enhanced_changelog = changelog_file.with_suffix(".md.tmp2")
    backup_file = changelog_file.with_suffix(".md.backup")

    try:
        # Get project root
        project_root = Path(ChangelogUtils.get_project_root())
        original_cwd = Path.cwd()

        # Change to project root for auto-changelog
        import os

        os.chdir(project_root)

        try:
            # Validate git repository
            ChangelogUtils.validate_git_repo()
            ChangelogUtils.check_git_history()

            # Check prerequisites
            if not shutil.which("npx"):
                sys.exit(1)

            # Verify auto-changelog is available
            try:
                result = subprocess.run(["npx", "--yes", "-p", "auto-changelog", "auto-changelog", "--version"], capture_output=True, check=True, text=True)
            except subprocess.CalledProcessError:
                sys.exit(1)

            # Verify configuration files exist
            if not Path(".auto-changelog").exists():
                sys.exit(1)

            template_path = Path("docs/templates/changelog.hbs")
            if not template_path.exists():
                sys.exit(1)

            # Backup existing changelog
            if changelog_file.exists():
                shutil.copy2(changelog_file, backup_file)

            # Get repository URL
            try:
                repo_url = ChangelogUtils.get_repository_url()
            except GitRepoError:
                repo_url = "https://github.com/acgetchell/delaunay"  # Default fallback

            # Step 1: Run auto-changelog
            try:
                result = subprocess.run(
                    ["npx", "--yes", "-p", "auto-changelog", "auto-changelog", "--stdout"], capture_output=True, check=True, text=True, cwd=project_root
                )
                temp_changelog.write_text(result.stdout, encoding="utf-8")
            except subprocess.CalledProcessError as e:
                if e.stderr:
                    pass
                sys.exit(1)

            # Step 2: Post-process dates (ISO format -> YYYY-MM-DD)
            content = temp_changelog.read_text(encoding="utf-8")
            # Remove time portion from ISO dates
            processed_content = re.sub(r"T[0-9]{2}:[0-9]{2}:[0-9]{2}.*?Z", "", content)
            processed_changelog.write_text(processed_content, encoding="utf-8")

            # Step 3: Expand squashed PR commits
            ChangelogGenerator.expand_squashed_prs(processed_changelog, expanded_changelog, repo_url)

            # Step 4: Enhance AI commits with categorization
            if not shutil.which("python3"):
                sys.exit(1)

            # Find enhance_commits.py in the same directory as this script
            script_dir = Path(__file__).parent
            enhance_script = script_dir / "enhance_commits.py"

            if not enhance_script.exists():
                sys.exit(1)

            try:
                subprocess.run(["python3", str(enhance_script), str(expanded_changelog), str(enhanced_changelog)], check=True, cwd=project_root)
            except subprocess.CalledProcessError:
                sys.exit(1)

            # Step 5: Final cleanup - remove excessive blank lines
            content = enhanced_changelog.read_text(encoding="utf-8")
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

            # Write final changelog
            final_content = "\n".join(cleaned_lines)
            changelog_file.write_text(final_content, encoding="utf-8")

            # Clean up temporary files (unless debug mode)
            if not args.debug:
                for temp_file in [temp_changelog, processed_changelog, expanded_changelog, enhanced_changelog]:
                    if temp_file.exists():
                        temp_file.unlink()
            else:
                for temp_file in [temp_changelog, processed_changelog, expanded_changelog, enhanced_changelog]:
                    if temp_file.exists():
                        pass

            # Remove backup on success
            if backup_file.exists():
                backup_file.unlink()

            # Success messages
            with contextlib.suppress(builtins.BaseException):
                len([line for line in final_content.split("\n") if line.startswith("## ")])

        finally:
            # Always restore original working directory
            os.chdir(original_cwd)

    except (ChangelogError, GitRepoError, VersionError):
        # Restore backup if it exists
        if backup_file.exists() and changelog_file.exists():
            shutil.copy2(backup_file, changelog_file)
            backup_file.unlink()
        sys.exit(1)
    except KeyboardInterrupt:
        # Restore backup if it exists
        if backup_file.exists() and changelog_file.exists():
            shutil.copy2(backup_file, changelog_file)
            backup_file.unlink()
        sys.exit(1)
    except Exception:
        # Restore backup if it exists
        if backup_file.exists() and changelog_file.exists():
            shutil.copy2(backup_file, changelog_file)
            backup_file.unlink()
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
                    result = subprocess.run(["git", "--no-pager", "show", commit_sha, "--format=%s", "--no-patch"], capture_output=True, check=True, text=True)
                    commit_subject = result.stdout.strip()

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

                except subprocess.CalledProcessError:
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
