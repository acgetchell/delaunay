#!/usr/bin/env python3
"""Create annotated git tags from CHANGELOG.md sections.

Handles GitHub's 125KB tag-annotation size limit by falling back to a short
reference message when the changelog section is too large.

Usage:
    tag-release v1.2.3          # create annotated tag from CHANGELOG.md
    tag-release v1.2.3 --force  # recreate tag if it already exists
    tag-release v1.2.3 --debug  # verbose output

Ported from the la-stack project's tag_release.py.
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path

from subprocess_utils import (
    ExecutableNotFoundError,
    run_git_command,
    run_git_command_with_input,
)

# GitHub's maximum size for git tag annotations (bytes)
_GITHUB_TAG_ANNOTATION_LIMIT = 125_000

# ANSI color codes for terminal output
_GREEN = "\033[0;32m"
_BLUE = "\033[0;34m"
_YELLOW = "\033[1;33m"
_RESET = "\033[0m"

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SemVer validation
# ---------------------------------------------------------------------------

# SemVer 2.0.0 strict with required 'v' prefix
# Alphanumeric prerelease identifier: any [0-9A-Za-z-]+ containing at least one
# non-digit.  This permits identifiers like "1a" that start with a digit but are
# not purely numeric (SemVer 2.0.0 §9).
_ALNUM_ID = r"(?:(?=[0-9A-Za-z-]*[A-Za-z-])[0-9A-Za-z-]+)"
_SEMVER_RE = re.compile(
    r"^v"
    r"(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)"
    rf"(?:-(?:(?:0|[1-9]\d*)|{_ALNUM_ID})"
    rf"(?:\.(?:(?:0|[1-9]\d*)|{_ALNUM_ID}))*"
    r")?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


def validate_semver(tag_version: str) -> None:
    """Raise ``ValueError`` if *tag_version* is not valid ``vX.Y.Z`` SemVer."""
    if not _SEMVER_RE.match(tag_version):
        msg = f"Tag version should follow SemVer format 'vX.Y.Z' (e.g., v0.3.5, v1.2.3-rc.1). Got: {tag_version}"
        raise ValueError(msg)


def parse_version(tag_version: str) -> str:
    """Return version string without leading ``v``."""
    return tag_version.removeprefix("v")


# ---------------------------------------------------------------------------
# Changelog helpers
# ---------------------------------------------------------------------------


def find_changelog(start: Path | None = None) -> Path:
    """Locate ``CHANGELOG.md`` in *start* or its parent.

    Raises:
        FileNotFoundError: If ``CHANGELOG.md`` cannot be found.
    """
    base = start or Path.cwd()
    for candidate in (base / "CHANGELOG.md", base.parent / "CHANGELOG.md"):
        if candidate.is_file():
            return candidate
    msg = "CHANGELOG.md not found in current directory or parent directory."
    raise FileNotFoundError(msg)


def extract_changelog_section(changelog: Path, version: str) -> str:
    """Extract the changelog body for *version* (without ``v`` prefix).

    Raises:
        LookupError: If the version section is not found or empty.
    """
    content = changelog.read_text(encoding="utf-8")
    header_re = _version_header_re(version)

    lines = content.split("\n")
    section: list[str] = []
    collecting = False

    for line in lines:
        if re.match(r"^##\s", line):
            if collecting:
                break
            if header_re.match(line):
                collecting = True
                continue
        elif collecting:
            section.append(line)

    if not collecting:
        msg = f"No changelog section found for version {version}. Expected a heading like: ## [{version}] - YYYY-MM-DD"
        raise LookupError(msg)

    # Trim leading/trailing blank lines (O(n) index scan + slice)
    start = 0
    while start < len(section) and not section[start].strip():
        start += 1
    end = len(section)
    while end > start and not section[end - 1].strip():
        end -= 1
    section = section[start:end]

    body = "\n".join(section)
    if not body.strip():
        msg = f"Changelog section for version {version} is empty."
        raise LookupError(msg)
    return body


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _tag_exists(tag_version: str) -> bool:
    """Return ``True`` if *tag_version* already exists as a git tag."""
    try:
        run_git_command(["rev-parse", "-q", "--verify", f"refs/tags/{tag_version}"])
    except subprocess.CalledProcessError:
        return False
    else:
        return True


def _delete_tag(tag_version: str) -> None:
    run_git_command(["tag", "-d", tag_version])


def _get_repo_url() -> str:
    """Detect the GitHub HTTPS URL from the ``origin`` remote."""
    result = run_git_command(["remote", "get-url", "origin"])
    raw = result.stdout.strip()
    patterns = [
        r"^git@github\.com:(?P<slug>[^/]+/[^/]+?)(?:\.git)?/?$",
        r"^https://github\.com/(?P<slug>[^/]+/[^/]+?)(?:\.git)?/?$",
        r"^ssh://git@github\.com[:/](?P<slug>[^/]+/[^/]+?)(?:\.git)?/?$",
    ]
    for pat in patterns:
        m = re.match(pat, raw)
        if m:
            return f"https://github.com/{m.group('slug')}"
    return raw  # best-effort fallback


def _version_header_re(version: str) -> re.Pattern[str]:
    """Build the header regex for *version*, matching ``extract_changelog_section``."""
    return re.compile(rf"^##\s*\[?v?{re.escape(version)}\]?(?:$|\s|\()")


def _github_anchor(changelog: Path, version: str) -> str:
    """Build a GitHub-compatible heading anchor (matches ``github-slugger``)."""
    header_re = _version_header_re(version)
    try:
        for line in changelog.read_text(encoding="utf-8").splitlines():
            if header_re.match(line):
                heading = line.removeprefix("## ").strip()
                # Strip inline-link markup [text](url) → text
                heading = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", heading)
                # Strip reference-style brackets [text] → text
                heading = re.sub(r"\[([^\]]+)\]", r"\1", heading)
                heading = heading.lower()
                # Remove everything except letters, digits, spaces, hyphens
                heading = re.sub(r"[^a-z0-9\s-]", "", heading)
                # Replace whitespace runs with a single hyphen
                return re.sub(r"\s+", "-", heading)
    except OSError:
        pass
    return re.sub(r"[^a-z0-9-]", "", f"v{version}".lower())


# ---------------------------------------------------------------------------
# Core workflow
# ---------------------------------------------------------------------------


def create_tag(tag_version: str, *, force: bool = False) -> None:
    """Create an annotated git tag with changelog content.

    If the changelog section exceeds GitHub's 125KB limit, creates the tag
    with a short reference message instead.
    """
    validate_semver(tag_version)
    version = parse_version(tag_version)

    # Check for existing tag (but don't delete yet — validate first)
    tag_existed = _tag_exists(tag_version)
    if tag_existed and not force:
        print(f"{_YELLOW}Tag '{tag_version}' already exists.{_RESET}", file=sys.stderr)
        print(f"Use --force to recreate, or delete manually: git tag -d {tag_version}", file=sys.stderr)
        sys.exit(1)

    # Extract changelog section (before any mutation)
    changelog = find_changelog()
    section = extract_changelog_section(changelog, version)
    section_bytes = len(section.encode("utf-8"))

    # Check size limit
    if section_bytes > _GITHUB_TAG_ANNOTATION_LIMIT:
        print(f"{_YELLOW}⚠ Changelog section ({section_bytes:,} bytes) exceeds GitHub's tag limit ({_GITHUB_TAG_ANNOTATION_LIMIT:,} bytes){_RESET}")
        anchor = _github_anchor(changelog, version)
        repo_url = _get_repo_url()
        tag_message = (
            f"Version {version}\n\n"
            f"This release contains extensive changes. See full changelog:\n"
            f"<{repo_url}/blob/{tag_version}/CHANGELOG.md#{anchor}>\n\n"
            f"For detailed release notes, refer to CHANGELOG.md in the repository.\n"
        )
        is_truncated = True
        print(f"{_BLUE}→ Creating annotated tag with CHANGELOG.md reference{_RESET}")
    else:
        tag_message = section
        is_truncated = False
        print(f"{_BLUE}Tag message preview ({section_bytes:,} bytes):{_RESET}")
        preview = section.split("\n")[:20]
        print("----------------------------------------")
        print("\n".join(preview))
        if len(section.split("\n")) > 20:
            print("... (truncated for preview)")
        print("----------------------------------------")

    # Delete existing tag only after all validation succeeds
    if tag_existed and force:
        print(f"{_BLUE}Deleting existing tag '{tag_version}'...{_RESET}")
        _delete_tag(tag_version)

    # Create annotated tag
    label = "reference" if is_truncated else "full changelog"
    print(f"{_BLUE}Creating annotated tag '{tag_version}' with {label} content...{_RESET}")
    run_git_command_with_input(["tag", "-a", tag_version, "-F", "-"], input_data=tag_message)

    # Success
    print(f"{_GREEN}✓ Successfully created tag '{tag_version}'{_RESET}")
    print()
    print("Next steps:")
    if force:
        print(f"  1. Force-push the tag: {_BLUE}git push --force origin {tag_version}{_RESET}")
    else:
        print(f"  1. Push the tag: {_BLUE}git push origin {tag_version}{_RESET}")
    print(f"  2. Create GitHub release: {_BLUE}gh release create {tag_version} --notes-from-tag{_RESET}")
    if is_truncated:
        print(f"\n{_YELLOW}Note: Tag annotation references CHANGELOG.md due to size (>125KB).{_RESET}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for ``tag-release``."""
    parser = argparse.ArgumentParser(
        prog="tag-release",
        description="Create an annotated git tag from a CHANGELOG.md section.",
    )
    parser.add_argument("version", help="Tag version (e.g. v1.2.3)")
    parser.add_argument("--force", action="store_true", help="Recreate tag if it already exists")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    try:
        create_tag(args.version, force=args.force)
    except (
        ValueError,
        FileNotFoundError,
        LookupError,
        ExecutableNotFoundError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
