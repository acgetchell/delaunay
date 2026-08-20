#!/usr/bin/env -S uv run
"""Archive completed minor series from CHANGELOG.md into per-minor files.

Parses the full CHANGELOG.md (produced by git-cliff + postprocess-changelog)
into version blocks, groups them by minor series (X.Y), and writes:

  - ``docs/archive/changelog/X.Y.md`` for each completed minor series
  - A trimmed ``CHANGELOG.md`` containing only the preamble, Unreleased,
    the active minor series, and an Archives link section

The active minor is detected from the first tagged release heading after
Unreleased.  All other minors are archived.

Usage:
    archive-changelog                      # default: CHANGELOG.md
    archive-changelog path/to/CHANGELOG.md
    archive-changelog --archive-dir docs/archive/changelog
"""

import argparse
import logging
import os
import re
import secrets
import shutil
import stat
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, cast

from postprocess_changelog import normalize_entry_headings_text, postprocess_text

if TYPE_CHECKING:
    from collections.abc import Sequence

# Matches any bracketed level-2 heading reserved for changelog versions.
_BRACKETED_HEADING_RE = re.compile(r"^## \[")

_SEMVER_NUMERIC_IDENTIFIER = r"(?:0|[1-9]\d*)"
_SEMVER_PRERELEASE_IDENTIFIER = r"(?:0|[1-9]\d*|[0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*)"
_SEMVER_BUILD_IDENTIFIER = r"[0-9A-Za-z-]+"
_SEMVER_PATTERN = (
    rf"{_SEMVER_NUMERIC_IDENTIFIER}\.{_SEMVER_NUMERIC_IDENTIFIER}\.{_SEMVER_NUMERIC_IDENTIFIER}"
    rf"(?:-{_SEMVER_PRERELEASE_IDENTIFIER}(?:\.{_SEMVER_PRERELEASE_IDENTIFIER})*)?"
    rf"(?:\+{_SEMVER_BUILD_IDENTIFIER}(?:\.{_SEMVER_BUILD_IDENTIFIER})*)?"
)

# Matches one exact release heading with an optional ISO date.
_RELEASE_HEADING_RE = re.compile(rf"^## \[(?P<version>{_SEMVER_PATTERN})\](?: - (?P<date>\d{{4}}-\d{{2}}-\d{{2}}))?\s*$")
_UNRELEASED_HEADING_RE = re.compile(r"^## \[Unreleased\]\s*$")

# Matches a reference-style link definition: ``[label]: URL``
_LINK_DEF_RE = re.compile(r"^\[([^\]]+)\]:\s+\S+")

# Archive directory relative to the repository root.
_DEFAULT_ARCHIVE_DIR = "docs/archive/changelog"

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ParsedChangelog:
    """A changelog whose version-heading invariants have been established."""

    preamble: str
    unreleased: str | None
    version_blocks: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _StagedWrite:
    """One fully staged target and its optional pre-publication backup."""

    target: Path
    staged: Path
    backup: Path | None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _minor_key(version: str) -> str:
    """Return the ``X.Y`` minor key for a semver version string.

    Parameters:
        version: A version string like ``0.7.2`` or ``1.2.3-rc.1``.

    Returns:
        The first two numeric components joined by a dot (e.g. ``0.7``).

    Raises:
        ValueError: If *version* does not contain at least two dot-separated components.
    """
    parts = version.split(".")
    if len(parts) < 2:
        msg = f"Expected a version with at least two components (X.Y), got: {version!r}"
        raise ValueError(msg)
    return f"{parts[0]}.{parts[1]}"


def _version_sort_key(label: str) -> tuple[bool, tuple[int, ...], tuple[tuple[int, int, str], ...]]:
    """Return a sort key for a version label that orders by semantic version.

    Non-numeric labels (e.g. ``unreleased``) sort after all numeric versions.
    Numeric parts are compared as integers so that ``0.10`` sorts after ``0.9``.

    Parameters:
        label: A version label like ``0.7.2``, ``0.10``, or ``unreleased``.

    Returns:
        A tuple suitable for use as a sort key.
    """
    label_without_build = label.split("+", 1)[0]
    core, separator, prerelease = label_without_build.partition("-")
    parts = core.split(".")
    try:
        nums = tuple(int(p) for p in parts)
    except ValueError:
        # Non-numeric labels ("unreleased") sort last (True > False).
        return (True, (), ())

    if not separator:
        prerelease_key: tuple[tuple[int, int, str], ...] = ((2, 0, ""),)
    else:
        prerelease_key = tuple((0, int(part), "") if part.isdecimal() else (1, 0, part) for part in prerelease.split("."))

    return (False, nums, prerelease_key)


def _is_strictly_older(version: str, preceding_version: str) -> bool:
    """Return whether *version* has lower SemVer precedence than its predecessor."""
    version_key = _version_sort_key(version)
    preceding_key = _version_sort_key(preceding_version)
    return version_key < preceding_key


def _extract_link_defs(text: str) -> tuple[str, dict[str, str]]:
    """Separate trailing reference-style link definitions from changelog text.

    git-cliff appends reference-style link definitions at the bottom of
    CHANGELOG.md for every version heading.  When the changelog is split
    into per-version blocks these definitions must be distributed to the
    correct output files so that headings like ``## [0.7.2]`` resolve and
    no unused definitions trigger markdownlint MD053.

    Parameters:
        text: The full changelog text.

    Returns:
        A 2-tuple of (*cleaned_text*, *link_defs*) where *link_defs* maps
        lowercase labels to their full definition lines.
    """
    lines = text.rstrip("\n").split("\n")
    link_defs: dict[str, str] = {}

    # Walk backwards from the end, collecting link-def and blank lines.
    i = len(lines) - 1
    while i >= 0:
        line = lines[i]
        m = _LINK_DEF_RE.match(line)
        if m:
            link_defs[m.group(1).lower()] = line
            i -= 1
        elif line.strip() == "":
            i -= 1
        else:
            break

    cleaned = "\n".join(lines[: i + 1])
    return cleaned.rstrip("\n") + "\n", link_defs


def _parse_release_heading(heading_line: str, line_number: int) -> str:
    """Return the trusted SemVer label from one release heading."""
    release_match = _RELEASE_HEADING_RE.fullmatch(heading_line)
    if release_match is None:
        msg = f"Unrecognized changelog heading at line {line_number}: {heading_line!r}"
        raise ValueError(msg)

    release_date = release_match.group("date")
    if release_date is not None:
        try:
            date.fromisoformat(release_date)
        except ValueError as err:
            msg = f"Invalid release date at line {line_number}: {release_date!r}"
            raise ValueError(msg) from err

    return cast("str", release_match.group("version"))


def parse_changelog(text: str) -> ParsedChangelog:
    """Parse a full changelog into trusted preamble and version blocks.

    Parameters:
        text: The full contents of CHANGELOG.md.

    Returns:
        An immutable parsed changelog. ``unreleased`` is ``None`` when no
        ``## [Unreleased]`` block exists. Each item in ``version_blocks`` is a
        ``(semver_label, full_heading_block)`` pair in strict newest-first
        order.

    Raises:
        ValueError: If a reserved bracketed heading is unknown, a release
            label or date is invalid, headings are duplicated, ``Unreleased``
            is not first, or releases are not newest-first.
    """
    lines = text.split("\n")

    # Locate all ``## [`` headings.
    headings: list[int] = []
    for i, line in enumerate(lines):
        if _BRACKETED_HEADING_RE.match(line):
            headings.append(i)

    if not headings:
        return ParsedChangelog(text, None, ())

    preamble = "\n".join(lines[: headings[0]])

    unreleased: str | None = None
    version_blocks: list[tuple[str, str]] = []
    seen_versions: dict[str, int] = {}
    previous_version: str | None = None

    for heading_index, start in enumerate(headings):
        end = headings[heading_index + 1] if heading_index + 1 < len(headings) else len(lines)
        block = "\n".join(lines[start:end])

        heading_line = lines[start]
        line_number = start + 1
        if _UNRELEASED_HEADING_RE.fullmatch(heading_line):
            if unreleased is not None:
                msg = f"Duplicate Unreleased heading at line {line_number}"
                raise ValueError(msg)
            if heading_index != 0:
                msg = f"Unreleased heading must be the first version heading (line {line_number})"
                raise ValueError(msg)
            unreleased = block
            continue

        version = _parse_release_heading(heading_line, line_number)

        if version in seen_versions:
            msg = f"Duplicate release heading {version!r} at line {line_number}; first seen at line {seen_versions[version]}"
            raise ValueError(msg)

        if previous_version is not None and not _is_strictly_older(version, previous_version):
            msg = f"Release heading out of order at line {line_number}: {version!r} must be older than preceding {previous_version!r}"
            raise ValueError(msg)

        seen_versions[version] = line_number
        previous_version = version
        version_blocks.append((version, block))

    return ParsedChangelog(preamble, unreleased, tuple(version_blocks))


def group_by_minor(
    version_blocks: Sequence[tuple[str, str]],
) -> dict[str, list[tuple[str, str]]]:
    """Group version blocks by their ``X.Y`` minor key.

    Preserves insertion order (newest first within each minor).

    Parameters:
        version_blocks: List of ``(version, block_text)`` pairs.

    Returns:
        An ordered dict mapping minor keys to their version blocks.
    """
    groups: dict[str, list[tuple[str, str]]] = {}
    for ver, block in version_blocks:
        key = _minor_key(ver)
        groups.setdefault(key, []).append((ver, block))
    return groups


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _open_sibling_temporary(path: Path, suffix: str) -> tuple[int, Path]:
    """Create an owner-only collision-resistant temporary file beside *path*."""
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    for _attempt in range(100):
        candidate = path.with_name(f".{path.name}.{secrets.token_hex(12)}{suffix}")
        try:
            return os.open(candidate, flags, 0o600), candidate
        except FileExistsError:
            continue

    msg = f"Could not reserve a temporary file beside {path}"
    raise FileExistsError(msg)


def _stage_text(path: Path, text: str) -> Path:
    """Write and sync *text* to an unpublished sibling of *path*."""
    existing_mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    descriptor, staged_path = _open_sibling_temporary(path, ".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if existing_mode is not None:
            staged_path.chmod(existing_mode)
    except BaseException:
        staged_path.unlink(missing_ok=True)
        raise
    return staged_path


def _stage_backup(path: Path) -> Path:
    """Copy and sync *path* to an unpublished rollback backup."""
    descriptor, backup_path = _open_sibling_temporary(path, ".bak")
    os.close(descriptor)
    try:
        shutil.copyfile(path, backup_path)
        with backup_path.open("rb+") as handle:
            os.fsync(handle.fileno())
        shutil.copystat(path, backup_path)
    except BaseException:
        backup_path.unlink(missing_ok=True)
        raise
    return backup_path


def _replace_path(source: Path, destination: Path) -> None:
    """Replace *destination* with *source* through a testable seam."""
    source.replace(destination)


def _cleanup_temporary_paths(paths: Sequence[Path], preserved: set[Path] | None = None) -> None:
    """Best-effort cleanup for unpublished transaction files."""
    preserved = preserved or set()
    for path in paths:
        if path in preserved:
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError as err:
            LOGGER.warning("Could not remove transaction temporary file %s: %s", path, err)


def _remove_created_directories(directories: Sequence[Path]) -> None:
    """Remove transaction-created directories that are still empty."""
    for directory in sorted(set(directories), key=lambda path: len(path.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            # A concurrent writer or an incomplete rollback may have populated it.
            continue


def _ensure_parent_directory(path: Path) -> list[Path]:
    """Create *path* and return the directories created by this call."""
    missing: list[Path] = []
    candidate = path
    while not candidate.exists():
        missing.append(candidate)
        candidate = candidate.parent
    path.mkdir(parents=True, exist_ok=True)
    return missing


def _transaction_temporary_paths(staged_writes: Sequence[_StagedWrite]) -> list[Path]:
    """Return every temporary path owned by *staged_writes*."""
    return [path for item in staged_writes for path in (item.staged, item.backup) if path is not None]


def _stage_writes(writes: Sequence[tuple[Path, str]]) -> tuple[list[_StagedWrite], list[Path]]:
    """Prepare replacement files and rollback backups without publishing."""
    created_directories: list[Path] = []
    staged_writes: list[_StagedWrite] = []
    try:
        for target, text in writes:
            created_directories.extend(_ensure_parent_directory(target.parent))
            staged_path = _stage_text(target, text)
            try:
                backup_path = _stage_backup(target) if target.exists() else None
            except BaseException:
                staged_path.unlink(missing_ok=True)
                raise
            staged_writes.append(_StagedWrite(target, staged_path, backup_path))
    except BaseException:
        _cleanup_temporary_paths(_transaction_temporary_paths(staged_writes))
        _remove_created_directories(created_directories)
        raise
    return staged_writes, created_directories


def _rollback_committed(committed: Sequence[_StagedWrite]) -> tuple[list[OSError], set[Path]]:
    """Restore committed targets, returning rollback failures and saved backups."""
    rollback_errors: list[OSError] = []
    preserved_backups: set[Path] = set()
    for item in reversed(committed):
        try:
            if item.backup is None:
                item.target.unlink(missing_ok=True)
            else:
                _replace_path(item.backup, item.target)
        except OSError as rollback_error:
            rollback_error.add_note(f"Could not restore {item.target}")
            rollback_errors.append(rollback_error)
            if item.backup is not None:
                preserved_backups.add(item.backup)
    return rollback_errors, preserved_backups


def _write_texts_transactionally(writes: Sequence[tuple[Path, str]]) -> None:
    """Publish UTF-8 files as one rollback-capable transaction.

    All replacement files and backups are prepared before the first visible
    change. If a later replacement fails, every earlier replacement is restored
    before the original error is re-raised.
    """
    if not writes:
        return

    targets = [path for path, _text in writes]
    if len(set(targets)) != len(targets):
        msg = "A publication transaction cannot contain duplicate target paths"
        raise ValueError(msg)

    staged_writes, created_directories = _stage_writes(writes)

    committed: list[_StagedWrite] = []
    try:
        for item in staged_writes:
            _replace_path(item.staged, item.target)
            committed.append(item)
    except BaseException as publication_error:
        rollback_errors, preserved_backups = _rollback_committed(committed)
        _cleanup_temporary_paths(_transaction_temporary_paths(staged_writes), preserved_backups)
        _remove_created_directories(created_directories)

        if rollback_errors:
            publication_error.add_note("One or more rollback backups were preserved beside their target files")
            message = "Changelog publication failed and rollback was incomplete"
            raise BaseExceptionGroup(
                message,
                [publication_error, *rollback_errors],
            ) from None
        raise

    backup_paths = [item.backup for item in staged_writes if item.backup is not None]
    _cleanup_temporary_paths(backup_paths)


def _write_text_atomic(path: Path, text: str) -> None:
    """Atomically replace one UTF-8 file while preserving it on failure."""
    _write_texts_transactionally(((path, text),))


def _format_link_defs(link_defs: dict[str, str], labels: set[str]) -> str:
    """Return the subset of *link_defs* whose labels are in *labels*.

    The definitions are returned in reverse-sorted order (matching the
    convention that git-cliff uses: ``[unreleased]`` first, then newest
    version to oldest).
    """
    relevant = [link_defs[label] for label in sorted(link_defs, key=_version_sort_key, reverse=True) if label in labels]
    return "\n".join(relevant) if relevant else ""


def _render_archive(
    minor: str,
    blocks: Sequence[tuple[str, str]],
    link_defs: dict[str, str] | None = None,
) -> str:
    """Render a single minor-series archive without publishing it."""
    parts = [f"# Changelog - {minor}.x\n"]
    for _ver, block in blocks:
        parts.append(block)

    text = "\n".join(parts)

    # Append only the reference-style link definitions for this archive.
    if link_defs:
        versions = {ver.lower() for ver, _ in blocks}
        defs_text = _format_link_defs(link_defs, versions)
        if defs_text:
            text = text.rstrip("\n") + "\n\n" + defs_text

    # Normalize archive output too; archived blocks can preserve historical
    # commit-body indentation that no longer appears in the trimmed root file.
    return postprocess_text(text)


def write_archive(
    archive_dir: Path,
    minor: str,
    blocks: list[tuple[str, str]],
    link_defs: dict[str, str] | None = None,
) -> Path:
    """Write an archive file for a single minor series.

    Parameters:
        archive_dir: Directory for archive files.
        minor: The ``X.Y`` minor key.
        blocks: Version blocks belonging to this minor, newest first, using
            the ``(semver_label, full_heading_block)`` shape returned by
            ``parse_changelog``. The archive writer preserves each provided
            block verbatim after the generated archive title.
        link_defs: Optional mapping of lowercase labels to reference-style
            link definition lines.  Only definitions matching versions in
            *blocks* are included.

    Returns:
        The path of the written archive file.
    """
    archive_dir.mkdir(parents=True, exist_ok=True)
    path = archive_dir / f"{minor}.md"

    _write_text_atomic(path, _render_archive(minor, blocks, link_defs))
    return path


def _existing_archive_updates(archive_dir: Path, excluded: set[Path] | None = None) -> list[tuple[Path, str]]:
    """Return required postprocessing updates for existing archives."""
    if not archive_dir.is_dir():
        return []

    excluded = excluded or set()
    updates: list[tuple[Path, str]] = []
    for path in sorted(archive_dir.glob("*.md")):
        if path in excluded:
            continue
        text = path.read_text(encoding="utf-8")
        processed = normalize_entry_headings_text(text)
        if processed != text:
            updates.append((path, processed))
    return updates


def _postprocess_existing_archives(archive_dir: Path) -> None:
    """Apply changelog postprocessing to existing archives transactionally."""
    _write_texts_transactionally(_existing_archive_updates(archive_dir))


def build_root(
    preamble: str,
    unreleased: str | None,
    active_blocks: list[tuple[str, str]],
    archived_minors: list[str],
    archive_dir_rel: str,
) -> str:
    """Assemble the trimmed root CHANGELOG.md content.

    Parameters:
        preamble: Text before the first ``## `` heading.
        unreleased: The full Unreleased block, or ``None`` if absent.
        active_blocks: Version blocks for the active minor series.
        archived_minors: Sorted list of archived ``X.Y`` minor keys.
        archive_dir_rel: Relative path to the archive directory from the changelog file.

    Returns:
        The full text for the trimmed CHANGELOG.md.
    """
    parts: list[str] = [preamble]

    if unreleased:
        parts.append(unreleased)

    for _ver, block in active_blocks:
        parts.append(block)

    if archived_minors:
        # Build the Archives section.
        archive_lines = ["## Archives\n"]
        archive_lines.append("Older releases are archived by minor series:\n")
        archive_lines.extend(f"- [{minor}.x]({archive_dir_rel}/{minor}.md)" for minor in archived_minors)
        archive_lines.append("")
        parts.append("\n".join(archive_lines))

    return postprocess_text("\n".join(parts))


def _archive_dir_link_prefix(archive_dir: Path, changelog_parent: Path) -> str:
    """Return the Markdown link prefix from a changelog to its archive directory."""
    try:
        return archive_dir.relative_to(changelog_parent).as_posix()
    except ValueError:
        try:
            archive_dir_rel = Path(os.path.relpath(archive_dir, changelog_parent)).as_posix()
        except ValueError as err:
            archive_dir_rel = archive_dir.as_posix()
            LOGGER.warning(
                "Could not compute relative archive directory: %s; archive_dir=%s changelog_parent=%s; generated Markdown links use %s",
                err,
                archive_dir,
                changelog_parent,
                archive_dir_rel,
            )
        if archive_dir_rel == ".." or archive_dir_rel.startswith("../") or Path(archive_dir_rel).is_absolute():
            LOGGER.warning(
                "Archive directory %s is outside changelog directory %s; generated Markdown links use %s",
                archive_dir,
                changelog_parent,
                archive_dir_rel,
            )
        return archive_dir_rel


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def archive_changelog(
    changelog_path: Path,
    archive_dir: Path | None = None,
) -> None:
    """Split a changelog into root + per-minor archive files.

    Parameters:
        changelog_path: Path to the full CHANGELOG.md.
        archive_dir: Directory for archive files.  Defaults to
            ``docs/archive/changelog`` relative to *changelog_path*'s parent.
    """
    if archive_dir is None:
        archive_dir = changelog_path.parent / _DEFAULT_ARCHIVE_DIR

    text = changelog_path.read_text(encoding="utf-8")

    # Separate trailing reference-style link definitions before parsing
    # so they can be distributed to the correct output files.
    text, link_defs = _extract_link_defs(text)

    parsed = parse_changelog(text)

    if not parsed.version_blocks:
        _postprocess_existing_archives(archive_dir)
        return  # nothing to archive

    groups = group_by_minor(parsed.version_blocks)
    minor_keys = list(groups.keys())

    # Active minor = first minor that appears (newest release).
    active_minor = minor_keys[0]

    # Render every minor except the active one before publishing any output.
    archived_minors: list[str] = []
    planned_writes: list[tuple[Path, str]] = []
    for minor in minor_keys[1:]:
        archive_path = archive_dir / f"{minor}.md"
        planned_writes.append((archive_path, _render_archive(minor, groups[minor], link_defs)))
        archived_minors.append(minor)

    if not archived_minors:
        _postprocess_existing_archives(archive_dir)
        return  # only one minor series — nothing to archive yet

    archive_dir_rel = _archive_dir_link_prefix(archive_dir, changelog_path.parent)

    root_text = build_root(
        parsed.preamble,
        parsed.unreleased,
        groups[active_minor],
        sorted(archived_minors, key=_version_sort_key, reverse=True),
        archive_dir_rel,
    )

    # Append reference-style link definitions for active versions.
    if link_defs:
        labels: set[str] = {ver.lower() for ver, _ in groups[active_minor]}
        if parsed.unreleased is not None:
            labels.add("unreleased")
        defs_text = _format_link_defs(link_defs, labels)
        if defs_text:
            root_text = root_text.rstrip("\n") + "\n\n" + defs_text + "\n"

    planned_writes.append((changelog_path, root_text))
    planned_targets = {path for path, _text in planned_writes}
    planned_writes.extend(_existing_archive_updates(archive_dir, planned_targets))
    _write_texts_transactionally(planned_writes)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for ``archive-changelog``."""
    parser = argparse.ArgumentParser(
        prog="archive-changelog",
        description="Archive completed minor series from CHANGELOG.md.",
        suggest_on_error=True,
        color=False,
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="CHANGELOG.md",
        help="Path to CHANGELOG.md (default: CHANGELOG.md)",
    )
    parser.add_argument(
        "--archive-dir",
        default=None,
        help=f"Archive output directory (default: {_DEFAULT_ARCHIVE_DIR})",
    )
    args = parser.parse_args()

    changelog = Path(args.path)
    if not changelog.is_file():
        print(f"Error: {changelog} not found", file=sys.stderr)
        sys.exit(1)

    archive_dir = Path(args.archive_dir) if args.archive_dir else None
    try:
        archive_changelog(changelog, archive_dir)
    except BaseExceptionGroup as err:
        rollback_errors, unhandled = err.split((OSError, ValueError))
        if rollback_errors is not None:
            print(f"Error: {changelog}: {err}", file=sys.stderr)
        if unhandled is not None:
            raise unhandled from None
        raise SystemExit(1) from None
    except (OSError, ValueError) as err:
        print(f"Error: {changelog}: {err}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
