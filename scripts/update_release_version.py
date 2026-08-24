"""Update deterministic release-version references from one target Git tag."""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import check_docs_version_sync as version_sync
from benchmark_utils import published_stable_release_tags
from subprocess_utils import ExecutableNotFoundError

if TYPE_CHECKING:
    from collections.abc import Callable

_STABLE_TAG_RE = re.compile(r"^v(?P<major>0|[1-9][0-9]*)\.(?P<minor>0|[1-9][0-9]*)\.(?P<patch>0|[1-9][0-9]*)$")
_TOML_VERSION_RE = re.compile(r'^(?P<prefix>\s*version\s*=\s*")(?P<version>[^"]+)(?P<suffix>"\s*(?:#.*)?)$')
_CITATION_VERSION_RE = re.compile(
    r"^(?P<prefix>version:\s*(?P<quote>['\"]?))"
    r"(?P<version>[0-9A-Za-z][0-9A-Za-z.+-]*)"
    r"(?P<suffix>(?P=quote)\s*(?:#.*)?)$",
)
_CITATION_DATE_RE = re.compile(
    r"^(?P<prefix>date-released:\s*(?P<quote>['\"]?))"
    r"(?P<date>\d{4}-\d{2}-\d{2})"
    r"(?P<suffix>(?P=quote)\s*(?:#.*)?)$",
)


@dataclass(frozen=True, order=True, slots=True)
class ReleaseTag:
    """A stable release tag with SemVer ordering."""

    major: int
    minor: int
    patch: int
    tag: str = field(compare=False)

    def __post_init__(self) -> None:
        """Reject directly constructed values whose identity is contradictory."""
        match = _STABLE_TAG_RE.fullmatch(self.tag)
        if match is None:
            msg = f"release tag must be a stable tag in vX.Y.Z form, got {self.tag!r}"
            raise ValueError(msg)
        parsed = (int(match.group("major")), int(match.group("minor")), int(match.group("patch")))
        supplied = (self.major, self.minor, self.patch)
        if supplied != parsed:
            msg = f"release tag components {supplied} contradict emitted tag {self.tag!r} with components {parsed}"
            raise ValueError(msg)

    @property
    def version(self) -> str:
        """Return the package version without the leading ``v``."""
        return self.tag.removeprefix("v")


@dataclass(frozen=True, slots=True)
class UpdateSummary:
    """Files and release identities produced by an update."""

    target: ReleaseTag
    previous: ReleaseTag
    changed_paths: tuple[Path, ...]
    release_date: str


@dataclass(frozen=True, slots=True)
class LineReplacement:
    """One fail-closed scalar replacement on a known source line."""

    line_number: int
    pattern: re.Pattern[str]
    group: str
    replacement: str
    allowed: frozenset[str]
    context: str


def parse_release_tag(value: str, *, label: str = "release tag") -> ReleaseTag:
    """Parse one stable ``vX.Y.Z`` release tag."""
    match = _STABLE_TAG_RE.fullmatch(value)
    if match is None:
        msg = f"{label} must be a stable tag in vX.Y.Z form, got {value!r}"
        raise ValueError(msg)
    return ReleaseTag(
        major=int(match.group("major")),
        minor=int(match.group("minor")),
        patch=int(match.group("patch")),
        tag=value,
    )


def select_previous_release_tag(tag_names: list[str], target: ReleaseTag) -> ReleaseTag:
    """Select the newest published stable release before *target*."""
    stable_tags = [parse_release_tag(tag) for tag in tag_names if _STABLE_TAG_RE.fullmatch(tag) is not None]
    if not stable_tags:
        msg = "repository has no published stable vX.Y.Z GitHub releases"
        raise ValueError(msg)
    newer = [tag for tag in stable_tags if tag > target]
    if newer:
        latest = max(newer)
        msg = f"target {target.tag} is older than published stable GitHub release {latest.tag}"
        raise ValueError(msg)
    previous = [tag for tag in stable_tags if tag < target]
    if not previous:
        msg = f"could not find a published stable GitHub release before {target.tag}"
        raise ValueError(msg)
    return max(previous)


def infer_previous_release_tag(root: Path, target: ReleaseTag) -> ReleaseTag:
    """Infer the previous release from published stable GitHub releases."""
    return select_previous_release_tag(published_stable_release_tags(root), target)


def _validated_date(value: str) -> str:
    """Require one real ISO calendar date and return it unchanged."""
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        msg = f"release date must use YYYY-MM-DD, got {value!r}"
        raise ValueError(msg) from error
    if parsed.isoformat() != value:
        msg = f"release date must use canonical YYYY-MM-DD form, got {value!r}"
        raise ValueError(msg)
    return value


def _current_utc_date() -> str:
    """Return today's canonical UTC calendar date."""
    return datetime.now(UTC).date().isoformat()


def _replace_line_group(text: str, edit: LineReplacement) -> str:
    lines = text.splitlines(keepends=True)
    if not 1 <= edit.line_number <= len(lines):
        msg = f"{edit.context} has no line {edit.line_number}"
        raise ValueError(msg)
    original_line = lines[edit.line_number - 1]
    body = original_line.rstrip("\r\n")
    ending = original_line[len(body) :]
    match = edit.pattern.fullmatch(body)
    if match is None:
        msg = f"{edit.context}:{edit.line_number} has an unsupported version assignment: {body!r}"
        raise ValueError(msg)
    current = match.group(edit.group)
    if current not in edit.allowed:
        msg = f"{edit.context}:{edit.line_number} has unexpected version {current!r}; expected one of {sorted(edit.allowed)}"
        raise ValueError(msg)
    start, end = match.span(edit.group)
    lines[edit.line_number - 1] = f"{body[:start]}{edit.replacement}{body[end:]}{ending}"
    return "".join(lines)


def _replace_match_groups(match: re.Match[str], replacements: dict[str, str]) -> str:
    updated = match.group(0)
    spans = sorted(
        ((match.start(group) - match.start(), match.end(group) - match.start(), value) for group, value in replacements.items()),
        reverse=True,
    )
    for start, end, value in spans:
        updated = f"{updated[:start]}{value}{updated[end:]}"
    return updated


def _replace_dependency_versions(text: str, package_name: str, target: ReleaseTag, previous: ReleaseTag, path: Path) -> str:
    allowed = frozenset({target.version, previous.version})
    pattern = version_sync.dependency_regex(package_name)

    def replace(match: re.Match[str]) -> str:
        group = next(name for name in ("plain", "plain_literal", "table", "table_literal") if match.group(name) is not None)
        current = match.group(group)
        if current not in allowed:
            msg = f"{path} has unexpected {package_name} dependency version {current!r}; expected one of {sorted(allowed)}"
            raise ValueError(msg)
        return _replace_match_groups(match, {group: target.version})

    return pattern.sub(replace, text)


def _replace_cargo_add_versions(text: str, package_name: str, target: ReleaseTag, previous: ReleaseTag, path: Path) -> str:
    allowed = frozenset({target.version, previous.version})
    pattern = version_sync.cargo_add_regex(package_name)

    def replace(match: re.Match[str]) -> str:
        current = match.group("version")
        if current not in allowed:
            msg = f"{path} has unexpected cargo add version {current!r}; expected one of {sorted(allowed)}"
            raise ValueError(msg)
        return _replace_match_groups(match, {"version": target.version})

    return pattern.sub(replace, text)


def _replace_readme_links(text: str, target: ReleaseTag, previous: ReleaseTag, path: Path) -> str:
    allowed = frozenset({target.version, previous.version})

    def replace(match: re.Match[str]) -> str:
        if version_sync.readme_tag_link_is_performance_asset(match):
            return match.group(0)
        version = match.group("version")
        if version is not None and version not in allowed:
            msg = f"{path} has unexpected release-pinned link version {version!r}; expected one of {sorted(allowed)}"
            raise ValueError(msg)
        group = "version" if version is not None else "revision"
        replacement = target.version if version is not None else target.tag
        return _replace_match_groups(match, {group: replacement})

    return version_sync.README_TAG_LINK_RE.sub(replace, text)


def _replace_benchmark_tag_pairs(text: str, target: ReleaseTag, previous: ReleaseTag, path: Path) -> str:
    allowed_current = frozenset({target.tag, previous.tag})

    for line_number, line in enumerate(text.splitlines(), start=1):
        incomplete = version_sync.BENCHMARK_SINGLE_TAG_RE.search(line)
        if incomplete is not None:
            msg = f"{path}:{line_number} has a benchmark command missing the baseline tag after {incomplete.group('current')!r}"
            raise ValueError(msg)

    def replace(match: re.Match[str]) -> str:
        current = parse_release_tag(match.group("current"), label=f"{path} benchmark current tag")
        baseline = parse_release_tag(match.group("baseline"), label=f"{path} benchmark baseline tag")
        if baseline >= current:
            msg = f"{path} has benchmark baseline {baseline.tag} that is not older than current tag {current.tag}"
            raise ValueError(msg)
        if current.tag not in allowed_current:
            msg = f"{path} has unexpected benchmark current tag {current.tag}; expected {target.tag} or {previous.tag}"
            raise ValueError(msg)
        if current == target and baseline != previous:
            msg = f"{path} has non-adjacent benchmark pair {current.tag} against {baseline.tag}; expected baseline {previous.tag}"
            raise ValueError(msg)
        return _replace_match_groups(match, {"current": target.tag, "baseline": previous.tag})

    return version_sync.BENCHMARK_TAG_PAIR_RE.sub(replace, text)


def _read_text(path: Path) -> str:
    with path.open(encoding="utf-8", newline="") as stream:
        return stream.read()


def _metadata_updates(root: Path, target: ReleaseTag, previous: ReleaseTag, release_date: str) -> dict[Path, str]:
    allowed = frozenset({target.version, previous.version})
    cargo_toml = root / "Cargo.toml"
    cargo_lock = root / "Cargo.lock"
    pyproject = root / "pyproject.toml"
    uv_lock = root / "uv.lock"
    citation = root / "CITATION.cff"

    package = version_sync.read_cargo_package_info(cargo_toml)
    project = version_sync.read_python_project_info(pyproject)
    cargo_toml_line = version_sync.toml_table_key_line(cargo_toml, "package", "version")
    cargo_lock_reference = version_sync.cargo_lock_reference(cargo_lock, package)
    pyproject_reference = version_sync.pyproject_reference(pyproject, project)
    uv_lock_reference = version_sync.uv_lock_reference(uv_lock, project)
    citation_reference = version_sync.citation_reference(citation)

    updates = {
        cargo_toml: _replace_line_group(
            _read_text(cargo_toml),
            LineReplacement(cargo_toml_line, _TOML_VERSION_RE, "version", target.version, allowed, str(cargo_toml)),
        ),
        cargo_lock: _replace_line_group(
            _read_text(cargo_lock),
            LineReplacement(cargo_lock_reference.line, _TOML_VERSION_RE, "version", target.version, allowed, str(cargo_lock)),
        ),
        pyproject: _replace_line_group(
            _read_text(pyproject),
            LineReplacement(pyproject_reference.line, _TOML_VERSION_RE, "version", target.version, allowed, str(pyproject)),
        ),
        uv_lock: _replace_line_group(
            _read_text(uv_lock),
            LineReplacement(uv_lock_reference.line, _TOML_VERSION_RE, "version", target.version, allowed, str(uv_lock)),
        ),
        citation: _replace_line_group(
            _read_text(citation),
            LineReplacement(citation_reference.line, _CITATION_VERSION_RE, "version", target.version, allowed, str(citation)),
        ),
    }
    citation_line, current_date = version_sync.citation_release_date(citation)
    updates[citation] = _replace_line_group(
        updates[citation],
        LineReplacement(
            citation_line,
            _CITATION_DATE_RE,
            "date",
            release_date,
            frozenset({current_date, release_date}),
            str(citation),
        ),
    )
    return updates


def _prepare_updates(root: Path, target: ReleaseTag, previous: ReleaseTag, release_date: str) -> dict[Path, str]:
    updates = _metadata_updates(root, target, previous, release_date)
    changelog = root / "CHANGELOG.md"
    changelog_match = version_sync.changelog_release_date(changelog, target.version)
    if changelog_match is not None:
        changelog_line, changelog_date = changelog_match
        updates[changelog] = _replace_changelog_release_date(
            changelog,
            target,
            line=changelog_line,
            current_date=changelog_date,
            release_date=release_date,
        )
    package = version_sync.read_cargo_package_info(root / "Cargo.toml")
    for path in version_sync.iter_active_markdown_files(root):
        original = _read_text(path)
        updated = _replace_dependency_versions(original, package.name, target, previous, path)
        updated = _replace_cargo_add_versions(updated, package.name, target, previous, path)
        updated = _replace_benchmark_tag_pairs(updated, target, previous, path)
        if path == root / "README.md":
            updated = _replace_readme_links(updated, target, previous, path)
        updates[path] = updated
    return updates


def _replace_changelog_release_date(
    changelog: Path,
    target: ReleaseTag,
    *,
    line: int,
    current_date: str,
    release_date: str,
) -> str:
    """Return a changelog with one target release heading date synchronized."""
    heading_re = re.compile(rf"^(?P<prefix>## \[v?{re.escape(target.version)}\] - )(?P<date>\d{{4}}-\d{{2}}-\d{{2}})$")
    return _replace_line_group(
        _read_text(changelog),
        LineReplacement(
            line,
            heading_re,
            "date",
            release_date,
            frozenset({current_date, release_date}),
            str(changelog),
        ),
    )


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    """Replace ``path`` atomically with exact bytes while preserving its mode."""
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
        temporary.chmod(path.stat().st_mode)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_text_atomic(path: Path, text: str) -> None:
    _write_bytes_atomic(path, text.encode("utf-8"))


def _validate_updated_root(root: Path, target: ReleaseTag, previous: ReleaseTag) -> None:
    mismatches = version_sync.find_version_mismatches(root)
    if mismatches:
        details = "; ".join(
            f"{mismatch.reference.path.relative_to(root)}:{mismatch.reference.line} has {mismatch.reference.version}" for mismatch in mismatches
        )
        msg = f"release-version validation failed after updating to {target.tag}: {details}"
        raise ValueError(msg)
    for path in version_sync.iter_active_markdown_files(root):
        for match in version_sync.BENCHMARK_TAG_PAIR_RE.finditer(_read_text(path)):
            if match.group("current") != target.tag or match.group("baseline") != previous.tag:
                msg = f"{path} contains a benchmark tag pair that does not match {target.tag} against {previous.tag}"
                raise ValueError(msg)


def _validate_planned_updates(
    root: Path,
    updates: dict[Path, str],
    target: ReleaseTag,
    previous: ReleaseTag,
) -> None:
    """Validate the complete candidate tree before replacing repository files."""
    sources = {
        root / "Cargo.toml",
        root / "Cargo.lock",
        root / "pyproject.toml",
        root / "uv.lock",
        root / "CITATION.cff",
        *version_sync.iter_active_markdown_files(root),
    }
    changelog = root / "CHANGELOG.md"
    if changelog.is_file():
        sources.add(changelog)

    with tempfile.TemporaryDirectory(prefix="delaunay-release-plan-") as temporary_directory:
        candidate_root = Path(temporary_directory)
        for source in sources:
            destination = candidate_root / source.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(updates.get(source, _read_text(source)), encoding="utf-8", newline="")
        _validate_updated_root(candidate_root, target, previous)


def _publish_transaction(updates: dict[Path, str], validate: Callable[[], None]) -> tuple[Path, ...]:
    original_bytes = {path: path.read_bytes() for path in updates}
    original_text = {path: _read_text(path) for path in updates}
    changed = tuple(sorted((path for path, text in updates.items() if text != original_text[path]), key=str))
    replaced: list[Path] = []
    try:
        for path in changed:
            _write_text_atomic(path, updates[path])
            replaced.append(path)
        validate()
    except BaseException as primary:
        rollback_errors: list[str] = []
        for path in reversed(replaced):
            try:
                _write_bytes_atomic(path, original_bytes[path])
            except OSError as error:
                rollback_errors.append(f"{path}: {error}")
        if rollback_errors:
            msg = f"release-version update failed ({primary}); rollback also failed: {'; '.join(rollback_errors)}"
            raise RuntimeError(msg) from primary
        raise
    return changed


def update_release_version(
    root: Path,
    tag: str,
    *,
    previous: ReleaseTag | None = None,
) -> UpdateSummary:
    """Update release references transactionally and return a summary."""
    resolved_root = root.resolve()
    target = parse_release_tag(tag, label="target tag")
    previous_release = previous or infer_previous_release_tag(resolved_root, target)
    if previous_release >= target:
        msg = f"previous release {previous_release.tag} must be older than target {target.tag}"
        raise ValueError(msg)
    prepared_date = _validated_date(_current_utc_date())
    updates = _prepare_updates(resolved_root, target, previous_release, prepared_date)
    _validate_planned_updates(resolved_root, updates, target, previous_release)
    changed = _publish_transaction(updates, lambda: _validate_updated_root(resolved_root, target, previous_release))
    return UpdateSummary(target=target, previous=previous_release, changed_paths=changed, release_date=prepared_date)


def sync_changelog_release_date(
    root: Path,
    tag: str,
    *,
    previous: ReleaseTag | None = None,
) -> tuple[tuple[Path, ...], str]:
    """Synchronize a generated changelog heading from ``CITATION.cff``."""
    resolved_root = root.resolve()
    target = parse_release_tag(tag, label="target tag")
    previous_release = previous or infer_previous_release_tag(resolved_root, target)
    package = version_sync.read_cargo_package_info(resolved_root / "Cargo.toml")
    if package.version != target.version:
        msg = f"Cargo.toml version {package.version} does not match target {target.tag}"
        raise ValueError(msg)
    citation = resolved_root / "CITATION.cff"
    _, citation_date = version_sync.citation_release_date(citation)
    changelog = resolved_root / "CHANGELOG.md"
    changelog_match = version_sync.changelog_release_date(changelog, target.version)
    if changelog_match is None:
        msg = f"{changelog} has no generated release heading for {target.tag}"
        raise ValueError(msg)
    changelog_line, changelog_date = changelog_match
    updated = _replace_changelog_release_date(
        changelog,
        target,
        line=changelog_line,
        current_date=changelog_date,
        release_date=citation_date,
    )
    updates = {changelog: updated}
    _validate_planned_updates(resolved_root, updates, target, previous_release)
    changed = _publish_transaction(
        updates,
        lambda: _validate_updated_root(resolved_root, target, previous_release),
    )
    return changed, citation_date


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="Target stable release tag in vX.Y.Z form")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root to update (default: current directory)")
    parser.add_argument("--previous-release", help="previous stable published tag already resolved by a non-mutating preflight")
    parser.add_argument(
        "--print-previous-release",
        action="store_true",
        help="print the inferred previous stable published tag without changing files",
    )
    parser.add_argument(
        "--sync-changelog-date",
        action="store_true",
        help="Synchronize the generated changelog heading from CITATION.cff instead of updating release metadata",
    )
    args = parser.parse_args(argv)
    if args.print_previous_release and (args.sync_changelog_date or args.previous_release is not None):
        parser.error("--print-previous-release cannot be combined with update or synchronization options")
    return args


def main(argv: list[str] | None = None) -> int:
    """Update deterministic release metadata with fail-closed diagnostics."""
    args = parse_args(argv)
    try:
        target = parse_release_tag(args.tag, label="target tag")
        if args.print_previous_release:
            print(infer_previous_release_tag(args.root.resolve(), target).tag)
            return 0
        previous = parse_release_tag(args.previous_release, label="previous release") if args.previous_release is not None else None
        if args.sync_changelog_date:
            changed_paths, release_date = sync_changelog_release_date(args.root, args.tag, previous=previous)
            if changed_paths:
                print(f"Synchronized CHANGELOG.md release date to {release_date}.")
            else:
                print(f"CHANGELOG.md release date already matches {release_date}.")
            return 0
        summary = update_release_version(args.root, args.tag, previous=previous)
    except (ExecutableNotFoundError, OSError, RuntimeError, subprocess.SubprocessError, TypeError, ValueError, tomllib.TOMLDecodeError) as error:
        print(f"failed to update release version: {error}", file=sys.stderr)
        return 1

    if summary.changed_paths:
        for path in summary.changed_paths:
            print(f"Updated {path.relative_to(args.root.resolve())}")
    else:
        print(f"Release-version references already match {summary.target.tag}.")
    print(f"Previous release: {summary.previous.tag}")
    print(f"CITATION.cff UTC release date: {summary.release_date}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
