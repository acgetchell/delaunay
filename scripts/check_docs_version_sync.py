"""Check release-version references against the Cargo package version."""

from __future__ import annotations

import argparse
import os
import re
import sys
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, TypeGuard

if TYPE_CHECKING:
    from collections.abc import Sequence

SKIP_DIRS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tmp_pycache",
        ".venv",
        "archive",
        "target",
        "tests",
    }
)
SKIP_MARKDOWN_FILES = frozenset({"CHANGELOG.md"})

type ParsedObject = dict[str, object]


def _is_parsed_object(value: object) -> TypeGuard[ParsedObject]:
    """Return true when a parsed TOML value is an object with string keys."""
    return isinstance(value, dict) and all(isinstance(key, str) for key in value)


def _require_parsed_object(value: object, context: str) -> ParsedObject:
    """Return *value* as a TOML object or raise with context."""
    if not _is_parsed_object(value):
        msg = f"{context} is not a TOML object"
        raise TypeError(msg)
    return value


def _read_toml(path: Path) -> ParsedObject:
    """Parse *path* as TOML and return its root table."""
    data: object = tomllib.loads(path.read_text(encoding="utf-8"))
    return _require_parsed_object(data, str(path))


def _require_table(data: ParsedObject, key: str, path: Path) -> ParsedObject:
    """Return a required child TOML table."""
    table = data.get(key)
    if not _is_parsed_object(table):
        msg = f"{path} is missing a [{key}] table"
        raise TypeError(msg)
    return table


def _require_string(data: ParsedObject, key: str, context: str) -> str:
    """Return a required string field."""
    value = data.get(key)
    if not isinstance(value, str):
        msg = f"{context} is missing a string {key}"
        raise TypeError(msg)
    return value


@dataclass(frozen=True, slots=True)
class PackageInfo:
    """Cargo package identity that defines the expected release version."""

    name: str
    version: str


@dataclass(frozen=True, slots=True)
class PythonProjectInfo:
    """Python support-package identity used to locate its uv lock entry."""

    name: str
    version: str


class ReferenceKind(StrEnum):
    """A release surface whose version must match Cargo.toml."""

    BENCHMARK_CURRENT_TAG = "benchmark workflow current tag"
    CARGO_LOCK = "Cargo.lock root package"
    CITATION = "CITATION.cff version"
    DEPENDENCY_SNIPPET = "documentation dependency snippet"
    PYPROJECT = "pyproject.toml project"
    README_TAG_LINK = "README tag-pinned link"
    UV_LOCK = "uv.lock editable package"


@dataclass(frozen=True, slots=True)
class VersionReference:
    """A parsed release-version reference with source location."""

    path: Path
    line: int
    version: str
    kind: ReferenceKind
    text: str


@dataclass(frozen=True, slots=True)
class VersionMismatch:
    """A release-version reference that does not match Cargo.toml."""

    reference: VersionReference
    package: PackageInfo


def _read_cargo_package_info(cargo_toml: Path) -> PackageInfo:
    """Read the Cargo package name and version."""
    package = _require_table(_read_toml(cargo_toml), "package", cargo_toml)
    return PackageInfo(
        name=_require_string(package, "name", f"{cargo_toml} [package]"),
        version=_require_string(package, "version", f"{cargo_toml} [package]"),
    )


def _read_python_project_info(pyproject_toml: Path) -> PythonProjectInfo:
    """Read the Python support package name and version."""
    project = _require_table(_read_toml(pyproject_toml), "project", pyproject_toml)
    return PythonProjectInfo(
        name=_require_string(project, "name", f"{pyproject_toml} [project]"),
        version=_require_string(project, "version", f"{pyproject_toml} [project]"),
    )


def _toml_table_key_line(path: Path, table_name: str, key: str) -> int:
    """Return the line number for *key* in a TOML table."""
    current_table: str | None = None
    key_re = re.compile(rf"^{re.escape(key)}\s*=")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current_table = stripped.strip("[]")
        elif current_table == table_name and key_re.match(stripped):
            return line_number
    msg = f"{path} [{table_name}] is missing {key}"
    raise TypeError(msg)


def _version_reference(path: Path, line: int, version: str, kind: ReferenceKind) -> VersionReference:
    """Build a version reference and include the source line text."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not 1 <= line <= len(lines):
        msg = f"{path} has no line {line} for {kind}"
        raise TypeError(msg)
    return VersionReference(path=path, line=line, version=version, kind=kind, text=lines[line - 1].strip())


def _package_entries(path: Path) -> list[ParsedObject]:
    """Return TOML ``[[package]]`` entries from a lockfile."""
    packages = _read_toml(path).get("package")
    if not isinstance(packages, list):
        msg = f"{path} is missing [[package]] entries"
        raise TypeError(msg)
    entries: list[ParsedObject] = []
    for index, package in enumerate(packages, start=1):
        entries.append(_require_parsed_object(package, f"{path} [[package]] entry {index}"))
    return entries


def _array_table_key_line(path: Path, table_name: str, table_index: int, key: str) -> int:
    """Return the line for *key* inside the requested array-table entry."""
    current_index = -1
    in_target_table = False
    key_re = re.compile(rf"^{re.escape(key)}\s*=")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if stripped == f"[[{table_name}]]":
            current_index += 1
            in_target_table = current_index == table_index
        elif stripped.startswith("[["):
            in_target_table = False
        elif in_target_table and key_re.match(stripped):
            return line_number
    msg = f"{path} [[{table_name}]] entry {table_index + 1} is missing {key}"
    raise TypeError(msg)


def _single_package_reference(
    path: Path,
    entries: list[ParsedObject],
    candidate_indices: list[int],
    package_name: str,
    kind: ReferenceKind,
) -> VersionReference:
    """Return the only matching package reference or raise on ambiguity."""
    if len(candidate_indices) != 1:
        msg = f"{path} must contain exactly one {kind} named {package_name!r}; found {len(candidate_indices)}"
        raise TypeError(msg)
    index = candidate_indices[0]
    version = _require_string(entries[index], "version", f"{path} [[package]] entry {index + 1}")
    line = _array_table_key_line(path, "package", index, "version")
    return _version_reference(path, line, version, kind)


def _cargo_lock_reference(path: Path, package: PackageInfo) -> VersionReference:
    """Return the root package reference from Cargo.lock."""
    entries = _package_entries(path)
    candidate_indices = [index for index, entry in enumerate(entries) if entry.get("name") == package.name and "source" not in entry]
    return _single_package_reference(path, entries, candidate_indices, package.name, ReferenceKind.CARGO_LOCK)


def _pyproject_reference(path: Path, project: PythonProjectInfo) -> VersionReference:
    """Return the Python project version reference."""
    line = _toml_table_key_line(path, "project", "version")
    return _version_reference(path, line, project.version, ReferenceKind.PYPROJECT)


def _uv_lock_reference(path: Path, project: PythonProjectInfo) -> VersionReference:
    """Return the editable Python project reference from uv.lock."""
    entries = _package_entries(path)
    candidate_indices: list[int] = []
    for index, entry in enumerate(entries):
        source = entry.get("source")
        if entry.get("name") == project.name and _is_parsed_object(source) and isinstance(source.get("editable"), str):
            candidate_indices.append(index)
    return _single_package_reference(path, entries, candidate_indices, project.name, ReferenceKind.UV_LOCK)


_CITATION_VERSION_RE = re.compile(r"^version:\s*(?P<quote>['\"]?)(?P<version>[0-9A-Za-z][0-9A-Za-z.+-]*)(?P=quote)\s*(?:#.*)?$")


def _citation_reference(path: Path) -> VersionReference:
    """Return the top-level CITATION.cff version reference."""
    references: list[VersionReference] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.startswith("version:"):
            continue
        match = _CITATION_VERSION_RE.fullmatch(line)
        if match is None:
            msg = f"{path}:{line_number}: top-level version must be a non-empty scalar"
            raise TypeError(msg)
        references.append(_version_reference(path, line_number, match.group("version"), ReferenceKind.CITATION))
    if len(references) != 1:
        msg = f"{path} must contain exactly one top-level version; found {len(references)}"
        raise TypeError(msg)
    return references[0]


def _iter_markdown_files(root: Path) -> list[Path]:
    """Return active Markdown files that can carry current release references."""
    markdown_files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [dirname for dirname in dirnames if not (set((Path(dirpath) / dirname).relative_to(root).parts) & SKIP_DIRS)]
        markdown_files.extend(Path(dirpath) / filename for filename in filenames if filename.endswith(".md") and filename not in SKIP_MARKDOWN_FILES)
    return sorted(markdown_files)


def _dependency_regex(package_name: str) -> re.Pattern[str]:
    """Build a regex for Cargo dependency snippets naming *package_name*."""
    escaped_name = re.escape(package_name)
    return re.compile(rf'(?<![\w.-]){escaped_name}\s*=\s*(?:"(?P<plain>[^"]+)"|\{{[^}}]*version\s*=\s*"(?P<table>[^"]+)"[^}}]*\}})')


def _dependency_references(path: Path, package_name: str) -> list[VersionReference]:
    """Return dependency snippet references in a Markdown file."""
    dependency_re = _dependency_regex(package_name)
    references: list[VersionReference] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        for match in dependency_re.finditer(line):
            version = match.group("plain") or match.group("table")
            references.append(
                VersionReference(
                    path=path,
                    line=line_number,
                    version=version,
                    kind=ReferenceKind.DEPENDENCY_SNIPPET,
                    text=line.strip(),
                )
            )
    return references


_README_TAG_LINK_RE = re.compile(
    r"https://(?:github\.com/acgetchell/delaunay/(?:blob|raw|tree)/|raw\.githubusercontent\.com/acgetchell/delaunay/)"
    r"(?:v(?P<version>[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?)"
    r"|(?P<revision>[0-9a-f]{7,40}))(?=/|$|[^0-9A-Za-z._+-])"
)


def _readme_tag_references(path: Path) -> list[VersionReference]:
    """Return release-pinned README links that should track the package version."""
    references: list[VersionReference] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        references.extend(
            VersionReference(
                path,
                line_number,
                match.group("version") or match.group("revision"),
                ReferenceKind.README_TAG_LINK,
                line.strip(),
            )
            for match in _README_TAG_LINK_RE.finditer(line)
        )
    return references


_BENCHMARK_CURRENT_TAG_RE = re.compile(
    r"just perf-(?:github-assets|local|release)\s+v"
    r"(?P<version>[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?)(?=\s|`)"
)


def _benchmark_current_tag_references(path: Path) -> list[VersionReference]:
    """Return current-release tag arguments in benchmark workflow examples."""
    references: list[VersionReference] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        references.extend(
            VersionReference(path, line_number, match.group("version"), ReferenceKind.BENCHMARK_CURRENT_TAG, line.strip())
            for match in _BENCHMARK_CURRENT_TAG_RE.finditer(line)
        )
    return references


def _version_references(root: Path, package: PackageInfo) -> list[VersionReference]:
    """Collect all current-release references that should match Cargo.toml."""
    pyproject_path = root / "pyproject.toml"
    project = _read_python_project_info(pyproject_path)
    references = [
        _cargo_lock_reference(root / "Cargo.lock", package),
        _pyproject_reference(pyproject_path, project),
        _uv_lock_reference(root / "uv.lock", project),
        _citation_reference(root / "CITATION.cff"),
    ]
    for path in _iter_markdown_files(root):
        references.extend(_dependency_references(path, package.name))
        references.extend(_benchmark_current_tag_references(path))
    references.extend(_readme_tag_references(root / "README.md"))
    return references


def find_version_mismatches(root: Path) -> list[VersionMismatch]:
    """Return release-version references that differ from Cargo.toml."""
    package = _read_cargo_package_info(root / "Cargo.toml")
    return [VersionMismatch(reference=reference, package=package) for reference in _version_references(root, package) if reference.version != package.version]


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="check-docs-version-sync",
        description="Check release-version references against Cargo.toml.",
        suggest_on_error=True,
        color=False,
    )
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd(), help="Repository root to check (default: current directory)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Check release-version references against the Cargo package version."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    root = args.root.resolve()
    try:
        mismatches = find_version_mismatches(root)
    except (OSError, TypeError, tomllib.TOMLDecodeError) as error:
        print(f"Could not check release-version synchronization: {error}", file=sys.stderr)
        return 1

    if not mismatches:
        return 0

    print("Release-version references are out of sync with Cargo.toml:", file=sys.stderr)
    for mismatch in mismatches:
        reference = mismatch.reference
        rel_path = reference.path.relative_to(root)
        print(
            f"  {rel_path}:{reference.line}: {reference.kind} found {reference.version}, expected {mismatch.package.version}: {reference.text}",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
