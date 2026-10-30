"""Temporary v0.2.4 scope/retention bridge for upstream #171, retired by #676.

The installed package still extracts, scans, verifies and maps every input and
publishes each complete report pair. These process-local hooks only choose the
Rust-block rules, assign fixture ownership and retain a failed generation.
"""

import json
import tomllib
from contextlib import contextmanager
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, cast

import yaml
from research_repo_tools import semgrep, semgrep_scan
from research_repo_tools.files import publish_directory, replace_many

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from research_repo_tools.config import Config
    from research_repo_tools.semgrep import FixtureCase
    from research_repo_tools.semgrep_scan import ScanInput


class _NoReportError(RuntimeError):
    """Roll back the empty candidate used by v0.2.4 for failed native scans."""


def rustdoc_paths(rule: dict[str, object]) -> dict[str, list[str]] | None:
    """Validate explicit Rust-block policy without changing native rule scopes."""
    metadata = rule.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{rule.get('id')}: metadata must be a mapping")  # noqa: TRY004 - shared CLI reports invalid configuration as ValueError.
    value = metadata.get("delaunay-rustdoc-paths")
    if value is None:
        return None
    if not isinstance(value, dict) or not value or set(value) - {"include", "exclude"}:
        raise ValueError(f"{rule.get('id')}: invalid delaunay-rustdoc-paths")
    languages = rule.get("languages")
    if not isinstance(languages, list) or "rust" not in languages:
        raise ValueError(f"{rule.get('id')}: Rust-block policy requires a Rust rule")
    result: dict[str, list[str]] = {}
    for key, patterns in value.items():
        if not isinstance(patterns, list) or not patterns or any(not isinstance(pattern, str) or not pattern for pattern in patterns):
            raise ValueError(f"{rule.get('id')}: {key} requires nonempty path patterns")
        result[key] = patterns
    return result


def documentation_rules(configuration: Path) -> list[dict[str, object]]:
    """Select explicitly opted-in Rust rules with their documentation path scope."""
    selected: list[dict[str, object]] = []
    for rule in semgrep.rules(configuration).values():
        paths = rustdoc_paths(rule)
        if paths is not None:
            selected.append({**rule, "languages": ["rust"], "paths": paths})
    if not selected:
        message = "Rust blocks require explicit metadata.delaunay-rustdoc-paths rules"
        raise ValueError(message)
    return selected


def extraction_names(root: Path, names: tuple[str, ...]) -> tuple[str, ...]:
    """Apply explicit consumer exclusions to added blocks, never native inputs."""
    document = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = document.get("tool", {}).get("delaunay", {}).get("semgrep", {}).get("rustdoc-exclude", [])
    if not isinstance(patterns, list) or any(not isinstance(pattern, str) or not pattern for pattern in patterns):
        message = "tool.delaunay.semgrep.rustdoc-exclude must be a list of nonempty globs"
        raise ValueError(message)
    return tuple(name for name in names if not any(PurePosixPath(name).full_match(pattern) for pattern in patterns))


@contextmanager
def scoped_semgrep() -> Iterator[None]:
    """Install bounded v0.2.4 hooks, restoring every shared entry point on exit."""
    original_groups = semgrep_scan._scan_groups  # noqa: SLF001 - version-guarded bridge until #676.
    original_cases = semgrep._documentation_cases  # noqa: SLF001 - version-guarded bridge until #676.
    original_scan = semgrep_scan.scan

    def groups(root: Path, names: tuple[str, ...], temporary: Path, configuration: Path, rust_docs: bool) -> list[tuple[Path, Path, list[ScanInput], bool]]:
        # The pinned private helper is unannotated; its verified v0.2.4 shape is
        # the only unchecked typing boundary in this bridge.
        result = cast("list[tuple[Path, Path, list[ScanInput], bool]]", original_groups(root, names, temporary, configuration, rust_docs=False))
        if rust_docs:
            result.extend(original_groups(root, extraction_names(root, names), temporary, configuration, rust_docs=True)[1:])
        if rust_docs and len(result) > 1:
            # All shared block groups use this generated configuration. Original
            # files retain their original configuration and complete rule set.
            (temporary / "rust-rules.yaml").write_text(yaml.safe_dump({"rules": documentation_rules(configuration)}, sort_keys=False), encoding="utf-8")
        return result

    def cases(path: Path, source: dict[str, dict[str, object]], namespace: str) -> list[FixtureCase]:
        # The shared fixture planner uses language solely to assign ownership;
        # native execution still loads the unchanged original rule definitions.
        ownership = {name: rule if rustdoc_paths(rule) is not None else {**rule, "languages": ["generic"]} for name, rule in source.items()}
        return original_cases(path, ownership, namespace)

    def scan(  # noqa: PLR0913 - preserve the installed public scan signature exactly.
        settings: Config,
        *,
        include: tuple[str, ...],
        exclude: tuple[str, ...] = (),
        output: str = "target/security/semgrep",
        rust_docs: bool = False,
        batch_size: int | None = None,
        inline_suppressions: bool | None = None,
        jobs: int | None = None,
        report_category: str | None = None,
        report_layout: str | None = None,
        target_timeout: int | None = None,
    ) -> int:
        destination = settings.path(output)
        marker = destination.with_name(destination.name + ".attempt.json")
        replace_many({marker: b'{"published": false}\n'})
        previous_publisher = semgrep_scan.publish_directory
        published = False

        @contextmanager
        def retain_previous(path: Path) -> Iterator[Path]:
            nonlocal published
            try:
                with publish_directory(path) as candidate:
                    yield candidate
                    if not any(candidate.iterdir()):
                        raise _NoReportError
            except _NoReportError:
                return
            published = True

        try:
            vars(semgrep_scan)["publish_directory"] = retain_previous
            code = original_scan(
                settings,
                include=include,
                exclude=exclude,
                output=output,
                rust_docs=rust_docs,
                batch_size=batch_size,
                inline_suppressions=inline_suppressions,
                jobs=jobs,
                report_category=report_category,
                report_layout=report_layout,
                target_timeout=target_timeout,
            )
            replace_many({marker: (json.dumps({"published": published, "exit_code": code}) + "\n").encode()})
            return code
        finally:
            vars(semgrep_scan)["publish_directory"] = previous_publisher

    try:
        vars(semgrep_scan)["_scan_groups"] = groups
        vars(semgrep)["_documentation_cases"] = cases
        vars(semgrep_scan)["scan"] = scan
        yield
    finally:
        vars(semgrep_scan)["_scan_groups"] = original_groups
        vars(semgrep)["_documentation_cases"] = original_cases
        vars(semgrep_scan)["scan"] = original_scan
