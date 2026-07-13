"""Tests for documentation and package-version synchronization checks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import check_docs_version_sync

if TYPE_CHECKING:
    from pathlib import Path


_CARGO_TOML = '[package]\nname = "delaunay"\nversion = "1.2.3"'
_VERSION = "1.2.3"


def _write_project(
    root: Path,
    *,
    metadata_version: str = _VERSION,
    readme: str | None = None,
) -> None:
    """Write a minimal project tree for version-sync tests."""
    readme_text = (
        readme
        if readme is not None
        else (
            f'delaunay = "{_VERSION}"\n'
            f"[doc](https://github.com/acgetchell/delaunay/blob/v{_VERSION}/README.md)\n"
            f"[raw](https://raw.githubusercontent.com/acgetchell/delaunay/v{_VERSION}/README.md)\n"
        )
    )
    files = {
        "Cargo.toml": f"{_CARGO_TOML}\n",
        "Cargo.lock": f'version = 4\n\n[[package]]\nname = "delaunay"\nversion = "{metadata_version}"\n',
        "pyproject.toml": f'[project]\nname = "delaunay-scripts"\nversion = "{metadata_version}"\n',
        "uv.lock": f'version = 1\n\n[[package]]\nname = "delaunay-scripts"\nversion = "{metadata_version}"\nsource = {{ editable = "." }}\n',
        "CITATION.cff": f"cff-version: 1.2.0\nversion: {metadata_version}\n",
        "README.md": readme_text,
    }
    for filename, content in files.items():
        (root / filename).write_text(content, encoding="utf-8")


def test_find_version_mismatches_accepts_matching_dependency_snippets(tmp_path: Path) -> None:
    """Dependency snippets matching Cargo.toml do not produce mismatches."""
    _write_project(
        tmp_path,
        readme='delaunay = "1.2.3"\ndelaunay = { version = "1.2.3", features = ["diagnostics"] }\nla-stack = "0.4.1"',
    )

    assert check_docs_version_sync.find_version_mismatches(tmp_path) == []


def test_find_version_mismatches_reports_stale_dependency_snippets(tmp_path: Path) -> None:
    """Stale active documentation snippets are reported with source context."""
    _write_project(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "install.md").write_text(
        'delaunay = { version = "1.2.2", features = ["diagnostics"] }\n',
        encoding="utf-8",
    )

    mismatches = check_docs_version_sync.find_version_mismatches(tmp_path)

    assert len(mismatches) == 1
    assert mismatches[0].reference.path == docs / "install.md"
    assert mismatches[0].reference.line == 1
    assert mismatches[0].reference.version == "1.2.2"
    assert mismatches[0].package.name == "delaunay"
    assert mismatches[0].package.version == "1.2.3"


def test_find_version_mismatches_handles_reordered_inline_table_keys(tmp_path: Path) -> None:
    """Inline Cargo dependency tables can put version after other keys."""
    _write_project(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    install_doc = docs / "install.md"
    install_doc.write_text(
        'delaunay = { features = ["diagnostics"], version = "1.2.2" }\n',
        encoding="utf-8",
    )

    mismatches = check_docs_version_sync.find_version_mismatches(tmp_path)

    assert len(mismatches) == 1
    assert mismatches[0].reference.path == install_doc
    assert mismatches[0].reference.line == 1
    assert mismatches[0].reference.version == "1.2.2"


def test_find_version_mismatches_reports_all_release_metadata(tmp_path: Path) -> None:
    """Cargo, Python, uv, and citation metadata must agree."""
    _write_project(
        tmp_path,
        metadata_version="1.2.2",
    )

    mismatches = check_docs_version_sync.find_version_mismatches(tmp_path)

    assert [(mismatch.reference.kind, mismatch.reference.path.name, mismatch.reference.line, mismatch.reference.version) for mismatch in mismatches] == [
        (check_docs_version_sync.ReferenceKind.CARGO_LOCK, "Cargo.lock", 5, "1.2.2"),
        (check_docs_version_sync.ReferenceKind.PYPROJECT, "pyproject.toml", 3, "1.2.2"),
        (check_docs_version_sync.ReferenceKind.UV_LOCK, "uv.lock", 5, "1.2.2"),
        (check_docs_version_sync.ReferenceKind.CITATION, "CITATION.cff", 2, "1.2.2"),
    ]


def test_find_version_mismatches_reports_readme_tag_links(tmp_path: Path) -> None:
    """Release-pinned README links must follow the current version."""
    _write_project(
        tmp_path,
        readme=(
            "[doc](https://github.com/acgetchell/delaunay/blob/v1.2.2/README.md)\n"
            "[raw](https://raw.githubusercontent.com/acgetchell/delaunay/v1.2.1/README.md)\n"
            "[stale-commit](https://github.com/acgetchell/delaunay/blob/abc1234/README.md)\n"
            "[moving](https://github.com/acgetchell/delaunay/blob/main/README.md)\n"
        ),
    )

    mismatches = check_docs_version_sync.find_version_mismatches(tmp_path)

    assert [mismatch.reference.kind for mismatch in mismatches] == [check_docs_version_sync.ReferenceKind.README_TAG_LINK] * 3
    assert [mismatch.reference.line for mismatch in mismatches] == [1, 2, 3]
    assert [mismatch.reference.version for mismatch in mismatches] == ["1.2.2", "1.2.1", "abc1234"]


@pytest.mark.parametrize("tag", ["v1.2.3.4", "v1.2.3.extra", "v1.2.3_suffix"])
def test_readme_tag_references_reject_longer_non_semver_tags(tmp_path: Path, tag: str) -> None:
    """Malformed version-looking README URLs are not partial matches."""
    readme = tmp_path / "README.md"
    readme.write_text(
        f"[invalid](https://github.com/acgetchell/delaunay/blob/{tag}/README.md)\n",
        encoding="utf-8",
    )

    assert check_docs_version_sync._readme_tag_references(readme) == []


@pytest.mark.parametrize("recipe", ["performance-github-assets", "performance-local", "performance-release"])
def test_find_version_mismatches_reports_stale_benchmark_current_tags(tmp_path: Path, recipe: str) -> None:
    """The first explicit benchmark release tag is the current release tag."""
    _write_project(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    workflows = docs / "workflows.md"
    workflows.write_text(
        f"| Release workflow | `just {recipe} v1.2.2 v1.2.1` |\n"
        "```bash\njust performance-release v1.2.3 v1.2.2\n```\n"
        "Historical v1.2.1 behavior remains documented.\n",
        encoding="utf-8",
    )

    mismatches = check_docs_version_sync.find_version_mismatches(tmp_path)

    assert len(mismatches) == 1
    assert mismatches[0].reference.kind is check_docs_version_sync.ReferenceKind.BENCHMARK_CURRENT_TAG
    assert mismatches[0].reference.path == workflows
    assert mismatches[0].reference.line == 1
    assert mismatches[0].reference.version == "1.2.2"


def test_benchmark_current_tag_references_ignore_baselines_and_historical_prose(tmp_path: Path) -> None:
    """Only the current-tag argument is checked for benchmark examples."""
    benchmarking = tmp_path / "BENCHMARKING.md"
    benchmarking.write_text(
        "just performance-release v1.2.3 v1.2.2\nThe v1.2.2 harness compares against v1.2.1.\n",
        encoding="utf-8",
    )

    references = check_docs_version_sync._benchmark_current_tag_references(benchmarking)

    assert [(reference.line, reference.version) for reference in references] == [(1, "1.2.3")]


@pytest.mark.parametrize(
    "version",
    ["1.2.3", "1.2.3-rc.1", "1.2.3+build.7", "1.2.3-rc.1+build.7"],
)
def test_readme_tag_references_accept_semver_suffixes(tmp_path: Path, version: str) -> None:
    """README release links may use SemVer prerelease and build suffixes."""
    readme = tmp_path / "README.md"
    readme.write_text(
        f"[tagged](https://github.com/acgetchell/delaunay/blob/v{version}/README.md)\n",
        encoding="utf-8",
    )

    references = check_docs_version_sync._readme_tag_references(readme)

    assert [(reference.line, reference.version) for reference in references] == [(1, version)]


def test_find_version_mismatches_ignores_historical_docs_and_test_fixtures(tmp_path: Path) -> None:
    """Archived docs, generated changelog, and test fixtures are not active release surfaces."""
    _write_project(tmp_path)
    archive = tmp_path / "docs" / "archive"
    archive.mkdir(parents=True)
    fixtures = tmp_path / "tests" / "fixtures"
    fixtures.mkdir(parents=True)
    stale_snippet = 'delaunay = "0.1.0"\njust performance-release v0.1.0 v0.0.9\n'
    (tmp_path / "CHANGELOG.md").write_text(stale_snippet, encoding="utf-8")
    (archive / "old.md").write_text(stale_snippet, encoding="utf-8")
    (fixtures / "example.md").write_text(stale_snippet, encoding="utf-8")

    assert check_docs_version_sync.find_version_mismatches(tmp_path) == []


def test_find_version_mismatches_rejects_missing_editable_uv_package(tmp_path: Path) -> None:
    """The checker requires uv.lock to include the editable local support package."""
    _write_project(tmp_path)
    (tmp_path / "uv.lock").write_text(
        'version = 1\n\n[[package]]\nname = "delaunay-scripts"\nversion = "1.2.3"\nsource = { registry = "https://pypi.org/simple" }\n',
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match=r"exactly one uv\.lock editable package"):
        check_docs_version_sync.find_version_mismatches(tmp_path)


def test_find_version_mismatches_rejects_malformed_citation_version(tmp_path: Path) -> None:
    """Malformed CITATION.cff versions fail before a release can continue."""
    _write_project(tmp_path)
    (tmp_path / "CITATION.cff").write_text('cff-version: 1.2.0\nversion: "\n', encoding="utf-8")

    with pytest.raises(TypeError, match=r"CITATION\.cff:2: top-level version"):
        check_docs_version_sync.find_version_mismatches(tmp_path)


def test_main_prints_mismatches(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI reports mismatches on stderr and exits nonzero."""
    _write_project(tmp_path, metadata_version="1.2.2")

    exit_code = check_docs_version_sync.main([str(tmp_path)])

    assert exit_code == 1
    assert "Release-version references are out of sync" in capsys.readouterr().err
