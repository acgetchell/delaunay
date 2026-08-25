"""Tests for documentation and package-version synchronization checks."""

from typing import TYPE_CHECKING

import pytest

import check_docs_version_sync

if TYPE_CHECKING:
    from pathlib import Path


_CARGO_TOML = '[package]\nname = "delaunay"\nversion = "1.2.3"'
_VERSION = "1.2.3"
_ZENODO_CONCEPT_DOI = "10.5281/zenodo.16931097"


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
        "CITATION.cff": (f"cff-version: 1.2.0\nversion: {metadata_version}\ndoi: {_ZENODO_CONCEPT_DOI}\ndate-released: 2026-01-02\n"),
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


def test_find_version_mismatches_reports_single_quoted_dependency_snippets(tmp_path: Path) -> None:
    """TOML literal-string dependency versions remain release-owned surfaces."""
    _write_project(
        tmp_path,
        readme="delaunay = '1.2.2'\ndelaunay = { features = ['diagnostics'], version = '1.2.1' }\n",
    )

    mismatches = check_docs_version_sync.find_version_mismatches(tmp_path)

    assert [mismatch.reference.version for mismatch in mismatches] == ["1.2.2", "1.2.1"]


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


def test_find_version_mismatches_reports_stale_cargo_add_commands(tmp_path: Path) -> None:
    """Stale cargo-add commands are reported without matching other packages."""
    _write_project(
        tmp_path,
        readme=("cargo add delaunay@1.2.3\n`cargo add --features diagnostics delaunay@1.2.2`\ncargo add la-stack@0.4.1\n"),
    )

    mismatches = check_docs_version_sync.find_version_mismatches(tmp_path)

    assert len(mismatches) == 1
    assert mismatches[0].reference.kind is check_docs_version_sync.ReferenceKind.CARGO_ADD
    assert mismatches[0].reference.path == tmp_path / "README.md"
    assert mismatches[0].reference.line == 2
    assert mismatches[0].reference.version == "1.2.2"


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


@pytest.mark.parametrize("recipe", ["performance-github-assets", "performance-release"])
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


def test_benchmark_current_tag_references_validate_pair_and_ignore_historical_prose(tmp_path: Path) -> None:
    """A valid baseline relationship is checked without scanning historical prose."""
    benchmarking = tmp_path / "BENCHMARKING.md"
    benchmarking.write_text(
        "just performance-release v1.2.3 v1.2.2\nThe v1.2.2 harness compares against v1.2.1.\n",
        encoding="utf-8",
    )

    references = check_docs_version_sync._benchmark_current_tag_references(benchmarking)

    assert [(reference.line, reference.version) for reference in references] == [(1, "1.2.3")]


@pytest.mark.parametrize("baseline", ["v1.2", "v1.2.3", "v1.2.4"])
def test_find_version_mismatches_rejects_invalid_benchmark_baseline(tmp_path: Path, baseline: str) -> None:
    """Malformed, equal, and newer benchmark baselines fail closed."""
    _write_project(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    workflow = docs / "workflow.md"
    workflow.write_text(f"just performance-release v1.2.3 {baseline}\n", encoding="utf-8")

    with pytest.raises(TypeError, match="benchmark baseline tag") as exc_info:
        check_docs_version_sync.find_version_mismatches(tmp_path)

    assert "workflow.md:1" in str(exc_info.value)


@pytest.mark.parametrize("current", ["1.2.3", "v1.2"])
def test_find_version_mismatches_rejects_invalid_benchmark_current_tag(tmp_path: Path, current: str) -> None:
    """Numeric current tags cannot evade stable vX.Y.Z parsing."""
    _write_project(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    workflow = docs / "workflow.md"
    workflow.write_text(f"just performance-release {current} v1.2.2\n", encoding="utf-8")

    with pytest.raises(TypeError, match="benchmark current tag") as exc_info:
        check_docs_version_sync.find_version_mismatches(tmp_path)

    assert "workflow.md:1" in str(exc_info.value)


def test_find_version_mismatches_rejects_benchmark_command_missing_baseline(tmp_path: Path) -> None:
    """A lone explicit current tag cannot evade the complete command contract."""
    _write_project(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    workflow = docs / "workflow.md"
    workflow.write_text("`just performance-release v1.2.3`\n", encoding="utf-8")

    with pytest.raises(TypeError, match="missing its baseline tag") as exc_info:
        check_docs_version_sync.find_version_mismatches(tmp_path)

    assert "workflow.md:1" in str(exc_info.value)


def test_find_version_mismatches_leaves_performance_readme_links_to_their_owner(tmp_path: Path) -> None:
    """Performance publication, not metadata updates, owns tagged evidence links."""
    _write_project(
        tmp_path,
        readme=(
            "[license](https://github.com/acgetchell/delaunay/blob/v1.2.3/LICENSE)\n"
            "[report](https://github.com/acgetchell/delaunay/blob/v1.2.2/docs/PERFORMANCE.md)\n"
            "[csv](https://github.com/acgetchell/delaunay/blob/v1.2.2/docs/assets/bench/release-performance.csv)\n"
            "[provenance](https://github.com/acgetchell/delaunay/blob/v1.2.2/docs/archive/performance/data/v1.2.2-vs-v1.2.1.provenance.json)\n"
        ),
    )

    assert check_docs_version_sync.find_version_mismatches(tmp_path) == []


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
    (tmp_path / "CHANGELOG.md").write_text(
        f"# Changelog\n\n## [{_VERSION}] - 2026-01-02\n\n{stale_snippet}",
        encoding="utf-8",
    )
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


def test_find_version_mismatches_rejects_same_named_non_root_editable_uv_package(tmp_path: Path) -> None:
    """A same-named package outside the repository root is not authoritative."""
    _write_project(tmp_path)
    (tmp_path / "uv.lock").write_text(
        'version = 1\n\n[[package]]\nname = "delaunay-scripts"\nversion = "1.2.3"\nsource = { editable = "../other" }\n',
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match=r"exactly one uv\.lock editable package"):
        check_docs_version_sync.find_version_mismatches(tmp_path)


@pytest.mark.parametrize(
    "version_line",
    ['version: "\n', f"version: {_VERSION}#not-a-comment\n", f'version: "{_VERSION}"#not-a-comment\n'],
)
def test_find_version_mismatches_rejects_malformed_citation_version(tmp_path: Path, version_line: str) -> None:
    """Malformed CITATION.cff versions fail before a release can continue."""
    _write_project(tmp_path)
    (tmp_path / "CITATION.cff").write_text(
        f"cff-version: 1.2.0\n{version_line}doi: {_ZENODO_CONCEPT_DOI}\ndate-released: 2026-01-02\n",
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match=r"CITATION\.cff:2: top-level version"):
        check_docs_version_sync.find_version_mismatches(tmp_path)


@pytest.mark.parametrize(
    ("date_line", "message"),
    [
        ("", "missing top-level date-released"),
        ("date-released: 2026-7-28\n", "must use YYYY-MM-DD"),
        ("date-released: 2026-02-30\n", "not a valid ISO date"),
        ("date-released: 2026-01-02#not-a-comment\n", "must use YYYY-MM-DD"),
        ('date-released: "2026-01-02"#not-a-comment\n', "must use YYYY-MM-DD"),
    ],
)
def test_release_date_rejects_missing_or_malformed_citation_value(tmp_path: Path, date_line: str, message: str) -> None:
    """Citation release dates are required and calendar-valid."""
    _write_project(tmp_path)
    citation = tmp_path / "CITATION.cff"
    citation.write_text(f"cff-version: 1.2.0\nversion: {_VERSION}\n{date_line}", encoding="utf-8")

    with pytest.raises(TypeError, match=message) as exc_info:
        check_docs_version_sync.find_version_mismatches(tmp_path)

    assert "CITATION.cff" in str(exc_info.value)


def test_release_date_rejects_duplicate_citation_values_with_lines(tmp_path: Path) -> None:
    """Duplicate top-level citation dates report every conflicting line."""
    _write_project(tmp_path)
    citation = tmp_path / "CITATION.cff"
    citation.write_text(
        f"cff-version: 1.2.0\nversion: {_VERSION}\ndate-released: 2026-01-02\ndate-released: 2026-01-03\n",
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="duplicate top-level date-released") as exc_info:
        check_docs_version_sync.find_version_mismatches(tmp_path)

    message = str(exc_info.value)
    assert "CITATION.cff:3" in message
    assert "CITATION.cff:4" in message


@pytest.mark.parametrize(
    ("doi_lines", "message"),
    [
        ("", "missing top-level doi"),
        ("doi: 10.5281/zenodo.99999999\n", "must remain the Zenodo concept DOI"),
        (
            f"doi: {_ZENODO_CONCEPT_DOI}\ndoi: {_ZENODO_CONCEPT_DOI}\n",
            "duplicate top-level doi values",
        ),
    ],
)
def test_citation_requires_exactly_one_zenodo_concept_doi(tmp_path: Path, doi_lines: str, message: str) -> None:
    """Release metadata keeps the stable concept DOI across versions."""
    _write_project(tmp_path)
    (tmp_path / "CITATION.cff").write_text(
        f"cff-version: 1.2.0\nversion: {_VERSION}\n{doi_lines}date-released: 2026-01-02\n",
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match=message):
        check_docs_version_sync.find_version_mismatches(tmp_path)


def test_release_date_must_match_generated_changelog_heading(tmp_path: Path) -> None:
    """The CFF date and current-version changelog heading use one UTC date."""
    _write_project(tmp_path)
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [1.2.3] - 2026-01-03\n",
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="release date mismatch") as exc_info:
        check_docs_version_sync.find_version_mismatches(tmp_path)

    message = str(exc_info.value)
    assert "CITATION.cff:4" in message
    assert "CHANGELOG.md:3" in message


def test_release_date_allows_target_heading_before_changelog_generation(tmp_path: Path) -> None:
    """Metadata can be bumped before changelog-unreleased generates its heading."""
    _write_project(tmp_path)
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [1.2.2] - 2026-01-02\n",
        encoding="utf-8",
    )

    assert check_docs_version_sync.find_version_mismatches(tmp_path) == []


def test_final_release_requires_current_changelog_heading(tmp_path: Path) -> None:
    """The strict final gate rejects the tolerant pre-generation state."""
    _write_project(tmp_path)
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [1.2.2] - 2026-01-02\n",
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="final release validation requires exactly one generated heading"):
        check_docs_version_sync.find_version_mismatches(tmp_path, final_release=True)


def test_final_release_accepts_exactly_one_matching_changelog_heading(tmp_path: Path) -> None:
    """The strict final gate accepts one current heading with the citation date."""
    _write_project(tmp_path)
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [1.2.3] - 2026-01-02\n",
        encoding="utf-8",
    )

    assert check_docs_version_sync.find_version_mismatches(tmp_path, final_release=True) == []


def test_release_date_rejects_malformed_current_changelog_heading(tmp_path: Path) -> None:
    """A current-version heading reports its malformed date with line context."""
    _write_project(tmp_path)
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [1.2.3] - 2026-1-02\n",
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="must end with YYYY-MM-DD") as exc_info:
        check_docs_version_sync.find_version_mismatches(tmp_path)

    assert "CHANGELOG.md:3" in str(exc_info.value)


def test_release_date_rejects_duplicate_current_changelog_headings(tmp_path: Path) -> None:
    """Duplicate current-version headings report every conflicting line."""
    _write_project(tmp_path)
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [1.2.3] - 2026-01-02\n\n## [1.2.3] - 2026-01-02\n",
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="duplicate release headings") as exc_info:
        check_docs_version_sync.find_version_mismatches(tmp_path)

    message = str(exc_info.value)
    assert "CHANGELOG.md:3" in message
    assert "CHANGELOG.md:5" in message


def test_release_date_accepts_matching_generated_changelog_heading(tmp_path: Path) -> None:
    """A matching generated release heading passes the date gate."""
    _write_project(tmp_path)
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [1.2.3] - 2026-01-02\n",
        encoding="utf-8",
    )

    assert check_docs_version_sync.find_version_mismatches(tmp_path) == []


def test_main_prints_mismatches(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI reports mismatches on stderr and exits nonzero."""
    _write_project(tmp_path, metadata_version="1.2.2")

    exit_code = check_docs_version_sync.main([str(tmp_path)])

    assert exit_code == 1
    assert "Release-version references are out of sync" in capsys.readouterr().err


def test_main_final_release_reports_missing_changelog_heading(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI exposes strict final-release changelog validation."""
    _write_project(tmp_path)

    exit_code = check_docs_version_sync.main(["--final-release", str(tmp_path)])

    assert exit_code == 1
    assert "final release validation requires" in capsys.readouterr().err
