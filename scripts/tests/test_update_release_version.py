"""Tests for transactional release-version updates."""

from typing import TYPE_CHECKING

import pytest

import check_docs_version_sync
import update_release_version

if TYPE_CHECKING:
    from pathlib import Path

_CONCEPT_DOI = "10.5281/zenodo.16931097"


@pytest.fixture(autouse=True)
def _fixed_utc_date(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep release-date expectations deterministic unless a test advances UTC."""
    monkeypatch.setattr(update_release_version, "_current_utc_date", lambda: "2026-08-20")


def _write_project(root: Path, *, metadata_version: str = "1.2.2", dependency_version: str = "1.2.2") -> None:
    files = {
        "Cargo.toml": f'[package]\nname = "delaunay"\nversion = "{metadata_version}"\n',
        "Cargo.lock": (f'version = 4\n\n[[package]]\nname = "either"\nversion = "1.17.0"\n\n[[package]]\nname = "delaunay"\nversion = "{metadata_version}"\n'),
        "pyproject.toml": (
            f'[project]\nname = "delaunay-scripts"\nversion = "{metadata_version}"\nrequires-python = ">=3.14"\n\n[dependency-groups]\ndev = ["pytest>=9"]\n'
        ),
        "uv.lock": (f'version = 1\n\n[[package]]\nname = "delaunay-scripts"\nversion = "{metadata_version}"\nsource = {{ editable = "." }}\n'),
        "CITATION.cff": (f'cff-version: 1.2.0\nversion: {metadata_version}\ndate-released: 2026-07-13\ndoi: "{_CONCEPT_DOI}"\n'),
        "README.md": (
            f"cargo add delaunay@{dependency_version}\n"
            f'delaunay = "{dependency_version}"\n'
            f'delaunay = {{ version = "{dependency_version}", features = ["diagnostics"] }}\n'
            f"[license](https://github.com/acgetchell/delaunay/blob/v{metadata_version}/LICENSE)\n"
            f"[performance](https://github.com/acgetchell/delaunay/blob/v{metadata_version}/docs/PERFORMANCE.md)\n"
            f"[csv](https://github.com/acgetchell/delaunay/blob/v{metadata_version}/docs/assets/bench/release-performance.csv)\n"
        ),
        "CHANGELOG.md": "# Changelog\n\n## [1.2.2] - 2026-07-13\n",
    }
    for filename, content in files.items():
        (root / filename).write_text(content, encoding="utf-8")
    docs = root / "docs"
    docs.mkdir()
    (docs / "BENCHMARKING.md").write_text(
        "just performance-release v1.2.2 v1.2.1\nHistorical v1.2.1 behavior remains documented.\n",
        encoding="utf-8",
    )


def _previous() -> update_release_version.ReleaseTag:
    return update_release_version.parse_release_tag("v1.2.2")


def _file_snapshots(root: Path) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_update_release_version_updates_current_surfaces_without_dependency_upgrades(tmp_path: Path) -> None:
    _write_project(tmp_path)

    summary = update_release_version.update_release_version(
        tmp_path,
        "v1.2.3",
        previous=_previous(),
    )

    assert summary.target.tag == "v1.2.3"
    assert summary.previous.tag == "v1.2.2"
    assert summary.release_date == "2026-08-20"
    assert summary.changed_paths
    assert 'name = "either"\nversion = "1.17.0"' in (tmp_path / "Cargo.lock").read_text(encoding="utf-8")
    assert 'name = "delaunay"\nversion = "1.2.3"' in (tmp_path / "Cargo.lock").read_text(encoding="utf-8")
    assert 'name = "delaunay-scripts"\nversion = "1.2.3"' in (tmp_path / "uv.lock").read_text(encoding="utf-8")
    citation = (tmp_path / "CITATION.cff").read_text(encoding="utf-8")
    assert "version: 1.2.3" in citation
    assert "date-released: 2026-08-20" in citation
    assert f'doi: "{_CONCEPT_DOI}"' in citation
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "cargo add delaunay@1.2.3" in readme
    assert 'delaunay = "1.2.3"' in readme
    assert 'version = "1.2.3"' in readme
    assert "blob/v1.2.3/LICENSE" in readme
    assert "blob/v1.2.2/docs/PERFORMANCE.md" in readme
    assert "blob/v1.2.2/docs/assets/bench/release-performance.csv" in readme
    benchmarking = (tmp_path / "docs" / "BENCHMARKING.md").read_text(encoding="utf-8")
    assert "just performance-release v1.2.3 v1.2.2" in benchmarking
    assert "Historical v1.2.1 behavior remains documented." in benchmarking
    assert "## [1.2.2] - 2026-07-13" in (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8")
    assert check_docs_version_sync.find_version_mismatches(tmp_path) == []


def test_update_release_version_updates_single_quoted_dependency_snippets(tmp_path: Path) -> None:
    _write_project(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text("delaunay = '1.2.2'\ndelaunay = { version = '1.2.2', features = ['diagnostics'] }\n", encoding="utf-8")

    update_release_version.update_release_version(
        tmp_path,
        "v1.2.3",
        previous=_previous(),
    )

    assert readme.read_text(encoding="utf-8") == "delaunay = '1.2.3'\ndelaunay = { version = '1.2.3', features = ['diagnostics'] }\n"


def test_update_release_version_is_content_idempotent_on_the_same_utc_day(tmp_path: Path) -> None:
    _write_project(tmp_path)
    kwargs = {"previous": _previous()}

    first = update_release_version.update_release_version(tmp_path, "v1.2.3", **kwargs)
    second = update_release_version.update_release_version(tmp_path, "v1.2.3", **kwargs)

    assert first.changed_paths
    assert second.changed_paths == ()


def test_successful_update_preserves_crlf_line_endings(tmp_path: Path) -> None:
    _write_project(tmp_path)
    cargo_lock = tmp_path / "Cargo.lock"
    cargo_lock.write_bytes(cargo_lock.read_bytes().replace(b"\n", b"\r\n"))

    summary = update_release_version.update_release_version(
        tmp_path,
        "v1.2.3",
        previous=_previous(),
    )

    content = cargo_lock.read_bytes()
    assert cargo_lock in summary.changed_paths
    assert b'\r\nname = "delaunay"\r\nversion = "1.2.3"\r\n' in content
    assert b"\n" not in content.replace(b"\r\n", b"")


def test_successful_update_preserves_crlf_benchmark_commands(tmp_path: Path) -> None:
    _write_project(tmp_path)
    benchmarking = tmp_path / "docs" / "BENCHMARKING.md"
    benchmarking.write_bytes(benchmarking.read_bytes().replace(b"\n", b"\r\n"))

    summary = update_release_version.update_release_version(
        tmp_path,
        "v1.2.3",
        previous=_previous(),
    )

    content = benchmarking.read_bytes()
    assert benchmarking in summary.changed_paths
    assert b"just performance-release v1.2.3 v1.2.2\r\n" in content
    assert b"\n" not in content.replace(b"\r\n", b"")


def test_update_release_version_advances_existing_release_dates_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project(tmp_path, metadata_version="1.2.3", dependency_version="1.2.3")
    citation = tmp_path / "CITATION.cff"
    citation.write_text(citation.read_text(encoding="utf-8").replace("2026-07-13", "2026-08-20"), encoding="utf-8")
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("# Changelog\n\n## [1.2.3] - 2026-08-20\n", encoding="utf-8")
    benchmarking = tmp_path / "docs" / "BENCHMARKING.md"
    benchmarking.write_text("just performance-release v1.2.3 v1.2.2\n", encoding="utf-8")
    monkeypatch.setattr(update_release_version, "_current_utc_date", lambda: "2026-08-21")

    summary = update_release_version.update_release_version(
        tmp_path,
        "v1.2.3",
        previous=_previous(),
    )

    assert summary.changed_paths == (changelog, citation)
    assert "date-released: 2026-08-21" in citation.read_text(encoding="utf-8")
    assert "## [1.2.3] - 2026-08-21" in changelog.read_text(encoding="utf-8")


def test_select_previous_release_tag_uses_latest_stable_published_tag() -> None:
    target = update_release_version.parse_release_tag("v1.3.0")

    previous = update_release_version.select_previous_release_tag(
        ["v1.1.9", "v1.2.0-rc.1", "not-a-release", "v1.2.0"],
        target,
    )

    assert previous.tag == "v1.2.0"


def test_select_previous_release_tag_ignores_already_published_target() -> None:
    target = update_release_version.parse_release_tag("v1.3.0")

    previous = update_release_version.select_previous_release_tag(["v1.2.0", "v1.3.0"], target)

    assert previous.tag == "v1.2.0"


@pytest.mark.parametrize("target", ["1.2.3", "v1.2", "v01.2.3", "v1.2.3-rc.1"])
def test_parse_release_tag_rejects_non_stable_tag_forms(target: str) -> None:
    with pytest.raises(ValueError, match=r"stable tag in vX\.Y\.Z form"):
        update_release_version.parse_release_tag(target)


def test_release_tag_direct_construction_rejects_contradictory_components() -> None:
    with pytest.raises(ValueError, match="contradict emitted tag"):
        update_release_version.ReleaseTag(major=1, minor=2, patch=2, tag="v1.2.3")


def test_select_previous_release_tag_rejects_missing_history() -> None:
    target = update_release_version.parse_release_tag("v1.2.3")

    with pytest.raises(ValueError, match="no published stable"):
        update_release_version.select_previous_release_tag([], target)


def test_select_previous_release_tag_rejects_target_older_than_published_release() -> None:
    target = update_release_version.parse_release_tag("v1.2.3")

    with pytest.raises(ValueError, match="older than published"):
        update_release_version.select_previous_release_tag(["v1.2.3", "v1.3.0"], target)


def test_infer_previous_release_tag_uses_published_github_releases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(update_release_version, "published_stable_release_tags", lambda root: ["v1.2.1", "v1.2.2"] if root == tmp_path else [])

    previous = update_release_version.infer_previous_release_tag(tmp_path, update_release_version.parse_release_tag("v1.2.3"))

    assert previous.tag == "v1.2.2"


def test_unexpected_version_fails_before_writing(tmp_path: Path) -> None:
    _write_project(tmp_path, dependency_version="1.0.0")
    originals = {path: path.read_text(encoding="utf-8") for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="unexpected delaunay dependency version"):
        update_release_version.update_release_version(
            tmp_path,
            "v1.2.3",
            previous=_previous(),
        )

    assert {path: path.read_text(encoding="utf-8") for path in originals} == originals


@pytest.mark.parametrize("baseline", ["v1.2", "v1.2.2", "v1.2.4"])
def test_invalid_benchmark_baseline_fails_before_writing(tmp_path: Path, baseline: str) -> None:
    _write_project(tmp_path)
    workflow = tmp_path / "docs/BENCHMARKING.md"
    workflow.write_text(f"just performance-release v1.2.2 {baseline}\n", encoding="utf-8")
    originals = _file_snapshots(tmp_path)

    with pytest.raises(ValueError, match="benchmark baseline"):
        update_release_version.update_release_version(
            tmp_path,
            "v1.2.3",
            previous=_previous(),
        )

    assert _file_snapshots(tmp_path) == originals


def test_nonadjacent_target_benchmark_baseline_fails_before_writing(tmp_path: Path) -> None:
    _write_project(tmp_path)
    workflow = tmp_path / "docs/BENCHMARKING.md"
    workflow.write_text("just performance-release v1.2.3 v1.2.1\n", encoding="utf-8")
    originals = _file_snapshots(tmp_path)

    with pytest.raises(ValueError, match=r"non-adjacent benchmark pair.*expected baseline v1\.2\.2"):
        update_release_version.update_release_version(
            tmp_path,
            "v1.2.3",
            previous=_previous(),
        )

    assert _file_snapshots(tmp_path) == originals


@pytest.mark.parametrize("current", ["1.2.2", "v1.2"])
def test_malformed_benchmark_current_tag_fails_before_writing(tmp_path: Path, current: str) -> None:
    _write_project(tmp_path)
    workflow = tmp_path / "docs/BENCHMARKING.md"
    workflow.write_text(f"just performance-release {current} v1.2.1\n", encoding="utf-8")
    originals = _file_snapshots(tmp_path)

    with pytest.raises(ValueError, match="benchmark current tag"):
        update_release_version.update_release_version(
            tmp_path,
            "v1.2.3",
            previous=_previous(),
        )

    assert _file_snapshots(tmp_path) == originals


def test_benchmark_command_missing_baseline_fails_before_writing(tmp_path: Path) -> None:
    _write_project(tmp_path)
    workflow = tmp_path / "docs/BENCHMARKING.md"
    workflow.write_text("`just performance-release v1.2.2`\n", encoding="utf-8")
    originals = _file_snapshots(tmp_path)

    with pytest.raises(ValueError, match="missing the baseline tag"):
        update_release_version.update_release_version(
            tmp_path,
            "v1.2.3",
            previous=_previous(),
        )

    assert _file_snapshots(tmp_path) == originals


def test_non_root_editable_uv_package_fails_before_writing(tmp_path: Path) -> None:
    _write_project(tmp_path)
    uv_lock = tmp_path / "uv.lock"
    uv_lock.write_text(
        'version = 1\n\n[[package]]\nname = "delaunay-scripts"\nversion = "1.2.2"\nsource = { editable = "../other" }\n',
        encoding="utf-8",
    )
    originals = _file_snapshots(tmp_path)

    with pytest.raises(TypeError, match=r"exactly one uv\.lock editable package"):
        update_release_version.update_release_version(
            tmp_path,
            "v1.2.3",
            previous=_previous(),
        )

    assert _file_snapshots(tmp_path) == originals


def test_validation_failure_precedes_every_repository_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_project(tmp_path)
    originals = {path: path.read_text(encoding="utf-8") for path in tmp_path.rglob("*") if path.is_file()}
    writes: list[Path] = []

    def fail_validation(*_args: object) -> None:
        msg = "simulated validation failure"
        raise ValueError(msg)

    def record_write(path: Path, text: str) -> None:
        writes.append(path)
        path.write_text(text, encoding="utf-8")

    monkeypatch.setattr(update_release_version, "_validate_updated_root", fail_validation)
    monkeypatch.setattr(update_release_version, "_write_text_atomic", record_write)

    with pytest.raises(ValueError, match="simulated validation failure"):
        update_release_version.update_release_version(
            tmp_path,
            "v1.2.3",
            previous=_previous(),
        )

    assert writes == []
    assert {path: path.read_text(encoding="utf-8") for path in originals} == originals


def test_mid_write_failure_rolls_back_every_release_surface(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_project(tmp_path)
    originals = _file_snapshots(tmp_path)
    real_write = update_release_version._write_text_atomic
    calls = 0

    def fail_second_write(path: Path, text: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            msg = "simulated mid-write failure"
            raise OSError(msg)
        real_write(path, text)

    monkeypatch.setattr(update_release_version, "_write_text_atomic", fail_second_write)

    with pytest.raises(OSError, match="simulated mid-write failure"):
        update_release_version.update_release_version(
            tmp_path,
            "v1.2.3",
            previous=_previous(),
        )

    assert _file_snapshots(tmp_path) == originals


def test_mid_write_failure_restores_original_crlf_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_project(tmp_path)
    cargo_lock = tmp_path / "Cargo.lock"
    cargo_lock.write_bytes(cargo_lock.read_bytes().replace(b"\n", b"\r\n"))
    originals = _file_snapshots(tmp_path)
    real_write = update_release_version._write_text_atomic
    calls = 0

    def fail_second_write(path: Path, text: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            msg = "simulated mid-write failure"
            raise OSError(msg)
        real_write(path, text)

    monkeypatch.setattr(update_release_version, "_write_text_atomic", fail_second_write)

    with pytest.raises(OSError, match="simulated mid-write failure"):
        update_release_version.update_release_version(
            tmp_path,
            "v1.2.3",
            previous=_previous(),
        )

    assert _file_snapshots(tmp_path) == originals


def test_post_write_validation_failure_rolls_back_every_release_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project(tmp_path)
    originals = _file_snapshots(tmp_path)
    real_validate = update_release_version._validate_updated_root
    calls = 0

    def fail_final_validation(root: Path, target: update_release_version.ReleaseTag, previous: update_release_version.ReleaseTag) -> None:
        nonlocal calls
        calls += 1
        real_validate(root, target, previous)
        if calls == 2:
            msg = "simulated post-write validation failure"
            raise ValueError(msg)

    monkeypatch.setattr(update_release_version, "_validate_updated_root", fail_final_validation)

    with pytest.raises(ValueError, match="simulated post-write validation failure"):
        update_release_version.update_release_version(
            tmp_path,
            "v1.2.3",
            previous=_previous(),
        )

    assert _file_snapshots(tmp_path) == originals


def test_rollback_failure_reports_primary_and_restore_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_project(tmp_path)
    real_write = update_release_version._write_text_atomic
    calls = 0

    def fail_rollback(_path: Path, _content: bytes) -> None:
        msg = "simulated rollback failure"
        raise OSError(msg)

    def fail_second_write(path: Path, text: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            monkeypatch.setattr(update_release_version, "_write_bytes_atomic", fail_rollback)
            msg = "simulated primary write failure"
            raise OSError(msg)
        real_write(path, text)

    monkeypatch.setattr(update_release_version, "_write_text_atomic", fail_second_write)

    with pytest.raises(RuntimeError, match=r"simulated primary write failure.*rollback also failed.*simulated rollback failure"):
        update_release_version.update_release_version(
            tmp_path,
            "v1.2.3",
            previous=_previous(),
        )


def test_sync_changelog_release_date_uses_citation_date(tmp_path: Path) -> None:
    _write_project(tmp_path)
    update_release_version.update_release_version(
        tmp_path,
        "v1.2.3",
        previous=_previous(),
    )
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("# Changelog\n\n## [1.2.3] - 2026-08-21\n", encoding="utf-8")

    changed, release_date = update_release_version.sync_changelog_release_date(
        tmp_path,
        "v1.2.3",
        previous=_previous(),
    )

    assert changed == (changelog,)
    assert release_date == "2026-08-20"
    assert "## [1.2.3] - 2026-08-20" in changelog.read_text(encoding="utf-8")
    assert check_docs_version_sync.find_version_mismatches(tmp_path) == []


def test_main_supports_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="0"):
        update_release_version.main(["--help"])

    assert "Target stable release tag" in capsys.readouterr().out


def test_main_prints_previous_release_without_mutating_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_project(tmp_path)
    originals = _file_snapshots(tmp_path)
    monkeypatch.setattr(update_release_version, "published_stable_release_tags", lambda root: ["v1.2.1", "v1.2.2"] if root == tmp_path else [])

    assert update_release_version.main(["v1.2.3", "--root", str(tmp_path), "--print-previous-release"]) == 0
    assert capsys.readouterr().out == "v1.2.2\n"
    assert _file_snapshots(tmp_path) == originals


def test_previous_release_preflight_failure_leaves_files_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_project(tmp_path)
    originals = _file_snapshots(tmp_path)

    def fail_lookup(*_args: object) -> update_release_version.ReleaseTag:
        msg = "GitHub release lookup failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(update_release_version, "infer_previous_release_tag", fail_lookup)

    assert update_release_version.main(["v1.2.3", "--root", str(tmp_path), "--print-previous-release"]) == 1
    assert "GitHub release lookup failed" in capsys.readouterr().err
    assert _file_snapshots(tmp_path) == originals


def test_main_uses_preflighted_previous_release_without_remote_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project(tmp_path)

    def unexpected_lookup(*_args: object) -> None:
        msg = "published release lookup must not repeat after preflight"
        raise AssertionError(msg)

    monkeypatch.setattr(update_release_version, "infer_previous_release_tag", unexpected_lookup)

    assert (
        update_release_version.main(
            [
                "v1.2.3",
                "--root",
                str(tmp_path),
                "--previous-release",
                "v1.2.2",
            ]
        )
        == 0
    )


def test_main_retries_are_idempotent_on_the_same_utc_day(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_project(tmp_path)

    def fixed_previous(root: Path, target: update_release_version.ReleaseTag) -> update_release_version.ReleaseTag:
        assert root == tmp_path
        assert target.tag == "v1.2.3"
        return _previous()

    monkeypatch.setattr(update_release_version, "infer_previous_release_tag", fixed_previous)
    argv = ["v1.2.3", "--root", str(tmp_path)]

    assert update_release_version.main(argv) == 0
    after_first = _file_snapshots(tmp_path)
    assert update_release_version.main(argv) == 0

    assert _file_snapshots(tmp_path) == after_first
    output = capsys.readouterr().out
    assert "CITATION.cff UTC release date: 2026-08-20" in output
    assert "Release-version references already match v1.2.3." in output
