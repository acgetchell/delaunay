"""Tests for changelog-utils generate workflow.

These focus on the plumbing around `changelog-utils generate` (invoked via `just changelog`)
so that the release workflow in docs/RELEASING.md stays reliable.

We avoid calling external tools (npx/auto-changelog, AI helpers) by patching those steps,
but still exercise the file-based pipeline and cleanup behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import changelog_utils
from changelog_utils import ChangelogError, _cleanup_final_output, _cleanup_temp_files, _post_process_dates


def _file_paths(tmp_path: Path) -> dict[str, Path]:
    changelog_file = tmp_path / "CHANGELOG.md"
    return {
        "changelog": changelog_file,
        "temp": tmp_path / "CHANGELOG.md.tmp",
        "processed": tmp_path / "CHANGELOG.md.processed",
        "expanded": tmp_path / "CHANGELOG.md.processed.expanded",
        "enhanced": tmp_path / "CHANGELOG.md.tmp2",
        "backup": tmp_path / "CHANGELOG.md.backup",
    }


def test_post_process_dates_strips_iso_time_component(tmp_path: Path) -> None:
    file_paths = _file_paths(tmp_path)
    file_paths["temp"].write_text(
        "\n".join(
            [
                "## v1.0.0 - 2025-01-01T12:34:56Z",
                "## v1.0.1 - 2025-01-02T12:34:56.123Z",
                "## v1.0.2 - 2025-01-03T12:34:56+00:00",
                "## v1.0.3 - 2025-01-04T12:34:56-07:00",
                "",
            ]
        ),
        encoding="utf-8",
    )

    _post_process_dates(file_paths)

    processed = file_paths["processed"].read_text(encoding="utf-8")
    assert "T12:34:56" not in processed
    assert "## v1.0.0 - 2025-01-01" in processed
    assert "## v1.0.1 - 2025-01-02" in processed
    assert "## v1.0.2 - 2025-01-03" in processed
    assert "## v1.0.3 - 2025-01-04" in processed


def test_cleanup_final_output_collapses_multiple_blank_lines(tmp_path: Path) -> None:
    file_paths = _file_paths(tmp_path)
    file_paths["enhanced"].write_text("a\n\n\n\n\nb\n\n\nc\n", encoding="utf-8")

    _cleanup_final_output(file_paths)

    out = file_paths["changelog"].read_text(encoding="utf-8")
    assert "a" in out
    assert "b" in out
    assert "c" in out
    assert "\n\n\n" not in out, "Expected no sequences of 2+ blank lines"


def test_cleanup_temp_files_non_debug_removes_intermediates_and_backup(tmp_path: Path) -> None:
    file_paths = _file_paths(tmp_path)

    # Create all intermediate files + backup.
    for key in ["temp", "processed", "expanded", "enhanced", "backup"]:
        file_paths[key].write_text(key, encoding="utf-8")
        assert file_paths[key].exists()

    _cleanup_temp_files(file_paths, debug_mode=False)

    for key in ["temp", "processed", "expanded", "enhanced", "backup"]:
        assert not file_paths[key].exists(), f"Expected {key} to be deleted"


def test_cleanup_temp_files_debug_keeps_intermediates_but_removes_backup(tmp_path: Path) -> None:
    file_paths = _file_paths(tmp_path)

    for key in ["temp", "processed", "expanded", "enhanced", "backup"]:
        file_paths[key].write_text(key, encoding="utf-8")
        assert file_paths[key].exists()

    _cleanup_temp_files(file_paths, debug_mode=True)

    for key in ["temp", "processed", "expanded", "enhanced"]:
        assert file_paths[key].exists(), f"Expected {key} to be preserved in debug mode"

    assert not file_paths["backup"].exists(), "Expected backup to be removed even in debug mode"


def test_execute_changelog_generation_writes_final_output_and_cleans_up(tmp_path: Path, monkeypatch) -> None:
    # Arrange: run from a different directory than the project root.
    workdir = tmp_path / "workdir"
    project_root = tmp_path / "project"
    workdir.mkdir()
    project_root.mkdir()
    monkeypatch.chdir(workdir)

    # Existing changelog (to verify backup/overwrite behavior).
    (project_root / "CHANGELOG.md").write_text("# Old changelog\n", encoding="utf-8")

    monkeypatch.setattr(changelog_utils.ChangelogUtils, "get_project_root", lambda: str(project_root))

    # Avoid external dependencies.
    monkeypatch.setattr(changelog_utils, "_validate_prerequisites", lambda: None)
    monkeypatch.setattr(changelog_utils, "_get_repository_url", lambda: "https://github.com/acgetchell/delaunay")

    sample = """# Changelog

## [v0.1.0] - 2025-01-01T00:00:00Z

### Changed

All tests pass: 123 passed

- **Bump MSRV to 1.92.0** [`abcdef123`](https://github.com/acgetchell/delaunay/commit/abcdef123)
  Details that should not appear in the breaking-changes summary.

- **Regular change** [`123456789`](https://github.com/acgetchell/delaunay/commit/123456789)

"""

    def fake_run_auto_changelog(file_paths: dict[str, Path], _project_root: Path) -> None:
        file_paths["temp"].write_text(sample, encoding="utf-8")

    def fake_expand_squashed_commits(file_paths: dict[str, Path], _repo_url: str) -> None:
        # Skip the git-dependent expansion logic; keep the file pipeline intact.
        file_paths["expanded"].write_text(file_paths["processed"].read_text(encoding="utf-8"), encoding="utf-8")

    def fake_enhance_with_ai(file_paths: dict[str, Path], _project_root: Path) -> None:
        # Skip the external AI categorization step; add extra blank lines to ensure cleanup runs.
        expanded = file_paths["expanded"].read_text(encoding="utf-8")
        file_paths["enhanced"].write_text(expanded + "\n\n\n", encoding="utf-8")

    monkeypatch.setattr(changelog_utils, "_run_auto_changelog", fake_run_auto_changelog)
    monkeypatch.setattr(changelog_utils, "_expand_squashed_commits", fake_expand_squashed_commits)
    monkeypatch.setattr(changelog_utils, "_enhance_with_ai", fake_enhance_with_ai)

    # Avoid printing during tests.
    monkeypatch.setattr(changelog_utils, "_show_success_message", lambda _file_paths: None)

    # Act
    changelog_utils._execute_changelog_generation(debug_mode=False)

    # Assert: working directory is restored.
    assert Path.cwd() == workdir

    # Assert: final output exists and contains expected transformations.
    final = (project_root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "T00:00:00Z" not in final
    assert "All tests pass:" not in final
    assert "\n\n\n" not in final

    # Assert: intermediates are cleaned up (default mode).
    for name in [
        "CHANGELOG.md.backup",
        "CHANGELOG.md.tmp",
        "CHANGELOG.md.processed",
        "CHANGELOG.md.processed.expanded",
        "CHANGELOG.md.tmp2",
    ]:
        assert not (project_root / name).exists(), f"Expected {name} to be removed"


def test_execute_changelog_generation_restores_backup_on_failure(tmp_path: Path, monkeypatch) -> None:
    workdir = tmp_path / "workdir"
    project_root = tmp_path / "project"
    workdir.mkdir()
    project_root.mkdir()
    monkeypatch.chdir(workdir)

    original = "# Original changelog\n\n## v0.0.1\n- old\n"
    (project_root / "CHANGELOG.md").write_text(original, encoding="utf-8")

    monkeypatch.setattr(changelog_utils.ChangelogUtils, "get_project_root", lambda: str(project_root))
    monkeypatch.setattr(changelog_utils, "_validate_prerequisites", lambda: None)
    monkeypatch.setattr(changelog_utils, "_get_repository_url", lambda: "https://github.com/acgetchell/delaunay")

    # Simulate a failure *after* the changelog was overwritten.
    def fail_cleanup_final_output(file_paths: dict[str, Path]) -> None:
        file_paths["changelog"].write_text("# CORRUPTED\n", encoding="utf-8")
        msg = "boom"
        raise ChangelogError(msg)

    monkeypatch.setattr(changelog_utils, "_run_auto_changelog", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(changelog_utils, "_post_process_dates", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(changelog_utils, "_expand_squashed_commits", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(changelog_utils, "_enhance_with_ai", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(changelog_utils, "_post_process_release_notes", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(changelog_utils, "_cleanup_final_output", fail_cleanup_final_output)

    with pytest.raises(SystemExit) as excinfo:
        changelog_utils._execute_changelog_generation(debug_mode=False)

    assert excinfo.value.code == 1

    # Backups should be restored and then removed.
    assert (project_root / "CHANGELOG.md").read_text(encoding="utf-8") == original
    assert not (project_root / "CHANGELOG.md.backup").exists()
