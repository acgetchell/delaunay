"""Recorded citation dates make prospective changelog generation repeatable."""

from pathlib import Path

import pytest
from research_repo_tools.just_inspect import dry_run

import release_changelog as adapter

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("quotation", ["", "'", '"'])
def test_generation_and_preview_reuse_recorded_date(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quotation: str) -> None:
    citation = tmp_path / "CITATION.cff"
    contents = f"version: 0.9.0\ndate-released: {quotation}2000-02-29{quotation}\n"
    citation.write_text(contents, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    calls: list[list[str]] = []

    def shared_cli(argv: list[str]) -> int:
        calls.append(argv)
        return 0

    monkeypatch.setattr(adapter, "tooling_main", shared_cli)
    for extra in ([], ["--date", ""], ["--dry-run"]):
        assert adapter.main(["--tag", "v0.9.0", *extra]) == 0
        expected = ["changelog", "generate", "--tag", "v0.9.0", "--date", "2000-02-29"]
        assert calls[-1] == [*expected, *(["--dry-run"] if extra == ["--dry-run"] else [])]
    assert citation.read_text(encoding="utf-8") == contents


@pytest.mark.parametrize("arguments", [["--tag", "v0.9.0", "--date", "2026-10-09"], []])
def test_explicit_date_and_untagged_preview_forward_shared_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arguments: list[str]) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[list[str]] = []

    def shared_cli(argv: list[str]) -> int:
        calls.append(argv)
        return 7

    monkeypatch.setattr(adapter, "tooling_main", shared_cli)
    assert adapter.main([*arguments, "--dry-run"]) == 7
    assert calls == [["changelog", "generate", *arguments, "--dry-run"]]
    assert not (tmp_path / "CITATION.cff").exists()


@pytest.mark.parametrize(
    "contents",
    [
        None,
        "[",
        "- item\n",
        "version: 0.9.0\n",
        "date-released: 2000-02-29\n",
        "version: 0.8.0\ndate-released: 2000-02-29\n",
        "version: 0.9.0\ndate-released: 2001-02-29\n",
        "version: 0.9.0\ndate-released: 20000229\n",
        "version: 0.9.0\ndate-released: \n",
        "version: 0.9.0\ndate-released: [2000-02-29]\n",
        "version: 0.9.0\ndate-released: 2000-02-29\ndate-released: 2000-03-01\n",
        "version: 0.9.0\nversion: 0.9.1\ndate-released: 2000-02-29\n",
    ],
)
def test_invalid_metadata_stops_before_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contents: str | None) -> None:
    citation = tmp_path / "CITATION.cff"
    if contents is not None:
        citation.write_text(contents, encoding="utf-8")
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_bytes(b"retained changelog\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(adapter, "tooling_main", lambda _argv: pytest.fail("invalid metadata must not generate anything"))
    assert adapter.main(["--tag", "v0.9.0"]) == 1
    assert changelog.read_bytes() == b"retained changelog\n"
    assert not (tmp_path / "docs").exists()
    if contents is not None:
        assert citation.read_text(encoding="utf-8") == contents


def test_shared_recipe_names_and_optional_arguments() -> None:
    implicit = dry_run(ROOT, "changelog-release", ["v0.9.0"]).stderr
    explicit = dry_run(ROOT, "changelog-release", ["v0.9.0", "2000-02-29"]).stderr
    assert "release_changelog.py --tag 'v0.9.0' --date ''" in implicit
    assert "release_changelog.py --tag 'v0.9.0' --date '2000-02-29'" in explicit
    assert 'release_changelog.py --dry-run "$@"' in dry_run(ROOT, "changelog-preview", ["--tag", "v0.9.0"]).stderr
