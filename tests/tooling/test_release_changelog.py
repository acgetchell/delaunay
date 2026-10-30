"""Exercise the installed citation-date default and thin consumer recipes."""

from pathlib import Path

import pytest
from research_repo_tools import changelog, config
from research_repo_tools.just_inspect import dry_run

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("quotation", ["", "'", '"'])
def test_generation_uses_installed_citation_date(tmp_path: Path, quotation: str) -> None:
    """The actual consumer configuration selects the shared citation contract."""
    citation = tmp_path / "CITATION.cff"
    citation.write_text(f"version: 0.9.0\ndate-released: {quotation}2000-02-29{quotation}\n", encoding="utf-8")
    settings = config.load(path=ROOT / "pyproject.toml", root=tmp_path)
    assert changelog.resolve_date(settings, tag="v0.9.0") == "2000-02-29"
    assert changelog.resolve_date(settings, tag="v0.9.0", released="2026-10-09") == "2026-10-09"
    assert changelog.resolve_date(settings) is None


@pytest.mark.parametrize(
    "contents",
    [
        "version: 0.8.0\ndate-released: 2000-02-29\n",
        "version: 0.9.0\ndate-released: 2001-02-29\n",
        "version: 0.9.0\n",
        "version: 0.9.0\ndate-released: 20000229\n",
        "version: 0.9.0\ndate-released: 2000-02-29\ndate-released: 2000-03-01\n",
    ],
)
def test_invalid_prepared_metadata_is_rejected(tmp_path: Path, contents: str) -> None:
    """Published parsing retains fail-closed version/date/duplicate handling."""
    (tmp_path / "CITATION.cff").write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=r"citation|calendar"):
        changelog.resolve_date(config.load(path=ROOT / "pyproject.toml", root=tmp_path), tag="v0.9.0")


def test_release_recipes_use_the_shared_generator() -> None:
    """Optional dates and previews no longer reach a local changelog adapter."""
    implicit = dry_run(ROOT, "changelog-release", ["v0.9.0"]).stderr
    assert "research-repo-tools changelog generate" in implicit
    assert "release_changelog" not in implicit
    assert 'research-repo-tools changelog generate --dry-run "$@"' in dry_run(ROOT, "changelog-preview", ["--tag", "v0.9.0"]).stderr
