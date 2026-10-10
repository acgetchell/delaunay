"""Consumer policy integrations with the published research-repo-tools APIs."""

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from research_repo_tools.cli import main
from research_repo_tools.config import load
from research_repo_tools.notebook_testing import isolated_project
from research_repo_tools.paper_dates import read_source_date
from research_repo_tools.sarif import split

import profiling_metadata

if TYPE_CHECKING:
    import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_profile_capture_retains_consumer_labels_and_native_declarations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("PROFILE_METADATA_MODE", "development")
    monkeypatch.setenv("PROFILE_METADATA_FILTER", "construction")
    monkeypatch.delenv("BENCH_FILTER_VALUE", raising=False)
    captured: dict[Path, bytes] = {}
    monkeypatch.setattr(profiling_metadata, "replace_many", captured.update)
    assert profiling_metadata.main([]) == 0
    payload = json.loads(captured[ROOT / "profiling-results/environment.json"])
    assert payload["context"]["mode"] == "development"
    assert payload["context"]["filter"] == "construction"
    assert payload["declarations"]["cargo-manifest"]["text"] == (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    assert payload["declarations"]["rust-toolchain"]["text"] == (ROOT / "rust-toolchain.toml").read_text(encoding="utf-8")
    assert payload["source"]["context"]["release"] == "v0.8.2"


def test_retained_validation_paper_satisfies_consumer_policy() -> None:
    assert read_source_date(ROOT / "papers/validation.tex").raw
    assert (
        main(
            [
                "--root",
                str(ROOT),
                "papers",
                "check",
                "papers/validation.pdf",
                "--require-text",
                "Validation Architecture in delaunay",
                "--require-text",
                "REFERENCES",
                "--forbid-text",
                r"\today",
                "--forbid-text",
                "Manuscript submitted to ACM",
            ]
        )
        == 0
    )


def test_codacy_policy_selects_only_repository_opengrep_rules(tmp_path: Path) -> None:
    source = tmp_path / "codacy.sarif"
    source.write_text(
        json.dumps(
            {
                "version": "2.1.0",
                "runs": [
                    {
                        "tool": {"driver": {"name": "Opengrep", "rules": [{"id": "default.rule"}, {"id": "delaunay.policy"}]}},
                        "results": [
                            {"ruleId": "default.rule", "ruleIndex": 0, "message": {"text": "default"}},
                            {"ruleId": "delaunay.policy", "ruleIndex": 1, "message": {"text": "consumer"}},
                        ],
                        "properties": {"consumer": "retained"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    policy = load(root=ROOT).sarif
    assert policy is not None
    outputs = split(source, tmp_path / "selected", policy)
    assert len(outputs) == 1
    run = json.loads(outputs[0].payload)["runs"][0]
    assert run["results"] == [{"ruleId": "delaunay.policy", "ruleIndex": 0, "message": {"text": "consumer"}}]
    assert run["properties"] == {"consumer": "retained"}
    assert outputs[0].category.startswith("codacy-opengrep-")


def test_managed_release_credentials_are_scoped_to_supported_binary_sync() -> None:
    action = yaml.safe_load((ROOT / ".github/actions/setup-just/action.yml").read_text(encoding="utf-8"))
    steps = action["runs"]["steps"]
    authenticated = [step for step in steps if "GH_TOKEN" in step.get("env", {}) or "GITHUB_TOKEN" in step.get("env", {})]
    assert len(authenticated) == 1
    assert authenticated[0]["run"] == ("uv run --locked --no-sync --no-python-downloads --no-env-file research-repo-tools toolchain sync-binaries")
    assert "prebuilt_tools" not in json.dumps(action)
    assert "uv sync --locked --managed-python --only-group tooling" in json.dumps(steps)


def test_shared_notebook_fixture_borrows_locked_environment_and_preserves_source(tmp_path: Path) -> None:
    paths = (Path("notebooks/00_quickstart.ipynb"),)
    original = (ROOT / paths[0]).read_bytes()
    with isolated_project(ROOT, paths, parent=tmp_path, environment=Path(sys.prefix)) as project:
        notebook = project.root / paths[0]
        notebook.write_text(
            json.dumps(
                {
                    "cells": [
                        {
                            "cell_type": "code",
                            "execution_count": None,
                            "id": "consumer-smoke",
                            "metadata": {},
                            "outputs": [],
                            "source": "value: int = 1\nassert value == 1",
                        }
                    ],
                    "metadata": {},
                    "nbformat": 4,
                    "nbformat_minor": 5,
                }
            ),
            encoding="utf-8",
        )
        result = project.execute(paths[0], timeout=30)
        assert result.returncode == 0
        assert result.report["status"] == "passed"
        assert json.loads(notebook.read_bytes())["cells"][0]["outputs"] == []
    assert (ROOT / paths[0]).read_bytes() == original
