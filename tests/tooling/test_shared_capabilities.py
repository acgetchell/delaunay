"""Consumer policy integrations with the published research-repo-tools APIs."""

import json
import subprocess
import sys
import tomllib
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from research_repo_tools.config import load
from research_repo_tools.notebook_testing import isolated_project
from research_repo_tools.process import run_command
from research_repo_tools.sarif import split

import profiling_metadata

ROOT = Path(__file__).resolve().parents[2]


def normalize_codacy_fixture(tmp_path: Path, document: dict[str, object]) -> Path:
    """Execute the exact compatibility code from the production workflow."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/codacy.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["codacy-quality-scan"]["steps"]
    step = next(step for step in steps if step["name"] == "Normalize Codacy SARIF rule indices")
    script = step["run"].split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    source = tmp_path / "raw.sarif"
    source.write_text(json.dumps(document), encoding="utf-8")
    output = tmp_path / "normalized.sarif"
    run_command(sys.executable, ["-", str(source), str(output)], input=script)
    return output


@pytest.mark.parametrize("suite", tuple(profiling_metadata.DIAGNOSTIC_TARGETS))
def test_profile_capture_retains_consumer_labels_and_native_declarations(monkeypatch: pytest.MonkeyPatch, suite: str) -> None:
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("PROFILE_METADATA_MODE", "development")
    selected_filter = "memory_profiling/memory_usage_2d/1000" if suite == "allocations" else "construction"
    if suite == "traversal":
        selected_filter = "queries/vertices/Δ 🚀"
    monkeypatch.setenv("PROFILE_METADATA_FILTER", selected_filter)
    monkeypatch.setenv("PROFILE_METADATA_SUITE", suite)
    monkeypatch.setenv("PROFILE_METADATA_TITLE", "Profiling 🚀")
    monkeypatch.delenv("BENCH_FILTER_VALUE", raising=False)
    captured: dict[Path, bytes] = {}
    monkeypatch.setattr(profiling_metadata, "replace_many", captured.update)
    assert profiling_metadata.main([]) == 0
    payload = json.loads(captured[ROOT / "profiling-results/environment.json"])
    assert payload["context"]["mode"] == "development"
    assert payload["context"]["title"] == "Profiling 🚀"
    assert payload["context"]["filter"] == selected_filter
    assert payload["declarations"]["cargo-manifest"]["text"] == (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    assert payload["declarations"]["rust-toolchain"]["text"] == (ROOT / "rust-toolchain.toml").read_text(encoding="utf-8")
    version = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))["package"]["version"]
    assert payload["source"]["context"]["release"] == f"v{version}"
    command = json.loads(payload["source"]["context"]["command"])
    assert command[6] == profiling_metadata.DIAGNOSTIC_TARGETS[suite]
    assert command[-1] == selected_filter
    assert ("count-allocations" in command) is (suite == "allocations")
    assert ("--exact" in command) is (suite == "allocations")
    harness = json.loads(payload["source"]["context"]["harness-inventory"])
    assert [name for name in harness if name.startswith("benches/") and not name.startswith("benches/common/")] == [f"benches/{command[6]}.rs"]


@pytest.mark.parametrize(
    "manifest",
    [None, "[package\n", "[package]\nname = 'delaunay'\n", "[package]\nversion = 123\n", "[package]\nversion = ''\n"],
    ids=["missing", "malformed", "missing-version", "invalid-version-type", "empty-version"],
)
def test_profile_metadata_cli_rejects_invalid_manifest_before_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], manifest: str | None
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest_path = tmp_path / "Cargo.toml"
    if manifest is not None:
        manifest_path.write_text(manifest, encoding="utf-8")
    output = tmp_path / "profiling-results/environment.json"
    output.parent.mkdir()
    output.write_bytes(b"previous metadata\n")
    capture = Mock(return_value=b"unexpected metadata\n")
    monkeypatch.setattr(profiling_metadata, "capture_profile", capture)

    assert profiling_metadata.main([]) == 1

    capture.assert_not_called()
    assert output.read_bytes() == b"previous metadata\n"
    assert not (tmp_path / "target").exists()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "delaunay-profile-metadata:" in captured.err
    expected_path = repr(str(manifest_path)) if manifest is None else str(manifest_path)
    assert expected_path in captured.err
    assert "No such file" in captured.err if manifest is None else "package.version" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("stage", ["capture", "publication"])
def test_profile_metadata_cli_reports_failure_and_cleans_temporary_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], stage: str
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Cargo.toml").write_text('[package]\nversion = "0.8.3"\n', encoding="utf-8")
    (tmp_path / "tooling").mkdir()
    (tmp_path / "tooling/profiling-inputs.toml").write_bytes((ROOT / "tooling/profiling-inputs.toml").read_bytes())
    output = tmp_path / "profiling-results/environment.json"
    output.parent.mkdir()
    output.write_bytes(b"previous metadata\n")
    capture = Mock(return_value=b"new metadata\n")
    if stage == "capture":
        capture.side_effect = subprocess.TimeoutExpired(["rustc", "--version"], 10, output=b"probe output\n", stderr=b"probe failure\n")
    publisher = Mock(side_effect=PermissionError(13, "publication denied", str(output)))
    monkeypatch.setattr(profiling_metadata, "capture_profile", capture)
    monkeypatch.setattr(profiling_metadata, "replace_many", publisher)

    assert profiling_metadata.main([]) == 1

    capture.assert_called_once()
    assert output.read_bytes() == b"previous metadata\n"
    assert not list((tmp_path / "target").glob("profiling-context-*"))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "delaunay-profile-metadata:" in captured.err
    assert "Traceback" not in captured.err
    if stage == "capture":
        publisher.assert_not_called()
        assert "rustc --version" in captured.err
        assert "10" in captured.err
        assert "probe output\n" in captured.err
        assert "probe failure\n" in captured.err
    else:
        publisher.assert_called_once_with({output: b"new metadata\n"})
        assert repr(str(output)) in captured.err
        assert "publication denied" in captured.err


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


def test_codacy_workflow_preserves_findings_with_missing_and_category_local_indices(tmp_path: Path) -> None:
    document: dict[str, object] = {
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "Opengrep (reported by Codacy)",
                        "rules": [{"id": "default.rule"}, {"id": "delaunay.security"}, {"id": "delaunay.quality"}],
                    }
                },
                "results": [
                    {"ruleId": "delaunay.local", "ruleIndex": -1, "message": {"text": "no descriptor"}},
                    {"ruleId": "delaunay.quality", "ruleIndex": 0, "message": {"text": "category-local index"}},
                    {"ruleId": "delaunay.security", "ruleIndex": 1, "message": {"text": "security"}},
                    {"ruleId": "default.rule", "ruleIndex": 0, "message": {"text": "default"}},
                ],
                "properties": {"consumer": "retained"},
                "invocations": [{"executionSuccessful": True}],
            }
        ],
    }
    source = normalize_codacy_fixture(tmp_path, document)
    normalized = json.loads(source.read_bytes())["runs"][0]
    assert len(normalized["results"]) == 4
    assert "ruleIndex" not in normalized["results"][0]
    assert normalized["results"][1]["ruleIndex"] == 2
    policy = load(root=ROOT).sarif
    assert policy is not None
    outputs = split(source, tmp_path / "selected", policy)
    assert len(outputs) == 1
    run = json.loads(outputs[0].payload)["runs"][0]
    assert run["results"] == [
        {"ruleId": "delaunay.local", "message": {"text": "no descriptor"}},
        {"ruleId": "delaunay.quality", "ruleIndex": 1, "message": {"text": "category-local index"}},
        {"ruleId": "delaunay.security", "ruleIndex": 0, "message": {"text": "security"}},
    ]
    assert run["tool"]["driver"]["rules"] == [{"id": "delaunay.security"}, {"id": "delaunay.quality"}]
    assert run["properties"] == {"consumer": "retained"}
    assert run["invocations"] == [{"executionSuccessful": True}]


@pytest.mark.parametrize("rule_index", [True, -2, 99])
def test_codacy_workflow_retains_rejection_of_invalid_indices(tmp_path: Path, rule_index: bool | int) -> None:
    source = normalize_codacy_fixture(
        tmp_path,
        {
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {"driver": {"name": "Opengrep (reported by Codacy)", "rules": [{"id": "delaunay.rule"}]}},
                    "results": [{"ruleId": "delaunay.rule", "ruleIndex": rule_index, "message": {"text": "invalid"}}],
                }
            ],
        },
    )
    policy = load(root=ROOT).sarif
    assert policy is not None
    with pytest.raises(ValueError, match="not a valid rule index"):
        split(source, tmp_path / "selected", policy)


def test_managed_release_credentials_are_scoped_to_supported_binary_sync() -> None:
    action = yaml.safe_load((ROOT / ".github/actions/setup-just/action.yml").read_text(encoding="utf-8"))
    steps = action["runs"]["steps"]
    authenticated = [step for step in steps if "GH_TOKEN" in step.get("env", {}) or "GITHUB_TOKEN" in step.get("env", {})]
    assert len(authenticated) == 1
    assert authenticated[0]["run"] == ("uv run --locked --no-sync --no-python-downloads --no-env-file research-repo-tools toolchain sync-binaries")
    assert "prebuilt_tools" not in json.dumps(action)
    assert "uv sync --locked --managed-python --only-group tooling" in json.dumps(steps)


@pytest.mark.filterwarnings("error:Proactor event loop does not implement add_reader family of methods required for zmq:RuntimeWarning")
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
        fixture_source = notebook.read_bytes()
        result = project.execute(paths[0], timeout=30)
        assert result.returncode == 0
        assert result.report["status"] == "passed"
        assert notebook.read_bytes() == fixture_source
        assert json.loads(notebook.read_bytes())["cells"][0]["outputs"] == []
    assert (ROOT / paths[0]).read_bytes() == original
