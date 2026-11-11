"""Execute isolated workflow handoff checks with only provider transport modeled."""

import json
import runpy
from pathlib import Path
from typing import Any

import pytest
import yaml
from research_repo_tools import release_drafts, release_validation
from research_repo_tools.release_assets import ReleaseTarget

ROOT = Path(__file__).resolve().parents[2]
SHA = "a" * 40
REPOSITORY = "acgetchell/delaunay"
TAG = "v0.8.5"


@pytest.fixture
def handoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Model provider records and downloads, keeping the workflow's checks intact."""
    monkeypatch.chdir(tmp_path)
    for key, value in {
        "GITHUB_REPOSITORY": REPOSITORY,
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_SHA": SHA,
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": f"refs/tags/{TAG}",
        "RELEASE_TAG": TAG,
        "RELEASE_COMMIT": SHA,
        "VALIDATED_ATTEMPT": "1",
        "HANDOFF_ID": "456",
        "HANDOFF_SHA256": "b" * 64,
        "ARTIFACT_ID": "456",
        "ARTIFACT_DIGEST": "b" * 64,
        "PACKAGE_NAME": "delaunay-0.8.5.crate",
    }.items():
        monkeypatch.setenv(key, value)
    run = {
        "id": 123,
        "workflow_id": 7,
        "run_attempt": 1,
        "event": "push",
        "head_branch": TAG,
        "head_sha": SHA,
        "path": ".github/workflows/prepare-release.yml",
    }
    artifact = {
        "id": 456,
        "expired": False,
        "name": "draft-handoff-123-1",
        "digest": "sha256:" + "b" * 64,
        "workflow_run": {"id": 123, "head_sha": SHA},
    }
    state: dict[str, Any] = {"run": run, "artifact": artifact, "history": [run], "calls": [], "rerun": False}

    def api(endpoint: str, *, repository: str) -> dict[str, Any]:
        assert repository == REPOSITORY
        if endpoint == "actions/runs/123":
            return state["run"].copy()
        assert endpoint == "actions/artifacts/456"
        return state["artifact"]

    def download(repository: str, observed: dict[str, Any], directory: Path, names: tuple[str, ...]) -> None:
        assert repository == REPOSITORY
        assert observed == artifact
        state["calls"].append("download")
        directory.mkdir()
        for name in names:
            content = (ROOT / "tooling/release-evidence.toml").read_bytes() if name == "evidence.toml" else b"original inert bytes"
            (directory / name).write_bytes(content)
        if state["rerun"]:
            state["run"]["run_attempt"] = 2

    monkeypatch.setattr(release_validation, "api", api)
    monkeypatch.setattr(release_validation, "read_pages", lambda *_args: state["history"])
    monkeypatch.setattr(release_validation, "download_artifact", download)

    def prepare(_root: Path, repository: str, tag: str, commit: str, **kwargs: object) -> ReleaseTarget:
        assert (repository, tag, commit) == (REPOSITORY, TAG, SHA)
        assert kwargs["expected"] == b"original inert bytes"
        assert kwargs["notes"] == b"original inert bytes"
        assert kwargs["approve"] is True
        state["calls"].append("prepare")
        return ReleaseTarget(repository, tag, 42, commit, tag)

    monkeypatch.setattr(release_drafts, "prepare_release_draft", prepare)
    return state


@pytest.mark.parametrize("failure", ["none", "attempt", "digest", "artifact-id", "commit", "history", "rerun"])
def test_original_draft_writer_authenticates_handoff(handoff: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    """Only the original tag push and immutable producer handoff may prepare a draft."""
    if failure == "attempt":
        monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    elif failure in {"digest", "artifact-id"}:
        handoff["artifact"]["digest" if failure == "digest" else "id"] = "replacement"
    elif failure == "commit":
        handoff["run"]["head_sha"] = "c" * 40
    elif failure == "history":
        handoff["history"].append(handoff["run"] | {"id": 122})
    elif failure == "rerun":
        handoff["rerun"] = True
    workflow = yaml.safe_load((ROOT / ".github/workflows/prepare-release.yml").read_bytes())
    step = next(step for step in workflow["jobs"]["prepare"]["steps"] if "run" in step)
    script = tmp_path / "writer.py"
    script.write_text(step["run"], encoding="utf-8")
    if failure == "none":
        runpy.run_path(str(script), run_name="__main__")
        assert handoff["calls"] == ["download", "prepare"]
        assert json.loads((tmp_path / "target.json").read_bytes())["release_id"] == 42
    else:
        with pytest.raises(ValueError, match=r"original exact tag push|identity differs|producer attempt|prior or unproven|producer changed"):
            runpy.run_path(str(script), run_name="__main__")
        assert "prepare" not in handoff["calls"]
        assert not (tmp_path / "target.json").exists()


@pytest.mark.parametrize("failure", ["none", "attempt", "digest", "commit", "event", "rerun"])
def test_cargo_signer_authenticates_original_producer(handoff: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    """The signer downloads only the exact current main producer's inert bytes."""
    handoff["artifact"]["name"] = "cargo-producer-123-1"
    handoff["run"].update(head_branch="main", path=".github/workflows/ci.yml")
    if failure == "attempt":
        monkeypatch.setenv("VALIDATED_ATTEMPT", "2")
    elif failure == "digest":
        handoff["artifact"]["digest"] = "sha256:" + "c" * 64
    elif failure == "commit":
        handoff["artifact"]["workflow_run"]["head_sha"] = "c" * 40
    elif failure == "event":
        handoff["run"]["event"] = "pull_request"
    elif failure == "rerun":
        handoff["rerun"] = True
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_bytes())
    step = next(step for step in workflow["jobs"]["sign-package"]["steps"] if step.get("name") == "Authenticate the original package handoff")
    script = tmp_path / "signer.py"
    script.write_text(step["run"], encoding="utf-8")
    if failure == "none":
        runpy.run_path(str(script), run_name="__main__")
        assert sorted(path.name for path in (tmp_path / "cargo-promotion").iterdir()) == ["delaunay-0.8.5.crate", "validation.json"]
    else:
        with pytest.raises(ValueError, match=r"different attempt|handoff identity changed|producer run changed|attempt changed"):
            runpy.run_path(str(script), run_name="__main__")
    if failure not in {"none", "rerun"}:
        assert not handoff["calls"]
