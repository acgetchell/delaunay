"""Execute isolated workflow handoff checks with only provider transport modeled."""

import base64
import hashlib
import json
import runpy
from pathlib import Path
from typing import Any

import pytest
import yaml
from research_repo_tools import draft_workflow, release_receipts, release_validation
from research_repo_tools.evidence import deterministic_json
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
        "name": "Prepare GitHub release",
        "status": "in_progress",
        "pull_requests": [],
        "repository": {"id": 17, "full_name": REPOSITORY},
        "head_repository": {"id": 17, "full_name": REPOSITORY},
    }
    artifact = {
        "id": 456,
        "expired": False,
        "name": "draft-handoff-123-1",
        "digest": "sha256:" + "b" * 64,
        "size_in_bytes": 100,
        "workflow_run": {"id": 123, "head_sha": SHA, "head_branch": TAG, "repository_id": 17, "head_repository_id": 17},
    }
    state: dict[str, Any] = {"run": run, "artifact": artifact, "history": [run], "calls": [], "rerun": False}

    payload = (ROOT / run["path"]).read_bytes()
    blob = b"blob " + str(len(payload)).encode() + b"\0" + payload
    responses = {
        "actions/runs/123": run,
        "actions/artifacts/456": artifact,
        "actions/workflows/prepare-release.yml": {"id": 7, "name": run["name"], "path": run["path"], "state": "active"},
        f"contents/{run['path']}?ref={SHA}": {
            "type": "file",
            "path": run["path"],
            "encoding": "base64",
            "content": base64.b64encode(payload).decode(),
            "size": len(payload),
            "sha": hashlib.sha1(blob, usedforsecurity=False).hexdigest(),
        },
    }
    jobs = [
        {
            "id": 99,
            "name": "verify",
            "run_id": 123,
            "run_attempt": 1,
            "head_sha": SHA,
            "head_branch": TAG,
            "labels": ["ubuntu-latest"],
            "status": "completed",
            "conclusion": "success",
            "steps": [
                {"name": name, "status": "completed", "conclusion": "success"}
                for name in ("Verify tagged source and freeze native evidence and notes", "Retain inert verified handoff")
            ],
        }
    ]

    def api(endpoint: str, *, repository: str) -> dict[str, Any]:
        assert repository == REPOSITORY
        return responses[endpoint].copy()

    def pages(_read: object, endpoint: str, field: str) -> list[dict[str, Any]]:
        return {("actions/workflows/7/runs", "workflow_runs"): state["history"], ("actions/runs/123/attempts/1/jobs", "jobs"): jobs}[endpoint, field]

    planned = deterministic_json(
        {
            "schema": "research-repo-tools/draft-handoff/v1",
            "repository": REPOSITORY,
            "source": {"ref": f"refs/tags/{TAG}", "commit": SHA},
            "event": "push",
            "title": TAG,
            "notes": "original notes",
            "evidence_policy": (ROOT / "tooling/release-evidence.toml").read_text(encoding="utf-8"),
            "selection": {"original": "native selection"},
            "checks": [{"name": "portable", "app_id": 15368}],
        }
    )

    def download(repository: str, observed: dict[str, Any], directory: Path, names: tuple[str, ...], **_kwargs: object) -> None:
        assert repository == REPOSITORY
        assert observed["id"] == artifact["id"]
        state["calls"].append("download")
        directory.mkdir()
        for name in names:
            (directory / name).write_bytes(planned if name == "handoff.json" else b"original inert bytes")
        if state["rerun"]:
            state["run"]["run_attempt"] = 2

    monkeypatch.setattr(release_validation, "api", api)
    monkeypatch.setattr(release_validation, "download_artifact", download)
    monkeypatch.setattr(draft_workflow, "api", api)
    monkeypatch.setattr(draft_workflow, "read_pages", pages)
    monkeypatch.setattr(release_receipts, "read_pages", pages)
    monkeypatch.setattr(draft_workflow, "download_artifact", download)
    monkeypatch.setattr(draft_workflow, "_require_draft", lambda *_args: None)

    def prepare(_root: Path, repository: str, tag: str, commit: str, **kwargs: Any) -> ReleaseTarget:
        assert (repository, tag, commit) == (REPOSITORY, TAG, SHA)
        assert json.loads(kwargs["expected"]) == {"original": "native selection"}
        assert kwargs["notes"] == b"original notes"
        assert kwargs["approve"] is True
        kwargs["verify_inputs"]()
        kwargs["reserve_receipt"](b"original pending journal")
        state["calls"].append("prepare")
        return ReleaseTarget(repository, tag, 42, commit, tag)

    monkeypatch.setattr(draft_workflow, "_prepare_release_draft", prepare)
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
        output = tmp_path / "draft-state"
        assert json.loads((output / "target.json").read_bytes())["release_id"] == 42
        assert (output / "draft-receipt.json").read_bytes() == b"original pending journal"
        assert (output / "producer.json").is_file()
        assert json.loads((output / "handoff.json").read_bytes())["selection"] == {"original": "native selection"}
    else:
        with pytest.raises(ValueError, match=r"original exact tag push|original tag-push run|artifact ID|authoritative SHA-256"):
            runpy.run_path(str(script), run_name="__main__")
        assert "prepare" not in handoff["calls"]
        assert not (tmp_path / "draft-state").exists()


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
