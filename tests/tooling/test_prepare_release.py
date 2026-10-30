"""Exercise draft orchestration with real installed lifecycle checks and fake providers."""

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from research_repo_tools import config, release_assets, release_interfaces
from research_repo_tools.cli import main as shared_main

import prepare_release

ROOT = Path(__file__).resolve().parents[2]
TAG = "v0.8.3"
COMMIT = "a" * 40
NOTES = "Release notes\n"
REPOSITORY = "acgetchell/delaunay"


@pytest.fixture
def provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Model transport/evidence while retaining the installed release-target parser."""
    state: dict[str, Any] = {"releases": [], "writes": [], "evidence": [], "protected": True, "commit": COMMIT, "checks": "success"}
    for key, value in {"GITHUB_REPOSITORY": REPOSITORY, "GITHUB_EVENT_NAME": "push", "GITHUB_REF": f"refs/tags/{TAG}", "GITHUB_SHA": COMMIT}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(prepare_release, "resolve_revision", lambda *_args: COMMIT)
    monkeypatch.setattr(prepare_release, "require_release_checkout", lambda *_args: None)
    monkeypatch.setattr(release_assets, "require_release_checkout", lambda *_args: None)
    settings = config.load(root=ROOT)
    assert settings.publishing is not None
    required_checks = settings.publishing.required_checks
    monkeypatch.setattr(prepare_release.config, "load", lambda *_args, **_kwargs: settings)
    monkeypatch.setattr(prepare_release, "load_evidence_policy", lambda _path: object())
    monkeypatch.setattr(prepare_release, "check_metadata", lambda *_args: None)
    monkeypatch.setattr(prepare_release.changelog, "notes", lambda *_args: (NOTES, tmp_path / "CHANGELOG.md", TAG))

    def evidence(*_args: object, **kwargs: object) -> bytes:
        state["evidence"].append(kwargs)
        if state.get("bad-evidence"):
            msg = "missing retained exact-commit evidence"
            raise ValueError(msg)
        return b'{"frozen":true}\n'

    monkeypatch.setattr(prepare_release, "verify_release_evidence", evidence)

    def read(endpoint: str, **_kwargs: object) -> Any:
        if endpoint == f"commits/{COMMIT}/check-runs?filter=latest&per_page=100":
            checks = [
                {"id": index, "name": item.name, "head_sha": COMMIT, "app": {"id": item.app_id}, "status": "completed", "conclusion": state["checks"]}
                for index, item in enumerate(required_checks, 1)
            ]
            return [{"total_count": len(checks), "check_runs": checks}]
        routes = {
            "": lambda: {"default_branch": "main"},
            "branches/main": lambda: {"protected": state["protected"]},
            f"compare/{COMMIT}...main": lambda: {"status": "identical"},
            "releases?per_page=100": lambda: state.get("inventory", [state["releases"]]),
            "releases/42": lambda: state["releases"][0],
            "releases/42/assets?per_page=100": lambda: [state["releases"][0]["assets"]],
            f"git/ref/tags/{TAG}": lambda: {"ref": f"refs/tags/{TAG}", "object": {"type": "commit", "sha": state["commit"]}},
        }
        return routes[endpoint]()

    def command(executable: str, args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if executable == "git":
            result = f"{state['commit']}\trefs/tags/{TAG}^{{}}\n"
        elif args[:2] == ["release", "create"]:
            state["writes"].append(args)
            assert Path(args[args.index("--notes-file") + 1]).read_text(encoding="utf-8") == NOTES
            state["releases"].append(draft())
            result = ""
        else:
            result = json.dumps(read(args[-1].removeprefix(f"repos/{REPOSITORY}/")))
        return subprocess.CompletedProcess([executable, *args], 0, stdout=result, stderr="")

    def binary_command(executable: str, args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        result = command(executable, args, **kwargs)
        return subprocess.CompletedProcess(result.args, result.returncode, result.stdout.encode(), b"")

    monkeypatch.setattr(prepare_release, "api", read)
    monkeypatch.setattr(prepare_release, "run_command", command)
    monkeypatch.setattr(release_assets, "run_command_bytes", binary_command)
    return state


def draft() -> dict[str, Any]:
    """Provider response including an independently retained benchmark attachment."""
    return {
        "id": 42,
        "url": f"https://api.github.com/repos/{REPOSITORY}/releases/42",
        "tag_name": TAG,
        "name": TAG,
        "body": NOTES,
        "draft": True,
        "immutable": False,
        "prerelease": False,
        "assets": [{"id": 7, "name": "benchmark.tar.gz", "size": 100, "state": "uploaded"}],
    }


def test_successful_preparation_and_safe_retry(tmp_path: Path, provider: dict[str, Any]) -> None:
    assert prepare_release.prepare(tmp_path, TAG) == 42
    assert len(provider["writes"]) == 1
    arguments = provider["writes"][0]
    assert arguments[:3] == ["release", "create", TAG]
    assert arguments[arguments.index("--target") + 1] == COMMIT
    assert "--draft" in arguments
    assert "--verify-tag" in arguments
    before = json.dumps(provider["releases"])
    assert prepare_release.prepare(tmp_path, TAG) == 42
    assert len(provider["writes"]) == 1
    assert json.dumps(provider["releases"]) == before
    assert provider["evidence"][1] == {"expected": b'{"frozen":true}\n'}


def test_declared_asset_cli_captures_the_draft_identity(tmp_path: Path, provider: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the installed CLI and real consumer asset declaration without uploads."""
    provider["releases"] = [draft()]
    monkeypatch.setattr(release_interfaces, "resolve_revision", lambda *_args: COMMIT)
    monkeypatch.setattr(release_interfaces, "require_release_checkout", lambda *_args: None)
    captured = tmp_path / "captured.json"
    arguments = [
        "--root",
        str(ROOT),
        "release",
        "assets",
        "preflight",
        "tooling/release-assets.toml",
        "--checkout",
        str(tmp_path),
        "--commit",
        COMMIT,
        "--tag",
        TAG,
        "--output",
        str(captured),
    ]
    assert shared_main(arguments) == 0
    target = release_assets.parse_release_target(captured.read_bytes())
    assert (target.repository, target.release_id, target.tag, target.commit, target.expected_title) == (REPOSITORY, 42, TAG, COMMIT, TAG)
    assert not provider["writes"]


@pytest.mark.parametrize(
    "failure", ["evidence", "checks", "protected", "commit", "inventory", "notes", "published", "immutable", "prerelease", "title", "ambiguous"]
)
def test_failures_never_create_or_modify_releases(tmp_path: Path, provider: dict[str, Any], failure: str) -> None:
    if failure == "evidence":
        provider["bad-evidence"] = True
    elif failure == "checks":
        provider["checks"] = "failure"
    elif failure in {"protected", "commit"}:
        provider[failure] = False if failure == "protected" else "b" * 40
    elif failure == "inventory":
        provider["inventory"] = []
    else:
        release = draft()
        if failure == "notes":
            release["body"] = "conflicting notes"
        elif failure == "published":
            release["draft"] = False
        elif failure == "title":
            release["name"] = "conflicting title"
        elif failure != "ambiguous":
            release[failure] = True
        provider["releases"] = [release, release.copy()] if failure == "ambiguous" else [release]
    before = json.dumps(provider["releases"])
    with pytest.raises(ValueError, match=r"evidence|checks|protected|tag|notes|draft|mutable|title|ambiguous|published|prerelease|inventory"):
        prepare_release.prepare(tmp_path, TAG)
    assert not provider["writes"]
    assert json.dumps(provider["releases"]) == before


@pytest.mark.parametrize("variable", ["GITHUB_REPOSITORY", "GITHUB_EVENT_NAME", "GITHUB_REF", "GITHUB_SHA"])
def test_tag_context_is_required(tmp_path: Path, provider: dict[str, Any], monkeypatch: pytest.MonkeyPatch, variable: str) -> None:
    monkeypatch.setenv(variable, "wrong")
    with pytest.raises(ValueError, match="tag-push"):
        prepare_release.prepare(tmp_path, TAG)
    assert not provider["writes"]


def test_preparation_workflow_has_no_registry_or_publication_authority() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/prepare-release.yml").read_bytes())
    assert workflow[True] == {"push": {"tags": ["v*"]}}
    assert workflow["concurrency"]["cancel-in-progress"] is False
    job = workflow["jobs"]["prepare"]
    assert job["timeout-minutes"] == 30
    assert job["permissions"] == {"contents": "write", "actions": "read", "checks": "read"}
    assert "environment" not in job
    steps = job["steps"]
    assert "Require protected main" in steps[1]["name"]
    assert "merge-base --is-ancestor" in steps[1]["run"]
    assert "prepare_release.py" in steps[-1]["run"]
    bootstrap = next(step for step in steps if step.get("name") == "Install locked draft tooling")
    assert "--no-install-project" in bootstrap["run"]
    assert "GH_TOKEN" not in bootstrap.get("env", {})
    assert "uv sync" not in steps[-1]["run"]
