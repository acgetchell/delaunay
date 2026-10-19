"""Exercise the consumer's release integration without credentials or uploads."""

import json
import os
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml
from research_repo_tools import changelog, config, release_publishing
from research_repo_tools.cli import main
from research_repo_tools.just_inspect import dry_run, inspect_justfile
from research_repo_tools.process import run_command

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/publish-crates.yml"
POLICY = config.load(root=ROOT)
VERSION = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))["package"]["version"]
TAG = f"v{VERSION}"
SHA = "a" * 40
CHECKS = ("build (ubuntu-latest)", "build (macos-latest)", "build (windows-latest)", "Clippy SARIF Analysis", "Repository Rule SARIF Analysis")


def test_policy_binds_native_checks_assets_and_declared_dates() -> None:
    """Consumer policy retains native science gates and deliberate release dates."""
    assert POLICY.publishing is not None
    assert (POLICY.publishing.registry, POLICY.publishing.package, POLICY.publishing.repository) == ("crates-io", "delaunay", "acgetchell/delaunay")
    assert tuple((check.name, check.app_id) for check in POLICY.publishing.required_checks) == tuple((name, 15368) for name in CHECKS)
    assert POLICY.publishing.required_assets == ("delaunay-v{version}-criterion-baseline.tar.gz",)
    assert POLICY.release.date_policy == "declared"


@pytest.fixture
def release_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Supply read-only provider evidence to the installed shared CLI."""
    event = {
        "action": "published",
        "repository": {"full_name": "acgetchell/delaunay"},
        "release": {"id": 42, "tag_name": TAG, "draft": False, "prerelease": False},
    }
    event_file = tmp_path / "event.json"
    event_file.write_text(json.dumps(event), encoding="utf-8")
    for key, value in {
        "GITHUB_EVENT_NAME": "release",
        "GITHUB_REPOSITORY": "acgetchell/delaunay",
        "GITHUB_SHA": SHA,
        "GITHUB_REF": f"refs/tags/{TAG}",
        "GITHUB_EVENT_PATH": str(event_file),
    }.items():
        monkeypatch.setenv(key, value)

    def git(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert args in (["--no-pager", "rev-parse", "HEAD"], ["--no-pager", "status", "--porcelain", "--untracked-files=all"])
        return subprocess.CompletedProcess(args, 0, stdout=SHA if "rev-parse" in args else "", stderr="")

    checks = [
        {"id": index, "name": name, "head_sha": SHA, "app": {"id": 15368}, "status": "completed", "conclusion": "success"}
        for index, name in enumerate(CHECKS, 1)
    ]
    responses: dict[str, object] = {
        "": {"full_name": "acgetchell/delaunay", "default_branch": "main"},
        "branches/main": {"name": "main", "protected": True},
        f"compare/{SHA}...main": {"status": "identical"},
        f"commits/{SHA}/check-runs?filter=latest&per_page=100": [{"total_count": len(checks), "check_runs": checks}],
    }
    monkeypatch.setattr(release_publishing, "run_git_command", git)
    monkeypatch.setattr(release_publishing, "_api", lambda _root, _repo, endpoint, **_kwargs: responses[endpoint])
    monkeypatch.setattr(release_publishing, "_remote_commit", lambda *_args: SHA)
    release = release_publishing.GitHubRelease(
        repository="acgetchell/delaunay",
        tag=TAG,
        identifier=42,
        draft=False,
        immutable=True,
        prerelease=False,
        assets=((f"delaunay-{TAG}-criterion-baseline.tar.gz", 7, 100, "digest"),),
    )
    monkeypatch.setattr(release_publishing, "lookup_release", lambda *_args: release)
    return {"event": event, "path": event_file, "checks": checks}


def test_installed_gate_accepts_consumer_metadata_and_native_results(release_event: dict[str, Any]) -> None:
    """The pinned shared contract accepts this repository's real metadata."""
    assert main(["--root", str(ROOT), "release", "gate", TAG]) == 0


@pytest.mark.parametrize("name", CHECKS)
@pytest.mark.parametrize("state", ["missing", "failure", "pending"])
def test_each_native_gate_is_required(release_event: dict[str, Any], name: str, state: str) -> None:
    """Every consumer-selected native check must succeed on the release commit."""
    check = next(item for item in release_event["checks"] if item["name"] == name)
    if state == "missing":
        check["name"] = "unrelated check"
    else:
        check.update(status="in_progress" if state == "pending" else "completed", conclusion=None if state == "pending" else "failure")
    assert main(["--root", str(ROOT), "release", "gate", TAG]) == 1


@pytest.mark.parametrize("mismatch", ["repository", "commit", "version", "event", "draft", "prerelease", "asset"])
def test_wrong_release_identity_or_missing_asset_blocks_gate(release_event: dict[str, Any], monkeypatch: pytest.MonkeyPatch, mismatch: str) -> None:
    """Provider mismatches cannot become a release approval."""
    tag = TAG
    if mismatch == "repository":
        monkeypatch.setenv("GITHUB_REPOSITORY", "other/repository")
    elif mismatch == "commit":
        monkeypatch.setattr(release_publishing, "_remote_commit", lambda *_args: "b" * 40)
    elif mismatch == "version":
        tag = "v99.0.0"
    elif mismatch == "event":
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    elif mismatch == "asset":
        monkeypatch.setattr(
            release_publishing,
            "lookup_release",
            lambda *_args: release_publishing.GitHubRelease("acgetchell/delaunay", TAG, 42, draft=False, immutable=True, prerelease=False, assets=()),
        )
    else:
        release_event["event"]["release"][mismatch] = True
        release_event["path"].write_text(json.dumps(release_event["event"]), encoding="utf-8")
    assert main(["--root", str(ROOT), "release", "gate", tag]) == 1


@pytest.mark.parametrize("immutable", [True, False, None])
def test_real_recipe_requires_provider_immutable_state(tmp_path: Path, immutable: bool | None) -> None:
    """Run the consumer's actual jq gate, with only provider/CLI boundaries stubbed."""
    payload = tmp_path / "event.json"
    payload.write_text(json.dumps({"release": {"id": 42}}), encoding="utf-8")
    # Notes have their own real-shell tests; isolate the final event/immutable
    # script from dependency expansion here.
    script = "#!/usr/bin/env bash" + dry_run(ROOT, "release-gate", [TAG]).stderr.rsplit("#!/usr/bin/env bash", 1)[1]
    provider = (
        f'gh() {{ [[ "$1 $2" == "api repos/acgetchell/delaunay/releases/42" ]] || return 90; echo {shlex.quote(json.dumps({"immutable": immutable}))}; }}\n'
    )
    result = run_command(
        "bash",
        ["--noprofile", "--norc", "-euc", "set -o pipefail\nuv() { return 0; }\n" + provider + script],
        env={**os.environ, "GITHUB_EVENT_PATH": payload.as_posix()},
        check=False,
    )
    assert (result.returncode == 0) == (immutable is True), result.stderr


@pytest.mark.parametrize("mismatch", ["none", "title", "notes"])
def test_notes_precheck_uses_generated_body_and_title(tmp_path: Path, mismatch: str) -> None:
    """Keep Markdown bytes intact through the real generated-notes comparison."""
    response = {"tagName": TAG, "name": "wrong" if mismatch == "title" else TAG, "body": "wrong" if mismatch == "notes" else changelog.notes(POLICY, TAG)[0]}
    payload = tmp_path / "release.json"
    payload.write_text(json.dumps(response), encoding="utf-8")
    runner = (
        'uv() { if [[ "$*" == "run --locked --no-sync --no-python-downloads research-repo-tools --version" ]]; then return 0; fi; '
        '[[ "$1 $2 $3 $4 $5" == "run --locked --group dev research-repo-tools" ]] || return 90; shift 5; '
        f'{shlex.quote(sys.executable)} -m research_repo_tools --root {shlex.quote(str(ROOT))} "$@"; }}\n'
        f"gh() {{ cat {shlex.quote(payload.as_posix())}; }}\n"
    )
    script = dry_run(ROOT, "_release-notes-check", [TAG]).stderr
    result = run_command("bash", ["--noprofile", "--norc", "-euc", runner + script], cwd=ROOT, check=False)
    assert (result.returncode == 0) == (mismatch == "none"), result.stderr


def test_publish_recipe_reaches_installed_explicit_approval_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace only irreversible publication while exercising the real parser."""
    requested: list[str] = []

    def publish(settings: config.Config, tag: str) -> None:
        assert settings.root == ROOT
        requested.append(tag)

    monkeypatch.setattr(release_publishing, "publish_reviewed_release", publish)
    command = shlex.split(dry_run(ROOT, "release-publish", [TAG]).stderr.splitlines()[-1])
    assert command[:6] == ["uv", "run", "--locked", "--group", "dev", "research-repo-tools"]
    assert main(["--root", str(ROOT), *command[6:]]) == 0
    assert requested == [TAG]


def test_workflow_limits_credentials_and_verifies_clean_packages() -> None:
    """Credential-free builds precede approval, revalidation and a serialized upload."""
    workflow = yaml.safe_load(WORKFLOW.read_bytes())
    assert workflow[True] == {"release": {"types": ["published"]}}
    assert workflow["concurrency"] == {"group": "publish-crates-delaunay", "cancel-in-progress": False}
    jobs = workflow["jobs"]
    assert jobs["publish"]["needs"] == "verify"
    assert jobs["publish"]["environment"] == "crates-io"
    assert jobs["verify"]["permissions"] == {"contents": "read", "checks": "read"}
    assert jobs["publish"]["permissions"] == {"contents": "read", "checks": "read", "id-token": "write"}
    assert "github.repository == 'acgetchell/delaunay'" in jobs["verify"]["if"]
    assert jobs["verify"]["steps"][-1]["run"] == "just release-package"
    assert "CARGO_REGISTRY_TOKEN" not in yaml.safe_dump({"env": workflow["env"], "verify": jobs["verify"]})
    for job in jobs.values():
        assert job["steps"][0]["with"] == {"ref": "${{ github.sha }}", "fetch-depth": 0, "persist-credentials": False}
        assert job["steps"][1]["with"]["cache"] == "false"
    steps = jobs["publish"]["steps"]
    assert 'just release-gate "$RELEASE_TAG"' in steps[2]["run"]
    assert 'release registry "$RELEASE_TAG" --expect absent' in steps[2]["run"]
    assert steps[3]["id"] == "auth"
    assert steps[4]["run"].endswith("cargo publish --locked --package delaunay --registry crates-io --no-verify")
    assert [index for index, step in enumerate(steps) if "CARGO_REGISTRY_TOKEN" in step.get("env", {})] == [4]
    assert steps[5]["if"] == "${{ !cancelled() && steps.auth.outcome == 'success' }}"
    assert steps[5]["run"] == 'just release-verify "$RELEASE_TAG"'
    assert "continue-on-error" not in WORKFLOW.read_text(encoding="utf-8")
    commands = dry_run(ROOT, "release-package").stderr
    assert commands.count("--package delaunay --registry crates-io --dry-run") == 2
    assert "--all-features" in commands
    assert "--allow-dirty" not in commands
    assert "--no-verify" not in commands
    gate_dependencies = {item["recipe"] for item in inspect_justfile(ROOT).recipes["release-gate"]["dependencies"]}
    assert "_release-notes-check" in gate_dependencies


def test_account_payloads_match_workflow_identity_and_require_tag_review() -> None:
    """Settings payloads must authorize only the pinned action and reviewed tags."""
    settings = ROOT / ".github/settings"
    environment = json.loads((settings / "crates-io-environment.json").read_bytes())
    assert environment["reviewers"] == [{"type": "User", "id": 434849}]
    assert environment["can_admins_bypass"] is False
    assert environment["deployment_branch_policy"] == {"protected_branches": False, "custom_branch_policies": True}
    assert json.loads((settings / "crates-io-tags.json").read_bytes()) == {"name": "v*", "type": "tag"}
    workflow = yaml.safe_load(WORKFLOW.read_bytes())
    auth = next(step["uses"] for step in workflow["jobs"]["publish"]["steps"] if step.get("id") == "auth")
    allowed = json.loads((settings / "actions-selected.json").read_bytes())
    assert [pattern for pattern in allowed["patterns_allowed"] if pattern.startswith("rust-lang/crates-io-auth-action@")] == [auth]
    assert allowed["verified_allowed"] is False
    auto_merge = (ROOT / ".github/workflows/dependabot-auto-merge.yml").read_text(encoding="utf-8")
    assert '".github/workflows/publish-crates.yml"' not in auto_merge


@pytest.mark.parametrize(("failure", "expected_exit"), [("none", 0), ("evidence", 1), ("present", 1), ("unavailable", 2)])
def test_preupload_shell_requires_evidence_and_proven_absence(tmp_path: Path, failure: str, expected_exit: int) -> None:
    """An existing version or unavailable registry blocks the actual upload sequence."""
    workflow = yaml.safe_load(WORKFLOW.read_bytes())
    script = workflow["jobs"]["publish"]["steps"][2]["run"]
    marker = tmp_path / "upload-reached"
    gate = "just() { return 1; }" if failure == "evidence" else "just() { return 0; }"
    registry_exit = expected_exit if failure in {"present", "unavailable"} else 0
    runner = f"{gate}\nuv() {{ return {registry_exit}; }}\n"
    result = run_command(
        "bash",
        ["--noprofile", "--norc", "-euc", runner + script + f"\ntouch {shlex.quote(marker.as_posix())}"],
        env={**os.environ, "RELEASE_TAG": TAG},
        check=False,
    )
    assert result.returncode == expected_exit, result.stderr
    assert marker.exists() == (failure == "none")


def test_packaging_failure_stops_the_recipe(tmp_path: Path) -> None:
    """Dependency or native Cargo failure prevents subsequent package verification."""
    calls = tmp_path / "calls"
    runner = f"uv() {{ echo invoked >> {shlex.quote(calls.as_posix())}; return 17; }}\n"
    result = run_command("bash", ["--noprofile", "--norc", "-euc", runner + dry_run(ROOT, "release-package").stderr], check=False)
    assert result.returncode == 17
    assert calls.read_text(encoding="utf-8").splitlines() == ["invoked"]
