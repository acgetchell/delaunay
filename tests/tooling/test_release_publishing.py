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


@pytest.mark.parametrize("failure", [False, True])
def test_real_publish_recipe_requires_retained_evidence_before_publication(tmp_path: Path, *, failure: bool) -> None:
    """A missing main artifact must leave the GitHub draft unpublished."""
    (tmp_path / "justfile").write_text(f"import '{(ROOT / 'justfile').as_posix()}'\n", encoding="utf-8")
    calls = tmp_path / "calls"
    provider = json.dumps({"tagName": TAG, "name": TAG, "body": "generated notes"})
    runner = (
        "uv() {\n"
        '  case "$*" in\n'
        '    *"changelog notes "*) printf "%s\\n" "generated notes" ;;\n'
        '    *"release_artifacts.py evidence "*)\n'
        f'      printf "%s\\n" evidence >> {shlex.quote(calls.as_posix())}\n'
        f"      return {17 if failure else 0} ;;\n"
        '    *"release publish "*)\n'
        f'      printf "%s\\n" publish >> {shlex.quote(calls.as_posix())} ;;\n'
        "    *) return 0 ;;\n"
        "  esac\n"
        "}\n"
        f"gh() {{ printf '%s\\n' {shlex.quote(provider)}; }}\n"
        "export -f uv gh\n"
    )
    result = run_command("bash", ["--noprofile", "--norc", "-euc", runner + f"just release-publish {shlex.quote(TAG)}"], cwd=tmp_path, check=False)
    assert (result.returncode == 0) is not failure, result.stderr
    assert calls.read_text(encoding="utf-8").splitlines() == (["evidence"] if failure else ["evidence", "publish"])


def test_workflow_limits_credentials_and_verifies_clean_packages() -> None:
    """Credential-free builds precede approval, revalidation and a serialized upload."""
    workflow = yaml.safe_load(WORKFLOW.read_bytes())
    assert workflow[True] == {"release": {"types": ["published"]}}
    assert workflow["concurrency"] == {"group": "publish-crates-delaunay", "cancel-in-progress": False}
    jobs = workflow["jobs"]
    assert jobs["publish"]["needs"] == "verify"
    assert jobs["publish"]["environment"] == "crates-io"
    assert jobs["verify"]["permissions"] == {"contents": "read", "checks": "read", "actions": "read"}
    assert jobs["publish"]["permissions"] == {"contents": "read", "checks": "read", "actions": "read", "id-token": "write"}
    assert "github.repository == 'acgetchell/delaunay'" in jobs["verify"]["if"]
    assert jobs["verify"]["steps"][-1]["run"] == "just release-artifact verify"
    assert "CARGO_REGISTRY_TOKEN" not in yaml.safe_dump({"env": workflow["env"], "verify": jobs["verify"]})
    for job in jobs.values():
        assert job["steps"][0]["with"] == {"ref": "${{ github.sha }}", "fetch-depth": 0, "persist-credentials": False}
    assert jobs["verify"]["steps"][1]["with"]["cache"] == "false"
    steps = jobs["publish"]["steps"]
    gate = next(step for step in steps if step.get("name", "").startswith("Recheck evidence"))
    auth = next(step for step in steps if step.get("id") == "auth")
    upload = next(step for step in steps if "CARGO_REGISTRY_TOKEN" in step.get("env", {}))
    visibility = steps[-1]
    assert steps.index(gate) < steps.index(auth) < steps.index(upload) < steps.index(visibility)
    assert 'release-gate "$RELEASE_TAG"' in gate["run"]
    assert 'release registry "$RELEASE_TAG" --expect absent' in gate["run"]
    assert upload["run"] == "cargo publish --locked --package delaunay --registry crates-io --no-verify"
    assert [step for step in steps if "CARGO_REGISTRY_TOKEN" in step.get("env", {})] == [upload]
    assert visibility["if"] == "${{ !cancelled() && steps.auth.outcome == 'success' }}"
    assert visibility["run"].endswith('release-verify "$RELEASE_TAG"')
    assert "continue-on-error" not in WORKFLOW.read_text(encoding="utf-8")
    commands = dry_run(ROOT, "release-package").stderr
    assert commands.count("--package delaunay --registry crates-io --dry-run") == 2
    assert "--all-features" in commands
    assert "--allow-dirty" not in commands
    assert "--no-verify" not in commands
    gate_dependencies = {item["recipe"] for item in inspect_justfile(ROOT).recipes["release-gate"]["dependencies"]}
    assert "_release-notes-check" in gate_dependencies


def test_oidc_job_installs_only_minimal_locked_publication_tools() -> None:
    """The credential-capable job excludes local code and managed development builds."""
    workflow = yaml.safe_load(WORKFLOW.read_bytes())
    steps = workflow["jobs"]["publish"]["steps"]
    rendered = yaml.safe_dump(steps)
    assert "setup-just" not in rendered
    assert "toolchain sync" not in rendered
    assert "toolchain run" not in rendered
    setup_uv = next(step for step in steps if step.get("uses", "").startswith("astral-sh/setup-uv@"))
    shared_setup = yaml.safe_load((ROOT / ".github/actions/setup-just/action.yml").read_bytes())
    assert setup_uv["uses"] == shared_setup["runs"]["steps"][0]["uses"]
    assert setup_uv["with"] == {"version-file": "pyproject.toml", "enable-cache": False}
    bootstrap = next(step for step in steps if step.get("name") == "Install locked publication tooling")
    assert "uv sync --locked --managed-python --only-group tooling --no-install-project" in bootstrap["run"]
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert manifest["dependency-groups"]["tooling"] == ["research-repo-tools[python-tools]==0.2.3"]


def test_oidc_rust_setup_exports_the_declared_minimal_toolchain(tmp_path: Path) -> None:
    """Exercise channel selection and Actions environment output without installation."""
    workflow = yaml.safe_load(WORKFLOW.read_bytes())
    script = next(step["run"] for step in workflow["jobs"]["publish"]["steps"] if step.get("name") == "Install the declared minimal Rust toolchain")
    output = tmp_path / "environment"
    calls = tmp_path / "rustup-args"
    runner = (
        'uv() { [[ "$1 $2 $3 $4" == "run --locked --no-sync python" ]] || return 90; shift 4; '
        f'{shlex.quote(Path(sys.executable).as_posix())} "$@"; }}\n'
        f"rustup() {{ printf '%s\\n' \"$@\" > {shlex.quote(calls.as_posix())}; }}\n"
    )
    result = run_command("bash", ["--noprofile", "--norc", "-euc", runner + script], cwd=ROOT, env={**os.environ, "GITHUB_ENV": output.as_posix()}, check=False)
    assert result.returncode == 0, result.stderr
    channel = tomllib.loads((ROOT / "rust-toolchain.toml").read_text(encoding="utf-8"))["toolchain"]["channel"]
    assert calls.read_text(encoding="utf-8").splitlines() == ["toolchain", "install", channel, "--profile", "minimal", "--no-self-update"]
    assert output.read_text(encoding="utf-8") == f"RUSTUP_TOOLCHAIN={channel}\n"


@pytest.mark.parametrize("recipe", ["release-gate", "release-verify"])
def test_oidc_release_commands_cannot_resynchronize_dev(recipe: str) -> None:
    """Real Just expansion must retain the minimal environment after setup."""
    workflow = yaml.safe_load(WORKFLOW.read_bytes())
    commands = [line for step in workflow["jobs"]["publish"]["steps"] for line in step.get("run", "").splitlines()]
    command = shlex.split(next(line for line in commands if line.endswith(f'{recipe} "$RELEASE_TAG"')))
    assert command[:4] == ["just", "--set", "rrt", "uv run --locked --no-sync research-repo-tools"]
    command[-1] = TAG
    command.insert(1, "--dry-run")
    result = run_command(command[0], command[1:], cwd=ROOT)
    rendered = result.stdout + result.stderr
    assert f"research-repo-tools release {'gate' if recipe == 'release-gate' else 'verify'}" in rendered
    assert "--group dev" not in rendered
    assert "uv sync" not in rendered


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
    updates = tomllib.loads((ROOT / ".github/action-updates.toml").read_text(encoding="utf-8"))["actions"]
    assert "rust-lang/crates-io-auth-action" not in updates
    assert "astral-sh/setup-uv" not in updates
    assert "acgetchell/research-repo-tools/.github/workflows/dependabot-approve.yml" not in updates
    auto_merge = (ROOT / ".github/workflows/dependabot-auto-merge.yml").read_text(encoding="utf-8")
    assert '".github/workflows/publish-crates.yml"' not in auto_merge


@pytest.mark.parametrize("mode", ["normal", "quiet", "failed"])
def test_promotion_output_survives_real_just_and_reaches_approval(tmp_path: Path, mode: str) -> None:
    """Exercise workflow shells and Just without using stdout as evidence."""
    workflow = yaml.safe_load(WORKFLOW.read_bytes())
    verify = workflow["jobs"]["verify"]["steps"][-1]
    approval = next(step for step in workflow["jobs"]["publish"]["steps"] if "PROMOTION_SELECTION" in step.get("env", {}))
    root = tmp_path / "consumer with spaces"
    root.mkdir()
    (root / "justfile").write_text(f"import '{(ROOT / 'justfile').as_posix()}'\n", encoding="utf-8")
    output = root / "github output"
    frozen = json.dumps({"run_id": 123, "run_attempt": 1, "artifact": {"id": 456, "digest": "sha256:" + "a" * 64}}, sort_keys=True)
    # Only the producer is substituted. The CLI round-trip tests independently
    # exercise validation and file emission using the installed shared helpers.
    producer = (
        "uv() {\n"
        '  [[ "$*" == "run --locked --no-sync python tooling/python/release_artifacts.py verify" ]] || return 90\n'
        '  if [[ "$FIXTURE_EXIT" != 0 ]]; then return "$FIXTURE_EXIT"; fi\n'
        '  printf "selection=%s\\n" "$FIXTURE_SELECTION" >> "$GITHUB_OUTPUT"\n'
        '  printf "%s\\n" "human-readable confirmation"\n'
        "}\n"
        "export -f uv\n"
    )
    script = verify["run"]
    if mode == "quiet":
        script = "just --quiet " + script.removeprefix("just ")
    environment = {
        **{
            key: value
            for key, value in os.environ.items()
            if key not in {"BASH_ENV", "SHELLOPTS", "BASHOPTS", "JUST_QUIET", "JUST_DRY_RUN", "PROMOTION_SELECTION"}
        },
        "GITHUB_OUTPUT": str(output),
        "FIXTURE_SELECTION": frozen,
        "FIXTURE_EXIT": "17" if mode == "failed" else "0",
    }
    result = run_command("bash", ["--noprofile", "--norc", "-euc", producer + script], cwd=root, env=environment, check=False)
    assert result.returncode == (17 if mode == "failed" else 0), result.stderr
    if mode == "failed":
        assert not output.exists()
        return
    assert output.read_text(encoding="utf-8").splitlines() == ["selection=" + frozen]
    environment["PROMOTION_SELECTION"] = output.read_text(encoding="utf-8").split("=", 1)[1].rstrip("\n")
    consumer = (
        "uv() {\n"
        '  [[ "$*" == "run --locked --no-sync python tooling/python/release_artifacts.py verify" ]] || return 90\n'
        '  [[ "$PROMOTION_SELECTION" == "$FIXTURE_SELECTION" ]]\n'
        "}\n"
    )
    approved = run_command("bash", ["--noprofile", "--norc", "-euc", consumer + approval["run"]], cwd=root, env=environment, check=False)
    assert approved.returncode == 0, approved.stderr


@pytest.mark.parametrize(("failure", "expected_exit"), [("none", 0), ("evidence", 1), ("present", 1), ("unavailable", 2)])
def test_preupload_shell_requires_evidence_and_proven_absence(tmp_path: Path, failure: str, expected_exit: int) -> None:
    """An existing version or unavailable registry blocks the actual upload sequence."""
    workflow = yaml.safe_load(WORKFLOW.read_bytes())
    script = next(step["run"] for step in workflow["jobs"]["publish"]["steps"] if step.get("name", "").startswith("Recheck evidence"))
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


@pytest.mark.parametrize(
    "failure_index",
    [
        pytest.param(0, id="toolchain"),
        pytest.param(1, id="metadata"),
        pytest.param(2, id="default-package"),
        pytest.param(3, id="all-features-package"),
        pytest.param(None, id="success"),
    ],
)
def test_packaging_stops_on_failure_and_completes_on_success(tmp_path: Path, failure_index: int | None) -> None:
    """Reach each packaging stage and preserve its failure without running later work."""
    expected_calls = [
        "run --locked --no-sync --no-python-downloads research-repo-tools toolchain check",
        "run --locked --group dev research-repo-tools toolchain run -- research-repo-tools validation cargo-metadata --package delaunay",
        "run --locked --group dev research-repo-tools toolchain run -- cargo publish --locked --package delaunay --registry crates-io --dry-run",
        "run --locked --group dev research-repo-tools toolchain run -- cargo publish --locked --package delaunay --registry crates-io --dry-run --all-features",
    ]
    failing_call = "" if failure_index is None else expected_calls[failure_index]
    calls = tmp_path / "calls"
    runner = f"""uv() {{
    printf '%s\\n' "$*" >> {shlex.quote(calls.as_posix())}
    if [[ "$*" == {shlex.quote(failing_call)} ]]; then return 17; fi
    return 0
}}
"""
    result = run_command("bash", ["--noprofile", "--norc", "-euc", runner + dry_run(ROOT, "release-package").stderr], check=False)
    assert result.returncode == (0 if failure_index is None else 17), result.stderr
    expected_count = len(expected_calls) if failure_index is None else failure_index + 1
    assert calls.read_text(encoding="utf-8").splitlines() == expected_calls[:expected_count]
