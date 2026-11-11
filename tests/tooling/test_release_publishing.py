"""Exercise the consumer's release integration without credentials or uploads."""

import hashlib
import json
import os
import shlex
import subprocess
import sys
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from research_repo_tools import changelog, config, dependency_provenance, release_assets, release_publishing
from research_repo_tools.cli import main
from research_repo_tools.just_inspect import dry_run, inspect_justfile
from research_repo_tools.process import run_command

import release_candidate

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/publish-crates.yml"
POLICY = config.load(root=ROOT)
VERSION = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))["package"]["version"]
TAG = f"v{VERSION}"
SHA = "a" * 40
CHECKS = ("portable", "build (ubuntu-latest)", "build (macos-latest)", "build (windows-latest)", "Clippy SARIF Analysis", "Repository Rule SARIF Analysis")


def _checkout_git(executable: str | Path, args: list[str], *, cwd: Path, timeout: int, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
    """Supply clean read-only Git observations to the installed checkout verifier."""
    options = ("--no-pager", "--no-optional-locks", "--no-replace-objects", "-c", "core.fsmonitor=false", "-c", "diff.autoRefreshIndex=false")
    assert cwd.samefile(ROOT)
    assert timeout == 30
    assert Path(executable).is_absolute()
    assert tuple(args[: len(options)]) == options
    responses = {
        ("rev-parse", "HEAD"): f"{SHA}\n".encode("ascii"),
        ("ls-files", "-v", "-z"): b"H Cargo.toml\0",
        ("submodule", "foreach", "--quiet", "--recursive", shlex.join((Path(executable).as_posix(), *options, "ls-files", "-v", "-z"))): b"",
        ("status", "--porcelain", "-z", "--untracked-files=all", "--ignore-submodules=none"): b"",
    }
    return subprocess.CompletedProcess(args, 0, stdout=responses[tuple(args[len(options) :])], stderr=b"")


@pytest.mark.parametrize("recipe_name", ["release-assets", "_release-candidate"])
def test_release_bootstrap_excludes_noncanonical_checkout_paths(
    recipe_name: str, tmp_path: Path, native_console_scripts: Callable[[Path, str, dict[str, str]], Path]
) -> None:
    """Probe managed tools outside the observed checkout in both recipes."""
    recipe = inspect_justfile(ROOT).recipes[recipe_name]
    script = "\n".join("".join(line) for line in recipe["body"])
    checkout = tmp_path / "checkout"
    (checkout / "tests").mkdir(parents=True)
    uv_pin = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["uv"]["required-version"]
    (checkout / "pyproject.toml").write_text(
        f'[project]\nname = "release-fixture"\nversion = "{VERSION}"\nrequires-python = ">=3.15"\n[tool.uv]\nrequired-version = "{uv_pin}"\n',
        encoding="utf-8",
    )
    (checkout / "tooling/python").mkdir(parents=True)
    (checkout / "tooling/python/release_candidate.py").write_bytes((ROOT / "tooling/python/release_candidate.py").read_bytes())
    (checkout / ".python-version").write_bytes((ROOT / ".python-version").read_bytes())
    outside = native_console_scripts(
        tmp_path / "trusted tools",
        "import json, os\nfrom pathlib import Path\n"
        f"def git_main():\n    print({SHA!r})\n"
        "def uv_main():\n"
        '    Path(os.environ["UV_PROBE_LOG"]).write_text(json.dumps(os.get_exec_path()), encoding="utf-8")\n'
        f"    print('uv {uv_pin.removeprefix('==')}')\n"
        "def gh_main():\n    raise AssertionError('unexpected GitHub access')\n",
        {"git": "git_main", "uv": "uv_main", "gh": "gh_main"},
    )
    # Exercise a directory alias without requiring Windows symlink privileges.
    alias = checkout / "tests" / ".."
    search_path = os.pathsep.join((str(checkout / ".venv" / "bin"), str(alias / "bin"), ".", "", str(outside), os.environ["PATH"]))
    args = ["preflight", TAG] if recipe_name == "release-assets" else ["verify"]
    probe = tmp_path / "uv-path.json"
    environment = os.environ | {"PATH": search_path, "UV_PROBE_LOG": str(probe)}
    result = run_command(sys.executable, ["-I", "-c", script, *args, "--help"], cwd=checkout, env=environment, check=False)
    assert result.returncode == 0, result.stderr
    assert ("--commit COMMIT" if recipe_name == "release-assets" else "--selection SELECTION") in result.stdout
    paths = [Path(entry) for entry in json.loads(probe.read_bytes())]
    assert outside.resolve() in paths
    assert all(path.is_absolute() and not path.resolve().is_relative_to(checkout.resolve()) for path in paths)


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
    monkeypatch.setattr(release_assets, "run_command_bytes", _checkout_git)
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

    def published(root: Path, *, packages: tuple[str, ...] | None = None) -> tuple[dependency_provenance.DependencyIdentity, ...]:
        assert root == ROOT
        assert packages == ()
        return (dependency_provenance.DependencyIdentity("research-repo-tools", "0.2.8", "https://pypi.org/simple"),)

    monkeypatch.setattr(dependency_provenance, "check_release_dependencies", published)
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


@dataclass
class PublicationProvider:
    """Provider state and retained owner bytes for the final publication boundary."""

    accepted: Path
    release: release_assets.GitHubRelease
    calls: list[str]
    failure: str = "none"


@pytest.fixture
def sealed_publication(release_event: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> PublicationProvider:
    """Retain original assertions and model only owner inspection and provider transport."""
    accepted = tmp_path / "acceptance.json"
    accepted.write_bytes(b"reviewed full acceptance")
    payloads = {role: role.encode() for role in release_candidate.ROLES}
    payloads["acceptance"] = accepted.read_bytes()
    assets = [
        release_candidate.AssetAssertion(f"candidate-{role}.json", index, hashlib.sha256(payloads[role]).hexdigest())
        for index, role in enumerate(release_candidate.ROLES, 1)
    ]
    target = release_assets.ReleaseTarget("acgetchell/delaunay", TAG, 42, SHA, TAG)
    original = release_candidate.DraftTransfer(target, "a" * 64, "b" * 64, release_candidate.OriginalAssets(*assets))
    attached = release_candidate.AssetAssertion(release_candidate.archive_name(TAG), 5, original.assets.archive.sha256)
    transfer = release_candidate.PublicationTransfer(original, attached)
    sealed = tmp_path / "publication-transfer.json"
    sealed.write_bytes(release_candidate.serialize_transfer(transfer))
    index = release_candidate.AssetAssertion(f"delaunay-{TAG}-candidate.json", 6, hashlib.sha256(sealed.read_bytes()).hexdigest())
    (tmp_path / "index-assertion.json").write_text(json.dumps({"name": index.name, "id": index.identifier, "sha256": index.sha256}), encoding="utf-8")
    release = release_assets.GitHubRelease(
        repository=target.repository,
        tag=TAG,
        identifier=42,
        draft=True,
        immutable=False,
        prerelease=False,
        title=TAG,
        assets=(*tuple((asset.name, asset.identifier, 100, asset.sha256) for _, asset in transfer.role_assertions()), (index.name, 6, 100, index.sha256)),
    )
    state = PublicationProvider(accepted, release, [])
    monkeypatch.setattr(release_publishing, "lookup_release", lambda *_args: state.release)

    def gate(name: str, result: Path, *_args: object) -> Path:
        state.calls.append(name)
        if state.failure == name:
            message = "original evidence unavailable"
            raise ValueError(message)
        return result

    monkeypatch.setattr(release_candidate, "verify_existing", lambda *args: gate("acceptance", accepted, *args))
    monkeypatch.setattr(release_candidate, "preparation_target", lambda *args: gate("receipt", sealed, *args))
    monkeypatch.setattr(release_candidate, "seal", lambda *args: gate("assets", sealed, *args))
    monkeypatch.setattr(release_candidate, "shared_main", lambda *_args: pytest.fail("publication must not select a fresh draft by tag"))
    monkeypatch.setattr(release_candidate, "publish_release_asset", lambda *_args, **_kwargs: pytest.fail("publication must never repair attachments"))

    def publish(executable: str, arguments: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        assert cwd == ROOT
        assert executable == "gh"
        assert arguments == ["api", "--method", "PATCH", "repos/acgetchell/delaunay/releases/42", "--field", "draft=false"]
        state.calls.append("publish")
        if state.failure in {"late-draft", "uncertain"}:
            if state.failure == "late-draft":
                state.release = replace(state.release, identifier=43)
            raise subprocess.CalledProcessError(1, [executable, *arguments], stderr="original release unavailable")
        state.release = replace(state.release, draft=False, immutable=True)
        if state.failure == "after-asset":
            state.release = replace(
                state.release, assets=tuple((name, identifier + 10, size, digest) for name, identifier, size, digest in state.release.assets)
            )
        return subprocess.CompletedProcess([executable, *arguments], 0, "{}", "")

    monkeypatch.setattr(release_candidate, "run_command", publish)
    return state


@pytest.mark.parametrize(
    "failure",
    [
        "none",
        "acceptance",
        "receipt",
        "assets",
        "draft",
        "asset",
        "missing-asset",
        "missing-index",
        "index-id",
        "acceptance-digest",
        "late-draft",
        "uncertain",
        "after-asset",
    ],
)
def test_publish_requires_original_acceptance_receipt_and_assets(sealed_publication: PublicationProvider, tmp_path: Path, failure: str) -> None:
    """Exercise shared final gates and the original-ID mutation without reselection or repair."""
    state = sealed_publication
    state.failure = failure
    if failure == "draft":
        state.release = replace(state.release, identifier=43)
    elif failure == "asset":
        state.release = replace(state.release, assets=tuple((name, identifier + 10, size, digest) for name, identifier, size, digest in state.release.assets))
    elif failure == "missing-asset":
        state.release = replace(state.release, assets=())
    elif failure == "missing-index":
        state.release = replace(state.release, assets=state.release.assets[:-1])
    elif failure == "index-id":
        name, identifier, size, digest = state.release.assets[-1]
        state.release = replace(state.release, assets=(*state.release.assets[:-1], (name, identifier + 1, size, digest)))
    elif failure == "acceptance-digest":
        state.accepted.write_bytes(b"different acceptance")
    if failure == "none":
        release_candidate.publish_draft(ROOT, tmp_path / "owner", "reviewed", TAG)
        assert state.calls == ["acceptance", "receipt", "assets", "acceptance", "publish"]
        assert state.release.immutable
        assert not state.release.draft
    elif failure in {"late-draft", "uncertain", "after-asset"}:
        with pytest.raises(release_assets.ReleasePublicationUnknownError, match=r"publication outcome unknown.*release 42"):
            release_candidate.publish_draft(ROOT, tmp_path / "owner", "reviewed", TAG)
        assert state.calls.count("publish") == 1
    else:
        with pytest.raises(ValueError, match=r"original evidence|reviewed release|asset identity|reviewed acceptance|required release asset"):
            release_candidate.publish_draft(ROOT, tmp_path / "owner", "reviewed", TAG)
        assert "publish" not in state.calls


def test_workflow_limits_credentials_and_verifies_clean_packages() -> None:
    """Credential-free builds precede approval, revalidation and a serialized upload."""
    workflow = yaml.safe_load(WORKFLOW.read_bytes())
    assert set(workflow[True]["workflow_dispatch"]["inputs"]) == {"tag", "candidate", "acceptance"}
    assert workflow["concurrency"] == {"group": "publish-crates-delaunay", "cancel-in-progress": False}
    jobs = workflow["jobs"]
    assert jobs["publish"]["needs"] == "verify"
    assert jobs["publish"]["environment"] == "crates-io"
    assert jobs["verify"]["permissions"] == {"contents": "read", "checks": "read", "actions": "read"}
    assert jobs["publish"]["permissions"] == {"contents": "read", "checks": "read", "actions": "read", "id-token": "write"}
    assert "github.repository == 'acgetchell/delaunay'" in jobs["verify"]["if"]
    assert jobs["verify"]["steps"][-1]["id"] == "approval"
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
    assert 'release-receive "$RUNNER_TEMP/accepted-candidate" --published --expected' in gate["run"]
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
    assert manifest["dependency-groups"]["tooling"] == ["research-repo-tools[python-tools]==0.2.8"]


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


@pytest.mark.parametrize("recipe", ["_release-notes-check", "release-verify"])
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
    assert ("changelog notes" if recipe == "_release-notes-check" else "release verify") in rendered
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


@pytest.mark.parametrize("mismatch", ["none", "attempt", "digest", "run", "expired"])
def test_approval_transport_requires_original_attempt_and_digest(tmp_path: Path, mismatch: str) -> None:
    """Execute the workflow's real artifact metadata check before downloading approval."""
    workflow = yaml.safe_load(WORKFLOW.read_bytes())
    gate = next(step for step in workflow["jobs"]["publish"]["steps"] if step.get("name") == "Require this exact approval receipt")
    artifact = {
        "expired": mismatch == "expired",
        "digest": "sha256:" + ("b" if mismatch == "digest" else "a") * 64,
        "workflow_run": {"id": 999 if mismatch == "run" else 123},
    }
    runner = f"gh() {{ printf '%s' {shlex.quote(json.dumps(artifact))}; }}\n"
    result = run_command(
        "bash",
        ["--noprofile", "--norc", "-euc", runner + gate["run"]],
        cwd=tmp_path,
        env=os.environ
        | {
            "VALIDATED_ATTEMPT": "2" if mismatch == "attempt" else "1",
            "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_RUN_ID": "123",
            "GITHUB_REPOSITORY": "acgetchell/delaunay",
            "ARTIFACT_ID": "456",
            "ARTIFACT_DIGEST": "a" * 64,
        },
        check=False,
    )
    assert (result.returncode == 0) is (mismatch == "none"), result.stderr
    verify = workflow["jobs"]["verify"]
    artifact_step = verify["steps"][-1]
    assert artifact_step["with"]["path"] == "${{ runner.temp }}/accepted-candidate/acceptance.json"
    assert verify["outputs"]["artifact-id"] == "${{ steps.approval.outputs.artifact-id }}"
    approval = next(step for step in workflow["jobs"]["publish"]["steps"] if step.get("name", "").startswith("Recheck evidence"))
    assert '--expected "$RUNNER_TEMP/approved-acceptance/acceptance.json"' in approval["run"]
    assert approval["env"]["CANDIDATE_ACCEPTANCE"] == "${{ inputs.acceptance }}"


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
        env={**os.environ, "RELEASE_TAG": TAG, "RUNNER_TEMP": str(tmp_path)},
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
        "run --locked --no-sync --group dev research-repo-tools toolchain run -- research-repo-tools validation cargo-metadata --package delaunay",
        "run --locked --no-sync --group dev research-repo-tools toolchain run -- cargo publish --locked --package delaunay --registry crates-io --dry-run",
        (
            "run --locked --no-sync --group dev research-repo-tools toolchain run -- "
            "cargo publish --locked --package delaunay --registry crates-io --dry-run --all-features"
        ),
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
