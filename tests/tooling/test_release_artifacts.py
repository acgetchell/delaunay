"""Consumer Cargo evidence, real archives and direct approval handoff checks."""

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from research_repo_tools import release_evidence, release_validation
from research_repo_tools.process import resolve_executable, run_command
from research_repo_tools.release_assets import GitHubRelease, ReleaseTarget, serialize_release_target
from research_repo_tools.release_interfaces import load_evidence_policy

import cargo_release as promotion

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parents[2]
POLICY = load_evidence_policy(ROOT / "tooling/release-evidence.toml")
COMMIT = "a" * 40
VERSION = "0.8.2"
NAME = f"delaunay-{VERSION}.crate"
TOOLS = {"cargo": "cargo 1.99.1", "rustc": "rustc 1.99.1\nhost: x86_64-unknown-linux-gnu"}
NATIVE_RUN = subprocess.run


def crate_bytes(*, dirty: bool = False, commit: str = COMMIT, version: str = VERSION) -> bytes:
    """Build independent Cargo source/version metadata and a binary source member."""
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        records = {
            ".cargo_vcs_info.json": json.dumps({"git": {"sha1": commit, "dirty": dirty}, "path_in_vcs": ""}).encode(),
            "Cargo.toml": f'[package]\nname = "delaunay"\nversion = "{version}"\n'.encode(),
            "src/lib.rs": b"// byte fixture: \xff\r\n",
        }
        for name, content in records.items():
            entry = tarfile.TarInfo(f"delaunay-{version}/{name}")
            entry.size = len(content)
            archive.addfile(entry, io.BytesIO(content))
    return output.getvalue()


@dataclass
class CargoFixture:
    root: Path
    run: dict[str, Any]
    jobs: list[dict[str, Any]]
    artifact: dict[str, Any]
    archive: bytes
    manifest: dict[str, Any] = field(default_factory=dict)
    calls: list[list[str]] = field(default_factory=list)
    changed_package: bool = False
    rerun_during_packaging: bool = False
    missing_artifact: bool = False
    damaged_zip: bool = False
    certificate: dict[str, str] = field(
        default_factory=lambda: {"runInvocationURI": "https://github.com/acgetchell/delaunay/actions/runs/123/attempts/1", "buildTrigger": "push"}
    )

    def payload(self) -> bytes:
        """Keep transport identity independent of the clock."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(zipfile.ZipInfo(NAME), self.archive)
            archive.writestr(zipfile.ZipInfo(promotion.MANIFEST), json.dumps(self.manifest))
            archive.writestr(zipfile.ZipInfo("validation.sigstore.json"), b"fixture signature")
        return buffer.getvalue()

    def read(self, endpoint: str, **_kwargs: object) -> dict[str, Any]:
        """Model provider observations, leaving all trust decisions installed."""
        if endpoint == "actions/workflows/ci.yml":
            return {"id": 7, "path": ".github/workflows/ci.yml", "name": "CI", "state": "active"}
        if endpoint == "actions/runs/123":
            return self.run
        if endpoint.startswith("actions/workflows/7/runs?"):
            return {"workflow_runs": [self.run]}
        if "/jobs?" in endpoint:
            return {"jobs": self.jobs}
        if "/artifacts?" in endpoint or endpoint == "actions/artifacts/456":
            payload = self.payload()
            self.artifact.update(digest=f"sha256:{hashlib.sha256(payload).hexdigest()}", size_in_bytes=len(payload))
            return self.artifact if endpoint == "actions/artifacts/456" else {"artifacts": [] if self.missing_artifact else [self.artifact]}
        msg = f"unexpected provider endpoint: {endpoint}"
        raise ValueError(msg)

    def evidence(self, *, expected: bytes | None = None) -> bytes:
        """Run the actual installed pre-tag verifier against the consumer policy."""
        return release_evidence.verify_release_evidence(POLICY, VERSION, COMMIT, self.read, expected=expected)

    def command(self, executable: str, args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        """Model Cargo's native files; no crate upload is permitted."""
        self.calls.append([executable, *args])
        if executable == "cargo" and args[0] == "package":
            assert args[1:8] == ["--locked", "--package", "delaunay", "--registry", "crates-io", "--no-verify", "--target-dir"]
            destination = Path(args[8]) / "package" / NAME
            destination.parent.mkdir(parents=True)
            destination.write_bytes(crate_bytes(commit="b" * 40) if self.changed_package else self.archive)
            if self.rerun_during_packaging:
                self.run["run_attempt"] = 2
            result = ""
        elif executable == "cargo" and args[0] == "metadata":
            result = json.dumps({"target_directory": str(self.root / "target")})
        elif (executable, args) == ("cargo", ["--version"]):
            result = TOOLS["cargo"]
        else:
            assert (executable, args) == ("rustc", ["--version", "--verbose"])
            result = TOOLS["rustc"]
        return subprocess.CompletedProcess([executable, *args], 0, stdout=result, stderr="")


@pytest.fixture
def cargo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CargoFixture:
    """Use the shipped consumer policy and real shared manifest/ZIP verification."""
    (tmp_path / "tooling").mkdir()
    (tmp_path / "tooling/release-evidence.toml").write_bytes((ROOT / "tooling/release-evidence.toml").read_bytes())
    (tmp_path / "Cargo.toml").write_text(f'[package]\nname = "delaunay"\nversion = "{VERSION}"\n', encoding="utf-8")
    repository = {"id": 99, "full_name": promotion.REPOSITORY}
    assert POLICY.signing_job is not None
    run = {
        "id": 123,
        "run_attempt": 1,
        "head_sha": COMMIT,
        "head_branch": "main",
        "repository": repository,
        "head_repository": repository.copy(),
        "workflow_id": 7,
        "name": "CI",
        "path": ".github/workflows/ci.yml",
        "event": "push",
        "pull_requests": [],
        "status": "completed",
        "conclusion": "success",
    }
    jobs = [
        {
            "id": identifier,
            "name": name,
            "run_id": 123,
            "run_attempt": 1,
            "head_sha": COMMIT,
            "head_branch": "main",
            "status": "completed",
            "conclusion": "success",
            "labels": list(dict(POLICY.validation.runner_labels).get(name, ())),
            "steps": [{"name": step, "status": "completed", "conclusion": "success"} for step in steps],
        }
        for identifier, (name, steps) in enumerate((*POLICY.validation.required_steps, POLICY.signing_job), 1)
    ]
    artifact = {
        "id": 456,
        "name": "cargo-package-123-1",
        "expired": False,
        "workflow_run": {"id": 123, "head_sha": COMMIT, "head_branch": "main", "repository_id": 99, "head_repository_id": 99},
    }
    fixture = CargoFixture(tmp_path, run, jobs, artifact, crate_bytes())
    archive = tmp_path / NAME
    archive.write_bytes(fixture.archive)
    fixture.manifest = json.loads(release_evidence.create_evidence_manifest(POLICY, VERSION, COMMIT, 123, 1, fixture.read, archive=archive, tools=TOOLS))
    monkeypatch.setattr(promotion, "identity", lambda _root: (COMMIT, VERSION))
    monkeypatch.setattr(promotion, "api", fixture.read)
    monkeypatch.setattr(promotion, "run_command", fixture.command)
    event = {"action": "published", "repository": repository, "release": {"id": 789, "tag_name": f"v{VERSION}", "draft": False, "prerelease": False}}
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event), encoding="utf-8")
    for key, value in {
        "GITHUB_REPOSITORY": promotion.REPOSITORY,
        "GITHUB_EVENT_NAME": "release",
        "GITHUB_SHA": COMMIT,
        "GITHUB_REF": f"refs/tags/v{VERSION}",
        "GITHUB_EVENT_PATH": str(event_path),
    }.items():
        monkeypatch.setenv(key, value)

    def download(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        assert args == ["gh", "api", f"repos/{promotion.REPOSITORY}/actions/artifacts/456/zip"]
        kwargs["stdout"].write(b"corrupt ZIP" if fixture.damaged_zip else fixture.payload())
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(release_validation.subprocess, "run", download)

    def attestation(*arguments: str) -> str:
        assert arguments[:2] == ("attestation", "verify")
        assert arguments[arguments.index("--signer-workflow") + 1] == "acgetchell/delaunay/.github/workflows/ci.yml"
        assert arguments[arguments.index("--signer-digest") + 1] == COMMIT
        assert arguments[arguments.index("--source-digest") + 1] == COMMIT
        assert arguments[arguments.index("--source-ref") + 1] == "refs/heads/main"
        assert "--deny-self-hosted-runners" in arguments
        assert Path(arguments[arguments.index("--bundle") + 1]).read_bytes() == b"fixture signature"
        return json.dumps([{"verificationResult": {"signature": {"certificate": fixture.certificate}}}])

    monkeypatch.setattr(release_validation, "gh", attestation)
    return fixture


def test_pre_tag_evidence_and_frozen_cargo_reproduction(cargo: CargoFixture) -> None:
    """No tag/event/Cargo is needed to inspect the shared evidence."""
    selected = cargo.evidence()
    assert json.loads(selected)["artifact"]["id"] == 456
    assert not cargo.calls
    assert promotion.verify(cargo.root, expected=selected) == selected
    assert [call[1] for call in cargo.calls if call[0] == "cargo"] == ["package", "--version"]
    assert promotion.verify(cargo.root, expected=selected) == selected


@pytest.mark.parametrize("failure", ["none", "actor", "attempt", "mutable", "bytes", "tools"])
def test_candidate_approval_preserves_cargo_reproduction(cargo: CargoFixture, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    """Exercise the new accepted-dispatch boundary using the real signed Cargo verifier."""
    accepted = cargo.root / "accepted"
    accepted.mkdir()
    target = ReleaseTarget(promotion.REPOSITORY, f"v{VERSION}", 789, COMMIT, f"v{VERSION}")
    (accepted / "target.json").write_bytes(serialize_release_target(target))
    native = cargo.evidence()
    (accepted / "acceptance.json").write_text(json.dumps({"native": json.loads(native)}), encoding="utf-8")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_ACTOR", "other" if failure == "actor" else "acgetchell")
    monkeypatch.setenv("GITHUB_TRIGGERING_ACTOR", "acgetchell")
    monkeypatch.setenv("GITHUB_WORKFLOW_REF", f"{promotion.REPOSITORY}/.github/workflows/publish-crates.yml@refs/tags/v{VERSION}")
    release = GitHubRelease(
        promotion.REPOSITORY, f"v{VERSION}", 789, draft=False, immutable=failure != "mutable", prerelease=False, assets=(), title=f"v{VERSION}"
    )
    monkeypatch.setattr(promotion, "check_reviewed_release", lambda *_args, **_kwargs: release)
    if failure == "attempt":
        cargo.run["run_attempt"] = 2
    elif failure == "bytes":
        cargo.changed_package = True
    elif failure == "tools":
        monkeypatch.setattr(promotion, "tool_identity", lambda _root: TOOLS | {"cargo": "different Cargo"})
    if failure == "none":
        assert promotion.verify(cargo.root, candidate=accepted) == native
    else:
        with pytest.raises(ValueError, match=r"accepted exact-tag|immutable published|attempt|reproduction|archive source/package/version"):
            promotion.verify(cargo.root, candidate=accepted)
    assert any(call[:2] == ["cargo", "package"] for call in cargo.calls) is (failure in {"none", "bytes", "tools"})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("runInvocationURI", "https://github.com/acgetchell/delaunay/actions/runs/123/attempts/2"),
        ("runInvocationURI", "https://github.com/other/repo/actions/runs/123/attempts/1"),
        ("buildTrigger", "pull_request"),
    ],
)
def test_signature_requires_original_main_attempt(cargo: CargoFixture, field: str, value: str) -> None:
    """A cryptographically verified certificate must still name the original producer."""
    cargo.certificate[field] = value
    with pytest.raises(ValueError, match="attestation certificate run, attempt or expected event differs"):
        cargo.evidence()
    assert not cargo.calls


@pytest.mark.parametrize("quiet", [False, True])
def test_isolated_native_cli_uses_signed_evidence_without_a_tag(
    cargo: CargoFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
    native_console_scripts: Callable[[Path, str, dict[str, str]], Path],
    *,
    quiet: bool,
) -> None:
    """Exercise the native half of candidate acceptance through its installed CLI."""
    monkeypatch.setattr(subprocess, "run", NATIVE_RUN)
    root = cargo.root
    uv_pin = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["uv"]["required-version"]
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "release-fixture"\nversion = "0.0.0"\nrequires-python = ">=3.15"\ndependencies = []\n[tool.uv]\nrequired-version = "{uv_pin}"\n',
        encoding="utf-8",
        newline="\n",
    )
    (root / ".python-version").write_bytes((ROOT / ".python-version").read_bytes())
    (root / "justfile").write_text(
        "[positional-arguments]\n"
        '[script("uvx", "--no-config", "--isolated", "--managed-python", "--python", "3.15", '
        '"--default-index", "https://pypi.org/simple", "--from", "research-repo-tools==0.2.8", "python", "-I")]\n'
        "native-evidence *args:\n"
        "    import sys\n"
        "    from pathlib import Path\n"
        "    from research_repo_tools.cli import main\n"
        "    from research_repo_tools.config import load\n"
        "    from research_repo_tools.release_runtime import isolated_release_runtime\n"
        "    with isolated_release_runtime(load(root=Path.cwd())):\n"
        f'        sys.exit(main(["release", "evidence", "verify", "tooling/release-evidence.toml", '
        f'"--checkout", str(Path.cwd()), "--commit", "{COMMIT}", "--version", "{VERSION}", *sys.argv[1:]]))\n',
        encoding="utf-8",
    )
    cargo.read("actions/artifacts/456")
    provider = root / "provider.json"
    provider.write_text(json.dumps({"run": cargo.run, "jobs": cargo.jobs, "artifact": cargo.artifact}), encoding="utf-8")
    (root / "archive.zip").write_bytes(cargo.payload())
    executables = native_console_scripts(
        tmp_path_factory.mktemp("isolated-provider-runtime"),
        "import json, os, subprocess, sys\nfrom pathlib import Path\n"
        "def uvx_main():\n"
        '    assert sys.argv[1:12] == ["--no-config", "--isolated", "--managed-python", "--python", "3.15", '
        '"--default-index", "https://pypi.org/simple", "--from", "research-repo-tools==0.2.8", "python", "-I"]\n'
        '    return subprocess.run([os.environ["FIXTURE_PYTHON"], *sys.argv[11:]], check=False).returncode\n'
        "def uv_main():\n"
        '    if sys.argv[1:] == ["--version"]:\n'
        '        checkout = Path(os.environ["FIXTURE_CHECKOUT"]).resolve()\n'
        "        assert not Path.cwd().resolve().is_relative_to(checkout)\n"
        "        assert all(not Path(path).resolve().is_relative_to(checkout) for path in os.get_exec_path())\n"
        '    return subprocess.run([os.environ["FIXTURE_UV"], *sys.argv[1:]], check=False).returncode\n'
        "def git_main():\n"
        '    with Path(os.environ["FIXTURE_GIT_CALLS"]).open("a", encoding="utf-8") as calls:\n'
        '        calls.write(json.dumps(sys.argv[1:]) + "\\n")\n'
        '    assert not any("refs/tags" in arg for arg in sys.argv)\n'
        f'    if "rev-parse" in sys.argv:\n        print("{COMMIT}")\n'
        "def gh_main():\n"
        '    if sys.argv[1:3] == ["attestation", "verify"]:\n'
        '        print(json.dumps([{"verificationResult": {"signature": {"certificate": {\n'
        '            "runInvocationURI": "https://github.com/acgetchell/delaunay/actions/runs/123/attempts/1", "buildTrigger": "push"\n'
        "        }}}}]))\n        return 0\n"
        '    data = json.loads(Path(os.environ["PROVIDER"]).read_bytes())\n    endpoint = sys.argv[-1]\n'
        '    if endpoint.endswith("/zip"):\n        sys.stdout.buffer.write(Path(os.environ["ARCHIVE"]).read_bytes())\n        return 0\n'
        '    if endpoint.endswith("workflows/ci.yml"):\n        result = {"id":7,"path":".github/workflows/ci.yml","name":"CI","state":"active"}\n'
        '    elif "/runs?" in endpoint:\n        result = {"workflow_runs":[data["run"]]}\n'
        '    elif "/jobs?" in endpoint:\n        result = {"jobs":data["jobs"]}\n'
        '    elif "/artifacts?" in endpoint:\n        result = {"artifacts":[data["artifact"]]}\n'
        '    elif endpoint.endswith("/artifacts/456"):\n        result = data["artifact"]\n'
        '    elif endpoint.endswith("/runs/123"):\n        result = data["run"]\n'
        "    else:\n        raise AssertionError(endpoint)\n    print(json.dumps(result))\n",
        {"uv": "uv_main", "uvx": "uvx_main", "git": "git_main", "gh": "gh_main"},
    )
    output = root / "actions-output"
    output.write_bytes(b"prior=kept\n")
    selected = root / "selected.json"
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GITHUB_")}
    environment.update(
        {
            "PATH": os.pathsep.join((str(root), str(executables), os.environ["PATH"])),
            "PROVIDER": str(provider),
            "ARCHIVE": str(root / "archive.zip"),
            "FIXTURE_GIT_CALLS": str(root / "git-calls"),
            "FIXTURE_PYTHON": sys.executable,
            "FIXTURE_UV": str(resolve_executable("uv", cwd=ROOT)),
            "FIXTURE_CHECKOUT": str(root),
        }
    )
    result = run_command(
        "just",
        [*(["--quiet"] if quiet else []), "native-evidence", "--output", str(selected), "--github-output", str(output)],
        cwd=root,
        env=environment,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(selected.read_bytes())["run"]["id"] == 123
    records = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())
    assert records["prior"] == "kept"
    assert json.loads(records["identity"]) == json.loads(selected.read_bytes())
    assert records["commit"] == COMMIT
    assert "refs/tags" not in (root / "git-calls").read_text(encoding="utf-8")
    assert not cargo.calls


@pytest.mark.parametrize("field", ["event", "head_sha", "head_branch", "workflow_id", "path", "name", "repository", "head_repository", "conclusion"])
def test_untrusted_main_never_repackages(cargo: CargoFixture, field: str) -> None:
    cargo.run[field] = {"full_name": "other/repo", "id": 8} if "repository" in field else "wrong"
    with pytest.raises(ValueError, match=r"repository|identity|main-push|successful|positive|workflow"):
        promotion.verify(cargo.root)
    assert not cargo.calls


@pytest.mark.parametrize("job_index", range(len(POLICY.validation.required_steps)))
@pytest.mark.parametrize("failure", ["missing", "failure", "cancelled", "skipped", "partial-rerun", "skipped-step", "wrong-platform"])
def test_all_declared_jobs_are_required(cargo: CargoFixture, job_index: int, failure: str) -> None:
    job = cargo.jobs[job_index]
    if failure == "missing":
        cargo.jobs.remove(job)
    elif failure in {"failure", "cancelled", "skipped"}:
        job["conclusion"] = failure
    elif failure == "partial-rerun":
        job["run_attempt"] = 2
    elif failure == "wrong-platform":
        job["labels"] = ["unrelated"]
    else:
        job["steps"][0]["conclusion"] = "skipped"
    with pytest.raises(ValueError, match=r"attempt|succeed|step|platform"):
        promotion.verify(cargo.root)
    assert not cargo.calls


@pytest.mark.parametrize("failure", ["missing", "expired", "wrong-run", "wrong-commit", "wrong-repository", "damaged-zip"])
def test_bad_artifacts_block_reproduction(cargo: CargoFixture, failure: str) -> None:
    if failure == "missing":
        cargo.missing_artifact = True
    elif failure == "expired":
        cargo.artifact["expired"] = True
    elif failure == "damaged-zip":
        cargo.damaged_zip = True
    else:
        field_name = {"wrong-run": "id", "wrong-commit": "head_sha", "wrong-repository": "repository_id"}[failure]
        cargo.artifact["workflow_run"][field_name] = "wrong"
    with pytest.raises(ValueError, match=r"artifact|ZIP"):
        promotion.verify(cargo.root)
    assert not cargo.calls


@pytest.mark.parametrize("failure", ["package", "rerun", "frozen-selection", "tools"])
def test_reproduction_and_approval_cannot_switch_sources(cargo: CargoFixture, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    selected = cargo.evidence()
    cargo.changed_package = failure == "package"
    cargo.rerun_during_packaging = failure == "rerun"
    if failure == "frozen-selection":
        cargo.artifact["digest"] = "sha256:" + "b" * 64
        selected = selected.replace(b'"run_attempt": 1', b'"run_attempt": 2')
    if failure == "tools":
        monkeypatch.setattr(promotion, "tool_identity", lambda _root: TOOLS | {"cargo": "wrong"})
    with pytest.raises(ValueError, match=r"differ|attempt|identity"):
        promotion.verify(cargo.root, expected=selected)


@pytest.mark.parametrize("variable", ["GITHUB_EVENT_NAME", "GITHUB_REF", "GITHUB_SHA", "GITHUB_REPOSITORY"])
def test_published_event_boundary_is_required(cargo: CargoFixture, monkeypatch: pytest.MonkeyPatch, variable: str) -> None:
    monkeypatch.setenv(variable, "wrong")
    with pytest.raises(ValueError, match=r"release event|tag ref"):
        promotion.verify(cargo.root)
    assert not cargo.calls


@pytest.mark.parametrize("failure", ["none", "empty", "malformed", "changed", "unavailable", "output"])
def test_real_cli_freezes_and_exports_approval_evidence(cargo: CargoFixture, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    monkeypatch.chdir(cargo.root)
    monkeypatch.setattr(sys, "argv", ["cargo_release.py", "verify"])
    output, summary = cargo.root / "output", cargo.root / "summary"
    output.write_bytes(b"existing=kept\r\n")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.delenv("PROMOTION_SELECTION", raising=False)
    assert promotion.main() == 0
    assert output.read_bytes().startswith(b"existing=kept\r\nselection=")
    frozen = output.read_text(encoding="utf-8").splitlines()[1].split("=", 1)[1]
    assert json.loads(frozen)["run"]["id"] == 123
    if failure in {"empty", "malformed"}:
        frozen = "" if failure == "empty" else "null"
    elif failure == "changed":
        frozen = frozen.replace('"run_attempt": 1', '"run_attempt": 2')
    elif failure == "unavailable":
        cargo.missing_artifact = True
    elif failure == "output":
        output.unlink()
        output.mkdir()
    monkeypatch.setenv("PROMOTION_SELECTION", frozen)
    before = output.read_bytes() if output.is_file() else None
    assert promotion.main() == (0 if failure == "none" else 1)
    if failure != "none" and before is not None:
        assert output.read_bytes() == before


@pytest.mark.parametrize("mismatch", [False, True])
def test_producer_retains_both_dry_run_archive_identity(cargo: CargoFixture, monkeypatch: pytest.MonkeyPatch, *, mismatch: bool) -> None:
    for key, value in {"GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/main", "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1"}.items():
        monkeypatch.setenv(key, value)
    packages = cargo.root / "target/package"
    for directory, data in (("tmp-crate", cargo.archive), ("tmp-registry", b"wrong" if mismatch else cargo.archive)):
        destination = packages / directory / NAME
        destination.parent.mkdir(parents=True)
        destination.write_bytes(data)
    output = cargo.root / "retained"
    if mismatch:
        with pytest.raises(ValueError, match="archives differ"):
            promotion.create(cargo.root, output)
        assert not output.exists()
    else:
        promotion.create(cargo.root, output)
        assert (output / NAME).read_bytes() == cargo.archive
        assert json.loads((output / promotion.MANIFEST).read_bytes()) == cargo.manifest
        with pytest.raises(FileExistsError):
            promotion.create(cargo.root, output)


def test_workflow_policy_covers_portable_native_and_package_gates() -> None:
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_bytes())
    assert ci["jobs"]["build"]["needs"] == ["portable", "prepare", "native"]
    platforms = {"ubuntu-latest", "macos-latest", "windows-latest"}
    stages = {"python-docs", "rust-builds", "rust-integrations", "rust-tests"}
    expected = {"portable", "package"} | {f"prepare ({host})" for host in platforms} | {f"native-{stage} ({host})" for host in platforms for stage in stages}
    assert {name for name, _ in POLICY.validation.required_steps} == expected
    runner_labels = dict(POLICY.validation.runner_labels)
    for name, required_steps in POLICY.validation.required_steps:
        identifier = "native" if name.startswith("native-") else "prepare" if name.startswith("prepare (") else name
        job = ci["jobs"][identifier]
        assert set(required_steps) <= {step.get("name") for step in job["steps"]}
        if identifier in {"native", "prepare"}:
            host = name.rsplit("(", 1)[1].removesuffix(")")
            assert runner_labels[name] == (host,)
            assert job["runs-on"] == "${{ matrix.os }}"
            assert host in job["strategy"]["matrix"]["os"]
            if identifier == "native":
                assert set(required_steps) == {"Set up shared tooling", "Verify native Rust host", "Run native validation"}
                assert name.removeprefix("native-").split(" ", 1)[0] in job["strategy"]["matrix"]["stage"]
        else:
            assert runner_labels[name] == (job["runs-on"],)
    package = ci["jobs"]["package"]
    assert package["needs"] == "build"
    assert package["permissions"] == {"contents": "read", "actions": "read"}
    retain = next(step for step in package["steps"] if step.get("name") == "Retain verified Cargo package")
    assert retain["env"]["GH_TOKEN"] == "${{ github.token }}"  # noqa: S105 - Actions expression, never a credential.
    upload = next(step for step in package["steps"] if step.get("name") == "Upload verified Cargo package")
    assert upload["with"]["name"] == "cargo-producer-${{ github.run_id }}-${{ github.run_attempt }}"
    assert upload["with"]["retention-days"] == 30
    signer = ci["jobs"]["sign-package"]
    assert signer["needs"] == "package"
    assert signer["permissions"] == {"contents": "read", "actions": "read", "id-token": "write", "attestations": "write"}
    assert not any(step.get("uses", "").startswith(("actions/checkout@", "$/")) for step in signer["steps"])
    assert signer["steps"][-1]["with"]["name"] == "cargo-package-${{ github.run_id }}-${{ github.run_attempt }}"


@pytest.mark.parametrize("limit", ["MAX_CRATE_BYTES", "MAX_CRATE_MEMBERS", "MAX_METADATA_BYTES"])
def test_installed_archive_bounds_preserve_consumer_contract(cargo: CargoFixture, monkeypatch: pytest.MonkeyPatch, limit: str) -> None:
    monkeypatch.setattr(release_evidence, limit, 1)
    with pytest.raises(ValueError, match=r"bound|inventory"):
        promotion.verify(cargo.root)
    assert not cargo.calls


def test_signed_download_diagnostics_are_redacted(cargo: CargoFixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def failure(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.CalledProcessError(1, ["gh", "api"], stderr=b"https://example.invalid/a.zip?sig=private-signature: reset")

    monkeypatch.chdir(cargo.root)
    monkeypatch.setattr(sys, "argv", ["cargo_release.py", "verify"])
    monkeypatch.setattr(release_validation.subprocess, "run", failure)
    assert promotion.main() == 1
    captured = capsys.readouterr()
    assert "private-signature" not in captured.err
    assert "Cargo promotion failed" in captured.err
    assert not cargo.calls
