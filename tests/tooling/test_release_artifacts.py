"""Exercise Cargo promotion with installed shared verification and real archives."""

import gzip
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml
from research_repo_tools import release_validation

import release_artifacts as promotion

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "a" * 40
VERSION = "0.8.2"
NAME = f"delaunay-{VERSION}.crate"
TOOLS = {"cargo": "cargo 1.99.1", "rustc": "rustc 1.99.1\nhost: x86_64-unknown-linux-gnu"}


def crate_bytes(*, dirty: bool = False, commit: str = COMMIT, version: str = VERSION) -> bytes:
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
    manifest: dict[str, Any]
    archive: bytes
    event: dict[str, Any]
    calls: list[list[str]] = field(default_factory=list)
    changed_package: bool = False
    rerun_during_packaging: bool = False
    missing_artifact: bool = False
    damaged_zip: bool = False

    def payload(self) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            # Fixed ZIP timestamps keep immutable identity independent of the clock.
            archive.writestr(zipfile.ZipInfo(NAME), self.archive)
            archive.writestr(zipfile.ZipInfo(promotion.MANIFEST), json.dumps(self.manifest))
        return buffer.getvalue()

    def read(self, endpoint: str) -> dict[str, Any]:
        if endpoint == "actions/workflows/ci.yml":
            return {"id": 7, "path": ".github/workflows/ci.yml", "name": "CI", "state": "active"}
        if endpoint.startswith("actions/workflows/7/runs?"):
            return {"workflow_runs": [self.run]}
        if "/jobs?" in endpoint:
            return {"jobs": self.jobs}
        if "/artifacts?" in endpoint:
            payload = self.payload()
            self.artifact.update(digest=f"sha256:{hashlib.sha256(payload).hexdigest()}", size_in_bytes=len(payload))
            return {"artifacts": [] if self.missing_artifact else [self.artifact]}
        raise AssertionError(endpoint)

    def command(self, executable: str, args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        self.calls.append([executable, *args])
        if executable == "cargo" and args[0] == "package":
            assert args[1:8] == ["--locked", "--package", "delaunay", "--registry", "crates-io", "--no-verify", "--target-dir"]
            destination = Path(args[8]) / "package" / NAME
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"changed archive" if self.changed_package else self.archive)
            if self.rerun_during_packaging:
                self.run["run_attempt"] = 2
            result = ""
        elif executable == "cargo" and args[0] == "metadata":
            result = json.dumps({"target_directory": str(self.root / "target")})
        elif args == ["--version"]:
            result = TOOLS["cargo"]
        else:
            assert (executable, args) == ("rustc", ["--version", "--verbose"])
            result = TOOLS["rustc"]
        return subprocess.CompletedProcess([executable, *args], 0, stdout=result, stderr="")


@pytest.fixture
def cargo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CargoFixture:
    (tmp_path / "Cargo.toml").write_text(f'[package]\nname = "delaunay"\nversion = "{VERSION}"\n', encoding="utf-8")
    repository = {"id": 99, "full_name": promotion.REPOSITORY}
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
            "steps": [{"name": step, "status": "completed", "conclusion": "success"} for step in steps],
        }
        for identifier, (name, steps) in enumerate(promotion.POLICY.required_steps, 1)
    ]
    artifact = {
        "id": 456,
        "name": "cargo-package-1",
        "expired": False,
        "workflow_run": {"id": 123, "head_sha": COMMIT, "head_branch": "main", "repository_id": 99, "head_repository_id": 99},
    }
    archive = crate_bytes()
    manifest = {
        "schema": 1,
        "repository": promotion.REPOSITORY,
        "commit": COMMIT,
        "version": VERSION,
        "run_id": 123,
        "run_attempt": 1,
        "file": NAME,
        "sha256": hashlib.sha256(archive).hexdigest(),
        "tools": TOOLS.copy(),
    }
    event = {"action": "published", "release": {"id": 789, "tag_name": f"v{VERSION}", "draft": False, "prerelease": False}}
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event), encoding="utf-8")
    fixture = CargoFixture(tmp_path, run, jobs, artifact, manifest, archive, event)
    monkeypatch.setattr(promotion, "_git", lambda _root, *args: COMMIT if args == ("rev-parse", "HEAD") else "")
    monkeypatch.setattr(promotion, "_read", fixture.read)
    monkeypatch.setattr(promotion, "run_command", fixture.command)
    for key, value in {
        "GITHUB_REPOSITORY": promotion.REPOSITORY,
        "GITHUB_EVENT_NAME": "release",
        "GITHUB_SHA": COMMIT,
        "GITHUB_REF": f"refs/tags/v{VERSION}",
        "GITHUB_EVENT_PATH": str(event_path),
    }.items():
        monkeypatch.setenv(key, value)

    def download(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        # Only the transport is modeled: the installed v0.2.3 helper verifies
        # the actual ZIP digest, exact inventory and byte-preserving extraction.
        assert args == ["gh", "api", f"repos/{promotion.REPOSITORY}/actions/artifacts/456/zip"]
        kwargs["stdout"].write(b"corrupt ZIP" if fixture.damaged_zip else fixture.payload())
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(release_validation.subprocess, "run", download)
    return fixture


def test_promotion_and_frozen_revalidation_use_installed_helpers(cargo: CargoFixture) -> None:
    selected = promotion.verify(cargo.root)
    assert selected["artifact"]["id"] == 456
    assert selected["jobs"] == {job["name"]: job["id"] for job in cargo.jobs}
    assert promotion.verify(cargo.root, expected=json.loads(json.dumps(selected))) == selected
    assert [call[1] for call in cargo.calls if call[0] == "cargo"] == ["--version", "package", "--version", "package"]


@pytest.mark.parametrize("failure", ["none", "empty", "malformed", "changed", "unavailable"])
def test_cli_hands_frozen_evidence_to_approval_or_fails_without_output(cargo: CargoFixture, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    """Round-trip the actual Actions output, including lost or invalid handoffs."""
    monkeypatch.chdir(cargo.root)
    monkeypatch.setattr(sys, "argv", ["release_artifacts.py", "verify"])
    monkeypatch.delenv("PROMOTION_SELECTION", raising=False)
    output = cargo.root / "verify output"
    output.write_bytes(b"existing=kept\r\n")
    summary = cargo.root / "verify summary"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert promotion.main() == 0
    records = output.read_text(encoding="utf-8").splitlines()
    assert len(records) == 2
    assert output.read_bytes().startswith(b"existing=kept\r\nselection=")
    key, frozen = records[1].split("=", 1)
    assert key == "selection"
    selected = json.loads(frozen)
    assert (selected["commit"], selected["run_id"], selected["run_attempt"], selected["artifact"]["id"]) == (COMMIT, 123, 1, 456)
    assert frozen in summary.read_text(encoding="utf-8")

    if failure == "empty":
        frozen = ""
    elif failure == "malformed":
        frozen = "null"
    elif failure == "changed":
        selected["run_id"] += 1
        frozen = json.dumps(selected)
    elif failure == "unavailable":
        cargo.missing_artifact = True
    cargo.calls.clear()
    approved_output = cargo.root / "approved output"
    approved_summary = cargo.root / "approved summary"
    monkeypatch.setenv("GITHUB_OUTPUT", str(approved_output))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(approved_summary))
    monkeypatch.setenv("PROMOTION_SELECTION", frozen)
    assert promotion.main() == (0 if failure == "none" else 1)
    if failure == "none":
        assert approved_output.read_bytes() == output.read_bytes().removeprefix(b"existing=kept\r\n")
    else:
        assert not approved_output.exists()
        assert not approved_summary.exists()
        assert not any(call[1] == "package" for call in cargo.calls)


@pytest.mark.parametrize("destination", ["missing", "directory"])
def test_cli_output_export_failure_cannot_report_success(
    cargo: CargoFixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], destination: str
) -> None:
    monkeypatch.chdir(cargo.root)
    monkeypatch.setattr(sys, "argv", ["release_artifacts.py", "verify"])
    monkeypatch.delenv("PROMOTION_SELECTION", raising=False)
    output = cargo.root / "github output"
    summary = cargo.root / "github summary"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    if destination == "missing":
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    else:
        output.mkdir()
        monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    assert promotion.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Cargo promotion failed:" in captured.err
    assert not output.is_file()
    assert not summary.exists()


def test_artifact_fixture_identity_does_not_change_with_the_clock(cargo: CargoFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = cargo.payload()
    monkeypatch.setattr(time, "time", lambda: 946684800.0)
    assert cargo.payload() == payload


@pytest.mark.parametrize("field", ["event", "head_sha", "head_branch", "workflow_id", "path", "name", "repository", "head_repository", "conclusion"])
def test_untrusted_or_failed_main_run_never_repackages(cargo: CargoFixture, field: str) -> None:
    cargo.run[field] = {"full_name": "other/repo", "id": 8} if "repository" in field else "wrong"
    with pytest.raises(ValueError, match=r"repository|identity|main-push|successful|positive integer"):
        promotion.verify(cargo.root)
    assert not cargo.calls


@pytest.mark.parametrize("job_index", range(4))
@pytest.mark.parametrize("failure", ["missing", "failed", "partial-rerun", "skipped-step"])
def test_every_native_and_package_gate_is_required(cargo: CargoFixture, job_index: int, failure: str) -> None:
    job = cargo.jobs[job_index]
    if failure == "missing":
        cargo.jobs.remove(job)
    elif failure == "failed":
        job["conclusion"] = "failure"
    elif failure == "partial-rerun":
        job["run_attempt"] = 2
    else:
        job["steps"][0]["conclusion"] = "skipped"
    with pytest.raises(ValueError, match=r"attempt|succeed|step"):
        promotion.verify(cargo.root)
    assert not cargo.calls


@pytest.mark.parametrize("failure", ["missing", "expired", "wrong-run", "wrong-commit", "wrong-repository", "damaged-zip"])
def test_immutable_artifact_failures_block_repackaging(cargo: CargoFixture, failure: str) -> None:
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


@pytest.mark.parametrize("field", ["commit", "version", "run_id", "run_attempt", "repository", "sha256", "tools", "file"])
def test_producer_manifest_must_match_actual_source_bytes_and_tools(cargo: CargoFixture, field: str) -> None:
    cargo.manifest[field] = "wrong"
    with pytest.raises(ValueError, match="manifest differs"):
        promotion.verify(cargo.root)
    assert not any(call[1] == "package" for call in cargo.calls)


@pytest.mark.parametrize("failure", ["package", "rerun", "frozen-selection"])
def test_changes_during_promotion_or_approval_cannot_switch_sources(cargo: CargoFixture, failure: str) -> None:
    selected = promotion.verify(cargo.root)
    cargo.calls.clear()
    cargo.changed_package = failure == "package"
    cargo.rerun_during_packaging = failure == "rerun"
    if failure == "frozen-selection":
        cargo.artifact["id"] = 987
    with pytest.raises(ValueError, match=r"differs|attempt|selection changed"):
        promotion.verify(cargo.root, expected=selected)
    assert all("publish" not in call for call in cargo.calls)


@pytest.mark.parametrize("kwargs", [{"dirty": True}, {"commit": "b" * 40}, {"version": "99.0.0"}])
def test_package_metadata_cannot_claim_another_source(cargo: CargoFixture, kwargs: dict[str, Any]) -> None:
    package = cargo.root / NAME
    package.write_bytes(crate_bytes(**kwargs))
    with pytest.raises((ValueError, KeyError)):
        promotion._check_crate(package, COMMIT, VERSION)


def test_nested_crate_expansion_is_bounded_before_tar_parsing(cargo: CargoFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    # A small, valid ZIP can contain a gzip member with enormous expanded data.
    # Lower the bound for this real compressed fixture rather than allocating a bomb.
    payload = gzip.decompress(cargo.archive)
    monkeypatch.setattr(promotion, "MAX_CRATE_BYTES", len(payload))
    cargo.archive = gzip.compress(payload + b"\0", mtime=0)
    cargo.manifest["sha256"] = hashlib.sha256(cargo.archive).hexdigest()
    with pytest.raises(ValueError, match="expanded size"):
        promotion.verify(cargo.root)
    assert not any(call[1] == "package" for call in cargo.calls)


@pytest.mark.parametrize("limit", ["MAX_CRATE_MEMBERS", "MAX_METADATA_BYTES"])
def test_nested_crate_inventory_and_metadata_are_bounded(cargo: CargoFixture, monkeypatch: pytest.MonkeyPatch, limit: str) -> None:
    monkeypatch.setattr(promotion, limit, 1)
    with pytest.raises(ValueError, match=r"inventory|metadata size"):
        promotion.verify(cargo.root)
    assert not any(call[1] == "package" for call in cargo.calls)


def test_nested_crate_at_the_expansion_and_inventory_limits_is_accepted(cargo: CargoFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(promotion, "MAX_CRATE_BYTES", len(gzip.decompress(cargo.archive)))
    monkeypatch.setattr(promotion, "MAX_CRATE_MEMBERS", 3)
    assert promotion.verify(cargo.root)["artifact"]["id"] == 456


@pytest.mark.parametrize("name", ["src/./lib.rs", "src//lib.rs", "src/../lib.rs", "src\\lib.rs", ".cargo_vcs_info.json"])
def test_nested_crate_rejects_aliased_and_duplicate_paths(cargo: CargoFixture, name: str) -> None:
    payload = io.BytesIO(gzip.decompress(cargo.archive))
    with tarfile.open(fileobj=payload, mode="a") as archive:
        archive.addfile(tarfile.TarInfo(f"delaunay-{VERSION}/{name}"))
    cargo.archive = gzip.compress(payload.getvalue(), mtime=0)
    cargo.manifest["sha256"] = hashlib.sha256(cargo.archive).hexdigest()
    with pytest.raises(ValueError, match="unique regular package files"):
        promotion.verify(cargo.root)
    assert not any(call[1] == "package" for call in cargo.calls)


@pytest.mark.parametrize("failure", ["command", "timeout"])
def test_cli_redacts_signed_download_urls_on_failure(
    cargo: CargoFixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failure: str
) -> None:
    def download_failure(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        detail = b'Get "https://example.invalid/archive.zip?sig=private-signature&se=expiry": connection reset'
        if failure == "command":
            raise subprocess.CalledProcessError(1, ["gh", "api"], stderr=detail)
        raise subprocess.TimeoutExpired(["gh", "api"], 120, stderr=detail)

    monkeypatch.chdir(cargo.root)
    monkeypatch.setattr(sys, "argv", ["release_artifacts.py", "verify"])
    monkeypatch.setattr(release_validation.subprocess, "run", download_failure)
    assert promotion.main() == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "connection reset" in output.err
    assert "https://example.invalid/archive.zip?<redacted>" in output.err
    assert "private-signature" not in output.err
    assert "se=expiry" not in output.err
    assert not any(call[1] == "package" for call in cargo.calls)


@pytest.mark.parametrize("variable", ["GITHUB_EVENT_NAME", "GITHUB_REF", "GITHUB_SHA", "GITHUB_REPOSITORY"])
def test_release_event_mismatches_are_rejected(cargo: CargoFixture, monkeypatch: pytest.MonkeyPatch, variable: str) -> None:
    monkeypatch.setenv(variable, "wrong")
    with pytest.raises(ValueError, match="release event"):
        promotion.verify(cargo.root)
    assert not cargo.calls


def test_producer_retains_archive_and_roundtrips_manifest(cargo: CargoFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in {"GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/main", "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1"}.items():
        monkeypatch.setenv(key, value)
    packaged = cargo.root / "target" / "package" / "tmp-crate" / NAME
    packaged.parent.mkdir(parents=True)
    packaged.write_bytes(cargo.archive)
    registry = cargo.root / "target" / "package" / "tmp-registry" / NAME
    registry.parent.mkdir(parents=True)
    registry.write_bytes(cargo.archive)
    output = cargo.root / "retained"
    promotion.create(cargo.root, output)
    assert (output / NAME).read_bytes() == cargo.archive
    assert json.loads((output / promotion.MANIFEST).read_bytes()) == cargo.manifest
    with pytest.raises(FileExistsError):
        promotion.create(cargo.root, output)


def test_producer_rejects_mismatched_dry_run_archives(cargo: CargoFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in {"GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/main", "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1"}.items():
        monkeypatch.setenv(key, value)
    packages = cargo.root / "target" / "package"
    for directory, content in (("tmp-crate", cargo.archive), ("tmp-registry", b"stale")):
        destination = packages / directory / NAME
        destination.parent.mkdir(parents=True)
        destination.write_bytes(content)
    with pytest.raises(ValueError, match="archives differ"):
        promotion.create(cargo.root, cargo.root / "retained")
    assert not (cargo.root / "retained").exists()


def test_producer_rejects_local_release_or_pr_events(cargo: CargoFixture) -> None:
    with pytest.raises(ValueError, match="trusted main push"):
        promotion.create(cargo.root, cargo.root / "retained")
    assert not (cargo.root / "retained").exists()


def test_workflows_bind_full_native_matrix_and_frozen_approval_selection() -> None:
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_bytes())
    package = ci["jobs"]["package"]
    assert package["needs"] == "build"
    assert "github.event_name == 'push'" in package["if"]
    assert "github.ref == 'refs/heads/main'" in package["if"]
    steps = {step.get("name"): step for step in package["steps"]}
    assert steps["Verify clean default and all-feature Cargo packages"]["run"] == "just release-package"
    assert steps["Upload verified Cargo package"]["with"]["retention-days"] == 30
    assert steps["Upload verified Cargo package"]["with"]["name"] == "cargo-package-${{ github.run_attempt }}"
    for job_name, required_steps in promotion.POLICY.required_steps:
        job = ci["jobs"]["build" if job_name.startswith("build (") else job_name]
        assert set(required_steps) <= {step.get("name") for step in job["steps"]}
    workflow = yaml.safe_load((ROOT / ".github/workflows/publish-crates.yml").read_bytes())
    verify = workflow["jobs"]["verify"]
    assert verify["outputs"]["selection"] == "${{ steps.promotion.outputs.selection }}"
    assert verify["steps"][-1]["run"] == "just release-artifact verify"
    publish = workflow["jobs"]["publish"]
    assert publish["environment"] == "crates-io"
    approval = next(step for step in publish["steps"] if "PROMOTION_SELECTION" in step.get("env", {}))
    assert approval["env"]["PROMOTION_SELECTION"] == "${{ needs.verify.outputs.selection }}"
    assert publish["steps"].index(approval) < next(index for index, step in enumerate(publish["steps"]) if step.get("id") == "auth")
    assert "release-package" not in yaml.safe_dump(workflow)
