"""Exercise local ownership and actual hosted handoff code without credentials."""

import hashlib
import json
import runpy
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from research_repo_tools import files as shared_files
from research_repo_tools.release_assets import GitHubRelease, ReleaseTarget, serialize_release_target

import release_candidate as candidate

ROOT = Path(__file__).resolve().parents[2]
SHA = "a" * 40
TAG = "v0.8.5"


@pytest.fixture
def transfer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Supply explicit independent assertions and provider bytes, never timings."""
    target = ReleaseTarget(candidate.REPOSITORY, TAG, 42, SHA, TAG)
    data: dict[str, Any] = {
        "schema": 1,
        "target": json.loads(serialize_release_target(target)),
        "acquisition": "b" * 64,
        "publication": "c" * 64,
        "assets": {},
    }
    payloads = {role: (role + "\r\n").encode() + b"\xff" for role in candidate.ROLES}
    for identifier, (role, payload) in enumerate(payloads.items(), 1):
        data["assets"][role] = {"name": f"candidate-{role}.bin", "id": identifier, "sha256": hashlib.sha256(payload).hexdigest()}
    for key, value in {
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REPOSITORY": candidate.REPOSITORY,
        "GITHUB_REF": f"refs/tags/{TAG}",
        "GITHUB_SHA": SHA,
        "RELEASE_TAG": TAG,
        "GITHUB_ACTOR": "acgetchell",
        "GITHUB_TRIGGERING_ACTOR": "acgetchell",
        "GITHUB_WORKFLOW_REF": f"{candidate.REPOSITORY}/.github/workflows/release-benchmarks.yml@refs/tags/{TAG}",
    }.items():
        monkeypatch.setenv(key, value)
    calls: list[str] = []

    def release(*_args: object, **_kwargs: object) -> GitHubRelease:
        calls.append("identity")
        return GitHubRelease(
            repository=candidate.REPOSITORY,
            tag=TAG,
            identifier=data["target"]["release_id"],
            draft=True,
            immutable=False,
            prerelease=False,
            assets=tuple((item["name"], item["id"], len(payloads[role]), item["sha256"]) for role, item in data["assets"].items()),
            title=TAG,
        )

    def download(_root: Path, _repository: str, _tag: str, name: str, destination: Path, *, expected_sha256: str, **_kwargs: object) -> None:
        calls.append("download")
        role = next(role for role, item in data["assets"].items() if item["name"] == name)
        assert hashlib.sha256(payloads[role]).hexdigest() == expected_sha256
        destination.write_bytes(payloads[role])

    def upload(_root: Path, _repository: str, _tag: str, path: Path, *, target: ReleaseTarget, **_kwargs: object) -> None:
        assert target.release_id == 42
        assert path.read_bytes() == payloads["archive"]
        calls.append("upload")

    monkeypatch.setattr("research_repo_tools.release_assets.revalidate_release_target", release)
    monkeypatch.setattr("research_repo_tools.release_assets.download_release_asset", download)
    monkeypatch.setattr("research_repo_tools.release_assets.publish_release_asset", upload)
    return {"data": data, "payloads": payloads, "calls": calls, "target": target, "release": release, "download": download}


def workflow_python(tmp_path: Path, job: str) -> Path:
    """Run the exact trusted inline block; provider functions are substituted only."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/release-benchmarks.yml").read_bytes())
    step = next(item for item in workflow["jobs"][job]["steps"] if "<<'PY'" in item.get("run", ""))
    script = tmp_path / f"{job}.py"
    script.write_text(step["run"].split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0], encoding="utf-8", newline="\n")
    return script


@pytest.mark.parametrize("failure", ["none", "ref", "commit", "asset-id", "archive", "acceptance"])
def test_actual_writer_handoff_rejects_replacement_before_upload(
    transfer: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    data, payloads = transfer["data"], transfer["payloads"]
    monkeypatch.setenv("CANDIDATE_TRANSFER", json.dumps(data))
    runpy.run_path(str(workflow_python(tmp_path, "fetch")), run_name="__main__")
    assert {path.stem.split(".")[0]: path.read_bytes() for path in (tmp_path / "incoming-candidate").iterdir()} == payloads
    directory = tmp_path / "accepted-candidate"
    directory.mkdir()
    (directory / "acceptance.json").write_bytes(payloads["acceptance"])
    (directory / f"delaunay-{TAG}-criterion-baseline.tar.gz").write_bytes(payloads["archive"])
    if failure in {"ref", "commit"}:
        monkeypatch.setenv("GITHUB_REF" if failure == "ref" else "GITHUB_SHA", "changed")
    elif failure == "asset-id":
        data["assets"]["policy"]["id"] += 100
    elif failure in {"archive", "acceptance"}:
        path = directory / ("acceptance.json" if failure == "acceptance" else f"delaunay-{TAG}-criterion-baseline.tar.gz")
        path.write_bytes(path.read_bytes() + b"changed")
    script = workflow_python(tmp_path, "attach")
    if failure == "none":
        runpy.run_path(str(script), run_name="__main__")
        assert transfer["calls"].count("upload") == 1
    else:
        with pytest.raises(ValueError, match=r"dispatch differs|asset ID changed|verified artifact differs"):
            runpy.run_path(str(script), run_name="__main__")
        assert "upload" not in transfer["calls"]


@pytest.mark.parametrize("failure", ["duplicate", "missing", "extra", "path", "bool-id", "digest", "alias", "attached", "nonfinite"])
def test_transfer_parser_rejects_invalid_assertions(transfer: dict[str, Any], failure: str) -> None:
    data = transfer["data"]
    if failure == "missing":
        del data["assets"]["archive"]
    elif failure == "extra":
        data["unexpected"] = True
    elif failure == "path":
        data["assets"]["archive"]["name"] = "../archive"
    elif failure == "bool-id":
        data["assets"]["archive"]["id"] = True
    elif failure == "digest":
        data["acquisition"] = "short"
    elif failure == "alias":
        data["assets"]["archive"]["id"] = 1
    elif failure == "attached":
        data["assets"]["attached"] = {"name": f"delaunay-{TAG}-criterion-baseline.tar.gz", "id": 9, "sha256": "d" * 64}
    elif failure == "nonfinite":
        data["target"]["release_id"] = float("inf")
    payload = json.dumps(data)
    if failure == "duplicate":
        payload = payload.replace('"schema": 1', '"schema": 1, "schema": 1')
    diagnostics = {
        "duplicate": "duplicate JSON key",
        "missing": "four original assets",
        "extra": "invalid candidate transfer schema",
        "path": "safe names",
        "bool-id": "positive exact IDs",
        "digest": "full SHA-256",
        "alias": "distinct names and IDs",
        "attached": "canonical attached archive",
        "nonfinite": "nonfinite JSON number",
    }
    with pytest.raises(ValueError, match=diagnostics[failure]):
        candidate.parse_transfer(payload.encode())


def test_draft_rejects_publication_transfer_before_reading_transport(transfer: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = transfer["data"]
    data["assets"]["attached"] = {"name": f"delaunay-{TAG}-criterion-baseline.tar.gz", "id": 9, "sha256": data["assets"]["archive"]["sha256"]}
    monkeypatch.setattr(candidate, "check_release_dependencies", lambda *_args: pytest.fail("wrong transfer must stop before checking dependencies"))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="draft verification requires the original four-asset transfer"):
        candidate.receive(tmp_path / "checkout", output, json.dumps(data).encode(), source=tmp_path / "missing transport")
    assert not output.exists()


def test_transfer_variants_keep_wire_schema(transfer: dict[str, Any]) -> None:
    data = transfer["data"]
    draft = candidate.parse_transfer(json.dumps(data).encode())
    assert isinstance(draft, candidate.DraftTransfer)
    assert json.loads(candidate.serialize_transfer(draft)) == data
    data["assets"]["attached"] = {"name": f"delaunay-{TAG}-criterion-baseline.tar.gz", "id": 9, "sha256": data["assets"]["archive"]["sha256"]}
    publication = candidate.parse_transfer(json.dumps(data).encode())
    assert isinstance(publication, candidate.PublicationTransfer)
    assert publication.original == draft
    assert json.loads(candidate.serialize_transfer(publication)) == data


def test_sealing_retains_the_original_history_index(transfer: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A retry must retain the first verified index ID even when replacement bytes match."""
    root, owner = tmp_path / "checkout", tmp_path / "owner"
    root.mkdir()
    final = candidate.final_directory(owner, SHA, "reviewed")
    final.mkdir(parents=True)
    data, payloads = transfer["data"], transfer["payloads"]
    (final / "transfer.json").write_bytes(json.dumps(data).encode())
    data["assets"]["attached"] = {"name": candidate.archive_name(TAG), "id": 5, "sha256": data["assets"]["archive"]["sha256"]}
    payloads["attached"] = payloads["archive"]
    monkeypatch.setattr(candidate, "resolve_revision", lambda *_args: SHA)
    monkeypatch.setattr(candidate, "revalidate_release_target", transfer["release"])
    monkeypatch.setattr(candidate, "download_release_asset", transfer["download"])

    def upload(_root: Path, _repository: str, _tag: str, path: Path, **_kwargs: object) -> None:
        assert path.name == f"delaunay-{TAG}-candidate.json"
        payloads["index"] = path.read_bytes()
        data["assets"].setdefault("index", {"name": path.name, "id": 6, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    monkeypatch.setattr(candidate, "publish_release_asset", upload)
    publication = candidate.seal(root, owner, "reviewed").read_bytes()
    assertion = (final / "index-assertion.json").read_bytes()
    assert json.loads(assertion) == {"name": f"delaunay-{TAG}-candidate.json", "id": 6, "sha256": hashlib.sha256(publication).hexdigest()}
    assert candidate.seal(root, owner, "reviewed").read_bytes() == publication
    data["assets"]["index"]["id"] = 7
    with pytest.raises(ValueError, match=r"immutable output differs: index-assertion\.json"):
        candidate.seal(root, owner, "reviewed")
    assert (final / "index-assertion.json").read_bytes() == assertion


@pytest.mark.parametrize("key", ["GITHUB_ACTOR", "GITHUB_TRIGGERING_ACTOR", "GITHUB_EVENT_NAME", "GITHUB_WORKFLOW_REF", "GITHUB_SHA", "GITHUB_REF"])
def test_dispatch_authentication_precedes_provider_access(transfer: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    monkeypatch.setenv(key, "untrusted")
    with pytest.raises(ValueError, match="maintainer's exact workflow/tag dispatch"):
        candidate.receive(tmp_path / "checkout", tmp_path / "output", json.dumps(transfer["data"]).encode(), source=tmp_path)
    assert transfer["calls"] == []
    assert not (tmp_path / "output").exists()


def test_staging_retry_keeps_original_draft_before_any_upload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, owner = tmp_path / "checkout", tmp_path / "owner"
    root.mkdir()
    owner.mkdir()
    final = candidate.final_directory(owner, SHA, "reviewed")
    final.mkdir(parents=True)
    for name in ("policy.json", "acceptance.json", "archive.tar.gz"):
        (final / name).write_bytes(name.encode())
    for name in ("publication.json", "acquisition.sha256", "publication.sha256"):
        (owner / name).write_bytes(b"a" * 64)
    monkeypatch.setattr(candidate, "verify", lambda *_args: final / "acceptance.json")
    monkeypatch.setattr(candidate, "resolve_revision", lambda *_args: SHA)
    monkeypatch.setattr(candidate, "archive", lambda *_args: final / "archive.tar.gz")
    monkeypatch.setattr(candidate, "parse_candidate_policy", lambda *_args: SimpleNamespace(tag=TAG))
    original = ReleaseTarget(candidate.REPOSITORY, TAG, 42, SHA, TAG)
    prepared = tmp_path / "prepared-target.json"
    monkeypatch.setattr(candidate, "preparation_target", lambda *_args: prepared)
    prepared.write_bytes(serialize_release_target(original))
    monkeypatch.setattr(candidate, "revalidate_release_target", lambda *_args, **_kwargs: None)

    def interrupted(*_args: object, **_kwargs: object) -> None:
        assert (final / "target.json").read_bytes() == serialize_release_target(original)
        message = "interrupted upload"
        raise OSError(message)

    monkeypatch.setattr(candidate, "publish_release_asset", interrupted)
    with pytest.raises(OSError, match="interrupted upload"):
        candidate.stage(root, owner, "reviewed")

    def replaced(_root: Path, target: ReleaseTarget, **_kwargs: object) -> None:
        assert target == original
        message = "original draft no longer exists"
        raise ValueError(message)

    monkeypatch.setattr(candidate, "revalidate_release_target", replaced)
    monkeypatch.setattr(candidate, "publish_release_asset", lambda *_args, **_kwargs: pytest.fail("replacement draft must never receive an upload"))
    with pytest.raises(ValueError, match="original draft no longer exists"):
        candidate.stage(root, owner, "reviewed")
    prepared.write_bytes(serialize_release_target(replace(original, release_id=43)))
    with pytest.raises(ValueError, match="immutable output differs"):
        candidate.stage(root, owner, "reviewed")


def test_staging_rejects_malformed_owner_digest_before_provider_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, owner = tmp_path / "checkout", tmp_path / "owner"
    final = candidate.final_directory(owner, SHA, "reviewed")
    final.mkdir(parents=True)
    (final / "policy.json").write_bytes(b"policy")
    (final / "acceptance.json").write_bytes(b"reviewed")
    (owner / "acquisition.sha256").write_bytes(b"invalid")
    (owner / "publication.sha256").write_bytes(b"a" * 64)
    prepared = tmp_path / "target.json"
    monkeypatch.setattr(candidate, "preparation_target", lambda *_args: prepared)
    prepared.write_bytes(serialize_release_target(ReleaseTarget(candidate.REPOSITORY, TAG, 42, SHA, TAG)))
    monkeypatch.setattr(candidate, "resolve_revision", lambda *_args: SHA)
    monkeypatch.setattr(candidate, "verify", lambda *_args: pytest.fail("invalid assertions must stop before revalidation"))
    monkeypatch.setattr(candidate, "revalidate_release_target", lambda *_args, **_kwargs: pytest.fail("invalid assertions must stop before provider access"))
    with pytest.raises(ValueError, match="full SHA-256 assertions"):
        candidate.stage(root, owner, "reviewed")
    assert not (final / "target.json").exists()


def test_owner_records_reject_overwrite_and_path_escape(tmp_path: Path) -> None:
    candidate.freeze(tmp_path, {"record": b"original"})
    candidate.freeze(tmp_path, {"record": b"original"})
    with pytest.raises(ValueError, match="immutable output differs"):
        candidate.freeze(tmp_path, {"record": b"replacement", "other": b"new"})
    with pytest.raises(ValueError, match="within their record"):
        candidate.freeze(tmp_path, {"../escaped": b"new"})
    assert (tmp_path / "record").read_bytes() == b"original"
    assert not (tmp_path / "other").exists()


def test_owner_record_serializes_competing_freezes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second writer arriving after the last check cannot replace owner assertions."""
    replace_path = shared_files._replace_path
    competing: list[str] = []

    def interleave(source: Path, destination: Path) -> None:
        if destination == tmp_path / "record" and not competing:
            competing.append("started")
            with pytest.raises(ValueError, match="owner record update"):
                candidate.freeze(tmp_path, {"record": b"competing assertion"})
            competing.append("rejected")
        replace_path(source, destination)

    monkeypatch.setattr(shared_files, "_replace_path", interleave)
    candidate.freeze(tmp_path, {"record": b"original assertion"})
    assert competing == ["started", "rejected"]
    assert (tmp_path / "record").read_bytes() == b"original assertion"
    candidate.freeze(tmp_path, {"record": b"original assertion"})
    assert sorted(path.name for path in tmp_path.iterdir()) == ["record"]


def test_interrupted_owner_lock_is_preserved(tmp_path: Path) -> None:
    """An abandoned lock requires explicit reconciliation, never automatic takeover."""
    lock = tmp_path / ".candidate-owner.lock"
    lock.mkdir()
    with pytest.raises(ValueError, match="active or interrupted"):
        candidate.freeze(tmp_path, {"record": b"new assertion"})
    assert lock.is_dir()
    assert not (tmp_path / "record").exists()


def test_failed_owner_write_releases_lock_after_rollback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared rollback still owns partial writes and a caught failure permits retry."""
    candidate.freeze(tmp_path, {"original": b"retained"})
    replace_path = shared_files._replace_path

    def interrupted(source: Path, destination: Path) -> None:
        if destination == tmp_path / "second":
            message = "interrupted owner write"
            raise OSError(message)
        replace_path(source, destination)

    with monkeypatch.context() as scope:
        scope.setattr(shared_files, "_replace_path", interrupted)
        with pytest.raises(OSError, match="interrupted owner write"):
            candidate.freeze(tmp_path, {"first": b"first", "second": b"second"})
    assert sorted(path.name for path in tmp_path.iterdir()) == ["original"]
    assert (tmp_path / "original").read_bytes() == b"retained"
    candidate.freeze(tmp_path, {"first": b"first", "second": b"second"})
    assert (tmp_path / "first").read_bytes() == b"first"
    assert (tmp_path / "second").read_bytes() == b"second"


@pytest.mark.parametrize("matches", [True, False])
def test_original_draft_receipt_matches_pretag_native_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, matches: bool) -> None:
    """The retrieved original journal cannot introduce a different native acceptance."""
    root, owner = tmp_path / "checkout", tmp_path / "owner"
    final = candidate.final_directory(owner, SHA, "reviewed")
    final.mkdir(parents=True)
    native = b'{"original":"native selection"}\n'
    (final / "native.json").write_bytes(native)
    monkeypatch.setattr(candidate, "resolve_revision", lambda *_args: SHA)
    monkeypatch.setattr(candidate, "require_release_checkout", lambda *_args: None)
    monkeypatch.setattr(candidate, "release_tag", lambda *_args: TAG)
    target = serialize_release_target(ReleaseTarget(candidate.REPOSITORY, TAG, 42, SHA, TAG))

    def retrieve(arguments: list[str]) -> int:
        assert arguments[2:6] == ["release", "assets", "retrieve", "tooling/draft-receipt.toml"]
        directory = Path(arguments[-1])
        directory.mkdir()
        (directory / "target.json").write_bytes(target)
        (directory / "draft-receipt.json").write_text(
            json.dumps({"selection_sha256": hashlib.sha256(native if matches else b"replacement").hexdigest()}), encoding="utf-8"
        )
        return 0

    monkeypatch.setattr(candidate, "shared_main", retrieve)
    if matches:
        assert candidate.preparation_target(root, owner, "reviewed").read_bytes() == target
    else:
        with pytest.raises(ValueError, match="different native evidence"):
            candidate.preparation_target(root, owner, "reviewed")


@pytest.mark.parametrize("field", ["repository", "tag", "commit", "expected_title", "release_id"])
def test_initial_stage_requires_the_exact_prepared_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    root, owner = tmp_path / "checkout", tmp_path / "owner"
    root.mkdir()
    owner.mkdir()
    final = candidate.final_directory(owner, SHA, "reviewed")
    final.mkdir(parents=True)
    monkeypatch.setattr(candidate, "verify", lambda *_args: final / "acceptance.json")
    monkeypatch.setattr(candidate, "resolve_revision", lambda *_args: SHA)
    monkeypatch.setattr(candidate, "archive", lambda *_args: final / "archive.tar.gz")
    monkeypatch.setattr(candidate, "parse_candidate_policy", lambda *_args: SimpleNamespace(tag=TAG))
    (final / "policy.json").write_bytes(b"owner policy")
    (final / "acceptance.json").write_bytes(b"reviewed acceptance")
    for name in ("acquisition.sha256", "publication.sha256"):
        (owner / name).write_bytes(b"a" * 64)
    target = ReleaseTarget(candidate.REPOSITORY, TAG, 42, SHA, TAG)
    changes = {"repository": "other/repo", "tag": "v9.9.9", "commit": "f" * 40, "expected_title": "edited", "release_id": 43}
    supplied = replace(target, **{field: changes[field]})
    prepared = tmp_path / "target.json"
    monkeypatch.setattr(candidate, "preparation_target", lambda *_args: prepared)
    prepared.write_bytes(serialize_release_target(supplied))

    def original(_root: Path, observed: ReleaseTarget, **_kwargs: object) -> None:
        assert observed.release_id == 43
        message = "original prepared draft no longer exists"
        raise ValueError(message)

    monkeypatch.setattr(candidate, "revalidate_release_target", original)
    monkeypatch.setattr(candidate, "publish_release_asset", lambda *_args, **_kwargs: pytest.fail("a different prepared target must receive no uploads"))
    diagnostic = "original prepared draft no longer exists" if field == "release_id" else "staging target differs"
    with pytest.raises(ValueError, match=diagnostic):
        candidate.stage(root, owner, "reviewed")


@pytest.mark.parametrize("failure", ["none", "missing-attached", "changed-attached", "title", "mutable", "changed-acceptance", "reviewed-digest"])
def test_publication_pins_canonical_archive_and_full_acceptance(
    transfer: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    data, payloads = transfer["data"], transfer["payloads"]
    data["assets"]["attached"] = {"name": f"delaunay-{TAG}-criterion-baseline.tar.gz", "id": 9, "sha256": data["assets"]["archive"]["sha256"]}
    payloads["attached"] = payloads["archive"]
    if failure == "missing-attached":
        del data["assets"]["attached"]
        del payloads["attached"]
    assertions = json.dumps(data).encode()
    if failure == "changed-attached":
        data["assets"]["attached"]["id"] = 10
    if failure == "changed-acceptance":
        data["assets"]["acceptance"]["sha256"] = "f" * 64
    root = tmp_path / "checkout"
    root.mkdir()
    monkeypatch.setenv("GITHUB_WORKFLOW_REF", f"{candidate.REPOSITORY}/.github/workflows/publish-crates.yml@refs/tags/{TAG}")
    monkeypatch.setattr(candidate, "check_release_dependencies", lambda *_args: None)
    monkeypatch.setattr(candidate, "load_config", lambda **_kwargs: None)

    def published(*_args: object, **_kwargs: object) -> GitHubRelease:
        observed = transfer["release"]()
        assert isinstance(observed, GitHubRelease)
        return replace(observed, draft=False, immutable=failure != "mutable", title="wrong" if failure == "title" else TAG)

    monkeypatch.setattr(candidate, "check_reviewed_release", published)
    monkeypatch.setattr(candidate, "download_release_asset", transfer["download"])
    monkeypatch.setattr(candidate, "parse_candidate_policy", lambda *_args: None)
    monkeypatch.setattr(candidate, "load_evidence_policy", lambda *_args: None)
    monkeypatch.setattr(candidate, "verify_candidate_archive", lambda *_args, **_kwargs: None)
    selections: list[bytes | None] = []

    def verify(*_args: object, expected: bytes | None, **_kwargs: object) -> bytes:
        selections.append(expected)
        accepted = payloads["acceptance"]
        assert isinstance(accepted, bytes)
        return accepted

    monkeypatch.setattr(candidate, "verify_candidate_release", verify)
    output = tmp_path / "accepted"
    digest = "f" * 64 if failure == "reviewed-digest" else hashlib.sha256(payloads["acceptance"]).hexdigest()
    if failure == "none":
        archive = candidate.receive(root, output, assertions, source=candidate.ReviewedAcceptance(digest))
        assert archive.read_bytes() == payloads["archive"]
        assert (output / "acceptance.json").read_bytes() == payloads["acceptance"]
        approved = candidate.ReviewedAcceptance(digest, (output / "acceptance.json").read_bytes())
        assert candidate.receive(root, output, assertions, source=approved).read_bytes() == payloads["archive"]
        assert selections == [None, payloads["acceptance"]]
    else:
        diagnostic = {
            "missing-attached": "independently selected canonical archive",
            "changed-attached": "asset identity changed",
            "title": "release identity changed",
            "mutable": "must be immutable",
            "changed-acceptance": "asset identity changed",
            "reviewed-digest": "differs from the independently reviewed digest",
        }[failure]
        with pytest.raises(ValueError, match=diagnostic):
            candidate.receive(root, output, assertions, source=candidate.ReviewedAcceptance(digest))
        assert selections == ([None] if failure == "reviewed-digest" else [])
        assert not (output / "acceptance.json").exists()


@pytest.mark.parametrize("failure", ["missing", "malformed", "replaced"])
def test_approval_receipt_requires_independent_identity(failure: str) -> None:
    reviewed = b"owner-reviewed full acceptance\r\n"
    digest = hashlib.sha256(reviewed).hexdigest()
    selected = "" if failure == "missing" else "a" * 63 if failure == "malformed" else digest
    expected = reviewed + b"replacement" if failure == "replaced" else None
    diagnostic = "approval receipt differs" if failure == "replaced" else "independently reviewed full acceptance"
    with pytest.raises(ValueError, match=diagnostic):
        candidate.ReviewedAcceptance(selected, expected)


def test_explicit_review_selection_preserves_previous_acceptance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, owner = tmp_path / "checkout", tmp_path / "owner"
    root.mkdir()
    owner.mkdir()
    for name in ("policy.json", "acquisition.sha256", "publication.sha256"):
        (owner / name).write_bytes(b"a" * 64)
    monkeypatch.setattr(candidate, "check_release_dependencies", lambda *_args: None)
    monkeypatch.setattr(candidate, "resolve_revision", lambda *_args: SHA)
    # Binding behavior is covered by the full synthetic acquisition test.
    monkeypatch.setattr(candidate, "parse_candidate_policy", lambda *_args: ReleaseTarget(candidate.REPOSITORY, TAG, 42, SHA, TAG))
    monkeypatch.setattr(candidate, "serialize_candidate_policy", serialize_release_target)
    requests: list[list[str]] = []

    def shared(arguments: list[str]) -> int:
        requests.append(arguments)
        output = Path(arguments[arguments.index("--output") + 1])
        selected = Path(arguments[arguments.index("--policy") + 1]).parent.name
        value = json.dumps({"native": {"selection": selected}}).encode()
        if "--acceptance" in arguments:
            assert Path(arguments[arguments.index("--acceptance") + 1]).read_bytes() == value
        output.write_bytes(value)
        return 0

    monkeypatch.setattr(candidate, "shared_main", shared)
    first = candidate.verify(root, owner)
    assert candidate.verify(root, owner) == first
    second = candidate.verify(root, owner, "retry-2")
    assert json.loads(first.read_bytes())["native"] == {"selection": "reviewed"}
    assert json.loads(second.read_bytes())["native"] == {"selection": "retry-2"}
    assert ["--acceptance" in arguments for arguments in requests] == [False, True, False]


def test_renewed_upload_keeps_original_publication_assets(transfer: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A named native renewal changes only independent acceptance, never published assets."""
    root, owner = tmp_path / "checkout", tmp_path / "owner"
    root.mkdir()
    original = candidate.require_draft_transfer(candidate.parse_transfer(json.dumps(transfer["data"]).encode()))
    published = candidate.PublicationTransfer(original, candidate.AssetAssertion(candidate.archive_name(TAG), 5, original.assets.archive.sha256))
    final = candidate.final_directory(owner, SHA, "reviewed")
    final.mkdir(parents=True)
    publication = candidate.serialize_transfer(published)
    (final.parent / "publication-transfer.json").write_bytes(publication)
    original_acceptance = b"original reviewed acceptance"
    (final / "acceptance.json").write_bytes(original_acceptance)
    renewed = candidate.final_directory(owner, SHA, "native-renewal")
    renewed.mkdir()
    renewed_acceptance = b"separately reviewed native renewal"
    (renewed / "acceptance.json").write_bytes(renewed_acceptance)
    monkeypatch.setattr(candidate, "resolve_revision", lambda *_args: SHA)
    monkeypatch.setattr(candidate, "require_release_checkout", lambda *_args: None)
    monkeypatch.setattr(candidate, "release_tag", lambda *_args: TAG)
    monkeypatch.setattr(candidate, "verify_existing", lambda _root, _owner, selection: candidate.final_directory(owner, SHA, selection) / "acceptance.json")
    checked: list[candidate.PublicationTransfer] = []
    monkeypatch.setattr(candidate, "check_published_transfer", lambda _root, observed: checked.append(observed))

    def dispatch(executable: str, arguments: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert executable == "gh"
        assert arguments[:7] == ["workflow", "run", "publish-crates.yml", "--repo", candidate.REPOSITORY, "--ref", TAG]
        selected = next(argument.removeprefix("candidate=@") for argument in arguments if argument.startswith("candidate=@"))
        assert Path(selected).read_bytes() == publication
        assert arguments[-2:] == ["-f", f"acceptance={hashlib.sha256(renewed_acceptance).hexdigest()}"]
        return subprocess.CompletedProcess([executable, *arguments], 0, "", "")

    monkeypatch.setattr(candidate, "run_command", dispatch)
    candidate.upload_dispatch(root, owner, "native-renewal", TAG)
    assert checked == [published]
    assert (final / "acceptance.json").read_bytes() == original_acceptance
    assert (final.parent / "publication-transfer.json").read_bytes() == publication
