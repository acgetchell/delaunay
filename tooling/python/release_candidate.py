"""Delaunay acquisition ownership and explicit release handoffs.

The installed RRT APIs own evidence, candidate, archive and provider validation.
This adapter owns the local record selection and maintainer dispatch contract.
Based on la-stack's release_candidate orchestration at a0fa01fad73a01a903ef7899289067023291d4ed;
Delaunay retains its independent Cargo archive and reproduction guarantee.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Literal

from research_repo_tools.candidate_publication import plan_candidate_publication
from research_repo_tools.candidate_reuse import (
    CandidatePolicy,
    acquisition_assertion_digest,
    parse_candidate_policy,
    plan_candidate_archive,
    serialize_candidate_policy,
    verify_candidate_archive,
    verify_candidate_release,
)
from research_repo_tools.cli import main as shared_main
from research_repo_tools.common_measurement import validate_run_policy
from research_repo_tools.complete_runs import run_identity
from research_repo_tools.config import load as load_config
from research_repo_tools.dependency_provenance import check_release_dependencies
from research_repo_tools.evidence import Evidence, deterministic_json, parse_evidence
from research_repo_tools.files import replace_many
from research_repo_tools.measurement import load_measurement, resolve_revision
from research_repo_tools.performance_progress import ProgressReporter
from research_repo_tools.performance_release import load_performance_release, load_preparation, measure_release, serialize_preparation
from research_repo_tools.process import run_command
from research_repo_tools.publication import publish_publication
from research_repo_tools.release_assets import (
    GitHubRelease,
    ReleasePublicationUnknownError,
    ReleaseTarget,
    download_release_asset,
    lookup_release,
    parse_release_target,
    publish_release_asset,
    require_release_checkout,
    revalidate_release_target,
    serialize_release_target,
)
from research_repo_tools.release_interfaces import load_evidence_policy
from research_repo_tools.release_metadata import read_package_info
from research_repo_tools.release_publishing import ReviewedRelease, check_reviewed_release
from research_repo_tools.release_validation import api, parse_release_json
from research_repo_tools.scheduled_runs import ScheduledRun
from research_repo_tools.suite_runs import parse_suite_run

SETTINGS = "tooling/performance-release.toml"
REPOSITORY = "acgetchell/delaunay"
ROLES = ("policy", "publication", "acceptance", "archive")
type AssetRole = Literal["policy", "publication", "acceptance", "archive", "attached"]


def release_tag(root: Path) -> str:
    """Derive the canonical release identity from the observed package."""
    tag = "v" + read_package_info(root).version
    if re.fullmatch(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", tag) is None:
        message = "release context requires a canonical stable package version"
        raise ValueError(message)
    return tag


def selection_directory(root: Path) -> Path:
    """Address one original selection by checkout and tag, never by latest time."""
    checkout = hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()
    directory = Path.home() / ".local/state/delaunay/releases" / checkout / release_tag(root)
    return owner_directory(root, "", directory)


def owner_directory(root: Path, checkpoint: str, selected: Path | None = None) -> Path:
    """Keep the operator's immutable record outside candidate-controlled files."""
    identity = hashlib.sha256(str((root / checkpoint).resolve()).encode("utf-8")).hexdigest()
    directory = (selected if selected is not None else Path.home() / ".local/state/delaunay/acquisitions" / release_tag(root) / identity).expanduser().resolve()
    if directory.is_relative_to(root) or root.is_relative_to(directory):
        message = "acquisition owner directory must be separate from the checkout"
        raise ValueError(message)
    return directory


def freeze(directory: Path, files: dict[str, bytes]) -> None:
    """Serialize owner-record updates; retries cannot replace earlier bytes."""
    directory = directory.resolve(strict=True)
    updates = {directory / name: data for name, data in files.items()}
    if any(not path.resolve().is_relative_to(directory) for path in updates):
        message = "owner files must remain within their record directory"
        raise ValueError(message)

    def unchanged() -> None:
        for path, data in updates.items():
            if path.exists() and path.read_bytes() != data:
                raise ValueError(f"immutable output differs: {path.name}")

    # Shared replacement is failure-atomic but does not serialize writers.
    # Atomic directory creation works on every supported host and fails closed
    # after an unclean termination; never infer that an existing lock is stale.
    lock = directory / ".candidate-owner.lock"
    try:
        lock.mkdir(mode=0o700)
    except FileExistsError as error:
        raise ValueError(f"owner record update is active or interrupted: {lock}; inspect the original operation before removing its lock") from error
    try:
        unchanged()
        replace_many(updates, verify=unchanged)
    finally:
        lock.rmdir()


def remember_owner(root: Path, owner: Path) -> None:
    """Register only original Step 1 assertions, before report promotion."""
    policy = parse_candidate_policy((owner / "policy.json").read_bytes())
    if policy.tag != release_tag(root):
        message = "acquisition tag differs from the package release context"
        raise ValueError(message)
    directory = selection_directory(root)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    record = {
        "schema": 1,
        "repository": REPOSITORY,
        "checkout": str(root.resolve()),
        "tag": policy.tag,
        "owner": str(owner.resolve()),
        "acquisition": (owner / "acquisition.sha256").read_text(encoding="ascii").strip(),
        "publication": (owner / "publication.sha256").read_text(encoding="ascii").strip(),
    }
    freeze(directory, {"owner.json": json.dumps(record, sort_keys=True).encode("utf-8") + b"\n"})


def selected_owner(root: Path, explicit: str = "") -> Path:
    """Resolve the original external selection; explicit recovery never replaces it."""
    if explicit:
        return owner_directory(root, "", Path(explicit))
    path = selection_directory(root) / "owner.json"
    try:
        data = parse_release_json(path.read_bytes())
    except FileNotFoundError as error:
        message = "no original Step 1 owner selection; restore its external owner.json or pass the original owner directory explicitly"
        raise ValueError(message) from error
    fields = {"schema", "repository", "checkout", "tag", "owner", "acquisition", "publication"}
    if (
        not isinstance(data, dict)
        or set(data) != fields
        or type(data["schema"]) is not int
        or data["schema"] != 1
        or (data["repository"], data["checkout"], data["tag"]) != (REPOSITORY, str(root.resolve()), release_tag(root))
        or not isinstance(data["owner"], str)
        or not Path(data["owner"]).is_absolute()
    ):
        message = "original owner selection differs from this checkout and release"
        raise ValueError(message)
    owner = owner_directory(root, "", Path(data["owner"]))
    for field in ("acquisition", "publication"):
        digest = data[field]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            message = "owner selection requires original SHA-256 assertions"
            raise ValueError(message)
        if (owner / f"{field}.sha256").read_text(encoding="ascii").strip() != digest:
            message = "original owner assertions changed; restore the retained selection"
            raise ValueError(message)
    policy = parse_candidate_policy((owner / "policy.json").read_bytes())
    if policy.tag != data["tag"] or acquisition_assertion_digest(policy) != data["acquisition"]:
        message = "original owner policy differs from the retained selection"
        raise ValueError(message)
    if hashlib.sha256((owner / "publication.json").read_bytes()).hexdigest() != data["publication"]:
        message = "original publication receipt differs from the retained selection"
        raise ValueError(message)
    return owner


def capture(root: Path, owner: Path, *, retained_scratch: bool = False) -> dict[str, str]:
    """Finish the trusted owner's receipt from its already retained original bytes."""
    settings = load_performance_release(root, SETTINGS)
    preparation = load_preparation(root, settings)
    if serialize_preparation(preparation) != (owner / "preparation.json").read_bytes():
        message = "independent preparation changed since acquisition"
        raise ValueError(message)
    if retained_scratch:
        # Explicit owner recovery after a failed callback; never infer this
        # authority from an incoming artifact or silently fall back to scratch.
        preparation.verify(root)
        freeze(owner, {"run.json": (root / settings.payload).read_bytes(), "evidence.json": (root / settings.manifest).read_bytes()})
    payload, manifest = (owner / "run.json").read_bytes(), (owner / "evidence.json").read_bytes()
    # Restore only identical/missing scratch evidence for a failed capture retry.
    freeze(root, {settings.payload: payload, settings.manifest: manifest})
    evidence = parse_evidence(payload, manifest)
    run = validate_run_policy(evidence, load_measurement(root, settings.measurement), release=True)
    if not isinstance(run, ScheduledRun):
        message = "candidate acquisition requires the declared trial schedule"
        raise TypeError(message)
    policy = CandidatePolicy(
        settings.repository,
        preparation.pair.current,
        None,
        "local",
        run_identity(evidence),
        run.plan,
        dict(evidence.sources)["current"],
        tuple(
            (index, suite, phase, source) for index, trial in enumerate(run.trials, 1) for suite, phase, _sample, source in parse_suite_run(trial).observations
        ),
        report=settings.report,
    )
    acquisition = acquisition_assertion_digest(policy)
    freeze(owner, {"policy.json": serialize_candidate_policy(policy), "acquisition.sha256": (acquisition + "\n").encode("ascii")})
    receipt_path = owner / "publication.json"
    if not receipt_path.exists():
        _plan, receipt = plan_candidate_publication(root, settings.payload, settings.manifest, policy, acquisition_sha256=acquisition)
        freeze(owner, {"publication.json": receipt})
    publication = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    freeze(owner, {"publication.sha256": (publication + "\n").encode("ascii")})
    return {"owner_directory": str(owner), "acquisition_sha256": acquisition, "publication_sha256": publication}


def acquire(root: Path, owner: Path, checkpoint: str, *, resume: bool, progress: ProgressReporter) -> dict[str, str]:
    """Acquire under owner control, freezing assertions before report promotion."""
    settings = load_performance_release(root, SETTINGS)
    preparation = load_preparation(root, settings)
    check_release_dependencies(root)
    if (selection_directory(root) / "owner.json").exists() and selected_owner(root) != owner.resolve():
        message = "this checkout/tag already selects a different original owner; use that record or explicitly reconcile the release before acquisition"
        raise ValueError(message)
    if resume:
        if not owner.is_dir():
            message = "resume requires the original acquisition owner record"
            raise ValueError(message)
    else:
        try:
            owner.mkdir(mode=0o700, parents=True)
        except FileExistsError as error:
            if not owner.exists():
                raise
            message = (
                f"acquisition owner record already exists: {owner}; preserve it and select a new checkpoint/owner directory, "
                "or use performance-resume if the original checkpoint still exists"
            )
            raise ValueError(message) from error
    freeze(owner, {"preparation.json": serialize_preparation(preparation)})

    def before_promotion() -> None:
        freeze(owner, {"run.json": (root / settings.payload).read_bytes(), "evidence.json": (root / settings.manifest).read_bytes()})
        capture(root, owner)
        remember_owner(root, owner)

    measure_release(
        root,
        settings,
        allow_git_mutations=False,
        checkpoint=checkpoint,
        resume=resume,
        progress=progress,
        before_promotion=before_promotion,
        release_intent=True,
    )
    return {
        "owner_directory": str(owner),
        "acquisition_sha256": (owner / "acquisition.sha256").read_text(encoding="ascii").strip(),
        "publication_sha256": (owner / "publication.sha256").read_text(encoding="ascii").strip(),
    }


def final_directory(owner: Path, commit: str, selection: str) -> Path:
    """Name an explicit immutable review slot; never replace an earlier selection."""
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", selection) is None:
        message = "final evidence requires a full commit and a short lowercase review selection name"
        raise ValueError(message)
    return owner / "final" / commit / selection


def verify(root: Path, owner: Path, selection: str = "reviewed") -> Path:
    """Bind the separately trusted record to clean HEAD and signed native CI."""
    check_release_dependencies(root)
    commit = resolve_revision(root, "HEAD")
    policy = replace(parse_candidate_policy((owner / "policy.json").read_bytes()), commit=commit)
    final = final_directory(owner, commit, selection)
    final.mkdir(parents=True, exist_ok=True)
    freeze(final, {"policy.json": serialize_candidate_policy(policy)})
    accepted = final / "acceptance.json"
    with tempfile.TemporaryDirectory(prefix="delaunay-candidate-verification-") as temporary:
        output = Path(temporary).resolve() / "acceptance.json"
        arguments = [
            "--root",
            str(root),
            "release",
            "evidence",
            "candidate",
            "tooling/release-evidence.toml",
            "--checkout",
            str(root),
            "--policy",
            str(final / "policy.json"),
            "--payload",
            str(owner / "run.json"),
            "--manifest",
            str(owner / "evidence.json"),
            "--publication-receipt",
            str(owner / "publication.json"),
            "--acquisition-sha256",
            (owner / "acquisition.sha256").read_text(encoding="ascii").strip(),
            "--publication-sha256",
            (owner / "publication.sha256").read_text(encoding="ascii").strip(),
            "--output",
            str(output),
            "--format",
            "summary",
        ]
        if accepted.exists():
            arguments.extend(("--acceptance", str(accepted)))
        status = shared_main(arguments)
        if status:
            raise RuntimeError(f"candidate verification failed with exit {status}")
        freeze(final, {"acceptance.json": output.read_bytes()})
        freeze(final, {"native.json": deterministic_json(parse_release_json(accepted.read_bytes())["native"])})
    return accepted


def verify_existing(root: Path, owner: Path, selection: str) -> Path:
    """Revalidate an already reviewed commit without silently creating a selection."""
    accepted = final_directory(owner, resolve_revision(root, "HEAD"), selection) / "acceptance.json"
    expected = accepted.read_bytes()
    if verify(root, owner, selection).read_bytes() != expected:
        message = "reviewed acceptance changed; repeat explicit release-evidence review"
        raise ValueError(message)
    return accepted


def archive(owner: Path, commit: str, selection: str = "reviewed") -> Path:
    """Derive an immutable release archive without acquisition or native work."""
    final = final_directory(owner, commit, selection)
    policy = parse_candidate_policy((final / "policy.json").read_bytes())
    if policy.commit != commit:
        message = "selected candidate policy differs from the independently selected commit"
        raise ValueError(message)
    prefix = final.relative_to(owner).as_posix()
    output = f"{prefix}/delaunay-{policy.tag}-criterion-baseline.tar.gz"
    publish_publication(plan_candidate_archive(owner, "run.json", "evidence.json", f"{prefix}/policy.json", f"{prefix}/acceptance.json", output))
    verify_candidate_archive(owner / output, policy, (final / "acceptance.json").read_bytes())
    return owner / output


@dataclass(frozen=True, slots=True)
class AssetAssertion:
    """A provider asset name, exact numeric ID, and independently selected digest."""

    name: str
    identifier: int
    sha256: str

    def __post_init__(self) -> None:
        """Validate provider values before carrying them as assertions."""
        if (
            not isinstance(self.name, str)
            or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,199}", self.name) is None
            or type(self.identifier) is not int
            or self.identifier <= 0
        ):
            message = "candidate assets require safe names and positive exact IDs"
            raise ValueError(message)
        if not isinstance(self.sha256, str) or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            message = "candidate asset requires a full SHA-256 assertion"
            raise ValueError(message)


def transfer_asset(raw: object) -> AssetAssertion:
    """Parse one exact inert asset selection."""
    if not isinstance(raw, dict) or set(raw) != {"name", "id", "sha256"}:
        message = "invalid candidate asset assertion"
        raise ValueError(message)
    return AssetAssertion(raw["name"], raw["id"], raw["sha256"])


@dataclass(frozen=True, slots=True)
class OriginalAssets:
    """Every original role is required exactly once."""

    policy: AssetAssertion
    publication: AssetAssertion
    acceptance: AssetAssertion
    archive: AssetAssertion

    def __post_init__(self) -> None:
        """Check the cross-role identity constraint."""
        assets = (self.policy, self.publication, self.acceptance, self.archive)
        if len({asset.name for asset in assets}) != len(assets) or len({asset.identifier for asset in assets}) != len(assets):
            message = "candidate assets must have distinct names and IDs"
            raise ValueError(message)

    def role_assertions(self) -> tuple[tuple[AssetRole, AssetAssertion], ...]:
        """Return the original assets with their fixed transport roles."""
        return (("policy", self.policy), ("publication", self.publication), ("acceptance", self.acceptance), ("archive", self.archive))


@dataclass(frozen=True, slots=True)
class DraftTransfer:
    """The original four-asset selection, before attachment."""

    target: ReleaseTarget
    acquisition: str
    publication: str
    assets: OriginalAssets


@dataclass(frozen=True, slots=True)
class PublicationTransfer:
    """The original selection plus an independently selected canonical attachment."""

    original: DraftTransfer
    attached: AssetAssertion

    def __post_init__(self) -> None:
        """Require a distinct attachment with the accepted archive bytes."""
        if any(self.attached.name == asset.name or self.attached.identifier == asset.identifier for _, asset in self.original.assets.role_assertions()):
            message = "candidate assets must have distinct names and IDs"
            raise ValueError(message)
        if self.attached.name != archive_name(self.original.target.tag) or self.attached.sha256 != self.original.assets.archive.sha256:
            message = "canonical attached archive must match the originally accepted bytes"
            raise ValueError(message)

    def role_assertions(self) -> tuple[tuple[AssetRole, AssetAssertion], ...]:
        """Return every asset selected for publication."""
        return (*self.original.assets.role_assertions(), ("attached", self.attached))


type Transfer = DraftTransfer | PublicationTransfer


def archive_name(tag: str) -> str:
    """Name the canonical attached benchmark archive."""
    return f"delaunay-{tag}-criterion-baseline.tar.gz"


def transfer_digest(value: object) -> str:
    """Require an exact SHA-256 assertion before preparing transport."""
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        message = "candidate transfer requires full SHA-256 assertions"
        raise ValueError(message)
    return value


def parse_transfer(payload: bytes) -> Transfer:
    """Reject malformed or ambiguous operator assertions before provider access."""

    data: object = parse_release_json(payload)
    if (
        not isinstance(data, dict)
        or set(data) != {"schema", "target", "acquisition", "publication", "assets"}
        or type(data["schema"]) is not int
        or data["schema"] != 1
    ):
        message = "invalid candidate transfer schema"
        raise ValueError(message)
    target = parse_release_target(json.dumps(data["target"]).encode("utf-8"))
    if target.repository != REPOSITORY or target.expected_title != target.tag:
        message = "candidate transfer must select this repository and its exact release title"
        raise ValueError(message)

    raw_assets = data["assets"]
    if not isinstance(raw_assets, dict) or set(raw_assets) not in (set(ROLES), {*ROLES, "attached"}):
        message = "candidate transfer requires four original assets and an optional attached archive"
        raise ValueError(message)
    assets = OriginalAssets(
        policy=transfer_asset(raw_assets["policy"]),
        publication=transfer_asset(raw_assets["publication"]),
        acceptance=transfer_asset(raw_assets["acceptance"]),
        archive=transfer_asset(raw_assets["archive"]),
    )
    original = DraftTransfer(target, transfer_digest(data["acquisition"]), transfer_digest(data["publication"]), assets)
    if "attached" in raw_assets:
        return PublicationTransfer(original, transfer_asset(raw_assets["attached"]))
    return original


def require_draft_transfer(transfer: Transfer) -> DraftTransfer:
    """Require the original four-asset dispatch selection."""
    if isinstance(transfer, DraftTransfer):
        return transfer
    message = "draft verification requires the original four-asset transfer"
    raise ValueError(message)


def require_publication_transfer(transfer: Transfer) -> PublicationTransfer:
    """Require the canonical attachment before any publication operation."""
    if isinstance(transfer, PublicationTransfer):
        return transfer
    message = "publication requires the independently selected canonical archive"
    raise ValueError(message)


def serialize_transfer(transfer: Transfer) -> bytes:
    """Preserve the schema-1 wire shape of a validated owner selection."""
    original = transfer.original if isinstance(transfer, PublicationTransfer) else transfer
    assets = transfer.role_assertions() if isinstance(transfer, PublicationTransfer) else transfer.assets.role_assertions()
    record = {
        "schema": 1,
        "target": json.loads(serialize_release_target(original.target)),
        "acquisition": original.acquisition,
        "publication": original.publication,
        "assets": {role: {"name": asset.name, "id": asset.identifier, "sha256": asset.sha256} for role, asset in assets},
    }
    return json.dumps(record, sort_keys=True).encode("utf-8") + b"\n"


def preparation_target(root: Path, owner: Path, selection: str) -> Path:
    """Retrieve the original authenticated target through the installed supplier."""
    commit = resolve_revision(root, "HEAD")
    require_release_checkout(root, commit)
    output = final_directory(owner, commit, selection) / "preparation"
    status = shared_main(
        [
            "--root",
            str(root),
            "release",
            "assets",
            "retrieve",
            "tooling/draft-receipt.toml",
            "--commit",
            commit,
            "--tag",
            release_tag(root),
            "--output",
            str(output),
        ]
    )
    if status:
        raise RuntimeError(f"original preparation retrieval failed with exit {status}")
    receipt = parse_release_json((output / "draft-receipt.json").read_bytes())
    native = (output.parent / "native.json").read_bytes()
    if receipt["selection_sha256"] != hashlib.sha256(native).hexdigest():
        message = "draft preparation selected different native evidence from the pre-tag acceptance"
        raise ValueError(message)
    return output / "target.json"


def stage(root: Path, owner: Path, selection: str) -> Path:
    """Upload inert original records, retaining dispatch assertions independently."""
    acquisition = transfer_digest((owner / "acquisition.sha256").read_text(encoding="ascii").strip())
    publication = transfer_digest((owner / "publication.sha256").read_text(encoding="ascii").strip())
    accepted = verify_existing(root, owner, selection)
    commit = resolve_revision(root, "HEAD")
    final = final_directory(owner, commit, selection)
    policy = parse_candidate_policy((final / "policy.json").read_bytes())
    target_path = final / "target.json"
    target = parse_release_target(preparation_target(root, owner, selection).read_bytes())
    if (target.repository, target.tag, target.commit, target.expected_title) != (REPOSITORY, policy.tag, commit, policy.tag):
        message = "staging target differs from the accepted candidate"
        raise ValueError(message)
    freeze(final, {"target.json": serialize_release_target(target)})
    target = parse_release_target(target_path.read_bytes())
    revalidate_release_target(root, target, checkout=root)
    artifact = archive(owner, commit, selection)
    inputs = {"policy": final / "policy.json", "publication": owner / "publication.json", "acceptance": accepted, "archive": artifact}
    transport = final / "transport"
    transport.mkdir(exist_ok=True)
    assertions: dict[str, dict[str, str | int]] = {}
    for role, source in inputs.items():
        suffix = "tar.gz" if role == "archive" else "json"
        name = f"delaunay-{policy.tag}-{selection}-candidate-{role}.{suffix}"
        payload = source.read_bytes()
        freeze(transport, {name: payload})
        publish_release_asset(root, REPOSITORY, policy.tag, transport / name, target=target, checkout=root)
        release = revalidate_release_target(root, target, checkout=root)
        entry = next((item for item in release.assets if item[0] == name), None)
        if entry is None:
            message = f"staged candidate asset is missing from the prepared release: {name}"
            raise ValueError(message)
        assertions[role] = {"name": name, "id": entry[1], "sha256": hashlib.sha256(payload).hexdigest()}
    payload = (
        json.dumps(
            {
                "schema": 1,
                "target": json.loads(serialize_release_target(target)),
                "acquisition": acquisition,
                "publication": publication,
                "assets": assertions,
            },
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    require_draft_transfer(parse_transfer(payload))
    freeze(final, {"transfer.json": payload})
    return final / "transfer.json"


def seal(root: Path, owner: Path, selection: str) -> Path:
    """Independently select the canonical attached asset before publication."""
    final = final_directory(owner, resolve_revision(root, "HEAD"), selection)
    transfer = require_draft_transfer(parse_transfer((final / "transfer.json").read_bytes()))
    release = revalidate_release_target(root, transfer.target, checkout=root)
    for _role, asset in transfer.assets.role_assertions():
        entries = [entry for entry in release.assets if entry[0] == asset.name]
        if len(entries) != 1 or entries[0][1] != asset.identifier or (entries[0][3] is not None and entries[0][3] != asset.sha256):
            message = "original staged candidate asset identity changed"
            raise ValueError(message)
    name = archive_name(transfer.target.tag)
    entry = next((item for item in release.assets if item[0] == name), None)
    if entry is None:
        message = "canonical accepted benchmark archive is not yet attached"
        raise ValueError(message)
    digest = transfer.assets.archive.sha256
    publication = PublicationTransfer(transfer, AssetAssertion(name, entry[1], digest))
    with tempfile.TemporaryDirectory(prefix="delaunay-seal-") as temporary:
        download_release_asset(root, REPOSITORY, transfer.target.tag, name, Path(temporary) / "archive.tar.gz", expected_sha256=digest, allow_draft=True)
    if entry not in revalidate_release_target(root, transfer.target, checkout=root).assets:
        message = "attached archive identity changed during verification"
        raise ValueError(message)
    freeze(final, {"publication-transfer.json": serialize_transfer(publication)})
    freeze(final.parent, {"publication-transfer.json": serialize_transfer(publication)})
    # Published readers authenticate this index through the immutable release.
    # Promotion still uses the independently retained local/dispatch assertion.
    index = f"delaunay-{transfer.target.tag}-candidate.json"
    freeze(final, {index: serialize_transfer(publication)})
    publish_release_asset(root, REPOSITORY, transfer.target.tag, final / index, target=transfer.target, checkout=root)
    release = revalidate_release_target(root, transfer.target, checkout=root)
    entry = next((item for item in release.assets if item[0] == index), None)
    if entry is None:
        message = "published-history index is missing from the prepared release"
        raise ValueError(message)
    index_digest = hashlib.sha256(serialize_transfer(publication)).hexdigest()
    with tempfile.TemporaryDirectory(prefix="delaunay-index-seal-") as temporary:
        download_release_asset(root, REPOSITORY, transfer.target.tag, index, Path(temporary) / "index.json", expected_sha256=index_digest, allow_draft=True)
    if entry not in revalidate_release_target(root, transfer.target, checkout=root).assets:
        message = "published-history index identity changed during verification"
        raise ValueError(message)
    freeze(final, {"index-assertion.json": deterministic_json({"name": index, "id": entry[1], "sha256": index_digest})})
    return final / "publication-transfer.json"


def read_published_archive(root: Path, tag: str, commit: str, archive: Path, directory: Path) -> Evidence:
    """Read a published v2 archive without fabricating legacy metadata or final-source measurements.

    The immutable published release authenticates historical assets. This is an
    inspection API, never a replacement for independent pre-publication approval.
    Native CI need not remain available forever to inspect a published history.
    """
    release = lookup_release(root, REPOSITORY, tag)
    if release.draft or release.prerelease or not release.immutable:
        message = "candidate history requires an immutable stable published release"
        raise ValueError(message)
    index = directory / "candidate.json"
    download_release_asset(root, REPOSITORY, tag, f"delaunay-{tag}-candidate.json", index)
    transfer = require_publication_transfer(parse_transfer(index.read_bytes()))
    target = transfer.original.target
    if (target.repository, target.tag, target.commit, target.release_id, target.expected_title) != (REPOSITORY, tag, commit, release.identifier, release.title):
        message = "published candidate differs from the independently requested tag/commit/release"
        raise ValueError(message)
    for _role, asset in transfer.role_assertions():
        entries = [entry for entry in release.assets if entry[0] == asset.name]
        if len(entries) != 1 or entries[0][1] != asset.identifier or (entries[0][3] is not None and entries[0][3] != asset.sha256):
            message = "published candidate asset identity differs from its immutable index"
            raise ValueError(message)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != transfer.attached.sha256:
        message = "downloaded archive differs from the published candidate selection"
        raise ValueError(message)
    for role, asset in (("policy", transfer.original.assets.policy), ("acceptance", transfer.original.assets.acceptance)):
        download_release_asset(root, REPOSITORY, tag, asset.name, directory / f"{role}.json", expected_sha256=asset.sha256)
    policy = parse_candidate_policy((directory / "policy.json").read_bytes())
    if acquisition_assertion_digest(policy) != transfer.original.acquisition:
        message = "published policy differs from the original acquisition assertion"
        raise ValueError(message)
    evidence = verify_candidate_archive(archive, policy, (directory / "acceptance.json").read_bytes(), target=target)
    if lookup_release(root, REPOSITORY, tag) != release:
        message = "immutable release changed during archive inspection"
        raise ValueError(message)
    return evidence


def benchmark_dispatch(root: Path, owner: Path, selection: str, tag: str) -> None:
    """Stage verified inert records and explicitly authorize their exact-tag transfer."""
    if tag != release_tag(root):
        message = "benchmark dispatch differs from the package release"
        raise ValueError(message)
    transfer = stage(root, owner, selection)
    with tempfile.TemporaryDirectory(prefix="delaunay-benchmark-dispatch-") as temporary:
        snapshot = Path(temporary) / "transfer.json"
        snapshot.write_bytes(transfer.read_bytes())
        run_command(
            "gh",
            ["workflow", "run", "release-benchmarks.yml", "--repo", REPOSITORY, "--ref", tag, "-f", f"tag={tag}", "-F", f"candidate=@{snapshot}"],
            cwd=root,
        )


def publish_draft(root: Path, owner: Path, selection: str, tag: str) -> None:
    """Revalidate the original acceptance and canonical attachment before publication."""
    if tag != release_tag(root):
        message = "publication tag differs from the package release"
        raise ValueError(message)
    accepted = verify_existing(root, owner, selection).read_bytes()
    preparation_target(root, owner, selection)
    sealed = seal(root, owner, selection)
    transfer = require_publication_transfer(parse_transfer(sealed.read_bytes()))
    index = transfer_asset(parse_release_json((sealed.parent / "index-assertion.json").read_bytes()))
    target = transfer.original.target
    if index.name != f"delaunay-{tag}-candidate.json" or index.sha256 != hashlib.sha256(serialize_transfer(transfer)).hexdigest():
        message = "published-history index differs from the sealed transfer"
        raise ValueError(message)
    if hashlib.sha256(accepted).hexdigest() != transfer.original.assets.acceptance.sha256:
        message = "publication transfer differs from the reviewed acceptance"
        raise ValueError(message)
    # Sealing can take time. Recheck that same acceptance after attachment work.
    if verify_existing(root, owner, selection).read_bytes() != accepted:
        message = "reviewed acceptance changed before draft publication"
        raise ValueError(message)
    reviewed = check_reviewed_release(load_config(root=root), tag, event=ReviewedRelease(tag, target.commit, target.release_id), draft=True)
    check_transfer_assets(reviewed, transfer, index=index)
    # RRT 0.2.8's release publish command captures a new target by tag. Its asset
    # publisher can recreate a missing attachment. Neither operation preserves
    # this already sealed selection. Keep the shared read-only gates, then make
    # only the lifecycle request to the original ID, without attachment repair.
    try:
        run_command("gh", ["api", "--method", "PATCH", f"repos/{REPOSITORY}/releases/{target.release_id}", "--field", "draft=false"], cwd=root)
        check_published_transfer(root, transfer, index=index)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise ReleasePublicationUnknownError(
            f"publication outcome unknown for {REPOSITORY} release {target.release_id}; inspect the original release and assets before retrying"
        ) from error


def record_path(root: Path, owner: Path, selection: str, role: Literal["transfer", "publication"]) -> Path:
    """Locate an already retained transfer for this exact clean release commit."""
    commit = resolve_revision(root, "HEAD")
    require_release_checkout(root, commit)
    filename = "transfer.json" if role == "transfer" else "publication-transfer.json"
    final = final_directory(owner, commit, selection)
    path = (final if role == "transfer" else final.parent) / filename
    parsed = parse_transfer(path.read_bytes())
    transfer = require_draft_transfer(parsed) if role == "transfer" else require_publication_transfer(parsed).original
    if (transfer.target.tag, transfer.target.commit) != (release_tag(root), commit):
        message = "saved transfer differs from the current release checkout"
        raise ValueError(message)
    return path


def upload_dispatch(root: Path, owner: Path, selection: str, tag: str) -> None:
    """Dispatch registry publication using only the existing reviewed acceptance."""
    if tag != release_tag(root):
        message = "dispatch tag differs from the package release"
        raise ValueError(message)
    publication = record_path(root, owner, selection, "publication")
    transfer_bytes = publication.read_bytes()
    transfer = require_publication_transfer(parse_transfer(transfer_bytes))
    accepted = final_directory(owner, resolve_revision(root, "HEAD"), selection) / "acceptance.json"
    expected = accepted.read_bytes()  # Missing review must never create a new selection.
    if verify_existing(root, owner, selection).read_bytes() != expected:
        message = "reviewed acceptance changed before publication dispatch"
        raise ValueError(message)
    check_published_transfer(root, transfer)
    if publication.read_bytes() != transfer_bytes or accepted.read_bytes() != expected:
        message = "publication assertions changed before dispatch"
        raise ValueError(message)
    # Snapshot the same original bytes sent by the prior explicit CLI procedure.
    with tempfile.TemporaryDirectory(prefix="delaunay-publication-dispatch-") as temporary:
        snapshot = Path(temporary) / "publication-transfer.json"
        snapshot.write_bytes(transfer_bytes)
        run_command(
            "gh",
            [
                "workflow",
                "run",
                "publish-crates.yml",
                "--repo",
                REPOSITORY,
                "--ref",
                tag,
                "-f",
                f"tag={tag}",
                "-F",
                f"candidate=@{snapshot}",
                "-f",
                f"acceptance={hashlib.sha256(expected).hexdigest()}",
            ],
            cwd=root,
        )


def require_dispatch(transfer: Transfer) -> None:
    """Bind independent workflow input to the allowed operator, workflow and tag."""
    if isinstance(transfer, PublicationTransfer):
        workflow = "publish-crates.yml"
        target = transfer.original.target
    else:
        workflow = "release-benchmarks.yml"
        target = transfer.target
    expected = {
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REPOSITORY": REPOSITORY,
        "GITHUB_REF": f"refs/tags/{target.tag}",
        "GITHUB_SHA": target.commit,
        "GITHUB_WORKFLOW_REF": f"{REPOSITORY}/.github/workflows/{workflow}@refs/tags/{target.tag}",
        "GITHUB_ACTOR": "acgetchell",
        "GITHUB_TRIGGERING_ACTOR": "acgetchell",
        "RELEASE_TAG": target.tag,
    }
    if any(os.environ.get(key) != value for key, value in expected.items()):
        message = "candidate handoff requires the maintainer's exact workflow/tag dispatch"
        raise ValueError(message)


def transport_name(role: AssetRole) -> str:
    """Use fixed local paths independent of provider asset names."""
    return "archive.tar.gz" if role == "archive" else f"{role}.json"


def copy_draft_assets(transfer: DraftTransfer, directory: Path, incoming: Path) -> None:
    """Check and retain the original inert transport bytes."""
    for role, asset in transfer.assets.role_assertions():
        name = transport_name(role)
        source = incoming / name
        if source.is_symlink():
            message = "transported candidate bytes differ from independent dispatch assertions"
            raise ValueError(message)
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != asset.sha256:
            message = "transported candidate bytes differ from independent dispatch assertions"
            raise ValueError(message)
        freeze(directory, {name: data})


def download_published_assets(root: Path, transfer: PublicationTransfer, directory: Path) -> None:
    """Fetch the original and canonical assets into distinct role paths."""
    for role, asset in transfer.role_assertions():
        download_release_asset(root, REPOSITORY, transfer.original.target.tag, asset.name, directory / transport_name(role), expected_sha256=asset.sha256)


def check_published_transfer(root: Path, transfer: PublicationTransfer, *, index: AssetAssertion | None = None) -> None:
    """Require the immutable release and exact original and canonical assets."""
    target = transfer.original.target
    release = check_reviewed_release(
        load_config(root=root),
        target.tag,
        event=ReviewedRelease(target.tag, target.commit, target.release_id),
    )
    if not release.immutable:
        message = "published candidate release must be immutable"
        raise ValueError(message)
    check_transfer_assets(release, transfer, index=index)


def check_transfer_assets(release: GitHubRelease, transfer: PublicationTransfer, *, index: AssetAssertion | None = None) -> None:
    """Keep the captured release and every original asset at both boundaries."""
    target = transfer.original.target
    if (release.repository, release.tag, release.identifier, release.title) != (
        REPOSITORY,
        target.tag,
        target.release_id,
        target.expected_title,
    ):
        message = "candidate release identity changed"
        raise ValueError(message)
    assets = tuple(asset for _role, asset in transfer.role_assertions())
    if index is not None:
        assets += (index,)
    for asset in assets:
        entry = next((item for item in release.assets if item[0] == asset.name), None)
        if entry is None or entry[1] != asset.identifier or (entry[3] is not None and entry[3] != asset.sha256):
            message = "candidate asset identity changed"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class ReviewedAcceptance:
    """Independent owner digest, optionally paired with the frozen approval bytes."""

    sha256: str
    expected: bytes | None = None

    def __post_init__(self) -> None:
        """Reject missing review identity or a different approval receipt."""
        if re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            message = "publication requires the independently reviewed full acceptance SHA-256"
            raise ValueError(message)
        if self.expected is not None and (not isinstance(self.expected, bytes) or hashlib.sha256(self.expected).hexdigest() != self.sha256):
            message = "approval receipt differs from independently reviewed acceptance"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class DraftHandoff:
    """Verify an original selection using inert transport."""

    transfer: DraftTransfer
    incoming: Path


@dataclass(frozen=True, slots=True)
class PublicationHandoff:
    """Verify the immutable published selection using independent owner approval."""

    transfer: PublicationTransfer
    review: ReviewedAcceptance


def parse_handoff(payload: bytes, source: Path | ReviewedAcceptance) -> DraftHandoff | PublicationHandoff:
    """Pair the transfer with its permitted source before authenticating the dispatch."""
    transfer = parse_transfer(payload)
    if isinstance(source, ReviewedAcceptance):
        return PublicationHandoff(require_publication_transfer(transfer), source)
    return DraftHandoff(require_draft_transfer(transfer), source)


def receive(root: Path, destination: Path, payload: bytes, *, source: Path | ReviewedAcceptance) -> Path:
    """Read-only hosted acceptance, pinned to original bytes and reviewed CI."""
    handoff = parse_handoff(payload, source)
    require_dispatch(handoff.transfer)
    transfer = handoff.transfer.original if isinstance(handoff, PublicationHandoff) else handoff.transfer
    check_release_dependencies(root)
    destination = owner_directory(root, "", destination)
    destination.mkdir(parents=True, exist_ok=True)

    def release_identity() -> None:
        if isinstance(handoff, PublicationHandoff):
            check_published_transfer(root, handoff.transfer)

    release_identity()
    with tempfile.TemporaryDirectory(prefix="delaunay-transfer-") as temporary:
        directory = Path(temporary)
        if isinstance(handoff, PublicationHandoff):
            download_published_assets(root, handoff.transfer, directory)
        else:
            copy_draft_assets(handoff.transfer, directory, handoff.incoming)
        release_identity()
        policy = parse_candidate_policy((directory / "policy.json").read_bytes())
        accepted = (directory / "acceptance.json").read_bytes()
        archive_path = directory / "archive.tar.gz"
        evidence = verify_candidate_archive(archive_path, policy, accepted, target=transfer.target)
        actual = verify_candidate_release(
            root,
            evidence,
            policy,
            load_evidence_policy(root / "tooling/release-evidence.toml"),
            partial(api, repository=REPOSITORY),
            acquisition_sha256=transfer.acquisition,
            publication_receipt=(directory / "publication.json").read_bytes(),
            publication_sha256=transfer.publication,
            # A published release may need explicitly reviewed newer native CI.
            # Before approval, the independent owner digest authenticates the
            # newly verified result. After approval, pin those exact bytes.
            expected=handoff.review.expected if isinstance(handoff, PublicationHandoff) else accepted,
        )
        if isinstance(handoff, PublicationHandoff) and hashlib.sha256(actual).hexdigest() != handoff.review.sha256:
            message = "verified acceptance differs from the independently reviewed digest"
            raise ValueError(message)
        release_identity()
        name = archive_name(transfer.target.tag)
        freeze(
            destination,
            {
                "transfer.json": payload,
                "acceptance.json": actual,
                "target.json": serialize_release_target(transfer.target),
                name: archive_path.read_bytes(),
            },
        )
    return destination / name


def dispatch_source(options: argparse.Namespace) -> Path | ReviewedAcceptance:
    """Parse mutually exclusive draft transport and published approval inputs."""
    if options.published:
        if options.incoming is not None:
            message = "published verification requires live immutable assets"
            raise ValueError(message)
        return ReviewedAcceptance(os.environ.get("CANDIDATE_ACCEPTANCE", ""), options.expected.read_bytes() if options.expected is not None else None)
    if options.incoming is None or options.expected is not None:
        message = "draft verification requires inert files and no publication approval override"
        raise ValueError(message)
    return Path(options.incoming)


def argument_parser() -> argparse.ArgumentParser:
    """Declare explicit recovery inputs alongside automatic retained context."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    measurement = commands.add_parser("acquire")
    measurement.add_argument("--checkpoint", required=True)
    measurement.add_argument("--resume", action="store_true")
    measurement.add_argument("--owner-directory", type=Path)
    measurement.add_argument("--progress", choices=("human", "json", "none"), default="human")
    measurement.add_argument("--progress-cadence", type=float, default=300)
    for name in ("capture", "verify", "archive", "stage", "seal", "record", "upload", "benchmarks", "publish"):
        command = commands.add_parser(name)
        if name == "record":
            command.add_argument("role", choices=("transfer", "publication"))
        if name in {"upload", "benchmarks", "publish"}:
            command.add_argument("tag")
        command.add_argument("owner_directory", nargs="?", default="")
        if name == "capture":
            command.add_argument("--retained-scratch", action="store_true", help="explicitly retry retention of the original locally controlled acquisition")
        else:
            command.add_argument("--selection", default="reviewed", help="immutable review record; choose a new name to review a different native CI attempt")
    receiver = commands.add_parser("receive")
    receiver.add_argument("destination", type=Path)
    receiver.add_argument("--published", action="store_true")
    receiver.add_argument("--incoming", type=Path)
    receiver.add_argument("--expected", type=Path)
    return parser


def run_owner_operation(root: Path, options: argparse.Namespace) -> None:
    """Route operations after resolving their independent acquisition owner."""
    owner = selected_owner(root, options.owner_directory)
    if options.operation == "capture":
        print(json.dumps(capture(root, owner, retained_scratch=options.retained_scratch)))
        remember_owner(root, owner)
    elif options.operation == "verify":
        accepted = verify(root, owner, options.selection)
        print(json.dumps({"acceptance": str(accepted), "sha256": hashlib.sha256(accepted.read_bytes()).hexdigest()}))
    elif options.operation == "archive":
        print(archive(owner, resolve_revision(root, "HEAD"), options.selection))
    elif options.operation == "stage":
        print(stage(root, owner, options.selection))
    elif options.operation == "seal":
        print(seal(root, owner, options.selection))
    elif options.operation == "record":
        print(record_path(root, owner, options.selection, options.role))
    elif options.operation == "upload":
        upload_dispatch(root, owner, options.selection, options.tag)
    elif options.operation == "benchmarks":
        benchmark_dispatch(root, owner, options.selection, options.tag)
    elif options.operation == "publish":
        publish_draft(root, owner, options.selection, options.tag)
    else:
        message = f"unknown candidate owner operation: {options.operation}"
        raise ValueError(message)


def main() -> None:
    """Expose only repository orchestration; shared APIs own every evidence check."""
    options = argument_parser().parse_args()
    root = Path.cwd().resolve()
    if options.operation == "acquire":
        owner = owner_directory(root, options.checkpoint, options.owner_directory)
        print(f"Independent acquisition record: {owner}", flush=True)
        result = acquire(
            root,
            owner,
            options.checkpoint,
            resume=options.resume,
            progress=ProgressReporter(mode=options.progress, cadence=options.progress_cadence),
        )
        print(json.dumps(result))
    elif options.operation == "receive":
        print(receive(root, options.destination, os.environ["CANDIDATE_TRANSFER"].encode("utf-8"), source=dispatch_source(options)))
    else:
        run_owner_operation(root, options)


if __name__ == "__main__":
    main()
