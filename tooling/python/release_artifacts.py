"""Delaunay Cargo evidence over research-repo-tools v0.2.3 promotion primitives.

Cargo still packages source during upload. Release jobs reproduce the retained
archive without building and require identical bytes before obtaining a token.
Only the trusted main producer performs default/all-feature package builds.
"""

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from typing import Any

from research_repo_tools.ci import export_environment
from research_repo_tools.files import replace_many
from research_repo_tools.process import ExecutableNotFoundError, format_exception_diagnostics, run_command
from research_repo_tools.release_validation import ValidationPolicy, download_artifact, select_main_run, validate_jobs

REPOSITORY = "acgetchell/delaunay"
MANIFEST = "validation.json"
# Bound the nested gzip/tar as well as the shared helper's outer ZIP limits.
MAX_CRATE_BYTES = 64 * 1024 * 1024
MAX_CRATE_MEMBERS = 4096
MAX_METADATA_BYTES = 1024 * 1024
RECOVERY = (
    "Rerun ALL jobs of the original main-push CI run for this exact commit within "
    "GitHub's 30-day rerun window, then retry. After that window, land a new reviewed "
    "commit and release it. Preserve published tags and versions; there is no rebuild fallback."
)
POLICY = ValidationPolicy(
    REPOSITORY,
    ".github/workflows/ci.yml",
    "CI",
    (
        *((f"build ({host})", ("Verify native Rust host", "Run CI checks")) for host in ("ubuntu-latest", "macos-latest", "windows-latest")),
        (
            "package",
            (
                "Verify clean default and all-feature Cargo packages",
                "Retain verified Cargo package",
                "Upload verified Cargo package",
            ),
        ),
    ),
)


def _object(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        msg = "expected a JSON object"
        raise TypeError(msg)
    return value


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            msg = f"duplicate JSON key: {key}"
            raise ValueError(msg)
        result[key] = value
    return result


def _json(payload: str | bytes) -> dict[str, Any]:
    return _object(json.loads(payload, object_pairs_hook=_unique))


def _positive(value: object) -> int:
    if type(value) is not int or value <= 0:
        msg = "expected a positive integer identity"
        raise ValueError(msg)
    return value


def _encode(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str:
    return run_command("git", ["--no-pager", *args], cwd=root).stdout.strip()


def _identity(root: Path) -> tuple[str, str]:
    commit = _git(root, "rev-parse", "HEAD")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None or _git(root, "status", "--porcelain", "--untracked-files=all"):
        msg = "Cargo promotion requires a clean full-commit checkout"
        raise ValueError(msg)
    package = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))["package"]
    version = package["version"]
    if (
        package["name"] != "delaunay"
        or not isinstance(version, str)
        or re.fullmatch(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", version) is None
    ):
        msg = "expected the stable delaunay Cargo package"
        raise ValueError(msg)
    return commit, version


def _tool_identity(root: Path) -> dict[str, str]:
    return {
        "cargo": run_command("cargo", ["--version"], cwd=root).stdout.strip(),
        "rustc": run_command("rustc", ["--version", "--verbose"], cwd=root).stdout.strip(),
    }


def _check_crate(path: Path, commit: str, version: str) -> None:
    """Inspect Cargo metadata without extracting or executing package contents."""
    prefix = f"delaunay-{version}"
    with gzip.open(path, "rb") as compressed:
        payload = compressed.read(MAX_CRATE_BYTES + 1)
    if len(payload) > MAX_CRATE_BYTES:
        msg = "Cargo archive expanded size exceeds the release bound"
        raise ValueError(msg)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        names: set[str] = set()
        metadata: dict[str, bytes] = {}
        for member in archive:
            if len(names) >= MAX_CRATE_MEMBERS:
                msg = "Cargo archive inventory exceeds the release bound"
                raise ValueError(msg)
            path_name = PurePosixPath(member.name)
            if (
                member.name in names
                or not member.isfile()
                or member.sparse is not None
                or not member.name.startswith(f"{prefix}/")
                or ".." in path_name.parts
                or str(path_name) != member.name
                or "\\" in member.name
                or not 0 <= member.size <= len(payload) - member.offset_data
            ):
                msg = "Cargo archive must contain unique regular package files"
                raise ValueError(msg)
            names.add(member.name)
            if member.name in (f"{prefix}/.cargo_vcs_info.json", f"{prefix}/Cargo.toml"):
                if member.size > MAX_METADATA_BYTES:
                    msg = "Cargo archive metadata size exceeds the release bound"
                    raise ValueError(msg)
                content = archive.extractfile(member)
                if content is not None:
                    metadata[path_name.name] = content.read()
        if set(metadata) != {".cargo_vcs_info.json", "Cargo.toml"}:
            msg = "Cargo archive lacks source or package metadata"
            raise ValueError(msg)
        identity = _json(metadata[".cargo_vcs_info.json"])
        package = tomllib.loads(metadata["Cargo.toml"].decode("utf-8"))["package"]
        git = _object(identity.get("git"))
        if (
            git.get("sha1") != commit
            or git.get("dirty", False) is not False
            or identity.get("path_in_vcs") != ""
            or (package["name"], package["version"]) != ("delaunay", version)
        ):
            msg = "Cargo archive source/version differs from the clean release commit"
            raise ValueError(msg)


def create(root: Path, output: Path) -> None:
    """Retain the archive just verified by both credential-free Cargo dry runs."""
    commit, version = _identity(root)
    if (os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_EVENT_NAME"), os.environ.get("GITHUB_REF"), os.environ.get("GITHUB_SHA")) != (
        REPOSITORY,
        "push",
        "refs/heads/main",
        commit,
    ):
        msg = "only the exact trusted main push can produce Cargo promotion evidence"
        raise ValueError(msg)
    metadata = _json(run_command("cargo", ["metadata", "--locked", "--no-deps", "--format-version", "1"], cwd=root).stdout)
    name = f"delaunay-{version}.crate"
    # Pinned Cargo 1.99.1 stages publish dry runs in these two directories,
    # rather than cargo package's top-level package/NAME output. Require both
    # native dry-run copies to agree; never pick a stale cargo package result.
    packages = Path(metadata["target_directory"]) / "package"
    source = packages / "tmp-crate" / name
    if source.read_bytes() != (packages / "tmp-registry" / name).read_bytes():
        msg = "Cargo dry-run source and registry archives differ"
        raise ValueError(msg)
    _check_crate(source, commit, version)
    evidence = {
        "schema": 1,
        "repository": REPOSITORY,
        "commit": commit,
        "version": version,
        "run_id": _positive(int(os.environ["GITHUB_RUN_ID"])),
        "run_attempt": _positive(int(os.environ["GITHUB_RUN_ATTEMPT"])),
        "file": name,
        "sha256": _digest(source),
        "tools": _tool_identity(root),
    }
    if _identity(root) != (commit, version):
        msg = "source changed while retaining the verified package"
        raise ValueError(msg)
    output.mkdir(parents=True, exist_ok=False)
    replace_many({output / name: source.read_bytes(), output / MANIFEST: _encode(evidence)})


def _read(endpoint: str) -> dict[str, Any]:
    return _json(run_command("gh", ["api", f"repos/{REPOSITORY}/{endpoint}"]).stdout)


def _pages(endpoint: str, field: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    separator = "&" if "?" in endpoint else "?"
    for page in range(1, 101):
        entries = _read(f"{endpoint}{separator}per_page=100&page={page}")[field]
        if not isinstance(entries, list):
            msg = f"invalid GitHub {field} inventory"
            raise TypeError(msg)
        result.extend(_object(entry) for entry in entries)
        if len(entries) < 100:
            return result
    msg = "GitHub inventory exceeds the bounded page limit"
    raise ValueError(msg)


def _candidate(commit: str, *, attempts: int = 30) -> dict[str, Any]:
    run = select_main_run(POLICY, commit, _read, attempts=attempts)
    jobs = validate_jobs(POLICY, run, _pages(f"actions/runs/{run['id']}/attempts/{run['run_attempt']}/jobs", "jobs"))
    name = f"cargo-package-{run['run_attempt']}"
    artifacts = [item for item in _pages(f"actions/runs/{run['id']}/artifacts", "artifacts") if item.get("name") == name]
    if len(artifacts) != 1:
        msg = f"missing or ambiguous Cargo artifact; {RECOVERY}"
        raise ValueError(msg)
    artifact = artifacts[0]
    association = _object(artifact.get("workflow_run"))
    expected = {
        "id": run["id"],
        "head_sha": commit,
        "head_branch": "main",
        "repository_id": run["repository"]["id"],
        "head_repository_id": run["repository"]["id"],
    }
    if artifact.get("expired") is not False or any(association.get(key) != value for key, value in expected.items()):
        msg = f"expired or mismatched Cargo artifact; {RECOVERY}"
        raise ValueError(msg)
    if not isinstance(artifact.get("digest"), str) or re.fullmatch(r"sha256:[0-9a-f]{64}", artifact["digest"]) is None:
        msg = "Cargo artifact lacks its authoritative ZIP digest"
        raise ValueError(msg)
    selected = {key: artifact[key] for key in ("id", "name", "digest", "size_in_bytes")}
    _positive(selected["id"])
    _positive(selected["size_in_bytes"])
    return {"run_id": run["id"], "run_attempt": run["run_attempt"], "jobs": jobs, "artifact": selected}


def _download_verified(selection: dict[str, Any], retained: Path, *, reproduce_root: Path | None = None) -> dict[str, Any]:
    """Check the retained package on any host; reproduction also binds local tools."""
    commit, version = selection["commit"], selection["tag"].removeprefix("v")
    name = f"delaunay-{version}.crate"
    download_artifact(REPOSITORY, _object(selection["artifact"]), retained, (name, MANIFEST))
    evidence = _json((retained / MANIFEST).read_bytes())
    recorded_tools = evidence.get("tools")
    if (
        not isinstance(recorded_tools, dict)
        or set(recorded_tools) != {"cargo", "rustc"}
        or any(not isinstance(value, str) or not value.strip() for value in recorded_tools.values())
    ):
        msg = "Cargo artifact manifest differs from the required producer tool identities"
        raise ValueError(msg)
    wanted = {
        "schema": 1,
        "repository": REPOSITORY,
        "commit": commit,
        "version": version,
        "run_id": selection["run_id"],
        "run_attempt": selection["run_attempt"],
        "file": name,
        "sha256": _digest(retained / name),
        "tools": recorded_tools if reproduce_root is None else _tool_identity(reproduce_root),
    }
    if evidence != wanted:
        msg = "Cargo artifact manifest differs from its archive, source, tools or validation attempt"
        raise ValueError(msg)
    _check_crate(retained / name, commit, version)
    return evidence


def _unchanged(root: Path, selection: dict[str, Any]) -> None:
    if _identity(root) != (selection["commit"], selection["tag"].removeprefix("v")) or _candidate(selection["commit"], attempts=1) != {
        key: selection[key] for key in ("run_id", "run_attempt", "jobs", "artifact")
    }:
        msg = "Cargo source or validation evidence changed during promotion"
        raise ValueError(msg)


def evidence(root: Path, *, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    """Verify retained main evidence locally before a release is published.

    This works across hosts without Cargo or a release event. The publication
    workflow separately requires that event and byte-identical repackaging with
    the producer's tools before credentials are obtained.
    """
    commit, version = _identity(root)
    selection = {"schema": 1, "tag": f"v{version}", "commit": commit, **_candidate(commit)}
    if expected is not None and selection != expected:
        msg = "Cargo validation selection changed; inspect the original run before publishing"
        raise ValueError(msg)
    with tempfile.TemporaryDirectory(prefix="delaunay-evidence-") as directory:
        _download_verified(selection, Path(directory) / "retained")
        _unchanged(root, selection)
    return selection


def verify(root: Path, *, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    """Verify exact main evidence and byte-identical repackaging without builds.

    The first job hands its frozen selection to the environment-gated job.
    After approval, reruns and changed artifacts are rejected instead of selecting
    a new validation source. The enclosing workflow retains the live release gate.
    """
    commit, version = _identity(root)
    event = _json(Path(os.environ["GITHUB_EVENT_PATH"]).read_bytes())
    release = _object(event.get("release"))
    observed_event = (
        os.environ.get("GITHUB_REPOSITORY"),
        os.environ.get("GITHUB_EVENT_NAME"),
        os.environ.get("GITHUB_REF"),
        os.environ.get("GITHUB_SHA"),
        event.get("action"),
    )
    if (
        observed_event
        != (
            REPOSITORY,
            "release",
            f"refs/tags/v{version}",
            commit,
            "published",
        )
        or release.get("tag_name") != f"v{version}"
        or release.get("draft") is not False
        or release.get("prerelease") is not False
    ):
        msg = "Cargo promotion requires the matching published stable release event"
        raise ValueError(msg)
    selection = {"schema": 1, "tag": f"v{version}", "commit": commit, "release_id": _positive(release.get("id")), **_candidate(commit)}
    if expected is not None and selection != expected:
        msg = "Cargo validation selection changed after approval; inspect the original run and retry the whole publication workflow"
        raise ValueError(msg)
    name = f"delaunay-{version}.crate"
    with tempfile.TemporaryDirectory(prefix="delaunay-promotion-") as directory:
        temporary = Path(directory)
        retained = temporary / "retained"
        retained_evidence = _download_verified(selection, retained, reproduce_root=root)
        run_command(
            "cargo",
            ["package", "--locked", "--package", "delaunay", "--registry", "crates-io", "--no-verify", "--target-dir", str(temporary / "cargo")],
            cwd=root,
            timeout=600,
        )
        if _digest(temporary / "cargo" / "package" / name) != retained_evidence["sha256"]:
            msg = "Cargo repackaging differs from the main-verified archive; upload is blocked"
            raise ValueError(msg)
        _unchanged(root, selection)
    return selection


def main() -> int:
    """Run producer retention or release verification; never upload a crate."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    producer = commands.add_parser("create", help="retain the main-push verified archive")
    producer.add_argument("output", type=Path)
    local = commands.add_parser("evidence", help="verify retained main evidence before publication, without Cargo")
    local.add_argument("--identity", default="", help="optional previously selected evidence JSON")
    commands.add_parser("verify", help="verify/reproduce exact main evidence without building")
    args = parser.parse_args()
    try:
        root = Path.cwd()
        if args.mode == "create":
            create(root, args.output)
        elif args.mode == "evidence":
            selection = evidence(root, expected=_json(args.identity) if args.identity else None)
            print(_encode(selection).decode("utf-8"), end="")
        else:
            frozen = os.environ.get("PROMOTION_SELECTION")
            selection = verify(root, expected=None if frozen is None else _json(frozen))
            export_environment(
                Path(os.environ["GITHUB_OUTPUT"]),
                ("selection",),
                environment={"selection": _encode(selection).decode("utf-8").removesuffix("\n")},
            )
            with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8", newline="\n") as summary:
                summary.write("### Verified Cargo promotion\n\n```json\n" + _encode(selection).decode("utf-8") + "```\n")
            print(f"Verified Cargo promotion: main run {selection['run_id']}, attempt {selection['run_attempt']}, artifact {selection['artifact']['id']}.")
    except (
        ExecutableNotFoundError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        EOFError,
        tarfile.TarError,
        zipfile.BadZipFile,
        zlib.error,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        # gh can include redirected, signed artifact URLs in transport errors.
        detail = re.sub(r'(https?://[^\s?"<>]+)\?[^\s"<>]+', r"\1?<redacted>", format_exception_diagnostics(error))
        print(f"Cargo promotion failed: {detail}\n{RECOVERY}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
