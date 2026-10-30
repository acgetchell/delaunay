"""Cargo-only retention and reproduction around the installed evidence verifier.

The shared package owns run selection, jobs, archives, manifests and frozen
identity. This adapter owns Cargo's two dry-run locations, credential-free byte
reproduction and the published-event boundary. It never uploads a crate.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from functools import partial
from pathlib import Path

from research_repo_tools.evidence import deterministic_json
from research_repo_tools.files import replace_many
from research_repo_tools.measurement import resolve_revision
from research_repo_tools.process import ExecutableNotFoundError, format_exception_diagnostics, run_command
from research_repo_tools.release_assets import require_release_checkout
from research_repo_tools.release_evidence import create_evidence_manifest, verify_release_evidence
from research_repo_tools.release_interfaces import load_evidence_policy, write_release_outputs
from research_repo_tools.release_metadata import read_package_info
from research_repo_tools.release_publishing import validate_event
from research_repo_tools.release_validation import api, parse_release_json, redact_release_diagnostic

REPOSITORY = "acgetchell/delaunay"
MANIFEST = "validation.json"
RECOVERY = (
    "Rerun ALL jobs of the original main-push CI run for this exact commit within "
    "GitHub's 30-day rerun window, then retry. After that window, land a new reviewed "
    "commit and release it. Preserve published tags and versions; there is no rebuild fallback."
)


def identity(root: Path) -> tuple[str, str]:
    """Observe the clean Delaunay Cargo source without executing its code."""
    commit = resolve_revision(root, "HEAD")
    require_release_checkout(root, commit)
    package = read_package_info(root)
    if package.name != "delaunay":
        msg = "expected the delaunay Cargo package"
        raise ValueError(msg)
    return commit, package.version


def tool_identity(root: Path) -> dict[str, str]:
    """Capture independently observed producer/reproduction tools."""
    return {
        "cargo": run_command("cargo", ["--version"], cwd=root).stdout.strip(),
        "rustc": run_command("rustc", ["--version", "--verbose"], cwd=root).stdout.strip(),
    }


def create(root: Path, output: Path) -> None:
    """Retain the exact archive verified by both native Cargo dry runs."""
    commit, version = identity(root)
    policy = load_evidence_policy(root / "tooling/release-evidence.toml")
    if (os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_EVENT_NAME"), os.environ.get("GITHUB_REF"), os.environ.get("GITHUB_SHA")) != (
        REPOSITORY,
        "push",
        "refs/heads/main",
        commit,
    ):
        msg = "only the exact trusted main push can produce Cargo evidence"
        raise ValueError(msg)
    metadata = parse_release_json(run_command("cargo", ["metadata", "--locked", "--no-deps", "--format-version", "1"], cwd=root).stdout)
    name = f"delaunay-{version}.crate"
    packages = Path(metadata["target_directory"]) / "package"
    # Cargo 1.99.1 uses these two dry-run directories, not package/NAME.
    source = packages / "tmp-crate" / name
    archive = source.read_bytes()
    if archive != (packages / "tmp-registry" / name).read_bytes():
        msg = "Cargo dry-run source and registry archives differ"
        raise ValueError(msg)
    manifest = create_evidence_manifest(
        policy,
        version,
        commit,
        int(os.environ["GITHUB_RUN_ID"]),
        int(os.environ["GITHUB_RUN_ATTEMPT"]),
        partial(api, repository=REPOSITORY),
        archive=source,
        tools=tool_identity(root),
    )
    if identity(root) != (commit, version) or source.read_bytes() != archive:
        msg = "Cargo source or archive changed during retention"
        raise ValueError(msg)
    output.mkdir(parents=True, exist_ok=False)
    replace_many({output / name: archive, output / MANIFEST: manifest})


def verify(root: Path, *, expected: bytes | None = None) -> bytes:
    """Require a published release event and reproduce its frozen Cargo evidence."""
    commit, version = identity(root)
    if (os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_EVENT_NAME"), os.environ.get("GITHUB_SHA")) != (REPOSITORY, "release", commit):
        msg = "Cargo promotion requires the matching published release event"
        raise ValueError(msg)
    event = validate_event(parse_release_json(Path(os.environ["GITHUB_EVENT_PATH"]).read_bytes()), REPOSITORY, commit, os.environ["GITHUB_REF"])
    if event.tag != f"v{version}":
        msg = "Cargo release event differs from package version"
        raise ValueError(msg)
    policy = load_evidence_policy(root / "tooling/release-evidence.toml")
    read = partial(api, repository=REPOSITORY)
    # Reject bad/missing approval evidence before executing Cargo. Reuse the
    # exact selection for the post-reproduction verification.
    selected = verify_release_evidence(policy, version, commit, read, expected=expected)
    with tempfile.TemporaryDirectory(prefix="delaunay-cargo-reproduction-") as directory:
        target = Path(directory)
        run_command(
            "cargo",
            ["package", "--locked", "--package", "delaunay", "--registry", "crates-io", "--no-verify", "--target-dir", str(target)],
            cwd=root,
            timeout=600,
        )
        verified = verify_release_evidence(
            policy,
            version,
            commit,
            read,
            expected=selected,
            reproduced_archive=target / "package" / f"delaunay-{version}.crate",
            reproduced_tools=tool_identity(root),
        )
    if identity(root) != (commit, version):
        msg = "Cargo source changed during reproduction"
        raise ValueError(msg)
    return verified


def main() -> int:
    """Retain a package or export a verified selection directly to Actions."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    producer = commands.add_parser("create")
    producer.add_argument("output", type=Path)
    commands.add_parser("verify")
    args = parser.parse_args()
    try:
        root = Path.cwd()
        if args.mode == "create":
            create(root, args.output)
        else:
            frozen = os.environ.get("PROMOTION_SELECTION")
            selected = verify(root, expected=None if frozen is None else frozen.encode("utf-8"))
            with tempfile.TemporaryDirectory(prefix="delaunay-cargo-selection-") as temporary:
                write_release_outputs(
                    Path(temporary).resolve() / "selection.json",
                    selected,
                    github_output=Path(os.environ["GITHUB_OUTPUT"]),
                    values={"selection": deterministic_json(parse_release_json(selected)).decode("utf-8").strip().replace("\n", "")},
                )
            with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8", newline="\n") as summary:
                summary.write("### Verified Cargo promotion\n\n```json\n" + selected.decode("utf-8") + "```\n")
            print("Verified frozen Cargo evidence and byte-identical reproduction.")
    except (ExecutableNotFoundError, OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        print(f"Cargo promotion failed: {redact_release_diagnostic(format_exception_diagnostics(error))}\n{RECOVERY}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
