"""Delaunay release policy over the shared, captured-target publication API."""

import argparse
import os
from pathlib import Path

from research_repo_tools.files import replace_many
from research_repo_tools.measurement import resolve_revision
from research_repo_tools.release_assets import (
    parse_release_target,
    preflight_release_target,
    publish_release_asset,
    revalidate_release_target,
    serialize_release_target,
)


def release_benchmarks(mode: str, root: Path, target_path: Path, repository: str, tag: str) -> None:
    """Bind the tagged benchmark checkout to a trusted preflight handoff.

    The workflow owns the handoff file and the installed runtime across checkout
    changes. Reading JSON alone does not establish that it came from preflight.
    Delaunay refuses existing baseline assets even when their bytes match.
    """
    asset = root / f"delaunay-{tag}-criterion-baseline.tar.gz"
    if mode == "preflight":
        target = preflight_release_target(root, repository, tag, resolve_revision(root, tag))
    elif mode == "publish":
        target = parse_release_target(target_path.read_bytes())
        if (target.repository, target.tag) != (repository, tag):
            msg = "release request differs from the trusted preflight target"
            raise ValueError(msg)
        if resolve_revision(root, "HEAD") != target.commit:
            msg = "HEAD differs from the benchmarked tag"
            raise ValueError(msg)
    else:
        msg = f"unknown release benchmark mode: {mode}"
        raise ValueError(msg)
    release = revalidate_release_target(root, target)
    if any(entry[0] == asset.name for entry in release.assets):
        msg = "baseline asset already exists; inspect the draft asset before retrying"
        raise ValueError(msg)
    if mode == "preflight":
        replace_many({target_path: serialize_release_target(target)})
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8", newline="\n") as output:
            output.write(f"commit={target.commit}\n")
    else:
        publish_release_asset(root, repository, tag, asset, target=target, publish=True)


def main() -> None:
    """Run the explicit workflow phase using its trusted handoff location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "publish"))
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    release_benchmarks(args.mode, Path.cwd(), args.target, os.environ["GITHUB_REPOSITORY"], os.environ["RELEASE_TAG"])


if __name__ == "__main__":
    main()
