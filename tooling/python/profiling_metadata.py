"""Delaunay profiling labels over the shared host and provenance capture API."""

import argparse
import json
import os
import tempfile
import tomllib
from pathlib import Path

from research_repo_tools.files import replace_many
from research_repo_tools.host_metadata import capture_profile


def main(argv: list[str] | None = None) -> int:
    """Capture the selected scientific profiling mode without parsing tool output."""
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    root = Path.cwd()
    version = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))["package"]["version"]
    mode = os.environ.get("PROFILE_METADATA_MODE") or ("development" if os.environ.get("PROFILING_DEV_MODE") == "1" else "production")
    context = {
        "title": os.environ.get("PROFILE_METADATA_TITLE", "Profiling Environment"),
        "code-ref": os.environ.get("GITHUB_REF_NAME", "local"),
        "cargo-profile": "perf",
        "filter": os.environ.get("BENCH_FILTER_VALUE") or os.environ.get("PROFILE_METADATA_FILTER", "All benchmarks"),
        "mode": mode,
    }
    scratch = root / "target"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="profiling-context-", dir=scratch) as temporary:
        configuration = Path(temporary) / "profile.toml"
        configuration.write_text(
            'schema = 1\ncargo-manifest = "Cargo.toml"\nrust-toolchain = "rust-toolchain.toml"\n'
            'measurement = "tooling/profiling-inputs.toml"\n'
            f"release = {json.dumps('v' + version)}\n[context]\n" + "".join(f"{name} = {json.dumps(value)}\n" for name, value in context.items()),
            encoding="utf-8",
        )
        payload = capture_profile(root, configuration.relative_to(root).as_posix())
    replace_many({root / "profiling-results/environment.json": payload})
    return 0
