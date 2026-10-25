"""Delaunay profiling labels over the shared host and provenance capture API."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

from research_repo_tools.files import replace_many
from research_repo_tools.host_metadata import capture_profile
from research_repo_tools.process import ExecutableNotFoundError, format_exception_diagnostics


def _write_metadata(root: Path) -> None:
    manifest = root / "Cargo.toml"
    try:
        version = tomllib.loads(manifest.read_text(encoding="utf-8"))["package"]["version"]
        if not isinstance(version, str) or not version.strip():
            message = "package.version must be a nonempty string"
            raise ValueError(message)
    except (ValueError, KeyError, TypeError) as error:
        message = f"Cannot read package.version from {manifest}: {error}"
        raise ValueError(message) from error
    mode = os.environ.get("PROFILE_METADATA_MODE") or ("development" if os.environ.get("PROFILING_DEV_MODE") == "1" else "production")
    context = {
        "title": os.environ.get("PROFILE_METADATA_TITLE", "Profiling Environment"),
        "code-ref": os.environ.get("GITHUB_REF_NAME", "local"),
        "cargo-profile": "perf",
        "filter": os.environ.get("BENCH_FILTER_VALUE") or os.environ.get("PROFILE_METADATA_FILTER", "All benchmarks"),
        "mode": mode,
        "suite": os.environ.get("PROFILE_METADATA_SUITE", "local selection"),
    }
    scratch = root / "target"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="profiling-context-", dir=scratch) as temporary:
        configuration = Path(temporary) / "profile.toml"
        configuration.write_text(
            'schema = 1\ncargo-manifest = "Cargo.toml"\nrust-toolchain = "rust-toolchain.toml"\n'
            'measurement = "tooling/profiling-inputs.toml"\n'
            f"release = {json.dumps('v' + version, ensure_ascii=False)}\n[context]\n"
            + "".join(f"{name} = {json.dumps(value, ensure_ascii=False)}\n" for name, value in context.items()),
            encoding="utf-8",
        )
        payload = capture_profile(root, configuration.relative_to(root).as_posix())
    replace_many({root / "profiling-results/environment.json": payload})


def main(argv: list[str] | None = None) -> int:
    """Capture the selected scientific profiling mode without parsing tool output."""
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    try:
        _write_metadata(Path.cwd())
    except (ExecutableNotFoundError, OSError, ValueError, RuntimeError, subprocess.SubprocessError, ExceptionGroup) as error:
        print(f"delaunay-profile-metadata: {format_exception_diagnostics(error)}", file=sys.stderr)
        return 1
    return 0
