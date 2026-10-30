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
from research_repo_tools.measurement import MeasurementConfig, load_measurement
from research_repo_tools.process import ExecutableNotFoundError, format_exception_diagnostics

DIAGNOSTIC_TARGETS = {
    "timing": "construction_scaling",
    "allocations": "memory_footprint",
    **{name: name for name in ("circumsphere_queries", "hull_queries", "memory_footprint", "traversal", "validation_components")},
}


def _selected_measurement(root: Path, destination: Path, suite: str, selected_filter: str) -> None:
    """Bind metadata to the actual focused target, instrumentation and filter."""
    if suite not in DIAGNOSTIC_TARGETS:
        raise ValueError(f"Unknown diagnostic suite: {suite}")
    base = load_measurement(root, "tooling/profiling-inputs.toml")
    if not isinstance(base, MeasurementConfig):
        message = "profiling inputs must declare a single native command"
        raise TypeError(message)
    target = DIAGNOSTIC_TARGETS[suite]
    command = [*base.command[:-1], target]
    if suite == "allocations":
        command.extend(("--features", "count-allocations"))
    command.extend(("--", "--noplot"))
    if suite == "allocations":
        command.extend(("^memory_profiling/",) if selected_filter == "memory_profiling" else ("--exact", selected_filter))
    elif selected_filter:
        command.append(selected_filter)
    harness = [f"benches/{target}.rs", *(path for path in base.harness if not path.startswith("benches/") or path.startswith("benches/common/"))]
    destination.write_text(
        "schema = 1\n"
        + "".join(
            f"{name} = {json.dumps(value, ensure_ascii=False)}\n" for name, value in (("command", command), ("sources", base.sources), ("harness", harness))
        )
        + "[probes]\n"
        + "".join(f"{name} = {json.dumps(command, ensure_ascii=False)}\n" for name, command in base.probes),
        encoding="utf-8",
    )


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
    suite = os.environ.get("PROFILE_METADATA_SUITE", "timing")
    selected_filter = os.environ.get("BENCH_FILTER_VALUE") or os.environ.get("PROFILE_METADATA_FILTER", "")
    if suite == "allocations" and not selected_filter:
        selected_filter = "memory_profiling"
    context = {
        "title": os.environ.get("PROFILE_METADATA_TITLE", "Profiling Environment"),
        "code-ref": os.environ.get("GITHUB_REF_NAME", "local"),
        "cargo-profile": "perf",
        "filter": selected_filter or "All benchmarks",
        "mode": mode,
        "suite": suite,
    }
    scratch = root / "target"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="profiling-context-", dir=scratch) as temporary:
        measurement = Path(temporary) / "measurement.toml"
        _selected_measurement(root, measurement, suite, selected_filter)
        configuration = Path(temporary) / "profile.toml"
        configuration.write_text(
            'schema = 1\ncargo-manifest = "Cargo.toml"\nrust-toolchain = "rust-toolchain.toml"\n'
            f"measurement = {json.dumps(measurement.relative_to(root).as_posix())}\n"
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
    except (ExecutableNotFoundError, OSError, TypeError, ValueError, RuntimeError, subprocess.SubprocessError, ExceptionGroup) as error:
        print(f"delaunay-profile-metadata: {format_exception_diagnostics(error)}", file=sys.stderr)
        return 1
    return 0
