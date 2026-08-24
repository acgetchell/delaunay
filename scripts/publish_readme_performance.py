"""Publish a compact README snapshot from retained release-performance data."""

import argparse
import math
import os
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from benchmark_utils import render_performance_bundle
from performance_artifacts import ArtifactPaths, PerformanceBundle, PerformanceRow, load_bundle

if TYPE_CHECKING:
    from collections.abc import Callable

_MARKER_BEGIN = "<!-- PERFORMANCE_RELEASE_TABLE:BEGIN -->"
_MARKER_END = "<!-- PERFORMANCE_RELEASE_TABLE:END -->"
_ASSET_CSV = Path("docs/assets/bench/release-performance.csv")
_ASSET_PROVENANCE = Path("docs/assets/bench/release-performance.provenance.json")


@dataclass(frozen=True, slots=True)
class PublicationSummary:
    """Destinations and release identity for one README publication."""

    current_tag: str
    baseline_tag: str
    changed_paths: tuple[Path, ...]


def _cargo_package_tag(cargo_toml: Path) -> str:
    """Return the stable tag implied by Cargo package metadata."""
    data = tomllib.loads(cargo_toml.read_text(encoding="utf-8"))
    package = data.get("package")
    if not isinstance(package, dict):
        msg = f"{cargo_toml} is missing a [package] table"
        raise TypeError(msg)
    version = package.get("version")
    if not isinstance(version, str):
        msg = f"{cargo_toml} [package] is missing a string version"
        raise TypeError(msg)
    return f"v{version.removeprefix('v')}"


def _comparable_groups(rows: tuple[PerformanceRow, ...]) -> dict[str, list[PerformanceRow]]:
    """Group comparable rows without discarding their validated identities."""
    groups: dict[str, list[PerformanceRow]] = {}
    for row in rows:
        if row.coverage_status == "comparable":
            groups.setdefault(row.group, []).append(row)
    return groups


def _geometric_mean_speedup(rows: list[PerformanceRow]) -> float:
    """Return the geometric mean of baseline/current median ratios."""
    logarithms: list[float] = []
    for row in rows:
        if row.baseline is None or row.current is None:
            msg = f"comparable row {row.benchmark_id} is missing one timing estimate"
            raise ValueError(msg)
        logarithms.append(math.log(row.baseline.median_ns) - math.log(row.current.median_ns))
    mean_logarithm = math.fsum(logarithms) / len(logarithms)
    try:
        speedup = math.exp(mean_logarithm)
    except OverflowError as error:
        msg = "geometric mean speedup is outside the finite floating-point range"
        raise ValueError(msg) from error
    if not math.isfinite(speedup) or speedup <= 0.0:
        msg = "geometric mean speedup is outside the positive finite floating-point range"
        raise ValueError(msg)
    return speedup


def render_readme_block(bundle: PerformanceBundle) -> str:
    """Render a compact, deterministic release snapshot from validated rows."""
    context = bundle.context
    current = context.release.current
    baseline = context.release.baseline
    groups = _comparable_groups(bundle.sorted_rows)
    if not groups:
        msg = "README publication requires at least one comparable benchmark row"
        raise ValueError(msg)

    asset_base = f"https://github.com/acgetchell/delaunay/blob/{current}"
    lines = [
        _MARKER_BEGIN,
        "",
        f"Latest retained release comparison: **{current} vs {baseline}**.",
        "",
        "| Benchmark group | Comparable cases | Geometric mean median speedup |",
        "|-----------------|-----------------:|------------------------------:|",
    ]
    for group in sorted(groups):
        rows = groups[group]
        lines.append(f"| `{group}` | {len(rows)} | {_geometric_mean_speedup(rows):.3f}x |")
    lines.extend(
        (
            "",
            (
                "Speedup is baseline median divided by current median; values above 1.000x are faster. "
                "These descriptive aggregates are not statistical-significance claims."
            ),
            "",
            (
                f"[Full report]({asset_base}/docs/PERFORMANCE.md) · "
                f"[CSV]({asset_base}/{_ASSET_CSV.as_posix()}) · "
                f"[provenance]({asset_base}/{_ASSET_PROVENANCE.as_posix()})"
            ),
            "",
            _MARKER_END,
        )
    )
    return "\n".join(lines)


def _replace_marked_block(readme: str, replacement: str) -> str:
    """Replace exactly one ordered README publication block."""
    begin_count = readme.count(_MARKER_BEGIN)
    end_count = readme.count(_MARKER_END)
    if begin_count != 1 or end_count != 1:
        msg = f"README performance markers must be unique (begin={begin_count}, end={end_count})"
        raise ValueError(msg)
    begin = readme.index(_MARKER_BEGIN)
    end = readme.index(_MARKER_END, begin) + len(_MARKER_END)
    return f"{readme[:begin]}{replacement}{readme[end:]}"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Replace one file atomically while preserving its existing mode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        temporary.chmod(path.stat().st_mode if path.exists() else 0o644)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _publish_transaction(payloads: dict[Path, bytes], validate: Callable[[], None]) -> tuple[Path, ...]:
    """Publish changed payloads together and roll back caught failures."""
    originals = {path: path.read_bytes() if path.exists() else None for path in payloads}
    changed = tuple(sorted((path for path, payload in payloads.items() if payload != originals[path]), key=str))
    replaced: list[Path] = []
    try:
        for path in changed:
            _write_bytes_atomic(path, payloads[path])
            replaced.append(path)
        validate()
    except BaseException as primary:
        rollback_errors: list[str] = []
        for path in reversed(replaced):
            try:
                original = originals[path]
                if original is None:
                    path.unlink(missing_ok=True)
                else:
                    _write_bytes_atomic(path, original)
            except OSError as error:
                rollback_errors.append(f"{path}: {error}")
        if rollback_errors:
            msg = f"README performance publication failed ({primary}); rollback also failed: {'; '.join(rollback_errors)}"
            raise RuntimeError(msg) from primary
        raise
    return changed


def _contained_destination(root: Path, path: Path, *, label: str) -> Path:
    """Resolve one write destination and require repository containment."""
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        msg = f"{label} must be contained by repository root {root}, got {resolved}"
        raise ValueError(msg) from error
    return resolved


def publish_readme_performance(
    root: Path,
    *,
    artifacts: ArtifactPaths | None = None,
    readme: Path | None = None,
) -> PublicationSummary:
    """Validate retained evidence and publish README-owned files transactionally."""
    resolved_root = root.resolve()
    source = artifacts or ArtifactPaths(
        csv=resolved_root / "target/bench-reports/performance.csv",
        provenance=resolved_root / "target/bench-reports/performance.provenance.json",
    )
    readme_path = _contained_destination(resolved_root, readme or resolved_root / "README.md", label="README destination")
    bundle = load_bundle(source)
    bundle.require_promotable()
    expected_current = _cargo_package_tag(resolved_root / "Cargo.toml")
    if bundle.context.release.current != expected_current:
        msg = (
            f"retained performance data is for {bundle.context.release.current}, but Cargo.toml is {expected_current}; "
            "run `just performance-release` for the current release"
        )
        raise ValueError(msg)
    if bundle.context.measurement_mode != "local-worktrees":
        msg = "README publication requires locally retained release measurements"
        raise ValueError(msg)

    current = bundle.context.release.current
    baseline = bundle.context.release.baseline
    archive_stem = f"{current}-vs-{baseline}"
    durable = ArtifactPaths(
        csv=resolved_root / "docs/archive/performance/data" / f"{archive_stem}.csv",
        provenance=resolved_root / "docs/archive/performance/data" / f"{archive_stem}.provenance.json",
    )
    source_csv = source.csv.read_bytes()
    source_provenance = source.provenance.read_bytes()
    if durable.csv.read_bytes() != source_csv or durable.provenance.read_bytes() != source_provenance:
        msg = "retained performance data does not match the exact bundle promoted by `just performance-release`"
        raise ValueError(msg)

    performance_report = resolved_root / "docs/PERFORMANCE.md"
    if not performance_report.is_file():
        msg = f"{performance_report} is missing; run `just performance-release` before publishing the README snapshot"
        raise ValueError(msg)
    durable_evidence = ArtifactPaths(
        csv=durable.csv.relative_to(resolved_root),
        provenance=durable.provenance.relative_to(resolved_root),
    )
    expected_report = render_performance_bundle(bundle, evidence_paths=durable_evidence, evidence_state="promoted")
    if performance_report.read_text(encoding="utf-8") != expected_report:
        msg = "docs/PERFORMANCE.md is not the canonical rendering of the retained and promoted performance bundle"
        raise ValueError(msg)

    updated_readme = _replace_marked_block(readme_path.read_text(encoding="utf-8"), render_readme_block(bundle))
    asset_paths = ArtifactPaths(
        csv=_contained_destination(resolved_root, resolved_root / _ASSET_CSV, label="README CSV asset destination"),
        provenance=_contained_destination(
            resolved_root,
            resolved_root / _ASSET_PROVENANCE,
            label="README provenance asset destination",
        ),
    )
    payloads = {
        asset_paths.csv: source_csv,
        asset_paths.provenance: source_provenance,
        readme_path: updated_readme.encode("utf-8"),
    }

    def validate() -> None:
        if load_bundle(asset_paths) != bundle:
            msg = "published README performance assets do not round-trip to the retained bundle"
            raise ValueError(msg)
        if readme_path.read_text(encoding="utf-8") != updated_readme:
            msg = "published README performance block does not match the validated plan"
            raise ValueError(msg)

    changed = _publish_transaction(payloads, validate)
    return PublicationSummary(current_tag=current, baseline_tag=baseline, changed_paths=changed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root (default: current directory)")
    parser.add_argument("--artifact-csv", type=Path, default=Path("target/bench-reports/performance.csv"))
    parser.add_argument("--artifact-provenance", type=Path, default=Path("target/bench-reports/performance.provenance.json"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    return parser.parse_args(argv)


def _under_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main(argv: list[str] | None = None) -> int:
    """Publish README performance evidence without running measurements."""
    args = parse_args(argv)
    root = args.root.resolve()
    try:
        summary = publish_readme_performance(
            root,
            artifacts=ArtifactPaths(
                csv=_under_root(root, args.artifact_csv),
                provenance=_under_root(root, args.artifact_provenance),
            ),
            readme=_under_root(root, args.readme),
        )
    except (OSError, RuntimeError, TypeError, ValueError, tomllib.TOMLDecodeError) as error:
        print(f"performance-readme: {error}", file=sys.stderr)
        return 1

    if summary.changed_paths:
        for path in summary.changed_paths:
            print(f"Updated {path.relative_to(root)}")
    else:
        print("README performance publication is already current.")
    print(f"README performance report: {summary.current_tag} vs {summary.baseline_tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
