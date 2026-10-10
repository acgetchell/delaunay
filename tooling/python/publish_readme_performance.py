"""Publish a compact README snapshot from retained release-performance data."""

import argparse
import math
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from research_repo_tools.process import format_exception_diagnostics
from research_repo_tools.publication import MarkerPair, plan_publication, publish_publication

from benchmark_utils import DOCS_PERFORMANCE_REPORT, render_performance_bundle
from performance_artifacts import ArtifactPaths, PerformanceBundle, PerformanceRow, load_bundle_bytes

_MARKER_BEGIN = "<!-- PERFORMANCE_RELEASE_TABLE:BEGIN -->"
_MARKER_END = "<!-- PERFORMANCE_RELEASE_TABLE:END -->"
_ASSET_PAYLOAD = Path("docs/assets/bench/release-performance.comparison.json")
_ASSET_PROVENANCE = Path("docs/assets/bench/release-performance.evidence.json")


@dataclass(frozen=True, slots=True)
class PublicationSummary:
    """Destinations and release identity for one README publication."""

    current_tag: str
    baseline_tag: str
    changed_paths: tuple[Path, ...]


def _cargo_package_tag(payload: bytes) -> str:
    """Return the stable tag implied by Cargo package metadata."""
    data = tomllib.loads(payload.decode("utf-8"))
    package = data.get("package")
    if not isinstance(package, dict):
        msg = "Cargo.toml is missing a [package] table"
        raise TypeError(msg)
    version = package.get("version")
    if not isinstance(version, str):
        msg = "Cargo.toml [package] is missing a string version"
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
                f"[Full report]({asset_base}/{DOCS_PERFORMANCE_REPORT.as_posix()}) · "
                f"[timing evidence]({asset_base}/{_ASSET_PAYLOAD.as_posix()}) · "
                f"[provenance]({asset_base}/{_ASSET_PROVENANCE.as_posix()})"
            ),
            "",
            _MARKER_END,
        )
    )
    return "\n".join(lines)


def _contained_destination(root: Path, path: Path, *, label: str) -> Path:
    """Check containment while preserving symlink spelling for shared validation."""
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        msg = f"{label} must be contained by repository root {root}, got {resolved}"
        raise ValueError(msg) from error
    return path.absolute()


def publish_readme_performance(
    root: Path,
    *,
    artifacts: ArtifactPaths | None = None,
    readme: Path | None = None,
) -> PublicationSummary:
    """Validate retained evidence and publish README-owned files transactionally."""
    resolved_root = root.resolve()
    source = artifacts or ArtifactPaths(
        payload=resolved_root / "target/bench-reports/performance.comparison.json",
        provenance=resolved_root / "target/bench-reports/performance.evidence.json",
    )
    readme_path = _contained_destination(resolved_root, readme or resolved_root / "README.md", label="README destination")
    source_evidence = source.payload.read_bytes()
    source_provenance = source.provenance.read_bytes()
    cargo_path = resolved_root / "Cargo.toml"
    cargo_payload = cargo_path.read_bytes()
    bundle = load_bundle_bytes(source_evidence, source_provenance, source=str(source.payload))
    bundle.require_promotable()
    expected_current = _cargo_package_tag(cargo_payload)
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
        payload=_contained_destination(
            resolved_root,
            resolved_root / "docs/archive/performance/data" / f"{archive_stem}.comparison.json",
            label="promoted performance payload",
        ),
        provenance=_contained_destination(
            resolved_root,
            resolved_root / "docs/archive/performance/data" / f"{archive_stem}.evidence.json",
            label="promoted performance provenance",
        ),
    )
    performance_report = _contained_destination(
        resolved_root,
        resolved_root / DOCS_PERFORMANCE_REPORT,
        label="promoted performance report",
    )
    missing_durable = [path for path in (durable.payload, durable.provenance) if not path.is_file()]
    if missing_durable:
        rendered = ", ".join(path.relative_to(resolved_root).as_posix() for path in missing_durable)
        msg = f"promoted performance evidence is missing: {rendered}; run `just performance-release` before publishing the README snapshot"
        raise ValueError(msg)
    durable_payload = durable.payload.read_bytes()
    durable_provenance = durable.provenance.read_bytes()
    if durable_payload != source_evidence or durable_provenance != source_provenance:
        msg = "retained performance data does not match the exact bundle promoted by `just performance-release`"
        raise ValueError(msg)

    if not performance_report.is_file():
        report_label = performance_report.relative_to(resolved_root).as_posix()
        msg = f"{report_label} is missing; run `just performance-release` before publishing the README snapshot"
        raise ValueError(msg)
    durable_evidence = ArtifactPaths(
        payload=durable.payload.relative_to(resolved_root),
        provenance=durable.provenance.relative_to(resolved_root),
    )
    expected_report = render_performance_bundle(bundle, evidence_paths=durable_evidence, evidence_state="promoted")
    report_payload = performance_report.read_bytes()
    if report_payload.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n") != expected_report:
        msg = f"{DOCS_PERFORMANCE_REPORT.as_posix()} is not the canonical rendering of the retained and promoted performance bundle"
        raise ValueError(msg)

    asset_paths = ArtifactPaths(
        payload=_contained_destination(resolved_root, resolved_root / _ASSET_PAYLOAD, label="README payload asset destination"),
        provenance=_contained_destination(
            resolved_root,
            resolved_root / _ASSET_PROVENANCE,
            label="README provenance asset destination",
        ),
    )
    inputs = {
        source.payload.relative_to(resolved_root).as_posix(): source_evidence,
        source.provenance.relative_to(resolved_root).as_posix(): source_provenance,
        durable.payload.relative_to(resolved_root).as_posix(): durable_payload,
        durable.provenance.relative_to(resolved_root).as_posix(): durable_provenance,
        performance_report.relative_to(resolved_root).as_posix(): report_payload,
        cargo_path.relative_to(resolved_root).as_posix(): cargo_payload,
    }
    block = render_readme_block(bundle).removeprefix(_MARKER_BEGIN).removesuffix(_MARKER_END).strip("\n")
    plan = plan_publication(
        resolved_root,
        readme_path.relative_to(resolved_root).as_posix(),
        MarkerPair(_MARKER_BEGIN, _MARKER_END),
        block,
        inputs=inputs,
        figures={
            asset_paths.payload.relative_to(resolved_root).as_posix(): source_evidence,
            asset_paths.provenance.relative_to(resolved_root).as_posix(): source_provenance,
        },
    )
    changed = publish_publication(plan)
    return PublicationSummary(current_tag=current, baseline_tag=baseline, changed_paths=changed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root (default: current directory)")
    parser.add_argument("--artifact-payload", type=Path, default=Path("target/bench-reports/performance.comparison.json"))
    parser.add_argument("--artifact-provenance", type=Path, default=Path("target/bench-reports/performance.evidence.json"))
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
                payload=_under_root(root, args.artifact_payload),
                provenance=_under_root(root, args.artifact_provenance),
            ),
            readme=_under_root(root, args.readme),
        )
    except (OSError, RuntimeError, TypeError, ValueError, ExceptionGroup) as error:
        print(f"performance-readme: {format_exception_diagnostics(error)}", file=sys.stderr)
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
