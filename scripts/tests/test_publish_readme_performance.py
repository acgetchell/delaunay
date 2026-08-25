"""Tests for README publication from retained performance evidence."""

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

import performance_artifacts
import publish_readme_performance

if TYPE_CHECKING:
    from pathlib import Path


def _bundle(
    *,
    current: str = "v0.8.1",
    baseline: str = "v0.8.0",
) -> performance_artifacts.PerformanceBundle:
    host = performance_artifacts.HostIdentity(status="recorded", cpu="Test CPU", operating_system="Test OS", architecture="test")
    toolchain = performance_artifacts.ToolchainState(
        rustc="rustc 1.98.0",
        criterion_version="0.7.0",
        cargo_profile="perf",
        cargo_lock_sha256="a" * 64,
        harness_sha256="b" * 64,
        configuration_sha256="c" * 64,
        measurement_plan_sha256="d" * 64,
    )
    return performance_artifacts.PerformanceBundle(
        context=performance_artifacts.ArtifactContext(
            release=performance_artifacts.ReleasePair(current=current, baseline=baseline),
            statistic="median",
            suite="release-signal",
            scope="release-signal",
            measurement_mode="local-worktrees",
            current_source=performance_artifacts.SourceState(
                version=current,
                commit="a" * 40,
                ref="HEAD",
                revision_timestamp="2026-08-23T12:00:00-07:00",
                git_clean=False,
                source_state_sha256="c" * 64,
            ),
            baseline_source=performance_artifacts.SourceState(
                version=baseline,
                commit="b" * 40,
                ref=baseline,
                revision_timestamp="2026-08-01T12:00:00-07:00",
                git_clean=True,
                source_state_sha256="d" * 64,
            ),
            current_commands=(("just", "bench-latest"),),
            baseline_commands=(("cargo", "bench", "--save-baseline", baseline),),
            current_completed_targets=performance_artifacts.RELEASE_SIGNAL_TARGETS,
            baseline_completed_targets=performance_artifacts.RELEASE_SIGNAL_TARGETS,
            current_acquisition_commands=(),
            baseline_acquisition_commands=(),
            current_toolchain=toolchain,
            baseline_toolchain=toolchain,
            current_measurement_host=host,
            baseline_measurement_host=host,
            current_artifact=performance_artifacts.MeasurementArtifact(
                origin="local-run",
                content_sha256="e" * 64,
                sample_name="new",
            ),
            baseline_artifact=performance_artifacts.MeasurementArtifact(
                origin="local-run",
                content_sha256="f" * 64,
                sample_name=baseline,
            ),
            publication_host=host,
        ),
        rows=(
            performance_artifacts.PerformanceRow(
                suite="release-signal",
                scope="release-signal",
                benchmark_id="validation/validate_3d/750",
                group="validation",
                benchmark="validate_3d/750",
                coverage_status="comparable",
                coverage_note="",
                baseline=performance_artifacts.TimingEstimate(2_000_000.0, 1_800_000.0, 2_200_000.0, 0.95),
                current=performance_artifacts.TimingEstimate(1_000_000.0, 900_000.0, 1_100_000.0, 0.95),
            ),
        ),
    )


def _write_project(root: Path, *, bundle: performance_artifacts.PerformanceBundle | None = None) -> performance_artifacts.ArtifactPaths:
    retained = bundle or _bundle()
    (root / "Cargo.toml").write_text('[package]\nname = "delaunay"\nversion = "0.8.1"\n', encoding="utf-8")
    (root / "README.md").write_text(
        "# Fixture\n\n<!-- PERFORMANCE_RELEASE_TABLE:BEGIN -->\n\nNo release comparison published.\n\n<!-- PERFORMANCE_RELEASE_TABLE:END -->\n",
        encoding="utf-8",
    )
    source = performance_artifacts.ArtifactPaths(
        csv=root / "target/bench-reports/performance.csv",
        provenance=root / "target/bench-reports/performance.provenance.json",
    )
    performance_artifacts.write_bundle(source, retained)
    archive_stem = f"{retained.context.release.current}-vs-{retained.context.release.baseline}"
    durable = performance_artifacts.ArtifactPaths(
        csv=root / "docs/archive/performance/data" / f"{archive_stem}.csv",
        provenance=root / "docs/archive/performance/data" / f"{archive_stem}.provenance.json",
    )
    performance_artifacts.write_bundle(durable, retained)
    report = publish_readme_performance.render_performance_bundle(
        retained,
        evidence_paths=performance_artifacts.ArtifactPaths(
            csv=durable.csv.relative_to(root),
            provenance=durable.provenance.relative_to(root),
        ),
        evidence_state="promoted",
    )
    (root / "docs/PERFORMANCE.md").write_text(report, encoding="utf-8")
    return source


def test_publish_readme_performance_uses_retained_bundle_without_measurement_commands(tmp_path: Path) -> None:
    source = _write_project(tmp_path)

    summary = publish_readme_performance.publish_readme_performance(tmp_path, artifacts=source, readme=tmp_path / "README.md")

    assert summary.current_tag == "v0.8.1"
    assert summary.baseline_tag == "v0.8.0"
    assert summary.changed_paths
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "**v0.8.1 vs v0.8.0**" in readme
    assert "| `validation` | 1 | 2.000x |" in readme
    assert "blob/v0.8.1/docs/PERFORMANCE.md" in readme
    assert "blob/v0.8.1/docs/assets/bench/release-performance.csv" in readme
    published = performance_artifacts.ArtifactPaths(
        csv=tmp_path / "docs/assets/bench/release-performance.csv",
        provenance=tmp_path / "docs/assets/bench/release-performance.provenance.json",
    )
    assert published.csv.read_bytes() == source.csv.read_bytes()
    assert published.provenance.read_bytes() == source.provenance.read_bytes()
    assert performance_artifacts.load_bundle(published) == _bundle()


def test_publish_readme_performance_is_content_idempotent(tmp_path: Path) -> None:
    source = _write_project(tmp_path)

    first = publish_readme_performance.publish_readme_performance(tmp_path, artifacts=source, readme=tmp_path / "README.md")
    second = publish_readme_performance.publish_readme_performance(tmp_path, artifacts=source, readme=tmp_path / "README.md")

    assert first.changed_paths
    assert second.changed_paths == ()


def test_promoted_report_validation_is_independent_of_checkout_path(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    _write_project(first)
    _write_project(second)

    first_report = (first / "docs/PERFORMANCE.md").read_text(encoding="utf-8")
    second_report = (second / "docs/PERFORMANCE.md").read_text(encoding="utf-8")
    assert first_report == second_report
    assert str(first) not in first_report
    assert str(second) not in second_report


def test_publish_readme_performance_rejects_stale_current_release_before_writing(tmp_path: Path) -> None:
    source = _write_project(tmp_path, bundle=_bundle(current="v0.8.0", baseline="v0.7.8"))
    original = (tmp_path / "README.md").read_bytes()

    with pytest.raises(ValueError, match=r"Cargo\.toml is v0\.8\.1"):
        publish_readme_performance.publish_readme_performance(tmp_path, artifacts=source, readme=tmp_path / "README.md")

    assert (tmp_path / "README.md").read_bytes() == original
    assert not (tmp_path / "docs/assets/bench").exists()


@pytest.mark.parametrize("artifact", ["csv", "provenance"])
def test_publish_readme_performance_requires_complete_promoted_bundle(tmp_path: Path, artifact: str) -> None:
    source = _write_project(tmp_path)
    suffix = ".csv" if artifact == "csv" else ".provenance.json"
    durable = tmp_path / "docs/archive/performance/data" / f"v0.8.1-vs-v0.8.0{suffix}"
    durable.unlink()
    original = (tmp_path / "README.md").read_bytes()

    with pytest.raises(ValueError, match=r"promoted performance evidence is missing: .*just performance-release"):
        publish_readme_performance.publish_readme_performance(tmp_path, artifacts=source, readme=tmp_path / "README.md")

    assert (tmp_path / "README.md").read_bytes() == original
    assert not (tmp_path / "docs/assets/bench").exists()


@pytest.mark.parametrize("artifact", ["csv", "provenance"])
def test_publish_readme_performance_requires_exact_promoted_bundle(tmp_path: Path, artifact: str) -> None:
    source = _write_project(tmp_path)
    suffix = ".csv" if artifact == "csv" else ".provenance.json"
    durable = tmp_path / "docs/archive/performance/data" / f"v0.8.1-vs-v0.8.0{suffix}"
    durable.write_text(f"not the retained {artifact}\n", encoding="utf-8")
    original = (tmp_path / "README.md").read_bytes()

    with pytest.raises(ValueError, match="does not match the exact bundle"):
        publish_readme_performance.publish_readme_performance(tmp_path, artifacts=source, readme=tmp_path / "README.md")

    assert (tmp_path / "README.md").read_bytes() == original
    assert not (tmp_path / "docs/assets/bench").exists()


def test_publish_readme_performance_requires_promoted_report(tmp_path: Path) -> None:
    source = _write_project(tmp_path)
    report = tmp_path / "docs/PERFORMANCE.md"
    report.unlink()
    original = (tmp_path / "README.md").read_bytes()

    with pytest.raises(ValueError, match=r"docs/PERFORMANCE\.md.*missing"):
        publish_readme_performance.publish_readme_performance(tmp_path, artifacts=source)

    assert (tmp_path / "README.md").read_bytes() == original
    assert not (tmp_path / "docs/assets/bench").exists()


def test_publish_readme_performance_rejects_stale_promoted_report(tmp_path: Path) -> None:
    source = _write_project(tmp_path)
    report = tmp_path / "docs/PERFORMANCE.md"
    report.write_text(f"{report.read_text(encoding='utf-8')}\nstale\n", encoding="utf-8")
    original = (tmp_path / "README.md").read_bytes()

    with pytest.raises(ValueError, match="not the canonical rendering"):
        publish_readme_performance.publish_readme_performance(tmp_path, artifacts=source)

    assert (tmp_path / "README.md").read_bytes() == original
    assert not (tmp_path / "docs/assets/bench").exists()


@pytest.mark.parametrize("path_form", ["absolute", "traversal"])
def test_main_rejects_external_readme_before_any_write(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    path_form: str,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = _write_project(root)
    outside = tmp_path / "README.md"
    outside.write_text("outside repository\n", encoding="utf-8")
    original = outside.read_bytes()

    exit_code = publish_readme_performance.main(
        [
            "--root",
            str(root),
            "--artifact-csv",
            str(source.csv),
            "--artifact-provenance",
            str(source.provenance),
            "--readme",
            str(outside) if path_form == "absolute" else "../README.md",
        ]
    )

    assert exit_code == 1
    assert "README destination must be contained" in capsys.readouterr().err
    assert outside.read_bytes() == original
    assert not (root / "docs/assets/bench").exists()


def test_publish_readme_performance_rejects_symlinked_asset_destination(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    source = _write_project(root)
    (root / "docs/assets").symlink_to(outside, target_is_directory=True)
    original = (root / "README.md").read_bytes()

    with pytest.raises(ValueError, match="asset destination must be contained"):
        publish_readme_performance.publish_readme_performance(root, artifacts=source, readme=root / "README.md")

    assert (root / "README.md").read_bytes() == original
    assert list(outside.iterdir()) == []


def test_geometric_mean_speedup_is_stable_for_reciprocal_extreme_ratios() -> None:
    bundle = _bundle()
    row = bundle.rows[0]
    tiny = performance_artifacts.TimingEstimate(1e-308, 1e-308, 1e-308, 0.95)
    huge = performance_artifacts.TimingEstimate(1e308, 1e308, 1e308, 0.95)
    reciprocal = replace(
        bundle,
        rows=(
            replace(row, benchmark_id="validation/extreme-fast", benchmark="extreme-fast", baseline=huge, current=tiny),
            replace(row, benchmark_id="validation/extreme-slow", benchmark="extreme-slow", baseline=tiny, current=huge),
        ),
    )

    block = publish_readme_performance.render_readme_block(reciprocal)

    assert "| `validation` | 2 | 1.000x |" in block


def test_publish_readme_performance_rolls_back_all_destinations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_project(tmp_path)
    readme = tmp_path / "README.md"
    original = readme.read_bytes()
    real_write = publish_readme_performance._write_bytes_atomic
    failed = False

    def fail_once(path: Path, payload: bytes) -> None:
        nonlocal failed
        if path.name == "release-performance.provenance.json" and not failed:
            failed = True
            msg = "simulated publication failure"
            raise OSError(msg)
        real_write(path, payload)

    monkeypatch.setattr(publish_readme_performance, "_write_bytes_atomic", fail_once)

    with pytest.raises(OSError, match="simulated publication failure"):
        publish_readme_performance.publish_readme_performance(tmp_path, artifacts=source, readme=readme)

    assert readme.read_bytes() == original
    assert not (tmp_path / "docs/assets/bench/release-performance.csv").exists()
    assert not (tmp_path / "docs/assets/bench/release-performance.provenance.json").exists()


def test_publish_readme_performance_rolls_back_post_write_validation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_project(tmp_path)
    readme = tmp_path / "README.md"
    assets = tmp_path / "docs/assets/bench"
    assets.mkdir(parents=True)
    csv = assets / "release-performance.csv"
    provenance = assets / "release-performance.provenance.json"
    csv.write_bytes(b"old csv\n")
    provenance.write_bytes(b"old provenance\n")
    originals = {path: path.read_bytes() for path in (readme, csv, provenance)}
    real_load = publish_readme_performance.load_bundle
    calls = 0

    def fail_published_readback(paths: performance_artifacts.ArtifactPaths) -> performance_artifacts.PerformanceBundle:
        nonlocal calls
        calls += 1
        if calls == 2:
            msg = "simulated post-write validation failure"
            raise ValueError(msg)
        return real_load(paths)

    monkeypatch.setattr(publish_readme_performance, "load_bundle", fail_published_readback)

    with pytest.raises(ValueError, match="simulated post-write validation failure"):
        publish_readme_performance.publish_readme_performance(tmp_path, artifacts=source, readme=readme)

    assert {path: path.read_bytes() for path in originals} == originals


def test_publish_readme_performance_reports_post_validation_and_rollback_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_project(tmp_path)
    real_load = publish_readme_performance.load_bundle
    real_write = publish_readme_performance._write_bytes_atomic
    load_calls = 0
    write_calls = 0

    def fail_published_readback(paths: performance_artifacts.ArtifactPaths) -> performance_artifacts.PerformanceBundle:
        nonlocal load_calls
        load_calls += 1
        if load_calls == 2:
            msg = "simulated post-write validation failure"
            raise ValueError(msg)
        return real_load(paths)

    def fail_first_rollback(path: Path, payload: bytes) -> None:
        nonlocal write_calls
        write_calls += 1
        if write_calls == 4:
            msg = "simulated rollback failure"
            raise OSError(msg)
        real_write(path, payload)

    monkeypatch.setattr(publish_readme_performance, "load_bundle", fail_published_readback)
    monkeypatch.setattr(publish_readme_performance, "_write_bytes_atomic", fail_first_rollback)

    with pytest.raises(RuntimeError, match=r"post-write validation failure.*rollback also failed.*rollback failure"):
        publish_readme_performance.publish_readme_performance(tmp_path, artifacts=source, readme=tmp_path / "README.md")
