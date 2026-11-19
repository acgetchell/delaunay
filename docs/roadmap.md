# Roadmap

This page tracks release preparation and future work. The
[limitations guide](limitations.md) describes supported operations and
geometric models. Historical task snapshots live in [`archive/`](archive/).

## v0.8.5 Release Preparation

The v0.8.5 candidate builds on the published
[v0.8.4 baseline](https://github.com/acgetchell/delaunay/releases/tag/v0.8.4).
The [release tracker #696](https://github.com/acgetchell/delaunay/issues/696)
tracks native validation, benchmark and package evidence, and publication.

- [`TriangleGluing`](topology.md#intrinsic-2d-triangle-gluings) represents and
  edits intrinsic 2D surfaces with explicit edge identities, including parallel
  edges and loops.
- Rust 1.100.0 is the minimum supported toolchain. The numerical backend is
  `la-stack` 0.5.4. The [adoption record](la_stack_adoption.md) retains the
  measurements and the bounds of the numerical comparisons.
- The release workflow uses `research-repo-tools` 0.3.0. It carries one retained
  acquisition through reports, README publication, and release attachment.
  Cargo publication uses the explicit dispatch and approval in
  [RELEASING.md](RELEASING.md).
- The [rollback performance report](insertion_rollback_performance.md) records
  the touched-record journal measurements and the 4D smoke guard results for
  [#510](https://github.com/acgetchell/delaunay/issues/510).

<!-- Preserve the former release headings' public deep links. -->
<a id="v084-release-preparation"></a>
<a id="v083-release-preparation"></a>
<a id="v083-released-baseline"></a>
<a id="v078-pre-v080-cleanup-released"></a>
<a id="v080-paper-facing-api-and-topology-push-released"></a>
<a id="v081-validation-evidence-and-performance-follow-up"></a>
<a id="v082-corrected-benchmark-baseline"></a>

## Release Sequence

The [changelog](../CHANGELOG.md) and its [archives](archives/changelog/) own
completed release history. The [v0.8.0 reviewer artifact](archive/ARTIFACT.md)
records that release's reproduction procedure. The
[retained benchmark runs](performance-runs/) identify the exact measured releases
and preserve the evidence for each comparison.

### v0.9.0 and later horizon

Scientific correctness and supporting evidence determine readiness for these
future candidates.

- **Broader geometry:** constrained Delaunay triangulations
  ([#299](https://github.com/acgetchell/delaunay/issues/299)), Voronoi diagrams
  ([#63](https://github.com/acgetchell/delaunay/issues/63)), and weakly-visible
  hull facets ([#136](https://github.com/acgetchell/delaunay/issues/136)).
- **Concurrent topology mutation:** establish correctness, conflict, rollback,
  and performance evidence before adopting a scheduler
  ([#495](https://github.com/acgetchell/delaunay/issues/495)).
- **High-dimensional hull buffers:** investigate allocation beyond D=7
  ([#106](https://github.com/acgetchell/delaunay/issues/106)).
- **Public builder polish:** use downstream experience before finalizing the
  construction API for v1.0.0
  ([#497](https://github.com/acgetchell/delaunay/issues/497)).
- **Spherical integration:** extend the bounded `S^2`/`S^3` prototype to the
  broader spherical construction, realization, and editing contract
  ([#414](https://github.com/acgetchell/delaunay/issues/414)).
- **Toroidal scaling:** make `T^4`/`T^5` periodic quotient construction scalable
  and independently validated before enabling it
  ([#416](https://github.com/acgetchell/delaunay/issues/416)).

## Ongoing Performance Monitoring

- **2D–5D shared large-scale monitoring:** keep
  `just debug-large-scale-{2,3,4,5}d [n] [repair_every]` aligned so
  performance work is measured across the supported small-dimensional range
  instead of tuned for one dimension at another's expense. The `justfile` owns
  current defaults; the [benchmark guide](../benches/README.md#release-mode-debug-defaults)
  owns workload sizes and dated maintainer-hardware timings.
- **Criterion performance canaries:** keep smaller `ci_performance_suite`
  canaries for the same construction path so PR regression checks remain
  practical under Criterion's repeated sampling model.
- **Heavy manual probes:** treat thousands-point 4D and larger 5D workloads as
  explicit characterization work. The completed
  [5D feasibility investigation #342](https://github.com/acgetchell/delaunay/issues/342)
  is historical context, not an open target or an input-size guarantee.
- **Release evidence:** preserve the legacy [performance report](performance.md)
  and dated investigations with their original identities. Publish replacement
  reports and README summaries only from validated retained evidence through
  their named workflows; a documentation refresh does not regenerate timings.

## Not Implemented Today

See [feature gaps](limitations.md#feature-gaps) for current operational limits and
[`archive/todo_2026-04-23.md`](archive/todo_2026-04-23.md) for the retired
post-v0.7.5 task snapshot.
