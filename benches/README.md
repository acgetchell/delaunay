# Performance Benchmarks

This directory contains the Criterion and release-mode benchmark harnesses used
to keep Delaunay construction, topology operations, validation, and geometric
predicates fast across 2D-5D.

## Benchmark Suite Overview

| Benchmark | Purpose | Scale | Typical Runtime | Used By |
|-----------|---------|-------|-----------------|---------|
| `ci_performance_suite.rs` | Public workflow regression contract | Calibrated 2D-5D canaries | ~5-10 min | CI, baselines, `just perf-no-regressions` |
| `circumsphere_containment.rs` | Compare circumsphere predicate methods | 2D-5D fixed, 3D random, edge cases | ~5 min | Predicate tuning, summaries |
| `cold_path_predicates.rs` | Track hot/cold predicate paths | 2D-5D hot queries, near-boundary cases | ~2-5 min | Predicate optimization work |
| `profiling_suite.rs` | Large-scale construction, memory, query, validation profiling | 2D/3D 10k, 4D 3k, 5D 1k | ~2-3 hr | Manual/monthly |
| `tds_clone.rs` | `Tds::clone()` snapshot cost | Deterministic 2D-5D triangulations | ~1-3 min | Rollback design baselines |
| `topology_guarantee_construction.rs` | Cost of topology guarantee modes | 2D-5D construction cases | ~5-15 min | Manual topology policy work |

## Selection Guide

| Use Case | Command |
|----------|---------|
| Final local correctness check | `just ci` |
| Fast local PR performance guard with temporary same-machine baseline | `just perf-no-regressions` |
| Full CI benchmark suite only | `just bench-ci` |
| Persist/update the default local baseline artifact | `just perf-baseline` |
| Generate a scratch baseline without replacing the default | `just perf-baseline-to <out> [ref]` |
| Persist/update the default local baseline from a release/ref | `just perf-baseline v0.7.5` |
| Compare against an existing baseline | `just perf-compare <file>` |
| Release performance summary | `just bench-perf-summary` |
| Smoke-test benchmark harnesses | `just bench-smoke` |
| Predicate comparison | `cargo bench --profile perf --bench circumsphere_containment -- --noplot` |
| Predicate cold-path work | `cargo bench --profile perf --bench cold_path_predicates -- --noplot` |
| Large-scale scaling suite | `cargo bench --profile perf --bench profiling_suite -- --noplot` |
| One-dimension acceptance/profiling run | `just debug-large-scale-{2,3,4,5}d [n] [repair_every]` |
| Deep profiling | `cargo bench --profile perf --bench profiling_suite --features count-allocations` |

## Profiles And Local Guards

Benchmarks that publish or compare performance data use Cargo's `perf` profile:

```bash
just bench
just bench-ci
just perf-baseline
just perf-compare
just perf-no-regressions
cargo bench --profile perf --bench ci_performance_suite
```

The `perf` profile inherits from release and restores ThinLTO with
`codegen-units = 1`. `just ci` intentionally uses the normal validation path
instead: it catches formatting, lint, test, documentation, example, and
benchmark-harness compile errors rather than publishing benchmark data.

`just perf-no-regressions [threshold]` is the recommended fast performance
check before pushing a PR that touches Rust or benchmark code:

```bash
just ci
just perf-no-regressions
```

The recipe first generates a temporary same-machine dev-mode baseline for the
current GitHub `main` ref, then runs `ci_performance_suite` for the current
checkout with the shared dev-mode Criterion settings
(`--sample-size 10 --measurement-time 2 --warm-up-time 1 --noplot`) and compares
the two at the default 7.5% threshold. The temporary baseline checkout and
artifact directory are removed after the comparison.

`just perf-baseline` is optional: use it only when you intentionally want to
persist or refresh `baseline-artifact/baseline_results.txt` for later manual
comparisons.

To generate a local baseline without replacing the default persistent artifact,
write it to another directory and compare directly:

```bash
just perf-baseline-to /tmp/delaunay-main-baseline
just perf-compare /tmp/delaunay-main-baseline/baseline_results.txt
```

Use `just bench-smoke` only to check that benchmark harnesses still compile and
run with minimal samples. Smoke output is not performance data.

## CI Performance Suite

```bash
cargo bench --profile perf --bench ci_performance_suite
```

`ci_performance_suite.rs` is the stable public workflow contract. It covers:

- construction via `DelaunayTriangulation::new_with_options`
- adversarial construction
- convex hull extraction
- boundary facet traversal
- validation Levels 1-4
- incremental vertex insertion into prepared triangulations
- explicit 4D bistellar flip roundtrips

The current calibrated fixture sizes are:

| Dimension | Fixture vertices | Insert batch |
|-----------|------------------|--------------|
| 2D | 2,000 | 10 |
| 3D | 700 | 10 |
| 4D | 50 | 6 |
| 5D | 25 | 4 |

The same fixture sizes are reused for construction, adversarial construction,
validation, hull extraction, boundary traversal, and insertion bases. This keeps
the suite orthogonal: one benchmark contract covers useful real operations
without carrying separate toy construction-only cases.

## Circumsphere Containment

```bash
cargo bench --profile perf --bench circumsphere_containment -- --noplot
cargo bench --bench circumsphere_containment -- --test
```

This suite compares three predicate paths:

- `insphere`: determinant-based robust predicate path
- `insphere_distance`: explicit circumcenter plus distance comparison
- `insphere_lifted`: lifted-paraboloid determinant formulation

Fresh local perf-profile run on maintainer Apple M4 Max hardware
(2026-05-14, Rust 1.95.0):

| 3D random batch, 1000 queries | Mean |
|-------------------------------|------|
| `insphere` | 8.389 ms |
| `insphere_distance` | 39.6 µs |
| `insphere_lifted` | 23.2 µs |

| Fixed simplex | `insphere` | `insphere_distance` | `insphere_lifted` | Winner |
|---------------|------------|----------------------|-------------------|--------|
| 2D | 17.7 ns | 23.6 ns | 8.3 ns | `insphere_lifted` |
| 3D | 2.93 µs | 26.8 ns | 18.4 ns | `insphere_lifted` |
| 4D | 6.84 µs | 58.3 ns | 3.98 µs | `insphere_distance` |
| 5D | 10.6 µs | 97.8 ns | 6.18 µs | `insphere_distance` |

| Edge-case family | 2D winner | 3D winner | 4D winner | 5D winner |
|------------------|-----------|-----------|-----------|-----------|
| Boundary vertex | `insphere` (~1.3 ns) | `insphere` (~1.5 ns) | `insphere` (~1.6 ns) | `insphere` (~2.4 ns) |
| Far vertex | `insphere_lifted` (~8.3 ns) | `insphere_lifted` (~18.4 ns) | `insphere_distance` (~57.3 ns) | `insphere_distance` (~83.6 ns) |
| Near boundary | `insphere_lifted` (~8.3 ns) | `insphere_lifted` (~18.4 ns) | `insphere_distance` (~57.0 ns) | `insphere_distance` (~83.3 ns) |

The same run reported 1000/1000 agreement for each pairwise method comparison
and 1000/1000 agreement across all three methods on the random consistency
checks.

Interpretation:

- `insphere_lifted` is currently fastest for 2D/3D fixed non-boundary cases and
  for the 3D random batch.
- `insphere_distance` is currently fastest for 4D/5D fixed non-boundary cases.
- `insphere` has very fast boundary-vertex short-circuit behavior, but its
  robust determinant path is slower on non-boundary 3D-5D cases.
- Keep `insphere` as the correctness reference unless a caller has a narrowly
  measured workload-specific reason to choose a comparison helper.

## Cold Path Predicates

```bash
cargo bench --profile perf --bench cold_path_predicates -- --noplot
```

`cold_path_predicates.rs` tracks the fast f64 filter path and the exact
Bareiss fallback path used by `insphere`, `insphere_lifted`, and the orientation
predicate they invoke. The hot group uses well-separated random queries; the
near-boundary group deliberately pushes queries close to the circumsphere so
Stage 2 costs remain visible.

## Profiling Suite

```bash
cargo bench --profile perf --bench profiling_suite -- --noplot
BENCH_LARGE_SCALE=1 cargo bench --profile perf --bench profiling_suite -- --noplot
cargo bench --profile perf --bench profiling_suite --features count-allocations -- memory_profiling
PROFILING_DEV_MODE=1 cargo bench --profile perf --bench profiling_suite --features count-allocations
```

`profiling_suite.rs` is the manual optimization harness. It now owns the former
large-scale scaling workload and the deeper profiling diagnostics:

- construction through the default batch construction path
- process RSS deltas during construction
- memory allocation profiling with the optional `count-allocations` feature
- neighbor queries, vertex iteration, and cell iteration
- circumsphere query latency
- algorithmic bottlenecks
- validation component costs

Current point-count families:

| Dimension | Default counts | Large-scale override |
|-----------|----------------|----------------------|
| 2D | 1,000 / 5,000 / 10,000 | same |
| 3D | 1,000 / 5,000 / 10,000 | same |
| 4D | 1,000 / 3,000 | 1,000 / 5,000 / 10,000 with `BENCH_LARGE_SCALE=1` |
| 5D | 500 / 1,000 | same |

Use filters to scope a run:

```bash
cargo bench --profile perf --bench profiling_suite -- "construction/3D"
cargo bench --profile perf --bench profiling_suite -- "queries/neighbors"
cargo bench --profile perf --bench profiling_suite -- "queries/vertices"
cargo bench --profile perf --bench profiling_suite -- "memory_profiling"
```

`PROFILING_DEV_MODE=1` reduces the auxiliary diagnostics while iterating with
profilers such as samply. `BENCH_LARGE_SCALE=1` enables the 4D 5,000 and 10,000
vertex construction, memory, validation, and traversal cases; reserve that for
dedicated profiling hardware.

### Release-mode Debug Defaults

The `just debug-large-scale-{2,3,4,5}d [n] [repair_every]` helpers are
single-run acceptance/profiling harnesses, not Criterion benchmarks. They insert
all vertices, run repair, and validate Levels 1-4. The defaults were calibrated
locally on maintainer Apple M4 Max hardware to land near one minute per
dimension:

| Dimension | Default vertices | Generated Simplices | Total wall time |
|-----------|------------------|---------------------|-----------------|
| 2D | 36,000 | 71,887 | ~48.1 s |
| 3D | 8,000 | 52,308 | ~51.9 s |
| 4D | 900 | 21,620 | ~57.4 s |
| 5D | 140 | 8,296 | ~51.8 s |

For 5D, the debug default is 140 vertices: 150 vertices measured about
61 seconds after strict inverse postcondition checks were enabled, while 140
vertices keeps the acceptance/profiling run comfortably below one minute and
allows denser progress reporting on the same hardware.

## TDS Clone

```bash
cargo bench --profile perf --bench tds_clone -- --noplot
```

This benchmark prebuilds deterministic 2D-5D triangulations and measures only
`Tds::clone()` on the resulting topology snapshots. Use it when comparing
rollback snapshot behavior against future journaled or localized rollback
implementations.

## Topology Guarantee Construction

```bash
cargo bench --profile perf --bench topology_guarantee_construction -- --noplot
cargo bench --profile perf --bench topology_guarantee_construction -- "topology_guarantee_construction/4d"
```

This suite compares construction cost under:

- `TopologyGuarantee::Pseudomanifold`
- `TopologyGuarantee::PLManifold`
- `TopologyGuarantee::PLManifoldStrict`

It is manual-only and useful when changing insertion, repair, or validation
paths that affect topology guarantees.

## Generated Summaries

`benches/PERFORMANCE_RESULTS.md` is generated by `benchmark-utils` and is used
for release-oriented summaries:

```bash
uv run benchmark-utils generate-summary
uv run benchmark-utils generate-summary --run-benchmarks --profile perf
uv run benchmark-utils compare --baseline baseline-artifact/baseline_results.txt
uv run benchmark-utils compare-tags --old-tag vX.Y.Z --new-tag vA.B.C
```

Generated summaries should come from fresh perf-profile runs when they are used
as release evidence. For routine PR work, use `just ci` plus
`just perf-no-regressions`.
