# Performance Benchmarks

This directory contains the Criterion and release-mode benchmark harnesses used
to keep Delaunay construction, topology operations, validation, and geometric
predicates fast across 2D-5D.

## Benchmark Suite Overview

| Benchmark | Purpose | Scale | Typical Runtime | Used By |
|-----------|---------|-------|-----------------|---------|
| `allocation_hot_paths.rs` | Construction/query/barycenter allocation contracts | Calibrated 2D-5D canary fixtures | ~1-2 min | Manual allocation checks |
| `ci_performance_suite.rs` | Public workflow regression contract | Calibrated 2D-5D canaries | ~5-10 min | CI, baselines, `just perf-no-regressions` |
| `circumsphere_containment.rs` | Compare circumsphere predicate methods | 2D-5D fixed, 3D random, edge cases | ~5 min | Predicate tuning, summaries |
| `cold_path_predicates.rs` | Track hot/cold predicate paths | 2D-5D hot queries, near-boundary cases | ~2-5 min | Predicate optimization work |
| `delaunay_repair.rs` | Flip-based Delaunay repair plus transaction-pressure cases | 2D-5D repair-convergent fixtures | ~2-5 min | Repair tuning |
| `pachner_stress.rs` | Unified Pachner move API stress | Accepted 4D microcases plus 3D/4D Monte Carlo sequences | Manual | Monte-Carlo move workflow tuning |
| `pl_manifold_repair.rs` | Over-shared facet and targeted topology repair | 2D/3D synthetic repair fixtures | <1 min | PL-manifold repair tuning |
| `profiling_suite.rs` | Large-scale construction, memory, query, validation profiling | 2D/3D 10k, 4D 3k, 5D 1k | ~2-3 hr | Manual/monthly |
| `delete_vertex.rs` | Vertex deletion and rollback cost | 2D-5D fixed cases | ~1-5 min | Vertex deletion |
| `locate.rs` | Point-location facet-walk latency (no-hint vs exact-hint) | 2D-5D fixed cases | ~1-3 min | Locate/walk tuning |
| `tds_clone.rs` | `Tds::clone()` snapshot cost | Deterministic 2D-5D triangulations | ~1-3 min | Rollback design baselines |
| `topology_guarantee_construction.rs` | Cost of topology guarantee modes | 2D-5D construction cases | ~5-15 min | Manual topology policy work |

## Selection Guide

| Use Case | Command |
|----------|---------|
| Final local correctness check | `just ci` |
| Quick local large-scale wall-clock guard | `just perf-large-scale-smoke` |
| Fast local PR performance guard with temporary same-machine baseline | `just perf-no-regressions` |
| Compare current branch against a local release/ref baseline | `just perf-vs-ref v0.7.8` |
| Full CI benchmark suite only | `just bench-ci` |
| Allocation-contract microbenchmarks | `just bench-allocations` |
| Persist/update the default local baseline artifact | `just perf-baseline` |
| Generate a scratch baseline without replacing the default | `just perf-baseline-to <out> [ref]` |
| Persist/update the default local baseline from a release/ref | `just perf-baseline v0.7.5` |
| Compare against an existing baseline | `just perf-compare <file>` |
| Release performance summary | `just bench-perf-summary` |
| Durable latest-version benchmark baseline | GitHub Release asset `delaunay-vX.Y.Z-criterion-baseline.tar.gz` |
| Smoke-test benchmark harnesses | `just bench-smoke` |
| Allocation hot-path contracts | `cargo bench --profile perf --bench allocation_hot_paths --features count-allocations -- --noplot` |
| Predicate comparison | `cargo bench --profile perf --bench circumsphere_containment -- --noplot` |
| Predicate cold-path work | `cargo bench --profile perf --bench cold_path_predicates -- --noplot` |
| Flip-based Delaunay repair | `cargo bench --profile perf --bench delaunay_repair -- --noplot` |
| Flip-repair transaction pressure | `cargo bench --profile perf --bench delaunay_repair -- repair_transaction_pressure --noplot` |
| Unified Pachner move stress | `cargo bench --profile perf --bench pachner_stress -- --noplot` |
| PL-manifold repair path | `cargo bench --profile perf --features bench --bench pl_manifold_repair -- --noplot` |
| Large-scale scaling suite | `cargo bench --profile perf --bench profiling_suite -- --noplot` |
| Vertex deletion mutation baseline | `cargo bench --profile perf --bench delete_vertex -- --noplot` |
| Point-location facet walk | `cargo bench --profile perf --bench locate -- --noplot` |
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
Workspace-wide benchmark recipes enable `--features bench` so feature-gated
repair fixtures are compiled; direct commands for those harnesses must pass the
same feature explicitly.

`just perf-large-scale-smoke [max_secs]` is a coarse local wall-clock guard for
the release-mode large-scale harness. It runs the same 2D-5D cases as
`just debug-large-scale-{2,3,4,5}d` with their local defaults, caps each test
runtime at 60 seconds by default, and reports all failing dimensions before
exiting. At the end it also prints a compact 2D-5D summary table with the
reported insertion wall time, total wall time, and pass/fail status so you do
not have to scroll through the full nextest output. This is useful while
iterating, but it is not a Criterion comparison and should not be treated as
published benchmark data. Run it before pushing Rust or benchmark changes to
catch obvious local performance drift early.

For ordinary Rust or benchmark pushes, use the quick smoke guard with the usual
correctness checks:

```bash
just ci
just perf-large-scale-smoke
```

`just perf-no-regressions [threshold]` is the fuller branch-vs-main comparison
for performance-sensitive changes and PR-ready work:

```bash
just perf-no-regressions
```

The recipe first generates a temporary same-machine dev-mode baseline for the
current GitHub `main` ref, then runs `ci_performance_suite` for the current
checkout with the shared dev-mode Criterion settings
(`--sample-size 10 --measurement-time 2 --warm-up-time 1 --noplot`) and compares
the two at the default 7.5% threshold. The temporary baseline checkout and
artifact directory are removed after the comparison.

To compare the current branch against a specific release or ref, use
`just perf-vs-ref`:

```bash
just perf-vs-ref v0.7.8
```

This reuses the same cached same-machine baseline flow as
`just perf-no-regressions`, but keys the cache and report path by the requested
ref. It reports overall performance using the total matched benchmark time,
while individual benchmark regressions remain warnings in the report.

`just perf-baseline` is optional: use it only when you intentionally want to
persist or refresh `baseline-artifact/baseline_results.txt` for later manual
same-machine comparisons. Local baseline directories are ignored by git, so
they are the right place to keep developer-machine numbers without mixing them
with Ubuntu GitHub Actions release baselines.

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

- construction via `DelaunayTriangulation::builder(...).construction_options(...).build()`
- adversarial construction
- convex hull extraction
- boundary facet traversal
- validation Levels 1-4
- incremental vertex insertion into prepared triangulations
- explicit 2D-5D bistellar flip roundtrips

The current calibrated fixture sizes are:

| Dimension | Fixture vertices | Insert batch |
|-----------|------------------|--------------|
| 2D | 4,000 | 10 |
| 3D | 750 | 10 |
| 4D | 75 | 6 |
| 5D | 25 | 4 |

The same fixture sizes are reused for construction, adversarial construction,
validation, hull extraction, boundary traversal, and insertion bases. This keeps
the suite orthogonal: one benchmark contract covers useful real operations
without carrying separate toy construction-only cases. The normal construction
cases target roughly one second on release hardware; adversarial construction
uses the same vertex counts and may take longer.

During setup, the suite prints machine-readable `api_benchmark` manifest lines
and `api_benchmark_metric` construction lines. `benchmark-utils` stores those
as sidecars under `target/criterion/` so generated summaries can distinguish
the current `ci_performance_suite` contract from stale Criterion directories and
display generated simplex counts beside the current construction timings.

For a quick metric-only refresh while investigating a construction case, set
`DELAUNAY_BENCH_EXPORT_METRICS=1`. With a `tds_new` Criterion filter, the
benchmark emits only the selected construction metric and exits before sampling:

```bash
DELAUNAY_BENCH_EXPORT_METRICS=1 \
  cargo bench --profile perf --bench ci_performance_suite -- \
  "tds_new_3d/tds_new/750"
```

Use `just bench-perf-summary` for release summaries; it runs the full perf
profile summary workflow and captures the construction metrics automatically.

## Pachner Stress

```bash
cargo bench --profile perf --bench pachner_stress -- --noplot
```

`pachner_stress.rs` contains two layers:

- accepted-move microcases for the unified 4D Pachner API facade
- manual Monte Carlo stress cases for 3D and 4D long-run topology stability

The Monte Carlo cases default to 10,000 vertices in 3D and 1,000 vertices in
4D, with 100,000 attempted random Pachner moves and topology validation every
1,000 attempts. They validate topology plus the Level 4 embedding invariant
that arbitrary Pachner moves are expected to preserve; Level 5 Delaunay
validity is not a postcondition of random topology edits. The move stream runs
through the `markov-chain-monte-carlo` delayed proposal API with a flat target,
so successfully planned Pachner proposals commit with 100% acceptance while
invalid local candidates are recorded as no-proposal self-loops. Each measured
sequence emits a `pachner_stress_metric` line with accepted/rejected attempts,
proposal diagnostics, validation time, final simplex count, and RSS memory
counters. Validation failures include recent MCMC trace rows so long chains can
be diagnosed by step, outcome, and topology size.

Useful overrides:

```bash
DELAUNAY_PACHNER_STRESS_ATTEMPTS=10000 \
DELAUNAY_PACHNER_STRESS_VERTICES_4D=250 \
cargo bench --profile perf --bench pachner_stress -- "monte_carlo/4d"
```

Supported override families are `DELAUNAY_PACHNER_STRESS_VERTICES`,
`DELAUNAY_PACHNER_STRESS_ATTEMPTS`,
`DELAUNAY_PACHNER_STRESS_VALIDATE_EVERY`,
`DELAUNAY_PACHNER_STRESS_KEY_REFRESH_EVERY`, and
`DELAUNAY_PACHNER_STRESS_SEED`. Append `_3D` or `_4D` for a
dimension-specific value.

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
- neighbor queries, vertex iteration, and simplex iteration
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
| 3D | 7,500 | Recomputed by harness | Near one minute |
| 4D | 800 | 19,141 | ~51.8 s |
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

## PL-Manifold Repair

```bash
cargo bench --profile perf --features bench --bench pl_manifold_repair -- --noplot
```

This benchmark measures `repair_facet_oversharing` on controlled 3D fixtures
where each cluster has one codimension-1 facet shared by three tetrahedra. The
repair removes the deliberately skinny third tetrahedron and then removes its
unique apex as an orphan vertex.

It also measures the targeted `repair_pl_manifold_topology` stages added for
boundary-ridge multiplicity, ridge-link, and vertex-link repair. Those cases use
validated 2D/3D fixtures so fixture construction and contract checks stay
outside Criterion's measured closures. Vertex-link cases use a smaller maximum
cluster count because each torus-link cluster consumes more of the default
targeted repair budget.

The harness requires `--features bench` because normal public constructors must
reject over-shared facets and targeted non-PL-manifold topology. `TopologyGuarantee::Pseudomanifold`
is not a bypass for these fixtures: pseudomanifold topology still requires
facet degree 1 or 2.

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
for release-oriented summaries. Durable per-version comparison baselines are
stored as GitHub Release assets named
`delaunay-vX.Y.Z-criterion-baseline.tar.gz`; the performance-regression
workflow downloads the latest stable release asset and compares the current
Ubuntu GitHub Actions run against that released-version Ubuntu baseline. Use
local baselines for developer-machine comparisons.

```bash
uv run benchmark-utils generate-summary
uv run benchmark-utils generate-summary --run-benchmarks --profile perf
uv run benchmark-utils write-baseline --ref vX.Y.Z --output baseline_results.txt
uv run benchmark-utils compare --baseline baseline-artifact/baseline_results.txt
uv run benchmark-utils compare-tags --old-tag vX.Y.Z --new-tag vA.B.C
```

Generated summaries should come from fresh perf-profile runs when they are used
as release evidence. For routine PR work, use `just ci` plus
`just perf-no-regressions`. Release-baseline comparisons write
`benches/main_vs_release_compare_results.txt`; PR/ref comparisons write
`benches/worktree_vs_<ref>_compare_results.txt`. The comparison commands print
a short pass/regression/error status and the report path before exiting. The
local PR/ref guard fails on total matched-time regressions, while individual
regressions and improvements are surfaced in the report.

The generated `Triangulation Data Structure Performance` section is intentionally
first: it is built from the current `target/criterion` construction results,
the `ci_performance_suite` metric sidecar, and the latest run metadata sidecar.
That makes the Criterion run date and generated simplex counts apply to the
public API tables that follow.
