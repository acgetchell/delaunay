# Performance Benchmarks

This directory contains the Criterion and release-mode benchmark harnesses used
to keep Delaunay construction, topology operations, validation, and geometric
predicates fast across 2D-5D.

## Benchmarking Model

The benchmark system answers three different questions:

1. **Are the scientific invariants maintained?** Use the normal validation
   gate, invariant tests, and known-answer checks. Benchmark harnesses should
   fail before publishing timings when the measured workflow violates
   triangulation, predicate, topology, or diagnostic invariants.
2. **Did this change move performance?** Use same-machine local comparisons and
   targeted benchmarks while developing.
3. **What can we say about a release?** Use release-signal benchmarks, durable
   release artifacts, environment metadata, and one curated committed report.

This split is the reusable pattern for Delaunay and sibling scientific crates:
invariants first, local evidence second, release evidence third. The exact
benchmark targets can differ by domain, but the command roles should stay
boring and consistent.

## How The Workflow Fits Together

Use `just ci` as the invariant validation gate and routine pre-commit check. It
catches formatting, lint, test, documentation, example, and benchmark-harness
compile errors. It does not publish benchmark data and does not replace
measured before/after evidence for performance-sensitive code.

Use same-machine local comparisons for branch and PR work. `just
perf-no-regressions` and `just perf-vs-ref <ref>` run the dev-mode
`ci_performance_suite`, generate or reuse text baselines, and write reports
under `benches/`. These commands are intentionally cheaper and less formal than
the release-signal workflow.

Use release-signal benchmarks for release evidence. They are slower, more
formal, and tied to release artifacts and report metadata. Publishing a GitHub
Release triggers `.github/workflows/release-benchmarks.yml`, which attaches
`delaunay-vX.Y.Z-criterion-baseline.tar.gz` to the release. That archive
contains `PERFORMANCE_RESULTS.md`, `baseline_results.txt`, raw Criterion data
under `criterion/`, and `metadata.json`.

Common maintainer flows:

- Before committing or handing off code: run `just ci`.
- Before pushing performance-sensitive Rust or benchmark changes: run
  `just perf-large-scale-smoke`, then `just perf-no-regressions` when the work
  is PR-ready.
- During release PR preparation: run `just performance-release`, then
  `just performance-readme`, to update the curated report, archive its
  predecessor, and publish the matching README snapshot.
- After a GitHub Release publishes: confirm the release benchmark workflow
  attached `delaunay-vX.Y.Z-criterion-baseline.tar.gz`; use
  `just performance-github-assets 'current-tag' 'baseline-tag'` when you need a
  report from stored release assets.

Criterion saved-baseline reports use Cargo's Criterion output under
`target/criterion/`. Save a baseline with `just bench-save-baseline <tag>` from
the baseline checkout, then run fresh
release-signal measurements from the current checkout with `just bench-latest`
and render a Markdown comparison with `just bench-compare <baseline>`.
`just bench-latest-vs-last` combines the fresh measurement and report rendering
for the common local saved-baseline case. Use `just performance-local` when you
want the tool to manage isolated baseline/current worktrees for you.

The canonical `performance-*` release recipes consume local worktrees,
published release assets, or retained inputs:

- `just performance-local` runs local release-signal benchmarks in isolated
  temporary worktrees and retains a Markdown/CSV/provenance bundle under
  `target/bench-reports/performance.*` without promoting documentation.
- `just performance-github-assets` compares stored GitHub Release assets
  without local Cargo benchmark runs and retains
  `target/bench-reports/github-assets-performance.*`. New archives must carry
  complete versioned measurement metadata bound to the requested clean tag.
  Existing legacy archives remain loadable as provenance-limited absolute
  timing evidence. Ratios are always suppressed because two independently
  hosted release runs are separate measurement sessions.
- `just performance-release` performs the local measurement and retention
  workflow, reload-validates the pair, promotes `docs/PERFORMANCE.md`, and
  archives both the prior report and the exact promoted CSV/provenance pair.
- `just performance-doc` reloads a retained CSV/provenance pair and performs
  only the documentation promotion. It runs no Cargo benchmarks or measurement
  worktrees and rejects incomplete, invalid, stale, same-version, or
  scientifically non-comparable inputs.
- `just performance-readme` validates the retained promoted bundle, then
  atomically publishes a compact README table and the canonical README-owned
  CSV/provenance pair. It runs no benchmarks.

Promotion uses per-file atomic replacement and rolls back caught failures. A
hard process or machine interruption can stop between replacements. Inspect
`docs/PERFORMANCE.md`, `docs/archive/performance/` and its `data/` directory,
`README.md`, and `docs/assets/bench/`, then rerun the owning idempotent command:
`just performance-doc` or `just performance-readme` for an already-retained
bundle.

Exact-renaming `perf-*` compatibility aliases are intentionally absent. The
remaining `perf-*` recipes are Delaunay-specific regression, stress, and
profiling workflows with distinct semantics.

Use explicit tag pairs only for release repair or regeneration. Both tags are
required together; a partial pair fails before fetching or benchmarking:

```bash
TAG=vX.Y.Z
PREVIOUS_TAG=vX.Y.W
just performance-release "$TAG" "$PREVIOUS_TAG"
```

Do not use release-comparison commands as a routine pre-`just ci` step.
Temp-worktree commands apply tracked changes from the current checkout by
default; untracked benchmark or script files must be added to git before they
can affect the generated report.

## Release Comparison Artifacts

Artifact ownership is deliberately narrow:

| Path | Tracked | Producer | Role |
|------|---------|----------|------|
| `target/bench-reports/performance.md` | No | `performance-local`, `performance-release`, `performance-doc` | Rendered local or release comparison |
| `target/bench-reports/performance.csv` | No | `performance-local`, `performance-release` | Scratch copy of canonical versioned timing and coverage data |
| `target/bench-reports/performance.provenance.json` | No | `performance-local`, `performance-release` | Scratch provenance and CSV binding evidence |
| `target/bench-reports/github-assets-performance.*` | No | `performance-github-assets` | Provenance-validated bundle sourced from release archives |
| `docs/PERFORMANCE.md` | Yes | `performance-release`, `performance-doc` | Latest curated distinct-release report |
| `docs/archive/performance/` | Yes | `performance-release`, `performance-doc` | Older curated distinct-release reports |
| `docs/archive/performance/data/` | Yes | `performance-release`, `performance-doc` | Exact CSV/provenance evidence for each new promoted report |
| `README.md` | Yes | `performance-readme` | Compact snapshot of the latest retained and promoted comparison |
| `docs/assets/bench/release-performance.*` | Yes | `performance-readme` | Exact README-owned CSV/provenance snapshot |
| `delaunay-vX.Y.Z-criterion-baseline.tar.gz` | Release asset | release benchmark workflow | Raw Criterion data and versioned measurement metadata |

CSV, not Parquet, is canonical for the retained comparison because this dataset
is a small audit record that should remain reviewable in diffs and readable
without Polars or PyArrow. Jupyter notebooks may derive Parquet caches when that
materially helps larger analyses, but those caches are disposable and must be
reproducible from the validated CSV; `performance-doc` does not accept them.

The CSV schema is versioned and deterministic. The provenance JSON binds its
exact digest and row count to the release pair, suite/scope selection, commands,
source states, toolchains, separate per-revision host identities, confidence
levels, Criterion content digests, sample names, and any acquisition commands.
One-sided and environmentally non-comparable coverage is retained explicitly
rather than silently dropped. Ratios require compatible hosts, toolchains,
harnesses, normalized measurement plans, completed-target manifests, and
confidence levels. Broad configuration digests remain provenance but do not
block ordinary cross-release comparisons. When a release adds or removes a
benchmark target, the harness and measurement-plan comparison is restricted to
the canonical targets completed by both revisions; the union must still cover
the current release-signal plan. Promotion requires complete coverage for an
unchanged target plan, or an explicitly valid target transition with one-sided
rows retained, plus at least one comparable row. Scratch reports identify their
adjacent retained pair; promoted reports identify the exact durable pair under
`docs/archive/performance/data/`. The Markdown report is always rendered from a
successfully reloaded pair, so it does not become an independent source of
timing data.

The default release-signal suite is deliberately curated. It favors stable,
release-relevant public behavior over every exploratory benchmark in this
directory. Broader math-kernel benchmark coverage is tracked separately in
[#513](https://github.com/acgetchell/delaunay/issues/513) so future additions
can decide whether each kernel belongs in `bench-latest`, `profiling_suite`,
allocation checks, or targeted diagnostics.

## Benchmark Suite Overview

| Benchmark | Purpose | Scale | Typical Runtime | Used By |
|-----------|---------|-------|-----------------|---------|
| `allocation_hot_paths.rs` | Bootstrap/insert/query/barycenter allocations | Calibrated 2D-5D canaries | ~1-2 min | Manual allocation checks |
| `checkpoint_serialization.rs` | Manifest and JSON checkpoint write/load | 64-vertex 2D owner | <1 min | Checkpoint tuning |
| `ci_performance_suite.rs` | Public workflow regression contract | Calibrated 2D-5D canaries | ~5-10 min | CI, baselines, `just perf-no-regressions` |
| `circumsphere_containment.rs` | Circumsphere predicates and solves | 2D-5D predicates, 3D LU/exact solves | ~5 min | Predicate/circumcenter tuning |
| `cold_path_predicates.rs` | Track predicate paths | Hot, centered, and certified exact cases in 2D-5D | ~2-5 min | Predicate tuning |
| `delaunay_repair.rs` | Flip-based Delaunay repair plus transaction-pressure cases | 2D-5D repair-convergent fixtures | ~2-5 min | Repair tuning |
| `pachner_stress.rs` | Unified Pachner move API stress | Accepted 4D microcases plus forward/inverse round trips | Manual | Pachner move workflow tuning |
| `pl_manifold_repair.rs` | Over-shared facet and targeted topology repair | 2D/3D synthetic repair fixtures | <1 min | PL-manifold repair tuning |
| `profiling_suite.rs` | Large-scale construction, memory, query, validation profiling | 2D/3D 10k, 4D 3k, 5D 1k | ~2-3 hr | Manual/monthly |
| `realization_validation.rs` | Level 4 narrow phase and whole validation | Realistic/near-degenerate 2D-5D | ~1-5 min | Level 4 tuning |
| `delete_vertex.rs` | Vertex deletion and rollback cost | 2D-5D fixed cases | ~1-5 min | Vertex deletion |
| `locate.rs` | Point-location facet-walk latency (no-hint vs exact-hint) | 2D-5D fixed cases | ~1-3 min | Locate/walk tuning |
| `tds_clone.rs` | `Tds::clone()` snapshot cost | Deterministic 2D-5D triangulations | ~1-3 min | Rollback design baselines |
| `topology_guarantee_construction.rs` | Cost of PL-manifold validation audit cadences | 2D-5D construction cases | ~45-60 min | Manual topology policy work |

## Selection Guide

| Use Case | Command |
|----------|---------|
| Final local invariant validation gate | `just ci` |
| Measure checkpoint manifest and JSON write/load paths | `cargo bench --bench checkpoint_serialization --features bench -- --noplot` |
| Quick local large-scale wall-clock guard | `just perf-large-scale-smoke` |
| Fast local PR performance guard with cached same-machine main baseline | `just perf-no-regressions` |
| Compare current branch against a local release/ref baseline | `just perf-vs-ref v0.7.8` |
| Full CI benchmark suite only | `just bench-ci` |
| Run curated release-signal Criterion measurements | `just bench-latest` |
| Compare latest measurements against saved `last` Criterion baseline | `just bench-latest-vs-last` |
| Render a Markdown report from existing Criterion results | `just bench-compare [baseline] [suite] [scope]` |
| Save a named Criterion baseline | `just bench-save-baseline <tag> [suite]` |
| Save the previous-release Criterion baseline as `last` | `just bench-save-baseline last` |
| Compare current tree against latest published release locally | `just performance-local` |
| Compare stored GitHub Release benchmark assets | `just performance-github-assets [current-tag baseline-tag]` |
| Measure, retain, validate, and promote release docs | `just performance-release [current-tag baseline-tag]` |
| Publish the retained comparison snapshot to the README | `just performance-readme` |
| Promote docs from a retained CSV/provenance pair | `just performance-doc` |
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
| Unified Pachner move stress | `just pachner-stress` |
| PL-manifold repair path | `cargo bench --profile perf --features bench --bench pl_manifold_repair -- --noplot` |
| Large-scale scaling suite | `cargo bench --profile perf --bench profiling_suite -- --noplot` |
| Level 4 realization validation | `cargo bench --profile perf --bench realization_validation -- --noplot` |
| Vertex deletion mutation baseline | `cargo bench --profile perf --bench delete_vertex -- --noplot` |
| Point-location facet walk | `cargo bench --profile perf --bench locate -- --noplot` |
| One-dimension acceptance/profiling run | `just debug-large-scale-{2,3,4,5}d [n] [repair_every]` |
| Isolated 32k-vertex 2D Level 4 acceptance run | `DELAUNAY_LARGE_DEBUG_VALIDATION=realization just debug-large-scale-2d 32000` |
| Deep profiling | `cargo bench --profile perf --bench profiling_suite --features count-allocations` |

The checkpoint benchmark uses one deterministic 2D owner because its purpose is
to isolate digest construction and codec overhead, whose dimension-generic
implementation is shared across dimensions. Correctness and exact replay remain
covered separately by the 2D–5D checkpoint tests; expanding the timed matrix
would primarily repeat topology construction cost rather than sharpen this
serialization signal.

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
invariant checks:

```bash
just ci
just perf-large-scale-smoke
```

`just perf-no-regressions [threshold]` is the fuller branch-vs-main comparison
for performance-sensitive changes and PR-ready work:

```bash
just perf-no-regressions
```

The recipe resolves the current GitHub `main` ref, reuses or refreshes a
cached same-machine dev-mode baseline under
`baseline-artifacts/perf-no-regressions/`, then runs `ci_performance_suite` for
the current checkout with the shared dev-mode Criterion settings
(`--sample-size 10 --measurement-time 2 --warm-up-time 1 --noplot`) and compares
the two at the default 7.5% threshold. The cache is keyed by the resolved
`main` commit and local Rust compiler, so repeated checks do not rerun the
baseline unless `main`, the compiler, or the benchmark contract changes.

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

Use the `bench-latest` family when you want a Criterion saved-baseline report
that mirrors the sibling `la-stack` workflow:

```bash
# In the baseline checkout, usually the previous release:
just bench-save-baseline last

# In the current checkout:
just bench-latest-vs-last
just bench-compare last
```

`just bench-latest` runs Delaunay's curated release-signal set:
`ci_performance_suite`, `circumsphere_containment`, `cold_path_predicates`,
`locate`, and `realization_validation`. This set covers public construction and
validation workflows, circumsphere/insphere query families, hot and cold
predicate paths, point location, and Level 4 exact intersection validation
without pulling in broad profiling, allocation-only runs, or the manual
topology policy suite. Use `just bench-compare <baseline>` to render
`target/bench-reports/performance.md` from existing Criterion `new` output and a
saved baseline such as `last` or `v0.7.8`. If the baseline and current
checkouts are easy to confuse, prefer `just performance-local`. Use
`uv run --locked benchmark-utils bench-compare --scope all-benches` only when you
explicitly want an exploratory report over every Criterion result already
present under `target/criterion/`.

Use the `performance-*` family for retained release-to-release reports:

```bash
just performance-local
just performance-github-assets
just performance-release
just performance-readme
just performance-doc
just performance-release "$TAG" "$PREVIOUS_TAG"
```

`performance-local` and `performance-github-assets` retain Markdown, CSV, and
provenance JSON bundles without changing tracked documentation.
`performance-release` measures locally, retains and reload-validates the bundle,
then promotes tracked docs. The canonical release flow immediately follows it
with `performance-readme` to publish the matching README table and durable
assets. `performance-doc` performs only the reload, rendering, and promotion
from retained inputs. Explicit `<current-tag>
<baseline-tag>` arguments repair or regenerate a specific distinct release
pair; neither tag may be supplied alone.

## CI Performance Suite

```bash
cargo bench --profile perf --bench ci_performance_suite
```

`ci_performance_suite.rs` is the stable public workflow contract. It covers:

- construction via `DelaunayTriangulationBuilder::new(...).construction_options(...).build()`
- adversarial construction
- convex hull extraction
- boundary facet traversal
- cumulative validation Levels 1-5, including Level 1-2 TDS structure, Level 3
  topology, Level 4 realization, and Level 5 Delaunay predicate checks
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
just pachner-stress
just pachner-stress-3d
just pachner-stress-4d
just bench-pachner-stress
```

`pachner_stress.rs` contains two Criterion layers:

- accepted-move microcases for the unified 4D Pachner API facade
- forward/inverse round-trip cases for the same 4D move supports

The `just pachner-stress` recipe runs 9,000 vertices in 3D and 1,000 vertices in
4D; the dimension-specific recipes use the same defaults. These recipes default
to 100 attempted Pachner steps with topology validation every 10 attempts. The
direct `pachner-stress` binary has larger soak-test defaults: 10,000
vertices in 3D or 1,000 in 4D, 100,000 attempts, and validation every 1,000
attempts.

The default `round-trip` mode commits a forward move and its inverse when a
candidate is locally valid. The `random-walk` mode commits accepted moves over
an evolving triangulation. Both modes write progress CSV and summary JSON under
`target/pachner_stress/` and also emit parseable stdout telemetry: one
`pachner_stress_source` line for the prepared triangulation,
`pachner_stress_progress` lines after each validation cadence, and a final
`pachner_stress_metric` line with accepted/rejected attempts, proposal
diagnostics, validation time, and final topology size.

Use `just bench-pachner-stress*` for Criterion timing evidence. Criterion
requires at least 10 samples. The benchmark recipe measures stable accepted
move fixtures and corresponding forward/inverse round trips.

The stress cases currently validate topology scope only (Levels 1-3). The
large Level 4 realization overlap scan is deferred to dedicated realization
validation, and Level 5 Delaunay validity is not a postcondition of arbitrary
topology edits. Invalid local candidates are counted as candidate misses or
proposal rejections, while successfully planned Pachner proposals commit
directly.

Useful overrides:

```bash
just pachner-stress-4d 10000 250 1000 target/pachner_stress/4d random-walk
just bench-pachner-stress 25
```

The benchmark accepts `MOVES_PER_SAMPLE` for local fixture batch size. The
`just bench-pachner-stress*` recipes accept a Criterion sample count and enforce
Criterion's minimum of 10 samples.

## Circumsphere Containment

```bash
cargo bench --profile perf --bench circumsphere_containment -- --noplot
cargo bench --bench circumsphere_containment -- --test
```

This suite compares three predicate paths:

- `insphere`: determinant-based robust predicate path
- `insphere_distance`: explicit circumcenter plus distance comparison
- `insphere_lifted`: lifted-paraboloid determinant formulation

The `circumcenter/solve_path` group separately compares a regular 3D LU solve
with a near-coplanar fixture that LU classifies as numerically singular and the
exact rounded solve resolves. Both fixtures are validated before timing.

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
predicate they invoke. The hot group uses well-separated random queries. The
historically named `near_boundary` group samples around the circumsphere center
and remains only for saved-baseline compatibility; it does not establish exact
fallback. The `exact_fallback` group validates known cospherical 2D-4D fixtures
before timing them, making it the benchmark's correctness-certified Stage-2
signal. The 5D hot group remains Stage-2-dominant because the fast determinant
bound is unavailable at that matrix size.

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

This benchmark measures the internal facet-over-sharing stage on controlled 3D fixtures
where each cluster has one codimension-1 facet shared by three tetrahedra. The
repair removes the deliberately skinny third tetrahedron and then removes its
unique apex as an orphan vertex.

It also measures the targeted stages used by the public consuming
`repair_pl_manifold_tds` transformation for
boundary-ridge multiplicity, ridge-link, and vertex-link repair. Those cases use
validated 2D/3D fixtures so fixture construction and contract checks stay
outside Criterion's measured closures. Vertex-link cases use a smaller maximum
cluster count because each torus-link cluster consumes more of the default
targeted repair budget.

The harness requires `--features bench` because normal public constructors must
reject over-shared facets and targeted non-PL-manifold topology. `TopologyGuarantee::Pseudomanifold`
is not a bypass for these fixtures: pseudomanifold topology still requires
facet degree 1 or 2.

## Topology Validation Policy Construction

```bash
cargo bench --profile perf --bench topology_guarantee_construction -- --noplot
cargo bench --profile perf --bench topology_guarantee_construction -- "topology_guarantee_construction/4d"
```

This suite compares PL-manifold construction cost under two audit cadences:

- `TopologyGuarantee::PLManifold` with `ValidationPolicy::ExplicitOnly`
- `TopologyGuarantee::PLManifold` with `ValidationPolicy::Always`

The pseudomanifold mode is excluded because its weaker Level 3 proof cannot
authorize every repair needed to preserve the Level 5 Delaunay invariant on
these large random construction fixtures.

Each mode is validated once as an untimed preflight; Criterion measures only
construction and the selected insertion-time audit cadence. The suite is
excluded from the pull-request regression set, remains available for manual
topology-policy work, and is also archived by the release benchmark workflow.

## Generated Summaries

`benches/PERFORMANCE_RESULTS.md` is generated by `benchmark-utils` and is used
for release-oriented summaries. Durable per-version comparison baselines are
stored as GitHub Release assets named
`delaunay-vX.Y.Z-criterion-baseline.tar.gz`; the performance-regression
workflow downloads the latest stable release asset and compares the current
Ubuntu GitHub Actions run against that released-version Ubuntu baseline. Use
local baselines for developer-machine comparisons, and use `docs/PERFORMANCE.md`
for the single curated release-to-release comparison that should stay visible in
active docs.

```bash
uv run --locked benchmark-utils generate-summary
uv run --locked benchmark-utils generate-summary --run-benchmarks --profile perf
uv run --locked benchmark-utils bench-compare last
uv run --locked benchmark-utils performance-local
uv run --locked benchmark-utils performance-github-assets
uv run --locked benchmark-utils performance-release
uv run --locked benchmark-utils performance-doc
uv run --locked benchmark-utils write-baseline --ref vX.Y.Z --output baseline_results.txt
uv run --locked benchmark-utils compare --baseline baseline-artifact/baseline_results.txt
uv run --locked benchmark-utils compare-tags --old-tag vX.Y.Z --new-tag vA.B.C
```

Generated summaries should come from fresh perf-profile runs when they are used
as release evidence. For routine PR work, use `just ci` plus
`just perf-no-regressions`. Release-baseline comparisons write
`benches/main_vs_release_compare_results.txt`; PR/ref comparisons write
`benches/worktree_vs_<ref>_compare_results.txt`. The comparison commands print
a short pass/regression/error status and the report path before exiting. The
local PR/ref guard fails on total matched-time regressions, while individual
regressions and improvements are surfaced in the report.

Direct `bench-compare` reports describe the local raw Criterion path and live
host memory/target information. Retained artifact-backed reports instead carry
the fields serialized for both measurements: source revisions, exact commands,
Cargo profile, OS, CPU, architecture, rustc and Criterion versions, harness and
configuration digests, Criterion sample/content identity, confidence levels,
and release-archive digests when applicable.

The generated `Triangulation Data Structure Performance` section is intentionally
first: it is built from the current `target/criterion` construction results,
the `ci_performance_suite` metric sidecar, and the latest run metadata sidecar.
That makes the Criterion run date and generated simplex counts apply to the
public API tables that follow.
