# Performance Benchmarks

This directory contains the Criterion and release-mode benchmark harnesses used
to keep Delaunay construction, topology operations, validation, and geometric
predicates fast across 2D-5D.

## Benchmarking Model

Routine CI and release measurements share the curated public API plan:
`ci_performance_suite`, `circumsphere_containment`, and `locate`. Each target
is sampled once per revision. Untimed discovery and scientific preflight precede
timing. Internal diagnostics remain opt-in; `just bench all` composes every
registered target, including instrumented allocation diagnostics.

## How The Workflow Fits Together

Use `just check` and focused tests during development, following the
[validation matrix](../docs/dev/commands.md#validation-command-selection).
For a fresh working-tree comparison, run:

```bash
just performance-local
# Optional topology-policy experiment, outside the release-signal suite:
just performance-local topology
```

These commands call the installed shared schema-4 acquisition engine. Read-only
source exports preserve each revision's native harness, including baseline-only
files; no Git worktrees or automatic tag fetching are needed. The current tree
includes tracked and nonignored new files. Tags must already exist locally.
Concurrent input changes reject publication. Both phases retain independent
canonical discovery inventories, gates, count policies, exact raw JSON, source,
harness, workload, compiler, host and declared environment identities.

`tooling/measurement-release-signal.toml` owns the 173-case release selection and
unchanged 10/15/20/25/100 sample counts. The separate topology declaration retains
its 12-sample 4D group. Discovery and exhaustive count maps reject missing or
unexpected IDs. Each phase executes its own correctness gate before either
phase is timed. The native command also saves `retained`; its benchmark,
estimate, sample and Tukey bytes must equal the fresh files.

## Release Comparison Artifacts

| Path | Owner / role |
|------|--------------|
| `target/bench-reports/performance.run.json` | Fresh complete-run/v3, containing both complete phases and raw files |
| `target/bench-reports/performance.evidence.json` | Digest-bound original provenance and acquisition declarations |
| `target/bench-reports/performance.md` | Local comparison, with union rows and valid absolute values |
| `target/bench-reports/preparation.json` | Independent release/ref assertions captured before release measurement |
| `docs/performance_current.md` | Future current complete-run report, written only by explicit promotion |
| `docs/performance-runs/` | Future immutable complete-run history and latest selection |
| `docs/performance.md`, `docs/archive/performance/` | Unchanged historical reports and evidence |
| `target/bench-reports/github-assets-performance.*` | Historical Delaunay archive interpretation; separate hosted sessions suppress ratios |
| `delaunay-vX.Y.Z-criterion-baseline.tar.gz` | Exact tagged single-run archive and Delaunay construction diagnostics |

Incompatible harnesses, workloads, sampling policies, tools or captured hosts
suppress ratios while preserving complete absolute results and one-sided rows.
A later host probe cannot make earlier measurements comparable. Marginal timing
intervals are not paired ratio intervals or significance tests.

Release preparation, acquisition/promotion and README publication are separate
retry boundaries, using the installed commands and consumer TOML:

```bash
just performance-prepare
just performance-release
just performance-readme --preview
just performance-readme
# Rerender the retained current report, without native benchmark commands:
just performance-doc
```

Release promotion is strict: different release labels, complete scientific
coverage and ratio eligibility are required. Equal-label working-tree runs need
distinct source identities and remain local evidence. The README configuration
selects four construction rows; the full report retains every measured case.
The report publisher and README publisher verify their captured inputs before
transactional replacement. Failed publication retains the previous generation;
inspect any reported recovery paths before retrying after interruption.

For explicit tag pairs, use the composable acquisition and promotion commands:

```bash
just performance measure tooling/measurement-release-signal.toml vX.Y.Z vX.Y.W \
  --payload target/bench-reports/performance.run.json \
  --manifest target/bench-reports/performance.evidence.json
just performance-doc --payload target/bench-reports/performance.run.json \
  --manifest target/bench-reports/performance.evidence.json
```

The default paired declaration uses the v0.8.3+ native case contract, including
the current location query IDs. To acquire an original v0.8.2 baseline, select
`tooling/measurement-release-signal-v082.toml` in the explicit `performance measure`
command. It preserves that baseline's 20 different locate IDs and all original
counts, retaining union coverage and suppressing incompatible ratios. Its
absolute evidence cannot pass strict release/README promotion. The regular
prepared release workflow assumes the package version has been advanced beyond
the latest published release, as described in the release guide.

Single-checkout `bench-latest`, saved-baseline, summary and tagged-archive paths
retain the Delaunay adapter: the published complete-run acquisition command
requires two phases. These paths preserve construction counters, the native
sampling policy, private staging and exact saved-copy verification without
sampling either revision twice. Historical-asset interpretation retains its
nested Delaunay schema and hosted-session policy; it cannot promote old evidence
as a complete run. See the [adoption boundary](../docs/dev/tooling_adoption.md#declarative-independent-performance-adoption).

`just bench-preflight` exercises the curated fixtures once without statistical
sampling. `just bench-latest` collects one fresh raw run; `just bench-perf-summary`
only renders it. Save named baselines using `just bench-save-baseline last`,
then use `just bench-latest-vs-last` in the current checkout. The explicit release
benchmark workflow preserves its captured release ID/tag commit, refuses an
existing baseline asset and leaves the release as a draft for maintainer review.

## Math-Kernel Coverage Audit (#513)

The release plan in `tooling/python/benchmark_utils.py` selects three public API
targets: `ci_performance_suite`, `circumsphere_containment`, and `locate`.
`cold_path_predicates`, `realization_validation`, and `math_kernels` isolate
implementation costs for manual debugging. They are excluded from `bench-latest`
and automated performance workflows. The coverage below distinguishes the
public workflow signal from diagnostic microbenchmarks.

| Kernel family | Existing coverage | Targeted diagnostic |
|---------------|----------------------------|-------------------|
| Orientation / exact fallback | Public predicate workflows; manual `cold_path_predicates` | `math/orientation` |
| Barycentric intersection | Public validation workflows; manual `realization_validation` | Retain existing cases |
| Circumsphere / insphere | `circumsphere_containment`; manual `cold_path_predicates` | `math/geometry` radii |
| Quality / volume | Indirect public-workflow coverage | `math/geometry` quality, volume, facet measure, and surface measure |
| Ridge / Euler / manifold | `ci_performance_suite` validation and proof boundaries | `math/topology` |
| Spherical distance | None | `math/spherical_distance` adapter and upstream angle |
| Locate / hull / visible facets | `locate`, `ci_performance_suite` | Retain existing cases |
| Barycenter | None; allocation coverage is manual | `math/geometry/*/barycenter` |
| Allocation hot paths | None; instrumentation is manual | Retain separate allocation contracts |

The existing orientation coverage is indirect through insphere and explicit
through `predicates/sos_degenerate` (2D-6D). Direct ordinary orientation now has
separate well-conditioned, nonzero cancellation, and zero-determinant cases.
Circumsphere coverage already compares three predicate families,
boundary/far/near-boundary queries, 3D circumcenter solve paths, certified
cospherical exact cases, and SoS; the focused path diagnostics remain manual.

`math/spherical_distance` compares `SphericalMetric::try_distance` with the
released la-stack slice angle on identical canonical coordinates in intrinsic
dimensions 2-5 (ambient lengths 3-6). Dense integer Gram and planar angle
oracles check correctness outside timing. Fixtures include nearly coincident
and antipodal directions, a subnormal angle, and extreme radii. With
`count-allocations`, setup also asserts zero allocations for both paths.
The upstream exterior-product calculation takes quadratic work in the ambient
length and constant auxiliary storage. These small-dimension measurements do
not establish scaling performance for large ambient dimensions.
To reproduce the #610 implementation comparison, prepare an isolated copy of
the current checkout, retaining its `benches/math_kernels.rs`, `Cargo.toml`,
and `Cargo.lock` for both runs. The corrected local revision predates this
benchmark group, so checking out that entire revision cannot produce the
required case IDs. Read its implementation with:

```bash
git --no-pager show \
  06284cfe4a699cc81bc1aa334524e5dcac438707:src/topology/spaces/spherical.rs
```

In the isolated copy, replace only `SphericalMetric::try_distance` with that
revision's complete method and remove the now-unused `VectorAngle` import.
Keep the current benchmark harness, canonical fixtures, dependency lockfile,
toolchain, and features unchanged. Run the command below from that copy to
save the corrected-local baseline; running it on the unchanged current adapter
would mislabel the baseline:

```bash
uv run --locked --group dev research-repo-tools toolchain run -- \
  cargo bench --locked --profile perf --features count-allocations \
  --bench math_kernels -- math/spherical_distance --sample-size 30 \
  --warm-up-time 0.5 --measurement-time 1 --noplot \
  --save-baseline spherical-local-610
```

Restore the current method and import in the same isolated copy, then repeat
the command with `--baseline spherical-local-610` instead of
`--save-baseline spherical-local-610`. Preserve the Criterion output directory
between runs. The baseline is the corrected local norm-weighted calculation;
the inaccurate `acos(dot)` path is excluded. The downstream benchmark measures
radius validation and arc-length conversion separately from the upstream API.

Criterion runs cases sequentially, so record execution order with the results.
For small timing differences, repeat the comparison with alternating
before/after run order to assess clock, temperature, and cache drift. A single
run provides descriptive local measurements, not a paired-sampling guarantee.
For direct upstream-versus-adapter comparisons, filter their case IDs into
separate invocations and reverse that execution order on repeated runs.

`realization_narrow_phase/shared_face_boundary` and
`near_degenerate_shared_face` already isolate exact barycentric intersection
decisions in 2D-5D. Whole Level 4 validation covers their integration. Defer
isolated LP primitives and exhaustive invalid intersection families to focused
correctness tests; the public narrow phase is the useful timed unit.

Topology previously had aggregate release coverage, with manual component work
in `validation_components/validation_components_*`, `topology_guarantee_construction`,
`pl_manifold_repair`, and `tds_clone/topology_view_construction`. The new target
isolates Euler enumeration, unique ridges, boundary and link checks, and Level 3
validation. Large random-mesh scaling remains in profiling.

Locate already compares `no_hint` and `exact_hint`. Hull release coverage
includes `boundary_facets`, `convex_hull`, and `convex_hull_queries` for outside,
visible, and nearest-visible queries. Non-Euclidean and adversarial walk
distributions remain separate investigations. Periodic-chart barycenters are
also deferred until a representative stable query workload is selected.

`allocation_hot_paths` checks bootstrap, insertion, iterators, incidence,
simplex vertices, barycenters, UUID iteration, facet keys, hinted locate,
inradius, and surface measure in 2D-5D; `memory_footprint` supplies allocation/RSS
diagnostics. Measure allocation checks borrow prebuilt geometry and exclude
fixture and facet-view construction. Keep
allocation assertions and process RSS out of release timings. Allocation counts
are useful local contracts, but instrumentation overhead and process-level noise
do not establish a release latency signal. The added barycenter cases measure
latency without that instrumentation.

`math/geometry` directly times volume, facet measure, surface measure, radius,
and quality calculations. Other small helpers, such as norms and radius queries
with a precomputed center, remain indirectly covered through geometric workflows.
Defer additional microcases until profiling identifies a distinct cost worth
tracking.

All reviewed repair, deletion, Pachner, clone, checkpoint, topology-policy,
allocation, and large-scale profiling targets remain manual diagnostics. Their
absence from the public API release plan is intentional, not missing release
execution. Promote a kernel only after representative same-machine runs establish
a useful, repeatable signal and its added release runtime is justified. A future
promotion must update the shared measurement plan and its coverage tests together.

### Targeted Math Kernels

```bash
# Execute every fixture and operation once; no timing evidence is produced.
cargo bench --profile perf --bench math_kernels -- --test

# Measure all kernel families, or select one with a Criterion filter.
cargo bench --profile perf --bench math_kernels -- --noplot
cargo bench --profile perf --bench math_kernels -- 'math/orientation' --noplot
cargo bench --profile perf --bench math_kernels -- 'math/geometry' --noplot
cargo bench --profile perf --bench math_kernels -- 'math/topology' --noplot

# Save and compare a local baseline around a kernel implementation change.
cargo bench --profile perf --bench math_kernels -- --save-baseline before --noplot
cargo bench --profile perf --bench math_kernels -- --baseline before --noplot
```

`math_kernels` defaults to 30 samples, one second of warm-up, and two seconds of
measurement per case; Criterion CLI overrides remain available. Its 170 cases
are investigations, not new performance claims. Use the same filter, profile,
hardware, and sampling settings before and after an implementation change.
`just bench-compile` includes this target in the ordinary benchmark compile
gate; `just bench all` runs it with the broader suite. No new release command or
saved-baseline suite alias is needed.

The orientation setup checks the actual `la-stack` determinant/error-bound
decision on the same homogeneous matrix used by `simplex_orientation`.
Well-conditioned 2D-3D cases must use the filter; 4D-5D are exact-path references
because their homogeneous matrices exceed the direct filter's 4×4 limit. Both cancellation
and zero cases must require exact arithmetic. This measures ordinary exact
orientation separately from SoS, whose nonzero symbolic ordering has its own
existing benchmark.

Geometry fixtures cover 2D-6D with axis lengths `(1, ..., 1, h)` and `h = 1` or
`h = 1/1024`. Volume, opposite-origin facet measure, total boundary measure,
circumradius, inradius,
both quality metrics, and barycenter coordinates are checked against analytical values. The thin fixtures stay
inside the helpers' supported nondegenerate domain; they do not time rejected
inputs. Each owning triangulation passes cumulative Level 5 validation before
sampling. The barycenter case measures one public query, including its ordinary
handle checks and coordinate work, without allocation instrumentation.

Topology fixtures cone the boundary of a cross-polytope to its center. They have
`2D + 1` vertices, `2^D` maximal simplices (4, 8, 16, and 32), Euler
characteristic one, and both interior and boundary links. Setup verifies the
entire f-vector and boundary f-vector analytically, not just the Euler sum,
and checks cumulative Level 5 validity. Component scans use an untimed clone
demoted through the public `into_tds` boundary; the original owner retains its
construction proof for `is_valid_topology`. Closed-boundary and vertex-link
cases reuse a facet index built outside timing, as their names state. Euler
timing includes face enumeration, where the substantive work occurs, rather
than only summing an already computed f-vector. Ridge timing consumes the full
deduplicating iterator. Geometry throughput counts queries; topology throughput
counts maximal simplices processed per scan.

Standalone vertex-link timing is limited to 2D-3D. Public TDS demotion discards
construction provenance, and the standalone checker cannot certify links of
dimension three or higher from local incidence alone. The 4D-5D fixtures retain
coverage through the original owner's `is_valid_topology`; do not bypass that
proof boundary or benchmark its rejection as successful validation.

These symmetric fixtures isolate kernels and supply independent answers; they
do not model large random meshes, all supported topology spaces, or every
conditioning regime. Use the focused profiling targets for scale and mixed workloads,
`realization_validation` for barycentric narrow-phase work, and the existing
predicate suites for in-sphere and SoS investigations. All measured fallible
operations abort on error so a failed path cannot silently publish a fast result.

## Benchmark Suite Overview

| Benchmark | Purpose | Scale | Typical Runtime | Used By |
|-----------|---------|-------|-----------------|---------|
| `allocation_hot_paths.rs` | Bootstrap/insert/query/barycenter allocations | Calibrated 2D-5D canaries | ~1-2 min | Manual allocation checks |
| `checkpoint_serialization.rs` | Manifest and JSON checkpoint write/load | 64-vertex 2D owner | <1 min | Checkpoint tuning |
| `ci_performance_suite.rs` | Public workflow regression contract | Calibrated 2D-5D canaries | ~5-10 min | CI, baselines, `just performance-local` |
| `circumsphere_containment.rs` | Circumsphere predicates and solves | 2D-5D predicates, 3D LU/exact solves | ~5 min | Predicate/circumcenter tuning |
| `cold_path_predicates.rs` | Track predicate paths | Hot, centered, and certified exact cases in 2D-5D | ~2-5 min | Predicate tuning |
| `delaunay_repair.rs` | Flip-based Delaunay repair plus transaction-pressure cases | 2D-5D repair-convergent fixtures | ~2-5 min | Repair tuning |
| `pachner_stress.rs` | Unified Pachner move API stress | Accepted 4D microcases plus forward/inverse round trips | Manual | Pachner move workflow tuning |
| `pl_manifold_repair.rs` | Over-shared facet and targeted topology repair | 2D/3D synthetic repair fixtures | <1 min | PL-manifold repair tuning |
| `construction_scaling.rs` | Construction scaling | Original profiling scales | Hours; filter explicitly | Manual only |
| `memory_footprint.rs` | RSS and allocations | Original profiling scales | Hours; filter explicitly | Manual only |
| `traversal.rs` | Neighbor and element traversal | Original profiling scales | Filter explicitly | Manual only |
| `validation_components.rs` | Validation layer costs | Ordinary/adversarial 2D–5D | Filter explicitly | Manual only |
| `circumsphere_queries.rs` | Batched circumsphere queries | 3D 1k/10k/30k | Filter explicitly | Manual only |
| `hull_queries.rs` | Boundary and hull extraction | 3D 10k | Filter explicitly | Manual only |
| `realization_validation.rs` | Level 4 narrow phase and whole validation | Realistic/near-degenerate 2D-5D | ~1-5 min | Level 4 tuning |
| `delete_vertex.rs` | Vertex deletion and rollback cost | 2D/3D seeded cases; 4D/5D inverse-k1; 2D-5D rollback | ~1-5 min | Vertex deletion |
| `locate.rs` | Point-location facet-walk latency (no-hint vs exact-hint) | 2D-5D fixed cases | ~1-3 min | Locate/walk tuning |
| `math_kernels.rs` | Geometry and topology kernels | 170 cases: geometry 2D-6D, others 2D-5D | Several minutes | Manual kernel investigations |
| `tds_clone.rs` | Owner clone and topology-view cost | Deterministic 2D-5D triangulations | ~1-3 min | Rollback and incidence baselines |
| `topology_guarantee_construction.rs` | Cost of PL-manifold validation audit cadences | 2D-5D construction cases | ~45-60 min | Manual topology policy work |

### Checkpoint Serialization Workload Identity

The current checkpoint workload identity is
`checkpoint-serialization/u32-payloads-v1`. Its 8×8 fixture stores the row-major
index `row * 8 + column` as each vertex's `u32` payload and stores the simplex
vertex count (three for this 2D triangulation) as each simplex's `u32` payload.
This is an intentional incompatible replacement for the earlier unit-payload
workload: unit-payload timings and saved Criterion baselines must not be compared
with this identity. Delete or rename an old saved checkpoint baseline and create
a fresh `u32-payloads-v1` baseline before drawing performance conclusions.

### Vertex Deletion Scope

The 2D/3D success cases pin their seed, fixture family, and input vertex index;
setup fails if that exact deletion no longer succeeds. The 4D/5D success cases
measure inverse-k1 removal of the interior point `[0.1; D]` from a subdivided
canonical simplex: six vertices and five simplices in 4D, seven vertices and six
simplices in 5D. They use the default public Delaunay builder and deletion policy,
with preflight validation through Levels 1-5 and an exact vertex-star check.
The IDs include `inverse_k1` and actual counts; these cases replace the earlier
general 4D/5D success workloads and require fresh timing baselines.

General fan deletion revokes the construction provenance required to certify
larger high-dimensional PL vertex links, so arbitrary 4D/5D mesh deletion is
outside this success benchmark's scope. The separate minimal-simplex rollback
cases remain in 2D-5D. All fixture construction and preflight checks stay outside
timing; the measured public deletion retains its own invariant checks.

## Selection Guide

| Use Case | Command |
|----------|---------|
| Final local invariant validation gate | `just ci` |
| Measure checkpoint manifest and JSON write/load paths | `cargo bench --bench checkpoint_serialization --features bench -- --noplot` |
| Quick local large-scale wall-clock guard | `just perf-large-scale-smoke` |
| Full CI benchmark suite only | `just bench-ci` |
| Run curated release-signal Criterion measurements | `just bench-latest` |
| Compare latest measurements against saved `last` Criterion baseline | `just bench-latest-vs-last` |
| Render a Markdown report from existing Criterion results | `just bench-compare [baseline] [suite] [scope]` |
| Save a named Criterion baseline | `just bench-save-baseline <tag> [suite]` |
| Save the previous-release Criterion baseline as `last` | `just bench-save-baseline last` |
| Compare current tree against latest published release locally | `just performance-local` |
| Compare stored GitHub Release benchmark assets | `just performance-github-assets [current-tag baseline-tag]` |
| Measure, retain, validate, and promote release docs | `just performance-prepare`, then `just performance-release` |
| Publish the retained comparison snapshot to the README | `just performance-readme` |
| Promote docs from a retained comparison JSON/evidence pair | `just performance-doc` |
| Allocation-contract microbenchmarks | `just bench-allocations` |
| Release performance summary | `just bench-perf-summary` |
| Durable latest-version benchmark baseline | GitHub Release asset `delaunay-vX.Y.Z-criterion-baseline.tar.gz` |
| Smoke-test benchmark harnesses | `just bench-smoke` |
| Allocation hot-path contracts | `cargo bench --profile perf --bench allocation_hot_paths --features count-allocations -- --noplot` |
| Predicate comparison | `cargo bench --profile perf --bench circumsphere_containment -- --noplot` |
| Predicate cold-path work | `cargo bench --profile perf --bench cold_path_predicates -- --noplot` |
| Orientation, quality, volume, radius, barycenter, or topology kernels | `cargo bench --profile perf --bench math_kernels -- --noplot` |
| New math-kernel fixture/operation smoke check | `cargo bench --profile perf --bench math_kernels -- --test` |
| Flip-based Delaunay repair | `cargo bench --profile perf --bench delaunay_repair -- --noplot` |
| Flip-repair transaction pressure | `cargo bench --profile perf --bench delaunay_repair -- repair_transaction_pressure --noplot` |
| Unified Pachner move stress | `just pachner-stress` |
| PL-manifold repair path | `cargo bench --profile perf --features bench --bench pl_manifold_repair -- --noplot` |
| Large-scale scaling suite | `cargo bench --profile perf --bench construction_scaling -- --noplot` |
| Level 4 realization validation | `cargo bench --profile perf --bench realization_validation -- --noplot` |
| Vertex deletion mutation baseline | `cargo bench --profile perf --bench delete_vertex -- --noplot` |
| Point-location facet walk | `cargo bench --profile perf --bench locate -- --noplot` |
| One-dimension acceptance/profiling run | `just debug-large-scale-{2,3,4,5}d [n] [repair_every]` |
| Isolated 32k-vertex 2D Level 4 acceptance run | `DELAUNAY_LARGE_DEBUG_VALIDATION=realization just debug-large-scale-2d 32000` |
| Deep profiling | `cargo bench --profile perf --bench memory_footprint --features count-allocations` |

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
just performance-local
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
the release-mode large-scale harness. It fixes the 2D/3D/4D/5D vertex counts at
32,000/9,000/1,000/160 and checks structure/topology (Levels 1–3) plus the Level 5
Delaunay property. These differ from the debug helpers' local count and
validation defaults; see the [command guide](../docs/dev/commands.md#benchmark-profiles).
The guard caps each test runtime at 60 seconds by default and reports all failing dimensions before
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

Use `just performance-local` for fresh same-machine comparisons. Legacy text
baselines, warm caches, GitHub Actions baseline-artifact polling and numeric
warning thresholds are retired. A saved Criterion baseline remains available
for targeted diagnostics through `bench-save-baseline` and `bench-compare`.

Use `just bench-smoke` to compile and execute each public API fixture once,
without statistical sampling. It produces no performance data. Use
`just bench-compile` to compile every diagnostic harness without executing it.

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
`ci_performance_suite`, `circumsphere_containment`, and `locate`. This set covers
public construction, mutation, import, validation, hull, predicate, and
point-location workflows. Internal path/component timing, broad profiling,
allocation instrumentation, and topology-policy experiments remain manual.
Use `just bench-compare <baseline>` to render
`target/bench-reports/performance.md` from existing Criterion `new` output and a
saved baseline such as `last` or `v0.7.8`. If the baseline and current
checkouts are easy to confuse, prefer `just performance-local`. Use
`uv run --locked benchmark-utils bench-compare --scope all-benches` only when you
explicitly want an exploratory report over every Criterion result already
present under `target/criterion/`.

The paired acquisition and publication commands are documented in
[How The Workflow Fits Together](#how-the-workflow-fits-together).

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
- explicit connectivity import and raw-TDS promotion under `Pseudomanifold`
- Level 5 certification of a `PLManifold` owner retaining construction provenance

The `explicit_import/import_pseudomanifold_*` and
`proof_boundaries/promote_pseudomanifold_*` cases intentionally measure the
incidence contract in every dimension. Arbitrary connectivity cannot recover
high-dimensional PL-manifold construction provenance. The independent
`proof_boundaries/certify_pl_manifold_*` cases retain their proof-bearing owner
from point construction. These names distinguish the corrected contracts from
historical timings; do not relabel old evidence as the new cases.

The current calibrated fixture sizes are:

| Dimension | Fixture vertices | Insert batch |
|-----------|------------------|--------------|
| 2D | 500 | 10 |
| 3D | 100 | 10 |
| 4D | 30 | 6 |
| 5D | 12 | 4 |

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
  "tds_new_3d/tds_new/100"
```

Use `just bench-latest` to measure the public API plan and capture construction
metrics, then `just bench-perf-summary` to render a release summary. Re-rendering
the summary does not repeat measurements.

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
to 100 workload steps with topology validation every 10 steps. `--attempts`
retains its flag spelling but counts workload steps: one forward/inverse pair in
`round-trip` mode or one candidate/proposal cycle in `random-walk` mode. The
direct `pachner-stress` binary has larger soak-test defaults: 10,000 vertices in
3D or 1,000 in 4D, 100,000 steps, and validation every 1,000 steps.

The default `round-trip` mode commits a forward move and its inverse when a
candidate is locally valid. The `random-walk` mode commits accepted moves over
an evolving triangulation. Both modes write progress CSV and summary JSON under
`target/pachner_stress/` and also emit parseable stdout telemetry: one
`pachner_stress_source` line for the prepared triangulation,
`pachner_stress_progress` lines after each validation cadence, and a final
`pachner_stress_metric` line with proposal attempts, accepted mutations,
proposal rejections, validation time, and final topology size. Summary and
progress schema version 2 names the units explicitly: `configured_steps` and
`completed_steps` describe workload progress, while `proposal_attempts` and
`accepted_mutations` describe actual Pachner work. Interim acceptance rates use
completed proposal attempts as their denominator.

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

Historical local perf-profile run on maintainer Apple M4 Max hardware
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

Interpretation of that historical run:

- `insphere_lifted` was fastest for 2D/3D fixed non-boundary cases and
  for the 3D random batch.
- `insphere_distance` was fastest for 4D/5D fixed non-boundary cases.
- `insphere` had fast boundary-vertex short-circuit behavior, while its
  robust determinant path was slower on non-boundary 3D-5D cases.

Current `insphere` and `insphere_lifted` share the filtered-exact core. These
May 2026 timings do not rank the current implementations; rerun representative
workloads with the current toolchain before choosing a helper for performance.

## Cold Path Predicates

```bash
cargo bench --profile perf --bench cold_path_predicates -- --noplot
```

`cold_path_predicates.rs` tracks the fast filter and exact fallback paths used
by `insphere`, `insphere_lifted`, and their orientation predicates. Orientation
uses a determinant error bound through D = 3 and an exact integer determinant
otherwise. In-sphere uses query-relative interval filtering through D = 6 and
an exact rational fallback. The hot group uses well-separated random queries. The
historically named `near_boundary` group samples around the circumsphere center
and remains only for saved-baseline compatibility; it does not establish exact
fallback. The `exact_fallback` group validates known cospherical 2D-4D fixtures
before timing them, making it the benchmark's correctness-certified Stage-2
signal. The 5D hot group requires exact orientation, but its in-sphere interval
filter can still succeed.

## Profiling Suite

The former monolith is retired after the parity inventory below. Each retained
family has an independently runnable target. Filtering and `--list` initialize
only selected callbacks; fixture construction and independent validation happen
before the measured closure. Read-only operations share immutable fixtures.

| Former group | Classification | Focused destination / retained cases |
|--------------|----------------|-------------------------------------|
| `construction/{D}D/{n}v/construct` | Partial overlap with CI construction; larger inputs | `construction_scaling`, all 10 default dimension/count pairs |
| `memory/{D}D/{n}v/construction_memory_delta` | Unique process RSS | `memory_footprint`, all 10 pairs |
| `validation/{D}D/{n}v/validate_topology` | Partial overlap with validation canaries; larger inputs | `validation_components`, all 10 pairs |
| `queries/neighbors/{D}D/{n}v/*` | Unique full-owner neighbor sweep | `traversal`, all 10 pairs |
| `queries/vertices/{D}D/{n}v/*` | Unique vertex iteration | `traversal`, all 10 pairs |
| `queries/simplices/{D}D/{n}v/*` | Unique simplex-key iteration | `traversal`, all 10 pairs |
| `memory_profiling/*` | Unique batch allocation footprint | `memory_footprint`, all 7 production cases with `count-allocations` |
| `query_latency/*` | Partial overlap; actual triangulation simplices | `circumsphere_queries`, 1k/10k/30k fixtures, 1,000 queries |
| `validation_components_*/*` | Unique layer/conditioning experiment | `validation_components`, all 40 ordinary/adversarial cases |
| `algorithmic_bottlenecks/boundary_facets_3d/{n}` | Partial overlap with CI public API; larger input | `hull_queries`, 10k (3k in development mode) |
| `algorithmic_bottlenecks/convex_hull_3d/{n}` | Partial overlap; larger input | `hull_queries`, shares the boundary fixture |

No identical measurement is retained in two targets. None of these diagnostics
is added to `bench-latest`, release assets or routine performance CI. The 3D
circumsphere/hull experiments keep their original dimension-specific scope;
construction, RSS, traversal and validation cover 2D–5D. Existing
`realization_validation`, `allocation_hot_paths`, `locate` and
`ci_performance_suite` keep their distinct smaller inputs and API contracts.

```bash
cargo bench --profile perf --bench construction_scaling -- "construction/3D"
cargo bench --profile perf --bench traversal -- "queries/neighbors/2D/1000v"
cargo bench --profile perf --bench validation_components -- "validation_components_2d"
cargo bench --profile perf --bench circumsphere_queries -- "query_latency"
cargo bench --profile perf --bench hull_queries -- "boundary_facets"
cargo bench --profile perf --bench memory_footprint --features count-allocations -- "memory_profiling/memory_usage_2d/1000"
```

Default scale counts remain 1k/5k/10k in 2D and 3D, 1k/3k in 4D, and
500/1k in 5D. `BENCH_LARGE_SCALE=1` replaces the 4D 1k/3k counts with
1k/5k/10k.
`PROFILING_DEV_MODE=1` reduces auxiliary memory, circumsphere and hull inputs.
The 5D 1,000-point allocation case can take hours even in development mode.
`BENCH_SAMPLE_SIZE`, `BENCH_WARMUP_SECS` and `BENCH_MEASUREMENT_TIME` configure
manual timing. `BENCH_PERCENTILE` selects the allocation summary percentile.

RSS uses the current process, `sysinfo` and KiB units. Process startup, allocator
retention and host activity affect its deltas; they are not allocation counts.
The optional allocator reports operation counts and bytes separately from
instrumented timing. Keep these observations distinct from API latency and
from general memory-scaling guarantees.

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
full-topology clone cost with the current touched-record rollback journals and
any future localized rollback implementation.

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
excluded from the pull-request and curated release-signal sets and remains
available for manual topology-policy work.

## Generated Summaries

`benches/PERFORMANCE_RESULTS.md` is generated by `benchmark-utils` for absolute
release summaries. Durable release archives contain raw Criterion data and
versioned metadata. The CI comparison measures both revisions in one job;
imported release archives provide absolute timings and suppress ratios because
they come from separate sessions. `docs/performance_current.md` owns the current
curated release comparison, with immutable history under `docs/performance-runs/`,
when retained evidence satisfies publication policy. `docs/performance.md`
remains unchanged historical evidence.

```bash
uv run --locked benchmark-utils generate-summary
uv run --locked benchmark-utils generate-summary --run-benchmarks --profile perf
uv run --locked benchmark-utils bench-compare last
just performance-local
uv run --locked benchmark-utils performance-github-assets
just performance-release
just performance-doc
```

Generated summaries must come from fresh perf-profile runs when used as release
evidence. Use `just performance-local` for a retained comparison. The report
keeps missing and non-comparable rows visible and applies no regression threshold.

Direct `bench-compare` reports describe the local raw Criterion path and live
host memory/target information. Retained artifact-backed reports instead carry
the fields serialized for both measurements: source revisions, exact commands,
Cargo profile, OS, CPU, architecture, rustc and Criterion versions, harness and
configuration digests, Criterion sample/content identity, confidence levels,
and release-archive digests when applicable.

The public-API summary uses the current `ci_performance_suite` metrics and run
metadata to display generated simplex counts beside construction timings.
