# Numerical Robustness Guide

Delaunay triangulation can be sensitive to floating-point roundoff when point
sets are near-degenerate (nearly coplanar/cospherical), contain extreme
coordinate magnitudes, or contain duplicates/near-duplicates.

This document summarizes the robustness tools available in this crate and how
to apply them. It intentionally complements
[`limitations.md`](limitations.md): that page describes the operational scope
and tested scale envelope, while this guide explains the numerical mechanisms
and input hygiene choices behind those limits.

For a runnable companion, see
[`examples/numerical_robustness.rs`](../examples/numerical_robustness.rs):

```bash
cargo run --release --example numerical_robustness
```

## Coordinate input model

The currently supported caller-visible coordinate scalar is `f64`. This is a
deliberate correctness boundary: the crate's linear algebra backend and
geometric predicates are designed around f64 fast filters with exact-arithmetic
fallbacks, while topology and manifold validation stay combinatorial where
possible.

Exact arithmetic is already part of the predicate pipeline, but exact
coordinates are not currently a public input type. If exact-coordinate input is
supported in the future, it should be introduced as an explicit documented
coordinate model/API rather than as arbitrary scalar genericity.

## Robustness toolbox

### Exact predicates (v0.7.1+)

Orientation and insphere predicates use staged evaluation:

1. **Provable f64 fast filter (D ≤ 4)** — `la_stack::Matrix::det_errbound()` computes a
   rigorous Shewchuk-style error bound from the f64 determinant. If the bound certifies
   the sign, no allocation is needed. For D ≥ 5 (where `det_errbound` is not available),
   the predicate falls through directly to exact arithmetic.
2. **Exact sign** — ordinary finite matrices use
   `la_stack::Matrix::det_sign_exact`. Relative in-sphere construction instead
   converts the original binary64 coordinates to `la-stack`'s re-exported
   `BigRational` before subtracting or squaring, then eliminates that exact
   derived matrix. This distinction matters: sending already-rounded `f64`
   differences or squared norms to an exact determinant would only compute the
   exact sign of the wrong matrix.
3. **Indeterminate or symbolic fallback** — if exact arithmetic cannot run
   (for example due to non-finite entries or unsupported insphere matrix size),
   robust predicates return `BOUNDARY` / `DEGENERATE` where appropriate, while
   `AdaptiveKernel` applies deterministic Simulation of Simplicity (SoS)
   tie-breaking for degenerate finite inputs.

This applies to `simplex_orientation`, `insphere`, `insphere_lifted`, `robust_orientation`,
and `robust_insphere`.

**Dimension limits:** the stack-allocated matrix dispatch supports up to 7×7 matrices
(`MAX_STACK_MATRIX_DIM = 7`). This means:

- f64 fast filter: D ≤ 4 (`det_errbound()` is unavailable above 4D)
- Exact orientation: D ≤ 6 (matrix is (D+1)×(D+1))
- Public exact-insphere / `ExactPredicates` support: D ≤ 6 (the relative lifted
  matrix is `(D+1)×(D+1)`)

For D ≥ 7, `robust_insphere` falls back to a distance-based classification and
preserves unresolved `BOUNDARY` results. Complete symbolic expansion shares the
D ≤ 6 exact support boundary. Treat D ≥ 7 triangulation as experimental;
explicit repair APIs are not available through the `ExactPredicates` gate there.

### Robust predicates (`geometry::robust_predicates`)

The crate includes robust orientation and insphere predicates (e.g.
`robust_orientation`, `robust_insphere`) for near-degenerate configurations.
These predicates layer additional strategies on top of exact predicates: opt-in
consistency checking against `insphere_distance`, symbolic perturbation for
tie-breaking, and deterministic geometric fallbacks when the exact insphere
matrix is outside the supported range.

Most users won't call these functions directly; instead, select a kernel.

### Kernel selection (`geometry::kernel`)

Kernels control which predicate implementations are used by the triangulation algorithms:

- `AdaptiveKernel<T>` **(default)**: provably correct predicates with zero configuration
  in the supported exact dimensions. Uses exact arithmetic (fast filter + Bareiss)
  for orientation and insphere, and adds Simulation of Simplicity (`SoS`) so
  degenerate ties are broken deterministically. Orientation and insphere queries
  return ±1 rather than 0 for finite ordered inputs, including repeated
  coordinate values. Best choice for Delaunay triangulation. Implements
  `ExactPredicates` through D ≤ 6.
- `RobustKernel<T>`: exact-arithmetic predicates that preserve explicit
  `BOUNDARY`/`DEGENERATE` signals and can run opt-in diagnostic consistency checks.
  Prefer this when your application needs to detect cospherical/coplanar/collinear
  configurations directly (SoS would mask these). Implements `ExactPredicates`
  through D ≤ 6.
- `FastKernel<T>`: lean filtered-exact predicates that preserve explicit
  `BOUNDARY`/`DEGENERATE` signals without `RobustKernel`'s opt-in diagnostic
  consistency check or higher-dimensional fallback. Implements
  `ExactPredicates` through D ≤ 6.

### `ExactPredicates` marker trait (v0.7.3+)

The `ExactPredicates` marker trait identifies kernels whose `orientation` and
`in_sphere` predicates return the mathematically correct sign in the supported
dimension, including near-degenerate configurations. `AdaptiveKernel`,
`RobustKernel`, and `FastKernel` implement this trait through D ≤ 6.

The public `DelaunayRefinementBuilder` flip-repair workflow and point-driven
Delaunay builders require `K: ExactPredicates`. This is enforced at compile
time, preventing kernels without the supported exact-predicate contract from
entering construction or flip repair. Incremental insertion through the builder
retains that bound while it crosses the Levels 1–5 publication boundary.
Insertion into an already published owner can use its stored `Kernel`; exact
flip repair remains a separate, explicitly bounded operation.

Dimension-bound exactness is intentional: orientation and relative-coordinate
insphere have exact determinant support through D ≤ 6, which is also the
current `ExactPredicates` boundary.

`DelaunayTriangulationBuilder::new(&vertices).build()` and
`DelaunayIncrementalBuilder::new()` use `AdaptiveKernel`. To opt into a
different kernel for incremental construction, use
`DelaunayIncrementalBuilder::with_kernel(...)`.

```rust
use delaunay::prelude::geometry::RobustKernel;
use delaunay::prelude::construction::{DelaunayTriangulation, DelaunayTriangulationBuilder, vertex};

let kernel = RobustKernel::<f64>::new();

let vertices = vec![
    vertex![0.0, 0.0, 0.0]?,
    vertex![1.0, 0.0, 0.0]?,
    vertex![0.0, 1.0, 0.0]?,
    vertex![0.0, 0.0, 1.0]?,
];

let dt: DelaunayTriangulation<RobustKernel<f64>, (), (), 3> =
    DelaunayTriangulationBuilder::new(&vertices)
        .build_with_kernel(&kernel)?;

assert!(dt.is_valid_delaunay().is_ok());
```

### Identity-based SoS perturbation via canonical vertex ordering

The `SoS` (Simulation of Simplicity) implementation assigns a distinct symbolic
coordinate perturbation ordered by slice position and coordinate index, then
computes the complete sparse determinant polynomial exactly. The first point in
the array gets the lowest-order perturbation terms. If different call sites present the
same vertex set in different orders, SoS tie-breaking can produce inconsistent
signs for the same geometric query — leading to flip cycles, invalid conflict
regions, or non-deterministic triangulations.

To eliminate this, **all kernel call sites canonically sort simplex vertices by
`VertexKey` identity** (`vk.data().as_ffi()`) before passing them to orientation
or insphere predicates. This makes the existing slice-position SoS identity-based
by construction: a vertex's perturbation priority depends only on its stable key,
not on how the simplex happened to store its vertices.

**Convention for contributors:**

- **Insphere calls:** sort all D+1 simplex vertices by `VertexKey` before calling
  `kernel.in_sphere()`. The test point is separate and not sorted.
- **Orientation for facet comparison:** sort the D facet vertices by `VertexKey`;
  the extra vertex (opposite or query) is always appended last.
- **Orientation for degeneracy check:** sort all D+1 vertices by `VertexKey`.

Helper functions in `src/core/util/canonical_points.rs` implement these patterns:

- `sorted_simplex_points(tds, simplex)` — collects simplex vertices in canonical order
- `sorted_facet_points_with_extra(tds, facet_keys, extra)` — collects facet
  vertices in canonical order, then appends `extra` at position D

Both return `Option`, with `None` indicating an unresolvable vertex key.

When adding new kernel call sites, **always** use canonical ordering. Failure to
do so will re-introduce order-dependent SoS behavior.

### Transactional insertion, retries, and skips

Incremental insertion is transactional: if an insertion attempt fails, the triangulation is
rolled back to the pre-insertion state.

Some geometric degeneracies are retryable via a small deterministic perturbation with
**progressive magnitude**: each retry multiplies the perturbation by ×10, spanning
several orders of magnitude across the retry budget. The base magnitude is
scale-invariant — it is proportional to the local feature size (nearest-vertex distance)
and uses ≈√machine_epsilon as the base factor (`1e-8` for `f64`).
With the default 3 retries, the ladder is:

- attempt 1: `1e-7 × local_scale`
- attempt 2: `1e-6 × local_scale`
- attempt 3: `1e-5 × local_scale`

If all retries are exhausted, strict insertion APIs return an
`InsertionError` and the triangulation is unchanged. The explicitly named
best-effort API reports the same event as `InsertionOutcome::Skipped { .. }`
with telemetry.

**Note:** With the default `AdaptiveKernel`, SoS resolves most orientation degeneracies
symbolically, so perturbation retries are rarely needed. The primary remaining retryable
cases involve cavity/topology failures rather than predicate degeneracies.

Use `insert_best_effort_with_statistics()` to observe this behavior:

```rust
use delaunay::prelude::construction::{DelaunayIncrementalBuilder, vertex};
use delaunay::prelude::insertion::InsertionOutcome;

let mut dt: DelaunayIncrementalBuilder<_, (), (), 3> =
    DelaunayIncrementalBuilder::new();

let (outcome, stats) = dt
    .insert_best_effort_with_statistics(vertex![0.5, 0.5, 0.5]?)?;

if stats.used_perturbation() {
    println!("used perturbation (attempts={})", stats.attempts);
}

match outcome {
    InsertionOutcome::Inserted { vertex_key, hint: _ } => {
        println!("inserted: {vertex_key:?}");
    }
    InsertionOutcome::Skipped { error } => {
        println!("skipped: {error}");
        println!("retryable? {}", error.is_retryable());
    }
}
```

### Flip-based repair and Delaunay verification (v0.7.3+)

`DelaunayTriangulation` runs flip-based repair passes to restore the local Delaunay property
after insertion. The primary pass uses the caller's kernel, matching the insertion path. If
that pass is non-convergent or fails its postcondition, insertion replays repair with
`RobustKernel` before returning an error. Each pass therefore uses one consistent predicate
policy; the robust replay is an explicit fallback attempt rather than an in-place predicate
override.

Since v0.7.3, the exact flip-repair boundary requires `K: ExactPredicates` at
compile time. `AdaptiveKernel`, `RobustKernel`, and `FastKernel` satisfy that
contract through D ≤ 6, so public conversion is available for each policy in
those dimensions.

To convert a Levels 1–4 triangulation explicitly, use the consuming workflow:

- `DelaunayRefinementBuilder::new(tri).repair_by_flips().build()`
- enable bounded rebuild recovery with
  `.repair_by_flips().fallback_rebuild(true).build()`

After construction or conversion, verify the Delaunay property via
`dt.is_valid_delaunay()`. Complete Euclidean point-set triangulations use a
fast local certificate when possible and otherwise fall back to the exact
global empty-circumsphere check.

For full-stack diagnostics (Levels 1-5), use `dt.validate()` or `dt.validation_report()`;
see `docs/construction_and_validation.md`.

### Exact circumcenter computation (v0.7.3+)

Circumcenter computation keeps ordinary well-conditioned systems on the
allocation-free `f64` LU path. When that solve is near singular, the cold path
converts the original point coordinates to `la-stack`'s re-exported
`BigRational`, forms every difference and squared-norm term rationally, solves
the exact derived system, and only then rounds the center to finite `f64`.
Forming the system first in `f64` and passing it to an exact solver is
insufficient because cancellation may already have erased the affine offset the
solver is meant to recover.

## Duplicate vertex handling

Duplicate or near-duplicate vertices are a common source of geometric degeneracy: they
produce zero-volume simplices whose orientation determinant is exactly zero, breaking
SoS perturbation, Pachner moves, and Delaunay repair. This crate applies a three-layer
defense-in-depth strategy so that duplicate vertices are caught early and never reach
the triangulation interior.

### Layer 1: Hilbert-sort preprocessing dedup (batch construction)

When vertices are inserted via batch construction
(`DelaunayTriangulationBuilder::new(&vertices).build()`,
`DelaunayTriangulationBuilder::new(&vertices).build_with_kernel(&kernel)`,
etc.) using the default
`InsertionOrderStrategy::Hilbert`, the
Hilbert ordering pass quantizes each coordinate to a fixed-width integer grid before
computing the space-filling curve index. After sorting, vertices that map to the same
quantized grid cell are adjacent and are removed in a single linear sweep.

The quantization resolution is `min(128/D, 31)` bits per coordinate, giving:

- 2D: 31 bits/coord → ~10⁻⁹ relative resolution
- 3D: 31 bits/coord → ~10⁻⁹ relative resolution
- 4D: 31 bits/coord → ~10⁻⁹ relative resolution
- 5D: 25 bits/coord → ~10⁻⁸ relative resolution

This layer is **unconditional** when Hilbert ordering is active (the default)
and runs in O(n log n) time with zero extra allocation (the quantized
coordinates are already computed during Hilbert index generation). It removes
the vast majority of exact and near-duplicate vertices before any insertion
occurs, regardless of `DedupPolicy`.

See `order_vertices_hilbert` (called from `order_vertices_by_strategy`) in
[`src/delaunay/construction.rs`](../src/delaunay/construction.rs).

### Layer 2: Per-insertion duplicate coordinate check

Every call to `insert_transactional` checks the incoming vertex against existing
vertices before attempting insertion. When a hash-grid spatial index is
available and its simplex size covers the current tolerance, this is an amortized
local lookup; otherwise it falls back to a linear scan.

The check uses a scale-aware **distance** tolerance, not a fixed squared-distance
threshold. For `f64`, the relative factor is `1e-10`.
The actual tolerance is estimated from a nearby simplex span or local feature
scale, with a small ULP-scaled floor for translated coordinate systems. The
comparison is overflow-safe: it compares squared distances against
`tolerance²` when possible and falls back to square roots for extreme scales.

If a duplicate is detected, strict APIs return
`InsertionError::DuplicateCoordinates` and the triangulation is unchanged.
`insert_best_effort_with_statistics()` instead returns
`InsertionOutcome::Skipped { error: DuplicateCoordinates { .. } }` with skip
telemetry.

This layer catches duplicates that survive Hilbert dedup (e.g. when using
`InsertionOrderStrategy::Input`) and also protects single-vertex `insert_vertex()` calls.

See `duplicate_coordinates_error` in
[`src/triangulation/insertion.rs`](../src/triangulation/insertion.rs).

### Layer 3: Simplex-level coordinate uniqueness validation

As a post-hoc safety net, cumulative `Tds::validate()` (Levels 1–2) includes a Level 1
`SimplexCoordinateUniqueness` check that scans every simplex for pairs of vertices with
identical coordinates. This uses exact `OrderedFloat`-based comparison (NaN-aware,
+0.0 == -0.0) via `coords_equal_exact`.

Unlike the per-insertion check (which uses a distance tolerance), this validation
detects only exact floating-point matches — it is a strict invariant that should
never be violated if Layers 1 and 2 are working correctly.

If violated, the error is
`TdsError::DuplicateCoordinatesInSimplex { simplex_id, message }`.

See `validate_simplex_coordinate_uniqueness` in
[`src/core/tds/validation.rs`](../src/core/tds/validation.rs).

### User-facing dedup utilities

For explicit preprocessing, the crate provides public deduplication functions in
`delaunay::prelude`:

- `dedup_vertices_exact(&[Vertex])` — removes exact coordinate duplicates (O(n²))
- `try_dedup_vertices_epsilon(&[Vertex], epsilon)` — fallible epsilon dedup
  that removes near-duplicates within Euclidean distance `epsilon` (O(n²)) and
  rejects negative, NaN, or infinite tolerances with a typed error
- `filter_vertices_excluding(&[Vertex], &[Vertex])` — excludes vertices matching
  reference coordinates (e.g. an initial simplex)

These are useful when you need fine-grained control over deduplication before
construction, or when using a non-Hilbert insertion order.

### Choosing a `DedupPolicy`

`DedupPolicy` is a **performance-tuning** knob, not a correctness requirement.
Layer 1 is active when using the default Hilbert ordering, and Layer 2 is always
active regardless of this setting.

- `DedupPolicy::Off` *(default)*: rely on the built-in Hilbert dedup (Layer 1)
  and per-insertion checks (Layer 2). This is sufficient for most use cases.
- `DedupPolicy::Exact`: additionally apply `dedup_vertices_exact` before
  construction. This is a performance optimisation for inputs with many exact
  duplicates — it avoids paying per-vertex insertion overhead for each one.
- `DedupPolicy::try_epsilon(value)`: additionally apply epsilon deduplication
  with the parsed tolerance before construction.

The default (`Off`) is recommended because Hilbert dedup is free (zero extra cost)
and per-insertion checks handle any remaining cases.

## Practical recommendations

- Start with the default `AdaptiveKernel` (`DelaunayTriangulationBuilder::new(&vertices).build()` /
  `DelaunayIncrementalBuilder::new()`).
  This handles near-degenerate configurations correctly out of the box.
- If you need explicit `BOUNDARY`/`DEGENERATE` signals (e.g. to detect and handle cospherical
  configurations yourself), switch to `RobustKernel`.
- Use finite, reasonably scaled coordinates. Extreme magnitudes and tiny local
  feature sizes are supported better than before, but they still increase exact
  arithmetic and duplicate-detection costs.
- Use `FastKernel` when explicit degeneracy signals are useful but the
  diagnostic consistency check and higher-dimensional fallbacks are not. For
  direct incremental insertion or exploratory batch construction, consider
  setting `DelaunayRepairPolicy::EveryN(n)` (e.g. `n = 10`) instead of the
  default `EveryInsertion` repair policy. Batch construction exposes this
  through `ConstructionOptions::with_batch_repair_policy(...)` and still
  performs final repair/validation. This reduces the frequency of the automatic
  robust-fallback repair pass while still maintaining the Delaunay property
  periodically. The consuming `DelaunayRefinementBuilder` flip-repair workflow
  is available through D ≤ 6 because `FastKernel` carries the
  dimension-bounded exact predicate proof.
- If you see retryable insertion errors, frequent perturbation retries, or skipped vertices,
  preprocess your input (dedup / rescale if appropriate).
- Treat `InsertionOutcome::Skipped { .. }` from the best-effort API as an expected outcome on
  pathological data; decide at the application level whether to drop the skipped point,
  perturb or rescale your point set, or re-run with a different kernel.

## Current limitations

- **D ≥ 5 performance:** for dimensions 5 and above, `det_errbound()` is not
  available, so predicates fall through directly to exact Bareiss arithmetic on
  more calls. This is correct but slower than the fast-filter path used for
  D ≤ 4.
- **D ≥ 7 exact repair:** the public exact-insphere / `ExactPredicates` repair
  contract stops at D ≤ 6.
  See [`limitations.md`](limitations.md) for the current dimension envelope.
- **Non-finite input:** robust predicates return typed errors or
  `BOUNDARY`/`DEGENERATE` when exact arithmetic cannot run. Clean or reject
  non-finite coordinates before construction when possible.

Historical investigation notes live in `docs/archive/`.
