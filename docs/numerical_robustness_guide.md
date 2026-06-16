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
2. **Exact sign** — via `la_stack::Matrix::det_sign_exact`. Uses exact `BigRational`
   Bareiss elimination. Provably correct for finite matrix entries.
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
- Public exact-insphere / `ExactPredicates` support: D ≤ 5

For D ≥ 6, `robust_insphere` falls back to a distance-based classification when
possible and then to symbolic perturbation/tie-breaking for unresolved boundary
cases. Treat D ≥ 6 triangulation as experimental; explicit repair APIs are not
available through the `ExactPredicates` gate there.

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
  return ±1 rather than 0 for distinct finite inputs. The exception is truly
  identical points (same f64 coordinates), where all SoS cofactors vanish and
  orientation returns 0. Best choice for Delaunay triangulation. Implements
  `ExactPredicates` through D ≤ 5.
- `RobustKernel<T>`: exact-arithmetic predicates that preserve explicit
  `BOUNDARY`/`DEGENERATE` signals and can run opt-in diagnostic consistency checks.
  Prefer this when your application needs to detect cospherical/coplanar/collinear
  configurations directly (SoS would mask these). Implements `ExactPredicates`
  through D ≤ 5.
- `FastKernel<T>`: raw f64 arithmetic, no robustness guarantees. Only suitable for 2D with
  well-conditioned input. Does **not** implement `ExactPredicates`. Construction and
  insertion work (automatic repair uses a `RobustKernel` fallback internally), but the
  explicit public repair methods (`repair_delaunay_with_flips`,
  `repair_delaunay_with_flips_advanced`) are compile-time blocked.

### `ExactPredicates` marker trait (v0.7.3+)

The `ExactPredicates` marker trait identifies kernels whose `orientation` and
`in_sphere` predicates return the mathematically correct sign in the supported
dimension, including near-degenerate configurations. Both `AdaptiveKernel` and
`RobustKernel` implement this trait through D ≤ 5; `FastKernel` does not.

The explicit public repair methods (`repair_delaunay_with_flips`,
`repair_delaunay_with_flips_advanced`, `rebuild_with_heuristic`) require
`K: ExactPredicates`. This is enforced at compile time, preventing silent
misclassification from floating-point-only predicates that can lead to infinite flip
cycles, invalid topology, or non-Delaunay results. Construction and insertion do **not**
require the bound — the internal repair path uses the caller's kernel first and falls
back to `RobustKernel` automatically.

Dimension-bound exactness is intentional: orientation has exact determinant
support through D ≤ 6, but the public exact repair contract is tied to exact
insphere support and the currently tested triangulation envelope, so
`ExactPredicates` stops at D ≤ 5.

`DelaunayTriangulationBuilder::new(&vertices).build::<()>()` and
`DelaunayTriangulation::empty()` use `AdaptiveKernel`. To opt into a different
kernel, use the explicit-kernel constructors:

```rust
use delaunay::prelude::geometry::RobustKernel;
use delaunay::prelude::construction::{DelaunayTriangulation};

let kernel = RobustKernel::<f64>::new();

let vertices = vec![
    delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
    delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
    delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
    delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
];

let dt: DelaunayTriangulation<RobustKernel<f64>, (), (), 3> =
    DelaunayTriangulation::try_with_kernel(&kernel, &vertices)?;

assert!(dt.is_valid().is_ok());
```

### Identity-based SoS perturbation via canonical vertex ordering

The `SoS` (Simulation of Simplicity) implementation assigns perturbation priority
by slice position: the first point in the array gets the lowest-order perturbation
term, the second gets the next, and so on. If different call sites present the
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
use delaunay::prelude::construction::{DelaunayTriangulation};
use delaunay::prelude::insertion::InsertionOutcome;

let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();

let (outcome, stats) = dt
    .insert_best_effort_with_statistics(delaunay::prelude::Vertex::<(), _>::try_new([0.5, 0.5, 0.5])?)?;

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
after insertion. The repair code uses the same kernel predicates as the insertion path —
there is no separate "robust predicate override". This unified predicate pipeline ensures
consistent sign decisions and eliminates flip cycles caused by predicate disagreements.

Since v0.7.3, all repair entry points require `K: ExactPredicates` at compile
time. This prevents accidental use of `FastKernel` for flip repair, which would
produce incorrect results on near-degenerate inputs. It also means explicit
repair APIs are currently available only in dimensions covered by the
`ExactPredicates` marker, D ≤ 5.

You can also run repair manually:

- `dt.repair_delaunay_with_flips()`
- `dt.repair_delaunay_with_flips_advanced(DelaunayRepairHeuristicConfig::default())`

After construction (or repair), verify the Delaunay property via `dt.is_valid()`
(which uses local flip predicates).

For full-stack diagnostics (Levels 1-4), use `dt.validate()` or `dt.validation_report()`;
see `docs/validation.md`.

### Exact circumcenter computation (v0.7.3+)

Circumcenter computation falls back to exact arithmetic when the simplex is
near-singular (ill-conditioned linear system). This uses
`la_stack::Matrix::solve_exact_rounded_f64()` — BigRational Gaussian elimination
with explicit finite `f64` rounding. This replaces the previous zero-tolerance
LU fallback which could fail on degenerate simplices.

## Duplicate vertex handling

Duplicate or near-duplicate vertices are a common source of geometric degeneracy: they
produce zero-volume simplices whose orientation determinant is exactly zero, breaking
SoS perturbation, Pachner moves, and Delaunay repair. This crate applies a three-layer
defense-in-depth strategy so that duplicate vertices are caught early and never reach
the triangulation interior.

### Layer 1: Hilbert-sort preprocessing dedup (batch construction)

When vertices are inserted via batch construction
(`DelaunayTriangulationBuilder::new(&vertices).build::<()>()`,
`DelaunayTriangulation::try_with_kernel()`, etc.) using the default
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
[`src/delaunay/triangulation.rs`](../src/delaunay/triangulation.rs).

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
`InsertionOrderStrategy::Input`) and also protects single-vertex `insert()` calls.

See `duplicate_coordinates_error` in
[`src/core/triangulation.rs`](../src/core/triangulation.rs).

### Layer 3: Simplex-level coordinate uniqueness validation

As a post-hoc safety net, `Tds::validate()` (Level 2 validation) includes a
`SimplexCoordinateUniqueness` check that scans every simplex for pairs of vertices with
identical coordinates. This uses exact `OrderedFloat`-based comparison (NaN-aware,
+0.0 == -0.0) via `coords_equal_exact`.

Unlike the per-insertion check (which uses a distance tolerance), this validation
detects only exact floating-point matches — it is a strict invariant that should
never be violated if Layers 1 and 2 are working correctly.

If violated, the error is
`TdsError::DuplicateCoordinatesInSimplex { simplex_id, message }`.

See `validate_simplex_coordinate_uniqueness` in
[`src/core/tds.rs`](../src/core/tds.rs).

### User-facing dedup utilities

For explicit preprocessing, the crate provides public deduplication functions in
`delaunay::prelude`:

- `dedup_vertices_exact(&[Vertex])` — removes exact coordinate duplicates (O(n²))
- `dedup_vertices_epsilon(&[Vertex], epsilon)` — removes near-duplicates within
  Euclidean distance `epsilon` (O(n²))
- `try_dedup_vertices_epsilon(&[Vertex], epsilon)` — fallible epsilon dedup
  that rejects negative, NaN, or infinite tolerances with a typed error
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
- `DedupPolicy::try_epsilon(value)`: additionally apply
  `dedup_vertices_epsilon` with the parsed tolerance before construction.

The default (`Off`) is recommended because Hilbert dedup is free (zero extra cost)
and per-insertion checks handle any remaining cases.

## Practical recommendations

- Start with the default `AdaptiveKernel` (`DelaunayTriangulationBuilder::new(&vertices).build::<()>()` /
  `DelaunayTriangulation::empty()`).
  This handles near-degenerate configurations correctly out of the box.
- If you need explicit `BOUNDARY`/`DEGENERATE` signals (e.g. to detect and handle cospherical
  configurations yourself), switch to `RobustKernel`.
- Use finite, reasonably scaled coordinates. Extreme magnitudes and tiny local
  feature sizes are supported better than before, but they still increase exact
  arithmetic and duplicate-detection costs.
- Treat `FastKernel` as a low-level or exploratory tool. If you use it for
  direct incremental insertion or exploratory batch construction, consider
  setting `DelaunayRepairPolicy::EveryN(n)` (e.g. `n = 10`) instead of the
  default `EveryInsertion` repair policy. Batch construction exposes this
  through `ConstructionOptions::with_batch_repair_policy(...)` and still
  performs final repair/validation. This reduces the frequency of the automatic
  robust-fallback repair pass while still maintaining the Delaunay property
  periodically. The explicit repair methods (`repair_delaunay_with_flips`,
  etc.) are not available with `FastKernel` — use `AdaptiveKernel` or
  `RobustKernel` if you need manual repair control.
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
- **D ≥ 6 exact repair:** exact orientation is available through D ≤ 6, but
  the public exact-insphere / `ExactPredicates` repair contract stops at D ≤ 5.
  See [`limitations.md`](limitations.md) for the current dimension envelope.
- **Non-finite input:** robust predicates return typed errors or
  `BOUNDARY`/`DEGENERATE` when exact arithmetic cannot run. Clean or reject
  non-finite coordinates before construction when possible.

Historical investigation notes live in `docs/archive/`.
