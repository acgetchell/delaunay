# Numerical Robustness Guide

Delaunay triangulation can be sensitive to floating-point roundoff when point sets are near-degenerate
(nearly coplanar/cospherical), contain extreme coordinate magnitudes, or contain duplicates/near-duplicates.

This document summarizes the robustness tools available in this crate and how to apply them.

## Robustness toolbox

### Exact predicates (v0.7.1+)

Orientation and insphere predicates use a two-stage evaluation:

1. **Provable f64 fast filter (D ≤ 4)** — `la_stack::Matrix::det_errbound()` computes a
   rigorous Shewchuk-style error bound from the f64 determinant.  If the bound certifies
   the sign, no allocation is needed.  For D ≥ 5 (where `det_errbound` is not available),
   the predicate falls through directly to exact arithmetic.
2. **Exact sign** — via `la_stack::Matrix::det_sign_exact`.  Uses exact `BigRational`
   Bareiss elimination.  Provably correct for finite matrix entries.
3. **Indeterminate fallback** — if exact arithmetic cannot run (non-finite entries),
   the predicate returns `BOUNDARY` / `DEGENERATE`.

This applies to `simplex_orientation`, `insphere`, `insphere_lifted`, `robust_orientation`,
and `robust_insphere`.

**Dimension limits:** the stack-allocated matrix dispatch supports up to 7×7 matrices
(`MAX_STACK_MATRIX_DIM = 7`). This means:

- Exact orientation: D ≤ 6 (matrix is (D+1)×(D+1))
- Exact insphere: D ≤ 5 (matrix is (D+2)×(D+2))

For D ≥ 6, `robust_insphere` falls back to symbolic perturbation and centroid-based
tie-breaking.

### Robust predicates (`geometry::robust_predicates`)

The crate includes robust orientation and insphere predicates (e.g. `robust_orientation`,
`robust_insphere`) for near-degenerate configurations. These predicates layer additional
strategies on top of exact predicates: consistency checking against `insphere_distance`,
symbolic perturbation for tie-breaking, and deterministic geometric fallbacks.

Most users won't call these functions directly; instead, select a kernel.

### Kernel selection (`geometry::kernel`)

Kernels control which predicate implementations are used by the triangulation algorithms:

- `AdaptiveKernel<T>` **(default)**: provably correct predicates with zero configuration.
  Uses exact arithmetic (fast filter + Bareiss) for orientation, and adds Simulation of
  Simplicity (`SoS`) for insphere so cospherical ties are broken deterministically (every
  insphere query returns ±1, never 0/BOUNDARY). Best choice for Delaunay triangulation.
- `RobustKernel<T>`: exact-arithmetic predicates that preserve explicit
  `BOUNDARY`/`DEGENERATE` signals and run diagnostic consistency checks. Prefer this when
  your application needs to detect cospherical/coplanar configurations directly.
- `FastKernel<T>`: raw f64 arithmetic, no robustness guarantees. Only suitable for 2D with
  well-conditioned input.

The convenience constructors (`DelaunayTriangulation::new()`, `::empty()`, etc.) use
`AdaptiveKernel`. To opt into a different kernel, use the explicit-kernel constructors:

```rust
use delaunay::geometry::kernel::RobustKernel;
use delaunay::prelude::triangulation::*;

let kernel = RobustKernel::<f64>::new();

let vertices = vec![
    vertex!([0.0, 0.0, 0.0]),
    vertex!([1.0, 0.0, 0.0]),
    vertex!([0.0, 1.0, 0.0]),
    vertex!([0.0, 0.0, 1.0]),
];

let dt: DelaunayTriangulation<RobustKernel<f64>, (), (), 3> =
    DelaunayTriangulation::with_kernel(&kernel, &vertices).unwrap();

assert!(dt.is_valid().is_ok());
```

### Transactional insertion, retries, and skips

Incremental insertion is transactional: if an insertion attempt fails, the triangulation is
rolled back to the pre-insertion state.

Some geometric degeneracies are retryable via a small deterministic perturbation. If retries
are exhausted, the vertex is skipped and you get `InsertionOutcome::Skipped { .. }`
(the triangulation is unchanged).

Use `insert_with_statistics()` to observe this behavior:

```rust
use delaunay::prelude::triangulation::*;

let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();

let (outcome, stats) = dt.insert_with_statistics(vertex!([0.5, 0.5, 0.5])).unwrap();

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
`la_stack::Matrix::solve_exact_f64()` — BigRational Gaussian elimination that
returns exact `f64`-rounded results. This replaces the previous zero-tolerance LU
fallback which could fail on degenerate simplices.

## Practical recommendations

- Start with the default `AdaptiveKernel` (`DelaunayTriangulation::new()` / `::empty()`).
  This handles near-degenerate configurations correctly out of the box.
- If you need explicit `BOUNDARY`/`DEGENERATE` signals (e.g. to detect and handle cospherical
  configurations yourself), switch to `RobustKernel`.
- If you see retryable insertion errors, frequent perturbation retries, or skipped vertices,
  preprocess your input (dedup / rescale if appropriate).
- Treat `InsertionOutcome::Skipped { .. }` as an expected outcome on pathological data; decide
  at the application level whether to drop the vertex, perturb/rescale your point set, or
  re-run with a different kernel.

## Current limitations

- **D ≥ 5 performance:** for dimensions 5 and above, `det_errbound()` is not
  available, so predicates fall through directly to exact Bareiss arithmetic on
  every call.  This is correct but slower than the fast-filter path used for
  D ≤ 4.  Tracked in [#257](https://github.com/acgetchell/delaunay/issues/257).

Historical investigation notes live in `docs/archive/`.
