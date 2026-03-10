# Numerical Robustness Guide

Delaunay triangulation can be sensitive to floating-point roundoff when point sets are near-degenerate
(nearly coplanar/cospherical), contain extreme coordinate magnitudes, or contain duplicates/near-duplicates.

This document summarizes the robustness tools available in this crate and how to apply them.

## Robustness toolbox

### Exact predicates (v0.7.1+)

Orientation and insphere predicates use a three-stage evaluation:

1. **f64 fast filter** — if the determinant is well outside an adaptive tolerance band,
   the sign is resolved immediately with no allocation.
2. **Exact Bareiss** — via `la_stack::Matrix::det_sign_exact`, the determinant sign is
   computed in exact `BigRational` arithmetic (Bareiss fraction-free elimination).
   Provably correct for finite matrix entries.
3. **Indeterminate fallback** — if exact arithmetic cannot run (non-finite entries),
   the predicate returns `BOUNDARY` / `DEGENERATE`.

This applies to `simplex_orientation`, `insphere`, `insphere_lifted`, `robust_orientation`,
and `robust_insphere`. For D ≤ 4, la-stack's fast filter uses provable Shewchuk-style
error bounds and resolves most queries without allocating.

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

- `FastKernel<T>`: standard floating-point predicates (default; fastest).
- `RobustKernel<T>`: robust predicates with configurable presets (slower, but more stable).

`RobustKernel::new()` uses the balanced preset `config_presets::general_triangulation()`.
You can pick other presets via `RobustKernel::with_config(...)`.

```rust
use delaunay::geometry::kernel::RobustKernel;
use delaunay::geometry::robust_predicates::config_presets;
use delaunay::prelude::triangulation::*;

let kernel = RobustKernel::with_config(config_presets::degenerate_robust::<f64>());

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

### Flip-based repair and Delaunay verification

`DelaunayTriangulation` can run flip-based repair passes to restore the local Delaunay property
after insertion. You can also run them manually:

- `dt.repair_delaunay_with_flips()`
- `dt.repair_delaunay_with_flips_advanced(DelaunayRepairHeuristicConfig::default())`

After construction (or repair), you can verify the Delaunay property via `dt.is_valid()`
(which uses local flip predicates).

For full stack diagnostics (Levels 1-4), use `dt.validate()` or `dt.validation_report()`;
see `docs/validation.md`.

## Practical recommendations

- Start with the default fast path (`DelaunayTriangulation::new()` / `::empty()`).
- If you see retryable insertion errors, frequent perturbation retries, or skipped vertices:
  - preprocess your input (dedup / rescale if appropriate), and/or
  - switch to `RobustKernel` (and optionally a different `config_presets::*` preset).
- Treat `InsertionOutcome::Skipped { .. }` as an expected outcome on pathological data; decide
  at the application level whether to drop the vertex, perturb/rescale your point set, or
  re-run with a different kernel.

## Current limitations

- **Heuristic fast-filter tolerance:** the f64 fast filter uses an adaptive tolerance
  based on the matrix infinity norm.  This is a heuristic, not a provable error bound.
  For ill-conditioned matrices, the fast filter may return early with a result that exact
  arithmetic would override.  la-stack #44 tracks adding Shewchuk-style bounds to close
  this gap.  In practice, the exact Bareiss stage catches these cases.
- **No SoS yet:** when the exact determinant is truly zero (exact degeneracy), predicates
  return `BOUNDARY` / `DEGENERATE`.  The flip-based repair has no tie-breaking strategy
  for these cases, which can cause flip cycles at scale.  #233 tracks adding Simulation
  of Simplicity (SoS) perturbation.

Historical investigation notes live in `docs/archive/`.
