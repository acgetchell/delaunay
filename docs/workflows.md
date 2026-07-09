# Workflows: Construction API and Pachner Moves

This document provides small, practical recipes for working with triangulations.

- **Builder API**: construct and maintain Delaunay triangulations via `DelaunayTriangulation`.
- **Pachner Move API**: explicitly edit triangulation topology via local Pachner moves.

For the full design discussion (and more extensive examples), see [`api_design.md`](api_design.md).
For validation semantics and configuration details, see [`validation.md`](validation.md).
For the theoretical background and rationale behind the invariants, see [`invariants.md`](invariants.md).

Examples that derive `thiserror::Error` assume the example crate includes
`thiserror`; run `cargo add thiserror` alongside `delaunay` when copying those
snippets into an application.

## Builder API: the happy path

For most use cases, construction is a single call:

```rust
use delaunay::prelude::construction::{DelaunayResult, DelaunayTriangulationBuilder, vertex};

fn main() -> DelaunayResult<()> {
    let vertices = vec![
        vertex![0.0, 0.0, 0.0]?,
        vertex![1.0, 0.0, 0.0]?,
        vertex![0.0, 1.0, 0.0]?,
        vertex![0.0, 0.0, 1.0]?,
    ];

    let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;

    // Optional verification (see docs/validation.md for when to use each):
    assert!(dt.as_triangulation().validate_realization().is_ok()); // Levels 1-4 (Valid Realization)
    assert!(dt.is_valid_delaunay().is_ok()); // Level 5 only (Geometric Predicates: Delaunay)
    Ok(())
}
```

## Builder API: topology guarantees and automatic validation

Two knobs are commonly used for insertion-time safety vs performance:

- `TopologyGuarantee`: what Level 3 Intrinsic PL Topology invariants are enforced.
- `ValidationPolicy`: when Level 3 Intrinsic PL Topology validation runs automatically during incremental insertion.

Use the `try_set_*` policy setters when changing both axes programmatically; they
return a typed error for incoherent combinations such as
`TopologyGuarantee::PLManifold` with `ValidationPolicy::Never`.

See [`validation.md`](validation.md) for details.

```rust
use delaunay::prelude::construction::{DelaunayTriangulation, TopologyGuarantee};
use delaunay::prelude::validation::ValidationPolicy;

let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();

// Enforce stricter topology checks.
dt.set_topology_guarantee(TopologyGuarantee::PLManifoldStrict);

// In tests/debugging, validate global Level 3 and changed-scope Level 4 after every insertion.
dt.set_validation_policy(ValidationPolicy::Always);
```

### What the topology guarantees mean (quick summary)

- `TopologyGuarantee::Pseudomanifold`:
  validates facet degree (each facet is incident to 1 or 2 simplices) and a closed boundary
  ("no boundary of boundary").
- `TopologyGuarantee::PLManifold` *(default)*:
  adds **ridge-link validation during insertion** and requires a **vertex-link validation pass at
  construction completion** to certify full PL-manifoldness.
- `TopologyGuarantee::PLManifoldStrict`:
  runs **vertex-link validation after every insertion** (slowest, maximum safety).

See [`validation.md`](validation.md) for the precise invariants and which methods validate which
levels.

## Builder API: flip-based Delaunay repair (details)

The Builder API is designed to construct Delaunay triangulations, and (by default) schedules local
flip-based repair passes during construction. Batch construction uses `ConstructionOptions`, whose
default repair cadence is `DelaunayRepairPolicy::EveryInsertion` plus final repair/validation. That
cadence reflects the current #341 3D scale acceptance path: the release-mode
`just debug-large-scale-3d 7500 1` harness is the current roughly one-minute
maintainer-hardware envelope for final Levels 1–5 validation. The explicit
`just debug-large-scale-3d 10000 1` run is a heavier characterization probe
that has also passed the same final validation checks. Direct incremental insertion keeps the lower-level
`DelaunayRepairPolicy` default at `EveryInsertion`.
The explicit repair methods (`repair_delaunay_with_flips`, `repair_delaunay_with_flips_advanced`,
`rebuild_with_heuristic`) require `K: ExactPredicates` at compile time. `AdaptiveKernel` and
`RobustKernel` implement this trait; `FastKernel` does not. See
[`numerical_robustness_guide.md`](numerical_robustness_guide.md) for kernel selection guidance.

```rust
use delaunay::prelude::construction::{DelaunayRepairPolicy, DelaunayTriangulation};

let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();

// Default:
assert_eq!(dt.delaunay_repair_policy(), DelaunayRepairPolicy::EveryInsertion);

// Disable automatic repairs (manual repair is still available):
dt.set_delaunay_repair_policy(DelaunayRepairPolicy::Never);
```

You can also run a global repair pass manually:

```rust
use delaunay::prelude::construction::{
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, vertex,
};
use delaunay::prelude::geometry::CoordinateConversionError;
use delaunay::prelude::repair::DelaunayRepairError;

#[derive(Debug, thiserror::Error)]
enum RepairExampleError {
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    Repair(#[from] DelaunayRepairError),
    #[error(transparent)]
    Coordinate(#[from] CoordinateConversionError),
}

fn main() -> Result<(), RepairExampleError> {
    let vertices = vec![
        vertex![0.0, 0.0, 0.0]?,
        vertex![1.0, 0.0, 0.0]?,
        vertex![0.0, 1.0, 0.0]?,
        vertex![0.0, 0.0, 1.0]?,
    ];

    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build()?;

    let _stats = dt.repair_delaunay_with_flips()?;
    Ok(())
}
```

### Topology and kernel requirements

Flip-based repair requires a PL-manifold topology guarantee. If your triangulation is configured as
`TopologyGuarantee::Pseudomanifold`, `repair_delaunay_with_flips()` returns an error.

Additionally, all explicit repair methods require `K: ExactPredicates` (compile-time bound).
The default `AdaptiveKernel` satisfies this. `FastKernel` does not — its automatic
insertion-time repair uses a `RobustKernel` fallback internally, but the public repair
methods are not available.

### Repair attempts and diagnostics

Internally, standard flip-based repair uses two bounded attempts:

1. Attempt 1: FIFO queue order seeded from the requested local frontier, or from
   all simplices when the caller explicitly requests a global repair.
2. Attempt 2: LIFO queue order with a full re-seed of the repair queue. This
   runs only after attempt 1 fails to converge or fails its postcondition.

After an attempt completes, repair verifies the Delaunay postcondition with the
same flip predicates used by the repair loop. A postcondition failure is treated
similarly to non-convergence and triggers the second attempt or a caller-level
fallback.

The public advanced repair path can then try a robust-kernel pass and, if that
still fails, a deterministic heuristic rebuild.

If repair fails to converge within the flip budget, you get
`DelaunayRepairError::NonConvergent { .. }`, which contains a `DelaunayRepairDiagnostics` payload
(facets checked, flips performed, max queue length, ambiguous predicate counts + samples, cycle
detections, etc.).

```rust
use delaunay::prelude::construction::{
    DelaunayResult, DelaunayTriangulationBuilder, vertex,
};
use delaunay::prelude::repair::DelaunayRepairError;

fn main() -> DelaunayResult<()> {
    let vertices = vec![
        vertex![0.0, 0.0, 0.0]?,
        vertex![1.0, 0.0, 0.0]?,
        vertex![0.0, 1.0, 0.0]?,
        vertex![0.0, 0.0, 1.0]?,
    ];

    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build()?;

    match dt.repair_delaunay_with_flips() {
        Ok(_stats) => {}
        Err(DelaunayRepairError::NonConvergent { diagnostics, .. }) => {
            eprintln!("repair non-convergent: {diagnostics}");
        }
        Err(err) => {
            eprintln!("repair failed: {err}");
        }
    }
    Ok(())
}
```

## Pachner Move API: simplex barycenter insert point

For a k=1 insert into an existing simplex, use `simplex_barycenter` to derive a
topology-aware interior point from the live triangulation. In Euclidean
triangulations this is the arithmetic average of the simplex vertices; in
periodic image-point triangulations the method lifts through stored periodic
offsets before averaging and canonicalizing back into the domain.

```rust
use delaunay::prelude::construction::{DelaunayResult, DelaunayTriangulationBuilder, vertex};
use delaunay::prelude::pachner::{PachnerMove, PachnerMoves};

fn main() -> DelaunayResult<()> {
    let vertices = vec![
        vertex![0.0, 0.0, 0.0]?,
        vertex![1.0, 0.0, 0.0]?,
        vertex![0.0, 1.0, 0.0]?,
        vertex![0.0, 0.0, 1.0]?,
    ];
    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    let Some((simplex_key, _)) = dt.simplices().next() else {
        return Ok(());
    };

    let barycenter = dt.simplex_barycenter(simplex_key)?;
    dt.propose_pachner(PachnerMove::K1Insert {
        simplex_key,
        vertex: vertex!(*barycenter.coords())?,
    })?
    .attempt_on(&mut dt)?;
    Ok(())
}
```

### Advanced repair with heuristic rebuild

If you want a stronger "try harder" path, call
`repair_delaunay_with_flips_advanced(DelaunayRepairHeuristicConfig)`, which returns a
`DelaunayRepairOutcome`.

This method:

1. Runs the standard flip-repair.
2. If it fails (non-convergent or postcondition failure), tries a robust-kernel repair pass.
3. If it still fails, rebuilds the triangulation from the **current vertex set** using a shuffled
   insertion order and a perturbation seed, then runs a final flip-repair pass.

If a heuristic rebuild is used, the outcome records the seeds in `outcome.heuristic`.
You can provide explicit seeds for reproducibility; otherwise deterministic defaults are derived
from the current vertex set.

```rust
use delaunay::prelude::construction::{
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, vertex,
};
use delaunay::prelude::geometry::CoordinateConversionError;
use delaunay::prelude::repair::{DelaunayRepairError, DelaunayRepairHeuristicConfig};

#[derive(Debug, thiserror::Error)]
enum RepairExampleError {
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    Repair(#[from] DelaunayRepairError),
    #[error(transparent)]
    Coordinate(#[from] CoordinateConversionError),
}

fn main() -> Result<(), RepairExampleError> {
    let vertices = vec![
        vertex![0.0, 0.0, 0.0]?,
        vertex![1.0, 0.0, 0.0]?,
        vertex![0.0, 1.0, 0.0]?,
        vertex![0.0, 0.0, 1.0]?,
    ];

    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build()?;

    let outcome = dt
        .repair_delaunay_with_flips_advanced(DelaunayRepairHeuristicConfig::default())?;

    if let Some(seeds) = outcome.heuristic {
        eprintln!("heuristic rebuild used: {seeds:?}");
    }
    Ok(())
}
```

## Builder API: toroidal construction modes

Toroidal construction has two modes. `.try_canonicalized_toroidal([..])` wraps vertices into the
fundamental domain before Euclidean construction. `.try_toroidal([..])`
uses the image-point method to build a true periodic quotient in the validated
2D and compact 3D cases.

```rust
use delaunay::prelude::construction::{
    DelaunayResult, DelaunayTriangulationBuilder, vertex,
};

fn main() -> DelaunayResult<()> {
    // 2D Euclidean triangulation after wrapping inputs into a unit square domain
    let vertices = vec![
        vertex![0.1, 0.1]?,
        vertex![0.9, 0.9]?,
        vertex![0.5, 0.5]?,
    ];

    let mut dt = DelaunayTriangulationBuilder::new(&vertices)
        .try_canonicalized_toroidal([1.0, 1.0])
        ? // input coordinate canonicalization
        .build()?;

    // Subsequent insertions are standard Euclidean insertions; canonicalize
    // additional points at the call site if they come from the same domain.
    dt.insert_vertex(vertex![0.2, 0.3]?)?;
    dt.insert_vertex(vertex![0.9, 0.7]?)?;
    Ok(())
}
```

**Key points:**

- **Domain wrapping**: Initial vertex coordinates are canonicalized (wrapped) to the
  fundamental domain `[0, period)` for each dimension before Euclidean construction
- **Manifold topology**: `.try_canonicalized_toroidal([..])` does not assign closed toroidal
  topology or rewire opposite boundary facets; use `.try_toroidal([..])` for a true quotient
- **Construction modes**:
  - `.try_canonicalized_toroidal([..])`: Euclidean construction after wrapping inputs
  - `.try_toroidal([..])`: periodic image-point construction; currently validated as a true
    toroidal quotient in 2D and compact 3D; 4D/5D fail fast pending issue #416

For more details, see `docs/topology.md` and the toroidal section in the main `README.md`.

## Builder API: auxiliary vertex and simplex data

Vertices and simplices can carry user-defined auxiliary data (`U` for vertices, `V` for simplices).
Vertex data is attached at construction time via `vertex![...; data = ...]`, read via the `data()`
accessor, and modified post-construction via `set_vertex_data` / `set_simplex_data`.

```rust
use delaunay::prelude::construction::{
    DelaunayResult, DelaunayTriangulationBuilder, Vertex, vertex,
};

fn main() -> DelaunayResult<()> {
    // Attach integer labels at construction time
    let vertices: [Vertex<i32, 2>; 3] = [
        vertex![0.0, 0.0; data = 10i32]?,
        vertex![1.0, 0.0; data = 20]?,
        vertex![0.0, 1.0; data = 30]?,
    ];
    let mut dt = DelaunayTriangulationBuilder::new(&vertices).simplex_data_type::<i32>().build()?;

    // Read vertex data
    for (_key, vertex) in dt.vertices() {
        println!("data = {:?}", vertex.data()); // Some(10), Some(20), or Some(30)
    }

    // Modify vertex data (O(1), does not affect geometry or topology)
    let Some((key, _)) = dt.vertices().next() else {
        return Ok(());
    };
    let prev = dt.set_vertex_data(key, Some(99))?;
    assert!(prev.is_some()); // returns the old Option<U>

    // Simplex data works the same way
    let Some((simplex_key, _)) = dt.simplices().next() else {
        return Ok(());
    };
    dt.set_simplex_data(simplex_key, Some(42))?;
    assert_eq!(dt.simplex(simplex_key).map(|s| s.data()), Some(Some(&42)));
    Ok(())
}
```

`set_vertex_data` and `set_simplex_data` are checked O(1) operations — they modify only the
user-data field, return the previous payload on success, and fail with a typed mutation error
if the supplied key no longer exists. Successful calls do not invalidate geometry, topology, or
Delaunay invariants.

For algorithm-local state keyed by existing vertices or simplices, prefer the
caller-owned secondary-map aliases instead of mutating stored user data:

```rust
use delaunay::prelude::collections::{SimplexSecondaryMap, VertexSecondaryMap};

let mut visited_simplices: SimplexSecondaryMap<bool> = SimplexSecondaryMap::new();
let mut vertex_order: VertexSecondaryMap<usize> = VertexSecondaryMap::new();

for (simplex_key, _) in dt.simplices() {
    visited_simplices.insert(simplex_key, false);
}
for (order, (vertex_key, _)) in dt.vertices().enumerate() {
    vertex_order.insert(vertex_key, order);
}
```

## Builder API: insertion statistics

If you need strict observability where duplicate or retry-exhausted skipped
insertions become errors, use `insert_with_statistics()`. If you intentionally
want to keep going after skipped vertices, use the explicitly best-effort
`insert_best_effort_with_statistics()`.

```rust
use delaunay::prelude::construction::{DelaunayResult, DelaunayTriangulation, vertex};
use delaunay::prelude::insertion::InsertionOutcome;

fn main() -> DelaunayResult<()> {
    let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();

    let (outcome, stats) = dt.insert_best_effort_with_statistics(vertex![0.5, 0.5, 0.5]?)?;

    if stats.used_perturbation() {
        println!("used perturbation (attempts={})", stats.attempts);
    }

    match outcome {
        InsertionOutcome::Inserted { vertex_key, hint: _ } => {
            println!("inserted: {vertex_key:?}");
        }
        InsertionOutcome::Skipped { error } => {
            println!("skipped: {error}");
        }
    }
    Ok(())
}
```

For guidance on retry/skip behavior and choosing `RobustKernel`, see
[`numerical_robustness_guide.md`](numerical_robustness_guide.md).

## Builder API: deleting a vertex

Vertex deletion is supported and preserves Levels 1–3. It uses an inverse k=1 fast path when
possible and fan retriangulation otherwise, then runs flip-based Delaunay repair when the active
`DelaunayRepairPolicy` allows it. If automatic repair is disabled, deletion still runs Level 4
Level 4 realization validation and the Level 5 Delaunay predicate, rolling back on any violation. If post-deletion repair, validation, or
orientation canonicalization fails, the operation rolls back to the pre-deletion triangulation.

```rust
use delaunay::prelude::construction::{
    DelaunayResult, DelaunayTriangulationBuilder, vertex,
};

fn main() -> DelaunayResult<()> {
    let vertices = vec![
        vertex![0.0, 0.0, 0.0]?,
        vertex![1.0, 0.0, 0.0]?,
        vertex![0.0, 1.0, 0.0]?,
        vertex![0.0, 0.0, 1.0]?,
        vertex![0.2, 0.2, 0.2]?,
    ];

    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    let Some((vertex_key, _)) = dt.vertices().next() else {
        return Ok(());
    };

    let _simplices_removed = dt.delete_vertex(vertex_key)?;

    // Topology should still be valid:
    assert!(dt.as_triangulation().validate().is_ok());

    // If automatic repair is enabled, successful deletion has already attempted to
    // restore the Delaunay property.
    assert!(dt.is_valid_delaunay().is_ok());
    Ok(())
}
```

When automatic repair fails after the mutation, `delete_vertex` reports
`DeleteVertexError::InvariantViolation { source:
Box::new(InvariantError::Delaunay(DelaunayTriangulationValidationError::RepairOperationFailed {
operation: DelaunayRepairOperation::VertexRemoval, source })) }`, preserving the underlying
`DelaunayRepairError` for callers that need to inspect the exact repair failure.
Successful deletions invalidate internal locate hints so stale simplex handles
are not reused. The spatial index is retained, but the deleted vertex entry is
removed; later spatial lookups still validate candidate keys against the live
TDS before using them.

## Pachner Move API: minimal local move example

The Pachner Move API exposes explicit local bistellar moves. These operations do **not** automatically restore
(or preserve) Level 5 Geometric Predicates such as Delaunay.

After using flips, you typically:

1. validate Intrinsic PL Topology (Level 3), and
2. optionally repair / verify Level 5 Geometric Predicates.

See [`api_design.md`](api_design.md) for the full construction vs local move API design.

```rust
use delaunay::prelude::construction::{
    DelaunayResult, DelaunayTriangulationBuilder, vertex,
};
use delaunay::prelude::pachner::{PachnerMove, PachnerMoves};

fn main() -> DelaunayResult<()> {
    let vertices = vec![
        vertex![0.0, 0.0, 0.0]?,
        vertex![1.0, 0.0, 0.0]?,
        vertex![0.0, 1.0, 0.0]?,
        vertex![0.0, 0.0, 1.0]?,
    ];
    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build()?;

    // k=1: split a simplex by inserting a vertex.
    let Some((simplex_key, _)) = dt.simplices().next() else {
        return Ok(());
    };
    let info = dt
        .propose_pachner(PachnerMove::K1Insert {
            simplex_key,
            vertex: vertex![0.1, 0.1, 0.1]?,
        })?
        .attempt_on(&mut dt)?;
    let inserted_vertex = info.inserted_face_vertices[0];

    // k=1 inverse: remove the inserted vertex (collapse its star).
    let removed = dt
        .propose_pachner(PachnerMove::K1Remove {
            vertex_key: inserted_vertex,
        })?
        .attempt_on(&mut dt)?;
    assert!(!removed.removed_simplices.is_empty());

    // Validate the stack (Levels 1–3) after topological edits.
    assert!(dt.as_triangulation().validate().is_ok());

    // If you need Delaunay after edits (requires K: ExactPredicates):
    // dt.repair_delaunay_with_flips()?;
    // assert!(dt.is_valid_delaunay().is_ok());
    Ok(())
}
```
