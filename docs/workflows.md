# Workflows: Builder API and Edit API

This document provides small, practical recipes for working with triangulations.

- **Builder API**: construct and maintain Delaunay triangulations via `DelaunayTriangulation`.
- **Edit API**: explicitly edit triangulation topology via bistellar flips.

For the full design discussion (and more extensive examples), see [`api_design.md`](api_design.md).
For validation semantics and configuration details, see [`validation.md`](validation.md).
For the theoretical background and rationale behind the invariants, see [`invariants.md`](invariants.md).

## Builder API: the happy path

For most use cases, construction is a single call:

```rust
use delaunay::prelude::construction::{
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError,
};

fn main() -> Result<(), DelaunayTriangulationConstructionError> {
    let vertices = vec![
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    ];

    let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;

    // Optional verification (see docs/validation.md for when to use each):
    assert!(dt.is_valid().is_ok()); // Level 4 only (Delaunay property)
    Ok(())
}
```

## Builder API: topology guarantees and automatic validation

Two knobs are commonly used for insertion-time safety vs performance:

- `TopologyGuarantee`: what Level 3 topology invariants are enforced.
- `ValidationPolicy`: when Level 3 topology validation runs automatically during incremental insertion.

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

// In tests/debugging, validate Level 3 after every insertion.
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
maintainer-hardware envelope for final Levels 1–4 validation. The explicit
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
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError,
};
use delaunay::prelude::repair::DelaunayRepairError;

#[derive(Debug, thiserror::Error)]
enum RepairExampleError {
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    Repair(#[from] DelaunayRepairError),
}

fn main() -> Result<(), RepairExampleError> {
    let vertices = vec![
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    ];

    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;

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
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError,
};
use delaunay::prelude::repair::DelaunayRepairError;

#[derive(Debug, thiserror::Error)]
enum RepairExampleError {
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
}

fn main() -> Result<(), RepairExampleError> {
    let vertices = vec![
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    ];

    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;

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
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError,
};
use delaunay::prelude::repair::{DelaunayRepairError, DelaunayRepairHeuristicConfig};

#[derive(Debug, thiserror::Error)]
enum RepairExampleError {
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    Repair(#[from] DelaunayRepairError),
}

fn main() -> Result<(), RepairExampleError> {
    let vertices = vec![
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    ];

    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;

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
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError,
};
use delaunay::prelude::insertion::InsertionError;

#[derive(Debug, thiserror::Error)]
enum ToroidalExampleError {
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    Insertion(#[from] InsertionError),
}

fn main() -> Result<(), ToroidalExampleError> {
    // 2D canonicalized toroidal triangulation with unit square domain
    let vertices = vec![
        delaunay::prelude::Vertex::<(), _>::try_new([0.1, 0.1])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.9, 0.9])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.5, 0.5])?,
    ];

    let mut dt = DelaunayTriangulationBuilder::new(&vertices)
        .try_canonicalized_toroidal([1.0, 1.0])
        .expect("unit toroidal domain is valid") // canonicalized toroidal construction
        .build::<()>()?;

    // Insert more points - they'll be wrapped to [0,1)×[0,1)
    dt.insert(delaunay::prelude::Vertex::<(), _>::try_new([1.2, 0.3])?)?; // wraps to [0.2, 0.3]
    dt.insert(delaunay::prelude::Vertex::<(), _>::try_new([-0.1, 0.7])?)?; // wraps to [0.9, 0.7]
    Ok(())
}
```

**Key points:**

- **Domain wrapping**: Vertex coordinates are automatically canonicalized (wrapped) to the
  fundamental domain `[0, period)` for each dimension
- **Distance computation**: Topology-aware operations can use the toroidal metric when the
  triangulation carries toroidal domain metadata
- **Construction modes**:
  - `.try_canonicalized_toroidal([..])`: canonicalized construction (wrap into fundamental domain)
  - `.try_toroidal([..])`: periodic image-point construction; currently validated as a true
    toroidal quotient in 2D and compact 3D; 4D/5D fail fast pending issue #416

For more details, see `docs/topology.md` and the toroidal section in the main `README.md`.

## Builder API: auxiliary vertex and simplex data

Vertices and simplices can carry user-defined auxiliary data (`U` for vertices, `V` for simplices).
Data is attached at construction time via `Vertex::try_new_with_data()`, read via the `data()` accessor,
and modified post-construction via `set_vertex_data` / `set_simplex_data`.

```rust
use delaunay::prelude::construction::{
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, Vertex,
};

fn main() -> Result<(), DelaunayTriangulationConstructionError> {
    // Attach integer labels at construction time
    let vertices: [Vertex<i32, 2>; 3] = [
        delaunay::prelude::Vertex::<_, _>::try_new_with_data([0.0, 0.0], 10i32)?,
        delaunay::prelude::Vertex::<_, _>::try_new_with_data([1.0, 0.0], 20)?,
        delaunay::prelude::Vertex::<_, _>::try_new_with_data([0.0, 1.0], 30)?,
    ];
    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;

    // Read vertex data
    for (_key, vertex) in dt.vertices() {
        println!("data = {:?}", vertex.data()); // Some(10), Some(20), or Some(30)
    }

    // Modify vertex data (O(1), does not affect geometry or topology)
    let Some((key, _)) = dt.vertices().next() else {
        return Ok(());
    };
    let prev = dt.set_vertex_data(key, Some(99));
    assert!(prev.is_some()); // returns the old Option<U>

    // Simplex data works the same way
    let Some((simplex_key, _)) = dt.simplices().next() else {
        return Ok(());
    };
    dt.set_simplex_data(simplex_key, Some(42));
    assert_eq!(dt.tds().simplex(simplex_key).map(|s| s.data()), Some(Some(&42)));
    Ok(())
}
```

`set_vertex_data` and `set_simplex_data` are safe O(1) operations — they modify only the
user-data field and do not invalidate geometry, topology, or Delaunay invariants.

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
use delaunay::prelude::construction::{DelaunayTriangulation};
use delaunay::prelude::insertion::{InsertionError, InsertionOutcome};

fn main() -> Result<(), InsertionError> {
    let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();

    let (outcome, stats) = dt.insert_best_effort_with_statistics(delaunay::prelude::Vertex::<(), _>::try_new([0.5, 0.5, 0.5])?)?;

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

## Builder API: removing a vertex

Vertex removal is supported and preserves Levels 1–3. It uses an inverse k=1 fast path when
possible and fan retriangulation otherwise, then runs flip-based Delaunay repair when the active
`DelaunayRepairPolicy` allows it. If post-removal repair or orientation canonicalization fails,
the operation rolls back to the pre-removal triangulation.

```rust
use delaunay::prelude::construction::{
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError,
};
use delaunay::prelude::tds::InvariantError;

#[derive(Debug, thiserror::Error)]
enum RemovalExampleError {
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    Invariant(#[from] InvariantError),
}

fn main() -> Result<(), RemovalExampleError> {
    let vertices = vec![
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.2, 0.2, 0.2])?,
    ];

    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    let Some((vertex_key, _)) = dt.vertices().next() else {
        return Ok(());
    };

    let _simplices_removed = dt.remove_vertex(vertex_key)?;

    // Topology should still be valid:
    assert!(dt.as_triangulation().validate().is_ok());

    // If automatic repair is enabled, successful removal has already attempted to
    // restore the Delaunay property.
    assert!(dt.is_valid().is_ok());
    Ok(())
}
```

When automatic repair fails after the mutation, `remove_vertex` reports
`InvariantError::Delaunay(DelaunayTriangulationValidationError::RepairOperationFailed { operation:
DelaunayRepairOperation::VertexRemoval, source })`, preserving the underlying
`DelaunayRepairError` for callers that need to inspect the exact repair failure.
Successful removals invalidate internal locate hints and the spatial index so subsequent queries do
not observe stale topology-dependent cache entries.

## Edit API: minimal flip example

The Edit API exposes explicit bistellar flips. These operations do **not** automatically restore
(or preserve) the Delaunay property.

After using flips, you typically:

1. validate topology (Level 3), and
2. optionally repair / verify the Delaunay property.

See [`api_design.md`](api_design.md) for the full Builder vs Edit API design.

```rust
use delaunay::prelude::construction::{DelaunayTriangulationBuilder};
use delaunay::prelude::flips::*;

#[derive(Debug, thiserror::Error)]
enum FlipExampleError {
    #[error(transparent)]
    Construction(#[from] delaunay::prelude::construction::DelaunayTriangulationConstructionError),
    #[error(transparent)]
    Flip(#[from] FlipError),
}

fn main() -> Result<(), FlipExampleError> {
    let vertices = vec![
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    ];
    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;

    // k=1: split a simplex by inserting a vertex.
    let Some((simplex_key, _)) = dt.simplices().next() else {
        return Ok(());
    };
    let info = dt.flip_k1_insert(simplex_key, delaunay::prelude::Vertex::<(), _>::try_new([0.1, 0.1, 0.1])?)?;
    let inserted_vertex = info.inserted_face_vertices[0];

    // k=1 inverse: remove the inserted vertex (collapse its star).
    let _ = dt.flip_k1_remove(inserted_vertex)?;

    // Validate the stack (Levels 1–3) after topological edits.
    assert!(dt.as_triangulation().validate().is_ok());

    // If you need Delaunay after edits (requires K: ExactPredicates):
    // dt.repair_delaunay_with_flips()?;
    // assert!(dt.is_valid().is_ok());
    Ok(())
}
```
