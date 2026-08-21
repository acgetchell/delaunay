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
use delaunay::prelude::construction::{DelaunayTriangulationDraft, TopologyGuarantee};
use delaunay::prelude::validation::ValidationPolicy;

let mut draft: DelaunayTriangulationDraft<_, (), (), 3> =
    DelaunayTriangulationDraft::with_topology_guarantee(TopologyGuarantee::PLManifold);

// In tests/debugging, validate global Levels 1–4 after every insertion.
draft.try_set_validation_policy(ValidationPolicy::Always)?;
```

### What the topology guarantees mean (quick summary)

- `TopologyGuarantee::Pseudomanifold`:
  validates facet degree (each facet is incident to 1 or 2 simplices) and a closed boundary
  ("no boundary of boundary").
- `TopologyGuarantee::PLManifold` *(default)*:
  adds **ridge- and vertex-link validation**. Every full Level 3 audit checks the
  same PL-manifold contract; `ValidationPolicy::Always` repeats complete Levels
  1–4 audits after every mutation.

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
that has also passed the same final validation checks. Published incremental
insertion keeps the invariant-preserving repair cadence fixed at every
insertion. `DelaunayRepairPolicy` remains configurable through
`ConstructionOptions` while a batch candidate is unpublished; the terminal
still certifies Level 5 before returning it.

At the domain boundary, the strict terminal composes three proof-bearing
operations: construct or deserialize and certify a Levels 1–2 `Tds`; consume it
to prove Level 3 topology and Level 4 realization in `Triangulation`; then run
bounded flip repair and Level 5 certification before publishing
`DelaunayTriangulation`. Each promotion checks only the invariant newly owned
by that layer. Local repairs may be interleaved with insertion as a performance
optimization, but no intermediate state is stored in the stronger owner. The
mathematical basis is incremental topological flipping for regular
triangulations; the implementation retains explicit work budgets and typed
non-convergence because the cited result is not an unconditional bound for
every local schedule and supported geometry. See
[Bistellar (Pachner) Moves and Delaunay Repair](../REFERENCES.md#bistellar-pachner-moves-and-delaunay-repair).

The consuming `delaunayize` conversion requires `K: ExactPredicates` at
compile time. `AdaptiveKernel`, `RobustKernel`, and `FastKernel` implement this
trait through D ≤ 5. See
[`numerical_robustness_guide.md`](numerical_robustness_guide.md) for kernel selection guidance.

```rust
use delaunay::prelude::construction::{
    ConstructionOptions, DelaunayRepairPolicy, DelaunayTriangulationBuilder,
};
use std::num::NonZeroUsize;

fn main() {
    let Some(every_four) = NonZeroUsize::new(4) else {
        return;
    };
    let options = ConstructionOptions::default()
        .with_batch_repair_policy(DelaunayRepairPolicy::EveryN(every_four));

    let _builder = DelaunayTriangulationBuilder::<(), (), 3>::new(&[])
        .construction_options(options);
}
```

To repair and certify a general Levels 1–4 triangulation explicitly, use the
consuming conversion:

```rust
use delaunay::prelude::construction::{
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, vertex,
};
use delaunay::prelude::delaunayize::{DelaunayRefinementBuilder, DelaunayizeError};
use delaunay::prelude::geometry::CoordinateConversionError;
use delaunay::RefinementError;

#[derive(Debug, thiserror::Error)]
enum ConversionExampleError {
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    Delaunayize(#[from] DelaunayizeError),
    #[error(transparent)]
    Coordinate(#[from] CoordinateConversionError),
}

fn main() -> Result<(), ConversionExampleError> {
    let vertices = vec![
        vertex![0.0, 0.0, 0.0]?,
        vertex![1.0, 0.0, 0.0]?,
        vertex![0.0, 1.0, 0.0]?,
        vertex![0.0, 0.0, 1.0]?,
    ];

    let tri = DelaunayTriangulationBuilder::new(&vertices).build_triangulation()?;
    let converted = DelaunayRefinementBuilder::new(tri)
        .repair_by_flips()
        .build()
        .map_err(RefinementError::into_reason)?;
    assert!(converted.triangulation.validate().is_ok());
    Ok(())
}
```

### Topology and kernel requirements

Flip-based conversion requires a PL-manifold topology guarantee. Passing a
`Triangulation` carrying `TopologyGuarantee::Pseudomanifold` returns a
`DelaunayizeRefinementError` whose reason is
`DelaunayizeError::FlipTopologyNotAdmissible` and whose owner is the unchanged
input triangulation.

Additionally, flip-repair refinement requires `K: ExactPredicates` (compile-time bound).
The default `AdaptiveKernel`, `RobustKernel`, and `FastKernel` satisfy this
through D ≤ 5. Their policies differ in tie handling, diagnostics, and
higher-dimensional fallback rather than in the exactness of a returned sign
inside that supported envelope.

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

If requested, the refinement builder can follow failed bounded repair with a
vertex-set rebuild before final certification.

If repair fails to converge within the flip budget, you get
`DelaunayRepairError::NonConvergent { .. }`, which contains a `DelaunayRepairDiagnostics` payload
(facets checked, flips performed, max queue length, ambiguous predicate counts + samples, cycle
detections, etc.).

```rust
use delaunay::prelude::construction::{
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, vertex,
};
use delaunay::prelude::delaunayize::{DelaunayRefinementBuilder, DelaunayizeError};
use delaunay::prelude::geometry::CoordinateConversionError;
use delaunay::prelude::repair::DelaunayRepairError;

#[derive(Debug, thiserror::Error)]
enum DiagnosticExampleError {
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    Coordinate(#[from] CoordinateConversionError),
}

fn main() -> Result<(), DiagnosticExampleError> {
    let vertices = vec![
        vertex![0.0, 0.0, 0.0]?,
        vertex![1.0, 0.0, 0.0]?,
        vertex![0.0, 1.0, 0.0]?,
        vertex![0.0, 0.0, 1.0]?,
    ];

    let tri = DelaunayTriangulationBuilder::new(&vertices).build_triangulation()?;

    match DelaunayRefinementBuilder::new(tri)
        .repair_by_flips()
        .build()
    {
        Ok(_converted) => {}
        Err(failure) => {
            let (tri, reason) = failure.into_parts();
            match reason {
                DelaunayizeError::DelaunayRepairFailed {
                    source: DelaunayRepairError::NonConvergent { diagnostics, .. },
                } => {
                    eprintln!("repair non-convergent: {diagnostics}");
                    assert!(tri.validate_realization().is_ok());
                }
                reason => eprintln!("repair failed: {reason}"),
            }
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
    let mut dt = DelaunayTriangulationBuilder::new(&vertices)
        .build()?
        .into_triangulation();
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

### Opt-in fallback rebuild

If you want a stronger "try harder" path, enable fallback rebuild on the
consuming conversion with `.fallback_rebuild(true)`.

This workflow:

1. Runs the standard flip-repair.
2. If repair fails, rebuilds a candidate from the **current vertex set**.
3. Restores simplex payloads whose vertex-UUID signature still identifies one
   original simplex.
4. Publishes only after cumulative Levels 1–5 certification succeeds.

The outcome reports whether rebuild fallback was used.

```rust
use delaunay::prelude::construction::{
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, vertex,
};
use delaunay::prelude::delaunayize::{DelaunayRefinementBuilder, DelaunayizeError};
use delaunay::prelude::geometry::CoordinateConversionError;
use delaunay::RefinementError;

#[derive(Debug, thiserror::Error)]
enum RepairExampleError {
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    Delaunayize(#[from] DelaunayizeError),
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

    let tri = DelaunayTriangulationBuilder::new(&vertices).build_triangulation()?;
    let converted = DelaunayRefinementBuilder::new(tri)
        .repair_by_flips()
        .fallback_rebuild(true)
        .build()
        .map_err(RefinementError::into_reason)?;
    eprintln!("fallback rebuild used: {}", converted.outcome.used_fallback_rebuild);
    Ok(())
}
```

## Builder API: toroidal construction

`.try_toroidal([..])` uses the image-point method to build a periodic quotient
in the validated `T^2` and compact `T^3` cases.

```rust
use delaunay::prelude::construction::{
    DelaunayResult, DelaunayTriangulationBuilder, vertex,
};
use delaunay::prelude::geometry::RobustKernel;

fn main() -> DelaunayResult<()> {
    let vertices = vec![
        vertex![0.2, 0.3]?,
        vertex![0.8, 0.1]?,
        vertex![0.5, 0.7]?,
        vertex![0.1, 0.9]?,
        vertex![0.6, 0.4]?,
        vertex![0.3, 0.5]?,
        vertex![0.9, 0.2]?,
    ];

    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .try_toroidal([1.0, 1.0])?
        .build_with_kernel(&RobustKernel::new())?;

    dt.validate()?;
    assert!(dt.global_topology().is_periodic());
    Ok(())
}
```

**Key points:**

- **Domain wrapping**: Construction canonicalizes input coordinates into the
  fundamental domain before generating periodic image points.
- **Manifold topology**: Opposite boundary facets are identified in the quotient,
  which is represented as closed toroidal topology.
- **Validated dimensions**: `T^2` and compact `T^3` are release-covered;
  `T^4`/`T^5` fail fast pending issue #416.

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
use delaunay::prelude::construction::{DelaunayResult, DelaunayTriangulationDraft, vertex};
use delaunay::prelude::insertion::InsertionOutcome;

fn main() -> DelaunayResult<()> {
    let mut dt: DelaunayTriangulationDraft<_, (), (), 3> =
        DelaunayTriangulationDraft::new();

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
realization validation and the Level 5 Delaunay predicate, rolling back on any violation. If
post-deletion repair, validation, or orientation canonicalization fails, the operation rolls back to
the pre-deletion triangulation.

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
Box::new(InvariantError::Delaunay { source:
DelaunayTriangulationValidationError::RepairOperationFailed {
operation: DelaunayRepairOperation::VertexRemoval, source } }) }`, preserving the underlying
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
    let mut dt = DelaunayTriangulationBuilder::new(&vertices)
        .build()?
        .into_triangulation();

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
    assert!(dt.validate().is_ok());

    // If you need Delaunay after edits (requires K: ExactPredicates), consume
    // `dt` with `DelaunayRefinementBuilder::new(dt).repair_by_flips().build()`;
    // failure returns `dt` for inspection or a differently configured retry.
    Ok(())
}
```
