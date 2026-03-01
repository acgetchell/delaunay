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
use delaunay::prelude::triangulation::*;

let vertices = vec![
    vertex!([0.0, 0.0, 0.0]),
    vertex!([1.0, 0.0, 0.0]),
    vertex!([0.0, 1.0, 0.0]),
    vertex!([0.0, 0.0, 1.0]),
];

let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();

// Optional verification (see docs/validation.md for when to use each):
dt.is_valid().unwrap(); // Level 4 only (Delaunay property)
```

## Builder API: topology guarantees and automatic validation

Two knobs are commonly used for insertion-time safety vs performance:

- `TopologyGuarantee`: what Level 3 topology invariants are enforced.
- `ValidationPolicy`: when Level 3 topology validation runs automatically during incremental insertion.

See [`validation.md`](validation.md) for details.

```rust
use delaunay::prelude::triangulation::*;

let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();

// Enforce stricter topology checks.
dt.set_topology_guarantee(TopologyGuarantee::PLManifoldStrict);

// In tests/debugging, validate Level 3 after every insertion.
dt.set_validation_policy(ValidationPolicy::Always);
```

### What the topology guarantees mean (quick summary)

- `TopologyGuarantee::Pseudomanifold`:
  validates facet degree (each facet is incident to 1 or 2 cells) and a closed boundary
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
flip-based repair passes after insertions.

Automatic repair scheduling is controlled by `DelaunayRepairPolicy` (default: `EveryInsertion`).

```rust
use delaunay::prelude::triangulation::*;

let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();

// Default:
assert_eq!(dt.delaunay_repair_policy(), DelaunayRepairPolicy::EveryInsertion);

// Disable automatic repairs (manual repair is still available):
dt.set_delaunay_repair_policy(DelaunayRepairPolicy::Never);
```

You can also run a global repair pass manually:

```rust
use delaunay::prelude::triangulation::*;

let vertices = vec![
    vertex!([0.0, 0.0, 0.0]),
    vertex!([1.0, 0.0, 0.0]),
    vertex!([0.0, 1.0, 0.0]),
    vertex!([0.0, 0.0, 1.0]),
];

let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();

let _stats = dt.repair_delaunay_with_flips().unwrap();
```

### Topology requirement

Flip-based repair requires a PL-manifold topology guarantee. If your triangulation is configured as
`TopologyGuarantee::Pseudomanifold`, `repair_delaunay_with_flips()` returns an error.

### Repair attempts and diagnostics

Internally, flip-based repair performs multiple attempts using different queue-ordering and
robustness settings. **Currently, this is up to three attempts**:

1. Attempt 1: FIFO queue order (fast path; uses the triangulation's kernel predicates).
2. Attempt 2: LIFO queue order, with robust predicates enabled for ambiguous boundary
   classifications (and a full re-seed of the repair queue).
3. Attempt 3: FIFO queue order again (alternate ordering), still with robust predicates enabled
   for ambiguous boundary classifications.

Note: in debug/test builds for D ≥ 3, attempt 1 may also enable robust predicates for ambiguous
boundary classifications.

After an attempt completes, the algorithm verifies the Delaunay postcondition via local flip
predicates. A postcondition failure is treated similarly to non-convergence and triggers retries.

If repair fails to converge within the flip budget, you get
`DelaunayRepairError::NonConvergent { .. }`, which contains a `DelaunayRepairDiagnostics` payload
(facets checked, flips performed, max queue length, ambiguous predicate counts + samples, cycle
detections, etc.).

```rust
use delaunay::core::algorithms::flips::DelaunayRepairError;
use delaunay::prelude::triangulation::*;

let vertices = vec![
    vertex!([0.0, 0.0, 0.0]),
    vertex!([1.0, 0.0, 0.0]),
    vertex!([0.0, 1.0, 0.0]),
    vertex!([0.0, 0.0, 1.0]),
];

let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();

match dt.repair_delaunay_with_flips() {
    Ok(_stats) => {}
    Err(DelaunayRepairError::NonConvergent { diagnostics, .. }) => {
        eprintln!("repair non-convergent: {diagnostics}");
    }
    Err(err) => {
        eprintln!("repair failed: {err}");
    }
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
use delaunay::core::delaunay_triangulation::DelaunayRepairHeuristicConfig;
use delaunay::prelude::triangulation::*;

let vertices = vec![
    vertex!([0.0, 0.0, 0.0]),
    vertex!([1.0, 0.0, 0.0]),
    vertex!([0.0, 1.0, 0.0]),
    vertex!([0.0, 0.0, 1.0]),
];

let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();

let outcome = dt
    .repair_delaunay_with_flips_advanced(DelaunayRepairHeuristicConfig::default())
    .unwrap();

if let Some(seeds) = outcome.heuristic {
    eprintln!("heuristic rebuild used: {seeds:?}");
}
```

## Builder API: toroidal (periodic) triangulations

Toroidal triangulations handle periodic boundary conditions. Use
`DelaunayTriangulationBuilder` to construct them:

```rust
use delaunay::prelude::triangulation::*;

// 2D periodic triangulation with unit square domain
let vertices = vec![
    vertex!([0.1, 0.1]),
    vertex!([0.9, 0.9]),
    vertex!([0.5, 0.5]),
];

let mut dt = DelaunayTriangulationBuilder::new(&vertices)
    .toroidal([1.0, 1.0]) // Phase 1: canonicalized toroidal construction
    .build::<()>()
    .unwrap();

// Insert more points - they'll be wrapped to [0,1)×[0,1)
dt.insert(vertex!([1.2, 0.3])).unwrap(); // wraps to [0.2, 0.3]
dt.insert(vertex!([-0.1, 0.7])).unwrap(); // wraps to [0.9, 0.7]
```

**Key points:**

- **Domain wrapping**: Vertex coordinates are automatically canonicalized (wrapped) to the
  fundamental domain `[0, period)` for each dimension
- **Distance computation**: Distances are computed accounting for periodic boundaries (toroidal
  metric)
- **Construction modes**:
  - `.toroidal([..])`: Phase 1 canonicalized construction (wrap into fundamental domain)
  - `.toroidal_periodic([..])`: Phase 2 periodic image-point construction (true toroidal quotient)

For more details, see `docs/topology.md` and the toroidal section in the main `README.md`.

## Builder API: insertion statistics

If you need observability (or you want to handle skipped vertices explicitly), use
`insert_with_statistics()`.

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
    }
}
```

For guidance on retry/skip behavior and choosing `RobustKernel`, see
[`numerical_robustness_guide.md`](numerical_robustness_guide.md).

## Builder API: removing a vertex

Vertex removal is supported and preserves Levels 1–3, but it may not preserve the Delaunay
property in all cases. If you need the Delaunay property after removals, run a repair pass and/or
validate explicitly.

```rust
use delaunay::prelude::triangulation::*;

let vertices = vec![
    vertex!([0.0, 0.0, 0.0]),
    vertex!([1.0, 0.0, 0.0]),
    vertex!([0.0, 1.0, 0.0]),
    vertex!([0.0, 0.0, 1.0]),
    vertex!([0.2, 0.2, 0.2]),
];

let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
let vertex_to_remove = dt.vertices().next().unwrap().1.clone();

let _cells_removed = dt.remove_vertex(&vertex_to_remove).unwrap();

// Topology should still be valid:
assert!(dt.as_triangulation().validate().is_ok());

// If you need Delaunay after edits:
// dt.repair_delaunay_with_flips().unwrap();
// dt.is_valid().unwrap();
```

## Edit API: minimal flip example

The Edit API exposes explicit bistellar flips. These operations do **not** automatically restore
(or preserve) the Delaunay property.

After using flips, you typically:

1. validate topology (Level 3), and
2. optionally repair / verify the Delaunay property.

See [`api_design.md`](api_design.md) for the full Builder vs Edit API design.

```rust
use delaunay::prelude::triangulation::*;
use delaunay::prelude::triangulation::flips::*;

let vertices = vec![
    vertex!([0.0, 0.0, 0.0]),
    vertex!([1.0, 0.0, 0.0]),
    vertex!([0.0, 1.0, 0.0]),
    vertex!([0.0, 0.0, 1.0]),
];
let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();

// k=1: split a cell by inserting a vertex.
let cell_key = dt.cells().next().unwrap().0;
let info = dt.flip_k1_insert(cell_key, vertex!([0.1, 0.1, 0.1])).unwrap();
let inserted_vertex = info.inserted_face_vertices[0];

// k=1 inverse: remove the inserted vertex (collapse its star).
let _ = dt.flip_k1_remove(inserted_vertex).unwrap();

// Validate the stack (Levels 1–3) after topological edits.
assert!(dt.as_triangulation().validate().is_ok());

// If you need Delaunay after edits:
// dt.repair_delaunay_with_flips().unwrap();
// dt.is_valid().unwrap();
```
