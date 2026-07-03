# API Design: Construction and Local Move Editing

This document explains the public API design for working with Delaunay
triangulations: construction and vertex lifecycle APIs for maintaining Delaunay
triangulations, and the Pachner move API for explicit local topology edits.

## Overview

The library provides two distinct APIs for different use cases:

1. **Builder API** (`DelaunayTriangulation::insert_vertex` / `::delete_vertex`)
   - High-level construction and maintenance of Delaunay triangulations
   - Automatically maintains the Delaunay property (empty circumsphere)
   - Designed for building triangulations from point sets
   - Uses cavity-based insertion and fan retriangulation

2. **Pachner Move API** (`prelude::pachner::PachnerMoves` trait)
   - Explicit local topology editing via Pachner move requests
   - Explicit control over individual topology operations
   - Does **not** automatically restore the Delaunay property
   - Designed for topology manipulation, research, and custom algorithms
   - Keeps raw flip primitives out of preludes; import `delaunay::flips`
     directly only when testing or documenting the primitive layer itself

Examples that derive `thiserror::Error` assume the example crate includes
`thiserror`; run `cargo add thiserror` alongside `delaunay` when copying those
snippets into an application.

## When to Use Each API

### Use the Builder API when

- Building a Delaunay triangulation from a set of points
- Adding/deleting vertices while maintaining the Delaunay property
- You need automatic geometric property preservation
- Working with standard computational geometry workflows

**Example use cases:**

- Computing convex hulls
- Nearest-neighbor queries
- Supplying triangulation data to downstream Voronoi/dual-simplex analysis
  (the crate does not yet extract Voronoi diagrams directly)
- Mesh generation
- Scientific simulations requiring Delaunay meshes

### Use the Pachner Move API when

- Implementing custom topological algorithms
- Researching bistellar flip sequences
- Building non-Delaunay triangulations with specific properties
- Experimenting with topology transformations
- You need explicit control over topology changes

**Example use cases:**

- Implementing custom Delaunay repair strategies
- Topology optimization algorithms
- Research on triangulation properties
- Building triangulations with non-Delaunay constraints
- Educational demonstrations of bistellar flip theory

## Builder API Reference

### Simple Construction: `DelaunayTriangulationBuilder`

For most use cases, the builder with default options is sufficient:

```rust
use delaunay::prelude::construction::{
    DelaunayResult, DelaunayTriangulationBuilder, vertex,
};

fn main() -> DelaunayResult<()> {
    // Simple construction from vertices (Euclidean space, default options)
    let vertices = vec![
        vertex![0.0, 0.0, 0.0]?,
        vertex![1.0, 0.0, 0.0]?,
        vertex![0.0, 1.0, 0.0]?,
        vertex![0.0, 0.0, 1.0]?,
    ];
    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build()?;

    // Incremental insertion (maintains Delaunay property)
    let new_vertex = vertex![0.5, 0.5, 0.5]?;
    dt.insert_vertex(new_vertex)?;

    // Vertex deletion (topology-preserving, with automatic repair when enabled)
    if let Some((vertex_key, _)) = dt.vertices().next() {
        dt.delete_vertex(vertex_key)?;
    }
    Ok(())
}
```

### Advanced Construction: `DelaunayTriangulationBuilder`

For advanced configuration (domain wrapping, toroidal topology, custom validation policies, etc.),
use `DelaunayTriangulationBuilder`:

```rust
use delaunay::prelude::construction::{
    DelaunayResult, DelaunayTriangulationBuilder, TopologyGuarantee, vertex,
};
use delaunay::prelude::validation::ValidationPolicy;

fn main() -> DelaunayResult<()> {
    // Euclidean triangulation of points canonicalized into a toroidal domain.
    let vertices = vec![
        vertex![0.1, 0.1]?,
        vertex![0.9, 0.9]?,
        vertex![0.5, 0.5]?,
    ];

    let mut dt = DelaunayTriangulationBuilder::new(&vertices)
        .try_canonicalized_toroidal([1.0, 1.0])
        ? // Wrap input coordinates before Euclidean construction.
        .topology_guarantee(TopologyGuarantee::PLManifoldStrict)
        .build()?;

    dt.set_validation_policy(ValidationPolicy::Always);

    // Works like any other DelaunayTriangulation
    dt.insert_vertex(vertex![0.25, 0.75]?)?;
    Ok(())
}
```

**When to use the Builder:**

- **Toroidal construction**: Use `.try_toroidal()` for periodic image-point construction or
  `.try_canonicalized_toroidal()` for Euclidean construction after wrapping input coordinates
  into explicit domain periods.
  The periodic image-point path is release-validated in 2D and compact 3D; 4D/5D
  fail fast pending scalable quotient construction in issue #416.
- **Custom topology guarantees**: Set stricter or more relaxed manifold checks
- **Custom validation policies**: Configure `ValidationPolicy` via
  `dt.try_set_validation_policy(...)` before or after build when callers need
  typed feedback for incompatible policy/guarantee pairs. The compatibility
  `dt.set_validation_policy(...)` setter leaves the previous policy unchanged
  for incoherent combinations.
- **Custom repair policies**: Configure Delaunay repair behavior

See `docs/topology.md` for more on toroidal triangulations and `docs/validation.md`
for topology guarantee and validation policy details.

### Key Characteristics

- **Automatic property preservation**: Insertion maintains the Delaunay
  empty-circumsphere property; deletion runs flip-based repair when the active
  `DelaunayRepairPolicy` permits it
- **Cavity-based insertion**: New vertices are inserted by identifying conflicting simplices, removing them, and filling the cavity
- **Transactional vertex deletion**: Vertex deletion uses an inverse k=1 fast path
  when possible and fan-based retriangulation otherwise. If post-deletion
  Delaunay repair or orientation canonicalization fails, the triangulation and
  internal caches are restored to their pre-deletion state.
- **Auxiliary data**: Vertices and simplices carry optional user data (`U` / `V`). Read via `vertex.data()` /
  `simplex.data()`, write via checked `dt.set_vertex_data(key, data)?` /
  `dt.set_simplex_data(key, data)?` calls (O(1), invariant-preserving, typed failure for stale keys).
  See [`workflows.md`](workflows.md) for examples.
- **Error handling**: Operations fail gracefully if they would violate invariants (see
  [`invariants.md`](invariants.md)). Mutating operations that invoke repair use
  typed repair diagnostics where available, for example
  `RepairOperationFailed { operation, source }`.
- **Validation**: The active `ValidationPolicy` (set with
  `dt.try_set_validation_policy(...)` or `dt.set_validation_policy(...)`) governs automatic topology and
  changed-scope embedding guards for subsequent construction/modification operations

### Simplex Barycenters For Local Editing

`DelaunayTriangulation::simplex_barycenter(simplex_key)` computes a topology-aware interior point for
a live `D`-simplex. In Euclidean triangulations it returns the arithmetic average of the simplex
vertices. In periodic image-point triangulations it lifts vertices through their stored periodic
offsets before averaging, then canonicalizes the result back into the topology domain.

Use the returned point when a workflow needs a deterministic local-editing coordinate, especially for
k=1 Pachner insert proposals. The method revalidates the detached `SimplexKey` against the live
triangulation and returns `SimplexBarycenterError` for stale keys, malformed simplex arity, missing
vertices, offset mismatches, topology lift/canonicalization failures, and invalid averaged points.

## Pachner Move API Reference

The local edit API is exposed through the `PachnerMoves` trait in
`prelude::pachner`:

The canonical public workflow is fluent and staged: parse a raw
`PachnerMove` into a provenanced `PachnerProposal`, then dry-run or attempt the
proposal through the proposal object. This keeps mutation explicit while
preserving owner/generation evidence between stages.

```rust
use delaunay::prelude::construction::{
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, vertex,
};
use delaunay::prelude::geometry::CoordinateConversionError;
use delaunay::prelude::pachner::{
    EdgeKey, FacetHandle, FlipError, PachnerMove, PachnerMoves, TriangleHandle,
};

#[derive(Debug, thiserror::Error)]
enum ExampleError {
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    Flip(#[from] FlipError),
    #[error(transparent)]
    Coordinate(#[from] CoordinateConversionError),
}

fn main() -> Result<(), ExampleError> {
    // Start with a valid triangulation
    let vertices = vec![
        vertex![0.0, 0.0, 0.0]?,
        vertex![1.0, 0.0, 0.0]?,
        vertex![0.0, 1.0, 0.0]?,
        vertex![0.0, 0.0, 1.0]?,
    ];
    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build()?;

    // k=1 move: Insert a vertex into a simplex (splits simplex into D+1 simplices)
    let Some((simplex_key, _)) = dt.simplices().next() else {
        return Ok(());
    };
    let info = dt
        .propose_pachner(PachnerMove::K1Insert {
            simplex_key,
            vertex: vertex![0.25, 0.25, 0.25]?,
        })?
        .attempt_on(&mut dt)?;

    // k=1 inverse: Remove a vertex (collapses its star)
    let vertex_key = info.inserted_face_vertices[0];
    dt.propose_pachner(PachnerMove::K1Remove { vertex_key })?
        .attempt_on(&mut dt)?;

    // k=2 move: Flip a facet (2 simplices ↔ D simplices)
    let facet = /* FacetHandle */;
    let info = dt
        .propose_pachner(PachnerMove::K2 { facet })?
        .attempt_on(&mut dt)?;

    // k=2 inverse: Flip from an edge star (D simplices ↔ 2 simplices)
    let edge = EdgeKey::try_new(info.inserted_face_vertices[0], info.inserted_face_vertices[1])?;
    dt.propose_pachner(PachnerMove::K2Inverse { edge })?
        .attempt_on(&mut dt)?;

    // k=3 move: Flip a ridge (3 simplices ↔ D-1 simplices, requires D ≥ 3)
    let ridge = /* RidgeHandle */;
    let info = dt
        .propose_pachner(PachnerMove::K3 { ridge })?
        .attempt_on(&mut dt)?;

    // k=3 inverse: Flip from a triangle star (D-1 simplices ↔ 3 simplices)
    let triangle = TriangleHandle::try_new(
        info.inserted_face_vertices[0],
        info.inserted_face_vertices[1],
        info.inserted_face_vertices[2],
    )?;
    dt.propose_pachner(PachnerMove::K3Inverse { triangle })?
        .attempt_on(&mut dt)?;
    Ok(())
}
```

### Available Flip Operations

#### k=1 Moves (Simplex Split/Merge)

- **Forward (`PachnerMove::K1Insert`)**: Insert a vertex into a simplex, splitting it into D+1 simplices
  - Valid for D ≥ 1
  - Replaces 1 simplex with D+1 simplices
  - Removed face: the entire simplex (D-simplex)
  - Inserted face: the new vertex (0-simplex)

- **Inverse (`PachnerMove::K1Remove`)**: Remove a vertex, collapsing its star
  - Requires the vertex star to be collapsible (star of D+1 simplices forming a ball)
  - Replaces D+1 simplices with 1 simplex

#### k=2 Moves (Facet Flip)

- **Forward (`PachnerMove::K2`)**: Flip a facet shared by 2 simplices
  - Valid for D ≥ 2
  - Replaces 2 simplices with D simplices
  - Removed face: the shared facet ((D-1)-simplex)
  - Inserted face: an edge (1-simplex)

- **Inverse (`PachnerMove::K2Inverse`)**: Flip from an edge star
  - Requires an edge with star of D simplices
  - Replaces D simplices with 2 simplices

#### k=3 Moves (Ridge Flip)

- **Forward (`PachnerMove::K3`)**: Flip a ridge
  - Valid for D ≥ 3
  - Replaces 3 simplices with D-1 simplices
  - Removed face: a ridge ((D-2)-simplex)
  - Inserted face: a triangle (2-simplex)

- **Inverse (`PachnerMove::K3Inverse`)**: Flip from a triangle star
  - Requires a triangle with star of D-1 simplices
  - Replaces D-1 simplices with 3 simplices

### Key Characteristics

- **Explicit control**: You specify exactly which flip to perform
- **Provenanced proposals**: Raw `PachnerMove` values are parsed into
  `PachnerProposal` values before dry-run or mutation
- **No automatic property preservation**: The Delaunay property is **not** maintained automatically
- **Reversible**: Each forward move has a corresponding inverse
- **Geometric validation**: Flips check for degeneracy and manifold preservation
- **Flexible**: Can be used to build custom repair or optimization algorithms

### Proposal Provenance

`PachnerMove` is a raw detached request. It can be stored, randomized, or queued,
but it is not proof that its handles are still live or that they came from the
target triangulation. `propose_pachner(...)` is the raw-to-provenanced
boundary: it validates the local move preconditions, then stamps the resulting
`PachnerProposal` with the current topology owner and structural generation
while carrying the proven feasibility report inward.

Two runtime-only TDS primitives provide that provenance:

- `TopologyOwnerId` is an opaque identity for one live topology owner. Ordinary
  clones and deserialization get fresh identities, while internal rollback
  snapshots preserve identity so failure-atomic mutation paths can restore the
  same owner.
- The topology generation increments on structural mutation. It is an
  invalidation stamp for caches, proposals, and detached topology artifacts; it
  is not serialized.

`PachnerProposal::can_attempt_on(...)` and `PachnerProposal::attempt_on(...)`
are the dry-run and mutation paths. They reject proposals from another owner
with `FlipError::WrongTopologyOwner` and proposals from an older generation with
`FlipError::StaleTopologyProposal` before interpreting runtime-local keys.
`can_attempt_on(...)` returns the feasibility proof stored in the proposal after
that provenance check; `attempt_on(...)` still revalidates through the selected
primitive mutation path before changing topology.

This design supports future concurrent proposal workflows: worker threads can
compute or filter candidate moves against an immutable snapshot, then a
coordinator can attempt selected proposals against the canonical owner and treat
losing stale proposals as typed, expected failures. It does not by itself make
topology mutation concurrent; shared mutable access still needs an explicit
synchronization or transaction design.

### Important Caveats

⚠️ **The Pachner Move API does not preserve the Delaunay property automatically.**

After applying flips, you should:

1. Manually verify the Delaunay property if needed:

   ```rust
   assert!(dt.as_triangulation().validate_embedding().is_ok()); // Check Level 4 (faithful embedding)
   assert!(dt.is_valid_delaunay().is_ok()); // Check Level 5 (Delaunay property)
   ```

2. Consider running a repair pass if you need the Delaunay property again (requires `K: ExactPredicates`):
   - `dt.repair_delaunay_with_flips()` (flip-based repair)
   - `dt.repair_delaunay_with_flips_advanced(DelaunayRepairHeuristicConfig::default())` (includes a heuristic rebuild fallback)

## Combining Both APIs

You can mix both APIs in the same workflow:

```rust
use delaunay::prelude::construction::{
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, vertex,
};
use delaunay::prelude::geometry::CoordinateConversionError;
use delaunay::prelude::insertion::InsertionError;
use delaunay::prelude::pachner::{FacetHandle, FlipError, PachnerMove, PachnerMoves};

#[derive(Debug, thiserror::Error)]
enum ExampleError {
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    Insertion(#[from] InsertionError),
    #[error(transparent)]
    Flip(#[from] FlipError),
    #[error(transparent)]
    Coordinate(#[from] CoordinateConversionError),
}

fn main() -> Result<(), ExampleError> {
    // 1. Build initial triangulation (Builder API)
    let vertices = vec![
        vertex![0.0, 0.0, 0.0]?,
        vertex![1.0, 0.0, 0.0]?,
        vertex![0.0, 1.0, 0.0]?,
        vertex![0.0, 0.0, 1.0]?,
    ];
    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build()?;

    // 2. Add vertices using Builder API (maintains Delaunay)
    dt.insert_vertex(vertex![0.5, 0.5, 0.5]?)?;

    // 3. Make custom topology edits (Pachner Move API)
    let facet = /* ... */;
    dt.propose_pachner(PachnerMove::K2 { facet })?
        .attempt_on(&mut dt)?;

    // 4. Verify Delaunay property if needed
    if let Err(e) = dt.is_valid_delaunay() {
        eprintln!("Warning: Delaunay property violated after manual edit: {}", e);
        // Optionally restore using Builder API or custom repair
    }
    Ok(())
}
```

## Validation and Guarantees

Both APIs work with the same validation framework but have different guarantees:

### Builder API Guarantees

- ✅ Maintains **structural invariants** (Level 1-2)
- ✅ Maintains **manifold topology** (Level 3, controlled by `TopologyGuarantee`)
- ✅ Designed to maintain **faithful embedding** (Level 4) and **Delaunay property** (Level 5)
- ✅ Fails gracefully if invariants cannot be maintained

### Pachner Move API Guarantees

- ✅ Maintains **structural invariants** (Level 1-2)
- ✅ Checks **geometric degeneracy** (prevents degenerate flips)
- ⚠️ Does **not** automatically maintain Delaunay property
- ⚠️ User is responsible for property preservation

### Validation Levels

Use the appropriate validation level for your needs:

```rust
// Level 2: Structural only (fast)
assert!(dt.tds().is_valid().is_ok());

// Level 3: + Manifold topology
assert!(dt.as_triangulation().is_valid_topology().is_ok());

// Level 4: + Faithful embedding
assert!(dt.as_triangulation().validate_embedding().is_ok());

// Level 5: + Delaunay property (most comprehensive)
assert!(dt.is_valid_delaunay().is_ok());

// Full diagnostic report
let report = dt.validation_report();
```

## Implementation Details

### Internal Organization

- **Builder API**: Implemented in `delaunay::construction`, `delaunay::builder`,
  and `core::algorithms::incremental_insertion`
- **Pachner Move API**: Implemented in `delaunay::pachner` over the primitive
  `delaunay::flips` trait and `core::algorithms::flips` internals
- **Low-level primitives**: Context builders and flip application functions are `pub(crate)` in `core::algorithms::flips`

### Borrowed Views, Handles, Snapshots, And Rollback State

Topology APIs use names to make ownership visible:

- `*View` values borrow the canonical owner or are lifetime-bound to it, so they
  cannot outlive the storage they observe. Examples include `FacetView<'tds>`,
  `EdgeView<'tds>`, `RidgeView<'tds>`, `RidgeLinkView<'tds>`,
  `IncidenceView<'tds>`, `EdgeIndex<'tds>`, `SimplexNeighborIndex<'tds>`,
  and `TriangulationAdjacency<'tds>`.
- Borrowed slices over canonical storage follow the same rule. For example,
  `Tds::simplex_vertices(simplex_key)` validates the key relation, then returns
  the simplex's stored `&[VertexKey]` instead of copying detached keys into a
  buffer.
- `*Handle` and `*Key` values are detached, copyable runtime references. They
  may be queued, stored, or returned from snapshots, but callers must validate
  them against a live owner before reading through them. Examples include
  `VertexKey`, `SimplexKey`, `FacetHandle`, `RidgeHandle`, `EdgeKey`, and
  `TriangleHandle`.
- Proof-bearing runtime candidates such as `RidgeCandidate<D>` may validate
  local arity, uniqueness, and canonical ordering without borrowing an owner,
  but they are still detached storage-local values. Convert them to
  `RidgeQuery<'tds>` before asking live-TDS questions that may have an empty
  answer, or to `RidgeView<'tds>` when the API requires an existing ridge.
  `RidgeView` construction proves the candidate vertices are live and have a
  non-empty incident simplex star.
- Toroidal covering-space identities such as `LiftedVertexId` and
  `LiftedLinkEdge` live under `topology::spaces::toroidal`. They are runtime
  graph identities, not TDS storage entries or durable IDs. They preserve
  periodic image identity for link traversal and validation; collapsing them to
  bare `VertexKey`s is an explicit quotient-space operation.
- Owned snapshots are allowed only when the data must cross a persistence,
  detached-analysis, or cache boundary. `TdsSnapshot`/`RawTdsSnapshot` are the
  durable UUID persistence boundary. `ConvexHull` is a logically immutable hull
  snapshot that stores `FacetHandle`s, while `ConvexHull::try_facets(triangulation)`
  returns borrowed `FacetView` values and `ConvexHull::facet_handles()` exposes
  the detached handles explicitly.
- Transactional rollback state may own cloned topology or exact mutation
  records while an operation is in flight. `Tds::clone_for_rollback`,
  `Tds::clone_from_for_rollback`, `SimplexIncidenceRemoval`, and flip trial
  workspaces are rollback state, not long-lived public views. Replacing
  full-TDS clone rollback with a journaled or localized design remains tracked
  by #364.

### Simplex-Local Incidence Query Vocabulary

The public incidence-query surface names topology by simplex dimension, not by
one downstream move type:

| Concept | Simplex dimension | Current public shape |
|---|---:|---|
| Vertex | 0 | `VertexKey`, `adjacent_simplices(vertex)` |
| Edge | 1 | `EdgeKey`, `EdgeView`, `incident_edges(vertex)` |
| Ridge | `D - 2` | `RidgeCandidate<D>`, `RidgeQuery<'tds>`, `RidgeView<'tds>` |
| Facet | `D - 1` | `FacetHandle`, `FacetView<'tds>`, `FacetToSimplicesIndex<'tds, ...>` |
| Cell | `D` | `SimplexKey`, `Simplex<V, D>` |

In 2D, an edge is also a cell facet. The first public edge-to-facet bridge is
therefore 2D-specific:

```rust
dt.try_incident_facets_to_edge_2d(edge)
dt.try_interior_facet_for_edge_2d(edge)
```

`try_incident_facets_to_edge_2d` parses the detached edge key against the
current TDS and returns the current simplex-local facet handles for that edge:
one handle for a boundary edge and two for an interior edge in a valid 2D PL
manifold. `try_interior_facet_for_edge_2d` returns one of those handles only
when the edge has exactly two incident 2D facets, making it suitable for
consumer code that needs a `FacetHandle` for a 2D k=2 local move. On
deliberately invalid low-level topology, non-manifold edge multiplicity is
visible through `try_incident_facets_to_edge_2d`; the narrower
`try_interior_facet_for_edge_2d` still returns `Ok(None)` because the edge is not
a two-sided 2D move support.

These queries are read-only and do not expose a mutable cache. Implementations
may use neighbor walks, maintained TDS incidence, or lifetime-bound derived
indexes internally, but the public contract is stable: detached `*Key` and
`*Handle` inputs are revalidated against the current live owner. Stale keys and
corrupted incidence metadata return typed parse errors rather than being
silently conflated with empty topology. Higher-dimensional incidence should
generalize through simplex-key and ridge/facet/cell vocabulary instead of
treating edge-to-facet as universal.

Runtime generation or identity checks remain appropriate for detached handles,
owned snapshots, serialization boundaries, persistent performance caches, and
tests that intentionally construct inconsistent topology. They should not be
used as a substitute for lifetimes when a value is truly a view over live
canonical storage.

Algorithms follow the same phase split. Read-only traversal, classification,
and validation should work through borrowed views or lifetime-bound indexes
where practical. Mutating topology APIs should take `&mut Tds`/`&mut
Triangulation` directly, or execute behind a transaction guard that holds that
mutable borrow for the mutation or rollback window. Handles and keys may appear
inside that guard as short-lived, validated commit identifiers; they are not
proof that topology still exists by themselves. Keep views in lexical scopes
that end before the mutation so Rust enforces both existence and mutable versus
immutable access.

### Design Rationale

The separation serves several purposes:

1. **Clear contracts**: Builder API guarantees Delaunay property; Pachner Move API does not
2. **Safety**: Low-level flip primitives are not exposed to prevent accidental misuse
3. **Flexibility**: Pachner Move API enables research and custom algorithms without restricting the design
4. **Documentation**: Clear distinction between "construction" and "manipulation" workflows

## Examples

See the following examples for practical demonstrations:

- `examples/topology_editing.rs` - 2D+3D example showing both APIs
- `examples/triangulation_and_hull.rs` - 3D/4D Builder API, traversal, and convex hull queries
- `examples/delaunayize_repair.rs` - Delaunayize workflow (2D/3D/4D, flip-then-repair, custom config)

## Delaunayize Workflow

The `delaunay::delaunayize` module provides a single entrypoint for the
common "repair topology then restore Delaunay" workflow:

```rust
use delaunay::prelude::construction::{
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, vertex,
};
use delaunay::prelude::delaunayize::{
    DelaunayizeConfig, DelaunayizeError, delaunayize_by_flips,
};
use delaunay::prelude::geometry::CoordinateConversionError;

#[derive(Debug, thiserror::Error)]
enum ExampleError {
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    Delaunayize(#[from] DelaunayizeError),
    #[error(transparent)]
    Coordinate(#[from] CoordinateConversionError),
}

fn main() -> Result<(), ExampleError> {
    let vertices = vec![
        vertex![0.0, 0.0, 0.0]?,
        vertex![1.0, 0.0, 0.0]?,
        vertex![0.0, 1.0, 0.0]?,
        vertex![0.0, 0.0, 1.0]?,
    ];
    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build()?;

    let outcome = delaunayize_by_flips(&mut dt, DelaunayizeConfig::default())?;
    assert!(outcome.topology_repair.succeeded);
    Ok(())
}
```

### Steps

1. **PL-manifold topology repair** — bounded deterministic removal of simplices
   that cause facet over-sharing (codimension-1 facet degree > 2).
2. **Delaunay flip repair** — k=2/k=3 bistellar flips to restore the
   empty-circumsphere property.
3. **Optional fallback rebuild** — rebuilds from the vertex set when both
   repair passes fail (`DelaunayizeConfig { fallback_rebuild: true, .. }`).
   If a failed topology repair is recovered by fallback rebuild,
   `outcome.topology_repair.succeeded` remains `false`; use
   `outcome.used_fallback_rebuild` to distinguish successful rebuild recovery
   from direct repair success.

### Configuration

`DelaunayizeConfig` controls:

- `topology_max_iterations` (default 64): max repair iterations.
- `topology_max_simplices_removed` (default 10,000): max simplices removed.
- `fallback_rebuild` (default false): rebuild from vertices on failure,
  restoring simplex data for rebuilt simplices whose sorted vertex UUID set still
  matches exactly one original simplex.
- `delaunay_max_flips` (default `None`): optional per-attempt flip budget.

### Data Preservation

`PlManifoldRepairStats` carries `removed_simplices` and `removed_vertices`
(identified by UUID) so callers can recover user data from entities removed
during topology repair. The fallback rebuild path also preserves simplex payloads
when a rebuilt simplex has the same vertex UUID set as exactly one original simplex;
changed or ambiguous simplices receive no payload.

### Explicitly Deferred

- Dedicated targeted repair stages for boundary-ridge multiplicity,
  ridge-link manifoldness, and vertex-link manifoldness (#304).

## Further Reading

- **Bistellar flip theory**: See `REFERENCES.md` for academic citations
- **Validation framework**: See `docs/validation.md` for detailed validation guide
- **Invariant rationale**: See [`invariants.md`](invariants.md) for theory and implementation pointers
- **Topology analysis**: See `docs/topology.md` for topological concepts
- **API implementation**: See `delaunay::pachner` and `delaunay::flips` module documentation
