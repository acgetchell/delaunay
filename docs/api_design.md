# API Design: Two-Track Approach

This document explains the dual API design for working with Delaunay triangulations:
the **Builder API** for constructing and maintaining Delaunay triangulations, and
the **Edit API** for explicit topology editing via bistellar flips.

## Overview

The library provides two distinct APIs for different use cases:

1. **Builder API** (`DelaunayTriangulation::insert` / `::remove_vertex`)
   - High-level construction and maintenance of Delaunay triangulations
   - Automatically maintains the Delaunay property (empty circumsphere)
   - Designed for building triangulations from point sets
   - Uses cavity-based insertion and fan retriangulation

2. **Edit API** (`prelude::flips::BistellarFlips` trait)
   - Low-level topology editing via bistellar (Pachner) flips
   - Explicit control over individual topology operations
   - Does **not** automatically restore the Delaunay property
   - Designed for topology manipulation, research, and custom algorithms

Examples that derive `thiserror::Error` assume the example crate includes
`thiserror`; run `cargo add thiserror` alongside `delaunay` when copying those
snippets into an application.

## When to Use Each API

### Use the Builder API when

- Building a Delaunay triangulation from a set of points
- Adding/removing vertices while maintaining the Delaunay property
- You need automatic geometric property preservation
- Working with standard computational geometry workflows

**Example use cases:**

- Computing convex hulls
- Nearest-neighbor queries
- Supplying triangulation data to downstream Voronoi/dual-simplex analysis
  (the crate does not yet extract Voronoi diagrams directly)
- Mesh generation
- Scientific simulations requiring Delaunay meshes

### Use the Edit API when

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
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, Vertex,
};
use delaunay::prelude::geometry::CoordinateConversionError;
use delaunay::prelude::insertion::InsertionError;
use delaunay::prelude::tds::InvariantError;

#[derive(Debug, thiserror::Error)]
enum ExampleError {
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    Insertion(#[from] InsertionError),
    #[error(transparent)]
    Topology(#[from] InvariantError),
    #[error(transparent)]
    Coordinate(#[from] CoordinateConversionError),
}

fn main() -> Result<(), ExampleError> {
    // Simple construction from vertices (Euclidean space, default options)
    let vertices = vec![
        Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
        Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
        Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
        Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    ];
    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;

    // Incremental insertion (maintains Delaunay property)
    let new_vertex = Vertex::<(), _>::try_new([0.5, 0.5, 0.5])?;
    dt.insert(new_vertex)?;

    // Vertex removal (topology-preserving, with automatic repair when enabled)
    if let Some((vertex_key, _)) = dt.vertices().next() {
        dt.remove_vertex(vertex_key)?;
    }
    Ok(())
}
```

### Advanced Construction: `DelaunayTriangulationBuilder`

For advanced configuration (toroidal topology, custom validation policies, etc.),
use `DelaunayTriangulationBuilder`:

```rust
use delaunay::prelude::construction::{
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, TopologyGuarantee,
    Vertex,
};
use delaunay::prelude::geometry::CoordinateConversionError;
use delaunay::prelude::insertion::InsertionError;
use delaunay::prelude::validation::ValidationPolicy;

#[derive(Debug, thiserror::Error)]
enum ExampleError {
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    Insertion(#[from] InsertionError),
    #[error(transparent)]
    Coordinate(#[from] CoordinateConversionError),
}

fn main() -> Result<(), ExampleError> {
    // Canonicalized toroidal triangulation in 2D
    let vertices = vec![
        Vertex::<(), _>::try_new([0.1, 0.1])?,
        Vertex::<(), _>::try_new([0.9, 0.9])?,
        Vertex::<(), _>::try_new([0.5, 0.5])?,
    ];

    let mut dt = DelaunayTriangulationBuilder::new(&vertices)
        .try_canonicalized_toroidal([1.0, 1.0])
        ? // Canonicalized toroidal construction
        .topology_guarantee(TopologyGuarantee::PLManifoldStrict)
        .build::<()>()?;

    dt.set_validation_policy(ValidationPolicy::Always);

    // Works like any other DelaunayTriangulation
    dt.insert(Vertex::<(), _>::try_new([0.25, 0.75])?)?;
    Ok(())
}
```

**When to use the Builder:**

- **Toroidal construction**: Use `.try_toroidal()` for periodic image-point construction or
  `.try_canonicalized_toroidal()` for canonicalized construction with explicit domain periods.
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
  empty-circumsphere property; removal runs flip-based repair when the active
  `DelaunayRepairPolicy` permits it
- **Cavity-based insertion**: New vertices are inserted by identifying conflicting simplices, removing them, and filling the cavity
- **Transactional vertex removal**: Vertex removal uses an inverse k=1 fast path
  when possible and fan-based retriangulation otherwise. If post-removal
  Delaunay repair or orientation canonicalization fails, the triangulation and
  internal caches are restored to their pre-removal state.
- **Auxiliary data**: Vertices and simplices carry optional user data (`U` / `V`). Read via `vertex.data()` /
  `simplex.data()`, write via `dt.set_vertex_data(key, data)` / `dt.set_simplex_data(key, data)` (O(1),
  invariant-preserving). See [`workflows.md`](workflows.md) for examples.
- **Error handling**: Operations fail gracefully if they would violate invariants (see
  [`invariants.md`](invariants.md)). Mutating operations that invoke repair use
  typed repair diagnostics where available, for example
  `RepairOperationFailed { operation, source }`.
- **Validation**: The active `ValidationPolicy` (set with
  `dt.try_set_validation_policy(...)` or `dt.set_validation_policy(...)`) governs automatic topology validation for
  subsequent construction/modification operations

## Edit API Reference

The Edit API is exposed through the `BistellarFlips` trait in `prelude::flips`:

```rust
use delaunay::prelude::construction::{
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, Vertex,
};
use delaunay::prelude::flips::*;
use delaunay::prelude::geometry::CoordinateConversionError;

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
        Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
        Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
        Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
        Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    ];
    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;

    // k=1 move: Insert a vertex into a simplex (splits simplex into D+1 simplices)
    let Some((simplex_key, _)) = dt.simplices().next() else {
        return Ok(());
    };
    let info = dt.flip_k1_insert(simplex_key, Vertex::<(), _>::try_new([0.25, 0.25, 0.25])?)?;

    // k=1 inverse: Remove a vertex (collapses its star)
    let vertex_key = info.inserted_face_vertices[0];
    dt.flip_k1_remove(vertex_key)?;

    // k=2 move: Flip a facet (2 simplices ↔ D simplices)
    let facet = /* FacetHandle */;
    let info = dt.flip_k2(facet)?;

    // k=2 inverse: Flip from an edge star (D simplices ↔ 2 simplices)
    let edge = EdgeKey::try_new(info.inserted_face_vertices[0], info.inserted_face_vertices[1])?;
    dt.flip_k2_inverse_from_edge(edge)?;

    // k=3 move: Flip a ridge (3 simplices ↔ D-1 simplices, requires D ≥ 3)
    let ridge = /* RidgeHandle */;
    let info = dt.flip_k3(ridge)?;

    // k=3 inverse: Flip from a triangle star (D-1 simplices ↔ 3 simplices)
    let triangle = TriangleHandle::try_new(
        info.inserted_face_vertices[0],
        info.inserted_face_vertices[1],
        info.inserted_face_vertices[2],
    )?;
    dt.flip_k3_inverse_from_triangle(triangle)?;
    Ok(())
}
```

### Available Flip Operations

#### k=1 Moves (Simplex Split/Merge)

- **Forward (`flip_k1_insert`)**: Insert a vertex into a simplex, splitting it into D+1 simplices
  - Valid for D ≥ 1
  - Replaces 1 simplex with D+1 simplices
  - Removed face: the entire simplex (D-simplex)
  - Inserted face: the new vertex (0-simplex)

- **Inverse (`flip_k1_remove`)**: Remove a vertex, collapsing its star
  - Requires the vertex star to be collapsible (star of D+1 simplices forming a ball)
  - Replaces D+1 simplices with 1 simplex

#### k=2 Moves (Facet Flip)

- **Forward (`flip_k2`)**: Flip a facet shared by 2 simplices
  - Valid for D ≥ 2
  - Replaces 2 simplices with D simplices
  - Removed face: the shared facet ((D-1)-simplex)
  - Inserted face: an edge (1-simplex)

- **Inverse (`flip_k2_inverse_from_edge`)**: Flip from an edge star
  - Requires an edge with star of D simplices
  - Replaces D simplices with 2 simplices

#### k=3 Moves (Ridge Flip)

- **Forward (`flip_k3`)**: Flip a ridge
  - Valid for D ≥ 3
  - Replaces 3 simplices with D-1 simplices
  - Removed face: a ridge ((D-2)-simplex)
  - Inserted face: a triangle (2-simplex)

- **Inverse (`flip_k3_inverse_from_triangle`)**: Flip from a triangle star
  - Requires a triangle with star of D-1 simplices
  - Replaces D-1 simplices with 3 simplices

### Key Characteristics

- **Explicit control**: You specify exactly which flip to perform
- **No automatic property preservation**: The Delaunay property is **not** maintained automatically
- **Reversible**: Each forward move has a corresponding inverse
- **Geometric validation**: Flips check for degeneracy and manifold preservation
- **Flexible**: Can be used to build custom repair or optimization algorithms

### Important Caveats

⚠️ **The Edit API does not preserve the Delaunay property automatically.**

After applying flips, you should:

1. Manually verify the Delaunay property if needed:

   ```rust
   assert!(dt.is_valid().is_ok()); // Check Level 4 (Delaunay property)
   ```

2. Consider running a repair pass if you need the Delaunay property again (requires `K: ExactPredicates`):
   - `dt.repair_delaunay_with_flips()` (flip-based repair)
   - `dt.repair_delaunay_with_flips_advanced(DelaunayRepairHeuristicConfig::default())` (includes a heuristic rebuild fallback)

## Combining Both APIs

You can mix both APIs in the same workflow:

```rust
use delaunay::prelude::construction::{
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, Vertex,
};
use delaunay::prelude::flips::*;
use delaunay::prelude::geometry::CoordinateConversionError;
use delaunay::prelude::insertion::InsertionError;

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
        Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
        Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
        Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
        Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    ];
    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;

    // 2. Add vertices using Builder API (maintains Delaunay)
    dt.insert(Vertex::<(), _>::try_new([0.5, 0.5, 0.5])?)?;

    // 3. Make custom topology edits (Edit API)
    let facet = /* ... */;
    dt.flip_k2(facet)?;

    // 4. Verify Delaunay property if needed
    if let Err(e) = dt.is_valid() {
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
- ✅ Designed to maintain **Delaunay property** (Level 4)
- ✅ Fails gracefully if invariants cannot be maintained

### Edit API Guarantees

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
assert!(dt.as_triangulation().is_valid().is_ok());

// Level 4: + Delaunay property (most comprehensive)
assert!(dt.is_valid().is_ok());

// Full diagnostic report
let report = dt.validation_report();
```

## Implementation Details

### Internal Organization

- **Builder API**: Implemented in `delaunay::construction`, `delaunay::builder`,
  and `core::algorithms::incremental_insertion`
- **Edit API**: Implemented in `delaunay::flips` (public trait) and `core::algorithms::flips` (internal implementation)
- **Low-level primitives**: Context builders and flip application functions are `pub(crate)` in `core::algorithms::flips`

### Design Rationale

The separation serves several purposes:

1. **Clear contracts**: Builder API guarantees Delaunay property; Edit API does not
2. **Safety**: Low-level flip primitives are not exposed to prevent accidental misuse
3. **Flexibility**: Edit API enables research and custom algorithms without restricting the design
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
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, Vertex,
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
        Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
        Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
        Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
        Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    ];
    let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;

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
- **API implementation**: See `delaunay::flips` module documentation
