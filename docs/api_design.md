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

2. **Edit API** (`prelude::triangulation::flips::BistellarFlips` trait)
   - Low-level topology editing via bistellar (Pachner) flips
   - Explicit control over individual topology operations
   - Does **not** automatically restore the Delaunay property
   - Designed for topology manipulation, research, and custom algorithms

## When to Use Each API

### Use the Builder API when

- Building a Delaunay triangulation from a set of points
- Adding/removing vertices while maintaining the Delaunay property
- You need automatic geometric property preservation
- Working with standard computational geometry workflows

**Example use cases:**

- Computing convex hulls
- Nearest-neighbor queries
- Voronoi diagram construction
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

The Builder API is exposed directly on `DelaunayTriangulation`:

```rust
use delaunay::prelude::triangulation::*;

// Construction from a batch of vertices
let vertices = vec![
    vertex!([0.0, 0.0, 0.0]),
    vertex!([1.0, 0.0, 0.0]),
    vertex!([0.0, 1.0, 0.0]),
    vertex!([0.0, 0.0, 1.0]),
];
let mut dt: DelaunayTriangulation<_, (), (), 3> = 
    DelaunayTriangulation::new(&vertices).unwrap();

// Incremental insertion (maintains Delaunay property)
let new_vertex = vertex!([0.5, 0.5, 0.5]);
dt.insert(new_vertex).unwrap();

// Vertex removal (maintains Delaunay property via fan retriangulation)
let vertex_key = /* ... */;
dt.remove_vertex(vertex_key).unwrap();
```

### Key Characteristics

- **Automatic property preservation**: The Delaunay empty-circumsphere property is maintained automatically
- **Cavity-based insertion**: New vertices are inserted by identifying conflicting cells, removing them, and filling the cavity
- **Fan retriangulation**: Vertex removal uses fan-based retriangulation of the vertex star
- **Error handling**: Operations fail gracefully if they would violate invariants (see
  [`invariants.md`](invariants.md)).
- **Validation**: Use `ValidationPolicy` to control automatic topology validation during construction

## Edit API Reference

The Edit API is exposed through the `BistellarFlips` trait in `prelude::triangulation::flips`:

```rust
use delaunay::prelude::triangulation::*;
use delaunay::prelude::triangulation::flips::*;

// Start with a valid triangulation
let vertices = vec![
    vertex!([0.0, 0.0, 0.0]),
    vertex!([1.0, 0.0, 0.0]),
    vertex!([0.0, 1.0, 0.0]),
    vertex!([0.0, 0.0, 1.0]),
];
let mut dt: DelaunayTriangulation<_, (), (), 3> = 
    DelaunayTriangulation::new(&vertices).unwrap();

// k=1 move: Insert a vertex into a cell (splits cell into D+1 cells)
let cell_key = dt.cells().next().unwrap().0;
let info = dt.flip_k1_insert(cell_key, vertex!([0.25, 0.25, 0.25])).unwrap();

// k=1 inverse: Remove a vertex (collapses its star)
let vertex_key = info.inserted_face_vertices[0];
dt.flip_k1_remove(vertex_key).unwrap();

// k=2 move: Flip a facet (2 cells ↔ D cells)
let facet = /* FacetHandle */;
let info = dt.flip_k2(facet).unwrap();

// k=2 inverse: Flip from an edge star (D cells ↔ 2 cells)
let edge = EdgeKey::new(info.inserted_face_vertices[0], info.inserted_face_vertices[1]);
dt.flip_k2_inverse_from_edge(edge).unwrap();

// k=3 move: Flip a ridge (3 cells ↔ D-1 cells, requires D ≥ 3)
let ridge = /* RidgeHandle */;
let info = dt.flip_k3(ridge).unwrap();

// k=3 inverse: Flip from a triangle star (D-1 cells ↔ 3 cells)
let triangle = TriangleHandle::new(
    info.inserted_face_vertices[0],
    info.inserted_face_vertices[1],
    info.inserted_face_vertices[2],
);
dt.flip_k3_inverse_from_triangle(triangle).unwrap();
```

### Available Flip Operations

#### k=1 Moves (Cell Split/Merge)

- **Forward (`flip_k1_insert`)**: Insert a vertex into a cell, splitting it into D+1 cells
  - Valid for D ≥ 1
  - Replaces 1 cell with D+1 cells
  - Removed face: the entire cell (D-simplex)
  - Inserted face: the new vertex (0-simplex)

- **Inverse (`flip_k1_remove`)**: Remove a vertex, collapsing its star
  - Requires the vertex star to be collapsible (star of D+1 cells forming a ball)
  - Replaces D+1 cells with 1 cell

#### k=2 Moves (Facet Flip)

- **Forward (`flip_k2`)**: Flip a facet shared by 2 cells
  - Valid for D ≥ 2
  - Replaces 2 cells with D cells
  - Removed face: the shared facet ((D-1)-simplex)
  - Inserted face: an edge (1-simplex)

- **Inverse (`flip_k2_inverse_from_edge`)**: Flip from an edge star
  - Requires an edge with star of D cells
  - Replaces D cells with 2 cells

#### k=3 Moves (Ridge Flip)

- **Forward (`flip_k3`)**: Flip a ridge
  - Valid for D ≥ 3
  - Replaces 3 cells with D-1 cells
  - Removed face: a ridge ((D-2)-simplex)
  - Inserted face: a triangle (2-simplex)

- **Inverse (`flip_k3_inverse_from_triangle`)**: Flip from a triangle star
  - Requires a triangle with star of D-1 cells
  - Replaces D-1 cells with 3 cells

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
   dt.is_valid().unwrap();  // Check Level 4 (Delaunay property)
   ```

2. Consider running a repair pass if you need the Delaunay property again:
   - `dt.repair_delaunay_with_flips()` (flip-based repair)
   - `dt.repair_delaunay_with_flips_advanced(DelaunayRepairHeuristicConfig::default())` (includes a heuristic rebuild fallback)

## Combining Both APIs

You can mix both APIs in the same workflow:

```rust
use delaunay::prelude::triangulation::*;
use delaunay::prelude::triangulation::flips::*;

// 1. Build initial triangulation (Builder API)
let vertices = vec![
    vertex!([0.0, 0.0, 0.0]),
    vertex!([1.0, 0.0, 0.0]),
    vertex!([0.0, 1.0, 0.0]),
    vertex!([0.0, 0.0, 1.0]),
];
let mut dt: DelaunayTriangulation<_, (), (), 3> = 
    DelaunayTriangulation::new(&vertices).unwrap();

// 2. Add vertices using Builder API (maintains Delaunay)
dt.insert(vertex!([0.5, 0.5, 0.5])).unwrap();

// 3. Make custom topology edits (Edit API)
let facet = /* ... */;
dt.flip_k2(facet).unwrap();

// 4. Verify Delaunay property if needed
if let Err(e) = dt.is_valid() {
    eprintln!("Warning: Delaunay property violated after manual edit: {}", e);
    // Optionally restore using Builder API or custom repair
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
dt.tds().is_valid().unwrap();

// Level 3: + Manifold topology
dt.as_triangulation().is_valid().unwrap();

// Level 4: + Delaunay property (most comprehensive)
dt.is_valid().unwrap();

// Full diagnostic report
let report = dt.validation_report();
```

## Implementation Details

### Internal Organization

- **Builder API**: Implemented in `core::delaunay_triangulation` and `core::algorithms::incremental_insertion`
- **Edit API**: Implemented in `triangulation::flips` (public trait) and `core::algorithms::flips` (internal implementation)
- **Low-level primitives**: Context builders and flip application functions are `pub(crate)` in `core::algorithms::flips`

### Design Rationale

The separation serves several purposes:

1. **Clear contracts**: Builder API guarantees Delaunay property; Edit API does not
2. **Safety**: Low-level flip primitives are not exposed to prevent accidental misuse
3. **Flexibility**: Edit API enables research and custom algorithms without restricting the design
4. **Documentation**: Clear distinction between "construction" and "manipulation" workflows

## Examples

See the following examples for practical demonstrations:

- `examples/topology_editing_2d_3d.rs` - 2D+3D example showing both APIs
- `examples/pachner_roundtrip_4d.rs` - Advanced 4D example with all flip types
- `examples/triangulation_3d_20_points.rs` - Builder API usage for construction

## Further Reading

- **Bistellar flip theory**: See `REFERENCES.md` for academic citations
- **Validation framework**: See `docs/validation.md` for detailed validation guide
- **Invariant rationale**: See [`invariants.md`](invariants.md) for theory and implementation pointers
- **Topology analysis**: See `docs/topology.md` for topological concepts
- **API implementation**: See `triangulation::flips` module documentation
