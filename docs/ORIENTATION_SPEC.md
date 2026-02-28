# Coherent Orientation Technical Specification

## Overview

This document specifies how to implement coherent orientation as a first-class invariant
in the TDS layer, following CGAL's approach of maintaining orientation through vertex ordering.

## Background: Simplicial Orientation

### Mathematical Foundation

A D-dimensional simplex with vertices v₀, v₁, ..., vₐ has an **orientation** determined by the sign of the determinant:

```text
│ x₀   y₀   z₀  ...  1 │
│ x₁   y₁   z₁  ...  1 │
│ x₂   y₂   z₂  ...  1 │
│ ...  ...  ... ...  ...│
│ xₐ   yₐ   zₐ  ...  1 │
```

- **Positive orientation**: determinant > 0
- **Negative orientation**: determinant < 0
- **Degenerate**: determinant ≈ 0 (points are coplanar/collinear)

### Permutation Parity

Swapping any two vertices **reverses** the orientation:

- Even permutation (0, 2, 4, ... swaps) → same orientation
- Odd permutation (1, 3, 5, ... swaps) → opposite orientation

## CGAL's Approach: Vertex Ordering

CGAL maintains orientation by **ordering vertices in cells**:

1. **Convention**: All cells store vertices in **positive orientation order**
2. **Facet orientation**: When two cells C₁ and C₂ share a facet F:
   - Extract vertices of F from C₁'s vertex list (in C₁'s order)
   - Extract vertices of F from C₂'s vertex list (in C₂'s order)
   - The two orderings must be **opposite parity** (differ by odd permutation)
   - This ensures C₁ and C₂ induce opposite orientations on their shared facet

### Example: 2D Triangle Mesh

```text
Cell C₁: vertices [v0, v1, v2] (counterclockwise = positive)
Cell C₂: vertices [v0, v2, v3] (counterclockwise = positive)

Shared edge (facet in 2D): {v0, v2}

From C₁: extract opposite v1 → edge is [v0, v2]
From C₂: extract opposite v3 → edge is [v2, v0] (reversed!)

[v0, v2] vs [v2, v0] differ by 1 swap -> ✓ opposite orientations
```

### Example: 3D Tetrahedral Mesh

```text
Cell C₁: vertices [v0, v1, v2, v3] (positive orientation)
Cell C₂: vertices [v0, v2, v1, v4] (positive orientation)

Shared triangle (facet in 3D): {v0, v1, v2}

From C₁: opposite v3 → facet is [v0, v1, v2]
From C₂: opposite v4 → facet is [v0, v2, v1] (v1 and v2 swapped!)

[v0, v1, v2] vs [v0, v2, v1] differ by 1 swap -> ✓ opposite orientations
```

## Implementation Strategy

### Phase 1: Canonical Vertex Ordering

#### 1.1 Define Canonical Order

We already have `simplex_orientation()` in `src/geometry/predicates.rs` that computes orientation from vertex coordinates.

**Strategy**: When creating a cell, ensure vertices are ordered to produce **positive orientation**.

```rust
// In src/core/triangulation_data_structure.rs

/// Ensure vertices are ordered to produce positive orientation.
/// Returns vertices in canonical (positive orientation) order.
fn canonicalize_vertex_order<T, U, const D: usize>(
    vertices: &[VertexKey],
    tds: &Tds<T, U, V, D>,
) -> Result<Vec<VertexKey>, TdsValidationError>
where
    T: CoordinateScalar,
    U: DataType,
{
    if vertices.len() != D + 1 {
        return Err(TdsValidationError::InconsistentDataStructure { ... });
    }
    
    // Extract points from vertices
    let points: Vec<Point<T, D>> = vertices
        .iter()
        .map(|&vk| tds.get_vertex_by_key(vk).unwrap().point().clone())
        .collect();
    
    // Check orientation
    match simplex_orientation(&points)? {
        Orientation::POSITIVE => Ok(vertices.to_vec()),
        Orientation::NEGATIVE => {
            // Swap first two vertices to flip orientation
            let mut canonical = vertices.to_vec();
            canonical.swap(0, 1);
            Ok(canonical)
        }
        Orientation::DEGENERATE => {
            Err(TdsValidationError::DegenerateCell { ... })
        }
    }
}
```

**Key insight**: Swapping **any two vertices** flips orientation. We conventionally swap vertices 0 and 1.

#### 1.2 Apply During Cell Creation

Update `Tds::insert_cell()` and related methods to canonicalize vertices:

```rust
pub fn insert_cell(&mut self, vertices: Vec<VertexKey>, data: Option<V>) 
    -> Result<CellKey, TdsValidationError>
where
    T: CoordinateScalar,
{
    // Canonicalize vertex order to ensure positive orientation
    let canonical_vertices = canonicalize_vertex_order(&vertices, self)?;
    
    // Create cell with canonical ordering
    let cell = Cell::new(canonical_vertices, data)?;
    // ... rest of insertion logic
}
```

### Phase 2: Validation - Check Coherent Orientation

#### 2.1 Algorithm: `is_coherently_oriented()`

```rust
/// Check if all adjacent cells induce opposite orientations on shared facets.
pub fn is_coherently_oriented(&self) -> bool 
where
    T: CoordinateScalar,
{
    for (cell_key, cell) in self.cells.iter() {
        let cell_vertices = cell.vertices();
        
        // Check each facet (opposite each vertex)
        for i in 0..cell_vertices.len() {
            // Get neighbor opposite vertex i
            let Some(neighbor_key) = cell.neighbor(i).flatten() else {
                continue; // Boundary facet, no neighbor
            };
            
            let neighbor_cell = &self.cells[neighbor_key];
            
            // Extract facet vertices from current cell (excluding vertex i)
            let facet_from_cell: Vec<VertexKey> = cell_vertices.iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, &vk)| vk)
                .collect();
            
            // Find which vertex in neighbor is opposite this facet
            let neighbor_vertices = neighbor_cell.vertices();
            let opposite_in_neighbor = find_opposite_vertex(neighbor_vertices, &facet_from_cell)?;
            
            // Extract facet vertices from neighbor (excluding opposite vertex)
            let facet_from_neighbor: Vec<VertexKey> = neighbor_vertices.iter()
                .enumerate()
                .filter(|(j, _)| *j != opposite_in_neighbor)
                .map(|(_, &vk)| vk)
                .collect();
            
            // Check if orderings have opposite parity
            if !have_opposite_parity(&facet_from_cell, &facet_from_neighbor) {
                return false;
            }
        }
    }
    
    true
}
```

#### 2.2 Helper: Check Permutation Parity

```rust
/// Check if two vertex orderings of the same facet have opposite parity.
/// Returns true if they differ by an odd number of swaps.
fn have_opposite_parity(ordering1: &[VertexKey], ordering2: &[VertexKey]) -> bool {
    assert_eq!(ordering1.len(), ordering2.len());
    
    // Build a mapping from ordering2's vertex keys to their positions
    let pos_map: FastHashMap<VertexKey, usize> = ordering2
        .iter()
        .enumerate()
        .map(|(i, &vk)| (vk, i))
        .collect();
    
    // Compute the permutation signature
    let permutation: Vec<usize> = ordering1
        .iter()
        .map(|&vk| pos_map[&vk])
        .collect();
    
    // Count inversions to determine parity
    let inversion_count = count_inversions(&permutation);
    
    // Odd number of inversions = odd permutation = opposite orientation
    inversion_count % 2 == 1
}

/// Count inversions in a permutation (number of pairs (i,j) where i < j but perm[i] > perm[j])
fn count_inversions(perm: &[usize]) -> usize {
    let mut count = 0;
    for i in 0..perm.len() {
        for j in (i + 1)..perm.len() {
            if perm[i] > perm[j] {
                count += 1;
            }
        }
    }
    count
}
```

**Complexity**: O(D²) per facet, O(D³) per cell, O(N×D³) total for N cells.
For practical D ≤ 5, this is acceptable for validation.

### Phase 3: Integration with Validation Hierarchy

#### 3.1 Add Error Variant

```rust
// In TdsValidationError enum
#[error("Orientation invariant violated: cells {cell1_uuid} and {cell2_uuid} induce same orientation on shared facet")]
OrientationViolation {
    cell1_key: CellKey,
    cell1_uuid: Uuid,
    cell2_key: CellKey,
    cell2_uuid: Uuid,
    facet_vertices: Vec<VertexKey>,
},
```

#### 3.2 Add to InvariantKind

```rust
pub enum InvariantKind {
    // ... existing variants
    /// Coherent combinatorial orientation (adjacent cells induce opposite facet orientations)
    CoherentOrientation,
}
```

#### 3.3 Update Validation Methods

```rust
// In Tds::is_valid()
pub fn is_valid(&self) -> Result<(), TdsValidationError>
where
    T: CoordinateScalar,
{
    // ... existing Level 2 checks ...
    
    // Level 2: Coherent orientation
    if !self.is_coherently_oriented() {
        // Find the violating pair for error reporting
        let (cell1, cell2, facet) = self.find_orientation_violation()?;
        return Err(TdsValidationError::OrientationViolation {
            cell1_key: cell1,
            cell1_uuid: self.cells[cell1].uuid(),
            cell2_key: cell2,
            cell2_uuid: self.cells[cell2].uuid(),
            facet_vertices: facet,
        });
    }
    
    Ok(())
}
```

### Phase 4: Flip Operations Preserve Orientation

#### 4.1 Audit Existing Flips

Check all Pachner moves in `src/core/algorithms/flips.rs`:

- k=1 flip (1→D+1)
- k=2 flip (2→D)
- k=3 flip (3→D-1)
- Inverse flips

#### 4.2 Add Debug Assertions

At entry and exit of each flip:

```rust
pub fn apply_k2_flip<K, U, V, const D: usize>(
    tds: &mut Tds<K::Scalar, U, V, D>,
    // ... params
) -> Result<FlipResult, FlipError>
where
    K: Kernel<D>,
    U: DataType,
    V: DataType,
{
    debug_assert!(
        tds.is_coherently_oriented(),
        "TDS orientation violated BEFORE k=2 flip"
    );
    
    // ... flip implementation ...
    
    debug_assert!(
        tds.is_coherently_oriented(),
        "TDS orientation violated AFTER k=2 flip"
    );
    
    Ok(result)
}
```

#### 4.3 Ensure New Cells Use Canonical Ordering

Flips now apply explicit per-cell orientation handling in `src/core/algorithms/flips.rs`
(`apply_bistellar_flip_with_k`):

1. Build candidate replacement cells
2. Compute orientation for each candidate and swap slots when needed so each new cell has canonical local orientation
3. Insert via `tds.insert_cell_with_mapping(...)` (insertion keeps the provided slot order; it does not canonicalize orientation implicitly)

After rewiring and removing old cells, flips run:

- `tds.normalize_coherent_orientation()` to reconcile residual neighbor-parity inconsistencies
- `debug_assert!(tds.is_coherently_oriented())` as a post-flip invariant check

So a flip must: (a) delete old cells, (b) insert new cells with explicit per-cell
orientation handling before insertion, and (c) run coherence normalization/assertion
after topology updates.

## Testing Strategy

### Unit Tests (`tests/tds_orientation.rs`)

```rust
#[test]
fn test_vertex_ordering_canonicalization_2d() {
    // Create TDS with 2D triangle
    // Test positive and negative input orderings
    // Verify all cells have positive orientation
}

#[test]
fn test_coherent_orientation_2d_mesh() {
    // Build small 2D triangle mesh
    // Verify is_coherently_oriented() returns true
    // Manually flip one cell's vertices
    // Verify is_coherently_oriented() returns false
}

#[test]
fn test_coherent_orientation_3d_tetrahedra() {
    // Build 3D tetrahedral mesh
    // Verify orientation coherence
}

#[test]
fn test_coherent_orientation_4d() {
    // Build 4D simplicial mesh
    // Verify orientation coherence
}

#[test]
fn test_opposite_parity_detection() {
    // Unit test for have_opposite_parity()
    // Test even/odd permutations
}
```

### Property Tests (`tests/proptest_orientation.rs`)

```rust
proptest! {
    #[test]
    fn prop_orientation_preserved_across_flips(
        vertices in generate_random_vertices(10, 4)
    ) {
        let mut dt = DelaunayTriangulation::new(&vertices)?;
        assert!(dt.tds().is_coherently_oriented());
        
        // Perform random flips
        for _ in 0..100 {
            if let Ok(_) = perform_random_flip(&mut dt) {
                assert!(dt.tds().is_coherently_oriented());
            }
        }
    }
}
```

## Performance Considerations

### Overhead Analysis

1. **Cell creation**: O(D) for canonicalization (one orientation test + possible swap)
   - Negligible compared to circumsphere tests during insertion

2. **Validation**: O(N×D³) for full orientation check
   - Only runs on-demand via `validate()`
   - Zero cost in release builds (debug_assert in flips)

3. **Debug assertions**: Zero cost in release builds (compiled out)

### When to Check

- **Eager (debug only)**: `debug_assert!` at flip entry/exit
- **Lazy (on-demand)**: `validate()` / `is_coherently_oriented()`
- **Never in release hot path**: No per-mutation overhead

## Open Questions

### Q1: Should we support both orientations?

**Answer**: No. Choose positive orientation as canonical. Simplifies implementation and matches CGAL.

### Q2: What about degenerate simplices?

**Answer**: Reject during cell creation. `canonicalize_vertex_order()` returns error on `DEGENERATE`.

### Q3: Performance in 5D?

**Answer**:

- Cell creation: O(6) orientation test = acceptable
- Validation: O(N×5³) = O(125N) = acceptable for validation (not hot path)
- Debug assertions compile out in release

## References

- CGAL Triangulation Documentation: <https://doc.cgal.org/latest/Triangulation/index.html>
- Orientation predicates: `src/geometry/predicates.rs::simplex_orientation()`
- Permutation parity: <https://en.wikipedia.org/wiki/Parity_of_a_permutation>
