//! Euler characteristic computation for triangulated spaces.
//!
//! This module implements dimensional-generic Euler characteristic calculation
//! using the formula: χ = Σ(-1)^k · `f_k` where `f_k` is the number of `k`-simplices.
//!
//! # Examples
//!
//! ```rust
//! use delaunay::prelude::*;
//! use delaunay::topology::characteristics::euler;
//!
//! let vertices = vec![
//!     vertex!([0.0, 0.0, 0.0]),
//!     vertex!([1.0, 0.0, 0.0]),
//!     vertex!([0.0, 1.0, 0.0]),
//!     vertex!([0.0, 0.0, 1.0]),
//! ];
//! let dt = DelaunayTriangulation::new(&vertices).unwrap();
//!
//! let counts = euler::count_simplices(dt.tds()).unwrap();
//! let chi = euler::euler_characteristic(&counts);
//! assert_eq!(chi, 1);  // Single tetrahedron has χ = 1
//! ```

use crate::core::{
    collections::{FacetToCellsMap, FastHashSet, MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer},
    traits::{BoundaryAnalysis, DataType},
    triangulation_data_structure::{Tds, VertexKey},
};
use crate::geometry::traits::coordinate::CoordinateScalar;
use crate::topology::traits::topological_space::TopologyError;

/// Counts of k-simplices for all dimensions 0 ≤ k ≤ D.
///
/// Stores the f-vector (f₀, f₁, ..., `f_D`) where `f_k` is the number of
/// `k`-dimensional simplices:
/// - `f₀` = vertices (`0`-simplices)
/// - `f₁` = edges (`1`-simplices)
/// - `f₂` = triangular faces (`2`-simplices)
/// - `f₃` = tetrahedral cells (`3`-simplices)
/// - `f_D` = `D`-dimensional cells
///
/// In the topology literature this is commonly called the **f-vector**.
///
/// # Examples
///
/// ```rust
/// use delaunay::topology::characteristics::euler::FVector;
///
/// // 2D triangle: 3 vertices, 3 edges, 1 face
/// let counts = FVector {
///     by_dim: vec![3, 3, 1],
/// };
///
/// assert_eq!(counts.count(0), 3);  // vertices
/// assert_eq!(counts.count(1), 3);  // edges
/// assert_eq!(counts.count(2), 1);  // faces
/// assert_eq!(counts.dimension(), 2);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FVector {
    /// `by_dim[k]` = `f_k` = number of `k`-simplices
    pub by_dim: Vec<usize>,
}

impl FVector {
    /// Get the number of `k`-simplices.
    ///
    /// Returns 0 if `k` is out of range.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::topology::characteristics::euler::FVector;
    ///
    /// let counts = FVector {
    ///     by_dim: vec![4, 6, 4, 1],  // 3D tetrahedron
    /// };
    ///
    /// assert_eq!(counts.count(0), 4);  // 4 vertices
    /// assert_eq!(counts.count(1), 6);  // 6 edges
    /// assert_eq!(counts.count(2), 4);  // 4 faces
    /// assert_eq!(counts.count(3), 1);  // 1 cell
    /// assert_eq!(counts.count(4), 0);  // out of range
    /// ```
    #[must_use]
    #[inline]
    pub fn count(&self, k: usize) -> usize {
        self.by_dim.get(k).copied().unwrap_or(0)
    }

    /// Get the dimension (maximum `k` where `f_k` > 0).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::topology::characteristics::euler::FVector;
    ///
    /// let counts_3d = FVector {
    ///     by_dim: vec![4, 6, 4, 1],
    /// };
    /// assert_eq!(counts_3d.dimension(), 3);
    ///
    /// let counts_2d = FVector {
    ///     by_dim: vec![3, 3, 1],
    /// };
    /// assert_eq!(counts_2d.dimension(), 2);
    /// ```
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.by_dim.len().saturating_sub(1)
    }
}

/// Topological classification of a triangulation.
///
/// Classifies the global topological structure to determine
/// the expected Euler characteristic.
///
/// # Variants
///
/// - `Empty`: No cells (χ = 0)
/// - `SingleSimplex(D)`: One D-simplex (χ = 1)
/// - `Ball(D)`: D-ball with boundary (χ = 1)
/// - `ClosedSphere(D)`: Closed D-sphere without boundary (χ = 1 + (-1)^D)
/// - `Unknown`: Cannot determine topology
///
/// # Examples
///
/// ```rust
/// use delaunay::topology::characteristics::euler::TopologyClassification;
///
/// let ball = TopologyClassification::Ball(3);
/// assert_eq!(format!("{:?}", ball), "Ball(3)");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyClassification {
    /// Empty triangulation (no cells).
    Empty,

    /// Single D-simplex (first cell).
    ///
    /// The value represents the dimension D.
    SingleSimplex(usize),

    /// Topological D-ball (has boundary).
    ///
    /// Most finite Delaunay triangulations fall into this category.
    /// The value represents the dimension D.
    Ball(usize),

    /// Closed D-sphere (no boundary).
    ///
    /// Rare for finite triangulations; requires special construction
    /// (e.g., periodic boundary conditions).
    /// The value represents the dimension D.
    ClosedSphere(usize),

    /// Cannot determine or doesn't fit known categories.
    Unknown,
}

/// Count all k-simplices in the triangulation.
///
/// Computes the complete f-vector (f₀, f₁, ..., `f_D`) where `f_k` is the
/// number of `k`-dimensional simplices.
///
/// # Algorithm
///
/// - `f₀` (vertices): Direct count from Tds - O(1)
/// - `f_D` (cells): Direct count from Tds - O(1)
/// - `f_{D-1}` (facets): Use `build_facet_to_cells_map()` - O(N·D²)
/// - Intermediate `k`: Enumerate combinations from cells - O(N · C(D+1, k+1))
///
/// For practical dimensions (D ≤ 5), this is efficient.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::*;
/// use delaunay::topology::characteristics::euler;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0]),
///     vertex!([1.0, 0.0]),
///     vertex!([0.5, 1.0]),
/// ];
/// let dt = DelaunayTriangulation::new(&vertices).unwrap();
///
/// let counts = euler::count_simplices(dt.tds()).unwrap();
/// assert_eq!(counts.count(0), 3);  // 3 vertices
/// assert_eq!(counts.count(1), 3);  // 3 edges
/// assert_eq!(counts.count(2), 1);  // 1 face
/// ```
///
/// # Errors
///
/// Returns `TopologyError::Counting` if simplex enumeration fails.
pub fn count_simplices<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
) -> Result<FVector, TopologyError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    // Handle empty triangulation without building any facet map.
    let mut by_dim = vec![0usize; D + 1];
    by_dim[0] = tds.number_of_vertices();
    by_dim[D] = tds.number_of_cells();
    if by_dim[D] == 0 {
        return Ok(FVector { by_dim });
    }

    // Build the facet map once, then compute counts from it.
    let facet_to_cells = tds
        .build_facet_to_cells_map()
        .map_err(|e| TopologyError::Counting(format!("Failed to build facet map: {e}")))?;

    Ok(count_simplices_with_facet_to_cells_map(
        tds,
        &facet_to_cells,
    ))
}

pub(crate) fn count_simplices_with_facet_to_cells_map<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    facet_to_cells: &FacetToCellsMap,
) -> FVector
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let mut by_dim = vec![0usize; D + 1];

    // f₀: vertices (O(1))
    by_dim[0] = tds.number_of_vertices();

    // f_D: D-cells (O(1))
    by_dim[D] = tds.number_of_cells();

    // Handle empty triangulation
    if by_dim[D] == 0 {
        return FVector { by_dim };
    }

    // f_{D-1}: (D-1)-facets from precomputed map
    by_dim[D - 1] = facet_to_cells.len();

    // Intermediate dimensions (1 ≤ k ≤ D-2): enumerate combinations.
    //
    // We keep a set per k and fill them in a single pass over cells, which is faster than
    // re-iterating all cells once per k.
    // Skip if D <= 2 (no intermediate dimensions)
    if D > 2 {
        let mut intermediate_simplex_sets: Vec<
            FastHashSet<SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>>,
        > = (0..(D - 2)).map(|_| FastHashSet::default()).collect();

        // Pre-sort each cell's vertex keys once so every generated combination is already
        // in canonical (sorted) order, avoiding per-combination sorting.
        let mut sorted_vertex_keys: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::new();

        for (_cell_key, cell) in tds.cells() {
            sorted_vertex_keys.clear();
            sorted_vertex_keys.extend(cell.vertices().iter().copied());
            sorted_vertex_keys.sort();

            for simplex_dimension in 1..=D - 2 {
                let simplex_set =
                    &mut intermediate_simplex_sets[simplex_dimension.saturating_sub(1)];
                let simplex_size = simplex_dimension + 1; // k-simplex has k+1 vertices
                insert_simplices_of_size(&sorted_vertex_keys, simplex_size, simplex_set);
            }
        }

        for simplex_dimension in 1..=D - 2 {
            by_dim[simplex_dimension] =
                intermediate_simplex_sets[simplex_dimension.saturating_sub(1)].len();
        }
    }

    FVector { by_dim }
}

fn insert_simplices_of_size(
    vertex_keys: &[VertexKey],
    simplex_size: usize,
    simplex_set: &mut FastHashSet<SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>>,
) {
    let n = vertex_keys.len();
    if n < simplex_size {
        return;
    }

    // We expect `vertex_keys` to be in sorted (canonical) order.
    //
    // With sorted input, each combination produced by increasing indices is already sorted, so we
    // can insert it directly without per-combination sorting.
    debug_assert!(vertex_keys.windows(2).all(|w| w[0] <= w[1]));

    // Generate all C(n, simplex_size) combinations using the standard lexicographic algorithm.
    //
    // We maintain `indices[0..simplex_size]` as strictly increasing positions into `vertex_keys`.
    // To advance to the next combination:
    // 1) Find the rightmost index that can be incremented without running out of room (the pivot).
    // 2) Increment it.
    // 3) Reset all subsequent indices to consecutive values.
    let mut indices: SmallBuffer<usize, MAX_PRACTICAL_DIMENSION_SIZE> = (0..simplex_size).collect();

    'outer: loop {
        // Extract current combination (already sorted due to sorted input + increasing indices).
        let mut simplex_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::new();
        for &vertex_index in &indices {
            simplex_vertices.push(vertex_keys[vertex_index]);
        }
        simplex_set.insert(simplex_vertices);

        // Generate next combination.
        // The maximum valid value at position `i` is `i + n - simplex_size`.
        let mut pivot = simplex_size;
        while pivot > 0 {
            pivot -= 1;
            if indices[pivot] != pivot + n - simplex_size {
                break;
            }
            if pivot == 0 {
                break 'outer;
            }
        }

        indices[pivot] += 1;
        for position in (pivot + 1)..simplex_size {
            indices[position] = indices[position - 1] + 1;
        }
    }
}

/// Count simplices on the boundary (convex hull) only.
///
/// This computes simplex counts for just the boundary facets,
/// which form a (D-1)-dimensional simplicial complex.
///
/// # Algorithm
///
/// 1. Extract all boundary facets using `boundary_facets()`
/// 2. Collect unique vertices that appear on the boundary
/// 3. Count (D-1)-cells (the boundary facets themselves)
/// 4. For intermediate dimensions k < D-1, enumerate k-simplices
///    from the boundary facets' vertex combinations
///
/// # Expected Euler Characteristics
///
/// The boundary forms a (D-1)-sphere S^(D-1):
/// - 2D triangulation: boundary is S¹ (circle) with χ = 0
/// - 3D triangulation: boundary is S² (sphere) with χ = 2
/// - 4D triangulation: boundary is S³ (3-sphere) with χ = 0
/// - 5D triangulation: boundary is S⁴ (4-sphere) with χ = 2
/// - Generally: χ(S^k) = 1 + (-1)^k
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::*;
/// use delaunay::topology::characteristics::euler;
///
/// // 3D tetrahedron - boundary is S² (sphere)
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let dt = DelaunayTriangulation::new(&vertices).unwrap();
///
/// let boundary_counts = euler::count_boundary_simplices(dt.tds()).unwrap();
/// let boundary_chi = euler::euler_characteristic(&boundary_counts);
/// assert_eq!(boundary_chi, 2);  // S² has χ = 2
/// ```
///
/// # Errors
///
/// Returns `TopologyError::Counting` if boundary enumeration fails.
pub fn count_boundary_simplices<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
) -> Result<FVector, TopologyError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    // Get boundary facets
    let boundary_facets: Vec<_> = tds
        .boundary_facets()
        .map_err(|e| TopologyError::Counting(format!("Failed to get boundary facets: {e}")))?
        .collect();

    if boundary_facets.is_empty() {
        // No boundary - return zero counts for (D-1)-dimensional complex
        return Ok(FVector { by_dim: vec![0; D] });
    }

    // Collect unique vertices on the boundary
    let mut boundary_vertices = FastHashSet::default();
    for facet in &boundary_facets {
        let cell = facet
            .cell()
            .map_err(|e| TopologyError::Counting(format!("Failed to get facet cell: {e}")))?;
        let facet_index = usize::from(facet.facet_index());

        // Add all vertex keys except the opposite vertex
        for (i, &v_key) in cell.vertices().iter().enumerate() {
            if i != facet_index {
                boundary_vertices.insert(v_key);
            }
        }
    }

    let num_boundary_vertices = boundary_vertices.len();
    let num_boundary_facets = boundary_facets.len(); // These are (D-1)-simplices

    // Initialize counts for (D-1)-dimensional complex
    // by_dim[0] = vertices, by_dim[1] = edges, ..., by_dim[D-1] = (D-1)-cells
    let mut by_dim = vec![0; D];
    by_dim[0] = num_boundary_vertices;
    by_dim[D - 1] = num_boundary_facets;

    // Count intermediate k-simplices (1 ≤ k < D-1) by enumerating combinations
    // from boundary facets.
    //
    // We keep a set per k and fill them in a single pass over boundary facets, which is faster than
    // re-iterating all facets once per k.
    // Skip if D <= 2 (no intermediate dimensions in boundary)
    if D > 2 {
        let mut intermediate_simplex_sets: Vec<
            FastHashSet<SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>>,
        > = (0..(D - 2)).map(|_| FastHashSet::default()).collect();

        for facet in &boundary_facets {
            let cell = facet
                .cell()
                .map_err(|e| TopologyError::Counting(format!("Failed to get facet cell: {e}")))?;
            let facet_index = usize::from(facet.facet_index());

            // Collect vertex keys for this facet (excluding opposite vertex).
            //
            // We sort once so every generated combination is already in canonical order, avoiding
            // per-combination sorting.
            let mut facet_vertex_keys: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
                SmallBuffer::new();
            for (vertex_position, &v_key) in cell.vertices().iter().enumerate() {
                if vertex_position != facet_index {
                    facet_vertex_keys.push(v_key);
                }
            }
            facet_vertex_keys.sort();

            for simplex_dimension in 1..=D - 2 {
                let simplex_set =
                    &mut intermediate_simplex_sets[simplex_dimension.saturating_sub(1)];
                let simplex_size = simplex_dimension + 1; // k-simplex has k+1 vertices
                insert_simplices_of_size(&facet_vertex_keys, simplex_size, simplex_set);
            }
        }

        for simplex_dimension in 1..=D - 2 {
            by_dim[simplex_dimension] =
                intermediate_simplex_sets[simplex_dimension.saturating_sub(1)].len();
        }
    }

    Ok(FVector { by_dim })
}

/// Compute Euler characteristic from simplex counts.
///
/// Uses the alternating sum formula: χ = Σ(-1)^k · `f_k`
///
/// # Formula
///
/// ```text
/// χ = f₀ - f₁ + f₂ - f₃ + ... ± f_D
///   = Σ(k=0 to D) (-1)^k · f_k
/// ```
///
/// # Examples
///
/// ```rust
/// use delaunay::topology::characteristics::euler::{FVector, euler_characteristic};
///
/// // 2D triangle: V=3, E=3, F=1 → χ = 3-3+1 = 1
/// let counts = FVector {
///     by_dim: vec![3, 3, 1],
/// };
/// assert_eq!(euler_characteristic(&counts), 1);
///
/// // 3D tetrahedron: V=4, E=6, F=4, C=1 → χ = 4-6+4-1 = 1
/// let counts_3d = FVector {
///     by_dim: vec![4, 6, 4, 1],
/// };
/// assert_eq!(euler_characteristic(&counts_3d), 1);
/// ```
#[must_use]
#[allow(clippy::cast_possible_wrap)] // Simplex counts won't exceed isize::MAX in practice
pub fn euler_characteristic(counts: &FVector) -> isize {
    counts
        .by_dim
        .iter()
        .enumerate()
        .map(|(k, &f_k)| {
            let sign = if k % 2 == 0 { 1 } else { -1 };
            sign * (f_k as isize)
        })
        .sum()
}

/// Classify the triangulation topologically.
///
/// Determines the topological type based on the number of cells
/// and boundary structure.
///
/// # Classification Logic
///
/// - No cells → `Empty`
/// - One cell → `SingleSimplex(D)`
/// - Has boundary → `Ball(D)`
/// - No boundary → `ClosedSphere(D)` (rare)
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::*;
/// use delaunay::topology::characteristics::euler::{classify_triangulation, TopologyClassification};
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let dt = DelaunayTriangulation::new(&vertices).unwrap();
///
/// let classification = classify_triangulation(dt.tds()).unwrap();
/// assert_eq!(classification, TopologyClassification::SingleSimplex(3));
/// ```
///
/// # Errors
///
/// Returns `TopologyError::Classification` if boundary detection fails.
pub fn classify_triangulation<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
) -> Result<TopologyClassification, TopologyError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let num_cells = tds.number_of_cells();

    // Empty triangulation
    if num_cells == 0 {
        return Ok(TopologyClassification::Empty);
    }

    // Single simplex
    if num_cells == 1 {
        return Ok(TopologyClassification::SingleSimplex(D));
    }

    // Check boundary
    let has_boundary = tds
        .number_of_boundary_facets()
        .map_err(|e| TopologyError::Classification(format!("Failed to count boundary: {e}")))?
        > 0;

    if has_boundary {
        // Has boundary → topological ball
        Ok(TopologyClassification::Ball(D))
    } else {
        // No boundary → closed manifold (assume sphere for now)
        Ok(TopologyClassification::ClosedSphere(D))
    }
}

/// Get expected χ for a topological classification.
///
/// # Expected Values
///
/// - `Empty`: χ = 0
/// - `SingleSimplex(_)`: χ = 1
/// - `Ball(_)`: χ = 1
/// - `ClosedSphere(d)`: χ = 1 + (-1)^d
/// - `Unknown`: None
///
/// # Examples
///
/// ```rust
/// use delaunay::topology::characteristics::euler::{TopologyClassification, expected_chi_for};
///
/// assert_eq!(expected_chi_for(&TopologyClassification::Empty), Some(0));
/// assert_eq!(expected_chi_for(&TopologyClassification::Ball(3)), Some(1));
/// assert_eq!(expected_chi_for(&TopologyClassification::ClosedSphere(2)), Some(2));
/// assert_eq!(expected_chi_for(&TopologyClassification::ClosedSphere(3)), Some(0));
/// assert_eq!(expected_chi_for(&TopologyClassification::Unknown), None);
/// ```
#[must_use]
pub fn expected_chi_for(classification: &TopologyClassification) -> Option<isize> {
    match classification {
        TopologyClassification::Empty => Some(0),
        TopologyClassification::SingleSimplex(_) | TopologyClassification::Ball(_) => Some(1),
        TopologyClassification::ClosedSphere(d) => {
            // χ(S^d) = 1 + (-1)^d
            Some(1 + if d % 2 == 0 { 1 } else { -1 })
        }
        TopologyClassification::Unknown => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplex_counts() {
        let counts = FVector {
            by_dim: vec![3, 3, 1],
        };

        assert_eq!(counts.count(0), 3);
        assert_eq!(counts.count(1), 3);
        assert_eq!(counts.count(2), 1);
        assert_eq!(counts.count(3), 0); // out of range
        assert_eq!(counts.dimension(), 2);
    }

    #[test]
    fn test_insert_simplices_of_size() {
        use slotmap::SlotMap;

        let mut vertex_slots: SlotMap<VertexKey, ()> = SlotMap::default();
        let v0 = vertex_slots.insert(());
        let v1 = vertex_slots.insert(());
        let v2 = vertex_slots.insert(());
        let v3 = vertex_slots.insert(());

        // n < simplex_size => no combinations.
        let mut simplex_set = FastHashSet::default();
        insert_simplices_of_size(&[v0, v1], 3, &mut simplex_set);
        assert!(simplex_set.is_empty());

        // simplex_size == n => exactly one combination.
        let mut simplex_set = FastHashSet::default();
        let mut keys = vec![v2, v0, v1]; // deliberately unsorted
        keys.sort();
        insert_simplices_of_size(&keys, 3, &mut simplex_set);
        assert_eq!(simplex_set.len(), 1);

        let mut expected: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> = SmallBuffer::new();
        expected.push(v0);
        expected.push(v1);
        expected.push(v2);
        expected.sort();
        assert!(simplex_set.contains(&expected));

        // simplex_size == 1 => n singleton combinations.
        let mut simplex_set = FastHashSet::default();
        let mut keys = vec![v3, v1, v0]; // deliberately unsorted
        keys.sort();
        insert_simplices_of_size(&keys, 1, &mut simplex_set);
        assert_eq!(simplex_set.len(), keys.len());

        for &vk in &keys {
            let mut singleton: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
                SmallBuffer::new();
            singleton.push(vk);
            assert!(simplex_set.contains(&singleton));
        }

        // C(3, 2) = 3 combinations.
        let mut simplex_set = FastHashSet::default();
        let mut keys = vec![v0, v2, v1]; // deliberately unsorted
        keys.sort();
        insert_simplices_of_size(&keys, 2, &mut simplex_set);
        assert_eq!(simplex_set.len(), 3);

        let expected_pairs = [(v0, v1), (v0, v2), (v1, v2)];
        for (a, b) in expected_pairs {
            let mut pair: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> = SmallBuffer::new();
            pair.push(a);
            pair.push(b);
            pair.sort();
            assert!(simplex_set.contains(&pair));
        }
    }

    #[test]
    fn test_euler_characteristic_2d() {
        // 2D triangle: V=3, E=3, F=1 → χ = 3-3+1 = 1
        let counts = FVector {
            by_dim: vec![3, 3, 1],
        };
        assert_eq!(euler_characteristic(&counts), 1);
    }

    #[test]
    fn test_euler_characteristic_3d() {
        // 3D tetrahedron: V=4, E=6, F=4, C=1 → χ = 4-6+4-1 = 1
        let counts = FVector {
            by_dim: vec![4, 6, 4, 1],
        };
        assert_eq!(euler_characteristic(&counts), 1);
    }

    #[test]
    fn test_expected_chi_for() {
        assert_eq!(expected_chi_for(&TopologyClassification::Empty), Some(0));
        assert_eq!(
            expected_chi_for(&TopologyClassification::SingleSimplex(3)),
            Some(1)
        );
        assert_eq!(expected_chi_for(&TopologyClassification::Ball(3)), Some(1));
        assert_eq!(
            expected_chi_for(&TopologyClassification::ClosedSphere(2)),
            Some(2)
        );
        assert_eq!(
            expected_chi_for(&TopologyClassification::ClosedSphere(3)),
            Some(0)
        );
        assert_eq!(expected_chi_for(&TopologyClassification::Unknown), None);
    }
}
