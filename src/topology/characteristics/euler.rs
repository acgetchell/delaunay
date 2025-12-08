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
    collections::{FastHashSet, MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer},
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
/// # Examples
///
/// ```rust
/// use delaunay::topology::characteristics::euler::SimplexCounts;
///
/// // 2D triangle: 3 vertices, 3 edges, 1 face
/// let counts = SimplexCounts {
///     by_dim: vec![3, 3, 1],
/// };
///
/// assert_eq!(counts.count(0), 3);  // vertices
/// assert_eq!(counts.count(1), 3);  // edges
/// assert_eq!(counts.count(2), 1);  // faces
/// assert_eq!(counts.dimension(), 2);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimplexCounts {
    /// `by_dim[k]` = `f_k` = number of `k`-simplices
    pub by_dim: Vec<usize>,
}

impl SimplexCounts {
    /// Get the number of `k`-simplices.
    ///
    /// Returns 0 if `k` is out of range.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::topology::characteristics::euler::SimplexCounts;
    ///
    /// let counts = SimplexCounts {
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
    /// use delaunay::topology::characteristics::euler::SimplexCounts;
    ///
    /// let counts_3d = SimplexCounts {
    ///     by_dim: vec![4, 6, 4, 1],
    /// };
    /// assert_eq!(counts_3d.dimension(), 3);
    ///
    /// let counts_2d = SimplexCounts {
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
) -> Result<SimplexCounts, TopologyError>
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
        return Ok(SimplexCounts { by_dim });
    }

    // f_{D-1}: (D-1)-facets using existing infrastructure
    by_dim[D - 1] = tds
        .build_facet_to_cells_map()
        .map_err(|e| TopologyError::Counting(format!("Failed to build facet map: {e}")))?
        .len();

    // Intermediate dimensions: enumerate combinations
    // Skip if D <= 2 (no intermediate dimensions)
    if D > 2 {
        for (k, item) in by_dim.iter_mut().enumerate().take(D - 1).skip(1) {
            *item = count_k_simplices(tds, k);
        }
    }

    Ok(SimplexCounts { by_dim })
}

/// Count k-dimensional simplices by enumerating combinations.
///
/// For each D-cell, generates all C(D+1, k+1) vertex combinations of size k+1,
/// then deduplicates using a hash set with canonical ordering.
///
/// # Arguments
///
/// * `tds` - The triangulation data structure
/// * `k` - Dimension of simplices to count (0 < k < D-1)
///
/// # Returns
///
/// The number of unique k-simplices.
///
/// # Complexity
///
/// O(N · C(D+1, k+1)) where N is the number of cells.
fn count_k_simplices<T, U, V, const D: usize>(tds: &Tds<T, U, V, D>, k: usize) -> usize
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let mut k_simplex_set = FastHashSet::default();
    let simplex_size = k + 1; // k-simplex has k+1 vertices

    for (_cell_key, cell) in tds.cells() {
        let vertex_keys = cell.vertices();

        // Generate all C(D+1, k+1) combinations of vertex keys using indices
        let n = vertex_keys.len();
        let mut indices: SmallBuffer<usize, MAX_PRACTICAL_DIMENSION_SIZE> =
            (0..simplex_size).collect();

        'outer: loop {
            // Extract current combination
            let mut k_simplex: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
                SmallBuffer::new();
            for &i in &indices {
                k_simplex.push(vertex_keys[i]);
            }
            k_simplex.sort();
            k_simplex_set.insert(k_simplex);

            // Generate next combination using standard algorithm
            let mut i = simplex_size;
            while i > 0 {
                i -= 1;
                if indices[i] != i + n - simplex_size {
                    break;
                }
                if i == 0 {
                    break 'outer;
                }
            }

            indices[i] += 1;
            for j in (i + 1)..simplex_size {
                indices[j] = indices[j - 1] + 1;
            }
        }
    }

    k_simplex_set.len()
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
/// use delaunay::topology::characteristics::euler::{SimplexCounts, euler_characteristic};
///
/// // 2D triangle: V=3, E=3, F=1 → χ = 3-3+1 = 1
/// let counts = SimplexCounts {
///     by_dim: vec![3, 3, 1],
/// };
/// assert_eq!(euler_characteristic(&counts), 1);
///
/// // 3D tetrahedron: V=4, E=6, F=4, C=1 → χ = 4-6+4-1 = 1
/// let counts_3d = SimplexCounts {
///     by_dim: vec![4, 6, 4, 1],
/// };
/// assert_eq!(euler_characteristic(&counts_3d), 1);
/// ```
#[must_use]
#[allow(clippy::cast_possible_wrap)] // Simplex counts won't exceed isize::MAX in practice
pub fn euler_characteristic(counts: &SimplexCounts) -> isize {
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
        let counts = SimplexCounts {
            by_dim: vec![3, 3, 1],
        };

        assert_eq!(counts.count(0), 3);
        assert_eq!(counts.count(1), 3);
        assert_eq!(counts.count(2), 1);
        assert_eq!(counts.count(3), 0); // out of range
        assert_eq!(counts.dimension(), 2);
    }

    #[test]
    fn test_euler_characteristic_2d() {
        // 2D triangle: V=3, E=3, F=1 → χ = 3-3+1 = 1
        let counts = SimplexCounts {
            by_dim: vec![3, 3, 1],
        };
        assert_eq!(euler_characteristic(&counts), 1);
    }

    #[test]
    fn test_euler_characteristic_3d() {
        // 3D tetrahedron: V=4, E=6, F=4, C=1 → χ = 4-6+4-1 = 1
        let counts = SimplexCounts {
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
