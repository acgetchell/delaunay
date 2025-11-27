//! Delaunay triangulation layer with incremental insertion.
//!
//! This layer adds Delaunay-specific operations on top of the generic
//! `Triangulation` struct, following CGAL's architecture.

use core::iter::Sum;
use core::ops::{AddAssign, SubAssign};

use num_traits::NumCast;

use crate::core::algorithms::incremental_insertion::{
    InsertionError, fill_cavity, wire_cavity_neighbors,
};
use crate::core::algorithms::locate::{
    LocateResult, extract_cavity_boundary, find_conflict_region, locate,
};
use crate::core::traits::data_type::DataType;
use crate::core::triangulation::Triangulation;
use crate::core::triangulation_data_structure::{
    CellKey, Tds, TriangulationConstructionError, VertexKey,
};
use crate::core::vertex::Vertex;
use crate::geometry::kernel::Kernel;
use crate::geometry::traits::coordinate::CoordinateScalar;

/// Delaunay triangulation with incremental insertion support.
///
/// # Type Parameters
/// - `K`: Geometric kernel implementing predicates
/// - `U`: User data type for vertices
/// - `V`: User data type for cells
/// - `D`: Dimension of the triangulation
///
/// # Phase 3 TODO
/// Implement incremental insertion with:
/// - Point location (facet walking)
/// - Conflict region computation (local BFS)
/// - Cavity extraction and filling
/// - Local neighbor wiring (no global `assign_neighbors`)
pub struct DelaunayTriangulation<K, U, V, const D: usize>
where
    K: Kernel<D>,
    U: DataType,
    V: DataType,
{
    /// The underlying generic triangulation.
    tri: Triangulation<K, U, V, D>,
    /// Hint for next `locate()` call (last inserted cell)
    last_inserted_cell: Option<CellKey>,
}

// Simplified API for common case (f64 with FastKernel)
impl<U, V, const D: usize> DelaunayTriangulation<crate::geometry::kernel::FastKernel<f64>, U, V, D>
where
    U: DataType,
    V: DataType,
{
    /// Create a Delaunay triangulation from vertices using default settings (f64, fast predicates).
    ///
    /// This is the recommended constructor for most users. It uses efficient cavity-based insertion:
    /// 1. Build initial simplex (D+1 vertices) using Bowyer-Watson
    /// 2. Insert remaining vertices incrementally with locate → conflict → cavity → wire
    ///
    /// # Errors
    /// Returns error if initial simplex cannot be constructed or insertion fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulation::new(&vertices).unwrap();
    /// assert_eq!(dt.number_of_vertices(), 5);
    /// ```
    pub fn new(vertices: &[Vertex<f64, U, D>]) -> Result<Self, TriangulationConstructionError> {
        Self::with_kernel(crate::geometry::kernel::FastKernel::new(), vertices)
    }
}

// Generic implementation for all kernels
impl<K, U, V, const D: usize> DelaunayTriangulation<K, U, V, D>
where
    K: Kernel<D>,
    K::Scalar: AddAssign + SubAssign + Sum + NumCast,
    U: DataType,
    V: DataType,
{
    /// Create an empty Delaunay triangulation with the given kernel.
    ///
    /// This creates a triangulation with no vertices or cells. Use [`insert`](Self::insert)
    /// to add vertices incrementally.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
    /// use delaunay::geometry::kernel::FastKernel;
    ///
    /// let dt: DelaunayTriangulation<FastKernel<f64>, (), (), 4> =
    ///     DelaunayTriangulation::new_empty(FastKernel::new());
    /// assert_eq!(dt.number_of_vertices(), 0);
    /// assert_eq!(dt.number_of_cells(), 0);
    /// ```
    #[must_use]
    pub fn new_empty(kernel: K) -> Self {
        Self {
            tri: Triangulation::new_empty(kernel),
            last_inserted_cell: None,
        }
    }

    /// Create a Delaunay triangulation from vertices with an explicit kernel (advanced usage).
    ///
    /// Most users should use [`DelaunayTriangulation::new()`] instead, which uses fast predicates
    /// by default. Use this method only if you need:
    /// - Custom coordinate precision (f32, custom types)
    /// - Explicit robust/exact arithmetic predicates
    /// - Specialized kernel implementations
    ///
    /// This uses the efficient cavity-based algorithm:
    /// 1. Build initial simplex (D+1 vertices) using Bowyer-Watson
    /// 2. Insert remaining vertices incrementally with locate → conflict → cavity → wire
    ///
    /// # Errors
    /// Returns error if initial simplex cannot be constructed or insertion fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
    /// use delaunay::geometry::kernel::RobustKernel;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0, 1.0]),
    /// ];
    ///
    /// // Use robust kernel for exact arithmetic
    /// let dt: DelaunayTriangulation<RobustKernel<f64>, (), (), 4> =
    ///     DelaunayTriangulation::with_kernel(
    ///         RobustKernel::new(),
    ///         &vertices
    ///     ).unwrap();
    /// assert_eq!(dt.number_of_vertices(), 5);
    /// ```
    pub fn with_kernel(
        kernel: K,
        vertices: &[Vertex<K::Scalar, U, D>],
    ) -> Result<Self, TriangulationConstructionError>
    where
        K::Scalar: CoordinateScalar,
    {
        if vertices.len() < D + 1 {
            return Err(TriangulationConstructionError::InsufficientVertices {
                dimension: D,
                source: crate::core::cell::CellValidationError::InsufficientVertices {
                    actual: vertices.len(),
                    expected: D + 1,
                    dimension: D,
                },
            });
        }

        // Build initial simplex using existing Tds::new (Bowyer-Watson)
        let initial_vertices = &vertices[..=D];
        let tds = Tds::new(initial_vertices)?;

        let mut dt = Self {
            tri: Triangulation { kernel, tds },
            last_inserted_cell: None,
        };

        // Insert remaining vertices incrementally
        for vertex in vertices.iter().skip(D + 1) {
            dt.insert(*vertex)
                .map_err(|e| TriangulationConstructionError::FailedToAddVertex {
                    message: format!("Incremental insertion failed: {e}"),
                })?;
        }

        Ok(dt)
    }

    /// Returns the number of vertices in the triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0, 1.0]),
    ///     vertex!([0.2, 0.2, 0.2, 0.2]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulation::new(&vertices).unwrap();
    /// assert_eq!(dt.number_of_vertices(), 6);
    /// ```
    #[must_use]
    pub fn number_of_vertices(&self) -> usize {
        self.tri.number_of_vertices()
    }

    /// Returns the number of cells in the triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulation::new(&vertices).unwrap();
    /// // One 4-simplex in 4D
    /// assert_eq!(dt.number_of_cells(), 1);
    /// ```
    #[must_use]
    pub fn number_of_cells(&self) -> usize {
        self.tri.number_of_cells()
    }

    /// Returns the dimension of the triangulation.
    ///
    /// Returns the dimension `D` as an `i32`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulation::new(&vertices).unwrap();
    /// assert_eq!(dt.dim(), 4);
    /// ```
    #[must_use]
    pub fn dim(&self) -> i32 {
        self.tri.dim()
    }

    /// Insert a vertex into the Delaunay triangulation using incremental cavity-based algorithm.
    ///
    /// # Algorithm
    /// 1. Insert vertex into Tds
    /// 2. Locate cell containing the point
    /// 3. Find conflict region (cells whose circumspheres contain the point)
    /// 4. Extract cavity boundary
    /// 5. Remove conflict cells
    /// 6. Fill cavity (create new cells)
    /// 7. Wire neighbors locally
    ///
    /// # Errors
    /// Returns error if:
    /// - Point location fails
    /// - Point is outside convex hull (hull extension not yet implemented)
    /// - Point is on a facet, edge, or vertex (not yet implemented)
    /// - Conflict region computation fails
    /// - Cavity boundary extraction fails
    /// - Cavity filling or neighbor wiring fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
    /// use delaunay::geometry::kernel::FastKernel;
    /// use delaunay::vertex;
    ///
    /// // Create initial triangulation with 5 vertices (4-simplex)
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0, 1.0]),
    /// ];
    /// let mut dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulation::new(&vertices).unwrap();
    /// assert_eq!(dt.number_of_vertices(), 5);
    ///
    /// // Insert additional interior vertex
    /// dt.insert(vertex!([0.2, 0.2, 0.2, 0.2])).unwrap();
    /// assert_eq!(dt.number_of_vertices(), 6);
    /// assert!(dt.number_of_cells() > 1);
    /// ```
    pub fn insert(&mut self, vertex: Vertex<K::Scalar, U, D>) -> Result<VertexKey, InsertionError>
    where
        K::Scalar: CoordinateScalar,
    {
        // 1. Insert vertex into Tds
        let v_key = self.tri.tds.insert_vertex_with_mapping(vertex)?;

        // 2. Locate containing cell
        let point = self
            .tri
            .tds
            .get_vertex_by_key(v_key)
            .ok_or_else(|| InsertionError::CavityFilling {
                message: "Vertex key invalid immediately after insertion".to_string(),
            })?
            .point();
        let location = locate(
            &self.tri.tds,
            &self.tri.kernel,
            point,
            self.last_inserted_cell,
        )?;

        let start_cell = match location {
            LocateResult::InsideCell(cell_key) => cell_key,
            LocateResult::Outside => {
                // TODO: Handle hull extension
                return Err(InsertionError::CavityFilling {
                    message: "Point outside hull - hull extension not yet implemented".to_string(),
                });
            }
            _ => {
                // TODO: Handle other cases (OnFacet, OnEdge, OnVertex)
                return Err(InsertionError::CavityFilling {
                    message: format!("Unhandled location result: {location:?}"),
                });
            }
        };

        // 3. Find conflict region
        let conflict_cells =
            find_conflict_region(&self.tri.tds, &self.tri.kernel, point, start_cell)?;

        // 4. Extract cavity boundary
        let boundary_facets = extract_cavity_boundary(&self.tri.tds, &conflict_cells)?;

        // 5. Fill cavity BEFORE removing old cells (so boundary cells still exist)
        let new_cells = fill_cavity(&mut self.tri.tds, v_key, &boundary_facets)?;

        // 6. Wire neighbors (while both old and new cells exist)
        wire_cavity_neighbors(&mut self.tri.tds, &new_cells, &boundary_facets)?;

        // 7. Remove conflict cells (now that new cells are wired up)
        let _removed_count = self.tri.tds.remove_cells_by_keys(&conflict_cells);

        // 8. Cache last inserted cell for next locate hint
        self.last_inserted_cell = new_cells.first().copied();

        Ok(v_key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vertex;

    /// Macro to generate incremental insertion tests across dimensions.
    ///
    /// This macro generates tests that verify incremental insertion by:
    /// 1. Creating a minimal simplex (D+1 vertices)
    /// 2. Inserting one additional interior vertex
    /// 3. Verifying the triangulation has the expected structure
    ///
    /// # Usage
    /// ```ignore
    /// test_incremental_insertion!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [0.5, 0.5]);
    /// ```
    macro_rules! test_incremental_insertion {
        ($dim:expr, [$($simplex_coords:expr),+ $(,)?], $interior_point:expr) => {
            pastey::paste! {
                #[test]
                fn [<test_incremental_insertion_ $dim d>]() {
                    // Build initial simplex (D+1 vertices)
                    let mut vertices: Vec<Vertex<f64, (), $dim>> = vec![
                        $(vertex!($simplex_coords)),+
                    ];

                    // Add interior point to be inserted incrementally
                    vertices.push(vertex!($interior_point));

                    let expected_vertices = vertices.len();

                    let dt: DelaunayTriangulation<_, (), (), $dim> =
                        DelaunayTriangulation::new(&vertices).unwrap();

                    assert_eq!(dt.number_of_vertices(), expected_vertices,
                        "{}D: Expected {} vertices", $dim, expected_vertices);
                    assert!(dt.number_of_cells() > 1,
                        "{}D: Expected multiple cells, got {}", $dim, dt.number_of_cells());
                }
            }
        };
    }

    // 2D: Triangle + interior point
    test_incremental_insertion!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [0.5, 0.5]);

    // 3D: Tetrahedron + interior point
    test_incremental_insertion!(
        3,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ],
        [0.2, 0.2, 0.2]
    );

    // 4D: 4-simplex + interior point
    test_incremental_insertion!(
        4,
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ],
        [0.2, 0.2, 0.2, 0.2]
    );

    // 5D: 5-simplex + interior point
    test_incremental_insertion!(
        5,
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ],
        [0.2, 0.2, 0.2, 0.2, 0.2]
    );

    // =========================================================================
    // new_empty() tests
    // =========================================================================

    #[test]
    fn test_new_empty_creates_empty_triangulation() {
        use crate::geometry::kernel::FastKernel;

        let dt: DelaunayTriangulation<FastKernel<f64>, (), (), 3> =
            DelaunayTriangulation::new_empty(FastKernel::new());

        assert_eq!(dt.number_of_vertices(), 0);
        assert_eq!(dt.number_of_cells(), 0);
        // dim() returns -1 for empty triangulation
        assert_eq!(dt.dim(), -1);
    }

    #[test]
    fn test_new_empty_then_construct() {
        use crate::geometry::kernel::FastKernel;

        let dt: DelaunayTriangulation<FastKernel<f64>, (), (), 2> =
            DelaunayTriangulation::new_empty(FastKernel::new());

        assert_eq!(dt.number_of_vertices(), 0);

        // Note: Currently can't insert into empty triangulation
        // This will be supported when hull extension is implemented
    }

    // =========================================================================
    // with_kernel() tests
    // =========================================================================

    #[test]
    fn test_with_kernel_fast_kernel() {
        use crate::geometry::kernel::FastKernel;

        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        let dt: DelaunayTriangulation<FastKernel<f64>, (), (), 2> =
            DelaunayTriangulation::with_kernel(FastKernel::new(), &vertices).unwrap();

        assert_eq!(dt.number_of_vertices(), 3);
        assert_eq!(dt.number_of_cells(), 1);
    }

    #[test]
    fn test_with_kernel_robust_kernel() {
        use crate::geometry::kernel::RobustKernel;

        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        let dt: DelaunayTriangulation<RobustKernel<f64>, (), (), 2> =
            DelaunayTriangulation::with_kernel(RobustKernel::new(), &vertices).unwrap();

        assert_eq!(dt.number_of_vertices(), 3);
        assert_eq!(dt.number_of_cells(), 1);
    }

    #[test]
    fn test_with_kernel_insufficient_vertices_2d() {
        use crate::geometry::kernel::FastKernel;

        let vertices = vec![vertex!([0.0, 0.0]), vertex!([1.0, 0.0])];

        let result: Result<DelaunayTriangulation<FastKernel<f64>, (), (), 2>, _> =
            DelaunayTriangulation::with_kernel(FastKernel::new(), &vertices);

        assert!(result.is_err());
        match result {
            Err(TriangulationConstructionError::InsufficientVertices { dimension, .. }) => {
                assert_eq!(dimension, 2);
            }
            _ => panic!("Expected InsufficientVertices error"),
        }
    }

    #[test]
    fn test_with_kernel_insufficient_vertices_3d() {
        use crate::geometry::kernel::FastKernel;

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
        ];

        let result: Result<DelaunayTriangulation<FastKernel<f64>, (), (), 3>, _> =
            DelaunayTriangulation::with_kernel(FastKernel::new(), &vertices);

        assert!(result.is_err());
        match result {
            Err(TriangulationConstructionError::InsufficientVertices { dimension, .. }) => {
                assert_eq!(dimension, 3);
            }
            _ => panic!("Expected InsufficientVertices error"),
        }
    }

    #[test]
    fn test_with_kernel_f32_coordinates() {
        use crate::geometry::kernel::FastKernel;

        let vertices = vec![
            vertex!([0.0f32, 0.0f32]),
            vertex!([1.0f32, 0.0f32]),
            vertex!([0.0f32, 1.0f32]),
        ];

        let dt: DelaunayTriangulation<FastKernel<f32>, (), (), 2> =
            DelaunayTriangulation::with_kernel(FastKernel::new(), &vertices).unwrap();

        assert_eq!(dt.number_of_vertices(), 3);
        assert_eq!(dt.number_of_cells(), 1);
    }

    // =========================================================================
    // Query method tests
    // =========================================================================

    #[test]
    fn test_number_of_vertices_minimal_simplex() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();

        assert_eq!(dt.number_of_vertices(), 4);
    }

    #[test]
    fn test_number_of_cells_minimal_simplex() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Minimal 3D simplex has exactly 1 tetrahedron
        assert_eq!(dt.number_of_cells(), 1);
    }

    #[test]
    fn test_number_of_cells_after_insertion() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        assert_eq!(dt.number_of_cells(), 1);

        // Insert interior point - should create 3 triangles
        dt.insert(vertex!([0.3, 0.3])).unwrap();
        assert_eq!(dt.number_of_cells(), 3);
    }

    #[test]
    fn test_dim_returns_correct_dimension() {
        let vertices_2d = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt_2d: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices_2d).unwrap();
        assert_eq!(dt_2d.dim(), 2);

        let vertices_3d = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt_3d: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices_3d).unwrap();
        assert_eq!(dt_3d.dim(), 3);

        let vertices_4d = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0]),
        ];
        let dt_4d: DelaunayTriangulation<_, (), (), 4> =
            DelaunayTriangulation::new(&vertices_4d).unwrap();
        assert_eq!(dt_4d.dim(), 4);
    }

    // =========================================================================
    // insert() tests
    // =========================================================================

    #[test]
    fn test_insert_single_interior_point_2d() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        assert_eq!(dt.number_of_vertices(), 3);
        assert_eq!(dt.number_of_cells(), 1);

        let v_key = dt.insert(vertex!([0.3, 0.3])).unwrap();

        // Verify insertion succeeded
        assert_eq!(dt.number_of_vertices(), 4);
        assert_eq!(dt.number_of_cells(), 3);

        // Verify the returned key can access the vertex
        assert!(dt.tri.tds.get_vertex_by_key(v_key).is_some());
    }

    #[test]
    fn test_insert_multiple_sequential_points_2d() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Insert 3 interior points sequentially
        dt.insert(vertex!([0.3, 0.3])).unwrap();
        assert_eq!(dt.number_of_vertices(), 4);

        dt.insert(vertex!([0.5, 0.2])).unwrap();
        assert_eq!(dt.number_of_vertices(), 5);

        dt.insert(vertex!([0.2, 0.5])).unwrap();
        assert_eq!(dt.number_of_vertices(), 6);

        // All vertices should be present
        assert!(dt.number_of_cells() > 1);
    }

    #[test]
    fn test_insert_multiple_sequential_points_3d() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Insert 3 interior points sequentially (well inside the tetrahedron)
        dt.insert(vertex!([0.1, 0.1, 0.1])).unwrap();
        assert_eq!(dt.number_of_vertices(), 5);

        dt.insert(vertex!([0.15, 0.15, 0.1])).unwrap();
        assert_eq!(dt.number_of_vertices(), 6);

        dt.insert(vertex!([0.1, 0.15, 0.15])).unwrap();
        assert_eq!(dt.number_of_vertices(), 7);

        assert!(dt.number_of_cells() > 1);
    }

    #[test]
    fn test_insert_updates_last_inserted_cell() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Initially no last_inserted_cell
        assert!(dt.last_inserted_cell.is_none());

        // After insertion, should have a cached cell
        dt.insert(vertex!([0.3, 0.3])).unwrap();
        assert!(dt.last_inserted_cell.is_some());
    }

    #[test]
    fn test_new_with_exact_minimum_vertices() {
        // 2D: exactly 3 vertices (minimum for 2D simplex)
        let vertices_2d = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt_2d: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices_2d).unwrap();
        assert_eq!(dt_2d.number_of_vertices(), 3);
        assert_eq!(dt_2d.number_of_cells(), 1);

        // 3D: exactly 4 vertices (minimum for 3D simplex)
        let vertices_3d = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt_3d: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices_3d).unwrap();
        assert_eq!(dt_3d.number_of_vertices(), 4);
        assert_eq!(dt_3d.number_of_cells(), 1);
    }
}
