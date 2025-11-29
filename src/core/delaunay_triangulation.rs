//! Delaunay triangulation layer with incremental insertion.
//!
//! This layer adds Delaunay-specific operations on top of the generic
//! `Triangulation` struct, following CGAL's architecture.

use core::iter::Sum;
use core::ops::{AddAssign, SubAssign};
use std::num::NonZeroUsize;

use num_traits::NumCast;

use crate::core::algorithms::incremental_insertion::{
    InsertionError, extend_hull, fill_cavity, wire_cavity_neighbors,
};
use crate::core::algorithms::locate::{
    LocateResult, extract_cavity_boundary, find_conflict_region, locate,
};
use crate::core::cell::Cell;
use crate::core::collections::{MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer};
use crate::core::facet::{AllFacetsIter, BoundaryFacetsIter};
use crate::core::traits::data_type::DataType;
use crate::core::triangulation::Triangulation;
use crate::core::triangulation_data_structure::{
    CellKey, Tds, TriangulationConstructionError, TriangulationValidationError,
    TriangulationValidationReport, ValidationOptions, VertexKey,
};
use crate::core::util::DelaunayValidationError;
use crate::core::vertex::Vertex;
use crate::geometry::kernel::{FastKernel, Kernel};
use crate::geometry::traits::coordinate::CoordinateScalar;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Delaunay triangulation with incremental insertion support.
///
/// # Type Parameters
/// - `K`: Geometric kernel implementing predicates
/// - `U`: User data type for vertices
/// - `V`: User data type for cells
/// - `D`: Dimension of the triangulation
///
/// # Implementation
///
/// Uses efficient incremental cavity-based insertion algorithm:
/// - ✅ Point location (facet walking) - [`locate`]
/// - ✅ Conflict region computation (local BFS) - [`find_conflict_region`]
/// - ✅ Cavity extraction and filling - [`extract_cavity_boundary`], [`fill_cavity`]
/// - ✅ Local neighbor wiring - [`wire_cavity_neighbors`]
/// - ✅ Hull extension for outside points - [`extend_hull`]
///
/// [`locate`]: crate::core::algorithms::locate::locate
/// [`find_conflict_region`]: crate::core::algorithms::locate::find_conflict_region
/// [`extract_cavity_boundary`]: crate::core::algorithms::locate::extract_cavity_boundary
/// [`fill_cavity`]: crate::core::algorithms::incremental_insertion::fill_cavity
/// [`wire_cavity_neighbors`]: crate::core::algorithms::incremental_insertion::wire_cavity_neighbors
/// [`extend_hull`]: crate::core::algorithms::incremental_insertion::extend_hull
#[derive(Clone, Debug)]
pub struct DelaunayTriangulation<K, U, V, const D: usize>
where
    K: Kernel<D>,
    U: DataType,
    V: DataType,
{
    /// The underlying generic triangulation.
    pub(crate) tri: Triangulation<K, U, V, D>,
    /// Hint for next `locate()` call (last inserted cell)
    last_inserted_cell: Option<CellKey>,
}

// Most common case: f64 with FastKernel, no vertex or cell data
impl<const D: usize> DelaunayTriangulation<FastKernel<f64>, (), (), D> {
    /// Create a Delaunay triangulation from vertices with no data (most common case).
    ///
    /// This is the simplest constructor for the most common use case:
    /// - f64 coordinates
    /// - Fast floating-point predicates  
    /// - No vertex data
    /// - No cell data
    ///
    /// No type annotations needed! The compiler can infer everything.
    ///
    /// # Errors
    /// Returns error if initial simplex cannot be constructed or insertion fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::DelaunayTriangulation;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// // No type annotations needed!
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// assert_eq!(dt.number_of_vertices(), 4);
    /// ```
    pub fn new(vertices: &[Vertex<f64, (), D>]) -> Result<Self, TriangulationConstructionError> {
        Self::with_kernel(FastKernel::<f64>::new(), vertices)
    }

    /// Create an empty Delaunay triangulation with no data (most common case).
    ///
    /// No type annotations needed! The compiler can infer everything.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::DelaunayTriangulation;
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();
    /// assert_eq!(dt.number_of_vertices(), 0);
    /// ```
    #[must_use]
    pub fn empty() -> Self {
        Self::with_empty_kernel(FastKernel::<f64>::new())
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
    /// Create an empty Delaunay triangulation with the given kernel (advanced usage).
    ///
    /// Most users should use [`DelaunayTriangulation::empty()`] instead, which uses fast predicates
    /// by default. Use this method only if you need custom coordinate precision or specialized kernels.
    ///
    /// This creates a triangulation with no vertices or cells. Use [`insert`](Self::insert)
    /// to add vertices incrementally.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
    /// use delaunay::geometry::kernel::RobustKernel;
    ///
    /// let dt: DelaunayTriangulation<RobustKernel<f64>, (), (), 4> =
    ///     DelaunayTriangulation::with_empty_kernel(RobustKernel::new());
    /// assert_eq!(dt.number_of_vertices(), 0);
    /// assert_eq!(dt.number_of_cells(), 0);
    /// ```
    #[must_use]
    pub fn with_empty_kernel(kernel: K) -> Self {
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
    /// 1. Build initial simplex (D+1 vertices) directly
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

        // Build initial simplex directly (no Bowyer-Watson)
        let initial_vertices = &vertices[..=D];
        let tds = Self::build_initial_simplex(initial_vertices)?;

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

    /// Build initial D-simplex from D+1 vertices without using Bowyer-Watson.
    ///
    /// This creates a Tds with a single cell containing all D+1 vertices,
    /// with no neighbor relationships (all boundary facets).
    ///
    /// # Arguments
    /// - `vertices`: Exactly D+1 vertices to form the initial simplex
    ///
    /// # Returns
    /// A Tds containing one D-cell with all vertices, ready for incremental insertion.
    ///
    /// # Errors
    /// Returns error if:
    /// - Wrong number of vertices (must be exactly D+1)
    /// - Vertex or cell insertion fails
    /// - Duplicate UUIDs detected
    fn build_initial_simplex(
        vertices: &[Vertex<K::Scalar, U, D>],
    ) -> Result<Tds<K::Scalar, U, V, D>, TriangulationConstructionError>
    where
        K::Scalar: CoordinateScalar,
    {
        if vertices.len() != D + 1 {
            return Err(TriangulationConstructionError::InsufficientVertices {
                dimension: D,
                source: crate::core::cell::CellValidationError::InsufficientVertices {
                    actual: vertices.len(),
                    expected: D + 1,
                    dimension: D,
                },
            });
        }

        // Create empty Tds
        let mut tds = Tds::empty();

        // Insert all vertices and collect their keys
        let mut vertex_keys = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
        for vertex in vertices {
            let vkey = tds.insert_vertex_with_mapping(*vertex)?;
            vertex_keys.push(vkey);
        }

        // Create single D-cell from all vertices
        // Note: Cell::new() handles vertex ordering/orientation internally
        let cell = Cell::new(vertex_keys, None).map_err(|e| {
            TriangulationConstructionError::FailedToCreateCell {
                message: format!("Failed to create initial simplex cell: {e}"),
            }
        })?;

        // Insert the cell
        let _cell_key = tds.insert_cell_with_mapping(cell)?;

        // Assign incident cells to vertices (each vertex points to this one cell)
        // This is required for proper Tds structure
        tds.assign_incident_cells()
            .map_err(TriangulationConstructionError::ValidationError)?;

        // Cache the initial cell key for first insert() hint
        // (This will be returned but caller will set last_inserted_cell)

        Ok(tds)
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

    /// Returns an iterator over all cells in the triangulation.
    ///
    /// This method provides access to the cells stored in the underlying
    /// triangulation data structure. The iterator yields `(CellKey, &Cell)`
    /// pairs for each cell in the triangulation.
    ///
    /// # Returns
    ///
    /// An iterator over `(CellKey, &Cell<K::Scalar, U, V, D>)` pairs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// for (cell_key, cell) in dt.cells() {
    ///     println!("Cell {:?} has {} vertices", cell_key, cell.number_of_vertices());
    /// }
    /// ```
    pub fn cells(&self) -> impl Iterator<Item = (CellKey, &Cell<K::Scalar, U, V, D>)> {
        self.tri.tds.cells()
    }

    /// Returns a reference to the underlying triangulation data structure.
    ///
    /// This provides access to the purely combinatorial Tds layer for
    /// advanced operations and performance testing.
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
    /// let tds = dt.tds();
    /// assert_eq!(tds.number_of_vertices(), 5);
    /// ```
    #[must_use]
    pub const fn tds(&self) -> &Tds<K::Scalar, U, V, D> {
        &self.tri.tds
    }

    /// Returns a reference to the underlying `Triangulation` (kernel + tds).
    ///
    /// This is useful when you need to pass the triangulation to methods that
    /// expect a `&Triangulation`, such as `ConvexHull::from_triangulation()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
    /// use delaunay::geometry::algorithms::convex_hull::ConvexHull;
    /// use delaunay::vertex;
    ///
    /// let vertices: Vec<_> = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();
    /// assert_eq!(hull.facet_count(), 4);
    /// ```
    #[must_use]
    pub const fn triangulation(&self) -> &Triangulation<K, U, V, D> {
        &self.tri
    }

    /// Returns an iterator over all facets in the triangulation.
    ///
    /// Delegates to the underlying `Triangulation` layer. This provides
    /// efficient access to all facets without pre-allocating a vector.
    ///
    /// # Returns
    ///
    /// An iterator yielding `FacetView` objects for all facets.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// let facet_count = dt.facets().count();
    /// assert_eq!(facet_count, 4); // Tetrahedron has 4 facets
    /// ```
    pub fn facets(&self) -> AllFacetsIter<'_, K::Scalar, U, V, D> {
        self.tri.facets()
    }

    /// Returns an iterator over boundary (hull) facets in the triangulation.
    ///
    /// Boundary facets are those that belong to exactly one cell. This method
    /// computes the facet-to-cells map internally for convenience.
    ///
    /// # Returns
    ///
    /// An iterator yielding `FacetView` objects for boundary facets only.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// let boundary_count = dt.boundary_facets().count();
    /// assert_eq!(boundary_count, 4); // All facets are on boundary
    /// ```
    pub fn boundary_facets(&self) -> BoundaryFacetsIter<'_, K::Scalar, U, V, D> {
        self.tri.boundary_facets()
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
        // Copy the point before borrowing tds mutably
        let point = *self
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
            &point,
            self.last_inserted_cell,
        )?;

        // Handle different location results
        match location {
            LocateResult::InsideCell(start_cell) => {
                // Interior vertex: use cavity-based insertion

                // 3. Find conflict region
                let conflict_cells =
                    find_conflict_region(&self.tri.tds, &self.tri.kernel, &point, start_cell)?;

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
            LocateResult::Outside => {
                // Exterior vertex: extend convex hull
                let new_cells = extend_hull(&mut self.tri.tds, &self.tri.kernel, v_key, &point)?;

                // Cache last inserted cell for next locate hint
                self.last_inserted_cell = new_cells.first().copied();

                Ok(v_key)
            }
            _ => {
                // TODO: Handle other cases (OnFacet, OnEdge, OnVertex)
                Err(InsertionError::CavityFilling {
                    message: format!("Unhandled location result: {location:?}"),
                })
            }
        }
    }

    /// Validate the combinatorial structure of the triangulation.
    ///
    /// This validates the underlying Tds topology including:
    /// - Vertex and cell mapping consistency
    /// - Neighbor relationships
    /// - Facet sharing
    /// - No duplicate cells
    ///
    /// # Errors
    ///
    /// Returns error if any structural invariant is violated.
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
    /// let dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// // Verify triangulation structure is valid
    /// assert!(dt.is_valid().is_ok());
    /// ```
    pub fn is_valid(&self) -> Result<(), TriangulationValidationError>
    where
        K::Scalar: CoordinateScalar,
    {
        self.tri.tds.is_valid()
    }

    /// Validate vertex mapping consistency.
    ///
    /// Checks that all vertex keys in cells correspond to valid vertices
    /// and that the incident cell pointers are correct.
    ///
    /// # Errors
    ///
    /// Returns error if vertex mappings are inconsistent.
    pub fn validate_vertex_mappings(&self) -> Result<(), TriangulationValidationError> {
        self.tri.tds.validate_vertex_mappings()
    }

    /// Create a `DelaunayTriangulation` from a deserialized `Tds` with a default kernel.
    ///
    /// This is useful when you've serialized just the `Tds` and want to reconstruct
    /// the `DelaunayTriangulation` with default kernel settings.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
    /// use delaunay::core::triangulation_data_structure::Tds;
    /// use delaunay::geometry::kernel::FastKernel;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// // Serialize just the Tds
    /// let json = serde_json::to_string(dt.tds()).unwrap();
    ///
    /// // Deserialize Tds and reconstruct DelaunayTriangulation
    /// let tds: Tds<f64, (), (), 4> = serde_json::from_str(&json).unwrap();
    /// let reconstructed = DelaunayTriangulation::from_tds(tds, FastKernel::new());
    /// assert_eq!(reconstructed.number_of_vertices(), 5);
    /// ```
    #[must_use]
    pub const fn from_tds(tds: Tds<K::Scalar, U, V, D>, kernel: K) -> Self {
        Self {
            tri: Triangulation { kernel, tds },
            last_inserted_cell: None,
        }
    }

    /// Validate cell mapping consistency.
    ///
    /// Checks that all cell neighbor pointers are valid and correspond
    /// to existing cells.
    ///
    /// # Errors
    ///
    /// Returns error if cell mappings are inconsistent.
    pub fn validate_cell_mappings(&self) -> Result<(), TriangulationValidationError> {
        self.tri.tds.validate_cell_mappings()
    }

    /// Validate that there are no duplicate cells.
    ///
    /// Checks that no two cells share the exact same set of vertices.
    ///
    /// # Errors
    ///
    /// Returns error if duplicate cells are found.
    pub fn validate_no_duplicate_cells(&self) -> Result<(), TriangulationValidationError>
    where
        K::Scalar: CoordinateScalar,
    {
        self.tri.tds.validate_no_duplicate_cells()
    }

    /// Validate facet sharing invariants.
    ///
    /// Checks that each facet is shared by at most 2 cells (1 for boundary facets,
    /// 2 for interior facets).
    ///
    /// # Errors
    ///
    /// Returns error if facet sharing invariants are violated.
    pub fn validate_facet_sharing(&self) -> Result<(), TriangulationValidationError>
    where
        K::Scalar: CoordinateScalar,
    {
        self.tri.tds.validate_facet_sharing()
    }

    /// Validate neighbor consistency.
    ///
    /// Checks that neighbor relationships are mutual and reference shared facets.
    ///
    /// # Errors
    ///
    /// Returns error if neighbor relationships are inconsistent.
    pub fn validate_neighbors(&self) -> Result<(), TriangulationValidationError>
    where
        K::Scalar: CoordinateScalar,
    {
        self.tri.tds.validate_neighbors()
    }

    /// Generate a comprehensive validation report.
    ///
    /// This runs all validation checks and returns detailed diagnostics.
    /// Use this for debugging triangulation issues.
    ///
    /// # Parameters
    ///
    /// - `options`: Configuration for validation (e.g., whether to check Delaunay property)
    ///
    /// # Errors
    ///
    /// Returns error if any validation check fails.
    pub fn validation_report(
        &self,
        options: ValidationOptions,
    ) -> Result<(), TriangulationValidationReport>
    where
        K::Scalar: CoordinateScalar,
    {
        self.tri.tds.validation_report(options)
    }

    /// Validate that the triangulation satisfies the Delaunay property.
    ///
    /// This checks that no vertex is inside the circumsphere of any cell,
    /// which is the defining property of a Delaunay triangulation.
    ///
    /// # Performance Warning
    ///
    /// This is an **O(N×V)** operation where N is the number of cells and V is the
    /// number of vertices. Use primarily for testing and validation, not in hot paths.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - A cell violates the Delaunay property
    /// - The triangulation has structural issues
    /// - Geometric predicates fail
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
    /// let dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// // Verify Delaunay property holds
    /// assert!(dt.validate_delaunay().is_ok());
    /// ```
    pub fn validate_delaunay(&self) -> Result<(), TriangulationValidationError>
    where
        K::Scalar: CoordinateScalar,
    {
        crate::core::util::is_delaunay(&self.tri.tds).map_err(|err| match err {
            DelaunayValidationError::DelaunayViolation { cell_key } => {
                let cell_uuid = self
                    .tri
                    .tds
                    .cell_uuid_from_key(cell_key)
                    .unwrap_or_else(uuid::Uuid::nil);
                TriangulationValidationError::DelaunayViolation {
                    message: format!(
                        "Cell {cell_uuid} (key: {cell_key:?}) violates Delaunay property"
                    ),
                }
            }
            DelaunayValidationError::TriangulationState { source } => source,
            DelaunayValidationError::InvalidCell { source } => {
                TriangulationValidationError::InvalidCell {
                    cell_id: uuid::Uuid::nil(),
                    source,
                }
            }
            DelaunayValidationError::NumericPredicateError {
                cell_key,
                vertex_key,
                source,
            } => TriangulationValidationError::InconsistentDataStructure {
                message: format!(
                    "Numeric predicate failure while validating Delaunay property for cell {cell_key:?}, vertex {vertex_key:?}: {source}"
                ),
            },
        })
    }
}

// Custom Serialize implementation that only serializes the Tds
impl<K, U, V, const D: usize> Serialize for DelaunayTriangulation<K, U, V, D>
where
    K: Kernel<D>,
    U: DataType,
    V: DataType,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Only serialize the Tds; kernel can be reconstructed on deserialization
        self.tri.tds.serialize(serializer)
    }
}

// Custom Deserialize for the common case: FastKernel<f64>
impl<'de, const D: usize> Deserialize<'de> for DelaunayTriangulation<FastKernel<f64>, (), (), D>
where
    Tds<f64, (), (), D>: Deserialize<'de>,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
    {
        let tds = Tds::deserialize(deserializer)?;
        Ok(Self::from_tds(tds, FastKernel::new()))
    }
}

/// Policy controlling when global Delaunay validation runs during triangulation.
///
/// This policy is interpreted by insertion algorithms to schedule validation passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DelaunayCheckPolicy {
    /// Run global Delaunay validation only at the end of triangulation.
    #[default]
    EndOnly,
    /// Run global Delaunay validation after every N successful insertions,
    /// in addition to a final pass at the end.
    EveryN(NonZeroUsize),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vertex::VertexBuilder;
    use crate::geometry::kernel::{FastKernel, RobustKernel};
    use crate::geometry::point::Point;
    use crate::geometry::traits::coordinate::Coordinate;
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
    // empty() / with_empty_kernel() tests
    // =========================================================================

    #[test]
    fn test_empty_creates_empty_triangulation() {
        let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();

        assert_eq!(dt.number_of_vertices(), 0);
        assert_eq!(dt.number_of_cells(), 0);
        // dim() returns -1 for empty triangulation
        assert_eq!(dt.dim(), -1);
    }

    #[test]
    fn test_empty_then_construct() {
        let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::empty();

        assert_eq!(dt.number_of_vertices(), 0);

        // Note: Currently can't insert into empty triangulation
        // This will be supported when hull extension is implemented
    }

    // =========================================================================
    // with_kernel() tests
    // =========================================================================

    #[test]
    fn test_with_kernel_fast_kernel() {
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
    // build_initial_simplex() tests
    // =========================================================================

    /// Macro to generate `build_initial_simplex` tests across dimensions.
    ///
    /// This macro generates tests that verify `build_initial_simplex` by:
    /// 1. Creating D+1 affinely independent vertices
    /// 2. Calling `build_initial_simplex` directly
    /// 3. Verifying the Tds has correct structure (vertices, cells, dimension)
    ///
    /// # Usage
    /// ```ignore
    /// test_build_initial_simplex!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    /// ```
    macro_rules! test_build_initial_simplex {
        ($dim:expr, [$($simplex_coords:expr),+ $(,)?]) => {
            pastey::paste! {
                #[test]
                fn [<test_build_initial_simplex_ $dim d>]() {
                    // Build initial simplex (D+1 vertices)
                    let vertices: Vec<Vertex<f64, (), $dim>> = vec![
                        $(vertex!($simplex_coords)),+
                    ];

                    let expected_vertices = vertices.len();
                    assert_eq!(expected_vertices, $dim + 1,
                        "Test must provide exactly D+1 vertices for {}D simplex", $dim);

                    let tds = DelaunayTriangulation::<FastKernel<f64>, (), (), $dim>::build_initial_simplex(&vertices)
                        .unwrap();

                    // Verify structure
                    assert_eq!(tds.number_of_vertices(), expected_vertices,
                        "{}D: Expected {} vertices", $dim, expected_vertices);
                    assert_eq!(tds.number_of_cells(), 1,
                        "{}D: Expected 1 cell", $dim);
                    assert_eq!(tds.dim(), $dim as i32,
                        "{}D: Expected dimension {}", $dim, $dim);

                    // Verify all vertices are present
                    assert_eq!(tds.vertices().count(), expected_vertices,
                        "{}D: All vertices should be in Tds", $dim);

                    // Verify the single cell has correct number of vertices
                    let (_, cell) = tds.cells().next()
                        .expect(&format!("{}D: Should have exactly one cell", $dim));
                    assert_eq!(cell.number_of_vertices(), expected_vertices,
                        "{}D: Cell should have {} vertices", $dim, expected_vertices);

                    // Verify incident cells are assigned
                    for (_, vertex) in tds.vertices() {
                        assert!(vertex.incident_cell.is_some(),
                            "{}D: All vertices should have incident cell assigned", $dim);
                    }

                    // Verify initial simplex has no neighbors (all boundary facets)
                    if let Some(neighbors) = cell.neighbors() {
                        assert!(neighbors.iter().all(|n| n.is_none()),
                            "{}D: Initial simplex should have no neighbors (all boundary)", $dim);
                    }
                }
            }
        };
    }

    // 2D: Triangle
    test_build_initial_simplex!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);

    // 3D: Tetrahedron
    test_build_initial_simplex!(
        3,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
    );

    // 4D: 4-simplex
    test_build_initial_simplex!(
        4,
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    );

    // 5D: 5-simplex
    test_build_initial_simplex!(
        5,
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ]
    );

    #[test]
    fn test_build_initial_simplex_insufficient_vertices() {
        // Try to build 3D simplex with only 2 vertices (need 4)
        let vertices = vec![vertex!([0.0, 0.0, 0.0]), vertex!([1.0, 0.0, 0.0])];

        let result =
            DelaunayTriangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices);

        assert!(result.is_err());
        match result {
            Err(TriangulationConstructionError::InsufficientVertices { dimension, .. }) => {
                assert_eq!(dimension, 3);
            }
            _ => panic!("Expected InsufficientVertices error"),
        }
    }

    #[test]
    fn test_build_initial_simplex_too_many_vertices() {
        // Try to build 2D simplex with 4 vertices (need exactly 3)
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
            vertex!([0.5, 0.5]),
        ];

        let result =
            DelaunayTriangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices);

        assert!(result.is_err());
        match result {
            Err(TriangulationConstructionError::InsufficientVertices { .. }) => {}
            _ => panic!("Expected InsufficientVertices error for wrong count"),
        }
    }

    #[test]
    fn test_build_initial_simplex_with_user_data() {
        // Build vertices with user data
        let v1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0]))
            .data(42_usize)
            .build()
            .unwrap();
        let v2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0]))
            .data(43_usize)
            .build()
            .unwrap();
        let v3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0]))
            .data(44_usize)
            .build()
            .unwrap();

        let vertices = vec![v1, v2, v3];
        let tds = DelaunayTriangulation::<FastKernel<f64>, usize, (), 2>::build_initial_simplex(
            &vertices,
        )
        .unwrap();

        assert_eq!(tds.number_of_vertices(), 3);
        assert_eq!(tds.number_of_cells(), 1);

        // Verify user data is preserved
        let data_values: Vec<_> = tds
            .vertices()
            .filter_map(|(_, v)| v.data.as_ref())
            .copied()
            .collect();
        assert_eq!(data_values.len(), 3);
        assert!(data_values.contains(&42));
        assert!(data_values.contains(&43));
        assert!(data_values.contains(&44));
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

    #[test]
    fn test_tds_accessor_provides_readonly_access() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Access TDS via immutable reference
        let tds = dt.tds();
        assert_eq!(tds.number_of_vertices(), 3);
        assert_eq!(tds.number_of_cells(), 1);

        // Verify we can call other TDS methods
        assert!(tds.is_valid().is_ok());
        assert!(tds.cell_keys().next().is_some());
    }

    #[test]
    fn test_internal_tds_access() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();

        assert_eq!(dt.number_of_vertices(), 4);

        // Internal code can access TDS directly for mutations
        let tds = &mut dt.tri.tds;
        assert_eq!(tds.number_of_vertices(), 4);
        assert_eq!(tds.number_of_cells(), 1);

        // Can call mutating methods like remove_duplicate_cells
        let result = tds.remove_duplicate_cells();
        assert!(result.is_ok());
    }

    #[test]
    fn test_tds_accessor_reflects_insertions() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Before insertion
        assert_eq!(dt.tds().number_of_vertices(), 3);

        // Insert a new vertex
        dt.insert(vertex!([0.3, 0.3])).unwrap();

        // After insertion, TDS accessor reflects the change
        assert_eq!(dt.tds().number_of_vertices(), 4);
        assert!(dt.tds().number_of_cells() > 1);
    }

    #[test]
    fn test_tds_accessors_maintain_validation_invariants() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0]),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 4> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Verify TDS is valid through accessor
        assert!(dt.tds().is_valid().is_ok());

        // Insert additional vertex
        dt.insert(vertex!([0.2, 0.2, 0.2, 0.2])).unwrap();

        // TDS should still be valid after mutation
        assert!(dt.tds().is_valid().is_ok());
        assert!(dt.tds().validate_vertex_mappings().is_ok());
        assert!(dt.tds().validate_cell_mappings().is_ok());
    }
}
