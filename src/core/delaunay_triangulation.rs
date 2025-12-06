//! Delaunay triangulation layer with incremental insertion.
//!
//! This layer adds Delaunay-specific operations on top of the generic
//! `Triangulation` struct, following CGAL's architecture.

use core::iter::Sum;
use core::ops::{AddAssign, SubAssign};
use std::num::NonZeroUsize;

use num_traits::NumCast;

use crate::core::algorithms::incremental_insertion::InsertionError;
use crate::core::cell::Cell;
use crate::core::facet::{AllFacetsIter, BoundaryFacetsIter};
use crate::core::traits::data_type::DataType;
use crate::core::triangulation::Triangulation;
use crate::core::triangulation_data_structure::{
    CellKey, Tds, TriangulationConstructionError, TriangulationValidationError,
    TriangulationValidationReport, VertexKey,
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
/// # Delaunay Property Note
///
/// The triangulation satisfies **structural validity** (all TDS invariants) but may
/// contain local violations of the empty circumsphere property in rare cases. In this
/// implementation, this arises from using an incremental Bowyer–Watson–style algorithm
/// without topology-changing post-processing (bistellar flips).
///
/// Most triangulations satisfy the Delaunay property. Violations typically occur with:
/// - Near-degenerate point configurations
/// - Specific geometric arrangements
///
/// For applications requiring strict Delaunay guarantees:
/// - Run [`validate_delaunay`](Self::validate_delaunay) in tests or debug builds
/// - Use smaller point sets (violations are rarer)
/// - Filter degenerate configurations when possible
/// - Monitor for bistellar flip implementation (planned for v0.7.0+)
///
/// See: [Issue #120 Investigation](https://github.com/acgetchell/delaunay/blob/main/docs/issue_120_investigation.md)
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
    /// Use this when you want to build a triangulation incrementally by inserting vertices
    /// one at a time. The triangulation will automatically bootstrap itself when you
    /// insert the (D+1)th vertex, creating the initial simplex.
    ///
    /// No type annotations needed! The compiler can infer everything.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// // Start with empty triangulation
    /// let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();
    /// assert_eq!(dt.number_of_vertices(), 0);
    /// assert_eq!(dt.number_of_cells(), 0);
    ///
    /// // Insert vertices one by one
    /// dt.insert(vertex!([0.0, 0.0, 0.0])).unwrap();
    /// dt.insert(vertex!([1.0, 0.0, 0.0])).unwrap();
    /// dt.insert(vertex!([0.0, 1.0, 0.0])).unwrap();
    /// dt.insert(vertex!([0.0, 0.0, 1.0])).unwrap(); // Initial simplex created automatically
    /// assert_eq!(dt.number_of_cells(), 1);
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
    /// to add vertices incrementally. The triangulation will automatically bootstrap itself when
    /// you insert the (D+1)th vertex, creating the initial simplex.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::geometry::kernel::RobustKernel;
    ///
    /// // Start with empty triangulation using robust kernel
    /// let mut dt: DelaunayTriangulation<RobustKernel<f64>, (), (), 4> =
    ///     DelaunayTriangulation::with_empty_kernel(RobustKernel::new());
    /// assert_eq!(dt.number_of_vertices(), 0);
    /// assert_eq!(dt.number_of_cells(), 0);
    ///
    /// // Insert vertices incrementally
    /// dt.insert(vertex!([0.0, 0.0, 0.0, 0.0])).unwrap();
    /// dt.insert(vertex!([1.0, 0.0, 0.0, 0.0])).unwrap();
    /// dt.insert(vertex!([0.0, 1.0, 0.0, 0.0])).unwrap();
    /// dt.insert(vertex!([0.0, 0.0, 1.0, 0.0])).unwrap();
    /// dt.insert(vertex!([0.0, 0.0, 0.0, 1.0])).unwrap(); // Initial simplex created
    /// assert_eq!(dt.number_of_cells(), 1);
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
        let tds = Triangulation::<K, U, V, D>::build_initial_simplex(initial_vertices)?;

        let mut dt = Self {
            tri: Triangulation { kernel, tds },
            last_inserted_cell: None,
        };

        // Insert remaining vertices incrementally
        // Note: Vertices causing geometric degeneracies are automatically skipped
        for vertex in vertices.iter().skip(D + 1) {
            // Skip vertices that fail insertion due to geometric degeneracy
            // The triangulation remains valid (manifold) by skipping problematic vertices
            let _ = dt.insert(*vertex);
            // Errors are logged by insert_transactional(), triangulation stays valid
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
    /// use delaunay::prelude::*;
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

    /// Returns an iterator over all vertices in the triangulation.
    ///
    /// This method provides access to the vertices stored in the underlying
    /// triangulation data structure. The iterator yields `(VertexKey, &Vertex)`
    /// pairs for each vertex in the triangulation.
    ///
    /// # Returns
    ///
    /// An iterator over `(VertexKey, &Vertex<K::Scalar, U, D>)` pairs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
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
    /// for (vertex_key, vertex) in dt.vertices() {
    ///     println!("Vertex {:?} at {:?}", vertex_key, vertex.point());
    /// }
    /// ```
    pub fn vertices(&self) -> impl Iterator<Item = (VertexKey, &Vertex<K::Scalar, U, D>)> {
        self.tri.vertices()
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

    /// Returns a mutable reference to the underlying triangulation data structure.
    ///
    /// This provides mutable access to the purely combinatorial Tds layer for
    /// advanced operations and testing of internal algorithms.
    ///
    /// # Safety
    ///
    /// Modifying the Tds directly can break Delaunay invariants. Use this only
    /// when you know what you're doing (typically in tests or specialized algorithms).
    #[cfg(test)]
    pub(crate) const fn tds_mut(&mut self) -> &mut Tds<K::Scalar, U, V, D> {
        &mut self.tri.tds
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
    /// assert_eq!(hull.number_of_facets(), 4);
    /// ```
    #[must_use]
    pub const fn triangulation(&self) -> &Triangulation<K, U, V, D> {
        &self.tri
    }

    /// Returns a mutable reference to the underlying `Triangulation`.
    ///
    /// # ⚠️ WARNING - ADVANCED USE ONLY
    ///
    /// This method provides direct mutable access to the internal triangulation state.
    /// **Modifying the triangulation through this reference can break Delaunay invariants
    /// and leave the data structure in an inconsistent state.**
    ///
    /// ## When to Use
    ///
    /// This is primarily intended for:
    /// - **Testing internal algorithms** (topology validation, repair mechanisms)
    /// - **Advanced library development** (implementing custom triangulation operations)
    /// - **Research prototyping** (experimenting with new algorithms)
    ///
    /// ## What Can Go Wrong
    ///
    /// Direct mutations can violate critical invariants:
    /// - **Delaunay property**: Cells may no longer satisfy the empty circumsphere condition
    /// - **Manifold topology**: Facets may become over-shared or improperly connected
    /// - **Neighbor consistency**: Cell neighbor pointers may become invalid
    /// - **Hint caching**: Location hints may point to deleted cells
    ///
    /// After direct modification, you should:
    /// 1. Call `detect_local_facet_issues()` and `repair_local_facet_issues()` if you modified topology
    /// 2. Call `is_valid()` to verify structural consistency
    /// 3. Verify Delaunay property manually (if needed)
    ///
    /// ## Safe Alternatives
    ///
    /// For most use cases, prefer these safe, high-level methods:
    /// - [`insert()`](Self::insert) - Add vertices (maintains all invariants)
    /// - [`remove_vertex()`](Self::remove_vertex) - Remove vertices safely
    /// - [`tds()`](Self::tds) - Read-only access to the data structure
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let mut dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// // ⚠️ Advanced use: direct access for testing validation
    /// let tri = dt.triangulation_mut();
    /// // ... perform internal algorithm testing ...
    ///
    /// // Always validate after direct modifications
    /// assert!(dt.is_valid().is_ok());
    /// ```
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // mutable refs from const fn not widely supported
    pub fn triangulation_mut(&mut self) -> &mut Triangulation<K, U, V, D> {
        &mut self.tri
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
    /// This method handles all stages of triangulation construction:
    /// - **Bootstrap (< D+1 vertices)**: Accumulates vertices without creating cells
    /// - **Initial simplex (D+1 vertices)**: Automatically builds the first D-cell
    /// - **Incremental (> D+1 vertices)**: Uses cavity-based insertion with point location
    ///
    /// # Algorithm
    /// 1. Insert vertex into Tds
    /// 2. Check vertex count:
    ///    - If < D+1: Return (bootstrap phase)
    ///    - If == D+1: Build initial simplex from all vertices
    ///    - If > D+1: Continue with steps 3-7
    /// 3. Locate cell containing the point
    /// 4. Find conflict region (cells whose circumspheres contain the point)
    /// 5. Extract cavity boundary
    /// 6. Fill cavity (create new cells)
    /// 7. Wire neighbors locally
    /// 8. Remove conflict cells
    ///
    /// # Errors
    /// Returns error if:
    /// - Duplicate UUID detected
    /// - Initial simplex construction fails (when reaching D+1 vertices)
    /// - Point is on a facet, edge, or vertex (degenerate cases not yet implemented)
    /// - Conflict region computation fails
    /// - Cavity boundary extraction fails
    /// - Cavity filling or neighbor wiring fails
    ///
    /// Note: Points outside the convex hull are handled automatically via hull extension.
    ///
    /// # Examples
    ///
    /// Incremental insertion from empty triangulation:
    ///
    /// ```rust
    /// use delaunay::prelude::DelaunayTriangulation;
    /// use delaunay::vertex;
    ///
    /// // Start with empty triangulation
    /// let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();
    /// assert_eq!(dt.number_of_vertices(), 0);
    /// assert_eq!(dt.number_of_cells(), 0);
    ///
    /// // Insert vertices one by one - bootstrap phase (no cells yet)
    /// dt.insert(vertex!([0.0, 0.0, 0.0])).unwrap();
    /// dt.insert(vertex!([1.0, 0.0, 0.0])).unwrap();
    /// dt.insert(vertex!([0.0, 1.0, 0.0])).unwrap();
    /// assert_eq!(dt.number_of_vertices(), 3);
    /// assert_eq!(dt.number_of_cells(), 0); // Still no cells
    ///
    /// // 4th vertex triggers initial simplex creation
    /// dt.insert(vertex!([0.0, 0.0, 1.0])).unwrap();
    /// assert_eq!(dt.number_of_vertices(), 4);
    /// assert_eq!(dt.number_of_cells(), 1); // First cell created!
    ///
    /// // Further insertions use cavity-based algorithm
    /// dt.insert(vertex!([0.2, 0.2, 0.2])).unwrap();
    /// assert_eq!(dt.number_of_vertices(), 5);
    /// assert!(dt.number_of_cells() > 1);
    /// ```
    ///
    /// Using batch construction (traditional approach):
    ///
    /// ```rust
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
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
        // Fully delegate to Triangulation layer
        // Triangulation handles:
        // - Manifold maintenance (conflict cells, cavity, repairs)
        // - Bootstrap and initial simplex
        // - Location and conflict region computation
        //
        // DelaunayTriangulation adds:
        // - Kernel (provides in-sphere predicate for Delaunay property)
        // - Hint caching for performance
        // - Future: Delaunay property restoration after removal
        let (v_key, hint) = self.tri.insert(vertex, None, self.last_inserted_cell)?;
        self.last_inserted_cell = hint;
        Ok(v_key)
    }

    /// Removes a vertex and retriangulates the resulting cavity using fan triangulation.
    ///
    /// This operation delegates to `Triangulation::remove_vertex()` which:
    /// 1. Finds all cells containing the vertex
    /// 2. Removes those cells (creating a cavity)
    /// 3. Fills the cavity with fan triangulation
    /// 4. Wires neighbors and rebuilds vertex-cell incidence
    /// 5. Removes the vertex
    ///
    /// The triangulation remains topologically valid after removal. However, the fan
    /// triangulation may temporarily violate the Delaunay property in some cases.
    ///
    /// **Future Enhancement**: Delaunay-aware cavity retriangulation will be added when
    /// iterative flip operations are implemented. For now, occasional Delaunay violations
    /// after removal are expected and will be addressed by the global flip refinement system.
    ///
    /// # Arguments
    ///
    /// * `vertex` - Reference to the vertex to remove
    ///
    /// # Returns
    ///
    /// The number of cells that were removed along with the vertex.
    ///
    /// # Errors
    ///
    /// Returns error if the vertex-cell incidence cannot be rebuilt, indicating data structure corruption.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    ///     vertex!([1.0, 1.0]),
    /// ];
    /// let mut dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// // Get a vertex to remove
    /// let vertex_to_remove = dt.vertices().next().unwrap().1.clone();
    /// let cells_before = dt.number_of_cells();
    ///
    /// // Remove the vertex and all cells containing it
    /// let cells_removed = dt.remove_vertex(&vertex_to_remove).unwrap();
    /// println!("Removed {} cells along with the vertex", cells_removed);
    ///
    /// assert!(dt.is_valid().is_ok());
    /// ```
    pub fn remove_vertex(
        &mut self,
        vertex: &Vertex<K::Scalar, U, D>,
    ) -> Result<usize, TriangulationValidationError>
    where
        K::Scalar: CoordinateScalar,
    {
        // Delegate to Triangulation layer
        // Future: Could add Delaunay property restoration after removal
        self.tri.remove_vertex(vertex)
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

    /// Generate a comprehensive validation report for structural invariants.
    ///
    /// This runs all structural validation checks (mappings, duplicate cells,
    /// facet sharing, neighbor consistency) and returns detailed diagnostics.
    ///
    /// **Note**: This does NOT check the Delaunay property. Use
    /// [`validate_delaunay()`](Self::validate_delaunay) separately for geometric validation.
    ///
    /// # Errors
    ///
    /// Returns error if any structural validation check fails.
    pub fn validation_report(&self) -> Result<(), TriangulationValidationReport>
    where
        K::Scalar: CoordinateScalar,
    {
        self.tri.tds.validation_report()
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
                TriangulationValidationError::InconsistentDataStructure {
                    message: format!(
                        "Delaunay property violated: Cell {cell_uuid} (key: {cell_key:?}) violates empty circumsphere invariant"
                    ),
                }
            }
            DelaunayValidationError::TriangulationState { source } => source,
            DelaunayValidationError::InvalidCell { source } => {
                // CellValidationError variants don't contain cell UUID context,
                // so we use nil() for the cell_id field
                TriangulationValidationError::InvalidCell {
                    cell_id: uuid::Uuid::nil(),
                    source,
                }
            }
            DelaunayValidationError::NumericPredicateError {
                cell_key,
                vertex_key,
                source,
            } => {
                // Include cell UUID for better debugging and log correlation
                let cell_uuid = self
                    .tri
                    .tds
                    .cell_uuid_from_key(cell_key)
                    .unwrap_or_else(uuid::Uuid::nil);
                TriangulationValidationError::InconsistentDataStructure {
                    message: format!(
                        "Numeric predicate failure while validating Delaunay property for cell {cell_uuid} (key: {cell_key:?}), vertex {vertex_key:?}: {source}"
                    ),
                }
            }
        })
    }

    /// Fixes invalid facet sharing by removing problematic cells using geometric quality metrics.
    ///
    /// Deprecated: This performs an O(N·D) global scan. Prefer the localized O(k·D) flow:
    /// ```ignore
    /// let affected_cells: Vec<CellKey> = /* cells that were just modified */;
    /// if let Some(issues) = triangulation.detect_local_facet_issues(&affected_cells) {
    ///     triangulation.repair_local_facet_issues(&issues)?;
    /// }
    /// ```
    ///
    /// Delegates to the underlying Triangulation layer.
    ///
    /// # Returns
    ///
    /// Number of cells removed.
    ///
    /// # Errors
    ///
    /// Returns error if facet map cannot be built or topology repair fails.
    #[deprecated(
        since = "0.5.5",
        note = "Use detect_local_facet_issues() + repair_local_facet_issues() for O(k·D) instead of global O(N·D)."
    )]
    pub fn fix_invalid_facet_sharing(&mut self) -> Result<usize, TriangulationValidationError>
    where
        K::Scalar: CoordinateScalar,
    {
        // Delegate to Triangulation layer
        #[allow(deprecated)]
        {
            self.tri.fix_invalid_facet_sharing()
        }
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

/// Custom `Deserialize` implementation for the common case: `FastKernel<f64>` with no custom data.
///
/// This specialization provides convenient deserialization for the most common use case:
/// triangulations with `f64` coordinates, `FastKernel`, and no custom vertex/cell data.
///
/// # Why This Specialization?
///
/// Kernels are stateless and can be reconstructed on deserialization. We only serialize
/// the `Tds` (which contains all the geometric and topological data), then reconstruct
/// the kernel wrapper on deserialization.
///
/// This specialization is limited to `FastKernel<f64>` because:
/// - It's the most common configuration (matches `DelaunayTriangulation::new()` default)
/// - Rust doesn't allow overlapping `impl` blocks for generic types
/// - Custom kernels are rare and can deserialize manually
///
/// # Usage with Custom Kernels
///
/// If you're using a custom kernel (e.g., `RobustKernel`) or custom data types,
/// deserialize the `Tds` directly and reconstruct with [`from_tds()`](Self::from_tds):
///
/// ```rust
/// # use delaunay::prelude::*;
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Create and serialize a triangulation
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let dt = DelaunayTriangulation::<FastKernel<f64>, (), (), 3>::new(&vertices)?;
/// let json = serde_json::to_string(&dt)?;
///
/// // Deserialize with custom kernel
/// let tds: Tds<f64, (), (), 3> = serde_json::from_str(&json)?;
/// let dt_robust = DelaunayTriangulation::from_tds(tds, RobustKernel::new());
/// # Ok(())
/// # }
/// ```
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
/// **Status**: Experimental API. Currently defined but not yet wired into insertion logic.
///
/// # Future Usage
///
/// This policy will be interpreted by insertion algorithms to schedule validation passes.
/// Planned integration points include:
/// - Configuration field on `DelaunayTriangulation` to control validation frequency
/// - Argument to higher-level construction routines (e.g., `new_with_policy`)
/// - Periodic `validate_delaunay()` calls during incremental insertion
///
/// Until wired in, users should call `validate_delaunay()` explicitly as needed.
///
/// # Examples (Future API)
///
/// ```ignore
/// // Once implemented:
/// let mut dt = DelaunayTriangulation::with_policy(
///     vertices,
///     DelaunayCheckPolicy::EveryN(NonZeroUsize::new(100).unwrap())
/// )?;
/// // Validation runs automatically every 100 insertions
/// ```
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
    use crate::geometry::kernel::{FastKernel, RobustKernel};
    use crate::vertex;

    /// Macro to generate comprehensive triangulation construction tests across dimensions.
    ///
    /// This macro generates tests that verify all construction patterns:
    /// 1. **Batch construction** - Creating a simplex with D+1 vertices + incremental insertion
    /// 2. **Bootstrap from empty** - Accumulating vertices until D+1, then auto-creating simplex
    /// 3. **Cavity-based continuation** - Verifying cavity algorithm works after bootstrap
    /// 4. **Equivalence testing** - Bootstrap and batch produce identical structures
    ///
    /// # Usage
    /// ```ignore
    /// test_incremental_insertion!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [0.5, 0.5]);
    /// ```
    macro_rules! test_incremental_insertion {
        ($dim:expr, [$($simplex_coords:expr),+ $(,)?], $interior_point:expr) => {
            pastey::paste! {
                // Test 1: Batch construction with incremental insertion
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

                // Test 2: Bootstrap from empty triangulation
                #[test]
                fn [<test_bootstrap_from_empty_ $dim d>]() {
                    // Start with empty triangulation
                    let mut dt: DelaunayTriangulation<_, (), (), $dim> = DelaunayTriangulation::empty();
                    assert_eq!(dt.number_of_vertices(), 0);
                    assert_eq!(dt.number_of_cells(), 0);

                    let vertices = vec![$(vertex!($simplex_coords)),+];
                    assert_eq!(vertices.len(), $dim + 1, "Test should provide exactly D+1 vertices");

                    // Insert D vertices - should accumulate without creating cells
                    for (i, vertex) in vertices.iter().take($dim).enumerate() {
                        dt.insert(*vertex).unwrap();
                        assert_eq!(dt.number_of_vertices(), i + 1,
                            "{}D: After inserting vertex {}, expected {} vertices", $dim, i, i + 1);
                        assert_eq!(dt.number_of_cells(), 0,
                            "{}D: Should have 0 cells during bootstrap (have {} vertices < D+1)",
                            $dim, i + 1);
                    }

                    // Insert (D+1)th vertex - should trigger initial simplex creation
                    dt.insert(*vertices.last().unwrap()).unwrap();
                    assert_eq!(dt.number_of_vertices(), $dim + 1);
                    assert_eq!(dt.number_of_cells(), 1,
                        "{}D: Should have exactly 1 cell after inserting D+1 vertices", $dim);

                    // Verify triangulation is valid
                    assert!(dt.is_valid().is_ok(),
                        "{}D: Triangulation should be valid after bootstrap", $dim);
                }

                // Test 3: Bootstrap continues with cavity-based insertion
                #[test]
                fn [<test_bootstrap_continues_with_cavity_ $dim d>]() {
                    // Start with empty, bootstrap to initial simplex, then continue with cavity-based
                    let mut dt: DelaunayTriangulation<_, (), (), $dim> = DelaunayTriangulation::empty();

                    let initial_vertices = vec![$(vertex!($simplex_coords)),+];

                    // Bootstrap: insert D+1 vertices
                    for vertex in &initial_vertices {
                        dt.insert(*vertex).unwrap();
                    }
                    assert_eq!(dt.number_of_cells(), 1);

                    // Continue with cavity-based insertion (vertex D+2 onward)
                    dt.insert(vertex!($interior_point)).unwrap();
                    assert_eq!(dt.number_of_vertices(), $dim + 2);
                    assert!(dt.number_of_cells() > 1,
                        "{}D: Should have multiple cells after cavity-based insertion", $dim);

                    // Verify triangulation remains valid
                    assert!(dt.is_valid().is_ok());
                }

                // Test 4: Bootstrap equivalent to batch construction
                #[test]
                fn [<test_bootstrap_equivalent_to_batch_ $dim d>]() {
                    // Compare bootstrap path vs batch construction
                    let vertices = vec![$(vertex!($simplex_coords)),+];

                    // Path A: Bootstrap from empty
                    let mut dt_bootstrap: DelaunayTriangulation<_, (), (), $dim> =
                        DelaunayTriangulation::empty();
                    for vertex in &vertices {
                        dt_bootstrap.insert(*vertex).unwrap();
                    }

                    // Path B: Batch construction
                    let dt_batch: DelaunayTriangulation<_, (), (), $dim> =
                        DelaunayTriangulation::new(&vertices).unwrap();

                    // Both should produce identical structure
                    assert_eq!(dt_bootstrap.number_of_vertices(), dt_batch.number_of_vertices(),
                        "{}D: Bootstrap and batch should have same vertex count", $dim);
                    assert_eq!(dt_bootstrap.number_of_cells(), dt_batch.number_of_cells(),
                        "{}D: Bootstrap and batch should have same cell count", $dim);

                    // Both should be valid
                    assert!(dt_bootstrap.is_valid().is_ok());
                    assert!(dt_batch.is_valid().is_ok());
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
    fn test_empty_supports_incremental_insertion() {
        // Verify empty triangulation supports incremental insertion via bootstrap
        let mut dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::empty();
        assert_eq!(dt.number_of_vertices(), 0);

        // Can now insert into empty triangulation - bootstrap phase
        dt.insert(vertex!([0.0, 0.0])).unwrap();
        dt.insert(vertex!([1.0, 0.0])).unwrap();
        assert_eq!(dt.number_of_cells(), 0); // Still in bootstrap

        dt.insert(vertex!([0.0, 1.0])).unwrap();
        assert_eq!(dt.number_of_cells(), 1); // Initial simplex created
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

    #[test]
    fn test_bootstrap_with_custom_kernel() {
        // Verify bootstrap works with RobustKernel
        let mut dt: DelaunayTriangulation<RobustKernel<f64>, (), (), 3> =
            DelaunayTriangulation::with_empty_kernel(RobustKernel::new());

        assert_eq!(dt.number_of_vertices(), 0);

        // Bootstrap with robust predicates
        dt.insert(vertex!([0.0, 0.0, 0.0])).unwrap();
        dt.insert(vertex!([1.0, 0.0, 0.0])).unwrap();
        dt.insert(vertex!([0.0, 1.0, 0.0])).unwrap();
        assert_eq!(dt.number_of_cells(), 0); // Still bootstrapping

        dt.insert(vertex!([0.0, 0.0, 1.0])).unwrap();
        assert_eq!(dt.number_of_cells(), 1); // Initial simplex created

        assert!(dt.is_valid().is_ok());
    }
}
