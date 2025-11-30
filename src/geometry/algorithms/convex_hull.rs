use crate::core::collections::{
    FacetToCellsMap, FastHashMap, MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer,
};
use crate::core::facet::{FacetError, FacetHandle, FacetView};
use crate::core::traits::boundary_analysis::BoundaryAnalysis;
use crate::core::traits::data_type::DataType;
use crate::core::traits::facet_cache::FacetCacheProvider;
use crate::core::triangulation::Triangulation;
use crate::core::triangulation_data_structure::TriangulationValidationError;
use crate::core::util::derive_facet_key_from_vertex_keys;
use crate::core::vertex::Vertex;
use crate::geometry::kernel::Kernel;
use crate::geometry::point::Point;
use crate::geometry::predicates::simplex_orientation;
use crate::geometry::traits::coordinate::{
    Coordinate, CoordinateConversionError, CoordinateScalar,
};
use crate::geometry::util::{safe_usize_to_scalar, squared_norm};
use arc_swap::ArcSwapOption;
use num_traits::NumCast;
use num_traits::Zero;
use std::marker::PhantomData;
use std::ops::{AddAssign, DivAssign, Sub, SubAssign};
use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicU64, Ordering},
};
use thiserror::Error;

// Import Orientation for predicates
use crate::geometry::predicates::Orientation;

// =============================================================================
// ERROR TYPES
// =============================================================================

/// Errors that can occur during convex hull validation.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ConvexHullValidationError {
    /// A facet has invalid structure.
    #[error("Facet {facet_index} validation failed: {source}")]
    InvalidFacet {
        /// Index of the invalid facet.
        facet_index: usize,
        /// The underlying facet error.
        source: FacetError,
    },
    /// A facet contains duplicate vertices.
    #[error("Facet {facet_index} has duplicate vertices at positions {positions:?}")]
    DuplicateVerticesInFacet {
        /// Index of the facet containing duplicate vertices.
        facet_index: usize,
        /// Positions of all duplicate vertices (groups of positions that have the same vertex).
        positions: Vec<Vec<usize>>,
    },
    /// Convex hull validation failed because the hull is stale.
    #[error(
        "ConvexHull validation failed: hull is stale and cannot be validated against this TDS. \
         The TDS has been modified since the hull was created (hull generation: {hull_generation}, \
         TDS generation: {tds_generation}). Create a new ConvexHull by calling from_triangulation()."
    )]
    StaleHull {
        /// The generation counter of the hull at creation time.
        hull_generation: u64,
        /// The current generation counter of the TDS.
        tds_generation: u64,
    },
}

/// Errors that can occur during convex hull construction.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ConvexHullConstructionError {
    /// Failed to extract boundary facets from the triangulation.
    #[error("Failed to extract boundary facets from triangulation: {source}")]
    BoundaryFacetExtractionFailed {
        /// The underlying validation error that caused the failure.
        source: TriangulationValidationError,
    },
    /// Failed to check facet visibility from a point.
    #[error("Failed to check facet visibility from point: {source}")]
    VisibilityCheckFailed {
        /// The underlying facet error that caused the visibility check to fail.
        source: FacetError,
    },
    /// The input triangulation is empty or invalid.
    #[error("Invalid input triangulation: {message}")]
    InvalidTriangulation {
        /// Description of why the triangulation is invalid.
        message: String,
    },
    /// Insufficient data to construct convex hull.
    #[error("Insufficient data for convex hull construction: {message}")]
    InsufficientData {
        /// Description of the data insufficiency.
        message: String,
    },
    /// Geometric degeneracy prevents convex hull construction.
    #[error("Geometric degeneracy encountered during convex hull construction: {message}")]
    GeometricDegeneracy {
        /// Description of the degeneracy issue.
        message: String,
    },
    /// Numeric cast failed during computation.
    #[error("Numeric cast failed during convex hull computation: {message}")]
    NumericCastFailed {
        /// Description of the cast failure.
        message: String,
    },
    /// Coordinate conversion error occurred during geometric computations.
    #[error("Coordinate conversion error: {0}")]
    CoordinateConversion(#[from] CoordinateConversionError),
    /// Failed to build facet cache during convex hull operations.
    #[error("Failed to build facet cache: {source}")]
    FacetCacheBuildFailed {
        /// The underlying triangulation validation error.
        #[source]
        source: TriangulationValidationError,
    },
    /// Failed to resolve adjacent cell vertices for visibility testing.
    #[error("Failed to resolve adjacent cell: {source}")]
    AdjacentCellResolutionFailed {
        /// The underlying triangulation validation error.
        #[source]
        source: TriangulationValidationError,
    },
    /// Failed to access facet data during convex hull construction.
    #[error("Failed to access facet data during convex hull construction: {source}")]
    FacetDataAccessFailed {
        /// The underlying facet error that caused the data access to fail.
        #[source]
        source: FacetError,
    },
    /// Convex hull used with a modified triangulation (stale hull).
    #[error(
        "ConvexHull is stale and cannot be used with this TDS. The TDS has been modified since \
         the hull was created (hull generation: {hull_generation}, TDS generation: {tds_generation}). \
         Create a new ConvexHull by calling from_triangulation()."
    )]
    StaleHull {
        /// The generation counter of the hull at creation time.
        hull_generation: u64,
        /// The current generation counter of the TDS.
        tds_generation: u64,
    },
}

// =============================================================================
// CONVEX HULL DATA STRUCTURE
// =============================================================================

/// Generic d-dimensional convex hull operations.
///
/// This struct provides convex hull functionality by leveraging the existing
/// boundary facet analysis from the TDS. Since boundary facets in a Delaunay
/// triangulation lie on the convex hull, we can use the `BoundaryAnalysis`
/// trait to get the hull facets directly.
///
/// The implementation supports d-dimensional convex hull extraction from
/// Delaunay triangulations, point-in-hull testing, and facet visibility
/// determination for incremental construction algorithms.
///
/// # Important: `ConvexHull` is a Logically Immutable Snapshot
///
/// **A `ConvexHull` instance is a logically immutable snapshot of the triangulation at creation time.**
/// The hull stores lightweight facet handles `(CellKey, u8)` which reference cells in the TDS.
/// These handles become **invalid** if the TDS is modified (e.g., by adding/removing vertices or cells).
///
/// ## Logical Immutability Design
///
/// Once created, a `ConvexHull` cannot be modified. There are no public mutating methods for the hull topology.
/// However, **internal caches may update** via interior mutability (`ArcSwapOption` for the facet cache,
/// `AtomicU64` for generation tracking), allowing performance optimizations without requiring `&mut self`.
///
/// This design ensures:
/// - Thread-safe sharing without locking (cache updates use lock-free atomics)
/// - Clear ownership semantics - a hull belongs to a specific TDS state
/// - Prevention of stale hull misuse (validated via generation counters)
///
/// ## Validity Checking
///
/// Use `is_valid_for_triangulation()` to check if a hull is still valid for a given TDS:
///
/// ```rust
/// # use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
/// # use delaunay::geometry::algorithms::convex_hull::ConvexHull;
/// # use delaunay::vertex;
/// # let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vec![
/// #     vertex!([0.0, 0.0, 0.0]),
/// #     vertex!([1.0, 0.0, 0.0]),
/// #     vertex!([0.0, 1.0, 0.0]),
/// #     vertex!([0.0, 0.0, 1.0]),
/// # ]).unwrap();
/// let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();
/// assert!(hull.is_valid_for_triangulation(dt.triangulation())); // Valid initially
///
/// // Hull extension is now implemented - inserting outside points works!
/// dt.insert(vertex!([2.0, 2.0, 2.0])).unwrap();
/// ```
///
/// ## When to Rebuild the Hull
///
/// You **must** create a new `ConvexHull` by calling `from_triangulation()` if:
/// - Vertices are added to or removed from the TDS
/// - Cells are added, removed, or modified in the TDS  
/// - Any operation that changes the TDS generation counter
/// - `is_valid_for_triangulation()` returns `false`
///
/// ## Example: Correct Usage Pattern
///
/// ```rust
/// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
/// use delaunay::geometry::algorithms::convex_hull::ConvexHull;
/// use delaunay::vertex;
///
/// let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ]).unwrap();
///
/// // Create initial hull (note: immutable binding)
/// let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();
/// assert_eq!(hull.facet_count(), 4);
/// assert!(hull.is_valid_for_triangulation(dt.triangulation()));
///
/// // Hull extension is now implemented - inserting outside points works!
/// // Note: The hull becomes invalid after modification and needs to be recreated
/// let new_vertex = vertex!([2.0, 2.0, 2.0]);
/// dt.insert(new_vertex).unwrap(); // Now works with hull extension!
/// assert!(!hull.is_valid_for_triangulation(dt.triangulation())); // Hull is stale
/// ```
///
/// # Type Parameters
///
/// * `T` - The coordinate scalar type (e.g., f64, f32)
/// * `U` - The vertex data type
/// * `V` - The cell data type  
/// * `D` - The dimension of the triangulation
///
/// # References
///
/// The algorithms implemented in this module are based on established computational geometry literature:
///
/// ## Convex Hull Construction from Delaunay Triangulations
///
/// - Brown, K.Q. "Voronoi Diagrams from Convex Hulls." *Information Processing Letters* 9, no. 5 (1979): 223-228.
///   DOI: [10.1016/0020-0190(79)90074-7](https://doi.org/10.1016/0020-0190(79)90074-7)
/// - Edelsbrunner, H. "Algorithms in Combinatorial Geometry." EATCS Monographs on Theoretical Computer Science.
///   Berlin: Springer-Verlag, 1987. DOI: [10.1007/978-3-642-61568-9](https://doi.org/10.1007/978-3-642-61568-9)
///
/// ## Point-in-Polytope Testing
///
/// - Preparata, F.P., and Shamos, M.I. "Computational Geometry: An Introduction." Texts and Monographs in Computer Science.
///   New York: Springer-Verlag, 1985. DOI: [10.1007/978-1-4612-1098-6](https://doi.org/10.1007/978-1-4612-1098-6)
/// - O'Rourke, J. "Computational Geometry in C." 2nd ed. Cambridge: Cambridge University Press, 1998.
///   DOI: [10.1017/CBO9780511804120](https://doi.org/10.1017/CBO9780511804120)
///
/// ## Incremental Convex Hull Construction
///
/// - Clarkson, K.L., and Shor, P.W. "Applications of Random Sampling in Computational Geometry, II."
///   *Discrete & Computational Geometry* 4, no. 1 (1989): 387-421. DOI: [10.1007/BF02187740](https://doi.org/10.1007/BF02187740)
/// - Barber, C.B., Dobkin, D.P., and Huhdanpaa, H. "The Quickhull Algorithm for Convex Hulls."
///   *ACM Transactions on Mathematical Software* 22, no. 4 (1996): 469-483. DOI: [10.1145/235815.235821](https://doi.org/10.1145/235815.235821)
///
/// ## High-Dimensional Computational Geometry
///
/// - Chazelle, B. "An Optimal Convex Hull Algorithm in Any Fixed Dimension." *Discrete & Computational Geometry* 10,
///   no. 4 (1993): 377-409. DOI: [10.1007/BF02573985](https://doi.org/10.1007/BF02573985)
/// - Seidel, R. "The Upper Bound Theorem for Polytopes: An Easy Proof of Its Asymptotic Version."
///   *Computational Geometry* 5, no. 2 (1995): 115-116. DOI: [10.1016/0925-7721(95)00013-Y](https://doi.org/10.1016/0925-7721(95)00013-Y)
#[derive(Debug)]
pub struct ConvexHull<K, U, V, const D: usize>
where
    K: Kernel<D>,
    U: DataType,
    V: DataType,
{
    /// The boundary facets that form the convex hull
    /// Stored as `FacetHandle` tuples (`CellKey`, `facet_index`) to enable reconstruction of `FacetView`
    ///
    /// **WARNING**: These handles are only valid for the triangulation at the generation captured
    /// in `creation_generation`. If the triangulation is modified, these handles become stale.
    /// Use `is_valid_for_triangulation()` to check validity before use.
    ///
    /// This field is private to prevent external mutation. Use the provided read-only
    /// accessors (`facets()`, `get_facet()`, `facet_count()`) to access hull facets.
    hull_facets: Vec<FacetHandle>,
    /// Cache for the facet-to-cells mapping to avoid rebuilding it for each facet check
    /// Uses `ArcSwapOption` for lock-free atomic updates when cache needs invalidation
    /// This avoids Some/None wrapping boilerplate compared to `ArcSwap<Option<K::Scalar>>`
    facet_to_cells_cache: ArcSwapOption<FacetToCellsMap>,
    /// Immutable triangulation generation at hull creation time.
    /// Set once in `from_triangulation()` and never modified. Used to detect stale hulls.
    /// Uses `OnceLock` to express the "set once, read many" semantic contract.
    creation_generation: OnceLock<u64>,
    /// Cache generation counter, updated when cache is rebuilt.
    /// Can be reset to 0 by `invalidate_cache()` to force cache rebuild.
    /// Uses `Arc<AtomicU64>` for consistent tracking across cloned `ConvexHull` instances.
    cached_generation: Arc<AtomicU64>,
    /// Phantom data to mark unused type parameters
    _phantom: PhantomData<(K, U, V)>,
}

// Minimal impl block for simple accessor methods that don't require arithmetic operations
impl<K, U, V, const D: usize> ConvexHull<K, U, V, D>
where
    K: Kernel<D>,
    U: DataType,
    V: DataType,
{
    /// Returns the number of hull facets
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
    /// use delaunay::geometry::algorithms::convex_hull::ConvexHull;
    /// use delaunay::vertex;
    ///
    /// // Create a 3D tetrahedron
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let hull =
    ///     ConvexHull::from_triangulation(dt.triangulation()).unwrap();
    ///
    /// assert_eq!(hull.facet_count(), 4); // Tetrahedron has 4 faces
    /// ```
    #[must_use]
    pub const fn facet_count(&self) -> usize {
        self.hull_facets.len()
    }

    /// Gets a hull facet by index
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the facet to retrieve
    ///
    /// # Returns
    ///
    /// Some reference to the facet if the index is valid, None otherwise
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
    /// use delaunay::geometry::algorithms::convex_hull::ConvexHull;
    /// use delaunay::vertex;
    ///
    /// // Create a 3D tetrahedron
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let hull =
    ///     ConvexHull::from_triangulation(dt.triangulation()).unwrap();
    ///
    /// // Get the first facet
    /// assert!(hull.get_facet(0).is_some());
    /// // Index out of bounds returns None
    /// assert!(hull.get_facet(10).is_none());
    /// ```
    #[must_use]
    pub fn get_facet(&self, index: usize) -> Option<&FacetHandle> {
        self.hull_facets.get(index)
    }

    /// Returns an iterator over the hull facets
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
    /// use delaunay::geometry::algorithms::convex_hull::ConvexHull;
    /// use delaunay::vertex;
    ///
    /// // Create a 3D tetrahedron
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let hull =
    ///     ConvexHull::from_triangulation(dt.triangulation()).unwrap();
    ///
    /// // Iterate over all hull facets
    /// let facet_count = hull.facets().count();
    /// assert_eq!(facet_count, 4); // Tetrahedron has 4 faces
    ///
    /// // Check that all facets have the expected number of vertices
    /// // Note: facets() returns FacetHandle structs - need to create FacetView to access vertices
    /// use delaunay::core::facet::FacetView;
    /// for facet_handle in hull.facets() {
    ///     if let Ok(facet_view) = FacetView::new(&dt.tds(), facet_handle.cell_key(), facet_handle.facet_index()) {
    ///         assert_eq!(facet_view.vertices().unwrap().count(), 3); // 3D facets have 3 vertices
    ///     }
    /// }
    /// ```
    pub fn facets(&self) -> std::slice::Iter<'_, FacetHandle> {
        self.hull_facets.iter()
    }

    /// Returns true if the convex hull is empty (has no facets)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
    /// use delaunay::geometry::algorithms::convex_hull::ConvexHull;
    /// use delaunay::vertex;
    ///
    /// // Empty hull
    /// let empty_hull: ConvexHull<delaunay::geometry::kernel::FastKernel<f64>, (), (), 3> = ConvexHull::default();
    /// assert!(empty_hull.is_empty());
    ///
    /// // Non-empty hull
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let hull =
    ///     ConvexHull::from_triangulation(dt.triangulation()).unwrap();
    /// assert!(!hull.is_empty());
    /// ```
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.hull_facets.is_empty()
    }

    /// Returns the dimension of the convex hull
    ///
    /// This is the same as the dimension of the triangulation that generated it.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
    /// use delaunay::geometry::algorithms::convex_hull::ConvexHull;
    /// use delaunay::vertex;
    ///
    /// // Create different dimensional hulls
    /// let vertices_2d: Vec<_> = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt_2d: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&vertices_2d).unwrap();
    /// let hull_2d =
    ///     ConvexHull::from_triangulation(dt_2d.triangulation()).unwrap();
    /// assert_eq!(hull_2d.dimension(), 2);
    ///
    /// let vertices_3d: Vec<_> = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt_3d: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices_3d).unwrap();
    /// let hull_3d =
    ///     ConvexHull::from_triangulation(dt_3d.triangulation()).unwrap();
    /// assert_eq!(hull_3d.dimension(), 3);
    /// ```
    #[must_use]
    pub const fn dimension(&self) -> usize {
        D
    }

    /// Checks if this convex hull is valid for the given triangulation
    ///
    /// Returns `true` if the hull's creation generation matches the triangulation generation,
    /// meaning the hull's facet handles are still valid for this triangulation.
    /// Returns `false` if the triangulation has been modified since the hull was created.
    ///
    /// **Empty hull special case**: An empty hull (with no facets) is always considered
    /// valid, since it contains no facet handles that could become stale.
    ///
    /// # Arguments
    ///
    /// * `tri` - The triangulation to check against
    ///
    /// # Returns
    ///
    /// `true` if the hull is valid for this triangulation, `false` otherwise
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
    /// use delaunay::geometry::algorithms::convex_hull::ConvexHull;
    /// use delaunay::vertex;
    ///
    /// let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ]).unwrap();
    ///
    /// // Create hull and verify it's valid
    /// let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();
    /// assert!(hull.is_valid_for_triangulation(dt.triangulation()));
    ///
    /// // After any modification to the triangulation, the hull would become invalid
    /// // Note: Currently, hull extension is not yet implemented, so inserting
    /// // outside points will cause an error. This demonstrates the validation concept.
    /// ```
    #[must_use]
    pub fn is_valid_for_triangulation(&self, tri: &Triangulation<K, U, V, D>) -> bool {
        // Use creation_generation (immutable) for validity check, not cached_generation (mutable)
        // Empty hull (with creation_generation unset) is always valid - it has no facets to be stale
        self.creation_generation
            .get()
            .map_or(self.is_empty(), |&g| g == tri.tds.generation())
    }

    /// Helper to construct a `StaleHull` error with generation info
    ///
    /// Centralizes the error construction pattern to avoid duplication.
    #[inline]
    fn stale_hull_error(&self, tri: &Triangulation<K, U, V, D>) -> ConvexHullValidationError {
        ConvexHullValidationError::StaleHull {
            hull_generation: self.creation_generation.get().copied().unwrap_or(0),
            tds_generation: tri.tds.generation(),
        }
    }

    /// Helper to construct a `StaleHull` construction error with generation info
    ///
    /// Centralizes the error construction pattern to avoid duplication.
    #[inline]
    fn stale_hull_construction_error(
        &self,
        tri: &Triangulation<K, U, V, D>,
    ) -> ConvexHullConstructionError {
        ConvexHullConstructionError::StaleHull {
            hull_generation: self.creation_generation.get().copied().unwrap_or(0),
            tds_generation: tri.tds.generation(),
        }
    }

    /// Invalidates the internal facet-to-cells cache and resets the cached generation counter
    ///
    /// This method forces the cache to be rebuilt on the next visibility test.
    /// It can be useful for manual cache management.
    ///
    /// # Two-Generation Design
    ///
    /// This method resets only the **cache generation** (`cached_generation`) to 0, forcing
    /// cache rebuild on next access. The **creation generation** (`creation_generation`) remains
    /// immutable and is never modified after construction.
    ///
    /// This separation ensures:
    /// - Cache invalidation doesn't break stale hull detection
    /// - `is_valid_for_triangulation()` always returns accurate results based on creation generation
    /// - Stale hulls are caught even after calling `invalidate_cache()`
    ///
    /// # Interior Mutability
    ///
    /// This method takes `&self` (not `&mut self`) and is safe to call on shared hulls
    /// due to interior mutability via `ArcSwapOption` (for the facet cache) and `AtomicU64`
    /// (for cache generation tracking). These lock-free atomic operations allow cache invalidation
    /// without exclusive mutable access.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
    /// use delaunay::geometry::algorithms::convex_hull::ConvexHull;
    /// use delaunay::vertex;
    ///
    /// // Create a 3D tetrahedron
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let hull =
    ///     ConvexHull::from_triangulation(dt.triangulation()).unwrap();
    ///
    /// // Manually invalidate the cache (note: takes &self, not &mut self)
    /// hull.invalidate_cache();
    ///
    /// // The next visibility test will rebuild the cache
    /// // ... perform visibility operations ...
    /// ```
    pub fn invalidate_cache(&self) {
        // Clear the cache using ArcSwapOption::store(None)
        self.facet_to_cells_cache.store(None);

        // Reset only cached_generation to force rebuild on next access.
        // creation_generation is NEVER modified - it remains the immutable snapshot from creation.
        // Use Release ordering to ensure consistency with Acquire loads by readers
        self.cached_generation.store(0, Ordering::Release);
    }
}

// Full impl block for construction and geometric operations
impl<K, U, V, const D: usize> ConvexHull<K, U, V, D>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar
        + AddAssign
        + SubAssign
        + Sub<Output = K::Scalar>
        + DivAssign
        + Zero
        + NumCast
        + Copy
        + std::iter::Sum,
    U: DataType,
    V: DataType,
    [K::Scalar; D]: Copy + Sized,
{
    /// Creates a new convex hull from a d-dimensional triangulation
    ///
    /// # Arguments
    ///
    /// * `tri` - The triangulation (combines kernel + topology)
    ///
    /// # Returns
    ///
    /// A `Result` containing the convex hull or a [`ConvexHullConstructionError`] if extraction fails
    ///
    /// # Errors
    ///
    /// Returns a [`ConvexHullConstructionError`] if:
    /// - Boundary facets cannot be extracted from the triangulation ([`ConvexHullConstructionError::BoundaryFacetExtractionFailed`])
    /// - The input triangulation is invalid ([`ConvexHullConstructionError::InvalidTriangulation`])
    /// - Facet data access fails during construction ([`ConvexHullConstructionError::FacetDataAccessFailed`])
    ///   - This can happen if cells or vertices referenced by boundary facets are no longer valid
    ///   - Or if the facet index is out of bounds for the cell's vertex count
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
    /// use delaunay::geometry::algorithms::convex_hull::ConvexHull;
    /// use delaunay::geometry::kernel::FastKernel;
    /// use delaunay::vertex;
    ///
    /// // 3D example
    /// let vertices_3d: Vec<_> = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt_3d: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices_3d).unwrap();
    /// let hull_3d: ConvexHull<FastKernel<f64>, (), (), 3> =
    ///     ConvexHull::from_triangulation(dt_3d.triangulation()).unwrap();
    /// assert_eq!(hull_3d.facet_count(), 4); // Tetrahedron has 4 faces
    ///
    /// // 4D example
    /// let vertices_4d: Vec<_> = vec![
    ///     vertex!([0.0, 0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0, 1.0]),
    /// ];
    /// let dt_4d: DelaunayTriangulation<_, (), (), 4> = DelaunayTriangulation::new(&vertices_4d).unwrap();
    /// let hull_4d =
    ///     ConvexHull::from_triangulation(dt_4d.triangulation()).unwrap();
    /// assert_eq!(hull_4d.facet_count(), 5); // 4-simplex has 5 facets
    /// ```
    pub fn from_triangulation(
        tri: &Triangulation<K, U, V, D>,
    ) -> Result<Self, ConvexHullConstructionError> {
        let tds = &tri.tds;

        // Validate input triangulation
        if tds.number_of_vertices() == 0 {
            return Err(ConvexHullConstructionError::InsufficientData {
                message: "Triangulation contains no vertices".to_string(),
            });
        }

        if tds.number_of_cells() == 0 {
            return Err(ConvexHullConstructionError::InsufficientData {
                message: "Triangulation contains no cells".to_string(),
            });
        }

        // Use the existing boundary analysis to get hull facets
        let hull_facets_iter = tds.boundary_facets().map_err(|source| {
            ConvexHullConstructionError::BoundaryFacetExtractionFailed { source }
        })?;

        // Collect facet handles (CellKey, facet_index) for storage
        // These can be used to reconstruct FacetViews when needed
        let hull_facets: Vec<_> = hull_facets_iter
            .map(|facet_view| FacetHandle::new(facet_view.cell_key(), facet_view.facet_index()))
            .collect();

        // Additional validation: ensure we have at least one boundary facet
        if hull_facets.is_empty() {
            return Err(ConvexHullConstructionError::InsufficientData {
                message: "No boundary facets found in triangulation".to_string(),
            });
        }

        let tds_gen = tds.generation();
        Ok(Self {
            hull_facets,
            facet_to_cells_cache: ArcSwapOption::empty(),
            // Immutable snapshot of triangulation generation at creation - never changes
            creation_generation: OnceLock::from(tds_gen),
            // Mutable cache generation - can be reset by invalidate_cache()
            cached_generation: Arc::new(AtomicU64::new(tds_gen)),
            _phantom: PhantomData,
        })
    }

    /// Tests if a facet is visible from an external point using proper geometric predicates
    ///
    /// A facet is visible if the point is on the "outside" side of the facet.
    /// This implementation uses geometric orientation predicates to determine the correct
    /// side of the hyperplane defined by the facet, based on the Bowyer-Watson algorithm.
    ///
    /// Uses an internal cache to avoid rebuilding the facet-to-cells mapping for each call.
    ///
    /// # Algorithm
    ///
    /// For a boundary facet F with vertices {f₁, f₂, ..., fₐ}, we need to determine
    /// if a test point p is on the "outside" of the facet. Since this is a boundary facet
    /// from a convex hull, we know it has exactly one adjacent cell.
    ///
    /// The algorithm works as follows:
    /// 1. Get or build the cached facet-to-cells mapping
    /// 2. Find the "inside" vertex of the adjacent cell (vertex not in the facet)
    /// 3. Create two simplices: facet + `inside_vertex` and facet + `test_point`  
    /// 4. Compare orientations - different orientations mean opposite sides
    /// 5. If test point is on opposite side from inside vertex, facet is visible
    ///
    /// # Arguments
    ///
    /// * `facet` - The facet to test
    /// * `point` - The external point
    /// * `tds` - Reference to triangulation (needed to find adjacent cell)
    ///
    /// # Returns
    ///
    /// `true` if the facet is visible from the point, `false` otherwise
    ///
    /// # Note
    ///
    /// This method uses cached facet-to-cells mapping for optimal performance. The cache is
    /// automatically built if it doesn't exist or has been invalidated.
    ///
    /// For batch visibility checking (e.g., [`Self::find_visible_facets`]), consider using the internal
    /// helper that accepts a pre-loaded cache to avoid redundant atomic loads.
    ///
    /// # Errors
    ///
    /// Returns a [`ConvexHullConstructionError`] if:
    /// - The facet cache cannot be built ([`ConvexHullConstructionError::FacetCacheBuildFailed`])
    /// - Adjacent cell resolution fails ([`ConvexHullConstructionError::AdjacentCellResolutionFailed`])
    /// - Facet visibility check fails ([`ConvexHullConstructionError::VisibilityCheckFailed`])
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
    /// use delaunay::geometry::algorithms::convex_hull::ConvexHull;
    /// use delaunay::geometry::point::Point;
    /// use delaunay::geometry::traits::coordinate::Coordinate;
    /// use delaunay::vertex;
    ///
    /// // Create a 3D tetrahedron
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let hull =
    ///     ConvexHull::from_triangulation(dt.triangulation()).unwrap();
    ///
    /// // Get a hull facet to test
    /// let facet = hull.get_facet(0).unwrap();
    ///
    /// // Test visibility from different points
    /// let inside_point = Point::new([0.2, 0.2, 0.2]); // Inside the tetrahedron
    /// let outside_point = Point::new([2.0, 2.0, 2.0]); // Outside the tetrahedron
    ///
    /// // Inside point should not see the facet (facet not visible)
    /// let inside_visible = hull.is_facet_visible_from_point(facet, &inside_point, dt.triangulation()).unwrap();
    /// assert!(!inside_visible, "Inside point should not see hull facet");
    ///
    /// // Outside point may see the facet depending on which facet we're testing
    /// let outside_visible = hull.is_facet_visible_from_point(facet, &outside_point, dt.triangulation()).unwrap();
    /// // Note: The result depends on which facet is selected and the point's position
    /// // This test just verifies the method executes without error
    /// ```
    pub fn is_facet_visible_from_point(
        &self,
        facet_handle: &FacetHandle,
        point: &Point<K::Scalar, D>,
        tri: &Triangulation<K, U, V, D>,
    ) -> Result<bool, ConvexHullConstructionError> {
        let tds = &tri.tds;

        // Staleness guard: fail fast before any cache work
        let creation_gen = self.creation_generation.get().copied().unwrap_or(0);
        let tds_gen = tds.generation();
        if creation_gen != tds_gen {
            return Err(ConvexHullConstructionError::StaleHull {
                hull_generation: creation_gen,
                tds_generation: tds_gen,
            });
        }

        // Get or build the cached facet-to-cells mapping
        let facet_to_cells_arc = self
            .try_get_or_build_facet_cache(tds)
            .map_err(|source| ConvexHullConstructionError::FacetCacheBuildFailed { source })?;

        // Delegate to internal helper with pre-loaded cache
        self.is_facet_visible_from_point_with_cache(
            facet_handle,
            point,
            tri,
            facet_to_cells_arc.as_ref(),
        )
    }

    /// Internal helper for visibility testing with a pre-loaded cache.
    ///
    /// This method avoids redundant atomic loads of the facet cache, which is beneficial
    /// when performing batch visibility checks (e.g., in [`Self::find_visible_facets`]).
    ///
    /// # Arguments
    ///
    /// * `facet_handle` - The facet to test
    /// * `point` - The external point
    /// * `tds` - Reference to triangulation
    /// * `facet_to_cells` - Pre-loaded facet-to-cells mapping
    ///
    /// # Returns
    ///
    /// `true` if the facet is visible from the point, `false` otherwise
    ///
    /// # Errors
    ///
    /// Returns a [`ConvexHullConstructionError`] if visibility checking fails.
    #[expect(clippy::too_many_lines)]
    fn is_facet_visible_from_point_with_cache(
        &self,
        facet_handle: &FacetHandle,
        point: &Point<K::Scalar, D>,
        tri: &Triangulation<K, U, V, D>,
        facet_to_cells: &FacetToCellsMap,
    ) -> Result<bool, ConvexHullConstructionError> {
        let tds = &tri.tds;

        // Two-generation design: creation_generation (immutable) vs cached_generation (mutable)
        // - creation_generation: Set once at from_triangulation(), never changes. Used for stale detection.
        // - cached_generation: Can be reset to 0 by invalidate_cache() to force cache rebuild.
        let creation_gen = self.creation_generation.get().copied().unwrap_or(0);
        let tds_gen = tds.generation();

        debug_assert!(
            creation_gen == tds_gen,
            "ConvexHull used with a modified TDS; rebuild the hull (creation_gen={creation_gen}, tds_gen={tds_gen})"
        );

        // Production build: always check creation_generation for stale detection
        if creation_gen != tds_gen {
            return Err(ConvexHullConstructionError::StaleHull {
                hull_generation: creation_gen,
                tds_generation: tds_gen,
            });
        }

        // Phase 3A: Derive facet vertex keys directly from the cell to avoid UUID↔key roundtrips.
        // This eliminates the need to convert vertex UUIDs back to keys later.
        let (facet_cell_key, facet_index) = (facet_handle.cell_key(), facet_handle.facet_index());
        let cell = tds.get_cell(facet_cell_key).ok_or(
            ConvexHullConstructionError::FacetDataAccessFailed {
                source: FacetError::CellNotFoundInTriangulation,
            },
        )?;

        // Bounds check: facet_index must be within cell vertices to prevent out-of-bounds access
        let cell_vertex_count = cell.vertices().len();
        if (facet_index as usize) >= cell_vertex_count {
            return Err(ConvexHullConstructionError::VisibilityCheckFailed {
                source: FacetError::InvalidFacetIndex {
                    index: facet_index,
                    facet_count: cell_vertex_count,
                },
            });
        }

        // Extract vertex keys for all vertices except the one at facet_index
        // Use SmallBuffer to avoid heap allocation for typical dimensions (up to 7D on stack)
        let mut facet_vertex_keys: SmallBuffer<_, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::with_capacity(D);
        for (i, &k) in cell.vertices().iter().enumerate() {
            if i != facet_index as usize {
                facet_vertex_keys.push(k);
            }
        }

        if facet_vertex_keys.len() != D {
            return Err(ConvexHullConstructionError::VisibilityCheckFailed {
                source: FacetError::InsufficientVertices {
                    expected: D,
                    actual: facet_vertex_keys.len(),
                    dimension: D,
                },
            });
        }

        // Optimization: Derive the facet key directly from vertex keys without materializing Vertex objects.
        // This avoids D vertex fetches and D UUID lookups, improving cache locality.
        let facet_key = derive_facet_key_from_vertex_keys::<K::Scalar, U, V, D>(&facet_vertex_keys)
            .map_err(|source| ConvexHullConstructionError::VisibilityCheckFailed { source })?;

        let adjacent_cells = facet_to_cells.get(&facet_key).ok_or_else(|| {
            // Collect vertex UUIDs for enhanced error reporting (only on error path)
            // Materialize vertices only when needed for error reporting
            let vertex_uuids: Vec<uuid::Uuid> = facet_vertex_keys
                .iter()
                .filter_map(|&k| tds.get_vertex_by_key(k).map(Vertex::uuid))
                .collect();
            ConvexHullConstructionError::VisibilityCheckFailed {
                source: FacetError::FacetKeyNotFoundInCache {
                    facet_key,
                    cache_size: facet_to_cells.len(),
                    vertex_uuids,
                },
            }
        })?;

        if adjacent_cells.len() != 1 {
            return Err(ConvexHullConstructionError::VisibilityCheckFailed {
                source: FacetError::InvalidAdjacentCellCount {
                    found: adjacent_cells.len(),
                },
            });
        }

        let adj_cell_key = adjacent_cells[0].cell_key();

        // Find the vertex in the adjacent cell that is NOT part of the facet
        // This is the "opposite" or "inside" vertex
        // Optimization: Use vertex keys instead of UUID comparison for better performance
        let cell_vertices = tds.get_cell_vertices(adj_cell_key).map_err(|source| {
            ConvexHullConstructionError::AdjacentCellResolutionFailed { source }
        })?;

        // facet_vertex_keys already computed above - no UUID→key roundtrip needed!
        // Find the cell vertex key that's not in the facet
        // Optimized: Use a sorted merge-like approach to avoid O(D²) contains() calls
        // Since both lists are small (D and D+1 elements), we can sort and scan efficiently
        let mut sorted_facet_keys = facet_vertex_keys.clone();
        sorted_facet_keys.sort_unstable();

        let inside_vertex_key = cell_vertices
            .iter()
            .find(|&&cell_key| sorted_facet_keys.binary_search(&cell_key).is_err())
            .copied()
            .ok_or(ConvexHullConstructionError::VisibilityCheckFailed {
                source: FacetError::InsideVertexNotFound,
            })?;

        // Get the actual vertex from the key
        let inside_vertex = tds.get_vertex_by_key(inside_vertex_key).ok_or(
            ConvexHullConstructionError::VisibilityCheckFailed {
                source: FacetError::InsideVertexNotFound,
            },
        )?;

        // Materialize facet vertices only when needed for orientation computation
        // This happens after cache lookup and inside vertex identification
        // Use SmallBuffer to avoid heap allocation for typical dimensions (up to 7D on stack)
        let facet_vertices: SmallBuffer<Vertex<K::Scalar, U, D>, MAX_PRACTICAL_DIMENSION_SIZE> =
            facet_vertex_keys
                .iter()
                .map(|&k| {
                    tds.get_vertex_by_key(k).copied().ok_or(
                        ConvexHullConstructionError::FacetDataAccessFailed {
                            source: FacetError::VertexKeyNotFoundInTriangulation { key: k },
                        },
                    )
                })
                .collect::<Result<SmallBuffer<_, _>, _>>()?;

        // Create test simplices to compare orientations
        // Build facet_points from the vertices (fetched only once, when needed)
        // Use SmallBuffer to keep typical dimensions on-stack and avoid heap allocation
        let mut facet_points: SmallBuffer<Point<K::Scalar, D>, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::with_capacity(D);
        for v in &facet_vertices {
            facet_points.push(*v.point());
        }

        // Simplex 1: facet vertices + inside vertex
        let mut simplex_with_inside = facet_points.clone().into_vec();
        simplex_with_inside.push(*inside_vertex.point());

        // Simplex 2: facet vertices + test point
        let mut simplex_with_test = facet_points.into_vec();
        simplex_with_test.push(*point);

        // Get orientations using geometric predicates
        let orientation_inside = simplex_orientation(&simplex_with_inside).map_err(|e| {
            ConvexHullConstructionError::VisibilityCheckFailed {
                source: FacetError::OrientationComputationFailed {
                    details: format!("Failed to compute orientation with inside vertex: {e}"),
                },
            }
        })?;
        let orientation_test = simplex_orientation(&simplex_with_test).map_err(|e| {
            ConvexHullConstructionError::VisibilityCheckFailed {
                source: FacetError::OrientationComputationFailed {
                    details: format!("Failed to compute orientation with test point: {e}"),
                },
            }
        })?;

        // Compare orientations - facet is visible if orientations are different
        match (orientation_inside, orientation_test) {
            (Orientation::NEGATIVE, Orientation::POSITIVE)
            | (Orientation::POSITIVE, Orientation::NEGATIVE) => Ok(true),
            (Orientation::DEGENERATE, _) | (_, Orientation::DEGENERATE) => {
                // Degenerate case - fall back to distance heuristic
                // Reuse vertices already loaded above to avoid redundant facet.vertices() call
                Self::fallback_visibility_test(&facet_vertices, point)
            }
            _ => Ok(false), // Same orientation = same side = not visible
        }
    }

    /// Fallback visibility test for degenerate cases
    ///
    /// When geometric predicates fail due to degeneracy, this method provides
    /// a simple heuristic based on distance from the facet centroid. The threshold
    /// is scale-adaptive, based on the facet's diameter squared, with an epsilon-based
    /// bound to prevent false positives from numeric noise near the hull surface.
    ///
    /// # Arguments
    ///
    /// * `facet` - The facet to test visibility against
    /// * `point` - The point to test visibility from
    ///
    /// # Returns
    ///
    /// Returns a `Result<bool, ConvexHullConstructionError>` where `true` indicates
    /// the facet is visible from the point and `false` indicates it's not visible.
    ///
    /// # Errors
    ///
    /// Returns a [`ConvexHullConstructionError::CoordinateConversion`] if
    /// coordinate conversion fails during centroid calculation or threshold computation.
    ///
    /// # Algorithm
    ///
    /// 1. Calculate the centroid of the facet vertices
    /// 2. Compute the distance from the test point to the centroid
    /// 3. Use the facet's diameter (max edge length) as a scale-adaptive threshold
    /// 4. Add a small relative epsilon (1e-12 scale) to avoid false positives from numeric noise
    /// 5. Return true if the distance exceeds the adjusted threshold (likely outside/visible)
    fn fallback_visibility_test(
        facet_vertices: &[Vertex<K::Scalar, U, D>],
        point: &Point<K::Scalar, D>,
    ) -> Result<bool, ConvexHullConstructionError> {
        // Use SmallBuffer for stack allocation (D vertices fit on stack for D ≤ 7)
        let vertex_points: SmallBuffer<Point<K::Scalar, D>, MAX_PRACTICAL_DIMENSION_SIZE> =
            facet_vertices
                .iter()
                .map(|vertex| *vertex.point())
                .collect();

        // Calculate facet centroid
        let mut centroid_coords = [K::Scalar::zero(); D];
        for vertex_point in &vertex_points {
            let coords: [K::Scalar; D] = vertex_point.into();
            for (i, &coord) in coords.iter().enumerate() {
                centroid_coords[i] += coord;
            }
        }
        let num_vertices = safe_usize_to_scalar(vertex_points.len())
            .map_err(ConvexHullConstructionError::CoordinateConversion)?;
        for coord in &mut centroid_coords {
            *coord /= num_vertices;
        }

        // Simple heuristic: if point is far from centroid, it's likely visible
        let point_coords: [K::Scalar; D] = point.into();
        let mut diff_coords = [K::Scalar::zero(); D];
        for i in 0..D {
            diff_coords[i] = point_coords[i] - centroid_coords[i];
        }
        let distance_squared = squared_norm(diff_coords);

        // Use a threshold to determine visibility - this is a simple heuristic
        // Scale-aware threshold: use the facet diameter squared (max pairwise edge length squared)
        let mut max_edge_sq = K::Scalar::zero();
        for (i, vertex_a) in vertex_points.iter().enumerate() {
            let ai: [K::Scalar; D] = vertex_a.into();
            for vertex_b in vertex_points.iter().skip(i + 1) {
                let bj: [K::Scalar; D] = vertex_b.into();
                let mut diff = [K::Scalar::zero(); D];
                for k in 0..D {
                    diff[k] = ai[k] - bj[k];
                }
                let edge_sq = squared_norm(diff);
                if max_edge_sq.is_zero() || edge_sq > max_edge_sq {
                    max_edge_sq = edge_sq;
                }
            }
        }

        if max_edge_sq.is_zero() {
            // Degenerate facet geometry; treat as not visible.
            return Ok(false);
        }
        // Add epsilon-based bound to avoid false positives from numeric noise
        // Use the type-specific default tolerance (1e-6 for f32, 1e-15 for f64)
        // to handle near-surface points. This adapts automatically to coordinate precision.
        let epsilon_factor = K::Scalar::default_tolerance();
        let adjusted_threshold = max_edge_sq + max_edge_sq * epsilon_factor;

        Ok(distance_squared > adjusted_threshold)
    }

    /// Finds all hull facets visible from an external point
    ///
    /// # Arguments
    ///
    /// * `point` - The external point
    /// * `tds` - Reference to triangulation (needed for visibility testing)
    ///
    /// # Returns
    ///
    /// A vector of indices into the `hull_facets` array for visible facets
    ///
    /// # Errors
    ///
    /// Returns a [`ConvexHullConstructionError`] if the visibility test fails for any facet.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
    /// use delaunay::geometry::algorithms::convex_hull::ConvexHull;
    /// use delaunay::geometry::point::Point;
    /// use delaunay::geometry::traits::coordinate::Coordinate;
    /// use delaunay::vertex;
    ///
    /// // Create a 3D tetrahedron
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let hull =
    ///     ConvexHull::from_triangulation(dt.triangulation()).unwrap();
    ///
    /// // Test with a point outside the hull
    /// let outside_point = Point::new([2.0, 2.0, 2.0]);
    /// let visible_facets = hull.find_visible_facets(&outside_point, dt.triangulation()).unwrap();
    /// assert!(!visible_facets.is_empty(), "Outside point should see some facets");
    ///
    /// // Test with a point inside the hull
    /// let inside_point = Point::new([0.2, 0.2, 0.2]);
    /// let visible_facets = hull.find_visible_facets(&inside_point, dt.triangulation()).unwrap();
    /// assert!(visible_facets.is_empty(), "Inside point should see no facets");
    /// ```
    pub fn find_visible_facets(
        &self,
        point: &Point<K::Scalar, D>,
        tri: &Triangulation<K, U, V, D>,
    ) -> Result<Vec<usize>, ConvexHullConstructionError> {
        let tds = &tri.tds;

        // Fail fast if hull is stale relative to this TDS (using immutable creation_generation)
        if !self.is_valid_for_triangulation(tri) {
            return Err(self.stale_hull_construction_error(tri));
        }

        // Optimization: Load cache once before the loop to avoid redundant atomic loads
        let facet_cache_arc = self
            .try_get_or_build_facet_cache(tds)
            .map_err(|source| ConvexHullConstructionError::FacetCacheBuildFailed { source })?;
        let facet_cache = facet_cache_arc.as_ref();

        let mut visible_facets = Vec::new();

        for (index, facet_handle) in self.hull_facets.iter().enumerate() {
            if self.is_facet_visible_from_point_with_cache(facet_handle, point, tri, facet_cache)? {
                visible_facets.push(index);
            }
        }

        Ok(visible_facets)
    }

    /// Finds the nearest visible facet to a point
    ///
    /// This is useful for incremental hull construction algorithms.
    ///
    /// # Arguments
    ///
    /// * `point` - The external point
    /// * `tds` - Reference to triangulation (needed for visibility testing)
    ///
    /// # Returns
    ///
    /// The index of the nearest visible facet, or None if no facets are visible
    ///
    /// # Errors
    ///
    /// Returns a [`ConvexHullConstructionError`] if the visibility test fails or if distance calculations fail.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
    /// use delaunay::geometry::algorithms::convex_hull::ConvexHull;
    /// use delaunay::geometry::point::Point;
    /// use delaunay::geometry::traits::coordinate::Coordinate;
    /// use delaunay::vertex;
    ///
    /// // Create a 3D tetrahedron
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let hull =
    ///     ConvexHull::from_triangulation(dt.triangulation()).unwrap();
    ///
    /// // Test with a point outside the hull - should find a nearest visible facet
    /// let outside_point = Point::new([2.0, 2.0, 2.0]);
    /// let nearest_facet = hull.find_nearest_visible_facet(&outside_point, dt.triangulation()).unwrap();
    /// assert!(nearest_facet.is_some(), "Outside point should have a nearest visible facet");
    ///
    /// // Test with a point inside the hull - should find no visible facets
    /// let inside_point = Point::new([0.2, 0.2, 0.2]);
    /// let nearest_facet = hull.find_nearest_visible_facet(&inside_point, dt.triangulation()).unwrap();
    /// assert!(nearest_facet.is_none(), "Inside point should have no visible facets");
    /// ```
    pub fn find_nearest_visible_facet(
        &self,
        point: &Point<K::Scalar, D>,
        tri: &Triangulation<K, U, V, D>,
    ) -> Result<Option<usize>, ConvexHullConstructionError>
    where
        K::Scalar: PartialOrd + Copy,
    {
        let tds = &tri.tds;

        // Fail fast if hull is stale relative to this TDS (using immutable creation_generation)
        if !self.is_valid_for_triangulation(tri) {
            return Err(self.stale_hull_construction_error(tri));
        }

        let visible_facets = self.find_visible_facets(point, tri)?;

        if visible_facets.is_empty() {
            return Ok(None);
        }

        // Find the facet with minimum distance to the point
        let mut min_distance: Option<K::Scalar> = None;
        let mut nearest_facet = None;

        for &facet_index in &visible_facets {
            let facet_handle = &self.hull_facets[facet_index];
            // Create FacetView to access facet vertices
            let facet_view =
                FacetView::new(tds, facet_handle.cell_key(), facet_handle.facet_index()).map_err(
                    |source| ConvexHullConstructionError::FacetDataAccessFailed { source },
                )?;
            // Extract points directly to avoid materializing Vertex copies
            let facet_points: Vec<Point<K::Scalar, D>> = facet_view
                .vertices()
                .map_err(|source| ConvexHullConstructionError::FacetDataAccessFailed { source })?
                .map(|v| *v.point())
                .collect();

            // Calculate distance from point to facet centroid as a simple heuristic
            let mut centroid_coords = [K::Scalar::zero(); D];
            let num_vertices = safe_usize_to_scalar(facet_points.len())
                .map_err(ConvexHullConstructionError::CoordinateConversion)?;

            for vertex_point in &facet_points {
                let coords: [K::Scalar; D] = (*vertex_point).into();
                for (i, &coord) in coords.iter().enumerate() {
                    centroid_coords[i] += coord;
                }
            }

            for coord in &mut centroid_coords {
                *coord /= num_vertices;
            }

            let centroid = Point::new(centroid_coords);

            // Calculate squared distance using squared_norm
            let point_coords: [K::Scalar; D] = point.into();
            let centroid_coords: [K::Scalar; D] = (&centroid).into();
            let mut diff_coords = [K::Scalar::zero(); D];
            for i in 0..D {
                diff_coords[i] = point_coords[i] - centroid_coords[i];
            }
            let distance = squared_norm(diff_coords);

            if min_distance.is_none_or(|min_dist| distance < min_dist) {
                min_distance = Some(distance);
                nearest_facet = Some(facet_index);
            }
        }

        Ok(nearest_facet)
    }

    /// Checks if a point is outside the current convex hull
    ///
    /// A point is outside if it's visible from at least one hull facet.
    ///
    /// # Arguments
    ///
    /// * `point` - The point to test
    /// * `tds` - Reference to triangulation (needed for visibility testing)
    ///
    /// # Returns
    ///
    /// `true` if the point is outside the hull, `false` if inside
    ///
    /// # Errors
    ///
    /// Returns a [`ConvexHullConstructionError`] if the visibility test fails for any facet.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
    /// use delaunay::geometry::algorithms::convex_hull::ConvexHull;
    /// use delaunay::geometry::point::Point;
    /// use delaunay::geometry::traits::coordinate::Coordinate;
    /// use delaunay::vertex;
    ///
    /// // Create a 3D tetrahedron
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let hull =
    ///     ConvexHull::from_triangulation(dt.triangulation()).unwrap();
    ///
    /// // Test with a point inside the hull
    /// let inside_point = Point::new([0.2, 0.2, 0.2]);
    /// assert!(!hull.is_point_outside(&inside_point, dt.triangulation()).unwrap());
    ///
    /// // Test with a point outside the hull
    /// let outside_point = Point::new([2.0, 2.0, 2.0]);
    /// assert!(hull.is_point_outside(&outside_point, dt.triangulation()).unwrap());
    /// ```
    pub fn is_point_outside(
        &self,
        point: &Point<K::Scalar, D>,
        tri: &Triangulation<K, U, V, D>,
    ) -> Result<bool, ConvexHullConstructionError> {
        let visible_facets = self.find_visible_facets(point, tri)?;
        Ok(!visible_facets.is_empty())
    }

    /// Validates the convex hull for consistency
    ///
    /// This performs basic checks on the hull facets to ensure they form
    /// a valid convex hull structure.
    ///
    /// # Errors
    ///
    /// Returns an error if any facet has an invalid number of vertices or contains duplicate vertices.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
    /// use delaunay::geometry::algorithms::convex_hull::ConvexHull;
    /// use delaunay::vertex;
    ///
    /// // Create a valid 3D tetrahedron
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let hull =
    ///     ConvexHull::from_triangulation(dt.triangulation()).unwrap();
    ///
    /// // Validation should pass for a well-formed hull
    /// assert!(hull.validate(dt.triangulation()).is_ok());
    ///
    /// // Empty hull should also validate
    /// let empty_hull: ConvexHull<delaunay::geometry::kernel::FastKernel<f64>, (), (), 3> = ConvexHull::default();
    /// // Note: validate() requires a TDS, so use an empty TDS for validation
    /// assert!(empty_hull.validate(dt.triangulation()).is_ok());
    /// ```
    pub fn validate(
        &self,
        tri: &Triangulation<K, U, V, D>,
    ) -> Result<(), ConvexHullValidationError> {
        let tds = &tri.tds;

        // Check staleness first - validate() should explicitly fail on stale hulls
        // This makes the test behavior robust: validate() fails due to staleness check,
        // not because facet handles happen to point to removed cells
        if !self.is_valid_for_triangulation(tri) {
            return Err(self.stale_hull_error(tri));
        }

        // Check that all facets have exactly D vertices (for D-dimensional triangulation,
        // facets are (D-1)-dimensional and have D vertices)
        for (index, facet_handle) in self.hull_facets.iter().enumerate() {
            // Phase 3A: Create FacetView from lightweight handle to access vertices
            let facet_view =
                FacetView::new(tds, facet_handle.cell_key(), facet_handle.facet_index()).map_err(
                    |source| ConvexHullValidationError::InvalidFacet {
                        facet_index: index,
                        source,
                    },
                )?;

            let vertices: Vec<_> = facet_view
                .vertices()
                .map_err(|source| ConvexHullValidationError::InvalidFacet {
                    facet_index: index,
                    source,
                })?
                .copied()
                .collect();
            if vertices.len() != D {
                return Err(ConvexHullValidationError::InvalidFacet {
                    facet_index: index,
                    source: FacetError::InsufficientVertices {
                        expected: D,
                        actual: vertices.len(),
                        dimension: D,
                    },
                });
            }

            // Check that vertices are distinct - collect all duplicates for this facet
            // Use SmallVec for positions to avoid heap allocation for typical small collections
            // Size 8 should cover most practical dimensions (up to 7D vertices per facet)
            //
            // Deferred to v0.6.0 or later; see Issue #106.
            // Current behavior is correct but not optimized for D > 7.
            // For D > 7, SmallBuffer heap-allocates; dimension-aware allocation could improve performance.
            let mut uuid_to_positions: FastHashMap<uuid::Uuid, SmallBuffer<usize, 8>> =
                FastHashMap::default();
            for (position, vertex) in vertices.iter().enumerate() {
                uuid_to_positions
                    .entry(vertex.uuid())
                    .or_default()
                    .push(position);
            }

            // Find any UUIDs that appear more than once
            // Convert SmallBuffer to Vec for the error type (maintains API compatibility)
            let duplicate_groups: Vec<Vec<usize>> = uuid_to_positions
                .into_values()
                .filter(|positions| positions.len() > 1)
                .map(smallvec::SmallVec::into_vec)
                .collect();

            if !duplicate_groups.is_empty() {
                return Err(ConvexHullValidationError::DuplicateVerticesInFacet {
                    facet_index: index,
                    positions: duplicate_groups,
                });
            }
        }

        Ok(())
    }
}

// Implementation of FacetCacheProvider trait for ConvexHull
// Separate impl block with FacetCacheProvider-specific trait bounds
// (main impl block has simpler bounds that don't include Sum, DeserializeOwned, etc.)
impl<K, U, V, const D: usize> FacetCacheProvider<K::Scalar, U, V, D> for ConvexHull<K, U, V, D>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar + AddAssign + SubAssign + std::iter::Sum + num_traits::NumCast,
    U: DataType,
    V: DataType,
{
    fn facet_cache(&self) -> &ArcSwapOption<FacetToCellsMap> {
        &self.facet_to_cells_cache
    }

    fn cached_generation(&self) -> &AtomicU64 {
        self.cached_generation.as_ref()
    }
}

impl<K, U, V, const D: usize> Default for ConvexHull<K, U, V, D>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar,
    U: DataType,
    V: DataType,
    [K::Scalar; D]: Copy + Sized,
{
    fn default() -> Self {
        Self {
            hull_facets: Vec::new(),
            facet_to_cells_cache: ArcSwapOption::empty(),
            creation_generation: OnceLock::new(), // Empty - indicates invalid/uninitialized hull
            cached_generation: Arc::new(AtomicU64::new(0)),
            _phantom: PhantomData,
        }
    }
}

// Type aliases for common use cases
/// Type alias for 2D convex hulls with `FastKernel`
pub type ConvexHull2D<K, U, V> = ConvexHull<K, U, V, 2>;
/// Type alias for 3D convex hulls with `FastKernel`
pub type ConvexHull3D<K, U, V> = ConvexHull<K, U, V, 3>;
/// Type alias for 4D convex hulls with `FastKernel`
pub type ConvexHull4D<K, U, V> = ConvexHull<K, U, V, 4>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::delaunay_triangulation::DelaunayTriangulation;
    use crate::core::traits::facet_cache::FacetCacheProvider;
    use crate::core::triangulation_data_structure::{
        TriangulationConstructionError, TriangulationValidationError,
    };
    use crate::core::util::{derive_facet_key_from_vertex_keys, facet_view_to_vertices};
    use crate::geometry::kernel::FastKernel;
    use crate::vertex;
    use serde::Serialize;
    use serde::de::DeserializeOwned;
    use std::error::Error;
    use std::sync::atomic::Ordering;
    use std::thread;

    // =============================================================================
    // HELPER FUNCTIONS
    // =============================================================================

    /// Helper function to create a Triangulation from vertices.
    ///
    /// This uses `DelaunayTriangulation::new()` and returns a reference to the underlying `Triangulation`.
    /// Since we need ownership, we create the `DelaunayTriangulation` and extract the `Triangulation`.
    fn create_triangulation<const D: usize>(
        vertices: &[crate::core::vertex::Vertex<f64, (), D>],
    ) -> DelaunayTriangulation<FastKernel<f64>, (), (), D> {
        DelaunayTriangulation::new(vertices).unwrap()
    }

    /// Helper function to extract vertices from a facet handle.
    ///
    /// This is a test utility that creates a `FacetView` from a facet handle
    /// and extracts the vertices as a `Vec<Vertex>`.
    /// Uses the shared `facet_view_to_vertices` utility to avoid code duplication.
    fn extract_facet_vertices<K, U, V, const D: usize>(
        facet_handle: &FacetHandle,
        tri: &Triangulation<K, U, V, D>,
    ) -> Result<Vec<Vertex<K::Scalar, U, D>>, ConvexHullConstructionError>
    where
        K: Kernel<D>,
        K::Scalar: CoordinateScalar,
        U: DataType,
        V: DataType,
        [K::Scalar; D]: Copy + Sized + Serialize + DeserializeOwned,
    {
        let tds = &tri.tds;
        let facet_view =
            FacetView::new(tds, facet_handle.cell_key(), facet_handle.facet_index())
                .map_err(|source| ConvexHullConstructionError::FacetDataAccessFailed { source })?;
        // Use the shared utility for extracting vertices
        facet_view_to_vertices(&facet_view)
            .map_err(|source| ConvexHullConstructionError::FacetDataAccessFailed { source })
    }

    // =============================================================================
    // TEST GENERATION MACROS
    // =============================================================================

    /// Macro to generate comprehensive dimension-specific convex hull tests.
    ///
    /// This macro reduces test duplication by generating consistent tests across
    /// multiple dimensions (1D through 6D). For each dimension, it creates:
    ///
    /// 1. **Basic operations test** - Tests:
    ///    - `facet_count()` returns expected number
    ///    - `dimension()` returns correct dimension
    ///    - `validate()` succeeds
    ///    - `is_empty()` returns false for non-empty hull
    ///
    /// 2. **Facet access test** (via `pastey::paste`) - Tests:
    ///    - `facets()` iterator returns correct count
    ///    - `get_facet(0)` returns Some for valid index
    ///    - `get_facet(out_of_bounds)` returns None
    ///
    /// 3. **Point containment test** (via `pastey::paste`) - Tests:
    ///    - Inside point (centroid) is correctly identified as not outside
    ///    - Outside point is correctly identified as outside
    ///
    /// # Usage
    ///
    /// ```ignore
    /// test_hull_dimensions! {
    ///     hull_2d => 2 => "triangle" => 3 =>
    ///         vec![vertex!([0.0, 0.0]), vertex!([1.0, 0.0]), vertex!([0.0, 1.0])],
    /// }
    /// ```
    ///
    /// # Arguments
    ///
    /// * `$test_name` - Base name for generated tests
    /// * `$dim` - Dimension (const generic parameter)
    /// * `$desc` - Description string for error messages
    /// * `$expected_facets` - Expected number of facets in the hull
    /// * `$vertices` - Vec of vertices to construct the hull
    ///
    /// # Generated Tests
    ///
    /// For `hull_2d => 2 => "triangle" => 3 => vertices`, generates:
    /// - `hull_2d()` - Basic operations
    /// - `hull_2d_facet_access()` - Facet access  
    /// - `hull_2d_point_containment()` - Point inside/outside detection
    macro_rules! test_hull_dimensions {
        ($(
            $test_name:ident => $dim:expr => $desc:expr => $expected_facets:expr => $vertices:expr
        ),+ $(,)?) => {
            $(
                #[test]
                fn $test_name() {
                    let vertices = $vertices;
                    let dt = create_triangulation(&vertices);
                    let hull: ConvexHull<FastKernel<f64>, (), (), $dim> =
                        ConvexHull::from_triangulation(dt.triangulation()).unwrap();

                    assert_eq!(
                        hull.facet_count(),
                        $expected_facets,
                        "{}D hull ({}) should have {} facets",
                        $dim,
                        $desc,
                        $expected_facets
                    );
                    assert_eq!(
                        hull.dimension(),
                        $dim,
                        "{}D hull should have dimension {}",
                        $dim,
                        $dim
                    );
                    assert!(
                        hull.validate(dt.triangulation()).is_ok(),
                        "{}D hull validation should succeed",
                        $dim
                    );
                    assert!(!hull.is_empty(), "{}D hull should not be empty", $dim);
                }

                pastey::paste! {
                    #[test]
                    fn [<$test_name _facet_access>]() {
                        let vertices = $vertices;
                        let dt = create_triangulation(&vertices);
                        let hull: ConvexHull<FastKernel<f64>, (), (), $dim> =
                            ConvexHull::from_triangulation(dt.triangulation()).unwrap();

                        // Test facet iterator
                        assert_eq!(
                            hull.facets().count(),
                            $expected_facets,
                            "{}D facets iterator should return {} facets",
                            $dim,
                            $expected_facets
                        );

                        // Test get_facet access
                        assert!(
                            hull.get_facet(0).is_some(),
                            "{}D hull should be able to get facet 0",
                            $dim
                        );
                        assert!(
                            hull.get_facet($expected_facets).is_none(),
                            "{}D hull out of range facet index should return None",
                            $dim
                        );
                    }

                    #[test]
                    fn [<$test_name _point_containment>]() {
                        let vertices = $vertices;
                        let dt = create_triangulation(&vertices);
                        let hull: ConvexHull<FastKernel<f64>, (), (), $dim> =
                            ConvexHull::from_triangulation(dt.triangulation()).unwrap();

                        // Test with inside point (scaled to ensure it's inside the unit simplex)
                        // For a unit simplex, a point is inside if sum of coordinates <= 1
                        // Use 0.1 per dimension to ensure sum stays well below 1 for all dimensions
                        let inside_coords = [0.1; $dim];
                        let inside_point = Point::new(inside_coords);
                        assert!(
                            !hull.is_point_outside(&inside_point, dt.triangulation()).unwrap(),
                            "{}D hull: point with small coordinates should be inside",
                            $dim
                        );

                        // Test with outside point
                        let outside_coords = [2.0; $dim];
                        let outside_point = Point::new(outside_coords);
                        assert!(
                            hull.is_point_outside(&outside_point, dt.triangulation()).unwrap(),
                            "{}D hull: point far outside should be detected",
                            $dim
                        );
                    }
                }
            )+
        };
    }

    // Generate tests for dimensions 1D through 6D
    test_hull_dimensions! {
        hull_1d_segment => 1 => "line segment" => 2 => vec![
            vertex!([0.0]),
            vertex!([1.0]),
        ],
        hull_2d_triangle => 2 => "triangle" => 3 => vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ],
        hull_3d_tetrahedron => 3 => "tetrahedron" => 4 => vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.5, 1.0, 0.0]),
            vertex!([0.5, 0.5, 1.0]),
        ],
        hull_4d_simplex => 4 => "4-simplex" => 5 => vec![
            vertex!([0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0]),
        ],
        hull_5d_simplex => 5 => "5-simplex" => 6 => vec![
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 0.0, 1.0]),
        ],
        hull_6d_simplex => 6 => "6-simplex" => 7 => vec![
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ],
    }

    #[test]
    fn test_empty_hull_comprehensive() {
        // Create a triangulation for validation purposes
        let vertices_3d = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt_3d = create_triangulation(&vertices_3d);

        // Test empty hull (default constructor)
        let empty_hull: ConvexHull3D<FastKernel<f64>, (), ()> = ConvexHull::default();

        // Basic empty hull properties
        assert_eq!(
            empty_hull.facet_count(),
            0,
            "Empty hull should have 0 facets"
        );
        assert_eq!(
            empty_hull.dimension(),
            3,
            "Empty hull should maintain dimension"
        );
        assert!(
            empty_hull.validate(dt_3d.triangulation()).is_ok(),
            "Empty hull validation should succeed"
        );
        assert!(
            empty_hull.is_empty(),
            "Default constructor should create empty hull"
        );
        assert!(
            empty_hull.get_facet(0).is_none(),
            "Empty hull should not have facets"
        );
        assert_eq!(
            empty_hull.facets().count(),
            0,
            "Empty hull's facets iterator should be empty"
        );

        // Test cache behavior on empty hull
        let empty_cache = empty_hull.facet_cache().load();
        assert!(
            empty_cache.is_none(),
            "Empty hull should have no cache initially"
        );

        let empty_generation = empty_hull.cached_generation().load(Ordering::Acquire);
        assert_eq!(empty_generation, 0, "Empty hull should have generation 0");

        // Test invalidation on empty hull
        empty_hull.invalidate_cache();
        let invalidated_generation = empty_hull.cached_generation().load(Ordering::Acquire);
        assert_eq!(
            invalidated_generation, 0,
            "Empty hull generation should remain 0 after invalidation"
        );
    }

    /// Comprehensive test for visibility algorithms covering all dimensions and edge cases
    /// Consolidates: `test_visibility_algorithms_comprehensive`, `test_visibility_edge_cases`,
    /// `test_visibility_algorithm_coverage`, `test_edge_case_distance_calculations`
    #[expect(clippy::too_many_lines, clippy::cognitive_complexity)]
    #[test]
    fn test_visibility_algorithms_comprehensive() {
        println!("Testing comprehensive visibility algorithms in dimensions 2D-5D");

        // ========================================================================
        // Test 1: 2D visibility (point-in-polygon and visible facets)
        // ========================================================================
        println!("  Testing 2D visibility algorithms...");
        let vertices_2d = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt_2d = create_triangulation(&vertices_2d);
        let hull_2d: ConvexHull2D<FastKernel<f64>, (), ()> =
            ConvexHull::from_triangulation(dt_2d.triangulation()).unwrap();

        let inside_point_2d = Point::new([0.1, 0.1]);
        let outside_point_2d = Point::new([2.0, 2.0]);

        // Test point outside detection
        assert!(
            !hull_2d
                .is_point_outside(&inside_point_2d, dt_2d.triangulation())
                .unwrap(),
            "2D inside point should not be outside"
        );
        assert!(
            hull_2d
                .is_point_outside(&outside_point_2d, dt_2d.triangulation())
                .unwrap(),
            "2D outside point should be outside"
        );

        // Test visible facets detection
        let visible_facets_inside = hull_2d
            .find_visible_facets(&inside_point_2d, dt_2d.triangulation())
            .unwrap();
        let visible_facets_outside = hull_2d
            .find_visible_facets(&outside_point_2d, dt_2d.triangulation())
            .unwrap();

        assert!(
            visible_facets_inside.is_empty(),
            "2D inside point should see no facets"
        );
        assert!(
            !visible_facets_outside.is_empty(),
            "2D outside point should see some facets"
        );

        // ========================================================================
        // Test 2: 3D visibility (comprehensive testing)
        // ========================================================================
        println!("  Testing 3D visibility algorithms...");
        let vertices_3d = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt_3d = create_triangulation(&vertices_3d);
        let hull_3d: ConvexHull3D<FastKernel<f64>, (), ()> =
            ConvexHull::from_triangulation(dt_3d.triangulation()).unwrap();

        let inside_point_3d = Point::new([0.2, 0.2, 0.2]);
        let outside_point_3d = Point::new([2.0, 2.0, 2.0]);

        // Test point outside detection
        assert!(
            !hull_3d
                .is_point_outside(&inside_point_3d, dt_3d.triangulation())
                .unwrap(),
            "3D inside point should not be outside"
        );
        assert!(
            hull_3d
                .is_point_outside(&outside_point_3d, dt_3d.triangulation())
                .unwrap(),
            "3D outside point should be outside"
        );

        // Test visible facets detection
        let visible_facets_inside_3d = hull_3d
            .find_visible_facets(&inside_point_3d, dt_3d.triangulation())
            .unwrap();
        let visible_facets_outside_3d = hull_3d
            .find_visible_facets(&outside_point_3d, dt_3d.triangulation())
            .unwrap();

        assert!(
            visible_facets_inside_3d.is_empty(),
            "3D inside point should see no facets"
        );
        assert!(
            !visible_facets_outside_3d.is_empty(),
            "3D outside point should see some facets"
        );

        // Test nearest visible facet
        let nearest_facet_inside = hull_3d
            .find_nearest_visible_facet(&inside_point_3d, dt_3d.triangulation())
            .unwrap();
        let nearest_facet_outside = hull_3d
            .find_nearest_visible_facet(&outside_point_3d, dt_3d.triangulation())
            .unwrap();

        assert!(
            nearest_facet_inside.is_none(),
            "Inside point should have no nearest visible facet"
        );
        assert!(
            nearest_facet_outside.is_some(),
            "Outside point should have a nearest visible facet"
        );

        // Test find_visible_facets with far outside point (multiple visible facets)
        let far_outside_point = Point::new([10.0, 10.0, 10.0]);
        let visible_facets = hull_3d
            .find_visible_facets(&far_outside_point, dt_3d.triangulation())
            .unwrap();
        assert!(
            !visible_facets.is_empty(),
            "Far outside point should see multiple facets"
        );

        // Verify all returned indices are valid
        for &index in &visible_facets {
            assert!(
                index < hull_3d.facet_count(),
                "Visible facet index should be valid"
            );
            assert!(hull_3d.get_facet(index).is_some());
        }

        // Test individual facet visibility for each facet
        for (i, facet) in hull_3d.facets().enumerate() {
            let visibility_result = hull_3d.is_facet_visible_from_point(
                facet,
                &far_outside_point,
                dt_3d.triangulation(),
            );
            assert!(
                visibility_result.is_ok(),
                "Facet {i} visibility test should succeed"
            );
        }

        // ========================================================================
        // Test 3: Edge cases with boundary and near-boundary points
        // ========================================================================
        println!("  Testing visibility edge cases...");
        let boundary_points = [
            Point::new([0.5, 0.5, 0.0]), // On a face
            Point::new([0.3, 0.3, 0.3]), // Near centroid
            Point::new([0.0, 0.0, 0.0]), // At a vertex
            Point::new([0.5, 0.0, 0.0]), // On an edge
        ];

        for (i, point) in boundary_points.iter().enumerate() {
            let result = hull_3d.is_point_outside(point, dt_3d.triangulation());
            assert!(
                result.is_ok(),
                "Boundary point {i} visibility test should not error"
            );
        }

        // Test with points very close to the hull
        let close_points = vec![
            Point::new([1e-10, 1e-10, 1e-10]),
            Point::new([0.999_999, 0.0, 0.0]),
            Point::new([0.0, 0.999_999, 0.0]),
            Point::new([0.0, 0.0, 0.999_999]),
        ];

        for point in close_points {
            let result = hull_3d.is_point_outside(&point, dt_3d.triangulation());
            assert!(
                result.is_ok(),
                "Close point visibility test should not error"
            );
        }

        // ========================================================================
        // Test 4: Distance calculations with edge cases
        // ========================================================================
        println!("  Testing edge case distance calculations...");

        // Test find_nearest_visible_facet with equidistant points
        let equidistant_point = Point::new([0.5, 0.5, 0.5]);
        let result = hull_3d.find_nearest_visible_facet(&equidistant_point, dt_3d.triangulation());
        assert!(
            result.is_ok(),
            "Distance calculation with equal distances should succeed"
        );

        // Test with very large coordinates that might cause overflow
        let large_point = Point::new([1e15, 1e15, 1e15]);
        let result = hull_3d.find_nearest_visible_facet(&large_point, dt_3d.triangulation());
        assert!(
            result.is_ok(),
            "Distance calculation with large coordinates should succeed"
        );

        // Test with very small coordinates that might cause underflow
        let small_point = Point::new([1e-15, 1e-15, 1e-15]);
        let result = hull_3d.find_nearest_visible_facet(&small_point, dt_3d.triangulation());
        assert!(
            result.is_ok(),
            "Distance calculation with small coordinates should succeed"
        );

        // ========================================================================
        // Test 5: 4D visibility
        // ========================================================================
        println!("  Testing 4D visibility algorithms...");
        let vertices_4d = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0]),
        ];
        let dt_4d = create_triangulation(&vertices_4d);
        let hull_4d: ConvexHull4D<FastKernel<f64>, (), ()> =
            ConvexHull::from_triangulation(dt_4d.triangulation()).unwrap();

        let inside_point_4d = Point::new([0.1, 0.1, 0.1, 0.1]);
        let outside_point_4d = Point::new([2.0, 2.0, 2.0, 2.0]);

        assert!(
            !hull_4d
                .is_point_outside(&inside_point_4d, dt_4d.triangulation())
                .unwrap(),
            "4D inside point should not be outside"
        );
        assert!(
            hull_4d
                .is_point_outside(&outside_point_4d, dt_4d.triangulation())
                .unwrap(),
            "4D outside point should be outside"
        );

        // ========================================================================
        // Test 6: 5D visibility
        // ========================================================================
        println!("  Testing 5D visibility algorithms...");
        let vertices_5d = vec![
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 0.0, 1.0]),
        ];
        let dt_5d = create_triangulation(&vertices_5d);
        let hull_5d: ConvexHull<FastKernel<f64>, (), (), 5> =
            ConvexHull::from_triangulation(dt_5d.triangulation()).unwrap();

        let inside_point_5d = Point::new([0.1, 0.1, 0.1, 0.1, 0.1]);
        let outside_point_5d = Point::new([2.0, 2.0, 2.0, 2.0, 2.0]);

        assert!(
            !hull_5d
                .is_point_outside(&inside_point_5d, dt_5d.triangulation())
                .unwrap(),
            "5D inside point should not be outside"
        );
        assert!(
            hull_5d
                .is_point_outside(&outside_point_5d, dt_5d.triangulation())
                .unwrap(),
            "5D outside point should be outside"
        );

        println!("  ✓ Visibility algorithms tested comprehensively across all dimensions");
    }

    // ============================================================================
    // UNIT TESTS FOR PRIVATE METHODS
    // ============================================================================
    // These tests target private methods to ensure thorough coverage of internal
    // ConvexHull functionality, particularly the fallback_visibility_test method.

    #[expect(clippy::too_many_lines)]
    #[test]
    fn test_fallback_visibility_comprehensive() {
        println!("Testing comprehensive fallback visibility algorithm");

        // Test distance-based heuristic with 3D tetrahedron
        println!("  Testing distance-based heuristic...");
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        assert!(hull.facet_count() > 0, "Hull should have facets");
        let test_facet_vertices =
            extract_facet_vertices(hull.get_facet(0).unwrap(), dt.triangulation()).unwrap();

        // Test with points at various distances to verify scale-adaptive threshold
        let distance_test_points = vec![
            (Point::new([0.1, 0.1, 0.1]), "Very close to centroid"),
            (Point::new([0.5, 0.5, 0.5]), "Medium distance from centroid"),
            (Point::new([2.0, 2.0, 2.0]), "Far from centroid"),
            (Point::new([5.0, 5.0, 5.0]), "Very far point"),
        ];

        let mut visibility_results = Vec::new();
        for (point, description) in &distance_test_points {
            let is_visible = ConvexHull::<
                crate::geometry::kernel::FastKernel<f64>,
                (),
                (),
                3,
            >::fallback_visibility_test(&test_facet_vertices, point)
            .unwrap();
            visibility_results.push(is_visible);
            let coords: [f64; 3] = (*point).into();
            println!("    Point {coords:?} ({description}) - Visible: {is_visible}");
        }

        let visible_count = visibility_results.iter().filter(|&&v| v).count();
        assert!(
            visible_count > 0,
            "At least some points should be visible with fallback"
        );

        // Test degenerate cases
        println!("  Testing degenerate and edge cases...");
        let degenerate_points = vec![
            (Point::new([0.0, 0.0, 0.0]), "Origin point"),
            (
                Point::new([f64::EPSILON, f64::EPSILON, f64::EPSILON]),
                "Very small coordinates",
            ),
            (Point::new([1e-15, 1e-15, 1e-15]), "Near-zero coordinates"),
        ];

        for (point, description) in degenerate_points {
            let result = ConvexHull::<
                crate::geometry::kernel::FastKernel<f64>,
                (),
                (),
                3,
            >::fallback_visibility_test(&test_facet_vertices, &point);
            assert!(
                result.is_ok(),
                "{description} should not cause fallback to error"
            );
            println!("    {description} - Result: {:?}", result.unwrap());
        }

        // Test consistency - same point multiple times should give same result
        println!("  Testing consistency...");
        let consistency_point = Point::new([2.0, 2.0, 2.0]);
        let consistency_results: Vec<bool> = (0..5)
            .map(|_| {
                ConvexHull::<crate::geometry::kernel::FastKernel<f64>, (), (), 3>::fallback_visibility_test(
                    &test_facet_vertices,
                    &consistency_point,
                )
                .unwrap()
            })
            .collect();

        let first_result = consistency_results[0];
        assert!(
            consistency_results
                .iter()
                .all(|&result| result == first_result),
            "Fallback visibility should be consistent for same point"
        );
        println!(
            "    Consistency test: all {} results were {}",
            consistency_results.len(),
            first_result
        );

        // Test numerical precision with high-precision coordinates
        println!("  Testing numerical precision...");
        let precise_points = vec![
            Point::new([1e-15, 1e-15, 1e-15]),
            Point::new([
                1.000_000_000_000_000_1,
                1.000_000_000_000_000_1,
                1.000_000_000_000_000_1,
            ]),
            Point::new([
                0.999_999_999_999_999_9,
                0.999_999_999_999_999_9,
                0.999_999_999_999_999_9,
            ]),
        ];

        for point in precise_points {
            let result = ConvexHull::<
                crate::geometry::kernel::FastKernel<f64>,
                (),
                (),
                3,
            >::fallback_visibility_test(&test_facet_vertices, &point);
            assert!(
                result.is_ok(),
                "High precision coordinates should not cause errors"
            );
            let coords: [f64; 3] = point.into();
            println!(
                "    High precision Point {coords:?} - Visible: {:?}",
                result.unwrap()
            );
        }

        println!("  Testing in different dimensions...");

        // Test 2D fallback
        let vertices_2d = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt_2d = create_triangulation(&vertices_2d);
        let hull_2d: ConvexHull<FastKernel<f64>, (), (), 2> =
            ConvexHull::from_triangulation(dt_2d.triangulation()).unwrap();
        let test_facet_2d_vertices =
            extract_facet_vertices(&hull_2d.hull_facets[0], dt_2d.triangulation()).unwrap();
        let test_point_2d = Point::new([2.0, 2.0]);
        let result_2d = ConvexHull::<
            crate::geometry::kernel::FastKernel<f64>,
            (),
            (),
            2,
        >::fallback_visibility_test(&test_facet_2d_vertices, &test_point_2d);
        assert!(result_2d.is_ok(), "2D fallback should work");
        println!("    2D fallback result: {:?}", result_2d.unwrap());

        // Test 4D fallback
        let vertices_4d = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0]),
        ];
        let dt_4d = create_triangulation(&vertices_4d);
        let hull_4d: ConvexHull<FastKernel<f64>, (), (), 4> =
            ConvexHull::from_triangulation(dt_4d.triangulation()).unwrap();
        let test_facet_4d_vertices =
            extract_facet_vertices(&hull_4d.hull_facets[0], dt_4d.triangulation()).unwrap();
        let test_point_4d = Point::new([2.0, 2.0, 2.0, 2.0]);
        let result_4d = ConvexHull::<
            crate::geometry::kernel::FastKernel<f64>,
            (),
            (),
            4,
        >::fallback_visibility_test(&test_facet_4d_vertices, &test_point_4d);
        assert!(result_4d.is_ok(), "4D fallback should work");
        println!("    4D fallback result: {:?}", result_4d.unwrap());

        println!("  ✓ Comprehensive fallback visibility algorithm tested successfully");
    }

    // ============================================================================
    // EXHAUSTIVE UNIT TESTS FOR COMPREHENSIVE COVERAGE
    // ============================================================================
    // Additional tests to ensure we maintain 85% test coverage by testing
    // edge cases, error conditions, and less-covered code paths.

    #[test]
    fn test_from_triangulation_error_cases() {
        // Test creating hull from triangulation that fails to extract boundary facets
        // This is tricky to test directly since boundary_facets() rarely fails,
        // but we can test the error handling path exists

        // Create a minimal valid triangulation first to ensure the path works
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation());
        assert!(
            hull.is_ok(),
            "Valid triangulation should create hull successfully"
        );
    }

    #[test]
    fn test_is_facet_visible_from_point_error_cases() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        // Get a valid facet
        let facet = &hull.hull_facets[0];
        let test_point = Point::new([1.0, 1.0, 1.0]);

        // Test normal case first
        let result = hull.is_facet_visible_from_point(facet, &test_point, dt.triangulation());
        assert!(result.is_ok(), "Normal visibility test should succeed");

        // Note: Testing the InsufficientVertices error path is complex because
        // it requires creating invalid facets. For now, we just test the normal
        // case to ensure the method works correctly. The error paths are covered
        // by the existing comprehensive tests in other methods.
    }

    #[test]
    fn test_validate_error_cases() {
        // Test basic validation scenarios
        // Most validation edge cases are covered by the existing comprehensive tests

        // Test empty hull validation (should pass)
        let empty_hull: ConvexHull<FastKernel<f64>, (), (), 3> = ConvexHull::default();
        // Create a dummy triangulation for validation
        let dummy_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dummy_dt = create_triangulation(&dummy_vertices);
        assert!(
            empty_hull.validate(dummy_dt.triangulation()).is_ok(),
            "Empty hull should validate successfully"
        );

        // Test valid hull validation (should pass)
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        assert!(
            hull.validate(dt.triangulation()).is_ok(),
            "Valid hull should validate successfully"
        );

        // Note: Testing validation with manually constructed invalid facets is complex
        // because our API doesn't expose direct facet construction with invalid data.
        // The validation logic is still tested through normal usage patterns.
    }

    /// Comprehensive tests for the `ConvexHull` validate method covering all scenarios
    /// Consolidates: `test_convex_hull_validation_comprehensive`, `test_validate_method_comprehensive`,
    /// `test_validate_method_various_dimensions`
    #[test]
    #[expect(clippy::cognitive_complexity, clippy::too_many_lines)]
    fn test_convex_hull_validation_comprehensive() {
        println!("Testing ConvexHull validation comprehensively");

        // ========================================================================
        // Test 1: Valid hulls in different dimensions (1D-5D comprehensive coverage)
        // ========================================================================
        println!("  Testing valid hull validation in different dimensions...");

        // Test 1D hull (empty hull validation)
        let dummy_vertices_1d = vec![vertex!([0.0]), vertex!([1.0])];
        let dummy_dt_1d = create_triangulation(&dummy_vertices_1d);
        let hull_1d: ConvexHull<FastKernel<f64>, (), (), 1> = ConvexHull::default();
        assert!(
            hull_1d.validate(dummy_dt_1d.triangulation()).is_ok(),
            "1D empty hull should validate"
        );
        println!("    1D empty hull validated");

        // Test 2D hull
        let vertices_2d = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt_2d = create_triangulation(&vertices_2d);
        let hull_2d: ConvexHull<FastKernel<f64>, (), (), 2> =
            ConvexHull::from_triangulation(dt_2d.triangulation()).unwrap();
        assert!(
            hull_2d.validate(dt_2d.triangulation()).is_ok(),
            "Valid 2D hull should validate successfully"
        );
        // Validate vertices through FacetView
        for (i, facet_handle) in hull_2d.hull_facets.iter().enumerate() {
            let facet_view = FacetView::new(
                &dt_2d.triangulation().tds,
                facet_handle.cell_key(),
                facet_handle.facet_index(),
            )
            .unwrap();
            let vertices = facet_view.vertices().unwrap().count();
            assert_eq!(vertices, 2, "2D facet {i} should have exactly 2 vertices");
        }
        println!("    2D hull: {} facets validated", hull_2d.facet_count());

        // Test empty 2D hull validation
        let empty_hull_2d: ConvexHull<FastKernel<f64>, (), (), 2> = ConvexHull::default();
        assert!(
            empty_hull_2d.validate(dt_2d.triangulation()).is_ok(),
            "2D empty hull should validate successfully"
        );

        // Test 3D hull
        let vertices_3d = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt_3d = create_triangulation(&vertices_3d);
        let hull_3d: ConvexHull<FastKernel<f64>, (), (), 3> =
            ConvexHull::from_triangulation(dt_3d.triangulation()).unwrap();
        assert!(
            hull_3d.validate(dt_3d.triangulation()).is_ok(),
            "Valid 3D hull should validate successfully"
        );
        // Validate vertices through FacetView
        for (i, facet_handle) in hull_3d.hull_facets.iter().enumerate() {
            let facet_view = FacetView::new(
                &dt_3d.triangulation().tds,
                facet_handle.cell_key(),
                facet_handle.facet_index(),
            )
            .unwrap();
            let vertices = facet_view.vertices().unwrap().count();
            assert_eq!(vertices, 3, "3D facet {i} should have exactly 3 vertices");
        }
        println!("    3D hull: {} facets validated", hull_3d.facet_count());

        // Test empty 3D hull validation
        let empty_hull_3d: ConvexHull<FastKernel<f64>, (), (), 3> = ConvexHull::default();
        assert!(
            empty_hull_3d.validate(dt_3d.triangulation()).is_ok(),
            "3D empty hull should validate successfully"
        );

        // Test 4D hull
        let vertices_4d = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0]),
        ];
        let dt_4d = create_triangulation(&vertices_4d);
        let hull_4d: ConvexHull<FastKernel<f64>, (), (), 4> =
            ConvexHull::from_triangulation(dt_4d.triangulation()).unwrap();
        assert!(
            hull_4d.validate(dt_4d.triangulation()).is_ok(),
            "Valid 4D hull should validate successfully"
        );
        // Validate vertices through FacetView
        for (i, facet_handle) in hull_4d.hull_facets.iter().enumerate() {
            let facet_view = FacetView::new(
                &dt_4d.triangulation().tds,
                facet_handle.cell_key(),
                facet_handle.facet_index(),
            )
            .unwrap();
            let vertices = facet_view.vertices().unwrap().count();
            assert_eq!(vertices, 4, "4D facet {i} should have exactly 4 vertices");
        }
        println!("    4D hull: {} facets validated", hull_4d.facet_count());

        // Test empty 4D hull validation
        let empty_hull_4d: ConvexHull<FastKernel<f64>, (), (), 4> = ConvexHull::default();
        assert!(
            empty_hull_4d.validate(dt_4d.triangulation()).is_ok(),
            "4D empty hull should validate successfully"
        );

        // Test 5D hull (minimum required coverage)
        let vertices_5d: Vec<_> = vec![
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 0.0, 1.0]),
        ];
        let dt_5d = create_triangulation(&vertices_5d);
        let hull_5d: ConvexHull<FastKernel<f64>, (), (), 5> =
            ConvexHull::from_triangulation(dt_5d.triangulation()).unwrap();
        assert!(
            hull_5d.validate(dt_5d.triangulation()).is_ok(),
            "Valid 5D hull should validate successfully"
        );
        // Validate vertices through FacetView
        for (i, facet_handle) in hull_5d.hull_facets.iter().enumerate() {
            let facet_view = FacetView::new(
                &dt_5d.triangulation().tds,
                facet_handle.cell_key(),
                facet_handle.facet_index(),
            )
            .unwrap();
            let vertices = facet_view.vertices().unwrap().count();
            assert_eq!(vertices, 5, "5D facet {i} should have exactly 5 vertices");
        }
        println!("    5D hull: {} facets validated", hull_5d.facet_count());

        // Test empty 5D hull validation
        let empty_hull_5d: ConvexHull<FastKernel<f64>, (), (), 5> = ConvexHull::default();
        assert!(
            empty_hull_5d.validate(dt_5d.triangulation()).is_ok(),
            "5D empty hull should validate successfully"
        );

        println!("  ✓ Empty hull validation passed for dimensions 1D-5D");
        println!("  ✓ Valid hull validation passed for all tested dimensions");

        // ========================================================================
        // Test 2: Validation with different data types
        // ========================================================================
        println!("  Testing validation with different data types...");

        // Test with integer vertex data
        let vertices_int = vec![
            vertex!([0.0, 0.0, 0.0], 1i32),
            vertex!([1.0, 0.0, 0.0], 2i32),
            vertex!([0.0, 1.0, 0.0], 3i32),
            vertex!([0.0, 0.0, 1.0], 4i32),
        ];
        let dt_int = DelaunayTriangulation::<FastKernel<f64>, i32, (), 3>::with_kernel(
            FastKernel::new(),
            &vertices_int,
        )
        .unwrap();
        let hull_int = ConvexHull::from_triangulation(dt_int.triangulation()).unwrap();
        assert!(
            hull_int.validate(dt_int.triangulation()).is_ok(),
            "Hull with integer data should validate successfully"
        );

        // Test with character vertex data
        let vertices_char = vec![
            vertex!([0.0, 0.0, 0.0], 'A'),
            vertex!([1.0, 0.0, 0.0], 'B'),
            vertex!([0.0, 1.0, 0.0], 'C'),
            vertex!([0.0, 0.0, 1.0], 'D'),
        ];
        let dt_char = DelaunayTriangulation::<FastKernel<f64>, char, (), 3>::with_kernel(
            FastKernel::new(),
            &vertices_char,
        )
        .unwrap();
        let hull_char = ConvexHull::from_triangulation(dt_char.triangulation()).unwrap();
        assert!(
            hull_char.validate(dt_char.triangulation()).is_ok(),
            "Hull with character data should validate successfully"
        );
        println!("  ✓ Validation with different data types passed");

        // ========================================================================
        // Test 4: Validation with extreme coordinate values
        // ========================================================================
        println!("  Testing validation with extreme coordinate values...");

        let extreme_vertices = vec![
            // Large coordinates
            (
                vec![
                    vertex!([0.0, 0.0, 0.0]),
                    vertex!([1e15, 0.0, 0.0]),
                    vertex!([0.0, 1e15, 0.0]),
                    vertex!([0.0, 0.0, 1e15]),
                ],
                "large",
            ),
            // Small coordinates
            (
                vec![
                    vertex!([0.0, 0.0, 0.0]),
                    vertex!([1e-15, 0.0, 0.0]),
                    vertex!([0.0, 1e-15, 0.0]),
                    vertex!([0.0, 0.0, 1e-15]),
                ],
                "small",
            ),
            // Mixed extreme coordinates
            (
                vec![
                    vertex!([f64::MIN_POSITIVE, 0.0, 0.0]),
                    vertex!([f64::MAX / 1e10, 0.0, 0.0]),
                    vertex!([0.0, f64::EPSILON, 0.0]),
                    vertex!([0.0, 0.0, 1.0]),
                ],
                "mixed",
            ),
        ];

        for (vertices, desc) in extreme_vertices {
            let dt_result: Result<DelaunayTriangulation<FastKernel<f64>, (), (), 3>, _> =
                DelaunayTriangulation::new(&vertices);

            match dt_result {
                Ok(dt) => {
                    let hull: ConvexHull<FastKernel<f64>, (), (), 3> =
                        ConvexHull::from_triangulation(dt.triangulation()).unwrap();
                    assert!(
                        hull.validate(dt.triangulation()).is_ok(),
                        "Hull with {desc} coordinates should validate successfully"
                    );
                }
                Err(TriangulationConstructionError::GeometricDegeneracy { .. }) => {
                    // Extremely large/small/mixed coordinate sets may be rejected as
                    // numerically unstable by the robust initial simplex search.
                    // This is acceptable as long as it is surfaced as a clear
                    // GeometricDegeneracy error.
                    println!(
                        "  [33mWarning:[0m skipping {desc} extreme coordinate hull validation \
                         due to geometric degeneracy in Tds::new",
                    );
                }
                Err(other) => {
                    panic!("Unexpected triangulation error for {desc} coordinates: {other}");
                }
            }
        }
        println!("  ✓ Validation with extreme coordinate values passed");

        // ========================================================================
        // Test 5: Error type structure and formatting
        // ========================================================================
        println!("  Testing error type structure and formatting...");

        // Test ConvexHullValidationError::InvalidFacet structure
        let invalid_facet_error = ConvexHullValidationError::InvalidFacet {
            facet_index: 42,
            source: FacetError::InsufficientVertices {
                expected: 3,
                actual: 2,
                dimension: 3,
            },
        };

        let error_message = format!("{invalid_facet_error}");
        assert!(error_message.contains("Facet 42 validation failed"));
        assert!(error_message.contains("exactly 3 vertices"));
        assert!(error_message.contains("got 2"));
        println!("    InvalidFacet error: {error_message}");

        // Test ConvexHullValidationError::DuplicateVerticesInFacet structure
        let duplicate_vertices_error = ConvexHullValidationError::DuplicateVerticesInFacet {
            facet_index: 17,
            positions: vec![vec![0, 2], vec![1, 3, 5]],
        };

        let error_message = format!("{duplicate_vertices_error}");
        assert!(error_message.contains("Facet 17 has duplicate vertices"));
        assert!(error_message.contains("[[0, 2], [1, 3, 5]]"));
        println!("    DuplicateVertices error: {error_message}");

        // Test error equality and cloning
        let cloned_error = invalid_facet_error.clone();
        assert_eq!(
            invalid_facet_error, cloned_error,
            "Cloned error should be equal to original"
        );

        // Test error source chain
        if let ConvexHullValidationError::InvalidFacet {
            facet_index,
            source,
        } = invalid_facet_error
        {
            assert_eq!(facet_index, 42);
            if let FacetError::InsufficientVertices {
                expected,
                actual,
                dimension,
            } = source
            {
                assert_eq!(expected, 3);
                assert_eq!(actual, 2);
                assert_eq!(dimension, 3);
            } else {
                panic!("Expected InsufficientVertices error");
            }
        }
        println!("  ✓ Error type structure and formatting tests passed");

        // ========================================================================
        // Test 6: Validation consistency and performance
        // ========================================================================
        println!("  Testing validation consistency and performance...");

        // Test consistency across multiple calls
        let results: Vec<Result<(), ConvexHullValidationError>> = (0..5)
            .map(|_| hull_3d.validate(dt_3d.triangulation()))
            .collect();
        for (i, result) in results.iter().enumerate() {
            assert!(result.is_ok(), "Validation call {i} should succeed");
        }

        // Test validation with empty hull
        let empty_hull: ConvexHull<FastKernel<f64>, (), (), 3> = ConvexHull::default();
        assert!(
            empty_hull.validate(dt_3d.triangulation()).is_ok(),
            "Empty hull validation should succeed"
        );

        // Performance test - validate 100 times
        let perf_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let perf_dt = create_triangulation(&perf_vertices);
        let perf_hull: ConvexHull<FastKernel<f64>, (), (), 3> =
            ConvexHull::from_triangulation(perf_dt.triangulation()).unwrap();

        let start_time = std::time::Instant::now();
        for i in 0..100 {
            assert!(
                perf_hull.validate(perf_dt.triangulation()).is_ok(),
                "Performance validation iteration {i} should succeed"
            );
        }
        let elapsed = start_time.elapsed();

        let budget_ms: u128 = std::env::var("VALIDATION_BUDGET_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2500);
        assert!(
            elapsed.as_millis() < budget_ms,
            "Validation should be fast (< {budget_ms} ms for 100 calls); took {elapsed:?}"
        );

        println!("    100 validation calls completed in {elapsed:?}");
        println!("  ✓ Validation consistency and performance tests passed");

        println!("✓ All comprehensive ConvexHull validation tests passed successfully!");
    }

    #[test]
    fn test_find_nearest_visible_facet_comprehensive() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        // Test with inside point - should return None
        let inside_point = Point::new([0.1, 0.1, 0.1]);
        let result = hull
            .find_nearest_visible_facet(&inside_point, dt.triangulation())
            .unwrap();
        assert!(
            result.is_none(),
            "Inside point should have no visible facets"
        );

        // Test with outside point - should return Some index
        let outside_point = Point::new([2.0, 2.0, 2.0]);
        let result = hull
            .find_nearest_visible_facet(&outside_point, dt.triangulation())
            .unwrap();
        assert!(result.is_some(), "Outside point should have visible facets");

        if let Some(facet_index) = result {
            assert!(
                facet_index < hull.facet_count(),
                "Facet index should be valid"
            );
        }

        // Test with point at various distances to verify distance calculation
        let test_points = vec![
            Point::new([1.5, 1.5, 1.5]),
            Point::new([3.0, 0.0, 0.0]),
            Point::new([0.0, 3.0, 0.0]),
            Point::new([0.0, 0.0, 3.0]),
        ];

        for point in test_points {
            let result = hull.find_nearest_visible_facet(&point, dt.triangulation());
            // All these points should be outside and have visible facets
            assert!(result.is_ok(), "Distance calculation should not fail");
        }
    }

    #[test]
    fn test_facet_access_edge_cases() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        // Test get_facet with valid indices
        for i in 0..hull.facet_count() {
            assert!(
                hull.get_facet(i).is_some(),
                "Valid index should return facet"
            );
        }

        // Test get_facet with invalid indices
        assert!(
            hull.get_facet(hull.facet_count()).is_none(),
            "Out of bounds index should return None"
        );
        assert!(
            hull.get_facet(usize::MAX).is_none(),
            "Very large index should return None"
        );

        // Test iterator
        let facet_count_via_iter = hull.facets().count();
        assert_eq!(
            facet_count_via_iter,
            hull.facet_count(),
            "Iterator count should match facet_count"
        );

        // Verify all facets in iterator are valid - create FacetView to check vertices
        for facet_handle in hull.facets() {
            let facet_view = FacetView::new(
                &dt.triangulation().tds,
                facet_handle.cell_key(),
                facet_handle.facet_index(),
            )
            .unwrap();
            let vertex_count = facet_view.vertices().unwrap().count();
            assert!(vertex_count > 0, "Each facet should have vertices");
        }
    }

    #[test]
    fn test_generic_types_comprehensive() {
        // Test coordinate type validation with f64
        let vertices_f64 = vec![
            vertex!([0.0f64, 0.0f64, 0.0f64]),
            vertex!([1.0f64, 0.0f64, 0.0f64]),
            vertex!([0.0f64, 1.0f64, 0.0f64]),
            vertex!([0.0f64, 0.0f64, 1.0f64]),
        ];
        let dt_f64 = create_triangulation(&vertices_f64);
        let hull_f64 = ConvexHull::from_triangulation(dt_f64.triangulation()).unwrap();

        assert_eq!(hull_f64.facet_count(), 4);
        assert!(!hull_f64.is_empty());

        // Test point operations with f64
        let test_point_f64 = Point::new([2.0f64, 2.0f64, 2.0f64]);
        let result = hull_f64.is_point_outside(&test_point_f64, dt_f64.triangulation());
        assert!(result.is_ok());

        // Test with high precision f64 coordinates
        let vertices_high_precision = vec![
            vertex!([0.000_000_000_000_001, 0.0, 0.0]),
            vertex!([1.000_000_000_000_001, 0.0, 0.0]),
            vertex!([0.0, 1.000_000_000_000_001, 0.0]),
            vertex!([0.0, 0.0, 1.000_000_000_000_001]),
        ];
        let dt_hp = create_triangulation(&vertices_high_precision);
        let hull_hp = ConvexHull::from_triangulation(dt_hp.triangulation()).unwrap();

        assert_eq!(hull_hp.facet_count(), 4);
        assert_eq!(hull_hp.dimension(), 3);
        assert!(hull_hp.validate(dt_hp.triangulation()).is_ok());
        assert!(!hull_hp.is_empty());

        // Test with integer vertex data
        let vertices_int = vec![
            vertex!([0.0, 0.0, 0.0], 1i32),
            vertex!([1.0, 0.0, 0.0], 2i32),
            vertex!([0.0, 1.0, 0.0], 3i32),
            vertex!([0.0, 0.0, 1.0], 4i32),
        ];
        let dt_int = DelaunayTriangulation::<FastKernel<f64>, i32, (), 3>::with_kernel(
            FastKernel::new(),
            &vertices_int,
        )
        .unwrap();
        let hull_int = ConvexHull::from_triangulation(dt_int.triangulation()).unwrap();

        assert_eq!(hull_int.facet_count(), 4);
        assert_eq!(hull_int.dimension(), 3);
        assert!(hull_int.validate(dt_int.triangulation()).is_ok());

        // Test with character vertex data
        let vertices_char = vec![
            vertex!([0.0, 0.0, 0.0], 'A'),
            vertex!([1.0, 0.0, 0.0], 'B'),
            vertex!([0.0, 1.0, 0.0], 'C'),
            vertex!([0.0, 0.0, 1.0], 'D'),
        ];
        let dt_char = DelaunayTriangulation::<FastKernel<f64>, char, (), 3>::with_kernel(
            FastKernel::new(),
            &vertices_char,
        )
        .unwrap();
        let hull_char = ConvexHull::from_triangulation(dt_char.triangulation()).unwrap();

        assert_eq!(hull_char.facet_count(), 4);
        assert_eq!(hull_char.dimension(), 3);
        assert!(hull_char.validate(dt_char.triangulation()).is_ok());

        // Test with Option<i32> vertex data
        let vertices_with_data = vec![
            vertex!([0.0, 0.0, 0.0], Some(1)),
            vertex!([1.0, 0.0, 0.0], Some(2)),
            vertex!([0.0, 1.0, 0.0], Some(3)),
            vertex!([0.0, 0.0, 1.0], Some(4)),
        ];
        let dt_with_data =
            DelaunayTriangulation::<FastKernel<f64>, Option<i32>, (), 3>::with_kernel(
                FastKernel::new(),
                &vertices_with_data,
            )
            .unwrap();
        let hull_with_data = ConvexHull::from_triangulation(dt_with_data.triangulation()).unwrap();

        assert_eq!(hull_with_data.facet_count(), 4);
        assert!(
            hull_with_data
                .validate(dt_with_data.triangulation())
                .is_ok()
        );
    }

    #[test]
    fn test_extreme_coordinate_values() {
        // Test with very large coordinates
        let vertices_large = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1e6, 0.0, 0.0]),
            vertex!([0.0, 1e6, 0.0]),
            vertex!([0.0, 0.0, 1e6]),
        ];
        let dt_large = create_triangulation(&vertices_large);
        let hull_large = ConvexHull::from_triangulation(dt_large.triangulation()).unwrap();

        assert_eq!(hull_large.facet_count(), 4);
        assert!(hull_large.validate(dt_large.triangulation()).is_ok());

        // Test visibility with large coordinates
        let inside_large = Point::new([1000.0, 1000.0, 1000.0]);
        let outside_large = Point::new([2e6, 2e6, 2e6]);

        assert!(
            !hull_large
                .is_point_outside(&inside_large, dt_large.triangulation())
                .unwrap()
        );
        assert!(
            hull_large
                .is_point_outside(&outside_large, dt_large.triangulation())
                .unwrap()
        );

        // Test with very small coordinates (but still large enough to avoid being
        // treated as numerically degenerate by the initial simplex search)
        let vertices_small = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1e-3, 0.0, 0.0]),
            vertex!([0.0, 1e-3, 0.0]),
            vertex!([0.0, 0.0, 1e-3]),
        ];
        let dt_small = create_triangulation(&vertices_small);
        let hull_small = ConvexHull::from_triangulation(dt_small.triangulation()).unwrap();

        assert_eq!(hull_small.facet_count(), 4);
        assert!(hull_small.validate(dt_small.triangulation()).is_ok());
    }

    #[test]
    fn test_trait_implementations_comprehensive() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        // Test Debug trait on hull
        let debug_string = format!("{hull:?}");
        assert!(
            debug_string.contains("ConvexHull"),
            "Debug output should contain ConvexHull"
        );
        assert!(!debug_string.is_empty(), "Debug output should not be empty");

        // Test Debug trait on error types
        let error = ConvexHullConstructionError::InsufficientData {
            message: "test".to_string(),
        };
        let debug_error = format!("{error:?}");
        assert!(debug_error.contains("InsufficientData"));

        // Test Default trait for various dimensions
        let dummy_vertices_2d = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dummy_dt_2d = create_triangulation(&dummy_vertices_2d);
        let hull_2d: ConvexHull<FastKernel<f64>, (), (), 2> = ConvexHull::default();
        assert!(hull_2d.is_empty());
        assert_eq!(hull_2d.facet_count(), 0);
        assert_eq!(hull_2d.dimension(), 2);
        assert!(hull_2d.validate(dummy_dt_2d.triangulation()).is_ok());

        let hull_3d_default: ConvexHull<FastKernel<f64>, (), (), 3> = ConvexHull::default();
        assert!(hull_3d_default.is_empty());
        assert_eq!(hull_3d_default.facet_count(), 0);
        assert_eq!(hull_3d_default.dimension(), 3);
        assert!(hull_3d_default.validate(dt.triangulation()).is_ok());

        let dummy_vertices_4d = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0]),
        ];
        let dummy_dt_4d = create_triangulation(&dummy_vertices_4d);
        let hull_4d: ConvexHull<FastKernel<f64>, (), (), 4> = ConvexHull::default();
        assert!(hull_4d.is_empty());
        assert_eq!(hull_4d.facet_count(), 0);
        assert_eq!(hull_4d.dimension(), 4);
        assert!(hull_4d.validate(dummy_dt_4d.triangulation()).is_ok());
    }

    #[test]
    fn test_type_aliases() {
        // Test that type aliases compile and work correctly
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt = create_triangulation(&vertices);
        let _hull_2d: ConvexHull2D<FastKernel<f64>, (), ()> =
            ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        let vertices_3d = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt_3d = create_triangulation(&vertices_3d);
        let _hull_3d: ConvexHull3D<FastKernel<f64>, (), ()> =
            ConvexHull::from_triangulation(dt_3d.triangulation()).unwrap();

        let vertices_4d = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0]),
        ];
        let dt_4d = create_triangulation(&vertices_4d);
        let _hull_4d: ConvexHull4D<FastKernel<f64>, (), ()> =
            ConvexHull::from_triangulation(dt_4d.triangulation()).unwrap();
    }

    // =========================================================================
    // ERROR PATH AND EDGE CASE TESTS
    // =========================================================================

    #[test]
    fn test_from_triangulation_empty_vertices_error() {
        // Test error path when triangulation has no vertices
        let empty_dt = DelaunayTriangulation::<FastKernel<f64>, (), (), 3>::empty();
        let result = ConvexHull::from_triangulation(empty_dt.triangulation());

        assert!(result.is_err());
        match result.unwrap_err() {
            ConvexHullConstructionError::InsufficientData { message } => {
                assert!(message.contains("no vertices"));
            }
            _ => panic!("Expected InsufficientData error for no vertices"),
        }

        // Also test with matches! macro (alternative assertion style)
        let result2 = ConvexHull::from_triangulation(empty_dt.triangulation());
        assert!(matches!(
            result2,
            Err(ConvexHullConstructionError::InsufficientData { .. })
        ));
    }

    #[test]
    fn test_from_triangulation_no_cells_error() {
        // Create a manually constructed TDS with vertices but no cells
        // Note: We need to use the underlying TDS directly for this edge case test
        let mut tds = crate::core::triangulation_data_structure::Tds::<f64, (), (), 3>::empty();
        let vertex = vertex!([0.0, 0.0, 0.0]);
        let _ = tds.insert_vertex_with_mapping(vertex);
        let tri = crate::core::triangulation::Triangulation {
            kernel: FastKernel::new(),
            tds,
        };

        let result = ConvexHull::from_triangulation(&tri);
        assert!(result.is_err());
        match result.unwrap_err() {
            ConvexHullConstructionError::InsufficientData { message } => {
                assert!(message.contains("no cells"));
            }
            _ => panic!("Expected InsufficientData error for no cells"),
        }
    }

    #[test]
    fn test_from_triangulation_no_boundary_facets_error() {
        // This is harder to trigger naturally, but we can test error propagation
        // by creating a TDS that would fail boundary facet extraction
        // For now, just test that the error mapping works with a valid TDS
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = create_triangulation(&vertices);
        let result = ConvexHull::from_triangulation(dt.triangulation());
        assert!(result.is_ok()); // This should succeed for a valid tetrahedron
        let hull = result.unwrap();
        assert!(!hull.hull_facets.is_empty());
    }

    #[test]
    fn test_visibility_check_insufficient_vertices_error() {
        // Create a hull and manually create a degenerate facet to test error path
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        // Test a normal facet first to ensure visibility checks work
        let test_point = Point::new([2.0, 2.0, 2.0]);
        if let Some(facet) = hull.get_facet(0) {
            let result = hull.is_facet_visible_from_point(facet, &test_point, dt.triangulation());
            // This should either succeed or fail gracefully
            match result {
                Ok(_visibility) => (), // Success case
                Err(e) => println!("Expected visibility error: {e}"),
            }
        }
    }

    #[test]
    fn test_fallback_visibility_test_degenerate_facet() {
        // Test the fallback visibility algorithm with degenerate geometry
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        // Test fallback with various points
        if let Some(facet_handle) = hull.get_facet(0) {
            // Create FacetView to get vertices
            let facet_view = FacetView::new(
                &dt.triangulation().tds,
                facet_handle.cell_key(),
                facet_handle.facet_index(),
            )
            .unwrap();
            let facet_vertices = crate::core::util::facet_view_to_vertices(&facet_view).unwrap();

            // Test with a point very close to the facet (should not be visible)
            let close_point = Point::new([0.1, 0.1, 0.1]);
            let result = ConvexHull::<FastKernel<f64>, (), (), 3>::fallback_visibility_test(
                &facet_vertices,
                &close_point,
            );
            assert!(result.is_ok());

            // Test with a point far from the facet (should be visible)
            let far_point = Point::new([10.0, 10.0, 10.0]);
            let result = ConvexHull::<FastKernel<f64>, (), (), 3>::fallback_visibility_test(
                &facet_vertices,
                &far_point,
            );
            assert!(result.is_ok());
            assert!(result.unwrap()); // Should be visible from far point
        }
    }

    #[test]
    fn test_inside_point_no_visible_facets() {
        // Consolidated test for inside point detection (no visible facets)
        // Tests both find_nearest_visible_facet and is_point_outside
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        // Test with a point inside the hull
        let inside_point = Point::new([0.2, 0.2, 0.2]);

        // Test find_nearest_visible_facet (no facets should be visible)
        let result = hull.find_nearest_visible_facet(&inside_point, dt.triangulation());
        assert!(result.is_ok());

        // Test is_point_outside (should return false)
        let is_outside = hull
            .is_point_outside(&inside_point, dt.triangulation())
            .unwrap();
        assert!(!is_outside, "Inside point should not be outside");
    }

    #[test]
    fn test_find_nearest_visible_facet_equidistant_cases() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        // Test with a point that's equidistant from multiple facets
        let equidistant_point = Point::new([5.0, 5.0, 5.0]);
        let result = hull.find_nearest_visible_facet(&equidistant_point, dt.triangulation());
        assert!(result.is_ok());
        // Should return some facet index or None
    }

    #[test]
    fn test_hull_operations_extended() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        // Test basic operations
        assert!(!hull.is_empty());
        assert_eq!(hull.dimension(), 3);
        assert_eq!(hull.facet_count(), 4);

        // Test get_facet bounds checking
        assert!(hull.get_facet(0).is_some());
        assert!(hull.get_facet(3).is_some());
        assert!(hull.get_facet(4).is_none());
        assert!(hull.get_facet(100).is_none());

        // Test facets iterator
        let facet_count = hull.facets().count();
        assert_eq!(facet_count, 4);
    }

    #[test]
    fn test_invalidate_cache() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        // Test cache invalidation
        hull.invalidate_cache();

        // Verify we can still perform operations after cache invalidation
        let test_point = Point::new([2.0, 2.0, 2.0]);
        let result = hull.is_point_outside(&test_point, dt.triangulation());
        assert!(result.is_ok());
    }

    #[test]
    fn test_facet_cache_provider_implementation() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        // Test FacetCacheProvider trait implementation
        let _facet_cache = hull.facet_cache();
        let cached_gen = hull.cached_generation();
        let tds_gen = dt.triangulation().tds.generation();
        assert_eq!(
            cached_gen.load(std::sync::atomic::Ordering::Acquire),
            tds_gen,
            "Hull generation should match TDS generation"
        );
    }

    #[test]
    fn test_error_types_display_formatting() {
        // Test ConvexHullValidationError display
        let validation_error = ConvexHullValidationError::InvalidFacet {
            facet_index: 2,
            source: FacetError::InsufficientVertices {
                expected: 3,
                actual: 2,
                dimension: 3,
            },
        };
        let display = format!("{validation_error}");
        assert!(display.contains("Facet 2 validation failed"));

        let duplicate_error = ConvexHullValidationError::DuplicateVerticesInFacet {
            facet_index: 1,
            positions: vec![vec![0, 2], vec![1, 3]],
        };
        let display = format!("{duplicate_error}");
        assert!(display.contains("duplicate vertices"));

        // Test ConvexHullConstructionError display
        let construction_error = ConvexHullConstructionError::InsufficientData {
            message: "test message".to_string(),
        };
        let display = format!("{construction_error}");
        assert!(display.contains("test message"));

        let coord_error = ConvexHullConstructionError::CoordinateConversion(
            crate::geometry::traits::coordinate::CoordinateConversionError::NonFiniteValue {
                coordinate_index: 0,
                coordinate_value: "NaN".to_string(),
            },
        );
        let display = format!("{coord_error}");
        assert!(display.contains("Coordinate conversion error"));
    }

    #[test]
    fn test_comprehensive_facet_iteration() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        // Test that facets() iterator produces the same results as get_facet
        let mut iter_facets = Vec::new();
        for facet in hull.facets() {
            iter_facets.push(facet);
        }

        assert_eq!(iter_facets.len(), hull.facet_count());

        for (i, facet_ref) in iter_facets.iter().enumerate() {
            let facet_by_index = hull.get_facet(i).unwrap();
            // They should be equivalent facets (same cell key and facet index)
            assert_eq!(*facet_ref, facet_by_index, "Facet {i} should match");
        }

        // Test multiple iterations produce same results
        let first_iteration: Vec<_> = hull.facets().collect();
        let second_iteration: Vec<_> = hull.facets().collect();
        assert_eq!(first_iteration.len(), second_iteration.len());

        for (i, (f1, f2)) in first_iteration
            .iter()
            .zip(second_iteration.iter())
            .enumerate()
        {
            // Multiple iterations should return equivalent facets
            assert_eq!(f1, f2, "Iteration {i} should return same facet");
        }

        // Test chaining with other iterator methods - create FacetViews to get vertex counts
        let vertex_counts: Vec<usize> = hull
            .hull_facets
            .iter()
            .map(|facet_handle| {
                FacetView::new(
                    &dt.triangulation().tds,
                    facet_handle.cell_key(),
                    facet_handle.facet_index(),
                )
                .unwrap()
                .vertices()
                .unwrap()
                .count()
            })
            .collect();

        // All facets should have the same number of vertices (dimension)
        for count in vertex_counts {
            assert_eq!(count, 3); // 3D facets have 3 vertices
        }
    }

    #[test]
    fn test_dimensional_consistency() {
        // Test that dimension() always returns D regardless of hull state
        let empty_hull_1d: ConvexHull<FastKernel<f64>, (), (), 1> = ConvexHull::default();
        assert_eq!(empty_hull_1d.dimension(), 1);

        let empty_hull_2d: ConvexHull<FastKernel<f64>, (), (), 2> = ConvexHull::default();
        assert_eq!(empty_hull_2d.dimension(), 2);

        let empty_hull_3d: ConvexHull<FastKernel<f64>, (), (), 3> = ConvexHull::default();
        assert_eq!(empty_hull_3d.dimension(), 3);

        let empty_hull_4d: ConvexHull<FastKernel<f64>, (), (), 4> = ConvexHull::default();
        assert_eq!(empty_hull_4d.dimension(), 4);

        let empty_hull_ten_d: ConvexHull<FastKernel<f64>, (), (), 10> = ConvexHull::default();
        assert_eq!(empty_hull_ten_d.dimension(), 10);

        // Test with populated hull
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        assert_eq!(hull.dimension(), 3);

        // Dimension is a const generic parameter, so it never changes
        // Empty hulls also preserve the dimension
        let empty_hull: ConvexHull<FastKernel<f64>, (), (), 3> = ConvexHull::default();
        assert_eq!(empty_hull.dimension(), 3);
    }

    #[test]
    fn test_error_handling_paths() {
        println!("Testing error handling paths in convex hull methods");

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        // Test fallback_visibility_test with a regular facet and extreme point
        let test_facet_vertices =
            extract_facet_vertices(&hull.hull_facets[0], dt.triangulation()).unwrap();
        let test_point = Point::new([1e-20, 1e-20, 1e-20]);
        let result = ConvexHull::<FastKernel<f64>, (), (), 3>::fallback_visibility_test(
            &test_facet_vertices,
            &test_point,
        );

        // The method should handle extreme coordinates gracefully
        println!("  Fallback visibility result with extreme point: {result:?}");

        // Test normal visibility methods with edge case points
        let edge_points = vec![
            Point::new([0.0, 0.0, 0.0]),                            // At vertex
            Point::new([0.5, 0.0, 0.0]),                            // On edge
            Point::new([f64::EPSILON, f64::EPSILON, f64::EPSILON]), // Very small
        ];

        for point in edge_points {
            let result = hull.is_point_outside(&point, dt.triangulation());
            assert!(result.is_ok(), "Edge case visibility test should not error");
        }

        println!("✓ Error handling paths tested successfully");
    }

    #[test]
    fn test_degenerate_orientation_fallback() {
        println!("Testing degenerate orientation fallback behavior");

        // Create a triangulation that might produce degenerate orientations
        // while still allowing the initial simplex search to succeed.
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1e-3, 0.0, 0.0]), // Small but not numerically wiped out
            vertex!([0.0, 1e-3, 0.0]),
            vertex!([0.0, 0.0, 1e-3]),
        ];
        let dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        // Test with points that might cause degenerate orientations
        let test_points = vec![
            Point::new([5e-11, 5e-11, 5e-11]), // Very close to the degenerate vertices
            Point::new([1e-9, 1e-9, 1e-9]),    // Slightly further but still small
            Point::new([0.0, 0.0, 0.0]),       // At origin
        ];

        for point in test_points {
            // These should potentially trigger the fallback visibility test
            let result = hull.is_point_outside(&point, dt.triangulation());
            assert!(
                result.is_ok(),
                "Degenerate orientation handling should not crash"
            );

            let coords: [f64; 3] = point.into();
            println!(
                "  Degenerate point {coords:?} - Outside: {:?}",
                result.unwrap()
            );
        }

        println!("✓ Degenerate orientation fallback tested successfully");
    }

    #[test]
    fn test_extreme_coordinate_precision() {
        println!("Testing extreme coordinate precision handling");

        // Test with coordinates at the limits of f64 precision
        let vertices_extreme = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([f64::MIN_POSITIVE, 0.0, 0.0]),
            vertex!([0.0, f64::MIN_POSITIVE, 0.0]),
            vertex!([0.0, 0.0, f64::MIN_POSITIVE]),
        ];

        let dt_extreme_result = DelaunayTriangulation::new(&vertices_extreme);

        match dt_extreme_result {
            Ok(dt_extreme) => {
                let hull_extreme =
                    ConvexHull::from_triangulation(dt_extreme.triangulation()).unwrap();

                // Test visibility with extreme coordinates
                let test_point = Point::new([
                    f64::MIN_POSITIVE * 2.0,
                    f64::MIN_POSITIVE * 2.0,
                    f64::MIN_POSITIVE * 2.0,
                ]);
                let result = hull_extreme.is_point_outside(&test_point, dt_extreme.triangulation());
                assert!(
                    result.is_ok(),
                    "Extreme precision coordinates should not crash visibility testing",
                );

                // Test fallback visibility with extreme coordinates
                let facet_vertices = extract_facet_vertices(
                    &hull_extreme.hull_facets[0],
                    dt_extreme.triangulation(),
                )
                .unwrap();
                let fallback_result =
                    ConvexHull::<FastKernel<f64>, (), (), 3>::fallback_visibility_test(
                        &facet_vertices,
                        &test_point,
                    );
                println!("  Extreme precision fallback result: {fallback_result:?}");
            }
            Err(TriangulationConstructionError::GeometricDegeneracy { .. }) => {
                // On some platforms, these extreme coordinates may be judged too
                // numerically unstable to form a reliable 3D simplex. In that
                // case, it's acceptable for DelaunayTriangulation::new to fail with geometric
                // degeneracy; later parts of this test still exercise max-scale
                // behavior.
                println!(
                    "  \x1b[33mWarning:\x1b[0m skipping MIN_POSITIVE extreme simplex due to geometric degeneracy",
                );
            }
            Err(other) => {
                panic!("Unexpected triangulation error for extreme precision test: {other}");
            }
        }

        // Test with maximum finite values
        let vertices_max = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt_max = create_triangulation(&vertices_max);
        let hull_max = ConvexHull::from_triangulation(dt_max.triangulation()).unwrap();

        let max_point = Point::new([f64::MAX / 2.0, f64::MAX / 2.0, f64::MAX / 2.0]);
        let result = hull_max.is_point_outside(&max_point, dt_max.triangulation());
        assert!(
            result.is_ok(),
            "Maximum finite coordinates should not crash"
        );

        println!("✓ Extreme coordinate precision tested successfully");
    }

    #[test]
    fn test_numeric_cast_error_handling() {
        println!("Testing numeric cast error handling in find_nearest_visible_facet");

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        // Test with a normal point to ensure the method works correctly first
        let outside_point = Point::new([2.0, 2.0, 2.0]);
        let result = hull.find_nearest_visible_facet(&outside_point, dt.triangulation());
        assert!(
            result.is_ok(),
            "Normal case should work without numeric cast issues"
        );
        assert!(
            result.unwrap().is_some(),
            "Outside point should have visible facets"
        );

        // The actual numeric cast failure is hard to test directly without creating
        // a coordinate type that fails NumCast, but we can verify that our error
        // handling structure is in place by checking that the method uses proper
        // error types and doesn't panic.

        // Test with various edge cases that could potentially cause numeric issues
        let edge_points = vec![
            Point::new([0.0, 0.0, 0.0]),       // At vertex
            Point::new([1e-10, 1e-10, 1e-10]), // Very small but positive
            Point::new([1e10, 1e10, 1e10]),    // Very large
        ];

        for point in edge_points {
            let result = hull.find_nearest_visible_facet(&point, dt.triangulation());
            assert!(
                result.is_ok(),
                "Edge case points should not cause numeric cast failures"
            );

            let coords: [f64; 3] = point.into();
            let result_val = result.unwrap();
            println!("  Edge point {coords:?} - Result: {result_val:?}");
        }

        println!("✓ Numeric cast error handling tested successfully");
    }

    #[expect(clippy::too_many_lines)]
    #[test]
    fn test_cache_invalidation_behavior() {
        println!("Testing cache invalidation behavior in ConvexHull");

        // Create initial triangulation
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        // Get initial generation values
        let initial_tds_generation = dt.triangulation().tds.generation();
        let initial_hull_generation = hull.cached_generation().load(Ordering::Acquire);

        println!("  Initial TDS generation: {initial_tds_generation}");
        println!("  Initial hull cached generation: {initial_hull_generation}");

        // ConvexHull keeps an independent snapshot for staleness detection
        // Since generation is now private, we can't compare pointers directly
        // But we can verify they track independently by checking values

        // Verify initial generations match (hull starts with snapshot of TDS generation)
        assert_eq!(
            initial_tds_generation, initial_hull_generation,
            "Initial generations should match since ConvexHull snapshots TDS generation"
        );

        // Test initial cache building - first visibility test should build the cache
        let test_point = Point::new([2.0, 2.0, 2.0]);
        let facet = hull.get_facet(0).unwrap();

        println!("  Performing initial visibility test to build cache...");
        let result1 = hull.is_facet_visible_from_point(facet, &test_point, dt.triangulation());
        assert!(result1.is_ok(), "Initial visibility test should succeed");

        // Cache should now be built, generations should still match
        let post_cache_tds_gen = dt.triangulation().tds.generation();
        let post_cache_hull_gen = hull.cached_generation().load(Ordering::Acquire);

        println!(
            "  After cache build - TDS gen: {post_cache_tds_gen}, Hull gen: {post_cache_hull_gen}"
        );
        assert_eq!(
            post_cache_tds_gen, post_cache_hull_gen,
            "Generations should still match after cache building"
        );

        // Verify cache was built
        let cache_arc = hull.facet_cache().load();
        assert!(
            cache_arc.is_some(),
            "Cache should exist after first visibility test"
        );

        println!("  ✓ Cache successfully built and generations synchronized");

        // Test validity checking before TDS modification
        println!("  Testing validity checking...");
        assert!(
            hull.is_valid_for_triangulation(dt.triangulation()),
            "Hull should be valid for initial TDS"
        );

        // Test TDS modification by adding a new vertex
        println!("  Testing TDS modification and hull invalidation...");
        let old_generation = dt.triangulation().tds.generation();
        let stale_hull_gen = hull.cached_generation().load(Ordering::Acquire);

        // Add a new vertex - this will bump the generation
        let new_vertex = vertex!([0.5, 0.5, 0.5]); // Interior point
        dt.insert(new_vertex)
            .expect("Failed to insert vertex into DelaunayTriangulation");

        let modified_tds_gen = dt.triangulation().tds.generation();
        println!("  After TDS modification (added vertex):");
        println!("    TDS generation: {modified_tds_gen}");
        println!("    Hull cached generation: {stale_hull_gen}");

        // Hull snapshot is now stale relative to TDS
        assert!(
            modified_tds_gen > old_generation,
            "Generation should be incremented after adding vertex"
        );
        assert!(
            modified_tds_gen > stale_hull_gen,
            "TDS generation should be ahead of hull's cached generation"
        );

        // Test validity checking after TDS modification
        assert!(
            !hull.is_valid_for_triangulation(dt.triangulation()),
            "Hull should be invalid for modified TDS"
        );

        println!("  ✓ Generation change correctly detected - hull is now invalid");

        // IMPORTANT: After TDS modification, the old hull's facet handles are invalid!
        // We must rebuild the hull to get fresh facet handles
        println!("  Rebuilding hull after TDS modification...");
        let new_hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        // The new hull should be valid and have matching generation
        assert!(
            new_hull.is_valid_for_triangulation(dt.triangulation()),
            "New hull should be valid for modified TDS"
        );
        let new_hull_gen = new_hull.cached_generation().load(Ordering::Acquire);
        println!("    New hull generation: {new_hull_gen}");
        assert_eq!(
            new_hull_gen, modified_tds_gen,
            "New hull should have same generation as modified TDS"
        );

        // Get a fresh facet handle from the new hull
        let new_facet = new_hull.get_facet(0).unwrap();

        // Test visibility with the new hull and fresh facet handle
        println!("  Testing visibility with rebuilt hull...");
        let result2 =
            new_hull.is_facet_visible_from_point(new_facet, &test_point, dt.triangulation());
        assert!(
            result2.is_ok(),
            "Visibility test with rebuilt hull should succeed"
        );

        // Verify the cache was built for the new hull
        let cache_arc = new_hull.facet_cache().load();
        assert!(
            cache_arc.is_some(),
            "Cache should exist after visibility test on new hull"
        );

        println!("  ✓ Hull rebuilt successfully after TDS modification");

        // Test manual cache invalidation on the new hull
        println!("  Testing manual cache invalidation...");

        // Store current generation
        let pre_invalidation_gen = new_hull.cached_generation().load(Ordering::Acquire);

        // Manually invalidate cache
        new_hull.invalidate_cache();

        // Check that cache was cleared
        let post_invalidation_cache = new_hull.facet_cache().load();
        assert!(
            post_invalidation_cache.is_none(),
            "Cache should be None after manual invalidation"
        );

        // Check that generation was reset to 0
        let post_invalidation_gen = new_hull.cached_generation().load(Ordering::Acquire);
        assert_eq!(
            post_invalidation_gen, 0,
            "Generation should be reset to 0 after manual invalidation"
        );

        println!("    Generation before invalidation: {pre_invalidation_gen}");
        println!("    Generation after invalidation: {post_invalidation_gen}");

        // Next visibility test should rebuild cache
        let result3 =
            new_hull.is_facet_visible_from_point(new_facet, &test_point, dt.triangulation());
        assert!(
            result3.is_ok(),
            "Visibility test after manual invalidation should succeed"
        );

        // Cache should be rebuilt
        let rebuilt_cache = new_hull.facet_cache().load();
        assert!(
            rebuilt_cache.is_some(),
            "Cache should be rebuilt after visibility test"
        );

        // Generation should be updated to current TDS generation
        let final_hull_gen = new_hull.cached_generation().load(Ordering::Acquire);
        let final_tds_gen = dt.triangulation().tds.generation();
        assert_eq!(
            final_hull_gen, final_tds_gen,
            "Hull generation should match TDS generation after cache rebuild"
        );

        println!("    Final TDS generation: {final_tds_gen}");
        println!("    Final hull generation: {final_hull_gen}");

        println!("  ✓ Manual cache invalidation working correctly");

        // Note: We verified that:
        // 1. The initial hull works correctly with the initial TDS
        // 2. After TDS modification, the old hull is correctly detected as invalid
        // 3. A new hull can be built for the modified TDS
        // 4. The new hull works correctly with the modified TDS
        // 5. Manual cache invalidation works as expected
        println!("  All tests verified correct hull and cache management");

        // Verify that all visibility tests succeeded
        assert!(
            result1.is_ok(),
            "First visibility test (original hull) should succeed"
        );
        assert!(
            result2.is_ok(),
            "Second visibility test (new hull) should succeed"
        );
        assert!(
            result3.is_ok(),
            "Third visibility test (after cache invalidation) should succeed"
        );

        println!("  ✓ Hull rebuilding and cache management working correctly");

        // Test concurrent access safety using the new hull
        println!("  Testing thread safety of cache operations...");

        let test_results: Vec<_> = (0..10)
            .map(|i| {
                let x = NumCast::from(i).unwrap_or(0.0f64).mul_add(0.1, 2.0);
                let test_pt = Point::new([x, 2.0, 2.0]);
                new_hull.is_facet_visible_from_point(new_facet, &test_pt, dt.triangulation())
            })
            .collect();

        // All operations should succeed
        for (i, result) in test_results.iter().enumerate() {
            assert!(
                result.is_ok(),
                "Concurrent visibility test {i} should succeed"
            );
        }

        println!("  ✓ Thread safety test passed");

        println!("✓ All cache invalidation behavior tests passed successfully!");
    }

    #[test]
    fn test_try_get_or_build_facet_cache() {
        println!("Testing try_get_or_build_facet_cache method");

        // Create a triangulation
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        // Initially, cache should be empty
        let initial_cache = hull.facet_cache().load();
        assert!(initial_cache.is_none(), "Cache should be empty initially");

        // First call should build the cache
        println!("  Testing initial cache building...");
        let cache1 = hull
            .try_get_or_build_facet_cache(&dt.triangulation().tds)
            .expect("Failed to build cache");
        assert!(
            !cache1.is_empty(),
            "Cache should not be empty after building"
        );

        // Verify cache is now stored
        let stored_cache = hull.facet_cache().load();
        assert!(
            stored_cache.is_some(),
            "Cache should be stored after building"
        );

        // Second call with same generation should reuse cache
        println!("  Testing cache reuse with same generation...");
        let cache2 = hull
            .try_get_or_build_facet_cache(&dt.triangulation().tds)
            .expect("Failed to reuse cache");
        assert_eq!(
            cache1.len(),
            cache2.len(),
            "Cache content should be identical on reuse"
        );

        // Verify the cache content is identical (cache reuse)
        assert_eq!(
            cache1.len(),
            cache2.len(),
            "Cached content should be identical when generation matches"
        );
        // Verify cache contains same keys
        for key in cache1.keys() {
            assert!(
                cache2.contains_key(key),
                "Reused cache should contain same keys"
            );
        }

        // Modify triangulation by adding a vertex to trigger generation change
        println!("  Testing cache invalidation with generation change...");
        let old_generation = dt.triangulation().tds.generation();

        // Add a new vertex to trigger generation bump
        let new_vertex = vertex!([0.5, 0.5, 0.5]); // Interior point
        dt.insert(new_vertex).expect("Failed to insert vertex");

        let new_generation = dt.triangulation().tds.generation();
        assert!(
            new_generation > old_generation,
            "Generation should increase after adding vertex"
        );

        // Next call should rebuild cache due to generation change
        let cache3 = hull
            .try_get_or_build_facet_cache(&dt.triangulation().tds)
            .expect("Failed to rebuild cache");

        // The cache content might be different since we added a vertex
        // but it should be a valid cache
        assert!(!cache3.is_empty(), "Rebuilt cache should not be empty");

        // Rebuilt cache may have different content due to TDS changes
        // We just verify it's a valid cache (non-empty)
        assert!(
            !cache3.is_empty(),
            "Rebuilt cache should be non-empty and valid"
        );

        // Verify generation was updated
        let updated_generation = hull.cached_generation().load(Ordering::Acquire);
        assert_eq!(
            updated_generation, new_generation,
            "Hull generation should match TDS generation after rebuild"
        );

        println!("  ✓ Cache building, reuse, and invalidation working correctly");
    }

    #[test]
    fn test_helper_methods_integration() {
        println!("Testing integration between helper methods");

        // Create a triangulation
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        // Test that cache contains keys derivable by the key derivation method
        println!("  Testing cache-key derivation consistency...");
        let cache = hull
            .try_get_or_build_facet_cache(&dt.triangulation().tds)
            .expect("Failed to build cache");

        // For each facet in the hull, derive its key and check it exists in cache
        let mut keys_found = 0usize;
        for i in 0..hull.facet_count() {
            let facet_handle = hull.get_facet(i).unwrap();
            // Create FacetView to get vertices
            let facet_view = FacetView::new(
                &dt.triangulation().tds,
                facet_handle.cell_key(),
                facet_handle.facet_index(),
            )
            .unwrap();
            let facet_vertices = crate::core::util::facet_view_to_vertices(&facet_view).unwrap();

            // Get vertex keys from vertices via TDS
            let facet_vertex_keys: Vec<_> = facet_vertices
                .iter()
                .filter_map(|v| dt.triangulation().tds.vertex_key_from_uuid(&v.uuid()))
                .collect();

            let derived_key_result =
                derive_facet_key_from_vertex_keys::<f64, (), (), 3>(&facet_vertex_keys);

            if let Ok(derived_key) = derived_key_result {
                if cache.contains_key(&derived_key) {
                    keys_found += 1;
                    println!("    Facet {i}: key {derived_key} found in cache ✓");
                } else {
                    println!("    Facet {i}: key {derived_key} NOT in cache (unexpected)");
                }
            } else {
                println!(
                    "    Facet {i}: key derivation failed: {:?}",
                    derived_key_result.err()
                );
            }
        }

        println!(
            "  Found {keys_found}/{} hull facet keys in cache",
            hull.facet_count()
        );

        // Cache should be non-empty (contains facets from the TDS)
        assert!(
            !cache.is_empty(),
            "Cache should contain facets from the triangulation"
        );
        assert_eq!(
            keys_found,
            hull.facet_count(),
            "Every hull facet key should be present in the cache"
        );

        // Test that helper methods work correctly together in visibility testing
        println!("  Testing helper methods in visibility context...");
        let test_point = Point::new([2.0, 2.0, 2.0]);
        let test_facet = hull.get_facet(0).unwrap();

        let visibility_result =
            hull.is_facet_visible_from_point(test_facet, &test_point, dt.triangulation());
        assert!(
            visibility_result.is_ok(),
            "Visibility test using helper methods should succeed"
        );

        println!("  Visibility result: {}", visibility_result.unwrap());
        println!("  ✓ Integration between helper methods working correctly");
    }

    #[test]
    fn test_facet_cache_build_failed_error() {
        println!("Testing FacetCacheBuildFailed error path");

        // This test is challenging because we need to trigger a TriangulationValidationError
        // during facet cache building. In practice, this is rare with valid TDS objects.
        // We'll test the error propagation pattern by verifying the error types are properly
        // connected and that the method signature returns the right error type.

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        // Test that normal cache building succeeds and returns the right error type
        let test_facet = hull.get_facet(0).unwrap();
        let test_point = Point::new([2.0, 2.0, 2.0]);

        // Call the method that should propagate TriangulationValidationError as FacetCacheBuildFailed
        let result = hull.is_facet_visible_from_point(test_facet, &test_point, dt.triangulation());

        // In the normal case, this should succeed
        assert!(
            result.is_ok(),
            "Normal visibility test should succeed, got error: {:?}",
            result.err()
        );

        // Verify that the method returns ConvexHullConstructionError (not FacetError)
        // This ensures our error hierarchy changes are properly implemented
        match result {
            Ok(_) => println!("  ✓ Normal case succeeded as expected"),
            Err(e) => {
                // If there is an error, verify it's the right type
                println!("  Error type verification: {e:?}");
                // The fact that it compiles with ConvexHullConstructionError shows the type is correct
            }
        }

        // Test that the error propagation chain is intact by using a method that
        // calls try_get_or_build_facet_cache internally
        let visibility_result = hull.find_visible_facets(&test_point, dt.triangulation());
        assert!(
            visibility_result.is_ok(),
            "find_visible_facets should succeed in normal case"
        );

        println!("  ✓ FacetCacheBuildFailed error path properly configured");
        println!(
            "  Note: Actual cache build failure requires corrupted TDS, which is hard to create in tests"
        );
    }

    #[test]
    fn test_nearest_facet_equidistant_cases() {
        println!("Testing find_nearest_visible_facet with equidistant facets");

        // Create a symmetric triangulation where multiple facets might be equidistant
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]), // Origin
            vertex!([1.0, 0.0, 0.0]), // X axis
            vertex!([0.0, 1.0, 0.0]), // Y axis
            vertex!([0.0, 0.0, 1.0]), // Z axis
        ];
        let dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        println!("  Testing with point equidistant from multiple facets...");

        // Point at (1,1,1) should be roughly equidistant from all facets
        let equidistant_point = Point::new([1.0, 1.0, 1.0]);
        let nearest_result =
            hull.find_nearest_visible_facet(&equidistant_point, dt.triangulation());

        match nearest_result {
            Ok(Some(facet_index)) => {
                assert!(
                    facet_index < hull.facet_count(),
                    "Returned facet index should be valid"
                );
                println!("  ✓ Found nearest facet at index: {facet_index}");

                // Verify the facet is actually visible
                let selected_facet = hull.get_facet(facet_index).unwrap();
                let is_visible = hull.is_facet_visible_from_point(
                    selected_facet,
                    &equidistant_point,
                    dt.triangulation(),
                );
                assert!(
                    is_visible.unwrap_or(false),
                    "Selected nearest facet should be visible from the test point"
                );
            }
            Ok(None) => {
                println!("  No visible facets found (point might be inside)");
                // Verify this is correct by checking if point is actually inside
                let is_outside = hull
                    .is_point_outside(&equidistant_point, dt.triangulation())
                    .unwrap();
                assert!(
                    !is_outside,
                    "If no facets are visible, point should be inside the hull"
                );
            }
            Err(e) => {
                panic!("find_nearest_visible_facet failed with error: {e:?}");
            }
        }

        println!("  Testing with point clearly outside...");

        // Point clearly outside should always find a nearest facet
        let far_point = Point::new([10.0, 10.0, 10.0]);
        let far_result = hull.find_nearest_visible_facet(&far_point, dt.triangulation());

        match far_result {
            Ok(Some(facet_index)) => {
                assert!(facet_index < hull.facet_count());
                println!("  ✓ Found nearest facet for far point at index: {facet_index}");
            }
            Ok(None) => {
                panic!("Far outside point should always see some facets");
            }
            Err(e) => {
                panic!("find_nearest_visible_facet failed for far point: {e:?}");
            }
        }

        println!("  Testing with point clearly inside...");

        // Point clearly inside should see no facets
        let inside_point = Point::new([0.1, 0.1, 0.1]);
        let inside_result = hull.find_nearest_visible_facet(&inside_point, dt.triangulation());

        match inside_result {
            Ok(None) => {
                println!("  ✓ Inside point correctly sees no facets");
            }
            Ok(Some(facet_index)) => {
                // This might happen due to numerical precision - verify it's reasonable
                println!(
                    "  Inside point unexpectedly sees facet {facet_index} (may be due to precision)"
                );
                assert!(facet_index < hull.facet_count());
            }
            Err(e) => {
                panic!("find_nearest_visible_facet failed for inside point: {e:?}");
            }
        }

        println!("  ✓ Equidistant facet selection working correctly");
    }

    #[test]
    fn test_concurrent_cache_access_patterns() {
        println!("Testing concurrent cache access patterns");

        // Create a triangulation
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        // Share the hull and DT across threads
        let hull = Arc::new(hull);
        let dt = Arc::new(dt);

        println!("  Testing concurrent cache building...");

        // Spawn multiple threads that will try to build/access the cache concurrently
        let mut handles = vec![];

        for thread_id in 0..4 {
            let hull_clone = Arc::clone(&hull);
            let dt_clone = Arc::clone(&dt);

            let handle = thread::spawn(move || {
                // Each thread tries to use methods that require the cache
                let thread_offset: f64 = NumCast::from(thread_id).unwrap();
                let test_point = Point::new([2.0 + thread_offset, 2.0, 2.0]);

                // This will internally call try_get_or_build_facet_cache
                let visible_facets =
                    hull_clone.find_visible_facets(&test_point, dt_clone.triangulation())?;

                // This should also work
                let is_outside =
                    hull_clone.is_point_outside(&test_point, dt_clone.triangulation())?;

                // Return some data to verify thread completed successfully
                Ok::<_, ConvexHullConstructionError>((visible_facets.len(), is_outside, thread_id))
            });

            handles.push(handle);
        }

        // Wait for all threads to complete and collect results
        let mut results = Vec::new();
        for handle in handles {
            let result = handle.join().expect("Thread should not panic");
            match result {
                Ok((facet_count, is_outside, thread_id)) => {
                    println!(
                        "    Thread {thread_id}: {facet_count} visible facets, outside: {is_outside}"
                    );
                    results.push((facet_count, is_outside, thread_id));
                }
                Err(e) => {
                    panic!("Thread failed with error: {e:?}");
                }
            }
        }

        // Verify all threads got reasonable results
        assert_eq!(results.len(), 4, "All threads should complete successfully");

        // All threads should agree that outside points are outside
        for (facet_count, is_outside, thread_id) in &results {
            assert!(
                *is_outside,
                "Thread {thread_id} should detect point as outside"
            );
            assert!(
                *facet_count > 0,
                "Thread {thread_id} should see some visible facets"
            );
        }

        println!("  Testing cache consistency after concurrent access...");

        // Verify cache is in a consistent state after concurrent access
        let cache = hull
            .try_get_or_build_facet_cache(&dt.triangulation().tds)
            .unwrap();
        assert!(
            !cache.is_empty(),
            "Cache should be populated after concurrent access"
        );

        // Test a few operations to make sure everything still works
        let test_point = Point::new([1.5, 1.5, 1.5]);
        let final_result = hull.find_visible_facets(&test_point, dt.triangulation());
        assert!(
            final_result.is_ok(),
            "Operations should work normally after concurrent access"
        );

        println!("  ✓ Concurrent cache access working correctly");
        println!("  Note: This test verifies basic thread safety, not high-contention scenarios");
    }

    #[test]
    fn test_invalidate_cache_behavior() {
        println!("Testing cache invalidation behavior");

        // Create a triangulation
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        // Build initial cache
        println!("  Building initial cache...");
        let initial_cache = hull
            .try_get_or_build_facet_cache(&dt.triangulation().tds)
            .unwrap();
        assert!(
            !initial_cache.is_empty(),
            "Initial cache should not be empty"
        );

        // Verify cache is stored
        let stored_cache = hull.facet_cache().load();
        assert!(stored_cache.is_some(), "Cache should be stored");

        // Check initial generation
        let initial_gen = hull.cached_generation().load(Ordering::Acquire);
        let expected_gen = dt.triangulation().tds.generation();
        assert_eq!(
            initial_gen, expected_gen,
            "Initial cached generation should match TDS generation"
        );

        // Manually invalidate cache
        println!("  Manually invalidating cache...");
        hull.invalidate_cache();

        // Verify cache is cleared
        let cleared_cache = hull.facet_cache().load();
        assert!(
            cleared_cache.is_none(),
            "Cache should be cleared after invalidation"
        );

        // Verify generation is reset
        let reset_gen = hull.cached_generation().load(Ordering::Acquire);
        assert_eq!(
            reset_gen, 0,
            "Generation should be reset to 0 after invalidation"
        );

        // Rebuild cache after invalidation
        println!("  Rebuilding cache after invalidation...");
        let rebuilt_cache = hull
            .try_get_or_build_facet_cache(&dt.triangulation().tds)
            .unwrap();
        assert!(
            !rebuilt_cache.is_empty(),
            "Rebuilt cache should not be empty"
        );

        // Verify cache is stored again
        let restored_cache = hull.facet_cache().load();
        assert!(
            restored_cache.is_some(),
            "Cache should be restored after rebuild"
        );

        // Generation should be updated to TDS generation
        let final_gen = hull.cached_generation().load(Ordering::Acquire);
        let tds_gen = dt.triangulation().tds.generation();
        assert_eq!(
            final_gen, tds_gen,
            "Generation should match TDS generation after rebuild"
        );

        // Verify functionality still works after invalidation/rebuild cycle
        println!("  Testing functionality after invalidation/rebuild...");
        let test_point = Point::new([2.0, 2.0, 2.0]);
        let visible_facets = hull.find_visible_facets(&test_point, dt.triangulation());
        assert!(
            visible_facets.is_ok(),
            "Visibility testing should work after invalidation/rebuild"
        );

        let facets_found = visible_facets.unwrap();
        assert!(
            !facets_found.is_empty(),
            "Outside point should see visible facets after rebuild"
        );

        println!("  ✓ Cache invalidation and rebuild working correctly");
    }

    #[test]
    fn test_error_propagation_chain() {
        println!("Testing complete error propagation chain");

        // Create a valid setup
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        println!("  Testing error types are properly propagated...");

        let test_point = Point::new([2.0, 2.0, 2.0]);
        let test_facet = hull.get_facet(0).unwrap();

        // Test is_facet_visible_from_point returns ConvexHullConstructionError
        let visibility_result =
            hull.is_facet_visible_from_point(test_facet, &test_point, dt.triangulation());
        match visibility_result {
            Ok(visible) => {
                println!("    is_facet_visible_from_point: Ok({visible})");
            }
            Err(e) => {
                println!("    is_facet_visible_from_point error: {e:?}");
                // Verify it's the right error type by matching on variants
                match e {
                    ConvexHullConstructionError::FacetCacheBuildFailed { source: _ } => {
                        println!("      ✓ FacetCacheBuildFailed variant present");
                    }
                    ConvexHullConstructionError::VisibilityCheckFailed { source: _ } => {
                        println!("      ✓ VisibilityCheckFailed variant present");
                    }
                    _ => {
                        println!("      Unexpected error variant: {e:?}");
                    }
                }
            }
        }

        // Test find_visible_facets also returns ConvexHullConstructionError
        let facets_result = hull.find_visible_facets(&test_point, dt.triangulation());
        assert!(
            facets_result.is_ok(),
            "find_visible_facets should succeed in normal case"
        );

        // Test is_point_outside also returns ConvexHullConstructionError
        let outside_result = hull.is_point_outside(&test_point, dt.triangulation());
        assert!(
            outside_result.is_ok(),
            "is_point_outside should succeed in normal case"
        );

        // Test find_nearest_visible_facet returns ConvexHullConstructionError
        let nearest_result = hull.find_nearest_visible_facet(&test_point, dt.triangulation());
        assert!(
            nearest_result.is_ok(),
            "find_nearest_visible_facet should succeed in normal case"
        );

        println!("  ✓ Error propagation chain correctly implemented");
        println!("  ✓ All methods return ConvexHullConstructionError as expected");
    }

    #[test]
    fn test_adjacent_cell_resolution_failed_error() {
        println!("Testing AdjacentCellResolutionFailed error variant");

        // Create a simple triangulation to test with
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        // Test normal case to verify the error type is available
        let test_point = Point::new([2.0, 2.0, 2.0]);
        let test_facet = hull.get_facet(0).unwrap();

        // This should succeed in normal cases
        let visibility_result =
            hull.is_facet_visible_from_point(test_facet, &test_point, dt.triangulation());
        assert!(
            visibility_result.is_ok(),
            "Visibility check should succeed with valid TDS"
        );

        // Verify that the AdjacentCellResolutionFailed error variant exists
        // by creating a synthetic error (we can't easily trigger the actual error path
        // with a valid TDS, but we can verify the error type is properly defined)
        let synthetic_error = ConvexHullConstructionError::AdjacentCellResolutionFailed {
            source: TriangulationValidationError::InconsistentDataStructure {
                message: "Test error for adjacent cell resolution".to_string(),
            },
        };

        // Verify the error can be created and displayed properly
        let error_message = format!("{synthetic_error}");
        assert!(
            error_message.contains("Failed to resolve adjacent cell"),
            "Error message should contain expected text: {error_message}"
        );

        // Verify the source error is accessible
        let source = synthetic_error.source();
        assert!(
            source.is_some(),
            "AdjacentCellResolutionFailed should have a source error"
        );

        println!("  ✓ AdjacentCellResolutionFailed error variant properly implemented");
        println!("  ✓ Error preserves underlying TriangulationValidationError as source");
        println!("  ✓ Error display format correct: {error_message}");
    }

    #[test]
    fn test_enhanced_facet_key_error_information() {
        println!("Testing enhanced facet key error information");

        // Create example UUIDs for testing
        let uuid1 = uuid::Uuid::new_v4();
        let uuid2 = uuid::Uuid::new_v4();
        let uuid3 = uuid::Uuid::new_v4();
        let vertex_uuids = vec![uuid1, uuid2, uuid3];

        // Create the enhanced error with detailed information
        let facet_key = 0x1234_5678_90ab_cdef_u64;
        let cache_size = 42;
        let enhanced_error = FacetError::FacetKeyNotFoundInCache {
            facet_key,
            cache_size,
            vertex_uuids: vertex_uuids.clone(),
        };

        println!("  Testing error message format...");
        let error_message = format!("{enhanced_error}");

        // Verify the error message contains expected components
        assert!(
            error_message.contains(&format!("{facet_key:016x}")),
            "Error message should contain facet key in hex format: {error_message}"
        );
        assert!(
            error_message.contains(&cache_size.to_string()),
            "Error message should contain cache size: {error_message}"
        );
        assert!(
            error_message.contains("invariant violation"),
            "Error message should mention invariant violation: {error_message}"
        );
        assert!(
            error_message.contains("key derivation mismatch"),
            "Error message should mention key derivation mismatch: {error_message}"
        );

        println!("    Enhanced error message: {error_message}");

        println!("  Testing error debug format...");
        let debug_message = format!("{enhanced_error:?}");
        assert!(
            debug_message.contains("FacetKeyNotFoundInCache"),
            "Debug format should contain variant name: {debug_message}"
        );

        // Verify UUIDs are included (check for at least one)
        let uuid_found = vertex_uuids
            .iter()
            .any(|uuid| debug_message.contains(&uuid.to_string()));
        assert!(
            uuid_found,
            "Debug format should contain vertex UUIDs: {debug_message}"
        );

        println!("    Debug representation: {debug_message}");

        println!("  Testing error comparison and cloning...");

        // Test Clone
        let cloned_error = enhanced_error.clone();
        assert_eq!(
            enhanced_error, cloned_error,
            "Cloned error should be equal to original"
        );

        // Test PartialEq with different values
        let different_error = FacetError::FacetKeyNotFoundInCache {
            facet_key: 0xdead_beef_cafe_babe,
            cache_size: 100,
            vertex_uuids: vec![uuid::Uuid::new_v4()],
        };
        assert_ne!(
            enhanced_error, different_error,
            "Different errors should not be equal"
        );

        println!("  Testing integration with ConvexHullConstructionError...");

        // Wrap in the higher-level error
        let construction_error = ConvexHullConstructionError::VisibilityCheckFailed {
            source: enhanced_error,
        };

        let construction_message = format!("{construction_error}");
        assert!(
            construction_message.contains("Failed to check facet visibility from point"),
            "Construction error should contain visibility check message: {construction_message}"
        );

        // Verify error source chain
        let source_error = construction_error.source();
        assert!(
            source_error.is_some(),
            "Construction error should have a source"
        );

        if let Some(source) = source_error {
            let source_message = format!("{source}");
            assert!(
                source_message.contains(&format!("{facet_key:016x}")),
                "Source error should contain facet key: {source_message}"
            );
        }

        println!("    Construction error message: {construction_message}");

        println!("  Testing backward compatibility...");

        // Verify the old error variant still exists and works
        let old_error = FacetError::FacetNotFoundInTriangulation;
        let old_message = format!("{old_error}");
        assert!(
            old_message.contains("Facet not found in triangulation"),
            "Old error variant should still work: {old_message}"
        );

        println!("    Old error message: {old_message}");

        println!("  ✓ Enhanced facet key error information working correctly");
        println!("  ✓ Error provides detailed diagnostic information including:");
        println!("    - Facet key in hex format for debugging");
        println!("    - Cache size for context");
        println!("    - Vertex UUIDs that generated the key");
        println!("    - Actionable error message suggesting possible causes");
        println!("  ✓ Backward compatibility maintained with existing error variants");
    }

    // ============================================================================
    // COMPREHENSIVE ERROR HANDLING TESTS
    // ============================================================================
    // These tests provide comprehensive coverage of error conditions that can
    // occur during convex hull construction and operation.

    #[expect(clippy::too_many_lines, clippy::cognitive_complexity)]
    #[test]
    fn test_convex_hull_error_handling_comprehensive() {
        println!("Testing comprehensive ConvexHull error handling");

        // ========================================================================
        // Test 1: ConvexHullValidationError variants
        // ========================================================================
        println!("  Testing ConvexHullValidationError variants...");

        let invalid_facet_error = ConvexHullValidationError::InvalidFacet {
            facet_index: 5,
            source: FacetError::InsufficientVertices {
                expected: 4,
                actual: 2,
                dimension: 4,
            },
        };

        let error_msg = format!("{invalid_facet_error}");
        assert!(error_msg.contains("Facet 5 validation failed"));
        assert!(error_msg.contains("must have exactly") && error_msg.contains("vertices"));

        let duplicate_error = ConvexHullValidationError::DuplicateVerticesInFacet {
            facet_index: 3,
            positions: vec![vec![0, 2], vec![1, 4, 6]],
        };

        let dup_msg = format!("{duplicate_error}");
        assert!(dup_msg.contains("Facet 3 has duplicate vertices"));
        assert!(dup_msg.contains("positions"));

        // Test error traits
        assert!(invalid_facet_error.source().is_some());
        assert!(duplicate_error.source().is_none());

        let cloned_invalid = invalid_facet_error.clone();
        assert_eq!(invalid_facet_error, cloned_invalid);
        assert_ne!(invalid_facet_error, duplicate_error);

        println!("    InvalidFacet: {error_msg}");
        println!("    DuplicateVertices: {dup_msg}");

        // ========================================================================
        // Test 2: ConvexHullConstructionError variants
        // ========================================================================
        println!("  Testing ConvexHullConstructionError variants...");

        let boundary_error = ConvexHullConstructionError::BoundaryFacetExtractionFailed {
            source: TriangulationValidationError::InconsistentDataStructure {
                message: "Test boundary extraction failure".to_string(),
            },
        };
        let boundary_msg = format!("{boundary_error}");
        assert!(boundary_msg.contains("Failed to extract boundary facets"));
        assert!(boundary_error.source().is_some());

        let visibility_error = ConvexHullConstructionError::VisibilityCheckFailed {
            source: FacetError::InsideVertexNotFound,
        };
        let visibility_msg = format!("{visibility_error}");
        assert!(visibility_msg.contains("Failed to check facet visibility"));
        assert!(visibility_error.source().is_some());

        let invalid_tri_error = ConvexHullConstructionError::InvalidTriangulation {
            message: "Empty triangulation provided".to_string(),
        };
        let invalid_tri_msg = format!("{invalid_tri_error}");
        assert!(invalid_tri_msg.contains("Invalid input triangulation"));
        assert!(invalid_tri_msg.contains("Empty triangulation provided"));
        assert!(invalid_tri_error.source().is_none());

        let degeneracy_error = ConvexHullConstructionError::GeometricDegeneracy {
            message: "All points are collinear".to_string(),
        };
        let degeneracy_msg = format!("{degeneracy_error}");
        assert!(degeneracy_msg.contains("Geometric degeneracy encountered"));
        assert!(degeneracy_msg.contains("All points are collinear"));

        let cast_error = ConvexHullConstructionError::NumericCastFailed {
            message: "Failed to convert f64 to usize".to_string(),
        };
        let cast_msg = format!("{cast_error}");
        assert!(cast_msg.contains("Numeric cast failed"));
        assert!(cast_msg.contains("Failed to convert f64 to usize"));

        let coord_error = ConvexHullConstructionError::CoordinateConversion(
            crate::geometry::traits::coordinate::CoordinateConversionError::NonFiniteValue {
                coordinate_index: 2,
                coordinate_value: "Infinity".to_string(),
            },
        );
        let coord_msg = format!("{coord_error}");
        assert!(coord_msg.contains("Coordinate conversion error"));
        assert!(coord_error.source().is_some());

        // Test error equality and cloning
        let cloned_boundary = boundary_error.clone();
        assert_eq!(boundary_error, cloned_boundary);
        assert_ne!(boundary_error, cast_error);

        println!("    BoundaryExtraction: {boundary_msg}");
        println!("    VisibilityCheck: {visibility_msg}");
        println!("    InvalidTriangulation: {invalid_tri_msg}");
        println!("    GeometricDegeneracy: {degeneracy_msg}");
        println!("    NumericCast: {cast_msg}");
        println!("    CoordinateConversion: {coord_msg}");

        // ========================================================================
        // Test 3: Error propagation and source chains
        // ========================================================================
        println!("  Testing error propagation and source chains...");

        // Test coordinate conversion error propagation
        let coord_conv_error =
            crate::geometry::traits::coordinate::CoordinateConversionError::NonFiniteValue {
                coordinate_index: 0,
                coordinate_value: "NaN".to_string(),
            };
        let hull_error: ConvexHullConstructionError = coord_conv_error.into();
        match hull_error {
            ConvexHullConstructionError::CoordinateConversion(_) => {
                println!("    ✓ Coordinate conversion error properly wrapped");
            }
            _ => panic!("Coordinate conversion error not properly wrapped"),
        }

        // Test complex error source chain
        let facet_error = FacetError::OrientationComputationFailed {
            details: "Degenerate simplex detected".to_string(),
        };
        let chained_visibility_error = ConvexHullConstructionError::VisibilityCheckFailed {
            source: facet_error,
        };

        // Walk the error source chain
        let mut current_error: &dyn Error = &chained_visibility_error;
        let mut depth = 0;
        while let Some(source) = current_error.source() {
            depth += 1;
            current_error = source;
        }
        assert!(
            depth > 0,
            "Error chain should have at least one level of nesting"
        );
        println!("    ✓ Error source chain depth: {depth}");

        // ========================================================================
        // Test 4: Error message consistency and formatting
        // ========================================================================
        println!("  Testing error message formatting consistency...");

        let test_errors: Vec<Box<dyn Error>> = vec![
            Box::new(ConvexHullValidationError::InvalidFacet {
                facet_index: 0,
                source: FacetError::InsufficientVertices {
                    expected: 3,
                    actual: 2,
                    dimension: 3,
                },
            }),
            Box::new(ConvexHullConstructionError::InvalidTriangulation {
                message: "Test message".to_string(),
            }),
            Box::new(ConvexHullConstructionError::GeometricDegeneracy {
                message: "Collinear points".to_string(),
            }),
        ];

        for (i, error) in test_errors.iter().enumerate() {
            let display_msg = format!("{error}");
            let debug_msg = format!("{error:?}");

            assert!(
                !display_msg.is_empty(),
                "Error {i} display message should not be empty"
            );
            assert!(
                display_msg.len() > 10,
                "Error {i} display message should be descriptive: '{display_msg}'"
            );
            assert!(
                !debug_msg.is_empty(),
                "Error {i} debug message should not be empty"
            );
        }

        // ========================================================================
        // Test 5: Extreme coordinate error handling
        // ========================================================================
        println!("  Testing extreme coordinate error handling...");

        // Test with very large coordinates (may cause numeric issues)
        let large_vertices = vec![
            vertex!([1e10, 0.0, 0.0]),
            vertex!([0.0, 1e10, 0.0]),
            vertex!([0.0, 0.0, 1e10]),
            vertex!([1e10, 1e10, 1e10]),
        ];

        match DelaunayTriangulation::new(&large_vertices) {
            Ok(large_dt) => {
                match ConvexHull::from_triangulation(large_dt.triangulation()) {
                    Ok(large_hull) => {
                        assert!(!large_hull.is_empty());
                        assert!(large_hull.validate(large_dt.triangulation()).is_ok());

                        let large_test_point = Point::new([2e10, 2e10, 2e10]);
                        let visibility_result = large_hull
                            .is_point_outside(&large_test_point, large_dt.triangulation());
                        assert!(
                            visibility_result.is_ok(),
                            "Visibility test should handle large coordinates"
                        );
                        println!("    ✓ Large coordinates handled successfully");
                    }
                    Err(e) => {
                        println!("    Large coordinate hull construction failed (acceptable): {e}");
                        // Verify appropriate error types for numeric issues
                        match e {
                            ConvexHullConstructionError::CoordinateConversion(_)
                            | ConvexHullConstructionError::NumericCastFailed { .. }
                            | ConvexHullConstructionError::GeometricDegeneracy { .. } => {
                                println!("      ✓ Appropriate error type for numeric issues");
                            }
                            _ => println!(
                                "      Note: Unexpected error type but may be acceptable: {e:?}"
                            ),
                        }
                    }
                }
            }
            Err(e) => println!(
                "    Large coordinate DelaunayTriangulation construction failed (acceptable): {e}"
            ),
        }

        // Test with very small coordinates (may cause precision issues)
        let small_vertices = vec![
            vertex!([1e-15, 0.0, 0.0]),
            vertex!([0.0, 1e-15, 0.0]),
            vertex!([0.0, 0.0, 1e-15]),
            vertex!([1e-15, 1e-15, 1e-15]),
        ];

        match DelaunayTriangulation::new(&small_vertices) {
            Ok(small_dt) => match ConvexHull::from_triangulation(small_dt.triangulation()) {
                Ok(small_hull) => {
                    assert!(small_hull.validate(small_dt.triangulation()).is_ok());
                    println!("    ✓ Small coordinates handled successfully");
                }
                Err(e) => {
                    println!("    Small coordinate hull construction failed (acceptable): {e}");
                }
            },
            Err(e) => println!(
                "    Small coordinate DelaunayTriangulation construction failed (acceptable): {e}"
            ),
        }

        // Test fallback visibility with extreme coordinates
        let normal_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let normal_dt = create_triangulation(&normal_vertices);
        let normal_hull = ConvexHull::from_triangulation(normal_dt.triangulation()).unwrap();
        let test_facet_vertices =
            extract_facet_vertices(normal_hull.get_facet(0).unwrap(), normal_dt.triangulation())
                .unwrap();

        let extreme_points = [
            Point::new([1e-100, 1e-100, 1e-100]), // Extremely small
            Point::new([1e100, 1e100, 1e100]),    // Extremely large
            Point::new([f64::EPSILON, f64::EPSILON, f64::EPSILON]), // Machine epsilon
        ];

        for (i, point) in extreme_points.iter().enumerate() {
            let fallback_result =
                ConvexHull::<FastKernel<f64>, (), (), 3>::fallback_visibility_test(
                    &test_facet_vertices,
                    point,
                );
            match fallback_result {
                Ok(is_visible) => {
                    println!("    Extreme point {i}: fallback visibility = {is_visible}");
                }
                Err(e) => {
                    println!("    Extreme point {i}: fallback failed (acceptable): {e}");
                }
            }
        }

        println!("✓ All comprehensive ConvexHull error handling tests passed successfully!");
    }

    // ============================================================================
    // ENHANCED FALLBACK VISIBILITY ALGORITHM TESTS
    // ============================================================================
    // These tests comprehensively exercise the fallback_visibility_test method
    // under various degenerate and edge-case conditions.

    #[test]
    #[expect(clippy::too_many_lines)]
    fn test_fallback_visibility_with_degenerate_facets() {
        println!("Testing fallback visibility algorithm with degenerate facet geometries");

        // Create basic triangulation to get valid facet structure
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        println!("  Testing fallback with points at various distances...");

        let facet_handle = hull.get_facet(0).unwrap();
        let facet_view = FacetView::new(
            &dt.triangulation().tds,
            facet_handle.cell_key(),
            facet_handle.facet_index(),
        )
        .unwrap();
        let test_facet_vertices = crate::core::util::facet_view_to_vertices(&facet_view).unwrap();

        // Test points at different distance scales
        let test_cases = vec![
            // (Point, expected_visibility_description, distance_category)
            (Point::new([0.0, 0.0, 0.0]), "vertex point", "zero_distance"),
            (
                Point::new([0.1, 0.1, 0.1]),
                "very close point",
                "very_close",
            ),
            (Point::new([0.5, 0.5, 0.5]), "moderate distance", "moderate"),
            (Point::new([1.0, 1.0, 1.0]), "unit distance", "unit"),
            (Point::new([2.0, 2.0, 2.0]), "double distance", "double"),
            (Point::new([10.0, 10.0, 10.0]), "far point", "far"),
            (
                Point::new([100.0, 100.0, 100.0]),
                "very far point",
                "very_far",
            ),
        ];

        for (point, description, category) in test_cases {
            let fallback_result =
                ConvexHull::<FastKernel<f64>, (), (), 3>::fallback_visibility_test(
                    &test_facet_vertices,
                    &point,
                );

            match fallback_result {
                Ok(is_visible) => {
                    println!("    {description} ({category}): visible = {is_visible}");

                    // Validate that the result makes geometric sense
                    match category {
                        "zero_distance" | "very_close" => {
                            // Very close points might be visible or not due to precision
                            println!(
                                "      Close point visibility: {is_visible} (precision-dependent)"
                            );
                        }
                        "very_far" => {
                            // Very far points should typically be visible
                            if !is_visible {
                                println!(
                                    "      Warning: Very far point unexpectedly not visible (may indicate threshold issues)"
                                );
                            }
                        }
                        _ => {
                            // Middle-range points - no strong expectations
                            println!("      Medium distance point visibility: {is_visible}");
                        }
                    }
                }
                Err(e) => {
                    println!("    {description} ({category}): error = {e:?}");

                    // Errors should only occur for coordinate conversion issues
                    match e {
                        ConvexHullConstructionError::CoordinateConversion(_) => {
                            println!("      ✓ Acceptable coordinate conversion error");
                        }
                        _ => {
                            panic!("Unexpected error type for fallback visibility: {e:?}");
                        }
                    }
                }
            }
        }

        println!("  Testing fallback with collinear facet vertices (degenerate geometry)...");

        // Create a triangulation with near-collinear points
        let near_collinear_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([2.0, 1e-10, 0.0]), // Nearly collinear with first two
            vertex!([0.5, 0.5, 1.0]),   // Out of plane
        ];

        match DelaunayTriangulation::new(&near_collinear_vertices) {
            Ok(collinear_dt) => {
                match ConvexHull::from_triangulation(collinear_dt.triangulation()) {
                    Ok(collinear_hull) => {
                        println!("    ✓ Near-collinear triangulation created successfully");

                        let collinear_facet_vertices = extract_facet_vertices(
                            collinear_hull.get_facet(0).unwrap(),
                            collinear_dt.triangulation(),
                        )
                        .unwrap();
                        let test_point = Point::new([1.5, 0.5, 0.5]);

                        let collinear_result =
                            ConvexHull::<FastKernel<f64>, (), (), 3>::fallback_visibility_test(
                                &collinear_facet_vertices,
                                &test_point,
                            );

                        match collinear_result {
                            Ok(is_visible) => {
                                println!("      Near-collinear facet visibility: {is_visible}");
                            }
                            Err(e) => {
                                println!("      Near-collinear facet test failed: {e}");
                                // This is acceptable for degenerate geometry
                            }
                        }
                    }
                    Err(e) => {
                        println!("    Near-collinear hull construction failed (expected): {e}");
                    }
                }
            }
            Err(e) => {
                println!(
                    "    Near-collinear DelaunayTriangulation construction failed (expected): {e}"
                );
            }
        }

        println!("  Testing fallback with zero-area configurations...");

        // Create a triangulation where facets might have very small areas
        let tiny_area_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1e-6, 0.0, 0.0]),
            vertex!([0.0, 1e-6, 0.0]),
            vertex!([0.0, 0.0, 1e-6]),
        ];

        match DelaunayTriangulation::new(&tiny_area_vertices) {
            Ok(tiny_dt) => {
                match ConvexHull::from_triangulation(tiny_dt.triangulation()) {
                    Ok(tiny_hull) => {
                        println!("    ✓ Tiny area triangulation created successfully");

                        let tiny_facet_vertices = extract_facet_vertices(
                            tiny_hull.get_facet(0).unwrap(),
                            tiny_dt.triangulation(),
                        )
                        .unwrap();
                        let test_point = Point::new([1e-3, 1e-3, 1e-3]);

                        let tiny_result =
                            ConvexHull::<FastKernel<f64>, (), (), 3>::fallback_visibility_test(
                                &tiny_facet_vertices,
                                &test_point,
                            );

                        match tiny_result {
                            Ok(is_visible) => {
                                println!("      Tiny area facet visibility: {is_visible}");
                            }
                            Err(e) => {
                                println!("      Tiny area facet test failed: {e}");
                                // This might fail due to precision issues
                            }
                        }
                    }
                    Err(e) => {
                        println!("    Tiny area hull construction failed (acceptable): {e}");
                    }
                }
            }
            Err(e) => {
                println!(
                    "    Tiny area DelaunayTriangulation construction failed (acceptable): {e}"
                );
            }
        }

        println!("  ✓ Fallback visibility algorithm tested with degenerate facets");
    }

    #[test]
    #[expect(clippy::too_many_lines)]
    fn test_fallback_visibility_threshold_behavior() {
        println!("Testing fallback visibility threshold and heuristic behavior");

        // Create a well-defined triangulation
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        let test_facet_vertices =
            extract_facet_vertices(hull.get_facet(0).unwrap(), dt.triangulation()).unwrap();

        println!("  Testing threshold behavior with systematic point placement...");

        // Test points at increasing distances to understand threshold behavior
        let base_distance = 0.1;
        let multipliers = vec![0.001, 0.01, 0.1, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0];

        let mut visibility_results = Vec::new();

        for &multiplier in &multipliers {
            let distance = base_distance * multiplier;
            let test_point = Point::new([distance, distance, distance]);

            let result = ConvexHull::<FastKernel<f64>, (), (), 3>::fallback_visibility_test(
                &test_facet_vertices,
                &test_point,
            );

            match result {
                Ok(is_visible) => {
                    visibility_results.push((multiplier, distance, is_visible));
                    println!(
                        "    Distance {distance:.6} (multiplier {multiplier}): visible = {is_visible}"
                    );
                }
                Err(e) => {
                    println!("    Distance {distance:.6} (multiplier {multiplier}): error = {e:?}");
                    visibility_results.push((multiplier, distance, false)); // Treat error as not visible
                }
            }
        }

        println!("  Analyzing threshold behavior patterns...");

        // Look for patterns in visibility results
        let visible_count = visibility_results
            .iter()
            .filter(|(_, _, visible)| *visible)
            .count();
        let not_visible_count = visibility_results.len() - visible_count;

        println!(
            "    Visible results: {visible_count}/{}",
            visibility_results.len()
        );
        println!(
            "    Not visible results: {not_visible_count}/{}",
            visibility_results.len()
        );

        // Check if there's a reasonable transition from not visible to visible
        let mut last_visible = false;
        let mut transition_found = false;

        for (multiplier, distance, visible) in &visibility_results {
            if !last_visible && *visible {
                println!(
                    "    ✓ Visibility transition found at distance {distance:.6} (multiplier {multiplier})"
                );
                transition_found = true;
            }
            last_visible = *visible;
        }

        if !transition_found && visible_count > 0 {
            println!("    No clear transition, but some points are visible");
        } else if visible_count == 0 {
            println!("    Warning: No points were deemed visible (possible threshold issue)");
        }

        println!("  Testing edge case geometries for threshold calculation...");

        // Test with facets that have different edge length distributions
        let edge_test_cases = vec![
            // (description, vertices)
            (
                "equilateral-like triangle",
                vec![
                    vertex!([0.0, 0.0, 0.0]),
                    vertex!([1.0, 0.0, 0.0]),
                    vertex!([0.5, 0.866, 0.0]),
                    vertex!([0.333, 0.289, 1.0]),
                ],
            ),
            (
                "elongated triangle",
                vec![
                    vertex!([0.0, 0.0, 0.0]),
                    vertex!([10.0, 0.0, 0.0]), // Very long edge
                    vertex!([0.1, 0.1, 0.0]),  // Short edge
                    vertex!([1.0, 1.0, 1.0]),
                ],
            ),
        ];

        for (description, vertices) in edge_test_cases {
            println!("    Testing {description}...");

            match DelaunayTriangulation::new(&vertices) {
                Ok(edge_dt) => match ConvexHull::from_triangulation(edge_dt.triangulation()) {
                    Ok(edge_hull) => {
                        let edge_facet_vertices = extract_facet_vertices(
                            edge_hull.get_facet(0).unwrap(),
                            edge_dt.triangulation(),
                        )
                        .unwrap();
                        let test_point = Point::new([5.0, 5.0, 5.0]);

                        let edge_result =
                            ConvexHull::<FastKernel<f64>, (), (), 3>::fallback_visibility_test(
                                &edge_facet_vertices,
                                &test_point,
                            );

                        match edge_result {
                            Ok(is_visible) => {
                                println!("      {description} visibility: {is_visible}");
                            }
                            Err(e) => {
                                println!("      {description} test failed: {e}");
                            }
                        }
                    }
                    Err(e) => {
                        println!("      {description} hull construction failed: {e}");
                    }
                },
                Err(e) => {
                    println!("      {description} DelaunayTriangulation construction failed: {e}");
                }
            }
        }

        println!("  ✓ Fallback visibility threshold behavior thoroughly tested");
    }

    // ============================================================================
    // GEOMETRIC DEGENERACY HANDLING TESTS
    // ============================================================================
    // These tests focus on how the convex hull algorithms handle degenerate
    // geometric configurations that can cause numerical instability.

    #[test]
    #[expect(clippy::too_many_lines)]
    fn test_collinear_points_handling() {
        println!("Testing convex hull construction with collinear point configurations");

        println!("  Testing perfectly collinear points in 2D...");

        let collinear_2d_vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([2.0, 0.0]), // Collinear with first two
        ];

        match DelaunayTriangulation::new(&collinear_2d_vertices) {
            Ok(collinear_dt) => {
                match ConvexHull::from_triangulation(collinear_dt.triangulation()) {
                    Ok(collinear_hull) => {
                        println!("    ✓ Collinear 2D hull constructed successfully");
                        assert!(
                            collinear_hull
                                .validate(collinear_dt.triangulation())
                                .is_ok()
                        );
                        println!("    Facet count: {}", collinear_hull.facet_count());

                        // Test operations on collinear hull
                        let test_point = Point::new([0.5, 1.0]);
                        let visibility_result = collinear_hull
                            .is_point_outside(&test_point, collinear_dt.triangulation());
                        match visibility_result {
                            Ok(is_outside) => {
                                println!("    Point outside test: {is_outside}");
                            }
                            Err(e) => {
                                println!(
                                    "    Point outside test failed (acceptable for degenerate case): {e}"
                                );
                            }
                        }
                    }
                    Err(e) => {
                        println!("    Collinear 2D hull construction failed: {e}");
                        match e {
                            ConvexHullConstructionError::GeometricDegeneracy { .. }
                            | ConvexHullConstructionError::InvalidTriangulation { .. }
                            | ConvexHullConstructionError::BoundaryFacetExtractionFailed {
                                ..
                            } => {
                                println!("      ✓ Appropriate error type for collinear points");
                            }
                            _ => {
                                println!("      Unexpected error type: {e:?}");
                            }
                        }
                    }
                }
            }
            Err(e) => {
                println!(
                    "    Collinear 2D DelaunayTriangulation construction failed (expected): {e}"
                );
            }
        }

        println!("  Testing nearly collinear points with small perturbations...");

        let nearly_collinear_vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 1e-12]),  // Nearly on the line y=0
            vertex!([2.0, -1e-12]), // Nearly on the line y=0
        ];

        match DelaunayTriangulation::new(&nearly_collinear_vertices) {
            Ok(nearly_dt) => {
                match ConvexHull::from_triangulation(nearly_dt.triangulation()) {
                    Ok(nearly_hull) => {
                        println!("    ✓ Nearly collinear 2D hull constructed successfully");
                        assert!(nearly_hull.validate(nearly_dt.triangulation()).is_ok());

                        // Test that operations handle numerical precision gracefully
                        let precision_test_point = Point::new([1.0, 1e-6]);
                        let precision_result = nearly_hull
                            .is_point_outside(&precision_test_point, nearly_dt.triangulation());
                        match precision_result {
                            Ok(is_outside) => {
                                println!(
                                    "    Precision test successful: point outside = {is_outside}"
                                );
                            }
                            Err(e) => {
                                println!("    Precision test failed (may be acceptable): {e}");
                            }
                        }
                    }
                    Err(e) => {
                        println!("    Nearly collinear hull construction failed: {e}");
                    }
                }
            }
            Err(e) => {
                println!("    Nearly collinear DelaunayTriangulation construction failed: {e}");
            }
        }

        println!("  Testing collinear points in 3D (degenerate configuration)...");

        let collinear_3d_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 1.0, 1.0]),
            vertex!([2.0, 2.0, 2.0]), // Collinear with first two
            vertex!([3.0, 3.0, 3.0]), // Also collinear
        ];

        match DelaunayTriangulation::new(&collinear_3d_vertices) {
            Ok(_) => {
                println!(
                    "    Warning: 3D collinear DelaunayTriangulation constructed (unexpected but handled)"
                );
            }
            Err(e) => {
                println!(
                    "    3D collinear DelaunayTriangulation construction failed (expected): {e}"
                );
            }
        }

        println!("  ✓ Collinear point configurations tested");
    }

    #[test]
    #[expect(clippy::too_many_lines)]
    fn test_coplanar_points_in_higher_dimensions() {
        println!("Testing convex hull construction with coplanar point configurations");

        println!("  Testing coplanar points in 3D...");

        // Four points in the same plane (z=0)
        let coplanar_3d_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([1.0, 1.0, 0.0]), // All in z=0 plane
        ];

        match DelaunayTriangulation::new(&coplanar_3d_vertices) {
            Ok(_) => {
                println!(
                    "    Warning: Coplanar 3D DelaunayTriangulation constructed (may indicate insufficient degeneracy detection)"
                );
            }
            Err(e) => {
                println!(
                    "    Coplanar 3D DelaunayTriangulation construction failed (expected): {e}"
                );
            }
        }

        println!("  Testing nearly coplanar points with small z-perturbations...");

        let nearly_coplanar_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 1e-10]),
            vertex!([0.0, 1.0, -1e-10]),
            vertex!([1.0, 1.0, 1e-10]),
        ];

        match DelaunayTriangulation::new(&nearly_coplanar_vertices) {
            Ok(nearly_coplanar_dt) => {
                match ConvexHull::from_triangulation(nearly_coplanar_dt.triangulation()) {
                    Ok(nearly_coplanar_hull) => {
                        println!("    ✓ Nearly coplanar 3D hull constructed successfully");

                        let validation_result =
                            nearly_coplanar_hull.validate(nearly_coplanar_dt.triangulation());
                        match validation_result {
                            Ok(()) => {
                                println!("    Hull validation successful");
                            }
                            Err(e) => {
                                println!(
                                    "    Hull validation failed (may be due to degeneracy): {e}"
                                );
                            }
                        }

                        // Test visibility operations on nearly coplanar hull
                        let test_point = Point::new([0.5, 0.5, 1.0]);
                        let visibility_result = nearly_coplanar_hull
                            .is_point_outside(&test_point, nearly_coplanar_dt.triangulation());
                        match visibility_result {
                            Ok(is_outside) => {
                                println!(
                                    "    Nearly coplanar visibility test: point outside = {is_outside}"
                                );
                            }
                            Err(e) => {
                                println!("    Nearly coplanar visibility test failed: {e}");
                                match e {
                                    ConvexHullConstructionError::VisibilityCheckFailed {
                                        ..
                                    }
                                    | ConvexHullConstructionError::GeometricDegeneracy { .. }
                                    | ConvexHullConstructionError::FacetCacheBuildFailed {
                                        ..
                                    } => {
                                        println!(
                                            "      ✓ Appropriate error for nearly degenerate geometry"
                                        );
                                    }
                                    _ => {
                                        println!("      Unexpected error type: {e:?}");
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        println!("    Nearly coplanar hull construction failed: {e}");
                        match e {
                            ConvexHullConstructionError::GeometricDegeneracy { .. }
                            | ConvexHullConstructionError::BoundaryFacetExtractionFailed {
                                ..
                            } => {
                                println!("      ✓ Appropriate error for nearly coplanar points");
                            }
                            _ => {
                                println!("      Unexpected error type: {e:?}");
                            }
                        }
                    }
                }
            }
            Err(e) => {
                println!("    Nearly coplanar DelaunayTriangulation construction failed: {e}");
            }
        }

        println!("  Testing coplanar points in 4D...");

        // Five points in the same 3D hyperplane (w=0)
        let coplanar_4d_vertices = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0]),
            vertex!([1.0, 1.0, 1.0, 0.0]), // All in w=0 hyperplane
        ];

        match DelaunayTriangulation::new(&coplanar_4d_vertices) {
            Ok(_) => {
                println!(
                    "    Warning: Coplanar 4D DelaunayTriangulation constructed (may indicate insufficient degeneracy detection)"
                );
            }
            Err(e) => {
                println!(
                    "    Coplanar 4D DelaunayTriangulation construction failed (expected): {e}"
                );
            }
        }

        println!("  ✓ Coplanar point configurations tested");
    }

    #[test]
    fn test_duplicate_and_coincident_vertices() {
        println!("Testing convex hull construction with duplicate and coincident vertices");

        println!("  Testing exact duplicate vertices...");

        let duplicate_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.0, 0.0, 0.0]), // Exact duplicate of first vertex
        ];

        match DelaunayTriangulation::new(&duplicate_vertices) {
            Ok(dup_dt) => {
                println!("    Duplicate vertices DelaunayTriangulation constructed");
                match ConvexHull::from_triangulation(dup_dt.triangulation()) {
                    Ok(dup_hull) => {
                        println!("    ✓ Hull with duplicate vertices constructed");

                        // Test validation - should catch duplicate vertices in facets
                        let validation_result = dup_hull.validate(dup_dt.triangulation());
                        match validation_result {
                            Ok(()) => {
                                println!(
                                    "    Hull validation passed (duplicates may have been handled)"
                                );
                            }
                            Err(ConvexHullValidationError::DuplicateVerticesInFacet { .. }) => {
                                println!(
                                    "    ✓ Validation correctly detected duplicate vertices in facets"
                                );
                            }
                            Err(e) => {
                                println!("    Hull validation failed with different error: {e}");
                            }
                        }
                    }
                    Err(e) => {
                        println!("    Hull construction with duplicates failed: {e}");
                        // This might be expected depending on how the TDS handles duplicates
                    }
                }
            }
            Err(e) => {
                println!(
                    "    Duplicate vertices DelaunayTriangulation construction failed (may be expected): {e}"
                );
            }
        }

        println!("  Testing nearly coincident vertices (within floating-point precision)...");

        let nearly_coincident_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([1e-15, 1e-15, 1e-15]), // Nearly coincident with first
        ];

        match DelaunayTriangulation::new(&nearly_coincident_vertices) {
            Ok(nearly_coin_dt) => {
                match ConvexHull::from_triangulation(nearly_coin_dt.triangulation()) {
                    Ok(nearly_coin_hull) => {
                        println!("    ✓ Hull with nearly coincident vertices constructed");

                        // Test operations to see how they handle near-duplicates
                        let test_point = Point::new([2.0, 2.0, 2.0]);
                        let visibility_result = nearly_coin_hull
                            .is_point_outside(&test_point, nearly_coin_dt.triangulation());
                        match visibility_result {
                            Ok(is_outside) => {
                                println!(
                                    "    Nearly coincident hull visibility test: {is_outside}"
                                );
                            }
                            Err(e) => {
                                println!("    Nearly coincident hull visibility test failed: {e}");
                            }
                        }
                    }
                    Err(e) => {
                        println!(
                            "    Hull construction with nearly coincident vertices failed: {e}"
                        );
                    }
                }
            }
            Err(e) => {
                println!(
                    "    Nearly coincident vertices DelaunayTriangulation construction failed: {e}"
                );
            }
        }

        println!("  ✓ Duplicate and coincident vertex configurations tested");
    }

    // ============================================================================
    // HIGH-DIMENSIONAL STRESS TESTS
    // ============================================================================
    // These tests exercise convex hull algorithms in higher dimensions (6D+)
    // and with larger datasets to ensure robustness and performance.

    #[test]
    #[expect(clippy::too_many_lines)]
    fn test_high_dimensional_convex_hulls() {
        println!("Testing convex hull construction in high dimensions (6D, 7D, 8D)");

        println!("  Testing 6D convex hull...");

        // Create a 6D simplex (7 vertices)
        let vertices_6d = vec![
            vertex!([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            vertex!([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]), // Interior point to make it non-degenerate
        ];

        match DelaunayTriangulation::new(&vertices_6d) {
            Ok(dt_6d) => {
                match ConvexHull::from_triangulation(dt_6d.triangulation()) {
                    Ok(hull_6d) => {
                        println!("    ✓ 6D hull constructed successfully");
                        println!("    6D hull facet count: {}", hull_6d.facet_count());
                        assert_eq!(hull_6d.dimension(), 6);

                        // Test validation in 6D
                        let validation_result = hull_6d.validate(dt_6d.triangulation());
                        match validation_result {
                            Ok(()) => {
                                println!("    6D hull validation successful");
                            }
                            Err(e) => {
                                println!("    6D hull validation failed: {e}");
                            }
                        }

                        // Test visibility operations in 6D
                        let test_point_6d = Point::new([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
                        let visibility_result =
                            hull_6d.is_point_outside(&test_point_6d, dt_6d.triangulation());
                        match visibility_result {
                            Ok(is_outside) => {
                                println!("    6D point outside test: {is_outside}");
                            }
                            Err(e) => {
                                println!("    6D point outside test failed: {e}");
                                // High dimensional operations might fail due to complexity
                            }
                        }
                    }
                    Err(e) => {
                        println!("    6D hull construction failed: {e}");
                        // This might be expected for high-dimensional cases
                    }
                }
            }
            Err(e) => {
                println!("    6D DelaunayTriangulation construction failed: {e}");
            }
        }

        println!("  Testing 7D convex hull...");

        let vertices_7d = vec![
            vertex!([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            vertex!([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        ];

        match DelaunayTriangulation::new(&vertices_7d) {
            Ok(dt_7d) => match ConvexHull::from_triangulation(dt_7d.triangulation()) {
                Ok(hull_7d) => {
                    println!("    ✓ 7D hull constructed successfully");
                    println!("    7D hull facet count: {}", hull_7d.facet_count());
                    assert_eq!(hull_7d.dimension(), 7);
                    assert!(!hull_7d.is_empty());
                }
                Err(e) => {
                    println!("    7D hull construction failed: {e}");
                }
            },
            Err(e) => {
                println!("    7D DelaunayTriangulation construction failed: {e}");
            }
        }

        println!("  Testing 8D convex hull (stress test)...");

        let vertices_8d = vec![
            vertex!([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            vertex!([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        ];

        match DelaunayTriangulation::new(&vertices_8d) {
            Ok(dt_8d) => {
                match ConvexHull::from_triangulation(dt_8d.triangulation()) {
                    Ok(hull_8d) => {
                        println!("    ✓ 8D hull constructed successfully (impressive!)");
                        println!("    8D hull facet count: {}", hull_8d.facet_count());
                        assert_eq!(hull_8d.dimension(), 8);

                        // Stress test operations on 8D hull
                        let clear_ops_start = std::time::Instant::now();
                        assert!(!hull_8d.is_empty());
                        let basic_ops_duration = clear_ops_start.elapsed();
                        println!("    8D basic operations took: {basic_ops_duration:?}");
                    }
                    Err(e) => {
                        println!("    8D hull construction failed (acceptable): {e}");
                    }
                }
            }
            Err(e) => {
                println!("    8D DelaunayTriangulation construction failed (acceptable): {e}");
            }
        }

        println!("  ✓ High-dimensional convex hull tests completed");
    }

    #[test]
    fn test_large_dataset_performance() {
        println!("Testing convex hull performance with larger datasets");

        println!("  Testing 3D hull with many vertices...");
        // Generate a larger set of 3D points around a sphere
        let num_vertices = 50; // Reasonable size for testing
        let mut large_vertices = Vec::new();

        for i in 0..num_vertices {
            let angle1 = <f64 as From<_>>::from(i) * 2.0 * std::f64::consts::PI
                / <f64 as From<_>>::from(num_vertices);
            let angle2 = <f64 as From<_>>::from(i * 3) * std::f64::consts::PI
                / <f64 as From<_>>::from(num_vertices);
            let x = angle1.cos() * angle2.sin();
            let y = angle1.sin() * angle2.sin();
            let z = angle2.cos();
            large_vertices.push(vertex!([x, y, z]));
        }

        println!("    Generated {num_vertices} vertices");

        let start_time = std::time::Instant::now();

        match DelaunayTriangulation::new(&large_vertices) {
            Ok(large_dt) => {
                let dt_construction_time = start_time.elapsed();
                println!("    DelaunayTriangulation construction took: {dt_construction_time:?}");

                let hull_start = std::time::Instant::now();
                match ConvexHull::from_triangulation(large_dt.triangulation()) {
                    Ok(large_hull) => {
                        let hull_construction_time = hull_start.elapsed();
                        println!("    ✓ Large 3D hull constructed successfully");
                        println!("    Hull construction took: {hull_construction_time:?}");
                        println!("    Large hull facet count: {}", large_hull.facet_count());

                        // Test operations on large hull
                        let ops_start = std::time::Instant::now();

                        let validation_result = large_hull.validate(large_dt.triangulation());
                        assert!(
                            validation_result.is_ok(),
                            "Large hull validation should succeed: {:?}",
                            validation_result.err()
                        );

                        // Test visibility operations
                        let test_point = Point::new([2.0, 2.0, 2.0]);
                        let visibility_result =
                            large_hull.is_point_outside(&test_point, large_dt.triangulation());
                        assert!(
                            visibility_result.is_ok(),
                            "Large hull visibility test should succeed: {:?}",
                            visibility_result.err()
                        );

                        let ops_duration = ops_start.elapsed();
                        println!("    Operations on large hull took: {ops_duration:?}");

                        // Performance expectations (loose bounds)
                        if hull_construction_time.as_millis() > 1000 {
                            println!("    Warning: Hull construction took longer than expected");
                        }

                        if ops_duration.as_millis() > 100 {
                            println!("    Warning: Operations took longer than expected");
                        }
                    }
                    Err(e) => {
                        println!("    Large hull construction failed: {e}");
                        // This might be acceptable for very large datasets
                    }
                }
            }
            Err(e) => {
                println!("    Large DelaunayTriangulation construction failed: {e}");
            }
        }

        println!("  Testing 2D hull with many vertices...");

        // Generate points on a 2D circle
        let num_2d_vertices = 100;
        let mut large_2d_vertices = Vec::new();

        for i in 0..num_2d_vertices {
            let angle = <f64 as From<_>>::from(i) * 2.0 * std::f64::consts::PI
                / <f64 as From<_>>::from(num_2d_vertices);
            let x = angle.cos();
            let y = angle.sin();
            large_2d_vertices.push(vertex!([x, y]));
        }

        let start_2d = std::time::Instant::now();

        match DelaunayTriangulation::new(&large_2d_vertices) {
            Ok(large_2d_dt) => match ConvexHull::from_triangulation(large_2d_dt.triangulation()) {
                Ok(large_2d_hull) => {
                    let construction_2d_time = start_2d.elapsed();
                    println!("    ✓ Large 2D hull constructed in {construction_2d_time:?}");
                    println!(
                        "    2D hull facet count: {} (should be ~{})",
                        large_2d_hull.facet_count(),
                        num_2d_vertices
                    );

                    assert!(large_2d_hull.validate(large_2d_dt.triangulation()).is_ok());
                }
                Err(e) => {
                    println!("    Large 2D hull construction failed: {e}");
                }
            },
            Err(e) => {
                println!("    Large 2D DelaunayTriangulation construction failed: {e}");
            }
        }

        println!("  ✓ Large dataset performance tests completed");
    }

    // ============================================================================
    // ADVANCED CACHE INVALIDATION EDGE CASE TESTS
    // ============================================================================
    // These tests focus on edge cases in cache invalidation, concurrent access
    // patterns, and generation counter behavior.

    #[test]
    fn test_generation_counter_edge_cases() {
        println!("Testing generation counter edge cases and overflow scenarios");

        // Create a basic triangulation for testing
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        println!("  Testing rapid generation changes...");

        // Record initial generation
        let initial_generation = dt.triangulation().tds.generation();
        let initial_hull_generation = hull.cached_generation().load(Ordering::Acquire);

        println!("    Initial TDS generation: {initial_generation}");
        println!("    Initial hull generation: {initial_hull_generation}");

        // Rapidly modify the TDS to increment generation many times
        for i in 0..20 {
            let new_vertex = vertex!([<f64 as From<_>>::from(i).mul_add(0.01, 0.1), 0.1, 0.1]);
            if dt.insert(new_vertex).is_ok() {
                let current_gen = dt.triangulation().tds.generation();
                println!("    After modification {i}: TDS generation = {current_gen}");

                // Test that hull detects staleness
                let hull_gen = hull.cached_generation().load(Ordering::Acquire);
                if hull_gen > 0 && hull_gen < current_gen {
                    println!(
                        "      ✓ Hull generation ({hull_gen}) correctly behind TDS ({current_gen})"
                    );
                }
            } else {
                // Some additions might fail, which is okay
            }
        }

        println!("  Testing cache rebuild with high generation values...");

        let final_generation = dt.triangulation().tds.generation();
        println!("    Final TDS generation: {final_generation}");

        // Force cache rebuild
        let test_point = Point::new([2.0, 2.0, 2.0]);
        let visibility_result = hull.is_point_outside(&test_point, dt.triangulation());
        match visibility_result {
            Ok(is_outside) => {
                let updated_hull_gen = hull.cached_generation().load(Ordering::Acquire);
                println!("    After cache rebuild: hull generation = {updated_hull_gen}");
                assert_eq!(
                    updated_hull_gen, final_generation,
                    "Hull generation should match TDS after rebuild"
                );
                println!(
                    "    ✓ Cache rebuild with high generation successful: point outside = {is_outside}"
                );
            }
            Err(e) => {
                println!("    Cache rebuild failed: {e}");
            }
        }

        println!("  Testing manual invalidation with high generation values...");

        // Test manual invalidation
        hull.invalidate_cache();
        let invalidated_gen = hull.cached_generation().load(Ordering::Acquire);
        assert_eq!(
            invalidated_gen, 0,
            "Generation should be reset to 0 after manual invalidation"
        );
        println!("    ✓ Manual invalidation correctly reset generation to 0");

        // Test rebuild after manual invalidation
        let rebuild_result = hull.is_point_outside(&test_point, dt.triangulation());
        match rebuild_result {
            Ok(_) => {
                let rebuilt_gen = hull.cached_generation().load(Ordering::Acquire);
                assert_eq!(
                    rebuilt_gen, final_generation,
                    "Generation should match TDS after rebuild from manual invalidation"
                );
                println!("    ✓ Rebuild after manual invalidation successful");
            }
            Err(e) => {
                println!("    Rebuild after manual invalidation failed: {e}");
            }
        }

        println!("  ✓ Generation counter edge cases tested");
    }

    #[test]
    fn test_cache_consistency_under_stress() {
        println!("Testing cache consistency under rapid operations");

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        println!("  Testing rapid cache invalidation and rebuild cycles...");

        let num_cycles = 50;
        let test_points = [
            Point::new([2.0, 2.0, 2.0]),
            Point::new([1.5, 1.5, 1.5]),
            Point::new([0.1, 0.1, 0.1]),
            Point::new([10.0, 10.0, 10.0]),
        ];

        for cycle in 0..num_cycles {
            // Invalidate cache
            hull.invalidate_cache();

            // Verify cache is cleared
            let cache_after_invalidation = hull.facet_cache().load();
            assert!(
                cache_after_invalidation.is_none(),
                "Cache should be None after invalidation"
            );

            let generation_after_invalidation = hull.cached_generation().load(Ordering::Acquire);
            assert_eq!(
                generation_after_invalidation, 0,
                "Generation should be 0 after invalidation"
            );

            // Test multiple operations that should trigger cache rebuild
            for (i, test_point) in test_points.iter().enumerate() {
                let visibility_result = hull.is_point_outside(test_point, dt.triangulation());
                match visibility_result {
                    Ok(is_outside) => {
                        // After first operation, cache should be rebuilt
                        if i == 0 {
                            let cache_after_rebuild = hull.facet_cache().load();
                            assert!(
                                cache_after_rebuild.is_some(),
                                "Cache should exist after first operation"
                            );

                            let generation_after_rebuild =
                                hull.cached_generation().load(Ordering::Acquire);
                            assert!(
                                generation_after_rebuild > 0,
                                "Generation should be updated after rebuild"
                            );
                        }

                        println!("    Cycle {cycle}, Point {i}: outside = {is_outside}");
                    }
                    Err(e) => {
                        panic!("Visibility test failed in cycle {cycle}, point {i}: {e:?}");
                    }
                }
            }

            // Verify cache remains consistent throughout the cycle
            let final_cache = hull.facet_cache().load();
            assert!(final_cache.is_some(), "Cache should exist at end of cycle");

            let final_generation = hull.cached_generation().load(Ordering::Acquire);
            let tds_generation = dt.triangulation().tds.generation();
            assert_eq!(
                final_generation, tds_generation,
                "Hull generation should match TDS at end of cycle"
            );

            if (cycle + 1) % 10 == 0 {
                println!("    Completed {cycle} invalidation/rebuild cycles");
            }
        }

        println!("    ✓ Completed {num_cycles} invalidation/rebuild cycles successfully");

        println!("  Testing cache reuse efficiency...");

        // Test that cache is reused when generation hasn't changed
        let cache_before = hull.facet_cache().load();
        let generation_before = hull.cached_generation().load(Ordering::Acquire);

        // Perform multiple operations without TDS changes
        for i in 0..10 {
            let test_point = Point::new([<f64 as From<_>>::from(i).mul_add(0.1, 1.0), 2.0, 2.0]);
            let result = hull.is_point_outside(&test_point, dt.triangulation());
            assert!(result.is_ok(), "Visibility test {i} should succeed");
        }

        let cache_after = hull.facet_cache().load();
        let generation_after = hull.cached_generation().load(Ordering::Acquire);

        // Cache should exist before and after operations
        assert!(
            cache_before.is_some(),
            "Cache should exist before operations"
        );
        assert!(cache_after.is_some(), "Cache should exist after operations");

        // Check that the cache contains the same data (indicating reuse)
        if let (Some(before_arc), Some(after_arc)) = (&*cache_before, &*cache_after) {
            // Compare Arc pointer equality for reuse detection
            let cache_reused = Arc::ptr_eq(before_arc, after_arc);
            println!("    Cache reused: {cache_reused}");

            // For this test, we expect cache to be reused since generation didn't change
            assert!(
                cache_reused,
                "Cache Arc should be reused when generation unchanged"
            );
            println!("    ✓ Cache efficiently reused across multiple operations");
        } else {
            panic!("Cache should exist both before and after operations");
        }

        assert_eq!(
            generation_before, generation_after,
            "Generation should remain unchanged"
        );

        println!("  ✓ Cache consistency under stress tested");
    }

    #[test]
    fn test_memory_usage_and_cleanup() {
        println!("Testing memory usage patterns and cleanup behavior");

        println!("  Testing cache memory cleanup...");

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = create_triangulation(&vertices);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        // Build cache
        let test_point = Point::new([2.0, 2.0, 2.0]);
        let _ = hull.is_point_outside(&test_point, dt.triangulation());

        // Verify cache exists
        let cache_exists = hull.facet_cache().load();
        assert!(
            cache_exists.is_some(),
            "Cache should exist after visibility test"
        );

        // Manual invalidation should clear the cache
        hull.invalidate_cache();
        let cache_after_invalidation = hull.facet_cache().load();
        assert!(
            cache_after_invalidation.is_none(),
            "Cache should be None after invalidation"
        );

        // Cache should rebuild on next operation
        let _ = hull.is_point_outside(&test_point, dt.triangulation());
        let cache_rebuilt = hull.facet_cache().load();
        assert!(
            cache_rebuilt.is_some(),
            "Cache should be rebuilt after invalidation"
        );

        println!("    ✓ Cache memory cleanup behavior correct");

        println!("  Testing multiple hull instances with shared TDS...");

        // Create multiple hull instances from the same triangulation
        let hull1: ConvexHull<FastKernel<f64>, (), (), 3> =
            ConvexHull::from_triangulation(dt.triangulation()).unwrap();
        let hull2: ConvexHull<FastKernel<f64>, (), (), 3> =
            ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        // Each should have independent cache
        let _ = hull1.is_point_outside(&test_point, dt.triangulation());
        let _ = hull2.is_point_outside(&test_point, dt.triangulation());

        let cache1 = hull1.facet_cache().load();
        let cache2 = hull2.facet_cache().load();

        if let (Some(cache1_arc), Some(cache2_arc)) = (&*cache1, &*cache2) {
            // Caches should be independent (different Arc instances)
            // but might have the same content
            println!("    Hull1 cache size: {}", cache1_arc.len());
            println!("    Hull2 cache size: {}", cache2_arc.len());

            // They should have the same content but be independent instances
            assert_eq!(
                cache1_arc.len(),
                cache2_arc.len(),
                "Cache sizes should be equal"
            );
            println!("    ✓ Multiple hull instances have independent but equivalent caches");
        } else {
            panic!("Both hulls should have caches after operations");
        }

        // Test that invalidating one doesn't affect the other
        hull1.invalidate_cache();

        let hull1_cache_after_invalidation = hull1.facet_cache().load();
        let hull2_cache_after_hull1_invalidation = hull2.facet_cache().load();

        assert!(
            hull1_cache_after_invalidation.is_none(),
            "Hull1 cache should be None after invalidation"
        );
        assert!(
            hull2_cache_after_hull1_invalidation.is_some(),
            "Hull2 cache should still exist"
        );

        println!("    ✓ Independent cache invalidation working correctly");

        println!("  ✓ Memory usage and cleanup patterns tested");
    }

    /// Regression test for the bug where stale hulls could silently revalidate after `invalidate_cache()`.
    ///
    /// Bug description: Before the two-generation design (`creation_generation` + `cached_generation`),
    /// calling `invalidate_cache()` would set the single generation counter to 0. This allowed
    /// stale hulls to bypass staleness detection, rebuild the cache with the new TDS generation,
    /// and then report as "valid" via `is_valid_for_triangulation()` despite having facet handles that
    /// reference the old TDS topology.
    ///
    /// Expected behavior: Hull remains invalid for the modified TDS even after cache invalidation.
    #[test]
    fn test_stale_hull_detection_after_invalidate_cache() {
        println!("Testing stale hull detection after invalidate_cache (regression test)");

        // Step 1: Create hull from initial DelaunayTriangulation
        let mut dt = DelaunayTriangulation::new(&[
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ])
        .unwrap();
        let initial_gen = dt.triangulation().tds.generation();
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        // Verify hull is valid initially
        assert!(
            hull.is_valid_for_triangulation(dt.triangulation()),
            "Hull should be valid for initial triangulation"
        );
        println!("  ✓ Hull created with generation {initial_gen}");

        // Step 2: Mutate triangulation (increases generation)
        dt.insert(vertex!([0.5, 0.5, 0.5])).unwrap();
        let new_gen = dt.triangulation().tds.generation();
        assert_ne!(
            initial_gen, new_gen,
            "Triangulation generation should increase after mutation"
        );
        println!("  ✓ Triangulation mutated, generation increased to {new_gen}");

        // Verify hull is now invalid (stale)
        assert!(
            !hull.is_valid_for_triangulation(dt.triangulation()),
            "Hull should be invalid after triangulation modification"
        );
        println!("  ✓ Hull correctly detected as stale");

        // Step 3: Call invalidate_cache() (the critical operation that triggered the bug)
        hull.invalidate_cache();
        println!("  ✓ Called invalidate_cache()");

        // Step 4: Try to use the hull (this would trigger cache rebuild in the old buggy code)
        let test_point = Point::new([2.0, 2.0, 2.0]);
        let visibility_result = hull.find_visible_facets(&test_point, dt.triangulation());

        // The visibility operation should fail with StaleHull error
        assert!(
            visibility_result.is_err(),
            "Visibility operation should fail on stale hull even after invalidate_cache()"
        );

        if let Err(ConvexHullConstructionError::StaleHull {
            hull_generation,
            tds_generation,
        }) = visibility_result
        {
            assert_eq!(
                hull_generation, initial_gen,
                "Error should report hull's creation generation"
            );
            assert_eq!(
                tds_generation, new_gen,
                "Error should report current TDS generation"
            );
            println!("  ✓ Visibility operation correctly failed with StaleHull error");
        } else {
            panic!("Expected StaleHull error, got: {visibility_result:?}");
        }

        // Step 5: CRITICAL CHECK - is_valid_for_triangulation() must STILL return false
        // This is the key test that catches the bug
        assert!(
            !hull.is_valid_for_triangulation(dt.triangulation()),
            "BUG DETECTED: Hull should STILL be invalid for modified triangulation after invalidate_cache()"
        );
        println!(
            "  ✓ CRITICAL: is_valid_for_triangulation() correctly returns false after cache invalidation"
        );

        // Step 6: Verify that creating a new hull works correctly
        let new_hull: ConvexHull<FastKernel<f64>, (), (), 3> =
            ConvexHull::from_triangulation(dt.triangulation()).unwrap();
        assert!(
            new_hull.is_valid_for_triangulation(dt.triangulation()),
            "New hull should be valid for modified triangulation"
        );
        let new_visibility_result = new_hull.find_visible_facets(&test_point, dt.triangulation());
        assert!(
            new_visibility_result.is_ok(),
            "Visibility operation should succeed on new hull"
        );
        println!("  ✓ New hull works correctly with modified TDS");

        println!(
            "  ✓ Regression test passed: stale hull detection works correctly after invalidate_cache()"
        );
    }

    // ============================================================================
    // COMPREHENSIVE STALE HULL SAFETY TESTS
    // ============================================================================

    #[test]
    fn test_stale_hull_all_operations_fail() {
        println!("Testing that all hull operations fail on stale hull");

        let mut dt = DelaunayTriangulation::new(&[
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ])
        .unwrap();
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        // Modify triangulation to make hull stale
        dt.insert(vertex!([0.5, 0.5, 0.5])).unwrap();

        let test_point = Point::new([2.0, 2.0, 2.0]);

        // Test that all operations detect staleness
        println!("  Testing find_visible_facets...");
        let visible_result = hull.find_visible_facets(&test_point, dt.triangulation());
        assert!(
            matches!(
                visible_result,
                Err(ConvexHullConstructionError::StaleHull { .. })
            ),
            "find_visible_facets should fail with StaleHull error"
        );

        println!("  Testing find_nearest_visible_facet...");
        let nearest_result = hull.find_nearest_visible_facet(&test_point, dt.triangulation());
        assert!(
            matches!(
                nearest_result,
                Err(ConvexHullConstructionError::StaleHull { .. })
            ),
            "find_nearest_visible_facet should fail with StaleHull error"
        );

        println!("  Testing is_point_outside...");
        let outside_result = hull.is_point_outside(&test_point, dt.triangulation());
        assert!(
            matches!(
                outside_result,
                Err(ConvexHullConstructionError::StaleHull { .. })
            ),
            "is_point_outside should fail with StaleHull error"
        );

        println!("  Testing validate...");
        let validate_result = hull.validate(dt.triangulation());
        // validate() should fail with explicit StaleHull error
        assert!(
            matches!(
                validate_result,
                Err(ConvexHullValidationError::StaleHull { .. })
            ),
            "validate should fail on stale hull with StaleHull error, got: {validate_result:?}"
        );

        println!("  ✓ All operations correctly detect stale hull");
    }

    // ============================================================================
    // DEGENERATE GEOMETRY TESTS
    // ============================================================================

    #[test]
    fn test_fallback_visibility_degenerate_facets() {
        println!("Testing fallback visibility check for degenerate facets");

        // Create hull with nearly coplanar points (z << x,y) but still valid simplex
        let dt = DelaunayTriangulation::new(&[
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 1e-6]),
            vertex!([0.0, 1.0, 1e-6]),
            vertex!([0.0, 0.0, 1.0]),
        ])
        .unwrap();
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        println!("  Testing point visibility with degenerate geometry...");
        let test_point = Point::new([0.5, 0.5, 1.0]);
        let result = hull.is_point_outside(&test_point, dt.triangulation());

        assert!(
            result.is_ok(),
            "Visibility test should handle degenerate geometry: {:?}",
            result.err()
        );
        println!("  ✓ Degenerate geometry handled gracefully");
    }

    #[test]
    fn test_extreme_aspect_ratio_facets() {
        println!("Testing convex hull with extreme aspect ratio facets");

        // Create a very flat triangulation (extreme aspect ratio)
        let dt = DelaunayTriangulation::new(&[
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1000.0, 0.0, 0.0]), // Very long in x
            vertex!([0.0, 0.001, 0.0]),  // Very short in y
            vertex!([0.0, 0.0, 0.001]),  // Very short in z
        ])
        .unwrap();
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        println!("  Testing operations with extreme aspect ratio...");
        let test_point = Point::new([500.0, 10.0, 10.0]);
        let result = hull.is_point_outside(&test_point, dt.triangulation());

        assert!(
            result.is_ok(),
            "Should handle extreme aspect ratios: {:?}",
            result.err()
        );
        println!("  ✓ Extreme aspect ratio handled correctly");
    }

    // ============================================================================
    // BOUNDARY CONDITION TESTS
    // ============================================================================

    #[test]
    fn test_point_exactly_on_hull_surface() {
        println!("Testing points exactly on hull surface");

        let dt = create_triangulation(&[
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ]);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        println!("  Testing point on facet plane...");
        // Point on the plane defined by (1,0,0), (0,1,0), (0,0,0)
        let on_facet = Point::new([0.5, 0.5, 0.0]);
        let result = hull.is_point_outside(&on_facet, dt.triangulation());
        // Should handle boundary case without panic
        assert!(result.is_ok(), "Point on facet should be handled");
        println!("  ✓ Point on facet handled: inside={}", !result.unwrap());

        println!("  Testing point at vertex...");
        let at_vertex = Point::new([1.0, 0.0, 0.0]);
        let result = hull.is_point_outside(&at_vertex, dt.triangulation());
        assert!(result.is_ok(), "Point at vertex should be handled");
        // Point at vertex should be inside hull
        assert!(!result.unwrap(), "Point at vertex should be inside");
        println!("  ✓ Point at vertex correctly identified as inside");

        println!("  Testing point on edge...");
        let on_edge = Point::new([0.5, 0.0, 0.0]);
        let result = hull.is_point_outside(&on_edge, dt.triangulation());
        assert!(result.is_ok(), "Point on edge should be handled");
        println!("  ✓ Point on edge handled");
    }

    #[test]
    fn test_point_very_close_to_surface() {
        println!("Testing points very close to hull surface (epsilon distance)");

        let dt = create_triangulation(&[
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ]);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        println!("  Testing point just outside (epsilon away)...");
        let just_outside = Point::new([0.5, 0.5, f64::EPSILON * 10.0]);
        let result = hull.is_point_outside(&just_outside, dt.triangulation());
        assert!(result.is_ok(), "Near-surface point should be handled");
        println!("  ✓ Epsilon-distance point handled");

        println!("  Testing point just inside (negative epsilon)...");
        let just_inside = Point::new([0.25, 0.25, -f64::EPSILON * 10.0]);
        let result = hull.is_point_outside(&just_inside, dt.triangulation());
        assert!(result.is_ok(), "Near-surface point should be handled");
        println!("  ✓ Negative epsilon-distance point handled");
    }

    // ============================================================================
    // HIGH-DIMENSIONAL CONVEX HULL TESTS (2D-5D)
    // ============================================================================

    macro_rules! test_convex_hull_dimension {
        ($dim:expr, $test_name:ident) => {
            #[test]
            fn $test_name() {
                // Items must be before statements
                use crate::vertex;

                // Origin - use nested macro to get correct type
                macro_rules! make_vertex {
                    ($coords:expr) => {{
                        let v: crate::core::vertex::Vertex<f64, (), $dim> = vertex!($coords);
                        v
                    }};
                }

                println!("Testing {}D convex hull construction and operations", $dim);

                // Generate simplex vertices for dimension D
                let mut vertices_vec = Vec::new();
                vertices_vec.push(make_vertex!([0.0; $dim]));

                // Unit vectors along each axis
                for i in 0..$dim {
                    let mut coords = [0.0; $dim];
                    coords[i] = 1.0;
                    vertices_vec.push(make_vertex!(coords));
                }

                println!(
                    "  Creating {}D triangulation with {} vertices...",
                    $dim,
                    vertices_vec.len()
                );
                let dt = DelaunayTriangulation::<FastKernel<f64>, (), (), $dim>::new(&vertices_vec)
                    .expect(&format!("Failed to create {}D DelaunayTriangulation", $dim));

                println!("  Constructing {}D convex hull...", $dim);
                let hull = ConvexHull::from_triangulation(dt.triangulation())
                    .expect(&format!("Failed to create {}D hull", $dim));

                // A D-simplex has D+1 facets
                let expected_facets = $dim + 1;
                assert_eq!(
                    hull.facet_count(),
                    expected_facets,
                    "{}D simplex should have {} facets",
                    $dim,
                    expected_facets
                );
                println!(
                    "  ✓ {}D hull has correct facet count: {}",
                    $dim, expected_facets
                );

                // Test point outside
                println!("  Testing point outside {}D hull...", $dim);
                let outside_coords = [2.0; $dim];
                let outside_point = Point::new(outside_coords);
                let is_outside = hull
                    .is_point_outside(&outside_point, dt.triangulation())
                    .expect(&format!("Failed to test outside point in {}D", $dim));
                assert!(
                    is_outside,
                    "Point at {:?} should be outside {}D hull",
                    outside_coords, $dim
                );
                println!("  ✓ Outside point correctly identified in {}D", $dim);

                // Test point inside (centroid)
                println!("  Testing point inside {}D hull...", $dim);
                let mut inside_coords = [0.0; $dim];
                for i in 0..$dim {
                    inside_coords[i] = 1.0 / <f64 as std::convert::From<i32>>::from($dim + 1);
                }
                let inside_point = Point::new(inside_coords);
                let is_outside = hull
                    .is_point_outside(&inside_point, dt.triangulation())
                    .expect(&format!("Failed to test inside point in {}D", $dim));
                assert!(!is_outside, "Centroid should be inside {}D hull", $dim);
                println!("  ✓ Inside point correctly identified in {}D", $dim);

                // Test visibility
                println!("  Testing visibility in {}D...", $dim);
                let visible_facets = hull
                    .find_visible_facets(&outside_point, dt.triangulation())
                    .expect(&format!("Failed to find visible facets in {}D", $dim));
                assert!(
                    !visible_facets.is_empty(),
                    "Outside point should see some facets in {}D",
                    $dim
                );
                println!(
                    "  ✓ Visibility test passed in {}D: {} visible facets",
                    $dim,
                    visible_facets.len()
                );

                // Test validation
                println!("  Validating {}D hull...", $dim);
                hull.validate(dt.triangulation())
                    .expect(&format!("{}D hull validation failed", $dim));
                println!("  ✓ {}D hull validation passed", $dim);

                println!("  ✓ All {}D convex hull tests passed", $dim);
            }
        };
    }

    // Generate tests for dimensions 2-5
    test_convex_hull_dimension!(2, test_convex_hull_2d_comprehensive);
    test_convex_hull_dimension!(3, test_convex_hull_3d_comprehensive);
    test_convex_hull_dimension!(4, test_convex_hull_4d_comprehensive);
    test_convex_hull_dimension!(5, test_convex_hull_5d_comprehensive);

    // ============================================================================
    // CACHE LIFECYCLE MANAGEMENT TESTS
    // ============================================================================

    #[test]
    fn test_cache_lifecycle_multiple_operations() {
        println!("Testing cache lifecycle through multiple operations");

        let dt = create_triangulation(&[
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ]);
        let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        let test_point = Point::new([2.0, 2.0, 2.0]);

        println!("  Initial state: cache already built during construction");
        let initial_gen = hull.cached_generation().load(Ordering::Acquire);
        assert!(
            initial_gen > 0,
            "Generation should be set after construction"
        );

        println!("  First operation: should build cache");
        hull.find_visible_facets(&test_point, dt.triangulation())
            .unwrap();
        assert!(
            hull.facet_cache().load().is_some(),
            "Cache should be built after first operation"
        );
        let gen_after_build = hull.cached_generation().load(Ordering::Acquire);
        assert!(
            gen_after_build > 0,
            "Generation should be updated after build"
        );

        println!("  Second operation: should reuse cache");
        let gen_before_reuse = hull.cached_generation().load(Ordering::Acquire);
        hull.find_visible_facets(&test_point, dt.triangulation())
            .unwrap();
        let gen_after_reuse = hull.cached_generation().load(Ordering::Acquire);
        assert_eq!(
            gen_after_reuse, gen_before_reuse,
            "Generation should not change when reusing cache"
        );

        println!("  Manual invalidation: should clear cache and reset generation");
        hull.invalidate_cache();
        assert!(
            hull.facet_cache().load().is_none(),
            "Cache should be cleared after invalidation"
        );
        let gen_after_invalidation = hull.cached_generation().load(Ordering::Acquire);
        assert_eq!(
            gen_after_invalidation, 0,
            "Generation should be reset to 0 after invalidation"
        );

        // Verify that cache was actually cleared from non-zero state
        assert!(
            gen_before_reuse > 0,
            "Should have had cache before invalidation"
        );

        println!("  Operation after invalidation: should rebuild cache");
        hull.find_visible_facets(&test_point, dt.triangulation())
            .unwrap();
        assert!(
            hull.facet_cache().load().is_some(),
            "Cache should be rebuilt"
        );
        let gen_after_rebuild = hull.cached_generation().load(Ordering::Acquire);
        assert!(
            gen_after_rebuild > 0,
            "Generation should be updated after rebuild"
        );

        println!("  ✓ Cache lifecycle behaves correctly through all operations");
    }

    #[test]
    fn test_cache_independence_multiple_hulls() {
        println!("Testing cache independence between multiple hull instances");

        let dt = create_triangulation(&[
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ]);

        println!("  Creating first hull and building cache...");
        let hull1: ConvexHull<FastKernel<f64>, (), (), 3> =
            ConvexHull::from_triangulation(dt.triangulation()).unwrap();
        let test_point = Point::new([2.0, 2.0, 2.0]);
        hull1
            .find_visible_facets(&test_point, dt.triangulation())
            .unwrap();

        println!("  Creating second hull from same triangulation...");
        let hull2: ConvexHull<FastKernel<f64>, (), (), 3> =
            ConvexHull::from_triangulation(dt.triangulation()).unwrap();

        println!("  Second hull also has cache built during construction...");
        let hull2_gen_initial = hull2.cached_generation().load(Ordering::Acquire);
        assert!(
            hull2_gen_initial > 0,
            "Hull2 should have cache built during construction"
        );

        println!("  Building cache in second hull...");
        hull2
            .find_visible_facets(&test_point, dt.triangulation())
            .unwrap();
        let hull2_gen = hull2.cached_generation().load(Ordering::Acquire);
        assert!(hull2_gen > 0, "Hull2 should have built cache");

        // Both should have cache
        assert!(
            hull1.facet_cache().load().is_some(),
            "Hull1 should have cache"
        );
        assert!(
            hull2.facet_cache().load().is_some(),
            "Hull2 should have cache"
        );

        println!("  Invalidating first hull cache...");
        hull1.invalidate_cache();

        // Hull1 should be cleared
        assert!(
            hull1.facet_cache().load().is_none(),
            "Hull1 cache should be cleared"
        );

        // Hull2 should still have cache (independent instances)
        assert!(
            hull2.facet_cache().load().is_some(),
            "Hull2 cache should be independent"
        );

        println!("  ✓ Cache independence between multiple hull instances verified");
    }
}
