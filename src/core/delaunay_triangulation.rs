//! Delaunay triangulation layer with incremental insertion.
//!
//! This layer adds Delaunay-specific operations on top of the generic
//! `Triangulation` struct, following CGAL's architecture.

use core::iter::Sum;
use core::ops::{AddAssign, SubAssign};
use std::num::NonZeroUsize;

use num_traits::NumCast;

use crate::core::adjacency::{AdjacencyIndex, AdjacencyIndexBuildError};
use crate::core::algorithms::incremental_insertion::{
    InsertionError, InsertionOutcome, InsertionStatistics,
};
use crate::core::cell::Cell;
use crate::core::edge::EdgeKey;
use crate::core::facet::{AllFacetsIter, BoundaryFacetsIter};
use crate::core::traits::data_type::DataType;
use crate::core::triangulation::{
    Triangulation, TriangulationConstructionError, TriangulationValidationError, ValidationPolicy,
};
use crate::core::triangulation_data_structure::{
    CellKey, InvariantKind, InvariantViolation, Tds, TdsConstructionError, TdsValidationError,
    TriangulationValidationReport, VertexKey,
};
use crate::core::util::DelaunayValidationError;
use crate::core::vertex::Vertex;
use crate::geometry::kernel::{FastKernel, Kernel};
use crate::geometry::traits::coordinate::{CoordinateConversionError, CoordinateScalar};

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;
use uuid::Uuid;

/// Errors that can occur during Delaunay triangulation construction.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum DelaunayTriangulationConstructionError {
    /// Lower-layer construction error (Triangulation / TDS).
    #[error(transparent)]
    Triangulation(#[from] TriangulationConstructionError),
}

/// Errors that can occur during Delaunay triangulation validation (Level 4).
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum DelaunayTriangulationValidationError {
    /// Lower-layer validation error (Levels 1–3).
    #[error(transparent)]
    Triangulation(#[from] TriangulationValidationError),

    /// A cell violates the empty circumsphere property.
    #[error(
        "Delaunay property violated: Cell {cell_uuid} (key: {cell_key:?}) violates empty circumsphere invariant"
    )]
    DelaunayViolation {
        /// Key of the violating cell.
        cell_key: CellKey,
        /// UUID of the violating cell (or nil if the UUID mapping is unavailable).
        cell_uuid: Uuid,
    },

    /// Numeric predicate failure during Delaunay validation.
    #[error(
        "Numeric predicate failure while validating Delaunay property for cell {cell_uuid} (key: {cell_key:?}), vertex {vertex_key:?}: {source}"
    )]
    NumericPredicateError {
        /// The key of the cell whose circumsphere was being tested.
        cell_key: CellKey,
        /// UUID of the cell whose predicate evaluation failed (or nil if unavailable).
        cell_uuid: Uuid,
        /// The key of the vertex being classified relative to the circumsphere.
        vertex_key: VertexKey,
        /// Underlying robust predicate error (e.g., conversion failure).
        #[source]
        source: CoordinateConversionError,
    },
}

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
/// - Run [`is_valid`](Self::is_valid) (Level 4) in tests or debug builds
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
    pub fn new(
        vertices: &[Vertex<f64, (), D>],
    ) -> Result<Self, DelaunayTriangulationConstructionError> {
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
    ) -> Result<Self, DelaunayTriangulationConstructionError>
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
            }
            .into());
        }

        // Build initial simplex directly (no Bowyer-Watson)
        let initial_vertices = &vertices[..=D];
        let tds = Triangulation::<K, U, V, D>::build_initial_simplex(initial_vertices)?;

        let mut dt = Self {
            tri: Triangulation {
                kernel,
                tds,
                validation_policy: ValidationPolicy::default(),
            },
            last_inserted_cell: None,
        };

        // Insert remaining vertices incrementally.
        // Retryable geometric degeneracies are retried with perturbation and ultimately skipped
        // (transactional rollback) to keep the triangulation manifold. Duplicate/near-duplicate
        // coordinates are skipped immediately.
        for vertex in vertices.iter().skip(D + 1) {
            match dt
                .tri
                .insert_with_statistics(*vertex, None, dt.last_inserted_cell)
            {
                Ok((
                    InsertionOutcome::Inserted {
                        vertex_key: _v_key,
                        hint,
                    },
                    _stats,
                )) => {
                    // Cache hint for faster subsequent insertions.
                    dt.last_inserted_cell = hint;
                }
                Ok((InsertionOutcome::Skipped { error }, stats)) => {
                    // Keep going: this vertex was intentionally skipped (e.g. duplicate/near-duplicate
                    // coordinates, or an unsalvageable geometric degeneracy after retries).
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "SKIPPED: vertex insertion after {} attempts during construction: {error}",
                        stats.attempts
                    );
                    #[cfg(not(debug_assertions))]
                    {
                        let _ = (error, stats);
                    }
                }
                Err(e) => {
                    // Non-retryable failure: abort construction with a structured error.
                    return Err(match e {
                        // Preserve underlying construction errors (e.g. duplicate UUID).
                        InsertionError::Construction(source) => source,
                        InsertionError::CavityFilling { message } => {
                            TriangulationConstructionError::FailedToCreateCell { message }
                        }
                        InsertionError::NeighborWiring { message } => {
                            TriangulationConstructionError::from(
                                TdsConstructionError::ValidationError(
                                    TdsValidationError::InvalidNeighbors { message },
                                ),
                            )
                        }
                        InsertionError::TopologyValidation(source) => {
                            TriangulationConstructionError::from(
                                TdsConstructionError::ValidationError(source),
                            )
                        }
                        InsertionError::DuplicateUuid { entity, uuid } => {
                            TriangulationConstructionError::from(
                                TdsConstructionError::DuplicateUuid { entity, uuid },
                            )
                        }
                        InsertionError::DuplicateCoordinates { coordinates } => {
                            TriangulationConstructionError::DuplicateCoordinates { coordinates }
                        }

                        // Insertion-layer failures that are best surfaced during construction as a
                        // geometric degeneracy (e.g. numerical instability, hull visibility issues).
                        //
                        // NOTE: This match is intentionally exhaustive over `InsertionError`.
                        // When adding new insertion failure modes in the future, revisit whether they
                        // deserve a dedicated `TriangulationConstructionError` variant instead of being
                        // collapsed into `GeometricDegeneracy`.
                        //
                        // We intentionally preserve the high-level insertion failure *bucket* in the
                        // degeneracy message by capturing `e.to_string()` (rather than only
                        // `source.to_string()`), so callers/telemetry can distinguish e.g.
                        // "Conflict region error" vs "Location error" vs "Hull extension failed".
                        insertion_error @ (InsertionError::ConflictRegion(_)
                        | InsertionError::Location(_)
                        | InsertionError::NonManifoldTopology { .. }
                        | InsertionError::HullExtension { .. }
                        | InsertionError::TopologyValidationFailed { .. }) => {
                            TriangulationConstructionError::GeometricDegeneracy {
                                message: insertion_error.to_string(),
                            }
                        }
                    }
                    .into());
                }
            }
        }

        Ok(dt)
    }

    // TODO: Implement after bistellar flips + robust insertion (v0.7.0+)
    // /// Create a Delaunay triangulation with a specified topological space.
    // ///
    // /// This will allow constructing Delaunay triangulations on different topologies
    // /// (Euclidean, spherical, toroidal) with appropriate boundary conditions
    // /// and topology validation. This method should delegate to
    // /// [`Triangulation::with_topology`] after constructing the TDS.
    // ///
    // /// Requires:
    // /// - Bistellar flips for topology-preserving operations
    // /// - Robust Delaunay insertion that respects topology constraints
    // ///
    // /// # Examples (future)
    // ///
    // /// ```rust,ignore
    // /// use delaunay::prelude::*;
    // /// use delaunay::topology::spaces::ToroidalSpace;
    // ///
    // /// let space = ToroidalSpace::new([1.0, 1.0, 1.0]);
    // /// let dt = DelaunayTriangulation::with_topology(
    // ///     FastKernel::new(),
    // ///     space,
    // ///     &vertices
    // /// ).unwrap();
    // /// ```
    // pub fn with_topology<T>(
    //     kernel: K,
    //     topology: T,
    //     vertices: &[Vertex<K::Scalar, U, D>],
    // ) -> Result<Self, TriangulationConstructionError>
    // where
    //     K::Scalar: CoordinateScalar,
    //     T: TopologicalSpace,
    // {
    //     // Build TDS with Delaunay property, then delegate to Triangulation layer
    //     unimplemented!("Requires bistellar flips + robust insertion")
    // }

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
    /// 2. Run `dt.triangulation().validate()` (Levels 1–3) or `dt.validate()` (Levels 1–4) to verify structural/topological consistency
    /// 3. Reserve `dt.is_valid()` for Delaunay-only (Level 4) checks
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
    /// assert!(dt.validate().is_ok());
    /// ```
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // mutable refs from const fn not widely supported
    pub fn triangulation_mut(&mut self) -> &mut Triangulation<K, U, V, D> {
        &mut self.tri
    }

    /// Returns the insertion-time global topology validation policy used by the underlying
    /// triangulation.
    ///
    /// This policy controls when Level 3 (`Triangulation::is_valid()`) is run automatically
    /// during incremental insertion (as part of the topology safety net).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 2> =
    ///     DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// assert_eq!(
    ///     dt.validation_policy(),
    ///     delaunay::core::triangulation::ValidationPolicy::OnSuspicion
    /// );
    /// ```
    #[inline]
    #[must_use]
    pub const fn validation_policy(&self) -> ValidationPolicy {
        self.tri.validation_policy
    }

    /// Sets the insertion-time global topology validation policy used by the underlying
    /// triangulation.
    ///
    /// This affects subsequent incremental insertions. (Construction-time behavior is determined
    /// by the policy active during `new()` / `with_kernel()`.)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    ///
    /// let mut dt: DelaunayTriangulation<_, (), (), 2> =
    ///     DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// dt.set_validation_policy(delaunay::core::triangulation::ValidationPolicy::Always);
    /// assert_eq!(
    ///     dt.validation_policy(),
    ///     delaunay::core::triangulation::ValidationPolicy::Always
    /// );
    /// ```
    #[inline]
    pub const fn set_validation_policy(&mut self, policy: ValidationPolicy) {
        self.tri.validation_policy = policy;
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

    /// Builds an immutable adjacency index for fast repeated topology queries.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::build_adjacency_index`](crate::core::triangulation::Triangulation::build_adjacency_index).
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying triangulation data structure is internally inconsistent
    /// (e.g., a cell references a missing vertex key or a missing neighbor cell key).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// // A single 3D tetrahedron has 6 unique edges.
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let index = dt.build_adjacency_index().unwrap();
    ///
    /// assert_eq!(index.number_of_edges(), 6);
    /// ```
    #[inline]
    pub fn build_adjacency_index(&self) -> Result<AdjacencyIndex, AdjacencyIndexBuildError> {
        self.triangulation().build_adjacency_index()
    }

    /// Returns an iterator over all unique edges in the triangulation.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::edges`](crate::core::triangulation::Triangulation::edges).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// // A single 3D tetrahedron has 6 unique edges.
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let edges: std::collections::HashSet<_> = dt.edges().collect();
    /// assert_eq!(edges.len(), 6);
    /// ```
    pub fn edges(&self) -> impl Iterator<Item = EdgeKey> + '_ {
        self.triangulation().edges()
    }

    /// Returns an iterator over all unique edges using a precomputed [`AdjacencyIndex`].
    ///
    /// This avoids per-call deduplication and allocations.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::edges_with_index`](crate::core::triangulation::Triangulation::edges_with_index).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let index = dt.triangulation().build_adjacency_index().unwrap();
    ///
    /// let edges: std::collections::HashSet<_> = dt.edges_with_index(&index).collect();
    /// assert_eq!(edges.len(), 6);
    /// ```
    pub fn edges_with_index<'a>(
        &self,
        index: &'a AdjacencyIndex,
    ) -> impl Iterator<Item = EdgeKey> + 'a {
        self.triangulation().edges_with_index(index)
    }

    /// Returns an iterator over all unique edges incident to a vertex.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::incident_edges`](crate::core::triangulation::Triangulation::incident_edges).
    ///
    /// If `v` is not present in this triangulation, the iterator is empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let v0 = dt.vertices().next().unwrap().0;
    ///
    /// // In a tetrahedron, each vertex has degree 3.
    /// assert_eq!(dt.incident_edges(v0).count(), 3);
    /// ```
    pub fn incident_edges(&self, v: VertexKey) -> impl Iterator<Item = EdgeKey> + '_ {
        self.triangulation().incident_edges(v)
    }

    /// Returns an iterator over all unique edges incident to a vertex using a precomputed
    /// [`AdjacencyIndex`].
    ///
    /// If `v` is not present in the index, the iterator is empty.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::incident_edges_with_index`](crate::core::triangulation::Triangulation::incident_edges_with_index).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let index = dt.triangulation().build_adjacency_index().unwrap();
    /// let v0 = dt.vertices().next().unwrap().0;
    ///
    /// assert_eq!(dt.incident_edges_with_index(&index, v0).count(), 3);
    /// ```
    pub fn incident_edges_with_index<'a>(
        &self,
        index: &'a AdjacencyIndex,
        v: VertexKey,
    ) -> impl Iterator<Item = EdgeKey> + 'a {
        self.triangulation().incident_edges_with_index(index, v)
    }

    /// Returns an iterator over all neighbors of a cell.
    ///
    /// Boundary facets are omitted (only existing neighbors are yielded). If `c` is not
    /// present, the iterator is empty.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::cell_neighbors`](crate::core::triangulation::Triangulation::cell_neighbors).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// // A single tetrahedron has no cell neighbors.
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let cell_key = dt.cells().next().unwrap().0;
    /// assert_eq!(dt.cell_neighbors(cell_key).count(), 0);
    /// ```
    pub fn cell_neighbors(&self, c: CellKey) -> impl Iterator<Item = CellKey> + '_ {
        self.triangulation().cell_neighbors(c)
    }

    /// Returns an iterator over all neighbors of a cell using a precomputed [`AdjacencyIndex`].
    ///
    /// If `c` is not present in the index, the iterator is empty.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::cell_neighbors_with_index`](crate::core::triangulation::Triangulation::cell_neighbors_with_index).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// // Two tetrahedra sharing a triangular facet => each tetra has exactly one neighbor.
    /// let vertices: Vec<_> = vec![
    ///     // Shared triangle
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([2.0, 0.0, 0.0]),
    ///     vertex!([1.0, 2.0, 0.0]),
    ///     // Two apices
    ///     vertex!([1.0, 0.7, 1.5]),
    ///     vertex!([1.0, 0.7, -1.5]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let index = dt.triangulation().build_adjacency_index().unwrap();
    ///
    /// let cell_key = dt.cells().next().unwrap().0;
    /// assert_eq!(dt.cell_neighbors_with_index(&index, cell_key).count(), 1);
    /// ```
    pub fn cell_neighbors_with_index<'a>(
        &self,
        index: &'a AdjacencyIndex,
        c: CellKey,
    ) -> impl Iterator<Item = CellKey> + 'a {
        self.triangulation().cell_neighbors_with_index(index, c)
    }

    /// Returns a slice view of a cell's vertex keys.
    ///
    /// This is a zero-allocation accessor. If `c` is not present, returns `None`.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::cell_vertices`](crate::core::triangulation::Triangulation::cell_vertices).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// let cell_key = dt.cells().next().unwrap().0;
    /// let cell_vertices = dt.cell_vertices(cell_key).unwrap();
    /// assert_eq!(cell_vertices.len(), 3); // D+1 for a 2D simplex
    /// ```
    #[must_use]
    pub fn cell_vertices(&self, c: CellKey) -> Option<&[VertexKey]>
    where
        K::Scalar: CoordinateScalar,
    {
        self.triangulation().cell_vertices(c)
    }

    /// Returns a slice view of a vertex's coordinates.
    ///
    /// This is a zero-allocation accessor. If `v` is not present, returns `None`.
    ///
    /// This is a convenience wrapper around
    /// [`Triangulation::vertex_coords`](crate::core::triangulation::Triangulation::vertex_coords).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// // Find the key for a known vertex by matching coordinates.
    /// let v_key = dt
    ///     .vertices()
    ///     .find_map(|(vk, _)| (dt.vertex_coords(vk)? == [1.0, 0.0]).then_some(vk))
    ///     .unwrap();
    ///
    /// assert_eq!(dt.vertex_coords(v_key).unwrap(), [1.0, 0.0]);
    /// ```
    #[must_use]
    pub fn vertex_coords(&self, v: VertexKey) -> Option<&[K::Scalar]>
    where
        K::Scalar: CoordinateScalar,
    {
        self.triangulation().vertex_coords(v)
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

    /// Insert a vertex and return the insertion outcome plus statistics.
    ///
    /// This is a convenience wrapper around [`Triangulation::insert_with_statistics`] that also
    /// updates the internal `last_inserted_cell` hint cache.
    ///
    /// # Errors
    ///
    /// Returns `Err(InsertionError)` only for non-retryable structural failures.
    /// Retryable geometric degeneracies that exhaust all attempts return
    /// `Ok((InsertionOutcome::Skipped { .. }, stats))`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();
    ///
    /// let (outcome, stats) = dt
    ///     .insert_with_statistics(vertex!([0.0, 0.0, 0.0]))
    ///     .unwrap();
    ///
    /// assert!(stats.success());
    /// assert!(matches!(outcome, InsertionOutcome::Inserted { .. }));
    /// ```
    pub fn insert_with_statistics(
        &mut self,
        vertex: Vertex<K::Scalar, U, D>,
    ) -> Result<(InsertionOutcome, InsertionStatistics), InsertionError>
    where
        K::Scalar: CoordinateScalar,
    {
        let (outcome, stats) =
            self.tri
                .insert_with_statistics(vertex, None, self.last_inserted_cell)?;

        if let InsertionOutcome::Inserted { hint, .. } = &outcome {
            self.last_inserted_cell = *hint;
        }

        Ok((outcome, stats))
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
    /// // Vertex removal preserves Levels 1–3 but may not preserve the Delaunay property.
    /// assert!(dt.triangulation().validate().is_ok());
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
        self.tri.remove_vertex(vertex).map_err(Into::into)
    }

    /// Validates the Delaunay empty-circumsphere property (Level 4).
    ///
    /// This is the Delaunay layer's `is_valid`: it checks **only** the Delaunay property
    /// and intentionally does **not** run lower-layer validation.
    ///
    /// For cumulative validation across the whole hierarchy, use [`validate`](Self::validate).
    ///
    /// # Errors
    ///
    /// Returns a [`DelaunayTriangulationValidationError`] if the empty-circumsphere test fails, or if
    /// the underlying triangulation state is inconsistent and prevents geometric predicates
    /// from being evaluated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices_4d = [
    ///     vertex!([0.0, 0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulation::new(&vertices_4d).unwrap();
    ///
    /// // Level 4: Delaunay property only
    /// assert!(dt.is_valid().is_ok());
    /// ```
    pub fn is_valid(&self) -> Result<(), DelaunayTriangulationValidationError>
    where
        K::Scalar: CoordinateScalar,
    {
        let cell_uuid_or_nil = |key: CellKey| -> Uuid {
            self.tri
                .tds
                .cell_uuid_from_key(key)
                .unwrap_or_else(Uuid::nil)
        };

        crate::core::util::is_delaunay_property_only(&self.tri.tds).map_err(|err| match err {
            DelaunayValidationError::DelaunayViolation { cell_key } => {
                DelaunayTriangulationValidationError::DelaunayViolation {
                    cell_key,
                    cell_uuid: cell_uuid_or_nil(cell_key),
                }
            }
            DelaunayValidationError::TriangulationState { source } => {
                TriangulationValidationError::from(source).into()
            }
            DelaunayValidationError::InvalidCell { cell_key, source } => {
                // Attach the best-available cell UUID (nil only if mapping is unavailable).
                TriangulationValidationError::from(TdsValidationError::InvalidCell {
                    cell_id: cell_uuid_or_nil(cell_key),
                    source,
                })
                .into()
            }
            DelaunayValidationError::NumericPredicateError {
                cell_key,
                vertex_key,
                source,
            } => {
                // Include cell UUID for better debugging and log correlation
                DelaunayTriangulationValidationError::NumericPredicateError {
                    cell_key,
                    cell_uuid: cell_uuid_or_nil(cell_key),
                    vertex_key,
                    source,
                }
            }
        })
    }

    /// Performs cumulative validation for Levels 1–4.
    ///
    /// This validates:
    /// - **Levels 1–3** via [`Triangulation::validate`](crate::core::triangulation::Triangulation::validate)
    /// - **Level 4** via [`DelaunayTriangulation::is_valid`](Self::is_valid)
    ///
    /// # Errors
    ///
    /// Returns a [`DelaunayTriangulationValidationError`] if Levels 1–3 validation fails or if the
    /// Delaunay property check (Level 4) fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices_4d = [
    ///     vertex!([0.0, 0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulation::new(&vertices_4d).unwrap();
    ///
    /// // Levels 1–4: elements + structure + topology + Delaunay property
    /// assert!(dt.validate().is_ok());
    /// ```
    pub fn validate(&self) -> Result<(), DelaunayTriangulationValidationError>
    where
        K::Scalar: CoordinateScalar,
    {
        self.tri.validate()?;
        self.is_valid()
    }

    /// Create a `DelaunayTriangulation` from a deserialized `Tds` with a default kernel.
    ///
    /// This is useful when you've serialized just the `Tds` and want to reconstruct
    /// the `DelaunayTriangulation` with default kernel settings.
    ///
    /// # Notes
    ///
    /// - The internal `last_inserted_cell` "locate hint" is intentionally **not** persisted
    ///   across serialization boundaries. Constructing via `from_tds` (including the serde
    ///   `Deserialize` impl below) always resets it to `None`. This can make the first few
    ///   insertions after loading slightly slower, but is otherwise behaviorally irrelevant.
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
            tri: Triangulation {
                kernel,
                tds,
                validation_policy: ValidationPolicy::OnSuspicion,
            },
            last_inserted_cell: None,
        }
    }

    /// Generate a comprehensive validation report for the full validation hierarchy.
    ///
    /// This is intended for debugging/telemetry (e.g. `insert_with_statistics`) where
    /// you want to see *all* violated invariants, not just the first one.
    ///
    /// # Notes
    /// - If UUID↔key mappings are inconsistent, this returns only mapping failures (other
    ///   checks may produce misleading secondary errors).
    /// - This report is **cumulative** across Levels 1–4.
    ///
    /// # Errors
    ///
    /// Returns `Err(TriangulationValidationReport)` containing all violated invariants.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// // Returns Ok(()) on success; otherwise returns a report listing all violations.
    /// let report = dt.validation_report();
    /// assert!(report.is_ok());
    /// ```
    pub fn validation_report(&self) -> Result<(), TriangulationValidationReport>
    where
        K::Scalar: CoordinateScalar,
    {
        // Levels 1–3: reuse the Triangulation layer report.
        match self.tri.validation_report() {
            Ok(()) => {
                // Level 4 (Delaunay property)
                if let Err(e) = self.is_valid() {
                    return Err(TriangulationValidationReport {
                        violations: vec![InvariantViolation {
                            kind: InvariantKind::DelaunayProperty,
                            error: e.into(),
                        }],
                    });
                }
                Ok(())
            }
            Err(mut report) => {
                // If mappings are inconsistent, return the lower-layer report unchanged.
                if report.violations.iter().any(|v| {
                    matches!(
                        v.kind,
                        InvariantKind::VertexMappings | InvariantKind::CellMappings
                    )
                }) {
                    return Err(report);
                }

                // Level 4 (Delaunay property)
                if let Err(e) = self.is_valid() {
                    report.violations.push(InvariantViolation {
                        kind: InvariantKind::DelaunayProperty,
                        error: e.into(),
                    });
                }

                if report.violations.is_empty() {
                    Ok(())
                } else {
                    Err(report)
                }
            }
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
/// # Note on Locate Hint Persistence
///
/// The internal `last_inserted_cell` "locate hint" is intentionally **not** serialized.
/// Deserialization reconstructs a fresh triangulation via [`from_tds()`](Self::from_tds),
/// which resets the hint to `None`. This only affects performance for the first few
/// insertions after loading.
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
/// - Periodic `is_valid()` calls during incremental insertion
///
/// Until wired in, users should call `is_valid()` (Level 4 only) or `validate()` (Levels 1–4)
/// explicitly as needed.
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

    #[test]
    fn test_validation_policy_defaults_to_on_suspicion() {
        // empty() -> Triangulation::new_empty() -> ValidationPolicy::default()
        let dt_empty: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::empty();
        assert_eq!(dt_empty.validation_policy(), ValidationPolicy::OnSuspicion);

        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        // new() -> with_kernel() -> explicit validation_policy initialization
        let dt_new: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();
        assert_eq!(dt_new.validation_policy(), ValidationPolicy::OnSuspicion);

        // with_kernel() constructor path should also use the default policy
        let dt_with_kernel: DelaunayTriangulation<FastKernel<f64>, (), (), 2> =
            DelaunayTriangulation::with_kernel(FastKernel::new(), &vertices).unwrap();
        assert_eq!(
            dt_with_kernel.validation_policy(),
            ValidationPolicy::OnSuspicion
        );

        // from_tds() is a separate constructor path (const-friendly), and should also
        // default to OnSuspicion.
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let dt_from_tds: DelaunayTriangulation<FastKernel<f64>, (), (), 2> =
            DelaunayTriangulation::from_tds(tds, FastKernel::new());
        assert_eq!(
            dt_from_tds.validation_policy(),
            ValidationPolicy::OnSuspicion
        );
    }

    #[test]
    fn test_validation_policy_setter_and_getter_roundtrip() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Getter reflects the underlying Triangulation policy.
        assert_eq!(dt.validation_policy(), ValidationPolicy::OnSuspicion);
        assert_eq!(dt.tri.validation_policy, ValidationPolicy::OnSuspicion);

        dt.set_validation_policy(ValidationPolicy::Always);
        assert_eq!(dt.validation_policy(), ValidationPolicy::Always);
        assert_eq!(dt.tri.validation_policy, ValidationPolicy::Always);

        dt.set_validation_policy(ValidationPolicy::Never);
        assert_eq!(dt.validation_policy(), ValidationPolicy::Never);
        assert_eq!(dt.tri.validation_policy, ValidationPolicy::Never);

        dt.set_validation_policy(ValidationPolicy::OnSuspicion);
        assert_eq!(dt.validation_policy(), ValidationPolicy::OnSuspicion);
        assert_eq!(dt.tri.validation_policy, ValidationPolicy::OnSuspicion);
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
            Err(DelaunayTriangulationConstructionError::Triangulation(
                TriangulationConstructionError::InsufficientVertices { dimension, .. },
            )) => {
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
            Err(DelaunayTriangulationConstructionError::Triangulation(
                TriangulationConstructionError::InsufficientVertices { dimension, .. },
            )) => {
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
        assert!(dt.tds().validate().is_ok());
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

    // =========================================================================
    // Coverage-oriented tests (tarpaulin)
    // =========================================================================

    #[test]
    fn test_with_kernel_aborts_on_duplicate_uuid_in_insertion_loop() {
        let mut vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([2.0, 0.0]),
            vertex!([0.0, 2.0]),
            vertex!([0.25, 0.25]),
        ];

        // Ensure the duplicate UUID is introduced in the incremental insertion loop,
        // not during initial simplex construction.
        let dup_uuid = vertices[0].uuid();
        vertices[3].set_uuid(dup_uuid).unwrap();

        let result: Result<DelaunayTriangulation<FastKernel<f64>, (), (), 2>, _> =
            DelaunayTriangulation::with_kernel(FastKernel::new(), &vertices);

        match result.unwrap_err() {
            DelaunayTriangulationConstructionError::Triangulation(
                TriangulationConstructionError::Tds(TdsConstructionError::DuplicateUuid {
                    entity: _,
                    uuid,
                }),
            ) => {
                assert_eq!(uuid, dup_uuid);
            }
            other => panic!("Expected DuplicateUuid error, got {other:?}"),
        }
    }

    #[test]
    fn test_is_valid_reports_delaunay_violation_and_includes_cell_uuid() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([2.0, 0.0]),
            vertex!([0.0, 2.0]),
        ];

        let mut tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();

        // Add a vertex strictly inside the circumcircle of the triangle.
        // This does *not* change topology (no new cells), but it *does* violate the
        // Delaunay empty-circumsphere property.
        tds.insert_vertex_with_mapping(vertex!([0.5, 0.5])).unwrap();

        let dt: DelaunayTriangulation<FastKernel<f64>, (), (), 2> =
            DelaunayTriangulation::from_tds(tds, FastKernel::new());

        match dt.is_valid().unwrap_err() {
            DelaunayTriangulationValidationError::DelaunayViolation {
                cell_key,
                cell_uuid,
            } => {
                assert_ne!(cell_uuid, Uuid::nil());
                assert_eq!(
                    dt.tds()
                        .cell_uuid_from_key(cell_key)
                        .unwrap_or_else(Uuid::nil),
                    cell_uuid
                );
            }
            other => panic!("Expected DelaunayViolation, got {other:?}"),
        }
    }

    #[test]
    fn test_is_valid_maps_triangulation_state_error_for_dangling_vertex_key() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        let mut tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();

        // Insert a structurally-valid cell that references a non-existent vertex key.
        // This intentionally violates TDS invariants to exercise the validation error mapping.
        let vertex_keys: Vec<_> = tds.vertices().map(|(k, _)| k).collect();
        let v0 = vertex_keys[0];
        let v1 = vertex_keys[1];
        let dangling = VertexKey::default();

        let cell = Cell::new(vec![v0, v1, dangling], None).unwrap();
        let _ = tds.cells_mut().insert(cell);

        let dt: DelaunayTriangulation<FastKernel<f64>, (), (), 2> =
            DelaunayTriangulation::from_tds(tds, FastKernel::new());

        match dt.is_valid().unwrap_err() {
            DelaunayTriangulationValidationError::Triangulation(
                TriangulationValidationError::Tds(TdsValidationError::InconsistentDataStructure {
                    ..
                }),
            ) => {}
            other => panic!("Expected TriangulationState→Triangulation error, got {other:?}"),
        }
    }

    #[test]
    fn test_is_valid_maps_invalid_cell_to_triangulation_error() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        let mut tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();

        // Corrupt the (only) cell: neighbors buffer with the wrong length.
        let cell_key = tds.cell_keys().next().unwrap();
        let cell = tds.get_cell_by_key_mut(cell_key).unwrap();
        let mut bad_neighbors = crate::core::collections::NeighborBuffer::<Option<CellKey>>::new();
        bad_neighbors.resize(2, None); // expected D+1 = 3
        cell.neighbors = Some(bad_neighbors);

        let dt: DelaunayTriangulation<FastKernel<f64>, (), (), 2> =
            DelaunayTriangulation::from_tds(tds, FastKernel::new());

        match dt.is_valid().unwrap_err() {
            DelaunayTriangulationValidationError::Triangulation(
                TriangulationValidationError::Tds(TdsValidationError::InvalidCell { .. }),
            ) => {}
            other => panic!("Expected InvalidCell→Triangulation error, got {other:?}"),
        }
    }

    #[test]
    fn test_validation_report_ok_for_valid_triangulation() {
        let vertices = [
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
        assert!(dt.validation_report().is_ok());
    }

    #[test]
    fn test_validation_report_returns_mapping_failures_only() {
        let vertices = [
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Break UUID↔key mappings: remove one vertex UUID entry.
        let uuid = dt.tri.tds.vertices().next().unwrap().1.uuid();
        dt.tri.tds.uuid_to_vertex_key.remove(&uuid);

        let report = dt.validation_report().unwrap_err();
        assert!(!report.violations.is_empty());
        assert!(report.violations.iter().all(|v| {
            matches!(
                v.kind,
                InvariantKind::VertexMappings | InvariantKind::CellMappings
            )
        }));

        // Early-return on mapping failures: do not add derived invariants.
        assert!(
            report
                .violations
                .iter()
                .all(|v| v.kind != InvariantKind::DelaunayProperty)
        );
    }

    #[test]
    fn test_validation_report_includes_vertex_incidence_violation() {
        let vertices = [
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Corrupt a `Vertex::incident_cell` pointer.
        let vertex_key = dt.tri.tds.vertices().next().unwrap().0;
        dt.tri
            .tds
            .get_vertex_by_key_mut(vertex_key)
            .unwrap()
            .incident_cell = Some(CellKey::default());

        let report = dt.validation_report().unwrap_err();
        assert!(
            report
                .violations
                .iter()
                .any(|v| v.kind == InvariantKind::VertexIncidence)
        );
    }

    #[test]
    fn test_validation_report_includes_delaunay_property_violation() {
        // Construct a non-Delaunay configuration without introducing mapping errors.
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([2.0, 0.0]),
            vertex!([0.0, 2.0]),
        ];

        let mut tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        tds.insert_vertex_with_mapping(vertex!([0.5, 0.5])).unwrap();

        let dt: DelaunayTriangulation<FastKernel<f64>, (), (), 2> =
            DelaunayTriangulation::from_tds(tds, FastKernel::new());

        let report = dt.validation_report().unwrap_err();
        assert!(
            report
                .violations
                .iter()
                .any(|v| v.kind == InvariantKind::DelaunayProperty)
        );
    }

    #[test]
    fn test_serde_roundtrip_uses_custom_deserialize_impl() {
        let vertices = [
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let dt: DelaunayTriangulation<FastKernel<f64>, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();

        let json = serde_json::to_string(&dt).unwrap();
        let roundtrip: DelaunayTriangulation<FastKernel<f64>, (), (), 3> =
            serde_json::from_str(&json).unwrap();

        assert_eq!(roundtrip.number_of_vertices(), dt.number_of_vertices());
        assert_eq!(roundtrip.number_of_cells(), dt.number_of_cells());

        // `last_inserted_cell` is a performance-only locate hint and is intentionally not
        // persisted across serde round-trips (it is reset to `None` in `from_tds`).
        assert!(roundtrip.last_inserted_cell.is_none());
    }

    // =========================================================================
    // Topology traversal forwarding tests (DelaunayTriangulation → Triangulation)
    // =========================================================================

    #[test]
    fn test_topology_traversal_methods_are_forwarded() {
        // Single tetrahedron: 4 vertices, 1 cell, 6 unique edges.
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let tri = dt.triangulation();

        let edges_dt: std::collections::HashSet<_> = dt.edges().collect();
        let edges_tri: std::collections::HashSet<_> = tri.edges().collect();
        assert_eq!(edges_dt, edges_tri);
        assert_eq!(edges_dt.len(), 6);

        let index = dt.build_adjacency_index().unwrap();
        let edges_dt_index: std::collections::HashSet<_> = dt.edges_with_index(&index).collect();
        let edges_tri_index: std::collections::HashSet<_> = tri.edges_with_index(&index).collect();
        assert_eq!(edges_dt_index, edges_tri_index);
        assert_eq!(edges_dt_index, edges_dt);

        let v0 = dt.vertices().next().unwrap().0;
        let incident_dt: std::collections::HashSet<_> = dt.incident_edges(v0).collect();
        let incident_tri: std::collections::HashSet<_> = tri.incident_edges(v0).collect();
        assert_eq!(incident_dt, incident_tri);
        assert_eq!(incident_dt.len(), 3);

        let incident_dt_index: std::collections::HashSet<_> =
            dt.incident_edges_with_index(&index, v0).collect();
        let incident_tri_index: std::collections::HashSet<_> =
            tri.incident_edges_with_index(&index, v0).collect();
        assert_eq!(incident_dt_index, incident_tri_index);
        assert_eq!(incident_dt_index, incident_dt);

        let cell_key = dt.cells().next().unwrap().0;
        let neighbors_dt: Vec<_> = dt.cell_neighbors(cell_key).collect();
        let neighbors_tri: Vec<_> = tri.cell_neighbors(cell_key).collect();
        assert_eq!(neighbors_dt, neighbors_tri);
        assert!(neighbors_dt.is_empty());

        let neighbors_dt_index: Vec<_> = dt.cell_neighbors_with_index(&index, cell_key).collect();
        let neighbors_tri_index: Vec<_> = tri.cell_neighbors_with_index(&index, cell_key).collect();
        assert_eq!(neighbors_dt_index, neighbors_tri_index);
        assert_eq!(neighbors_dt_index, neighbors_dt);

        // Geometry/topology accessors should be forwarded as well.
        let cell_vertices_dt = dt.cell_vertices(cell_key).unwrap();
        let cell_vertices_tri = tri.cell_vertices(cell_key).unwrap();
        assert_eq!(cell_vertices_dt, cell_vertices_tri);
        assert_eq!(cell_vertices_dt.len(), 4);

        let coords_dt = dt.vertex_coords(v0).unwrap();
        let coords_tri = tri.vertex_coords(v0).unwrap();
        assert_eq!(coords_dt, coords_tri);

        // Missing keys should behave the same as on `Triangulation`.
        assert!(dt.vertex_coords(VertexKey::default()).is_none());
        assert!(dt.cell_vertices(CellKey::default()).is_none());
    }
}
