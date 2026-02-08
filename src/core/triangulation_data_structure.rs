//! Data and operations on d-dimensional triangulation data structures.
//!
//! This module provides the `Tds` (Triangulation Data Structure) struct which represents
//! a D-dimensional finite simplicial complex with geometric vertices, cells, and their
//! topological relationships. The implementation closely follows the design principles
//! of [CGAL Triangulation](https://doc.cgal.org/latest/Triangulation/index.html).
//!
//! # Key Features
//!
//! - **Generic Coordinate Support**: Works with any floating-point type (`f32`, `f64`, etc.)
//!   that implements the `CoordinateScalar` trait
//! - **Arbitrary Dimensions**: Supports triangulations in any dimension D ≥ 1
//! - **Delaunay Triangulation**: Implements Bowyer-Watson algorithm for Delaunay triangulation
//! - **Hierarchical Cell Structure**: Stores maximal D-dimensional cells and infers lower-dimensional
//!   simplices (vertices, edges, facets) from the maximal cells
//! - **Neighbor Relationships**: Maintains adjacency information between cells for efficient
//!   traversal and geometric queries
//! - **Validation Support**: Comprehensive validation of triangulation properties including
//!   neighbor consistency and geometric validity
//! - **Serialization Support**: Full serde support for persistence and data exchange
//! - **Optimized Storage**: Internal key-based storage with UUIDs for external identity
//!
//! # Geometric Structure
//!
//! The triangulation data structure represents a finite simplicial complex where:
//!
//! - **0-cells**: Individual vertices embedded in D-dimensional Euclidean space
//! - **1-cells**: Edges connecting two vertices (inferred from maximal cells)
//! - **2-cells**: Triangular faces with three vertices (inferred from maximal cells)
//! - **...**
//! - **D-cells**: Maximal D-dimensional simplices with D+1 vertices (explicitly stored)
//!
//! For example, in 3D space:
//! - Vertices are 0-dimensional cells
//! - Edges are 1-dimensional cells (inferred from tetrahedra)
//! - Faces are 2-dimensional cells represented as `Facet`s
//! - Tetrahedra are 3-dimensional cells (maximal cells)
//!
//! # Delaunay Property
//!
//! When constructed via the Delaunay triangulation algorithm, the structure satisfies
//! the **empty circumsphere property**: no vertex lies inside the circumsphere of any
//! D-dimensional cell. This property ensures optimal geometric characteristics for
//! many applications including mesh generation, interpolation, and spatial analysis.
//!
//! # Topological Invariants
//!
//! Valid Delaunay triangulations maintain several critical topological invariants:
//!
//! - **Facet Sharing Invariant**: Every facet (D-1 dimensional face) is shared by exactly
//!   two cells, except for boundary facets which belong to exactly one cell. This ensures
//!   the triangulation forms a valid simplicial complex.
//! - **Neighbor Consistency**: Adjacent cells properly reference each other through their
//!   shared facets, maintaining bidirectional neighbor relationships.
//! - **Vertex Incidence**: Each vertex is incident to a well-defined set of cells that
//!   form a topologically valid star configuration around the vertex.
//! - **Delaunay Property**: No vertex lies inside the circumsphere of any D-dimensional cell.
//!
//! ## Invariant Enforcement
//!
//! | Invariant Type | Enforcement Location | Method |
//! |---|---|---|
//! | **Delaunay Property** | incremental insertion (`core::algorithms::incremental_insertion`) | Empty circumsphere test via `insphere()` (best-effort) |
//! | **Facet Sharing** | `Tds::is_valid()` / `Tds::validate()` | Each facet shared by ≤ 2 cells |
//! | **No Duplicate Cells** | `Tds::is_valid()` / `Tds::validate()` | No cells with identical vertex sets |
//! | **Neighbor Consistency** | `Tds::is_valid()` / `Tds::validate()` | Mutual neighbor relationships |
//! | **Cell Vertex Keys** | `Tds::is_valid()` / `Tds::validate()` | Cells reference only valid vertex keys |
//! | **Vertex Incidence** | `Tds::is_valid()` / `Tds::validate()` | `Vertex::incident_cell` is non-dangling and consistent (when present) |
//! | **Cell Validity** | `CellBuilder::validate()` (vertex count) + `cell.is_valid()` (comprehensive) | Construction + runtime validation |
//! | **Vertex Validity** | `Point::from()` (coordinates) + UUID auto-gen + `vertex.is_valid()` | Construction + runtime validation |
//!
//! The incremental insertion algorithm attempts to maintain the Delaunay property during
//! construction, but rare violations can remain. Structural invariants are enforced
//! **reactively** through validation methods. For a definitive Delaunay check, run
//! Level 4 validation via `DelaunayTriangulation::is_valid()` / `DelaunayTriangulation::validate()`.
//!
//! # Validation
//!
//! The TDS participates in a layered validation hierarchy:
//!
//! ## Validation Hierarchy (TDS Role)
//!
//! 1. **Level 1: Element Validity** - [`Cell::is_valid()`], [`Vertex::is_valid()`]
//!    - Basic data integrity (coordinates, UUIDs, initialization)
//! 2. **Level 2: TDS Structural Validity** - [`Tds::is_valid()`] ← **This module**
//!    - UUID ↔ Key mapping consistency
//!    - Cells reference only valid vertex keys (no stale/missing vertex keys)
//!    - `Vertex::incident_cell`, when present, must point at an existing cell that contains the vertex
//!    - Isolated vertices (not referenced by any cell) are allowed at this layer (`incident_cell` may be `None`)
//!    - No duplicate cells
//!    - Facet sharing invariant (≤2 cells per facet)
//!    - Neighbor consistency
//! 3. **Level 3: Manifold Topology** - [`Triangulation::is_valid()`]
//!    - Builds on Level 2, and rejects isolated vertices (every vertex must be incident to ≥ 1 cell)
//!    - Adds manifold-with-boundary + Euler characteristic
//! 4. **Level 4: Delaunay Property** - [`DelaunayTriangulation::is_valid()`]
//!    - Empty circumsphere property
//!
//! ## TDS Validation Methods
//!
//! - [`is_valid()`](Tds::is_valid) - Level 2 only (structural); returns first error, stops early
//! - [`validate()`](Tds::validate) - Levels 1–2 (elements + structural); returns first error, stops early
//!
//! For cumulative diagnostics across the full stack (Levels 1–4), use
//! [`DelaunayTriangulation::validation_report()`].
//!
//! ## Example: Using Validation
//!
//! ```rust
//! use delaunay::prelude::triangulation::*;
//!
//! let vertices = [
//!     vertex!([0.0, 0.0, 0.0]),
//!     vertex!([1.0, 0.0, 0.0]),
//!     vertex!([0.0, 1.0, 0.0]),
//!     vertex!([0.0, 0.0, 1.0]),
//! ];
//! let dt = DelaunayTriangulation::new(&vertices).unwrap();
//!
//! // Level 2: structural only (fast)
//! assert!(dt.tds().is_valid().is_ok());
//!
//! // Levels 1–2: elements + structural
//! assert!(dt.tds().validate().is_ok());
//!
//! // Full report across Levels 1–4
//! match dt.validation_report() {
//!     Ok(()) => println!("✓ All invariants satisfied"),
//!     Err(report) => {
//!         for violation in report.violations {
//!             eprintln!("Invariant: {:?}, Error: {}", violation.kind, violation.error);
//!         }
//!     }
//! }
//! ```
//!
//! See [`docs/validation.md`](https://github.com/acgetchell/delaunay/blob/main/docs/validation.md)
//! for a comprehensive validation guide.
//!
//! [`Cell::is_valid()`]: crate::core::cell::Cell::is_valid
//! [`Vertex::is_valid()`]: crate::core::vertex::Vertex::is_valid
//! [`Triangulation::is_valid()`]: crate::core::triangulation::Triangulation::is_valid
//! [`DelaunayTriangulation::is_valid()`]: crate::core::delaunay_triangulation::DelaunayTriangulation::is_valid
//! [`DelaunayTriangulation::validation_report()`]: crate::core::delaunay_triangulation::DelaunayTriangulation::validation_report
//!
//! # Examples
//!
//! ## Creating a 3D Triangulation
//!
//! ```rust
//! use delaunay::prelude::triangulation::*;
//!
//! // Create vertices for a tetrahedron
//! let vertices = [
//!     vertex!([0.0, 0.0, 0.0]),
//!     vertex!([1.0, 0.0, 0.0]),
//!     vertex!([0.0, 1.0, 0.0]),
//!     vertex!([0.0, 0.0, 1.0]),
//! ];
//!
//! // Create Delaunay triangulation
//! let dt = DelaunayTriangulation::new(&vertices).unwrap();
//!
//! // Query triangulation properties
//! assert_eq!(dt.number_of_vertices(), 4);
//! assert_eq!(dt.number_of_cells(), 1);
//! assert_eq!(dt.dim(), 3);
//! assert!(dt.validate().is_ok());
//! ```
//!
//! ## Adding Vertices to Existing Triangulation
//!
//! ```rust
//! use delaunay::prelude::triangulation::*;
//!
//! // Start with initial vertices
//! let initial_vertices = [
//!     vertex!([0.0, 0.0, 0.0]),
//!     vertex!([1.0, 0.0, 0.0]),
//!     vertex!([0.0, 1.0, 0.0]),
//!     vertex!([0.0, 0.0, 1.0]),
//! ];
//!
//! let mut dt = DelaunayTriangulation::new(&initial_vertices).unwrap();
//!
//! // Add a new vertex
//! let new_vertex = vertex!([0.2, 0.2, 0.2]);
//! dt.insert(new_vertex).unwrap();
//!
//! assert_eq!(dt.number_of_vertices(), 5);
//! assert!(dt.validate().is_ok());
//! ```
//!
//! ## 4D Triangulation
//!
//! ```rust
//! use delaunay::prelude::triangulation::*;
//!
//! // Create 4D triangulation with 5 vertices (needed for a 4-simplex)
//! let vertices_4d = [
//!     vertex!([0.0, 0.0, 0.0, 0.0]),  // Origin
//!     vertex!([1.0, 0.0, 0.0, 0.0]),  // Unit vector along first dimension
//!     vertex!([0.0, 1.0, 0.0, 0.0]),  // Unit vector along second dimension
//!     vertex!([0.0, 0.0, 1.0, 0.0]),  // Unit vector along third dimension
//!     vertex!([0.0, 0.0, 0.0, 1.0]),  // Unit vector along fourth dimension
//! ];
//!
//! let dt_4d = DelaunayTriangulation::new(&vertices_4d).unwrap();
//! assert_eq!(dt_4d.dim(), 4);
//! assert_eq!(dt_4d.number_of_vertices(), 5);
//! assert_eq!(dt_4d.number_of_cells(), 1);
//! assert!(dt_4d.validate().is_ok());
//! ```
//!
//! # References
//!
//! - [CGAL Triangulation Documentation](https://doc.cgal.org/latest/Triangulation/index.html)
//! - Bowyer, A. "Computing Dirichlet tessellations." The Computer Journal 24.2 (1981): 162-166
//! - Watson, D.F. "Computing the n-dimensional Delaunay tessellation with application to Voronoi polytopes." The Computer Journal 24.2 (1981): 167-172
//! - de Berg, M., et al. "Computational Geometry: Algorithms and Applications." 3rd ed. Springer-Verlag, 2008

#![forbid(unsafe_code)]

use super::{
    cell::{Cell, CellValidationError},
    facet::{FacetHandle, facet_key_from_vertices},
    traits::data_type::DataType,
    util::usize_to_u8,
    vertex::{Vertex, VertexValidationError},
};
use crate::core::collections::{
    CellKeySet, CellRemovalBuffer, CellVertexUuidBuffer, CellVerticesMap, Entry, FacetToCellsMap,
    FastHashMap, MAX_PRACTICAL_DIMENSION_SIZE, NeighborBuffer, SmallBuffer, StorageMap,
    UuidToCellKeyMap, UuidToVertexKeyMap, VertexKeyBuffer, VertexKeySet,
    fast_hash_map_with_capacity,
};
use crate::core::triangulation::TriangulationValidationError;
use crate::geometry::traits::coordinate::CoordinateScalar;
use serde::{
    Deserialize, Deserializer, Serialize,
    de::{self, MapAccess, Visitor},
};
use slotmap::new_key_type;
use std::{
    cmp::Ordering as CmpOrdering,
    fmt::{self, Debug},
    marker::PhantomData,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
use thiserror::Error;
use uuid::Uuid;

// =============================================================================
// CONSTRUCTION STATE TYPES
// =============================================================================

/// Represents the construction state of a triangulation.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::triangulation::*;
///
/// let state = TriangulationConstructionState::Incomplete(2);
/// assert!(matches!(state, TriangulationConstructionState::Incomplete(2)));
///
/// let default_state = TriangulationConstructionState::default();
/// assert!(matches!(
///     default_state,
///     TriangulationConstructionState::Incomplete(0)
/// ));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TriangulationConstructionState {
    /// The triangulation has insufficient vertices to form a complete D-dimensional triangulation.
    /// Contains the number of vertices currently stored.
    Incomplete(usize),
    /// The triangulation is complete and valid with at least D+1 vertices and proper cell structure.
    Constructed,
}

impl Default for TriangulationConstructionState {
    fn default() -> Self {
        Self::Incomplete(0)
    }
}

// =============================================================================
// ERROR TYPES
// =============================================================================

/// Errors that can occur during TDS construction operations.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::triangulation::*;
/// use uuid::Uuid;
///
/// let err = TdsConstructionError::DuplicateUuid {
///     entity: EntityKind::Vertex,
///     uuid: Uuid::nil(),
/// };
/// assert!(matches!(err, TdsConstructionError::DuplicateUuid { .. }));
/// ```
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum TdsConstructionError {
    /// Validation error during construction.
    #[error("Validation error during construction: {0}")]
    ValidationError(#[from] TdsValidationError),
    /// Attempted to insert an entity with a UUID that already exists.
    #[error("Duplicate UUID: {entity:?} with UUID {uuid} already exists")]
    DuplicateUuid {
        /// The type of entity.
        entity: EntityKind,
        /// The UUID that was duplicated.
        uuid: Uuid,
    },
}

/// Represents the type of entity in the triangulation.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::triangulation::*;
///
/// let kind = EntityKind::Cell;
/// assert_eq!(kind, EntityKind::Cell);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntityKind {
    /// A vertex entity.
    Vertex,
    /// A cell entity.
    Cell,
}

// REMOVED: TriangulationStatistics and TriangulationDiagnostics
// These were part of the deprecated Bowyer-Watson architecture that has been removed.
// Statistics tracking will be reimplemented for the incremental insertion algorithm
// in a future release.

/// Example usage
///
/// ```
/// use delaunay::prelude::triangulation::*;
///
/// // Build a simple 3D triangulation
/// let vertices = [
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let dt = DelaunayTriangulation::new(&vertices).unwrap();
/// assert_eq!(dt.number_of_vertices(), 4);
/// assert!(dt.is_valid().is_ok());
/// ```
///
/// Errors that can occur during triangulation validation (post-construction).
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum TdsValidationError {
    /// The triangulation contains an invalid vertex.
    #[error("Invalid vertex {vertex_id}: {source}")]
    InvalidVertex {
        /// The UUID of the invalid vertex.
        vertex_id: Uuid,
        /// The underlying vertex validation error.
        source: VertexValidationError,
    },
    /// The triangulation contains an invalid cell.
    #[error("Invalid cell {cell_id}: {source}")]
    InvalidCell {
        /// The UUID of the invalid cell.
        cell_id: Uuid,
        /// The underlying cell validation error.
        source: CellValidationError,
    },
    /// Neighbor relationships are invalid.
    #[error("Invalid neighbor relationships: {message}")]
    InvalidNeighbors {
        /// Description of the neighbor validation failure.
        message: String,
    },
    /// The triangulation contains duplicate cells.
    #[error("Duplicate cells detected: {message}")]
    DuplicateCells {
        /// Description of the duplicate cell validation failure.
        message: String,
    },
    /// Failed to create a cell during triangulation.
    #[error("Failed to create cell: {message}")]
    FailedToCreateCell {
        /// Description of the cell creation failure.
        message: String,
    },
    /// Cells are not neighbors as expected
    #[error("Cells {cell1:?} and {cell2:?} are not neighbors")]
    NotNeighbors {
        /// The first cell UUID.
        cell1: Uuid,
        /// The second cell UUID.
        cell2: Uuid,
    },
    /// Entity mapping inconsistency (vertex or cell).
    #[error("{entity:?} mapping inconsistency: {message}")]
    MappingInconsistency {
        /// The type of entity with the mapping issue.
        entity: EntityKind,
        /// Description of the mapping inconsistency.
        message: String,
    },
    /// Failed to retrieve vertex keys for a cell during neighbor assignment.
    #[error("Failed to retrieve vertex keys for cell {cell_id}: {message}")]
    VertexKeyRetrievalFailed {
        /// The UUID of the cell that failed.
        cell_id: Uuid,
        /// Description of the failure.
        message: String,
    },
    /// Internal data structure inconsistency during neighbor assignment.
    #[error("Internal data structure inconsistency: {message}")]
    InconsistentDataStructure {
        /// Description of the inconsistency.
        message: String,
    },

    /// Level 3 topology validation failed (manifold / Euler characteristic, etc.).
    ///
    /// This preserves the structured Level‑3 validation error when topology checks
    /// need to be surfaced through APIs that currently return [`TdsValidationError`]
    /// (notably the incremental insertion rollback path).
    ///

    /// Insufficient vertices to create a triangulation.
    #[error("Insufficient vertices for {dimension}D triangulation: {source}")]
    InsufficientVertices {
        /// The dimension that was attempted.
        dimension: usize,
        /// The underlying cell validation error.
        source: CellValidationError,
    },
    /// Facet operation failed during validation.
    #[error("Facet operation failed: {0}")]
    FacetError(#[from] super::facet::FacetError),
    /// Finalization failed during triangulation operations.
    #[error("Finalization failed: {message}")]
    FinalizationFailed {
        /// Description of the finalization failure, including underlying error details.
        message: String,
    },
}

/// Errors that can occur during TDS mutation operations.
///
/// This error is a thin wrapper around [`TdsValidationError`]. Mutation operations can fail
/// for the same reasons as validation (i.e., because an invariant would be violated or a
/// consistency check fails while attempting to perform the mutation).
///
/// The wrapper exists to make call sites and API docs semantically explicit, while also
/// allowing this error to evolve into a richer, dedicated type in a future release without
/// breaking the public API surface.
///
/// # Stability / conversion contract
///
/// `TdsMutationError` currently supports lossless conversion to and from [`TdsValidationError`]
/// via the provided `From`/`Into` impls. If this wrapper evolves to include mutation-specific
/// context (additional fields/variants), converting `TdsMutationError` into [`TdsValidationError`]
/// may become lossy.
///
/// Callers that want to preserve potential future mutation-specific details should avoid
/// converting back to [`TdsValidationError`] and instead propagate/handle `TdsMutationError`
/// directly.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::triangulation::*;
///
/// let validation = TdsValidationError::InvalidNeighbors {
///     message: "bad neighbors".to_string(),
/// };
/// let mutation: TdsMutationError = validation.clone().into();
/// let round_trip: TdsValidationError = mutation.clone().into();
/// assert_eq!(round_trip, validation);
/// ```
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error(transparent)]
pub struct TdsMutationError(pub TdsValidationError);

impl From<TdsValidationError> for TdsMutationError {
    fn from(err: TdsValidationError) -> Self {
        Self(err)
    }
}

impl From<TdsMutationError> for TdsValidationError {
    fn from(err: TdsMutationError) -> Self {
        err.0
    }
}

// Temporary internal alias to ease refactors within this module.
// This does not affect the public API.
type TdsError = TdsValidationError;

/// Classifies the kind of triangulation invariant that failed during validation.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::triangulation::*;
///
/// let kind = InvariantKind::Topology;
/// assert_eq!(kind, InvariantKind::Topology);
/// ```
///
/// This is used by [`TriangulationValidationReport`] to group related errors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InvariantKind {
    /// Per-vertex validity (finite coordinates, non-nil UUID, etc.).
    VertexValidity,
    /// Per-cell validity (vertex count, duplicate vertices, nil UUID, etc.).
    CellValidity,
    /// Vertex UUID↔key mapping invariants.
    VertexMappings,
    /// Cell UUID↔key mapping invariants.
    CellMappings,
    /// Cells reference only valid vertex keys (no stale/missing vertex keys).
    CellVertexKeys,
    /// Vertex incidence invariants (`Vertex::incident_cell` pointers are non-dangling + consistent).
    VertexIncidence,
    /// No duplicate maximal cells with identical vertex sets.
    DuplicateCells,
    /// Facet sharing invariants (each facet shared by at most 2 cells).
    FacetSharing,
    /// Neighbor topology and mutual-consistency invariants.
    NeighborConsistency,
    /// Triangulation/topology invariants (manifold-with-boundary, Euler characteristic).
    Topology,
    /// Delaunay empty-circumsphere property.
    DelaunayProperty,
}

/// A union error type that can represent any layer's validation failure.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::triangulation::*;
///
/// let err = InvariantError::Tds(TdsValidationError::InvalidNeighbors {
///     message: "bad neighbors".to_string(),
/// });
/// assert!(matches!(err, InvariantError::Tds(_)));
/// ```
///
/// This is used by [`TriangulationValidationReport`] so that diagnostic reporting can
/// preserve structured errors from each layer (TDS / topology / Delaunay) without
/// stringification.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum InvariantError {
    /// Level 1–2 (elements + TDS structure).
    #[error(transparent)]
    Tds(#[from] TdsValidationError),

    /// Level 3 (topology).
    #[error(transparent)]
    Triangulation(#[from] TriangulationValidationError),

    /// Level 4 (Delaunay property).
    #[error(transparent)]
    Delaunay(#[from] crate::core::delaunay_triangulation::DelaunayTriangulationValidationError),
}

/// A single invariant violation recorded during validation diagnostics.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::triangulation::*;
///
/// let violation = InvariantViolation {
///     kind: InvariantKind::Topology,
///     error: InvariantError::Tds(TdsValidationError::InvalidNeighbors {
///         message: "bad neighbors".to_string(),
///     }),
/// };
/// assert_eq!(violation.kind, InvariantKind::Topology);
/// ```
#[derive(Clone, Debug)]
pub struct InvariantViolation {
    /// The kind of invariant that failed.
    pub kind: InvariantKind,
    /// The detailed validation error explaining the failure.
    pub error: InvariantError,
}

/// Aggregate report of one or more validation failures.
///
/// This is returned by
/// [`DelaunayTriangulation::validation_report()`]
/// to surface all failed invariants at once for debugging and test diagnostics.
///
/// [`DelaunayTriangulation::validation_report()`]: crate::core::delaunay_triangulation::DelaunayTriangulation::validation_report
///
/// # Examples
///
/// ```rust
/// use delaunay::core::triangulation_data_structure::TriangulationValidationReport;
///
/// let report = TriangulationValidationReport { violations: Vec::new() };
/// assert!(report.is_empty());
/// ```
#[derive(Clone, Debug)]
pub struct TriangulationValidationReport {
    /// The ordered list of invariant violations that occurred.
    pub violations: Vec<InvariantViolation>,
}

impl TriangulationValidationReport {
    /// Returns `true` if no violations were recorded.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.violations.is_empty()
    }
}

// =============================================================================
// MACROS/HELPERS
// =============================================================================

// Define key types for storage maps using slotmap's new_key_type! macro
// These macros create unique, type-safe keys for accessing elements in storage maps

new_key_type! {
    /// Key type for accessing vertices in the storage map.
    ///
    /// This creates a unique, type-safe identifier for vertices stored in the
    /// triangulation's vertex storage. Each VertexKey corresponds to exactly
    /// one vertex and provides efficient, stable access even as vertices are
    /// added or removed from the triangulation.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let key = dt.tds().vertex_keys().next().unwrap();
    /// let _ = key;
    /// ```
    pub struct VertexKey;
}

new_key_type! {
    /// Key type for accessing cells in the storage map.
    ///
    /// This creates a unique, type-safe identifier for cells stored in the
    /// triangulation's cell storage. Each CellKey corresponds to exactly
    /// one cell and provides efficient, stable access even as cells are
    /// added or removed during triangulation operations.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let key = dt.tds().cell_keys().next().unwrap();
    /// let _ = key;
    /// ```
    pub struct CellKey;
}

#[derive(Clone, Debug)]
/// The `Tds` struct represents a triangulation data structure with vertices
/// and cells, where the vertices and cells are identified by UUIDs.
///
/// # Properties
///
/// - `vertices`: A storage map that stores vertices with stable keys for efficient access.
///   Each [`Vertex`] has a point of type T, vertex data of type U, and a constant D representing the dimension.
/// - `cells`: The `cells` property is a storage map that stores [`Cell`] objects with stable keys.
///   Each [`Cell`] has one or more [`Vertex`] objects with cell data of type V.
///   Note the dimensionality of the cell may differ from D, though the [`Tds`]
///   only stores cells of maximal dimensionality D and infers other lower
///   dimensional cells (cf. Facets) from the maximal cells and their vertices.
///
/// For example, in 3 dimensions:
///
/// - A 0-dimensional cell is a [`Vertex`].
/// - A 1-dimensional cell is an `Edge` given by the `Tetrahedron` and two
///   [`Vertex`] endpoints.
/// - A 2-dimensional cell is a `Facet` given by the `Tetrahedron` and the
///   opposite [`Vertex`].
/// - A 3-dimensional cell is a `Tetrahedron`, the maximal cell.
///
/// A similar pattern holds for higher dimensions.
///
/// In general, vertices are embedded into D-dimensional Euclidean space,
/// and so the [`Tds`] is a finite simplicial complex.
///
/// # Usage
///
/// The `Tds` struct is the primary entry point for creating and manipulating
/// Delaunay triangulations. It is initialized with a set of vertices and
/// automatically computes the triangulation.
///
/// ```rust
/// use delaunay::prelude::triangulation::*;
///
/// // Create vertices for a 2D triangulation
/// let vertices = [
///     vertex!([0.0, 0.0]),
///     vertex!([1.0, 0.0]),
///     vertex!([0.5, 1.0]),
/// ];
///
/// // Create a new triangulation
/// let dt = DelaunayTriangulation::new(&vertices).unwrap();
///
/// // Check the number of cells and vertices
/// assert_eq!(dt.number_of_cells(), 1);
/// assert_eq!(dt.number_of_vertices(), 3);
/// ```
pub struct Tds<T, U, V, const D: usize>
where
    U: DataType,
    V: DataType,
{
    /// Storage map for vertices, allowing stable keys and efficient access.
    vertices: StorageMap<VertexKey, Vertex<T, U, D>>,

    /// Storage map for cells, providing stable keys and efficient access.
    cells: StorageMap<CellKey, Cell<T, U, V, D>>,

    /// Fast mapping from Vertex UUIDs to their `VertexKeys` for efficient UUID → Key lookups.
    /// This optimizes the common operation of looking up vertex keys by UUID.
    /// For reverse Key → UUID lookups, we use direct storage map access: `vertices[key].uuid()`.
    ///
    /// SAFETY: External mutation of this map will violate TDS invariants.
    /// This should only be modified through TDS methods that maintain consistency.
    ///
    /// Note: Not serialized - reconstructed during deserialization from vertices.
    pub(crate) uuid_to_vertex_key: UuidToVertexKeyMap,

    /// Fast mapping from Cell UUIDs to their `CellKeys` for efficient UUID → Key lookups.
    /// This optimizes the common operation of looking up cell keys by UUID.
    /// For reverse Key → UUID lookups, we use direct storage map access: `cells[key].uuid()`.
    ///
    /// SAFETY: External mutation of this map will violate TDS invariants.
    /// This should only be modified through TDS methods that maintain consistency.
    ///
    /// Note: Not serialized - reconstructed during deserialization from cells.
    pub(crate) uuid_to_cell_key: UuidToCellKeyMap,

    /// The current construction state of the triangulation.
    /// This field tracks whether the triangulation has enough vertices to form a complete
    /// D-dimensional triangulation or if it's still being incrementally built.
    ///
    /// Note: Not serialized - only constructed triangulations should be serialized.
    pub construction_state: TriangulationConstructionState,

    /// Generation counter for invalidating caches.
    /// This counter is incremented whenever the triangulation structure is modified
    /// (vertices added, cells created/removed, etc.), allowing dependent caches to
    /// detect when they need to refresh.
    /// Uses `Arc<AtomicU64>` for thread-safe operations in concurrent contexts while allowing Clone.
    ///
    /// Note: Not serialized - generation is runtime-only.
    generation: Arc<AtomicU64>,
}

// =============================================================================
// CORE FUNCTIONALITY
// =============================================================================

// =============================================================================
// PURE COMBINATORIAL METHODS (No geometric operations)
// =============================================================================
// These methods work with the combinatorial structure only - vertices, cells,
// neighbors, facets, keys, and UUIDs. They do NOT require coordinate operations
// and are designed to be independent of geometry.
//
// Following CGAL's Triangulation_data_structure pattern, these methods operate
// on topology independently of geometry.
//
impl<T, U, V, const D: usize> Tds<T, U, V, D>
where
    U: DataType,
    V: DataType,
{
    /// Assigns neighbor relationships between cells based on shared facets with semantic ordering.
    ///
    /// This method efficiently builds neighbor relationships by using the `facet_key_from_vertices`
    /// function to compute unique keys for facets. Two cells are considered neighbors if they share
    /// exactly one facet (which contains D vertices for a D-dimensional triangulation).
    ///
    /// **Note**: This is a purely combinatorial operation that does not perform any coordinate
    /// operations. It works entirely with vertex keys, cell keys, and topological relationships.
    ///
    /// **Internal use only**: This method rebuilds ALL neighbor pointers from scratch, which is
    /// inefficient for most use cases. For external use, prefer
    /// [`repair_neighbor_pointers`](crate::core::algorithms::incremental_insertion::repair_neighbor_pointers)
    /// which provides more efficient surgical reconstruction by only fixing broken neighbor pointers.
    ///
    /// # Errors
    ///
    /// Returns `TdsValidationError` if neighbor assignment fails due to inconsistent
    /// data structures or invalid facet sharing patterns.
    fn assign_neighbors(&mut self) -> Result<(), TdsValidationError> {
        use crate::core::facet::facet_key_from_vertices;

        // Build facet mapping with vertex index information using optimized collections
        // facet_key -> [(cell_key, vertex_index_opposite_to_facet)]
        type FacetInfo = (CellKey, usize);
        // Use saturating arithmetic to avoid potential overflow on adversarial inputs
        let cap = self.cells.len().saturating_mul(D.saturating_add(1));
        let mut facet_map: FastHashMap<u64, SmallBuffer<FacetInfo, 2>> =
            fast_hash_map_with_capacity(cap);

        for (cell_key, cell) in &self.cells {
            let vertices = self.get_cell_vertices(cell_key).map_err(|err| {
                TdsError::VertexKeyRetrievalFailed {
                    cell_id: cell.uuid(),
                    message: format!(
                        "Failed to retrieve vertex keys for cell during neighbor assignment: {err}"
                    ),
                }
            })?;

            let mut facet_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
                SmallBuffer::with_capacity(vertices.len().saturating_sub(1));
            for i in 0..vertices.len() {
                facet_vertices.clear();
                for (j, &key) in vertices.iter().enumerate() {
                    if j != i {
                        facet_vertices.push(key);
                    }
                }
                let facet_key = facet_key_from_vertices(&facet_vertices);
                let facet_entry = facet_map.entry(facet_key).or_default();
                // Detect degenerate case early: more than 2 cells sharing a facet
                // Note: Check happens before push, so len() reflects current sharing count
                if facet_entry.len() >= 2 {
                    return Err(TdsError::InconsistentDataStructure {
                        message: format!(
                            "Facet with key {} already shared by {} cells; cannot add cell {} (would violate 2-manifold property)",
                            facet_key,
                            facet_entry.len(),
                            cell.uuid()
                        ),
                    });
                }
                facet_entry.push((cell_key, i));
            }
        }

        // For each cell, build an ordered neighbor array where neighbors[i] is opposite vertices[i]
        let mut cell_neighbors: FastHashMap<
            CellKey,
            SmallBuffer<Option<CellKey>, MAX_PRACTICAL_DIMENSION_SIZE>,
        > = fast_hash_map_with_capacity(self.cells.len());

        // Initialize each cell with a SmallBuffer of None values (one per vertex)
        for (cell_key, cell) in &self.cells {
            let vertex_count = cell.number_of_vertices();
            if vertex_count > MAX_PRACTICAL_DIMENSION_SIZE {
                return Err(TdsError::InconsistentDataStructure {
                    message: format!(
                        "Cell {} has {} vertices, which exceeds MAX_PRACTICAL_DIMENSION_SIZE={}. \
                         This would overflow the neighbors buffer.",
                        cell.uuid(),
                        vertex_count,
                        MAX_PRACTICAL_DIMENSION_SIZE
                    ),
                });
            }
            let mut neighbors = SmallBuffer::with_capacity(vertex_count);
            neighbors.resize(vertex_count, None);
            cell_neighbors.insert(cell_key, neighbors);
        }

        // For each facet that is shared by exactly two cells, establish neighbor relationships
        // Note: >2 cells per facet already caught by early check during map build (above)
        for (_facet_key, facet_infos) in facet_map {
            if facet_infos.len() != 2 {
                continue;
            }

            let (cell_key1, vertex_index1) = facet_infos[0];
            let (cell_key2, vertex_index2) = facet_infos[1];

            // Set neighbors with semantic constraint: neighbors[i] is opposite vertices[i]
            cell_neighbors.get_mut(&cell_key1).ok_or_else(|| {
                TdsError::InconsistentDataStructure {
                    message: format!("Cell key {cell_key1:?} not found in cell neighbors map"),
                }
            })?[vertex_index1] = Some(cell_key2);

            cell_neighbors.get_mut(&cell_key2).ok_or_else(|| {
                TdsError::InconsistentDataStructure {
                    message: format!("Cell key {cell_key2:?} not found in cell neighbors map"),
                }
            })?[vertex_index2] = Some(cell_key1);
        }

        // Apply updates
        for (cell_key, neighbors) in &cell_neighbors {
            if let Some(cell) = self.cells.get_mut(*cell_key) {
                if neighbors.iter().all(Option::is_none) {
                    cell.neighbors = None;
                } else {
                    let mut neighbor_buffer = SmallBuffer::new();
                    neighbor_buffer.extend(neighbors.iter().copied());
                    cell.neighbors = Some(neighbor_buffer);
                }
            }
        }

        // Topology changed; invalidate caches.
        self.bump_generation();

        Ok(())
    }

    /// Returns an iterator over all cells in the triangulation.
    ///
    /// This method provides read-only access to the cells collection without
    /// exposing the underlying storage implementation. The iterator yields
    /// `(CellKey, &Cell)` pairs for each cell in the triangulation.
    ///
    /// For direct key-based access, use [`get_cell`](Self::get_cell).
    ///
    /// # Returns
    ///
    /// An iterator over `(CellKey, &Cell<T, U, V, D>)` pairs.
    ///
    /// # Example
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// for (cell_key, cell) in dt.cells() {
    ///     println!("Cell {:?} has {} vertices", cell_key, cell.number_of_vertices());
    /// }
    /// ```
    pub fn cells(&self) -> impl Iterator<Item = (CellKey, &Cell<T, U, V, D>)> {
        self.cells.iter()
    }

    /// Returns an iterator over all cell values (without keys) in the triangulation.
    ///
    /// This is a convenience method that simplifies the common pattern of iterating over
    /// `cells().map(|(_, cell)| cell)`. It provides read-only access to cell objects
    /// when you don't need the cell keys.
    ///
    /// # Returns
    ///
    /// An iterator over `&Cell<T, U, V, D>` references.
    ///
    /// # Example
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// for (_, cell) in dt.cells() {
    ///     println!("Cell has {} vertices", cell.number_of_vertices());
    /// }
    /// ```
    pub fn cells_values(&self) -> impl Iterator<Item = &Cell<T, U, V, D>> {
        self.cells.values()
    }

    // REMOVED: last_triangulation_statistics()
    // This method was part of the deprecated Bowyer-Watson architecture.
    // Statistics tracking will be reimplemented for incremental insertion in a future release.

    /// Returns an iterator over all vertices in the triangulation.
    ///
    /// This method provides read-only access to the vertices collection without
    /// exposing the underlying storage implementation. The iterator yields
    /// `(VertexKey, &Vertex)` pairs for each vertex in the triangulation.
    ///
    /// For direct key-based access, use [`get_vertex_by_key`](Self::get_vertex_by_key).
    ///
    /// # Returns
    ///
    /// An iterator over `(VertexKey, &Vertex<T, U, D>)` pairs.
    ///
    /// # Example
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.5, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// for (vertex_key, vertex) in dt.vertices() {
    ///     println!("Vertex {:?} at {:?}", vertex_key, vertex.point());
    /// }
    /// ```
    pub fn vertices(&self) -> impl Iterator<Item = (VertexKey, &Vertex<T, U, D>)> {
        self.vertices.iter()
    }

    /// Returns an iterator over all vertex keys in the triangulation.
    ///
    /// # Returns
    ///
    /// An iterator over `VertexKey` values.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let keys: Vec<_> = dt.tds().vertex_keys().collect();
    /// assert_eq!(keys.len(), 3);
    /// ```
    pub fn vertex_keys(&self) -> impl Iterator<Item = VertexKey> + '_ {
        self.vertices.keys()
    }

    /// Returns an iterator over all cell keys in the triangulation.
    ///
    /// # Returns
    ///
    /// An iterator over `CellKey` values.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let keys: Vec<_> = dt.tds().cell_keys().collect();
    /// assert_eq!(keys.len(), 1);
    /// ```
    pub fn cell_keys(&self) -> impl Iterator<Item = CellKey> + '_ {
        self.cells.keys()
    }

    /// Returns a reference to a cell by its key.
    ///
    /// # Returns
    ///
    /// `Some(&Cell)` if the key exists, `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tds = dt.tds();
    /// let cell_key = tds.cell_keys().next().unwrap();
    /// let cell = tds.get_cell(cell_key).unwrap();
    /// assert_eq!(cell.number_of_vertices(), 3);
    /// ```
    #[must_use]
    pub fn get_cell(&self, key: CellKey) -> Option<&Cell<T, U, V, D>> {
        self.cells.get(key)
    }

    /// Checks if a cell key exists in the triangulation.
    ///
    /// # Returns
    ///
    /// `true` if the key exists, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tds = dt.tds();
    /// let cell_key = tds.cell_keys().next().unwrap();
    /// assert!(tds.contains_cell(cell_key));
    /// ```
    #[must_use]
    pub fn contains_cell(&self, key: CellKey) -> bool {
        self.cells.contains_key(key)
    }

    /// The function returns the number of vertices in the triangulation
    /// data structure.
    ///
    /// # Returns
    ///
    /// The number of [Vertex] objects in the [Tds].
    ///
    /// # Examples
    ///
    /// Count vertices in an empty triangulation:
    ///
    /// ```
    /// use delaunay::core::triangulation_data_structure::Tds;
    ///
    /// let tds = Tds::<f64, (), (), 3>::empty();
    /// assert_eq!(tds.number_of_vertices(), 0);
    /// ```
    ///
    /// Count vertices after adding them:
    ///
    /// ```no_run
    /// use delaunay::prelude::query::*;
    ///
    /// let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();
    /// let vertex1: Vertex<f64, (), 3> = vertex!([1.0, 2.0, 3.0]);
    /// let vertex2: Vertex<f64, (), 3> = vertex!([4.0, 5.0, 6.0]);
    ///
    /// dt.insert(vertex1).unwrap();
    /// assert_eq!(dt.number_of_vertices(), 1);
    ///
    /// dt.insert(vertex2).unwrap();
    /// assert_eq!(dt.number_of_vertices(), 2);
    /// ```
    ///
    /// Count vertices initialized from points:
    ///
    /// ```
    /// use delaunay::prelude::query::*;
    ///
    /// let points = [
    ///     Point::new([0.0, 0.0, 0.0]),
    ///     Point::new([1.0, 0.0, 0.0]),
    ///     Point::new([0.0, 1.0, 0.0]),
    ///     Point::new([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let vertices = Vertex::from_points(&points);
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// assert_eq!(dt.number_of_vertices(), 4);
    /// ```
    #[must_use]
    pub fn number_of_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// The `dim` function returns the dimensionality of the [Tds].
    ///
    /// # Returns
    ///
    /// The `dim` function returns the minimum value between the number of
    /// vertices minus one and the value of `D` as an [i32].
    ///
    /// # Examples
    ///
    /// Dimension of an empty triangulation:
    ///
    /// ```
    /// use delaunay::core::triangulation_data_structure::Tds;
    /// use delaunay::geometry::point::Point;
    /// use delaunay::geometry::traits::coordinate::Coordinate;
    ///
    /// let tds = Tds::<f64, (), (), 3>::empty();
    /// assert_eq!(tds.dim(), -1); // Empty triangulation
    /// ```
    ///
    /// Dimension progression as vertices are added:
    ///
    /// ```no_run
    /// use delaunay::prelude::query::*;
    ///
    /// let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();
    ///
    /// // Start empty
    /// assert_eq!(dt.dim(), -1);
    ///
    /// // Add vertices incrementally
    /// let vertex1: Vertex<f64, (), 3> = vertex!([0.0, 0.0, 0.0]);
    /// dt.insert(vertex1).unwrap();
    /// assert_eq!(dt.dim(), 0);
    ///
    /// let vertex2: Vertex<f64, (), 3> = vertex!([1.0, 0.0, 0.0]);
    /// dt.insert(vertex2).unwrap();
    /// assert_eq!(dt.dim(), 1);
    ///
    /// let vertex3: Vertex<f64, (), 3> = vertex!([0.0, 1.0, 0.0]);
    /// dt.insert(vertex3).unwrap();
    /// assert_eq!(dt.dim(), 2);
    ///
    /// let vertex4: Vertex<f64, (), 3> = vertex!([0.0, 0.0, 1.0]);
    /// dt.insert(vertex4).unwrap();
    /// assert_eq!(dt.number_of_vertices(), 4);
    /// assert_eq!(dt.dim(), 3);
    /// ```
    ///
    /// Different dimensional triangulations:
    ///
    /// ```
    /// use delaunay::prelude::query::*;
    ///
    /// // 2D triangulation
    /// let points_2d = [
    ///     Point::new([0.0, 0.0]),
    ///     Point::new([1.0, 0.0]),
    ///     Point::new([0.5, 1.0]),
    /// ];
    /// let vertices_2d = Vertex::from_points(&points_2d);
    /// let dt_2d = DelaunayTriangulation::new(&vertices_2d).unwrap();
    /// assert_eq!(dt_2d.dim(), 2);
    ///
    /// // 4D triangulation with 5 vertices (minimum for 4D simplex)
    /// let points_4d = [
    ///     Point::new([0.0, 0.0, 0.0, 0.0]),
    ///     Point::new([1.0, 0.0, 0.0, 0.0]),
    ///     Point::new([0.0, 1.0, 0.0, 0.0]),
    ///     Point::new([0.0, 0.0, 1.0, 0.0]),
    ///     Point::new([0.0, 0.0, 0.0, 1.0]),
    /// ];
    /// let vertices_4d = Vertex::from_points(&points_4d);
    /// let dt_4d = DelaunayTriangulation::new(&vertices_4d).unwrap();
    /// assert_eq!(dt_4d.dim(), 4);
    /// ```
    #[must_use]
    pub fn dim(&self) -> i32 {
        let nv = self.number_of_vertices();
        // Convert to i32 first, then subtract to handle empty case (0 - 1 = -1)
        let nv_i32 = i32::try_from(nv).unwrap_or(i32::MAX);
        let d_i32 = i32::try_from(D).unwrap_or(i32::MAX);
        nv_i32.saturating_sub(1).min(d_i32)
    }

    /// The function `number_of_cells` returns the number of cells in the [Tds].
    ///
    /// # Returns
    ///
    /// The number of [Cell]s in the [Tds].
    ///
    /// # Examples
    ///
    /// Count cells in a newly created triangulation:
    ///
    /// ```
    /// use delaunay::prelude::query::*;
    ///
    /// let points = [
    ///     Point::new([0.0, 0.0, 0.0]),
    ///     Point::new([1.0, 0.0, 0.0]),
    ///     Point::new([0.0, 1.0, 0.0]),
    ///     Point::new([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let vertices = Vertex::from_points(&points);
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// assert_eq!(dt.number_of_cells(), 1); // Cells are automatically created via triangulation
    /// ```
    ///
    /// Count cells after triangulation:
    ///
    /// ```
    /// use delaunay::prelude::query::*;
    ///
    /// let points = [
    ///     Point::new([0.0, 0.0, 0.0]),
    ///     Point::new([1.0, 0.0, 0.0]),
    ///     Point::new([0.0, 1.0, 0.0]),
    ///     Point::new([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let vertices = Vertex::from_points(&points);
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// assert_eq!(dt.number_of_cells(), 1); // One tetrahedron for 4 points in 3D
    /// ```
    ///
    /// Empty triangulation has no cells:
    ///
    /// ```
    /// use delaunay::core::triangulation_data_structure::Tds;
    ///
    /// let tds = Tds::<f64, (), (), 3>::empty();
    /// assert_eq!(tds.number_of_cells(), 0); // No cells for empty input
    /// ```
    #[must_use]
    pub fn number_of_cells(&self) -> usize {
        self.cells.len()
    }

    /// Increments the generation counter to invalidate dependent caches.
    ///
    /// This method should be called whenever the triangulation structure is modified
    /// (vertices added, cells created/removed, etc.). It uses relaxed memory ordering
    /// since it's just an invalidation counter.
    ///
    /// # Thread Safety
    ///
    /// This method is thread-safe due to the use of `Arc<AtomicU64>`.
    #[inline]
    fn bump_generation(&self) {
        // Relaxed is fine for an invalidation counter
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    /// Gets the current generation value.
    ///
    /// This can be used by external code to detect when the triangulation has changed.
    /// The generation counter is incremented on any structural modification.
    ///
    /// # Returns
    ///
    /// The current generation counter value.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::core::triangulation_data_structure::Tds;
    ///
    /// let tds: Tds<f64, (), (), 2> = Tds::empty();
    /// assert_eq!(tds.generation(), 0);
    /// ```
    #[inline]
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Relaxed)
    }

    // =========================================================================
    // QUERY OPERATIONS
    // =========================================================================

    /// Returns a mutable reference to the internal cells storage.
    ///
    /// # ⚠️ Warning: Dangerous Internal API
    ///
    /// This method exposes the concrete storage backend and **WILL BREAK TRIANGULATION INVARIANTS**
    /// if used incorrectly. It is intended **ONLY** for:
    /// - Internal crate implementation
    /// - Performance benchmarks that need to violate invariants deliberately
    /// - Integration tests validating storage backend behavior
    ///
    /// **DO NOT** use this in production code. Modifying cells through this method bypasses
    /// all safety checks and can leave the triangulation in an inconsistent state.
    ///
    /// # Returns
    ///
    /// A mutable reference to the storage map containing all cells.
    #[doc(hidden)]
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Dangerous internal API used only in tests to intentionally violate invariants",
        )
    )]
    pub(crate) const fn cells_mut(&mut self) -> &mut StorageMap<CellKey, Cell<T, U, V, D>> {
        &mut self.cells
    }

    /// Atomically inserts a vertex and creates the UUID-to-key mapping.
    ///
    /// This method ensures that both the vertex insertion and UUID mapping are
    /// performed together, maintaining data structure invariants.
    ///
    /// **⚠️ INTERNAL API WARNING**: This method bypasses atomicity guarantees for topology
    /// assignment operations (`assign_neighbors()` and `assign_incident_cells()`). It only
    /// ensures atomic vertex insertion and UUID mapping. If you need full atomicity including
    /// topology assignment, use `insert_vertex_with_topology_assignment()` instead.
    ///
    /// **Note:** This method does NOT check for duplicate coordinates. It only checks
    /// for UUID uniqueness. Duplicate coordinate checking is performed at a higher layer
    /// in `Triangulation::try_insert_impl()` before calling this method.
    ///
    /// # Arguments
    ///
    /// * `vertex` - The vertex to insert
    ///
    /// # Returns
    ///
    /// The `VertexKey` that can be used to access the inserted vertex.
    ///
    /// # Errors
    ///
    /// Returns [`TdsConstructionError::DuplicateUuid`] if a vertex with the
    /// same UUID already exists in the triangulation.
    ///
    /// # Examples
    ///
    ///
    /// See the unit tests for usage examples of this pub(crate) method.
    pub(crate) fn insert_vertex_with_mapping(
        &mut self,
        vertex: Vertex<T, U, D>,
    ) -> Result<VertexKey, TdsConstructionError> {
        let vertex_uuid = vertex.uuid();

        // Use Entry API for atomic check-and-insert
        match self.uuid_to_vertex_key.entry(vertex_uuid) {
            Entry::Occupied(_) => Err(TdsConstructionError::DuplicateUuid {
                entity: EntityKind::Vertex,
                uuid: vertex_uuid,
            }),
            Entry::Vacant(e) => {
                let vertex_key = self.vertices.insert(vertex);
                e.insert(vertex_key);
                // Topology changed; invalidate caches.
                self.bump_generation();
                Ok(vertex_key)
            }
        }
    }

    /// Atomically inserts a cell and creates the UUID-to-key mapping.
    ///
    /// This method ensures that both the cell insertion and UUID mapping are
    /// performed together, maintaining data structure invariants. This is preferred
    /// over separate `cells_mut().insert()` + `uuid_to_cell_key.insert()` calls
    /// which can leave the data structure in an inconsistent state if interrupted.
    ///
    /// # Arguments
    ///
    /// * `cell` - The cell to insert
    ///
    /// # Returns
    ///
    /// The `CellKey` that can be used to access the inserted cell.
    ///
    /// # Errors
    ///
    /// Returns [`TdsConstructionError::DuplicateUuid`] if a cell with the
    /// same UUID already exists in the triangulation.
    ///
    /// # Examples
    ///
    ///
    /// See the unit tests for usage examples of this pub(crate) method.
    pub(crate) fn insert_cell_with_mapping(
        &mut self,
        cell: Cell<T, U, V, D>,
    ) -> Result<CellKey, TdsConstructionError> {
        // Phase 3A: Validate structural invariants using vertices
        debug_assert_eq!(
            cell.number_of_vertices(),
            D + 1,
            "Cell should have exactly D+1 vertices for quick failure in dev"
        );
        if cell.number_of_vertices() != D + 1 {
            return Err(TdsConstructionError::ValidationError(
                TdsError::InconsistentDataStructure {
                    message: format!(
                        "Cell must have exactly {} vertices for {}-dimensional simplex, but has {}",
                        D + 1,
                        D,
                        cell.number_of_vertices()
                    ),
                },
            ));
        }

        // Phase 3A: Verify all vertex keys exist in the triangulation
        for &vkey in cell.vertices() {
            if !self.vertices.contains_key(vkey) {
                return Err(TdsConstructionError::ValidationError(
                    TdsError::InconsistentDataStructure {
                        message: format!(
                            "Cell references vertex key {vkey:?} that does not exist in the triangulation"
                        ),
                    },
                ));
            }
        }

        let cell_uuid = cell.uuid();

        // Use Entry API for atomic check-and-insert
        match self.uuid_to_cell_key.entry(cell_uuid) {
            Entry::Occupied(_) => Err(TdsConstructionError::DuplicateUuid {
                entity: EntityKind::Cell,
                uuid: cell_uuid,
            }),
            Entry::Vacant(e) => {
                let cell_key = self.cells.insert(cell);
                e.insert(cell_key);
                // Topology changed; invalidate caches.
                self.bump_generation();
                Ok(cell_key)
            }
        }
    }

    /// Gets vertex keys for a cell.
    ///
    /// **Phase 3A**: Cells now store `VertexKey` directly. This method performs O(D) validation
    /// and copying of keys for the requested cell and returns a stack-friendly buffer.
    ///
    /// This method provides:
    /// - O(1) cell lookup via storage map key
    /// - O(D) validation that all vertex keys exist in the triangulation
    /// - Direct key access without UUID→Key lookups (Phase 3A completed)
    /// - Stack-allocated buffer for D ≤ 7 to avoid heap allocation
    ///
    /// # Arguments
    ///
    /// * `cell_key` - The key of the cell whose vertex keys we need
    ///
    /// # Returns
    ///
    /// A `Result` containing a `VertexKeyBuffer` if the cell exists and all vertices are valid,
    /// or a `TdsValidationError` if the cell doesn't exist or vertices are missing.
    ///
    /// # Errors
    ///
    /// Returns a `TdsValidationError` if:
    /// - The cell with the given key doesn't exist
    /// - A vertex key from the cell doesn't exist in the vertex storage (TDS corruption)
    ///
    /// # Performance
    ///
    /// This uses direct storage map access with O(1) key lookup for the cell and O(D)
    /// validation for vertex keys. Uses stack-allocated buffer for D ≤ 7 to avoid heap
    /// allocation in the hot path.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tds = dt.tds();
    /// let cell_key = tds.cell_keys().next().unwrap();
    /// let keys = tds.get_cell_vertices(cell_key).unwrap();
    /// assert_eq!(keys.len(), 3);
    /// ```
    #[inline]
    pub fn get_cell_vertices(
        &self,
        cell_key: CellKey,
    ) -> Result<VertexKeyBuffer, TdsValidationError> {
        let cell = self
            .cells
            .get(cell_key)
            .ok_or_else(|| TdsError::InconsistentDataStructure {
                message: format!("Cell key {cell_key:?} not found in cells storage map"),
            })?;

        // Phase 3A: Cell now stores vertex keys directly
        // Validate and collect keys in one pass to avoid redundant iteration
        let cell_vertices = cell.vertices();
        let mut keys = VertexKeyBuffer::with_capacity(cell_vertices.len());
        for (idx, &vertex_key) in cell_vertices.iter().enumerate() {
            if !self.vertices.contains_key(vertex_key) {
                return Err(TdsError::InconsistentDataStructure {
                    message: format!(
                        "Cell {} (key {cell_key:?}) references non-existent vertex key {vertex_key:?} at position {idx}",
                        cell.uuid()
                    ),
                });
            }
            keys.push(vertex_key);
        }
        Ok(keys)
    }

    /// Helper function to get a cell key from a cell UUID using the optimized UUID→Key mapping.
    ///
    /// # Arguments
    ///
    /// * `cell_uuid` - The UUID of the cell to look up
    ///
    /// # Returns
    ///
    /// An `Option<CellKey>` if the cell is found, `None` otherwise.
    ///
    /// # Performance
    ///
    /// This uses `FastHashMap` for O(1) UUID→Key lookups.
    ///
    /// # Examples
    ///
    /// Successfully finding a cell key from a UUID:
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// // Create a triangulation with some vertices
    /// let vertices = [
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tds = dt.tds();
    ///
    /// // Get the first cell and its UUID
    /// let (cell_key, cell) = tds.cells().next().unwrap();
    /// let cell_uuid = cell.uuid();
    ///
    /// // Use the helper function to find the cell key from its UUID
    /// let found_key = tds.cell_key_from_uuid(&cell_uuid);
    /// assert_eq!(found_key, Some(cell_key));
    /// ```
    ///
    /// Returns `None` for non-existent UUID:
    ///
    /// ```
    /// use delaunay::core::triangulation_data_structure::Tds;
    /// use uuid::Uuid;
    ///
    /// let tds: Tds<f64, (), (), 3> = Tds::empty();
    /// let random_uuid = Uuid::new_v4();
    ///
    /// let result = tds.cell_key_from_uuid(&random_uuid);
    /// assert_eq!(result, None);
    /// ```
    #[inline]
    #[must_use]
    pub fn cell_key_from_uuid(&self, cell_uuid: &Uuid) -> Option<CellKey> {
        self.uuid_to_cell_key.get(cell_uuid).copied()
    }

    /// Helper function to get a vertex key from a vertex UUID using the optimized UUID→Key mapping.
    /// This provides efficient UUID→Key lookups in hot paths.
    ///
    /// # Arguments
    ///
    /// * `vertex_uuid` - The UUID of the vertex to look up
    ///
    /// # Returns
    ///
    /// An `Option<VertexKey>` if the vertex is found, `None` otherwise.
    ///
    /// # Performance
    ///
    /// This uses `FastHashMap` for O(1) UUID→Key lookups.
    ///
    /// # Examples
    ///
    /// Successfully finding a vertex key from a UUID:
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// // Create a triangulation with some vertices
    /// let vertices = [
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tds = dt.tds();
    ///
    /// // Get the first vertex and its UUID
    /// let (vertex_key, vertex) = tds.vertices().next().unwrap();
    /// let vertex_uuid = vertex.uuid();
    ///
    /// // Use the helper function to find the vertex key from its UUID
    /// let found_key = tds.vertex_key_from_uuid(&vertex_uuid);
    /// assert_eq!(found_key, Some(vertex_key));
    /// ```
    ///
    /// Returns `None` for non-existent UUID:
    ///
    /// ```
    /// use delaunay::core::triangulation_data_structure::Tds;
    /// use uuid::Uuid;
    ///
    /// let tds: Tds<f64, (), (), 3> = Tds::empty();
    /// let random_uuid = Uuid::new_v4();
    ///
    /// let result = tds.vertex_key_from_uuid(&random_uuid);
    /// assert_eq!(result, None);
    /// ```
    #[inline]
    #[must_use]
    pub fn vertex_key_from_uuid(&self, vertex_uuid: &Uuid) -> Option<VertexKey> {
        self.uuid_to_vertex_key.get(vertex_uuid).copied()
    }

    /// Helper function to get a cell UUID from a cell key using direct `storage map` access.
    /// This is the reverse of `cell_key_from_uuid()` for the less common Key→UUID direction.
    ///
    /// # Arguments
    ///
    /// * `cell_key` - The key of the cell to look up
    ///
    /// # Returns
    ///
    /// An `Option<Uuid>` if the cell is found, `None` otherwise.
    ///
    /// # Performance
    ///
    /// This uses direct `storage map` indexing for O(1) Key→UUID lookups.
    ///
    /// # Examples
    ///
    /// Successfully getting a UUID from a cell key:
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// // Create a triangulation with some vertices
    /// let vertices = [
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tds = dt.tds();
    ///
    /// // Get the first cell key and expected UUID
    /// let (cell_key, cell) = tds.cells().next().unwrap();
    /// let expected_uuid = cell.uuid();
    ///
    /// // Use the helper function to get UUID from the cell key
    /// let found_uuid = tds.cell_uuid_from_key(cell_key);
    /// assert_eq!(found_uuid, Some(expected_uuid));
    /// ```
    ///
    /// Round-trip conversion between UUID and key:
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// // Create a triangulation with some vertices
    /// let vertices = [
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tds = dt.tds();
    ///
    /// // Get the first cell's UUID
    /// let (_, cell) = tds.cells().next().unwrap();
    /// let original_uuid = cell.uuid();
    ///
    /// // Convert UUID to key, then key back to UUID
    /// let cell_key = tds.cell_key_from_uuid(&original_uuid).unwrap();
    /// let round_trip_uuid = tds.cell_uuid_from_key(cell_key).unwrap();
    /// assert_eq!(original_uuid, round_trip_uuid);
    /// ```
    #[inline]
    #[must_use]
    pub fn cell_uuid_from_key(&self, cell_key: CellKey) -> Option<Uuid> {
        self.cells.get(cell_key).map(super::cell::Cell::uuid)
    }

    /// Helper function to get a vertex UUID from a vertex key using direct `storage map` access.
    /// This is the reverse of `vertex_key_from_uuid()` for the less common Key→UUID direction.
    ///
    /// # Arguments
    ///
    /// * `vertex_key` - The key of the vertex to look up
    ///
    /// # Returns
    ///
    /// An `Option<Uuid>` if the vertex is found, `None` otherwise.
    ///
    /// # Performance
    ///
    /// This uses direct `storage map` indexing for O(1) Key→UUID lookups.
    ///
    /// # Examples
    ///
    /// Successfully getting a UUID from a vertex key:
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// // Create a triangulation with some vertices
    /// let vertices = [
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tds = dt.tds();
    ///
    /// // Get the first vertex key and expected UUID
    /// let (vertex_key, vertex) = tds.vertices().next().unwrap();
    /// let expected_uuid = vertex.uuid();
    ///
    /// // Use the helper function to get UUID from the vertex key
    /// let found_uuid = tds.vertex_uuid_from_key(vertex_key);
    /// assert_eq!(found_uuid, Some(expected_uuid));
    /// ```
    ///
    /// Round-trip conversion between UUID and key:
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// // Create a triangulation with some vertices
    /// let vertices = [
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tds = dt.tds();
    ///
    /// // Get the first vertex's UUID
    /// let (_, vertex) = tds.vertices().next().unwrap();
    /// let original_uuid = vertex.uuid();
    ///
    /// // Convert UUID to key, then key back to UUID
    /// let vertex_key = tds.vertex_key_from_uuid(&original_uuid).unwrap();
    /// let round_trip_uuid = tds.vertex_uuid_from_key(vertex_key).unwrap();
    /// assert_eq!(original_uuid, round_trip_uuid);
    /// ```
    #[inline]
    #[must_use]
    pub fn vertex_uuid_from_key(&self, vertex_key: VertexKey) -> Option<Uuid> {
        self.vertices
            .get(vertex_key)
            .map(super::vertex::Vertex::uuid)
    }

    // =========================================================================
    // KEY-BASED ACCESS METHODS (Phase 2 Optimization)
    // =========================================================================
    // These methods work directly with keys to avoid UUID lookups in hot paths.
    // They complement the existing UUID-based methods for internal algorithm use.

    /// Gets a mutable reference to a cell directly by its key.
    ///
    /// This method provides direct mutable access to cells, similar to [`get_vertex_by_key_mut()`](Self::get_vertex_by_key_mut).
    /// While this allows modifying cell data fields, callers should use safe topology setter APIs
    /// like [`set_neighbors_by_key()`](Self::set_neighbors_by_key) when modifying neighbor relationships.
    ///
    /// # Arguments
    ///
    /// * `cell_key` - The key of the cell to retrieve
    ///
    /// # Returns
    ///
    /// An `Option` containing a mutable reference to the cell if it exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let mut tds = dt.tds().clone();
    /// let cell_key = tds.cell_keys().next().unwrap();
    /// let cell = tds.get_cell_by_key_mut(cell_key).unwrap();
    /// assert_eq!(cell.number_of_vertices(), 3);
    /// ```
    #[inline]
    #[must_use]
    pub fn get_cell_by_key_mut(&mut self, cell_key: CellKey) -> Option<&mut Cell<T, U, V, D>> {
        self.cells.get_mut(cell_key)
    }

    /// Gets a vertex directly by its key without UUID lookup.
    ///
    /// This is a key-based optimization of the UUID-based vertex access.
    /// Use this method in internal algorithms to avoid UUID→Key conversion overhead.
    ///
    /// # Arguments
    ///
    /// * `vertex_key` - The key of the vertex to retrieve
    ///
    /// # Returns
    ///
    /// An `Option` containing a reference to the vertex if it exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tds = dt.tds();
    /// let vertex_key = tds.vertex_keys().next().unwrap();
    /// assert!(tds.get_vertex_by_key(vertex_key).is_some());
    /// ```
    #[inline]
    #[must_use]
    pub fn get_vertex_by_key(&self, vertex_key: VertexKey) -> Option<&Vertex<T, U, D>> {
        self.vertices.get(vertex_key)
    }

    /// Gets a mutable reference to a vertex directly by its key.
    ///
    /// # Arguments
    ///
    /// * `vertex_key` - The key of the vertex to retrieve
    ///
    /// # Returns
    ///
    /// An `Option` containing a mutable reference to the vertex if it exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let mut tds = dt.tds().clone();
    /// let vertex_key = tds.vertex_keys().next().unwrap();
    /// assert!(tds.get_vertex_by_key_mut(vertex_key).is_some());
    /// ```
    #[inline]
    #[must_use]
    pub fn get_vertex_by_key_mut(&mut self, vertex_key: VertexKey) -> Option<&mut Vertex<T, U, D>> {
        self.vertices.get_mut(vertex_key)
    }

    /// Checks if a cell key exists in the triangulation.
    ///
    /// # Arguments
    ///
    /// * `cell_key` - The key to check
    ///
    /// # Returns
    ///
    /// `true` if the cell exists, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tds = dt.tds();
    /// let cell_key = tds.cell_keys().next().unwrap();
    /// assert!(tds.contains_cell_key(cell_key));
    /// ```
    #[inline]
    #[must_use]
    pub fn contains_cell_key(&self, cell_key: CellKey) -> bool {
        self.cells.contains_key(cell_key)
    }

    /// Checks if a vertex key exists in the triangulation.
    ///
    /// # Arguments
    ///
    /// * `vertex_key` - The key to check
    ///
    /// # Returns
    ///
    /// `true` if the vertex exists, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tds = dt.tds();
    /// let vertex_key = tds.vertex_keys().next().unwrap();
    /// assert!(tds.contains_vertex_key(vertex_key));
    /// ```
    #[inline]
    #[must_use]
    pub fn contains_vertex_key(&self, vertex_key: VertexKey) -> bool {
        self.vertices.contains_key(vertex_key)
    }

    /// Removes a cell by its key, updating all necessary mappings.
    ///
    /// This is a key-based version of cell removal that avoids UUID lookups.
    ///
    /// # Safety Warning
    ///
    /// This method only removes the cell and updates the UUID→Key mapping.
    /// It does NOT maintain topology consistency. The caller MUST:
    /// 1. Call `assign_neighbors()` to rebuild neighbor relationships
    /// 2. Call `assign_incident_cells()` to update vertex-cell associations
    ///
    /// Failure to do so will leave the triangulation in an inconsistent state.
    ///
    /// # Arguments
    ///
    /// * `cell_key` - The key of the cell to remove
    ///
    /// # Returns
    ///
    /// The removed cell if it existed, `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let mut tds = dt.tds().clone();
    /// let cell_key = tds.cell_keys().next().unwrap();
    /// assert!(tds.remove_cell_by_key(cell_key).is_some());
    /// assert_eq!(tds.number_of_cells(), 0);
    /// ```
    pub fn remove_cell_by_key(&mut self, cell_key: CellKey) -> Option<Cell<T, U, V, D>> {
        if let Some(removed_cell) = self.cells.remove(cell_key) {
            // Also remove from UUID mapping
            self.uuid_to_cell_key.remove(&removed_cell.uuid());
            // Topology changed; invalidate caches
            self.bump_generation();
            Some(removed_cell)
        } else {
            None
        }
    }

    /// Removes multiple cells by their keys in a batch operation.
    ///
    /// This method performs a **local** topology update:
    /// - Removes the requested cells (and their UUID→Key mappings)
    /// - Clears neighbor back-references in adjacent surviving cells so no neighbor points at a removed key
    /// - Repairs `Vertex::incident_cell` for vertices that previously pointed at a removed cell
    ///
    /// It does **not** attempt to retriangulate the cavity created by the removals.
    ///
    /// # Performance
    ///
    /// When neighbor pointers are present and mutually consistent, this touches only the
    /// boundary of the removed region:
    /// - Time: typically `O(#removed_cells × (D+1)^2)`
    /// - Space: `O(#removed_cells × (D+1))` for temporary removal metadata
    ///
    /// In degraded states (e.g., after unsafe mutation where neighbor pointers are missing),
    /// it may fall back to a conservative scan to find replacement incident cells.
    ///
    /// # Arguments
    ///
    /// * `cell_keys` - The keys of cells to remove
    ///
    /// # Returns
    ///
    /// The number of cells successfully removed.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let mut tds = dt.tds().clone();
    /// let cell_key = tds.cell_keys().next().unwrap();
    /// let removed = tds.remove_cells_by_keys(&[cell_key]);
    /// assert_eq!(removed, 1);
    /// assert_eq!(tds.number_of_cells(), 0);
    /// ```
    pub fn remove_cells_by_keys(&mut self, cell_keys: &[CellKey]) -> usize {
        if cell_keys.is_empty() {
            return 0;
        }

        // Build a set for O(1) membership tests.
        let cells_to_remove: CellKeySet = cell_keys.iter().copied().collect();

        // 1) Clear neighbor back-references in surviving cells and collect candidate incidence.
        let (affected_vertices, candidate_incident) = self
            .collect_removal_frontier_and_clear_neighbor_back_references(
                cell_keys,
                &cells_to_remove,
            );

        // 2) Remove the cells and update UUID mappings.
        let removed_count = self.remove_cells_and_update_uuid_mappings(cell_keys);
        if removed_count == 0 {
            return 0;
        }

        // 3) Repair `incident_cell` pointers for vertices that referenced removed cells.
        self.repair_incident_cells_after_cell_removal(
            &affected_vertices,
            &cells_to_remove,
            &candidate_incident,
        );

        // Bump generation once for all removals (neighbors + incidence + cell storage).
        self.bump_generation();

        removed_count
    }

    /// Repairs locally degenerate cells by removing them and clearing dangling references
    /// (neighbor back-references + `incident_cell` pointers).
    ///
    /// This is a narrowly scoped repair primitive intended for test-only usage
    /// (including bench-feature tests).
    ///
    /// A cell is treated as degenerate if it:
    /// - fails basic per-cell validity (`Cell::is_valid()`),
    /// - references a missing vertex key, or
    /// - contains a neighbor pointer to a missing cell key.
    ///
    /// This method does **not** retriangulate cavities, insert new cells, or attempt to repair
    /// geometric degeneracy. It only removes cells and relies on [`Tds::remove_cells_by_keys`]
    /// to clear neighbor back-references and repair `incident_cell` pointers.
    ///
    /// Returns the number of cells removed.
    #[cfg(test)]
    pub(crate) fn repair_degenerate_cells(&mut self) -> usize {
        if self.cells.is_empty() {
            return 0;
        }

        // Collect keys first (cannot mutate while iterating).
        let mut to_remove: Vec<CellKey> = Vec::new();

        for (cell_key, cell) in self.cells() {
            if cell.is_valid().is_err() {
                to_remove.push(cell_key);
                continue;
            }

            if cell
                .vertices()
                .iter()
                .any(|&vkey| !self.vertices.contains_key(vkey))
            {
                to_remove.push(cell_key);
                continue;
            }

            if cell.neighbors().is_some_and(|neighbors| {
                neighbors
                    .iter()
                    .flatten()
                    .any(|&neighbor_key| !self.cells.contains_key(neighbor_key))
            }) {
                to_remove.push(cell_key);
            }
        }

        if to_remove.is_empty() {
            return 0;
        }

        self.remove_cells_by_keys(&to_remove)
    }

    fn collect_removal_frontier_and_clear_neighbor_back_references(
        &mut self,
        cell_keys: &[CellKey],
        cells_to_remove: &CellKeySet,
    ) -> (VertexKeySet, FastHashMap<VertexKey, CellKey>) {
        let mut affected_vertices: VertexKeySet = VertexKeySet::default();
        let mut candidate_incident: FastHashMap<VertexKey, CellKey> =
            fast_hash_map_with_capacity(cell_keys.len().saturating_mul(D.saturating_add(1)));

        for &cell_key in cell_keys {
            let Some((vertices, neighbors)) = self.cells.get(cell_key).map(|cell| {
                let mut vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
                    SmallBuffer::with_capacity(cell.number_of_vertices());
                vertices.extend(cell.vertices().iter().copied());

                let mut neighbors: SmallBuffer<Option<CellKey>, MAX_PRACTICAL_DIMENSION_SIZE> =
                    SmallBuffer::with_capacity(vertices.len());
                neighbors.resize(vertices.len(), None);

                if let Some(cell_neighbors) = cell.neighbors() {
                    for (slot, neighbor_opt) in neighbors.iter_mut().zip(cell_neighbors.iter()) {
                        *slot = *neighbor_opt;
                    }
                }

                (vertices, neighbors)
            }) else {
                continue;
            };

            for &vk in &vertices {
                affected_vertices.insert(vk);
            }

            for (facet_idx, neighbor_key_opt) in neighbors.iter().enumerate() {
                let Some(neighbor_key) = neighbor_key_opt else {
                    continue;
                };

                if cells_to_remove.contains(neighbor_key) {
                    continue; // neighbor is also being removed
                }

                // The neighbor across facet_idx shares the facet consisting of all vertices
                // except vertices[facet_idx].
                for (vertex_index, &vkey) in vertices.iter().enumerate() {
                    if vertex_index == facet_idx {
                        continue;
                    }
                    candidate_incident.entry(vkey).or_insert(*neighbor_key);
                }

                let Some(neighbor_cell) = self.cells.get_mut(*neighbor_key) else {
                    continue;
                };
                let Some(neighbors_buf) = neighbor_cell.neighbors.as_mut() else {
                    continue;
                };

                // Clear the back-reference in the neighbor cell's neighbor buffer.
                for slot in neighbors_buf.iter_mut() {
                    if *slot == Some(cell_key) {
                        *slot = None;
                    }
                }

                // Normalize: if all neighbor slots are None, store `None`.
                if neighbors_buf.iter().all(Option::is_none) {
                    neighbor_cell.neighbors = None;
                }
            }
        }

        (affected_vertices, candidate_incident)
    }

    fn remove_cells_and_update_uuid_mappings(&mut self, cell_keys: &[CellKey]) -> usize {
        let mut removed_count = 0;

        for &cell_key in cell_keys {
            if let Some(removed_cell) = self.cells.remove(cell_key) {
                self.uuid_to_cell_key.remove(&removed_cell.uuid());
                removed_count += 1;
            }
        }

        removed_count
    }

    fn repair_incident_cells_after_cell_removal(
        &mut self,
        affected_vertices: &VertexKeySet,
        cells_to_remove: &CellKeySet,
        candidate_incident: &FastHashMap<VertexKey, CellKey>,
    ) {
        // We only need to consider vertices that appeared in removed cells.
        let vertices_to_repair: Vec<VertexKey> = affected_vertices
            .iter()
            .copied()
            .filter(|&vk| {
                let Some(v) = self.vertices.get(vk) else {
                    return false;
                };

                match v.incident_cell {
                    None => true,
                    Some(cell_key) => {
                        cells_to_remove.contains(&cell_key) || !self.cells.contains_key(cell_key)
                    }
                }
            })
            .collect();

        let mut incident_updates: Vec<(VertexKey, Option<CellKey>)> =
            Vec::with_capacity(vertices_to_repair.len());

        for vk in vertices_to_repair {
            // Prefer a candidate cell discovered on the boundary of the removed region.
            let mut new_incident = candidate_incident.get(&vk).copied().filter(|&ck| {
                self.cells
                    .get(ck)
                    .is_some_and(|cell| cell.contains_vertex(vk))
            });

            // Conservative fallback: pick the first remaining cell that contains this vertex.
            // This is only hit if neighbor pointers were missing or the boundary had no surviving cell.
            if new_incident.is_none() {
                new_incident = self
                    .cells
                    .iter()
                    .find_map(|(cell_key, cell)| cell.contains_vertex(vk).then_some(cell_key));
            }

            incident_updates.push((vk, new_incident));
        }

        for (vk, new_incident) in incident_updates {
            if let Some(vertex) = self.vertices.get_mut(vk) {
                vertex.incident_cell = new_incident;
            }
        }
    }

    /// Finds all cells containing a specific vertex.
    ///
    /// This is an internal helper used by [`Tds::remove_vertex`](Self::remove_vertex) to efficiently
    /// collect the vertex star that needs to be removed.
    ///
    /// # Arguments
    ///
    /// * `vertex_key` - The key of the vertex to search for
    ///
    /// # Returns
    ///
    /// A `CellRemovalBuffer` containing the keys of all cells that contain the vertex.
    /// Uses stack-backed `SmallBuffer` for typical vertex stars (≤16 incident cells).
    ///
    /// # Performance
    ///
    /// Fast path (typical, valid triangulations):
    /// - Uses the vertex's `incident_cell` pointer as a seed and walks the neighbor graph across
    ///   facets that still contain the vertex.
    /// - Time: O(|star(v)| × (D+1))
    /// - Space: O(|star(v)|)
    ///
    /// Conservative fallback:
    /// - If `incident_cell` is missing/stale or neighbor pointers are unavailable, falls back to
    ///   scanning all cells.
    /// - Time: O(#cells)
    fn find_cells_containing_vertex(&self, vertex_key: VertexKey) -> CellRemovalBuffer {
        let fallback_scan = || {
            self.cells()
                .filter_map(|(cell_key, cell)| cell.contains_vertex(vertex_key).then_some(cell_key))
                .collect()
        };

        // Fast path: walk the star from the vertex's incident cell using neighbor pointers.
        let Some(vertex) = self.vertices.get(vertex_key) else {
            return fallback_scan();
        };

        let Some(start_cell_key) = vertex.incident_cell else {
            return fallback_scan();
        };

        let Some(start_cell) = self.cells.get(start_cell_key) else {
            return fallback_scan();
        };

        if !start_cell.contains_vertex(vertex_key) || start_cell.neighbors().is_none() {
            return fallback_scan();
        }

        let mut visited: CellKeySet = CellKeySet::default();
        let mut stack: CellRemovalBuffer = CellRemovalBuffer::new();
        let mut result: CellRemovalBuffer = CellRemovalBuffer::new();

        visited.insert(start_cell_key);
        stack.push(start_cell_key);

        while let Some(cell_key) = stack.pop() {
            result.push(cell_key);

            let Some(cell) = self.cells.get(cell_key) else {
                continue;
            };

            let Some(neighbors) = cell.neighbors() else {
                continue;
            };

            // Traverse only across facets that still contain the target vertex.
            for (facet_idx, neighbor_opt) in neighbors.iter().enumerate() {
                if cell
                    .vertices()
                    .get(facet_idx)
                    .is_some_and(|&vkey| vkey == vertex_key)
                {
                    // This facet excludes `vertex_key`, so crossing it would
                    // leave the vertex star.
                    continue;
                }

                let Some(neighbor_key) = neighbor_opt else {
                    continue;
                };

                if visited.contains(neighbor_key) {
                    continue;
                }

                let Some(neighbor_cell) = self.cells.get(*neighbor_key) else {
                    continue;
                };

                if !neighbor_cell.contains_vertex(vertex_key) {
                    continue;
                }

                visited.insert(*neighbor_key);
                stack.push(*neighbor_key);
            }
        }

        result
    }

    /// Removes a vertex and all cells containing it, maintaining data structure consistency.
    ///
    /// This is a composite operation that, on success:
    /// 1. Finds all cells containing the vertex (using `incident_cell` + neighbor-walk when available)
    /// 2. Removes all such cells (clearing neighbor back-references + repairing affected incidence pointers)
    /// 3. Removes the vertex itself
    ///
    /// This operation leaves the triangulation in a valid state (though potentially incomplete).
    /// This is the recommended way to remove a vertex from the triangulation.
    ///
    /// # Arguments
    ///
    /// * `vertex` - Reference to the vertex to remove
    ///
    /// # Returns
    ///
    /// `Ok(usize)` with the number of cells that were removed along with the vertex.
    ///
    /// # Errors
    ///
    /// Currently this operation is infallible and always returns `Ok(_)`. The `Result` is retained
    /// for API consistency with other mutation operations and to allow future checks to surface as
    /// [`TdsMutationError`].
    ///
    /// # Performance
    ///
    /// Vertex removal is a **local** topological change (deleting the vertex star).
    ///
    /// Typical case (valid triangulations with neighbors/`incident_cell` assigned):
    /// - Star discovery: O(|star(v)| × (D+1)) via neighbor-walk (no global scan)
    /// - Removal + repair: typically O(|star(v)| × (D+1)²) touching only the boundary of the star
    ///
    /// Conservative fallbacks:
    /// - If `incident_cell`/neighbor pointers are unavailable (e.g., after unsafe mutation), the
    ///   implementation falls back to a global cell scan to find the star and/or a replacement
    ///   incident cell for affected vertices.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
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
    /// assert!(dt.tds().validate().is_ok());
    /// ```
    #[expect(
        clippy::unnecessary_wraps,
        reason = "Keep Result for future mutation validation without changing the API"
    )]
    pub(crate) fn remove_vertex(
        &mut self,
        vertex: &Vertex<T, U, D>,
    ) -> Result<usize, TdsMutationError> {
        // Find the vertex key
        let Some(vertex_key) = self.vertex_key_from_uuid(&vertex.uuid()) else {
            return Ok(0); // Vertex not found, nothing to remove
        };

        // Find all cells containing this vertex
        let cells_to_remove = self.find_cells_containing_vertex(vertex_key);

        // Remove all cells containing the vertex.
        //
        // `remove_cells_by_keys()` clears neighbor back-references that would otherwise dangle and
        // incrementally repairs `incident_cell` pointers for vertices that referenced removed cells.
        let cells_removed = self.remove_cells_by_keys(&cells_to_remove);

        // Remove the vertex itself (inline instead of using deprecated method)
        self.vertices.remove(vertex_key);
        self.uuid_to_vertex_key.remove(&vertex.uuid());
        // Topology changed; invalidate caches
        self.bump_generation();

        Ok(cells_removed)
    }

    // =========================================================================
    // KEY-BASED NEIGHBOR OPERATIONS (Phase 2 Optimization)
    // =========================================================================

    /// Finds neighbor cell keys for a given cell without UUID lookups.
    ///
    /// This is the key-based version of neighbor retrieval that avoids
    /// UUID→Key conversions in the hot path.
    ///
    /// # Arguments
    ///
    /// * `cell_key` - The key of the cell whose neighbors to find
    ///
    /// # Returns
    ///
    /// A buffer of `Option<CellKey>` where `None` indicates no neighbor
    /// at that position (boundary facet). Uses stack allocation for typical dimensions.
    ///
    /// **Special case**: If the cell does not exist (invalid `cell_key`), returns a buffer
    /// filled with `None` values. This is a non-panicking fallback that allows callers to
    /// distinguish "cell missing" from "no neighbors assigned" by checking cell existence
    /// separately with `get_cell()` if needed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tds = dt.tds();
    /// let (cell_key, _) = tds.cells().next().unwrap();
    ///
    /// // Get neighbors for existing cell
    /// let neighbors = tds.find_neighbors_by_key(cell_key);
    /// assert_eq!(neighbors.len(), 3); // D+1 for 2D
    /// ```
    #[must_use]
    pub fn find_neighbors_by_key(&self, cell_key: CellKey) -> NeighborBuffer<Option<CellKey>> {
        let mut neighbors = NeighborBuffer::new();
        neighbors.resize(D + 1, None);

        let Some(cell) = self.get_cell(cell_key) else {
            return neighbors;
        };

        // Phase 3A: Cell now stores neighbors directly
        if let Some(ref neighbors_from_cell) = cell.neighbors {
            // Use zip to avoid potential OOB if neighbors_from_cell.len() > D+1 (malformed data)
            for (slot, neighbor_key_opt) in neighbors.iter_mut().zip(neighbors_from_cell.iter()) {
                *slot = *neighbor_key_opt;
            }
        }

        neighbors
    }

    /// Validates the topological invariant for neighbor relationships.
    ///
    /// **Critical Invariant**: For a cell, `neighbors[i]` must be opposite `vertices[i]`,
    /// meaning the two cells share a facet containing all vertices **except** vertex `i`.
    ///
    /// # Arguments
    ///
    /// * `cell_key` - The key of the cell to validate
    /// * `neighbors` - The neighbor keys to validate (must have length D+1)
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the topology is valid
    /// * `Err(TdsValidationError)` with details about which neighbors violate the invariant
    ///
    /// # Use Cases
    ///
    /// - Called by `set_neighbors_by_key()` to enforce correctness
    /// - Can be called by `is_valid()` to check entire triangulation
    /// - Useful during incremental construction to identify cells needing repair
    ///
    /// # Errors
    ///
    /// Returns `TdsValidationError` if topology validation fails.
    fn validate_neighbor_topology(
        &self,
        cell_key: CellKey,
        neighbors: &[Option<CellKey>],
    ) -> Result<(), TdsValidationError> {
        if neighbors.len() != D + 1 {
            return Err(TdsError::InvalidNeighbors {
                message: format!(
                    "Neighbor vector length {} != D+1 ({})",
                    neighbors.len(),
                    D + 1
                ),
            });
        }

        let cell = self
            .cells
            .get(cell_key)
            .ok_or_else(|| TdsError::InconsistentDataStructure {
                message: format!("Cell key {cell_key:?} not found"),
            })?;

        let cell_vertices = cell.vertices();

        for (i, neighbor_key_opt) in neighbors.iter().enumerate() {
            if let Some(neighbor_key) = neighbor_key_opt {
                let neighbor = self.cells.get(*neighbor_key).ok_or_else(|| {
                    TdsError::InvalidNeighbors {
                        message: format!(
                            "Neighbor at position {i} references non-existent cell {neighbor_key:?}"
                        ),
                    }
                })?;

                let neighbor_vertices = neighbor.vertices();

                // Count shared vertices and find missing vertex
                let mut shared_count = 0;
                let mut missing_vertex_idx = None;

                for (idx, &vkey) in cell_vertices.iter().enumerate() {
                    if neighbor_vertices.contains(&vkey) {
                        shared_count += 1;
                    } else if missing_vertex_idx.is_none() {
                        missing_vertex_idx = Some(idx);
                    }
                }

                // Validate the topological invariant
                if shared_count != D {
                    return Err(TdsError::InvalidNeighbors {
                        message: format!(
                            "Cell {:?} neighbor at position {i} shares {shared_count} vertices, expected {D}. \
                            Invariant: neighbor[{i}] must share facet opposite vertex[{i}] (all vertices except vertex {i})",
                            cell.uuid()
                        ),
                    });
                }

                if missing_vertex_idx != Some(i) {
                    return Err(TdsError::InvalidNeighbors {
                        message: format!(
                            "Cell {:?} neighbor at position {i} is opposite vertex {:?}, expected {i}. \
                            Invariant: neighbor[{i}] must be opposite vertex[{i}]",
                            cell.uuid(),
                            missing_vertex_idx
                        ),
                    });
                }
            }
        }

        Ok(())
    }

    /// Sets neighbor relationships using cell keys directly.
    ///
    /// **Phase 3A**: This method now stores `CellKey`s directly in `Cell.neighbors`.
    ///
    /// # Positional Semantics (Critical Topological Invariant)
    ///
    /// **`neighbors[i]` must be the neighbor opposite to `vertices[i]`**
    ///
    /// This means the two cells share facet `i`, which contains all vertices **except** vertex `i`.
    ///
    /// ## Example: 3D Tetrahedron
    ///
    /// For a cell with vertices `[v0, v1, v2, v3]`:
    /// - `neighbors[0]` shares facet `[v1, v2, v3]` (opposite v0)
    /// - `neighbors[1]` shares facet `[v0, v2, v3]` (opposite v1)
    /// - `neighbors[2]` shares facet `[v0, v1, v3]` (opposite v2)
    /// - `neighbors[3]` shares facet `[v0, v1, v2]` (opposite v3)
    ///
    /// **This invariant is always validated** via `validate_neighbor_topology()`.
    ///
    /// # Arguments
    ///
    /// * `cell_key` - The key of the cell to update
    /// * `neighbors` - The new neighbor keys (must have length D+1)
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful, or an error if validation fails.
    ///
    /// # Errors
    ///
    /// Returns a `TdsMutationError` if:
    /// - The cell with the given key doesn't exist
    /// - The neighbor vector length is not D+1
    /// - Any neighbor key references a non-existent cell
    /// - **The topological invariant is violated** (neighbor\[i\] not opposite vertex\[i\])
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let mut tds = dt.tds().clone();
    /// let cell_key = tds.cell_keys().next().unwrap();
    /// let neighbors = vec![None; 3];
    /// tds.set_neighbors_by_key(cell_key, &neighbors).unwrap();
    /// assert!(tds.get_cell(cell_key).unwrap().neighbors().is_none());
    /// ```
    pub fn set_neighbors_by_key(
        &mut self,
        cell_key: CellKey,
        neighbors: &[Option<CellKey>],
    ) -> Result<(), TdsMutationError> {
        // Validate the topological invariant before applying changes
        // (includes length check: neighbors.len() == D+1)
        self.validate_neighbor_topology(cell_key, neighbors)?;

        // Phase 3A: Store CellKeys directly, no UUID conversion needed
        let neighbors_vec = neighbors;

        // Get mutable reference and update, or return error if not found
        let cell = self.get_cell_by_key_mut(cell_key).ok_or_else(|| {
            TdsError::InconsistentDataStructure {
                message: format!("Cell with key {cell_key:?} not found"),
            }
        })?;

        // Phase 3A: Store neighbor keys directly in SmallBuffer
        // Normalize: if all neighbors are None, set cell.neighbors to None
        if neighbors_vec.iter().all(Option::is_none) {
            cell.neighbors = None;
        } else {
            let mut neighbor_buffer = SmallBuffer::new();
            neighbor_buffer.extend(neighbors_vec.iter().copied());
            cell.neighbors = Some(neighbor_buffer);
        }

        // Topology changed; invalidate caches
        self.bump_generation();
        Ok(())
    }

    /// Finds cells containing a vertex using its key directly.
    ///
    /// This method avoids UUID lookups when searching for cells that
    /// contain a specific vertex.
    ///
    /// # Arguments
    ///
    /// * `vertex_key` - The key of the vertex to search for
    ///
    /// # Returns
    ///
    /// A set of cell keys that contain the given vertex.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tds = dt.tds();
    /// let vertex_key = tds.vertex_keys().next().unwrap();
    /// let cells = tds.find_cells_containing_vertex_by_key(vertex_key);
    /// assert_eq!(cells.len(), 1);
    /// ```
    #[must_use]
    pub fn find_cells_containing_vertex_by_key(&self, vertex_key: VertexKey) -> CellKeySet {
        if self.get_vertex_by_key(vertex_key).is_none() {
            return CellKeySet::default();
        }

        let cells = self.find_cells_containing_vertex(vertex_key);
        cells.iter().copied().collect()
    }

    /// Assigns incident cells to vertices in the triangulation.
    ///
    /// This method establishes a mapping from each vertex to one of the cells that contains it,
    /// which is useful for various geometric queries and traversals. For each vertex, an arbitrary
    /// incident cell is selected from the cells that contain that vertex.
    ///
    /// Note: Many topology-mutating operations (like [`Tds::remove_cells_by_keys`](Self::remove_cells_by_keys))
    /// attempt to repair `incident_cell` incrementally for affected vertices. This method exists as a
    /// conservative **full rebuild** after bulk changes (deserialization, large repairs, etc.).
    ///
    /// # Returns
    ///
    /// `Ok(())` if incident cells were successfully assigned to all vertices,
    /// otherwise a `TdsMutationError`.
    ///
    /// # Errors
    ///
    /// Returns a `TdsMutationError` if a cell references a non-existent vertex key
    /// (`InconsistentDataStructure`).
    ///
    /// # Algorithm
    ///
    /// 1. Clear `incident_cell` for all vertices
    /// 2. Scan all cells; for each vertex encountered, assign the current cell as its incident cell
    ///    if it does not already have one
    ///
    /// # Performance
    ///
    /// This method rebuilds incidence **globally** by scanning all cells:
    /// - Time: O(#cells × (D+1))
    /// - Space: O(1) extra (no temporary vertex→cells map)
    ///
    /// It is intended for repair/validation paths after bulk topology changes, not as a per-step
    /// hot-path update.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let mut tds = dt.tds().clone();
    /// tds.assign_incident_cells().unwrap();
    /// let all_assigned = tds.vertices().all(|(_, v)| v.incident_cell.is_some());
    /// assert!(all_assigned);
    /// ```
    pub fn assign_incident_cells(&mut self) -> Result<(), TdsMutationError> {
        if self.cells.is_empty() {
            // No cells remain; all vertices must have incident_cell cleared to avoid
            // dangling pointers to previously removed cells.
            for vertex in self.vertices.values_mut() {
                vertex.incident_cell = None;
            }
            return Ok(());
        }

        // Reset incident_cell for all vertices before rebuilding the mapping. This
        // ensures vertices that no longer belong to any cell do not retain stale
        // incident_cell pointers after topology changes (e.g., vertex or cell removal).
        for vertex in self.vertices.values_mut() {
            vertex.incident_cell = None;
        }

        // Single-pass rebuild: assign the first cell encountered for each vertex.
        for (cell_key, cell) in &self.cells {
            for &vertex_key in cell.vertices() {
                let vertex = self.vertices.get_mut(vertex_key).ok_or_else(|| {
                    TdsError::InconsistentDataStructure {
                        message: format!(
                            "Vertex key {vertex_key:?} not found in vertices storage map during incident cell assignment"
                        ),
                    }
                })?;

                if vertex.incident_cell.is_none() {
                    vertex.incident_cell = Some(cell_key);
                }
            }
        }

        Ok(())
    }

    /// Creates a new empty triangulation data structure.
    ///
    ///
    /// This function creates an empty triangulation with no vertices and no cells.
    /// Use [`DelaunayTriangulation::empty()`](crate::core::delaunay_triangulation::DelaunayTriangulation::empty)
    /// for the high-level API, or this method for low-level Tds construction.
    ///
    /// # Returns
    ///
    /// An empty triangulation data structure with:
    /// - No vertices
    /// - No cells
    /// - Construction state set to `Incomplete(0)`
    /// - Dimension of -1 (empty)
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::core::triangulation_data_structure::Tds;
    /// use delaunay::core::triangulation_data_structure::TriangulationConstructionState;
    ///
    /// let tds: Tds<f64, (), (), 3> = Tds::empty();
    /// assert_eq!(tds.number_of_vertices(), 0);
    /// assert_eq!(tds.number_of_cells(), 0);
    /// assert_eq!(tds.dim(), -1);
    /// assert!(matches!(tds.construction_state, TriangulationConstructionState::Incomplete(0)));
    /// ```
    #[must_use]
    pub fn empty() -> Self {
        Self {
            vertices: StorageMap::with_key(),
            cells: StorageMap::with_key(),
            uuid_to_vertex_key: UuidToVertexKeyMap::default(),
            uuid_to_cell_key: UuidToCellKeyMap::default(),
            construction_state: TriangulationConstructionState::Incomplete(0),
            generation: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Clears all neighbor relationships between cells in the triangulation.
    ///
    /// This method removes all neighbor relationships by setting the `neighbors` field
    /// to `None` for every cell in the triangulation. This is useful for:
    /// - Benchmarking neighbor assignment in isolation
    /// - Testing triangulations in a known state without neighbors
    /// - Debugging neighbor-related algorithms
    /// - Implementing custom neighbor assignment algorithms
    ///
    /// This is the inverse operation of `assign_neighbors`,
    /// and is commonly used in benchmarks and testing scenarios where you want to
    /// measure the performance of neighbor assignment starting from a clean state.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// // Initially has neighbors assigned during construction
    /// // All cells have neighbors
    /// for (_, cell) in dt.cells() {
    ///     // Check that cells have properly assigned neighbors
    ///     println!("Cell has neighbors: {:?}", cell.neighbors().is_some());
    /// }
    /// ```
    #[inline]
    pub fn clear_all_neighbors(&mut self) {
        for cell in self.cells.values_mut() {
            cell.clear_neighbors();
        }
        // Topology changed; invalidate caches.
        self.bump_generation();
    }

    /// Builds a `FacetToCellsMap` with strict error handling.
    ///
    /// This method returns an error if any cell has missing vertex keys, ensuring
    /// complete and accurate facet topology information. This is the preferred method
    /// for building facet-to-cells mappings.
    ///
    /// # Returns
    ///
    /// A `Result` containing:
    /// - `Ok(FacetToCellsMap)`: A complete mapping of facet keys to cells
    /// - `Err(TdsValidationError)`: If any cell has missing vertex keys
    ///
    /// # Errors
    ///
    /// Returns a `TdsValidationError::InconsistentDataStructure` if any cell
    /// cannot resolve its vertex keys, which would indicate a corrupted triangulation state.
    ///
    /// # Performance
    ///
    /// O(N×F) time complexity where N is the number of cells and F is the
    /// number of facets per cell (typically D+1 for D-dimensional cells).
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::triangulation::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tds = dt.tds();
    /// let facet_map = tds.build_facet_to_cells_map().unwrap();
    /// assert!(!facet_map.is_empty());
    /// ```
    pub fn build_facet_to_cells_map(&self) -> Result<FacetToCellsMap, TdsValidationError> {
        // Ensure facet indices fit in u8 range
        debug_assert!(
            D <= 255,
            "Dimension D must be <= 255 to fit facet indices in u8 (indices 0..=D)"
        );

        let cap = self.cells.len().saturating_mul(D.saturating_add(1));
        let mut facet_to_cells: FacetToCellsMap = fast_hash_map_with_capacity(cap);

        // Preallocate facet_vertices buffer outside the loops to avoid per-iteration allocations
        let mut facet_vertices = Vec::with_capacity(D);

        // Iterate over all cells and their facets
        for (cell_id, _cell) in &self.cells {
            // Use direct key-based method to avoid UUID→Key lookups
            // The error from get_cell_vertices is already TdsValidationError
            let vertices = self.get_cell_vertices(cell_id)?;

            for i in 0..vertices.len() {
                // Clear and reuse the buffer instead of allocating a new one
                facet_vertices.clear();
                for (j, &key) in vertices.iter().enumerate() {
                    if i != j {
                        facet_vertices.push(key);
                    }
                }

                let facet_key = facet_key_from_vertices(&facet_vertices);
                let Ok(facet_index_u8) = usize_to_u8(i, facet_vertices.len()) else {
                    return Err(TdsError::InconsistentDataStructure {
                        message: format!("Facet index {i} exceeds u8 range for dimension {D}"),
                    });
                };

                facet_to_cells
                    .entry(facet_key)
                    .or_default()
                    .push(FacetHandle::new(cell_id, facet_index_u8));
            }
        }

        Ok(facet_to_cells)
    }

    /// Remove duplicate cells (cells with identical vertex sets)
    ///
    /// Returns the number of duplicate cells that were removed.
    ///
    /// After removing duplicate cells, this method rebuilds the topology
    /// (neighbor relationships and incident cells) to maintain data structure
    /// invariants and prevent stale references.
    ///
    /// # Errors
    ///
    /// Returns a `TdsMutationError` if:
    /// - Vertex keys cannot be retrieved for any cell (data structure corruption)
    /// - Neighbor assignment fails after cell removal
    /// - Incident cell assignment fails after cell removal
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::core::triangulation_data_structure::Tds;
    ///
    /// let mut tds: Tds<f64, (), (), 2> = Tds::empty();
    /// let removed = tds.remove_duplicate_cells().unwrap();
    /// assert_eq!(removed, 0);
    /// ```
    pub fn remove_duplicate_cells(&mut self) -> Result<usize, TdsMutationError> {
        let mut unique_cells = FastHashMap::default();
        let mut cells_to_remove = CellRemovalBuffer::new();

        // First pass: identify duplicate cells
        for cell_key in self.cells.keys() {
            let vertices = self.get_cell_vertices(cell_key)?;
            // Sort vertex UUIDs instead of keys for deterministic ordering
            // Note: Don't sort by VertexKey as slotmap::Key's Ord is implementation-defined
            let mut vertex_uuids: Vec<_> = vertices
                .iter()
                .map(|&key| self.vertices[key].uuid())
                .collect();
            vertex_uuids.sort_unstable();

            // Use Entry API for atomic check-and-insert
            match unique_cells.entry(vertex_uuids) {
                Entry::Occupied(_) => {
                    cells_to_remove.push(cell_key);
                }
                Entry::Vacant(e) => {
                    e.insert(cell_key);
                }
            }
        }

        let duplicate_count = cells_to_remove.len();

        // Second pass: remove duplicate cells and their corresponding UUID mappings
        for cell_key in &cells_to_remove {
            if let Some(removed_cell) = self.cells.remove(*cell_key) {
                // Remove from our optimized UUID-to-key mapping
                self.uuid_to_cell_key.remove(&removed_cell.uuid());
            }
        }

        if duplicate_count > 0 {
            // Rebuild topology to avoid stale references after cell removal.
            // This ensures vertices don't point to removed cells via incident_cell,
            // and neighbor arrays don't reference removed keys.
            //
            // NOTE: Both `assign_neighbors()` and `assign_incident_cells()` are full rebuilds
            // across all cells/vertices (O(#cells)). This is intentionally conservative and is
            // expected to be used in repair/cleanup paths rather than per-step hot loops.
            self.assign_neighbors()?;
            self.assign_incident_cells()?;

            // Generation already bumped by assign_neighbors(); avoid double increment
        }
        Ok(duplicate_count)
    }

    // =========================================================================
    // VALIDATION & CONSISTENCY CHECKS
    // =========================================================================
    // Note: Structural validation is topology-only. Only Level-1 element validation (coordinates)
    // requires `T: CoordinateScalar`.

    /// Validates the consistency of vertex UUID-to-key mappings.
    ///
    /// This helper function ensures that:
    /// 1. The number of entries in `vertex_uuid_to_key` matches the number of vertices
    /// 2. The number of entries in `vertex_key_to_uuid` matches the number of vertices
    /// 3. Every vertex UUID in the triangulation has a corresponding key mapping
    /// 4. Every vertex key in the triangulation has a corresponding UUID mapping
    /// 5. The mappings are bidirectional and consistent (UUID ↔ Key)
    ///
    /// # Returns
    ///
    /// `Ok(())` if all vertex mappings are consistent, otherwise a `TdsValidationError`.
    ///
    /// This corresponds to [`InvariantKind::VertexMappings`], which is included in
    /// [`Tds::is_valid`](Self::is_valid) and [`Tds::validate`](Self::validate), and is also surfaced by
    /// [`DelaunayTriangulation::validation_report()`](crate::core::delaunay_triangulation::DelaunayTriangulation::validation_report).
    ///
    /// # Errors
    ///
    /// Returns a `TdsValidationError::MappingInconsistency` with a descriptive message if:
    /// - The number of UUID-to-key mappings doesn't match the number of vertices
    /// - The number of key-to-UUID mappings doesn't match the number of vertices
    /// - A vertex exists without a corresponding UUID-to-key mapping
    /// - A vertex exists without a corresponding key-to-UUID mapping
    /// - The bidirectional mappings are inconsistent (UUID maps to key A, but key A maps to different UUID)
    ///
    fn validate_vertex_mappings(&self) -> Result<(), TdsValidationError> {
        if self.uuid_to_vertex_key.len() != self.vertices.len() {
            return Err(TdsError::MappingInconsistency {
                entity: EntityKind::Vertex,
                message: format!(
                    "Number of mapping entries ({}) doesn't match number of vertices ({})",
                    self.uuid_to_vertex_key.len(),
                    self.vertices.len()
                ),
            });
        }

        // Phase 1: Optimize validation by checking key-to-UUID direction first (direct storage map access)
        // then only doing UUID-to-key lookup verification when needed
        for (vertex_key, vertex) in &self.vertices {
            let vertex_uuid = vertex.uuid();

            // Check key-to-UUID direction first (direct storage map access - no hash lookup)
            if self.vertex_uuid_from_key(vertex_key) != Some(vertex_uuid) {
                return Err(TdsError::MappingInconsistency {
                    entity: EntityKind::Vertex,
                    message: format!(
                        "Inconsistent or missing key-to-UUID mapping for key {vertex_key:?}"
                    ),
                });
            }

            // Now verify UUID-to-key direction (requires hash lookup but we know it should exist)
            if self.uuid_to_vertex_key.get(&vertex_uuid) != Some(&vertex_key) {
                return Err(TdsError::MappingInconsistency {
                    entity: EntityKind::Vertex,
                    message: format!(
                        "Inconsistent or missing UUID-to-key mapping for UUID {vertex_uuid:?}"
                    ),
                });
            }
        }
        Ok(())
    }

    /// Validates the consistency of cell UUID-to-key mappings.
    ///
    /// This helper function ensures that:
    /// 1. The number of entries in `cell_uuid_to_key` matches the number of cells
    /// 2. The number of entries in `cell_key_to_uuid` matches the number of cells
    /// 3. Every cell UUID in the triangulation has a corresponding key mapping
    /// 4. Every cell key in the triangulation has a corresponding UUID mapping
    /// 5. The mappings are bidirectional and consistent (UUID ↔ Key)
    ///
    /// # Returns
    ///
    /// `Ok(())` if all cell mappings are consistent, otherwise a `TdsValidationError`.
    ///
    /// This corresponds to [`InvariantKind::CellMappings`], which is included in
    /// [`Tds::is_valid`](Self::is_valid) and [`Tds::validate`](Self::validate), and is also surfaced by
    /// [`DelaunayTriangulation::validation_report()`].
    ///
    /// [`DelaunayTriangulation::validation_report()`]: crate::core::delaunay_triangulation::DelaunayTriangulation::validation_report
    ///
    /// # Errors
    ///
    /// Returns a `TdsValidationError::MappingInconsistency` with a descriptive message if:
    /// - The number of UUID-to-key mappings doesn't match the number of cells
    /// - The number of key-to-UUID mappings doesn't match the number of cells
    /// - A cell exists without a corresponding UUID-to-key mapping
    /// - A cell exists without a corresponding key-to-UUID mapping
    /// - The bidirectional mappings are inconsistent (UUID maps to key A, but key A maps to different UUID)
    ///
    fn validate_cell_mappings(&self) -> Result<(), TdsValidationError> {
        if self.uuid_to_cell_key.len() != self.cells.len() {
            return Err(TdsError::MappingInconsistency {
                entity: EntityKind::Cell,
                message: format!(
                    "Number of mapping entries ({}) doesn't match number of cells ({})",
                    self.uuid_to_cell_key.len(),
                    self.cells.len()
                ),
            });
        }

        // Phase 1: Optimize validation by checking key-to-UUID direction first (direct storage map access)
        // then only doing UUID-to-key lookup verification when needed
        for (cell_key, cell) in &self.cells {
            let cell_uuid = cell.uuid();

            // Check key-to-UUID direction first (direct storage map access - no hash lookup)
            if self.cell_uuid_from_key(cell_key) != Some(cell_uuid) {
                return Err(TdsError::MappingInconsistency {
                    entity: EntityKind::Cell,
                    message: format!(
                        "Inconsistent or missing key-to-UUID mapping for key {cell_key:?}"
                    ),
                });
            }

            // Now verify UUID-to-key direction (requires hash lookup but we know it should exist)
            if self.uuid_to_cell_key.get(&cell_uuid) != Some(&cell_key) {
                return Err(TdsError::MappingInconsistency {
                    entity: EntityKind::Cell,
                    message: format!(
                        "Inconsistent or missing UUID-to-key mapping for UUID {cell_uuid:?}"
                    ),
                });
            }
        }
        Ok(())
    }

    /// Validates that all vertex keys referenced by cells actually exist in the vertices `storage map`.
    ///
    /// This is a defensive check for data structure corruption. In normal operation,
    /// this should never fail, but it's useful for catching bugs during development
    /// and for comprehensive validation.
    ///
    /// **Phase 3A**: With cells storing vertex keys directly, this validation ensures
    /// that no cell references a stale or invalid vertex key.
    ///
    /// # Returns
    ///
    /// `Ok(())` if all vertex keys in all cells are valid, otherwise a `TdsValidationError`.
    ///
    /// # Errors
    ///
    /// Returns `TdsValidationError::InconsistentDataStructure` if any cell
    /// references a vertex key that doesn't exist in the vertices `storage map`.
    fn validate_cell_vertex_keys(&self) -> Result<(), TdsValidationError> {
        for (cell_key, cell) in &self.cells {
            let cell_uuid = cell.uuid();
            for (vertex_idx, &vertex_key) in cell.vertices().iter().enumerate() {
                if !self.vertices.contains_key(vertex_key) {
                    return Err(TdsError::InconsistentDataStructure {
                        message: format!(
                            "Cell {cell_uuid} (key {cell_key:?}) references non-existent vertex key {vertex_key:?} at position {vertex_idx}"
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    /// Validates that `Vertex::incident_cell` pointers are non-dangling and internally consistent.
    ///
    /// Note: at the TDS structural layer (Level 2), isolated vertices (vertices not referenced by
    /// any cell) are allowed, so `Vertex::incident_cell` may be `None`.
    ///
    /// Level 3 topology validation (`Triangulation::is_valid`) rejects isolated vertices.
    ///
    /// However, any `incident_cell` pointer that *is* present must:
    /// - point to an existing cell key, and
    /// - reference a cell that actually contains the vertex.
    fn validate_vertex_incidence(&self) -> Result<(), TdsValidationError> {
        for (vertex_key, vertex) in &self.vertices {
            let Some(incident_cell_key) = vertex.incident_cell else {
                continue;
            };

            let Some(incident_cell) = self.cells.get(incident_cell_key) else {
                return Err(TdsError::InconsistentDataStructure {
                    message: format!(
                        "Vertex {vertex_key:?} has dangling incident_cell pointer to missing cell {incident_cell_key:?}"
                    ),
                });
            };

            if !incident_cell.contains_vertex(vertex_key) {
                return Err(TdsError::InconsistentDataStructure {
                    message: format!(
                        "Vertex {vertex_key:?} incident_cell {incident_cell_key:?} does not contain the vertex"
                    ),
                });
            }
        }

        Ok(())
    }

    /// Check for duplicate cells and return an error if any are found
    ///
    /// This is useful for validation where you want to detect duplicates
    /// without automatically removing them.
    ///
    /// **Implementation Note**: This method uses `Cell::vertex_uuids()` to get canonical
    /// vertex UUIDs for each cell, which are then sorted and compared for duplicate detection.
    ///
    /// # Errors
    ///
    /// Returns a [`TdsValidationError`] if cell vertex retrieval fails
    /// or if any duplicate cells are detected.
    ///
    /// This corresponds to [`InvariantKind::DuplicateCells`], which is included in
    /// [`Tds::is_valid`](Self::is_valid) and [`Tds::validate`](Self::validate), and is also surfaced by
    /// [`DelaunayTriangulation::validation_report()`].
    ///
    /// [`DelaunayTriangulation::validation_report()`]: crate::core::delaunay_triangulation::DelaunayTriangulation::validation_report
    fn validate_no_duplicate_cells(&self) -> Result<(), TdsValidationError> {
        // Use CellVertexUuidBuffer as HashMap key directly to avoid extra Vec allocation
        // Pre-size to avoid rehashing during insertion (minor optimization for hot path)
        let mut unique_cells: FastHashMap<CellVertexUuidBuffer, CellKey> =
            crate::core::collections::fast_hash_map_with_capacity(self.cells.len());
        let mut duplicates = Vec::new();

        for (cell_key, cell) in &self.cells {
            // Use Cell::vertex_uuids() helper to avoid duplicating VertexKey→UUID mapping logic
            // Convert CellValidationError to TdsValidationError for propagation
            let mut vertex_uuids =
                cell.vertex_uuids(self)
                    .map_err(|e| TdsError::InconsistentDataStructure {
                        message: format!("Failed to get vertex UUIDs for cell {cell_key:?}: {e}"),
                    })?;

            // Canonicalize by sorting UUIDs for backend-agnostic equality
            // Note: Don't sort by VertexKey as slotmap::Key's Ord is implementation-defined
            vertex_uuids.sort_unstable();

            // Use buffer directly as HashMap key (keeps stack allocation, avoids Vec copy)
            if let Some(existing_cell_key) = unique_cells.get(&vertex_uuids) {
                // Convert to Vec only for error message payload
                duplicates.push((cell_key, *existing_cell_key, vertex_uuids.to_vec()));
            } else {
                unique_cells.insert(vertex_uuids, cell_key);
            }
        }

        if !duplicates.is_empty() {
            let duplicate_descriptions: Vec<String> = duplicates
                .iter()
                .map(|(cell1, cell2, vertex_uuids)| {
                    format!("cells {cell1:?} and {cell2:?} with vertex UUIDs {vertex_uuids:?}")
                })
                .collect();

            return Err(TdsError::DuplicateCells {
                message: format!(
                    "Found {} duplicate cell(s): {}",
                    duplicates.len(),
                    duplicate_descriptions.join(", ")
                ),
            });
        }

        Ok(())
    }

    /// Validates that no facet is shared by more than 2 cells
    ///
    /// This is a critical property for valid triangulations. Each facet should be
    /// shared by at most 2 cells - boundary facets belong to 1 cell, and internal
    /// facets should be shared by exactly 2 adjacent cells.
    ///
    /// # Errors
    ///
    /// Returns a [`TdsValidationError`] if building the facet map fails
    /// or if any facet is shared by more than two cells.
    ///
    /// This corresponds to [`InvariantKind::FacetSharing`], which is included in
    /// [`Tds::is_valid`](Self::is_valid) and [`Tds::validate`](Self::validate), and is also surfaced by
    /// [`DelaunayTriangulation::validation_report()`].
    ///
    /// [`DelaunayTriangulation::validation_report()`]: crate::core::delaunay_triangulation::DelaunayTriangulation::validation_report
    pub(crate) fn validate_facet_sharing(&self) -> Result<(), TdsValidationError> {
        // Build a map from facet keys to the cells that contain them.
        // Use the strict version to ensure we catch any missing vertex keys.
        let facet_to_cells = self.build_facet_to_cells_map()?;
        Self::validate_facet_sharing_with_facet_to_cells_map(&facet_to_cells)
    }

    fn validate_facet_sharing_with_facet_to_cells_map(
        facet_to_cells: &FacetToCellsMap,
    ) -> Result<(), TdsValidationError> {
        // Check for facets shared by more than 2 cells.
        for (facet_key, cell_facet_pairs) in facet_to_cells {
            if cell_facet_pairs.len() > 2 {
                return Err(TdsError::InconsistentDataStructure {
                    message: format!(
                        "Facet with key {} is shared by {} cells, but should be shared by at most 2 cells in a valid triangulation",
                        facet_key,
                        cell_facet_pairs.len()
                    ),
                });
            }
        }

        Ok(())
    }

    /// Checks whether the triangulation data structure is structurally valid.
    ///
    /// This is a **Level 2 (TDS structural)** check in the validation hierarchy.
    /// It intentionally does **not** validate individual vertices/cells (Level 1),
    /// nor triangulation topology (Level 3), nor the Delaunay property (Level 4).
    ///
    /// # Structural invariants checked
    /// - Vertex UUID↔key mapping consistency
    /// - Cell UUID↔key mapping consistency
    /// - Cells reference only valid vertex keys (no stale/missing vertex keys)
    /// - `Vertex::incident_cell`, when present, must point at an existing cell that contains the vertex.
    /// - No duplicate cells (same vertex set)
    /// - Facet sharing invariant (each facet is shared by at most 2 cells)
    /// - Neighbor consistency (topology + mutual neighbors)
    ///
    /// # ⚠️ Performance Warning
    ///
    /// **This method can be expensive** for large triangulations:
    /// - **Time Complexity**: O(N×F + N×D²) where N is the number of cells and F = D+1 facets per cell
    /// - **Space Complexity**: O(N×F) for facet-to-cell mappings
    ///
    /// For a cumulative validator that also checks vertices/cells (Level 1), use
    /// [`Tds::validate`](Self::validate).
    ///
    /// # Errors
    ///
    /// Returns a [`TdsValidationError`] if any structural invariant fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
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
    /// // Level 2: TDS structural validation
    /// assert!(dt.tds().is_valid().is_ok());
    /// ```
    pub fn is_valid(&self) -> Result<(), TdsValidationError> {
        // Fast-fail: return the first violated invariant.
        // For full diagnostics across all structural invariants, use `validation_report()`.
        self.validate_vertex_mappings()?;
        self.validate_cell_mappings()?;

        // Defensive: ensure no cell references a stale/missing vertex key before
        // higher-level structural checks that assume key validity.
        self.validate_cell_vertex_keys()?;

        // Structural: ensure `incident_cell` pointers, when present, are non-dangling + consistent.
        self.validate_vertex_incidence()?;

        self.validate_no_duplicate_cells()?;

        // Build the facet-to-cells map once and share it between facet-sharing and neighbor validators.
        let facet_to_cells = self.build_facet_to_cells_map()?;
        Self::validate_facet_sharing_with_facet_to_cells_map(&facet_to_cells)?;
        self.validate_neighbors_with_facet_to_cells_map(&facet_to_cells)?;

        Ok(())
    }

    /// Performs cumulative validation for Levels 1–2.
    ///
    /// This validates:
    /// - **Level 1**: all vertices (`Vertex::is_valid`) and all cells (`Cell::is_valid`)
    /// - **Level 2**: structural invariants (`Tds::is_valid`)
    ///
    /// # Errors
    ///
    /// Returns a [`TdsValidationError`] if any vertex/cell is invalid or if any
    /// structural invariant fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::*;
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
    /// // Levels 1–2: elements + TDS structure
    /// assert!(dt.tds().validate().is_ok());
    /// ```
    pub fn validate(&self) -> Result<(), TdsValidationError>
    where
        T: CoordinateScalar,
    {
        for (_vertex_key, vertex) in &self.vertices {
            if let Err(source) = (*vertex).is_valid() {
                return Err(TdsError::InvalidVertex {
                    vertex_id: vertex.uuid(),
                    source,
                });
            }
        }

        for (cell_key, cell) in &self.cells {
            if let Err(source) = cell.is_valid() {
                let Some(cell_id) = self.cell_uuid_from_key(cell_key) else {
                    return Err(TdsError::InconsistentDataStructure {
                        message: format!(
                            "Cell key {cell_key:?} has no UUID mapping during validation",
                        ),
                    });
                };

                return Err(TdsError::InvalidCell { cell_id, source });
            }
        }

        self.is_valid()
    }

    /// Runs structural validation checks and returns a report containing **all** failed invariants.
    ///
    /// Unlike [`is_valid()`](Self::is_valid), this method does **not** stop at the
    /// first error. Instead it records a [`TdsValidationError`] for each
    /// invariant group that fails and returns them as a
    /// [`TriangulationValidationReport`].
    ///
    /// **Note**: If UUID↔key mappings are inconsistent, this returns only mapping-related
    /// failures. Additional checks may produce misleading secondary errors or panic.
    ///
    /// **Note**: If any cell references a stale/missing vertex key, this reports the
    /// key-reference failure (and any vertex-incidence failures) and skips derived
    /// invariants that assume key validity.
    ///
    /// This is primarily intended for debugging, diagnostics, and tests that
    /// want to surface every violated invariant at once.
    ///
    /// **Note**: This does NOT check the Delaunay property. Use
    /// `DelaunayTriangulation::is_valid()` (Level 4) or `DelaunayTriangulation::validate()` (Levels 1–4)
    /// for geometric validation.
    ///
    /// # Errors
    ///
    /// Returns a [`TriangulationValidationReport`] containing all invariant
    /// violations if any validation step fails.
    pub(crate) fn validation_report(&self) -> Result<(), TriangulationValidationReport> {
        let mut violations = Vec::new();

        // 1. Mapping consistency (vertex + cell UUID↔key mappings)
        if let Err(e) = self.validate_vertex_mappings() {
            violations.push(InvariantViolation {
                kind: InvariantKind::VertexMappings,
                error: e.into(),
            });
        }
        if let Err(e) = self.validate_cell_mappings() {
            violations.push(InvariantViolation {
                kind: InvariantKind::CellMappings,
                error: e.into(),
            });
        }

        // If mappings are inconsistent, additional checks may produce confusing
        // secondary errors or panic. In that case, stop here and return the
        // mapping-related failures only.
        if !violations.is_empty() {
            return Err(TriangulationValidationReport { violations });
        }

        // 2. Cell→vertex key references (no stale/missing vertex keys)
        let mut cell_vertex_keys_ok = true;
        if let Err(e) = self.validate_cell_vertex_keys() {
            cell_vertex_keys_ok = false;
            violations.push(InvariantViolation {
                kind: InvariantKind::CellVertexKeys,
                error: e.into(),
            });
        }

        // 3. Vertex incidence (non-dangling `incident_cell` pointers, when present)
        if let Err(e) = self.validate_vertex_incidence() {
            violations.push(InvariantViolation {
                kind: InvariantKind::VertexIncidence,
                error: e.into(),
            });
        }

        // If cell vertex keys are invalid, derived invariants may produce confusing secondary
        // errors or panic (many routines assume key validity). Stop here.
        if !cell_vertex_keys_ok {
            return Err(TriangulationValidationReport { violations });
        }

        // 4. Cell uniqueness (no duplicate cells with identical vertex sets)
        if let Err(e) = self.validate_no_duplicate_cells() {
            violations.push(InvariantViolation {
                kind: InvariantKind::DuplicateCells,
                error: e.into(),
            });
        }

        // 5–6. Facet sharing + neighbor consistency share the facet-to-cells map.
        match self.build_facet_to_cells_map() {
            Ok(facet_to_cells) => {
                if let Err(e) =
                    Self::validate_facet_sharing_with_facet_to_cells_map(&facet_to_cells)
                {
                    violations.push(InvariantViolation {
                        kind: InvariantKind::FacetSharing,
                        error: e.into(),
                    });
                }

                if let Err(e) = self.validate_neighbors_with_facet_to_cells_map(&facet_to_cells) {
                    violations.push(InvariantViolation {
                        kind: InvariantKind::NeighborConsistency,
                        error: e.into(),
                    });
                }
            }
            Err(e) => {
                // If we can't build the facet map, both facet-sharing and neighbor checks are blocked.
                //
                // We intentionally record *both* invariant kinds for diagnostic granularity.
                // This requires cloning the error once (so each violation owns an error), which
                // may allocate/copy string payloads, but this is on an error path and facet-map
                // build errors are expected to be rare and small.
                violations.push(InvariantViolation {
                    kind: InvariantKind::FacetSharing,
                    error: e.clone().into(),
                });
                violations.push(InvariantViolation {
                    kind: InvariantKind::NeighborConsistency,
                    error: e.into(),
                });
            }
        }

        if violations.is_empty() {
            Ok(())
        } else {
            Err(TriangulationValidationReport { violations })
        }
    }

    /// Validates global neighbor relationships for the triangulation.
    ///
    /// This method validates:
    /// - Topological invariant (neighbor\[i\] is opposite vertex\[i\]) via `validate_neighbor_topology()`
    /// - Mutual neighbor relationships (if A neighbors B, then B neighbors A)
    /// - Shared facet correctness (neighbors share exactly D vertices)
    ///
    /// # Performance / intended use
    ///
    /// This routine is intentionally thorough and defensive. It precomputes per-cell vertex sets
    /// and performs per-neighbor set-intersection + mirror-facet cross-checks, which can be
    /// relatively expensive for large triangulations.
    ///
    /// It is only invoked from explicit validation APIs (`is_valid()`, `validate()`,
    /// `validation_report()`) and is not intended for per-step hot paths.
    ///
    /// Some small optimizations keep the cost reasonable:
    /// - Early termination on validation failures
    /// - Precomputing per-cell vertex sets once
    /// - Counting intersections without allocating intermediate collections
    ///
    /// # Errors
    ///
    /// Returns a [`TdsValidationError`] if any neighbor relationship
    /// violates topological or consistency invariants.
    ///
    /// This corresponds to [`InvariantKind::NeighborConsistency`], which is included in
    /// [`Tds::is_valid`](Self::is_valid) and [`Tds::validate`](Self::validate), and is also surfaced by
    /// [`DelaunayTriangulation::validation_report()`].
    ///
    /// [`DelaunayTriangulation::validation_report()`]: crate::core::delaunay_triangulation::DelaunayTriangulation::validation_report
    ///
    /// Note: callers provide `facet_to_cells` so `is_valid()` and `validation_report()` can share
    /// the precomputed facet map between validators.
    fn validate_neighbors_with_facet_to_cells_map(
        &self,
        facet_to_cells: &FacetToCellsMap,
    ) -> Result<(), TdsValidationError> {
        self.validate_neighbor_pointers_match_facet_to_cells_map(facet_to_cells)?;

        let cell_vertices = self.build_cell_vertex_sets()?;
        self.validate_neighbors_with_precomputed_vertex_sets(&cell_vertices)
    }

    fn validate_neighbor_pointers_match_facet_to_cells_map(
        &self,
        facet_to_cells: &FacetToCellsMap,
    ) -> Result<(), TdsValidationError> {
        for (facet_key, cell_facet_pairs) in facet_to_cells {
            match cell_facet_pairs.as_slice() {
                [handle] => {
                    // Boundary facet: must not have a neighbor across this facet.
                    let cell_key = handle.cell_key();
                    let facet_index = handle.facet_index() as usize;

                    let cell = self.cells.get(cell_key).ok_or_else(|| {
                        TdsError::InconsistentDataStructure {
                            message: format!(
                                "Cell key {cell_key:?} not found during neighbor validation"
                            ),
                        }
                    })?;

                    if let Some(neighbors) = cell.neighbors() {
                        let neighbor = neighbors.get(facet_index).and_then(|n| *n);
                        if let Some(neighbor_key) = neighbor {
                            return Err(TdsError::InvalidNeighbors {
                                message: format!(
                                    "Boundary facet {facet_key} unexpectedly has a neighbor across cell {}[{facet_index}] -> {neighbor_key:?}",
                                    cell.uuid(),
                                ),
                            });
                        }
                    }
                }
                [a, b] => {
                    // Interior facet: both cells must be neighbors across the corresponding facet indices.
                    let first_cell_key = a.cell_key();
                    let first_facet_index = a.facet_index() as usize;
                    let second_cell_key = b.cell_key();
                    let second_facet_index = b.facet_index() as usize;

                    let first_cell = self.cells.get(first_cell_key).ok_or_else(|| {
                        TdsError::InconsistentDataStructure {
                            message: format!(
                                "Cell key {first_cell_key:?} not found during neighbor validation"
                            ),
                        }
                    })?;
                    let second_cell = self.cells.get(second_cell_key).ok_or_else(|| {
                        TdsError::InconsistentDataStructure {
                            message: format!(
                                "Cell key {second_cell_key:?} not found during neighbor validation"
                            ),
                        }
                    })?;

                    let first_neighbor = first_cell
                        .neighbors()
                        .and_then(|n| n.get(first_facet_index))
                        .and_then(|n| *n);
                    let second_neighbor = second_cell
                        .neighbors()
                        .and_then(|n| n.get(second_facet_index))
                        .and_then(|n| *n);

                    if first_neighbor != Some(second_cell_key)
                        || second_neighbor != Some(first_cell_key)
                    {
                        return Err(TdsError::InvalidNeighbors {
                            message: format!(
                                "Interior facet {facet_key} has inconsistent neighbor pointers: {}[{first_facet_index}] -> {first_neighbor:?}, {}[{second_facet_index}] -> {second_neighbor:?}",
                                first_cell.uuid(),
                                second_cell.uuid(),
                            ),
                        });
                    }
                }
                _ => {
                    // Non-manifold facet multiplicity should have been caught by facet-sharing validation.
                    return Err(TdsError::InconsistentDataStructure {
                        message: format!(
                            "Facet with key {facet_key} is shared by {} cells, but should be shared by at most 2 cells in a valid triangulation",
                            cell_facet_pairs.len()
                        ),
                    });
                }
            }
        }

        Ok(())
    }

    fn validate_neighbors_with_precomputed_vertex_sets(
        &self,
        cell_vertices: &CellVerticesMap,
    ) -> Result<(), TdsValidationError> {
        for (cell_key, cell) in &self.cells {
            // Phase 3A: Use neighbors (CellKey-based) instead of neighbor UUIDs
            let Some(neighbors_buf) = &cell.neighbors else {
                continue; // Skip cells without neighbors
            };

            // Convert SmallBuffer to Vec for validation
            let neighbors: Vec<Option<CellKey>> = neighbors_buf.iter().copied().collect();

            // Validate topological invariant (neighbor[i] opposite vertex[i])
            self.validate_neighbor_topology(cell_key, &neighbors)?;

            let this_vertices = cell_vertices.get(&cell_key).ok_or_else(|| {
                TdsError::InconsistentDataStructure {
                    message: format!(
                        "Cell {} (key {cell_key:?}) missing from precomputed vertex set map during neighbor validation",
                        cell.uuid()
                    ),
                }
            })?;

            for (facet_idx, neighbor_key_opt) in neighbors.iter().enumerate() {
                // Skip None neighbors (missing neighbors)
                let Some(neighbor_key) = neighbor_key_opt else {
                    continue;
                };

                // Early termination: check if neighbor exists
                let Some(neighbor_cell) = self.cells.get(*neighbor_key) else {
                    return Err(TdsError::InvalidNeighbors {
                        message: format!("Neighbor cell {neighbor_key:?} not found"),
                    });
                };

                let neighbor_vertices = cell_vertices.get(neighbor_key).ok_or_else(|| {
                    TdsError::InconsistentDataStructure {
                        message: format!(
                            "Neighbor cell {} (key {neighbor_key:?}) missing from precomputed vertex set map during neighbor validation",
                            neighbor_cell.uuid()
                        ),
                    }
                })?;

                Self::validate_shared_facet_count(
                    cell,
                    neighbor_cell,
                    this_vertices,
                    neighbor_vertices,
                )?;

                let mirror_idx = Self::compute_and_verify_mirror_facet(
                    cell,
                    facet_idx,
                    neighbor_cell,
                    this_vertices,
                )?;

                Self::validate_shared_facet_vertices(
                    cell,
                    facet_idx,
                    neighbor_cell,
                    mirror_idx,
                    this_vertices,
                    neighbor_vertices,
                )?;

                Self::validate_mutual_neighbor_back_reference(
                    cell_key,
                    cell,
                    facet_idx,
                    neighbor_cell,
                    mirror_idx,
                )?;
            }
        }

        Ok(())
    }

    fn build_cell_vertex_sets(&self) -> Result<CellVerticesMap, TdsValidationError> {
        // Pre-compute vertex keys for all cells to avoid repeated computation
        let mut cell_vertices: CellVerticesMap = fast_hash_map_with_capacity(self.cells.len());

        for cell_key in self.cells.keys() {
            // Use get_cell_vertices to ensure all vertex keys are present
            // The error is already TdsValidationError, so just propagate it
            let vertices = self.get_cell_vertices(cell_key)?;

            // Store the HashSet for containment checks
            let vertex_set: VertexKeySet = vertices.iter().copied().collect();
            cell_vertices.insert(cell_key, vertex_set);
        }

        Ok(cell_vertices)
    }

    fn validate_shared_facet_count(
        cell: &Cell<T, U, V, D>,
        neighbor_cell: &Cell<T, U, V, D>,
        this_vertices: &VertexKeySet,
        neighbor_vertices: &VertexKeySet,
    ) -> Result<(), TdsValidationError> {
        let shared_count = this_vertices.intersection(neighbor_vertices).count();

        if shared_count != D {
            return Err(TdsError::NotNeighbors {
                cell1: cell.uuid(),
                cell2: neighbor_cell.uuid(),
            });
        }

        Ok(())
    }

    fn compute_and_verify_mirror_facet(
        cell: &Cell<T, U, V, D>,
        facet_idx: usize,
        neighbor_cell: &Cell<T, U, V, D>,
        this_vertices: &VertexKeySet,
    ) -> Result<usize, TdsValidationError> {
        let mirror_idx = cell
            .mirror_facet_index(facet_idx, neighbor_cell)
            .ok_or_else(|| TdsError::InvalidNeighbors {
                message: format!(
                    "Could not find mirror facet: cell {:?}[{facet_idx}] -> neighbor {:?}",
                    cell.uuid(),
                    neighbor_cell.uuid()
                ),
            })?;

        // Defensive cross-check: verify the mirror index against shared-vertex analysis.
        // This adds overhead but guards against subtle logic bugs in `mirror_facet_index()`.
        //
        // If validation ever becomes performance-sensitive, this is a good candidate to
        // gate behind a "strict validation" option/flag.
        let expected_mirror_idx =
            Self::compute_expected_mirror_facet_index(cell, neighbor_cell, this_vertices)?;

        if mirror_idx != expected_mirror_idx {
            return Err(TdsError::InvalidNeighbors {
                message: format!(
                    "Mirror facet index mismatch: cell {:?}[{facet_idx}] -> neighbor {:?}; mirror_facet_index returned {mirror_idx} but shared-vertex analysis implies {expected_mirror_idx}",
                    cell.uuid(),
                    neighbor_cell.uuid()
                ),
            });
        }

        Ok(mirror_idx)
    }

    fn compute_expected_mirror_facet_index(
        cell: &Cell<T, U, V, D>,
        neighbor_cell: &Cell<T, U, V, D>,
        this_vertices: &VertexKeySet,
    ) -> Result<usize, TdsValidationError> {
        let mut expected_mirror_idx: Option<usize> = None;

        for (idx, &neighbor_vkey) in neighbor_cell.vertices().iter().enumerate() {
            if !this_vertices.contains(&neighbor_vkey) {
                if expected_mirror_idx.is_some() {
                    return Err(TdsError::InvalidNeighbors {
                        message: format!(
                            "Mirror facet is ambiguous: cell {:?} and neighbor {:?} differ by more than one vertex",
                            cell.uuid(),
                            neighbor_cell.uuid()
                        ),
                    });
                }
                expected_mirror_idx = Some(idx);
            }
        }

        expected_mirror_idx.ok_or_else(|| TdsError::InvalidNeighbors {
            message: format!(
                "Mirror facet could not be determined: cell {:?} and neighbor {:?} appear to share all vertices (duplicate cells?)",
                cell.uuid(),
                neighbor_cell.uuid()
            ),
        })
    }

    fn validate_shared_facet_vertices(
        cell: &Cell<T, U, V, D>,
        facet_idx: usize,
        neighbor_cell: &Cell<T, U, V, D>,
        mirror_idx: usize,
        this_vertices: &VertexKeySet,
        neighbor_vertices: &VertexKeySet,
    ) -> Result<(), TdsValidationError> {
        for (idx, &vkey) in cell.vertices().iter().enumerate() {
            if idx == facet_idx {
                continue;
            }
            if !neighbor_vertices.contains(&vkey) {
                return Err(TdsError::InvalidNeighbors {
                    message: format!(
                        "Shared facet mismatch: cell {:?}[{facet_idx}] -> neighbor {:?}[{mirror_idx}] is missing vertex {vkey:?} from the shared facet",
                        cell.uuid(),
                        neighbor_cell.uuid(),
                    ),
                });
            }
        }

        for (idx, &vkey) in neighbor_cell.vertices().iter().enumerate() {
            if idx == mirror_idx {
                continue;
            }
            if !this_vertices.contains(&vkey) {
                return Err(TdsError::InvalidNeighbors {
                    message: format!(
                        "Shared facet mismatch: neighbor {:?}[{mirror_idx}] -> cell {:?}[{facet_idx}] is missing vertex {vkey:?} from the shared facet",
                        neighbor_cell.uuid(),
                        cell.uuid(),
                    ),
                });
            }
        }

        Ok(())
    }

    fn validate_mutual_neighbor_back_reference(
        cell_key: CellKey,
        cell: &Cell<T, U, V, D>,
        facet_idx: usize,
        neighbor_cell: &Cell<T, U, V, D>,
        mirror_idx: usize,
    ) -> Result<(), TdsValidationError> {
        let Some(neighbor_neighbors) = &neighbor_cell.neighbors else {
            return Err(TdsError::InvalidNeighbors {
                message: format!(
                    "Neighbor relationship not mutual: {:?}[{facet_idx}] -> {:?} (neighbor has no neighbors)",
                    cell.uuid(),
                    neighbor_cell.uuid()
                ),
            });
        };

        let back_ref = neighbor_neighbors.get(mirror_idx).copied().flatten();
        if back_ref != Some(cell_key) {
            return Err(TdsError::InvalidNeighbors {
                message: format!(
                    "Neighbor relationship not mutual: {:?}[{facet_idx}] -> {:?}[{mirror_idx}] (expected back-reference)",
                    cell.uuid(),
                    neighbor_cell.uuid()
                ),
            });
        }

        Ok(())
    }
}

// =============================================================================
// TRAIT IMPLEMENTATIONS
// =============================================================================

/// Manual implementation of `PartialEq` for Tds
///
/// Two triangulation data structures are considered equal if they have:
/// - The same set of vertices (compared by coordinates)
/// - The same set of cells (compared by vertex sets)
/// - Consistent vertex and cell mappings
///
/// **Note:** Vertices with NaN coordinates are rejected during construction; equality assumes no NaNs.
/// The triangulation validates coordinates at construction time to ensure no NaN values are present.
///
/// Note: Buffer fields are ignored since they are transient data structures.
impl<T, U, V, const D: usize> PartialEq for Tds<T, U, V, D>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    fn eq(&self, other: &Self) -> bool {
        // Early exit if the basic counts don't match
        if self.vertices.len() != other.vertices.len()
            || self.cells.len() != other.cells.len()
            || self.uuid_to_vertex_key.len() != other.uuid_to_vertex_key.len()
            || self.uuid_to_cell_key.len() != other.uuid_to_cell_key.len()
        {
            return false;
        }

        // Compare vertices by collecting them into sorted vectors
        // We sort by coordinates to make comparison order-independent
        let mut self_vertices: Vec<_> = self.vertices.values().collect();
        let mut other_vertices: Vec<_> = other.vertices.values().collect();

        // Sort vertices by their coordinates for consistent comparison
        // CoordinateScalar guarantees PartialOrd; NaN validation occurs at construction time
        self_vertices.sort_by(|a, b| {
            let a_coords = *a.point().coords();
            let b_coords = *b.point().coords();
            a_coords
                .partial_cmp(&b_coords)
                .unwrap_or(CmpOrdering::Equal)
        });

        other_vertices.sort_by(|a, b| {
            let a_coords = *a.point().coords();
            let b_coords = *b.point().coords();
            a_coords
                .partial_cmp(&b_coords)
                .unwrap_or(CmpOrdering::Equal)
        });

        // Compare sorted vertex lists
        if self_vertices != other_vertices {
            return false;
        }

        // Compare cells using Cell::eq_by_vertices() which uses Vertex::PartialEq
        // This provides consistent behavior: Tds uses Vertex::PartialEq and Cell::eq_by_vertices()
        // which internally uses Vertex::PartialEq for semantic equality across TDS instances.

        // Collect cells into vectors for sorting
        let mut self_cells: Vec<_> = self.cells.values().collect();
        let mut other_cells: Vec<_> = other.cells.values().collect();

        // Sort cells for order-independent comparison using UUID-based keys
        // This avoids allocating coordinate vectors and is cheaper than coordinate comparison
        let sort_key = |cell: &Cell<T, U, V, D>, tds: &Self| -> Vec<Uuid> {
            let mut ids: Vec<Uuid> = cell
                .vertices()
                .iter()
                .filter_map(|&vkey| tds.get_vertex_by_key(vkey).map(Vertex::uuid))
                .collect();
            ids.sort_unstable();
            ids
        };

        self_cells.sort_by(|a, b| {
            sort_key(a, self)
                .partial_cmp(&sort_key(b, self))
                .unwrap_or(CmpOrdering::Equal)
        });

        other_cells.sort_by(|a, b| {
            sort_key(a, other)
                .partial_cmp(&sort_key(b, other))
                .unwrap_or(CmpOrdering::Equal)
        });

        // Compare sorted cell lists using Cell::eq_by_vertices
        if self_cells.len() != other_cells.len() {
            return false;
        }

        for (self_cell, other_cell) in self_cells.iter().zip(other_cells.iter()) {
            if !self_cell.eq_by_vertices(self, other_cell, other) {
                return false;
            }
        }

        // If we get here, the triangulations have the same structure
        // UUID→Key maps are derived from the vertices/cells, so if those match, the maps should be consistent
        // (We don't need to compare the maps directly since they're just indexing structures)

        true
    }
}

/// Eq implementation for Tds
///
/// This is a marker trait implementation that relies on the `PartialEq` implementation.
/// Since Tds represents a well-defined mathematical structure (triangulation),
/// the `PartialEq` relation is indeed an equivalence relation.
impl<T, U, V, const D: usize> Eq for Tds<T, U, V, D>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
}

/// Manual implementation of Serialize for Tds to include cell-vertex UUID mappings
///
/// Phase 3A: Cells store vertex keys that can't be serialized directly, so we need
/// to serialize the mapping of cell UUID → vertex UUIDs to enable deserialization
/// to rebuild the vertex keys for each cell.
impl<T, U, V, const D: usize> Serialize for Tds<T, U, V, D>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        // Build cell UUID → vertex UUIDs mapping
        // Note: Using Vec<Uuid> for serde compatibility (SmallVec isn't natively serializable)
        let cell_vertices: FastHashMap<Uuid, Vec<Uuid>> = self
            .cells
            .iter()
            .map(|(_cell_key, cell)| {
                let cell_uuid = cell.uuid();
                let vertex_uuids = cell.vertex_uuids(self).map_err(serde::ser::Error::custom)?;
                // Convert CellVertexUuidBuffer (SmallVec) to Vec for serde
                Ok((cell_uuid, vertex_uuids.to_vec()))
            })
            .collect::<Result<_, _>>()?;

        let mut state = serializer.serialize_struct("Tds", 3)?;
        state.serialize_field("vertices", &self.vertices)?;
        state.serialize_field("cells", &self.cells)?;
        state.serialize_field("cell_vertices", &cell_vertices)?;
        state.end()
    }
}

/// Manual implementation of Deserialize for Tds to handle trait bound conflicts
impl<'de, T, U, V, const D: usize> Deserialize<'de> for Tds<T, U, V, D>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    fn deserialize<D2>(deserializer: D2) -> Result<Self, D2::Error>
    where
        D2: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            Vertices,
            Cells,
            CellVertices,
        }

        struct TdsVisitor<T, U, V, const D: usize>(PhantomData<(T, U, V)>);

        impl<'de, T, U, V, const D: usize> Visitor<'de> for TdsVisitor<T, U, V, D>
        where
            T: CoordinateScalar,
            U: DataType,
            V: DataType,
        {
            type Value = Tds<T, U, V, D>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct Tds")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut vertices: Option<StorageMap<VertexKey, Vertex<T, U, D>>> = None;
                let mut cells: Option<StorageMap<CellKey, Cell<T, U, V, D>>> = None;
                let mut cell_vertices: Option<FastHashMap<Uuid, Vec<Uuid>>> = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Vertices => {
                            if vertices.is_some() {
                                return Err(de::Error::duplicate_field("vertices"));
                            }
                            vertices = Some(map.next_value()?);
                        }
                        Field::Cells => {
                            if cells.is_some() {
                                return Err(de::Error::duplicate_field("cells"));
                            }
                            cells = Some(map.next_value()?);
                        }
                        Field::CellVertices => {
                            if cell_vertices.is_some() {
                                return Err(de::Error::duplicate_field("cell_vertices"));
                            }
                            cell_vertices = Some(map.next_value()?);
                        }
                    }
                }

                let vertices = vertices.ok_or_else(|| de::Error::missing_field("vertices"))?;
                let mut cells = cells.ok_or_else(|| de::Error::missing_field("cells"))?;
                let cell_vertices =
                    cell_vertices.ok_or_else(|| de::Error::missing_field("cell_vertices"))?;

                // Rebuild UUID→Key mappings from the deserialized data
                let mut uuid_to_vertex_key = fast_hash_map_with_capacity(vertices.len());
                for (vertex_key, vertex) in &vertices {
                    uuid_to_vertex_key.insert(vertex.uuid(), vertex_key);
                }

                let mut uuid_to_cell_key = fast_hash_map_with_capacity(cells.len());
                for (cell_key, cell) in &cells {
                    uuid_to_cell_key.insert(cell.uuid(), cell_key);
                }

                // Phase 3A: Rebuild cell vertex keys from the cell_vertices mapping
                for (_cell_key, cell) in &mut cells {
                    let cell_uuid = cell.uuid();
                    if let Some(vertex_uuids) = cell_vertices.get(&cell_uuid) {
                        // Clear stale vertex keys from serialized data before rebuilding
                        // This prevents duplication: serialized keys + reconstructed keys
                        cell.clear_vertex_keys();

                        // Convert vertex UUIDs to vertex keys
                        for &vertex_uuid in vertex_uuids {
                            if let Some(&vertex_key) = uuid_to_vertex_key.get(&vertex_uuid) {
                                cell.push_vertex_key(vertex_key);
                            } else {
                                return Err(de::Error::custom(format!(
                                    "Vertex UUID {vertex_uuid} referenced by cell {cell_uuid} not found in vertices"
                                )));
                            }
                        }
                    } else {
                        return Err(de::Error::custom(format!(
                            "No vertex UUIDs found for cell {cell_uuid}"
                        )));
                    }
                }

                // Phase 3A: Rebuild incident_cell mappings and neighbors since Keys aren't serialized
                let mut tds = Tds {
                    vertices,
                    cells,
                    uuid_to_vertex_key,
                    uuid_to_cell_key,
                    // Initialize construction state to default when deserializing
                    // Since only constructed triangulations should be serialized,
                    // we assume the deserialized triangulation is constructed
                    construction_state: TriangulationConstructionState::Constructed,
                    // Initialize generation counter to 0 when deserializing
                    // NOTE: Generation counter reset on deserialization means cache generation
                    // comparisons across serialize/deserialize boundaries will be invalidated.
                    // This ensures cached data from before serialization is not incorrectly
                    // considered valid after deserialization.
                    generation: Arc::new(AtomicU64::new(0)),
                };

                // Rebuild topology; fail fast on any inconsistency.
                // Order: neighbors first, then incident cells (consistent with other call sites).
                tds.assign_neighbors().map_err(de::Error::custom)?;
                tds.assign_incident_cells().map_err(de::Error::custom)?;

                Ok(tds)
            }
        }

        const FIELDS: &[&str] = &["vertices", "cells", "cell_vertices"];
        deserializer.deserialize_struct("Tds", FIELDS, TdsVisitor(PhantomData))
    }
}

// =============================================================================
// SERDE HELPERS
// =============================================================================

// UUID-to-key mappings are skipped during serialization and reconstructed during
// deserialization from the vertices and cells data.

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::incremental_insertion::InsertionError;
    use crate::core::{delaunay_triangulation::DelaunayTriangulation, vertex::VertexBuilder};
    use crate::geometry::point::Point;
    use crate::geometry::traits::coordinate::Coordinate;
    use crate::vertex;
    use slotmap::KeyData;

    // =============================================================================
    // TEST HELPER FUNCTIONS
    // =============================================================================

    /// Test helper to create a vertex with a specific UUID for collision testing.
    /// This is only used in tests to create specific scenarios.
    #[cfg(test)]
    fn create_vertex_with_uuid<T, U, const D: usize>(
        point: Point<T, D>,
        uuid: Uuid,
        data: Option<U>,
    ) -> Vertex<T, U, D>
    where
        T: CoordinateScalar,
        U: DataType,
    {
        let mut vertex = data.map_or_else(
            || {
                VertexBuilder::default()
                    .point(point)
                    .build()
                    .expect("Failed to build vertex")
            },
            |data_value| {
                VertexBuilder::default()
                    .point(point)
                    .data(data_value)
                    .build()
                    .expect("Failed to build vertex")
            },
        );

        vertex.set_uuid(uuid).expect("Failed to set UUID");
        vertex
    }

    fn initial_simplex_vertices_3d() -> [Vertex<f64, (), 3>; 4] {
        [
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ]
    }

    // =============================================================================
    // VERTEX ADDITION TESTS - CONSOLIDATED
    // =============================================================================

    #[test]
    fn test_repair_degenerate_cells() {
        // Exercise the repair primitive by creating a cell with a neighbor pointer
        // to a missing cell key.
        use crate::core::cell::Cell;

        let vertices = [
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
            vertex!([1.0, 1.0]),
        ];
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let v_keys: Vec<_> = vertices
            .iter()
            .copied()
            .map(|v| tds.insert_vertex_with_mapping(v).unwrap())
            .collect();

        // A valid cell that should remain after repair.
        let good_cell = Cell::new(vec![v_keys[0], v_keys[1], v_keys[2]], None).unwrap();
        let good_cell_key = tds.insert_cell_with_mapping(good_cell).unwrap();

        // A second cell that will be made degenerate.
        let bad_cell = Cell::new(vec![v_keys[0], v_keys[1], v_keys[3]], None).unwrap();
        let bad_cell_key = tds.insert_cell_with_mapping(bad_cell).unwrap();

        // Insert and then remove a third cell so we get a real CellKey that no longer exists.
        // Removing *after* inserting bad_cell avoids key reuse affecting the test.
        let removed_target_cell = Cell::new(vec![v_keys[1], v_keys[2], v_keys[3]], None).unwrap();
        let removed_target_key = tds.insert_cell_with_mapping(removed_target_cell).unwrap();
        assert!(tds.remove_cell_by_key(removed_target_key).is_some());

        // Inject a dangling neighbor pointer using cells_mut() (violates invariants deliberately).
        {
            let bad_cell_mut = tds.cells_mut().get_mut(bad_cell_key).unwrap();
            let mut neighbors = crate::core::collections::NeighborBuffer::new();
            neighbors.push(Some(removed_target_key));
            neighbors.push(None);
            neighbors.push(None);
            bad_cell_mut.neighbors = Some(neighbors);
        }

        let removed_count = tds.repair_degenerate_cells();
        assert_eq!(
            removed_count, 1,
            "Expected exactly 1 degenerate cell removed (bad_cell with dangling neighbor), got {removed_count}",
        );

        assert_eq!(tds.number_of_cells(), 1);
        assert!(tds.cells.contains_key(good_cell_key));
        assert!(!tds.cells.contains_key(bad_cell_key));

        assert_eq!(tds.repair_degenerate_cells(), 0);
        assert!(tds.is_valid().is_ok());

        println!("✓ repair_degenerate_cells removes cells with dangling neighbor pointers");
    }

    #[test]
    fn test_add_vertex_basic_insertion_succeeds() {
        let initial_vertices = initial_simplex_vertices_3d();
        let mut dt = DelaunayTriangulation::new(&initial_vertices).unwrap();

        let vertex = vertex!([1.0, 2.0, 3.0]);
        let result = dt.insert(vertex);

        assert!(result.is_ok(), "Basic vertex addition should succeed");
        assert_eq!(dt.number_of_vertices(), 5);
    }

    #[test]
    fn test_add_vertex_duplicate_coordinates_rejected() {
        let initial_vertices = initial_simplex_vertices_3d();
        let mut dt = DelaunayTriangulation::new(&initial_vertices).unwrap();

        let vertex = vertex!([1.0, 2.0, 3.0]);
        let duplicate = vertex!([1.0, 2.0, 3.0]);
        dt.insert(vertex).unwrap();

        // Same coordinates again (distinct UUID, constructed via vertex! macro)
        let result = dt.insert(duplicate);
        assert!(
            matches!(result, Err(InsertionError::DuplicateCoordinates { .. })),
            "insert() should reject duplicate coordinates created via vertex! (before UUID), got: {result:?}"
        );
    }

    #[test]
    fn test_add_vertex_duplicate_uuid_rejected() {
        let initial_vertices = initial_simplex_vertices_3d();
        let mut dt = DelaunayTriangulation::new(&initial_vertices).unwrap();

        let vertex1 = vertex!([1.0, 2.0, 3.0]);
        let uuid1 = vertex1.uuid();
        dt.insert(vertex1).unwrap();

        let vertex2 = create_vertex_with_uuid(Point::new([4.0, 5.0, 6.0]), uuid1, None);
        let result = dt.insert(vertex2);
        assert!(
            matches!(
                result,
                Err(InsertionError::Construction(
                    crate::core::triangulation::TriangulationConstructionError::Tds(
                        TdsConstructionError::DuplicateUuid {
                            entity: EntityKind::Vertex,
                            ..
                        },
                    ),
                ))
            ),
            "Same UUID with different coordinates should fail with DuplicateUuid"
        );
    }

    #[test]
    fn test_add_vertex_increases_counts_and_leaves_tds_valid() {
        let initial_vertices = initial_simplex_vertices_3d();
        let mut dt = DelaunayTriangulation::new(&initial_vertices).unwrap();
        let initial_cell_count = dt.number_of_cells();

        let new_vertex = vertex!([0.5, 0.5, 0.5]);
        dt.insert(new_vertex).unwrap();

        assert_eq!(dt.number_of_vertices(), 5);
        assert!(
            dt.number_of_cells() >= initial_cell_count,
            "Cell count should not decrease"
        );
        assert!(
            dt.as_triangulation().tds.is_valid().is_ok(),
            "TDS should remain valid"
        );
    }

    #[test]
    fn test_add_vertex_is_accessible_by_uuid_and_coordinates() {
        let initial_vertices = initial_simplex_vertices_3d();
        let mut dt = DelaunayTriangulation::new(&initial_vertices).unwrap();

        let vertex = vertex!([1.0, 2.0, 3.0]);
        let uuid = vertex.uuid();
        dt.insert(vertex).unwrap();

        // Vertex should be findable by UUID.
        let vertex_key = dt.as_triangulation().tds.vertex_key_from_uuid(&uuid);
        assert!(
            vertex_key.is_some(),
            "Added vertex should be findable by UUID"
        );

        // Vertex should be in the vertices collection.
        let stored_vertex = dt
            .as_triangulation()
            .tds
            .get_vertex_by_key(vertex_key.unwrap())
            .unwrap();
        let coords = *stored_vertex.point().coords();
        let expected = [1.0, 2.0, 3.0];
        assert!(
            coords
                .iter()
                .zip(expected.iter())
                .all(|(a, b)| (a - b).abs() < 1e-10),
            "Stored coordinates should match: got {coords:?}, expected {expected:?}"
        );
    }

    // DELETED: test_add_vertex_rollback_on_algorithm_failure and test_add_vertex_atomic_rollback_on_topology_failure
    // These tests used deprecated internal Tds APIs (tds.add() and rollback_vertex_insertion()) that no longer exist.
    // The new cavity-based incremental insertion uses a different approach without explicit rollback.
    // Rollback is an internal implementation detail handled within DelaunayTriangulation::insert().

    // =============================================================================
    // VERTEX REMOVAL TESTS
    // =============================================================================

    #[test]
    fn test_remove_vertex_maintains_topology_consistency() {
        // Test that remove_vertex properly clears dangling neighbor references
        // Create a triangulation with multiple cells
        let vertices = [
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.5, 1.0]),
            vertex!([1.5, 1.0]),
        ];
        let mut dt = DelaunayTriangulation::new(&vertices).unwrap();

        // Verify initial state
        let initial_vertices = dt.number_of_vertices();
        let initial_cells = dt.number_of_cells();
        assert_eq!(initial_vertices, 4);
        assert!(initial_cells > 0);

        // Get a vertex to remove (not a corner vertex, to ensure we have remaining cells)
        let vertex_to_remove = *dt.vertices().next().unwrap().1;
        let vertex_uuid = vertex_to_remove.uuid();

        // Remove the vertex and all cells containing it
        let cells_removed = dt.remove_vertex(&vertex_to_remove).unwrap();

        // Verify the vertex was removed
        assert!(
            dt.as_triangulation()
                .tds
                .vertex_key_from_uuid(&vertex_uuid)
                .is_none(),
            "Vertex should be removed from TDS"
        );
        assert!(
            cells_removed > 0,
            "At least one cell should have been removed"
        );
        assert_eq!(
            dt.number_of_vertices(),
            initial_vertices - 1,
            "Vertex count should decrease by 1"
        );
        assert!(
            dt.number_of_cells() < initial_cells,
            "Cell count should decrease"
        );

        // CRITICAL: Verify that no dangling neighbor references exist
        // This is the key test for the bug fix
        for (cell_key, cell) in dt.cells() {
            if let Some(neighbors) = cell.neighbors() {
                for (i, neighbor_opt) in neighbors.iter().enumerate() {
                    if let Some(neighbor_key) = neighbor_opt {
                        assert!(
                            dt.as_triangulation().tds.cells.contains_key(*neighbor_key),
                            "Cell {cell_key:?} has dangling neighbor reference at index {i}: {neighbor_key:?}"
                        );
                    }
                }
            }
        }

        // Verify the TDS is valid (this should pass with the bug fix)
        assert!(
            dt.as_triangulation().tds.is_valid().is_ok(),
            "TDS should be valid after removing vertex"
        );

        println!("✓ remove_vertex maintains topology consistency");
    }

    #[test]
    fn test_remove_vertex_nonexistent() {
        // Test removing a vertex that doesn't exist
        let vertices = [
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let mut dt = DelaunayTriangulation::new(&vertices).unwrap();

        // Create a vertex that was never added
        let nonexistent_vertex = vertex!([5.0, 5.0]);

        let initial_vertices = dt.number_of_vertices();
        let initial_cells = dt.number_of_cells();

        // Remove should return 0 (no cells removed)
        let cells_removed = dt.remove_vertex(&nonexistent_vertex).unwrap();

        assert_eq!(cells_removed, 0, "No cells should be removed");
        assert_eq!(
            dt.number_of_vertices(),
            initial_vertices,
            "Vertex count should not change"
        );
        assert_eq!(
            dt.number_of_cells(),
            initial_cells,
            "Cell count should not change"
        );

        println!("✓ remove_vertex handles nonexistent vertex correctly");
    }

    #[test]
    fn test_remove_vertex_multiple_dimensions() {
        // Test remove_vertex in different dimensions

        // 2D test
        {
            let vertices_2d = [
                vertex!([0.0, 0.0]),
                vertex!([1.0, 0.0]),
                vertex!([0.0, 1.0]),
                vertex!([1.0, 1.0]),
            ];
            let mut dt_2d: DelaunayTriangulation<_, (), (), 2> =
                DelaunayTriangulation::new(&vertices_2d).unwrap();
            let vertex = *dt_2d.vertices().next().unwrap().1;
            let cells_removed = dt_2d.remove_vertex(&vertex).unwrap();
            assert!(cells_removed > 0);
            assert!(dt_2d.as_triangulation().tds.is_valid().is_ok());
        }

        // 3D test
        {
            let vertices_3d = [
                vertex!([0.0, 0.0, 0.0]),
                vertex!([1.0, 0.0, 0.0]),
                vertex!([0.0, 1.0, 0.0]),
                vertex!([0.0, 0.0, 1.0]),
                vertex!([1.0, 1.0, 1.0]),
            ];
            let mut dt_3d: DelaunayTriangulation<_, (), (), 3> =
                DelaunayTriangulation::new(&vertices_3d).unwrap();
            let vertex = *dt_3d.vertices().next().unwrap().1;
            let cells_removed = dt_3d.remove_vertex(&vertex).unwrap();
            assert!(cells_removed > 0);
            assert!(dt_3d.as_triangulation().tds.is_valid().is_ok());
        }

        // 4D test
        {
            let vertices_4d = [
                vertex!([0.0, 0.0, 0.0, 0.0]),
                vertex!([1.0, 0.0, 0.0, 0.0]),
                vertex!([0.0, 1.0, 0.0, 0.0]),
                vertex!([0.0, 0.0, 1.0, 0.0]),
                vertex!([0.0, 0.0, 0.0, 1.0]),
                vertex!([1.0, 1.0, 1.0, 1.0]),
            ];
            let mut dt_4d: DelaunayTriangulation<_, (), (), 4> =
                DelaunayTriangulation::new(&vertices_4d).unwrap();
            let vertex = *dt_4d.vertices().next().unwrap().1;
            let cells_removed = dt_4d.remove_vertex(&vertex).unwrap();
            assert!(cells_removed > 0);
            assert!(dt_4d.as_triangulation().tds.is_valid().is_ok());
        }

        println!("✓ remove_vertex works correctly in multiple dimensions");
    }

    #[test]
    fn test_remove_vertex_no_dangling_references() {
        // Test that after removing a vertex:
        // 1. No cells contain the deleted vertex
        // 2. No vertices have incident_cell pointing to a removed cell
        // 3. All remaining incident_cell pointers are valid

        let vertices = [
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            // Interior vertex to remove; offset from circumcenter to avoid degenerate configuration
            vertex!([0.2, 0.2, 0.2]),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Find the interior vertex by coordinates (order-independent)
        let interior_coords = [0.2, 0.2, 0.2];
        let (removed_vertex_key, removed_vertex_uuid) = dt
            .vertices()
            .find(|(_, v)| {
                v.point()
                    .coords()
                    .as_slice()
                    .iter()
                    .zip(&interior_coords)
                    .all(|(a, b)| (a - b).abs() < 1e-10)
            })
            .map(|(k, v)| (k, v.uuid()))
            .expect("Interior vertex should exist");

        let vertex_to_remove = *dt
            .vertices()
            .find(|(k, _)| *k == removed_vertex_key)
            .unwrap()
            .1;

        // Remove the vertex
        let cells_removed = dt.remove_vertex(&vertex_to_remove).unwrap();
        assert!(cells_removed > 0, "Should have removed at least one cell");

        // CRITICAL CHECK 1: No cells should contain the deleted vertex
        for (cell_key, cell) in dt.cells() {
            for &vk in cell.vertices() {
                assert_ne!(
                    vk, removed_vertex_key,
                    "Cell {cell_key:?} still references deleted vertex {removed_vertex_key:?}"
                );
            }
        }

        // CRITICAL CHECK 2: The vertex should no longer exist in TDS
        assert!(
            dt.as_triangulation()
                .tds
                .vertex_key_from_uuid(&removed_vertex_uuid)
                .is_none(),
            "Deleted vertex UUID should not be in mapping"
        );
        assert!(
            dt.as_triangulation()
                .tds
                .get_vertex_by_key(removed_vertex_key)
                .is_none(),
            "Deleted vertex key should not exist in storage"
        );

        // CRITICAL CHECK 3: All remaining vertices should have valid incident_cell pointers
        for (vertex_key, vertex) in dt.vertices() {
            if let Some(incident_cell_key) = vertex.incident_cell {
                assert!(
                    dt.as_triangulation()
                        .tds
                        .cells
                        .contains_key(incident_cell_key),
                    "Vertex {vertex_key:?} has dangling incident_cell pointer to {incident_cell_key:?}"
                );

                // Verify the incident cell actually contains this vertex
                let incident_cell = dt
                    .as_triangulation()
                    .tds
                    .get_cell(incident_cell_key)
                    .unwrap();
                assert!(
                    incident_cell.contains_vertex(vertex_key),
                    "Vertex {vertex_key:?} incident_cell {incident_cell_key:?} does not contain the vertex"
                );
            }
        }

        // CRITICAL CHECK 4: TDS should be valid
        assert!(
            dt.as_triangulation().tds.is_valid().is_ok(),
            "TDS should be valid after vertex removal"
        );

        println!("✓ remove_vertex leaves no dangling references to deleted vertex");
    }

    #[test]
    fn test_find_cells_containing_vertex() {
        // Test the helper function that finds all cells containing a specific vertex
        let vertices = [
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.5, 0.5, 0.5]), // Interior vertex
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Pick a vertex known to belong to at least one cell (from an existing cell)
        let first_cell_key = dt.cells().next().unwrap().0;
        let some_vertex_key = dt
            .as_triangulation()
            .tds
            .get_cell_vertices(first_cell_key)
            .unwrap()[0];
        let cells_with_vertex = dt
            .as_triangulation()
            .tds
            .find_cells_containing_vertex(some_vertex_key);

        // Verify the result
        assert!(
            !cells_with_vertex.is_empty(),
            "Vertex should be in at least one cell"
        );

        // Verify all returned cells actually contain the vertex
        for &cell_key in &cells_with_vertex {
            let cell = dt.as_triangulation().tds.get_cell(cell_key).unwrap();
            assert!(
                cell.contains_vertex(some_vertex_key),
                "Cell {cell_key:?} should contain vertex {some_vertex_key:?}"
            );
        }

        // Verify we found ALL cells containing this vertex
        let expected_count = dt
            .cells()
            .filter(|(_, cell)| cell.contains_vertex(some_vertex_key))
            .count();
        assert_eq!(
            cells_with_vertex.len(),
            expected_count,
            "Should find all cells containing the vertex"
        );

        // Test finding cells for another vertex
        let another_vertex_key = dt.vertices().nth(1).map(|(k, _)| k).unwrap();
        let cells_with_another = dt
            .as_triangulation()
            .tds
            .find_cells_containing_vertex(another_vertex_key);
        assert!(
            !cells_with_another.is_empty(),
            "Another vertex should also be in at least one cell"
        );

        println!(
            "✓ find_cells_containing_vertex correctly identifies all cells containing a vertex"
        );
    }

    #[test]
    fn test_assign_neighbors_errors_on_missing_vertex_key() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();
        let a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();

        let cell_key = tds
            .insert_cell_with_mapping(Cell::new(vec![a, b, c], None).unwrap())
            .unwrap();

        // Corrupt the cell by inserting a vertex key that doesn't exist in the TDS.
        let invalid_vkey = VertexKey::from(KeyData::from_ffi(u64::MAX));
        tds.get_cell_by_key_mut(cell_key)
            .unwrap()
            .push_vertex_key(invalid_vkey);

        let err = tds.assign_neighbors().unwrap_err();
        assert!(matches!(err, TdsError::VertexKeyRetrievalFailed { .. }));
    }

    #[test]
    fn test_assign_neighbors_errors_on_non_manifold_facet_sharing() {
        // Three triangles sharing the same edge (v_a,v_b) is non-manifold in 2D.
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();
        let v_a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v_b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v_c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(vertex!([0.0, -1.0]))
            .unwrap();
        let v_e = tds.insert_vertex_with_mapping(vertex!([2.0, 0.0])).unwrap();

        tds.insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_c], None).unwrap())
            .unwrap();
        tds.insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_d], None).unwrap())
            .unwrap();
        tds.insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_e], None).unwrap())
            .unwrap();

        let err = tds.assign_neighbors().unwrap_err();
        assert!(matches!(err, TdsError::InconsistentDataStructure { .. }));
    }

    #[test]
    fn test_remove_cells_by_keys_empty_is_noop() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();
        let gen_before = tds.generation();
        assert_eq!(tds.remove_cells_by_keys(&[]), 0);
        assert_eq!(tds.generation(), gen_before);
    }

    #[test]
    fn test_remove_cells_by_keys_clears_neighbor_pointers() {
        // Two triangles sharing an edge.
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();
        let a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let d = tds.insert_vertex_with_mapping(vertex!([1.0, 1.0])).unwrap();

        let cell1 = tds
            .insert_cell_with_mapping(Cell::new(vec![a, b, c], None).unwrap())
            .unwrap();
        let cell2 = tds
            .insert_cell_with_mapping(Cell::new(vec![b, d, c], None).unwrap())
            .unwrap();

        // Build neighbor pointers based on facet sharing.
        tds.assign_neighbors().unwrap();

        // Sanity: at least one neighbor pointer should exist before removal.
        assert!(
            tds.get_cell(cell1)
                .unwrap()
                .neighbors()
                .is_some_and(|n| n.iter().any(Option::is_some))
        );

        let gen_before = tds.generation();
        assert_eq!(tds.remove_cells_by_keys(&[cell2]), 1);
        assert_eq!(tds.generation(), gen_before + 1);

        // All remaining neighbor pointers must not reference the removed cell.
        for (_, cell) in tds.cells() {
            if let Some(neighbors) = cell.neighbors() {
                for neighbor_opt in neighbors {
                    assert_ne!(*neighbor_opt, Some(cell2));
                }
            }
        }
    }

    #[test]
    fn test_remove_cells_by_keys_repairs_incident_cells_for_affected_vertices() {
        // Two triangles sharing an edge (B,C). Remove one and ensure incident_cell pointers are
        // updated without requiring a full assign_incident_cells() rebuild.
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let d = tds.insert_vertex_with_mapping(vertex!([1.0, 1.0])).unwrap();

        let cell1 = tds
            .insert_cell_with_mapping(Cell::new(vec![a, b, c], None).unwrap())
            .unwrap();
        let cell2 = tds
            .insert_cell_with_mapping(Cell::new(vec![b, d, c], None).unwrap())
            .unwrap();

        tds.assign_neighbors().unwrap();

        // Force deterministic incident_cell pointers that require repair.
        tds.get_vertex_by_key_mut(a).unwrap().incident_cell = Some(cell1);
        tds.get_vertex_by_key_mut(b).unwrap().incident_cell = Some(cell1);
        tds.get_vertex_by_key_mut(c).unwrap().incident_cell = Some(cell1);
        tds.get_vertex_by_key_mut(d).unwrap().incident_cell = Some(cell2);

        assert_eq!(tds.remove_cells_by_keys(&[cell1]), 1);

        // A is now isolated (no remaining cells contain it) => incident_cell must be None.
        assert!(tds.get_vertex_by_key(a).unwrap().incident_cell.is_none());

        // B, C, D are still in cell2 => their incident_cell must be valid and contain them.
        for vk in [b, c, d] {
            let incident = tds
                .get_vertex_by_key(vk)
                .unwrap()
                .incident_cell
                .expect("vertex should have an incident cell after repair");
            assert!(tds.cells.contains_key(incident));
            let cell = tds.get_cell(incident).unwrap();
            assert!(cell.contains_vertex(vk));
        }

        // With remaining cells, isolated vertices are allowed at the TDS structural level.
        assert!(tds.is_valid().is_ok());

        // Neighbor pointers in the surviving cell must not reference the removed cell.
        let cell2_ref = tds.get_cell(cell2).unwrap();
        if let Some(neighbors) = cell2_ref.neighbors() {
            assert!(neighbors.iter().all(|n| *n != Some(cell1)));
        }
    }

    #[test]
    fn test_tds_remove_vertex_returns_zero_when_vertex_not_in_mapping() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();
        let v = vertex!([0.0, 0.0]);
        assert_eq!(tds.remove_vertex(&v).unwrap(), 0);
    }

    #[test]
    fn test_tds_remove_vertex_repairs_neighbors_and_incident_cells_incrementally() {
        // Two triangles sharing an edge (east,north). Remove the origin (only in cell1) and ensure:
        // - cell1 is removed
        // - neighbor back-references in cell2 are cleared
        // - incident_cell pointers remain valid for remaining vertices without a full rebuild
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let origin_key = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let east_key = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let north_key = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let diagonal_key = tds.insert_vertex_with_mapping(vertex!([1.0, 1.0])).unwrap();

        let cell1 = tds
            .insert_cell_with_mapping(
                Cell::new(vec![origin_key, east_key, north_key], None).unwrap(),
            )
            .unwrap();
        let cell2 = tds
            .insert_cell_with_mapping(
                Cell::new(vec![east_key, diagonal_key, north_key], None).unwrap(),
            )
            .unwrap();

        tds.assign_neighbors().unwrap();

        // Seed incident_cell pointers:
        // - ORIGIN points at cell1 so star discovery can use the neighbor-walk fast path.
        // - EAST/NORTH point at cell1 so removal must repair them to cell2.
        // - DIAGONAL points at cell2 and should remain valid.
        tds.get_vertex_by_key_mut(origin_key).unwrap().incident_cell = Some(cell1);
        tds.get_vertex_by_key_mut(east_key).unwrap().incident_cell = Some(cell1);
        tds.get_vertex_by_key_mut(north_key).unwrap().incident_cell = Some(cell1);
        tds.get_vertex_by_key_mut(diagonal_key)
            .unwrap()
            .incident_cell = Some(cell2);

        let vertex_to_remove = *tds.get_vertex_by_key(origin_key).unwrap();
        let removed = tds.remove_vertex(&vertex_to_remove).unwrap();
        assert_eq!(removed, 1);

        // The removed vertex should be gone.
        assert!(tds.vertex_key_from_uuid(&vertex_to_remove.uuid()).is_none());
        assert!(tds.get_vertex_by_key(origin_key).is_none());

        // cell2 should remain and must not reference cell1 as a neighbor.
        assert!(tds.cells.contains_key(cell2));
        let cell2_ref = tds.get_cell(cell2).unwrap();
        if let Some(neighbors) = cell2_ref.neighbors() {
            assert!(neighbors.iter().all(|n| *n != Some(cell1)));
        }

        // Remaining vertices must have valid incident_cell pointers (if present).
        for vertex_key in [east_key, north_key, diagonal_key] {
            let v = tds.get_vertex_by_key(vertex_key).unwrap();
            let Some(incident) = v.incident_cell else {
                panic!("vertex {vertex_key:?} should have an incident cell after removal");
            };
            assert!(tds.cells.contains_key(incident));
            assert!(tds.get_cell(incident).unwrap().contains_vertex(vertex_key));
        }
    }

    #[test]
    fn test_find_neighbors_by_key_returns_none_buffer_for_missing_cell() {
        let tds: Tds<f64, (), (), 2> = Tds::empty();
        let missing = CellKey::from(KeyData::from_ffi(u64::MAX));
        let neighbors = tds.find_neighbors_by_key(missing);
        assert_eq!(neighbors.len(), 3);
        assert!(neighbors.iter().all(Option::is_none));
    }

    #[test]
    fn test_set_neighbors_by_key_rejects_non_neighbor_and_wrong_slot() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let v_a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v_b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v_c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let v_d = tds.insert_vertex_with_mapping(vertex!([1.0, 1.0])).unwrap();
        let v_e = tds.insert_vertex_with_mapping(vertex!([2.0, 2.0])).unwrap();

        let cell1 = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_c], None).unwrap())
            .unwrap();

        // A cell that shares only one vertex with cell1 => not a neighbor in 2D.
        let cell_far = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_d, v_e], None).unwrap())
            .unwrap();
        let err = tds
            .set_neighbors_by_key(cell1, &[Some(cell_far), None, None])
            .unwrap_err()
            .0;
        assert!(matches!(err, TdsError::InvalidNeighbors { .. }));

        // A true facet-neighbor (shares {v_a,v_b}) placed at the wrong facet index.
        let cell2 = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_d], None).unwrap())
            .unwrap();
        let err = tds
            .set_neighbors_by_key(cell1, &[Some(cell2), None, None])
            .unwrap_err()
            .0;
        assert!(matches!(err, TdsError::InvalidNeighbors { .. }));
    }

    // =============================================================================
    // NEIGHBOR VALIDATION HELPER TESTS
    // =============================================================================

    #[test]
    fn test_build_cell_vertex_sets_success() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let v_a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v_b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v_c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let v_d = tds.insert_vertex_with_mapping(vertex!([1.0, 1.0])).unwrap();

        let cell1 = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_c], None).unwrap())
            .unwrap();
        let cell2 = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_c, v_d], None).unwrap())
            .unwrap();

        let map = tds.build_cell_vertex_sets().unwrap();
        assert_eq!(map.len(), 2);

        let expected1: VertexKeySet = [v_a, v_b, v_c].into_iter().collect();
        let expected2: VertexKeySet = [v_a, v_c, v_d].into_iter().collect();

        assert_eq!(map.get(&cell1), Some(&expected1));
        assert_eq!(map.get(&cell2), Some(&expected2));
    }

    #[test]
    fn test_build_cell_vertex_sets_errors_on_missing_vertex_key() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let v_a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v_b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v_c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();

        let cell = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_c], None).unwrap())
            .unwrap();

        // Corrupt the cell by inserting a vertex key that doesn't exist in the TDS.
        let invalid_vkey = VertexKey::from(KeyData::from_ffi(u64::MAX));
        tds.get_cell_by_key_mut(cell)
            .unwrap()
            .push_vertex_key(invalid_vkey);

        let err = tds.build_cell_vertex_sets().unwrap_err();
        assert!(matches!(
            err,
            TdsError::InconsistentDataStructure { message }
                if message.contains("references non-existent vertex key")
        ));
    }

    #[test]
    fn test_validate_shared_facet_count_ok_for_true_neighbors() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let v_a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v_b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v_c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let v_d = tds.insert_vertex_with_mapping(vertex!([1.0, 1.0])).unwrap();

        let cell1_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_c], None).unwrap())
            .unwrap();
        let cell2_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_d], None).unwrap())
            .unwrap();

        let cell1 = tds.get_cell(cell1_key).unwrap();
        let cell2 = tds.get_cell(cell2_key).unwrap();

        let this_vertices: VertexKeySet = [v_a, v_b, v_c].into_iter().collect();
        let neighbor_vertices: VertexKeySet = [v_a, v_b, v_d].into_iter().collect();

        assert!(
            Tds::validate_shared_facet_count(cell1, cell2, &this_vertices, &neighbor_vertices)
                .is_ok()
        );
    }

    #[test]
    fn test_validate_shared_facet_count_errors_for_non_neighbors() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let v_a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v_b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v_c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let v_d = tds.insert_vertex_with_mapping(vertex!([1.0, 1.0])).unwrap();
        let v_e = tds.insert_vertex_with_mapping(vertex!([2.0, 2.0])).unwrap();

        let cell1_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_c], None).unwrap())
            .unwrap();
        let cell_far_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_d, v_e], None).unwrap())
            .unwrap();

        let cell1 = tds.get_cell(cell1_key).unwrap();
        let cell_far = tds.get_cell(cell_far_key).unwrap();

        let this_vertices: VertexKeySet = [v_a, v_b, v_c].into_iter().collect();
        let far_vertices: VertexKeySet = [v_a, v_d, v_e].into_iter().collect();

        let cell1_uuid = cell1.uuid();
        let cell_far_uuid = cell_far.uuid();

        let err = Tds::validate_shared_facet_count(cell1, cell_far, &this_vertices, &far_vertices)
            .unwrap_err();

        assert!(matches!(
            err,
            TdsError::NotNeighbors { cell1: c1, cell2: c2 }
                if c1 == cell1_uuid && c2 == cell_far_uuid
        ));
    }

    #[test]
    fn test_compute_expected_mirror_facet_index_returns_unique_vertex_index() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let v_a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v_b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v_c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let v_d = tds.insert_vertex_with_mapping(vertex!([1.0, 1.0])).unwrap();

        let cell1_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_c], None).unwrap())
            .unwrap();
        // Put the unique vertex at index 0 to ensure we test the returned index.
        let cell2_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v_d, v_a, v_b], None).unwrap())
            .unwrap();

        let cell1 = tds.get_cell(cell1_key).unwrap();
        let cell2 = tds.get_cell(cell2_key).unwrap();

        let this_vertices: VertexKeySet = [v_a, v_b, v_c].into_iter().collect();

        let idx = Tds::compute_expected_mirror_facet_index(cell1, cell2, &this_vertices).unwrap();
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_compute_expected_mirror_facet_index_errors_when_ambiguous() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let v_a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v_b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v_c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let v_d = tds.insert_vertex_with_mapping(vertex!([1.0, 1.0])).unwrap();
        let v_e = tds.insert_vertex_with_mapping(vertex!([2.0, 2.0])).unwrap();

        let cell1_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_c], None).unwrap())
            .unwrap();
        // Shares only v_a -> differs by 2 vertices -> ambiguous mirror facet.
        let cell2_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_d, v_e], None).unwrap())
            .unwrap();

        let cell1 = tds.get_cell(cell1_key).unwrap();
        let cell2 = tds.get_cell(cell2_key).unwrap();

        let this_vertices: VertexKeySet = [v_a, v_b, v_c].into_iter().collect();

        let err =
            Tds::compute_expected_mirror_facet_index(cell1, cell2, &this_vertices).unwrap_err();

        assert!(matches!(
            err,
            TdsError::InvalidNeighbors { message } if message.contains("ambiguous")
        ));
    }

    #[test]
    fn test_compute_expected_mirror_facet_index_errors_when_duplicate_cells() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let v_a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v_b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v_c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();

        let cell1_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_c], None).unwrap())
            .unwrap();
        // Duplicate by vertices (different UUID) -> no unique vertex to identify mirror facet.
        let cell2_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_c], None).unwrap())
            .unwrap();

        let cell1 = tds.get_cell(cell1_key).unwrap();
        let cell2 = tds.get_cell(cell2_key).unwrap();

        let this_vertices: VertexKeySet = [v_a, v_b, v_c].into_iter().collect();

        let err =
            Tds::compute_expected_mirror_facet_index(cell1, cell2, &this_vertices).unwrap_err();

        assert!(matches!(
            err,
            TdsError::InvalidNeighbors { message } if message.contains("share all vertices")
        ));
    }

    #[test]
    fn test_compute_and_verify_mirror_facet_ok() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let v_a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v_b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v_c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let v_d = tds.insert_vertex_with_mapping(vertex!([1.0, 1.0])).unwrap();

        let cell1_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_c], None).unwrap())
            .unwrap();
        let cell2_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_d], None).unwrap())
            .unwrap();

        let cell1 = tds.get_cell(cell1_key).unwrap();
        let cell2 = tds.get_cell(cell2_key).unwrap();

        let this_vertices: VertexKeySet = [v_a, v_b, v_c].into_iter().collect();

        // Shared edge is (v_a, v_b). In cell1, that's opposite vertex index 2 (v_c).
        let mirror_idx =
            Tds::compute_and_verify_mirror_facet(cell1, 2, cell2, &this_vertices).unwrap();
        assert_eq!(mirror_idx, 2);
    }

    #[test]
    fn test_compute_and_verify_mirror_facet_errors_when_no_shared_facet() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let v_a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v_b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v_c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let v_d = tds.insert_vertex_with_mapping(vertex!([1.0, 1.0])).unwrap();

        let cell1_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_c], None).unwrap())
            .unwrap();
        let cell2_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_d], None).unwrap())
            .unwrap();

        let cell1 = tds.get_cell(cell1_key).unwrap();
        let cell2 = tds.get_cell(cell2_key).unwrap();

        let this_vertices: VertexKeySet = [v_a, v_b, v_c].into_iter().collect();

        // facet_idx=0 corresponds to edge (v_b, v_c) in cell1, which is not shared with cell2.
        let err =
            Tds::compute_and_verify_mirror_facet(cell1, 0, cell2, &this_vertices).unwrap_err();

        assert!(matches!(
            err,
            TdsError::InvalidNeighbors { message } if message.contains("Could not find mirror facet")
        ));
    }

    #[test]
    fn test_compute_and_verify_mirror_facet_errors_on_mismatch_with_cross_check() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let v_a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v_b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v_c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let v_d = tds.insert_vertex_with_mapping(vertex!([1.0, 1.0])).unwrap();

        let cell1_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_c], None).unwrap())
            .unwrap();
        let cell2_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_d], None).unwrap())
            .unwrap();

        let cell1 = tds.get_cell(cell1_key).unwrap();
        let cell2 = tds.get_cell(cell2_key).unwrap();

        // Intentionally WRONG vertex set (includes v_d, excludes v_b) to force the mismatch branch.
        // This is a unit-level test of the helper's defensive cross-check behavior.
        let this_vertices_wrong: VertexKeySet = [v_a, v_c, v_d].into_iter().collect();

        let err = Tds::compute_and_verify_mirror_facet(cell1, 2, cell2, &this_vertices_wrong)
            .unwrap_err();

        assert!(matches!(
            err,
            TdsError::InvalidNeighbors { message } if message.contains("index mismatch")
        ));
    }

    #[test]
    fn test_validate_neighbors_errors_on_mirror_facet_index_mismatch() {
        // This test exercises the same "mirror facet index mismatch" defensive branch, but via the
        // neighbor-validation loop used by `validate_neighbors_with_precomputed_vertex_sets()`.
        //
        // The mismatch is only reachable if the precomputed per-cell vertex-set map is inconsistent
        // with the cell's actual vertex buffer (e.g., a bug/corruption in the precompute step). To
        // simulate that scenario deterministically, we build the map normally and then corrupt the
        // entry for one cell before running the validation loop.
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let origin_key = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let east_key = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let north_key = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let diagonal_key = tds.insert_vertex_with_mapping(vertex!([1.0, 1.0])).unwrap();

        let cell1_key = tds
            .insert_cell_with_mapping(
                Cell::new(vec![origin_key, east_key, north_key], None).unwrap(),
            )
            .unwrap();
        let _cell2_key = tds
            .insert_cell_with_mapping(
                Cell::new(vec![origin_key, east_key, diagonal_key], None).unwrap(),
            )
            .unwrap();

        tds.assign_neighbors().unwrap();

        let mut cell_vertices = tds.build_cell_vertex_sets().unwrap();

        // Corrupt the vertex-set entry for cell1 so it no longer matches the actual cell's vertices.
        // (Drop `east_key`, add `diagonal_key`.)
        let corrupted_cell1_vertices: VertexKeySet =
            [origin_key, north_key, diagonal_key].into_iter().collect();
        cell_vertices.insert(cell1_key, corrupted_cell1_vertices);

        let err = tds
            .validate_neighbors_with_precomputed_vertex_sets(&cell_vertices)
            .unwrap_err();

        assert!(matches!(
            err,
            TdsError::InvalidNeighbors { message } if message.contains("index mismatch")
        ));
    }

    #[test]
    fn test_validate_shared_facet_vertices_ok() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let v_a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v_b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v_c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let v_d = tds.insert_vertex_with_mapping(vertex!([1.0, 1.0])).unwrap();

        let cell1_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_c], None).unwrap())
            .unwrap();
        let cell2_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_d], None).unwrap())
            .unwrap();

        let cell1 = tds.get_cell(cell1_key).unwrap();
        let cell2 = tds.get_cell(cell2_key).unwrap();

        let this_vertices: VertexKeySet = [v_a, v_b, v_c].into_iter().collect();
        let neighbor_vertices: VertexKeySet = [v_a, v_b, v_d].into_iter().collect();

        assert!(
            Tds::validate_shared_facet_vertices(
                cell1,
                2, // opposite v_c => shared edge {v_a,v_b}
                cell2,
                2, // opposite v_d => shared edge {v_a,v_b}
                &this_vertices,
                &neighbor_vertices,
            )
            .is_ok()
        );
    }

    #[test]
    fn test_validate_shared_facet_vertices_errors_when_mirror_index_wrong() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let v_a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v_b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v_c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let v_d = tds.insert_vertex_with_mapping(vertex!([1.0, 1.0])).unwrap();

        let cell1_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_c], None).unwrap())
            .unwrap();
        let cell2_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_d], None).unwrap())
            .unwrap();

        let cell1 = tds.get_cell(cell1_key).unwrap();
        let cell2 = tds.get_cell(cell2_key).unwrap();

        let this_vertices: VertexKeySet = [v_a, v_b, v_c].into_iter().collect();
        let neighbor_vertices: VertexKeySet = [v_a, v_b, v_d].into_iter().collect();

        // mirror_idx=0 is intentionally wrong here; it treats vertex v_a as the "opposite"
        // vertex, which makes v_d incorrectly part of the "shared facet".
        let err = Tds::validate_shared_facet_vertices(
            cell1,
            2, // correct for cell1
            cell2,
            0, // intentionally wrong for cell2
            &this_vertices,
            &neighbor_vertices,
        )
        .unwrap_err();

        assert!(matches!(
            err,
            TdsError::InvalidNeighbors { message } if message.contains("Shared facet mismatch")
        ));
    }

    #[test]
    fn test_validate_mutual_neighbor_back_reference_ok() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let v_a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v_b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v_c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let v_d = tds.insert_vertex_with_mapping(vertex!([1.0, 1.0])).unwrap();

        let cell1_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_c], None).unwrap())
            .unwrap();
        let cell2_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_d], None).unwrap())
            .unwrap();

        // Build neighbor pointers so mutual back-references exist.
        tds.assign_neighbors().unwrap();

        let cell1 = tds.get_cell(cell1_key).unwrap();
        let cell2 = tds.get_cell(cell2_key).unwrap();

        assert!(
            Tds::validate_mutual_neighbor_back_reference(
                cell1_key, cell1, 2, // opposite v_c => shared edge {v_a,v_b}
                cell2, 2, // opposite v_d => shared edge {v_a,v_b}
            )
            .is_ok()
        );
    }

    #[test]
    fn test_validate_mutual_neighbor_back_reference_errors_when_neighbor_has_no_neighbors() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let v_a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v_b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v_c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let v_d = tds.insert_vertex_with_mapping(vertex!([1.0, 1.0])).unwrap();

        let cell1_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_c], None).unwrap())
            .unwrap();
        let cell2_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_d], None).unwrap())
            .unwrap();

        // NOTE: We intentionally do NOT call assign_neighbors(), so neighbor_cell.neighbors is None.
        let cell1 = tds.get_cell(cell1_key).unwrap();
        let cell2 = tds.get_cell(cell2_key).unwrap();

        let err = Tds::validate_mutual_neighbor_back_reference(cell1_key, cell1, 2, cell2, 2)
            .unwrap_err();

        assert!(matches!(
            err,
            TdsError::InvalidNeighbors { message } if message.contains("neighbor has no neighbors")
        ));
    }

    #[test]
    fn test_validate_mutual_neighbor_back_reference_errors_when_back_reference_missing() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let v_a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v_b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v_c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let v_d = tds.insert_vertex_with_mapping(vertex!([1.0, 1.0])).unwrap();

        let cell1_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_c], None).unwrap())
            .unwrap();
        let cell2_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v_a, v_b, v_d], None).unwrap())
            .unwrap();

        // Build neighbor pointers first, then deliberately corrupt the back-reference.
        tds.assign_neighbors().unwrap();

        {
            let cell2_mut = tds.get_cell_by_key_mut(cell2_key).unwrap();
            let neighbors = cell2_mut
                .neighbors
                .as_mut()
                .expect("cell2 should have neighbors after assign_neighbors()");
            // For (v_a, v_b, v_d), the shared edge with cell1 is opposite v_d => index 2.
            neighbors[2] = None;
        }

        let cell1 = tds.get_cell(cell1_key).unwrap();
        let cell2 = tds.get_cell(cell2_key).unwrap();

        let err = Tds::validate_mutual_neighbor_back_reference(cell1_key, cell1, 2, cell2, 2)
            .unwrap_err();

        assert!(matches!(
            err,
            TdsError::InvalidNeighbors { message } if message.contains("expected back-reference")
        ));
    }

    #[test]
    fn test_assign_incident_cells_clears_incident_cell_when_no_cells() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();
        let vkey = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();

        // Corrupt incident_cell and ensure it gets cleared.
        tds.get_vertex_by_key_mut(vkey).unwrap().incident_cell =
            Some(CellKey::from(KeyData::from_ffi(u64::MAX)));
        assert!(tds.get_vertex_by_key(vkey).unwrap().incident_cell.is_some());

        tds.assign_incident_cells().unwrap();
        assert!(tds.get_vertex_by_key(vkey).unwrap().incident_cell.is_none());
    }

    #[test]
    fn test_build_facet_to_cells_map_errors_on_u8_facet_index_overflow() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();
        let a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();

        let cell_key = tds
            .insert_cell_with_mapping(Cell::new(vec![a, b, c], None).unwrap())
            .unwrap();

        // NOTE: This scenario is not realistic for valid triangulations:
        // - Valid D-cells always contain exactly D+1 vertices (and `D <= MAX_PRACTICAL_DIMENSION_SIZE`).
        // - Therefore facet indices are always within `0..=D` and trivially fit in `u8`.
        //
        // We intentionally *corrupt* the cell here by inflating its vertex buffer (still using valid
        // vertex keys) so facet indices exceed `u8::MAX`. This is a robustness test to ensure
        // `build_facet_to_cells_map()` fails fast on corrupted/invalid TDS state (e.g., unsafe internal
        // mutation or malformed serialized data).
        //
        // Inflate the vertex buffer (still using valid vertex keys) so facet indices exceed u8.
        {
            let cell = tds.get_cell_by_key_mut(cell_key).unwrap();
            while cell.number_of_vertices() <= usize::from(u8::MAX) + 1 {
                cell.push_vertex_key(a);
            }
        }

        let err = tds.build_facet_to_cells_map().unwrap_err();
        assert!(matches!(
            err,
            TdsError::InconsistentDataStructure { message }
                if message.contains("Facet index")
        ));
    }

    #[test]
    fn test_validate_vertex_and_cell_mappings_detect_inconsistencies() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();
        let a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let cell_key = tds
            .insert_cell_with_mapping(Cell::new(vec![a, b, c], None).unwrap())
            .unwrap();

        // Start from a consistent state.
        assert!(tds.validate_vertex_mappings().is_ok());
        assert!(tds.validate_cell_mappings().is_ok());

        // Break vertex mapping: remove one uuid entry (len mismatch).
        let uuid_a = tds.get_vertex_by_key(a).unwrap().uuid();
        tds.uuid_to_vertex_key.remove(&uuid_a);
        assert!(matches!(
            tds.validate_vertex_mappings(),
            Err(TdsError::MappingInconsistency {
                entity: EntityKind::Vertex,
                ..
            })
        ));

        // Restore length but make the UUID map point at the wrong key.
        tds.uuid_to_vertex_key.insert(uuid_a, b);
        assert!(matches!(
            tds.validate_vertex_mappings(),
            Err(TdsError::MappingInconsistency {
                entity: EntityKind::Vertex,
                ..
            })
        ));

        // Break cell mapping similarly.
        let uuid_cell = tds.get_cell(cell_key).unwrap().uuid();
        tds.uuid_to_cell_key.remove(&uuid_cell);
        assert!(matches!(
            tds.validate_cell_mappings(),
            Err(TdsError::MappingInconsistency {
                entity: EntityKind::Cell,
                ..
            })
        ));
    }

    #[test]
    fn test_validate_cell_vertex_keys_detects_missing_vertices() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();
        let a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();

        let cell_key = tds
            .insert_cell_with_mapping(Cell::new(vec![a, b, c], None).unwrap())
            .unwrap();

        let invalid_vkey = VertexKey::from(KeyData::from_ffi(u64::MAX));
        tds.get_cell_by_key_mut(cell_key)
            .unwrap()
            .push_vertex_key(invalid_vkey);

        let err = tds.validate_cell_vertex_keys().unwrap_err();
        assert!(matches!(err, TdsError::InconsistentDataStructure { .. }));

        // Now wired into structural validation: is_valid() should fail early with the
        // more precise "missing vertex key" diagnostic.
        let err = tds.is_valid().unwrap_err();
        assert!(matches!(
            err,
            TdsError::InconsistentDataStructure { message }
                if message.contains("references non-existent vertex key")
        ));
    }
}
