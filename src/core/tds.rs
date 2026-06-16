//! Data and operations on d-dimensional triangulation data structures.
//!
//! This module provides the `Tds` (Triangulation Data Structure): a key-based,
//! CGAL-inspired representation of the **combinatorial** topology of a D-dimensional
//! finite simplicial complex (vertices, simplices, and adjacency). The implementation
//! closely follows the design principles of
//! [CGAL Triangulation](https://doc.cgal.org/latest/Triangulation/index.html).
//!
//! The crate follows a layered architecture: `Tds` is topology-focused, while geometric
//! predicates and Delaunay-specific operations live in higher layers (`Triangulation` /
//! `DelaunayTriangulation`) and `core::algorithms`.
//!
//! # Key Features
//!
//! - **CGAL-style layering**: topology in `Tds`, geometry/predicates in higher layers
//! - **Relaxed access bounds**: read-only topology accessors and cache identity helpers do
//!   not require coordinate or payload trait bounds; mutation, validation, and serde paths
//!   add only the bounds they need
//! - **Arbitrary Dimensions**: Supports triangulations in any dimension D ≥ 1
//! - **Hierarchical Simplex Structure**: Stores maximal D-dimensional simplices and infers lower-dimensional
//!   simplices (vertices, edges, facets) from the maximal simplices
//! - **Neighbor Relationships**: Maintains adjacency information between simplices for efficient
//!   topological traversal
//! - **Validation Support**: Structural invariant validation (Level 2) plus cumulative element
//!   validation (Levels 1–2)
//! - **Serialization Support**: Serde support for persistence
//! - **Optimized Storage**: Internal key-based storage with UUIDs for external identity
//!
//! # Geometric Structure
//!
//! The triangulation data structure represents a finite simplicial complex where:
//!
//! - **0-simplices**: Individual vertices embedded in D-dimensional Euclidean space
//! - **1-simplices**: Edges connecting two vertices (inferred from maximal simplices)
//! - **2-simplices**: Triangular faces with three vertices (inferred from maximal simplices)
//! - **...**
//! - **D-simplices**: Maximal D-dimensional simplices with D+1 vertices (explicitly stored)
//!
//! For example, in 3D space:
//! - Vertices are 0-dimensional simplices
//! - Edges are 1-dimensional simplices (inferred from tetrahedra)
//! - Faces are 2-dimensional simplices represented as `Facet`s
//! - Tetrahedra are 3-dimensional simplices (maximal simplices)
//!
//! # Delaunay Property
//!
//! When constructed via the Delaunay triangulation algorithm, the structure satisfies
//! the **empty circumsphere property**: no vertex lies inside the circumsphere of any
//! D-dimensional simplex. This property ensures optimal geometric characteristics for
//! many applications including mesh generation, interpolation, and spatial analysis.
//!
//! # Topological Invariants
//!
//! Valid Delaunay triangulations maintain several critical topological invariants:
//!
//! - **Facet Sharing Invariant**: Every facet (D-1 dimensional face) is shared by exactly
//!   two simplices, except for boundary facets which belong to exactly one simplex. This ensures
//!   the triangulation forms a valid simplicial complex.
//! - **Neighbor Consistency**: Adjacent simplices properly reference each other through their
//!   shared facets, maintaining bidirectional neighbor relationships.
//! - **Vertex Incidence**: Each vertex is incident to a well-defined set of simplices that
//!   form a topologically valid star configuration around the vertex.
//! - **Delaunay Property**: No vertex lies inside the circumsphere of any D-dimensional simplex.
//!
//! ## Invariant Enforcement
//!
//! | Invariant Type | Enforcement Location | Method |
//! |---|---|---|
//! | **Delaunay Property** | incremental insertion (`core::algorithms::incremental_insertion`) | Empty circumsphere test via `insphere()` (best-effort) |
//! | **Facet Sharing** | `Tds::is_valid()` / `Tds::validate()` | Each facet shared by ≤ 2 simplices |
//! | **No Duplicate Simplices** | `Tds::is_valid()` / `Tds::validate()` | No simplices with identical vertex sets |
//! | **Neighbor Consistency** | `Tds::is_valid()` / `Tds::validate()` | Mutual neighbor relationships |
//! | **Coherent Orientation** | `Tds::is_valid()` / `Tds::validate()` | Adjacent simplices induce opposite facet orientations |
//! | **Simplex Vertex Keys** | `Tds::is_valid()` / `Tds::validate()` | Simplices reference only valid vertex keys |
//! | **Vertex Incidence** | `Tds::is_valid()` / `Tds::validate()` | `Vertex::incident_simplex` is non-dangling and consistent (when present) |
//! | **Simplex Validity** | `SimplexBuilder::validate()` (vertex count) + `simplex.is_valid()` (comprehensive) | Construction + runtime validation |
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
//! 1. **Level 1: Element Validity** - [`Simplex::is_valid()`], [`Vertex::is_valid()`]
//!    - Basic data integrity (coordinates, UUIDs, initialization)
//! 2. **Level 2: TDS Structural Validity** - [`Tds::is_valid()`] ← **This module**
//!    - UUID ↔ Key mapping consistency
//!    - Simplices reference only valid vertex keys (no stale/missing vertex keys)
//!    - `Vertex::incident_simplex`, when present, must point at an existing simplex that contains the vertex
//!    - Isolated vertices (not referenced by any simplex) are allowed at this layer (`incident_simplex` may be `None`)
//!    - No duplicate simplices
//!    - Coherent orientation (adjacent simplices induce opposite facet orientations)
//!    - Facet sharing invariant (≤2 simplices per facet)
//!    - Neighbor consistency
//! 3. **Level 3: Manifold Topology** - [`Triangulation::is_valid()`]
//!    - Builds on Level 2, and rejects isolated vertices (every vertex must be incident to ≥ 1 simplex)
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
//! use delaunay::prelude::*;
//!
//! # #[derive(Debug, thiserror::Error)]
//! # enum ExampleError {
//! #     #[error(transparent)]
//! #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
//! #     #[error(transparent)]
//! #     Insertion(#[from] delaunay::prelude::insertion::InsertionError),
//! #     #[error(transparent)]
//! #     Tds(#[from] delaunay::prelude::tds::TdsError),
//! #     #[error(transparent)]
//! #     TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
//! #     #[error(transparent)]
//! #     Invariant(#[from] delaunay::prelude::tds::InvariantError),
//! #     #[error(transparent)]
//! #     Facet(#[from] delaunay::prelude::tds::FacetError),
//! #     #[error(transparent)]
//! #     Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
//! #     #[error(transparent)]
//! #     Validation(#[from] delaunay::DelaunayTriangulationValidationError),
//! #     #[error(transparent)]
//! #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
//! # }
//! # fn main() -> Result<(), ExampleError> {
//! let vertices = [
//!     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
//!     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
//!     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
//!     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
//! ];
//! let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
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
//! # Ok(())
//! # }
//! ```
//!
//! See [`docs/validation.md`](https://github.com/acgetchell/delaunay/blob/main/docs/validation.md)
//! for a comprehensive validation guide.
//!
//! [`Simplex::is_valid()`]: crate::prelude::tds::Simplex::is_valid
//! [`Vertex::is_valid()`]: crate::prelude::Vertex::is_valid
//! [`Triangulation::is_valid()`]: crate::prelude::triangulation::Triangulation::is_valid
//! [`DelaunayTriangulation::is_valid()`]: crate::DelaunayTriangulation::is_valid
//! [`DelaunayTriangulation::validation_report()`]: crate::DelaunayTriangulation::validation_report
//!
//! # Examples
//!
//! ## Creating a 3D Triangulation
//!
//! ```rust
//! use delaunay::prelude::*;
//!
//! # #[derive(Debug, thiserror::Error)]
//! # enum ExampleError {
//! #     #[error(transparent)]
//! #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
//! #     #[error(transparent)]
//! #     Insertion(#[from] delaunay::prelude::insertion::InsertionError),
//! #     #[error(transparent)]
//! #     Tds(#[from] delaunay::prelude::tds::TdsError),
//! #     #[error(transparent)]
//! #     TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
//! #     #[error(transparent)]
//! #     Invariant(#[from] delaunay::prelude::tds::InvariantError),
//! #     #[error(transparent)]
//! #     Facet(#[from] delaunay::prelude::tds::FacetError),
//! #     #[error(transparent)]
//! #     Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
//! #     #[error(transparent)]
//! #     Validation(#[from] delaunay::DelaunayTriangulationValidationError),
//! #     #[error(transparent)]
//! #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
//! # }
//! # fn main() -> Result<(), ExampleError> {
//! // Create vertices for a tetrahedron
//! let vertices = [
//!     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
//!     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
//!     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
//!     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
//! ];
//!
//! // Create Delaunay triangulation
//! let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
//!
//! // Query triangulation properties
//! assert_eq!(dt.number_of_vertices(), 4);
//! assert_eq!(dt.number_of_simplices(), 1);
//! assert_eq!(dt.dim(), 3);
//! assert!(dt.validate().is_ok());
//! # Ok(())
//! # }
//! ```
//!
//! ## Adding Vertices to Existing Triangulation
//!
//! ```rust
//! use delaunay::prelude::*;
//!
//! # #[derive(Debug, thiserror::Error)]
//! # enum ExampleError {
//! #     #[error(transparent)]
//! #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
//! #     #[error(transparent)]
//! #     Insertion(#[from] delaunay::prelude::insertion::InsertionError),
//! #     #[error(transparent)]
//! #     Tds(#[from] delaunay::prelude::tds::TdsError),
//! #     #[error(transparent)]
//! #     TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
//! #     #[error(transparent)]
//! #     Invariant(#[from] delaunay::prelude::tds::InvariantError),
//! #     #[error(transparent)]
//! #     Facet(#[from] delaunay::prelude::tds::FacetError),
//! #     #[error(transparent)]
//! #     Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
//! #     #[error(transparent)]
//! #     Validation(#[from] delaunay::DelaunayTriangulationValidationError),
//! #     #[error(transparent)]
//! #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
//! # }
//! # fn main() -> Result<(), ExampleError> {
//! // Start with initial vertices
//! let initial_vertices = [
//!     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
//!     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
//!     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
//!     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
//! ];
//!
//! let mut dt = DelaunayTriangulationBuilder::new(&initial_vertices).build::<()>()?;
//!
//! // Add a new vertex
//! let new_vertex = delaunay::prelude::Vertex::<(), _>::try_new([0.2, 0.2, 0.2])?;
//! dt.insert(new_vertex)?;
//!
//! assert_eq!(dt.number_of_vertices(), 5);
//! assert!(dt.validate().is_ok());
//! # Ok(())
//! # }
//! ```
//!
//! ## 4D Triangulation
//!
//! ```rust
//! use delaunay::prelude::*;
//!
//! # #[derive(Debug, thiserror::Error)]
//! # enum ExampleError {
//! #     #[error(transparent)]
//! #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
//! #     #[error(transparent)]
//! #     Insertion(#[from] delaunay::prelude::insertion::InsertionError),
//! #     #[error(transparent)]
//! #     Tds(#[from] delaunay::prelude::tds::TdsError),
//! #     #[error(transparent)]
//! #     TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
//! #     #[error(transparent)]
//! #     Invariant(#[from] delaunay::prelude::tds::InvariantError),
//! #     #[error(transparent)]
//! #     Facet(#[from] delaunay::prelude::tds::FacetError),
//! #     #[error(transparent)]
//! #     Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
//! #     #[error(transparent)]
//! #     Validation(#[from] delaunay::DelaunayTriangulationValidationError),
//! #     #[error(transparent)]
//! #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
//! # }
//! # fn main() -> Result<(), ExampleError> {
//! // Create 4D triangulation with 5 vertices (needed for a 4-simplex)
//! let vertices_4d = [
//!     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0])?,  // Origin
//!     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0, 0.0])?,  // Unit vector along first dimension
//!     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0, 0.0])?,  // Unit vector along second dimension
//!     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0, 0.0])?,  // Unit vector along third dimension
//!     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 1.0])?,  // Unit vector along fourth dimension
//! ];
//!
//! let dt_4d = DelaunayTriangulationBuilder::new(&vertices_4d).build::<()>()?;
//! assert_eq!(dt_4d.dim(), 4);
//! assert_eq!(dt_4d.number_of_vertices(), 5);
//! assert_eq!(dt_4d.number_of_simplices(), 1);
//! assert!(dt_4d.validate().is_ok());
//! # Ok(())
//! # }
//! ```
//!
//! # References
//!
//! - [CGAL Triangulation Documentation](https://doc.cgal.org/latest/Triangulation/index.html)
//! - Bowyer, A. "Computing Dirichlet tessellations." The Computer Journal 24.2 (1981): 162-166
//! - Watson, D.F. "Computing the n-dimensional Delaunay tessellation with application to Voronoi polytopes." The Computer Journal 24.2 (1981): 167-172
//! - de Berg, M., et al. "Computational Geometry: Algorithms and Applications." 3rd ed. Springer-Verlag, 2008

#![forbid(unsafe_code)]

#[path = "tds_snapshot.rs"]
mod tds_snapshot;

use super::{
    facet::{FacetHandle, facet_key_from_vertices},
    simplex::{NeighborSlot, Simplex, SimplexValidationError},
    traits::data_type::DataType,
    util::{
        deduplication::coords_equal_exact, periodic_facet_key_from_lifted_vertices, usize_to_u8,
    },
    vertex::{Vertex, VertexValidationError},
};
use crate::core::algorithms::flips::FlipError;
use crate::core::collections::{
    Entry, FacetToSimplicesMap, FastHashMap, MAX_PRACTICAL_DIMENSION_SIZE, NeighborBuffer,
    SimplexKeySet, SimplexRemovalBuffer, SimplexVerticesMap, SmallBuffer, StorageMap,
    UuidToSimplexKeyMap, UuidToVertexKeyMap, VertexKeyBuffer, VertexKeySet,
    fast_hash_map_with_capacity,
};
use crate::core::validation::TriangulationValidationError;
use crate::validation::DelaunayTriangulationValidationError;
use slotmap::{Key, new_key_type};
use std::{
    cmp::Ordering as CmpOrdering,
    collections::VecDeque,
    fmt::Debug,
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
/// use delaunay::prelude::tds::TriangulationConstructionState;
///
/// let state = TriangulationConstructionState::Incomplete(2);
/// std::assert_matches!(state, TriangulationConstructionState::Incomplete(2));
///
/// let default_state = TriangulationConstructionState::default();
/// std::assert_matches!(
///     default_state,
///     TriangulationConstructionState::Incomplete(0)
/// );
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TriangulationConstructionState {
    /// The triangulation has insufficient vertices to form a complete D-dimensional triangulation.
    /// Contains the number of vertices currently stored.
    Incomplete(usize),
    /// The triangulation is complete and valid with at least D+1 vertices and proper simplex structure.
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
/// use delaunay::prelude::tds::{EntityKind, TdsConstructionError};
/// use uuid::Uuid;
///
/// let err = TdsConstructionError::DuplicateUuid {
///     entity: EntityKind::Vertex,
///     uuid: Uuid::nil(),
/// };
/// std::assert_matches!(err, TdsConstructionError::DuplicateUuid { .. });
/// ```
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum TdsConstructionError {
    /// Validation error during construction.
    #[error("Validation error during construction: {0}")]
    ValidationError(#[from] TdsError),
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
/// use delaunay::prelude::tds::EntityKind;
///
/// let kind = EntityKind::Simplex;
/// assert_eq!(kind, EntityKind::Simplex);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntityKind {
    /// A vertex entity.
    Vertex,
    /// A simplex entity.
    Simplex,
}

/// Geometric orientation/predicate errors.
///
/// These errors indicate floating-point or geometric degeneracy issues
/// (e.g., nearly coplanar input producing a zero or negative determinant)
/// rather than internal data structure bugs. They are retryable via
/// coordinate perturbation.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::tds::GeometricError;
///
/// let err = GeometricError::DegenerateOrientation {
///     message: "det=0".to_string(),
/// };
/// std::assert_matches!(err, GeometricError::DegenerateOrientation { .. });
/// ```
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum GeometricError {
    /// Geometric orientation degeneracy detected during orientation canonicalization.
    ///
    /// This indicates a geometry-related issue (e.g., nearly coplanar input points
    /// producing a zero determinant, or a kernel predicate evaluation failure)
    /// rather than an internal data structure bug.
    #[error("Degenerate geometric orientation: {message}")]
    DegenerateOrientation {
        /// Description of the degeneracy.
        message: String,
    },
    /// Negative geometric orientation detected after canonicalization.
    ///
    /// A simplex has `det < 0` even after orientation canonicalization passes.  This
    /// typically indicates floating-point sign instability for near-degenerate input
    /// (the fast kernel gives inconsistent sign results across calls) rather than a
    /// data-structure corruption bug.
    #[error("Negative geometric orientation: {message}")]
    NegativeOrientation {
        /// Description of the negative-orientation condition.
        message: String,
    },
}

// REMOVED: TriangulationStatistics and TriangulationDiagnostics
// These were part of the deprecated Bowyer-Watson architecture that has been removed.
// Statistics tracking will be reimplemented for the incremental insertion algorithm
// in a future release.

/// Example usage
///
/// ```
/// use delaunay::prelude::*;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Insertion(#[from] delaunay::prelude::insertion::InsertionError),
/// #     #[error(transparent)]
/// #     Tds(#[from] delaunay::prelude::tds::TdsError),
/// #     #[error(transparent)]
/// #     TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
/// #     #[error(transparent)]
/// #     Invariant(#[from] delaunay::prelude::tds::InvariantError),
/// #     #[error(transparent)]
/// #     Facet(#[from] delaunay::prelude::tds::FacetError),
/// #     #[error(transparent)]
/// #     Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
/// #     #[error(transparent)]
/// #     Validation(#[from] delaunay::DelaunayTriangulationValidationError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// // Build a simple 3D triangulation
/// let vertices = [
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
/// assert_eq!(dt.number_of_vertices(), 4);
/// assert!(dt.is_valid().is_ok());
/// # Ok(())
/// # }
/// ```
///
/// Which side of a neighbor relationship is missing a shared-facet vertex.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SharedFacetMismatchSide {
    /// The source simplex's facet vertex is missing from the neighbor.
    SourceFacet,
    /// The neighbor simplex's facet vertex is missing from the source simplex.
    NeighborFacet,
}

/// Structured reason why neighbor relationships failed validation.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum NeighborValidationError {
    /// A neighbor buffer has the wrong arity for the triangulation dimension.
    #[error("Neighbor vector length {actual} != expected {expected} during {context}")]
    LengthMismatch {
        /// Observed number of neighbor slots.
        actual: usize,
        /// Expected number of neighbor slots.
        expected: usize,
        /// Validation context.
        context: String,
    },
    /// A neighbor buffer contains an unassigned facet slot.
    #[error(
        "Simplex {simplex_uuid} (key {simplex_key:?}) has unassigned neighbor slot at facet {facet_index} during {context}"
    )]
    UnassignedNeighborSlot {
        /// Simplex containing the unassigned slot.
        simplex_key: SimplexKey,
        /// UUID of the simplex containing the unassigned slot.
        simplex_uuid: Uuid,
        /// Facet slot that has not been assigned as boundary or neighbor.
        facet_index: usize,
        /// Validation context.
        context: String,
    },
    /// A non-periodic simplex points to itself as a neighbor.
    #[error(
        "Simplex {simplex_uuid} (key {simplex_key:?}) has non-periodic self-neighbor at facet {facet_index}"
    )]
    NonPeriodicSelfNeighbor {
        /// Simplex whose neighbor pointer references itself.
        simplex_key: SimplexKey,
        /// UUID of the simplex whose neighbor pointer references itself.
        simplex_uuid: Uuid,
        /// Facet slot containing the self-neighbor.
        facet_index: usize,
    },
    /// A neighbor pointer references a missing simplex key.
    #[error(
        "Simplex {simplex_uuid} (key {simplex_key:?}) facet {facet_index} references missing neighbor {neighbor_key:?} during {context}"
    )]
    MissingNeighborSimplex {
        /// Simplex containing the stale neighbor pointer.
        simplex_key: SimplexKey,
        /// UUID of the simplex containing the stale neighbor pointer.
        simplex_uuid: Uuid,
        /// Facet slot containing the stale neighbor pointer.
        facet_index: usize,
        /// Missing neighbor key.
        neighbor_key: SimplexKey,
        /// Validation context.
        context: String,
    },
    /// A neighbor pointer references a simplex removed by the current local edit.
    #[error(
        "Simplex {simplex_uuid} (key {simplex_key:?}) facet {facet_index} references removed neighbor {neighbor_key:?}"
    )]
    ReferencedRemovedNeighbor {
        /// Simplex containing the stale local-edit neighbor pointer.
        simplex_key: SimplexKey,
        /// UUID of the simplex containing the stale local-edit neighbor pointer.
        simplex_uuid: Uuid,
        /// Facet slot containing the removed neighbor pointer.
        facet_index: usize,
        /// Removed neighbor key.
        neighbor_key: SimplexKey,
    },
    /// A neighbor pair does not share exactly the facet opposite the slot.
    #[error(
        "Simplex {simplex_uuid} (key {simplex_key:?}) facet {facet_index} shares {shared_count} vertices with neighbor, expected {expected}"
    )]
    SharedVertexCountMismatch {
        /// Simplex containing the invalid neighbor pointer.
        simplex_key: SimplexKey,
        /// UUID of the simplex containing the invalid neighbor pointer.
        simplex_uuid: Uuid,
        /// Facet slot being checked.
        facet_index: usize,
        /// Number of shared vertices observed.
        shared_count: usize,
        /// Expected number of shared vertices.
        expected: usize,
    },
    /// A neighbor is opposite a different vertex slot than the pointer position.
    #[error(
        "Simplex {simplex_uuid} (key {simplex_key:?}) neighbor at facet {facet_index} is opposite {observed_opposite:?}, expected {expected_opposite}"
    )]
    OppositeVertexMismatch {
        /// Simplex containing the invalid neighbor pointer.
        simplex_key: SimplexKey,
        /// UUID of the simplex containing the invalid neighbor pointer.
        simplex_uuid: Uuid,
        /// Facet slot being checked.
        facet_index: usize,
        /// Observed opposite vertex slot.
        observed_opposite: Option<usize>,
        /// Expected opposite vertex slot.
        expected_opposite: usize,
    },
    /// The facet incidence map is missing a facet implied by a simplex.
    #[error(
        "Simplex {simplex_uuid} (key {simplex_key:?}) facet {facet_index} key {facet_key} is missing from facet incidence"
    )]
    FacetIncidenceMissing {
        /// Simplex whose facet was missing from the incidence map.
        simplex_key: SimplexKey,
        /// UUID of the simplex whose facet was missing from the incidence map.
        simplex_uuid: Uuid,
        /// Facet index.
        facet_index: usize,
        /// Canonical facet key.
        facet_key: u64,
    },
    /// A facet incidence entry exists but does not include the edited simplex/facet.
    #[error(
        "Simplex {simplex_uuid} (key {simplex_key:?}) facet {facet_index} key {facet_key} does not reference the edited facet"
    )]
    FacetIncidenceDoesNotReferenceSimplex {
        /// Simplex being edited.
        simplex_key: SimplexKey,
        /// UUID of the simplex being edited.
        simplex_uuid: Uuid,
        /// Facet index being edited.
        facet_index: usize,
        /// Canonical facet key.
        facet_key: u64,
    },
    /// A facet incidence entry has non-manifold multiplicity.
    #[error(
        "Simplex {simplex_uuid} (key {simplex_key:?}) facet {facet_index} key {facet_key} is shared by {simplex_count} simplices"
    )]
    FacetIncidenceMultiplicity {
        /// Simplex being edited.
        simplex_key: SimplexKey,
        /// UUID of the simplex being edited.
        simplex_uuid: Uuid,
        /// Facet index being edited.
        facet_index: usize,
        /// Canonical facet key.
        facet_key: u64,
        /// Number of simplices incident to the facet.
        simplex_count: usize,
    },
    /// A proposed neighbor pointer disagrees with facet incidence.
    #[error(
        "Simplex {simplex_uuid} (key {simplex_key:?}) facet {facet_index} proposed neighbor {proposed_neighbor:?} does not match expected {expected_neighbor:?}"
    )]
    NeighborIncidenceMismatch {
        /// Simplex being edited.
        simplex_key: SimplexKey,
        /// UUID of the simplex being edited.
        simplex_uuid: Uuid,
        /// Facet index being edited.
        facet_index: usize,
        /// Proposed neighbor pointer.
        proposed_neighbor: Option<SimplexKey>,
        /// Neighbor expected from facet incidence.
        expected_neighbor: Option<SimplexKey>,
    },
    /// A facet slot index is outside the neighbor buffer.
    #[error("Neighbor facet index {facet_index} out of bounds for {slot_count} neighbor slots")]
    NeighborSlotOutOfBounds {
        /// Invalid facet index.
        facet_index: usize,
        /// Number of available neighbor slots.
        slot_count: usize,
    },
    /// The mirror facet could not be found between adjacent simplices.
    #[error(
        "Could not determine mirror facet during {context}: simplex {simplex_uuid}[{facet_index}] -> neighbor {neighbor_uuid}"
    )]
    MirrorFacetMissing {
        /// UUID of the source simplex.
        simplex_uuid: Uuid,
        /// Facet index in the source simplex.
        facet_index: usize,
        /// UUID of the neighbor simplex.
        neighbor_uuid: Uuid,
        /// Validation context.
        context: String,
    },
    /// Shared-vertex analysis found more than one possible mirror facet.
    #[error(
        "Mirror facet is ambiguous: simplex {simplex_uuid} and neighbor {neighbor_uuid} differ by more than one vertex"
    )]
    MirrorFacetAmbiguous {
        /// UUID of the source simplex.
        simplex_uuid: Uuid,
        /// UUID of the neighbor simplex.
        neighbor_uuid: Uuid,
    },
    /// Shared-vertex analysis found that two simplices share every vertex.
    #[error(
        "Mirror facet could not be determined: simplex {simplex_uuid} and neighbor {neighbor_uuid} share all vertices"
    )]
    MirrorFacetDuplicateSimplices {
        /// UUID of the source simplex.
        simplex_uuid: Uuid,
        /// UUID of the neighbor simplex.
        neighbor_uuid: Uuid,
    },
    /// A computed mirror facet disagrees with shared-vertex analysis.
    #[error(
        "Mirror facet index mismatch: simplex {simplex_uuid}[{facet_index}] -> neighbor {neighbor_uuid}; observed {observed_mirror_index}, expected {expected_mirror_index}"
    )]
    MirrorFacetIndexMismatch {
        /// UUID of the source simplex.
        simplex_uuid: Uuid,
        /// Facet index in the source simplex.
        facet_index: usize,
        /// UUID of the neighbor simplex.
        neighbor_uuid: Uuid,
        /// Mirror index returned by simplex logic.
        observed_mirror_index: usize,
        /// Mirror index implied by shared-vertex analysis.
        expected_mirror_index: usize,
    },
    /// A shared facet is missing a vertex on one side of a neighbor pair.
    #[error(
        "Shared facet mismatch ({side:?}): simplex {simplex_uuid}[{facet_index}] and neighbor {neighbor_uuid}[{mirror_index}] are missing vertex {missing_vertex:?}"
    )]
    SharedFacetMissingVertex {
        /// Which side exposed the missing vertex.
        side: SharedFacetMismatchSide,
        /// UUID of the source simplex.
        simplex_uuid: Uuid,
        /// Facet index in the source simplex.
        facet_index: usize,
        /// UUID of the neighbor simplex.
        neighbor_uuid: Uuid,
        /// Mirror facet index in the neighbor simplex.
        mirror_index: usize,
        /// Missing vertex key.
        missing_vertex: VertexKey,
    },
    /// A neighbor does not carry the required reciprocal pointer.
    #[error(
        "Neighbor back-reference mismatch during {context}: simplex {simplex_uuid}[{facet_index}] -> {neighbor_key:?} should be mirrored by {neighbor_uuid}[{mirror_index}] -> {simplex_key:?}, found {observed:?}"
    )]
    BackReferenceMismatch {
        /// Source simplex key.
        simplex_key: SimplexKey,
        /// Source simplex UUID.
        simplex_uuid: Uuid,
        /// Source facet index.
        facet_index: usize,
        /// Neighbor simplex key.
        neighbor_key: SimplexKey,
        /// Neighbor simplex UUID.
        neighbor_uuid: Uuid,
        /// Mirror facet index in the neighbor.
        mirror_index: usize,
        /// Observed back-reference, or `None` if absent.
        observed: Option<SimplexKey>,
        /// Validation context.
        context: String,
    },
    /// A reciprocal update would overwrite another back-reference.
    #[error(
        "Neighbor simplex {neighbor_uuid}[{mirror_index}] already references {existing_back_ref:?}; refusing to overwrite with {requested_back_ref:?}"
    )]
    ExistingBackReferenceConflict {
        /// Neighbor UUID.
        neighbor_uuid: Uuid,
        /// Mirror facet index in the neighbor.
        mirror_index: usize,
        /// Existing back-reference.
        existing_back_ref: SimplexKey,
        /// Requested back-reference.
        requested_back_ref: SimplexKey,
    },
    /// A boundary facet has a neighbor pointer.
    #[error(
        "Boundary facet {facet_key} unexpectedly has neighbor {neighbor_key:?} across simplex {simplex_uuid}[{facet_index}]"
    )]
    BoundaryFacetHasNeighbor {
        /// Boundary facet key.
        facet_key: u64,
        /// Simplex containing the boundary facet.
        simplex_key: SimplexKey,
        /// UUID of the simplex containing the boundary facet.
        simplex_uuid: Uuid,
        /// Facet index in the simplex.
        facet_index: usize,
        /// Unexpected neighbor key.
        neighbor_key: SimplexKey,
    },
    /// A boundary facet has inadmissible self-adjacency.
    #[error(
        "Boundary facet {facet_key} has non-periodic self-neighbor across simplex {simplex_uuid}[{facet_index}]"
    )]
    BoundaryFacetHasNonPeriodicSelfNeighbor {
        /// Boundary facet key.
        facet_key: u64,
        /// Simplex containing the boundary facet.
        simplex_key: SimplexKey,
        /// UUID of the simplex containing the boundary facet.
        simplex_uuid: Uuid,
        /// Facet index in the simplex.
        facet_index: usize,
    },
    /// An interior facet's two incident simplices do not point to each other.
    #[error(
        "Interior facet {facet_key} has inconsistent neighbor pointers: {first_simplex_uuid}[{first_facet_index}] -> {first_neighbor:?}, {second_simplex_uuid}[{second_facet_index}] -> {second_neighbor:?}"
    )]
    InteriorFacetNeighborMismatch {
        /// Interior facet key.
        facet_key: u64,
        /// First incident simplex key.
        first_simplex_key: SimplexKey,
        /// First incident simplex UUID.
        first_simplex_uuid: Uuid,
        /// Facet index in the first simplex.
        first_facet_index: usize,
        /// Neighbor pointer observed in the first simplex.
        first_neighbor: Option<SimplexKey>,
        /// Second incident simplex key.
        second_simplex_key: SimplexKey,
        /// Second incident simplex UUID.
        second_simplex_uuid: Uuid,
        /// Facet index in the second simplex.
        second_facet_index: usize,
        /// Neighbor pointer observed in the second simplex.
        second_neighbor: Option<SimplexKey>,
    },
    /// A facet's vertex order could not be built for neighbor validation.
    #[error(
        "Could not build facet order during {context}: simplex {simplex_uuid} (key {simplex_key:?}) facet {facet_index}: {source}"
    )]
    FacetOrderUnavailable {
        /// Simplex whose facet order could not be built.
        simplex_key: SimplexKey,
        /// UUID of the simplex whose facet order could not be built.
        simplex_uuid: Uuid,
        /// Facet index whose order could not be built.
        facet_index: usize,
        /// Validation context.
        context: String,
        /// Underlying [`FlipError`] raised while deriving the facet order.
        #[source]
        source: Box<FlipError>,
    },
    /// Bistellar flip neighbor wiring failed while preserving TDS invariants.
    #[error("Flip neighbor wiring failed: {reason}")]
    FlipNeighborWiring {
        /// Structured flip wiring failure.
        #[source]
        reason: Box<crate::core::algorithms::flips::FlipNeighborWiringError>,
    },
    /// Neighbor validation failed in a context that is still being migrated to structured fields.
    #[error("{message}")]
    Other {
        /// Diagnostic detail.
        message: String,
    },
}

/// Errors that can occur during triangulation validation (post-construction).
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum TdsError {
    /// The triangulation contains an invalid vertex.
    #[error("Invalid vertex {vertex_id}: {source}")]
    InvalidVertex {
        /// The UUID of the invalid vertex.
        vertex_id: Uuid,
        /// The underlying vertex validation error.
        source: VertexValidationError,
    },
    /// The triangulation contains an invalid simplex.
    #[error("Invalid simplex {simplex_id}: {source}")]
    InvalidSimplex {
        /// The UUID of the invalid simplex.
        simplex_id: Uuid,
        /// The underlying simplex validation error.
        source: SimplexValidationError,
    },
    /// Neighbor relationships are invalid.
    #[error("Invalid neighbor relationships: {reason}")]
    InvalidNeighbors {
        /// Structured neighbor validation failure.
        #[source]
        reason: NeighborValidationError,
    },
    /// Coherent orientation invariant violated between adjacent simplices.
    #[error(
        "Orientation invariant violated between simplices {simplex1_uuid} and {simplex2_uuid}; shared facet orderings {facet_vertices:?} vs {simplex2_facet_vertices:?} (simplex1 facet index {simplex1_facet_index}, simplex2 facet index {simplex2_facet_index}, observed_odd_permutation={observed_odd_permutation}, expected_odd_permutation={expected_odd_permutation})"
    )]
    OrientationViolation {
        /// Key of the first simplex.
        simplex1_key: SimplexKey,
        /// UUID of the first simplex.
        simplex1_uuid: Uuid,
        /// Key of the second simplex.
        simplex2_key: SimplexKey,
        /// UUID of the second simplex.
        simplex2_uuid: Uuid,
        /// Facet index in the first simplex.
        simplex1_facet_index: usize,
        /// Facet index in the second simplex.
        simplex2_facet_index: usize,
        /// Vertex keys of the shared facet in `simplex1` ordering (excluding `simplex1_facet_index`).
        facet_vertices: Vec<VertexKey>,
        /// Vertex keys of the shared facet in `simplex2` ordering (excluding `simplex2_facet_index`).
        simplex2_facet_vertices: Vec<VertexKey>,
        /// Observed parity of the permutation from `facet_vertices` to `simplex2_facet_vertices`.
        observed_odd_permutation: bool,
        /// Expected odd-permutation parity under the coherent boundary-orientation convention.
        expected_odd_permutation: bool,
    },
    /// The triangulation contains duplicate simplices.
    #[error("Duplicate simplices detected: {message}")]
    DuplicateSimplices {
        /// Description of the duplicate simplex validation failure.
        message: String,
    },
    /// A simplex insertion or validation pass found a facet incident to too many simplices.
    ///
    /// During insertion preflight, the `candidate_*` fields identify the simplex that
    /// would exceed the PL-manifold facet multiplicity. During post-hoc
    /// validation, they identify one offending incident simplex from the over-shared
    /// facet.
    #[error(
        "Facet {facet_key} exceeds incident-simplex limit: observed {attempted_incident_count} incident simplices, max {max_incident_count}; candidate/offending simplex {candidate_simplex_uuid} facet {candidate_facet_index}; other incident simplices {existing_incident_count}"
    )]
    FacetSharingViolation {
        /// Canonical key of the over-shared facet.
        facet_key: u64,
        /// Number of other/pre-existing simplices already incident to the facet.
        existing_incident_count: usize,
        /// Number of incident simplices observed, or that would exist after candidate insertion.
        attempted_incident_count: usize,
        /// Maximum allowed number of incident simplices for a PL-manifold facet.
        max_incident_count: usize,
        /// UUID of the candidate or offending simplex.
        candidate_simplex_uuid: Uuid,
        /// Facet index on the candidate or offending simplex.
        candidate_facet_index: usize,
    },
    /// Failed to create a simplex during triangulation.
    #[error("Failed to create simplex: {message}")]
    FailedToCreateSimplex {
        /// Description of the simplex creation failure.
        message: String,
    },
    /// Simplices are not neighbors as expected
    #[error("Simplices {simplex1:?} and {simplex2:?} are not neighbors")]
    NotNeighbors {
        /// The first simplex UUID.
        simplex1: Uuid,
        /// The second simplex UUID.
        simplex2: Uuid,
    },
    /// Entity mapping inconsistency (vertex or simplex).
    #[error("{entity:?} mapping inconsistency: {message}")]
    MappingInconsistency {
        /// The type of entity with the mapping issue.
        entity: EntityKind,
        /// Description of the mapping inconsistency.
        message: String,
    },
    /// Failed to retrieve vertex keys for a simplex during neighbor assignment.
    #[error("Failed to retrieve vertex keys for simplex {simplex_id}: {message}")]
    VertexKeyRetrievalFailed {
        /// The UUID of the simplex that failed.
        simplex_id: Uuid,
        /// Description of the failure.
        message: String,
    },
    /// A simplex key was expected in storage but not found.
    ///
    /// This typically indicates a dangling simplex reference or stale key
    /// after topology mutations (simplex removal, cavity filling, etc.).
    #[error("Simplex key {simplex_key:?} not found: {context}")]
    SimplexNotFound {
        /// The simplex key that was not found in storage.
        simplex_key: SimplexKey,
        /// Description of the context where the lookup failed.
        context: String,
    },
    /// A vertex key was expected in storage but not found.
    ///
    /// This typically indicates a dangling vertex reference or stale key
    /// after topology mutations.
    #[error("Vertex key {vertex_key:?} not found: {context}")]
    VertexNotFound {
        /// The vertex key that was not found in storage.
        vertex_key: VertexKey,
        /// Description of the context where the lookup failed.
        context: String,
    },
    /// A dimensional invariant was violated (wrong vertex count, offset count, etc.).
    ///
    /// A simplex, facet, ridge, or link has a different number of elements than
    /// expected for the triangulation dimension `D`.
    #[error("Dimension mismatch: expected {expected}, got {actual} — {context}")]
    DimensionMismatch {
        /// The expected count.
        expected: usize,
        /// The actual count observed.
        actual: usize,
        /// Description of what was being checked.
        context: String,
    },
    /// An index exceeded the valid range for the target structure.
    #[error("Index out of bounds: index {index}, bound {bound} — {context}")]
    IndexOutOfBounds {
        /// The index that was out of bounds.
        index: usize,
        /// The exclusive upper bound.
        bound: usize,
        /// Description of what was being accessed.
        context: String,
    },
    /// Internal data structure inconsistency.
    ///
    /// This is the catch-all for structural invariant violations that do not
    /// fit a more specific variant (e.g. topology contradictions, error
    /// wrapping, operational failures). Prefer [`SimplexNotFound`],
    /// [`VertexNotFound`], [`DimensionMismatch`], or [`IndexOutOfBounds`]
    /// when applicable.
    ///
    /// [`SimplexNotFound`]: TdsError::SimplexNotFound
    /// [`VertexNotFound`]: TdsError::VertexNotFound
    /// [`DimensionMismatch`]: TdsError::DimensionMismatch
    /// [`IndexOutOfBounds`]: TdsError::IndexOutOfBounds
    #[error("Internal data structure inconsistency: {message}")]
    InconsistentDataStructure {
        /// Description of the inconsistency.
        message: String,
    },
    /// Geometric orientation/predicate error (e.g., degenerate or negative orientation).
    ///
    /// This wraps a [`GeometricError`] and indicates a floating-point or geometric
    /// degeneracy issue rather than an internal data structure bug.
    #[error(transparent)]
    Geometric(#[from] GeometricError),

    /// Facet operation failed during validation.
    #[error("Facet operation failed: {0}")]
    FacetError(#[from] super::facet::FacetError),
    /// A simplex contains two or more vertices with identical coordinates.
    ///
    /// This is distinct from [`SimplexValidationError::DuplicateVertices`] which checks
    /// for duplicate vertex *keys*. This variant detects the case where different
    /// vertex keys reference geometrically identical points — producing a zero-volume
    /// simplex that is catastrophic for `SoS` and Pachner moves.
    #[error("Duplicate coordinates in simplex {simplex_id}: {message}")]
    DuplicateCoordinatesInSimplex {
        /// UUID of the simplex containing duplicate-coordinate vertices.
        simplex_id: Uuid,
        /// Description of which vertices share coordinates.
        message: String,
    },
}

/// Discriminant for compact [`TdsError`] summaries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TdsErrorKind {
    /// A vertex failed validation.
    InvalidVertex,
    /// A simplex failed validation.
    InvalidSimplex,
    /// Neighbor relationships were invalid.
    InvalidNeighbors,
    /// Adjacent simplices violated coherent orientation.
    OrientationViolation,
    /// Duplicate maximal simplices were detected.
    DuplicateSimplices,
    /// A facet exceeded, or would exceed, the allowed incident-simplex count.
    FacetSharingViolation,
    /// Simplex creation failed.
    FailedToCreateSimplex,
    /// Expected neighbor relation was absent.
    NotNeighbors,
    /// UUID-to-key mapping was inconsistent.
    MappingInconsistency,
    /// Simplex vertex-key lookup failed.
    VertexKeyRetrievalFailed,
    /// A referenced simplex key was missing.
    SimplexNotFound,
    /// A referenced vertex key was missing.
    VertexNotFound,
    /// A dimension/count invariant was violated.
    DimensionMismatch,
    /// An index exceeded its valid bound.
    IndexOutOfBounds,
    /// Internal TDS state was inconsistent.
    InconsistentDataStructure,
    /// A geometric validation failure occurred.
    Geometric,
    /// A facet operation failed.
    FacetError,
    /// A simplex contained duplicate coordinates.
    DuplicateCoordinatesInSimplex,
}

impl From<&TdsError> for TdsErrorKind {
    fn from(source: &TdsError) -> Self {
        match source {
            TdsError::InvalidVertex { .. } => Self::InvalidVertex,
            TdsError::InvalidSimplex { .. } => Self::InvalidSimplex,
            TdsError::InvalidNeighbors { .. } => Self::InvalidNeighbors,
            TdsError::OrientationViolation { .. } => Self::OrientationViolation,
            TdsError::DuplicateSimplices { .. } => Self::DuplicateSimplices,
            TdsError::FacetSharingViolation { .. } => Self::FacetSharingViolation,
            TdsError::FailedToCreateSimplex { .. } => Self::FailedToCreateSimplex,
            TdsError::NotNeighbors { .. } => Self::NotNeighbors,
            TdsError::MappingInconsistency { .. } => Self::MappingInconsistency,
            TdsError::VertexKeyRetrievalFailed { .. } => Self::VertexKeyRetrievalFailed,
            TdsError::SimplexNotFound { .. } => Self::SimplexNotFound,
            TdsError::VertexNotFound { .. } => Self::VertexNotFound,
            TdsError::DimensionMismatch { .. } => Self::DimensionMismatch,
            TdsError::IndexOutOfBounds { .. } => Self::IndexOutOfBounds,
            TdsError::InconsistentDataStructure { .. } => Self::InconsistentDataStructure,
            TdsError::Geometric(_) => Self::Geometric,
            TdsError::FacetError(_) => Self::FacetError,
            TdsError::DuplicateCoordinatesInSimplex { .. } => Self::DuplicateCoordinatesInSimplex,
        }
    }
}

/// Errors that can occur during TDS mutation operations.
///
/// This error is a thin wrapper around [`TdsError`]. Mutation operations can fail
/// for the same reasons as validation (i.e., because an invariant would be violated or a
/// consistency check fails while attempting to perform the mutation).
///
/// The wrapper exists to make call sites and API docs semantically explicit, while also
/// allowing this error to evolve into a richer, dedicated type in a future release without
/// breaking the public API surface.
///
/// # Stability / conversion contract
///
/// `TdsMutationError` currently supports lossless conversion to and from [`TdsError`]
/// via the provided `From`/`Into` impls. If this wrapper evolves to include mutation-specific
/// context (additional fields/variants), converting `TdsMutationError` into [`TdsError`]
/// may become lossy.
///
/// Callers that want to preserve potential future mutation-specific details should avoid
/// converting back to [`TdsError`] and instead propagate/handle `TdsMutationError`
/// directly.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::tds::{NeighborValidationError, TdsError, TdsMutationError};
///
/// let err = TdsError::InvalidNeighbors {
///     reason: NeighborValidationError::Other {
///         message: "bad neighbors".to_string(),
///     },
/// };
/// let mutation: TdsMutationError = err.clone().into();
/// let round_trip: TdsError = mutation.clone().into();
/// assert_eq!(round_trip, err);
/// ```
#[derive(Clone, Debug, Error, PartialEq)]
#[error(transparent)]
#[must_use]
pub struct TdsMutationError(TdsError);

impl TdsMutationError {
    /// Returns a reference to the underlying [`TdsError`].
    #[must_use]
    pub const fn as_tds_error(&self) -> &TdsError {
        &self.0
    }

    /// Consumes this wrapper and returns the underlying [`TdsError`].
    #[must_use]
    pub fn into_inner(self) -> TdsError {
        self.0
    }
}

impl From<TdsError> for TdsMutationError {
    fn from(err: TdsError) -> Self {
        Self(err)
    }
}

impl From<TdsMutationError> for TdsError {
    fn from(err: TdsMutationError) -> Self {
        err.0
    }
}

/// Classifies the kind of triangulation invariant that failed during validation.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::tds::InvariantKind;
///
/// let kind = InvariantKind::Topology;
/// assert_eq!(kind, InvariantKind::Topology);
/// ```
///
/// This is used by [`TriangulationValidationReport`] to group related errors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum InvariantKind {
    /// Per-vertex validity (finite coordinates, non-nil UUID, etc.).
    VertexValidity,
    /// Per-simplex validity (vertex count, duplicate vertices, nil UUID, etc.).
    SimplexValidity,
    /// No simplices contain vertices with identical coordinates (geometric uniqueness).
    SimplexCoordinateUniqueness,
    /// Vertex UUID↔key mapping invariants.
    VertexMappings,
    /// Simplex UUID↔key mapping invariants.
    SimplexMappings,
    /// Simplices reference only valid vertex keys (no stale/missing vertex keys).
    SimplexVertexKeys,
    /// Vertex incidence invariants (`Vertex::incident_simplex` pointers are non-dangling + consistent).
    VertexIncidence,
    /// No duplicate maximal simplices with identical vertex sets.
    DuplicateSimplices,
    /// Facet sharing invariants (each facet shared by at most 2 simplices).
    FacetSharing,
    /// Neighbor topology and mutual-consistency invariants.
    NeighborConsistency,
    /// Coherent combinatorial orientation (adjacent simplices induce opposite facet orientations).
    CoherentOrientation,
    /// Simplex neighbor graph connectivity (single connected component).
    Connectedness,
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
/// use delaunay::prelude::tds::{InvariantError, NeighborValidationError, TdsError};
///
/// let err = InvariantError::Tds(TdsError::InvalidNeighbors {
///     reason: NeighborValidationError::Other {
///         message: "bad neighbors".to_string(),
///     },
/// });
/// std::assert_matches!(err, InvariantError::Tds(_));
/// ```
///
/// This is used by [`TriangulationValidationReport`] so that diagnostic reporting can
/// preserve structured errors from each layer (TDS / topology / Delaunay) without
/// stringification.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum InvariantError {
    /// Level 1–2 (elements + TDS structure).
    #[error(transparent)]
    Tds(#[from] TdsError),

    /// Level 3 (topology).
    #[error(transparent)]
    Triangulation(#[from] TriangulationValidationError),

    /// Level 4 (Delaunay property).
    #[error(transparent)]
    Delaunay(#[from] DelaunayTriangulationValidationError),
}

/// Validation layer reported by an [`InvariantErrorSummary`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum InvariantErrorSummaryKind {
    /// Level 1-2 TDS validation failed.
    Tds,
    /// Level 3 topology validation failed.
    Triangulation,
    /// Level 4 Delaunay validation failed.
    Delaunay,
}

/// Discriminant for compact Level 3 topology-validation summaries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TriangulationValidationErrorKind {
    /// A facet had invalid manifold multiplicity.
    ManifoldFacetMultiplicity,
    /// A boundary ridge had invalid boundary-facet multiplicity.
    BoundaryRidgeMultiplicity,
    /// A ridge link failed PL-manifold validation.
    RidgeLinkNotManifold,
    /// A vertex link failed PL-manifold validation.
    VertexLinkNotManifold,
    /// Euler characteristic did not match the expected classification.
    EulerCharacteristicMismatch,
    /// A vertex was not incident to any simplex.
    IsolatedVertex,
    /// The simplex-neighbor graph was disconnected.
    Disconnected,
    /// Positive-orientation promotion did not converge.
    OrientationPromotionNonConvergence,
}

impl From<&TriangulationValidationError> for TriangulationValidationErrorKind {
    fn from(source: &TriangulationValidationError) -> Self {
        match source {
            TriangulationValidationError::ManifoldFacetMultiplicity { .. } => {
                Self::ManifoldFacetMultiplicity
            }
            TriangulationValidationError::BoundaryRidgeMultiplicity { .. } => {
                Self::BoundaryRidgeMultiplicity
            }
            TriangulationValidationError::RidgeLinkNotManifold { .. } => Self::RidgeLinkNotManifold,
            TriangulationValidationError::VertexLinkNotManifold { .. } => {
                Self::VertexLinkNotManifold
            }
            TriangulationValidationError::EulerCharacteristicMismatch { .. } => {
                Self::EulerCharacteristicMismatch
            }
            TriangulationValidationError::IsolatedVertex { .. } => Self::IsolatedVertex,
            TriangulationValidationError::Disconnected { .. } => Self::Disconnected,
            TriangulationValidationError::OrientationPromotionNonConvergence { .. } => {
                Self::OrientationPromotionNonConvergence
            }
        }
    }
}

/// Discriminant for compact Level 4 Delaunay-validation summaries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DelaunayValidationErrorKind {
    /// Lower-layer TDS validation failed.
    Tds,
    /// Lower-layer topology validation failed.
    Triangulation,
    /// Delaunay verification failed.
    VerificationFailed,
    /// Legacy string-only repair validation failed.
    RepairFailed,
    /// Typed repair validation failed.
    RepairOperationFailed,
}

impl From<&DelaunayTriangulationValidationError> for DelaunayValidationErrorKind {
    fn from(source: &DelaunayTriangulationValidationError) -> Self {
        match source {
            DelaunayTriangulationValidationError::Tds(_) => Self::Tds,
            DelaunayTriangulationValidationError::Triangulation(_) => Self::Triangulation,
            DelaunayTriangulationValidationError::VerificationFailed { .. } => {
                Self::VerificationFailed
            }
            DelaunayTriangulationValidationError::RepairFailed { .. } => Self::RepairFailed,
            DelaunayTriangulationValidationError::RepairOperationFailed { .. } => {
                Self::RepairOperationFailed
            }
        }
    }
}

/// Nested discriminant preserved by an [`InvariantErrorSummary`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum InvariantErrorSummaryDetail {
    /// Level 1-2 TDS validation failed with the given kind.
    Tds(TdsErrorKind),
    /// Level 3 topology validation failed with the given kind.
    Triangulation(TriangulationValidationErrorKind),
    /// Level 4 Delaunay validation failed with the given kind.
    Delaunay(DelaunayValidationErrorKind),
}

/// Compact summary of an [`InvariantError`] for small by-value error payloads.
///
/// The conversion preserves the validation layer in
/// [`InvariantErrorSummaryKind`], the nested typed discriminant in
/// [`InvariantErrorSummaryDetail`], and the rendered diagnostic text. It
/// intentionally drops bulky typed payloads and source chains; keep the original
/// [`InvariantError`] when callers need the full structured validation context.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::tds::{
///     InvariantError, InvariantErrorSummary, InvariantErrorSummaryDetail,
///     InvariantErrorSummaryKind, TdsError, TdsErrorKind,
/// };
///
/// let source = InvariantError::Tds(TdsError::InconsistentDataStructure {
///     message: "dangling simplex key".to_string(),
/// });
/// let summary = InvariantErrorSummary::from(source);
///
/// assert_eq!(summary.kind, InvariantErrorSummaryKind::Tds);
/// assert_eq!(
///     summary.detail,
///     InvariantErrorSummaryDetail::Tds(TdsErrorKind::InconsistentDataStructure),
/// );
/// ```
#[must_use]
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("{message}")]
pub struct InvariantErrorSummary {
    /// Validation layer that produced the failure.
    pub kind: InvariantErrorSummaryKind,
    /// Nested structured error kind.
    pub detail: InvariantErrorSummaryDetail,
    /// Full diagnostic text from the original invariant error.
    pub message: String,
}

impl From<InvariantError> for InvariantErrorSummary {
    fn from(source: InvariantError) -> Self {
        let kind = match &source {
            InvariantError::Tds(_) => InvariantErrorSummaryKind::Tds,
            InvariantError::Triangulation(_) => InvariantErrorSummaryKind::Triangulation,
            InvariantError::Delaunay(_) => InvariantErrorSummaryKind::Delaunay,
        };
        let detail = match &source {
            InvariantError::Tds(source) => InvariantErrorSummaryDetail::Tds(source.into()),
            InvariantError::Triangulation(source) => {
                InvariantErrorSummaryDetail::Triangulation(source.into())
            }
            InvariantError::Delaunay(source) => {
                InvariantErrorSummaryDetail::Delaunay(source.into())
            }
        };
        Self {
            kind,
            detail,
            message: source.to_string(),
        }
    }
}

/// A single invariant violation recorded during validation diagnostics.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::tds::{
///     InvariantError, InvariantKind, InvariantViolation, NeighborValidationError, TdsError,
/// };
///
/// let violation = InvariantViolation {
///     kind: InvariantKind::Topology,
///     error: InvariantError::Tds(TdsError::InvalidNeighbors {
///         reason: NeighborValidationError::Other {
///             message: "bad neighbors".to_string(),
///         },
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
/// [`DelaunayTriangulation::validation_report()`]: crate::DelaunayTriangulation::validation_report
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::tds::TriangulationValidationReport;
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
        /// use delaunay::prelude::*;
        ///
        /// # #[derive(Debug, thiserror::Error)]
        /// # enum ExampleError {
        /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
        /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
        /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
        /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
        /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
        /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
        /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
        /// #     #[error(transparent)]
        /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
        /// # }
        /// # fn main() -> Result<(), ExampleError> {
        /// let vertices = [
        ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0])?,
        ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0])?,
        ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0])?,
        /// ];
        /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
        /// let Some(key) = dt.tds().vertex_keys().next() else {
        ///     return Ok(());
        /// };
        /// let _ = key;
        /// # Ok(())
        /// # }
        /// ```
    pub struct VertexKey;
}

new_key_type! {
    /// Key type for accessing simplices in the storage map.
    ///
    /// This creates a unique, type-safe identifier for simplices stored in the
    /// triangulation's simplex storage. Each SimplexKey corresponds to exactly
    /// one simplex and provides efficient, stable access even as simplices are
    /// added or removed during triangulation operations.
    ///
    /// # Examples
    ///
        /// ```
        /// use delaunay::prelude::*;
        ///
        /// # #[derive(Debug, thiserror::Error)]
        /// # enum ExampleError {
        /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
        /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
        /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
        /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
        /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
        /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
        /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
        /// #     #[error(transparent)]
        /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
        /// # }
        /// # fn main() -> Result<(), ExampleError> {
        /// let vertices = [
        ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0])?,
        ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0])?,
        ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0])?,
        /// ];
        /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
        /// let Some(key) = dt.tds().simplex_keys().next() else {
        ///     return Ok(());
        /// };
        /// let _ = key;
        /// # Ok(())
        /// # }
        /// ```
    pub struct SimplexKey;
}

#[derive(Debug)]
/// The `Tds` struct represents a triangulation data structure with vertices
/// and simplices, where the vertices and simplices are identified by UUIDs.
///
/// # Properties
///
/// - `vertices`: A storage map that stores vertices with stable keys for efficient access.
///   Each [`Vertex`] has validated coordinates, optional data of type `U`, and a constant `D` dimension.
/// - `simplices`: A storage map that stores maximal [`Simplex`] objects with stable keys.
///   Each [`Simplex`] stores [`VertexKey`]s (keys into `vertices`) and optional neighbor [`SimplexKey`]s,
///   plus simplex data of type `V`.
///   Note the dimensionality of the simplex may differ from D, though the [`Tds`]
///   only stores simplices of maximal dimensionality D and infers other lower
///   dimensional simplices (cf. Facets) from the maximal simplices and their vertices.
///
/// For example, in 3 dimensions:
///
/// - A 0-dimensional simplex is a [`Vertex`].
/// - A 1-dimensional simplex is an `Edge` given by the `Tetrahedron` and two
///   [`Vertex`] endpoints.
/// - A 2-dimensional simplex is a `Facet` given by the `Tetrahedron` and the
///   opposite [`Vertex`].
/// - A 3-dimensional simplex is a `Tetrahedron`, the maximal simplex.
///
/// A similar pattern holds for higher dimensions.
///
/// In typical usage, vertices carry coordinates in D-dimensional Euclidean space.
/// However, the core `Tds` API is designed to be **combinatorial**: most methods
/// operate purely on keys and adjacency.
///
/// # Usage
///
/// `Tds` is the low-level topology container used by
/// [`Triangulation`](crate::prelude::triangulation::Triangulation) and
/// [`crate::DelaunayTriangulation`].
///
/// Most users should construct triangulations via `DelaunayTriangulation` and access the
/// underlying `Tds` via `dt.tds()`. Use [`Tds::empty`](Self::empty) for low-level or test
/// scenarios where you want to manipulate the topology directly.
///
/// ```rust
/// use delaunay::prelude::*;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Insertion(#[from] delaunay::prelude::insertion::InsertionError),
/// #     #[error(transparent)]
/// #     Tds(#[from] delaunay::prelude::tds::TdsError),
/// #     #[error(transparent)]
/// #     TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
/// #     #[error(transparent)]
/// #     Invariant(#[from] delaunay::prelude::tds::InvariantError),
/// #     #[error(transparent)]
/// #     Facet(#[from] delaunay::prelude::tds::FacetError),
/// #     #[error(transparent)]
/// #     Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
/// #     #[error(transparent)]
/// #     Validation(#[from] delaunay::DelaunayTriangulationValidationError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// // Create vertices for a 2D triangulation
/// let vertices = [
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0])?,
///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0])?,
///     delaunay::prelude::Vertex::<(), _>::try_new([0.5, 1.0])?,
/// ];
///
/// // Create a new triangulation
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
///
/// // Check the number of simplices and vertices
/// assert_eq!(dt.number_of_simplices(), 1);
/// assert_eq!(dt.number_of_vertices(), 3);
/// # Ok(())
/// # }
/// ```
pub struct Tds<U, V, const D: usize> {
    /// Storage map for vertices, allowing stable keys and efficient access.
    vertices: StorageMap<VertexKey, Vertex<U, D>>,

    /// Storage map for simplices, providing stable keys and efficient access.
    simplices: StorageMap<SimplexKey, Simplex<V, D>>,

    /// Fast mapping from Vertex UUIDs to their `VertexKeys` for efficient UUID → Key lookups.
    /// This optimizes the common operation of looking up vertex keys by UUID.
    /// For reverse Key → UUID lookups, we use direct storage map access: `vertices[key].uuid()`.
    ///
    /// INVARIANT: External mutation of this map will violate TDS invariants.
    /// This should only be modified through TDS methods that maintain consistency.
    ///
    /// Note: Not serialized - reconstructed during deserialization from vertices.
    pub(crate) uuid_to_vertex_key: UuidToVertexKeyMap,

    /// Fast mapping from Simplex UUIDs to their `SimplexKeys` for efficient UUID → Key lookups.
    /// This optimizes the common operation of looking up simplex keys by UUID.
    /// For reverse Key → UUID lookups, we use direct storage map access: `simplices[key].uuid()`.
    ///
    /// INVARIANT: External mutation of this map will violate TDS invariants.
    /// This should only be modified through TDS methods that maintain consistency.
    ///
    /// Note: Not serialized - reconstructed during deserialization from simplices.
    pub(crate) uuid_to_simplex_key: UuidToSimplexKeyMap,

    /// The current construction state of the triangulation.
    /// This field tracks whether the triangulation has enough vertices to form a complete
    /// D-dimensional triangulation or if it's still being incrementally built.
    ///
    /// Note: Not serialized - only constructed triangulations should be serialized.
    pub(crate) construction_state: TriangulationConstructionState,

    /// Generation counter for invalidating caches.
    /// This counter is incremented whenever the triangulation structure is modified
    /// (vertices added, simplices created/removed, etc.), allowing dependent caches to
    /// detect when they need to refresh.
    /// Uses `Arc<AtomicU64>` for thread-safe operations in concurrent contexts while allowing Clone.
    ///
    /// Note: Not serialized - generation is runtime-only.
    generation: Arc<AtomicU64>,

    /// Runtime identity for cache/handle provenance checks.
    ///
    /// Cloning or deserializing a `Tds` creates a fresh identity so handles cached
    /// from another storage snapshot cannot be reused against the reconstructed
    /// storage by generation alone.
    ///
    /// Note: Not serialized - identity is runtime-only.
    identity: Arc<Uuid>,
}

/// Topology validation mode for checked TDS simplex insertion.
#[derive(Clone, Copy)]
enum SimplexInsertionTopologyCheck {
    /// Validate the candidate against all existing simplices.
    Checked,
    /// Skip global topology scans because the caller validated the local cavity.
    Prechecked,
}

impl<U, V, const D: usize> Clone for Tds<U, V, D>
where
    U: Clone,
    V: Clone,
{
    fn clone(&self) -> Self {
        Self {
            vertices: self.vertices.clone(),
            simplices: self.simplices.clone(),
            uuid_to_vertex_key: self.uuid_to_vertex_key.clone(),
            uuid_to_simplex_key: self.uuid_to_simplex_key.clone(),
            construction_state: self.construction_state.clone(),
            generation: Arc::new(AtomicU64::new(self.generation.load(Ordering::Relaxed))),
            identity: Arc::new(Uuid::new_v4()),
        }
    }
}

impl<U, V, const D: usize> Tds<U, V, D>
where
    U: Clone,
    V: Clone,
{
    /// Clones storage for an internal transactional snapshot while preserving
    /// the runtime identity promised to cache and handle provenance checks.
    pub(crate) fn clone_for_rollback(&self) -> Self {
        Self {
            vertices: self.vertices.clone(),
            simplices: self.simplices.clone(),
            uuid_to_vertex_key: self.uuid_to_vertex_key.clone(),
            uuid_to_simplex_key: self.uuid_to_simplex_key.clone(),
            construction_state: self.construction_state.clone(),
            generation: Arc::new(AtomicU64::new(self.generation.load(Ordering::Relaxed))),
            identity: Arc::clone(&self.identity),
        }
    }
}

// =============================================================================
// CORE FUNCTIONALITY
// =============================================================================

// =============================================================================
// PURE COMBINATORIAL METHODS (No geometric operations)
// =============================================================================
// These methods work with the combinatorial structure only - vertices, simplices,
// neighbors, facets, keys, and UUIDs. They do NOT require coordinate operations
// and are designed to be independent of geometry.
//
// Following CGAL's Triangulation_data_structure pattern, these methods operate
// on topology independently of geometry.
//
impl<U, V, const D: usize> Tds<U, V, D> {
    #[inline]
    fn allows_periodic_self_neighbor(simplex: &Simplex<V, D>) -> bool {
        let Some(offsets) = simplex.periodic_vertex_offsets() else {
            return false;
        };
        !offsets.is_empty() && offsets.len() == simplex.number_of_vertices()
    }
    fn periodic_facet_key_from_simplex_vertices(
        simplex: &Simplex<V, D>,
        vertices: &[VertexKey],
        facet_index: usize,
    ) -> Result<u64, TdsError> {
        if facet_index >= vertices.len() {
            return Err(TdsError::IndexOutOfBounds {
                index: facet_index,
                bound: vertices.len(),
                context: format!("facet index for simplex with {} vertices", vertices.len()),
            });
        }

        let Some(periodic_offsets) = simplex.periodic_vertex_offsets() else {
            // Non-periodic path: build facet_vertices only when needed
            let mut facet_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
                SmallBuffer::new();
            for (i, &vertex_key) in vertices.iter().enumerate() {
                if i != facet_index {
                    facet_vertices.push(vertex_key);
                }
            }
            return Ok(facet_key_from_vertices(&facet_vertices));
        };

        if periodic_offsets.len() != vertices.len() {
            return Err(TdsError::DimensionMismatch {
                expected: vertices.len(),
                actual: periodic_offsets.len(),
                context: "simplex periodic offset count vs vertex count".to_string(),
            });
        }

        let mut lifted_vertices: SmallBuffer<(VertexKey, [i8; D]), MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::new();
        for (vertex_idx, &vertex_key) in vertices.iter().enumerate() {
            lifted_vertices.push((vertex_key, periodic_offsets[vertex_idx]));
        }

        periodic_facet_key_from_lifted_vertices::<D>(&lifted_vertices, facet_index).map_err(
            |error| TdsError::InconsistentDataStructure {
                message: format!(
                    "Failed to derive periodic facet key for simplex {:?} facet {facet_index}: {error}",
                    simplex.uuid()
                ),
            },
        )
    }

    fn build_periodic_vertex_uuid_offsets(
        &self,
        simplex_key: SimplexKey,
        vertices: &[VertexKey],
    ) -> Result<SimplexUuidSortKey<D>, TdsError> {
        let simplex = self
            .simplices
            .get(simplex_key)
            .ok_or_else(|| TdsError::SimplexNotFound {
                simplex_key,
                context: "building periodic vertex identity (UUID/offset)".to_string(),
            })?;

        let periodic_offsets = simplex.periodic_vertex_offsets();
        if let Some(offsets) = periodic_offsets
            && offsets.len() != vertices.len()
        {
            return Err(TdsError::DimensionMismatch {
                expected: vertices.len(),
                actual: offsets.len(),
                context: format!("simplex {simplex_key:?} periodic offset count vs vertex count"),
            });
        }

        let mut vertex_uuid_offsets = SimplexUuidSortKey::<D>::new();
        for (vertex_idx, &vertex_key) in vertices.iter().enumerate() {
            let vertex = self
                .vertices
                .get(vertex_key)
                .ok_or_else(|| TdsError::VertexNotFound {
                    vertex_key,
                    context: format!(
                        "referenced by simplex {simplex_key:?} at index {vertex_idx} while building periodic vertex identity (UUID/offset)",
                    ),
                })?;
            let offset = periodic_offsets.map_or([0_i8; D], |offsets| offsets[vertex_idx]);
            vertex_uuid_offsets.push((vertex.uuid(), offset));
        }
        vertex_uuid_offsets.sort_unstable();

        Ok(vertex_uuid_offsets)
    }

    fn lifted_vertex_identities(
        simplex_key: SimplexKey,
        simplex: &Simplex<V, D>,
    ) -> Result<SmallBuffer<(VertexKey, [i8; D]), MAX_PRACTICAL_DIMENSION_SIZE>, TdsError> {
        let vertices = simplex.vertices();
        let periodic_offsets = simplex.periodic_vertex_offsets();
        if let Some(offsets) = periodic_offsets
            && offsets.len() != vertices.len()
        {
            return Err(TdsError::DimensionMismatch {
                expected: vertices.len(),
                actual: offsets.len(),
                context: format!(
                    "simplex {simplex_key:?} periodic offset count vs vertex count in neighbor topology validation"
                ),
            });
        }

        let mut lifted_vertices: SmallBuffer<(VertexKey, [i8; D]), MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::new();
        for (vertex_idx, &vertex_key) in vertices.iter().enumerate() {
            let offset = periodic_offsets.map_or([0_i8; D], |offsets| offsets[vertex_idx]);
            lifted_vertices.push((vertex_key, offset));
        }

        Ok(lifted_vertices)
    }

    fn matching_lifted_facet_index(
        simplex: &Simplex<V, D>,
        neighbor: &Simplex<V, D>,
    ) -> Result<Option<usize>, TdsError> {
        let simplex_vertices = simplex.vertices();
        let neighbor_vertices = neighbor.vertices();

        for simplex_facet_index in 0..simplex_vertices.len() {
            let simplex_facet_key = Self::periodic_facet_key_from_simplex_vertices(
                simplex,
                simplex_vertices,
                simplex_facet_index,
            )?;
            for neighbor_facet_index in 0..neighbor_vertices.len() {
                let neighbor_facet_key = Self::periodic_facet_key_from_simplex_vertices(
                    neighbor,
                    neighbor_vertices,
                    neighbor_facet_index,
                )?;
                if simplex_facet_key == neighbor_facet_key {
                    return Ok(Some(simplex_facet_index));
                }
            }
        }

        Ok(None)
    }

    /// Finds the neighbor facet that matches a source facet in lifted periodic coordinates.
    fn matching_lifted_mirror_facet_index(
        simplex: &Simplex<V, D>,
        facet_idx: usize,
        neighbor: &Simplex<V, D>,
        context: &str,
    ) -> Result<usize, TdsError> {
        let simplex_facet_key =
            Self::periodic_facet_key_from_simplex_vertices(simplex, simplex.vertices(), facet_idx)?;
        let mut mirror_idx = None;
        for neighbor_facet_idx in 0..neighbor.vertices().len() {
            let neighbor_facet_key = Self::periodic_facet_key_from_simplex_vertices(
                neighbor,
                neighbor.vertices(),
                neighbor_facet_idx,
            )?;
            if neighbor_facet_key == simplex_facet_key
                && mirror_idx.replace(neighbor_facet_idx).is_some()
            {
                return Err(TdsError::InvalidNeighbors {
                    reason: NeighborValidationError::MirrorFacetAmbiguous {
                        simplex_uuid: simplex.uuid(),
                        neighbor_uuid: neighbor.uuid(),
                    },
                });
            }
        }

        mirror_idx.ok_or_else(|| TdsError::InvalidNeighbors {
            reason: NeighborValidationError::MirrorFacetMissing {
                simplex_uuid: simplex.uuid(),
                facet_index: facet_idx,
                neighbor_uuid: neighbor.uuid(),
                context: context.to_string(),
            },
        })
    }

    pub(crate) fn facet_key_for_simplex_facet(
        &self,
        simplex_key: SimplexKey,
        facet_index: usize,
    ) -> Result<u64, TdsError> {
        let vertices = self.simplex_vertices(simplex_key)?;
        let simplex = self
            .simplices
            .get(simplex_key)
            .ok_or_else(|| TdsError::SimplexNotFound {
                simplex_key,
                context: format!("deriving facet key for index {facet_index}"),
            })?;
        Self::periodic_facet_key_from_simplex_vertices(simplex, &vertices, facet_index)
    }
    /// Assigns neighbor relationships between simplices based on shared facets with semantic ordering.
    ///
    /// This method efficiently builds neighbor relationships by using the `facet_key_from_vertices`
    /// function to compute unique keys for facets. Two simplices are considered neighbors if they share
    /// exactly one facet (which contains D vertices for a D-dimensional triangulation).
    ///
    /// **Note**: This is a purely combinatorial operation that does not perform any coordinate
    /// operations. It works entirely with vertex keys, simplex keys, and topological relationships.
    ///
    /// **Internal use only**: This method rebuilds ALL neighbor pointers from scratch, which is
    /// inefficient for most use cases. For external use, prefer
    /// [`repair_neighbor_pointers`](crate::prelude::insertion::repair_neighbor_pointers),
    /// which rebuilds neighbor pointers from facet incidence.
    ///
    /// # Errors
    ///
    /// Returns `TdsError` if neighbor assignment fails due to inconsistent
    /// data structures or invalid facet sharing patterns.
    pub(crate) fn assign_neighbors(&mut self) -> Result<(), TdsError> {
        // Build facet mapping with vertex index information using optimized collections
        // facet_key -> [(simplex_key, vertex_index_opposite_to_facet)]
        type FacetInfo = (SimplexKey, usize);
        // Use saturating arithmetic to avoid potential overflow on adversarial inputs
        let cap = self.simplices.len().saturating_mul(D.saturating_add(1));
        let mut facet_map: FastHashMap<u64, SmallBuffer<FacetInfo, 2>> =
            fast_hash_map_with_capacity(cap);

        for (simplex_key, simplex) in &self.simplices {
            let vertices = self.simplex_vertices(simplex_key).map_err(|err| {
                TdsError::VertexKeyRetrievalFailed {
                    simplex_id: simplex.uuid(),
                    message: format!(
                        "Failed to retrieve vertex keys for simplex during neighbor assignment: {err}"
                    ),
                }
            })?;

            for i in 0..vertices.len() {
                let facet_key =
                    Self::periodic_facet_key_from_simplex_vertices(simplex, &vertices, i)?;
                let facet_entry = facet_map.entry(facet_key).or_default();
                // Detect degenerate case early: more than 2 simplices sharing a facet
                // Note: Check happens before push, so len() reflects current sharing count
                if facet_entry.len() >= 2 {
                    return Err(TdsError::InconsistentDataStructure {
                        message: format!(
                            "Facet with key {} already shared by {} simplices; cannot add simplex {} (would violate 2-manifold property)",
                            facet_key,
                            facet_entry.len(),
                            simplex.uuid()
                        ),
                    });
                }
                facet_entry.push((simplex_key, i));
            }
        }

        // For each simplex, build an ordered neighbor array where neighbors[i] is opposite vertices[i]
        let mut simplex_neighbors: FastHashMap<
            SimplexKey,
            SmallBuffer<Option<SimplexKey>, MAX_PRACTICAL_DIMENSION_SIZE>,
        > = fast_hash_map_with_capacity(self.simplices.len());

        // Initialize each simplex with a SmallBuffer of None values (one per vertex)
        for (simplex_key, simplex) in &self.simplices {
            let vertex_count = simplex.number_of_vertices();
            if vertex_count > MAX_PRACTICAL_DIMENSION_SIZE {
                return Err(TdsError::InconsistentDataStructure {
                    message: format!(
                        "simplex {} vertex count ({vertex_count}) exceeds storage limit MAX_PRACTICAL_DIMENSION_SIZE ({MAX_PRACTICAL_DIMENSION_SIZE}) (would overflow neighbors buffer)",
                        simplex.uuid(),
                    ),
                });
            }
            let mut neighbors = SmallBuffer::with_capacity(vertex_count);
            neighbors.resize(vertex_count, None);
            simplex_neighbors.insert(simplex_key, neighbors);
        }

        // For each facet that is shared by exactly two simplices, establish neighbor relationships
        // Note: >2 simplices per facet already caught by early check during map build (above)
        for (_facet_key, facet_infos) in facet_map {
            if facet_infos.len() != 2 {
                continue;
            }

            let (simplex_key1, vertex_index1) = facet_infos[0];
            let (simplex_key2, vertex_index2) = facet_infos[1];

            // Set neighbors with semantic constraint: neighbors[i] is opposite vertices[i]
            simplex_neighbors.get_mut(&simplex_key1).ok_or_else(|| {
                TdsError::SimplexNotFound {
                    simplex_key: simplex_key1,
                    context: "assign_neighbors: simplex missing from local neighbors map"
                        .to_string(),
                }
            })?[vertex_index1] = Some(simplex_key2);

            simplex_neighbors.get_mut(&simplex_key2).ok_or_else(|| {
                TdsError::SimplexNotFound {
                    simplex_key: simplex_key2,
                    context: "assign_neighbors: simplex missing from local neighbors map"
                        .to_string(),
                }
            })?[vertex_index2] = Some(simplex_key1);
        }

        // Apply updates. Even simplices with only boundary facets receive an
        // assigned boundary buffer so assigned-boundary and unassigned states
        // remain distinct.
        for (simplex_key, neighbors) in &simplex_neighbors {
            if let Some(simplex) = self.simplices.get_mut(*simplex_key) {
                let simplex_id = simplex.uuid();
                simplex
                    .set_neighbors_from_keys(neighbors.iter().copied())
                    .map_err(|source| TdsError::InvalidSimplex { simplex_id, source })?;
            }
        }

        // Topology changed; invalidate caches.
        self.bump_generation();

        Ok(())
    }

    /// Returns an iterator over all simplices in the triangulation.
    ///
    /// This method provides read-only access to the simplices collection without
    /// exposing the underlying storage implementation. The iterator yields
    /// `(SimplexKey, &Simplex)` pairs for each simplex in the triangulation.
    ///
    /// For direct key-based access, use [`simplex`](Self::simplex).
    ///
    /// # Returns
    ///
    /// An iterator over `(SimplexKey, &Simplex<V, D>)` pairs.
    ///
    /// # Example
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)]
    /// #     Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)]
    /// #     TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)]
    /// #     Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)]
    /// #     Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)]
    /// #     Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Validation(#[from] delaunay::DelaunayTriangulationValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// for (simplex_key, simplex) in dt.simplices() {
    ///     println!("Simplex {:?} has {} vertices", simplex_key, simplex.number_of_vertices());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn simplices(&self) -> impl Iterator<Item = (SimplexKey, &Simplex<V, D>)> {
        self.simplices.iter()
    }

    /// Returns an iterator over all simplex values (without keys) in the triangulation.
    ///
    /// This is a convenience method that simplifies the common pattern of iterating over
    /// `simplices().map(|(_, simplex)| simplex)`. It provides read-only access to simplex objects
    /// when you don't need the simplex keys.
    ///
    /// # Returns
    ///
    /// An iterator over `&Simplex<V, D>` references.
    ///
    /// # Example
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)]
    /// #     Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)]
    /// #     TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)]
    /// #     Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)]
    /// #     Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)]
    /// #     Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Validation(#[from] delaunay::DelaunayTriangulationValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// for (_, simplex) in dt.simplices() {
    ///     println!("Simplex has {} vertices", simplex.number_of_vertices());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn simplices_values(&self) -> impl Iterator<Item = &Simplex<V, D>> {
        self.simplices.values()
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
    /// For direct key-based access, use [`vertex`](Self::vertex).
    ///
    /// # Returns
    ///
    /// An iterator over `(VertexKey, &Vertex<U, D>)` pairs.
    ///
    /// # Example
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Coordinates(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsMutation(#[from] delaunay::prelude::tds::TdsMutationError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.5, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// for (vertex_key, vertex) in dt.vertices() {
    ///     println!("Vertex {:?} at {:?}", vertex_key, vertex.point());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn vertices(&self) -> impl Iterator<Item = (VertexKey, &Vertex<U, D>)> {
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
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Coordinates(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsMutation(#[from] delaunay::prelude::tds::TdsMutationError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let keys: Vec<_> = dt.tds().vertex_keys().collect();
    /// assert_eq!(keys.len(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn vertex_keys(&self) -> impl Iterator<Item = VertexKey> + '_ {
        self.vertices.keys()
    }

    /// Returns an iterator over all simplex keys in the triangulation.
    ///
    /// # Returns
    ///
    /// An iterator over `SimplexKey` values.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Coordinates(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsMutation(#[from] delaunay::prelude::tds::TdsMutationError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let keys: Vec<_> = dt.tds().simplex_keys().collect();
    /// assert_eq!(keys.len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn simplex_keys(&self) -> impl Iterator<Item = SimplexKey> + '_ {
        self.simplices.keys()
    }

    /// Returns the concrete simplex-key iterator for internal iterator structs
    /// that need to store traversal state without allocating a key snapshot.
    pub(crate) fn simplex_key_iter(&self) -> slotmap::dense::Keys<'_, SimplexKey, Simplex<V, D>> {
        self.simplices.keys()
    }

    /// Returns a reference to a simplex by its key.
    ///
    /// # Returns
    ///
    /// `Some(&Simplex)` if the key exists, `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsMutation(#[from] delaunay::prelude::tds::TdsMutationError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let tds = dt.tds();
    /// let Some(simplex_key) = tds.simplex_keys().next() else {
    ///     return Ok(());
    /// };
    /// let Some(simplex) = tds.simplex(simplex_key) else {
    ///     return Ok(());
    /// };
    /// assert_eq!(simplex.number_of_vertices(), 3);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn simplex(&self, key: SimplexKey) -> Option<&Simplex<V, D>> {
        self.simplices.get(key)
    }

    /// Checks if a simplex key exists in the triangulation.
    ///
    /// # Returns
    ///
    /// `true` if the key exists, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsMutation(#[from] delaunay::prelude::tds::TdsMutationError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let tds = dt.tds();
    /// let Some(simplex_key) = tds.simplex_keys().next() else {
    ///     return Ok(());
    /// };
    /// assert!(tds.contains_simplex(simplex_key));
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn contains_simplex(&self, key: SimplexKey) -> bool {
        self.simplices.contains_key(key)
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
    /// use delaunay::prelude::tds::Tds;
    ///
    /// let tds = Tds::<(), (), 3>::empty();
    /// assert_eq!(tds.number_of_vertices(), 0);
    /// ```
    ///
    /// Count vertices after adding them:
    ///
    /// ```no_run
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();
    /// let vertex1: Vertex<(), 3> = delaunay::prelude::Vertex::<(), _>::try_new([1.0, 2.0, 3.0])?;
    /// let vertex2: Vertex<(), 3> = delaunay::prelude::Vertex::<(), _>::try_new([4.0, 5.0, 6.0])?;
    ///
    /// dt.insert(vertex1)?;
    /// assert_eq!(dt.number_of_vertices(), 1);
    ///
    /// dt.insert(vertex2)?;
    /// assert_eq!(dt.number_of_vertices(), 2);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Count vertices initialized from points:
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)] Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let points = [
    ///     Point::try_from([0.0, 0.0, 0.0])?,
    ///     Point::try_from([1.0, 0.0, 0.0])?,
    ///     Point::try_from([0.0, 1.0, 0.0])?,
    ///     Point::try_from([0.0, 0.0, 1.0])?,
    /// ];
    ///
    /// let vertices = Vertex::<(), 3>::from_validated_points(&points);
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// assert_eq!(dt.number_of_vertices(), 4);
    /// # Ok(())
    /// # }
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
    /// use delaunay::prelude::tds::Tds;
    /// use delaunay::prelude::geometry::Point;
    /// use delaunay::prelude::geometry::Coordinate;
    ///
    /// let tds = Tds::<(), (), 3>::empty();
    /// assert_eq!(tds.dim(), -1); // Empty triangulation
    /// ```
    ///
    /// Dimension progression as vertices are added:
    ///
    /// ```no_run
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Coordinates(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();
    ///
    /// // Start empty
    /// assert_eq!(dt.dim(), -1);
    ///
    /// // Add vertices incrementally
    /// let vertex1: Vertex<(), 3> = delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?;
    /// dt.insert(vertex1)?;
    /// assert_eq!(dt.dim(), 0);
    ///
    /// let vertex2: Vertex<(), 3> = delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?;
    /// dt.insert(vertex2)?;
    /// assert_eq!(dt.dim(), 1);
    ///
    /// let vertex3: Vertex<(), 3> = delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?;
    /// dt.insert(vertex3)?;
    /// assert_eq!(dt.dim(), 2);
    ///
    /// let vertex4: Vertex<(), 3> = delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?;
    /// dt.insert(vertex4)?;
    /// assert_eq!(dt.number_of_vertices(), 4);
    /// assert_eq!(dt.dim(), 3);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Different dimensional triangulations:
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)] Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// // 2D triangulation
    /// let points_2d = [
    ///     Point::try_from([0.0, 0.0])?,
    ///     Point::try_from([1.0, 0.0])?,
    ///     Point::try_from([0.5, 1.0])?,
    /// ];
    /// let vertices_2d = Vertex::<(), 2>::from_validated_points(&points_2d);
    /// let dt_2d = DelaunayTriangulationBuilder::new(&vertices_2d).build::<()>()?;
    /// assert_eq!(dt_2d.dim(), 2);
    ///
    /// // 4D triangulation with 5 vertices (minimum for 4D simplex)
    /// let points_4d = [
    ///     Point::try_from([0.0, 0.0, 0.0, 0.0])?,
    ///     Point::try_from([1.0, 0.0, 0.0, 0.0])?,
    ///     Point::try_from([0.0, 1.0, 0.0, 0.0])?,
    ///     Point::try_from([0.0, 0.0, 1.0, 0.0])?,
    ///     Point::try_from([0.0, 0.0, 0.0, 1.0])?,
    /// ];
    /// let vertices_4d = Vertex::<(), 4>::from_validated_points(&points_4d);
    /// let dt_4d = DelaunayTriangulationBuilder::new(&vertices_4d).build::<()>()?;
    /// assert_eq!(dt_4d.dim(), 4);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn dim(&self) -> i32 {
        let nv = self.number_of_vertices();
        // Convert to i32 first, then subtract to handle empty case (0 - 1 = -1)
        let nv_i32 = i32::try_from(nv).unwrap_or(i32::MAX);
        let d_i32 = i32::try_from(D).unwrap_or(i32::MAX);
        nv_i32.saturating_sub(1).min(d_i32)
    }

    /// Returns the current construction state of this triangulation data structure.
    ///
    /// The state is maintained by checked TDS and Delaunay construction paths. It
    /// is exposed read-only so callers can inspect incomplete vs. constructed
    /// topology without bypassing mutation invariants.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::tds::{Tds, TriangulationConstructionState};
    ///
    /// let tds: Tds<(), (), 3> = Tds::empty();
    /// std::assert_matches!(
    ///     tds.construction_state(),
    ///     TriangulationConstructionState::Incomplete(0)
    /// );
    /// ```
    #[inline]
    #[must_use]
    pub const fn construction_state(&self) -> &TriangulationConstructionState {
        &self.construction_state
    }

    /// The function `number_of_simplices` returns the number of simplices in the [Tds].
    ///
    /// # Returns
    ///
    /// The number of [Simplex]s in the [Tds].
    ///
    /// # Examples
    ///
    /// Count simplices in a newly created triangulation:
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)] Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let points = [
    ///     Point::try_from([0.0, 0.0, 0.0])?,
    ///     Point::try_from([1.0, 0.0, 0.0])?,
    ///     Point::try_from([0.0, 1.0, 0.0])?,
    ///     Point::try_from([0.0, 0.0, 1.0])?,
    /// ];
    ///
    /// let vertices = Vertex::<(), 3>::from_validated_points(&points);
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// assert_eq!(dt.number_of_simplices(), 1); // Simplices are automatically created via triangulation
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Count simplices after triangulation:
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)] Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let points = [
    ///     Point::try_from([0.0, 0.0, 0.0])?,
    ///     Point::try_from([1.0, 0.0, 0.0])?,
    ///     Point::try_from([0.0, 1.0, 0.0])?,
    ///     Point::try_from([0.0, 0.0, 1.0])?,
    /// ];
    ///
    /// let vertices = Vertex::<(), 3>::from_validated_points(&points);
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// assert_eq!(dt.number_of_simplices(), 1); // One tetrahedron for 4 points in 3D
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Empty triangulation has no simplices:
    ///
    /// ```
    /// use delaunay::prelude::tds::Tds;
    ///
    /// let tds = Tds::<(), (), 3>::empty();
    /// assert_eq!(tds.number_of_simplices(), 0); // No simplices for empty input
    /// ```
    #[must_use]
    pub fn number_of_simplices(&self) -> usize {
        self.simplices.len()
    }

    /// Returns `true` if the simplex neighbor graph is a single connected component.
    ///
    /// An empty triangulation (no simplices) is trivially connected.
    ///
    /// Connectivity is a **topology-layer** (Level 3) invariant: it is not checked
    /// by [`Tds::is_valid`] (Level 2), but it *is* checked by [`Triangulation::is_valid`].
    /// This method exposes the underlying BFS so that diagnostic code and the
    /// `Triangulation`-layer check can both reuse the same primitive without going
    /// through a full `Triangulation` wrapper.
    ///
    /// Time complexity: O(N · D), where N is the number of simplices (each simplex has at most
    /// D+1 neighbors, so the BFS visits at most N·(D+1) edges).
    ///
    /// [`Triangulation::is_valid`]: crate::prelude::triangulation::Triangulation::is_valid
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// assert!(dt.tds().is_connected());
    ///
    /// let empty = dt.tds().number_of_simplices() == 0 || dt.tds().is_connected();
    /// assert!(empty);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn is_connected(&self) -> bool {
        let total = self.simplices.len();
        if total == 0 {
            return true;
        }

        let Some(start) = self.simplex_keys().next() else {
            return true;
        };

        let mut visited: SimplexKeySet = SimplexKeySet::default();
        visited.reserve(total);
        let mut stack: Vec<SimplexKey> = Vec::with_capacity(total.min(64));
        stack.push(start);

        while let Some(ck) = stack.pop() {
            if !visited.insert(ck) {
                continue;
            }
            let Some(simplex) = self.simplices.get(ck) else {
                continue;
            };
            let Some(neighbors) = simplex.neighbor_keys() else {
                continue;
            };
            for n_opt in neighbors {
                let Some(nk) = n_opt else {
                    continue;
                };
                if self.simplices.contains_key(nk) && !visited.contains(&nk) {
                    stack.push(nk);
                }
            }
        }

        visited.len() == total
    }

    /// Increments the generation counter to invalidate dependent caches.
    ///
    /// This method should be called whenever the triangulation structure is modified
    /// (vertices added, simplices created/removed, etc.). It uses relaxed memory ordering
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
    /// use delaunay::prelude::tds::Tds;
    ///
    /// let tds: Tds<(), (), 2> = Tds::empty();
    /// assert_eq!(tds.generation(), 0);
    /// ```
    #[inline]
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Relaxed)
    }

    /// Returns the runtime identity used by cache owners to reject handles from another TDS.
    #[inline]
    pub(crate) const fn identity(&self) -> &Arc<Uuid> {
        &self.identity
    }

    /// Marks the triangulation topology as modified and invalidates generation-keyed caches.
    ///
    /// This is intended for crate-internal mutation paths that adjust simplex slot ordering
    /// without going through the standard insertion/removal APIs.
    #[inline]
    pub(crate) fn mark_topology_modified(&self) {
        self.bump_generation();
    }
}

impl<U, V, const D: usize> Tds<U, V, D> {
    // =========================================================================
    // QUERY OPERATIONS
    // =========================================================================

    /// Atomically inserts a vertex and creates the UUID-to-key mapping.
    ///
    /// This method ensures that both the vertex insertion and UUID mapping are
    /// performed together, maintaining data structure invariants.
    ///
    /// **⚠️ INTERNAL API WARNING**: This method bypasses atomicity guarantees for topology
    /// assignment operations (`assign_neighbors()` and `assign_incident_simplices()`). It only
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
        vertex: Vertex<U, D>,
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

    /// Atomically inserts a simplex and creates the UUID-to-key mapping.
    ///
    /// This method ensures that both the simplex insertion and UUID mapping are
    /// performed together, maintaining data structure invariants. This is preferred
    /// over separate raw storage insertion plus `uuid_to_simplex_key.insert()` calls, which can
    /// leave the data structure in an inconsistent state if interrupted.
    ///
    /// # Arguments
    ///
    /// * `simplex` - The simplex to insert
    ///
    /// # Returns
    ///
    /// The `SimplexKey` that can be used to access the inserted simplex.
    ///
    /// # Errors
    ///
    /// Returns [`TdsConstructionError::DuplicateUuid`] if a simplex with the
    /// same UUID already exists in the triangulation, or
    /// [`TdsConstructionError::ValidationError`] if the simplex arity does not
    /// match `D + 1`, or if any vertex key referenced by `simplex` is not present
    /// in this TDS.
    ///
    /// # Examples
    ///
    ///
    /// See the unit tests for usage examples of this pub(crate) method.
    pub(crate) fn insert_simplex_with_mapping(
        &mut self,
        simplex: Simplex<V, D>,
    ) -> Result<SimplexKey, TdsConstructionError> {
        self.insert_simplex_with_mapping_impl(simplex, SimplexInsertionTopologyCheck::Checked)
    }

    /// Atomically inserts a simplex after validating vertex provenance.
    ///
    /// Cavity filling, flips, and explicit construction should only build
    /// `simplex` from vertex keys that came from this TDS and are still live. This
    /// method still verifies that invariant in every build mode so stale keys
    /// fail with a typed error instead of corrupting TDS invariants.
    ///
    /// # Errors
    ///
    /// Returns [`TdsConstructionError::DuplicateUuid`] if a simplex with the same
    /// UUID already exists, [`TdsConstructionError::ValidationError`] if the
    /// arity is wrong, or [`TdsConstructionError::ValidationError`] if any
    /// referenced vertex key is not present in this TDS.
    pub(crate) fn insert_simplex_with_mapping_trusted_vertices(
        &mut self,
        simplex: Simplex<V, D>,
    ) -> Result<SimplexKey, TdsConstructionError> {
        self.insert_simplex_with_mapping_impl(simplex, SimplexInsertionTopologyCheck::Checked)
    }

    /// Inserts a caller-validated simplex without global topology scans.
    ///
    /// Hull-extension callers use this only after proving that candidate simplices
    /// are built from visible boundary facets around a fresh apex and before
    /// immediately wiring local neighbors and validating the affected topology.
    pub(crate) fn insert_simplex_with_mapping_prechecked_topology(
        &mut self,
        simplex: Simplex<V, D>,
    ) -> Result<SimplexKey, TdsConstructionError> {
        self.insert_simplex_with_mapping_impl(simplex, SimplexInsertionTopologyCheck::Prechecked)
    }

    /// Shared checked simplex-insertion path used by public and trusted internal wrappers.
    fn insert_simplex_with_mapping_impl(
        &mut self,
        simplex: Simplex<V, D>,
        topology_check: SimplexInsertionTopologyCheck,
    ) -> Result<SimplexKey, TdsConstructionError> {
        if simplex.number_of_vertices() != D + 1 {
            return Err(TdsConstructionError::ValidationError(
                TdsError::DimensionMismatch {
                    expected: D + 1,
                    actual: simplex.number_of_vertices(),
                    context: format!("{D}-dimensional simplex vertex count"),
                },
            ));
        }

        self.validate_simplex_vertices_exist(&simplex)?;

        let simplex_uuid = simplex.uuid();
        if self.uuid_to_simplex_key.contains_key(&simplex_uuid) {
            return Err(TdsConstructionError::DuplicateUuid {
                entity: EntityKind::Simplex,
                uuid: simplex_uuid,
            });
        }

        match topology_check {
            SimplexInsertionTopologyCheck::Checked => {
                self.validate_simplex_topology_safe_for_insertion(&simplex)?;
            }
            SimplexInsertionTopologyCheck::Prechecked => {}
        }

        let simplex_key = self.simplices.insert(simplex);
        self.uuid_to_simplex_key.insert(simplex_uuid, simplex_key);
        // Topology changed; invalidate caches.
        self.bump_generation();
        Ok(simplex_key)
    }

    /// Verifies that inserting `simplex` would not violate local topology invariants.
    fn validate_simplex_topology_safe_for_insertion(
        &self,
        simplex: &Simplex<V, D>,
    ) -> Result<(), TdsConstructionError> {
        self.validate_no_duplicate_simplex_on_insert(simplex)?;
        self.validate_facet_sharing_on_insert(simplex)?;
        Ok(())
    }

    /// Builds the duplicate-simplex identity for a not-yet-inserted simplex.
    fn candidate_periodic_vertex_uuid_offsets(
        &self,
        simplex: &Simplex<V, D>,
    ) -> Result<SimplexUuidSortKey<D>, TdsError> {
        let vertices = simplex.vertices();
        let periodic_offsets = simplex.periodic_vertex_offsets();
        if let Some(offsets) = periodic_offsets
            && offsets.len() != vertices.len()
        {
            return Err(TdsError::DimensionMismatch {
                expected: vertices.len(),
                actual: offsets.len(),
                context: format!(
                    "candidate simplex {} periodic offset count vs vertex count",
                    simplex.uuid()
                ),
            });
        }

        let mut vertex_uuid_offsets = SimplexUuidSortKey::<D>::new();
        for (vertex_idx, &vertex_key) in vertices.iter().enumerate() {
            let vertex = self
                .vertices
                .get(vertex_key)
                .ok_or_else(|| TdsError::VertexNotFound {
                    vertex_key,
                    context: format!(
                        "referenced by candidate simplex {} at index {vertex_idx} while building periodic vertex identity (UUID/offset)",
                        simplex.uuid()
                    ),
                })?;
            let offset = periodic_offsets.map_or([0_i8; D], |offsets| offsets[vertex_idx]);
            vertex_uuid_offsets.push((vertex.uuid(), offset));
        }
        vertex_uuid_offsets.sort_unstable();

        Ok(vertex_uuid_offsets)
    }

    /// Rejects a candidate simplex that duplicates an existing maximal simplex.
    fn validate_no_duplicate_simplex_on_insert(
        &self,
        simplex: &Simplex<V, D>,
    ) -> Result<(), TdsError> {
        let candidate_identity = self.candidate_periodic_vertex_uuid_offsets(simplex)?;

        for (existing_simplex_key, _existing_simplex) in &self.simplices {
            let vertices = self.simplex_vertices(existing_simplex_key)?;
            let existing_identity =
                self.build_periodic_vertex_uuid_offsets(existing_simplex_key, &vertices)?;

            if existing_identity == candidate_identity {
                return Err(TdsError::DuplicateSimplices {
                    message: format!(
                        "Refusing to insert duplicate simplex {} with same vertex UUIDs as existing simplex {existing_simplex_key:?}: {candidate_identity:?}",
                        simplex.uuid()
                    ),
                });
            }
        }

        Ok(())
    }

    /// Rejects a candidate simplex whose facets would be incident to a third simplex.
    fn validate_facet_sharing_on_insert(&self, simplex: &Simplex<V, D>) -> Result<(), TdsError> {
        let vertices = simplex.vertices();
        for candidate_facet_idx in 0..vertices.len() {
            let candidate_facet_key = Self::periodic_facet_key_from_simplex_vertices(
                simplex,
                vertices,
                candidate_facet_idx,
            )?;
            let mut incident_count = 0_usize;

            for (existing_simplex_key, existing_simplex) in &self.simplices {
                let existing_vertices = self.simplex_vertices(existing_simplex_key)?;
                for existing_facet_idx in 0..existing_vertices.len() {
                    let existing_facet_key = Self::periodic_facet_key_from_simplex_vertices(
                        existing_simplex,
                        &existing_vertices,
                        existing_facet_idx,
                    )?;
                    if existing_facet_key == candidate_facet_key {
                        incident_count += 1;
                        if incident_count >= 2 {
                            return Err(TdsError::FacetSharingViolation {
                                facet_key: candidate_facet_key,
                                existing_incident_count: incident_count,
                                attempted_incident_count: incident_count + 1,
                                max_incident_count: 2,
                                candidate_simplex_uuid: simplex.uuid(),
                                candidate_facet_index: candidate_facet_idx,
                            });
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Verifies every vertex key referenced by `simplex` is live in this TDS.
    fn validate_simplex_vertices_exist(
        &self,
        simplex: &Simplex<V, D>,
    ) -> Result<(), TdsConstructionError> {
        for &vkey in simplex.vertices() {
            if !self.vertices.contains_key(vkey) {
                return Err(TdsConstructionError::ValidationError(
                    TdsError::VertexNotFound {
                        vertex_key: vkey,
                        context: "referenced by simplex being inserted".to_string(),
                    },
                ));
            }
        }
        Ok(())
    }

    /// Inserts a simplex while intentionally bypassing topology safety checks in tests.
    #[cfg(test)]
    pub(crate) fn insert_simplex_bypassing_topology_checks_for_test(
        &mut self,
        simplex: Simplex<V, D>,
    ) -> Result<SimplexKey, TdsConstructionError> {
        self.insert_simplex_with_mapping_impl(simplex, SimplexInsertionTopologyCheck::Prechecked)
    }
}

impl<U, V, const D: usize> Tds<U, V, D> {
    /// Gets validated vertex keys for a simplex.
    ///
    /// This performs O(D) validation and copying of the requested simplex's
    /// vertex keys into a stack-friendly buffer.
    ///
    /// This method provides:
    /// - O(1) simplex lookup via storage map key
    /// - O(D) validation that all vertex keys exist in the triangulation
    /// - Direct key access without UUID→key lookups
    /// - Stack-allocated buffer for D ≤ 7 to avoid heap allocation
    ///
    /// # Arguments
    ///
    /// * `simplex_key` - The key of the simplex whose vertex keys we need
    ///
    /// # Returns
    ///
    /// A `Result` containing a `VertexKeyBuffer` if the simplex exists and all vertices are valid,
    /// or a `TdsError` if the simplex doesn't exist or vertices are missing.
    ///
    /// # Errors
    ///
    /// Returns a `TdsError` if:
    /// - The simplex with the given key doesn't exist
    /// - A vertex key from the simplex doesn't exist in the vertex storage (TDS corruption)
    ///
    /// # Performance
    ///
    /// This uses direct storage map access with O(1) key lookup for the simplex and O(D)
    /// validation for vertex keys. Uses stack-allocated buffer for D ≤ 7 to avoid heap
    /// allocation in the hot path.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let tds = dt.tds();
    /// let Some(simplex_key) = tds.simplex_keys().next() else {
    ///     return Ok(());
    /// };
    /// let keys = tds.simplex_vertices(simplex_key)?;
    /// assert_eq!(keys.len(), 3);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn simplex_vertices(&self, simplex_key: SimplexKey) -> Result<VertexKeyBuffer, TdsError> {
        let simplex = self
            .simplices
            .get(simplex_key)
            .ok_or_else(|| TdsError::SimplexNotFound {
                simplex_key,
                context: "simplex_vertices lookup".to_string(),
            })?;

        // Validate and collect keys in one pass to avoid redundant iteration.
        let simplex_vertices = simplex.vertices();
        let mut keys = VertexKeyBuffer::with_capacity(simplex_vertices.len());
        for (idx, &vertex_key) in simplex_vertices.iter().enumerate() {
            if !self.vertices.contains_key(vertex_key) {
                return Err(TdsError::VertexNotFound {
                    vertex_key,
                    context: format!(
                        "referenced by simplex {} (key {simplex_key:?}) at position {idx}",
                        simplex.uuid()
                    ),
                });
            }
            keys.push(vertex_key);
        }
        Ok(keys)
    }

    /// Helper function to get a simplex key from a simplex UUID using the optimized UUID→Key mapping.
    ///
    /// # Arguments
    ///
    /// * `simplex_uuid` - The UUID of the simplex to look up
    ///
    /// # Returns
    ///
    /// An `Option<SimplexKey>` if the simplex is found, `None` otherwise.
    ///
    /// # Performance
    ///
    /// This uses `FastHashMap` for O(1) UUID→Key lookups.
    ///
    /// # Examples
    ///
    /// Successfully finding a simplex key from a UUID:
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// // Create a triangulation with some vertices
    /// let vertices = [
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    /// ];
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let tds = dt.tds();
    ///
    /// // Get the first simplex and its UUID
    /// let Some((simplex_key, simplex)) = tds.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let simplex_uuid = simplex.uuid();
    ///
    /// // Use the helper function to find the simplex key from its UUID
    /// let found_key = tds.simplex_key_from_uuid(&simplex_uuid);
    /// assert_eq!(found_key, Some(simplex_key));
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Returns `None` for non-existent UUID:
    ///
    /// ```
    /// use delaunay::prelude::tds::Tds;
    /// use uuid::Uuid;
    ///
    /// let tds: Tds<(), (), 3> = Tds::empty();
    /// let random_uuid = Uuid::new_v4();
    ///
    /// let result = tds.simplex_key_from_uuid(&random_uuid);
    /// assert_eq!(result, None);
    /// ```
    #[inline]
    #[must_use]
    pub fn simplex_key_from_uuid(&self, simplex_uuid: &Uuid) -> Option<SimplexKey> {
        self.uuid_to_simplex_key.get(simplex_uuid).copied()
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
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// // Create a triangulation with some vertices
    /// let vertices = [
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    /// ];
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let tds = dt.tds();
    ///
    /// // Get the first vertex and its UUID
    /// let Some((vertex_key, vertex)) = tds.vertices().next() else {
    ///     return Ok(());
    /// };
    /// let vertex_uuid = vertex.uuid();
    ///
    /// // Use the helper function to find the vertex key from its UUID
    /// let found_key = tds.vertex_key_from_uuid(&vertex_uuid);
    /// assert_eq!(found_key, Some(vertex_key));
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Returns `None` for non-existent UUID:
    ///
    /// ```
    /// use delaunay::prelude::tds::Tds;
    /// use uuid::Uuid;
    ///
    /// let tds: Tds<(), (), 3> = Tds::empty();
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

    /// Helper function to get a simplex UUID from a simplex key using direct `storage map` access.
    /// This is the reverse of `simplex_key_from_uuid()` for the less common Key→UUID direction.
    ///
    /// # Arguments
    ///
    /// * `simplex_key` - The key of the simplex to look up
    ///
    /// # Returns
    ///
    /// An `Option<Uuid>` if the simplex is found, `None` otherwise.
    ///
    /// # Performance
    ///
    /// This uses direct `storage map` indexing for O(1) Key→UUID lookups.
    ///
    /// # Examples
    ///
    /// Successfully getting a UUID from a simplex key:
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// // Create a triangulation with some vertices
    /// let vertices = [
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    /// ];
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let tds = dt.tds();
    ///
    /// // Get the first simplex key and expected UUID
    /// let Some((simplex_key, simplex)) = tds.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let expected_uuid = simplex.uuid();
    ///
    /// // Use the helper function to get UUID from the simplex key
    /// let found_uuid = tds.simplex_uuid_from_key(simplex_key);
    /// assert_eq!(found_uuid, Some(expected_uuid));
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Round-trip conversion between UUID and key:
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// // Create a triangulation with some vertices
    /// let vertices = [
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    /// ];
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let tds = dt.tds();
    ///
    /// // Get the first simplex's UUID
    /// let Some((_, simplex)) = tds.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let original_uuid = simplex.uuid();
    ///
    /// // Convert UUID to key, then key back to UUID
    /// let Some(simplex_key) = tds.simplex_key_from_uuid(&original_uuid) else {
    ///     return Ok(());
    /// };
    /// let round_trip_uuid = tds.simplex_uuid_from_key(simplex_key);
    /// assert_eq!(Some(original_uuid), round_trip_uuid);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn simplex_uuid_from_key(&self, simplex_key: SimplexKey) -> Option<Uuid> {
        self.simplices
            .get(simplex_key)
            .map(super::simplex::Simplex::uuid)
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
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// // Create a triangulation with some vertices
    /// let vertices = [
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    /// ];
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let tds = dt.tds();
    ///
    /// // Get the first vertex key and expected UUID
    /// let Some((vertex_key, vertex)) = tds.vertices().next() else {
    ///     return Ok(());
    /// };
    /// let expected_uuid = vertex.uuid();
    ///
    /// // Use the helper function to get UUID from the vertex key
    /// let found_uuid = tds.vertex_uuid_from_key(vertex_key);
    /// assert_eq!(found_uuid, Some(expected_uuid));
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Round-trip conversion between UUID and key:
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// // Create a triangulation with some vertices
    /// let vertices = [
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    /// ];
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let tds = dt.tds();
    ///
    /// // Get the first vertex's UUID
    /// let Some((_, vertex)) = tds.vertices().next() else {
    ///     return Ok(());
    /// };
    /// let original_uuid = vertex.uuid();
    ///
    /// // Convert UUID to key, then key back to UUID
    /// let Some(vertex_key) = tds.vertex_key_from_uuid(&original_uuid) else {
    ///     return Ok(());
    /// };
    /// let round_trip_uuid = tds.vertex_uuid_from_key(vertex_key);
    /// assert_eq!(Some(original_uuid), round_trip_uuid);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn vertex_uuid_from_key(&self, vertex_key: VertexKey) -> Option<Uuid> {
        self.vertices
            .get(vertex_key)
            .map(super::vertex::Vertex::uuid)
    }

    // =========================================================================
    // KEY-BASED ACCESS METHODS
    // =========================================================================
    // These methods work directly with keys to avoid UUID lookups in hot paths.
    // They complement the existing UUID-based methods for internal algorithm use.

    /// Gets a mutable reference to a simplex directly by its key.
    ///
    /// This method provides direct mutable access to simplices, similar to [`vertex_mut()`](Self::vertex_mut).
    /// While this allows modifying simplex data fields, callers should use safe topology setter APIs
    /// like [`set_neighbors_by_key()`](Self::set_neighbors_by_key) when modifying neighbor relationships.
    ///
    /// # Arguments
    ///
    /// * `simplex_key` - The key of the simplex to retrieve
    ///
    /// # Returns
    ///
    /// An `Option` containing a mutable reference to the simplex if it exists.
    ///
    #[inline]
    #[must_use]
    pub(crate) fn simplex_mut(&mut self, simplex_key: SimplexKey) -> Option<&mut Simplex<V, D>> {
        self.simplices.get_mut(simplex_key)
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
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let tds = dt.tds();
    /// let Some(vertex_key) = tds.vertex_keys().next() else {
    ///     return Ok(());
    /// };
    /// assert!(tds.vertex(vertex_key).is_some());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn vertex(&self, vertex_key: VertexKey) -> Option<&Vertex<U, D>> {
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
    #[inline]
    #[must_use]
    pub(crate) fn vertex_mut(&mut self, vertex_key: VertexKey) -> Option<&mut Vertex<U, D>> {
        self.vertices.get_mut(vertex_key)
    }

    /// Sets the auxiliary data on a returning the previous value.
    ///
    /// This is a safe O(1) operation that modifies only the user-data field.
    /// It does not affect geometry, topology, or Delaunay invariants.
    ///
    /// # Arguments
    ///
    /// * `key` - The key of the vertex to modify
    /// * `data` - The new data value to set, or `None` to clear
    ///
    /// # Returns
    ///
    /// `None` if the key is not found. `Some(previous)` where `previous` is
    /// the old `Option<U>` value if the key exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices: [Vertex<i32, 2>; 3] = [
    ///     delaunay::prelude::Vertex::<_, _>::try_new_with_data([0.0, 0.0], 10i32)?,
    ///     delaunay::prelude::Vertex::<_, _>::try_new_with_data([1.0, 0.0], 20)?,
    ///     delaunay::prelude::Vertex::<_, _>::try_new_with_data([0.0, 1.0], 30)?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let mut tds = dt.tds().clone();
    /// let Some(key) = tds.vertex_keys().next() else {
    ///     return Ok(());
    /// };
    ///
    /// // Replace existing data
    /// let prev = tds.set_vertex_data(key, Some(99));
    /// assert!(prev.is_some()); // key was found
    ///
    /// // Verify new value
    /// let Some(vertex) = tds.vertex(key) else {
    ///     return Ok(());
    /// };
    /// assert_eq!(vertex.data(), Some(&99));
    ///
    /// // Clear data
    /// let prev = tds.set_vertex_data(key, None);
    /// assert_eq!(prev, Some(Some(99)));
    /// assert_eq!(tds.vertex(key).and_then(|vertex| vertex.data()), None);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn set_vertex_data(&mut self, key: VertexKey, data: Option<U>) -> Option<Option<U>> {
        let vertex = self.vertices.get_mut(key)?;
        let previous = vertex.data.take();
        vertex.data = data;
        Some(previous)
    }

    /// Sets the auxiliary data on a simplex, returning the previous value.
    ///
    /// This is a safe O(1) operation that modifies only the user-data field.
    /// It does not affect geometry, topology, or Delaunay invariants.
    ///
    /// # Arguments
    ///
    /// * `key` - The key of the simplex to modify
    /// * `data` - The new data value to set, or `None` to clear
    ///
    /// # Returns
    ///
    /// `None` if the key is not found. `Some(previous)` where `previous` is
    /// the old `Option<V>` value if the key exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<i32>()?;
    /// let mut tds = dt.tds().clone();
    /// let Some(key) = tds.simplex_keys().next() else {
    ///     return Ok(());
    /// };
    ///
    /// // Set data on a simplex that had no data
    /// let prev = tds.set_simplex_data(key, Some(42));
    /// assert_eq!(prev, Some(None)); // key found, previous was None
    ///
    /// // Verify new value
    /// let Some(simplex) = tds.simplex(key) else {
    ///     return Ok(());
    /// };
    /// assert_eq!(simplex.data(), Some(&42));
    ///
    /// // Clear data
    /// let prev = tds.set_simplex_data(key, None);
    /// assert_eq!(prev, Some(Some(42)));
    /// assert_eq!(tds.simplex(key).and_then(|simplex| simplex.data()), None);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn set_simplex_data(&mut self, key: SimplexKey, data: Option<V>) -> Option<Option<V>> {
        let simplex = self.simplices.get_mut(key)?;
        let previous = simplex.data.take();
        simplex.data = data;
        Some(previous)
    }

    /// Checks if a simplex key exists in the triangulation.
    ///
    /// # Arguments
    ///
    /// * `simplex_key` - The key to check
    ///
    /// # Returns
    ///
    /// `true` if the simplex exists, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let tds = dt.tds();
    /// let Some(simplex_key) = tds.simplex_keys().next() else {
    ///     return Ok(());
    /// };
    /// assert!(tds.contains_simplex_key(simplex_key));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn contains_simplex_key(&self, simplex_key: SimplexKey) -> bool {
        self.simplices.contains_key(simplex_key)
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
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let tds = dt.tds();
    /// let Some(vertex_key) = tds.vertex_keys().next() else {
    ///     return Ok(());
    /// };
    /// assert!(tds.contains_vertex_key(vertex_key));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn contains_vertex_key(&self, vertex_key: VertexKey) -> bool {
        self.vertices.contains_key(vertex_key)
    }
}

impl<U, V, const D: usize> Tds<U, V, D> {
    /// Removes multiple simplices by their keys in a batch operation.
    ///
    /// This method performs a **local** topology update:
    /// - Removes the requested simplices (and their UUID→Key mappings)
    /// - Clears neighbor back-references in adjacent surviving simplices so no neighbor points at a removed key
    /// - Repairs `Vertex::incident_simplex` for vertices that previously pointed at a removed simplex
    ///
    /// It does **not** attempt to retriangulate the cavity created by the removals.
    ///
    /// # Performance
    ///
    /// When neighbor pointers are present and mutually consistent, this touches only the
    /// boundary of the removed region:
    /// - Time: typically `O(#removed_simplices × (D+1)^2)`
    /// - Space: `O(#removed_simplices × (D+1))` for temporary removal metadata
    ///
    /// In degraded states (e.g., after unsafe mutation where neighbor pointers are missing),
    /// it may fall back to a conservative scan to find replacement incident simplices.
    ///
    /// # Arguments
    ///
    /// * `simplex_keys` - The keys of simplices to remove
    ///
    /// # Returns
    ///
    /// The number of simplices successfully removed.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let mut tds = dt.tds().clone();
    /// let Some(simplex_key) = tds.simplex_keys().next() else {
    ///     return Ok(());
    /// };
    /// let removed = tds.remove_simplices_by_keys(&[simplex_key]);
    /// assert_eq!(removed, 1);
    /// assert_eq!(tds.number_of_simplices(), 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn remove_simplices_by_keys(&mut self, simplex_keys: &[SimplexKey]) -> usize {
        if simplex_keys.is_empty() {
            return 0;
        }

        // Build a set for O(1) membership tests.
        let simplices_to_remove: SimplexKeySet = simplex_keys.iter().copied().collect();

        // 1) Clear neighbor back-references in surviving simplices and collect candidate incidence.
        let (affected_vertices, candidate_incident) = self
            .collect_removal_frontier_and_clear_neighbor_back_references(
                simplex_keys,
                &simplices_to_remove,
            );

        // 2) Remove the simplices and update UUID mappings.
        let removed_count = self.remove_simplices_and_update_uuid_mappings(simplex_keys);
        if removed_count == 0 {
            return 0;
        }

        // 3) Repair `incident_simplex` pointers for vertices that referenced removed simplices.
        self.repair_incident_simplices_after_simplex_removal(
            &affected_vertices,
            &simplices_to_remove,
            &candidate_incident,
        );

        // Bump generation once for all removals (neighbors + incidence + simplex storage).
        self.bump_generation();

        removed_count
    }

    fn collect_removal_frontier_and_clear_neighbor_back_references(
        &mut self,
        simplex_keys: &[SimplexKey],
        simplices_to_remove: &SimplexKeySet,
    ) -> (VertexKeySet, FastHashMap<VertexKey, SimplexKey>) {
        let mut affected_vertices: VertexKeySet = VertexKeySet::default();
        let mut candidate_incident: FastHashMap<VertexKey, SimplexKey> =
            fast_hash_map_with_capacity(simplex_keys.len().saturating_mul(D.saturating_add(1)));

        for &simplex_key in simplex_keys {
            let Some((vertices, neighbors)) = self.simplices.get(simplex_key).map(|simplex| {
                let mut vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
                    SmallBuffer::with_capacity(simplex.number_of_vertices());
                vertices.extend(simplex.vertices().iter().copied());

                let mut neighbors: SmallBuffer<Option<SimplexKey>, MAX_PRACTICAL_DIMENSION_SIZE> =
                    SmallBuffer::with_capacity(vertices.len());
                neighbors.resize(vertices.len(), None);

                if let Some(simplex_neighbors) = simplex.neighbor_keys() {
                    for (slot, neighbor_opt) in neighbors.iter_mut().zip(simplex_neighbors) {
                        *slot = neighbor_opt;
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

                if simplices_to_remove.contains(neighbor_key) {
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

                let Some(neighbor_simplex) = self.simplices.get_mut(*neighbor_key) else {
                    continue;
                };
                let Some(neighbors_buf) = neighbor_simplex.neighbor_slots_mut() else {
                    continue;
                };

                // Clear the back-reference in the neighbor simplex's neighbor buffer.
                for slot in neighbors_buf.iter_mut() {
                    if slot.simplex_key() == Some(simplex_key) {
                        *slot = NeighborSlot::Boundary;
                    }
                }
            }
        }

        (affected_vertices, candidate_incident)
    }

    fn remove_simplices_and_update_uuid_mappings(&mut self, simplex_keys: &[SimplexKey]) -> usize {
        let mut removed_count = 0;

        for &simplex_key in simplex_keys {
            if let Some(removed_simplex) = self.simplices.remove(simplex_key) {
                self.uuid_to_simplex_key.remove(&removed_simplex.uuid());
                removed_count += 1;
            }
        }

        removed_count
    }

    fn repair_incident_simplices_after_simplex_removal(
        &mut self,
        affected_vertices: &VertexKeySet,
        simplices_to_remove: &SimplexKeySet,
        candidate_incident: &FastHashMap<VertexKey, SimplexKey>,
    ) {
        // We only need to consider vertices that appeared in removed simplices.
        let vertices_to_repair: Vec<VertexKey> = affected_vertices
            .iter()
            .copied()
            .filter(|&vk| {
                let Some(v) = self.vertices.get(vk) else {
                    return false;
                };

                match v.incident_simplex() {
                    None => true,
                    Some(simplex_key) => {
                        simplices_to_remove.contains(&simplex_key)
                            || !self.simplices.contains_key(simplex_key)
                    }
                }
            })
            .collect();

        let mut incident_updates: Vec<(VertexKey, Option<SimplexKey>)> =
            Vec::with_capacity(vertices_to_repair.len());

        for vk in vertices_to_repair {
            // Prefer a candidate simplex discovered on the boundary of the removed region.
            let mut new_incident = candidate_incident.get(&vk).copied().filter(|&ck| {
                self.simplices
                    .get(ck)
                    .is_some_and(|simplex| simplex.contains_vertex(vk))
            });

            // Conservative fallback: pick the first remaining simplex that contains this vertex.
            // This is only hit if neighbor pointers were missing or the boundary had no surviving simplex.
            if new_incident.is_none() {
                new_incident = self.simplices.iter().find_map(|(simplex_key, simplex)| {
                    simplex.contains_vertex(vk).then_some(simplex_key)
                });
            }

            incident_updates.push((vk, new_incident));
        }

        for (vk, new_incident) in incident_updates {
            if let Some(vertex) = self.vertices.get_mut(vk) {
                vertex.set_incident_simplex(new_incident);
            }
        }
    }

    /// Finds all simplices containing a specific vertex.
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
    /// A `SimplexRemovalBuffer` containing the keys of all simplices that contain the vertex.
    /// Uses stack-backed `SmallBuffer` for typical vertex stars (≤16 incident simplices).
    ///
    /// # Performance
    ///
    /// Fast path (typical, valid triangulations):
    /// - Uses the vertex's `incident_simplex` pointer as a seed and walks the neighbor graph across
    ///   facets that still contain the vertex.
    /// - Time: O(|star(v)| × (D+1))
    /// - Space: O(|star(v)|)
    ///
    /// Conservative fallback:
    /// - If `incident_simplex` is missing/stale or neighbor pointers are unavailable, falls back to
    ///   scanning all simplices.
    /// - Time: O(#simplices)
    pub(crate) fn find_simplices_containing_vertex(
        &self,
        vertex_key: VertexKey,
    ) -> SimplexRemovalBuffer {
        let fallback_scan = || {
            self.simplices()
                .filter_map(|(simplex_key, simplex)| {
                    simplex.contains_vertex(vertex_key).then_some(simplex_key)
                })
                .collect()
        };

        // Fast path: walk the star from the vertex's incident simplex using neighbor pointers.
        let Some(vertex) = self.vertices.get(vertex_key) else {
            return fallback_scan();
        };

        let Some(start_simplex_key) = vertex.incident_simplex() else {
            return fallback_scan();
        };

        let Some(start_simplex) = self.simplices.get(start_simplex_key) else {
            return fallback_scan();
        };

        let Some(start_neighbor_slots) = start_simplex.neighbor_slots() else {
            return fallback_scan();
        };
        if !start_simplex.contains_vertex(vertex_key)
            || start_neighbor_slots.iter().any(|slot| slot.is_unassigned())
        {
            return fallback_scan();
        }

        let mut visited: SimplexKeySet = SimplexKeySet::default();
        let mut stack: SimplexRemovalBuffer = SimplexRemovalBuffer::new();
        let mut result: SimplexRemovalBuffer = SimplexRemovalBuffer::new();

        visited.insert(start_simplex_key);
        stack.push(start_simplex_key);

        while let Some(simplex_key) = stack.pop() {
            result.push(simplex_key);

            let Some(simplex) = self.simplices.get(simplex_key) else {
                return fallback_scan();
            };

            let Some(neighbors) = simplex.neighbor_slots() else {
                return fallback_scan();
            };
            if neighbors.iter().any(|slot| slot.is_unassigned()) {
                return fallback_scan();
            }

            // Traverse only across facets that still contain the target vertex.
            for (facet_idx, neighbor_slot) in neighbors.iter().copied().enumerate() {
                if simplex
                    .vertices()
                    .get(facet_idx)
                    .is_some_and(|&vkey| vkey == vertex_key)
                {
                    // This facet excludes `vertex_key`, so crossing it would
                    // leave the vertex star.
                    continue;
                }

                let NeighborSlot::Neighbor(neighbor_key) = neighbor_slot else {
                    continue;
                };

                if visited.contains(&neighbor_key) {
                    continue;
                }

                let Some(neighbor_simplex) = self.simplices.get(neighbor_key) else {
                    return fallback_scan();
                };

                if !neighbor_simplex.contains_vertex(vertex_key) {
                    return fallback_scan();
                }

                visited.insert(neighbor_key);
                stack.push(neighbor_key);
            }
        }

        result
    }

    /// Removes a vertex and all simplices containing it, maintaining data structure consistency.
    ///
    /// This is a composite operation that, on success:
    /// 1. Finds all simplices containing the vertex (using `incident_simplex` + neighbor-walk when available)
    /// 2. Removes all such simplices (clearing neighbor back-references + repairing affected incidence pointers)
    /// 3. Removes the vertex itself
    ///
    /// This operation leaves the triangulation in a valid state (though potentially incomplete).
    /// This is the recommended way to remove a vertex from the triangulation.
    ///
    /// # Arguments
    ///
    /// * `vertex_key` - Key of the vertex to remove
    ///
    /// # Returns
    ///
    /// `Ok(usize)` with the number of simplices that were removed along with the vertex.
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
    /// Typical case (valid triangulations with neighbors/`incident_simplex` assigned):
    /// - Star discovery: O(|star(v)| × (D+1)) via neighbor-walk (no global scan)
    /// - Removal + repair: typically O(|star(v)| × (D+1)²) touching only the boundary of the star
    ///
    /// Conservative fallbacks:
    /// - If `incident_simplex`/neighbor pointers are unavailable (e.g., after unsafe mutation), the
    ///   implementation falls back to a global simplex scan to find the star and/or a replacement
    ///   incident simplex for affected vertices.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 1.0])?,
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// // Get a vertex key to remove
    /// let Some((vertex_key, _)) = dt.vertices().next() else {
    ///     return Ok(());
    /// };
    /// let simplices_before = dt.number_of_simplices();
    ///
    /// // Remove the vertex and all simplices containing it
    /// let simplices_removed = dt.remove_vertex(vertex_key)?;
    /// println!("Removed {} simplices along with the vertex", simplices_removed);
    ///
    /// assert!(dt.tds().validate().is_ok());
    /// # let _ = simplices_before;
    /// # Ok(())
    /// # }
    /// ```
    #[expect(
        clippy::unnecessary_wraps,
        reason = "Keep Result for future mutation validation without changing the API"
    )]
    pub(crate) fn remove_vertex(
        &mut self,
        vertex_key: VertexKey,
    ) -> Result<usize, TdsMutationError> {
        // Look up the vertex to get its UUID before removal
        let Some(vertex) = self.vertex(vertex_key) else {
            return Ok(0); // Vertex not found, nothing to remove
        };
        let uuid = vertex.uuid();

        // Find all simplices containing this vertex
        let simplices_to_remove = self.find_simplices_containing_vertex(vertex_key);

        // Remove all simplices containing the vertex.
        //
        // `remove_simplices_by_keys()` clears neighbor back-references that would otherwise dangle and
        // incrementally repairs `incident_simplex` pointers for vertices that referenced removed simplices.
        let simplices_removed = self.remove_simplices_by_keys(&simplices_to_remove);

        // Remove the vertex itself
        self.vertices.remove(vertex_key);
        self.uuid_to_vertex_key.remove(&uuid);
        // Topology changed; invalidate caches
        self.bump_generation();

        Ok(simplices_removed)
    }

    /// Remove an isolated vertex (one with no incident simplices) from the TDS.
    ///
    /// This removes only the vertex and its UUID mapping. It does **not** touch
    /// any simplices. If the vertex has an incident simplex, this is a no-op.
    pub(crate) fn remove_isolated_vertex(&mut self, vertex_key: VertexKey) {
        let Some(vertex) = self.vertex(vertex_key) else {
            return;
        };
        // Only remove if truly isolated.
        if vertex.incident_simplex().is_some() {
            return;
        }
        let uuid = vertex.uuid();
        self.vertices.remove(vertex_key);
        self.uuid_to_vertex_key.remove(&uuid);
        self.bump_generation();
    }

    // =========================================================================
    // KEY-BASED NEIGHBOR OPERATIONS
    // =========================================================================

    /// Finds neighbor simplex keys for a given simplex without UUID lookups.
    ///
    /// This is the key-based version of neighbor retrieval that avoids
    /// UUID→Key conversions in the hot path.
    ///
    /// # Arguments
    ///
    /// * `simplex_key` - The key of the simplex whose neighbors to find
    ///
    /// # Returns
    ///
    /// A buffer of `Option<SimplexKey>` where `None` indicates no neighbor
    /// at that position (boundary facet). Uses stack allocation for typical dimensions.
    ///
    /// **Special case**: If the simplex does not exist (invalid `simplex_key`), returns a buffer
    /// filled with `None` values. This is a non-panicking fallback that allows callers to
    /// distinguish "simplex missing" from "no neighbors assigned" by checking simplex existence
    /// separately with `simplex()` if needed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let tds = dt.tds();
    /// let Some((simplex_key, _)) = tds.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// // Get neighbors for existing simplex
    /// let neighbors = tds.find_neighbors_by_key(simplex_key);
    /// assert_eq!(neighbors.len(), 3); // D+1 for 2D
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn find_neighbors_by_key(
        &self,
        simplex_key: SimplexKey,
    ) -> NeighborBuffer<Option<SimplexKey>> {
        let mut neighbors = NeighborBuffer::new();
        neighbors.resize(D + 1, None);

        let Some(simplex) = self.simplex(simplex_key) else {
            return neighbors;
        };

        if let Some(neighbors_from_simplex) = simplex.neighbor_keys() {
            // Use zip to avoid potential OOB if neighbors_from_simplex.len() > D+1 (malformed data)
            for (slot, neighbor_key_opt) in neighbors.iter_mut().zip(neighbors_from_simplex) {
                *slot = neighbor_key_opt;
            }
        }

        neighbors
    }

    /// Validates the topological invariant for neighbor relationships.
    ///
    /// **Critical Invariant**: For a simplex, `neighbors[i]` must be opposite `vertices[i]`,
    /// meaning the two simplices share a facet containing all vertices **except** vertex `i`.
    ///
    /// # Arguments
    ///
    /// * `simplex_key` - The key of the simplex to validate
    /// * `neighbors` - The neighbor keys to validate (must have length D+1)
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the topology is valid
    /// * `Err(TdsError)` with details about which neighbors violate the invariant
    ///
    /// # Use Cases
    ///
    /// - Called by `set_neighbors_by_key()` to enforce correctness
    /// - Can be called by `is_valid()` to check entire triangulation
    /// - Useful during incremental construction to identify simplices needing repair
    ///
    /// # Errors
    ///
    /// Returns `TdsError` if topology validation fails.
    fn validate_neighbor_topology(
        &self,
        simplex_key: SimplexKey,
        neighbors: &[Option<SimplexKey>],
    ) -> Result<(), TdsError> {
        if neighbors.len() != D + 1 {
            return Err(TdsError::InvalidNeighbors {
                reason: NeighborValidationError::LengthMismatch {
                    actual: neighbors.len(),
                    expected: D + 1,
                    context: "neighbor topology validation".to_string(),
                },
            });
        }

        let simplex = self
            .simplices
            .get(simplex_key)
            .ok_or_else(|| TdsError::SimplexNotFound {
                simplex_key,
                context: "validate_neighbor_topology".to_string(),
            })?;

        let simplex_lifted_vertices = Self::lifted_vertex_identities(simplex_key, simplex)?;

        for (i, neighbor_key_opt) in neighbors.iter().enumerate() {
            if let Some(neighbor_key) = neighbor_key_opt {
                // Self-adjacency: a simplex can be its own neighbor on a closed manifold (e.g.
                // a torus). In that case the invariant "neighbor[i] shares the facet opposite
                // vertex[i]" is trivially satisfied by the periodic identification.
                if *neighbor_key == simplex_key {
                    if Self::allows_periodic_self_neighbor(simplex) {
                        continue;
                    }
                    return Err(TdsError::InvalidNeighbors {
                        reason: NeighborValidationError::NonPeriodicSelfNeighbor {
                            simplex_key,
                            simplex_uuid: simplex.uuid(),
                            facet_index: i,
                        },
                    });
                }

                let neighbor = self.simplices.get(*neighbor_key).ok_or_else(|| {
                    TdsError::InvalidNeighbors {
                        reason: NeighborValidationError::MissingNeighborSimplex {
                            simplex_key,
                            simplex_uuid: simplex.uuid(),
                            facet_index: i,
                            neighbor_key: *neighbor_key,
                            context: "neighbor topology validation".to_string(),
                        },
                    }
                })?;

                let uses_periodic_offsets = simplex.periodic_vertex_offsets().is_some()
                    || neighbor.periodic_vertex_offsets().is_some();
                let (shared_count, missing_vertex_idx) = if uses_periodic_offsets {
                    // Periodic quotient simplices may be represented in different translated
                    // frames. Compare normalized lifted facet identities so offset-distinct
                    // vertices remain distinct while globally translated representatives match.
                    let matching_facet_index =
                        Self::matching_lifted_facet_index(simplex, neighbor)?;
                    (matching_facet_index.map_or(0, |_| D), matching_facet_index)
                } else {
                    let neighbor_lifted_vertices =
                        Self::lifted_vertex_identities(*neighbor_key, neighbor)?;

                    // Count shared vertices and find missing vertex
                    let mut shared_count = 0;
                    let mut missing_vertex_idx = None;

                    for (idx, simplex_vertex_identity) in simplex_lifted_vertices.iter().enumerate()
                    {
                        if neighbor_lifted_vertices
                            .iter()
                            .any(|neighbor_vertex_identity| {
                                neighbor_vertex_identity == simplex_vertex_identity
                            })
                        {
                            shared_count += 1;
                        } else if missing_vertex_idx.is_none() {
                            missing_vertex_idx = Some(idx);
                        }
                    }
                    (shared_count, missing_vertex_idx)
                };

                // Validate the topological invariant
                if shared_count != D {
                    return Err(TdsError::InvalidNeighbors {
                        reason: NeighborValidationError::SharedVertexCountMismatch {
                            simplex_key,
                            simplex_uuid: simplex.uuid(),
                            facet_index: i,
                            shared_count,
                            expected: D,
                        },
                    });
                }

                if missing_vertex_idx != Some(i) {
                    return Err(TdsError::InvalidNeighbors {
                        reason: NeighborValidationError::OppositeVertexMismatch {
                            simplex_key,
                            simplex_uuid: simplex.uuid(),
                            facet_index: i,
                            observed_opposite: missing_vertex_idx,
                            expected_opposite: i,
                        },
                    });
                }
            }
        }

        Ok(())
    }

    fn validate_neighbor_update_matches_facet_incidence(
        &self,
        simplex_key: SimplexKey,
        neighbors: &[Option<SimplexKey>],
    ) -> Result<(), TdsError> {
        let simplex = self
            .simplices
            .get(simplex_key)
            .ok_or_else(|| TdsError::SimplexNotFound {
                simplex_key,
                context: "validate_neighbor_update_matches_facet_incidence".to_string(),
            })?;

        let facet_to_simplices = self.build_facet_to_simplices_map()?;
        for (facet_idx, proposed_neighbor) in neighbors.iter().copied().enumerate() {
            let facet_key = self.facet_key_for_simplex_facet(simplex_key, facet_idx)?;
            let Some(simplex_facet_pairs) = facet_to_simplices.get(&facet_key) else {
                return Err(TdsError::InvalidNeighbors {
                    reason: NeighborValidationError::FacetIncidenceMissing {
                        simplex_key,
                        simplex_uuid: simplex.uuid(),
                        facet_index: facet_idx,
                        facet_key,
                    },
                });
            };

            let expected_neighbor = match simplex_facet_pairs.as_slice() {
                [_] => None,
                [a, b] => {
                    if a.simplex_key() == simplex_key && a.facet_index() as usize == facet_idx {
                        Some(b.simplex_key())
                    } else if b.simplex_key() == simplex_key
                        && b.facet_index() as usize == facet_idx
                    {
                        Some(a.simplex_key())
                    } else {
                        return Err(TdsError::InvalidNeighbors {
                            reason:
                                NeighborValidationError::FacetIncidenceDoesNotReferenceSimplex {
                                    simplex_key,
                                    simplex_uuid: simplex.uuid(),
                                    facet_index: facet_idx,
                                    facet_key,
                                },
                        });
                    }
                }
                _ => {
                    return Err(TdsError::InvalidNeighbors {
                        reason: NeighborValidationError::FacetIncidenceMultiplicity {
                            simplex_key,
                            simplex_uuid: simplex.uuid(),
                            facet_index: facet_idx,
                            facet_key,
                            simplex_count: simplex_facet_pairs.len(),
                        },
                    });
                }
            };

            if proposed_neighbor == Some(simplex_key)
                && expected_neighbor.is_none()
                && Self::allows_periodic_self_neighbor(simplex)
            {
                continue;
            }

            if proposed_neighbor != expected_neighbor {
                return Err(TdsError::InvalidNeighbors {
                    reason: NeighborValidationError::NeighborIncidenceMismatch {
                        simplex_key,
                        simplex_uuid: simplex.uuid(),
                        facet_index: facet_idx,
                        proposed_neighbor,
                        expected_neighbor,
                    },
                });
            }
        }

        Ok(())
    }

    fn set_simplex_neighbors_normalized(
        simplex: &mut Simplex<V, D>,
        neighbors: &[Option<SimplexKey>],
    ) -> Result<(), TdsError> {
        let simplex_id = simplex.uuid();
        simplex
            .set_neighbors_from_keys(neighbors.iter().copied())
            .map_err(|source| TdsError::InvalidSimplex { simplex_id, source })
    }

    fn ensure_neighbor_buffer(
        simplex: &mut Simplex<V, D>,
    ) -> Result<&mut SmallBuffer<NeighborSlot, MAX_PRACTICAL_DIMENSION_SIZE>, TdsError> {
        if simplex.neighbor_slots().is_none() {
            let simplex_id = simplex.uuid();
            simplex
                .set_neighbors_from_keys((0..=D).map(|_| None))
                .map_err(|source| TdsError::InvalidSimplex { simplex_id, source })?;
        }
        let simplex_id = simplex.uuid();
        simplex
            .try_ensure_neighbors_buffer_mut()
            .map_err(|source| TdsError::InvalidSimplex { simplex_id, source })
    }

    fn set_neighbor_slot(
        simplex: &mut Simplex<V, D>,
        facet_idx: usize,
        neighbor: Option<SimplexKey>,
    ) -> Result<(), TdsError> {
        let neighbors = Self::ensure_neighbor_buffer(simplex)?;
        let Some(slot) = neighbors.get_mut(facet_idx) else {
            return Err(TdsError::InvalidNeighbors {
                reason: NeighborValidationError::NeighborSlotOutOfBounds {
                    facet_index: facet_idx,
                    slot_count: neighbors.len(),
                },
            });
        };
        *slot = NeighborSlot::from_neighbor_key(neighbor);
        Ok(())
    }

    fn reciprocal_neighbor_updates_for_neighbor_update(
        &self,
        simplex_key: SimplexKey,
        neighbors: &[Option<SimplexKey>],
    ) -> Result<Vec<(SimplexKey, usize, Option<SimplexKey>)>, TdsError> {
        let simplex = self
            .simplices
            .get(simplex_key)
            .ok_or_else(|| TdsError::SimplexNotFound {
                simplex_key,
                context: "set_neighbors_by_key".to_string(),
            })?;
        let old_neighbors: Vec<Option<SimplexKey>> = simplex
            .neighbor_keys()
            .map_or_else(|| vec![None; D + 1], Iterator::collect);

        let mut reciprocal_updates = Vec::new();
        self.collect_stale_reciprocal_neighbor_updates(
            simplex_key,
            simplex,
            &old_neighbors,
            neighbors,
            &mut reciprocal_updates,
        )?;
        self.collect_new_reciprocal_neighbor_updates(
            simplex_key,
            simplex,
            neighbors,
            &mut reciprocal_updates,
        )?;
        Ok(reciprocal_updates)
    }

    fn collect_stale_reciprocal_neighbor_updates(
        &self,
        simplex_key: SimplexKey,
        simplex: &Simplex<V, D>,
        old_neighbors: &[Option<SimplexKey>],
        new_neighbors: &[Option<SimplexKey>],
        reciprocal_updates: &mut Vec<(SimplexKey, usize, Option<SimplexKey>)>,
    ) -> Result<(), TdsError> {
        for (facet_idx, old_neighbor_key) in old_neighbors.iter().copied().enumerate() {
            let Some(old_neighbor_key) = old_neighbor_key else {
                continue;
            };
            if old_neighbor_key == simplex_key
                || new_neighbors
                    .iter()
                    .copied()
                    .any(|neighbor_key| neighbor_key == Some(old_neighbor_key))
            {
                continue;
            }

            let old_neighbor_simplex =
                self.simplices
                    .get(old_neighbor_key)
                    .ok_or_else(|| TdsError::InvalidNeighbors {
                        reason: NeighborValidationError::MissingNeighborSimplex {
                            simplex_key,
                            simplex_uuid: simplex.uuid(),
                            facet_index: facet_idx,
                            neighbor_key: old_neighbor_key,
                            context: "clearing stale reciprocal neighbor".to_string(),
                        },
                    })?;
            let mirror_idx = simplex
                .mirror_facet_index(facet_idx, old_neighbor_simplex)
                .ok_or_else(|| TdsError::InvalidNeighbors {
                    reason: NeighborValidationError::MirrorFacetMissing {
                        simplex_uuid: simplex.uuid(),
                        facet_index: facet_idx,
                        neighbor_uuid: old_neighbor_simplex.uuid(),
                        context: "clearing old back-reference".to_string(),
                    },
                })?;
            let back_ref = old_neighbor_simplex.neighbor_key(mirror_idx).flatten();
            if back_ref == Some(simplex_key) {
                reciprocal_updates.push((old_neighbor_key, mirror_idx, None));
            }
        }

        Ok(())
    }

    fn collect_new_reciprocal_neighbor_updates(
        &self,
        simplex_key: SimplexKey,
        simplex: &Simplex<V, D>,
        neighbors: &[Option<SimplexKey>],
        reciprocal_updates: &mut Vec<(SimplexKey, usize, Option<SimplexKey>)>,
    ) -> Result<(), TdsError> {
        for (facet_idx, neighbor_key) in neighbors.iter().copied().enumerate() {
            let Some(neighbor_key) = neighbor_key else {
                continue;
            };
            if neighbor_key == simplex_key {
                continue;
            }

            let neighbor_simplex =
                self.simplices
                    .get(neighbor_key)
                    .ok_or_else(|| TdsError::InvalidNeighbors {
                        reason: NeighborValidationError::MissingNeighborSimplex {
                            simplex_key,
                            simplex_uuid: simplex.uuid(),
                            facet_index: facet_idx,
                            neighbor_key,
                            context: "setting reciprocal neighbor".to_string(),
                        },
                    })?;
            let mirror_idx = simplex
                .mirror_facet_index(facet_idx, neighbor_simplex)
                .ok_or_else(|| TdsError::InvalidNeighbors {
                    reason: NeighborValidationError::MirrorFacetMissing {
                        simplex_uuid: simplex.uuid(),
                        facet_index: facet_idx,
                        neighbor_uuid: neighbor_simplex.uuid(),
                        context: "setting back-reference".to_string(),
                    },
                })?;
            let existing_back_ref = neighbor_simplex.neighbor_key(mirror_idx).flatten();
            if let Some(existing_back_ref) = existing_back_ref
                && existing_back_ref != simplex_key
            {
                return Err(TdsError::InvalidNeighbors {
                    reason: NeighborValidationError::ExistingBackReferenceConflict {
                        neighbor_uuid: neighbor_simplex.uuid(),
                        mirror_index: mirror_idx,
                        existing_back_ref,
                        requested_back_ref: simplex_key,
                    },
                });
            }
            reciprocal_updates.push((neighbor_key, mirror_idx, Some(simplex_key)));
        }

        Ok(())
    }

    /// Sets neighbor relationships using simplex keys directly.
    ///
    /// # Positional Semantics (Critical Topological Invariant)
    ///
    /// **`neighbors[i]` must be the neighbor opposite to `vertices[i]`**
    ///
    /// This means the two simplices share facet `i`, which contains all vertices **except** vertex `i`.
    ///
    /// ## Example: 3D Tetrahedron
    ///
    /// For a simplex with vertices `[v0, v1, v2, v3]`:
    /// - `neighbors[0]` shares facet `[v1, v2, v3]` (opposite v0)
    /// - `neighbors[1]` shares facet `[v0, v2, v3]` (opposite v1)
    /// - `neighbors[2]` shares facet `[v0, v1, v3]` (opposite v2)
    /// - `neighbors[3]` shares facet `[v0, v1, v2]` (opposite v3)
    ///
    /// **This invariant is always validated** via `validate_neighbor_topology()`.
    ///
    /// # Arguments
    ///
    /// * `simplex_key` - The key of the simplex to update
    /// * `neighbors` - The new neighbor keys (must have length D+1)
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful, or an error if validation fails.
    ///
    /// # Errors
    ///
    /// Returns a `TdsMutationError` if:
    /// - The simplex with the given key doesn't exist
    /// - The neighbor vector length is not D+1
    /// - Any neighbor key references a non-existent simplex
    /// - **The topological invariant is violated** (neighbor\[i\] not opposite vertex\[i\])
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsMutation(#[from] delaunay::prelude::tds::TdsMutationError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let mut tds = dt.tds().clone();
    /// let Some(simplex_key) = tds.simplex_keys().next() else {
    ///     return Ok(());
    /// };
    /// let neighbors = vec![None; 3];
    /// tds.set_neighbors_by_key(simplex_key, &neighbors)?;
    /// let Some(simplex) = tds.simplex(simplex_key) else {
    ///     return Ok(());
    /// };
    /// assert!(simplex
    ///     .neighbors()
    ///     .is_some_and(|mut neighbors| neighbors.all(|neighbor| neighbor.is_none())));
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_neighbors_by_key(
        &mut self,
        simplex_key: SimplexKey,
        neighbors: &[Option<SimplexKey>],
    ) -> Result<(), TdsMutationError> {
        // Validate the topological invariant before applying changes
        // (includes length check: neighbors.len() == D+1)
        self.validate_neighbor_topology(simplex_key, neighbors)?;
        self.validate_neighbor_update_matches_facet_incidence(simplex_key, neighbors)?;
        let reciprocal_updates =
            self.reciprocal_neighbor_updates_for_neighbor_update(simplex_key, neighbors)?;

        let simplex_uuid = {
            let simplex =
                self.simplex_mut(simplex_key)
                    .ok_or_else(|| TdsError::SimplexNotFound {
                        simplex_key,
                        context: "set_neighbors_by_key".to_string(),
                    })?;
            let simplex_uuid = simplex.uuid();
            Self::set_simplex_neighbors_normalized(simplex, neighbors)?;
            simplex_uuid
        };

        for (neighbor_key, mirror_idx, back_reference) in reciprocal_updates {
            let neighbor_simplex =
                self.simplices
                    .get_mut(neighbor_key)
                    .ok_or_else(|| TdsError::InvalidNeighbors {
                        reason: NeighborValidationError::MissingNeighborSimplex {
                            simplex_key,
                            simplex_uuid,
                            facet_index: mirror_idx,
                            neighbor_key,
                            context: "applying reciprocal neighbor update".to_string(),
                        },
                    })?;
            Self::set_neighbor_slot(neighbor_simplex, mirror_idx, back_reference)?;
        }

        // Topology changed; invalidate caches
        self.bump_generation();
        Ok(())
    }

    /// Finds simplices containing a vertex using its key directly.
    ///
    /// This method avoids UUID lookups when searching for simplices that
    /// contain a specific vertex.
    ///
    /// # Arguments
    ///
    /// * `vertex_key` - The key of the vertex to search for
    ///
    /// # Returns
    ///
    /// A set of simplex keys that contain the given vertex.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsMutation(#[from] delaunay::prelude::tds::TdsMutationError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let tds = dt.tds();
    /// let Some(vertex_key) = tds.vertex_keys().next() else {
    ///     return Ok(());
    /// };
    /// let simplices = tds.find_simplices_containing_vertex_by_key(vertex_key);
    /// assert_eq!(simplices.len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn find_simplices_containing_vertex_by_key(&self, vertex_key: VertexKey) -> SimplexKeySet {
        if self.vertex(vertex_key).is_none() {
            return SimplexKeySet::default();
        }

        let simplices = self.find_simplices_containing_vertex(vertex_key);
        simplices.iter().copied().collect()
    }

    /// Assigns incident simplices to vertices in the triangulation.
    ///
    /// This method establishes a mapping from each vertex to one of the simplices that contains it,
    /// which is useful for various geometric queries and traversals. For each an arbitrary
    /// incident simplex is selected from the simplices that contain that vertex.
    ///
    /// Note: Many topology-mutating operations (like [`Tds::remove_simplices_by_keys`](Self::remove_simplices_by_keys))
    /// attempt to repair `incident_simplex` incrementally for affected vertices. This method exists as a
    /// conservative **full rebuild** after bulk changes (deserialization, large repairs, etc.).
    ///
    /// # Returns
    ///
    /// `Ok(())` if incident simplices were successfully assigned to all vertices,
    /// otherwise a `TdsMutationError`.
    ///
    /// # Errors
    ///
    /// Returns a `TdsMutationError` if a simplex references a non-existent vertex key
    /// (`VertexNotFound`).
    ///
    /// # Algorithm
    ///
    /// 1. Clear `incident_simplex` for all vertices
    /// 2. Scan all simplices; for each vertex encountered, assign the current simplex as its incident simplex
    ///    if it does not already have one
    ///
    /// # Performance
    ///
    /// This method rebuilds incidence **globally** by scanning all simplices:
    /// - Time: O(#simplices × (D+1))
    /// - Space: O(1) extra (no temporary vertex→simplices map)
    ///
    /// It is intended for repair/validation paths after bulk topology changes, not as a per-step
    /// hot-path update.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsMutation(#[from] delaunay::prelude::tds::TdsMutationError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let mut tds = dt.tds().clone();
    /// tds.assign_incident_simplices()?;
    /// let all_assigned = tds.vertices().all(|(_, v)| v.incident_simplex().is_some());
    /// assert!(all_assigned);
    /// # Ok(())
    /// # }
    /// ```
    pub fn assign_incident_simplices(&mut self) -> Result<(), TdsMutationError> {
        if self.simplices.is_empty() {
            // No simplices remain; all vertices must have incident_simplex cleared to avoid
            // dangling pointers to previously removed simplices.
            for vertex in self.vertices.values_mut() {
                vertex.set_incident_simplex(None);
            }
            self.bump_generation();
            return Ok(());
        }

        // Reset incident_simplex for all vertices before rebuilding the mapping. This
        // ensures vertices that no longer belong to any simplex do not retain stale
        // incident_simplex pointers after topology changes (e.g. or simplex removal).
        for vertex in self.vertices.values_mut() {
            vertex.set_incident_simplex(None);
        }

        // Single-pass rebuild: assign the first simplex encountered for each vertex.
        for (simplex_key, simplex) in &self.simplices {
            for &vertex_key in simplex.vertices() {
                let Some(vertex) = self.vertices.get_mut(vertex_key) else {
                    // State has already been mutated (incident simplices cleared above),
                    // so bump generation before returning the error.
                    self.bump_generation();
                    return Err(TdsError::VertexNotFound {
                        vertex_key,
                        context: "incident simplex assignment".to_string(),
                    }
                    .into());
                };

                if vertex.incident_simplex().is_none() {
                    vertex.set_incident_simplex(Some(simplex_key));
                }
            }
        }

        self.bump_generation();
        Ok(())
    }

    /// Creates a new empty triangulation data structure.
    ///
    ///
    /// This function creates an empty triangulation with no vertices and no simplices.
    /// Use [`DelaunayTriangulation::empty()`](crate::DelaunayTriangulation::empty)
    /// for the high-level API, or this method for low-level Tds construction.
    ///
    /// # Returns
    ///
    /// An empty triangulation data structure with:
    /// - No vertices
    /// - No simplices
    /// - Construction state set to `Incomplete(0)`
    /// - Dimension of -1 (empty)
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::tds::Tds;
    /// use delaunay::prelude::tds::TriangulationConstructionState;
    ///
    /// let tds: Tds<(), (), 3> = Tds::empty();
    /// assert_eq!(tds.number_of_vertices(), 0);
    /// assert_eq!(tds.number_of_simplices(), 0);
    /// assert_eq!(tds.dim(), -1);
    /// std::assert_matches!(
    ///     tds.construction_state(),
    ///     TriangulationConstructionState::Incomplete(0)
    /// );
    /// ```
    #[must_use]
    pub fn empty() -> Self {
        Self {
            vertices: StorageMap::with_key(),
            simplices: StorageMap::with_key(),
            uuid_to_vertex_key: UuidToVertexKeyMap::default(),
            uuid_to_simplex_key: UuidToSimplexKeyMap::default(),
            construction_state: TriangulationConstructionState::Incomplete(0),
            generation: Arc::new(AtomicU64::new(0)),
            identity: Arc::new(Uuid::new_v4()),
        }
    }

    /// Clears all neighbor relationships between simplices in the triangulation.
    ///
    /// This method removes all neighbor relationships by setting the `neighbors` field
    /// to `None` for every simplex in the triangulation. This is useful for:
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
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    /// ];
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// // Initially has neighbors assigned during construction
    /// // All simplices have neighbors
    /// for (_, simplex) in dt.simplices() {
    ///     // Check that simplices have properly assigned neighbors
    ///     println!("Simplex has neighbors: {:?}", simplex.neighbors().is_some());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn clear_all_neighbors(&mut self) {
        for simplex in self.simplices.values_mut() {
            simplex.clear_neighbors();
        }
        // Topology changed; invalidate caches.
        self.bump_generation();
    }

    /// Normalizes simplex vertex ordering so all adjacent simplices satisfy coherent orientation.
    ///
    /// This computes a per-simplex flip assignment over each connected component of the
    /// simplex-neighbor graph, then applies slot swaps (`0 <-> 1`) to simplices that must flip.
    ///
    /// # Errors
    ///
    /// Returns [`TdsError`] if the traversal encounters structural problems:
    /// - [`SimplexNotFound`](TdsError::SimplexNotFound) — a simplex key referenced during BFS is missing from storage.
    /// - [`InvalidNeighbors`](TdsError::InvalidNeighbors) — a mirror facet cannot be derived between adjacent simplices.
    /// - [`InconsistentDataStructure`](TdsError::InconsistentDataStructure) — orientation constraints are contradictory
    ///   or a flip-assignment entry is unexpectedly absent.
    #[expect(
        clippy::too_many_lines,
        reason = "orientation normalization is unchanged; simplex nomenclature makes existing names longer"
    )]
    pub(crate) fn normalize_coherent_orientation(&mut self) -> Result<(), TdsError> {
        let mut flip_assignment: FastHashMap<SimplexKey, bool> =
            fast_hash_map_with_capacity(self.simplices.len());

        for root_simplex_key in self.simplices.keys() {
            if flip_assignment.contains_key(&root_simplex_key) {
                continue;
            }

            flip_assignment.insert(root_simplex_key, false);
            let mut queue = VecDeque::new();
            queue.push_back(root_simplex_key);

            while let Some(simplex_key) = queue.pop_front() {
                let this_flip_state = *flip_assignment.get(&simplex_key).ok_or_else(|| {
                    TdsError::InconsistentDataStructure {
                        message: format!(
                            "Missing flip assignment for simplex {simplex_key:?} during orientation normalization",
                        ),
                    }
                })?;

                let simplex =
                    self.simplices
                        .get(simplex_key)
                        .ok_or_else(|| TdsError::SimplexNotFound {
                            simplex_key,
                            context: "orientation normalization traversal".to_string(),
                        })?;
                let Some(neighbors) = simplex.neighbor_keys() else {
                    continue;
                };

                for (facet_idx, neighbor_key_opt) in neighbors.enumerate() {
                    let Some(neighbor_key) = neighbor_key_opt else {
                        continue;
                    };
                    if neighbor_key == simplex_key && Self::allows_periodic_self_neighbor(simplex) {
                        continue;
                    }

                    let neighbor_simplex =
                        self.simplices
                            .get(neighbor_key)
                            .ok_or_else(|| TdsError::SimplexNotFound {
                                simplex_key: neighbor_key,
                                context: format!(
                                    "neighbor of simplex {simplex_key:?} during orientation normalization"
                                ),
                            })?;
                    // Periodic-lifted adjacencies do not have a unique canonical orientation at this
                    // structural layer because the embedding depends on lattice representative choice.
                    // Skip normalization constraints for these pairs.
                    if simplex.periodic_vertex_offsets().is_some()
                        || neighbor_simplex.periodic_vertex_offsets().is_some()
                    {
                        continue;
                    }
                    let mirror_idx = simplex
                        .mirror_facet_index(facet_idx, neighbor_simplex)
                        .ok_or_else(|| TdsError::InvalidNeighbors {
                            reason: NeighborValidationError::MirrorFacetMissing {
                                simplex_uuid: simplex.uuid(),
                                facet_index: facet_idx,
                                neighbor_uuid: neighbor_simplex.uuid(),
                                context: "orientation normalization".to_string(),
                            },
                        })?;

                    let (currently_coherent, _, _) = Self::facet_permutation_parity(
                        simplex,
                        facet_idx,
                        neighbor_simplex,
                        mirror_idx,
                    )?;

                    // Flipping exactly one endpoint toggles the coherence state for this edge.
                    let requires_relative_flip = !currently_coherent;
                    let required_neighbor_flip_state = this_flip_state ^ requires_relative_flip;

                    if let Some(existing_neighbor_flip_state) = flip_assignment.get(&neighbor_key) {
                        if *existing_neighbor_flip_state != required_neighbor_flip_state {
                            return Err(TdsError::InconsistentDataStructure {
                                message: format!(
                                    "Contradictory orientation constraints while normalizing simplices {:?} and {:?}",
                                    simplex.uuid(),
                                    neighbor_simplex.uuid(),
                                ),
                            });
                        }
                    } else {
                        flip_assignment.insert(neighbor_key, required_neighbor_flip_state);
                        queue.push_back(neighbor_key);
                    }
                }
            }
        }

        let mut flipped_any = false;
        for (simplex_key, should_flip) in flip_assignment {
            if !should_flip {
                continue;
            }
            let simplex =
                self.simplices
                    .get_mut(simplex_key)
                    .ok_or_else(|| TdsError::SimplexNotFound {
                        simplex_key,
                        context: "applying orientation normalization".to_string(),
                    })?;
            if simplex.number_of_vertices() >= 2 {
                simplex.swap_vertex_slots(0, 1);
                flipped_any = true;
            }
        }
        if flipped_any {
            self.bump_generation();
        }

        Ok(())
    }

    /// Validates coherent orientation for simplices touched by a local mutation.
    ///
    /// This checks every adjacency owned by `simplices`, including adjacencies to
    /// simplices outside the supplied slice. It is intended for insertion and local
    /// repair paths that already know the mutation frontier and want Level-2
    /// orientation safety without a full-TDS traversal.
    pub(crate) fn validate_coherent_orientation_for_simplices(
        &self,
        simplices: &[SimplexKey],
    ) -> Result<(), TdsError> {
        for &simplex_key in simplices {
            let simplex =
                self.simplices
                    .get(simplex_key)
                    .ok_or_else(|| TdsError::SimplexNotFound {
                        simplex_key,
                        context: "local orientation validation scope".to_string(),
                    })?;
            let Some(neighbors) = simplex.neighbor_keys() else {
                continue;
            };

            for (facet_idx, neighbor_key_opt) in neighbors.enumerate() {
                let Some(neighbor_key) = neighbor_key_opt else {
                    continue;
                };
                if neighbor_key == simplex_key && Self::allows_periodic_self_neighbor(simplex) {
                    continue;
                }

                let neighbor_simplex =
                    self.simplices
                        .get(neighbor_key)
                        .ok_or_else(|| TdsError::SimplexNotFound {
                            simplex_key: neighbor_key,
                            context: format!(
                                "neighbor of simplex {simplex_key:?} during local orientation validation",
                            ),
                        })?;

                let (mirror_idx, uses_periodic_offsets) = Self::orientation_mirror_facet_index(
                    simplex,
                    facet_idx,
                    neighbor_simplex,
                    "local orientation validation",
                )?;
                let observed_back_reference = neighbor_simplex.neighbor_key(mirror_idx).flatten();
                if observed_back_reference != Some(simplex_key) {
                    return Err(TdsError::InvalidNeighbors {
                        reason: NeighborValidationError::BackReferenceMismatch {
                            simplex_key,
                            simplex_uuid: simplex.uuid(),
                            facet_index: facet_idx,
                            neighbor_key,
                            neighbor_uuid: neighbor_simplex.uuid(),
                            mirror_index: mirror_idx,
                            observed: observed_back_reference,
                            context: "local orientation validation".to_string(),
                        },
                    });
                }
                if uses_periodic_offsets {
                    continue;
                }

                let simplex1_facet_vertices =
                    Self::facet_vertices_in_simplex_order(simplex, facet_idx)?;
                let simplex2_facet_vertices =
                    Self::facet_vertices_in_simplex_order(neighbor_simplex, mirror_idx)?;
                let (coherent, observed_odd_permutation, expected_odd_permutation) =
                    Self::facet_permutation_parity(
                        simplex,
                        facet_idx,
                        neighbor_simplex,
                        mirror_idx,
                    )?;
                if !coherent {
                    return Err(TdsError::OrientationViolation {
                        simplex1_key: simplex_key,
                        simplex1_uuid: simplex.uuid(),
                        simplex2_key: neighbor_key,
                        simplex2_uuid: neighbor_simplex.uuid(),
                        simplex1_facet_index: facet_idx,
                        simplex2_facet_index: mirror_idx,
                        facet_vertices: simplex1_facet_vertices.into_iter().collect(),
                        simplex2_facet_vertices: simplex2_facet_vertices.into_iter().collect(),
                        observed_odd_permutation,
                        expected_odd_permutation,
                    });
                }
            }
        }

        Ok(())
    }
}

impl<U, V, const D: usize> Tds<U, V, D> {
    /// Builds a `FacetToSimplicesMap` with strict error handling.
    ///
    /// This method returns an error if any simplex has missing vertex keys, ensuring
    /// complete and accurate facet topology information. This is the preferred method
    /// for building facet-to-simplices mappings.
    ///
    /// # Returns
    ///
    /// A `Result` containing:
    /// - `Ok(FacetToSimplicesMap)`: A complete mapping of facet keys to simplices
    /// - `Err(TdsError)`: If any simplex has missing vertex keys
    ///
    /// # Errors
    ///
    /// Returns [`TdsError`] if the map cannot be built:
    /// - [`VertexNotFound`](TdsError::VertexNotFound) / [`SimplexNotFound`](TdsError::SimplexNotFound) — a simplex cannot resolve its vertex keys.
    /// - [`IndexOutOfBounds`](TdsError::IndexOutOfBounds) — a facet index exceeds the `u8` range.
    /// - [`DimensionMismatch`](TdsError::DimensionMismatch) — periodic offset count does not match vertex count.
    /// - [`InconsistentDataStructure`](TdsError::InconsistentDataStructure) — periodic facet key derivation fails.
    ///
    /// # Performance
    ///
    /// O(N×F) time complexity where N is the number of simplices and F is the
    /// number of facets per simplex (typically D+1 for D-dimensional simplices).
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let tds = dt.tds();
    /// let facet_map = tds.build_facet_to_simplices_map()?;
    /// assert!(!facet_map.is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub fn build_facet_to_simplices_map(&self) -> Result<FacetToSimplicesMap, TdsError> {
        if D > usize::from(u8::MAX) {
            return Err(TdsError::DimensionMismatch {
                expected: usize::from(u8::MAX),
                actual: D,
                context: "facet indices must fit in u8".to_string(),
            });
        }

        let cap = self.simplices.len().saturating_mul(D.saturating_add(1));
        let mut facet_to_simplices: FacetToSimplicesMap = fast_hash_map_with_capacity(cap);

        // Iterate over all simplices and their facets
        for (simplex_id, simplex) in &self.simplices {
            // Use direct key-based method to avoid UUID→Key lookups
            // The error from simplex_vertices is already TdsError
            let vertices = self.simplex_vertices(simplex_id)?;

            for i in 0..vertices.len() {
                let facet_key =
                    Self::periodic_facet_key_from_simplex_vertices(simplex, &vertices, i)?;
                let Ok(facet_index_u8) = usize_to_u8(i, vertices.len()) else {
                    return Err(TdsError::IndexOutOfBounds {
                        index: i,
                        bound: u8::MAX as usize + 1,
                        context: format!("facet index exceeds u8 range for {D}D"),
                    });
                };

                facet_to_simplices
                    .entry(facet_key)
                    .or_default()
                    .push(FacetHandle::from_validated(simplex_id, facet_index_u8));
            }
        }

        Ok(facet_to_simplices)
    }
}

impl<U, V, const D: usize> Tds<U, V, D> {
    /// Removes duplicate simplices with identical vertex sets.
    ///
    /// Returns the number of duplicate simplices that were removed.
    ///
    /// Duplicate removal is applied to a cloned trial [`Tds`], then the
    /// topology (neighbor relationships and incident simplices) is rebuilt to
    /// maintain data structure invariants and prevent stale references. If the
    /// rebuild or validation fails, the original structure is left unchanged.
    ///
    /// When duplicates are present, the rollback guarantee is implemented by
    /// cloning the current [`Tds`] before removal. This keeps failed mutations
    /// atomic, but the snapshot cost is linear in the size of the stored
    /// topology. The method therefore requires the stored coordinates and
    /// payloads to be cloneable so the trial structure can preserve them.
    ///
    /// # Errors
    ///
    /// Returns a [`TdsMutationError`] if:
    /// - Vertex keys cannot be retrieved for any simplex (data structure corruption)
    /// - Neighbor assignment fails after simplex removal
    /// - Incident simplex assignment fails after simplex removal
    /// - Validation fails after topology rebuild
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::tds::{Tds, TdsMutationError};
    ///
    /// # fn main() -> Result<(), TdsMutationError> {
    /// let mut tds: Tds<(), (), 2> = Tds::empty();
    /// let removed = tds.remove_duplicate_simplices()?;
    /// assert_eq!(removed, 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn remove_duplicate_simplices(&mut self) -> Result<usize, TdsMutationError>
    where
        U: Clone,
        V: Clone,
    {
        let mut unique_simplices = FastHashMap::default();
        let mut simplices_to_remove = SimplexRemovalBuffer::new();

        // First pass: identify duplicate simplices
        for simplex_key in self.simplices.keys() {
            let vertices = self.simplex_vertices(simplex_key)?;
            let vertex_uuid_offsets =
                self.build_periodic_vertex_uuid_offsets(simplex_key, &vertices)?;

            // Use Entry API for atomic check-and-insert
            match unique_simplices.entry(vertex_uuid_offsets) {
                Entry::Occupied(_) => {
                    simplices_to_remove.push(simplex_key);
                }
                Entry::Vacant(e) => {
                    e.insert(simplex_key);
                }
            }
        }

        let duplicate_count = simplices_to_remove.len();

        if duplicate_count == 0 {
            return Ok(0);
        }

        let original_generation = self.generation();
        let mut trial = self.clone_for_rollback();
        trial.generation = Arc::new(AtomicU64::new(original_generation));
        let removed = trial.remove_simplices_by_keys(&simplices_to_remove);
        let rebuild_result = (|| -> Result<(), TdsMutationError> {
            trial.assign_neighbors().map_err(TdsMutationError::from)?;
            trial.assign_incident_simplices()?;
            trial.is_valid().map_err(TdsMutationError::from)?;
            Ok(())
        })();

        if let Err(error) = rebuild_result {
            self.generation
                .store(original_generation, Ordering::Relaxed);
            return Err(error);
        }

        *self = trial;
        Ok(removed)
    }

    // =========================================================================
    // VALIDATION & CONSISTENCY CHECKS
    // =========================================================================
    // Note: Structural validation is topology-only. Level-1 element validation
    // checks validated f64 coordinate storage.

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
    /// `Ok(())` if all vertex mappings are consistent, otherwise a `TdsError`.
    ///
    /// This corresponds to [`InvariantKind::VertexMappings`], which is included in
    /// [`Tds::is_valid`](Self::is_valid) and [`Tds::validate`](Self::validate), and is also surfaced by
    /// [`DelaunayTriangulation::validation_report()`](crate::DelaunayTriangulation::validation_report).
    ///
    /// # Errors
    ///
    /// Returns a `TdsError::MappingInconsistency` with a descriptive message if:
    /// - The number of UUID-to-key mappings doesn't match the number of vertices
    /// - The number of key-to-UUID mappings doesn't match the number of vertices
    /// - A vertex exists without a corresponding UUID-to-key mapping
    /// - A vertex exists without a corresponding key-to-UUID mapping
    /// - The bidirectional mappings are inconsistent (UUID maps to key A, but key A maps to different UUID)
    ///
    pub(crate) fn validate_vertex_mappings(&self) -> Result<(), TdsError> {
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

        // Check the key-to-UUID direction first (direct storage map access),
        // then only do UUID-to-key lookup verification when needed.
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

    /// Validates the consistency of simplex UUID-to-key mappings.
    ///
    /// This helper function ensures that:
    /// 1. The number of entries in `simplex_uuid_to_key` matches the number of simplices
    /// 2. The number of entries in `simplex_key_to_uuid` matches the number of simplices
    /// 3. Every simplex UUID in the triangulation has a corresponding key mapping
    /// 4. Every simplex key in the triangulation has a corresponding UUID mapping
    /// 5. The mappings are bidirectional and consistent (UUID ↔ Key)
    ///
    /// # Returns
    ///
    /// `Ok(())` if all simplex mappings are consistent, otherwise a `TdsError`.
    ///
    /// This corresponds to [`InvariantKind::SimplexMappings`], which is included in
    /// [`Tds::is_valid`](Self::is_valid) and [`Tds::validate`](Self::validate), and is also surfaced by
    /// [`DelaunayTriangulation::validation_report()`].
    ///
    /// [`DelaunayTriangulation::validation_report()`]: crate::DelaunayTriangulation::validation_report
    ///
    /// # Errors
    ///
    /// Returns a `TdsError::MappingInconsistency` with a descriptive message if:
    /// - The number of UUID-to-key mappings doesn't match the number of simplices
    /// - The number of key-to-UUID mappings doesn't match the number of simplices
    /// - A simplex exists without a corresponding UUID-to-key mapping
    /// - A simplex exists without a corresponding key-to-UUID mapping
    /// - The bidirectional mappings are inconsistent (UUID maps to key A, but key A maps to different UUID)
    ///
    pub(crate) fn validate_simplex_mappings(&self) -> Result<(), TdsError> {
        if self.uuid_to_simplex_key.len() != self.simplices.len() {
            return Err(TdsError::MappingInconsistency {
                entity: EntityKind::Simplex,
                message: format!(
                    "Number of mapping entries ({}) doesn't match number of simplices ({})",
                    self.uuid_to_simplex_key.len(),
                    self.simplices.len()
                ),
            });
        }

        // Check the key-to-UUID direction first (direct storage map access),
        // then only do UUID-to-key lookup verification when needed.
        for (simplex_key, simplex) in &self.simplices {
            let simplex_uuid = simplex.uuid();

            // Check key-to-UUID direction first (direct storage map access - no hash lookup)
            if self.simplex_uuid_from_key(simplex_key) != Some(simplex_uuid) {
                return Err(TdsError::MappingInconsistency {
                    entity: EntityKind::Simplex,
                    message: format!(
                        "Inconsistent or missing key-to-UUID mapping for key {simplex_key:?}"
                    ),
                });
            }

            // Now verify UUID-to-key direction (requires hash lookup but we know it should exist)
            if self.uuid_to_simplex_key.get(&simplex_uuid) != Some(&simplex_key) {
                return Err(TdsError::MappingInconsistency {
                    entity: EntityKind::Simplex,
                    message: format!(
                        "Inconsistent or missing UUID-to-key mapping for UUID {simplex_uuid:?}"
                    ),
                });
            }
        }
        Ok(())
    }

    /// Validates that all vertex keys referenced by simplices actually exist in the vertices `storage map`.
    ///
    /// This is a defensive check for data structure corruption. In normal operation,
    /// this should never fail, but it's useful for catching bugs during development
    /// and for comprehensive validation.
    ///
    /// This ensures that no simplex references a stale or invalid vertex key.
    ///
    /// # Returns
    ///
    /// `Ok(())` if all vertex keys in all simplices are valid, otherwise a `TdsError`.
    ///
    /// # Errors
    ///
    /// Returns `TdsError::VertexNotFound` if any simplex
    /// references a vertex key that doesn't exist in the vertices `storage map`.
    pub(crate) fn validate_simplex_vertex_keys(&self) -> Result<(), TdsError> {
        for (simplex_key, simplex) in &self.simplices {
            let simplex_uuid = simplex.uuid();
            for (vertex_idx, &vertex_key) in simplex.vertices().iter().enumerate() {
                if !self.vertices.contains_key(vertex_key) {
                    return Err(TdsError::VertexNotFound {
                        vertex_key,
                        context: format!(
                            "referenced by simplex {simplex_uuid} (key {simplex_key:?}) at position {vertex_idx}"
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    /// Validates that `Vertex::incident_simplex` pointers are non-dangling and internally consistent.
    ///
    /// Note: at the TDS structural layer (Level 2), isolated vertices (vertices not referenced by
    /// any simplex) are allowed, so `Vertex::incident_simplex` may be `None`.
    ///
    /// Level 3 topology validation (`Triangulation::is_valid`) rejects isolated vertices.
    ///
    /// However, any `incident_simplex` pointer that *is* present must:
    /// - point to an existing simplex key, and
    /// - reference a simplex that actually contains the vertex.
    fn validate_vertex_incidence(&self) -> Result<(), TdsError> {
        for (vertex_key, vertex) in &self.vertices {
            let Some(incident_simplex_key) = vertex.incident_simplex() else {
                continue;
            };

            let Some(incident_simplex) = self.simplices.get(incident_simplex_key) else {
                return Err(TdsError::SimplexNotFound {
                    simplex_key: incident_simplex_key,
                    context: format!(
                        "dangling incident_simplex pointer from vertex {vertex_key:?}"
                    ),
                });
            };

            if !incident_simplex.contains_vertex(vertex_key) {
                return Err(TdsError::InconsistentDataStructure {
                    message: format!(
                        "Vertex {vertex_key:?} incident_simplex {incident_simplex_key:?} does not contain the vertex"
                    ),
                });
            }
        }

        Ok(())
    }

    /// Check for duplicate simplices and return an error if any are found
    ///
    /// This is useful for validation where you want to detect duplicates
    /// without automatically removing them.
    ///
    /// **Implementation Note**: This method uses `Simplex::vertex_uuids()` to get canonical
    /// vertex UUIDs for each simplex, which are then sorted and compared for duplicate detection.
    ///
    /// # Errors
    ///
    /// Returns a [`TdsError`] if simplex vertex retrieval fails
    /// or if any duplicate simplices are detected.
    ///
    /// This corresponds to [`InvariantKind::DuplicateSimplices`], which is included in
    /// [`Tds::is_valid`](Self::is_valid) and [`Tds::validate`](Self::validate), and is also surfaced by
    /// [`DelaunayTriangulation::validation_report()`].
    ///
    /// [`DelaunayTriangulation::validation_report()`]: crate::DelaunayTriangulation::validation_report
    fn validate_no_duplicate_simplices(&self) -> Result<(), TdsError> {
        // Include periodic per-vertex offsets in the duplicate key so periodic quotient simplices
        // with identical vertex sets but distinct lattice offsets are not collapsed.
        let mut unique_simplices: FastHashMap<SimplexUuidSortKey<D>, SimplexKey> =
            fast_hash_map_with_capacity(self.simplices.len());
        let mut duplicates = Vec::new();

        for (simplex_key, _simplex) in &self.simplices {
            let vertices = self.simplex_vertices(simplex_key)?;
            let vertex_uuid_offsets =
                self.build_periodic_vertex_uuid_offsets(simplex_key, &vertices)?;

            if let Some(existing_simplex_key) = unique_simplices.get(&vertex_uuid_offsets) {
                duplicates.push((
                    simplex_key,
                    *existing_simplex_key,
                    vertex_uuid_offsets.clone(),
                ));
            } else {
                unique_simplices.insert(vertex_uuid_offsets, simplex_key);
            }
        }

        if !duplicates.is_empty() {
            let duplicate_descriptions: Vec<String> = duplicates
                .iter()
                .map(|(simplex1, simplex2, vertex_uuids)| {
                    format!(
                        "simplices {simplex1:?} and {simplex2:?} with vertex UUIDs {vertex_uuids:?}"
                    )
                })
                .collect();

            return Err(TdsError::DuplicateSimplices {
                message: format!(
                    "Found {} duplicate simplex(s): {}",
                    duplicates.len(),
                    duplicate_descriptions.join(", ")
                ),
            });
        }

        Ok(())
    }

    /// Validates that no simplex contains vertices with identical coordinates.
    ///
    /// This is a geometric-level check complementing [`Simplex::try_new()`]'s vertex-key uniqueness
    /// check. Two different vertex keys can reference geometrically identical points, producing
    /// a zero-volume simplex that is catastrophic for `SoS` orientation and Pachner moves.
    ///
    /// Uses exact `OrderedFloat`-based coordinate comparison (NaN-aware, +0.0 == -0.0).
    ///
    /// # Errors
    ///
    /// Returns [`TdsError::DuplicateCoordinatesInSimplex`] on the first simplex found
    /// containing two vertices with identical coordinates.
    fn validate_simplex_coordinate_uniqueness(&self) -> Result<(), TdsError> {
        for (_simplex_key, simplex) in &self.simplices {
            let vkeys = simplex.vertices();
            // O(D²) pairwise comparison per simplex — acceptable since D is small (≤ 6).
            for i in 0..vkeys.len() {
                let Some(vi) = self.vertex(vkeys[i]) else {
                    continue; // Missing keys are caught by validate_simplex_vertex_keys
                };
                for j in (i + 1)..vkeys.len() {
                    // Same key → same vertex; skip to avoid a misleading
                    // "duplicate coordinates" error for what is really a
                    // duplicate-key issue (caught by Simplex::new).
                    if vkeys[i] == vkeys[j] {
                        continue;
                    }
                    let Some(vj) = self.vertex(vkeys[j]) else {
                        continue;
                    };
                    if coords_equal_exact(vi.point().coords(), vj.point().coords()) {
                        return Err(TdsError::DuplicateCoordinatesInSimplex {
                            simplex_id: simplex.uuid(),
                            message: format!(
                                "vertices {:?} and {:?} (keys {:?}, {:?}) have identical coordinates {:?}",
                                vi.uuid(),
                                vj.uuid(),
                                vkeys[i],
                                vkeys[j],
                                vi.point().coords(),
                            ),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Validates that no facet is shared by more than 2 simplices
    ///
    /// This is a critical property for valid triangulations. Each facet should be
    /// shared by at most 2 simplices - boundary facets belong to 1 simplex, and internal
    /// facets should be shared by exactly 2 adjacent simplices.
    ///
    /// # Errors
    ///
    /// Returns a [`TdsError`] if building the facet map fails
    /// or if any facet is shared by more than two simplices.
    ///
    /// This corresponds to [`InvariantKind::FacetSharing`], which is included in
    /// [`Tds::is_valid`](Self::is_valid) and [`Tds::validate`](Self::validate), and is also surfaced by
    /// [`DelaunayTriangulation::validation_report()`].
    ///
    /// [`DelaunayTriangulation::validation_report()`]: crate::DelaunayTriangulation::validation_report
    pub(crate) fn validate_facet_sharing(&self) -> Result<(), TdsError> {
        // Build a map from facet keys to the simplices that contain them.
        // Use the strict version to ensure we catch any missing vertex keys.
        let facet_to_simplices = self.build_facet_to_simplices_map()?;
        self.validate_facet_sharing_with_facet_to_simplices_map(&facet_to_simplices)
    }

    /// Checks whether all adjacent simplices induce opposite orientations on shared facets.
    ///
    /// This is a combinatorial check based on simplex vertex ordering and neighbor slots.
    /// It does not evaluate geometric predicates.
    ///
    /// Returns `false` on the first detected inconsistency or data-structure error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0])?,
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 2> =
    ///     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// assert!(dt.tds().is_coherently_oriented());
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn is_coherently_oriented(&self) -> bool {
        self.validate_coherent_orientation().is_ok()
    }

    /// Validates coherent combinatorial orientation for all adjacent simplex pairs.
    ///
    /// For two neighboring simplices that share a facet, this verifies the induced
    /// facet orientations are opposite (boundary-orientation convention).
    ///
    /// # Errors
    ///
    /// Returns [`TdsError`] on the first detected problem:
    /// - [`OrientationViolation`](TdsError::OrientationViolation) — adjacent simplices do not induce opposite facet orientations.
    /// - [`InvalidNeighbors`](TdsError::InvalidNeighbors) — a mirror facet cannot be derived, or a
    ///   neighbor's back-reference does not point to the originating simplex.
    /// - [`SimplexNotFound`](TdsError::SimplexNotFound) — a neighbor simplex key is missing from storage.
    /// - [`InconsistentDataStructure`](TdsError::InconsistentDataStructure) — permutation parity cannot be determined.
    /// - [`IndexOutOfBounds`](TdsError::IndexOutOfBounds) / [`DimensionMismatch`](TdsError::DimensionMismatch)
    ///   — facet-extraction helpers encounter invalid indices or periodic-offset count mismatches.
    fn validate_coherent_orientation(&self) -> Result<(), TdsError> {
        for (simplex_key, simplex) in &self.simplices {
            let Some(neighbors) = simplex.neighbor_keys() else {
                continue;
            };

            for (facet_idx, neighbor_key_opt) in neighbors.enumerate() {
                let Some(neighbor_key) = neighbor_key_opt else {
                    continue; // Boundary facet
                };

                // Periodic quotient triangulations may use self-neighbors.
                // Neighbor/topology validation handles admissibility checks.
                if neighbor_key == simplex_key && Self::allows_periodic_self_neighbor(simplex) {
                    continue;
                }

                let neighbor_simplex =
                    self.simplices
                        .get(neighbor_key)
                        .ok_or_else(|| TdsError::SimplexNotFound {
                            simplex_key: neighbor_key,
                            context: format!(
                                "neighbor of simplex {simplex_key:?} during orientation validation",
                            ),
                        })?;

                let (mirror_idx, uses_periodic_offsets) = Self::orientation_mirror_facet_index(
                    simplex,
                    facet_idx,
                    neighbor_simplex,
                    "orientation validation",
                )?;
                let back_neighbor = neighbor_simplex
                    .neighbor_key(mirror_idx)
                    .flatten()
                    .ok_or_else(|| TdsError::InvalidNeighbors {
                        reason: NeighborValidationError::BackReferenceMismatch {
                            simplex_key,
                            simplex_uuid: simplex.uuid(),
                            facet_index: facet_idx,
                            neighbor_key,
                            neighbor_uuid: neighbor_simplex.uuid(),
                            mirror_index: mirror_idx,
                            observed: None,
                            context: "orientation validation".to_string(),
                        },
                    })?;
                if back_neighbor != simplex_key {
                    return Err(TdsError::InvalidNeighbors {
                        reason: NeighborValidationError::BackReferenceMismatch {
                            simplex_key,
                            simplex_uuid: simplex.uuid(),
                            facet_index: facet_idx,
                            neighbor_key,
                            neighbor_uuid: neighbor_simplex.uuid(),
                            mirror_index: mirror_idx,
                            observed: Some(back_neighbor),
                            context: "orientation validation".to_string(),
                        },
                    });
                }
                if uses_periodic_offsets {
                    continue;
                }

                let simplex1_facet_vertices =
                    Self::facet_vertices_in_simplex_order(simplex, facet_idx)?;
                let simplex2_facet_vertices =
                    Self::facet_vertices_in_simplex_order(neighbor_simplex, mirror_idx)?;
                let (currently_coherent, observed_odd_permutation, expected_odd_permutation) =
                    Self::facet_permutation_parity(
                        simplex,
                        facet_idx,
                        neighbor_simplex,
                        mirror_idx,
                    )?;

                if !currently_coherent {
                    return Err(TdsError::OrientationViolation {
                        simplex1_key: simplex_key,
                        simplex1_uuid: simplex.uuid(),
                        simplex2_key: neighbor_key,
                        simplex2_uuid: neighbor_simplex.uuid(),
                        simplex1_facet_index: facet_idx,
                        simplex2_facet_index: mirror_idx,
                        facet_vertices: simplex1_facet_vertices.into_iter().collect(),
                        simplex2_facet_vertices: simplex2_facet_vertices.into_iter().collect(),
                        observed_odd_permutation,
                        expected_odd_permutation,
                    });
                }
            }
        }

        Ok(())
    }

    /// Resolves the mirror facet for orientation validation without forcing periodic parity checks.
    fn orientation_mirror_facet_index(
        simplex: &Simplex<V, D>,
        facet_idx: usize,
        neighbor_simplex: &Simplex<V, D>,
        context: &str,
    ) -> Result<(usize, bool), TdsError> {
        let uses_periodic_offsets = simplex.periodic_vertex_offsets().is_some()
            || neighbor_simplex.periodic_vertex_offsets().is_some();
        let mirror_idx = if uses_periodic_offsets {
            Self::matching_lifted_mirror_facet_index(simplex, facet_idx, neighbor_simplex, context)?
        } else {
            simplex
                .mirror_facet_index(facet_idx, neighbor_simplex)
                .ok_or_else(|| TdsError::InvalidNeighbors {
                    reason: NeighborValidationError::MirrorFacetMissing {
                        simplex_uuid: simplex.uuid(),
                        facet_index: facet_idx,
                        neighbor_uuid: neighbor_simplex.uuid(),
                        context: context.to_string(),
                    },
                })?
        };
        Ok((mirror_idx, uses_periodic_offsets))
    }

    fn facet_vertices_in_simplex_order(
        simplex: &Simplex<V, D>,
        omit_idx: usize,
    ) -> Result<SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>, TdsError> {
        if omit_idx >= simplex.number_of_vertices() {
            return Err(TdsError::IndexOutOfBounds {
                index: omit_idx,
                bound: simplex.number_of_vertices(),
                context: format!(
                    "facet index for simplex {:?} during orientation validation",
                    simplex.uuid(),
                ),
            });
        }

        let mut facet_vertices = SmallBuffer::new();
        for (idx, &vkey) in simplex.vertices().iter().enumerate() {
            if idx != omit_idx {
                facet_vertices.push(vkey);
            }
        }
        Ok(facet_vertices)
    }
    /// Build facet vertex identities in simplex-local order, including periodic offsets.
    ///
    /// Offsets are normalized by subtracting a deterministic anchor offset so the
    /// same lifted facet can be compared across neighboring simplices independent of a
    /// global translation. The anchor is selected lexicographically by
    /// `(vertex_key_value, offset)`.
    fn facet_vertex_identities_in_simplex_order(
        simplex: &Simplex<V, D>,
        omit_idx: usize,
    ) -> Result<SmallBuffer<(VertexKey, [i16; D]), MAX_PRACTICAL_DIMENSION_SIZE>, TdsError> {
        if omit_idx >= simplex.number_of_vertices() {
            return Err(TdsError::IndexOutOfBounds {
                index: omit_idx,
                bound: simplex.number_of_vertices(),
                context: format!(
                    "facet index for simplex {:?} during orientation validation",
                    simplex.uuid(),
                ),
            });
        }

        let periodic_offsets = simplex.periodic_vertex_offsets();
        if let Some(offsets) = periodic_offsets
            && offsets.len() != simplex.number_of_vertices()
        {
            return Err(TdsError::DimensionMismatch {
                expected: simplex.number_of_vertices(),
                actual: offsets.len(),
                context: format!(
                    "periodic offset count for simplex {:?} during orientation validation",
                    simplex.uuid(),
                ),
            });
        }

        let mut facet_identities: SmallBuffer<(VertexKey, [i16; D]), MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::new();
        for (idx, &vkey) in simplex.vertices().iter().enumerate() {
            if idx == omit_idx {
                continue;
            }

            let raw_offset = periodic_offsets.map_or([0_i8; D], |offsets| offsets[idx]);
            let mut offset = [0_i16; D];
            for axis in 0..D {
                offset[axis] = i16::from(raw_offset[axis]);
            }
            facet_identities.push((vkey, offset));
        }

        let mut anchor_key = u64::MAX;
        let mut anchor_offset = [0_i16; D];
        for (vkey, offset) in &facet_identities {
            let key_value = vkey.data().as_ffi();
            if key_value < anchor_key || (key_value == anchor_key && *offset < anchor_offset) {
                anchor_key = key_value;
                anchor_offset = *offset;
            }
        }
        for (_, offset) in &mut facet_identities {
            for axis in 0..D {
                offset[axis] -= anchor_offset[axis];
            }
        }

        Ok(facet_identities)
    }

    /// Derive observed and expected facet permutation parity between neighboring simplices.
    ///
    /// Returns `(currently_coherent, observed_odd_permutation, expected_odd_permutation)`.
    /// The expected odd parity follows the coherent boundary-orientation convention:
    /// odd is expected exactly when `(facet_idx + mirror_idx)` is even.
    fn facet_permutation_parity(
        simplex: &Simplex<V, D>,
        facet_idx: usize,
        neighbor_simplex: &Simplex<V, D>,
        mirror_idx: usize,
    ) -> Result<(bool, bool, bool), TdsError> {
        let simplex_facet_identities =
            Self::facet_vertex_identities_in_simplex_order(simplex, facet_idx)?;
        let neighbor_facet_identities =
            Self::facet_vertex_identities_in_simplex_order(neighbor_simplex, mirror_idx)?;

        let observed_odd_permutation = Self::permutation_is_odd(
            &simplex_facet_identities[..],
            &neighbor_facet_identities[..],
        )
        .ok_or_else(|| TdsError::InconsistentDataStructure {
            message: format!(
                "Could not derive facet-order permutation parity between simplices {:?} and {:?}",
                simplex.uuid(),
                neighbor_simplex.uuid(),
            ),
        })?;

        let expected_odd_permutation = (facet_idx + mirror_idx).is_multiple_of(2);
        Ok((
            observed_odd_permutation == expected_odd_permutation,
            observed_odd_permutation,
            expected_odd_permutation,
        ))
    }
    /// Returns whether the permutation mapping `source_order` to `target_order` is odd.
    ///
    /// Returns `None` if the orders are not permutations of each other.
    fn permutation_is_odd<Id: PartialEq>(source_order: &[Id], target_order: &[Id]) -> Option<bool> {
        if source_order.len() != target_order.len() {
            return None;
        }

        let mut target_positions: SmallBuffer<usize, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::with_capacity(source_order.len());
        let mut used_target_indices: SmallBuffer<bool, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::from_elem(false, target_order.len());

        for source_vertex in source_order {
            let mut matched_target_position = None;
            for (target_idx, target_vertex) in target_order.iter().enumerate() {
                if target_vertex == source_vertex && !used_target_indices[target_idx] {
                    matched_target_position = Some(target_idx);
                    used_target_indices[target_idx] = true;
                    break;
                }
            }
            target_positions.push(matched_target_position?);
        }

        if used_target_indices.iter().any(|used| !*used) {
            return None;
        }

        let mut is_odd = false;
        for i in 0..target_positions.len() {
            for j in (i + 1)..target_positions.len() {
                if target_positions[i] > target_positions[j] {
                    is_odd = !is_odd;
                }
            }
        }

        Some(is_odd)
    }

    /// Validate facet multiplicity from an already-built facet incidence map.
    ///
    /// This helper exists so [`Tds::is_valid`] and validation reports can share
    /// one O(N×F) facet-map construction while still emitting the same structured
    /// [`TdsError::FacetSharingViolation`] as insertion preflight. It returns the
    /// first facet incident to more than two simplices, identifying one offending
    /// incident simplex in the `candidate_*` fields.
    fn validate_facet_sharing_with_facet_to_simplices_map(
        &self,
        facet_to_simplices: &FacetToSimplicesMap,
    ) -> Result<(), TdsError> {
        // Check for facets shared by more than 2 simplices.
        for (facet_key, simplex_facet_pairs) in facet_to_simplices {
            let [_, _, candidate, ..] = simplex_facet_pairs.as_slice() else {
                continue;
            };
            let candidate_simplex =
                self.simplex(candidate.simplex_key())
                    .ok_or_else(|| TdsError::SimplexNotFound {
                        simplex_key: candidate.simplex_key(),
                        context: format!(
                            "facet-sharing validation for over-shared facet {facet_key}"
                        ),
                    })?;
            return Err(TdsError::FacetSharingViolation {
                facet_key: *facet_key,
                existing_incident_count: simplex_facet_pairs.len() - 1,
                attempted_incident_count: simplex_facet_pairs.len(),
                max_incident_count: 2,
                candidate_simplex_uuid: candidate_simplex.uuid(),
                candidate_facet_index: usize::from(candidate.facet_index()),
            });
        }

        Ok(())
    }

    /// Checks whether the triangulation data structure is structurally valid.
    ///
    /// This is a **Level 2 (TDS structural)** check in the validation hierarchy.
    /// It intentionally does **not** validate individual vertices/simplices (Level 1),
    /// nor triangulation topology (Level 3), nor the Delaunay property (Level 4).
    ///
    /// # Structural invariants checked
    /// - Vertex UUID↔key mapping consistency
    /// - Simplex UUID↔key mapping consistency
    /// - Simplices reference only valid vertex keys (no stale/missing vertex keys)
    /// - `Vertex::incident_simplex`, when present, must point at an existing simplex that contains the vertex.
    /// - No duplicate simplices (same vertex set)
    /// - Facet sharing invariant (each facet is shared by at most 2 simplices,
    ///   reported as [`TdsError::FacetSharingViolation`])
    /// - Neighbor consistency (topology + mutual neighbors)
    /// - Coherent orientation (adjacent simplices induce opposite facet orientations)
    ///
    /// # ⚠️ Performance Warning
    ///
    /// **This method can be expensive** for large triangulations:
    /// - **Time Complexity**: O(N×F + N×D²) where N is the number of simplices and F = D+1 facets per simplex
    /// - **Space Complexity**: O(N×F) for facet-to-simplex mappings
    ///
    /// For a cumulative validator that also checks vertices/simplices (Level 1), use
    /// [`Tds::validate`](Self::validate).
    ///
    /// # Errors
    ///
    /// Returns a [`TdsError`] if any structural invariant fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices_4d = [
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 1.0])?,
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulationBuilder::new(&vertices_4d).build::<()>()?;
    ///
    /// // Level 2: TDS structural validation
    /// assert!(dt.tds().is_valid().is_ok());
    /// # Ok(())
    /// # }
    /// ```
    pub fn is_valid(&self) -> Result<(), TdsError> {
        // Fast-fail: return the first violated invariant.
        // For full diagnostics across all structural invariants, use `validation_report()`.
        self.validate_vertex_mappings()?;
        self.validate_simplex_mappings()?;

        // Defensive: ensure no simplex references a stale/missing vertex key before
        // higher-level structural checks that assume key validity.
        self.validate_simplex_vertex_keys()?;

        // Structural: ensure `incident_simplex` pointers, when present, are non-dangling + consistent.
        self.validate_vertex_incidence()?;

        self.validate_no_duplicate_simplices()?;

        // Build the facet-to-simplices map once and share it between facet-sharing and neighbor validators.
        let facet_to_simplices = self.build_facet_to_simplices_map()?;
        self.validate_facet_sharing_with_facet_to_simplices_map(&facet_to_simplices)?;
        self.validate_neighbors_with_facet_to_simplices_map(&facet_to_simplices)?;
        self.validate_coherent_orientation()?;

        Ok(())
    }

    /// Performs cumulative validation for Levels 1–2.
    ///
    /// This validates:
    /// - **Level 1**: all vertices (`Vertex::is_valid`) and all simplices (`Simplex::is_valid`)
    /// - **Level 2**: structural invariants (`Tds::is_valid`)
    ///
    /// # Errors
    ///
    /// Returns a [`TdsError`] if any vertex/simplex is invalid or if any
    /// structural invariant fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Insertion(#[from] delaunay::prelude::insertion::InsertionError),
    /// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)] TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// #     #[error(transparent)] Invariant(#[from] delaunay::prelude::tds::InvariantError),
    /// #     #[error(transparent)] Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)] Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices_4d = [
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 1.0])?,
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulationBuilder::new(&vertices_4d).build::<()>()?;
    ///
    /// // Levels 1–2: elements + TDS structure
    /// assert!(dt.tds().validate().is_ok());
    /// # Ok(())
    /// # }
    /// ```
    pub fn validate(&self) -> Result<(), TdsError> {
        for (_vertex_key, vertex) in &self.vertices {
            if let Err(source) = (*vertex).is_valid() {
                return Err(TdsError::InvalidVertex {
                    vertex_id: vertex.uuid(),
                    source,
                });
            }
        }

        for (simplex_key, simplex) in &self.simplices {
            if let Err(source) = simplex.is_valid() {
                let Some(simplex_id) = self.simplex_uuid_from_key(simplex_key) else {
                    return Err(TdsError::InconsistentDataStructure {
                        message: format!(
                            "Simplex key {simplex_key:?} has no UUID mapping during validation",
                        ),
                    });
                };

                return Err(TdsError::InvalidSimplex { simplex_id, source });
            }
        }

        // Coordinate-level duplicate detection: different vertex keys with identical
        // coordinates produce zero-volume simplices that break SoS and Pachner moves.
        // Guard behind simplex-vertex-key validity so that stale keys are reported as
        // key-reference failures (by is_valid below) rather than confusing coordinate
        // errors.  Matches the pattern in validation_report().
        if self.validate_simplex_vertex_keys().is_ok() {
            self.validate_simplex_coordinate_uniqueness()?;
        }

        self.is_valid()
    }

    /// Runs structural validation checks and returns a report containing **all** failed invariants.
    ///
    /// Unlike [`is_valid()`](Self::is_valid), this method does **not** stop at the
    /// first error. Instead it records a [`TdsError`] for each
    /// invariant group that fails and returns them as a
    /// [`TriangulationValidationReport`].
    ///
    /// **Note**: If UUID↔key mappings are inconsistent, this returns only mapping-related
    /// failures. Additional checks may produce misleading secondary errors or panic.
    ///
    /// **Note**: If any simplex references a stale/missing vertex key, this reports the
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
    #[expect(
        clippy::too_many_lines,
        reason = "validation report aggregation is intentionally linear and simplex nomenclature makes existing names longer"
    )]
    pub(crate) fn validation_report(&self) -> Result<(), TriangulationValidationReport> {
        let mut violations = Vec::new();

        // 1. Mapping consistency (vertex + simplex UUID↔key mappings)
        if let Err(e) = self.validate_vertex_mappings() {
            violations.push(InvariantViolation {
                kind: InvariantKind::VertexMappings,
                error: e.into(),
            });
        }
        if let Err(e) = self.validate_simplex_mappings() {
            violations.push(InvariantViolation {
                kind: InvariantKind::SimplexMappings,
                error: e.into(),
            });
        }

        // If mappings are inconsistent, additional checks may produce confusing
        // secondary errors or panic. In that case, stop here and return the
        // mapping-related failures only.
        if !violations.is_empty() {
            return Err(TriangulationValidationReport { violations });
        }

        // 2. Simplex→vertex key references (no stale/missing vertex keys)
        let mut simplex_vertex_keys_ok = true;
        if let Err(e) = self.validate_simplex_vertex_keys() {
            simplex_vertex_keys_ok = false;
            violations.push(InvariantViolation {
                kind: InvariantKind::SimplexVertexKeys,
                error: e.into(),
            });
        }

        // 2b. Simplex coordinate uniqueness (no simplices with duplicate-coordinate vertices)
        if simplex_vertex_keys_ok && let Err(e) = self.validate_simplex_coordinate_uniqueness() {
            violations.push(InvariantViolation {
                kind: InvariantKind::SimplexCoordinateUniqueness,
                error: e.into(),
            });
        }

        // 3. Vertex incidence (non-dangling `incident_simplex` pointers, when present)
        if let Err(e) = self.validate_vertex_incidence() {
            violations.push(InvariantViolation {
                kind: InvariantKind::VertexIncidence,
                error: e.into(),
            });
        }

        // If simplex vertex keys are invalid, derived invariants may produce confusing secondary
        // errors or panic (many routines assume key validity). Stop here.
        if !simplex_vertex_keys_ok {
            return Err(TriangulationValidationReport { violations });
        }

        // 4. Simplex uniqueness (no duplicate simplices with identical vertex sets)
        if let Err(e) = self.validate_no_duplicate_simplices() {
            violations.push(InvariantViolation {
                kind: InvariantKind::DuplicateSimplices,
                error: e.into(),
            });
        }

        // 5–7. Facet sharing + neighbor consistency + coherent orientation.
        let mut neighbor_consistency_ok = false;
        match self.build_facet_to_simplices_map() {
            Ok(facet_to_simplices) => {
                if let Err(e) =
                    self.validate_facet_sharing_with_facet_to_simplices_map(&facet_to_simplices)
                {
                    violations.push(InvariantViolation {
                        kind: InvariantKind::FacetSharing,
                        error: e.into(),
                    });
                }

                if let Err(e) =
                    self.validate_neighbors_with_facet_to_simplices_map(&facet_to_simplices)
                {
                    violations.push(InvariantViolation {
                        kind: InvariantKind::NeighborConsistency,
                        error: e.into(),
                    });
                } else {
                    neighbor_consistency_ok = true;
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

        if neighbor_consistency_ok && let Err(e) = self.validate_coherent_orientation() {
            violations.push(InvariantViolation {
                kind: InvariantKind::CoherentOrientation,
                error: e.into(),
            });
        }

        // 8. Connectivity (topology-layer invariant; reported here for comprehensive diagnostics).
        //
        // Note: connectivity is NOT part of Level-2 `is_valid()` — it belongs at Level 3
        // (Triangulation::is_valid). It is included here in the diagnostic report so that
        // `DelaunayTriangulation::validation_report()` surfaces it together with all other
        // structural failures, even when the Triangulation wrapper is not available.
        if !self.is_connected() {
            violations.push(InvariantViolation {
                kind: InvariantKind::Connectedness,
                error: TdsError::InconsistentDataStructure {
                    message: format!(
                        "Disconnected triangulation: simplex neighbor graph is not a single \
                         connected component ({} simplices total)",
                        self.simplices.len()
                    ),
                }
                .into(),
            });
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
    /// This routine is intentionally thorough and defensive. It precomputes per-simplex vertex sets
    /// and performs per-neighbor set-intersection + mirror-facet cross-checks, which can be
    /// relatively expensive for large triangulations.
    ///
    /// It is only invoked from explicit validation APIs (`is_valid()`, `validate()`,
    /// `validation_report()`) and is not intended for per-step hot paths.
    ///
    /// Some small optimizations keep the cost reasonable:
    /// - Early termination on validation failures
    /// - Precomputing per-simplex vertex sets once
    /// - Counting intersections without allocating intermediate collections
    ///
    /// # Errors
    ///
    /// Returns a [`TdsError`] if any neighbor relationship
    /// violates topological or consistency invariants.
    ///
    /// This corresponds to [`InvariantKind::NeighborConsistency`], which is included in
    /// [`Tds::is_valid`](Self::is_valid) and [`Tds::validate`](Self::validate), and is also surfaced by
    /// [`DelaunayTriangulation::validation_report()`].
    ///
    /// [`DelaunayTriangulation::validation_report()`]: crate::DelaunayTriangulation::validation_report
    ///
    /// Note: callers provide `facet_to_simplices` so `is_valid()` and `validation_report()` can share
    /// the precomputed facet map between validators.
    fn validate_neighbors_with_facet_to_simplices_map(
        &self,
        facet_to_simplices: &FacetToSimplicesMap,
    ) -> Result<(), TdsError> {
        self.validate_neighbor_pointers_match_facet_to_simplices_map(facet_to_simplices)?;

        let simplex_vertices = self.build_simplex_vertex_sets()?;
        self.validate_neighbors_with_precomputed_vertex_sets(&simplex_vertices)
    }

    fn validate_neighbor_pointers_match_facet_to_simplices_map(
        &self,
        facet_to_simplices: &FacetToSimplicesMap,
    ) -> Result<(), TdsError> {
        for (facet_key, simplex_facet_pairs) in facet_to_simplices {
            match simplex_facet_pairs.as_slice() {
                [handle] => self.validate_boundary_facet_neighbor_pointer(*facet_key, *handle)?,
                [a, b] => self.validate_interior_facet_neighbor_pointer(*facet_key, *a, *b)?,
                _ => {
                    // Non-manifold facet multiplicity should have been caught by facet-sharing validation.
                    return Err(TdsError::InconsistentDataStructure {
                        message: format!(
                            "Facet with key {facet_key} is shared by {} simplices, but should be shared by at most 2 simplices in a valid triangulation",
                            simplex_facet_pairs.len()
                        ),
                    });
                }
            }
        }

        Ok(())
    }

    fn validate_boundary_facet_neighbor_pointer(
        &self,
        facet_key: u64,
        handle: FacetHandle,
    ) -> Result<(), TdsError> {
        let simplex_key = handle.simplex_key();
        let facet_index = handle.facet_index() as usize;
        let simplex = self
            .simplices
            .get(simplex_key)
            .ok_or_else(|| TdsError::SimplexNotFound {
                simplex_key,
                context: "neighbor validation (boundary facet)".to_string(),
            })?;

        let Some(neighbor_slots) = simplex.neighbor_slots() else {
            return Ok(());
        };
        let Some(neighbor_slot) = neighbor_slots.get(facet_index).copied() else {
            return Err(TdsError::InvalidNeighbors {
                reason: NeighborValidationError::LengthMismatch {
                    actual: neighbor_slots.len(),
                    expected: D + 1,
                    context: "neighbor validation (boundary facet)".to_string(),
                },
            });
        };

        match neighbor_slot {
            NeighborSlot::Unassigned => Err(TdsError::InvalidNeighbors {
                reason: NeighborValidationError::UnassignedNeighborSlot {
                    simplex_key,
                    simplex_uuid: simplex.uuid(),
                    facet_index,
                    context: "neighbor validation (boundary facet)".to_string(),
                },
            }),
            NeighborSlot::Boundary => Ok(()),
            NeighborSlot::Neighbor(neighbor) if neighbor == simplex_key => {
                if Self::allows_periodic_self_neighbor(simplex) {
                    return Ok(());
                }
                Err(TdsError::InvalidNeighbors {
                    reason: NeighborValidationError::BoundaryFacetHasNonPeriodicSelfNeighbor {
                        facet_key,
                        simplex_key,
                        simplex_uuid: simplex.uuid(),
                        facet_index,
                    },
                })
            }
            NeighborSlot::Neighbor(neighbor) => Err(TdsError::InvalidNeighbors {
                reason: NeighborValidationError::BoundaryFacetHasNeighbor {
                    facet_key,
                    simplex_key,
                    simplex_uuid: simplex.uuid(),
                    facet_index,
                    neighbor_key: neighbor,
                },
            }),
        }
    }

    fn validate_interior_facet_neighbor_pointer(
        &self,
        facet_key: u64,
        first: FacetHandle,
        second: FacetHandle,
    ) -> Result<(), TdsError> {
        let first_simplex_key = first.simplex_key();
        let first_facet_index = first.facet_index() as usize;
        let second_simplex_key = second.simplex_key();
        let second_facet_index = second.facet_index() as usize;

        let first_simplex =
            self.simplices
                .get(first_simplex_key)
                .ok_or_else(|| TdsError::SimplexNotFound {
                    simplex_key: first_simplex_key,
                    context: "neighbor validation (interior facet, first simplex)".to_string(),
                })?;
        let second_simplex =
            self.simplices
                .get(second_simplex_key)
                .ok_or_else(|| TdsError::SimplexNotFound {
                    simplex_key: second_simplex_key,
                    context: "neighbor validation (interior facet, second simplex)".to_string(),
                })?;

        let first_neighbor = first_simplex.neighbor_key(first_facet_index).flatten();
        let second_neighbor = second_simplex.neighbor_key(second_facet_index).flatten();

        if first_neighbor == Some(second_simplex_key) && second_neighbor == Some(first_simplex_key)
        {
            return Ok(());
        }

        Err(TdsError::InvalidNeighbors {
            reason: NeighborValidationError::InteriorFacetNeighborMismatch {
                facet_key,
                first_simplex_key,
                first_simplex_uuid: first_simplex.uuid(),
                first_facet_index,
                first_neighbor,
                second_simplex_key,
                second_simplex_uuid: second_simplex.uuid(),
                second_facet_index,
                second_neighbor,
            },
        })
    }

    fn validate_neighbors_with_precomputed_vertex_sets(
        &self,
        simplex_vertices: &SimplexVerticesMap,
    ) -> Result<(), TdsError> {
        for (simplex_key, simplex) in &self.simplices {
            let Some(neighbor_keys) = simplex.neighbor_keys() else {
                continue; // Skip simplices without neighbors
            };
            let neighbors_buf: NeighborBuffer<Option<SimplexKey>> = neighbor_keys.collect();

            // Validate topological invariant (neighbor[i] opposite vertex[i])
            self.validate_neighbor_topology(simplex_key, &neighbors_buf)?;

            let this_vertices = simplex_vertices.get(&simplex_key).ok_or_else(|| {
                TdsError::InconsistentDataStructure {
                    message: format!(
                        "Simplex {} (key {simplex_key:?}) missing from precomputed vertex set map during neighbor validation",
                        simplex.uuid()
                    ),
                }
            })?;

            for (facet_idx, neighbor_key_opt) in neighbors_buf.iter().copied().enumerate() {
                // Skip None neighbors (missing neighbors)
                let Some(neighbor_key) = neighbor_key_opt else {
                    continue;
                };

                // Self-adjacency is valid for periodic quotient triangulations.
                if neighbor_key == simplex_key {
                    if Self::allows_periodic_self_neighbor(simplex) {
                        continue;
                    }
                    return Err(TdsError::InvalidNeighbors {
                        reason: NeighborValidationError::NonPeriodicSelfNeighbor {
                            simplex_key,
                            simplex_uuid: simplex.uuid(),
                            facet_index: facet_idx,
                        },
                    });
                }

                // Early termination: check if neighbor exists
                let Some(neighbor_simplex) = self.simplices.get(neighbor_key) else {
                    return Err(TdsError::InvalidNeighbors {
                        reason: NeighborValidationError::MissingNeighborSimplex {
                            simplex_key,
                            simplex_uuid: simplex.uuid(),
                            facet_index: facet_idx,
                            neighbor_key,
                            context: "precomputed neighbor validation".to_string(),
                        },
                    });
                };

                if simplex.periodic_vertex_offsets().is_some()
                    || neighbor_simplex.periodic_vertex_offsets().is_some()
                {
                    continue;
                }

                let neighbor_vertices = simplex_vertices.get(&neighbor_key).ok_or_else(|| {
                    TdsError::InconsistentDataStructure {
                        message: format!(
                            "Neighbor simplex {} (key {neighbor_key:?}) missing from precomputed vertex set map during neighbor validation",
                            neighbor_simplex.uuid()
                        ),
                    }
                })?;

                Self::validate_shared_facet_count(
                    simplex,
                    neighbor_simplex,
                    this_vertices,
                    neighbor_vertices,
                )?;

                let mirror_idx = Self::verified_mirror_facet_index(
                    simplex,
                    facet_idx,
                    neighbor_simplex,
                    this_vertices,
                )?;

                Self::validate_shared_facet_vertices(
                    simplex,
                    facet_idx,
                    neighbor_simplex,
                    mirror_idx,
                    this_vertices,
                    neighbor_vertices,
                )?;

                Self::validate_mutual_neighbor_back_reference(
                    simplex_key,
                    simplex,
                    facet_idx,
                    neighbor_key,
                    neighbor_simplex,
                    mirror_idx,
                )?;
            }
        }

        Ok(())
    }

    fn build_simplex_vertex_sets(&self) -> Result<SimplexVerticesMap, TdsError> {
        // Pre-compute vertex keys for all simplices to avoid repeated computation
        let mut simplex_vertices: SimplexVerticesMap =
            fast_hash_map_with_capacity(self.simplices.len());

        for simplex_key in self.simplices.keys() {
            // Use simplex_vertices to ensure all vertex keys are present
            // The error is already TdsError, so just propagate it
            let vertices = self.simplex_vertices(simplex_key)?;

            // Store the HashSet for containment checks
            let vertex_set: VertexKeySet = vertices.iter().copied().collect();
            simplex_vertices.insert(simplex_key, vertex_set);
        }

        Ok(simplex_vertices)
    }

    fn validate_shared_facet_count(
        simplex: &Simplex<V, D>,
        neighbor_simplex: &Simplex<V, D>,
        this_vertices: &VertexKeySet,
        neighbor_vertices: &VertexKeySet,
    ) -> Result<(), TdsError> {
        let shared_count = this_vertices.intersection(neighbor_vertices).count();

        if shared_count != D {
            return Err(TdsError::NotNeighbors {
                simplex1: simplex.uuid(),
                simplex2: neighbor_simplex.uuid(),
            });
        }

        Ok(())
    }

    fn verified_mirror_facet_index(
        simplex: &Simplex<V, D>,
        facet_idx: usize,
        neighbor_simplex: &Simplex<V, D>,
        this_vertices: &VertexKeySet,
    ) -> Result<usize, TdsError> {
        let mirror_idx = simplex
            .mirror_facet_index(facet_idx, neighbor_simplex)
            .ok_or_else(|| TdsError::InvalidNeighbors {
                reason: NeighborValidationError::MirrorFacetMissing {
                    simplex_uuid: simplex.uuid(),
                    facet_index: facet_idx,
                    neighbor_uuid: neighbor_simplex.uuid(),
                    context: "neighbor validation".to_string(),
                },
            })?;

        // Defensive cross-check: verify the mirror index against shared-vertex analysis.
        // This adds overhead but guards against subtle logic bugs in `mirror_facet_index()`.
        //
        // If validation ever becomes performance-sensitive, this is a good candidate to
        // gate behind a "strict validation" option/flag.
        let expected_mirror_idx =
            Self::expected_mirror_facet_index(simplex, neighbor_simplex, this_vertices)?;

        if mirror_idx != expected_mirror_idx {
            return Err(TdsError::InvalidNeighbors {
                reason: NeighborValidationError::MirrorFacetIndexMismatch {
                    simplex_uuid: simplex.uuid(),
                    facet_index: facet_idx,
                    neighbor_uuid: neighbor_simplex.uuid(),
                    observed_mirror_index: mirror_idx,
                    expected_mirror_index: expected_mirror_idx,
                },
            });
        }

        Ok(mirror_idx)
    }

    fn expected_mirror_facet_index(
        simplex: &Simplex<V, D>,
        neighbor_simplex: &Simplex<V, D>,
        this_vertices: &VertexKeySet,
    ) -> Result<usize, TdsError> {
        let mut expected_mirror_idx: Option<usize> = None;

        for (idx, &neighbor_vkey) in neighbor_simplex.vertices().iter().enumerate() {
            if !this_vertices.contains(&neighbor_vkey) {
                if expected_mirror_idx.is_some() {
                    return Err(TdsError::InvalidNeighbors {
                        reason: NeighborValidationError::MirrorFacetAmbiguous {
                            simplex_uuid: simplex.uuid(),
                            neighbor_uuid: neighbor_simplex.uuid(),
                        },
                    });
                }
                expected_mirror_idx = Some(idx);
            }
        }

        expected_mirror_idx.ok_or_else(|| TdsError::InvalidNeighbors {
            reason: NeighborValidationError::MirrorFacetDuplicateSimplices {
                simplex_uuid: simplex.uuid(),
                neighbor_uuid: neighbor_simplex.uuid(),
            },
        })
    }

    fn validate_shared_facet_vertices(
        simplex: &Simplex<V, D>,
        facet_idx: usize,
        neighbor_simplex: &Simplex<V, D>,
        mirror_idx: usize,
        this_vertices: &VertexKeySet,
        neighbor_vertices: &VertexKeySet,
    ) -> Result<(), TdsError> {
        for (idx, &vkey) in simplex.vertices().iter().enumerate() {
            if idx == facet_idx {
                continue;
            }
            if !neighbor_vertices.contains(&vkey) {
                return Err(TdsError::InvalidNeighbors {
                    reason: NeighborValidationError::SharedFacetMissingVertex {
                        side: SharedFacetMismatchSide::SourceFacet,
                        simplex_uuid: simplex.uuid(),
                        facet_index: facet_idx,
                        neighbor_uuid: neighbor_simplex.uuid(),
                        mirror_index: mirror_idx,
                        missing_vertex: vkey,
                    },
                });
            }
        }

        for (idx, &vkey) in neighbor_simplex.vertices().iter().enumerate() {
            if idx == mirror_idx {
                continue;
            }
            if !this_vertices.contains(&vkey) {
                return Err(TdsError::InvalidNeighbors {
                    reason: NeighborValidationError::SharedFacetMissingVertex {
                        side: SharedFacetMismatchSide::NeighborFacet,
                        simplex_uuid: simplex.uuid(),
                        facet_index: facet_idx,
                        neighbor_uuid: neighbor_simplex.uuid(),
                        mirror_index: mirror_idx,
                        missing_vertex: vkey,
                    },
                });
            }
        }

        Ok(())
    }

    fn validate_mutual_neighbor_back_reference(
        simplex_key: SimplexKey,
        simplex: &Simplex<V, D>,
        facet_idx: usize,
        neighbor_key: SimplexKey,
        neighbor_simplex: &Simplex<V, D>,
        mirror_idx: usize,
    ) -> Result<(), TdsError> {
        let Some(back_ref) = neighbor_simplex.neighbor_key(mirror_idx) else {
            return Err(TdsError::InvalidNeighbors {
                reason: NeighborValidationError::BackReferenceMismatch {
                    simplex_key,
                    simplex_uuid: simplex.uuid(),
                    facet_index: facet_idx,
                    neighbor_key,
                    neighbor_uuid: neighbor_simplex.uuid(),
                    mirror_index: mirror_idx,
                    observed: None,
                    context: "neighbor validation; neighbor has no neighbor buffer".to_string(),
                },
            });
        };

        if back_ref != Some(simplex_key) {
            return Err(TdsError::InvalidNeighbors {
                reason: NeighborValidationError::BackReferenceMismatch {
                    simplex_key,
                    simplex_uuid: simplex.uuid(),
                    facet_index: facet_idx,
                    neighbor_key,
                    neighbor_uuid: neighbor_simplex.uuid(),
                    mirror_index: mirror_idx,
                    observed: back_ref,
                    context: "neighbor validation".to_string(),
                },
            });
        }

        Ok(())
    }
}

// =============================================================================
// TRAIT IMPLEMENTATIONS
// =============================================================================

type SimplexUuidSortKey<const D: usize> =
    SmallBuffer<(Uuid, [i8; D]), MAX_PRACTICAL_DIMENSION_SIZE>;
type SimplexUuidSortEntry<'a, V, const D: usize> = (SimplexUuidSortKey<D>, &'a Simplex<V, D>);

/// Builds stable simplex sort keys once so equality does not hide dangling
/// vertex references or allocate sort keys repeatedly during comparison.
fn simplex_uuid_sort_entries<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
) -> Option<Vec<SimplexUuidSortEntry<'_, V, D>>>
where
    U: DataType,
    V: DataType,
{
    tds.simplices
        .values()
        .map(|simplex| {
            let offsets = simplex.periodic_vertex_offsets();
            if let Some(offsets) = offsets
                && offsets.len() != simplex.number_of_vertices()
            {
                return None;
            }

            let mut ids = SimplexUuidSortKey::<D>::new();
            for (idx, &vkey) in simplex.vertices().iter().enumerate() {
                let uuid = tds.vertex(vkey).map(Vertex::uuid)?;
                let offset = offsets.map_or([0_i8; D], |offsets| offsets[idx]);
                ids.push((uuid, offset));
            }
            ids.sort_unstable();
            Some((ids, simplex))
        })
        .collect()
}

/// Manual implementation of `PartialEq` for Tds
///
/// Two triangulation data structures are considered equal if they have:
/// - The same set of vertices (compared by coordinates)
/// - The same set of simplices (compared by vertex sets)
/// - Consistent vertex and simplex mappings
///
/// **Note:** Vertices with NaN coordinates are rejected during construction; equality assumes no NaNs.
/// The triangulation validates coordinates at construction time to ensure no NaN values are present.
///
/// Note: Buffer fields are ignored since they are transient data structures.
impl<U, V, const D: usize> PartialEq for Tds<U, V, D>
where
    U: DataType,
    V: DataType,
{
    fn eq(&self, other: &Self) -> bool {
        // Early exit if the basic counts don't match
        if self.vertices.len() != other.vertices.len()
            || self.simplices.len() != other.simplices.len()
            || self.uuid_to_vertex_key.len() != other.uuid_to_vertex_key.len()
            || self.uuid_to_simplex_key.len() != other.uuid_to_simplex_key.len()
        {
            return false;
        }

        // Compare vertices by collecting them into sorted vectors
        // We sort by coordinates to make comparison order-independent
        let mut self_vertices: Vec<_> = self.vertices.values().collect();
        let mut other_vertices: Vec<_> = other.vertices.values().collect();

        // Sort vertices by their coordinates for consistent comparison
        // f64-backed vertices provide PartialOrd; NaN validation occurs at construction time
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

        // Compare simplices using Simplex::eq_by_vertices() which uses
        // Vertex::PartialEq. Missing vertex references make the structures
        // unequal rather than being silently dropped from sort keys.
        let (Some(mut self_simplices), Some(mut other_simplices)) = (
            simplex_uuid_sort_entries(self),
            simplex_uuid_sort_entries(other),
        ) else {
            return false;
        };

        self_simplices.sort_by(|(a_ids, _), (b_ids, _)| a_ids.cmp(b_ids));
        other_simplices.sort_by(|(a_ids, _), (b_ids, _)| a_ids.cmp(b_ids));

        // Compare sorted simplex lists using Simplex::eq_by_vertices
        if self_simplices.len() != other_simplices.len() {
            return false;
        }

        for ((_, self_simplex), (_, other_simplex)) in
            self_simplices.iter().zip(other_simplices.iter())
        {
            if !self_simplex.eq_by_vertices(self, other_simplex, other) {
                return false;
            }
        }

        // If we get here, the triangulations have the same structure
        // UUID→Key maps are derived from the vertices/simplices, so if those match, the maps should be consistent
        // (We don't need to compare the maps directly since they're just indexing structures)

        true
    }
}

/// Eq implementation for Tds
///
/// This is a marker trait implementation that relies on the `PartialEq` implementation.
/// Since Tds represents a well-defined mathematical structure (triangulation),
/// the `PartialEq` relation is indeed an equivalence relation.
impl<U, V, const D: usize> Eq for Tds<U, V, D>
where
    U: DataType,
    V: DataType,
{
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DelaunayTriangulation;
    use crate::builder::DelaunayTriangulationBuilder;
    use crate::core::algorithms::flips::DelaunayRepairError;
    use crate::core::algorithms::incremental_insertion::InsertionError;
    use crate::core::facet::FacetError;
    use crate::core::simplex::Simplex;
    use crate::core::util::uuid::UuidValidationError;
    use crate::core::validation::TriangulationValidationError;
    use crate::geometry::point::Point;
    use crate::repair::DelaunayRepairOperation;
    use crate::topology::characteristics::euler::TopologyClassification;
    use crate::validation::DelaunayTriangulationValidationError;
    use slotmap::KeyData;
    use std::assert_matches;
    use std::sync::Arc;

    // =============================================================================
    // TEST HELPER FUNCTIONS
    // =============================================================================

    fn vertex_with_uuid<U, const D: usize>(
        point: Point<D>,
        uuid: Uuid,
        data: Option<U>,
    ) -> Vertex<U, D>
    where
        U: DataType,
    {
        Vertex::try_new_with_uuid(point, uuid, data).expect("Failed to build vertex")
    }

    fn initial_simplex_vertices_3d() -> [Vertex<(), 3>; 4] {
        [
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ]
    }

    fn assert_tds_error_kind(source: &TdsError, expected: TdsErrorKind) {
        assert_eq!(TdsErrorKind::from(source), expected);
    }

    #[test]
    fn tds_error_kind_from_error_preserves_validation_variants() {
        let simplex_key = SimplexKey::from(KeyData::from_ffi(1));
        let other_simplex_key = SimplexKey::from(KeyData::from_ffi(2));
        let vertex_key = VertexKey::from(KeyData::from_ffi(3));
        let uuid = Uuid::new_v4();

        assert_tds_error_kind(
            &TdsError::InvalidVertex {
                vertex_id: uuid,
                source: VertexValidationError::InvalidUuid {
                    source: UuidValidationError::NilUuid,
                },
            },
            TdsErrorKind::InvalidVertex,
        );
        assert_tds_error_kind(
            &TdsError::InvalidSimplex {
                simplex_id: uuid,
                source: SimplexValidationError::DuplicateVertices,
            },
            TdsErrorKind::InvalidSimplex,
        );
        assert_tds_error_kind(
            &TdsError::InvalidNeighbors {
                reason: NeighborValidationError::Other {
                    message: "neighbor invariant failed".to_string(),
                },
            },
            TdsErrorKind::InvalidNeighbors,
        );
        assert_tds_error_kind(
            &TdsError::OrientationViolation {
                simplex1_key: simplex_key,
                simplex1_uuid: uuid,
                simplex2_key: other_simplex_key,
                simplex2_uuid: Uuid::new_v4(),
                simplex1_facet_index: 0,
                simplex2_facet_index: 1,
                facet_vertices: vec![vertex_key],
                simplex2_facet_vertices: vec![vertex_key],
                observed_odd_permutation: false,
                expected_odd_permutation: true,
            },
            TdsErrorKind::OrientationViolation,
        );
        assert_tds_error_kind(
            &TdsError::Geometric(GeometricError::DegenerateOrientation {
                message: "zero determinant".to_string(),
            }),
            TdsErrorKind::Geometric,
        );
        assert_tds_error_kind(
            &TdsError::FacetError(FacetError::InvalidFacetIndex {
                index: 4,
                facet_count: 4,
            }),
            TdsErrorKind::FacetError,
        );
        assert_tds_error_kind(
            &TdsError::DuplicateCoordinatesInSimplex {
                simplex_id: uuid,
                message: "two vertices share coordinates".to_string(),
            },
            TdsErrorKind::DuplicateCoordinatesInSimplex,
        );
    }

    #[test]
    fn tds_error_kind_from_error_preserves_lookup_and_operation_variants() {
        let simplex_key = SimplexKey::from(KeyData::from_ffi(1));
        let vertex_key = VertexKey::from(KeyData::from_ffi(3));
        let uuid = Uuid::new_v4();

        assert_tds_error_kind(
            &TdsError::DuplicateSimplices {
                message: "duplicate simplex vertex set".to_string(),
            },
            TdsErrorKind::DuplicateSimplices,
        );
        assert_tds_error_kind(
            &TdsError::FacetSharingViolation {
                facet_key: 42,
                existing_incident_count: 2,
                attempted_incident_count: 3,
                max_incident_count: 2,
                candidate_simplex_uuid: uuid,
                candidate_facet_index: 1,
            },
            TdsErrorKind::FacetSharingViolation,
        );
        assert_tds_error_kind(
            &TdsError::FailedToCreateSimplex {
                message: "simplex validation failed".to_string(),
            },
            TdsErrorKind::FailedToCreateSimplex,
        );
        assert_tds_error_kind(
            &TdsError::NotNeighbors {
                simplex1: uuid,
                simplex2: Uuid::new_v4(),
            },
            TdsErrorKind::NotNeighbors,
        );
        assert_tds_error_kind(
            &TdsError::MappingInconsistency {
                entity: EntityKind::Simplex,
                message: "uuid mapping was stale".to_string(),
            },
            TdsErrorKind::MappingInconsistency,
        );
        assert_tds_error_kind(
            &TdsError::VertexKeyRetrievalFailed {
                simplex_id: uuid,
                message: "simplex vertices unavailable".to_string(),
            },
            TdsErrorKind::VertexKeyRetrievalFailed,
        );
        assert_tds_error_kind(
            &TdsError::SimplexNotFound {
                simplex_key,
                context: "simplex lookup".to_string(),
            },
            TdsErrorKind::SimplexNotFound,
        );
        assert_tds_error_kind(
            &TdsError::VertexNotFound {
                vertex_key,
                context: "vertex lookup".to_string(),
            },
            TdsErrorKind::VertexNotFound,
        );
        assert_tds_error_kind(
            &TdsError::DimensionMismatch {
                expected: 4,
                actual: 3,
                context: "simplex arity".to_string(),
            },
            TdsErrorKind::DimensionMismatch,
        );
        assert_tds_error_kind(
            &TdsError::IndexOutOfBounds {
                index: 4,
                bound: 4,
                context: "facet index".to_string(),
            },
            TdsErrorKind::IndexOutOfBounds,
        );
        assert_tds_error_kind(
            &TdsError::InconsistentDataStructure {
                message: "dangling neighbor".to_string(),
            },
            TdsErrorKind::InconsistentDataStructure,
        );
    }

    #[test]
    fn triangulation_validation_error_kind_from_error_preserves_all_variants() {
        let vertex_key = VertexKey::from(KeyData::from_ffi(3));
        let cases = [
            (
                TriangulationValidationError::ManifoldFacetMultiplicity {
                    facet_key: 0xabc,
                    simplex_count: 3,
                },
                TriangulationValidationErrorKind::ManifoldFacetMultiplicity,
            ),
            (
                TriangulationValidationError::BoundaryRidgeMultiplicity {
                    ridge_key: 0xdef,
                    boundary_facet_count: 3,
                },
                TriangulationValidationErrorKind::BoundaryRidgeMultiplicity,
            ),
            (
                TriangulationValidationError::RidgeLinkNotManifold {
                    ridge_key: 0x123,
                    link_vertex_count: 4,
                    link_edge_count: 2,
                    max_degree: 3,
                    degree_one_vertices: 1,
                    connected: false,
                },
                TriangulationValidationErrorKind::RidgeLinkNotManifold,
            ),
            (
                TriangulationValidationError::VertexLinkNotManifold {
                    vertex_key,
                    link_vertex_count: 4,
                    link_simplex_count: 2,
                    boundary_facet_count: 1,
                    max_degree: 3,
                    connected: false,
                    interior_vertex: true,
                },
                TriangulationValidationErrorKind::VertexLinkNotManifold,
            ),
            (
                TriangulationValidationError::EulerCharacteristicMismatch {
                    computed: 0,
                    expected: 1,
                    classification: TopologyClassification::Ball(3),
                },
                TriangulationValidationErrorKind::EulerCharacteristicMismatch,
            ),
            (
                TriangulationValidationError::IsolatedVertex {
                    vertex_key,
                    vertex_uuid: Uuid::new_v4(),
                },
                TriangulationValidationErrorKind::IsolatedVertex,
            ),
            (
                TriangulationValidationError::Disconnected { simplex_count: 2 },
                TriangulationValidationErrorKind::Disconnected,
            ),
            (
                TriangulationValidationError::OrientationPromotionNonConvergence {
                    residual_count: 1,
                    sampled: vec![SimplexKey::from(KeyData::from_ffi(4))],
                },
                TriangulationValidationErrorKind::OrientationPromotionNonConvergence,
            ),
        ];

        for (source, expected) in cases {
            assert_eq!(TriangulationValidationErrorKind::from(&source), expected);
        }
    }

    #[test]
    fn delaunay_validation_error_kind_from_error_preserves_all_variants() {
        let cases = [
            (
                DelaunayTriangulationValidationError::from(TdsError::InconsistentDataStructure {
                    message: "dangling simplex".to_string(),
                }),
                DelaunayValidationErrorKind::Tds,
            ),
            (
                DelaunayTriangulationValidationError::from(
                    TriangulationValidationError::Disconnected { simplex_count: 2 },
                ),
                DelaunayValidationErrorKind::Triangulation,
            ),
            (
                DelaunayTriangulationValidationError::VerificationFailed {
                    message: "non-Delaunay facet".to_string(),
                },
                DelaunayValidationErrorKind::VerificationFailed,
            ),
            (
                DelaunayTriangulationValidationError::RepairFailed {
                    message: "repair did not converge".to_string(),
                },
                DelaunayValidationErrorKind::RepairFailed,
            ),
            (
                DelaunayTriangulationValidationError::RepairOperationFailed {
                    operation: DelaunayRepairOperation::VertexRemoval,
                    source: Box::new(DelaunayRepairError::PostconditionFailed {
                        message: "remaining violation".to_string(),
                    }),
                },
                DelaunayValidationErrorKind::RepairOperationFailed,
            ),
        ];

        for (source, expected) in cases {
            assert_eq!(DelaunayValidationErrorKind::from(&source), expected);
        }
    }

    #[test]
    fn test_facet_key_for_simplex_facet_maps_periodic_derivation_errors() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        tds.simplex_mut(simplex_key)
            .unwrap()
            .set_periodic_vertex_offsets(vec![[-128_i8, 0_i8], [127_i8, 0_i8], [0_i8, 0_i8]])
            .unwrap();

        let err = tds.facet_key_for_simplex_facet(simplex_key, 2).unwrap_err();
        assert_matches!(
            err,
            TdsError::InconsistentDataStructure { message }
                if message.contains("Failed to derive periodic facet key")
                    && message.contains("facet 2")
        );
    }

    #[test]
    fn test_facet_vertex_identities_anchor_uses_lexicographic_key_offset() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();

        // Deliberately corrupt for regression coverage: duplicate v_a with a smaller
        // periodic lift so anchor selection must use the (key, offset) tie-breaker.
        {
            let simplex = tds.simplex_mut(simplex_key).unwrap();
            simplex.push_vertex_key(v_a);
            simplex
                .set_periodic_vertex_offsets(vec![[5, 0], [9, 0], [8, 0], [1, 0]])
                .unwrap();
        }

        let simplex = tds.simplex(simplex_key).unwrap();
        let identities =
            Tds::<(), (), 2>::facet_vertex_identities_in_simplex_order(simplex, 2).unwrap();
        assert_eq!(identities.len(), 3);

        let mut offsets_for_a: Vec<[i16; 2]> = identities
            .iter()
            .filter_map(|(vkey, offset)| (*vkey == v_a).then_some(*offset))
            .collect();
        offsets_for_a.sort_unstable();
        assert_eq!(offsets_for_a, vec![[0, 0], [4, 0]]);

        let b_offset = identities
            .iter()
            .find_map(|(vkey, offset)| (*vkey == v_b).then_some(*offset))
            .expect("identity list should include v_b");
        assert_eq!(b_offset, [8, 0]);
    }

    #[test]
    fn test_facet_permutation_parity_derives_coherent_and_incoherent_cases() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();

        // Shared edge for facet_idx=2 is (v_a, v_b).
        let simplex: Simplex<(), 2> =
            Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap();

        // Coherent case: neighbor lists the shared edge in opposite order.
        let coherent_neighbor: Simplex<(), 2> =
            Simplex::try_new_with_data(vec![v_b, v_a, v_d], None).unwrap();
        let coherent_mirror_idx = simplex.mirror_facet_index(2, &coherent_neighbor).unwrap();
        let (currently_coherent, observed_odd_permutation, expected_odd_permutation) =
            Tds::<(), (), 2>::facet_permutation_parity(
                &simplex,
                2,
                &coherent_neighbor,
                coherent_mirror_idx,
            )
            .unwrap();
        assert!(currently_coherent);
        assert!(observed_odd_permutation);
        assert!(expected_odd_permutation);

        // Incoherent case: neighbor lists the shared edge in the same order.
        let incoherent_neighbor: Simplex<(), 2> =
            Simplex::try_new_with_data(vec![v_a, v_b, v_d], None).unwrap();
        let incoherent_mirror_idx = simplex.mirror_facet_index(2, &incoherent_neighbor).unwrap();
        let (currently_coherent, observed_odd_permutation, expected_odd_permutation) =
            Tds::<(), (), 2>::facet_permutation_parity(
                &simplex,
                2,
                &incoherent_neighbor,
                incoherent_mirror_idx,
            )
            .unwrap();
        assert!(!currently_coherent);
        assert!(!observed_odd_permutation);
        assert!(expected_odd_permutation);
    }

    #[test]
    fn test_facet_permutation_parity_smoke_4d() {
        let mut tds: Tds<(), (), 4> = Tds::empty();
        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_e = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v_f = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0, 1.0, 1.0]).unwrap(),
            )
            .unwrap();

        // Shared 3-face for facet_idx=4 is [v_a, v_b, v_c, v_d].
        let simplex: Simplex<(), 4> =
            Simplex::try_new_with_data(vec![v_a, v_b, v_c, v_d, v_e], None).unwrap();

        // Coherent case: odd permutation of the shared face.
        let coherent_neighbor: Simplex<(), 4> =
            Simplex::try_new_with_data(vec![v_b, v_a, v_c, v_d, v_f], None).unwrap();
        let coherent_mirror_idx = simplex.mirror_facet_index(4, &coherent_neighbor).unwrap();
        let (currently_coherent, observed_odd_permutation, expected_odd_permutation) =
            Tds::<(), (), 4>::facet_permutation_parity(
                &simplex,
                4,
                &coherent_neighbor,
                coherent_mirror_idx,
            )
            .unwrap();
        assert!(currently_coherent);
        assert!(observed_odd_permutation);
        assert!(expected_odd_permutation);

        // Incoherent case: identity ordering of the shared face.
        let incoherent_neighbor: Simplex<(), 4> =
            Simplex::try_new_with_data(vec![v_a, v_b, v_c, v_d, v_f], None).unwrap();
        let incoherent_mirror_idx = simplex.mirror_facet_index(4, &incoherent_neighbor).unwrap();
        let (currently_coherent, observed_odd_permutation, expected_odd_permutation) =
            Tds::<(), (), 4>::facet_permutation_parity(
                &simplex,
                4,
                &incoherent_neighbor,
                incoherent_mirror_idx,
            )
            .unwrap();
        assert!(!currently_coherent);
        assert!(!observed_odd_permutation);
        assert!(expected_odd_permutation);
    }

    // =============================================================================
    // VERTEX ADDITION TESTS - CONSOLIDATED
    // =============================================================================

    #[test]
    fn test_add_vertex_basic_insertion_succeeds() {
        let initial_vertices = initial_simplex_vertices_3d();
        let mut dt = DelaunayTriangulation::try_new(&initial_vertices).unwrap();

        let vertex = crate::core::vertex::Vertex::<(), _>::try_new([1.0, 2.0, 3.0]).unwrap();
        let result = dt.insert(vertex);

        assert!(result.is_ok(), "Basic vertex addition should succeed");
        assert_eq!(dt.number_of_vertices(), 5);
    }

    #[test]
    fn test_add_vertex_duplicate_coordinates_rejected() {
        let initial_vertices = initial_simplex_vertices_3d();
        let mut dt = DelaunayTriangulation::try_new(&initial_vertices).unwrap();

        let vertex = crate::core::vertex::Vertex::<(), _>::try_new([1.0, 2.0, 3.0]).unwrap();
        let duplicate = crate::core::vertex::Vertex::<(), _>::try_new([1.0, 2.0, 3.0]).unwrap();
        dt.insert(vertex).unwrap();

        // Same coordinates again (distinct UUID, constructed via Vertex smart constructors)
        let result = dt.insert(duplicate);
        assert!(
            matches!(result, Err(InsertionError::DuplicateCoordinates { .. })),
            "insert() should reject duplicate coordinates created via Vertex::try_new (before UUID), got: {result:?}"
        );
    }

    #[test]
    fn test_add_vertex_duplicate_uuid_rejected() {
        let initial_vertices = initial_simplex_vertices_3d();
        let mut dt = DelaunayTriangulation::try_new(&initial_vertices).unwrap();

        let vertex1 = crate::core::vertex::Vertex::<(), _>::try_new([1.0, 2.0, 3.0]).unwrap();
        let uuid1 = vertex1.uuid();
        dt.insert(vertex1).unwrap();

        let vertex2 = vertex_with_uuid(
            Point::try_new([4.0, 5.0, 6.0]).expect("finite point coordinates"),
            uuid1,
            None,
        );
        let result = dt.insert(vertex2);
        assert!(
            matches!(
                result,
                Err(InsertionError::DuplicateUuid {
                    entity: EntityKind::Vertex,
                    ..
                })
            ),
            "Same UUID with different coordinates should fail with DuplicateUuid"
        );
    }

    #[test]
    fn test_add_vertex_increases_counts_and_leaves_tds_valid() {
        let initial_vertices = initial_simplex_vertices_3d();
        let mut dt = DelaunayTriangulation::try_new(&initial_vertices).unwrap();
        let initial_simplex_count = dt.number_of_simplices();

        let new_vertex = crate::core::vertex::Vertex::<(), _>::try_new([0.5, 0.5, 0.5]).unwrap();
        dt.insert(new_vertex).unwrap();

        assert_eq!(dt.number_of_vertices(), 5);
        assert!(
            dt.number_of_simplices() >= initial_simplex_count,
            "Simplex count should not decrease"
        );
        assert!(
            dt.as_triangulation().tds.is_valid().is_ok(),
            "TDS should remain valid"
        );
    }

    #[test]
    fn test_add_vertex_is_accessible_by_uuid_and_coordinates() {
        let initial_vertices = initial_simplex_vertices_3d();
        let mut dt = DelaunayTriangulation::try_new(&initial_vertices).unwrap();

        let vertex = crate::core::vertex::Vertex::<(), _>::try_new([1.0, 2.0, 3.0]).unwrap();
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
            .vertex(vertex_key.unwrap())
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
        // Create a triangulation with multiple simplices
        let vertices = [
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.5, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.5, 1.0]).unwrap(),
        ];
        let mut dt = DelaunayTriangulation::try_new(&vertices).unwrap();

        // Verify initial state
        let initial_vertices = dt.number_of_vertices();
        let initial_simplices = dt.number_of_simplices();
        assert_eq!(initial_vertices, 4);
        assert!(initial_simplices > 0);

        // Get a vertex to remove (not a corner to ensure we have remaining simplices)
        let (vertex_key, vertex_ref) = dt.vertices().next().unwrap();
        let vertex_uuid = vertex_ref.uuid();

        // Remove the vertex and all simplices containing it
        let simplices_removed = dt.remove_vertex(vertex_key).unwrap();

        // Verify the vertex was removed
        assert!(
            dt.as_triangulation()
                .tds
                .vertex_key_from_uuid(&vertex_uuid)
                .is_none(),
            "Vertex should be removed from TDS"
        );
        assert!(
            simplices_removed > 0,
            "At least one simplex should have been removed"
        );
        assert_eq!(
            dt.number_of_vertices(),
            initial_vertices - 1,
            "Vertex count should decrease by 1"
        );
        assert!(
            dt.number_of_simplices() < initial_simplices,
            "Simplex count should decrease"
        );

        // CRITICAL: Verify that no dangling neighbor references exist
        // This is the key test for the bug fix
        for (simplex_key, simplex) in dt.simplices() {
            if let Some(neighbors) = simplex.neighbors() {
                for (i, neighbor_opt) in neighbors.enumerate() {
                    if let Some(neighbor_key) = neighbor_opt {
                        assert!(
                            dt.as_triangulation()
                                .tds
                                .simplices
                                .contains_key(neighbor_key),
                            "Simplex {simplex_key:?} has dangling neighbor reference at index {i}: {neighbor_key:?}"
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
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let mut dt = DelaunayTriangulation::try_new(&vertices).unwrap();

        // Use a key that was never added
        let nonexistent_key = VertexKey::from(KeyData::from_ffi(u64::MAX));

        let initial_vertices = dt.number_of_vertices();
        let initial_simplices = dt.number_of_simplices();

        // Remove should return 0 (no simplices removed)
        let simplices_removed = dt.remove_vertex(nonexistent_key).unwrap();

        assert_eq!(simplices_removed, 0, "No simplices should be removed");
        assert_eq!(
            dt.number_of_vertices(),
            initial_vertices,
            "Vertex count should not change"
        );
        assert_eq!(
            dt.number_of_simplices(),
            initial_simplices,
            "Simplex count should not change"
        );

        println!("✓ remove_vertex handles nonexistent vertex correctly");
    }

    #[test]
    fn test_remove_vertex_stale_key_is_idempotent() {
        // With the VertexKey API, callers can hold a key after removal and reuse it.
        // Double-remove must be a no-op returning Ok(0).
        let vertices = [
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        let vertex_key = dt.vertices().next().unwrap().0;

        // First removal succeeds.
        let simplices_removed = dt.remove_vertex(vertex_key).unwrap();
        assert!(simplices_removed > 0);
        let vertices_after = dt.number_of_vertices();
        let simplices_after = dt.number_of_simplices();

        // Second removal with the same (now stale) key is a no-op.
        let simplices_removed_again = dt.remove_vertex(vertex_key).unwrap();
        assert_eq!(
            simplices_removed_again, 0,
            "Stale key should remove nothing"
        );
        assert_eq!(dt.number_of_vertices(), vertices_after);
        assert_eq!(dt.number_of_simplices(), simplices_after);
    }

    #[test]
    fn test_remove_vertex_multiple_dimensions() {
        // Test remove_vertex in different dimensions

        // 2D test
        {
            let vertices_2d = [
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            ];
            let mut dt_2d: DelaunayTriangulation<_, (), (), 2> =
                DelaunayTriangulation::try_new(&vertices_2d).unwrap();
            let vertex_key = dt_2d.vertices().next().unwrap().0;
            let simplices_removed = dt_2d.remove_vertex(vertex_key).unwrap();
            assert!(simplices_removed > 0);
            assert!(dt_2d.as_triangulation().tds.is_valid().is_ok());
        }

        // 3D test
        {
            let vertices_3d = [
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
                crate::core::vertex::Vertex::<(), _>::try_new([0.2, 0.2, 0.2]).unwrap(),
            ];
            let mut dt_3d: DelaunayTriangulation<_, (), (), 3> =
                DelaunayTriangulation::try_new(&vertices_3d).unwrap();
            let vertex_key = dt_3d
                .vertices()
                .find(|(_, vertex)| {
                    let coords = vertex.point().coords();
                    coords
                        .iter()
                        .zip([0.2, 0.2, 0.2])
                        .all(|(coord, expected)| (*coord - expected).abs() < 1e-12)
                })
                .unwrap()
                .0;
            let simplices_removed = dt_3d.remove_vertex(vertex_key).unwrap();
            assert!(simplices_removed > 0);
            assert!(dt_3d.as_triangulation().tds.is_valid().is_ok());
        }

        // 4D test
        {
            let vertices_4d = [
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0]).unwrap(),
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0, 0.0]).unwrap(),
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0, 0.0]).unwrap(),
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0, 0.0]).unwrap(),
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 1.0]).unwrap(),
                crate::core::vertex::Vertex::<(), _>::try_new([0.2, 0.2, 0.2, 0.2]).unwrap(),
            ];
            let mut dt_4d: DelaunayTriangulation<_, (), (), 4> =
                DelaunayTriangulation::try_new(&vertices_4d).unwrap();
            let vertex_key = dt_4d
                .vertices()
                .find(|(_, vertex)| {
                    let coords = vertex.point().coords();
                    coords
                        .iter()
                        .zip([0.2, 0.2, 0.2, 0.2])
                        .all(|(coord, expected)| (*coord - expected).abs() < 1e-12)
                })
                .unwrap()
                .0;
            let simplices_removed = dt_4d.remove_vertex(vertex_key).unwrap();
            assert!(simplices_removed > 0);
            assert!(dt_4d.as_triangulation().tds.is_valid().is_ok());
        }

        println!("✓ remove_vertex works correctly in multiple dimensions");
    }

    #[test]
    fn test_remove_vertex_no_dangling_references() {
        // Test that after removing a vertex:
        // 1. No simplices contain the deleted vertex
        // 2. No vertices have incident_simplex pointing to a removed simplex
        // 3. All remaining incident_simplex pointers are valid

        let vertices = [
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            // Interior vertex to remove; offset from circumcenter to avoid degenerate configuration
            crate::core::vertex::Vertex::<(), _>::try_new([0.2, 0.2, 0.2]).unwrap(),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

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

        // Remove the vertex
        let simplices_removed = dt.remove_vertex(removed_vertex_key).unwrap();
        assert!(
            simplices_removed > 0,
            "Should have removed at least one simplex"
        );

        // CRITICAL CHECK 1: No simplices should contain the deleted vertex
        for (simplex_key, simplex) in dt.simplices() {
            for &vk in simplex.vertices() {
                assert_ne!(
                    vk, removed_vertex_key,
                    "Simplex {simplex_key:?} still references deleted vertex {removed_vertex_key:?}"
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
                .vertex(removed_vertex_key)
                .is_none(),
            "Deleted vertex key should not exist in storage"
        );

        // CRITICAL CHECK 3: All remaining vertices should have valid incident_simplex pointers
        for (vertex_key, vertex) in dt.vertices() {
            if let Some(incident_simplex_key) = vertex.incident_simplex() {
                assert!(
                    dt.as_triangulation()
                        .tds
                        .simplices
                        .contains_key(incident_simplex_key),
                    "Vertex {vertex_key:?} has dangling incident_simplex pointer to {incident_simplex_key:?}"
                );

                // Verify the incident simplex actually contains this vertex
                let incident_simplex = dt
                    .as_triangulation()
                    .tds
                    .simplex(incident_simplex_key)
                    .unwrap();
                assert!(
                    incident_simplex.contains_vertex(vertex_key),
                    "Vertex {vertex_key:?} incident_simplex {incident_simplex_key:?} does not contain the vertex"
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
    fn test_find_simplices_containing_vertex() {
        // Test the helper function that finds all simplices containing a specific vertex
        let vertices = [
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.5, 0.5, 0.5]).unwrap(), // Interior vertex
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        // Pick a vertex known to belong to at least one simplex (from an existing simplex)
        let first_simplex_key = dt.simplices().next().unwrap().0;
        let some_vertex_key = dt
            .as_triangulation()
            .tds
            .simplex_vertices(first_simplex_key)
            .unwrap()[0];
        let simplices_with_vertex = dt
            .as_triangulation()
            .tds
            .find_simplices_containing_vertex(some_vertex_key);

        // Verify the result
        assert!(
            !simplices_with_vertex.is_empty(),
            "Vertex should be in at least one simplex"
        );

        // Verify all returned simplices actually contain the vertex
        for &simplex_key in &simplices_with_vertex {
            let simplex = dt.as_triangulation().tds.simplex(simplex_key).unwrap();
            assert!(
                simplex.contains_vertex(some_vertex_key),
                "Simplex {simplex_key:?} should contain vertex {some_vertex_key:?}"
            );
        }

        // Verify we found ALL simplices containing this vertex
        let expected_count = dt
            .simplices()
            .filter(|(_, simplex)| simplex.contains_vertex(some_vertex_key))
            .count();
        assert_eq!(
            simplices_with_vertex.len(),
            expected_count,
            "Should find all simplices containing the vertex"
        );

        // Test finding simplices for another vertex
        let another_vertex_key = dt.vertices().nth(1).map(|(k, _)| k).unwrap();
        let simplices_with_another = dt
            .as_triangulation()
            .tds
            .find_simplices_containing_vertex(another_vertex_key);
        assert!(
            !simplices_with_another.is_empty(),
            "Another vertex should also be in at least one simplex"
        );

        println!(
            "✓ find_simplices_containing_vertex correctly identifies all simplices containing a vertex"
        );
    }

    #[test]
    fn test_assign_neighbors_errors_on_missing_vertex_key() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex_key = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, c], None).unwrap())
            .unwrap();

        // Corrupt the simplex by inserting a vertex key that doesn't exist in the TDS.
        let invalid_vkey = VertexKey::from(KeyData::from_ffi(u64::MAX));
        tds.simplex_mut(simplex_key)
            .unwrap()
            .push_vertex_key(invalid_vkey);

        let err = tds.assign_neighbors().unwrap_err();
        assert_matches!(err, TdsError::VertexKeyRetrievalFailed { .. });
    }

    #[test]
    fn test_assign_neighbors_errors_on_non_manifold_facet_sharing() {
        // Three triangles sharing the same edge (v_a,v_b) is non-manifold in 2D.
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, -1.0]).unwrap(),
            )
            .unwrap();
        let v_e = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([2.0, 0.0]).unwrap(),
            )
            .unwrap();

        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
        )
        .unwrap();
        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v_a, v_b, v_d], None).unwrap(),
        )
        .unwrap();
        tds.insert_simplex_bypassing_topology_checks_for_test(
            Simplex::try_new_with_data(vec![v_a, v_b, v_e], None).unwrap(),
        )
        .unwrap();

        let err = tds.assign_neighbors().unwrap_err();
        assert_matches!(err, TdsError::InconsistentDataStructure { .. });
    }

    #[test]
    fn test_remove_simplices_by_keys_empty_is_noop() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let gen_before = tds.generation();
        assert_eq!(tds.remove_simplices_by_keys(&[]), 0);
        assert_eq!(tds.generation(), gen_before);
    }

    #[test]
    fn test_remove_simplices_by_keys_clears_neighbor_pointers() {
        // Two triangles sharing an edge.
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex1 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, c], None).unwrap())
            .unwrap();
        let simplex2 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![b, d, c], None).unwrap())
            .unwrap();

        // Build neighbor pointers based on facet sharing.
        tds.assign_neighbors().unwrap();

        // Sanity: at least one neighbor pointer should exist before removal.
        assert!(
            tds.simplex(simplex1)
                .unwrap()
                .neighbors()
                .is_some_and(|mut n| n.any(|neighbor| neighbor.is_some()))
        );

        let gen_before = tds.generation();
        assert_eq!(tds.remove_simplices_by_keys(&[simplex2]), 1);
        assert_eq!(tds.generation(), gen_before + 1);

        // All remaining neighbor pointers must not reference the removed simplex.
        for (_, simplex) in tds.simplices() {
            if let Some(neighbors) = simplex.neighbors() {
                for neighbor_opt in neighbors {
                    assert_ne!(neighbor_opt, Some(simplex2));
                }
            }
        }
    }

    #[test]
    fn test_remove_simplices_by_keys_repairs_incident_simplices_for_affected_vertices() {
        // Two triangles sharing an edge (B,C). Remove one and ensure incident_simplex pointers are
        // updated without requiring a full assign_incident_simplices() rebuild.
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex1 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, c], None).unwrap())
            .unwrap();
        let simplex2 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![b, d, c], None).unwrap())
            .unwrap();

        tds.assign_neighbors().unwrap();

        // Force deterministic incident_simplex pointers that require repair.
        tds.vertex_mut(a)
            .unwrap()
            .set_incident_simplex(Some(simplex1));
        tds.vertex_mut(b)
            .unwrap()
            .set_incident_simplex(Some(simplex1));
        tds.vertex_mut(c)
            .unwrap()
            .set_incident_simplex(Some(simplex1));
        tds.vertex_mut(d)
            .unwrap()
            .set_incident_simplex(Some(simplex2));

        assert_eq!(tds.remove_simplices_by_keys(&[simplex1]), 1);

        // A is now isolated (no remaining simplices contain it) => incident_simplex must be None.
        assert!(tds.vertex(a).unwrap().incident_simplex().is_none());

        // B, C, D are still in simplex2 => their incident_simplex must be valid and contain them.
        for vk in [b, c, d] {
            let incident = tds
                .vertex(vk)
                .unwrap()
                .incident_simplex()
                .expect("vertex should have an incident simplex after repair");
            assert!(tds.simplices.contains_key(incident));
            let simplex = tds.simplex(incident).unwrap();
            assert!(simplex.contains_vertex(vk));
        }

        // With remaining simplices, isolated vertices are allowed at the TDS structural level.
        assert!(tds.is_valid().is_ok());

        // Neighbor pointers in the surviving simplex must not reference the removed simplex.
        let simplex2_ref = tds.simplex(simplex2).unwrap();
        if let Some(mut neighbors) = simplex2_ref.neighbors() {
            assert!(neighbors.all(|n| n != Some(simplex1)));
        }
    }

    #[test]
    fn test_tds_remove_vertex_returns_zero_when_vertex_not_in_mapping() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let missing_key = VertexKey::from(KeyData::from_ffi(u64::MAX));
        assert_eq!(tds.remove_vertex(missing_key).unwrap(), 0);
    }

    #[test]
    fn test_remove_isolated_vertex_noop_on_missing_key() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let missing = VertexKey::from(KeyData::from_ffi(u64::MAX));
        let gen_before = tds.generation();
        tds.remove_isolated_vertex(missing);
        assert_eq!(tds.generation(), gen_before, "No mutation expected");
    }

    #[test]
    fn test_remove_isolated_vertex_noop_when_incident_simplex_set() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let vk = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();

        // Manually set an incident simplex so the vertex appears non-isolated.
        let fake_simplex_key = SimplexKey::from(KeyData::from_ffi(1));
        tds.vertex_mut(vk)
            .unwrap()
            .set_incident_simplex(Some(fake_simplex_key));

        let gen_before = tds.generation();
        tds.remove_isolated_vertex(vk);
        assert_eq!(
            tds.generation(),
            gen_before,
            "Non-isolated vertex should not be removed"
        );
        assert!(tds.vertex(vk).is_some());
    }

    #[test]
    fn test_remove_isolated_vertex_removes_truly_isolated() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let vk = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 2.0]).unwrap(),
            )
            .unwrap();
        let uuid = tds.vertex(vk).unwrap().uuid();

        // No incident simplex set → truly isolated.
        assert!(tds.vertex(vk).unwrap().incident_simplex().is_none());

        let gen_before = tds.generation();
        tds.remove_isolated_vertex(vk);
        assert!(tds.generation() > gen_before);
        assert!(tds.vertex(vk).is_none(), "Vertex should be removed");
        assert!(
            tds.vertex_key_from_uuid(&uuid).is_none(),
            "UUID mapping should be removed"
        );
    }

    #[test]
    fn test_tds_remove_vertex_repairs_neighbors_and_incident_simplices_incrementally() {
        // Two triangles sharing an edge (east,north). Remove the origin (only in simplex1) and ensure:
        // - simplex1 is removed
        // - neighbor back-references in simplex2 are cleared
        // - incident_simplex pointers remain valid for remaining vertices without a full rebuild
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let origin_key = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let east_key = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let north_key = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let diagonal_key = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![origin_key, east_key, north_key], None).unwrap(),
            )
            .unwrap();
        let simplex2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![east_key, diagonal_key, north_key], None).unwrap(),
            )
            .unwrap();

        tds.assign_neighbors().unwrap();

        // Seed incident_simplex pointers:
        // - ORIGIN points at simplex1 so star discovery can use the neighbor-walk fast path.
        // - EAST/NORTH point at simplex1 so removal must repair them to simplex2.
        // - DIAGONAL points at simplex2 and should remain valid.
        tds.vertex_mut(origin_key)
            .unwrap()
            .set_incident_simplex(Some(simplex1));
        tds.vertex_mut(east_key)
            .unwrap()
            .set_incident_simplex(Some(simplex1));
        tds.vertex_mut(north_key)
            .unwrap()
            .set_incident_simplex(Some(simplex1));
        tds.vertex_mut(diagonal_key)
            .unwrap()
            .set_incident_simplex(Some(simplex2));

        let origin_uuid = tds.vertex(origin_key).unwrap().uuid();
        let removed = tds.remove_vertex(origin_key).unwrap();
        assert_eq!(removed, 1);

        // The removed vertex should be gone.
        assert!(tds.vertex_key_from_uuid(&origin_uuid).is_none());
        assert!(tds.vertex(origin_key).is_none());

        // simplex2 should remain and must not reference simplex1 as a neighbor.
        assert!(tds.simplices.contains_key(simplex2));
        let simplex2_ref = tds.simplex(simplex2).unwrap();
        if let Some(mut neighbors) = simplex2_ref.neighbors() {
            assert!(neighbors.all(|n| n != Some(simplex1)));
        }

        // Remaining vertices must have valid incident_simplex pointers (if present).
        for vertex_key in [east_key, north_key, diagonal_key] {
            let v = tds.vertex(vertex_key).unwrap();
            let Some(incident) = v.incident_simplex() else {
                panic!("vertex {vertex_key:?} should have an incident simplex after removal");
            };
            assert!(tds.simplices.contains_key(incident));
            assert!(tds.simplex(incident).unwrap().contains_vertex(vertex_key));
        }
    }

    #[test]
    fn test_find_neighbors_by_key_returns_none_buffer_for_missing_simplex() {
        let tds: Tds<(), (), 2> = Tds::empty();
        let missing = SimplexKey::from(KeyData::from_ffi(u64::MAX));
        let neighbors = tds.find_neighbors_by_key(missing);
        assert_eq!(neighbors.len(), 3);
        assert!(neighbors.iter().all(Option::is_none));
    }

    #[test]
    fn test_set_neighbors_by_key_rejects_non_neighbor_and_wrong_slot() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v_e = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([2.0, 2.0]).unwrap(),
            )
            .unwrap();

        let simplex1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();

        // A simplex that shares only one vertex with simplex1 => not a neighbor in 2D.
        let simplex_far = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_d, v_e], None).unwrap(),
            )
            .unwrap();
        let err = tds
            .set_neighbors_by_key(simplex1, &[Some(simplex_far), None, None])
            .unwrap_err()
            .0;
        assert_matches!(err, TdsError::InvalidNeighbors { .. });

        // A true facet-neighbor (shares {v_a,v_b}) placed at the wrong facet index.
        let simplex2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_b, v_a, v_d], None).unwrap(),
            )
            .unwrap();
        let err = tds
            .set_neighbors_by_key(simplex1, &[Some(simplex2), None, None])
            .unwrap_err()
            .0;
        assert_matches!(err, TdsError::InvalidNeighbors { .. });
    }

    #[test]
    fn test_set_neighbors_by_key_updates_reciprocal_back_reference() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let simplex2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_d], None).unwrap(),
            )
            .unwrap();

        tds.set_neighbors_by_key(simplex1, &[None, None, Some(simplex2)])
            .unwrap();

        assert_eq!(
            tds.simplex(simplex1).unwrap().neighbor_key(2).flatten(),
            Some(simplex2)
        );
        assert_eq!(
            tds.simplex(simplex2).unwrap().neighbor_key(2).flatten(),
            Some(simplex1)
        );
        let facet_to_simplices = tds.build_facet_to_simplices_map().unwrap();
        tds.validate_neighbors_with_facet_to_simplices_map(&facet_to_simplices)
            .unwrap();
    }

    #[test]
    fn test_set_neighbors_by_key_rejects_unpaired_interior_facet() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v_b, v_a, v_d], None).unwrap(),
        )
        .unwrap();
        tds.assign_neighbors().unwrap();

        let err = tds
            .set_neighbors_by_key(simplex1, &[None, None, None])
            .unwrap_err()
            .0;
        assert_matches!(err, TdsError::InvalidNeighbors { .. });
        assert!(tds.is_valid().is_ok());
    }

    // =============================================================================
    // NEIGHBOR VALIDATION HELPER TESTS
    // =============================================================================

    #[test]
    fn test_build_simplex_vertex_sets_success() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let simplex2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_c, v_d], None).unwrap(),
            )
            .unwrap();

        let map = tds.build_simplex_vertex_sets().unwrap();
        assert_eq!(map.len(), 2);

        let expected1: VertexKeySet = [v_a, v_b, v_c].into_iter().collect();
        let expected2: VertexKeySet = [v_a, v_c, v_d].into_iter().collect();

        assert_eq!(map.get(&simplex1), Some(&expected1));
        assert_eq!(map.get(&simplex2), Some(&expected2));
    }

    #[test]
    fn test_build_simplex_vertex_sets_errors_on_missing_vertex_key() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();

        // Corrupt the simplex by inserting a vertex key that doesn't exist in the TDS.
        let invalid_vkey = VertexKey::from(KeyData::from_ffi(u64::MAX));
        tds.simplex_mut(simplex)
            .unwrap()
            .push_vertex_key(invalid_vkey);

        let err = tds.build_simplex_vertex_sets().unwrap_err();
        assert_matches!(err, TdsError::VertexNotFound { .. });
    }

    #[test]
    fn test_validate_shared_facet_count_ok_for_true_neighbors() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let simplex2_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_d], None).unwrap(),
            )
            .unwrap();

        let simplex1 = tds.simplex(simplex1_key).unwrap();
        let simplex2 = tds.simplex(simplex2_key).unwrap();

        let this_vertices: VertexKeySet = [v_a, v_b, v_c].into_iter().collect();
        let neighbor_vertices: VertexKeySet = [v_a, v_b, v_d].into_iter().collect();

        assert!(
            Tds::<(), (), 2>::validate_shared_facet_count(
                simplex1,
                simplex2,
                &this_vertices,
                &neighbor_vertices
            )
            .is_ok()
        );
    }

    #[test]
    fn test_validate_shared_facet_count_errors_for_non_neighbors() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v_e = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([2.0, 2.0]).unwrap(),
            )
            .unwrap();

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let simplex_far_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_d, v_e], None).unwrap(),
            )
            .unwrap();

        let simplex1 = tds.simplex(simplex1_key).unwrap();
        let simplex_far = tds.simplex(simplex_far_key).unwrap();

        let this_vertices: VertexKeySet = [v_a, v_b, v_c].into_iter().collect();
        let far_vertices: VertexKeySet = [v_a, v_d, v_e].into_iter().collect();

        let simplex1_uuid = simplex1.uuid();
        let simplex_far_uuid = simplex_far.uuid();

        let err = Tds::<(), (), 2>::validate_shared_facet_count(
            simplex1,
            simplex_far,
            &this_vertices,
            &far_vertices,
        )
        .unwrap_err();

        assert_matches!(
            err,
            TdsError::NotNeighbors { simplex1: c1, simplex2: c2 }
                if c1 == simplex1_uuid && c2 == simplex_far_uuid
        );
    }

    #[test]
    fn test_expected_mirror_facet_index_returns_unique_vertex_index() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        // Put the unique vertex at index 0 to ensure we test the returned index.
        let simplex2_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_d, v_a, v_b], None).unwrap(),
            )
            .unwrap();

        let simplex1 = tds.simplex(simplex1_key).unwrap();
        let simplex2 = tds.simplex(simplex2_key).unwrap();

        let this_vertices: VertexKeySet = [v_a, v_b, v_c].into_iter().collect();

        let idx = Tds::<(), (), 2>::expected_mirror_facet_index(simplex1, simplex2, &this_vertices)
            .unwrap();
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_expected_mirror_facet_index_errors_when_ambiguous() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v_e = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([2.0, 2.0]).unwrap(),
            )
            .unwrap();

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        // Shares only v_a -> differs by 2 vertices -> ambiguous mirror facet.
        let simplex2_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_d, v_e], None).unwrap(),
            )
            .unwrap();

        let simplex1 = tds.simplex(simplex1_key).unwrap();
        let simplex2 = tds.simplex(simplex2_key).unwrap();

        let this_vertices: VertexKeySet = [v_a, v_b, v_c].into_iter().collect();

        let err = Tds::<(), (), 2>::expected_mirror_facet_index(simplex1, simplex2, &this_vertices)
            .unwrap_err();

        assert_matches!(
            err,
            TdsError::InvalidNeighbors {
                reason: NeighborValidationError::MirrorFacetAmbiguous { .. },
            }
        );
    }

    #[test]
    fn test_expected_mirror_facet_index_errors_when_duplicate_simplices() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        // Duplicate by vertices (different UUID) -> no unique vertex to identify mirror facet.
        let simplex2_key = tds
            .insert_simplex_bypassing_topology_checks_for_test(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();

        let simplex1 = tds.simplex(simplex1_key).unwrap();
        let simplex2 = tds.simplex(simplex2_key).unwrap();

        let this_vertices: VertexKeySet = [v_a, v_b, v_c].into_iter().collect();

        let err = Tds::<(), (), 2>::expected_mirror_facet_index(simplex1, simplex2, &this_vertices)
            .unwrap_err();

        assert_matches!(
            err,
            TdsError::InvalidNeighbors {
                reason: NeighborValidationError::MirrorFacetDuplicateSimplices { .. },
            }
        );
    }

    #[test]
    fn test_verified_mirror_facet_index_ok() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let simplex2_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_d], None).unwrap(),
            )
            .unwrap();

        let simplex1 = tds.simplex(simplex1_key).unwrap();
        let simplex2 = tds.simplex(simplex2_key).unwrap();

        let this_vertices: VertexKeySet = [v_a, v_b, v_c].into_iter().collect();

        // Shared edge is (v_a, v_b). In simplex1, that's opposite vertex index 2 (v_c).
        let mirror_idx =
            Tds::<(), (), 2>::verified_mirror_facet_index(simplex1, 2, simplex2, &this_vertices)
                .unwrap();
        assert_eq!(mirror_idx, 2);
    }

    #[test]
    fn test_verified_mirror_facet_index_errors_when_no_shared_facet() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let simplex2_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_d], None).unwrap(),
            )
            .unwrap();

        let simplex1 = tds.simplex(simplex1_key).unwrap();
        let simplex2 = tds.simplex(simplex2_key).unwrap();

        let this_vertices: VertexKeySet = [v_a, v_b, v_c].into_iter().collect();

        // facet_idx=0 corresponds to edge (v_b, v_c) in simplex1, which is not shared with simplex2.
        let err =
            Tds::<(), (), 2>::verified_mirror_facet_index(simplex1, 0, simplex2, &this_vertices)
                .unwrap_err();

        assert_matches!(
            err,
            TdsError::InvalidNeighbors {
                reason: NeighborValidationError::MirrorFacetMissing { .. },
            }
        );
    }

    #[test]
    fn test_verified_mirror_facet_index_errors_on_mismatch_with_cross_check() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let simplex2_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_d], None).unwrap(),
            )
            .unwrap();

        let simplex1 = tds.simplex(simplex1_key).unwrap();
        let simplex2 = tds.simplex(simplex2_key).unwrap();

        // Intentionally WRONG vertex set (includes v_d, excludes v_b) to force the mismatch branch.
        // This is a unit-level test of the helper's defensive cross-check behavior.
        let this_vertices_wrong: VertexKeySet = [v_a, v_c, v_d].into_iter().collect();

        let err = Tds::<(), (), 2>::verified_mirror_facet_index(
            simplex1,
            2,
            simplex2,
            &this_vertices_wrong,
        )
        .unwrap_err();

        assert_matches!(
            err,
            TdsError::InvalidNeighbors {
                reason: NeighborValidationError::MirrorFacetIndexMismatch { .. },
            }
        );
    }

    #[test]
    fn test_validate_neighbors_errors_on_mirror_facet_index_mismatch() {
        // This test exercises the same "mirror facet index mismatch" defensive branch, but via the
        // neighbor-validation loop used by `validate_neighbors_with_precomputed_vertex_sets()`.
        //
        // The mismatch is only reachable if the precomputed per-simplex vertex-set map is inconsistent
        // with the simplex's actual vertex buffer (e.g., a bug/corruption in the precompute step). To
        // simulate that scenario deterministically, we build the map normally and then corrupt the
        // entry for one simplex before running the validation loop.
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let origin_key = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let east_key = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let north_key = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let diagonal_key = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![origin_key, east_key, north_key], None).unwrap(),
            )
            .unwrap();
        let _simplex2_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![origin_key, east_key, diagonal_key], None).unwrap(),
            )
            .unwrap();

        tds.assign_neighbors().unwrap();

        let mut simplex_vertices = tds.build_simplex_vertex_sets().unwrap();

        // Corrupt the vertex-set entry for simplex1 so it no longer matches the actual simplex's vertices.
        // (Drop `east_key`, add `diagonal_key`.)
        let corrupted_simplex1_vertices: VertexKeySet =
            [origin_key, north_key, diagonal_key].into_iter().collect();
        simplex_vertices.insert(simplex1_key, corrupted_simplex1_vertices);

        let err = tds
            .validate_neighbors_with_precomputed_vertex_sets(&simplex_vertices)
            .unwrap_err();

        assert_matches!(
            err,
            TdsError::InvalidNeighbors {
                reason: NeighborValidationError::MirrorFacetIndexMismatch { .. },
            }
        );
    }

    #[test]
    fn test_validate_shared_facet_vertices_ok() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let simplex2_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_d], None).unwrap(),
            )
            .unwrap();

        let simplex1 = tds.simplex(simplex1_key).unwrap();
        let simplex2 = tds.simplex(simplex2_key).unwrap();

        let this_vertices: VertexKeySet = [v_a, v_b, v_c].into_iter().collect();
        let neighbor_vertices: VertexKeySet = [v_a, v_b, v_d].into_iter().collect();

        assert!(
            Tds::<(), (), 2>::validate_shared_facet_vertices(
                simplex1,
                2, // opposite v_c => shared edge {v_a,v_b}
                simplex2,
                2, // opposite v_d => shared edge {v_a,v_b}
                &this_vertices,
                &neighbor_vertices,
            )
            .is_ok()
        );
    }

    #[test]
    fn test_validate_shared_facet_vertices_errors_when_mirror_index_wrong() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let simplex2_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_d], None).unwrap(),
            )
            .unwrap();

        let simplex1 = tds.simplex(simplex1_key).unwrap();
        let simplex2 = tds.simplex(simplex2_key).unwrap();

        let this_vertices: VertexKeySet = [v_a, v_b, v_c].into_iter().collect();
        let neighbor_vertices: VertexKeySet = [v_a, v_b, v_d].into_iter().collect();

        // mirror_idx=0 is intentionally wrong here; it treats vertex v_a as the "opposite"
        // which makes v_d incorrectly part of the "shared facet".
        let err = Tds::<(), (), 2>::validate_shared_facet_vertices(
            simplex1,
            2, // correct for simplex1
            simplex2,
            0, // intentionally wrong for simplex2
            &this_vertices,
            &neighbor_vertices,
        )
        .unwrap_err();

        assert_matches!(
            err,
            TdsError::InvalidNeighbors {
                reason: NeighborValidationError::SharedFacetMissingVertex { .. },
            }
        );
    }

    #[test]
    fn test_validate_mutual_neighbor_back_reference_ok() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let simplex2_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_d], None).unwrap(),
            )
            .unwrap();

        // Build neighbor pointers so mutual back-references exist.
        tds.assign_neighbors().unwrap();

        let simplex1 = tds.simplex(simplex1_key).unwrap();
        let simplex2 = tds.simplex(simplex2_key).unwrap();

        assert!(
            Tds::<(), (), 2>::validate_mutual_neighbor_back_reference(
                simplex1_key,
                simplex1,
                2, // opposite v_c => shared edge {v_a,v_b}
                simplex2_key,
                simplex2,
                2, // opposite v_d => shared edge {v_a,v_b}
            )
            .is_ok()
        );
    }

    #[test]
    fn test_validate_mutual_neighbor_back_reference_errors_when_neighbor_has_no_neighbors() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let simplex2_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_d], None).unwrap(),
            )
            .unwrap();

        // NOTE: We intentionally do NOT call assign_neighbors(), so neighbor_simplex.neighbors is None.
        let simplex1 = tds.simplex(simplex1_key).unwrap();
        let simplex2 = tds.simplex(simplex2_key).unwrap();

        let err = Tds::<(), (), 2>::validate_mutual_neighbor_back_reference(
            simplex1_key,
            simplex1,
            2,
            simplex2_key,
            simplex2,
            2,
        )
        .unwrap_err();

        assert_matches!(
            err,
            TdsError::InvalidNeighbors {
                reason: NeighborValidationError::BackReferenceMismatch { observed: None, .. },
            }
        );
    }

    #[test]
    fn test_validate_mutual_neighbor_back_reference_errors_when_back_reference_missing() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let simplex2_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_d], None).unwrap(),
            )
            .unwrap();

        // Build neighbor pointers first, then deliberately corrupt the back-reference.
        tds.assign_neighbors().unwrap();

        {
            let simplex2_mut = tds.simplex_mut(simplex2_key).unwrap();
            let neighbors = simplex2_mut
                .neighbor_slots_mut()
                .expect("simplex2 should have neighbors after assign_neighbors()");
            // For (v_a, v_b, v_d), the shared edge with simplex1 is opposite v_d => index 2.
            neighbors[2] = NeighborSlot::Boundary;
        }

        let simplex1 = tds.simplex(simplex1_key).unwrap();
        let simplex2 = tds.simplex(simplex2_key).unwrap();

        let err = Tds::<(), (), 2>::validate_mutual_neighbor_back_reference(
            simplex1_key,
            simplex1,
            2,
            simplex2_key,
            simplex2,
            2,
        )
        .unwrap_err();

        assert_matches!(
            err,
            TdsError::InvalidNeighbors {
                reason: NeighborValidationError::BackReferenceMismatch { observed: None, .. },
            }
        );
    }

    #[test]
    fn test_orientation_validation_rejects_non_periodic_self_neighbor() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();

        {
            let simplex = tds.simplex_mut(simplex_key).unwrap();
            simplex
                .set_neighbors_from_keys(vec![Some(simplex_key), None, None])
                .unwrap();
        }

        let simplex = tds.simplex(simplex_key).unwrap();
        assert!(!Tds::<(), (), 2>::allows_periodic_self_neighbor(simplex));

        let err = tds.validate_coherent_orientation().unwrap_err();
        assert_matches!(
            err,
            TdsError::OrientationViolation {
                simplex1_key,
                simplex2_key,
                ..
            } if simplex1_key == simplex_key && simplex2_key == simplex_key
        );
        assert!(!tds.is_coherently_oriented());

        let err = tds.normalize_coherent_orientation().unwrap_err();
        assert_matches!(
            err,
            TdsError::InconsistentDataStructure { message }
                if message.contains("Contradictory orientation constraints")
        );
    }

    #[test]
    fn test_orientation_validation_allows_periodic_self_neighbor() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();

        {
            let simplex = tds.simplex_mut(simplex_key).unwrap();
            simplex
                .set_neighbors_from_keys(vec![Some(simplex_key), None, None])
                .unwrap();
            simplex
                .set_periodic_vertex_offsets(vec![[0, 0], [0, 0], [0, 0]])
                .unwrap();
        }

        let simplex = tds.simplex(simplex_key).unwrap();
        assert!(Tds::<(), (), 2>::allows_periodic_self_neighbor(simplex));
        assert!(tds.validate_coherent_orientation().is_ok());
        assert!(tds.is_coherently_oriented());
        assert!(tds.normalize_coherent_orientation().is_ok());
    }

    #[test]
    fn test_orientation_validation_rejects_one_sided_periodic_neighbor() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let simplex2_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_d], None).unwrap(),
            )
            .unwrap();

        {
            let simplex1 = tds.simplex_mut(simplex1_key).unwrap();
            simplex1
                .set_neighbors_from_keys(vec![None, None, Some(simplex2_key)])
                .unwrap();
            simplex1
                .set_periodic_vertex_offsets(vec![[0, 0], [0, 0], [0, 0]])
                .unwrap();
        }
        {
            let simplex2 = tds.simplex_mut(simplex2_key).unwrap();
            simplex2
                .set_neighbors_from_keys(vec![None, None, None])
                .unwrap();
            simplex2
                .set_periodic_vertex_offsets(vec![[0, 0], [0, 0], [0, 0]])
                .unwrap();
        }

        let err = tds.validate_coherent_orientation().unwrap_err();
        assert_matches!(
            err,
            TdsError::InvalidNeighbors {
                reason: NeighborValidationError::BackReferenceMismatch { observed: None, .. },
            }
        );
        assert!(!tds.is_coherently_oriented());

        let err = tds
            .validate_coherent_orientation_for_simplices(&[simplex1_key])
            .unwrap_err();
        assert_matches!(
            err,
            TdsError::InvalidNeighbors {
                reason: NeighborValidationError::BackReferenceMismatch { observed: None, .. },
            }
        );
    }

    #[test]
    fn test_boundary_facet_validation_rejects_unassigned_neighbor_slot() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        tds.assign_neighbors().unwrap();
        let facet_to_simplices = tds.build_facet_to_simplices_map().unwrap();

        {
            let simplex = tds.simplex_mut(simplex_key).unwrap();
            simplex.ensure_neighbors_buffer_mut()[1] = NeighborSlot::Unassigned;
        }

        let err = tds
            .validate_neighbor_pointers_match_facet_to_simplices_map(&facet_to_simplices)
            .unwrap_err();

        assert_matches!(
            err,
            TdsError::InvalidNeighbors {
                reason: NeighborValidationError::UnassignedNeighborSlot {
                    simplex_key: key,
                    facet_index: 1,
                    ..
                },
            } if key == simplex_key
        );
    }

    #[test]
    fn test_boundary_facet_validation_rejects_non_periodic_self_neighbor() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        tds.assign_neighbors().unwrap();
        let facet_to_simplices = tds.build_facet_to_simplices_map().unwrap();

        {
            let simplex = tds.simplex_mut(simplex_key).unwrap();
            simplex.ensure_neighbors_buffer_mut()[2] = NeighborSlot::Neighbor(simplex_key);
        }

        let err = tds
            .validate_neighbor_pointers_match_facet_to_simplices_map(&facet_to_simplices)
            .unwrap_err();

        assert_matches!(
            err,
            TdsError::InvalidNeighbors {
                reason: NeighborValidationError::BoundaryFacetHasNonPeriodicSelfNeighbor {
                    simplex_key: key,
                    facet_index: 2,
                    ..
                },
            } if key == simplex_key
        );
    }

    #[test]
    fn test_orientation_validation_rejects_one_way_neighbor_pointer() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let simplex2_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_b, v_a, v_d], None).unwrap(),
            )
            .unwrap();
        tds.assign_neighbors().unwrap();
        assert!(tds.validate_coherent_orientation().is_ok());

        let mirror_idx = {
            let simplex1 = tds.simplex(simplex1_key).unwrap();
            let mut neighbors = simplex1
                .neighbors()
                .expect("simplex1 should have neighbors after assign_neighbors()");
            let facet_idx = neighbors
                .position(|n| n == Some(simplex2_key))
                .expect("simplex1 should reference simplex2");
            let simplex2 = tds.simplex(simplex2_key).unwrap();
            simplex1
                .mirror_facet_index(facet_idx, simplex2)
                .expect("adjacent simplices should have a mirror facet")
        };

        {
            let simplex2 = tds.simplex_mut(simplex2_key).unwrap();
            let neighbors = simplex2
                .neighbor_slots_mut()
                .expect("simplex2 should have neighbors after assign_neighbors()");
            neighbors[mirror_idx] = NeighborSlot::Boundary;
        }

        let err = tds.validate_coherent_orientation().unwrap_err();
        assert_matches!(
            err,
            TdsError::InvalidNeighbors {
                reason: NeighborValidationError::BackReferenceMismatch { observed: None, .. },
            }
        );
        assert!(!tds.is_coherently_oriented());
    }

    #[test]
    fn test_local_orientation_validation_checks_neighbors_outside_scope() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([2.0, 2.0]).unwrap(),
            )
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([3.0, 3.0]).unwrap(),
            )
            .unwrap();

        let simplex1_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v3], None).unwrap(),
            )
            .unwrap();
        tds.assign_neighbors().unwrap();

        let err = tds
            .validate_coherent_orientation_for_simplices(&[simplex1_key])
            .unwrap_err();
        assert_matches!(err, TdsError::OrientationViolation { .. });
    }

    #[test]
    fn test_local_orientation_validation_errors_on_missing_scope_simplex() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        assert_eq!(tds.remove_simplices_by_keys(&[simplex_key]), 1);

        let err = tds
            .validate_coherent_orientation_for_simplices(&[simplex_key])
            .unwrap_err();
        assert_matches!(
            err,
            TdsError::SimplexNotFound {
                simplex_key: missing_key,
                ..
            } if missing_key == simplex_key
        );
    }

    macro_rules! test_normalize_repairs_incoherent_adjacent_pair {
        ($name:ident, $dim:literal) => {
            #[test]
            fn $name() {
                let mut tds: Tds<(), (), $dim> = Tds::empty();

                let mut vertex_keys = Vec::with_capacity($dim + 2);
                let mut seed = 1.0_f64;
                for idx in 0..($dim + 2) {
                    let mut coords = [0.0_f64; $dim];
                    if idx < $dim {
                        coords[idx] = 1.0;
                    } else {
                        for coord in &mut coords {
                            *coord = seed;
                            seed += 1.0;
                        }
                    }
                    vertex_keys.push(
                        tds.insert_vertex_with_mapping(
                            crate::core::vertex::Vertex::<(), _>::try_new(coords).unwrap(),
                        )
                        .unwrap(),
                    );
                }

                // Construct two adjacent simplices that share a facet but induce the same shared-facet
                // ordering, making orientation incoherent before normalization:
                // simplex1 = [v0..vD], simplex2 = [v0..v(D-1), v(D+1)].
                let simplex1_vertices: Vec<_> =
                    vertex_keys.iter().take($dim + 1).copied().collect();
                let mut simplex2_vertices: Vec<_> =
                    vertex_keys.iter().take($dim).copied().collect();
                simplex2_vertices.push(vertex_keys[$dim + 1]);

                let simplex1: Simplex<(), $dim> =
                    Simplex::try_new_with_data(simplex1_vertices, None).unwrap();
                let simplex2: Simplex<(), $dim> =
                    Simplex::try_new_with_data(simplex2_vertices, None).unwrap();

                tds.insert_simplex_with_mapping(simplex1).unwrap();
                tds.insert_simplex_with_mapping(simplex2).unwrap();
                tds.assign_neighbors().unwrap();

                let err = tds.validate_coherent_orientation().unwrap_err();
                assert_matches!(err, TdsError::OrientationViolation { .. });
                assert!(!tds.is_coherently_oriented());

                tds.normalize_coherent_orientation().unwrap();
                assert!(tds.validate_coherent_orientation().is_ok());
                assert!(tds.is_coherently_oriented());
            }
        };
    }

    test_normalize_repairs_incoherent_adjacent_pair!(
        test_normalize_repairs_incoherent_adjacent_pair_2d,
        2
    );
    test_normalize_repairs_incoherent_adjacent_pair!(
        test_normalize_repairs_incoherent_adjacent_pair_3d,
        3
    );
    test_normalize_repairs_incoherent_adjacent_pair!(
        test_normalize_repairs_incoherent_adjacent_pair_4d,
        4
    );
    test_normalize_repairs_incoherent_adjacent_pair!(
        test_normalize_repairs_incoherent_adjacent_pair_5d,
        5
    );

    #[test]
    fn test_assign_incident_simplices_clears_incident_simplex_when_no_simplices() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let vkey = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();

        // Corrupt incident_simplex and ensure it gets cleared.
        tds.vertex_mut(vkey)
            .unwrap()
            .set_incident_simplex(Some(SimplexKey::from(KeyData::from_ffi(u64::MAX))));
        assert!(tds.vertex(vkey).unwrap().incident_simplex().is_some());

        tds.assign_incident_simplices().unwrap();
        assert!(tds.vertex(vkey).unwrap().incident_simplex().is_none());
    }

    #[test]
    fn test_build_facet_to_simplices_map_errors_on_u8_facet_index_overflow() {
        let tds: Tds<(), (), 256> = Tds::empty();

        let err = tds.build_facet_to_simplices_map().unwrap_err();
        assert_matches!(
            err,
            TdsError::DimensionMismatch {
                expected: 255,
                actual: 256,
                ref context,
            } if context == "facet indices must fit in u8"
        );
    }

    #[test]
    fn test_validate_vertex_and_simplex_mappings_detect_inconsistencies() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let simplex_key = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, c], None).unwrap())
            .unwrap();

        // Start from a consistent state.
        assert!(tds.validate_vertex_mappings().is_ok());
        assert!(tds.validate_simplex_mappings().is_ok());

        // Break vertex mapping: remove one uuid entry (len mismatch).
        let uuid_a = tds.vertex(a).unwrap().uuid();
        tds.uuid_to_vertex_key.remove(&uuid_a);
        assert_matches!(
            tds.validate_vertex_mappings(),
            Err(TdsError::MappingInconsistency {
                entity: EntityKind::Vertex,
                ..
            })
        );

        // Restore length but make the UUID map point at the wrong key.
        tds.uuid_to_vertex_key.insert(uuid_a, b);
        assert_matches!(
            tds.validate_vertex_mappings(),
            Err(TdsError::MappingInconsistency {
                entity: EntityKind::Vertex,
                ..
            })
        );

        // Break simplex mapping similarly.
        let uuid_simplex = tds.simplex(simplex_key).unwrap().uuid();
        tds.uuid_to_simplex_key.remove(&uuid_simplex);
        assert_matches!(
            tds.validate_simplex_mappings(),
            Err(TdsError::MappingInconsistency {
                entity: EntityKind::Simplex,
                ..
            })
        );
    }

    #[test]
    fn test_validate_simplex_vertex_keys_detects_missing_vertices() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex_key = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, c], None).unwrap())
            .unwrap();

        let invalid_vkey = VertexKey::from(KeyData::from_ffi(u64::MAX));
        tds.simplex_mut(simplex_key)
            .unwrap()
            .push_vertex_key(invalid_vkey);

        let err = tds.validate_simplex_vertex_keys().unwrap_err();
        assert_matches!(err, TdsError::VertexNotFound { .. });

        // Now wired into structural validation: is_valid() should fail early with the
        // more precise "missing vertex key" diagnostic.
        let err = tds.is_valid().unwrap_err();
        assert_matches!(err, TdsError::VertexNotFound { .. });
    }

    // ---- Error variant Display / construction coverage ----

    #[test]
    fn test_geometric_error_display() {
        let deg = GeometricError::DegenerateOrientation {
            message: "det=0".to_string(),
        };
        assert!(deg.to_string().contains("det=0"));

        let neg = GeometricError::NegativeOrientation {
            message: "det<0".to_string(),
        };
        assert!(neg.to_string().contains("det<0"));
    }

    #[test]
    fn test_tds_error_new_variant_display() {
        let simplex_key = SimplexKey::from(KeyData::from_ffi(1));
        let vertex_key = VertexKey::from(KeyData::from_ffi(2));

        let err = TdsError::SimplexNotFound {
            simplex_key,
            context: "test lookup".to_string(),
        };
        assert!(err.to_string().contains("not found"));
        assert!(err.to_string().contains("test lookup"));

        let err = TdsError::VertexNotFound {
            vertex_key,
            context: "test vertex".to_string(),
        };
        assert!(err.to_string().contains("not found"));
        assert!(err.to_string().contains("test vertex"));

        let err = TdsError::DimensionMismatch {
            expected: 4,
            actual: 3,
            context: "simplex check".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains('4') && msg.contains('3') && msg.contains("simplex check"));

        let err = TdsError::IndexOutOfBounds {
            index: 10,
            bound: 5,
            context: "facet index".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("10") && msg.contains('5') && msg.contains("facet index"));
    }

    #[test]
    fn test_tds_error_geometric_variant_wraps_geometric_error() {
        let inner = GeometricError::DegenerateOrientation {
            message: "test".to_string(),
        };
        let err = TdsError::Geometric(inner.clone());
        assert!(err.to_string().contains("test"));
        assert_eq!(TdsError::from(inner.clone()), TdsError::Geometric(inner));
    }

    #[test]
    fn test_tds_mutation_error_accessors() {
        let inner = TdsError::InvalidNeighbors {
            reason: NeighborValidationError::Other {
                message: "test".to_string(),
            },
        };
        let mutation = TdsMutationError::from(inner.clone());

        // as_tds_error returns a reference to the inner error.
        assert_eq!(mutation.as_tds_error(), &inner);

        // into_inner consumes the wrapper and returns the inner error.
        let recovered: TdsError = mutation.into_inner();
        assert_eq!(recovered, inner);
    }

    #[test]
    fn test_invariant_error_from_tds_and_triangulation() {
        let tds_err = TdsError::InconsistentDataStructure {
            message: "test".to_string(),
        };
        let inv = InvariantError::from(tds_err);
        assert_matches!(inv, InvariantError::Tds(_));

        let tri_err = TriangulationValidationError::EulerCharacteristicMismatch {
            computed: 1,
            expected: 2,
            classification: TopologyClassification::Ball(3),
        };
        let inv = InvariantError::from(tri_err);
        assert_matches!(inv, InvariantError::Triangulation(_));
    }

    #[test]
    fn test_invariant_error_from_delaunay_validation_error() {
        let dt_err = DelaunayTriangulationValidationError::VerificationFailed {
            message: "test".to_string(),
        };
        let inv = InvariantError::from(dt_err);
        assert_matches!(inv, InvariantError::Delaunay(_));
    }

    #[test]
    fn test_invariant_error_summary_covers_validation_layers() {
        let cases = [
            (
                InvariantError::from(TdsError::InconsistentDataStructure {
                    message: "dangling simplex".to_string(),
                }),
                InvariantErrorSummaryKind::Tds,
                InvariantErrorSummaryDetail::Tds(TdsErrorKind::InconsistentDataStructure),
            ),
            (
                InvariantError::from(TriangulationValidationError::Disconnected {
                    simplex_count: 2,
                }),
                InvariantErrorSummaryKind::Triangulation,
                InvariantErrorSummaryDetail::Triangulation(
                    TriangulationValidationErrorKind::Disconnected,
                ),
            ),
            (
                InvariantError::from(DelaunayTriangulationValidationError::VerificationFailed {
                    message: "non-Delaunay facet".to_string(),
                }),
                InvariantErrorSummaryKind::Delaunay,
                InvariantErrorSummaryDetail::Delaunay(
                    DelaunayValidationErrorKind::VerificationFailed,
                ),
            ),
        ];

        for (source, expected_kind, expected_detail) in cases {
            let expected_message = source.to_string();
            let summary = InvariantErrorSummary::from(source);
            assert_eq!(summary.kind, expected_kind);
            assert_eq!(summary.detail, expected_detail);
            assert_eq!(summary.message, expected_message);
        }
    }

    #[test]
    fn test_invariant_kind_all_variants_are_distinct() {
        let kinds = [
            InvariantKind::VertexValidity,
            InvariantKind::SimplexValidity,
            InvariantKind::SimplexCoordinateUniqueness,
            InvariantKind::VertexMappings,
            InvariantKind::SimplexMappings,
            InvariantKind::SimplexVertexKeys,
            InvariantKind::VertexIncidence,
            InvariantKind::DuplicateSimplices,
            InvariantKind::FacetSharing,
            InvariantKind::NeighborConsistency,
            InvariantKind::CoherentOrientation,
            InvariantKind::Connectedness,
            InvariantKind::Topology,
            InvariantKind::DelaunayProperty,
        ];
        // All variants must be copyable and comparable.
        for (i, &a) in kinds.iter().enumerate() {
            assert_eq!(a, a);
            for &b in &kinds[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }

    #[test]
    fn test_invariant_violation_stores_kind_and_error() {
        let violation = InvariantViolation {
            kind: InvariantKind::NeighborConsistency,
            error: InvariantError::Tds(TdsError::InvalidNeighbors {
                reason: NeighborValidationError::Other {
                    message: "test".to_string(),
                },
            }),
        };
        assert_eq!(violation.kind, InvariantKind::NeighborConsistency);
        assert_matches!(violation.error, InvariantError::Tds(_));
    }

    #[test]
    fn test_entity_kind_debug_output() {
        assert_eq!(format!("{:?}", EntityKind::Vertex), "Vertex");
        assert_eq!(format!("{:?}", EntityKind::Simplex), "Simplex");
        assert_ne!(EntityKind::Vertex, EntityKind::Simplex);
    }

    #[test]
    fn test_tds_mutation_error_from_round_trips() {
        // Test the full round-trip: TdsError -> TdsMutationError -> TdsError
        let original = TdsError::SimplexNotFound {
            simplex_key: SimplexKey::from(KeyData::from_ffi(42)),
            context: "round trip".to_string(),
        };
        let mutation = TdsMutationError::from(original.clone());
        assert_eq!(mutation.to_string(), original.to_string());
        let round_tripped: TdsError = mutation.into();
        assert_eq!(round_tripped, original);
    }

    #[test]
    fn test_geometric_error_from_into_tds_error() {
        let geo = GeometricError::NegativeOrientation {
            message: "det<0".to_string(),
        };
        let tds_err: TdsError = geo.into();
        assert_matches!(
            tds_err,
            TdsError::Geometric(GeometricError::NegativeOrientation { .. })
        );
        // Display propagates via #[error(transparent)].
        assert!(tds_err.to_string().contains("det<0"));
    }

    #[test]
    fn test_is_connected_returns_false_for_isolated_simplices() {
        // Build a TDS with two triangles that have no neighbor wiring between them.
        // Since neither simplex's `neighbors` field is populated, BFS from either simplex
        // cannot reach the other → is_connected() must return false.
        let mut tds: Tds<(), (), 2> = Tds::empty();

        // Component A
        let a0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let a1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let a2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        // Component B (far away, no shared vertices)
        let b0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([10.0, 0.0]).unwrap(),
            )
            .unwrap();
        let b1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([11.0, 0.0]).unwrap(),
            )
            .unwrap();
        let b2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([10.0, 1.0]).unwrap(),
            )
            .unwrap();

        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![a0, a1, a2], None).unwrap(),
        )
        .unwrap();
        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![b0, b1, b2], None).unwrap(),
        )
        .unwrap();

        // Two simplices with neighbors = None: BFS from the first simplex finds no edges
        // and can never visit the second.
        assert!(
            !tds.is_connected(),
            "TDS with two isolated simplices (no neighbor wiring) must not be connected"
        );
    }

    // =========================================================================
    // COHERENT ORIENTATION NORMALIZATION & GENERATION COUNTER
    // =========================================================================

    #[test]
    fn test_normalize_coherent_orientation_produces_coherent_result() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();

        // Two triangles sharing edge v1-v2, with deliberately inconsistent vertex order
        let c0 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        let c1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v1, v2, v3], None).unwrap(),
            )
            .unwrap();

        // Assign neighbors: shared facet is edge v1-v2.
        // c0[v0,v1,v2]: facet opposite v0 (index 0) = edge [v1,v2] → neighbor c1
        // c1[v1,v2,v3]: facet opposite v3 (index 2) = edge [v1,v2] → neighbor c0
        tds.simplex_mut(c0)
            .unwrap()
            .set_neighbors_from_keys([Some(c1), None, None])
            .unwrap();
        tds.simplex_mut(c1)
            .unwrap()
            .set_neighbors_from_keys([None, None, Some(c0)])
            .unwrap();

        tds.normalize_coherent_orientation().unwrap();
        assert!(tds.is_coherently_oriented());
    }

    #[test]
    fn test_generation_counter_bumps_on_topology_modification() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        assert_eq!(tds.generation(), 0);

        let _v = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        assert!(tds.generation() > 0);

        let gen_before = tds.generation();
        tds.mark_topology_modified();
        assert!(tds.generation() > gen_before);
    }

    macro_rules! test_clone_identity_dimensions {
        ($($dim:expr),+ $(,)?) => {
            pastey::paste! {
                $(
                    #[test]
                    fn [<test_clone_uses_fresh_runtime_identity_ $dim d>]() {
                        let tds: Tds<(), (), $dim> = Tds::empty();
                        let cloned = tds.clone();

                        assert!(
                            !Arc::ptr_eq(tds.identity(), cloned.identity()),
                            "ordinary TDS clones must have distinct runtime identities"
                        );
                        assert_eq!(tds.generation(), cloned.generation());
                    }

                    #[test]
                    fn [<test_clone_for_rollback_preserves_identity_with_independent_generation_ $dim d>]() {
                        let mut tds: Tds<(), (), $dim> = Tds::empty();
                        let _v = tds
                            .insert_vertex_with_mapping(crate::core::vertex::Vertex::<(), _>::try_new([0.0_f64; $dim]).unwrap())
                            .unwrap();
                        let snapshot = tds.clone_for_rollback();
                        let snapshot_generation = snapshot.generation();

                        assert!(
                            Arc::ptr_eq(tds.identity(), snapshot.identity()),
                            "rollback snapshots should preserve runtime identity"
                        );

                        tds.mark_topology_modified();

                        assert_eq!(
                            snapshot.generation(),
                            snapshot_generation,
                            "rollback snapshots need an independent generation counter"
                        );
                    }
                )+
            }
        };
    }

    test_clone_identity_dimensions!(2, 3, 4, 5);

    #[test]
    fn test_remove_duplicate_simplices_removes_exact_duplicates() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        // Insert the same simplex twice
        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
        )
        .unwrap();
        tds.insert_simplex_bypassing_topology_checks_for_test(
            Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
        )
        .unwrap();
        assert_eq!(tds.number_of_simplices(), 2);
        let generation_before = tds.generation();

        let removed = tds.remove_duplicate_simplices().unwrap();
        assert_eq!(removed, 1);
        assert_eq!(tds.number_of_simplices(), 1);
        assert!(tds.generation() > generation_before);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_remove_duplicate_simplices_noop_when_no_duplicates() {
        let verts = initial_simplex_vertices_3d();
        let dt = DelaunayTriangulation::try_new(&verts).unwrap();
        let mut tds = dt.tds().clone();
        let generation_before = tds.generation();

        let removed = tds.remove_duplicate_simplices().unwrap();
        assert_eq!(removed, 0);
        assert_eq!(tds.generation(), generation_before);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_remove_duplicate_simplices_rolls_back_when_rebuild_fails() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, -1.0]).unwrap(),
            )
            .unwrap();
        let v4 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.5, -0.5]).unwrap(),
            )
            .unwrap();

        // Three distinct triangles share edge v0-v1, so global neighbor
        // assignment will reject the complex after duplicate removal starts.
        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
        )
        .unwrap();
        tds.insert_simplex_bypassing_topology_checks_for_test(
            Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
        )
        .unwrap();
        tds.insert_simplex_bypassing_topology_checks_for_test(
            Simplex::try_new_with_data(vec![v0, v1, v3], None).unwrap(),
        )
        .unwrap();
        tds.insert_simplex_bypassing_topology_checks_for_test(
            Simplex::try_new_with_data(vec![v0, v1, v4], None).unwrap(),
        )
        .unwrap();
        let before = tds.clone();
        let generation_before = tds.generation();

        let error = tds.remove_duplicate_simplices().unwrap_err();

        assert_matches!(
            error.into_inner(),
            TdsError::InconsistentDataStructure { .. }
        );
        assert_eq!(tds.number_of_simplices(), 4);
        assert_eq!(tds.generation(), generation_before);
        assert_eq!(tds, before);
    }

    // =========================================================================
    // VALIDATION ERROR PATHS
    // =========================================================================

    #[test]
    fn test_validate_vertex_incidence_detects_dangling_incident_simplex() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let ck = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        tds.assign_incident_simplices().unwrap();

        // Remove the simplex but leave the vertex's incident_simplex pointer dangling
        tds.simplices.remove(ck);

        let err = tds.validate_vertex_incidence().unwrap_err();
        assert_matches!(err, TdsError::SimplexNotFound { .. });
    }

    #[test]
    fn test_validate_simplex_coordinate_uniqueness_rejects_duplicate_coords() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        // Two distinct vertex keys with identical coordinates
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();

        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
        )
        .unwrap();

        let err = tds.validate_simplex_coordinate_uniqueness().unwrap_err();
        assert!(
            matches!(err, TdsError::DuplicateCoordinatesInSimplex { .. }),
            "Expected DuplicateCoordinatesInSimplex, got {err:?}"
        );
    }

    #[test]
    fn test_validate_facet_sharing_rejects_triple_shared_facet() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.5, 0.5]).unwrap(),
            )
            .unwrap();
        let v4 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.3, 0.3]).unwrap(),
            )
            .unwrap();

        // Three simplices sharing the v0-v1 edge (facet):
        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
        )
        .unwrap();
        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v0, v1, v3], None).unwrap(),
        )
        .unwrap();
        tds.insert_simplex_bypassing_topology_checks_for_test(
            Simplex::try_new_with_data(vec![v0, v1, v4], None).unwrap(),
        )
        .unwrap();

        let err = tds.validate_facet_sharing().unwrap_err();
        let message = err.to_string();
        assert!(
            matches!(
                &err,
                TdsError::FacetSharingViolation {
                    existing_incident_count: 2,
                    attempted_incident_count: 3,
                    max_incident_count: 2,
                    candidate_facet_index: 2,
                    ..
                }
            ),
            "Expected over-shared facet error, got {err:?}"
        );
        assert!(message.contains("exceeds incident-simplex limit"));
        assert!(!message.contains("inserting candidate simplex"));

        let err = tds.is_valid().unwrap_err();
        assert!(
            matches!(err, TdsError::FacetSharingViolation { .. }),
            "Expected is_valid to surface facet-sharing violation, got {err:?}"
        );

        let report = tds.validation_report().unwrap_err();
        let facet_violation = report
            .violations
            .iter()
            .find(|violation| violation.kind == InvariantKind::FacetSharing)
            .expect("validation_report should include the facet-sharing violation");
        assert!(
            matches!(
                &facet_violation.error,
                InvariantError::Tds(TdsError::FacetSharingViolation { .. })
            ),
            "Expected validation_report to preserve structured facet-sharing error, got {:?}",
            facet_violation.error
        );
    }

    #[test]
    fn test_validate_no_duplicate_simplices_detects_dupes() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
        )
        .unwrap();
        tds.insert_simplex_bypassing_topology_checks_for_test(
            Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
        )
        .unwrap();

        let err = tds.validate_no_duplicate_simplices().unwrap_err();
        assert_matches!(err, TdsError::DuplicateSimplices { .. });
    }

    #[test]
    fn test_tds_is_valid_passes_for_valid_simplex() {
        let verts = initial_simplex_vertices_3d();
        let dt = DelaunayTriangulation::try_new(&verts).unwrap();
        assert!(dt.tds().is_valid().is_ok());
        assert!(dt.tds().validate().is_ok());
    }

    #[test]
    fn test_tds_partial_eq_different_structures_not_equal() {
        let verts_a = [
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let verts_b = [
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([2.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 2.0]).unwrap(),
        ];
        let dt_a: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&verts_a).unwrap();
        let dt_b: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&verts_b).unwrap();
        assert_ne!(dt_a.tds(), dt_b.tds());
    }

    #[test]
    fn test_clear_all_neighbors_and_rebuild() {
        // Use 5 vertices so there are multiple simplices with actual neighbor pointers
        let vertices = [
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.5, 0.5, 0.5]).unwrap(),
        ];
        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let mut tds = dt.tds().clone();
        assert!(tds.number_of_simplices() > 1);

        // Multi-simplex: simplices that share facets have Some(neighbor) entries
        let has_any_neighbor = tds.simplices().any(|(_, simplex)| {
            simplex
                .neighbors()
                .is_some_and(|mut nb| nb.any(|neighbor| neighbor.is_some()))
        });
        assert!(has_any_neighbor);

        tds.clear_all_neighbors();

        // After clearing, no simplex should have neighbors
        for (_, simplex) in tds.simplices() {
            assert!(simplex.neighbors().is_none());
        }
    }

    #[test]
    fn test_build_facet_to_simplices_map_basic() {
        let verts = initial_simplex_vertices_3d();
        let dt = DelaunayTriangulation::try_new(&verts).unwrap();
        let tds = dt.tds();

        let facet_map = tds.build_facet_to_simplices_map().unwrap();
        // A single tetrahedron in 3D has 4 facets, all boundary (degree 1)
        assert_eq!(facet_map.len(), 4);
        for handles in facet_map.values() {
            assert_eq!(handles.len(), 1);
        }
    }

    #[test]
    fn test_find_simplices_containing_vertex_by_key() {
        let vertices = [
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.5, 0.5, 0.5]).unwrap(),
        ];
        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let tds = dt.tds();

        // Every vertex should be in at least one simplex
        for (vk, _) in tds.vertices() {
            let simplices = tds.find_simplices_containing_vertex_by_key(vk);
            assert!(
                !simplices.is_empty(),
                "Vertex {vk:?} should be in at least one simplex"
            );
        }

        // Stale key should return empty set
        let stale_key = VertexKey::from(KeyData::from_ffi(0xDEAD));
        assert!(
            tds.find_simplices_containing_vertex_by_key(stale_key)
                .is_empty()
        );
    }

    #[test]
    fn test_validation_report_accumulates_violations() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        // Create a simplex, then corrupt the UUID mapping
        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
        )
        .unwrap();
        // Corrupt: add a stray UUID mapping pointing to a non-existent key
        tds.uuid_to_simplex_key
            .insert(Uuid::new_v4(), SimplexKey::from(KeyData::from_ffi(0xBAD)));

        let report = tds.validation_report().unwrap_err();
        assert!(!report.is_empty());
        assert!(
            report
                .violations
                .iter()
                .any(|v| v.kind == InvariantKind::SimplexMappings),
            "Expected SimplexMappings violation"
        );
    }

    // =========================================================================
    // INSERT SIMPLEX WITH MAPPING: ERROR PATHS
    // =========================================================================

    #[test]
    fn test_insert_simplex_with_mapping_registers_uuid_mapping() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex = Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap();
        let simplex_uuid = simplex.uuid();
        let ck = tds.insert_simplex_with_mapping(simplex).unwrap();

        // UUID mapping should resolve back to the same key.
        assert_eq!(tds.simplex_key_from_uuid(&simplex_uuid), Some(ck));
        assert_eq!(tds.number_of_simplices(), 1);
    }

    #[test]
    fn test_insert_simplex_with_mapping_rejects_missing_vertex() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        // Use a stale key that doesn't exist in the TDS.
        let stale = VertexKey::from(KeyData::from_ffi(0xDEAD));

        let simplex = Simplex::try_new_with_data(vec![v0, v1, stale], None).unwrap();
        let err = tds.insert_simplex_with_mapping(simplex).unwrap_err();
        assert_matches!(
            err,
            TdsConstructionError::ValidationError(TdsError::VertexNotFound { .. })
        );
    }

    #[test]
    fn test_insert_simplex_with_mapping_trusted_vertices_rejects_missing_vertex() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let stale = VertexKey::from(KeyData::from_ffi(0xDEAD));

        let simplex = Simplex::try_new_with_data(vec![v0, v1, stale], None).unwrap();
        let err = tds
            .insert_simplex_with_mapping_trusted_vertices(simplex)
            .unwrap_err();
        assert_matches!(
            err,
            TdsConstructionError::ValidationError(TdsError::VertexNotFound { .. })
        );
    }

    #[test]
    fn test_insert_simplex_with_mapping_rejects_duplicate_uuid() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex_a = Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap();
        let uuid_a = simplex_a.uuid();
        tds.insert_simplex_with_mapping(simplex_a).unwrap();

        // Create a second simplex with the same UUID.
        let simplex_b = Simplex::try_new_with_uuid(vec![v0, v1, v2], uuid_a, None).unwrap();
        let err = tds.insert_simplex_with_mapping(simplex_b).unwrap_err();
        assert_matches!(err, TdsConstructionError::DuplicateUuid { .. });
    }

    #[test]
    fn test_insert_simplex_rejects_candidate_periodic_offset_count_mismatch() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let mut candidate = Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap();
        let candidate_uuid = candidate.uuid();
        let generation_before = tds.generation();
        candidate.periodic_vertex_offsets = Some(vec![[0_i8, 0_i8], [0_i8, 0_i8]].into());

        let err = tds.insert_simplex_with_mapping(candidate).unwrap_err();

        assert_matches!(
            err,
            TdsConstructionError::ValidationError(TdsError::DimensionMismatch {
                expected: 3,
                actual: 2,
                ..
            })
        );
        assert_eq!(tds.number_of_simplices(), 0);
        assert_eq!(tds.generation(), generation_before);
        assert_eq!(tds.simplex_key_from_uuid(&candidate_uuid), None);
    }

    #[test]
    fn test_insert_simplex_propagates_existing_periodic_facet_key_error() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();

        let existing = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        tds.simplex_mut(existing)
            .unwrap()
            .set_periodic_vertex_offsets(vec![[-128_i8, 0_i8], [127_i8, 0_i8], [0_i8, 0_i8]])
            .unwrap();

        let candidate = Simplex::try_new_with_data(vec![v0, v1, v3], None).unwrap();
        let candidate_uuid = candidate.uuid();
        let generation_before = tds.generation();

        let err = tds.insert_simplex_with_mapping(candidate).unwrap_err();

        assert_matches!(
            err,
            TdsConstructionError::ValidationError(TdsError::InconsistentDataStructure {
                message
            }) if message.contains("Failed to derive periodic facet key")
        );
        assert_eq!(tds.number_of_simplices(), 1);
        assert_eq!(tds.generation(), generation_before);
        assert_eq!(tds.simplex_key_from_uuid(&candidate_uuid), None);
    }

    #[test]
    fn test_insert_simplex_with_mapping_rejects_duplicate_simplex_without_mutation() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
        )
        .unwrap();
        let candidate = Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap();
        let candidate_uuid = candidate.uuid();
        let generation_before = tds.generation();

        let err = tds.insert_simplex_with_mapping(candidate).unwrap_err();

        assert_matches!(
            err,
            TdsConstructionError::ValidationError(TdsError::DuplicateSimplices { .. })
        );
        assert_eq!(tds.number_of_simplices(), 1);
        assert_eq!(tds.generation(), generation_before);
        assert_eq!(tds.simplex_key_from_uuid(&candidate_uuid), None);
    }

    #[test]
    fn test_insert_simplex_with_mapping_rejects_third_incident_facet_without_mutation() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, -1.0]).unwrap(),
            )
            .unwrap();
        let v4 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();

        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
        )
        .unwrap();
        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v0, v1, v3], None).unwrap(),
        )
        .unwrap();
        let candidate = Simplex::try_new_with_data(vec![v0, v1, v4], None).unwrap();
        let candidate_uuid = candidate.uuid();
        let generation_before = tds.generation();

        let err = tds.insert_simplex_with_mapping(candidate).unwrap_err();

        match err {
            TdsConstructionError::ValidationError(TdsError::FacetSharingViolation {
                existing_incident_count,
                attempted_incident_count,
                max_incident_count,
                candidate_simplex_uuid,
                candidate_facet_index,
                ..
            }) => {
                assert_eq!(existing_incident_count, 2);
                assert_eq!(attempted_incident_count, 3);
                assert_eq!(max_incident_count, 2);
                assert_eq!(candidate_simplex_uuid, candidate_uuid);
                assert_eq!(candidate_facet_index, 2);
            }
            other => panic!("expected structured facet-sharing violation, got {other:?}"),
        }
        assert_eq!(tds.number_of_simplices(), 2);
        assert_eq!(tds.generation(), generation_before);
        assert_eq!(tds.simplex_key_from_uuid(&candidate_uuid), None);
    }

    /// Verifies removed simplices are absent from subsequent insertion preflight scans.
    #[test]
    fn test_removed_simplices_do_not_block_future_simplex_insertions() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, -1.0]).unwrap(),
            )
            .unwrap();
        let v4 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();

        let removed = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        let second = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v3], None).unwrap(),
            )
            .unwrap();

        assert_eq!(tds.remove_simplices_by_keys(&[removed]), 1);

        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
        )
        .expect("removed duplicate simplex should not block reinsertion");
        assert_eq!(tds.remove_simplices_by_keys(&[second]), 1);
        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v0, v1, v4], None).unwrap(),
        )
        .expect("removed incident simplex should not block later facet sharing");
    }

    // =========================================================================
    // GET SIMPLEX VERTICES: ERROR PATH
    // =========================================================================

    #[test]
    fn test_simplex_vertices_errors_on_missing_vertex_key() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let ck = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();

        // Corrupt: remove a vertex that the simplex references.
        tds.vertices.remove(v2);
        tds.uuid_to_vertex_key.retain(|_, &mut vk| vk != v2);

        let err = tds.simplex_vertices(ck).unwrap_err();
        assert_matches!(err, TdsError::VertexNotFound { .. });
    }

    // =========================================================================
    // VALIDATE VERTEX INCIDENCE: INCONSISTENT INCIDENT_SIMPLEX
    // =========================================================================

    #[test]
    fn test_validate_vertex_incidence_detects_inconsistent_incident_simplex() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let _ck = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        tds.assign_incident_simplices().unwrap();

        // Create a second simplex that does NOT contain v0, then point v0 at it.
        let v3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([2.0, 2.0]).unwrap(),
            )
            .unwrap();
        let ck2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v1, v2, v3], None).unwrap(),
            )
            .unwrap();
        tds.vertex_mut(v0).unwrap().set_incident_simplex(Some(ck2));

        let err = tds.validate_vertex_incidence().unwrap_err();
        assert_matches!(err, TdsError::InconsistentDataStructure { .. });
    }

    #[test]
    fn test_validate_vertex_incidence_detects_dangling_simplex_key() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
        )
        .unwrap();
        tds.assign_incident_simplices().unwrap();

        // Point v0 at a non-existent simplex key.
        let dangling = SimplexKey::from(KeyData::from_ffi(0xDEAD));
        tds.vertex_mut(v0)
            .unwrap()
            .set_incident_simplex(Some(dangling));

        let err = tds.validate_vertex_incidence().unwrap_err();
        assert_matches!(err, TdsError::SimplexNotFound { .. });
    }

    // =========================================================================
    // FIND SIMPLICES CONTAINING VERTEX: FAST PATH VS FALLBACK
    // =========================================================================

    #[test]
    fn test_find_simplices_containing_vertex_fallback_when_no_incident_simplex() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
        )
        .unwrap();
        // Don't assign incident simplices — force fallback scan.
        let simplices = tds.find_simplices_containing_vertex_by_key(v0);
        assert_eq!(simplices.len(), 1);
    }

    #[test]
    fn test_find_simplices_containing_vertex_falls_back_on_unassigned_neighbor_slot() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([-1.0, 1.0]).unwrap(),
            )
            .unwrap();

        let first_simplex = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        let second_simplex = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v2, v3], None).unwrap(),
            )
            .unwrap();
        tds.assign_neighbors().unwrap();
        tds.assign_incident_simplices().unwrap();
        tds.vertex_mut(v0)
            .unwrap()
            .set_incident_simplex(Some(first_simplex));

        {
            let simplex = tds.simplex_mut(first_simplex).unwrap();
            simplex.ensure_neighbors_buffer_mut()[0] = NeighborSlot::Unassigned;
        }

        let simplices = tds.find_simplices_containing_vertex_by_key(v0);

        assert_eq!(simplices.len(), 2);
        assert!(simplices.contains(&first_simplex));
        assert!(simplices.contains(&second_simplex));
    }

    // =========================================================================
    // REMOVE SIMPLICES BY KEYS: BATCH REPAIR
    // =========================================================================

    #[test]
    fn test_remove_simplices_by_keys_batch_repairs_incidence_and_neighbors() {
        let vertices = [
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.5, 0.5, 0.5]).unwrap(),
        ];
        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let mut tds = dt.tds().clone();
        assert!(tds.number_of_simplices() > 1);

        // Remove the first simplex.
        let first_ck = tds.simplex_keys().next().unwrap();
        let removed = tds.remove_simplices_by_keys(&[first_ck]);
        assert_eq!(removed, 1);

        // All remaining vertex incident_simplex pointers should be valid.
        for (vk, v) in tds.vertices() {
            if let Some(ic) = v.incident_simplex() {
                assert!(
                    tds.contains_simplex(ic),
                    "Vertex {vk:?} has dangling incident_simplex after batch removal"
                );
            }
        }

        // No surviving simplex should have a neighbor pointer to the removed simplex.
        for (_, simplex) in tds.simplices() {
            if let Some(neighbors) = simplex.neighbors() {
                for nk in neighbors.flatten() {
                    assert_ne!(nk, first_ck, "Dangling neighbor pointer to removed simplex");
                }
            }
        }
    }

    // =========================================================================
    // ASSIGN INCIDENT SIMPLICES: ERROR ON DANGLING VERTEX KEY
    // =========================================================================

    #[test]
    fn test_assign_incident_simplices_errors_on_dangling_vertex_key() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let _ck = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();

        // Remove v2 from the vertex storage, leaving the simplex with a dangling reference.
        tds.vertices.remove(v2);
        tds.uuid_to_vertex_key.retain(|_, &mut vk| vk != v2);

        let err = tds.assign_incident_simplices().unwrap_err();
        assert_matches!(err.as_tds_error(), TdsError::VertexNotFound { .. });
    }

    // =========================================================================
    // NORMALIZE COHERENT ORIENTATION: SINGLE SIMPLEX (NO NEIGHBORS)
    // =========================================================================

    #[test]
    fn test_normalize_coherent_orientation_handles_single_simplex() {
        let verts = initial_simplex_vertices_3d();
        let dt = DelaunayTriangulation::try_new(&verts).unwrap();
        let mut tds = dt.tds().clone();
        assert_eq!(tds.number_of_simplices(), 1);

        // Single simplex with no neighbors: should succeed without flipping.
        assert!(tds.normalize_coherent_orientation().is_ok());
    }

    #[test]
    fn test_normalize_coherent_orientation_multi_simplex() {
        let vertices = [
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.5, 0.5, 0.5]).unwrap(),
        ];
        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let mut tds = dt.tds().clone();
        assert!(tds.number_of_simplices() > 1);

        // Should succeed for a valid multi-simplex triangulation.
        assert!(tds.normalize_coherent_orientation().is_ok());
    }

    // =========================================================================
    // VALIDATE SIMPLEX COORDINATE UNIQUENESS
    // =========================================================================

    #[test]
    fn test_validate_simplex_coordinate_uniqueness_passes_for_distinct_coords() {
        let verts = initial_simplex_vertices_3d();
        let dt = DelaunayTriangulation::try_new(&verts).unwrap();
        let tds = dt.tds();
        assert!(tds.validate_simplex_coordinate_uniqueness().is_ok());
    }

    // =========================================================================
    // GENERATION COUNTER
    // =========================================================================

    #[test]
    fn test_mark_topology_modified_bumps_generation() {
        let tds: Tds<(), (), 2> = Tds::empty();
        let gen_before = tds.generation();
        tds.mark_topology_modified();
        assert_eq!(tds.generation(), gen_before + 1);
    }

    // =========================================================================
    // REMOVE DUPLICATE SIMPLICES: WITH ACTUAL DUPLICATES
    // =========================================================================

    #[test]
    fn test_remove_duplicate_simplices_removes_actual_duplicates() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
        )
        .unwrap();
        // Insert a duplicate simplex (same vertex set, different UUID).
        tds.insert_simplex_bypassing_topology_checks_for_test(
            Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
        )
        .unwrap();
        assert_eq!(tds.number_of_simplices(), 2);
        let generation_before = tds.generation();

        let removed = tds.remove_duplicate_simplices().unwrap();
        assert_eq!(removed, 1);
        assert_eq!(tds.number_of_simplices(), 1);
        assert!(tds.generation() > generation_before);
        assert!(tds.is_valid().is_ok());
    }

    // =========================================================================
    // REMOVE SIMPLICES BY KEYS
    // =========================================================================

    #[test]
    fn test_remove_simplices_by_keys_returns_zero_for_missing() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let stale = SimplexKey::from(KeyData::from_ffi(0xDEAD));
        assert_eq!(tds.remove_simplices_by_keys(&[stale]), 0);
    }

    #[test]
    fn test_remove_simplices_by_keys_repairs_local_topology() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let removed_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        let surviving_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v2, v3], None).unwrap(),
            )
            .unwrap();
        tds.construction_state = TriangulationConstructionState::Constructed;
        tds.assign_neighbors().unwrap();
        tds.assign_incident_simplices().unwrap();

        assert!(
            tds.simplices
                .get(surviving_key)
                .unwrap()
                .neighbors()
                .unwrap()
                .any(|neighbor| neighbor == Some(removed_key))
        );

        let removed_uuid = tds.simplices.get(removed_key).unwrap().uuid();
        let removed_vertices = tds.simplices.get(removed_key).unwrap().vertices().to_vec();

        assert_eq!(tds.remove_simplices_by_keys(&[removed_key]), 1);

        assert_eq!(removed_vertices, vec![v0, v1, v2]);
        assert!(!tds.simplices.contains_key(removed_key));
        assert!(tds.simplices.contains_key(surviving_key));
        assert!(!tds.uuid_to_simplex_key.contains_key(&removed_uuid));
        if let Some(mut neighbors) = tds.simplices.get(surviving_key).unwrap().neighbors() {
            assert!(neighbors.all(|neighbor| neighbor != Some(removed_key)));
        }
        assert_ne!(
            tds.vertices.get(v0).unwrap().incident_simplex(),
            Some(removed_key)
        );
        assert_ne!(
            tds.vertices.get(v1).unwrap().incident_simplex(),
            Some(removed_key)
        );
        assert_ne!(
            tds.vertices.get(v2).unwrap().incident_simplex(),
            Some(removed_key)
        );
        assert_eq!(
            tds.vertices.get(v0).unwrap().incident_simplex(),
            Some(surviving_key)
        );
        assert_eq!(
            tds.vertices.get(v2).unwrap().incident_simplex(),
            Some(surviving_key)
        );
    }

    // =========================================================================
    // VALIDATE NEIGHBOR TOPOLOGY: ERROR PATHS
    // =========================================================================

    #[test]
    fn test_validate_neighbor_topology_rejects_wrong_length() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let ck = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();

        // Wrong length: 2 instead of D+1=3.
        let err = tds
            .validate_neighbor_topology(ck, &[None, None])
            .unwrap_err();
        assert_matches!(err, TdsError::InvalidNeighbors { .. });
    }

    // =========================================================================
    // SET VERTEX DATA / SET SIMPLEX DATA
    // =========================================================================

    #[test]
    fn test_set_vertex_data_replaces_existing() {
        let vertices: [Vertex<i32, 2>; 3] = [
            crate::core::vertex::Vertex::<_, _>::try_new_with_data([0.0, 0.0], 10i32).unwrap(),
            crate::core::vertex::Vertex::<_, _>::try_new_with_data([1.0, 0.0], 20).unwrap(),
            crate::core::vertex::Vertex::<_, _>::try_new_with_data([0.0, 1.0], 30).unwrap(),
        ];
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .build::<()>()
            .unwrap();
        let mut tds = dt.tds().clone();
        let key = tds.vertex_keys().next().unwrap();

        let prev = tds.set_vertex_data(key, Some(99));
        assert!(prev.unwrap().is_some()); // had data before
        assert_eq!(tds.vertex(key).unwrap().data, Some(99));
    }

    #[test]
    fn test_set_vertex_data_on_no_data_vertex() {
        // Vertices without data have U = (), so set_vertex_data sets ().
        let vertices = [
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let mut tds = dt.tds().clone();
        let key = tds.vertex_keys().next().unwrap();

        let prev = tds.set_vertex_data(key, Some(()));
        // Vertices constructed without explicit data have data = None
        assert_eq!(prev, Some(None));
        assert_eq!(tds.vertex(key).unwrap().data, Some(()));
    }

    #[test]
    fn test_set_vertex_data_invalid_key_returns_none() {
        let mut tds: Tds<i32, (), 2> = Tds::empty();
        let stale = VertexKey::from(KeyData::from_ffi(0xDEAD));
        assert!(tds.set_vertex_data(stale, Some(1)).is_none());
    }

    #[test]
    fn test_set_simplex_data_on_empty_simplex() {
        let vertices = [
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .build::<i32>()
            .unwrap();
        let mut tds = dt.tds().clone();
        let key = tds.simplex_keys().next().unwrap();

        let prev = tds.set_simplex_data(key, Some(42));
        assert_eq!(prev, Some(None)); // key found, no previous data
        assert_eq!(tds.simplex(key).unwrap().data, Some(42));
    }

    #[test]
    fn test_set_simplex_data_replaces_existing() {
        let vertices = [
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .build::<i32>()
            .unwrap();
        let mut tds = dt.tds().clone();
        let key = tds.simplex_keys().next().unwrap();

        tds.set_simplex_data(key, Some(1));
        let prev = tds.set_simplex_data(key, Some(2));
        assert_eq!(prev, Some(Some(1)));
        assert_eq!(tds.simplex(key).unwrap().data, Some(2));
    }

    #[test]
    fn test_set_simplex_data_invalid_key_returns_none() {
        let mut tds: Tds<(), i32, 2> = Tds::empty();
        let stale = SimplexKey::from(KeyData::from_ffi(0xDEAD));
        assert!(tds.set_simplex_data(stale, Some(1)).is_none());
    }

    #[test]
    fn test_set_vertex_data_preserves_triangulation_validity() {
        let vertices: [Vertex<i32, 2>; 3] = [
            crate::core::vertex::Vertex::<_, _>::try_new_with_data([0.0, 0.0], 1i32).unwrap(),
            crate::core::vertex::Vertex::<_, _>::try_new_with_data([1.0, 0.0], 2).unwrap(),
            crate::core::vertex::Vertex::<_, _>::try_new_with_data([0.0, 1.0], 3).unwrap(),
        ];
        let mut dt = DelaunayTriangulationBuilder::new(&vertices)
            .build::<()>()
            .unwrap();

        // Mutate every vertex's data through the DT wrapper.
        let keys: Vec<_> = dt.vertices().map(|(k, _)| k).collect();
        for (key, i) in keys.iter().zip(0i32..) {
            dt.set_vertex_data(*key, Some(i * 100));
        }

        // Triangulation must remain fully valid.
        assert!(dt.validate().is_ok());

        // Verify all data was updated.
        for (key, i) in keys.iter().zip(0i32..) {
            let v = dt.tds().vertex(*key).unwrap();
            assert_eq!(v.data, Some(i * 100));
        }
    }

    #[test]
    fn test_set_simplex_data_preserves_triangulation_validity() {
        let vertices = [
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.5, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.5, 0.5]).unwrap(),
        ];
        let mut dt = DelaunayTriangulationBuilder::new(&vertices)
            .build::<i32>()
            .unwrap();
        assert!(dt.number_of_simplices() > 1);

        // Mutate every simplex's data through the DT wrapper.
        let keys: Vec<_> = dt.simplices().map(|(k, _)| k).collect();
        for (key, i) in keys.iter().zip(0i32..) {
            dt.set_simplex_data(*key, Some(i));
        }

        // Triangulation must remain fully valid.
        assert!(dt.validate().is_ok());

        // Verify all data was updated.
        for (key, i) in keys.iter().zip(0i32..) {
            let c = dt.tds().simplex(*key).unwrap();
            assert_eq!(c.data, Some(i));
        }
    }

    #[test]
    fn test_set_vertex_data_via_delaunay_wrapper() {
        let vertices: [Vertex<i32, 2>; 3] = [
            crate::core::vertex::Vertex::<_, _>::try_new_with_data([0.0, 0.0], 10i32).unwrap(),
            crate::core::vertex::Vertex::<_, _>::try_new_with_data([1.0, 0.0], 20).unwrap(),
            crate::core::vertex::Vertex::<_, _>::try_new_with_data([0.0, 1.0], 30).unwrap(),
        ];
        let mut dt = DelaunayTriangulationBuilder::new(&vertices)
            .build::<()>()
            .unwrap();
        let key = dt.vertices().next().unwrap().0;

        // Set via Delaunay wrapper
        let prev = dt.set_vertex_data(key, Some(99));
        assert!(prev.unwrap().is_some());
        assert_eq!(dt.tds().vertex(key).unwrap().data, Some(99));

        // Clear via Delaunay wrapper
        let prev = dt.set_vertex_data(key, None);
        assert_eq!(prev, Some(Some(99)));
        assert_eq!(dt.tds().vertex(key).unwrap().data, None);
    }

    #[test]
    fn test_set_simplex_data_via_delaunay_wrapper() {
        let vertices = [
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let mut dt = DelaunayTriangulationBuilder::new(&vertices)
            .build::<i32>()
            .unwrap();
        let key = dt.simplices().next().unwrap().0;

        // Set via Delaunay wrapper
        let prev = dt.set_simplex_data(key, Some(42));
        assert_eq!(prev, Some(None));
        assert_eq!(dt.tds().simplex(key).unwrap().data, Some(42));

        // Clear via Delaunay wrapper
        let prev = dt.set_simplex_data(key, None);
        assert_eq!(prev, Some(Some(42)));
        assert_eq!(dt.tds().simplex(key).unwrap().data, None);
    }

    #[test]
    fn test_set_data_via_dt_does_not_invalidate_locate_hint() {
        let vertices: [Vertex<i32, 2>; 3] = [
            crate::core::vertex::Vertex::<_, _>::try_new_with_data([0.0, 0.0], 0i32).unwrap(),
            crate::core::vertex::Vertex::<_, _>::try_new_with_data([1.0, 0.0], 0).unwrap(),
            crate::core::vertex::Vertex::<_, _>::try_new_with_data([0.0, 1.0], 0).unwrap(),
        ];
        let mut dt = DelaunayTriangulationBuilder::new(&vertices)
            .build::<()>()
            .unwrap();

        // Insert a new vertex so the locate hint is populated.
        let extra =
            crate::core::vertex::Vertex::<_, _>::try_new_with_data([0.25, 0.25], 0i32).unwrap();
        dt.insert(extra).unwrap();

        // Data mutation should NOT clear the insertion hint.
        let key = dt.vertices().next().unwrap().0;
        let prev = dt.set_vertex_data(key, Some(999));
        assert!(prev.is_some(), "set_vertex_data should find the key");
        assert_eq!(
            dt.tds().vertex(key).unwrap().data,
            Some(999),
            "stored value should reflect the mutation"
        );

        // A subsequent insert should still succeed (hint not invalidated).
        let another =
            crate::core::vertex::Vertex::<_, _>::try_new_with_data([0.75, 0.1], 0i32).unwrap();
        assert!(dt.insert(another).is_ok());
        assert!(dt.validate().is_ok());
    }
}
