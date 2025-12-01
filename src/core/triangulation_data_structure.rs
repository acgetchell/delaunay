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
//! | **Delaunay Property** | `bowyer_watson::find_bad_cells()` | Empty circumsphere test using `insphere()` |
//! | **Facet Sharing** | `validate_facet_sharing()` | Each facet shared by ≤ 2 cells |
//! | **No Duplicate Cells** | `validate_no_duplicate_cells()` | No cells with identical vertex sets |
//! | **Neighbor Consistency** | `validate_neighbors()` | Mutual neighbor relationships |
//! | **Cell Validity** | `CellBuilder::validate()` (vertex count) + `cell.is_valid()` (comprehensive) | Construction + runtime validation |
//! | **Vertex Validity** | `Point::from()` (coordinates) + UUID auto-gen + `vertex.is_valid()` | Construction + runtime validation |
//!
//! The Delaunay property is enforced **proactively** during construction, while structural
//! invariants are enforced **reactively** through validation methods.
//!
//! # Examples
//!
//! ## Creating a 3D Triangulation
//!
//! ```rust
//! use delaunay::prelude::*;
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
//! assert!(dt.is_valid().is_ok());
//! ```
//!
//! ## Adding Vertices to Existing Triangulation
//!
//! ```rust
//! use delaunay::prelude::*;
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
//! let new_vertex = vertex!([0.5, 0.5, 0.5]);
//! dt.insert(new_vertex).unwrap();
//!
//! assert_eq!(dt.number_of_vertices(), 5);
//! assert!(dt.is_valid().is_ok());
//! ```
//!
//! ## 4D Triangulation
//!
//! ```rust
//! use delaunay::prelude::*;
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
//! assert!(dt_4d.is_valid().is_ok());
//! ```
//!
//! # References
//!
//! - [CGAL Triangulation Documentation](https://doc.cgal.org/latest/Triangulation/index.html)
//! - Bowyer, A. "Computing Dirichlet tessellations." The Computer Journal 24.2 (1981): 162-166
//! - Watson, D.F. "Computing the n-dimensional Delaunay tessellation with application to Voronoi polytopes." The Computer Journal 24.2 (1981): 167-172
//! - de Berg, M., et al. "Computational Geometry: Algorithms and Applications." 3rd ed. Springer-Verlag, 2008

// =============================================================================
// IMPORTS
// =============================================================================

// Standard library imports
use std::{
    cmp::Ordering as CmpOrdering,
    fmt::{self, Debug},
    iter::Sum,
    marker::PhantomData,
    ops::{AddAssign, SubAssign},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

// External crate imports
use serde::{
    Deserialize, Deserializer, Serialize,
    de::{self, MapAccess, Visitor},
};
use slotmap::new_key_type;
use thiserror::Error;
use uuid::Uuid;

// Crate-internal imports
use crate::core::collections::{
    CellKeySet, CellRemovalBuffer, CellVertexUuidBuffer, CellVerticesMap, Entry, FacetToCellsMap,
    FastHashMap, MAX_PRACTICAL_DIMENSION_SIZE, NeighborBuffer, SmallBuffer, StorageMap,
    UuidToCellKeyMap, UuidToVertexKeyMap, ValidCellsBuffer, VertexKeyBuffer, VertexKeySet,
    VertexToCellsMap, fast_hash_map_with_capacity,
};
use crate::core::util::DelaunayValidationError;
use crate::geometry::traits::coordinate::CoordinateScalar;

// num-traits imports
use num_traits::cast::NumCast;

// Parent module imports
use super::{
    cell::{Cell, CellValidationError},
    facet::{FacetHandle, facet_key_from_vertices},
    traits::data_type::DataType,
    util::usize_to_u8,
    vertex::Vertex,
};

// =============================================================================
// CONSTRUCTION STATE TYPES
// =============================================================================

/// Represents the construction state of a triangulation.
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

/// Errors that can occur during triangulation construction.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum TriangulationConstructionError {
    /// Failed to create a cell during triangulation construction.
    #[error("Failed to create cell during construction: {message}")]
    FailedToCreateCell {
        /// Description of the cell creation failure.
        message: String,
    },
    /// Insufficient vertices to create a triangulation.
    #[error("Insufficient vertices for {dimension}D triangulation: {source}")]
    InsufficientVertices {
        /// The dimension that was attempted.
        dimension: usize,
        /// The underlying cell validation error.
        source: CellValidationError,
    },
    /// Failed to add vertex during triangulation construction.
    #[error("Failed to add vertex during construction: {message}")]
    FailedToAddVertex {
        /// Description of the vertex addition failure.
        message: String,
    },
    /// Geometric degeneracy prevents triangulation construction.
    #[error("Geometric degeneracy encountered during construction: {message}")]
    GeometricDegeneracy {
        /// Description of the degeneracy issue.
        message: String,
    },
    /// Validation error during construction.
    #[error("Validation error during construction: {0}")]
    ValidationError(#[from] TriangulationValidationError),
    /// Attempted to insert an entity with a UUID that already exists.
    #[error("Duplicate UUID: {entity:?} with UUID {uuid} already exists")]
    DuplicateUuid {
        /// The type of entity.
        entity: EntityKind,
        /// The UUID that was duplicated.
        uuid: Uuid,
    },
    /// Attempted to insert a vertex with coordinates that already exist.
    #[error(
        "Duplicate coordinates: vertex with coordinates {coordinates} already exists in the triangulation"
    )]
    DuplicateCoordinates {
        /// String representation of the duplicate coordinates.
        coordinates: String,
    },
}

/// Represents the type of entity in the triangulation.
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
/// use delaunay::prelude::*;
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
pub enum TriangulationValidationError {
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
    /// The triangulation violates the Delaunay empty circumsphere property.
    #[error("Delaunay invariant violated: {message}")]
    DelaunayViolation {
        /// Human-readable description of the Delaunay violation(s).
        message: String,
    },
    /// Finalization failed during triangulation operations.
    #[error("Finalization failed: {message}")]
    FinalizationFailed {
        /// Description of the finalization failure, including underlying error details.
        message: String,
    },
}

/// Classifies the kind of triangulation invariant that failed during validation.
///
/// This is used by [`TriangulationValidationReport`] to group related errors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InvariantKind {
    /// Vertex UUID↔key mapping invariants.
    VertexMappings,
    /// Cell UUID↔key mapping invariants.
    CellMappings,
    /// No duplicate maximal cells with identical vertex sets.
    DuplicateCells,
    /// Per-cell validity (vertex count, duplicate vertices, nil UUID, etc.).
    CellValidity,
    /// Facet sharing invariants (each facet shared by at most 2 cells).
    FacetSharing,
    /// Neighbor topology and mutual-consistency invariants.
    NeighborConsistency,
    /// Delaunay empty circumsphere invariant.
    Delaunay,
}

/// A single invariant violation recorded during validation diagnostics.
#[derive(Clone, Debug)]
pub struct InvariantViolation {
    /// The kind of invariant that failed.
    pub kind: InvariantKind,
    /// The detailed validation error explaining the failure.
    pub error: TriangulationValidationError,
}

/// Aggregate report of one or more validation failures.
///
/// This is returned by [`Tds::validation_report`] to surface all failed
/// invariants at once for debugging and test diagnostics.
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

/// Configuration options for [`Tds::validation_report`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ValidationOptions {
    /// Whether to validate the Delaunay empty circumsphere invariant.
    pub check_delaunay: bool,
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
    pub struct VertexKey;
}

new_key_type! {
    /// Key type for accessing cells in the storage map.
    ///
    /// This creates a unique, type-safe identifier for cells stored in the
    /// triangulation's cell storage. Each CellKey corresponds to exactly
    /// one cell and provides efficient, stable access even as cells are
    /// added or removed during triangulation operations.
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
/// use delaunay::prelude::*;
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
    T: CoordinateScalar,
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
// NOTE: Currently T: CoordinateScalar is required because Cell and Vertex structs
// have this bound. In Phase 1.2, we will relax Cell/Vertex bounds to complete
// the separation.

impl<T, U, V, const D: usize> Tds<T, U, V, D>
where
    T: CoordinateScalar, // TODO: Remove in Phase 1.2 after relaxing Cell/Vertex bounds
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
    /// # Errors
    ///
    /// Returns `TriangulationValidationError` if neighbor assignment fails due to inconsistent
    /// data structures or invalid facet sharing patterns.
    pub fn assign_neighbors(&mut self) -> Result<(), TriangulationValidationError> {
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
                TriangulationValidationError::VertexKeyRetrievalFailed {
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
                    return Err(TriangulationValidationError::InconsistentDataStructure {
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
                return Err(TriangulationValidationError::InconsistentDataStructure {
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
                TriangulationValidationError::InconsistentDataStructure {
                    message: format!("Cell key {cell_key1:?} not found in cell neighbors map"),
                }
            })?[vertex_index1] = Some(cell_key2);

            cell_neighbors.get_mut(&cell_key2).ok_or_else(|| {
                TriangulationValidationError::InconsistentDataStructure {
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
    /// use delaunay::prelude::*;
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
    /// use delaunay::prelude::*;
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
    /// use delaunay::prelude::*;
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
    pub fn vertex_keys(&self) -> impl Iterator<Item = VertexKey> + '_ {
        self.vertices.keys()
    }

    /// Returns an iterator over all cell keys in the triangulation.
    ///
    /// # Returns
    ///
    /// An iterator over `CellKey` values.
    pub fn cell_keys(&self) -> impl Iterator<Item = CellKey> + '_ {
        self.cells.keys()
    }

    /// Returns a reference to a cell by its key.
    ///
    /// # Returns
    ///
    /// `Some(&Cell)` if the key exists, `None` otherwise.
    #[must_use]
    pub fn get_cell(&self, key: CellKey) -> Option<&Cell<T, U, V, D>> {
        self.cells.get(key)
    }

    /// Checks if a cell key exists in the triangulation.
    ///
    /// # Returns
    ///
    /// `true` if the key exists, `false` otherwise.
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
    /// use delaunay::prelude::*;
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
    /// use delaunay::prelude::*;
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
    /// use delaunay::prelude::*;
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
    /// use delaunay::prelude::*;
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
    /// use delaunay::prelude::*;
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
    /// use delaunay::prelude::*;
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
    #[inline]
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Relaxed)
    }
}

// =============================================================================
// QUERY OPERATIONS
// =============================================================================

impl<T, U, V, const D: usize> Tds<T, U, V, D>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
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
    #[allow(dead_code)]
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
    /// for UUID uniqueness. For public API use where coordinate uniqueness is required,
    /// prefer using the `add()` method instead, which enforces coordinate-uniqueness.
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
    /// Returns `TriangulationConstructionError::DuplicateUuid` if a vertex with the
    /// same UUID already exists in the triangulation.
    ///
    /// # Examples
    ///
    ///
    /// See the unit tests for usage examples of this pub(crate) method.
    pub(crate) fn insert_vertex_with_mapping(
        &mut self,
        vertex: Vertex<T, U, D>,
    ) -> Result<VertexKey, TriangulationConstructionError> {
        let vertex_uuid = vertex.uuid();

        // Use Entry API for atomic check-and-insert
        match self.uuid_to_vertex_key.entry(vertex_uuid) {
            Entry::Occupied(_) => Err(TriangulationConstructionError::DuplicateUuid {
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
    /// Returns `TriangulationConstructionError::DuplicateUuid` if a cell with the
    /// same UUID already exists in the triangulation.
    ///
    /// # Examples
    ///
    ///
    /// See the unit tests for usage examples of this pub(crate) method.
    pub(crate) fn insert_cell_with_mapping(
        &mut self,
        cell: Cell<T, U, V, D>,
    ) -> Result<CellKey, TriangulationConstructionError> {
        // Phase 3A: Validate structural invariants using vertices
        debug_assert_eq!(
            cell.number_of_vertices(),
            D + 1,
            "Cell should have exactly D+1 vertices for quick failure in dev"
        );
        if cell.number_of_vertices() != D + 1 {
            return Err(TriangulationConstructionError::ValidationError(
                TriangulationValidationError::InconsistentDataStructure {
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
                return Err(TriangulationConstructionError::ValidationError(
                    TriangulationValidationError::InconsistentDataStructure {
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
            Entry::Occupied(_) => Err(TriangulationConstructionError::DuplicateUuid {
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
    /// or a `TriangulationValidationError` if the cell doesn't exist or vertices are missing.
    ///
    /// # Errors
    ///
    /// Returns a `TriangulationValidationError` if:
    /// - The cell with the given key doesn't exist
    /// - A vertex key from the cell doesn't exist in the vertex storage (TDS corruption)
    ///
    /// # Performance
    ///
    /// This uses direct storage map access with O(1) key lookup for the cell and O(D)
    /// validation for vertex keys. Uses stack-allocated buffer for D ≤ 7 to avoid heap
    /// allocation in the hot path.
    #[inline]
    pub fn get_cell_vertices(
        &self,
        cell_key: CellKey,
    ) -> Result<VertexKeyBuffer, TriangulationValidationError> {
        let cell = self.cells.get(cell_key).ok_or_else(|| {
            TriangulationValidationError::InconsistentDataStructure {
                message: format!("Cell key {cell_key:?} not found in cells storage map"),
            }
        })?;

        // Phase 3A: Cell now stores vertex keys directly
        // Validate and collect keys in one pass to avoid redundant iteration
        let cell_vertices = cell.vertices();
        let mut keys = VertexKeyBuffer::with_capacity(cell_vertices.len());
        for (idx, &vertex_key) in cell_vertices.iter().enumerate() {
            if !self.vertices.contains_key(vertex_key) {
                return Err(TriangulationValidationError::InconsistentDataStructure {
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
    /// use delaunay::prelude::*;
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
    /// use delaunay::prelude::*;
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
    /// use delaunay::prelude::*;
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
    /// use delaunay::prelude::*;
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
    /// use delaunay::prelude::*;
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
    /// use delaunay::prelude::*;
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
    /// This method is optimized for removing multiple cells at once,
    /// updating the generation counter only once.
    ///
    /// # Safety Warning
    ///
    /// This method only removes the cells and updates the UUID→Key mappings.
    /// It does NOT maintain topology consistency. The caller MUST:
    /// 1. Call `assign_neighbors()` to rebuild neighbor relationships
    /// 2. Call `assign_incident_cells()` to update vertex-cell associations
    ///
    /// Failure to do so will leave the triangulation in an inconsistent state.
    ///
    /// # Arguments
    ///
    /// * `cell_keys` - The keys of cells to remove
    ///
    /// # Returns
    ///
    /// The number of cells successfully removed.
    pub fn remove_cells_by_keys(&mut self, cell_keys: &[CellKey]) -> usize {
        if cell_keys.is_empty() {
            return 0;
        }

        // Build a set for O(1) lookup when clearing neighbor references.
        let cells_to_remove: CellKeySet = cell_keys.iter().copied().collect();

        // Clear neighbor references in remaining cells that point to cells being removed.
        // This prevents dangling neighbor pointers to non-existent cells.
        for (cell_key, cell) in &mut self.cells {
            // Skip cells that will be removed
            if cells_to_remove.contains(&cell_key) {
                continue;
            }

            if let Some(neighbors) = &mut cell.neighbors {
                for neighbor_slot in neighbors.iter_mut() {
                    if neighbor_slot
                        .as_ref()
                        .is_some_and(|neighbor_key| cells_to_remove.contains(neighbor_key))
                    {
                        *neighbor_slot = None; // Clear dangling reference (becomes boundary)
                    }
                }
            }
        }

        let mut removed_count = 0;

        for &cell_key in cell_keys {
            if let Some(removed_cell) = self.cells.remove(cell_key) {
                // Also remove from UUID mapping
                self.uuid_to_cell_key.remove(&removed_cell.uuid());
                removed_count += 1;
            }
        }

        // Bump generation once for all removals
        if removed_count > 0 {
            self.bump_generation();
        }

        removed_count
    }

    /// Finds all cells containing a specific vertex.
    ///
    /// This is an internal helper used by `remove_vertex` and `rollback_vertex_insertion`
    /// to efficiently collect cells that need to be removed when a vertex is deleted.
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
    /// - Time complexity: O(n) where n is the number of cells
    /// - Space: Stack-allocated for ≤16 results, heap fallback for larger vertex stars
    ///
    /// # Related
    ///
    /// This method is optimized for **removal/rollback operations** where the result
    /// is consumed once for batch removal. For **set operations** (intersection, union,
    /// membership testing), use `find_cells_containing_vertex_by_key` which returns a
    /// `CellKeySet` (hash set) optimized for O(1) lookups and set operations.
    ///
    /// The two methods serve different use cases:
    /// - This method: Stack-allocated buffer → optimized for sequential removal
    /// - `find_cells_containing_vertex_by_key`: Hash set → optimized for set operations
    fn find_cells_containing_vertex(&self, vertex_key: VertexKey) -> CellRemovalBuffer {
        self.cells()
            .filter_map(|(cell_key, cell)| {
                if cell.contains_vertex(vertex_key) {
                    Some(cell_key)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Removes a vertex and all cells containing it, maintaining data structure consistency.
    ///
    /// This is an atomic operation that:
    /// 1. Finds all cells containing the vertex
    /// 2. Removes all such cells
    /// 3. Rebuilds vertex-cell incidence to prevent dangling `incident_cell` pointers
    /// 4. Removes the vertex itself
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
    /// `Ok(usize)` with the number of cells that were removed along with the vertex,
    /// or `Err(TriangulationValidationError)` if incident cell assignment fails.
    ///
    /// # Errors
    ///
    /// Returns `TriangulationValidationError` if the vertex-cell incidence cannot be rebuilt
    /// after removing cells. This indicates a corrupted data structure.
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
        vertex: &Vertex<T, U, D>,
    ) -> Result<usize, TriangulationValidationError> {
        // Find the vertex key
        let Some(vertex_key) = self.vertex_key_from_uuid(&vertex.uuid()) else {
            return Ok(0); // Vertex not found, nothing to remove
        };

        // Find all cells containing this vertex
        let cells_to_remove = self.find_cells_containing_vertex(vertex_key);

        // Remove all cells containing the vertex. Neighbor references in surviving
        // cells that pointed to these cells are cleared by `remove_cells_by_keys`.
        let cells_removed = self.remove_cells_by_keys(&cells_to_remove);

        // Rebuild vertex incidence before removing the vertex to ensure surviving vertices
        // no longer point at the deleted cells. This prevents dangling incident_cell references.
        // Any vertex whose incident_cell previously referenced one of the removed cells will
        // have its pointer updated to a valid remaining cell (or None if isolated).
        self.assign_incident_cells()?;

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
    /// use delaunay::prelude::*;
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
    /// * `Err(TriangulationValidationError)` with details about which neighbors violate the invariant
    ///
    /// # Use Cases
    ///
    /// - Called by `set_neighbors_by_key()` to enforce correctness
    /// - Can be called by `is_valid()` to check entire triangulation
    /// - Useful during incremental construction to identify cells needing repair
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
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tds = dt.tds();
    /// let (cell_key, cell) = tds.cells().next().unwrap();
    /// let neighbors = tds.find_neighbors_by_key(cell_key);
    ///
    /// // Validate specific cell's neighbors
    /// if let Err(e) = tds.validate_neighbor_topology(cell_key, &neighbors) {
    ///     eprintln!("Cell {:?} has invalid neighbors: {}", cell_key, e);
    ///     // Fix the neighbors...
    /// }
    /// ```
    /// # Errors
    ///
    /// Returns `TriangulationValidationError` if topology validation fails.
    pub fn validate_neighbor_topology(
        &self,
        cell_key: CellKey,
        neighbors: &[Option<CellKey>],
    ) -> Result<(), TriangulationValidationError> {
        if neighbors.len() != D + 1 {
            return Err(TriangulationValidationError::InvalidNeighbors {
                message: format!(
                    "Neighbor vector length {} != D+1 ({})",
                    neighbors.len(),
                    D + 1
                ),
            });
        }

        let cell = self.cells.get(cell_key).ok_or_else(|| {
            TriangulationValidationError::InconsistentDataStructure {
                message: format!("Cell key {cell_key:?} not found"),
            }
        })?;

        let cell_vertices = cell.vertices();

        for (i, neighbor_key_opt) in neighbors.iter().enumerate() {
            if let Some(neighbor_key) = neighbor_key_opt {
                let neighbor = self.cells.get(*neighbor_key).ok_or_else(|| {
                    TriangulationValidationError::InvalidNeighbors {
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
                    return Err(TriangulationValidationError::InvalidNeighbors {
                        message: format!(
                            "Cell {:?} neighbor at position {i} shares {shared_count} vertices, expected {D}. \
                            Invariant: neighbor[{i}] must share facet opposite vertex[{i}] (all vertices except vertex {i})",
                            cell.uuid()
                        ),
                    });
                }

                if missing_vertex_idx != Some(i) {
                    return Err(TriangulationValidationError::InvalidNeighbors {
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
    /// Returns a `TriangulationValidationError` if:
    /// - The cell with the given key doesn't exist
    /// - The neighbor vector length is not D+1
    /// - Any neighbor key references a non-existent cell
    /// - **The topological invariant is violated** (neighbor\[i\] not opposite vertex\[i\])
    pub fn set_neighbors_by_key(
        &mut self,
        cell_key: CellKey,
        neighbors: &[Option<CellKey>],
    ) -> Result<(), TriangulationValidationError> {
        // Validate the topological invariant before applying changes
        // (includes length check: neighbors.len() == D+1)
        self.validate_neighbor_topology(cell_key, neighbors)?;

        // Phase 3A: Store CellKeys directly, no UUID conversion needed
        let neighbors_vec = neighbors;

        // Get mutable reference and update, or return error if not found
        let cell = self.get_cell_by_key_mut(cell_key).ok_or_else(|| {
            TriangulationValidationError::InconsistentDataStructure {
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

    /// Builds a complete vertex→cells mapping for all vertices in the triangulation.
    ///
    /// This is a helper method that constructs a mapping from each vertex key to the
    /// set of cells containing that vertex. This map is reused by multiple operations
    /// including `assign_incident_cells` and optimized cell-finding operations.
    ///
    /// # Returns
    ///
    /// A `VertexToCellsMap` containing the mapping from vertex keys to cell keys.
    ///
    /// # Errors
    ///
    /// Returns a `TriangulationValidationError` if a cell references a non-existent vertex key.
    fn build_vertex_to_cells_map(&self) -> Result<VertexToCellsMap, TriangulationValidationError> {
        let mut vertex_to_cells: VertexToCellsMap =
            fast_hash_map_with_capacity(self.vertices.len());

        for (cell_key, cell) in &self.cells {
            let vertices = self.get_cell_vertices(cell_key).map_err(|e| {
                TriangulationValidationError::InconsistentDataStructure {
                    message: format!("Failed to get vertex keys for cell {}: {}", cell.uuid(), e),
                }
            })?;
            for &vertex_key in &vertices {
                vertex_to_cells
                    .entry(vertex_key)
                    .or_default()
                    .push(cell_key);
            }
        }

        Ok(vertex_to_cells)
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
    #[must_use]
    pub fn find_cells_containing_vertex_by_key(&self, vertex_key: VertexKey) -> CellKeySet {
        let mut containing_cells = CellKeySet::default();

        let Some(_target_vertex) = self.get_vertex_by_key(vertex_key) else {
            return containing_cells;
        };

        for (cell_key, cell) in &self.cells {
            // Phase 3A: Check if cell contains the vertex using vertices
            if cell.vertices().contains(&vertex_key) {
                containing_cells.insert(cell_key);
            }
        }

        containing_cells
    }

    /// Assigns incident cells to vertices in the triangulation.
    ///
    /// This method establishes a mapping from each vertex to one of the cells that contains it,
    /// which is useful for various geometric queries and traversals. For each vertex, an arbitrary
    /// incident cell is selected from the cells that contain that vertex.
    ///
    /// # Returns
    ///
    /// `Ok(())` if incident cells were successfully assigned to all vertices,
    /// otherwise a `TriangulationValidationError`.
    ///
    /// # Errors
    ///
    /// Returns a `TriangulationValidationError` if:
    /// - A cell references a non-existent vertex key (`InconsistentDataStructure`)
    /// - A cell key cannot be found in the cells storage map (`InconsistentDataStructure`)
    /// - A vertex key cannot be found in the vertices storage map (`InconsistentDataStructure`)
    ///
    /// # Algorithm
    ///
    /// 1. Build a mapping from vertex keys to lists of cell keys that contain each vertex
    /// 2. For each vertex that appears in at least one cell, assign the first cell as its incident cell
    /// 3. Update the vertex's `incident_cell` field with the `CellKey` of the selected cell (Phase 3)
    ///
    pub fn assign_incident_cells(&mut self) -> Result<(), TriangulationValidationError> {
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

        // Build vertex_to_cells mapping using the reusable helper
        let vertex_to_cells = self.build_vertex_to_cells_map()?;

        // Iterate over for (vertex_key, cell_keys) in vertex_to_cells
        for (vertex_key, cell_keys) in vertex_to_cells {
            if !cell_keys.is_empty() {
                // Phase 3: Use CellKey directly instead of converting to UUID
                let cell_key = cell_keys[0];

                // Verify the cell key is still valid
                if !self.cells.contains_key(cell_key) {
                    return Err(TriangulationValidationError::InconsistentDataStructure {
                        message: format!(
                            "Cell key {cell_key:?} not found in cells storage map during incident cell assignment"
                        ),
                    });
                }

                // Update the vertex's incident cell with the key
                let vertex = self.vertices.get_mut(vertex_key)
                    .ok_or_else(|| TriangulationValidationError::InconsistentDataStructure {
                        message: format!(
                            "Vertex key {vertex_key:?} not found in vertices storage map during incident cell assignment"
                        ),
                    })?;
                vertex.incident_cell = Some(cell_key);
            }
        }

        Ok(())
    }
}

// =============================================================================
// TRIANGULATION LOGIC
// =============================================================================

impl<T, U, V, const D: usize> Tds<T, U, V, D>
where
    T: CoordinateScalar + AddAssign<T> + SubAssign<T> + Sum + NumCast,
    U: DataType,
    V: DataType,
{
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
}

// =============================================================================
// NEIGHBOR & INCIDENT ASSIGNMENT
// =============================================================================
// Note: These methods have been moved to the minimal trait bounds impl block
// since they only require basic TDS functionality, not coordinate operations.

// Placeholder comment to maintain section structure

impl<T, U, V, const D: usize> Tds<T, U, V, D>
where
    T: CoordinateScalar + AddAssign<T> + SubAssign<T> + Sum + NumCast,
    U: DataType,
    V: DataType,
{
    /// Clears all neighbor relationships between cells in the triangulation.
    ///
    /// This method removes all neighbor relationships by setting the `neighbors` field
    /// to `None` for every cell in the triangulation. This is useful for:
    /// - Benchmarking neighbor assignment in isolation
    /// - Testing triangulations in a known state without neighbors
    /// - Debugging neighbor-related algorithms
    /// - Implementing custom neighbor assignment algorithms
    ///
    /// This is the inverse operation of [`assign_neighbors`](Self::assign_neighbors),
    /// and is commonly used in benchmarks and testing scenarios where you want to
    /// measure the performance of neighbor assignment starting from a clean state.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
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
}

// =============================================================================
// NEIGHBOR ASSIGNMENT (requires additional trait bounds)
// =============================================================================

// =============================================================================
// DUPLICATE REMOVAL & FACET MAPPING
// =============================================================================

impl<T, U, V, const D: usize> Tds<T, U, V, D>
where
    T: CoordinateScalar + AddAssign<T> + SubAssign<T> + Sum + NumCast,
    U: DataType,
    V: DataType,
{
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
    /// Returns a `TriangulationValidationError` if:
    /// - Vertex keys cannot be retrieved for any cell (data structure corruption)
    /// - Neighbor assignment fails after cell removal
    /// - Incident cell assignment fails after cell removal
    pub fn remove_duplicate_cells(&mut self) -> Result<usize, TriangulationValidationError> {
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
            // Rebuild topology to avoid stale references after cell removal
            // This ensures vertices don't point to removed cells via incident_cell,
            // and neighbor arrays don't reference removed keys
            self.assign_neighbors()?;
            self.assign_incident_cells()?;

            // Generation already bumped by assign_neighbors(); avoid double increment
        }
        Ok(duplicate_count)
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
    /// - `Err(TriangulationValidationError)`: If any cell has missing vertex keys
    ///
    /// # Errors
    ///
    /// Returns a `TriangulationValidationError::InconsistentDataStructure` if any cell
    /// cannot resolve its vertex keys, which would indicate a corrupted triangulation state.
    ///
    /// # Performance
    ///
    /// O(N×F) time complexity where N is the number of cells and F is the
    /// number of facets per cell (typically D+1 for D-dimensional cells).
    pub fn build_facet_to_cells_map(
        &self,
    ) -> Result<FacetToCellsMap, TriangulationValidationError> {
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
            // The error from get_cell_vertices is already TriangulationValidationError
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
                    return Err(TriangulationValidationError::InconsistentDataStructure {
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

    /// Fixes invalid facet sharing by removing problematic cells
    ///
    /// This method first checks if there are any invalid facet sharing issues using
    /// `validate_facet_sharing()`. If validation passes, no action is needed.
    /// Otherwise, it identifies facets that are shared by more than 2 cells (which is
    /// geometrically impossible in a valid triangulation) and removes the excess cells.
    /// It intelligently determines which cells actually contain the vertices of the facet
    /// and removes cells that don't properly contain those vertices.
    ///
    /// # Returns
    ///
    /// A `Result` containing the number of invalid cells that were removed during the cleanup process.
    /// Returns `Ok(0)` if no fixes were needed.
    ///
    /// # Errors
    ///
    /// Returns a `TriangulationValidationError` if:
    /// - Facet creation fails during the validation process
    /// - Neighbor or incident cell assignment fails during topology repair
    ///
    /// # Algorithm
    ///
    /// 1. Use `validate_facet_sharing()` to check if there are any issues
    /// 2. If validation passes, return early (no fix needed)
    /// 3. Otherwise, build a map from facet keys to the cells that contain them
    /// 4. For each facet shared by more than 2 cells:
    ///    - Extract the actual facet vertices from one of the cells using `Facet::vertices()`
    ///    - Verify which cells truly contain all vertices of that facet using `Cell::vertices()`
    ///    - Keep only the valid cells (up to 2) and remove invalid ones
    /// 5. Remove the excess/invalid cells and update the cell bimap accordingly
    /// 6. Clean up any resulting duplicate cells
    ///
    /// # Selection Strategy
    ///
    /// This is a purely topological repair that uses UUID ordering for deterministic
    /// cell selection when multiple valid cells share a facet. For quality-based selection,
    /// use `Triangulation::fix_invalid_facet_sharing()` instead.
    #[expect(clippy::too_many_lines)]
    #[allow(dead_code)]
    pub(crate) fn fix_invalid_facet_sharing_uuid_only(
        &mut self,
    ) -> Result<usize, TriangulationValidationError> {
        // Safety limit for iteration count to prevent infinite loops
        const MAX_FIX_FACET_ITERATIONS: usize = 10;

        // First check if there are any facet sharing issues using the validation function
        if self.validate_facet_sharing().is_ok() {
            // No facet sharing issues found, no fix needed
            return Ok(0);
        }

        // Iterate until all facet sharing issues are resolved
        // Multiple passes may be needed as removing cells can create new issues
        let mut total_removed = 0;

        #[allow(unused_variables)] // iteration only used in debug_assertions
        for iteration in 0..MAX_FIX_FACET_ITERATIONS {
            #[cfg(debug_assertions)]
            eprintln!("fix_invalid_facet_sharing: Starting iteration {iteration}");

            // Check if facet sharing is already valid at the start of this iteration
            if self.validate_facet_sharing().is_ok() {
                #[cfg(debug_assertions)]
                if iteration > 0 {
                    eprintln!(
                        "✓ Fixed invalid facet sharing after {iteration} iterations, removed {total_removed} total cells"
                    );
                }
                return Ok(total_removed);
            }

            // There are facet sharing issues, proceed with the fix
            // Use strict build - if it fails, we can't reliably repair the triangulation
            // The facet map is essential for identifying shared facets
            let facet_to_cells = match self.build_facet_to_cells_map() {
                Ok(map) => map,
                Err(e) => {
                    // If we can't build a facet map, we can't reliably repair
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "Warning: Facet map build failed during repair: {e}. \
                         Cannot safely repair without valid facet mapping."
                    );
                    return Err(e);
                }
            };
            let mut cells_to_remove: CellKeySet = CellKeySet::default();

            // Find facets that are shared by more than 2 cells and validate which ones are correct
            #[allow(unused_variables)] // facet_key used in debug_assertions
            for (facet_key, cell_facet_pairs) in facet_to_cells {
                #[cfg(debug_assertions)]
                if cell_facet_pairs.len() > 2 {
                    eprintln!(
                        "Iteration {}: Processing facet {} with {} sharing cells",
                        iteration,
                        facet_key,
                        cell_facet_pairs.len()
                    );
                }

                if cell_facet_pairs.len() > 2 {
                    let first_cell_key = cell_facet_pairs[0].cell_key();
                    let first_facet_index = cell_facet_pairs[0].facet_index();
                    if self.cells.contains_key(first_cell_key) {
                        // Use direct key-based method with proper error propagation
                        // The error is already TriangulationValidationError, so just propagate it
                        let vertices = self.get_cell_vertices(first_cell_key)?;
                        // Allocate facet_vertices buffer once, reuse for all facets
                        let mut facet_vertices = Vec::with_capacity(vertices.len() - 1);
                        let idx: usize = first_facet_index.into();
                        for (i, &key) in vertices.iter().enumerate() {
                            if i != idx {
                                facet_vertices.push(key);
                            }
                        }

                        // Build the facet vertex set once, outside the loop
                        let facet_vertices_set: VertexKeySet =
                            facet_vertices.iter().copied().collect();

                        let mut valid_cells = ValidCellsBuffer::new();
                        for facet_handle in &cell_facet_pairs {
                            let cell_key = facet_handle.cell_key();
                            if self.cells.contains_key(cell_key) {
                                // Use direct key-based method with proper error propagation
                                // The error is already TriangulationValidationError, so just propagate it
                                let cell_vertices_vec = self.get_cell_vertices(cell_key)?;
                                // Use iter().copied() to avoid moving the Vec
                                let cell_vertices: VertexKeySet =
                                    cell_vertices_vec.iter().copied().collect();

                                if facet_vertices_set.is_subset(&cell_vertices) {
                                    valid_cells.push(cell_key);
                                } else {
                                    cells_to_remove.insert(cell_key);
                                }
                            }
                        }

                        #[cfg(debug_assertions)]
                        eprintln!(
                            "Iteration {}: Facet {} has {} valid cells",
                            iteration,
                            facet_key,
                            valid_cells.len()
                        );

                        if valid_cells.len() > 2 {
                            #[cfg(debug_assertions)]
                            eprintln!(
                                "Iteration {}: Facet {} has {} valid cells, using UUID ordering to select 2",
                                iteration,
                                facet_key,
                                valid_cells.len()
                            );

                            // UUID-based selection for deterministic behavior
                            // For quality-based selection, use Triangulation::fix_invalid_facet_sharing()
                            valid_cells.sort_unstable_by_key(|&k| self.cells[k].uuid());
                            for &cell_key in valid_cells.iter().skip(2) {
                                if self.cells.contains_key(cell_key) {
                                    cells_to_remove.insert(cell_key);

                                    #[cfg(debug_assertions)]
                                    eprintln!("Removing cell {cell_key:?} (UUID-based selection)");
                                } else {
                                    #[cfg(debug_assertions)]
                                    eprintln!(
                                        "Cell {cell_key:?} already removed in previous iteration"
                                    );
                                }
                            }
                        }

                        if cfg!(debug_assertions) {
                            let total_cells = cell_facet_pairs.len();
                            let removed_count = total_cells - valid_cells.len().min(2);
                            if removed_count > 0 {
                                #[cfg(debug_assertions)]
                                eprintln!(
                                    "Warning: Facet {} was shared by {} cells, removing {} invalid cells (keeping {} valid)",
                                    facet_key,
                                    total_cells,
                                    removed_count,
                                    valid_cells.len().min(2)
                                );
                            }
                        }
                    }
                }
            }

            #[cfg(debug_assertions)]
            eprintln!(
                "Iteration {}: Facet processing complete, {} cells marked for removal",
                iteration,
                cells_to_remove.len()
            );

            // Remove the invalid/excess cells using batch API (handles UUID mapping and generation)
            let to_remove: Vec<CellKey> = cells_to_remove.into_iter().collect();
            let actually_removed = self.remove_cells_by_keys(&to_remove);

            #[cfg(debug_assertions)]
            eprintln!("Iteration {iteration}: Removed {actually_removed} cells directly");

            // Clean up any resulting duplicate cells
            #[cfg(debug_assertions)]
            eprintln!("Iteration {iteration}: Calling remove_duplicate_cells");
            let duplicate_cells_removed = match self.remove_duplicate_cells() {
                Ok(n) => n,
                Err(e) => {
                    // Count direct removals before retrying next pass
                    total_removed += actually_removed;
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "Iteration {iteration}: remove_duplicate_cells failed (will retry next iteration): {e}"
                    );
                    continue; // try again next pass
                }
            };

            #[cfg(debug_assertions)]
            eprintln!("Iteration {iteration}: Removed {duplicate_cells_removed} duplicate cells");

            // Topology was rebuilt inside remove_duplicate_cells() when duplicates were removed.
            // If no duplicates were found but cells were removed, we must rebuild topology ourselves
            // per the contract of remove_cells_by_keys().
            if actually_removed > 0 && duplicate_cells_removed == 0 {
                #[cfg(debug_assertions)]
                eprintln!("Iteration {iteration}: Rebuilding topology after removals");

                if let Err(e) = self.assign_neighbors() {
                    // Count removals before retrying
                    total_removed += actually_removed;
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "Iteration {iteration}: assign_neighbors failed (will retry next iteration): {e}"
                    );
                    continue;
                }

                if let Err(e) = self.assign_incident_cells() {
                    // Count removals before retrying
                    total_removed += actually_removed;
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "Iteration {iteration}: assign_incident_cells failed (will retry next iteration): {e}"
                    );
                    continue;
                }
            }

            let removed_this_iteration = actually_removed + duplicate_cells_removed;
            total_removed += removed_this_iteration;

            #[cfg(debug_assertions)]
            eprintln!(
                "Iteration {iteration}: Removed {removed_this_iteration} cells ({actually_removed} directly, {duplicate_cells_removed} duplicates)"
            );

            // If no cells were removed this iteration, or validation passes, we're done
            let validation_ok = self.validate_facet_sharing().is_ok();

            #[cfg(debug_assertions)]
            eprintln!(
                "Iteration {iteration}: validation_ok={validation_ok}, removed_this_iteration={removed_this_iteration}"
            );

            if removed_this_iteration == 0 || validation_ok {
                #[cfg(debug_assertions)]
                if iteration > 0 {
                    eprintln!(
                        "✓ Fixed invalid facet sharing after {} iterations, removed {} total cells",
                        iteration + 1,
                        total_removed
                    );
                }
                break;
            }

            #[cfg(debug_assertions)]
            eprintln!(
                "Iteration {}: Removed {} cells, {} total removed so far",
                iteration + 1,
                removed_this_iteration,
                total_removed
            );
        }

        // After loop, verify that facet sharing is actually fixed
        if self.validate_facet_sharing().is_err() {
            return Err(TriangulationValidationError::InconsistentDataStructure {
                message: format!(
                    "fix_invalid_facet_sharing: reached MAX_FIX_FACET_ITERATIONS={MAX_FIX_FACET_ITERATIONS} with remaining invalid facet sharing"
                ),
            });
        }

        Ok(total_removed)
    }
}

// =============================================================================
// VALIDATION & CONSISTENCY CHECKS
// =============================================================================

impl<T, U, V, const D: usize> Tds<T, U, V, D>
where
    T: CoordinateScalar + AddAssign<T> + SubAssign<T> + Sum + NumCast,
    U: DataType,
    V: DataType,
{
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
    /// `Ok(())` if all vertex mappings are consistent, otherwise a `TriangulationValidationError`.
    ///
    /// This corresponds to [`InvariantKind::VertexMappings`], which is reported by
    /// [`Tds::validation_report`](Self::validation_report) and is included in
    /// [`Tds::is_valid`](Self::is_valid).
    ///
    /// # Errors
    ///
    /// Returns a `TriangulationValidationError::MappingInconsistency` with a descriptive message if:
    /// - The number of UUID-to-key mappings doesn't match the number of vertices
    /// - The number of key-to-UUID mappings doesn't match the number of vertices
    /// - A vertex exists without a corresponding UUID-to-key mapping
    /// - A vertex exists without a corresponding key-to-UUID mapping
    /// - The bidirectional mappings are inconsistent (UUID maps to key A, but key A maps to different UUID)
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// // Validation should pass for a properly constructed triangulation
    /// assert!(dt.validate_vertex_mappings().is_ok());
    /// ```
    pub fn validate_vertex_mappings(&self) -> Result<(), TriangulationValidationError> {
        if self.uuid_to_vertex_key.len() != self.vertices.len() {
            return Err(TriangulationValidationError::MappingInconsistency {
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
                return Err(TriangulationValidationError::MappingInconsistency {
                    entity: EntityKind::Vertex,
                    message: format!(
                        "Inconsistent or missing key-to-UUID mapping for key {vertex_key:?}"
                    ),
                });
            }

            // Now verify UUID-to-key direction (requires hash lookup but we know it should exist)
            if self.uuid_to_vertex_key.get(&vertex_uuid) != Some(&vertex_key) {
                return Err(TriangulationValidationError::MappingInconsistency {
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
    /// `Ok(())` if all cell mappings are consistent, otherwise a `TriangulationValidationError`.
    ///
    /// This corresponds to [`InvariantKind::CellMappings`], which is reported by
    /// [`Tds::validation_report`](Self::validation_report) and is included in
    /// [`Tds::is_valid`](Self::is_valid).
    ///
    /// # Errors
    ///
    /// Returns a `TriangulationValidationError::MappingInconsistency` with a descriptive message if:
    /// - The number of UUID-to-key mappings doesn't match the number of cells
    /// - The number of key-to-UUID mappings doesn't match the number of cells
    /// - A cell exists without a corresponding UUID-to-key mapping
    /// - A cell exists without a corresponding key-to-UUID mapping
    /// - The bidirectional mappings are inconsistent (UUID maps to key A, but key A maps to different UUID)
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// // Validation should pass for a properly constructed triangulation
    /// assert!(dt.validate_cell_mappings().is_ok());
    /// ```
    pub fn validate_cell_mappings(&self) -> Result<(), TriangulationValidationError> {
        if self.uuid_to_cell_key.len() != self.cells.len() {
            return Err(TriangulationValidationError::MappingInconsistency {
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
                return Err(TriangulationValidationError::MappingInconsistency {
                    entity: EntityKind::Cell,
                    message: format!(
                        "Inconsistent or missing key-to-UUID mapping for key {cell_key:?}"
                    ),
                });
            }

            // Now verify UUID-to-key direction (requires hash lookup but we know it should exist)
            if self.uuid_to_cell_key.get(&cell_uuid) != Some(&cell_key) {
                return Err(TriangulationValidationError::MappingInconsistency {
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
    /// `Ok(())` if all vertex keys in all cells are valid, otherwise a `TriangulationValidationError`.
    ///
    /// # Errors
    ///
    /// Returns `TriangulationValidationError::InconsistentDataStructure` if any cell
    /// references a vertex key that doesn't exist in the vertices `storage map`.
    #[allow(dead_code)]
    fn validate_cell_vertex_keys(&self) -> Result<(), TriangulationValidationError> {
        for (cell_key, cell) in &self.cells {
            let cell_uuid = cell.uuid();
            for (vertex_idx, &vertex_key) in cell.vertices().iter().enumerate() {
                if !self.vertices.contains_key(vertex_key) {
                    return Err(TriangulationValidationError::InconsistentDataStructure {
                        message: format!(
                            "Cell {cell_uuid} (key {cell_key:?}) references non-existent vertex key {vertex_key:?} at position {vertex_idx}"
                        ),
                    });
                }
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
    /// Returns a [`TriangulationValidationError`] if cell vertex retrieval fails
    /// or if any duplicate cells are detected.
    ///
    /// This corresponds to [`InvariantKind::DuplicateCells`], which is reported by
    /// [`Tds::validation_report`](Self::validation_report) and is included in
    /// [`Tds::is_valid`](Self::is_valid).
    pub fn validate_no_duplicate_cells(&self) -> Result<(), TriangulationValidationError> {
        // Use CellVertexUuidBuffer as HashMap key directly to avoid extra Vec allocation
        // Pre-size to avoid rehashing during insertion (minor optimization for hot path)
        let mut unique_cells: FastHashMap<CellVertexUuidBuffer, CellKey> =
            crate::core::collections::fast_hash_map_with_capacity(self.cells.len());
        let mut duplicates = Vec::new();

        for (cell_key, cell) in &self.cells {
            // Use Cell::vertex_uuids() helper to avoid duplicating VertexKey→UUID mapping logic
            // Convert CellValidationError to TriangulationValidationError for propagation
            let mut vertex_uuids = cell.vertex_uuids(self).map_err(|e| {
                TriangulationValidationError::InconsistentDataStructure {
                    message: format!("Failed to get vertex UUIDs for cell {cell_key:?}: {e}"),
                }
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

            return Err(TriangulationValidationError::DuplicateCells {
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
    /// Returns a [`TriangulationValidationError`] if building the facet map fails
    /// or if any facet is shared by more than two cells.
    ///
    /// This corresponds to [`InvariantKind::FacetSharing`], which is reported by
    /// [`Tds::validation_report`](Self::validation_report) and is included in
    /// [`Tds::is_valid`](Self::is_valid).
    pub fn validate_facet_sharing(&self) -> Result<(), TriangulationValidationError> {
        // Build a map from facet keys to the cells that contain them
        // Use the strict version to ensure we catch any missing vertex keys
        let facet_to_cells = self.build_facet_to_cells_map()?;

        // Check for facets shared by more than 2 cells
        for (facet_key, cell_facet_pairs) in facet_to_cells {
            if cell_facet_pairs.len() > 2 {
                return Err(TriangulationValidationError::InconsistentDataStructure {
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

    /// Checks whether the triangulation data structure is valid.
    ///
    /// # ⚠️ Performance Warning
    ///
    /// **This method is computationally expensive** and should be used judiciously:
    /// - **Time Complexity**: O(N×F + N×D²) where N is the number of cells, F is facets per cell (D+1),
    ///   and D is the spatial dimension
    /// - **Space Complexity**: O(N×F) for building facet-to-cell mappings
    /// - For large triangulations (>10K cells), this can take significant time
    /// - Consider using this primarily for debugging, testing, or after major structural changes
    ///
    /// For production code, prefer individual validation methods like [`validate_cell_mappings()`](Self::validate_cell_mappings)
    /// when only specific checks are needed.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the triangulation is valid, otherwise a [`TriangulationValidationError`].
    ///
    /// # Errors
    ///
    /// Returns a [`TriangulationValidationError`] if:
    /// - Vertex or cell UUID-to-key mappings are inconsistent
    /// - Any cell is invalid (contains invalid vertices, has nil UUID, or contains duplicate vertices)
    /// - Duplicate cells exist (cells with identical vertex sets)
    /// - Any facet is shared by more than 2 cells
    /// - Neighbor relationships are not mutual between cells
    /// - Cells have too many neighbors for their dimension
    ///
    /// # Panics
    ///
    /// Panics if an internal invariant is violated such that
    /// `validation_report` returns an error with an empty violations list.
    /// This should never occur in a correctly implemented validator.
    ///
    /// # Validation Checks
    ///
    /// This function performs comprehensive validation including:
    /// 1. **Mapping consistency**: Validates vertex and cell UUID-to-key mappings
    /// 2. **Cell uniqueness**: Checks for duplicate cells with identical vertex sets
    /// 3. **Individual cell validation**: Calls [`is_valid()`](crate::core::cell::Cell::is_valid) on each cell
    /// 4. **Facet sharing validation**: Ensures no facet is shared by >2 cells
    /// 5. **Neighbor relationship validation**: Validates mutual neighbor relationships
    ///
    /// # Examples
    ///
    /// Basic usage with a 3D triangulation:
    ///
    /// ```
    /// use delaunay::prelude::*;
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
    /// assert!(dt.is_valid().is_ok());
    /// ```
    ///
    /// Empty triangulations are valid:
    ///
    /// ```
    /// use delaunay::core::triangulation_data_structure::Tds;
    ///
    /// let tds = Tds::<f64, (), (), 3>::empty();
    /// assert!(tds.is_valid().is_ok());
    /// ```
    ///
    /// This convenience method runs all **structural invariants** (vertex and cell
    /// mappings, duplicate cells, per-cell validity, facet sharing, and neighbor
    /// consistency) and returns only the first failure.
    ///
    /// It does **not** enable the expensive global Delaunay check; to include that,
    /// use [`Tds::validation_report`](Self::validation_report) with
    /// [`ValidationOptions::check_delaunay`] set to `true`, or call
    /// [`Tds::validate_delaunay`](Self::validate_delaunay) directly.
    pub fn is_valid(&self) -> Result<(), TriangulationValidationError> {
        // Delegate to the multi-invariant report API and return only the first
        // error for backward compatibility.
        match self.validation_report(ValidationOptions::default()) {
            Ok(()) => Ok(()),
            Err(report) => {
                let first = report
                    .violations
                    .into_iter()
                    .next()
                    .expect("validation_report returned an error with no violations");
                Err(first.error)
            }
        }
    }

    /// Validates only the Delaunay empty circumsphere invariant.
    ///
    /// This is a convenience wrapper around `core::util::is_delaunay` that maps
    /// `DelaunayValidationError` into `TriangulationValidationError`.
    ///
    /// # Deprecation Notice
    ///
    /// **This method is deprecated** and will be removed in a future version.
    /// The Delaunay validation logic belongs in the `DelaunayTriangulation` layer,
    /// not in the generic `Tds` (Triangulation Data Structure) layer.
    ///
    /// **Migration Guide**:
    /// - If using `DelaunayTriangulation`, call `dt.validate_delaunay()` instead
    /// - If using `Tds` directly, call `crate::core::util::is_delaunay(&tds)` instead
    ///
    /// This deprecation follows CGAL's architecture where Tds is purely combinatorial
    /// and Delaunay-specific operations live in the Delaunay triangulation layer.
    ///
    /// # Errors
    ///
    /// Returns a [`TriangulationValidationError`] if the triangulation violates
    /// the Delaunay property or if structural validation fails during the check.
    ///
    /// This corresponds to [`InvariantKind::Delaunay`], which is reported by
    /// [`Tds::validation_report`](Self::validation_report). It is **not** enabled by
    /// default in [`Tds::is_valid`](Self::is_valid) and must be requested via
    /// [`ValidationOptions::check_delaunay`] or by calling this method directly.
    #[deprecated(
        since = "0.6.0",
        note = "Use `DelaunayTriangulation::validate_delaunay()` or `crate::core::util::is_delaunay()` instead. Tds should be purely combinatorial."
    )]
    pub fn validate_delaunay(&self) -> Result<(), TriangulationValidationError>
    where
        T: std::ops::AddAssign<T> + std::ops::SubAssign<T> + std::iter::Sum + num_traits::NumCast,
    {
        crate::core::util::is_delaunay(self).map_err(|err| {
            match err {
                DelaunayValidationError::DelaunayViolation { cell_key } => {
                    let cell_uuid = self
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
                        cell_id: uuid::Uuid::nil(), // Best effort - cell UUID not available in error
                        source,
                    }
                }
                DelaunayValidationError::NumericPredicateError {
                    cell_key,
                    vertex_key,
                    source,
                } => TriangulationValidationError::InconsistentDataStructure {
                    message: format!(
                        "Numeric predicate failure while validating Delaunay property for cell {cell_key:?}, vertex {vertex_key:?}: {source}",
                    ),
                },
            }
        })
    }

    /// Runs all structural validation checks (and optionally the Delaunay invariant)
    /// and returns a report containing **all** failed invariants.
    ///
    /// Unlike [`is_valid()`](Self::is_valid), this method does **not** stop at the
    /// first error. Instead it records a [`TriangulationValidationError`] for each
    /// invariant group that fails and returns them as a
    /// [`TriangulationValidationReport`].
    ///
    /// This is primarily intended for debugging, diagnostics, and tests that
    /// want to surface every violated invariant at once.
    ///
    /// # Errors
    ///
    /// Returns a [`TriangulationValidationReport`] containing all invariant
    /// violations if any validation step fails.
    pub fn validation_report(
        &self,
        options: ValidationOptions,
    ) -> Result<(), TriangulationValidationReport>
    where
        T: std::ops::AddAssign<T> + std::ops::SubAssign<T> + std::iter::Sum + num_traits::NumCast,
    {
        let mut violations = Vec::new();

        // 1. Mapping consistency (vertex + cell UUID↔key mappings)
        if let Err(e) = self.validate_vertex_mappings() {
            violations.push(InvariantViolation {
                kind: InvariantKind::VertexMappings,
                error: e,
            });
        }
        if let Err(e) = self.validate_cell_mappings() {
            violations.push(InvariantViolation {
                kind: InvariantKind::CellMappings,
                error: e,
            });
        }

        // If mappings are inconsistent, additional checks may produce confusing
        // secondary errors or panic. In that case, stop here and return the
        // mapping-related failures only.
        if !violations.is_empty() {
            return Err(TriangulationValidationReport { violations });
        }

        // 2. Cell uniqueness (no duplicate cells with identical vertex sets)
        if let Err(e) = self.validate_no_duplicate_cells() {
            violations.push(InvariantViolation {
                kind: InvariantKind::DuplicateCells,
                error: e,
            });
        }

        // 3. Individual cell validation
        for (cell_id, cell) in &self.cells {
            if let Err(source) = cell.is_valid() {
                let error = self.cell_uuid_from_key(cell_id).map_or_else(
                    || TriangulationValidationError::InconsistentDataStructure {
                        message: format!(
                            "Cell key {cell_id:?} has no UUID mapping during validation",
                        ),
                    },
                    |cell_uuid| TriangulationValidationError::InvalidCell {
                        cell_id: cell_uuid,
                        source,
                    },
                );
                violations.push(InvariantViolation {
                    kind: InvariantKind::CellValidity,
                    error,
                });
            }
        }

        // 4. Facet sharing (no facet shared by more than 2 cells)
        if let Err(e) = self.validate_facet_sharing() {
            violations.push(InvariantViolation {
                kind: InvariantKind::FacetSharing,
                error: e,
            });
        }

        // 5. Neighbor relationships (mutual neighbors, correct shared facets)
        if let Err(e) = self.validate_neighbors() {
            violations.push(InvariantViolation {
                kind: InvariantKind::NeighborConsistency,
                error: e,
            });
        }

        // 6. Optional Delaunay property (empty circumsphere invariant)
        if options.check_delaunay {
            // Use crate::core::util::is_delaunay directly instead of deprecated method
            let result = crate::core::util::is_delaunay(self).map_err(|err| match err {
                DelaunayValidationError::DelaunayViolation { cell_key } => {
                    let cell_uuid = self
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
            });
            if let Err(e) = result {
                violations.push(InvariantViolation {
                    kind: InvariantKind::Delaunay,
                    error: e,
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
    /// This method is optimized for performance using:
    /// - Early termination on validation failures
    /// - `HashSet` reuse to avoid repeated allocations
    /// - Efficient intersection counting without creating intermediate collections
    ///
    /// # Errors
    ///
    /// Returns a [`TriangulationValidationError`] if any neighbor relationship
    /// violates topological or consistency invariants.
    ///
    /// This corresponds to [`InvariantKind::NeighborConsistency`], which is reported by
    /// [`Tds::validation_report`](Self::validation_report) and is included in
    /// [`Tds::is_valid`](Self::is_valid).
    pub fn validate_neighbors(&self) -> Result<(), TriangulationValidationError> {
        // Pre-compute vertex keys for all cells to avoid repeated computation
        let mut cell_vertices: CellVerticesMap = fast_hash_map_with_capacity(self.cells.len());

        for cell_key in self.cells.keys() {
            // Use get_cell_vertices to ensure all vertex keys are present
            // The error is already TriangulationValidationError, so just propagate it
            let vertices = self.get_cell_vertices(cell_key)?;

            // Store the HashSet for containment checks
            let vertex_set: VertexKeySet = vertices.iter().copied().collect();
            cell_vertices.insert(cell_key, vertex_set);
        }

        for (cell_key, cell) in &self.cells {
            // Phase 3A: Use neighbors (CellKey-based) instead of neighbor UUIDs
            let Some(neighbors_buf) = &cell.neighbors else {
                continue; // Skip cells without neighbors
            };

            // Convert SmallBuffer to Vec for validation
            let neighbors: Vec<Option<CellKey>> = neighbors_buf.iter().copied().collect();

            // Validate topological invariant (neighbor[i] opposite vertex[i])
            self.validate_neighbor_topology(cell_key, &neighbors)?;

            // Get this cell's vertices from pre-computed maps
            let this_vertices = &cell_vertices[&cell_key];

            for neighbor_key_opt in &neighbors {
                // Skip None neighbors (missing neighbors)
                let Some(neighbor_key) = neighbor_key_opt else {
                    continue;
                };

                // Early termination: check if neighbor exists
                let Some(neighbor_cell) = self.cells.get(*neighbor_key) else {
                    return Err(TriangulationValidationError::InvalidNeighbors {
                        message: format!("Neighbor cell {neighbor_key:?} not found"),
                    });
                };

                // Early termination: mutual neighbor check using linear search (faster for small neighbor lists)
                // Phase 3A: Check neighbors (CellKey-based) instead of neighbor UUIDs
                if let Some(neighbor_neighbors) = &neighbor_cell.neighbors {
                    if !neighbor_neighbors.contains(&Some(cell_key)) {
                        return Err(TriangulationValidationError::InvalidNeighbors {
                            message: format!(
                                "Neighbor relationship not mutual: {:?} → {:?}",
                                cell.uuid(),
                                neighbor_cell.uuid()
                            ),
                        });
                    }
                } else {
                    // Neighbor has no neighbors, so relationship cannot be mutual
                    return Err(TriangulationValidationError::InvalidNeighbors {
                        message: format!(
                            "Neighbor relationship not mutual: {:?} → {:?}",
                            cell.uuid(),
                            neighbor_cell.uuid()
                        ),
                    });
                }

                // Optimized shared facet check: count intersections without creating intermediate collections
                let neighbor_vertices = &cell_vertices[neighbor_key];
                let shared_count = this_vertices.intersection(neighbor_vertices).count();

                // Early termination: check shared vertex count
                if shared_count != D {
                    return Err(TriangulationValidationError::NotNeighbors {
                        cell1: cell.uuid(),
                        cell2: neighbor_cell.uuid(),
                    });
                }
            }
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
            let a_coords: [T; D] = (*a).into();
            let b_coords: [T; D] = (*b).into();
            a_coords
                .partial_cmp(&b_coords)
                .unwrap_or(CmpOrdering::Equal)
        });

        other_vertices.sort_by(|a, b| {
            let a_coords: [T; D] = (*a).into();
            let b_coords: [T; D] = (*b).into();
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

    // =============================================================================
    // VERTEX ADDITION TESTS - CONSOLIDATED
    // =============================================================================

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_add_vertex_comprehensive() {
        // Test successful vertex addition into existing triangulation
        {
            // Need at least D+1 vertices for a valid triangulation before inserting
            let initial_vertices = [
                vertex!([0.0, 0.0, 0.0]),
                vertex!([1.0, 0.0, 0.0]),
                vertex!([0.0, 1.0, 0.0]),
                vertex!([0.0, 0.0, 1.0]),
            ];
            let mut dt = DelaunayTriangulation::new(&initial_vertices).unwrap();
            let vertex = vertex!([1.0, 2.0, 3.0]);
            let result = dt.insert(vertex);
            assert!(result.is_ok(), "Basic vertex addition should succeed");
            assert_eq!(dt.number_of_vertices(), 5);
        }

        // Test duplicate coordinates error
        {
            let initial_vertices = [
                vertex!([0.0, 0.0, 0.0]),
                vertex!([1.0, 0.0, 0.0]),
                vertex!([0.0, 1.0, 0.0]),
                vertex!([0.0, 0.0, 1.0]),
            ];
            let mut dt = DelaunayTriangulation::new(&initial_vertices).unwrap();
            let vertex = vertex!([1.0, 2.0, 3.0]);
            dt.insert(vertex).unwrap();

            let result = dt.insert(vertex); // Same vertex again (same UUID)
            assert!(
                matches!(
                    result,
                    Err(InsertionError::Construction(
                        TriangulationConstructionError::DuplicateUuid { .. }
                    ))
                ),
                "Adding same vertex object should fail with DuplicateUuid, got: {result:?}"
            );
        }

        // Test duplicate UUID with different coordinates
        {
            let initial_vertices = [
                vertex!([0.0, 0.0, 0.0]),
                vertex!([1.0, 0.0, 0.0]),
                vertex!([0.0, 1.0, 0.0]),
                vertex!([0.0, 0.0, 1.0]),
            ];
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
                        TriangulationConstructionError::DuplicateUuid {
                            entity: EntityKind::Vertex,
                            ..
                        }
                    ))
                ),
                "Same UUID with different coordinates should fail with DuplicateUuid"
            );
        }

        // Test vertex addition increasing counts
        {
            let initial_vertices = [
                vertex!([0.0, 0.0, 0.0]),
                vertex!([1.0, 0.0, 0.0]),
                vertex!([0.0, 1.0, 0.0]),
                vertex!([0.0, 0.0, 1.0]),
            ];
            let mut dt = DelaunayTriangulation::new(&initial_vertices).unwrap();
            let initial_cell_count = dt.number_of_cells();

            // Add another vertex
            let new_vertex = vertex!([0.5, 0.5, 0.5]);
            dt.insert(new_vertex).unwrap();

            assert_eq!(dt.number_of_vertices(), 5);
            assert!(
                dt.number_of_cells() >= initial_cell_count,
                "Cell count should not decrease"
            );
            assert!(
                dt.triangulation().tds.is_valid().is_ok(),
                "TDS should remain valid"
            );
        }

        // Test that added vertices are properly accessible
        {
            let initial_vertices = [
                vertex!([0.0, 0.0, 0.0]),
                vertex!([1.0, 0.0, 0.0]),
                vertex!([0.0, 1.0, 0.0]),
                vertex!([0.0, 0.0, 1.0]),
            ];
            let mut dt = DelaunayTriangulation::new(&initial_vertices).unwrap();
            let vertex = vertex!([1.0, 2.0, 3.0]);
            let uuid = vertex.uuid();
            dt.insert(vertex).unwrap();

            // Vertex should be findable by UUID
            let vertex_key = dt.triangulation().tds.vertex_key_from_uuid(&uuid);
            assert!(
                vertex_key.is_some(),
                "Added vertex should be findable by UUID"
            );

            // Vertex should be in the vertices collection
            let stored_vertex = dt
                .triangulation()
                .tds
                .get_vertex_by_key(vertex_key.unwrap())
                .unwrap();
            let coords: [f64; 3] = stored_vertex.into();
            let expected = [1.0, 2.0, 3.0];
            assert!(
                coords
                    .iter()
                    .zip(expected.iter())
                    .all(|(a, b)| (a - b).abs() < 1e-10),
                "Stored coordinates should match: got {coords:?}, expected {expected:?}"
            );
        }
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
            dt.triangulation()
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
                            dt.triangulation().tds.cells.contains_key(*neighbor_key),
                            "Cell {cell_key:?} has dangling neighbor reference at index {i}: {neighbor_key:?}"
                        );
                    }
                }
            }
        }

        // Verify the TDS is valid (this should pass with the bug fix)
        assert!(
            dt.triangulation().tds.is_valid().is_ok(),
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
            assert!(dt_2d.triangulation().tds.is_valid().is_ok());
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
            assert!(dt_3d.triangulation().tds.is_valid().is_ok());
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
            assert!(dt_4d.triangulation().tds.is_valid().is_ok());
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

        // Get a vertex to remove and its key
        let vertex_to_remove = *dt.vertices().nth(4).unwrap().1; // Get the interior vertex
        let removed_vertex_key = dt
            .triangulation()
            .tds
            .vertex_key_from_uuid(&vertex_to_remove.uuid())
            .unwrap();
        let removed_vertex_uuid = vertex_to_remove.uuid();

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
            dt.triangulation()
                .tds
                .vertex_key_from_uuid(&removed_vertex_uuid)
                .is_none(),
            "Deleted vertex UUID should not be in mapping"
        );
        assert!(
            dt.triangulation()
                .tds
                .get_vertex_by_key(removed_vertex_key)
                .is_none(),
            "Deleted vertex key should not exist in storage"
        );

        // CRITICAL CHECK 3: All remaining vertices should have valid incident_cell pointers
        for (vertex_key, vertex) in dt.vertices() {
            if let Some(incident_cell_key) = vertex.incident_cell {
                assert!(
                    dt.triangulation().tds.cells.contains_key(incident_cell_key),
                    "Vertex {vertex_key:?} has dangling incident_cell pointer to {incident_cell_key:?}"
                );

                // Verify the incident cell actually contains this vertex
                let incident_cell = dt.triangulation().tds.get_cell(incident_cell_key).unwrap();
                assert!(
                    incident_cell.contains_vertex(vertex_key),
                    "Vertex {vertex_key:?} incident_cell {incident_cell_key:?} does not contain the vertex"
                );
            }
        }

        // CRITICAL CHECK 4: TDS should be valid
        assert!(
            dt.triangulation().tds.is_valid().is_ok(),
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
            .triangulation()
            .tds
            .get_cell_vertices(first_cell_key)
            .unwrap()[0];
        let cells_with_vertex = dt
            .triangulation()
            .tds
            .find_cells_containing_vertex(some_vertex_key);

        // Verify the result
        assert!(
            !cells_with_vertex.is_empty(),
            "Vertex should be in at least one cell"
        );

        // Verify all returned cells actually contain the vertex
        for &cell_key in &cells_with_vertex {
            let cell = dt.triangulation().tds.get_cell(cell_key).unwrap();
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
            .triangulation()
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
}
