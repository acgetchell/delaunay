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
//! | **Vertex Validity** | [`Point::try_new`](crate::geometry::point::Point::try_new) / [`Point`](crate::geometry::point::Point) coordinate conversion (coordinates) + UUID auto-gen + `vertex.is_valid()` | Construction + runtime validation |
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
//!     delaunay::vertex![0.0, 0.0, 0.0]?,
//!     delaunay::vertex![1.0, 0.0, 0.0]?,
//!     delaunay::vertex![0.0, 1.0, 0.0]?,
//!     delaunay::vertex![0.0, 0.0, 1.0]?,
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
//!     delaunay::vertex![0.0, 0.0, 0.0]?,
//!     delaunay::vertex![1.0, 0.0, 0.0]?,
//!     delaunay::vertex![0.0, 1.0, 0.0]?,
//!     delaunay::vertex![0.0, 0.0, 1.0]?,
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
//!     delaunay::vertex![0.0, 0.0, 0.0]?,
//!     delaunay::vertex![1.0, 0.0, 0.0]?,
//!     delaunay::vertex![0.0, 1.0, 0.0]?,
//!     delaunay::vertex![0.0, 0.0, 1.0]?,
//! ];
//!
//! let mut dt = DelaunayTriangulationBuilder::new(&initial_vertices).build::<()>()?;
//!
//! // Add a new vertex
//! let new_vertex = delaunay::vertex![0.2, 0.2, 0.2]?;
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
//!     delaunay::vertex![0.0, 0.0, 0.0, 0.0]?,  // Origin
//!     delaunay::vertex![1.0, 0.0, 0.0, 0.0]?,  // Unit vector along first dimension
//!     delaunay::vertex![0.0, 1.0, 0.0, 0.0]?,  // Unit vector along second dimension
//!     delaunay::vertex![0.0, 0.0, 1.0, 0.0]?,  // Unit vector along third dimension
//!     delaunay::vertex![0.0, 0.0, 0.0, 1.0]?,  // Unit vector along fourth dimension
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

use crate::core::collections::{
    MAX_PRACTICAL_DIMENSION_SIZE, SimplexKeySet, SmallBuffer, StorageMap, UuidToSimplexKeyMap,
    UuidToVertexKeyMap, VertexKeyBuffer,
};
use crate::core::tds::errors::{NeighborValidationError, TdsError, TriangulationConstructionState};
use crate::core::tds::incidence::VertexIncidenceIndex;
use crate::core::tds::{SimplexKey, VertexKey};
use crate::core::{
    facet::facet_key_from_vertices, simplex::Simplex,
    util::periodic_facet_key_from_lifted_vertices, vertex::Vertex,
};
use std::{
    fmt::Debug,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
use uuid::Uuid;

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
///     delaunay::vertex![0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0]?,
///     delaunay::vertex![0.5, 1.0]?,
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
    pub(in crate::core::tds) vertices: StorageMap<VertexKey, Vertex<U, D>>,

    /// Storage map for simplices, providing stable keys and efficient access.
    pub(in crate::core::tds) simplices: StorageMap<SimplexKey, Simplex<V, D>>,

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

    /// Maintained vertex-to-simplices incidence index.
    ///
    /// This is the canonical exact vertex-star relation for TDS storage. It is updated
    /// together with simplex insertion/removal and rebuilt during deserialization or
    /// bulk incidence repair. Isolated vertices are represented by present-but-empty
    /// entries.
    ///
    /// Note: Not serialized - reconstructed during deserialization from simplices.
    pub(in crate::core::tds) vertex_to_simplices: VertexIncidenceIndex,

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
    pub(in crate::core::tds) generation: Arc<AtomicU64>,

    /// Runtime identity for cache/handle provenance checks.
    ///
    /// Cloning or deserializing a `Tds` creates a fresh identity so handles cached
    /// from another storage snapshot cannot be reused against the reconstructed
    /// storage by generation alone.
    ///
    /// Note: Not serialized - identity is runtime-only.
    pub(in crate::core::tds) identity: Arc<Uuid>,
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
            vertex_to_simplices: self.vertex_to_simplices.clone(),
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
            vertex_to_simplices: self.vertex_to_simplices.clone(),
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
    pub(in crate::core::tds) fn allows_periodic_self_neighbor(simplex: &Simplex<V, D>) -> bool {
        let Some(offsets) = simplex.periodic_vertex_offsets() else {
            return false;
        };
        !offsets.is_empty() && offsets.len() == simplex.number_of_vertices()
    }
    pub(in crate::core::tds) fn periodic_facet_key_from_simplex_vertices(
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

    pub(in crate::core::tds) fn build_periodic_vertex_uuid_offsets(
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

    pub(in crate::core::tds) fn lifted_vertex_identities(
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

    pub(in crate::core::tds) fn matching_lifted_facet_index(
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
    pub(in crate::core::tds) fn matching_lifted_mirror_facet_index(
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
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
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
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
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
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.5, 1.0]?,
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
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
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
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
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
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
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
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
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
    /// let vertex1: Vertex<(), 3> = delaunay::vertex![1.0, 2.0, 3.0]?;
    /// let vertex2: Vertex<(), 3> = delaunay::vertex![4.0, 5.0, 6.0]?;
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
    /// let vertices = delaunay::try_vertices_from_points(&points)?;
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
    /// let vertex1: Vertex<(), 3> = delaunay::vertex![0.0, 0.0, 0.0]?;
    /// dt.insert(vertex1)?;
    /// assert_eq!(dt.dim(), 0);
    ///
    /// let vertex2: Vertex<(), 3> = delaunay::vertex![1.0, 0.0, 0.0]?;
    /// dt.insert(vertex2)?;
    /// assert_eq!(dt.dim(), 1);
    ///
    /// let vertex3: Vertex<(), 3> = delaunay::vertex![0.0, 1.0, 0.0]?;
    /// dt.insert(vertex3)?;
    /// assert_eq!(dt.dim(), 2);
    ///
    /// let vertex4: Vertex<(), 3> = delaunay::vertex![0.0, 0.0, 1.0]?;
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
    /// let vertices_2d = delaunay::try_vertices_from_points(&points_2d)?;
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
    /// let vertices_4d = delaunay::try_vertices_from_points(&points_4d)?;
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

    /// Refreshes the derived vertex count carried by an incomplete construction state.
    #[inline]
    pub(in crate::core::tds) fn refresh_incomplete_construction_state(&mut self) {
        if matches!(
            self.construction_state,
            TriangulationConstructionState::Incomplete(_)
        ) {
            self.construction_state =
                TriangulationConstructionState::Incomplete(self.vertices.len());
        }
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
    /// let vertices = delaunay::try_vertices_from_points(&points)?;
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
    /// let vertices = delaunay::try_vertices_from_points(&points)?;
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
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
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
    pub(in crate::core::tds) fn bump_generation(&self) {
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

impl<U, V, const D: usize> Tds<U, V, D> {}

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
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
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
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
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
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
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
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
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
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
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
        self.simplices.get(simplex_key).map(Simplex::uuid)
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
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
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
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
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
        self.vertices.get(vertex_key).map(Vertex::uuid)
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
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
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
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
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
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
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

    /// Registers an isolated vertex in the maintained vertex-to-simplices index.
    #[inline]
    pub(in crate::core::tds) fn insert_empty_vertex_incidence(
        &mut self,
        vertex_key: VertexKey,
    ) -> Result<(), TdsError> {
        self.vertex_to_simplices.insert_vertex(vertex_key)
    }

    /// Removes a vertex from the maintained vertex-to-simplices index.
    #[inline]
    pub(in crate::core::tds) fn remove_vertex_incidence(
        &mut self,
        vertex_key: VertexKey,
    ) -> Result<(), TdsError> {
        self.vertex_to_simplices.remove_isolated_vertex(vertex_key)
    }

    /// Registers a simplex under each of its vertices in the maintained incidence index.
    pub(in crate::core::tds) fn add_simplex_to_vertex_incidence(
        &mut self,
        simplex_key: SimplexKey,
        vertices: &[VertexKey],
    ) -> Result<(), TdsError> {
        self.vertex_to_simplices
            .insert_simplex(simplex_key, vertices)
    }

    /// Rebuilds the maintained vertex-to-simplices incidence index from simplex storage.
    ///
    /// # Errors
    ///
    /// Returns [`TdsError::VertexNotFound`] if a simplex references a missing vertex key.
    pub(in crate::core::tds) fn rebuild_vertex_to_simplices_index(
        &mut self,
    ) -> Result<(), TdsError> {
        let mut vertex_to_simplices =
            VertexIncidenceIndex::with_vertex_capacity(self.vertices.len());
        for vertex_key in self.vertices.keys() {
            vertex_to_simplices.insert_vertex(vertex_key)?;
        }

        for (simplex_key, simplex) in &self.simplices {
            vertex_to_simplices.insert_simplex(simplex_key, simplex.vertices())?;
        }

        self.vertex_to_simplices = vertex_to_simplices;
        Ok(())
    }

    /// Returns the maintained vertex-to-simplices incidence index.
    #[inline]
    pub(crate) const fn vertex_to_simplices_index(&self) -> &VertexIncidenceIndex {
        &self.vertex_to_simplices
    }

    /// Clears one vertex incidence buffer for tests that need corrupted storage.
    #[cfg(test)]
    pub(in crate::core) fn clear_vertex_incidence_for_test(&mut self, vertex_key: VertexKey) {
        self.vertex_to_simplices.clear_vertex_for_test(vertex_key);
    }

    /// Returns every simplex key whose simplex contains `vertex_key`.
    ///
    /// This is an exact lookup against the maintained vertex-to-simplices incidence
    /// index. It deliberately does not rely on `Vertex::incident_simplex` or neighbor
    /// pointers, so callers get complete results even while inspecting disconnected
    /// but otherwise structurally consistent TDS states.
    pub(crate) fn simplex_keys_containing_vertex(
        &self,
        vertex_key: VertexKey,
    ) -> impl Iterator<Item = SimplexKey> + '_ {
        self.vertex_to_simplices.simplex_keys(vertex_key)
    }
}

impl<U, V, const D: usize> Tds<U, V, D> {
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
            vertex_to_simplices: VertexIncidenceIndex::default(),
            construction_state: TriangulationConstructionState::Incomplete(0),
            generation: Arc::new(AtomicU64::new(0)),
            identity: Arc::new(Uuid::new_v4()),
        }
    }
}

// =============================================================================
// TRAIT IMPLEMENTATIONS
// =============================================================================

pub(in crate::core::tds) type SimplexUuidSortKey<const D: usize> =
    SmallBuffer<(Uuid, [i8; D]), MAX_PRACTICAL_DIMENSION_SIZE>;

// =============================================================================
// TESTS
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::simplex::Simplex;
    use std::assert_matches;
    use std::sync::Arc;

    #[test]
    fn test_empty_initializes_storage_identity_and_counts() {
        let tds: Tds<(), (), 3> = Tds::empty();

        assert_eq!(tds.number_of_vertices(), 0);
        assert_eq!(tds.number_of_simplices(), 0);
        assert_eq!(tds.dim(), -1);
        assert!(tds.vertices().next().is_none());
        assert!(tds.simplices().next().is_none());
        assert!(tds.vertex_to_simplices_index().is_empty());
        assert!(tds.vertex_key_from_uuid(&Uuid::new_v4()).is_none());
        assert!(tds.simplex_key_from_uuid(&Uuid::new_v4()).is_none());
        assert_matches!(
            tds.construction_state(),
            TriangulationConstructionState::Incomplete(0)
        );
    }

    #[test]
    fn test_incomplete_construction_state_tracks_vertex_count() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 0.0]).unwrap())
            .unwrap();
        assert_matches!(
            tds.construction_state(),
            TriangulationConstructionState::Incomplete(1)
        );

        let _v1 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([1.0, 0.0]).unwrap())
            .unwrap();
        assert_matches!(
            tds.construction_state(),
            TriangulationConstructionState::Incomplete(2)
        );

        tds.remove_isolated_vertex(v0).unwrap();
        assert_matches!(
            tds.construction_state(),
            TriangulationConstructionState::Incomplete(1)
        );
    }

    #[test]
    fn test_vertex_to_simplices_index_tracks_simplex_insertion() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 1.0]).unwrap())
            .unwrap();

        for vertex_key in [v0, v1, v2] {
            assert!(tds.vertex_to_simplices_index().contains_vertex(vertex_key));
            assert_eq!(
                tds.vertex_to_simplices_index()
                    .number_of_simplices(vertex_key),
                0
            );
        }

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();

        for vertex_key in [v0, v1, v2] {
            let simplices: SimplexKeySet = tds.simplex_keys_containing_vertex(vertex_key).collect();
            assert_eq!(simplices.len(), 1);
            assert!(simplices.contains(&simplex_key));
        }

        assert!(
            tds.simplex_keys_containing_vertex(VertexKey::default())
                .next()
                .is_none()
        );
    }

    #[test]
    fn test_vertex_to_simplices_index_returns_disconnected_vertex_star() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let shared = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 1.0]).unwrap())
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([-1.0, 0.0]).unwrap())
            .unwrap();
        let v4 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, -1.0]).unwrap())
            .unwrap();

        let first = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![shared, v1, v2], None).unwrap(),
            )
            .unwrap();
        let second = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![shared, v3, v4], None).unwrap(),
            )
            .unwrap();

        tds.vertex_mut(shared)
            .unwrap()
            .set_incident_simplex(Some(first));

        let simplices: SimplexKeySet = tds.simplex_keys_containing_vertex(shared).collect();
        assert_eq!(simplices.len(), 2);
        assert!(simplices.contains(&first));
        assert!(simplices.contains(&second));
    }

    #[test]
    fn test_facet_key_for_simplex_facet_maps_periodic_derivation_errors() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v_a = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 0.0]).unwrap())
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([1.0, 0.0]).unwrap())
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 1.0]).unwrap())
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
    fn test_generation_counter_bumps_on_topology_modification() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        assert_eq!(tds.generation(), 0);

        let _v = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 0.0]).unwrap())
            .unwrap();
        assert!(tds.generation() > 0);

        let gen_before = tds.generation();
        tds.mark_topology_modified();
        assert!(tds.generation() > gen_before);
    }

    // =========================================================================
    // GET SIMPLEX VERTICES: ERROR PATH
    // =========================================================================

    #[test]
    fn test_simplex_vertices_errors_on_missing_vertex_key() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 1.0]).unwrap())
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
    // VALIDATE SIMPLEX COORDINATE UNIQUENESS
    // =========================================================================

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
                            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0_f64; $dim]).unwrap())
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
}
