//! D-dimensional Facets Representation
//!
//! This module provides the `FacetView` struct which represents a facet of a d-dimensional simplex
//! (d-1 sub-simplex) within a triangulation. Each facet is defined in terms of a simplex and the
//! vertex opposite to it, similar to [CGAL](https://doc.cgal.org/latest/TDS_3/index.html#title3).
//!
//! # Key Features
//!
//! - **Lightweight**: `FacetView` stores borrowed references plus compact vertex-key buffers
//! - **Dimensional Simplicity**: Represents co-dimension 1 sub-simplexes of d-dimensional simplexes
//! - **Simplex Association**: Each facet resides within a specific simplex and is described by its opposite vertex
//! - **Support for Delaunay Triangulations**: Facilitates operations fundamental to the
//!   [Bowyer-Watson algorithm](https://en.wikipedia.org/wiki/Bowyer–Watson_algorithm)
//! - **On-demand Creation**: Facets are generated dynamically as needed rather than stored persistently in the TDS
//! - **Memory Efficient**: Parses facet storage once, then exposes infallible borrowed accessors
//! - **Runtime-local Identity**: Facet handles and views contain slotmap keys and are not durable
//!   interchange identifiers. Serialize a full [`Tds`] snapshot when topology must cross an I/O
//!   boundary.
//!
//! # Fundamental Invariant
//!
//! **A critical TDS invariant is that each facet is incident to one or two
//! simplices. One-sided incidence is not, by itself, a manifold boundary
//! classification: topology-aware triangulation queries decide whether a
//! one-sided facet is true boundary or an admissible closed self-identification.**
//!
//! This property ensures the triangulation forms a valid simplicial complex:
//! - **Two-sided facets**: shared by exactly 2 simplices (defines proper adjacency)
//! - **One-sided facets**: incident to exactly 1 simplex
//! - **Boundary facets**: topology-approved one-sided facets in spaces that admit boundary
//! - **Periodic self-identifications**: one-sided in quotient storage, but closed rather than boundary
//! - **Invalid configurations**: Facets shared by 0, 3, or more simplices indicate topological errors
//!
//! This invariant is fundamental to many algorithms and is actively validated during triangulation
//! construction and validation phases.
//!
//! For a comprehensive discussion of all topological invariants in Delaunay triangulations,
//! see the [Topological Invariants](crate::prelude::tds#topological-invariants)
//! section in the triangulation data structure documentation.
//!
//! # Examples
//!
//! ```rust
//! use delaunay::prelude::*;
//! use delaunay::prelude::tds::FacetView;
//!
//! # #[derive(Debug, thiserror::Error)]
//! # enum ExampleError {
//! #     #[error(transparent)]
//! #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
//! #     #[error(transparent)]
//! #     Facet(#[from] delaunay::prelude::tds::FacetError),
//! #     #[error(transparent)]
//! #     Tds(#[from] delaunay::prelude::tds::TdsError),
//! #     #[error(transparent)]
//! #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
//! # }
//! # fn main() -> Result<(), ExampleError> {
//! // Create vertices for a tetrahedron
//! let vertices = vec![
//!     delaunay::vertex![0.0, 0.0, 0.0]?,
//!     delaunay::vertex![1.0, 0.0, 0.0]?,
//!     delaunay::vertex![0.0, 1.0, 0.0]?,
//!     delaunay::vertex![0.0, 0.0, 1.0]?,
//! ];
//!
//! // Create a 3D triangulation
//! let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
//! let Some((simplex_key, _)) = dt.simplices().next() else {
//!     return Ok(());
//! };
//!
//! // Create a facet view (facet 0 excludes vertex 0)
//! let facet = FacetView::try_new(dt.tds(), simplex_key, 0)?;
//! assert_eq!(facet.vertices().count(), 3);  // Facet (triangle) in 3D has 3 vertices
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]

use super::collections::{
    FacetToSimplicesMap, FastHashMap, MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer,
    fast_hash_map_with_capacity,
};
use super::util::{stable_hash_u64_slice, usize_to_u8};
use super::{
    simplex::Simplex,
    tds::{NeighborValidationError, SimplexKey, Tds, TdsError, VertexKey},
    vertex::Vertex,
};
use crate::geometry::traits::coordinate::CoordinateConversionError;
use slotmap::Key;
use std::{
    fmt::{self, Debug},
    iter::FusedIterator,
    sync::Arc,
    vec::IntoIter,
};
use thiserror::Error;

// =============================================================================
// ERROR TYPES
// =============================================================================

/// Error type for facet operations.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::tds::FacetError;
///
/// let err = FacetError::FacetNotFoundInTriangulation;
/// std::assert_matches!(err, FacetError::FacetNotFoundInTriangulation);
/// ```
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum FacetError {
    /// The simplex does not contain the vertex.
    #[error("The simplex does not contain the vertex!")]
    SimplexDoesNotContainVertex,
    /// A vertex UUID was not found in the UUID-to-key mapping.
    #[error("Vertex UUID not found in mapping: {uuid}")]
    VertexNotFound {
        /// The UUID that was not found.
        uuid: uuid::Uuid,
    },
    /// Facet has insufficient vertices for the given dimension.
    #[error(
        "Facet must have exactly {expected} vertices for {dimension}D triangulation, got {actual}"
    )]
    InsufficientVertices {
        /// The expected number of vertices.
        expected: usize,
        /// The actual number of vertices.
        actual: usize,
        /// The dimension of the triangulation.
        dimension: usize,
    },
    /// Facet was not found in the triangulation.
    #[error("Facet not found in triangulation")]
    FacetNotFoundInTriangulation,
    /// Facet key was not found in the cache during lookup.
    #[error(
        "Facet key {facet_key:016x} not found in cache with {cache_size} entries - possible invariant violation or key derivation mismatch. Vertex UUIDs: {vertex_uuids:?}"
    )]
    FacetKeyNotFoundInCache {
        /// The facet key that was not found.
        facet_key: u64,
        /// The number of entries in the cache.
        cache_size: usize,
        /// The vertex UUIDs that generated the facet key.
        vertex_uuids: Vec<uuid::Uuid>,
    },
    /// Expected exactly one adjacent simplex for boundary facet.
    #[error("Expected exactly 1 adjacent simplex for boundary facet, found {found}")]
    InvalidAdjacentSimplexCount {
        /// The number of adjacent simplices found.
        found: usize,
    },
    /// Adjacent simplex was not found in the triangulation.
    #[error("Adjacent simplex not found")]
    AdjacentSimplexNotFound,
    /// Could not find inside vertex for boundary facet.
    #[error("Could not find inside vertex for boundary facet")]
    InsideVertexNotFound,
    /// Failed to compute geometric orientation.
    #[error("Failed to compute orientation during {context}: {source}")]
    OrientationComputationFailed {
        /// Orientation-computation context.
        context: String,
        /// Underlying coordinate conversion or predicate setup failure.
        #[source]
        source: CoordinateConversionError,
    },
    /// Invalid facet index for a simplex.
    #[error("Invalid facet index {index} for simplex with {facet_count} facets")]
    InvalidFacetIndex {
        /// The invalid facet index.
        index: u8,
        /// The number of facets in the simplex.
        facet_count: usize,
    },
    /// Invalid facet index that couldn't be converted to u8.
    #[error(
        "Invalid facet index {original_index} (too large for u8 conversion) for {facet_count} facets"
    )]
    InvalidFacetIndexOverflow {
        /// The original usize index that failed conversion.
        original_index: usize,
        /// The number of facets available.
        facet_count: usize,
    },
    /// Dimension exceeds the maximum representable facet index.
    #[error("Dimension {dimension} exceeds maximum {max_dimension} for u8 facet indices")]
    FacetIndexCapacityExceeded {
        /// The triangulation dimension.
        dimension: usize,
        /// The maximum supported dimension with the current facet-index representation.
        max_dimension: usize,
    },
    /// Simplex was not found in the triangulation.
    #[error("Simplex not found in triangulation (potential data corruption)")]
    SimplexNotFoundInTriangulation,
    /// Vertex key was not found in the triangulation.
    #[error("Vertex key not found in triangulation: {key:?}")]
    VertexKeyNotFoundInTriangulation {
        /// The vertex key that was not found.
        key: VertexKey,
    },
    /// A facet view was used with a different TDS than the one that produced it.
    #[error(
        "Facet view for simplex {simplex_key:?}, facet {facet_index} belongs to a different TDS"
    )]
    FacetOwnerMismatch {
        /// The simplex key stored in the foreign facet view.
        simplex_key: SimplexKey,
        /// The facet index stored in the foreign facet view.
        facet_index: u8,
    },
    /// A facet-to-simplices index was used with a different TDS than the one that produced it.
    #[error("Facet-to-simplices index belongs to a different TDS")]
    FacetIndexOwnerMismatch,
    /// Facet has invalid multiplicity (should be one-sided or two-sided).
    #[error(
        "Facet with key {facet_key:016x} has invalid multiplicity {found}, expected 1 (one-sided) or 2 (two-sided)"
    )]
    InvalidFacetMultiplicity {
        /// The facet key with invalid multiplicity.
        facet_key: u64,
        /// The actual multiplicity found.
        found: usize,
    },
    /// A two-sided facet incidence repeated the same simplex facet handle.
    #[error(
        "Facet with key {facet_key:016x} repeats incident simplex facet {handle:?}; expected distinct incident simplex facets"
    )]
    DuplicateFacetIncidentHandle {
        /// The facet key with duplicate incident handles.
        facet_key: u64,
        /// The repeated simplex facet handle.
        handle: FacetHandle,
    },
    /// An incident facet handle derives a different canonical facet key than its index entry.
    #[error(
        "Facet handle {handle:?} derives key {actual_facet_key:016x}, but index entry expected {expected_facet_key:016x}"
    )]
    FacetHandleKeyMismatch {
        /// The facet key under which the handle was stored.
        expected_facet_key: u64,
        /// The canonical facet key derived from the handle's live facet view.
        actual_facet_key: u64,
        /// The mismatched incident simplex facet handle.
        handle: FacetHandle,
    },
    /// A supplied boundary facet handle is not the parsed one-sided handle for its facet key.
    #[error(
        "Boundary facet handle {supplied_handle:?} is not the indexed one-sided handle {indexed_handle:?} for facet key {facet_key:016x}"
    )]
    BoundaryFacetHandleNotIndexed {
        /// The canonical facet key derived from the supplied handle.
        facet_key: u64,
        /// The handle supplied to the boundary-facet iterator.
        supplied_handle: FacetHandle,
        /// The one-sided handle stored in the parsed facet index.
        indexed_handle: FacetHandle,
    },
    /// Failed to retrieve boundary facets from triangulation.
    #[error("Failed to retrieve boundary facets: {source}")]
    BoundaryFacetRetrievalFailed {
        /// The underlying TDS validation error.
        #[source]
        source: Arc<TdsError>,
    },
    /// Failed to derive this facet's canonical key.
    #[error("Failed to derive canonical facet key: {source}")]
    FacetKeyDerivationFailed {
        /// The underlying TDS validation error.
        #[source]
        source: Arc<TdsError>,
    },
    /// Simplex operation failed due to validation error.
    #[error("Simplex operation failed: {source}")]
    SimplexOperationFailed {
        /// The underlying TDS validation error.
        #[source]
        source: Arc<TdsError>,
    },
}

// =============================================================================
// FACET HANDLE
// =============================================================================

/// A lightweight handle to a facet.
///
/// This provides a more readable and maintainable alternative to raw tuples throughout
/// the codebase. Facet handles are used to reference facets without storing full vertex data.
/// They are storage-local runtime handles: the embedded [`SimplexKey`] is regenerated when a
/// [`Tds`] is hydrated from a snapshot and must not be persisted as durable topology identity.
///
/// # Components
///
/// - `simplex_key`: The key of the simplex containing the facet
/// - `facet_index`: The facet index (0 to D, representing the vertex opposite to the facet)
///
/// # Usage
///
/// `FacetHandle` is commonly used in:
/// - Boundary-facet handles after topology-aware convex-hull extraction
/// - Facet visibility testing
/// - Cavity computation in Bowyer-Watson algorithm
/// - Any operation requiring lightweight facet references
///
/// Use [`FacetHandle::try_new`] for raw simplex-key/index inputs from callers, then
/// [`FacetHandle::view`] when borrowed facet access is needed. The crate-internal `from_validated`
/// constructor is reserved for code paths that have already proven the simplex key is live and the
/// facet index is in range.
///
/// # Example
///
/// ```rust
/// use delaunay::prelude::*;
/// use delaunay::prelude::tds::{FacetHandle, FacetView};
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Facet(#[from] delaunay::prelude::tds::FacetError),
/// #     #[error(transparent)]
/// #     Tds(#[from] delaunay::prelude::tds::TdsError),
/// #     #[error(transparent)]
/// #     Query(#[from] delaunay::query::QueryError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0, 1.0]?,
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
/// let Some((simplex_key, _)) = dt.simplices().next() else {
///     return Ok(());
/// };
///
/// // Create a facet handle
/// let handle = FacetHandle::try_new(dt.tds(), simplex_key, 0)?;
///
/// // Use it to create a FacetView
/// let facet = handle.view(dt.tds())?;
/// # let _ = facet;
/// # Ok(())
/// # }
/// ```
#[must_use]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FacetHandle {
    simplex_key: SimplexKey,
    facet_index: u8,
}

impl FacetHandle {
    /// Creates a new facet handle after validating it against a TDS.
    ///
    /// # Arguments
    ///
    /// * `simplex_key` - The key of the simplex containing the facet
    /// * `facet_index` - The facet index (0 to D)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::tds::FacetHandle;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Facet(#[from] delaunay::prelude::tds::FacetError),
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
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let handle = FacetHandle::try_new(dt.tds(), simplex_key, 0)?;
    /// assert_eq!(handle.facet_index(), 0);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`FacetError::SimplexNotFoundInTriangulation`] if `simplex_key`
    /// is not present in `tds`, or [`FacetError::InvalidFacetIndex`] if
    /// `facet_index` is outside the simplex facet range.
    pub fn try_new<U, V, const D: usize>(
        tds: &Tds<U, V, D>,
        simplex_key: SimplexKey,
        facet_index: u8,
    ) -> Result<Self, FacetError> {
        let simplex = tds
            .simplex(simplex_key)
            .ok_or(FacetError::SimplexNotFoundInTriangulation)?;
        let facet_count = simplex.number_of_vertices();
        if usize::from(facet_index) >= facet_count {
            return Err(FacetError::InvalidFacetIndex {
                index: facet_index,
                facet_count,
            });
        }

        Ok(Self::from_validated(simplex_key, facet_index))
    }

    /// Creates a facet handle from a simplex key and facet index already
    /// proven valid by the caller.
    #[inline]
    pub(crate) const fn from_validated(simplex_key: SimplexKey, facet_index: u8) -> Self {
        Self {
            simplex_key,
            facet_index,
        }
    }

    /// Returns the simplex key.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::tds::FacetHandle;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)]
    /// #     Tds(#[from] delaunay::prelude::tds::TdsError),
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
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let handle = FacetHandle::try_new(dt.tds(), simplex_key, 0)?;
    /// assert_eq!(handle.simplex_key(), simplex_key);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub const fn simplex_key(&self) -> SimplexKey {
        self.simplex_key
    }

    /// Returns the facet index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::tds::FacetHandle;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Facet(#[from] delaunay::prelude::tds::FacetError),
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
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let handle = FacetHandle::try_new(dt.tds(), simplex_key, 1)?;
    /// assert_eq!(handle.facet_index(), 1);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub const fn facet_index(&self) -> u8 {
        self.facet_index
    }

    /// Creates a borrowed [`FacetView`] for this handle in a live TDS.
    ///
    /// The handle stores only runtime-local slotmap identity. This method rechecks that the
    /// simplex key still resolves in `tds` and that the facet index is still in range before
    /// returning a view with borrowed accessors.
    ///
    /// # Errors
    ///
    /// Returns [`FacetError::SimplexNotFoundInTriangulation`] if this handle's simplex key
    /// is not present in `tds`, or [`FacetError::InvalidFacetIndex`] if this handle's facet
    /// index is outside the simplex facet range.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::tds::FacetHandle;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Facet(#[from] delaunay::prelude::tds::FacetError),
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
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let handle = FacetHandle::try_new(dt.tds(), simplex_key, 0)?;
    /// let view = handle.view(dt.tds())?;
    ///
    /// assert_eq!(view.handle(), handle);
    /// # Ok(())
    /// # }
    /// ```
    pub fn view<U, V, const D: usize>(
        self,
        tds: &Tds<U, V, D>,
    ) -> Result<FacetView<'_, U, V, D>, FacetError> {
        FacetView::try_new(tds, self.simplex_key, self.facet_index)
    }
}

// =============================================================================
// LIGHTWEIGHT FACET VIEW
// =============================================================================

/// Lightweight facet representation as a view into a triangulation data structure.
///
/// Lightweight facet implementation that replaces the heavyweight `Facet` struct
/// with an ~18x memory reduction.
///
/// `FacetView` represents a facet (d-1 dimensional face) of a d-dimensional simplex.
/// It validates the simplex key, facet index, and vertex keys at construction,
/// then carries borrowed references for infallible live-TDS access.
///
/// Mathematically, a facet of a D-simplex is the `(D - 1)`-simplex obtained by
/// omitting exactly one of the simplex's `D + 1` vertices. The omitted vertex is
/// called the **opposite vertex**. Thus a triangle has three edge facets, a
/// tetrahedron has four triangular facets, and in general a D-simplex has one
/// facet opposite each vertex.
///
/// Like [`FacetHandle`], this is a live view over one in-memory [`Tds`]. It is appropriate for
/// traversal, validation, and local algorithms, but it is not a persistence format. A
/// deserialized TDS receives fresh slotmap keys, so stored facet views or handles from another
/// process or snapshot cannot be reused.
///
/// # Memory Efficiency
///
/// Compared to the original `Facet<U, V, D>`:
/// - **Original**: Stores complete Simplex + Vertex objects (~hundreds of bytes)
/// - **`FacetView`**: Stores TDS/simplex references, a handle, and compact facet vertex buffers
/// - **Memory reduction**: avoids owning simplex and vertex payloads
///
/// # Type Parameters
///
/// - `'tds`: Lifetime of the triangulation data structure
/// - `U`: Vertex data type  
/// - `V`: Simplex data type
/// - `D`: Spatial dimension
///
/// # Examples
///
/// ```rust,no_run
/// use delaunay::prelude::tds::{FacetError, FacetView};
/// use delaunay::prelude::tds::{Tds, SimplexKey};
///
/// // This is a conceptual example showing FacetView usage
/// // In practice, tds and simplex_key would come from your triangulation
/// fn example_usage<'a>(
///     tds: &'a Tds<(), (), 3>,
///     simplex_key: SimplexKey,
/// ) -> Result<(), FacetError> {
///     // Create a facet view for the first facet of a simplex
///     let facet_view = FacetView::try_new(tds, simplex_key, 0)?;
///
///     // Access vertices through the view
///     for vertex in facet_view.vertices() {
///         println!("Vertex: {:?}", vertex.point());
///     }
///
///     // Get the opposite vertex
///     let opposite = facet_view.opposite_vertex();
///
///     // Compute facet key
///     let key = facet_view.key();
///     let _ = (opposite, key);
///     Ok(())
/// }
/// ```
#[must_use]
pub struct FacetView<'tds, U, V, const D: usize> {
    /// Reference to the triangulation data structure.
    tds: &'tds Tds<U, V, D>,
    /// Borrowed simplex containing this facet.
    simplex: &'tds Simplex<V, D>,
    /// Key of the simplex containing this facet.
    simplex_key: SimplexKey,
    /// Index of this facet within the simplex (0 <= `facet_index` < D+1).
    ///
    /// The `facet_index` indicates which vertex of the simplex is the "opposite vertex"
    /// (the vertex not included in the facet). For a D-dimensional simplex with D+1
    /// vertices, facet i excludes vertex i and includes all others.
    facet_index: u8,
    /// Vertex keys that define this facet, excluding the opposite vertex.
    facet_vertex_keys: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    /// Canonical key matching the TDS facet-to-simplices index.
    key: u64,
    /// Borrowed vertices that define this facet, in containing-simplex order.
    vertices: SmallBuffer<&'tds Vertex<U, D>, MAX_PRACTICAL_DIMENSION_SIZE>,
    /// Borrowed opposite vertex excluded from this facet.
    opposite_vertex: &'tds Vertex<U, D>,
}

impl<'tds, U, V, const D: usize> FacetView<'tds, U, V, D> {
    /// Returns the simplex key for this facet.
    #[inline]
    #[must_use]
    pub const fn simplex_key(&self) -> SimplexKey {
        self.simplex_key
    }

    /// Returns the facet index within the simplex.
    #[inline]
    #[must_use]
    pub const fn facet_index(&self) -> u8 {
        self.facet_index
    }

    /// Returns the TDS reference.
    #[inline]
    #[must_use]
    pub const fn tds(&self) -> &'tds Tds<U, V, D> {
        self.tds
    }

    /// Returns the copyable runtime handle represented by this view.
    ///
    /// This is infallible because a [`FacetView`] is constructed only after proving that the
    /// simplex key and facet index are valid for the borrowed TDS.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::tds::FacetHandle;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Facet(#[from] delaunay::prelude::tds::FacetError),
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
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let handle = FacetHandle::try_new(dt.tds(), simplex_key, 0)?;
    /// let view = handle.view(dt.tds())?;
    ///
    /// assert_eq!(view.handle(), handle);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub const fn handle(&self) -> FacetHandle {
        FacetHandle::from_validated(self.simplex_key, self.facet_index)
    }
}

impl<'tds, U, V, const D: usize> FacetView<'tds, U, V, D> {
    /// Creates a new `FacetView` for the specified facet of a simplex.
    ///
    /// # Arguments
    ///
    /// * `tds` - Reference to the triangulation data structure
    /// * `simplex_key` - The key of the simplex containing the facet
    /// * `facet_index` - The index of the facet within the simplex (0 to D)
    ///
    /// The `facet_index` is the index of the opposite vertex in the containing
    /// simplex. Constructing the facet means borrowing every simplex vertex
    /// except that opposite vertex, preserving the simplex's vertex order.
    ///
    /// # Returns
    ///
    /// A `Result<FacetView, FacetError>` containing the facet view if successful.
    ///
    /// # Errors
    ///
    /// Returns [`FacetError::SimplexNotFoundInTriangulation`] if
    /// `simplex_key` is not present in the TDS, or
    /// [`FacetError::InvalidFacetIndex`] if `facet_index` is not a valid facet
    /// of the simplex.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::tds::FacetView;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)]
    /// #     Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let facet = FacetView::try_new(dt.tds(), simplex_key, 0)?;
    /// assert_eq!(facet.facet_index(), 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_new(
        tds: &'tds Tds<U, V, D>,
        simplex_key: SimplexKey,
        facet_index: u8,
    ) -> Result<Self, FacetError> {
        let simplex = tds
            .simplex(simplex_key)
            .ok_or(FacetError::SimplexNotFoundInTriangulation)?;

        let vertex_count = simplex.number_of_vertices();
        if usize::from(facet_index) >= vertex_count {
            return Err(FacetError::InvalidFacetIndex {
                index: facet_index,
                facet_count: vertex_count,
            });
        }

        let mut facet_vertex_keys: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::with_capacity(vertex_count.saturating_sub(1));
        let mut vertices: SmallBuffer<&'tds Vertex<U, D>, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::with_capacity(vertex_count.saturating_sub(1));
        let mut opposite_vertex = None;
        let facet_index_usize = usize::from(facet_index);
        for (index, &vertex_key) in simplex.vertices().iter().enumerate() {
            let vertex = tds
                .vertex(vertex_key)
                .ok_or(FacetError::VertexKeyNotFoundInTriangulation { key: vertex_key })?;
            if index == facet_index_usize {
                opposite_vertex = Some(vertex);
            } else {
                facet_vertex_keys.push(vertex_key);
                vertices.push(vertex);
            }
        }
        let opposite_vertex = opposite_vertex.ok_or(FacetError::InvalidFacetIndex {
            index: facet_index,
            facet_count: vertex_count,
        })?;
        let key = Tds::<U, V, D>::periodic_facet_key_from_simplex_vertices(
            simplex,
            simplex.vertices(),
            facet_index_usize,
        )
        .map_err(|source| FacetError::FacetKeyDerivationFailed {
            source: Arc::new(source),
        })?;

        Ok(Self {
            tds,
            simplex,
            simplex_key,
            facet_index,
            facet_vertex_keys,
            key,
            vertices,
            opposite_vertex,
        })
    }

    /// Returns an iterator over the vertices that make up this facet.
    ///
    /// The facet vertices are all vertices of the containing simplex except
    /// the opposite vertex (at `facet_index`).
    ///
    /// This method is available without coordinate or payload trait bounds,
    /// enabling usage in lightweight operations that only inspect topology.
    ///
    /// This is infallible because [`Self::try_new`] validated the containing
    /// simplex, facet index, and vertex keys while borrowing the TDS.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::tds::FacetView;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)]
    /// #     Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// if let Some((simplex_key, _)) = dt.simplices().next() {
    ///     let facet = FacetView::try_new(dt.tds(), simplex_key, 0)?;
    ///     let vertex_iter = facet.vertices();
    ///     assert_eq!(vertex_iter.count(), 3); // 3D facet has 3 vertices
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn vertices(&self) -> impl ExactSizeIterator<Item = &'tds Vertex<U, D>> + '_ {
        self.vertices.iter().copied()
    }

    /// Returns the opposite vertex (the vertex not included in the facet).
    ///
    /// Returns a reference to the opposite vertex.
    ///
    /// This is infallible because [`Self::try_new`] validated and cached the
    /// opposite vertex while borrowing the TDS.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::tds::FacetView;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)]
    /// #     Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// let facet = FacetView::try_new(dt.tds(), simplex_key, 1)?;
    /// let opposite = facet.opposite_vertex();
    /// assert_eq!(opposite.point().coords().len(), 3);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub const fn opposite_vertex(&self) -> &'tds Vertex<U, D> {
        self.opposite_vertex
    }

    /// Returns the simplex containing this facet.
    ///
    /// Returns a reference to the containing simplex.
    ///
    /// This is infallible because [`Self::try_new`] validated the simplex key
    /// while borrowing the TDS.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::tds::FacetView;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)]
    /// #     Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// let facet = FacetView::try_new(dt.tds(), simplex_key, 2)?;
    /// let simplex = facet.simplex();
    /// assert_eq!(simplex.number_of_vertices(), 4);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub const fn simplex(&self) -> &'tds Simplex<V, D> {
        self.simplex
    }

    /// Returns the canonical key for this facet.
    ///
    /// The key matches the owning TDS facet-to-simplices index. For ordinary
    /// Euclidean facets this is the bare vertex-key hash; for periodic quotient
    /// facets it also incorporates normalized lattice offsets, so identified
    /// lifted images share a key.
    ///
    /// This is infallible because [`Self::try_new`] validated and cached the
    /// facet vertex keys while borrowing the TDS.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::tds::FacetView;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)]
    /// #     Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// let facet = FacetView::try_new(dt.tds(), simplex_key, 0)?;
    /// let facet_key = facet.key();
    /// let index = dt.tds().build_facet_to_simplices_index()?;
    /// assert!(index.get(&facet_key).is_some());
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub const fn key(&self) -> u64 {
        self.key
    }
}

/// Validated incident simplices for one canonical facet key.
///
/// A valid D-dimensional triangulation facet is incident to either one or two
/// D-simplices. This crate-internal value carries that multiplicity proof after
/// a raw incidence map has been parsed. Public callers observe it through
/// [`FacetIncidenceView`], which keeps the proof borrowed from the
/// owner-bound [`FacetToSimplicesIndex`].
///
/// One-sided incidence is not the same as manifold boundary: periodic quotient
/// triangulations can encode a closed self-identification with one incident
/// simplex and a self-neighbor pointer. Boundary classification belongs to the
/// topology layer because it depends on the
/// [`GlobalTopology`](crate::prelude::topology::spaces::GlobalTopology)
/// declared by the surrounding triangulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use]
pub(crate) struct FacetIncidence {
    kind: FacetIncidenceKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FacetIncidenceKind {
    OneSided(FacetHandle),
    TwoSided([FacetHandle; 2]),
}

impl FacetIncidence {
    /// Parses a raw facet-index entry into validated incidence for one canonical facet key.
    ///
    /// # Errors
    ///
    /// Returns [`FacetError::InvalidFacetMultiplicity`] or
    /// [`FacetError::DuplicateFacetIncidentHandle`] when the raw entry does not
    /// contain one handle or two distinct handles. Returns
    /// [`FacetError::FacetHandleKeyMismatch`] or a lower-level handle/view error
    /// when an incident handle does not resolve to the facet key under the TDS
    /// that produced the index.
    fn try_from_index_entry<U, V, const D: usize>(
        tds: &Tds<U, V, D>,
        facet_key: u64,
        handles: &SmallBuffer<FacetHandle, 2>,
    ) -> Result<Self, FacetError> {
        let incidence = Self::try_from_handles(facet_key, handles)?;
        match incidence.kind {
            FacetIncidenceKind::OneSided(handle) => {
                let handle = try_incident_facet_view_for_facet_key(tds, facet_key, handle)
                    .map(|_| handle)?;
                Ok(Self {
                    kind: FacetIncidenceKind::OneSided(handle),
                })
            }
            FacetIncidenceKind::TwoSided([first, second]) => {
                let first =
                    try_incident_facet_view_for_facet_key(tds, facet_key, first).map(|_| first)?;
                let second = try_incident_facet_view_for_facet_key(tds, facet_key, second)
                    .map(|_| second)?;
                Ok(Self {
                    kind: FacetIncidenceKind::TwoSided([first, second]),
                })
            }
        }
    }

    /// Parses raw incident handles into a multiplicity-proof facet incidence.
    ///
    /// # Errors
    ///
    /// Returns [`FacetError::InvalidFacetMultiplicity`] unless the entry has one
    /// handle or two handles, and returns
    /// [`FacetError::DuplicateFacetIncidentHandle`] when a two-sided entry
    /// repeats the same handle.
    fn try_from_handles(
        facet_key: u64,
        handles: &SmallBuffer<FacetHandle, 2>,
    ) -> Result<Self, FacetError> {
        match handles.as_slice() {
            [handle] => Ok(Self {
                kind: FacetIncidenceKind::OneSided(*handle),
            }),
            [first, second] if first != second => Ok(Self {
                kind: FacetIncidenceKind::TwoSided([*first, *second]),
            }),
            [handle, _] => Err(FacetError::DuplicateFacetIncidentHandle {
                facet_key,
                handle: *handle,
            }),
            _ => Err(FacetError::InvalidFacetMultiplicity {
                facet_key,
                found: handles.len(),
            }),
        }
    }

    /// Returns true when this facet is incident to exactly one D-simplex.
    ///
    /// One-sided facets are open incidence candidates only. Periodic quotient
    /// triangulations may use one-sided self-identifications for closed
    /// topology, so manifold boundary classification belongs to the topology
    /// layer.
    #[inline]
    #[must_use]
    pub(crate) const fn is_one_sided(self) -> bool {
        matches!(self.kind, FacetIncidenceKind::OneSided(_))
    }

    /// Returns the number of incident D-simplices.
    #[inline]
    #[must_use]
    pub(crate) const fn incident_simplex_count(self) -> usize {
        match self.kind {
            FacetIncidenceKind::OneSided(_) => 1,
            FacetIncidenceKind::TwoSided(_) => 2,
        }
    }

    /// Returns the handle when this is a one-sided facet.
    #[inline]
    #[must_use]
    pub(crate) const fn one_sided_handle(self) -> Option<FacetHandle> {
        match self.kind {
            FacetIncidenceKind::OneSided(handle) => Some(handle),
            FacetIncidenceKind::TwoSided(_) => None,
        }
    }

    /// Returns the handles when this is a two-sided facet.
    #[inline]
    #[must_use]
    pub(crate) const fn two_sided_handles(self) -> Option<[FacetHandle; 2]> {
        match self.kind {
            FacetIncidenceKind::OneSided(_) => None,
            FacetIncidenceKind::TwoSided(handles) => Some(handles),
        }
    }
}

/// Parses a raw incident handle as belonging to one canonical facet-key entry.
///
/// # Errors
///
/// Returns the same errors as [`FacetHandle::view`] when the handle cannot be
/// reborrowed from `tds`, or [`FacetError::FacetHandleKeyMismatch`] when the
/// live facet view derives a different canonical key than the index entry.
pub(crate) fn try_incident_facet_view_for_facet_key<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    expected_facet_key: u64,
    handle: FacetHandle,
) -> Result<FacetView<'_, U, V, D>, FacetError> {
    let facet = handle.view(tds)?;
    let actual_facet_key = facet.key();
    if actual_facet_key == expected_facet_key {
        return Ok(facet);
    }

    Err(FacetError::FacetHandleKeyMismatch {
        expected_facet_key,
        actual_facet_key,
        handle,
    })
}

/// Borrowed view over one parsed facet-incidence entry.
///
/// The view borrows the [`FacetToSimplicesIndex`] entry and carries the [`Tds`]
/// that produced that index. This keeps the parsed multiplicity proof, facet
/// key, and canonical owner together for the lifetime of the index borrow.
#[must_use]
pub struct FacetIncidenceView<'idx, 'tds, U, V, const D: usize> {
    tds: &'tds Tds<U, V, D>,
    facet_key: u64,
    incidence: &'idx FacetIncidence,
}

impl<U, V, const D: usize> Clone for FacetIncidenceView<'_, '_, U, V, D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<U, V, const D: usize> Copy for FacetIncidenceView<'_, '_, U, V, D> {}

impl<'tds, U, V, const D: usize> FacetIncidenceView<'_, 'tds, U, V, D> {
    /// Returns the TDS that produced the borrowed incidence index entry.
    #[inline]
    #[must_use]
    pub const fn tds(self) -> &'tds Tds<U, V, D> {
        self.tds
    }

    /// Returns the canonical facet key for this incidence entry.
    #[inline]
    #[must_use]
    pub const fn facet_key(self) -> u64 {
        self.facet_key
    }

    /// Returns true when this facet is incident to exactly one D-simplex.
    #[inline]
    #[must_use]
    pub const fn is_one_sided(self) -> bool {
        self.incidence.is_one_sided()
    }

    /// Returns the number of incident D-simplices.
    #[inline]
    #[must_use]
    pub const fn incident_simplex_count(self) -> usize {
        self.incidence.incident_simplex_count()
    }

    /// Returns the handle when this is a one-sided facet.
    #[inline]
    #[must_use]
    pub const fn one_sided_handle(self) -> Option<FacetHandle> {
        self.incidence.one_sided_handle()
    }

    /// Returns the handles when this is a two-sided facet.
    #[inline]
    #[must_use]
    pub const fn two_sided_handles(self) -> Option<[FacetHandle; 2]> {
        self.incidence.two_sided_handles()
    }
}

/// Owner-bound derived index from facet keys to validated incident simplex facets.
///
/// The index owns its derived map but borrows the [`Tds`] that produced it.
/// Passing this wrapper instead of a raw map prevents boundary queries from
/// accidentally pairing a [`FacetView`] with incidence data from a different
/// triangulation. It also parses raw multiplicities into borrowed
/// [`FacetIncidenceView`] entries, so public boundary queries can operate on
/// proof-bearing incidence without detaching it from the producing index.
#[derive(Clone, Debug)]
#[must_use]
pub struct FacetToSimplicesIndex<'tds, U, V, const D: usize> {
    tds: &'tds Tds<U, V, D>,
    map: FastHashMap<u64, FacetIncidence>,
}

impl<'tds, U, V, const D: usize> FacetToSimplicesIndex<'tds, U, V, D> {
    /// Parses a freshly built raw facet map and binds it to the TDS that produced it.
    ///
    /// # Errors
    ///
    /// Returns [`FacetError`] when any raw incidence entry has invalid
    /// multiplicity, duplicate handles, stale handles, or handles whose live
    /// facet key does not match the map entry.
    #[inline]
    pub(crate) fn try_from_map(
        tds: &'tds Tds<U, V, D>,
        map: &FacetToSimplicesMap,
    ) -> Result<Self, FacetError> {
        let mut parsed = fast_hash_map_with_capacity(map.len());
        for (facet_key, handles) in map {
            let incidence = FacetIncidence::try_from_index_entry(tds, *facet_key, handles)?;
            parsed.insert(*facet_key, incidence);
        }
        Ok(Self { tds, map: parsed })
    }

    /// Returns the borrowed TDS that produced this index.
    #[inline]
    #[must_use]
    pub const fn tds(&self) -> &'tds Tds<U, V, D> {
        self.tds
    }

    /// Returns the parsed incident simplex facets for a canonical facet key.
    #[inline]
    #[must_use]
    pub fn get<'idx>(
        &'idx self,
        facet_key: &u64,
    ) -> Option<FacetIncidenceView<'idx, 'tds, U, V, D>> {
        self.map.get(facet_key).map(|incidence| FacetIncidenceView {
            tds: self.tds,
            facet_key: *facet_key,
            incidence,
        })
    }

    /// Returns true when `facet_key` has one-sided incidence.
    #[inline]
    #[must_use]
    pub fn is_one_sided_facet_key(&self, facet_key: &u64) -> bool {
        self.map
            .get(facet_key)
            .is_some_and(|incidence| incidence.is_one_sided())
    }

    /// Returns the number of indexed facet keys.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns whether the index contains no facet keys.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Iterates over borrowed facet-incidence entries.
    #[inline]
    pub fn iter<'idx>(
        &'idx self,
    ) -> impl Iterator<Item = FacetIncidenceView<'idx, 'tds, U, V, D>> + 'idx {
        let tds = self.tds;
        self.map
            .iter()
            .map(move |(facet_key, incidence)| FacetIncidenceView {
                tds,
                facet_key: *facet_key,
                incidence,
            })
    }

    /// Iterates over handles for parsed one-sided facet incidences.
    #[inline]
    pub(crate) fn one_sided_handles(&self) -> impl Iterator<Item = FacetHandle> + '_ {
        self.map
            .values()
            .filter_map(|incidence| incidence.one_sided_handle())
    }
}

/// Local neighbor metadata for a one-sided facet occurrence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use]
pub(crate) enum OneSidedFacetAdjacency {
    /// The owning simplex has no neighbor across this facet.
    Open,
    /// The owning simplex points to itself with periodic vertex offsets.
    PeriodicSelfIdentification,
}

/// Classifies the local neighbor metadata for a parsed one-sided facet.
pub(crate) fn classify_one_sided_facet_adjacency<U, V, const D: usize>(
    facet: &FacetView<'_, U, V, D>,
) -> Result<OneSidedFacetAdjacency, TdsError> {
    let facet_key = facet.key();
    let simplex_key = facet.simplex_key();
    let facet_index = usize::from(facet.facet_index());
    let simplex = facet.simplex();

    if facet_index >= simplex.number_of_vertices() {
        return Err(TdsError::IndexOutOfBounds {
            index: facet_index,
            bound: simplex.number_of_vertices(),
            context: format!(
                "one-sided facet adjacency classification for simplex {simplex_key:?}"
            ),
        });
    }

    let Some(neighbor) = simplex.neighbor_key(facet_index) else {
        return Ok(OneSidedFacetAdjacency::Open);
    };
    let Some(neighbor_key) = neighbor else {
        return Ok(OneSidedFacetAdjacency::Open);
    };

    if neighbor_key == simplex_key {
        if simplex_allows_periodic_self_neighbor(simplex) {
            return Ok(OneSidedFacetAdjacency::PeriodicSelfIdentification);
        }
        return Err(TdsError::InvalidNeighbors {
            reason: NeighborValidationError::BoundaryFacetHasNonPeriodicSelfNeighbor {
                facet_key,
                simplex_key,
                simplex_uuid: simplex.uuid(),
                facet_index,
            },
        });
    }

    Err(TdsError::InvalidNeighbors {
        reason: NeighborValidationError::BoundaryFacetHasNeighbor {
            facet_key,
            simplex_key,
            simplex_uuid: simplex.uuid(),
            facet_index,
            neighbor_key,
        },
    })
}

/// Mirrors TDS validation's periodic self-neighbor allowance for boundary queries.
fn simplex_allows_periodic_self_neighbor<V, const D: usize>(simplex: &Simplex<V, D>) -> bool {
    let Some(offsets) = simplex.periodic_vertex_offsets() else {
        return false;
    };
    !offsets.is_empty() && offsets.len() == simplex.number_of_vertices()
}

// Trait implementations for FacetView
impl<U, V, const D: usize> Debug for FacetView<'_, U, V, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FacetView")
            .field("simplex_key", &self.simplex_key)
            .field("facet_index", &self.facet_index)
            .field("facet_vertex_keys", &self.facet_vertex_keys)
            .field("key", &self.key)
            .field("dimension", &D)
            .finish()
    }
}

impl<U, V, const D: usize> Clone for FacetView<'_, U, V, D> {
    fn clone(&self) -> Self {
        Self {
            tds: self.tds,
            simplex: self.simplex,
            simplex_key: self.simplex_key,
            facet_index: self.facet_index,
            facet_vertex_keys: self.facet_vertex_keys.clone(),
            key: self.key,
            vertices: self.vertices.clone(),
            opposite_vertex: self.opposite_vertex,
        }
    }
}

impl<U, V, const D: usize> PartialEq for FacetView<'_, U, V, D> {
    fn eq(&self, other: &Self) -> bool {
        // Two facet views are equal if they reference the same facet
        std::ptr::eq(self.tds, other.tds)
            && self.simplex_key == other.simplex_key
            && self.facet_index == other.facet_index
    }
}

impl<U, V, const D: usize> Eq for FacetView<'_, U, V, D> {}

/// Iterator over the facets of one simplex in a triangulation data structure.
///
/// This iterator is lifetime-bound to the owning [`Tds`], so the returned
/// [`FacetView`] values cannot outlive the topology they observe. Construction is
/// fallible because the caller supplies a runtime [`SimplexKey`]; per-item
/// `FacetError`s still surface during iteration if the TDS is structurally
/// inconsistent.
///
/// Iteration follows the standard simplex boundary construction: for a simplex
/// with vertices `[v0, ..., vD]`, item `i` is the facet opposite `vi`, containing
/// all other vertices. The iterator therefore has exactly `D + 1` items for a
/// well-formed D-simplex.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::*;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Facet(#[from] delaunay::prelude::tds::FacetError),
/// #     #[error(transparent)]
/// #     Tds(#[from] delaunay::prelude::tds::TdsError),
/// #     #[error(transparent)]
/// #     Query(#[from] delaunay::query::QueryError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0, 1.0]?,
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
/// let Some((simplex_key, _)) = dt.simplices().next() else {
///     return Ok(());
/// };
///
/// let facets = dt.tds().try_simplex_facets(simplex_key)?;
/// assert_eq!(facets.len(), 4);
/// # Ok(())
/// # }
/// ```
#[must_use]
#[derive(Clone)]
pub struct SimplexFacetsIter<'tds, U, V, const D: usize> {
    tds: &'tds Tds<U, V, D>,
    simplex_key: SimplexKey,
    next_facet_index: u16,
    facet_count: u16,
}

impl<'tds, U, V, const D: usize> SimplexFacetsIter<'tds, U, V, D> {
    /// Creates a new iterator over the facets of `simplex_key`.
    ///
    /// # Errors
    ///
    /// Returns [`FacetError::SimplexNotFoundInTriangulation`] if `simplex_key`
    /// does not identify a simplex in `tds`, or
    /// [`FacetError::InvalidFacetIndexOverflow`] if the simplex has more facets
    /// than can be represented by the public `u8` facet-index storage.
    pub(crate) fn try_new(
        tds: &'tds Tds<U, V, D>,
        simplex_key: SimplexKey,
    ) -> Result<Self, FacetError> {
        let simplex = tds
            .simplex(simplex_key)
            .ok_or(FacetError::SimplexNotFoundInTriangulation)?;
        let facet_count_usize = simplex.number_of_vertices();
        let max_facet_count = usize::from(u8::MAX) + 1;
        if facet_count_usize > max_facet_count {
            return Err(FacetError::InvalidFacetIndexOverflow {
                original_index: max_facet_count,
                facet_count: facet_count_usize,
            });
        }
        let facet_count = u16::try_from(facet_count_usize).map_err(|_| {
            FacetError::InvalidFacetIndexOverflow {
                original_index: max_facet_count,
                facet_count: facet_count_usize,
            }
        })?;

        Ok(Self {
            tds,
            simplex_key,
            next_facet_index: 0,
            facet_count,
        })
    }
}

impl<'tds, U, V, const D: usize> Iterator for SimplexFacetsIter<'tds, U, V, D> {
    type Item = Result<FacetView<'tds, U, V, D>, FacetError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_facet_index >= self.facet_count {
            return None;
        }

        let facet_index = usize_to_u8(
            usize::from(self.next_facet_index),
            usize::from(self.facet_count),
        );
        self.next_facet_index += 1;
        Some(facet_index.and_then(|idx| FacetView::try_new(self.tds, self.simplex_key, idx)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len();
        (remaining, Some(remaining))
    }
}

impl<U, V, const D: usize> ExactSizeIterator for SimplexFacetsIter<'_, U, V, D> {
    fn len(&self) -> usize {
        usize::from(self.facet_count.saturating_sub(self.next_facet_index))
    }
}

impl<U, V, const D: usize> FusedIterator for SimplexFacetsIter<'_, U, V, D> {}

/// Iterator over all facets in a triangulation data structure.
///
/// This iterator provides efficient access to all facets without allocating
/// a vector. It's particularly useful for performance-critical operations
/// like boundary detection and cavity analysis in triangulation insertion.
/// Each item is a `Result` so structurally invalid facet views are reported
/// to callers instead of being skipped.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::*;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Facet(#[from] delaunay::prelude::tds::FacetError),
/// #     #[error(transparent)]
/// #     Tds(#[from] delaunay::prelude::tds::TdsError),
/// #     #[error(transparent)]
/// #     Query(#[from] delaunay::query::QueryError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0, 1.0]?,
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
///
/// let count = dt.tds().facets()
///     .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))?;
/// assert_eq!(count, 4);
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct AllFacetsIter<'tds, U, V, const D: usize> {
    tds: &'tds Tds<U, V, D>,
    simplex_keys: slotmap::dense::Keys<'tds, SimplexKey, Simplex<V, D>>,
    state: AllFacetsIterState,
}

/// Encodes whether facet iteration is between simplices, inside one, or done.
#[derive(Clone)]
enum AllFacetsIterState {
    PendingError(FacetError),
    PendingSimplex,
    InSimplex {
        simplex_key: SimplexKey,
        next_facet_index: usize,
        facet_count: usize,
    },
    Exhausted,
}

impl<'tds, U, V, const D: usize> AllFacetsIter<'tds, U, V, D> {
    /// Creates a new iterator over all facets in the TDS.
    ///
    /// The iterator itself is infallible to construct. Structural or dimension
    /// errors are reported as iterator items.
    #[must_use]
    pub(crate) fn from_tds(tds: &'tds Tds<U, V, D>) -> Self {
        let state = if D > usize::from(u8::MAX) {
            AllFacetsIterState::PendingError(FacetError::FacetIndexCapacityExceeded {
                dimension: D,
                max_dimension: usize::from(u8::MAX),
            })
        } else {
            AllFacetsIterState::PendingSimplex
        };

        Self {
            tds,
            simplex_keys: tds.simplex_key_iter(),
            state,
        }
    }

    /// Creates a new iterator over all facets in the TDS.
    ///
    /// # Errors
    ///
    /// Returns [`FacetError::FacetIndexCapacityExceeded`] if `D > 255`, since
    /// facet indices are stored as `u8`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)]
    /// #     Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)]
    /// #     Query(#[from] delaunay::query::QueryError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// let mut iter = dt.tds().facets();
    /// assert!(iter.next().transpose()?.is_some());
    /// # Ok(())
    /// # }
    /// ```
    pub(crate) fn try_new(tds: &'tds Tds<U, V, D>) -> Result<Self, FacetError> {
        // Dimension check: facets per simplex = D+1, so D must be <= 255
        if D > usize::from(u8::MAX) {
            return Err(FacetError::FacetIndexCapacityExceeded {
                dimension: D,
                max_dimension: usize::from(u8::MAX),
            });
        }

        Ok(Self::from_tds(tds))
    }
}

impl<U, V, const D: usize> Tds<U, V, D> {
    /// Returns an iterator over all facets of one simplex in the TDS.
    ///
    /// This is the owner-bound API for per-simplex facet views. It constructs no
    /// `Vec`; callers that need an owned collection can collect the iterator and
    /// decide how to handle per-item [`FacetError`] values.
    ///
    /// # Errors
    ///
    /// Returns [`FacetError::SimplexNotFoundInTriangulation`] if `simplex_key`
    /// does not identify a simplex in this TDS, or [`FacetError::InvalidFacetIndex`]
    /// if the simplex has more facets than can be represented by the public `u8`
    /// facet-index storage.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)]
    /// #     Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// let facet_count = dt
    ///     .tds()
    ///     .try_simplex_facets(simplex_key)?
    ///     .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))?;
    ///
    /// assert_eq!(facet_count, 4);
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_simplex_facets(
        &self,
        simplex_key: SimplexKey,
    ) -> Result<SimplexFacetsIter<'_, U, V, D>, FacetError> {
        SimplexFacetsIter::try_new(self, simplex_key)
    }

    /// Returns an iterator over all facets in the TDS.
    ///
    /// This is the TDS-level counterpart to
    /// [`Triangulation::boundary_facets`](crate::Triangulation::boundary_facets).
    /// The iterator itself is infallible to construct; individual iterator items
    /// return [`FacetError`] if a facet view cannot be constructed from the
    /// current TDS state.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)]
    /// #     Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let facet_count = dt
    ///     .tds()
    ///     .facets()
    ///     .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))?;
    ///
    /// assert_eq!(facet_count, 4);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn facets(&self) -> AllFacetsIter<'_, U, V, D> {
        AllFacetsIter::from_tds(self)
    }
}

impl<'tds, U, V, const D: usize> Iterator for AllFacetsIter<'tds, U, V, D> {
    type Item = Result<FacetView<'tds, U, V, D>, FacetError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match &mut self.state {
                AllFacetsIterState::PendingError(error) => {
                    let error = error.clone();
                    self.state = AllFacetsIterState::Exhausted;
                    return Some(Err(error));
                }
                AllFacetsIterState::InSimplex {
                    simplex_key,
                    next_facet_index,
                    facet_count,
                } if *next_facet_index < *facet_count => {
                    let facet_index = *next_facet_index;
                    *next_facet_index += 1;

                    let facet_u8 = match usize_to_u8(facet_index, *facet_count) {
                        Ok(facet_u8) => facet_u8,
                        Err(err) => return Some(Err(err)),
                    };
                    return Some(FacetView::try_new(self.tds, *simplex_key, facet_u8));
                }
                AllFacetsIterState::Exhausted => return None,
                AllFacetsIterState::PendingSimplex | AllFacetsIterState::InSimplex { .. } => {
                    if let Some(next_simplex_key) = self.simplex_keys.next() {
                        if let Some(simplex) = self.tds.simplex(next_simplex_key) {
                            self.state = AllFacetsIterState::InSimplex {
                                simplex_key: next_simplex_key,
                                next_facet_index: 0,
                                facet_count: simplex.number_of_vertices(),
                            };
                        } else {
                            return Some(Err(FacetError::SimplexNotFoundInTriangulation));
                        }
                    } else {
                        self.state = AllFacetsIterState::Exhausted;
                        return None;
                    }
                }
            }
        }
    }
}

/// Iterator over topology-approved boundary facets in a triangulation.
///
/// This iterator yields facets whose keys were preclassified as true manifold
/// boundary by the topology layer.
/// It owns the topology-approved handles in deterministic storage order while
/// borrowing the TDS that produced them. Each item is a `Result` so facet-view
/// construction failures propagate to the caller during iteration.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::*;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Facet(#[from] delaunay::prelude::tds::FacetError),
/// #     #[error(transparent)]
/// #     Tds(#[from] delaunay::prelude::tds::TdsError),
/// #     #[error(transparent)]
/// #     Query(#[from] delaunay::query::QueryError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0, 1.0]?,
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
/// let count = dt.boundary_facets()?
///     .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))?;
/// assert_eq!(count, 4);
/// # Ok(())
/// # }
/// ```
#[must_use]
#[derive(Clone)]
pub struct BoundaryFacetsIter<'tds, U, V, const D: usize> {
    tds: &'tds Tds<U, V, D>,
    boundary_facet_handles: IntoIter<FacetHandle>,
}

impl<'tds, U, V, const D: usize> BoundaryFacetsIter<'tds, U, V, D> {
    /// Creates a new iterator over boundary facets.
    ///
    /// # Errors
    ///
    /// Returns [`FacetError::FacetIndexCapacityExceeded`] if this dimension
    /// cannot be represented by the current `u8` facet-index storage. Also
    /// returns [`FacetError`] if any supplied handle is stale, missing from the
    /// parsed index, not the indexed one-sided handle for its facet key, or no
    /// longer reborrows as a live [`FacetView`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)]
    /// #     Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)]
    /// #     Query(#[from] delaunay::query::QueryError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let mut iter = dt.boundary_facets()?;
    /// assert!(iter.next().transpose()?.is_some());
    /// # Ok(())
    /// # }
    /// ```
    pub(crate) fn try_new(
        facet_to_simplices_index: &FacetToSimplicesIndex<'tds, U, V, D>,
        mut boundary_facet_handles: Vec<FacetHandle>,
    ) -> Result<Self, FacetError> {
        let tds = facet_to_simplices_index.tds();
        AllFacetsIter::try_new(tds)?;
        for handle in &mut boundary_facet_handles {
            *handle = try_one_sided_handle_from_index(facet_to_simplices_index, *handle)?;
        }
        sort_handles_by_storage_order(&mut boundary_facet_handles);
        Ok(Self {
            tds,
            boundary_facet_handles: boundary_facet_handles.into_iter(),
        })
    }
}

impl<'tds, U, V, const D: usize> Iterator for BoundaryFacetsIter<'tds, U, V, D> {
    type Item = Result<FacetView<'tds, U, V, D>, FacetError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.boundary_facet_handles
            .next()
            .map(|handle| handle.view(self.tds))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.boundary_facet_handles.size_hint()
    }
}

impl<U, V, const D: usize> ExactSizeIterator for BoundaryFacetsIter<'_, U, V, D> {}

impl<U, V, const D: usize> FusedIterator for BoundaryFacetsIter<'_, U, V, D> {}

/// Iterator over one-sided facet incidences in a TDS.
///
/// This is a TDS-level incidence traversal, not a topology-aware boundary
/// query. Closed periodic self-identifications can be one-sided in the quotient
/// incidence index without being manifold boundary. The iterator owns a sorted
/// handle list derived from the parsed incidence index while borrowing the TDS
/// that produced it.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::*;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Facet(#[from] delaunay::prelude::tds::FacetError),
/// #     #[error(transparent)]
/// #     Tds(#[from] delaunay::prelude::tds::TdsError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0, 1.0]?,
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
///
/// let one_sided_count = dt
///     .tds()
///     .one_sided_facets()?
///     .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))?;
/// assert_eq!(one_sided_count, 4);
/// # Ok(())
/// # }
/// ```
#[must_use]
#[derive(Clone)]
pub struct OneSidedFacetsIter<'tds, U, V, const D: usize> {
    tds: &'tds Tds<U, V, D>,
    one_sided_facet_handles: IntoIter<FacetHandle>,
}

impl<'tds, U, V, const D: usize> OneSidedFacetsIter<'tds, U, V, D> {
    /// Creates a new iterator over one-sided facet incidences.
    ///
    /// # Errors
    ///
    /// Returns [`FacetError::FacetIndexCapacityExceeded`] if this dimension
    /// cannot be represented by the current `u8` facet-index storage.
    pub(crate) fn try_new(
        facet_to_simplices_index: &FacetToSimplicesIndex<'tds, U, V, D>,
    ) -> Result<Self, FacetError> {
        let tds = facet_to_simplices_index.tds();
        AllFacetsIter::try_new(tds)?;
        let mut one_sided_facet_handles = facet_to_simplices_index
            .one_sided_handles()
            .collect::<Vec<_>>();
        sort_handles_by_storage_order(&mut one_sided_facet_handles);
        Ok(Self {
            tds,
            one_sided_facet_handles: one_sided_facet_handles.into_iter(),
        })
    }
}

impl<'tds, U, V, const D: usize> Iterator for OneSidedFacetsIter<'tds, U, V, D> {
    type Item = Result<FacetView<'tds, U, V, D>, FacetError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.one_sided_facet_handles
            .next()
            .map(|handle| handle.view(self.tds))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.one_sided_facet_handles.size_hint()
    }
}

impl<U, V, const D: usize> ExactSizeIterator for OneSidedFacetsIter<'_, U, V, D> {}

impl<U, V, const D: usize> FusedIterator for OneSidedFacetsIter<'_, U, V, D> {}

/// Parses a supplied handle as the indexed one-sided handle for its facet key.
///
/// # Errors
///
/// Returns the same errors as [`FacetHandle::view`] when `handle` cannot be
/// reborrowed from the index's TDS. Returns
/// [`FacetError::FacetKeyNotFoundInCache`] when the handle's facet key is absent
/// from the parsed index, [`FacetError::BoundaryFacetHandleNotIndexed`] when a
/// different one-sided handle is indexed for that key, or
/// [`FacetError::InvalidAdjacentSimplexCount`] when the key is not one-sided.
fn try_one_sided_handle_from_index<U, V, const D: usize>(
    facet_to_simplices_index: &FacetToSimplicesIndex<'_, U, V, D>,
    handle: FacetHandle,
) -> Result<FacetHandle, FacetError> {
    let facet = handle.view(facet_to_simplices_index.tds())?;
    let facet_key = facet.key();
    let Some(incidence) = facet_to_simplices_index.get(&facet_key) else {
        let vertex_uuids = facet.vertices().map(Vertex::uuid).collect();
        return Err(FacetError::FacetKeyNotFoundInCache {
            facet_key,
            cache_size: facet_to_simplices_index.len(),
            vertex_uuids,
        });
    };
    match incidence.one_sided_handle() {
        Some(indexed_handle) if indexed_handle == handle => Ok(handle),
        Some(indexed_handle) => Err(FacetError::BoundaryFacetHandleNotIndexed {
            facet_key,
            supplied_handle: handle,
            indexed_handle,
        }),
        None => Err(FacetError::InvalidAdjacentSimplexCount {
            found: incidence.incident_simplex_count(),
        }),
    }
}

/// Sorts handles by storage key and local facet index for deterministic iteration.
fn sort_handles_by_storage_order(handles: &mut [FacetHandle]) {
    handles.sort_unstable_by_key(|handle| {
        (handle.simplex_key().data().as_ffi(), handle.facet_index())
    });
}

// =============================================================================
// FACET KEY GENERATION FUNCTIONS
// =============================================================================

/// Generates a canonical facet key from sorted 64-bit `VertexKey` arrays.
///
/// This function creates a deterministic facet key by:
/// 1. Converting `VertexKeys` to 64-bit integers using their internal `KeyData`
/// 2. Sorting the keys to ensure deterministic ordering regardless of input order
/// 3. Combining the keys using an efficient bitwise hash algorithm
///
/// The resulting key is guaranteed to be identical for any facet that contains
/// the same set of vertices, regardless of the order in which the vertices are provided.
///
/// # Arguments
///
/// * `vertices` - A slice of `VertexKeys` representing the vertices of the facet
///
/// # Returns
///
/// A `u64` hash value representing the canonical key of the facet
///
/// # Performance
///
/// This method is optimized for performance:
/// - Time Complexity: O(n log n) where n is the number of vertices (due to sorting)
/// - Space Complexity: O(n) for the temporary sorted array
/// - Uses efficient bitwise operations for hash combination
/// - Avoids heap allocation when possible
///
/// # Examples
///
/// ```
/// use delaunay::prelude::tds::facet_key_from_vertices;
/// use delaunay::prelude::tds::VertexKey;
/// use slotmap::Key;
///
/// // Create some vertex keys (normally these would come from a TDS)
/// let vertices = vec![
///     VertexKey::from(slotmap::KeyData::from_ffi(1u64)),
///     VertexKey::from(slotmap::KeyData::from_ffi(2u64)),
///     VertexKey::from(slotmap::KeyData::from_ffi(3u64)),
/// ];
///
/// // Generate facet key from vertex keys
/// let facet_key = facet_key_from_vertices(&vertices);
///
/// // The same vertices in different order should produce the same key
/// let mut reversed_keys = vertices.clone();
/// reversed_keys.reverse();
/// let facet_key_reversed = facet_key_from_vertices(&reversed_keys);
/// assert_eq!(facet_key, facet_key_reversed);
/// ```
///
/// # Algorithm Details
///
/// The hash combination uses a polynomial rolling hash approach:
/// 1. Start with an initial hash value
/// 2. For each sorted vertex key, combine it using: `hash = hash.wrapping_mul(PRIME).wrapping_add(key)`
/// 3. Apply a final avalanche step to improve bit distribution
///
/// This approach ensures:
/// - Good hash distribution across the output space
/// - Deterministic results independent of vertex ordering
/// - Efficient computation with minimal allocations
#[must_use]
pub fn facet_key_from_vertices(vertices: &[VertexKey]) -> u64 {
    // Handle empty case
    if vertices.is_empty() {
        return 0;
    }

    // Convert VertexKeys to u64 and sort for deterministic ordering
    let mut key_values: SmallBuffer<u64, MAX_PRACTICAL_DIMENSION_SIZE> =
        vertices.iter().map(|key| key.data().as_ffi()).collect();
    key_values.sort_unstable();

    // Use the shared stable hash function
    stable_hash_u64_slice(&key_values)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::construction::{
        ConstructionOptions, InitialSimplexStrategy, InsertionOrderStrategy,
    };
    use crate::core::tds::{Tds, VertexKey};
    use crate::core::triangulation::Triangulation;
    use crate::core::validation::TopologyGuarantee;
    use crate::core::vertex::Vertex;
    use crate::geometry::kernel::AdaptiveKernel;
    use crate::triangulation::DelaunayTriangulation;
    use crate::vertex;
    use slotmap::{KeyData, SlotMap};
    use std::assert_matches;
    use std::{collections::HashSet, mem};

    // =============================================================================
    // UNIT TESTS FOR HELPER FUNCTIONS
    // =============================================================================

    #[test]
    fn test_usize_to_u8_conversion() {
        // Test successful conversion
        assert_eq!(usize_to_u8(0, 4), Ok(0));
        assert_eq!(usize_to_u8(1, 4), Ok(1));
        assert_eq!(usize_to_u8(255, 256), Ok(255));

        // Test conversion at boundary
        assert_eq!(usize_to_u8(u8::MAX as usize, 256), Ok(u8::MAX));

        // Test failed conversion (index too large)
        let result = usize_to_u8(256, 10);
        assert!(result.is_err());
        if let Err(FacetError::InvalidFacetIndexOverflow {
            original_index,
            facet_count,
        }) = result
        {
            assert_eq!(original_index, 256); // Should preserve original value
            assert_eq!(facet_count, 10);
        } else {
            panic!("Expected InvalidFacetIndexOverflow error");
        }

        // Test failed conversion (very large index)
        let result = usize_to_u8(usize::MAX, 5);
        assert!(result.is_err());
        if let Err(FacetError::InvalidFacetIndexOverflow {
            original_index,
            facet_count,
        }) = result
        {
            assert_eq!(original_index, usize::MAX);
            assert_eq!(facet_count, 5);
        } else {
            panic!("Expected InvalidFacetIndexOverflow error");
        }
    }

    // =============================================================================
    // FACET CREATION TESTS
    // =============================================================================

    #[test]
    fn test_facet_error_handling() {
        // Create a 1D triangulation (2 vertices forming an edge)
        let vertices = vec![vertex!([0.0]).unwrap(), vertex!([1.0]).unwrap()];
        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        // Test invalid facet index (should be 0 or 1 for 1D, facet_index >= 2 is invalid)
        assert_matches!(
            FacetView::try_new(dt.tds(), simplex_key, 99),
            Err(FacetError::InvalidFacetIndex { .. })
        );
    }

    #[test]
    fn facet_new() {
        // Create a 3D triangulation with a tetrahedron
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        // Create facet view for facet 0 (excludes vertex 0)
        let facet = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();
        assert_eq!(facet.simplex_key(), simplex_key);
        assert_eq!(facet.facet_index(), 0);
    }

    #[test]
    fn facet_handle_view_roundtrip() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        let handle = FacetHandle::try_new(dt.tds(), simplex_key, 1).unwrap();
        let view = handle.view(dt.tds()).unwrap();

        assert_eq!(view.simplex_key(), simplex_key);
        assert_eq!(view.facet_index(), 1);
        assert_eq!(view.handle(), handle);
    }

    #[test]
    fn facet_handle_view_revalidates_against_tds() {
        let tds: Tds<(), (), 3> = Tds::empty();
        let handle = FacetHandle::from_validated(SimplexKey::default(), 0);

        assert_matches!(
            handle.view(&tds),
            Err(FacetError::SimplexNotFoundInTriangulation)
        );
    }

    #[test]
    fn test_facet_new_success_coverage() {
        // Test 2D case: Create a triangle (2D simplex with 3 vertices)
        let vertices_2d = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.5, 1.0]).unwrap(),
        ];
        let dt_2d = DelaunayTriangulation::try_new(&vertices_2d).unwrap();
        let simplex_key_2d = dt_2d.simplices().next().unwrap().0;
        let result_2d = FacetView::try_new(dt_2d.tds(), simplex_key_2d, 0);

        // Assert that the result is Ok
        assert!(result_2d.is_ok());
        let facet_2d = result_2d.unwrap();
        assert_eq!(facet_2d.vertices().count(), 2); // 2D facet should have 2 vertices

        // Test 1D case: Create an edge (1D simplex with 2 vertices)
        let vertices_1d = vec![vertex!([0.0]).unwrap(), vertex!([1.0]).unwrap()];
        let dt_1d = DelaunayTriangulation::try_new(&vertices_1d).unwrap();
        let simplex_key_1d = dt_1d.simplices().next().unwrap().0;
        let result_1d = FacetView::try_new(dt_1d.tds(), simplex_key_1d, 0);

        // Assert that the result is Ok
        assert!(result_1d.is_ok());
        let facet_1d = result_1d.unwrap();
        assert_eq!(facet_1d.vertices().count(), 1); // 1D facet should have 1 vertex
    }

    #[test]
    fn facet_new_with_incorrect_vertex() {
        // Create a 3D triangulation
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        // Test invalid facet index (3D simplex has vertices 0-3, facet index 4 is invalid)
        assert!(FacetView::try_new(dt.tds(), simplex_key, 4).is_err());
    }

    #[test]
    fn facet_vertices() {
        // Create a 3D triangulation with a tetrahedron
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        // Create facet view for facet 0 (excludes vertex 0)
        let facet = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();
        assert_eq!(facet.vertices().count(), 3);
    }

    // =============================================================================
    // EQUALITY AND ORDERING TESTS
    // =============================================================================

    #[test]
    fn facet_partial_eq() {
        // Create a 3D triangulation
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        // Create facet views with same facet index (should be equal)
        let facet1 = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();
        let facet2 = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();
        let facet3 = FacetView::try_new(dt.tds(), simplex_key, 1).unwrap();

        assert_eq!(facet1, facet2);
        assert_ne!(facet1, facet3);
    }

    #[test]
    fn facet_clone() {
        // Create a 3D triangulation
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        let facet = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();
        let cloned_facet = facet.clone();

        // Verify clones are equal
        assert_eq!(facet, cloned_facet);
        assert_eq!(facet.simplex_key(), cloned_facet.simplex_key());
        assert_eq!(facet.facet_index(), cloned_facet.facet_index());

        // Verify simplex and opposite vertex are accessible through both views
        let simplex1 = facet.simplex();
        let simplex2 = cloned_facet.simplex();
        assert_eq!(simplex1.uuid(), simplex2.uuid());

        let vertex1 = facet.opposite_vertex();
        let vertex2 = cloned_facet.opposite_vertex();
        assert_eq!(vertex1.uuid(), vertex2.uuid());
    }

    #[test]
    fn facet_debug() {
        // Create a 3D triangulation with a non-degenerate tetrahedron
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        let facet = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();
        let debug_str = format!("{facet:?}");

        assert!(debug_str.contains("FacetView"));
        assert!(debug_str.contains("simplex_key"));
        assert!(debug_str.contains("facet_index"));
        assert!(debug_str.contains("dimension"));
    }

    // =============================================================================
    // DIMENSIONAL AND GEOMETRIC TESTS
    // =============================================================================

    #[test]
    fn facet_with_typed_data() {
        // Create 3D triangulation with typed vertex data
        let vertices: Vec<Vertex<i32, 3>> = vec![
            vertex!([0.0, 0.0, 0.0]; data = 1).unwrap(),
            vertex!([1.0, 0.0, 0.0]; data = 2).unwrap(),
            vertex!([0.0, 1.0, 0.0]; data = 3).unwrap(),
            vertex!([0.0, 0.0, 1.0]; data = 4).unwrap(),
        ];
        let options = ConstructionOptions::default()
            .with_insertion_order(InsertionOrderStrategy::Input)
            .with_initial_simplex_strategy(InitialSimplexStrategy::First);
        let dt: DelaunayTriangulation<AdaptiveKernel<f64>, i32, (), 3> =
            DelaunayTriangulation::try_with_topology_guarantee_and_options(
                &AdaptiveKernel::new(),
                &vertices,
                TopologyGuarantee::DEFAULT,
                options,
            )
            .unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        // Create facet view for facet 0 (excludes vertex 0)
        let facet = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();

        let facet_vertices: Vec<_> = facet.vertices().collect();
        assert_eq!(facet_vertices.len(), 3); // 3D facet should have 3 vertices (D)
        let simplex = dt.tds().simplex(simplex_key).expect("simplex exists");
        for &vertex_key in simplex.vertices().iter().skip(1) {
            let expected_data = dt.tds().vertex(vertex_key).unwrap().data;
            assert!(
                facet_vertices.iter().any(|v| v.data == expected_data),
                "Expected facet vertex data {expected_data:?} not found"
            );
        }
    }

    /// Macro to generate dimension-specific facet tests for dimensions 2D-5D.
    ///
    /// This macro reduces test duplication by generating consistent tests across
    /// multiple dimensions. It creates tests for:
    /// - Basic facet view creation and vertex count validation
    /// - `FacetKey` computation and consistency
    /// - Facet equality tests
    ///
    /// For a D-dimensional simplex, a facet is (D-1)-dimensional and has D vertices.
    ///
    /// # Usage
    ///
    /// ```ignore
    /// test_facet_dimensions! {
    ///     facet_2d => 2 => "triangle" => 2 => vec![delaunay::vertex![0.0, 0.0]?, ...],
    /// }
    /// ```
    macro_rules! test_facet_dimensions {
        ($(
            $test_name:ident => $dim:expr => $desc:expr => $expected_facet_vertices:expr => $vertices:expr
        ),+ $(,)?) => {
            $(
                #[test]
                fn $test_name() {
                    // Test basic facet view creation
                    let vertices = $vertices;
                    let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
                    let simplex_key = dt.simplices().next().unwrap().0;

                    // Create facet view for facet 0 (excludes vertex 0)
                    let facet = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();

                    // Facet of D-dimensional simplex is (D-1)-dimensional with D vertices
                    assert_eq!(facet.vertices().count(), $expected_facet_vertices,
                        "Facet of {}D {} should have {} vertices", $dim, $desc, $expected_facet_vertices);
                }

                pastey::paste! {
                    #[test]
                    fn [<$test_name _key_consistency>]() {
                        // Test FacetKey computation consistency
                        let vertices = $vertices;
                        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
                        let simplex_key = dt.simplices().next().unwrap().0;

                        // Create same facet twice
                        let facet1 = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();
                        let facet2 = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();

                        assert_eq!(facet1.key(), facet2.key(),
                            "Same facet should produce same key");

                        // Create different facet
                        let facet3 = FacetView::try_new(dt.tds(), simplex_key, 1).unwrap();
                        assert_ne!(facet1.key(), facet3.key(),
                            "Different facets should produce different keys");
                    }

                    #[test]
                    fn [<$test_name _equality>]() {
                        // Test facet equality comparison
                        let vertices = $vertices;
                        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
                        let simplex_key = dt.simplices().next().unwrap().0;

                        let facet1 = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();
                        let facet2 = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();
                        let facet3 = FacetView::try_new(dt.tds(), simplex_key, 1).unwrap();

                        assert!(facet1 == facet2, "Same facet should be equal");
                        assert!(facet1 != facet3, "Different facets should not be equal");
                    }

                    #[test]
                    fn [<$test_name _all_facets>]() {
                        // Test iterating through all facets of a simplex
                        let vertices = $vertices;
                        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
                        let simplex_key = dt.simplices().next().unwrap().0;

                        // D+1 dimensional simplex should have D+1 facets (one opposite each vertex)
                        let expected_facets = $dim + 1;
                        let mut facet_keys = HashSet::new();

                        for i in 0..expected_facets {
                            let facet = FacetView::try_new(dt.tds(), simplex_key, u8::try_from(i).unwrap()).unwrap();
                            facet_keys.insert(facet.key());
                        }

                        assert_eq!(facet_keys.len(), expected_facets,
                            "{}D simplex should have {} unique facets", $dim, expected_facets);
                    }
                }
            )+
        };
    }

    // Generate tests for dimensions 2D through 5D
    test_facet_dimensions! {
        facet_2d_triangle => 2 => "triangle" => 2 => vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.5, 1.0]).unwrap(),
        ],
        facet_3d_tetrahedron => 3 => "tetrahedron" => 3 => vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ],
        facet_4d_simplex => 4 => "4-simplex" => 4 => vec![
            vertex!([0.0, 0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 0.0, 1.0]).unwrap(),
        ],
        facet_5d_simplex => 5 => "5-simplex" => 5 => vec![
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 0.0, 0.0, 1.0]).unwrap(),
        ],
    }

    // Keep 1D test separate as it's less common
    #[test]
    fn facet_1d_edge() {
        // Create 1D triangulation (edge with 2 vertices)
        let vertices = vec![vertex!([0.0]).unwrap(), vertex!([1.0]).unwrap()];
        let options = ConstructionOptions::default()
            .with_insertion_order(InsertionOrderStrategy::Input)
            .with_initial_simplex_strategy(InitialSimplexStrategy::First);
        let dt = DelaunayTriangulation::try_new_with_options(&vertices, options).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        // Create facet view for facet 0 (excludes vertex 0)
        let facet = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();

        // Facet of 1D edge is a point (0D) with 1 vertex
        assert_eq!(facet.vertices().count(), 1);
    }

    #[test]
    fn all_facets_iter_rejects_dimension_above_u8_facet_index_capacity() {
        let tds: Tds<(), (), 256> = Tds::empty();
        let Err(err) = AllFacetsIter::try_new(&tds) else {
            panic!("D=256 cannot fit facet indices in u8");
        };

        assert_matches!(
            err,
            FacetError::FacetIndexCapacityExceeded {
                dimension: 256,
                max_dimension: 255,
            }
        );
    }

    #[test]
    fn tds_facets_reports_dimension_capacity_as_iterator_item() {
        let tds: Tds<(), (), 256> = Tds::empty();
        let mut facets = tds.facets();

        assert_matches!(
            facets.next(),
            Some(Err(FacetError::FacetIndexCapacityExceeded {
                dimension: 256,
                max_dimension: 255,
            }))
        );
        assert!(facets.next().is_none());
    }

    #[test]
    fn try_simplex_facets_supports_d255_full_u8_index_range() {
        let mut tds: Tds<(), (), 255> = Tds::empty();
        let mut vertex_keys = Vec::with_capacity(usize::from(u8::MAX) + 1);
        for i in 0..=usize::from(u8::MAX) {
            let mut coords = [0.0; 255];
            coords[0] = f64::from(u32::try_from(i).unwrap());
            let vertex = vertex!(coords).unwrap();
            vertex_keys.push(tds.insert_vertex_with_mapping(vertex).unwrap());
        }
        let simplex_key = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vertex_keys, None).unwrap())
            .unwrap();

        let mut facets = tds.try_simplex_facets(simplex_key).unwrap();

        assert_eq!(facets.len(), usize::from(u8::MAX) + 1);
        for expected_index in 0..=u8::MAX {
            let facet = facets.next().unwrap().unwrap();
            assert_eq!(facet.facet_index(), expected_index);
        }
        assert!(facets.next().is_none());
    }

    /// Builds a deliberately corrupted 2D TDS whose lone simplex has more
    /// vertices than can be represented by the `u8` facet-index storage.
    fn overwide_simplex_tds() -> Tds<(), (), 2> {
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let mut tds =
            Triangulation::<AdaptiveKernel<f64>, (), (), 2>::build_initial_simplex(&vertices)
                .unwrap();
        let simplex_key = tds.simplex_keys().next().unwrap();
        let first_vertex = tds.simplex(simplex_key).unwrap().vertices()[0];

        {
            let simplex = tds.simplex_mut(simplex_key).unwrap();
            while simplex.number_of_vertices() <= usize::from(u8::MAX) + 1 {
                simplex.push_vertex_key(first_vertex);
            }
        }

        tds
    }

    fn tetrahedron_tds() -> Tds<(), (), 3> {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        Triangulation::<AdaptiveKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap()
    }

    fn first_facet_view(tds: &Tds<(), (), 3>) -> FacetView<'_, (), (), 3> {
        tds.facets().next().unwrap().unwrap()
    }

    #[test]
    fn all_facets_iter_yields_overflow_error() {
        let tds = overwide_simplex_tds();
        let mut iter = tds.facets();

        for facet in iter.by_ref().take(usize::from(u8::MAX) + 1) {
            assert!(
                facet.is_ok(),
                "facet indices up to u8::MAX remain representable"
            );
        }

        assert_matches!(
            iter.next(),
            Some(Err(FacetError::InvalidFacetIndexOverflow {
                original_index: 256,
                facet_count: 257,
            }))
        );
    }

    #[test]
    fn all_facets_iter_stays_exhausted_after_completion() {
        let tds = tetrahedron_tds();
        let mut iter = tds.facets();

        while iter.next().transpose().unwrap().is_some() {}

        assert!(iter.next().is_none());
    }

    #[test]
    fn boundary_facets_iter_yields_supplied_handles_in_storage_order() {
        let tds = tetrahedron_tds();
        let mut facet_to_simplices = FacetToSimplicesMap::default();
        for facet in tds.facets() {
            let facet = facet.unwrap();
            let mut incidents = SmallBuffer::new();
            let handle = FacetHandle::from_validated(facet.simplex_key(), facet.facet_index());
            incidents.push(handle);
            facet_to_simplices.insert(facet.key(), incidents);
        }
        let facet_to_simplices_index =
            FacetToSimplicesIndex::try_from_map(&tds, &facet_to_simplices).unwrap();
        let mut boundary_facet_handles = facet_to_simplices_index
            .one_sided_handles()
            .collect::<Vec<_>>();
        boundary_facet_handles.reverse();
        let mut iter =
            BoundaryFacetsIter::try_new(&facet_to_simplices_index, boundary_facet_handles).unwrap();

        assert_eq!(iter.len(), 4);
        for expected_index in 0..4 {
            let facet = iter.next().transpose().unwrap().unwrap();
            assert_eq!(usize::from(facet.facet_index()), expected_index);
        }
        assert!(iter.next().is_none());
    }

    #[test]
    fn one_sided_facets_iter_reports_len_and_storage_order() {
        let tds = tetrahedron_tds();
        let mut facet_to_simplices = FacetToSimplicesMap::default();
        for facet in tds.facets() {
            let facet = facet.unwrap();
            let mut incidents = SmallBuffer::new();
            let handle = FacetHandle::from_validated(facet.simplex_key(), facet.facet_index());
            incidents.push(handle);
            facet_to_simplices.insert(facet.key(), incidents);
        }
        let facet_to_simplices_index =
            FacetToSimplicesIndex::try_from_map(&tds, &facet_to_simplices).unwrap();
        let mut iter = OneSidedFacetsIter::try_new(&facet_to_simplices_index).unwrap();

        assert_eq!(iter.len(), 4);
        for expected_index in 0..4 {
            let facet = iter.next().transpose().unwrap().unwrap();
            assert_eq!(usize::from(facet.facet_index()), expected_index);
        }
        assert!(iter.next().is_none());
    }

    #[test]
    fn boundary_facets_iter_revalidates_supplied_handles() {
        let tds = tetrahedron_tds();
        let facet_to_simplices_index =
            FacetToSimplicesIndex::try_from_map(&tds, &FacetToSimplicesMap::default()).unwrap();
        let stale_handle =
            FacetHandle::from_validated(SimplexKey::from(KeyData::from_ffi(0xDEAD)), 0);
        let Err(error) = BoundaryFacetsIter::try_new(&facet_to_simplices_index, vec![stale_handle])
        else {
            panic!("expected stale boundary handle to be rejected");
        };

        assert_matches!(error, FacetError::SimplexNotFoundInTriangulation);
    }

    #[test]
    fn boundary_facets_iter_rejects_handle_missing_from_index() {
        let tds = tetrahedron_tds();
        let first_facet = first_facet_view(&tds);
        let handle = first_facet.handle();
        let facet_to_simplices_index =
            FacetToSimplicesIndex::try_from_map(&tds, &FacetToSimplicesMap::default()).unwrap();
        let Err(error) = BoundaryFacetsIter::try_new(&facet_to_simplices_index, vec![handle])
        else {
            panic!("expected missing boundary handle to be rejected");
        };

        assert_matches!(
            error,
            FacetError::FacetKeyNotFoundInCache {
                cache_size: 0,
                vertex_uuids,
                ..
            } if vertex_uuids.len() == 3
        );
    }

    #[test]
    fn boundary_facets_iter_rejects_same_key_handle_not_indexed() {
        let mut tds = tetrahedron_tds();
        let simplex_key = tds.simplex_keys().next().unwrap();
        let duplicated_vertex = tds.simplex(simplex_key).unwrap().vertices()[0];
        {
            let simplex = tds.simplex_mut(simplex_key).unwrap();
            simplex.push_vertex_key(duplicated_vertex);
        }

        let indexed_handle = FacetHandle::from_validated(simplex_key, 0);
        let supplied_handle = FacetHandle::from_validated(simplex_key, 4);
        let facet_key = indexed_handle.view(&tds).unwrap().key();
        assert_eq!(supplied_handle.view(&tds).unwrap().key(), facet_key);

        let mut incidents = SmallBuffer::new();
        incidents.push(indexed_handle);
        let mut facet_to_simplices = FacetToSimplicesMap::default();
        facet_to_simplices.insert(facet_key, incidents);
        let facet_to_simplices_index =
            FacetToSimplicesIndex::try_from_map(&tds, &facet_to_simplices).unwrap();
        let Err(error) =
            BoundaryFacetsIter::try_new(&facet_to_simplices_index, vec![supplied_handle])
        else {
            panic!("expected non-indexed boundary handle to be rejected");
        };

        assert_matches!(
            error,
            FacetError::BoundaryFacetHandleNotIndexed {
                facet_key: observed_facet_key,
                supplied_handle: observed_supplied_handle,
                indexed_handle: observed_indexed_handle,
            } if observed_facet_key == facet_key
                && observed_supplied_handle == supplied_handle
                && observed_indexed_handle == indexed_handle
        );
    }

    #[test]
    fn boundary_facets_iter_errors_on_empty_multiplicity() {
        let tds = tetrahedron_tds();
        let first_facet = first_facet_view(&tds);
        let mut facet_to_simplices = FacetToSimplicesMap::default();
        facet_to_simplices.insert(first_facet.key(), SmallBuffer::new());
        assert_matches!(
            FacetToSimplicesIndex::try_from_map(&tds, &facet_to_simplices),
            Err(FacetError::InvalidFacetMultiplicity { found: 0, .. })
        );
    }

    #[test]
    fn boundary_facets_iter_errors_on_overshared_multiplicity() {
        let tds = tetrahedron_tds();
        let first_facet = first_facet_view(&tds);
        let handle =
            FacetHandle::from_validated(first_facet.simplex_key(), first_facet.facet_index());
        let mut incidents = SmallBuffer::new();
        incidents.push(handle);
        incidents.push(handle);
        incidents.push(handle);
        let mut facet_to_simplices = FacetToSimplicesMap::default();
        facet_to_simplices.insert(first_facet.key(), incidents);
        assert_matches!(
            FacetToSimplicesIndex::try_from_map(&tds, &facet_to_simplices),
            Err(FacetError::InvalidFacetMultiplicity { found: 3, .. })
        );
    }

    #[test]
    fn facet_index_rejects_duplicate_two_sided_incident_handle() {
        let tds = tetrahedron_tds();
        let first_facet = first_facet_view(&tds);
        let handle =
            FacetHandle::from_validated(first_facet.simplex_key(), first_facet.facet_index());
        let mut incidents = SmallBuffer::new();
        incidents.push(handle);
        incidents.push(handle);
        let mut facet_to_simplices = FacetToSimplicesMap::default();
        facet_to_simplices.insert(first_facet.key(), incidents);

        assert_matches!(
            FacetToSimplicesIndex::try_from_map(&tds, &facet_to_simplices),
            Err(FacetError::DuplicateFacetIncidentHandle {
                facet_key,
                handle: repeated
            }) if facet_key == first_facet.key() && repeated == handle
        );
    }

    #[test]
    fn facet_index_rejects_handle_with_mismatched_facet_key() {
        let tds = tetrahedron_tds();
        let mut facets = tds.facets();
        let first = facets.next().unwrap().unwrap();
        let second = facets.next().unwrap().unwrap();
        let wrong_handle = second.handle();
        let mut incidents = SmallBuffer::new();
        incidents.push(wrong_handle);
        let mut facet_to_simplices = FacetToSimplicesMap::default();
        facet_to_simplices.insert(first.key(), incidents);

        assert_matches!(
            FacetToSimplicesIndex::try_from_map(&tds, &facet_to_simplices),
            Err(FacetError::FacetHandleKeyMismatch {
                expected_facet_key,
                actual_facet_key,
                handle,
            }) if expected_facet_key == first.key()
                && actual_facet_key == second.key()
                && handle == wrong_handle
        );
    }

    // =============================================================================
    // ERROR HANDLING TESTS
    // =============================================================================

    #[test]
    fn facet_error_display() {
        let simplex_error = FacetError::SimplexDoesNotContainVertex;

        assert_eq!(
            simplex_error.to_string(),
            "The simplex does not contain the vertex!"
        );
    }

    #[test]
    fn facet_error_debug() {
        let simplex_error = FacetError::SimplexDoesNotContainVertex;

        let simplex_debug = format!("{simplex_error:?}");

        assert!(simplex_debug.contains("SimplexDoesNotContainVertex"));
    }

    #[test]
    fn test_facet_key_consistency() {
        // Create 3D triangulation with a tetrahedron
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        // Create facet views for different facets
        let facet1 = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap(); // excludes vertex 0
        let facet2 = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap(); // same facet
        let facet3 = FacetView::try_new(dt.tds(), simplex_key, 1).unwrap(); // excludes vertex 1 (different facet)

        // Both facet1 and facet2 reference the same facet, so same key
        assert_eq!(
            facet1.key(),
            facet2.key(),
            "Keys should be consistent for the same facet"
        );

        // facet3 is a different facet, so different key
        assert_ne!(
            facet1.key(),
            facet3.key(),
            "Keys should be different for facets with different vertices"
        );
    }

    #[test]
    fn facet_vertices_empty_simplex() {
        // Test edge case of minimal simplex (1D edge with 2 vertices)
        let vertices = vec![vertex!([0.0]).unwrap(), vertex!([1.0]).unwrap()];
        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        // Create facet with vertex 0 as opposite - should have only vertex 1 in facet
        let facet = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();
        assert_eq!(facet.vertices().count(), 1);

        // Test the opposite case - vertex 1 as opposite should have only vertex 0 in facet
        let other_facet = FacetView::try_new(dt.tds(), simplex_key, 1).unwrap();
        assert_eq!(other_facet.vertices().count(), 1);
    }

    #[test]
    fn facet_vertices_ordering() {
        // Test that vertices are filtered correctly
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        // Create facet view for facet 2 (excludes vertex 2)
        let facet = FacetView::try_new(dt.tds(), simplex_key, 2).unwrap();

        // Should have all vertices except vertex at index 2
        assert_eq!(facet.vertices().count(), 3);
        // Verify we have exactly 3 vertices (the D vertices of the D-1 dimensional facet)
    }

    #[test]
    fn facet_eq_different_vertices() {
        // Create a 3D triangulation
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        let facet1 = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();
        let facet2 = FacetView::try_new(dt.tds(), simplex_key, 1).unwrap();
        let facet3 = FacetView::try_new(dt.tds(), simplex_key, 2).unwrap();
        let facet4 = FacetView::try_new(dt.tds(), simplex_key, 3).unwrap();

        // All facets should be different because they have different facet indices
        // (i.e., different opposite vertices)
        assert_ne!(facet1, facet2);
        assert_ne!(facet1, facet3);
        assert_ne!(facet1, facet4);
        assert_ne!(facet2, facet3);
        assert_ne!(facet2, facet4);
        assert_ne!(facet3, facet4);
    }

    #[test]
    fn facet_key_hash() {
        // Create a 3D triangulation
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        // Create two facet views that reference the same facet
        let facet1 = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();
        let facet2 = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();

        // Create a different facet
        let facet3 = FacetView::try_new(dt.tds(), simplex_key, 1).unwrap();

        // Test that facet keys are consistent for the same facet
        assert_eq!(facet1.key(), facet2.key());

        // Test that different facets have different keys
        assert_ne!(facet1.key(), facet3.key());
    }

    // =============================================================================
    // FACET KEY GENERATION TESTS
    // =============================================================================

    #[test]
    fn test_facet_key_from_vertices() {
        // Create a temporary SlotMap to generate valid VertexKeys
        let mut temp_vertices: SlotMap<VertexKey, ()> = SlotMap::with_key();
        let vertices = vec![
            temp_vertices.insert(()),
            temp_vertices.insert(()),
            temp_vertices.insert(()),
        ];
        let key1 = facet_key_from_vertices(&vertices);

        let mut reversed_keys = vertices;
        reversed_keys.reverse();
        let key2 = facet_key_from_vertices(&reversed_keys);

        assert_eq!(
            key1, key2,
            "Facet keys should be identical for the same vertices in different order"
        );

        // Test with different vertex keys
        let different_keys = vec![
            temp_vertices.insert(()),
            temp_vertices.insert(()),
            temp_vertices.insert(()),
        ];
        let key3 = facet_key_from_vertices(&different_keys);

        assert_ne!(
            key1, key3,
            "Different vertices should produce different keys"
        );

        // Test empty case
        let empty_keys: Vec<VertexKey> = vec![];
        let key_empty = facet_key_from_vertices(&empty_keys);
        assert_eq!(key_empty, 0, "Empty vertex keys should produce key 0");
    }

    // =============================================================================
    // FACET VIEW TESTS
    // =============================================================================

    #[test]
    fn test_facet_view_creation() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];

        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        // Test valid facet creation
        let facet_view = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();
        assert_eq!(facet_view.simplex_key(), simplex_key);
        assert_eq!(facet_view.facet_index(), 0);

        // Test invalid facet index
        let result = FacetView::try_new(dt.tds(), simplex_key, 10);
        assert_matches!(result, Err(FacetError::InvalidFacetIndex { .. }));
    }

    #[test]
    fn test_facet_view_vertices_iteration() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];

        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        let facet_view = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();

        // Facet opposite to vertex 0 should have 3 vertices (D vertices in D-1 facet)
        let facet_vertices: Vec<_> = facet_view.vertices().collect();
        assert_eq!(facet_vertices.len(), 3);

        let simplex = dt.tds().simplex(simplex_key).expect("simplex exists");
        let opposite_vertex = dt
            .tds()
            .vertex(simplex.vertices()[0])
            .expect("opposite vertex exists");

        // Facet vertices should not include the opposite vertex
        assert!(
            !facet_vertices
                .iter()
                .any(|v| v.uuid() == opposite_vertex.uuid())
        );
    }

    #[test]
    fn test_facet_view_opposite_vertex() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];

        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        let facet_view = FacetView::try_new(dt.tds(), simplex_key, 1).unwrap();
        let opposite = facet_view.opposite_vertex();

        // The opposite vertex should be the vertex at index 1
        let simplex = dt.tds().simplex(simplex_key).expect("simplex exists");
        let simplex_vertex_keys = simplex.vertices();
        let expected_vertex = dt
            .tds()
            .vertex(simplex_vertex_keys[1])
            .expect("vertex exists");
        assert_eq!(opposite.uuid(), expected_vertex.uuid());
    }

    #[test]
    fn test_facet_view_key_computation() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];

        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        let facet_view = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();
        let key = facet_view.key();

        // Key should be non-zero for valid facet
        assert_ne!(key, 0);
    }

    #[test]
    fn test_try_simplex_facets() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];

        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        let facet_views = dt.tds().try_simplex_facets(simplex_key).unwrap();
        let facet_count = facet_views.len();

        // 3D simplex (tetrahedron) should have 4 facets
        assert_eq!(facet_count, 4);

        // Each facet should have a different index
        for (i, facet_view) in facet_views.enumerate() {
            let facet_view = facet_view.unwrap();
            assert_eq!(
                facet_view.facet_index(),
                usize_to_u8(i, facet_count).unwrap()
            );
            assert_eq!(facet_view.simplex_key(), simplex_key);
        }
    }

    #[test]
    fn test_facet_view_equality() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];

        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        let facet_view1 = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();
        let facet_view2 = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();
        let facet_view3 = FacetView::try_new(dt.tds(), simplex_key, 1).unwrap();

        // Same facet should be equal
        assert_eq!(facet_view1, facet_view2);

        // Different facets should not be equal
        assert_ne!(facet_view1, facet_view3);
    }

    #[test]
    fn test_facet_view_debug() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];

        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        let facet_view = FacetView::try_new(dt.tds(), simplex_key, 1).unwrap();
        let debug_str = format!("{facet_view:?}");

        assert!(debug_str.contains("FacetView"));
        assert!(debug_str.contains("simplex_key"));
        assert!(debug_str.contains("facet_index"));
        assert!(debug_str.contains("dimension"));
    }

    #[test]
    fn test_facet_view_memory_efficiency() {
        let lightweight_size = mem::size_of::<FacetView<(), (), 3>>();
        let payload_independent_size = mem::size_of::<FacetView<[u8; 1024], [u8; 1024], 3>>();

        assert_eq!(
            lightweight_size, payload_independent_size,
            "FacetView must borrow vertex/simplex payloads rather than owning them"
        );
        assert!(
            lightweight_size <= 256,
            "FacetView should stay a compact borrowed view, got {lightweight_size} bytes"
        );
    }
}
