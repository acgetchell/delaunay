//! D-dimensional Facets Representation
//!
//! This module provides the `FacetView` struct which represents a facet of a d-dimensional simplex
//! (d-1 sub-simplex) within a triangulation. Each facet is defined in terms of a simplex and the
//! vertex opposite to it, similar to [CGAL](https://doc.cgal.org/latest/TDS_3/index.html#title3).
//!
//! # Key Features
//!
//! - **Lightweight**: `FacetView` is ~18x smaller than the deprecated `Facet` struct
//! - **Dimensional Simplicity**: Represents co-dimension 1 sub-simplexes of d-dimensional simplexes
//! - **Simplex Association**: Each facet resides within a specific simplex and is described by its opposite vertex
//! - **Support for Delaunay Triangulations**: Facilitates operations fundamental to the
//!   [Bowyer-Watson algorithm](https://en.wikipedia.org/wiki/Bowyer–Watson_algorithm)
//! - **On-demand Creation**: Facets are generated dynamically as needed rather than stored persistently in the TDS
//! - **Memory Efficient**: Stores only references and keys, accessing data on-demand from the TDS
//!
//! # Fundamental Invariant
//!
//! **A critical invariant of Delaunay triangulations is that each facet is shared by exactly two simplices,
//! except for boundary facets which belong to only one simplex.**
//!
//! This property ensures the triangulation forms a valid simplicial complex:
//! - **Interior facets**: Shared by exactly 2 simplices (defines proper adjacency)
//! - **Boundary facets**: Belong to exactly 1 simplex (lie on the convex hull)
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
//! # }
//! # fn main() -> Result<(), ExampleError> {
//! // Create vertices for a tetrahedron
//! let vertices = vec![
//!     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
//!     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).expect("finite vertex coordinates"),
//!     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).expect("finite vertex coordinates"),
//!     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).expect("finite vertex coordinates"),
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
//! assert_eq!(facet.vertices()?.count(), 3);  // Facet (triangle) in 3D has 3 vertices
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]

use super::collections::{FacetToSimplicesMap, MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer};
use super::util::{stable_hash_u64_slice, usize_to_u8};
use super::{
    simplex::Simplex,
    tds::{SimplexKey, Tds, TdsError, VertexKey},
    vertex::Vertex,
};
use crate::geometry::traits::coordinate::CoordinateConversionError;
use slotmap::Key;
use std::fmt::{self, Debug};
use std::sync::Arc;
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
    /// Facet has invalid multiplicity (should be 1 for boundary or 2 for internal).
    #[error(
        "Facet with key {facet_key:016x} has invalid multiplicity {found}, expected 1 (boundary) or 2 (internal)"
    )]
    InvalidFacetMultiplicity {
        /// The facet key with invalid multiplicity.
        facet_key: u64,
        /// The actual multiplicity found.
        found: usize,
    },
    /// Failed to retrieve boundary facets from triangulation.
    #[error("Failed to retrieve boundary facets: {source}")]
    BoundaryFacetRetrievalFailed {
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
///
/// # Components
///
/// - `simplex_key`: The key of the simplex containing the facet
/// - `facet_index`: The facet index (0 to D, representing the vertex opposite to the facet)
///
/// # Usage
///
/// `FacetHandle` is commonly used in:
/// - Boundary facet analysis (convex hull extraction)
/// - Facet visibility testing
/// - Cavity computation in Bowyer-Watson algorithm
/// - Any operation requiring lightweight facet references
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
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).expect("finite vertex coordinates"),
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
/// let facet = FacetView::try_new(dt.tds(), handle.simplex_key(), handle.facet_index())?;
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
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     Vertex::<(), _>::try_new([0.0, 0.0]).expect("finite vertex coordinates"),
    ///     Vertex::<(), _>::try_new([1.0, 0.0]).expect("finite vertex coordinates"),
    ///     Vertex::<(), _>::try_new([0.0, 1.0]).expect("finite vertex coordinates"),
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
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     Vertex::<(), _>::try_new([0.0, 0.0]).expect("finite vertex coordinates"),
    ///     Vertex::<(), _>::try_new([1.0, 0.0]).expect("finite vertex coordinates"),
    ///     Vertex::<(), _>::try_new([0.0, 1.0]).expect("finite vertex coordinates"),
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
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     Vertex::<(), _>::try_new([0.0, 0.0]).expect("finite vertex coordinates"),
    ///     Vertex::<(), _>::try_new([1.0, 0.0]).expect("finite vertex coordinates"),
    ///     Vertex::<(), _>::try_new([0.0, 1.0]).expect("finite vertex coordinates"),
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
}

// =============================================================================
// LIGHTWEIGHT FACET VIEW
// =============================================================================

/// Lightweight facet representation as a view into a triangulation data structure.
///
/// Lightweight facet implementation that replaces the heavyweight `Facet` struct
/// with an ~18x memory reduction.
///
/// `FacetView` represents a facet (d-1 dimensional face) of a d-dimensional simplex
/// without storing any data directly. Instead, it maintains references to the TDS
/// and uses keys to access data on-demand.
///
/// # Memory Efficiency
///
/// Compared to the original `Facet<U, V, D>`:
/// - **Original**: Stores complete Simplex + Vertex objects (~hundreds of bytes)
/// - **`FacetView`**: Stores TDS reference + `SimplexKey` + `facet_index` (~17 bytes)
/// - **Memory reduction: ~18x smaller**
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
///     // Access vertices through the view (lazy evaluation)
///     for vertex in facet_view.vertices()? {
///         println!("Vertex: {:?}", vertex.point());
///     }
///
///     // Get the opposite vertex
///     let opposite = facet_view.opposite_vertex()?;
///
///     // Compute facet key
///     let key = facet_view.key()?;
///     Ok(())
/// }
/// ```
#[must_use]
pub struct FacetView<'tds, U, V, const D: usize> {
    /// Reference to the triangulation data structure.
    tds: &'tds Tds<U, V, D>,
    /// Key of the simplex containing this facet.
    simplex_key: SimplexKey,
    /// Index of this facet within the simplex (0 <= `facet_index` < D+1).
    ///
    /// The `facet_index` indicates which vertex of the simplex is the "opposite vertex"
    /// (the vertex not included in the facet). For a D-dimensional simplex with D+1
    /// vertices, facet i excludes vertex i and includes all others.
    facet_index: u8,
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
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).expect("finite vertex coordinates"),
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
        // Validate simplex exists
        let simplex = tds
            .simplex(simplex_key)
            .ok_or(FacetError::SimplexNotFoundInTriangulation)?;

        // Validate facet index
        let vertex_count = simplex.number_of_vertices();
        if usize::from(facet_index) >= vertex_count {
            return Err(FacetError::InvalidFacetIndex {
                index: facet_index,
                facet_count: vertex_count,
            });
        }

        Ok(Self {
            tds,
            simplex_key,
            facet_index,
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
    /// # Returns
    ///
    /// A `Result` containing an iterator yielding references to vertices in the facet,
    /// or a `FacetError` if the simplex is no longer present in the TDS.
    ///
    /// # Errors
    ///
    /// Returns `FacetError::SimplexNotFoundInTriangulation` if the simplex key is no longer
    /// present in the TDS. This could happen if the TDS is modified after the `FacetView`
    /// is created, though this should not occur under normal usage patterns.
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
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).expect("finite vertex coordinates"),
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// if let Some((simplex_key, _)) = dt.simplices().next() {
    ///     let facet = FacetView::try_new(dt.tds(), simplex_key, 0)?;
    ///     let vertex_iter = facet.vertices()?;
    ///     assert_eq!(vertex_iter.count(), 3); // 3D facet has 3 vertices
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn vertices(&self) -> Result<impl Iterator<Item = &'tds Vertex<U, D>>, FacetError> {
        let simplex = self
            .tds
            .simplex(self.simplex_key)
            .ok_or(FacetError::SimplexNotFoundInTriangulation)?;
        let facet_index = usize::from(self.facet_index);

        // Collect first so missing vertex keys become an error, not silent drops.
        // Use SmallBuffer for stack allocation (D vertices fit on stack for D ≤ 7)
        let mut refs: SmallBuffer<&'tds Vertex<U, D>, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::with_capacity(simplex.number_of_vertices().saturating_sub(1));
        for (i, &vkey) in simplex.vertices().iter().enumerate() {
            if i == facet_index {
                continue;
            }
            refs.push(
                self.tds
                    .vertex(vkey)
                    .ok_or(FacetError::VertexKeyNotFoundInTriangulation { key: vkey })?,
            );
        }
        Ok(refs.into_iter())
    }

    /// Returns the opposite vertex (the vertex not included in the facet).
    ///
    /// # Returns
    ///
    /// A `Result` containing a reference to the opposite vertex.
    ///
    /// # Errors
    ///
    /// Returns [`FacetError::SimplexNotFoundInTriangulation`] if the simplex is no longer in the TDS,
    /// [`FacetError::InvalidFacetIndex`] if the facet index is outside the simplex's vertex list,
    /// or [`FacetError::VertexKeyNotFoundInTriangulation`] if the opposite vertex key no longer
    /// resolves to a stored vertex.
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
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).expect("finite vertex coordinates"),
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// let facet = FacetView::try_new(dt.tds(), simplex_key, 1)?;
    /// let opposite = facet.opposite_vertex()?;
    /// assert_eq!(opposite.point().coords().len(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn opposite_vertex(&self) -> Result<&'tds Vertex<U, D>, FacetError> {
        let simplex = self
            .tds
            .simplex(self.simplex_key)
            .ok_or(FacetError::SimplexNotFoundInTriangulation)?;

        let vertices = simplex.vertices();
        let facet_index = usize::from(self.facet_index);

        let vkey = vertices
            .get(facet_index)
            .ok_or(FacetError::InvalidFacetIndex {
                index: self.facet_index,
                facet_count: vertices.len(),
            })?;

        // Use get() to safely handle potentially invalid vertex keys
        self.tds
            .vertex(*vkey)
            .ok_or(FacetError::VertexKeyNotFoundInTriangulation { key: *vkey })
    }

    /// Returns the simplex containing this facet.
    ///
    /// # Returns
    ///
    /// A `Result` containing a reference to the containing simplex.
    ///
    /// # Errors
    ///
    /// Returns `FacetError::SimplexNotFoundInTriangulation` if the simplex is no longer in the TDS.
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
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).expect("finite vertex coordinates"),
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// let facet = FacetView::try_new(dt.tds(), simplex_key, 2)?;
    /// let simplex = facet.simplex()?;
    /// assert_eq!(simplex.number_of_vertices(), 4);
    /// # Ok(())
    /// # }
    /// ```
    pub fn simplex(&self) -> Result<&'tds Simplex<V, D>, FacetError> {
        self.tds
            .simplex(self.simplex_key)
            .ok_or(FacetError::SimplexNotFoundInTriangulation)
    }

    /// Computes a canonical key for this facet.
    ///
    /// The key is computed from the vertex keys of the facet vertices,
    /// providing a stable hash that's identical for any two facets
    /// containing the same vertices.
    ///
    /// # Returns
    ///
    /// A `Result` containing the facet key as a `u64`.
    ///
    /// # Errors
    ///
    /// Returns `FacetError` if vertex keys cannot be retrieved.
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
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).expect("finite vertex coordinates"),
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// let facet = FacetView::try_new(dt.tds(), simplex_key, 0)?;
    /// let facet_key = facet.key()?;
    /// let map = dt.tds().build_facet_to_simplices_map()?;
    /// assert!(map.contains_key(&facet_key));
    /// # Ok(())
    /// # }
    /// ```
    pub fn key(&self) -> Result<u64, FacetError> {
        self.tds
            .facet_key_for_simplex_facet(self.simplex_key, usize::from(self.facet_index))
            .map_err(|e| FacetError::SimplexOperationFailed {
                source: Arc::new(e),
            })
    }
}

// Trait implementations for FacetView
impl<U, V, const D: usize> Debug for FacetView<'_, U, V, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FacetView")
            .field("simplex_key", &self.simplex_key)
            .field("facet_index", &self.facet_index)
            .field("dimension", &D)
            .finish()
    }
}

#[expect(
    clippy::non_canonical_clone_impl,
    reason = "facet clone intentionally preserves cached view fields"
)]
impl<U, V, const D: usize> Clone for FacetView<'_, U, V, D> {
    fn clone(&self) -> Self {
        Self {
            tds: self.tds,
            simplex_key: self.simplex_key,
            facet_index: self.facet_index,
        }
    }
}

impl<U, V, const D: usize> Copy for FacetView<'_, U, V, D> {}

impl<U, V, const D: usize> PartialEq for FacetView<'_, U, V, D> {
    fn eq(&self, other: &Self) -> bool {
        // Two facet views are equal if they reference the same facet
        std::ptr::eq(self.tds, other.tds)
            && self.simplex_key == other.simplex_key
            && self.facet_index == other.facet_index
    }
}

impl<U, V, const D: usize> Eq for FacetView<'_, U, V, D> {}

/// Utility function to create multiple `FacetView`s for all facets of a simplex.
///
/// # Arguments
///
/// * `tds` - Reference to the triangulation data structure
/// * `simplex_key` - Key of the simplex to create facet views for
///
/// # Returns
///
/// A `Result` containing a `Vec` of `FacetView`s for all facets of the simplex.
///
/// # Errors
///
/// Returns `FacetError` if the simplex is not found or has invalid structure.
///
/// # Note
///
/// Removed unnecessary numeric bounds (`AddAssign`, `SubAssign`, `Sum`, `NumCast`, `Div`)
/// since this function doesn't perform any arithmetic operations.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::*;
/// use delaunay::prelude::tds::all_facets_for_simplex;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Facet(#[from] delaunay::prelude::tds::FacetError),
/// #     #[error(transparent)]
/// #     Tds(#[from] delaunay::prelude::tds::TdsError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).expect("finite vertex coordinates"),
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
/// let Some((simplex_key, _)) = dt.simplices().next() else {
///     return Ok(());
/// };
///
/// let facets = all_facets_for_simplex(dt.tds(), simplex_key)?;
/// assert_eq!(facets.len(), 4);
/// # Ok(())
/// # }
/// ```
pub fn all_facets_for_simplex<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_key: SimplexKey,
) -> Result<Vec<FacetView<'_, U, V, D>>, FacetError> {
    let simplex = tds
        .simplex(simplex_key)
        .ok_or(FacetError::SimplexNotFoundInTriangulation)?;

    let vertex_count = simplex.number_of_vertices();
    let mut facet_views = Vec::with_capacity(vertex_count);

    for facet_index in 0..vertex_count {
        let idx = facet_index; // usize
        let facet_view = FacetView::try_new(tds, simplex_key, usize_to_u8(idx, vertex_count)?)?;
        facet_views.push(facet_view);
    }

    Ok(facet_views)
}

/// Iterator over all facets in a triangulation data structure.
///
/// This iterator provides efficient access to all facets without allocating
/// a vector. It's particularly useful for performance-critical operations
/// like boundary detection and cavity analysis in triangulation insertion.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::tds::AllFacetsIter;
/// use delaunay::prelude::*;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Facet(#[from] delaunay::prelude::tds::FacetError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).expect("finite vertex coordinates"),
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
///
/// let count = AllFacetsIter::try_new(dt.tds())?.count();
/// assert_eq!(count, 4);
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct AllFacetsIter<'tds, U, V, const D: usize> {
    tds: &'tds Tds<U, V, D>,
    simplex_keys: std::vec::IntoIter<SimplexKey>,
    current_simplex_key: Option<SimplexKey>,
    current_facet_index: usize,
    current_simplex_facet_count: usize,
}

impl<'tds, U, V, const D: usize> AllFacetsIter<'tds, U, V, D> {
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
    /// use delaunay::prelude::tds::AllFacetsIter;
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Facet(#[from] delaunay::prelude::tds::FacetError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).expect("finite vertex coordinates"),
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// let mut iter = AllFacetsIter::try_new(dt.tds())?;
    /// assert!(iter.next().is_some());
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_new(tds: &'tds Tds<U, V, D>) -> Result<Self, FacetError> {
        // Dimension check: facets per simplex = D+1, so D must be <= 255
        if D > usize::from(u8::MAX) {
            return Err(FacetError::FacetIndexCapacityExceeded {
                dimension: D,
                max_dimension: usize::from(u8::MAX),
            });
        }

        // We collect here because we need an owned iterator to store in the struct
        // SimplexKey is just u64, so this is efficient
        #[expect(
            clippy::needless_collect,
            reason = "iterator owns a stable snapshot of simplex keys before yielding facets"
        )]
        let simplex_keys: Vec<SimplexKey> = tds.simplex_keys().collect();
        Ok(Self {
            tds,
            simplex_keys: simplex_keys.into_iter(),
            current_simplex_key: None,
            current_facet_index: 0,
            current_simplex_facet_count: 0,
        })
    }
}

impl<'tds, U, V, const D: usize> Iterator for AllFacetsIter<'tds, U, V, D> {
    type Item = FacetView<'tds, U, V, D>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // If we have a current simplex and more facets in it
            if let Some(simplex_key) = self.current_simplex_key
                && self.current_facet_index < self.current_simplex_facet_count
            {
                let facet_index = self.current_facet_index;
                self.current_facet_index += 1;

                // Create FacetView - we know this is valid since we're iterating within bounds
                let Ok(facet_u8) = usize_to_u8(facet_index, self.current_simplex_facet_count)
                else {
                    // Fail fast instead of silently skipping in release.
                    // If D can exceed 255, widen the index type.
                    return None;
                };
                if let Ok(facet_view) = FacetView::try_new(self.tds, simplex_key, facet_u8) {
                    return Some(facet_view);
                }
            }

            // Move to next simplex
            if let Some(next_simplex_key) = self.simplex_keys.next() {
                if let Some(simplex) = self.tds.simplex(next_simplex_key) {
                    self.current_simplex_key = Some(next_simplex_key);
                    self.current_facet_index = 0;
                    self.current_simplex_facet_count = simplex.number_of_vertices();
                    // Continue loop to process first facet of new simplex
                } else {
                    // Simplex not found, skip to next (continue is implicit at end of loop)
                }
            } else {
                // No more simplices
                return None;
            }
        }
    }
}

/// Iterator over boundary facets in a triangulation.
///
/// This iterator efficiently identifies and yields only the boundary facets
/// (facets that belong to only one simplex) without pre-computing all facets.
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
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).expect("finite vertex coordinates"),
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
/// let count = dt.tds().boundary_facets()?.count();
/// assert_eq!(count, 4);
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct BoundaryFacetsIter<'tds, U, V, const D: usize> {
    all_facets: AllFacetsIter<'tds, U, V, D>,
    facet_to_simplices_map: crate::core::collections::FacetToSimplicesMap,
}

impl<'tds, U, V, const D: usize> BoundaryFacetsIter<'tds, U, V, D> {
    /// Creates a new iterator over boundary facets.
    ///
    /// # Errors
    ///
    /// Returns [`FacetError::FacetIndexCapacityExceeded`] if this dimension
    /// cannot be represented by the current `u8` facet-index storage.
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
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).expect("finite vertex coordinates"),
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let mut iter = dt.tds().boundary_facets()?;
    /// assert!(iter.next().is_some());
    /// # Ok(())
    /// # }
    /// ```
    pub(crate) fn try_new(
        tds: &'tds Tds<U, V, D>,
        facet_to_simplices_map: FacetToSimplicesMap,
    ) -> Result<Self, FacetError> {
        Ok(Self {
            all_facets: AllFacetsIter::try_new(tds)?,
            facet_to_simplices_map,
        })
    }
}

impl<'tds, U, V, const D: usize> Iterator for BoundaryFacetsIter<'tds, U, V, D> {
    type Item = FacetView<'tds, U, V, D>;

    fn next(&mut self) -> Option<Self::Item> {
        // Find the next boundary facet
        self.all_facets.find(|facet_view| {
            // Check if this facet is a boundary facet using the precomputed map
            if let Ok(facet_key) = facet_view.key()
                && let Some(simplex_list) = self.facet_to_simplices_map.get(&facet_key)
            {
                // Boundary facets appear in exactly one simplex
                return simplex_list.len() == 1;
            }
            false
        })
    }
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
    use crate::core::validation::TopologyGuarantee;
    use crate::core::vertex::Vertex;
    use crate::geometry::kernel::AdaptiveKernel;
    use crate::triangulation::DelaunayTriangulation;
    use slotmap::SlotMap;
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
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
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
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        // Create facet view for facet 0 (excludes vertex 0)
        let facet = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();
        assert_eq!(facet.simplex_key(), simplex_key);
        assert_eq!(facet.facet_index(), 0);
    }

    #[test]
    fn test_facet_new_success_coverage() {
        // Test 2D case: Create a triangle (2D simplex with 3 vertices)
        let vertices_2d = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.5, 1.0]).unwrap(),
        ];
        let dt_2d = DelaunayTriangulation::new(&vertices_2d).unwrap();
        let simplex_key_2d = dt_2d.simplices().next().unwrap().0;
        let result_2d = FacetView::try_new(dt_2d.tds(), simplex_key_2d, 0);

        // Assert that the result is Ok
        assert!(result_2d.is_ok());
        let facet_2d = result_2d.unwrap();
        assert_eq!(facet_2d.vertices().unwrap().count(), 2); // 2D facet should have 2 vertices

        // Test 1D case: Create an edge (1D simplex with 2 vertices)
        let vertices_1d = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0]).unwrap(),
        ];
        let dt_1d = DelaunayTriangulation::new(&vertices_1d).unwrap();
        let simplex_key_1d = dt_1d.simplices().next().unwrap().0;
        let result_1d = FacetView::try_new(dt_1d.tds(), simplex_key_1d, 0);

        // Assert that the result is Ok
        assert!(result_1d.is_ok());
        let facet_1d = result_1d.unwrap();
        assert_eq!(facet_1d.vertices().unwrap().count(), 1); // 1D facet should have 1 vertex
    }

    #[test]
    fn facet_new_with_incorrect_vertex() {
        // Create a 3D triangulation
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        // Test invalid facet index (3D simplex has vertices 0-3, facet index 4 is invalid)
        assert!(FacetView::try_new(dt.tds(), simplex_key, 4).is_err());
    }

    #[test]
    fn facet_vertices() {
        // Create a 3D triangulation with a tetrahedron
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        // Create facet view for facet 0 (excludes vertex 0)
        let facet = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();
        assert_eq!(facet.vertices().unwrap().count(), 3);
    }

    // =============================================================================
    // EQUALITY AND ORDERING TESTS
    // =============================================================================

    #[test]
    fn facet_partial_eq() {
        // Create a 3D triangulation
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
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
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        let facet = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();
        let cloned_facet = facet;

        // Verify clones are equal
        assert_eq!(facet, cloned_facet);
        assert_eq!(facet.simplex_key(), cloned_facet.simplex_key());
        assert_eq!(facet.facet_index(), cloned_facet.facet_index());

        // Verify simplex and opposite vertex are accessible through both views
        let simplex1 = facet.simplex().unwrap();
        let simplex2 = cloned_facet.simplex().unwrap();
        assert_eq!(simplex1.uuid(), simplex2.uuid());

        let vertex1 = facet.opposite_vertex().unwrap();
        let vertex2 = cloned_facet.opposite_vertex().unwrap();
        assert_eq!(vertex1.uuid(), vertex2.uuid());
    }

    #[test]
    fn facet_debug() {
        // Create a 3D triangulation with a non-degenerate tetrahedron
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
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
            crate::core::vertex::Vertex::<_, _>::try_new_with_data([0.0, 0.0, 0.0], 1).unwrap(),
            crate::core::vertex::Vertex::<_, _>::try_new_with_data([1.0, 0.0, 0.0], 2).unwrap(),
            crate::core::vertex::Vertex::<_, _>::try_new_with_data([0.0, 1.0, 0.0], 3).unwrap(),
            crate::core::vertex::Vertex::<_, _>::try_new_with_data([0.0, 0.0, 1.0], 4).unwrap(),
        ];
        let options = ConstructionOptions::default()
            .with_insertion_order(InsertionOrderStrategy::Input)
            .with_initial_simplex_strategy(InitialSimplexStrategy::First);
        let dt: DelaunayTriangulation<AdaptiveKernel<f64>, i32, (), 3> =
            DelaunayTriangulation::with_topology_guarantee_and_options(
                &AdaptiveKernel::new(),
                &vertices,
                TopologyGuarantee::DEFAULT,
                options,
            )
            .unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        // Create facet view for facet 0 (excludes vertex 0)
        let facet = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();

        let facet_vertices: Vec<_> = facet.vertices().unwrap().collect();
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
    ///     facet_2d => 2 => "triangle" => 2 => vec![delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0]).expect("finite vertex coordinates"), ...],
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
                    let dt = DelaunayTriangulation::new(&vertices).unwrap();
                    let simplex_key = dt.simplices().next().unwrap().0;

                    // Create facet view for facet 0 (excludes vertex 0)
                    let facet = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();

                    // Facet of D-dimensional simplex is (D-1)-dimensional with D vertices
                    assert_eq!(facet.vertices().unwrap().count(), $expected_facet_vertices,
                        "Facet of {}D {} should have {} vertices", $dim, $desc, $expected_facet_vertices);
                }

                pastey::paste! {
                    #[test]
                    fn [<$test_name _key_consistency>]() {
                        // Test FacetKey computation consistency
                        let vertices = $vertices;
                        let dt = DelaunayTriangulation::new(&vertices).unwrap();
                        let simplex_key = dt.simplices().next().unwrap().0;

                        // Create same facet twice
                        let facet1 = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();
                        let facet2 = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();

                        assert_eq!(facet1.key().unwrap(), facet2.key().unwrap(),
                            "Same facet should produce same key");

                        // Create different facet
                        let facet3 = FacetView::try_new(dt.tds(), simplex_key, 1).unwrap();
                        assert_ne!(facet1.key().unwrap(), facet3.key().unwrap(),
                            "Different facets should produce different keys");
                    }

                    #[test]
                    fn [<$test_name _equality>]() {
                        // Test facet equality comparison
                        let vertices = $vertices;
                        let dt = DelaunayTriangulation::new(&vertices).unwrap();
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
                        let dt = DelaunayTriangulation::new(&vertices).unwrap();
                        let simplex_key = dt.simplices().next().unwrap().0;

                        // D+1 dimensional simplex should have D+1 facets (one opposite each vertex)
                        let expected_facets = $dim + 1;
                        let mut facet_keys = HashSet::new();

                        for i in 0..expected_facets {
                            let facet = FacetView::try_new(dt.tds(), simplex_key, u8::try_from(i).unwrap()).unwrap();
                            facet_keys.insert(facet.key().unwrap());
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
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.5, 1.0]).unwrap(),
        ],
        facet_3d_tetrahedron => 3 => "tetrahedron" => 3 => vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ],
        facet_4d_simplex => 4 => "4-simplex" => 4 => vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 1.0]).unwrap(),
        ],
        facet_5d_simplex => 5 => "5-simplex" => 5 => vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0, 1.0]).unwrap(),
        ],
    }

    // Keep 1D test separate as it's less common
    #[test]
    fn facet_1d_edge() {
        // Create 1D triangulation (edge with 2 vertices)
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0]).unwrap(),
        ];
        let options = ConstructionOptions::default()
            .with_insertion_order(InsertionOrderStrategy::Input)
            .with_initial_simplex_strategy(InitialSimplexStrategy::First);
        let dt = DelaunayTriangulation::new_with_options(&vertices, options).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        // Create facet view for facet 0 (excludes vertex 0)
        let facet = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();

        // Facet of 1D edge is a point (0D) with 1 vertex
        assert_eq!(facet.vertices().unwrap().count(), 1);
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
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        // Create facet views for different facets
        let facet1 = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap(); // excludes vertex 0
        let facet2 = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap(); // same facet
        let facet3 = FacetView::try_new(dt.tds(), simplex_key, 1).unwrap(); // excludes vertex 1 (different facet)

        // Both facet1 and facet2 reference the same facet, so same key
        assert_eq!(
            facet1.key().unwrap(),
            facet2.key().unwrap(),
            "Keys should be consistent for the same facet"
        );

        // facet3 is a different facet, so different key
        assert_ne!(
            facet1.key().unwrap(),
            facet3.key().unwrap(),
            "Keys should be different for facets with different vertices"
        );
    }

    #[test]
    fn facet_vertices_empty_simplex() {
        // Test edge case of minimal simplex (1D edge with 2 vertices)
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        // Create facet with vertex 0 as opposite - should have only vertex 1 in facet
        let facet = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();
        assert_eq!(facet.vertices().unwrap().count(), 1);

        // Test the opposite case - vertex 1 as opposite should have only vertex 0 in facet
        let other_facet = FacetView::try_new(dt.tds(), simplex_key, 1).unwrap();
        assert_eq!(other_facet.vertices().unwrap().count(), 1);
    }

    #[test]
    fn facet_vertices_ordering() {
        // Test that vertices are filtered correctly
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        // Create facet view for facet 2 (excludes vertex 2)
        let facet = FacetView::try_new(dt.tds(), simplex_key, 2).unwrap();

        // Should have all vertices except vertex at index 2
        assert_eq!(facet.vertices().unwrap().count(), 3);
        // Verify we have exactly 3 vertices (the D vertices of the D-1 dimensional facet)
    }

    #[test]
    fn facet_eq_different_vertices() {
        // Create a 3D triangulation
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
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
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        // Create two facet views that reference the same facet
        let facet1 = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();
        let facet2 = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();

        // Create a different facet
        let facet3 = FacetView::try_new(dt.tds(), simplex_key, 1).unwrap();

        // Test that facet keys are consistent for the same facet
        assert_eq!(facet1.key().unwrap(), facet2.key().unwrap());

        // Test that different facets have different keys
        assert_ne!(facet1.key().unwrap(), facet3.key().unwrap());
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
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];

        let dt = DelaunayTriangulation::new(&vertices).unwrap();
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
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];

        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        let facet_view = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();

        // Facet opposite to vertex 0 should have 3 vertices (D vertices in D-1 facet)
        let facet_vertices: Vec<_> = facet_view.vertices().unwrap().collect();
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
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];

        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        let facet_view = FacetView::try_new(dt.tds(), simplex_key, 1).unwrap();
        let opposite = facet_view.opposite_vertex().unwrap();

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
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];

        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        let facet_view = FacetView::try_new(dt.tds(), simplex_key, 0).unwrap();
        let key = facet_view.key().unwrap();

        // Key should be non-zero for valid facet
        assert_ne!(key, 0);
    }

    #[test]
    fn test_all_facets_for_simplex() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];

        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        let facet_views = all_facets_for_simplex(dt.tds(), simplex_key).unwrap();

        // 3D simplex (tetrahedron) should have 4 facets
        assert_eq!(facet_views.len(), 4);

        // Each facet should have a different index
        for (i, facet_view) in facet_views.iter().enumerate() {
            assert_eq!(
                facet_view.facet_index(),
                usize_to_u8(i, facet_views.len()).unwrap()
            );
            assert_eq!(facet_view.simplex_key(), simplex_key);
        }
    }

    #[test]
    fn test_facet_view_equality() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];

        let dt = DelaunayTriangulation::new(&vertices).unwrap();
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
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];

        let dt = DelaunayTriangulation::new(&vertices).unwrap();
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

        // FacetView should be around 17 bytes (8 byte ref + 8 byte SimplexKey + 1 byte facet_index)
        // Allow for some padding/alignment
        assert!(lightweight_size <= 24);
    }
}
