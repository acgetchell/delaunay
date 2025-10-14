//! D-dimensional Facets Representation
//!
//! This module provides the `FacetView` struct which represents a facet of a d-dimensional simplex
//! (d-1 sub-simplex) within a triangulation. Each facet is defined in terms of a cell and the
//! vertex opposite to it, similar to [CGAL](https://doc.cgal.org/latest/TDS_3/index.html#title3).
//!
//! # Key Features
//!
//! - **Lightweight**: `FacetView` is ~18x smaller than the deprecated `Facet` struct
//! - **Dimensional Simplicity**: Represents co-dimension 1 sub-simplexes of d-dimensional simplexes
//! - **Cell Association**: Each facet resides within a specific cell and is described by its opposite vertex
//! - **Support for Delaunay Triangulations**: Facilitates operations fundamental to the
//!   [Bowyer-Watson algorithm](https://en.wikipedia.org/wiki/Bowyer–Watson_algorithm)
//! - **On-demand Creation**: Facets are generated dynamically as needed rather than stored persistently in the TDS
//! - **Memory Efficient**: Stores only references and keys, accessing data on-demand from the TDS
//!
//! # Fundamental Invariant
//!
//! **A critical invariant of Delaunay triangulations is that each facet is shared by exactly two cells,
//! except for boundary facets which belong to only one cell.**
//!
//! This property ensures the triangulation forms a valid simplicial complex:
//! - **Interior facets**: Shared by exactly 2 cells (defines proper adjacency)
//! - **Boundary facets**: Belong to exactly 1 cell (lie on the convex hull)
//! - **Invalid configurations**: Facets shared by 0, 3, or more cells indicate topological errors
//!
//! This invariant is fundamental to many algorithms and is actively validated during triangulation
//! construction and validation phases.
//!
//! For a comprehensive discussion of all topological invariants in Delaunay triangulations,
//! see the [Topological Invariants](crate::core::triangulation_data_structure#topological-invariants)
//! section in the triangulation data structure documentation.
//!
//! # Examples
//!
//! ```rust
//! use delaunay::core::facet::FacetView;
//! use delaunay::core::triangulation_data_structure::Tds;
//! use delaunay::vertex;
//!
//! // Create vertices for a tetrahedron
//! let vertices = vec![
//!     vertex!([0.0, 0.0, 0.0]),
//!     vertex!([1.0, 0.0, 0.0]),
//!     vertex!([0.0, 1.0, 0.0]),
//!     vertex!([0.0, 0.0, 1.0]),
//! ];
//!
//! // Create a 3D triangulation
//! let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
//! let cell_key = tds.cell_keys().next().unwrap();
//!
//! // Create a facet view (facet 0 excludes vertex 0)
//! let facet = FacetView::new(&tds, cell_key, 0).unwrap();
//! assert_eq!(facet.vertices().unwrap().count(), 3);  // Facet (triangle) in 3D has 3 vertices
//! ```

// =============================================================================
// IMPORTS
// =============================================================================

use super::traits::data_type::DataType;
use super::util::{stable_hash_u64_slice, usize_to_u8};
use super::{
    cell::Cell,
    triangulation_data_structure::{CellKey, Tds, VertexKey},
    vertex::Vertex,
};
use crate::geometry::traits::coordinate::CoordinateScalar;
use serde::{Serialize, de::DeserializeOwned};
use slotmap::Key;
use std::fmt::{self, Debug};
use thiserror::Error;

// =============================================================================
// ERROR TYPES
// =============================================================================

/// Error type for facet operations.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum FacetError {
    /// The cell does not contain the vertex.
    #[error("The cell does not contain the vertex!")]
    CellDoesNotContainVertex,
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
    /// Expected exactly one adjacent cell for boundary facet.
    #[error("Expected exactly 1 adjacent cell for boundary facet, found {found}")]
    InvalidAdjacentCellCount {
        /// The number of adjacent cells found.
        found: usize,
    },
    /// Adjacent cell was not found in the triangulation.
    #[error("Adjacent cell not found")]
    AdjacentCellNotFound,
    /// Could not find inside vertex for boundary facet.
    #[error("Could not find inside vertex for boundary facet")]
    InsideVertexNotFound,
    /// Failed to compute geometric orientation.
    #[error("Failed to compute orientation: {details}")]
    OrientationComputationFailed {
        /// Details about the orientation computation failure.
        details: String,
    },
    /// Invalid facet index for a cell.
    #[error("Invalid facet index {index} for cell with {facet_count} facets")]
    InvalidFacetIndex {
        /// The invalid facet index.
        index: u8,
        /// The number of facets in the cell.
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
    /// Cell was not found in the triangulation.
    #[error("Cell not found in triangulation (potential data corruption)")]
    CellNotFoundInTriangulation,
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
}

// =============================================================================
// LIGHTWEIGHT FACET VIEW (Phase 3 Optimization)
// =============================================================================

/// Lightweight facet representation as a view into a triangulation data structure.
///
/// **Phase 3 Optimization**: This is the new lightweight facet implementation that
/// replaces the heavyweight `Facet` struct with an ~18x memory reduction.
///
/// `FacetView` represents a facet (d-1 dimensional face) of a d-dimensional cell
/// without storing any data directly. Instead, it maintains references to the TDS
/// and uses keys to access data on-demand.
///
/// # Memory Efficiency
///
/// Compared to the original `Facet<T, U, V, D>`:
/// - **Original**: Stores complete Cell + Vertex objects (~hundreds of bytes)
/// - **`FacetView`**: Stores TDS reference + `CellKey` + `facet_index` (~17 bytes)
/// - **Memory reduction: ~18x smaller**
///
/// # Type Parameters
///
/// - `'tds`: Lifetime of the triangulation data structure
/// - `T`: Coordinate scalar type
/// - `U`: Vertex data type  
/// - `V`: Cell data type
/// - `D`: Spatial dimension
///
/// # Examples
///
/// ```rust,no_run
/// use delaunay::core::facet::FacetView;
/// use delaunay::core::triangulation_data_structure::{Tds, CellKey};
///
/// // This is a conceptual example showing FacetView usage
/// // In practice, tds and cell_key would come from your triangulation
/// fn example_usage<'a>(tds: &'a Tds<f64, Option<()>, Option<()>, 3>, cell_key: CellKey) -> Result<(), Box<dyn std::error::Error>> {
///     // Create a facet view for the first facet of a cell
///     let facet_view = FacetView::new(tds, cell_key, 0)?;
///
///     // Access vertices through the view (lazy evaluation)
/// for vertex in facet_view.vertices().unwrap() {
///     println!("Vertex: {:?}", vertex.point());
/// }
///
///     // Get the opposite vertex
///     let opposite = facet_view.opposite_vertex()?;
///
///     // Compute facet key
///     let key = facet_view.key()?;
///     Ok(())
/// }
/// ```
pub struct FacetView<'tds, T, U, V, const D: usize>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
{
    /// Reference to the triangulation data structure.
    tds: &'tds Tds<T, U, V, D>,
    /// Key of the cell containing this facet.
    cell_key: CellKey,
    /// Index of this facet within the cell (0 <= `facet_index` < D+1).
    ///
    /// The `facet_index` indicates which vertex of the cell is the "opposite vertex"
    /// (the vertex not included in the facet). For a D-dimensional cell with D+1
    /// vertices, facet i excludes vertex i and includes all others.
    facet_index: u8,
}

impl<'tds, T, U, V, const D: usize> FacetView<'tds, T, U, V, D>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
{
    /// Returns the cell key for this facet.
    #[inline]
    #[must_use]
    pub const fn cell_key(&self) -> CellKey {
        self.cell_key
    }

    /// Returns the facet index within the cell.
    #[inline]
    #[must_use]
    pub const fn facet_index(&self) -> u8 {
        self.facet_index
    }

    /// Returns the TDS reference.
    #[inline]
    #[must_use]
    pub const fn tds(&self) -> &'tds Tds<T, U, V, D> {
        self.tds
    }

    /// Creates a new `FacetView` for the specified facet of a cell.
    ///
    /// # Arguments
    ///
    /// * `tds` - Reference to the triangulation data structure
    /// * `cell_key` - The key of the cell containing the facet
    /// * `facet_index` - The index of the facet within the cell (0 to D)
    ///
    /// # Returns
    ///
    /// A `Result<FacetView, FacetError>` containing the facet view if successful.
    ///
    /// # Errors
    ///
    /// Returns `FacetError` if:
    /// - `cell_key` is not found in the TDS
    /// - `facet_index` is out of bounds (>= D+1)
    pub fn new(
        tds: &'tds Tds<T, U, V, D>,
        cell_key: CellKey,
        facet_index: u8,
    ) -> Result<Self, FacetError> {
        // Validate cell exists
        let cell = tds
            .cells()
            .get(cell_key)
            .ok_or(FacetError::CellNotFoundInTriangulation)?;

        // Validate facet index
        let vertex_count = cell.vertices().len();
        if usize::from(facet_index) >= vertex_count {
            return Err(FacetError::InvalidFacetIndex {
                index: facet_index,
                facet_count: vertex_count,
            });
        }

        Ok(Self {
            tds,
            cell_key,
            facet_index,
        })
    }

    /// Returns an iterator over the vertices that make up this facet.
    ///
    /// The facet vertices are all vertices of the containing cell except
    /// the opposite vertex (at `facet_index`).
    ///
    /// This method is available with minimal trait bounds (only `CoordinateScalar`),
    /// enabling usage in lightweight operations that don't require arithmetic.
    ///
    /// # Returns
    ///
    /// A `Result` containing an iterator yielding references to vertices in the facet,
    /// or a `FacetError` if the cell is no longer present in the TDS.
    ///
    /// # Errors
    ///
    /// Returns `FacetError::CellNotFoundInTriangulation` if the cell key is no longer
    /// present in the TDS. This could happen if the TDS is modified after the `FacetView`
    /// is created, though this should not occur under normal usage patterns.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::core::facet::FacetView;
    /// use delaunay::core::triangulation_data_structure::Tds;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
    ///
    /// if let Some(cell_key) = tds.cell_keys().next() {
    ///     let facet = FacetView::new(&tds, cell_key, 0).unwrap();
    ///     let vertex_iter = facet.vertices().unwrap();
    ///     assert_eq!(vertex_iter.count(), 3); // 3D facet has 3 vertices
    /// }
    /// ```
    pub fn vertices(&self) -> Result<impl Iterator<Item = &'tds Vertex<T, U, D>>, FacetError> {
        let cell = self
            .tds
            .cells()
            .get(self.cell_key)
            .ok_or(FacetError::CellNotFoundInTriangulation)?;
        let facet_index = usize::from(self.facet_index);

        // Phase 3A: Use vertices and resolve via TDS
        // Use filter_map with get() to safely handle potentially invalid vertex keys
        Ok(cell
            .vertices()
            .iter()
            .enumerate()
            .filter(move |(i, _)| *i != facet_index)
            .filter_map(move |(_, &vkey)| self.tds.vertices().get(vkey)))
    }

    /// Returns the opposite vertex (the vertex not included in the facet).
    ///
    /// # Returns
    ///
    /// A `Result` containing a reference to the opposite vertex.
    ///
    /// # Errors
    ///
    /// Returns `FacetError::CellNotFoundInTriangulation` if the cell is no longer in the TDS.
    pub fn opposite_vertex(&self) -> Result<&'tds Vertex<T, U, D>, FacetError> {
        let cell = self
            .tds
            .cells()
            .get(self.cell_key)
            .ok_or(FacetError::CellNotFoundInTriangulation)?;

        // Phase 3A: Use vertices and resolve via TDS
        let vertices = cell.vertices();
        let facet_index = usize::from(self.facet_index);

        let vkey = vertices
            .get(facet_index)
            .ok_or(FacetError::InvalidFacetIndex {
                index: self.facet_index,
                facet_count: vertices.len(),
            })?;

        // Use get() to safely handle potentially invalid vertex keys
        self.tds
            .vertices()
            .get(*vkey)
            .ok_or(FacetError::CellNotFoundInTriangulation)
    }

    /// Returns the cell containing this facet.
    ///
    /// # Returns
    ///
    /// A `Result` containing a reference to the containing cell.
    ///
    /// # Errors
    ///
    /// Returns `FacetError::CellNotFoundInTriangulation` if the cell is no longer in the TDS.
    pub fn cell(&self) -> Result<&'tds Cell<T, U, V, D>, FacetError> {
        self.tds
            .cells()
            .get(self.cell_key)
            .ok_or(FacetError::CellNotFoundInTriangulation)
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
    pub fn key(&self) -> Result<u64, FacetError> {
        // Get vertex keys for the facet vertices
        let cell_vertices = self
            .tds
            .get_cell_vertices(self.cell_key)
            .map_err(|_| FacetError::CellNotFoundInTriangulation)?;
        let facet_index = usize::from(self.facet_index);

        // Collect vertex keys excluding the opposite vertex
        let facet_vertices: Vec<_> = cell_vertices
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != facet_index)
            .map(|(_, &key)| key)
            .collect();

        // Compute canonical key from vertex keys
        Ok(facet_key_from_vertices(&facet_vertices))
    }
}

// Trait implementations for FacetView
impl<T, U, V, const D: usize> Debug for FacetView<'_, T, U, V, D>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FacetView")
            .field("cell_key", &self.cell_key)
            .field("facet_index", &self.facet_index)
            .field("dimension", &D)
            .finish()
    }
}

#[allow(clippy::expl_impl_clone_on_copy)]
#[allow(clippy::non_canonical_clone_impl)]
impl<T, U, V, const D: usize> Clone for FacetView<'_, T, U, V, D>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
    [T; D]: Serialize + DeserializeOwned,
{
    fn clone(&self) -> Self {
        Self {
            tds: self.tds,
            cell_key: self.cell_key,
            facet_index: self.facet_index,
        }
    }
}

impl<T, U, V, const D: usize> Copy for FacetView<'_, T, U, V, D>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
{
}

impl<T, U, V, const D: usize> PartialEq for FacetView<'_, T, U, V, D>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
{
    fn eq(&self, other: &Self) -> bool {
        // Two facet views are equal if they reference the same facet
        std::ptr::eq(self.tds, other.tds)
            && self.cell_key == other.cell_key
            && self.facet_index == other.facet_index
    }
}

impl<T, U, V, const D: usize> Eq for FacetView<'_, T, U, V, D>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
{
}

/// Utility function to create multiple `FacetView`s for all facets of a cell.
///
/// # Arguments
///
/// * `tds` - Reference to the triangulation data structure
/// * `cell_key` - Key of the cell to create facet views for
///
/// # Returns
///
/// A `Result` containing a `Vec` of `FacetView`s for all facets of the cell.
///
/// # Errors
///
/// Returns `FacetError` if the cell is not found or has invalid structure.
///
/// # Note
///
/// Removed unnecessary numeric bounds (`AddAssign`, `SubAssign`, `Sum`, `NumCast`, `Div`)
/// since this function doesn't perform any arithmetic operations.
pub fn all_facets_for_cell<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    cell_key: CellKey,
) -> Result<Vec<FacetView<'_, T, U, V, D>>, FacetError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
{
    let cell = tds
        .cells()
        .get(cell_key)
        .ok_or(FacetError::CellNotFoundInTriangulation)?;

    let vertex_count = cell.vertices().len();
    let mut facet_views = Vec::with_capacity(vertex_count);

    for facet_index in 0..vertex_count {
        let idx = facet_index; // usize
        let facet_view = FacetView::new(tds, cell_key, usize_to_u8(idx, vertex_count)?)?;
        facet_views.push(facet_view);
    }

    Ok(facet_views)
}

/// Iterator over all facets in a triangulation data structure.
///
/// This iterator provides efficient access to all facets without allocating
/// a vector. It's particularly useful for performance-critical operations
/// like boundary detection and cavity analysis in triangulation insertion.
#[derive(Clone)]
pub struct AllFacetsIter<'tds, T, U, V, const D: usize>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
{
    tds: &'tds Tds<T, U, V, D>,
    cell_keys: std::vec::IntoIter<CellKey>,
    current_cell_key: Option<CellKey>,
    current_facet_index: usize,
    current_cell_facet_count: usize,
}

impl<'tds, T, U, V, const D: usize> AllFacetsIter<'tds, T, U, V, D>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
{
    /// Creates a new iterator over all facets in the TDS.
    #[must_use]
    pub fn new(tds: &'tds Tds<T, U, V, D>) -> Self {
        // We collect here because we need an owned iterator to store in the struct
        // CellKey is just u64, so this is efficient
        #[allow(clippy::needless_collect)]
        let cell_keys: Vec<CellKey> = tds.cell_keys().collect();
        Self {
            tds,
            cell_keys: cell_keys.into_iter(),
            current_cell_key: None,
            current_facet_index: 0,
            current_cell_facet_count: 0,
        }
    }
}

impl<'tds, T, U, V, const D: usize> Iterator for AllFacetsIter<'tds, T, U, V, D>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
{
    type Item = FacetView<'tds, T, U, V, D>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // If we have a current cell and more facets in it
            if let Some(cell_key) = self.current_cell_key
                && self.current_facet_index < self.current_cell_facet_count
            {
                let facet_index = self.current_facet_index;
                self.current_facet_index += 1;

                // Create FacetView - we know this is valid since we're iterating within bounds
                let Ok(facet_u8) = usize_to_u8(facet_index, self.current_cell_facet_count) else {
                    // Skip indices that cannot be represented; avoids silent truncation
                    debug_assert!(
                        false,
                        "Facet index {facet_index} exceeds u8 range (max 255). Consider widening FacetView index type for dimension {D}"
                    );
                    continue;
                };
                if let Ok(facet_view) = FacetView::new(self.tds, cell_key, facet_u8) {
                    return Some(facet_view);
                }
            }

            // Move to next cell
            if let Some(next_cell_key) = self.cell_keys.next() {
                if let Some(cell) = self.tds.cells().get(next_cell_key) {
                    self.current_cell_key = Some(next_cell_key);
                    self.current_facet_index = 0;
                    self.current_cell_facet_count = cell.vertices().len();
                    // Continue loop to process first facet of new cell
                } else {
                    // Cell not found, skip to next (continue is implicit at end of loop)
                }
            } else {
                // No more cells
                return None;
            }
        }
    }
}

/// Iterator over boundary facets in a triangulation.
///
/// This iterator efficiently identifies and yields only the boundary facets
/// (facets that belong to only one cell) without pre-computing all facets.
#[derive(Clone)]
pub struct BoundaryFacetsIter<'tds, T, U, V, const D: usize>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
{
    all_facets: AllFacetsIter<'tds, T, U, V, D>,
    facet_to_cells_map: crate::core::collections::FacetToCellsMap,
}

impl<'tds, T, U, V, const D: usize> BoundaryFacetsIter<'tds, T, U, V, D>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
{
    /// Creates a new iterator over boundary facets.
    #[must_use]
    pub fn new(
        tds: &'tds Tds<T, U, V, D>,
        facet_to_cells_map: crate::core::collections::FacetToCellsMap,
    ) -> Self {
        Self {
            all_facets: AllFacetsIter::new(tds),
            facet_to_cells_map,
        }
    }
}

impl<'tds, T, U, V, const D: usize> Iterator for BoundaryFacetsIter<'tds, T, U, V, D>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
{
    type Item = FacetView<'tds, T, U, V, D>;

    fn next(&mut self) -> Option<Self::Item> {
        // Find the next boundary facet
        self.all_facets.find(|facet_view| {
            // Check if this facet is a boundary facet using the precomputed map
            if let Ok(facet_key) = facet_view.key()
                && let Some(cell_list) = self.facet_to_cells_map.get(&facet_key)
            {
                // Boundary facets appear in exactly one cell
                return cell_list.len() == 1;
            }
            false
        })
    }
}

// =============================================================================
// DEPRECATED FACET TYPE REMOVED
// =============================================================================
// The heavyweight Facet<T, U, V, D> type has been deleted.
// All code should use FacetView instead for 18x memory reduction.
// FacetView is a lightweight view that references the TDS instead of
// storing complete Cell and Vertex objects.
//
// Migration guide:
// - Old: Facet::new(cell, vertex)
// - New: FacetView::new(&tds, cell_key, facet_index)
//
// - Old: facet.vertices() -> Vec<Vertex>
// - New: facet.vertices()? -> impl Iterator<Item = &Vertex>
//
// - Old: facet.cell() -> &Cell
// - New: facet.cell()? -> &Cell (requires TDS lookup)

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
/// use delaunay::core::facet::facet_key_from_vertices;
/// use delaunay::core::triangulation_data_structure::VertexKey;
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
    let mut key_values: Vec<u64> = vertices.iter().map(|key| key.data().as_ffi()).collect();
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
    use crate::core::triangulation_data_structure::{Tds, VertexKey};
    use crate::vertex;

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
        let vertices = vec![vertex!([0.0]), vertex!([1.0])];
        let tds: Tds<f64, Option<()>, Option<()>, 1> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        // Test invalid facet index (should be 0 or 1 for 1D, facet_index >= 2 is invalid)
        assert!(matches!(
            FacetView::new(&tds, cell_key, 99),
            Err(FacetError::InvalidFacetIndex { .. })
        ));
    }

    #[test]
    fn facet_new() {
        // Create a 3D triangulation with a tetrahedron
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        // Create facet view for facet 0 (excludes vertex 0)
        let facet = FacetView::new(&tds, cell_key, 0).unwrap();
        assert_eq!(facet.cell_key(), cell_key);
        assert_eq!(facet.facet_index(), 0);

        // Human readable output for cargo test -- --nocapture
        println!(
            "FacetView: cell_key={:?}, facet_index={}",
            facet.cell_key(),
            facet.facet_index()
        );
    }

    #[test]
    fn test_facet_new_success_coverage() {
        // Test 2D case: Create a triangle (2D cell with 3 vertices)
        let vertices_2d = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.5, 1.0]),
        ];
        let tds_2d: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_2d).unwrap();
        let cell_key_2d = tds_2d.cell_keys().next().unwrap();
        let result_2d = FacetView::new(&tds_2d, cell_key_2d, 0);

        // Assert that the result is Ok
        assert!(result_2d.is_ok());
        let facet_2d = result_2d.unwrap();
        assert_eq!(facet_2d.vertices().unwrap().count(), 2); // 2D facet should have 2 vertices

        // Test 1D case: Create an edge (1D cell with 2 vertices)
        let vertices_1d = vec![vertex!([0.0]), vertex!([1.0])];
        let tds_1d: Tds<f64, Option<()>, Option<()>, 1> = Tds::new(&vertices_1d).unwrap();
        let cell_key_1d = tds_1d.cell_keys().next().unwrap();
        let result_1d = FacetView::new(&tds_1d, cell_key_1d, 0);

        // Assert that the result is Ok
        assert!(result_1d.is_ok());
        let facet_1d = result_1d.unwrap();
        assert_eq!(facet_1d.vertices().unwrap().count(), 1); // 1D facet should have 1 vertex
    }

    #[test]
    fn facet_new_with_incorrect_vertex() {
        // Create a 3D triangulation
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        // Test invalid facet index (3D cell has vertices 0-3, facet index 4 is invalid)
        assert!(FacetView::new(&tds, cell_key, 4).is_err());
    }

    #[test]
    fn facet_vertices() {
        // Create a 3D triangulation with a tetrahedron
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        // Create facet view for facet 0 (excludes vertex 0)
        let facet = FacetView::new(&tds, cell_key, 0).unwrap();
        let facet_vertices: Vec<_> = facet.vertices().unwrap().collect();

        assert_eq!(facet_vertices.len(), 3);
        // Facet 0 should contain vertices 1, 2, 3 (all except vertex 0)

        // Human readable output for cargo test -- --nocapture
        println!(
            "FacetView: facet_index={}, vertex_count={}",
            facet.facet_index(),
            facet_vertices.len()
        );
    }

    // =============================================================================
    // EQUALITY AND ORDERING TESTS
    // =============================================================================

    #[test]
    fn facet_partial_eq() {
        // Create a 3D triangulation
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        // Create facet views with same facet index (should be equal)
        let facet1 = FacetView::new(&tds, cell_key, 0).unwrap();
        let facet2 = FacetView::new(&tds, cell_key, 0).unwrap();
        let facet3 = FacetView::new(&tds, cell_key, 1).unwrap();

        assert_eq!(facet1, facet2);
        assert_ne!(facet1, facet3);
    }

    // Note: PartialOrd is not implemented for FacetView as facet ordering
    // doesn't have semantic meaning in triangulation operations.
    // The old Facet::partial_ord test has been removed.

    #[test]
    fn facet_clone() {
        // Create a 3D triangulation
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        let facet = FacetView::new(&tds, cell_key, 0).unwrap();
        let cloned_facet = facet;

        // Verify clones are equal
        assert_eq!(facet, cloned_facet);
        assert_eq!(facet.cell_key(), cloned_facet.cell_key());
        assert_eq!(facet.facet_index(), cloned_facet.facet_index());

        // Verify cell and opposite vertex are accessible through both views
        let cell1 = facet.cell().unwrap();
        let cell2 = cloned_facet.cell().unwrap();
        assert_eq!(cell1.uuid(), cell2.uuid());

        let vertex1 = facet.opposite_vertex().unwrap();
        let vertex2 = cloned_facet.opposite_vertex().unwrap();
        assert_eq!(vertex1.uuid(), vertex2.uuid());
    }

    #[test]
    fn facet_debug() {
        // Create a 3D triangulation
        let vertices = vec![
            vertex!([1.0, 2.0, 3.0]),
            vertex!([4.0, 5.0, 6.0]),
            vertex!([7.0, 8.0, 9.0]),
            vertex!([10.0, 11.0, 12.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        let facet = FacetView::new(&tds, cell_key, 0).unwrap();
        let debug_str = format!("{facet:?}");

        assert!(debug_str.contains("FacetView"));
        assert!(debug_str.contains("cell_key"));
        assert!(debug_str.contains("facet_index"));
        assert!(debug_str.contains("dimension"));
    }

    // =============================================================================
    // DIMENSIONAL AND GEOMETRIC TESTS
    // =============================================================================

    #[test]
    fn facet_with_typed_data() {
        // Create 3D triangulation with typed vertex data
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0], 1),
            vertex!([1.0, 0.0, 0.0], 2),
            vertex!([0.0, 1.0, 0.0], 3),
            vertex!([0.0, 0.0, 1.0], 4),
        ];
        let tds: Tds<f64, i32, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        // Create facet view for facet 0 (excludes vertex 0)
        let facet = FacetView::new(&tds, cell_key, 0).unwrap();

        let facet_vertices: Vec<_> = facet.vertices().unwrap().collect();
        assert_eq!(facet_vertices.len(), 3); // 3D facet should have 3 vertices (D)
        assert!(facet_vertices.iter().any(|v| v.data == Some(2)));
        assert!(facet_vertices.iter().any(|v| v.data == Some(3)));
        assert!(facet_vertices.iter().any(|v| v.data == Some(4)));
    }

    #[test]
    fn facet_2d_triangle() {
        // Create 2D triangulation (triangle)
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.5, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        // Create facet view for facet 0 (excludes vertex 0)
        let facet = FacetView::new(&tds, cell_key, 0).unwrap();

        // Facet of 2D triangle is an edge (1D) with 2 vertices
        assert_eq!(facet.vertices().unwrap().count(), 2);
    }

    #[test]
    fn facet_1d_edge() {
        // Create 1D triangulation (edge with 2 vertices)
        let vertices = vec![vertex!([0.0]), vertex!([1.0])];
        let tds: Tds<f64, Option<()>, Option<()>, 1> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        // Create facet view for facet 0 (excludes vertex 0)
        let facet = FacetView::new(&tds, cell_key, 0).unwrap();

        // Facet of 1D edge is a point (0D) with 1 vertex
        assert_eq!(facet.vertices().unwrap().count(), 1);
    }

    #[test]
    fn facet_4d_simplex() {
        // Create 4D triangulation (4-simplex with 5 vertices)
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 4> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        // Create facet view for facet 0 (excludes vertex 0)
        let facet = FacetView::new(&tds, cell_key, 0).unwrap();

        // Facet of 4D simplex is a 3D tetrahedron with 4 vertices
        assert_eq!(facet.vertices().unwrap().count(), 4);
    }

    // =============================================================================
    // ERROR HANDLING TESTS
    // =============================================================================

    #[test]
    fn facet_error_display() {
        let cell_error = FacetError::CellDoesNotContainVertex;

        assert_eq!(
            cell_error.to_string(),
            "The cell does not contain the vertex!"
        );
    }

    #[test]
    fn facet_error_debug() {
        let cell_error = FacetError::CellDoesNotContainVertex;

        let cell_debug = format!("{cell_error:?}");

        assert!(cell_debug.contains("CellDoesNotContainVertex"));
    }

    #[test]
    fn test_facet_key_consistency() {
        // Create 3D triangulation with a tetrahedron
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        // Create facet views for different facets
        let facet1 = FacetView::new(&tds, cell_key, 0).unwrap(); // excludes vertex 0
        let facet2 = FacetView::new(&tds, cell_key, 0).unwrap(); // same facet
        let facet3 = FacetView::new(&tds, cell_key, 1).unwrap(); // excludes vertex 1 (different facet)

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
    fn facet_vertices_empty_cell() {
        // Test edge case of minimal cell (1D edge with 2 vertices)
        let vertices = vec![vertex!([0.0]), vertex!([1.0])];
        let tds: Tds<f64, Option<()>, Option<()>, 1> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        // Create facet with vertex 0 as opposite - should have only vertex 1 in facet
        let facet = FacetView::new(&tds, cell_key, 0).unwrap();
        assert_eq!(facet.vertices().unwrap().count(), 1);

        // Test the opposite case - vertex 1 as opposite should have only vertex 0 in facet
        let other_facet = FacetView::new(&tds, cell_key, 1).unwrap();
        assert_eq!(other_facet.vertices().unwrap().count(), 1);
    }

    #[test]
    fn facet_vertices_ordering() {
        // Test that vertices are filtered correctly
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        // Create facet view for facet 2 (excludes vertex 2)
        let facet = FacetView::new(&tds, cell_key, 2).unwrap();

        // Should have all vertices except vertex at index 2
        assert_eq!(facet.vertices().unwrap().count(), 3);
        // Verify we have exactly 3 vertices (the D vertices of the D-1 dimensional facet)
    }

    #[test]
    fn facet_eq_different_vertices() {
        // Create a 3D triangulation
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        let facet1 = FacetView::new(&tds, cell_key, 0).unwrap();
        let facet2 = FacetView::new(&tds, cell_key, 1).unwrap();
        let facet3 = FacetView::new(&tds, cell_key, 2).unwrap();
        let facet4 = FacetView::new(&tds, cell_key, 3).unwrap();

        // All facets should be different because they have different facet indices
        // (i.e., different opposite vertices)
        assert_ne!(facet1, facet2);
        assert_ne!(facet1, facet3);
        assert_ne!(facet1, facet4);
        assert_ne!(facet2, facet3);
        assert_ne!(facet2, facet4);
        assert_ne!(facet3, facet4);
    }

    // Note: Hash is not implemented for FacetView as it contains a reference.
    // Use FacetView::key() to get a hashable u64 key for facet identity.
    // The old Facet::hash test has been removed.
    #[test]
    fn facet_key_hash() {
        // Create a 3D triangulation
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        // Create two facet views that reference the same facet
        let facet1 = FacetView::new(&tds, cell_key, 0).unwrap();
        let facet2 = FacetView::new(&tds, cell_key, 0).unwrap();

        // Create a different facet
        let facet3 = FacetView::new(&tds, cell_key, 1).unwrap();

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
        use slotmap::SlotMap;
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
    // PHASE 3: FACET VIEW TESTS
    // =============================================================================

    #[test]
    fn test_facet_view_creation() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cells().keys().next().unwrap();

        // Test valid facet creation
        let facet_view = FacetView::new(&tds, cell_key, 0).unwrap();
        assert_eq!(facet_view.cell_key(), cell_key);
        assert_eq!(facet_view.facet_index(), 0);

        // Test invalid facet index
        let result = FacetView::new(&tds, cell_key, 10);
        assert!(matches!(result, Err(FacetError::InvalidFacetIndex { .. })));
    }

    #[test]
    fn test_facet_view_vertices_iteration() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cells().keys().next().unwrap();

        let facet_view = FacetView::new(&tds, cell_key, 0).unwrap();

        // Facet opposite to vertex 0 should have 3 vertices (D vertices in D-1 facet)
        let facet_vertices: Vec<_> = facet_view.vertices().unwrap().collect();
        assert_eq!(facet_vertices.len(), 3);

        // Get original vertices for comparison
        let original_vertices = vertices;
        let opposite_vertex = &original_vertices[0];

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
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cells().keys().next().unwrap();

        let facet_view = FacetView::new(&tds, cell_key, 1).unwrap();
        let opposite = facet_view.opposite_vertex().unwrap();

        // The opposite vertex should be the vertex at index 1
        let cell_vertex_keys = tds.cells()[cell_key].vertices();
        let expected_vertex = &tds.vertices()[cell_vertex_keys[1]];
        assert_eq!(opposite.uuid(), expected_vertex.uuid());
    }

    #[test]
    fn test_facet_view_key_computation() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cells().keys().next().unwrap();

        let facet_view = FacetView::new(&tds, cell_key, 0).unwrap();
        let key = facet_view.key().unwrap();

        // Key should be non-zero for valid facet
        assert_ne!(key, 0);
    }

    #[test]
    fn test_all_facets_for_cell() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cells().keys().next().unwrap();

        let facet_views = all_facets_for_cell(&tds, cell_key).unwrap();

        // 3D cell (tetrahedron) should have 4 facets
        assert_eq!(facet_views.len(), 4);

        // Each facet should have a different index
        for (i, facet_view) in facet_views.iter().enumerate() {
            assert_eq!(
                facet_view.facet_index(),
                usize_to_u8(i, facet_views.len()).unwrap()
            );
            assert_eq!(facet_view.cell_key(), cell_key);
        }
    }

    #[test]
    fn test_facet_view_equality() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cells().keys().next().unwrap();

        let facet_view1 = FacetView::new(&tds, cell_key, 0).unwrap();
        let facet_view2 = FacetView::new(&tds, cell_key, 0).unwrap();
        let facet_view3 = FacetView::new(&tds, cell_key, 1).unwrap();

        // Same facet should be equal
        assert_eq!(facet_view1, facet_view2);

        // Different facets should not be equal
        assert_ne!(facet_view1, facet_view3);
    }

    #[test]
    fn test_facet_view_debug() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cells().keys().next().unwrap();

        let facet_view = FacetView::new(&tds, cell_key, 1).unwrap();
        let debug_str = format!("{facet_view:?}");

        assert!(debug_str.contains("FacetView"));
        assert!(debug_str.contains("cell_key"));
        assert!(debug_str.contains("facet_index"));
        assert!(debug_str.contains("dimension"));
    }

    #[test]
    fn test_facet_view_memory_efficiency() {
        use std::mem;

        // This test demonstrates the memory efficiency of FacetView
        // The deprecated heavyweight Facet struct has been removed.
        let lightweight_size = mem::size_of::<FacetView<f64, Option<()>, Option<()>, 3>>();

        println!("Lightweight FacetView size: {lightweight_size} bytes");

        // FacetView should be around 17 bytes (8 byte ref + 8 byte CellKey + 1 byte facet_index)
        // Allow for some padding/alignment
        assert!(lightweight_size <= 24);

        // Document actual size for reference
        // On 64-bit systems: typically 17 bytes (reference + CellKey + u8)
    }
}
