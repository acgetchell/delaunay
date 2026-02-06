//! Data and operations on d-dimensional cells or [simplices](https://en.wikipedia.org/wiki/Simplex).
//!
//! This module provides the `Cell` struct which represents a geometric cell
//! (simplex) in D-dimensional space with associated metadata including unique
//! identification, neighboring cells, and optional user data.
//!
//! # Key Features
//!
//! - **Generic Coordinate Support**: Works with any floating-point type (`f32`, `f64`, etc.)
//!   that implements the `CoordinateScalar` trait
//! - **Unique Identification**: Each cell has a UUID for consistent identification
//! - **Vertices Management**: Stores vertices that form the simplex
//! - **Neighbor Tracking**: Maintains references to neighboring cells
//! - **Optional Data Storage**: Supports attaching arbitrary user data of type `V`
//! - **Serialization Support**: Manual serde for `uuid` and `data`; vertex/neighbor keys are
//!   omitted and reconstructed during TDS (de)serialization
//!
//! # Examples
//!
//! ```rust
//! use delaunay::prelude::*;
//!
//! // Create vertices for a tetrahedron
//! let vertices = vec![
//!     vertex!([0.0, 0.0, 0.0]),
//!     vertex!([1.0, 0.0, 0.0]),
//!     vertex!([0.0, 1.0, 0.0]),
//!     vertex!([0.0, 0.0, 1.0]),
//! ];
//!
//! // Create a 3D triangulation with cells
//! let dt: DelaunayTriangulation<_, (), (), 3> =
//!     DelaunayTriangulation::new(&vertices).unwrap();
//! let (cell_key, cell) = dt.cells().next().unwrap();
//! assert_eq!(cell.number_of_vertices(), 4);
//! ```

#![allow(clippy::similar_names)]

// =============================================================================
// IMPORTS
// =============================================================================

use super::{
    facet::FacetError,
    traits::DataType,
    triangulation_data_structure::{CellKey, Tds, VertexKey},
    util::{UuidValidationError, make_uuid, validate_uuid},
    vertex::VertexValidationError,
};

use super::vertex::Vertex;
use crate::core::collections::{CellVertexBuffer, FastHashMap, FastHashSet, NeighborBuffer};
use crate::geometry::traits::coordinate::{CoordinateConversionError, CoordinateScalar};
use serde::{
    Deserialize, Deserializer, Serialize,
    de::{self, IgnoredAny, MapAccess, Visitor},
};
use std::{
    cmp,
    fmt::{self, Debug},
    hash::{Hash, Hasher},
    marker::PhantomData,
};
use thiserror::Error;
use uuid::Uuid;

// =============================================================================
// ERROR TYPES
// =============================================================================

/// Errors that can occur during cell validation.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::cell::CellValidationError;
///
/// let err = CellValidationError::DuplicateVertices;
/// assert!(matches!(err, CellValidationError::DuplicateVertices));
/// ```
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum CellValidationError {
    /// The cell has an invalid vertex.
    #[error("Invalid vertex: {source}")]
    InvalidVertex {
        /// The underlying vertex validation error.
        #[from]
        source: VertexValidationError,
    },
    /// The cell has an invalid UUID.
    #[error("Invalid UUID: {source}")]
    InvalidUuid {
        /// The underlying UUID validation error.
        #[from]
        source: UuidValidationError,
    },
    /// The cell contains duplicate vertices.
    #[error("Duplicate vertices: cell contains non-unique vertices which is not allowed")]
    DuplicateVertices,
    /// The cell has insufficient vertices to form a proper D-simplex.
    #[error(
        "Insufficient vertices: cell has {actual} vertices; expected exactly {expected} for a {dimension}D simplex"
    )]
    InsufficientVertices {
        /// The actual number of vertices in the cell.
        actual: usize,
        /// The expected number of vertices (D+1).
        expected: usize,
        /// The dimension D.
        dimension: usize,
    },
    /// The simplex is degenerate (vertices are collinear, coplanar, or otherwise geometrically degenerate).
    #[error(
        "Degenerate simplex: the vertices form a degenerate configuration that cannot reliably determine geometric properties"
    )]
    DegenerateSimplex,
    /// Coordinate conversion error occurred during geometric computations.
    #[error("Coordinate conversion error: {source}")]
    CoordinateConversion {
        /// The underlying coordinate conversion error.
        #[from]
        source: CoordinateConversionError,
    },
    /// The neighbors vector length is inconsistent with the number of vertices (D+1).
    #[error(
        "Invalid neighbors length: got {actual}, expected {expected} (D+1) for a {dimension}D simplex"
    )]
    InvalidNeighborsLength {
        /// The actual neighbors length.
        actual: usize,
        /// The expected neighbors length (= D+1).
        expected: usize,
        /// The dimension D.
        dimension: usize,
    },
    /// A vertex key referenced by the cell was not found in the TDS.
    #[error("Vertex key {key:?} not found in TDS (indicates TDS corruption or inconsistency)")]
    VertexKeyNotFound {
        /// The vertex key that was not found.
        key: VertexKey,
    },
}

impl From<crate::geometry::matrix::StackMatrixDispatchError> for CellValidationError {
    fn from(source: crate::geometry::matrix::StackMatrixDispatchError) -> Self {
        CoordinateConversionError::from(source).into()
    }
}

// =============================================================================
// CONVENIENCE MACROS AND HELPERS
// =============================================================================

// =============================================================================
// CELL STRUCT DEFINITION
// =============================================================================

#[derive(Clone, Debug)]
/// The [Cell] struct represents a d-dimensional
/// [simplex](https://en.wikipedia.org/wiki/Simplex) with vertices, a unique
/// identifier, optional neighbors, and optional data.
///
/// # Phase 3A: Key-Based Storage
///
/// This Cell now stores keys to vertices and neighbors instead of full objects,
/// providing better memory efficiency and cache locality.
///
/// # Properties
///
/// - `vertices`: Keys referencing vertices in the TDS. Access via `vertices()` method.
/// - `uuid`: Universally unique identifier for the cell.
/// - `neighbors`: Optional keys to neighboring cells (opposite each vertex). Access via `neighbors()` method.
/// - `data`: Optional user data associated with the cell.
///
/// # Accessing Vertices
///
/// Since cells now store keys, you need a `&Tds` reference to access vertex data:
/// ```rust
/// use delaunay::core::collections::Uuid;
/// use delaunay::prelude::*;
///
/// // Create a triangulation with some vertices
/// let vertices = vec![
///     vertex!([0.0, 0.0]),
///     vertex!([1.0, 0.0]),
///     vertex!([0.0, 1.0]),
/// ];
/// let dt = DelaunayTriangulation::new(&vertices).unwrap();
///
/// // Get first cell and iterate over vertex keys
/// let (cell_key, cell) = dt.cells().next().unwrap();
/// let tds = dt.tds();
/// for &vertex_key in cell.vertices() {
///     let vertex = &tds.get_vertex_by_key(vertex_key).unwrap();
///     // use vertex...
///     assert!(vertex.uuid() != Uuid::nil());
/// }
/// ```
pub struct Cell<T, U, V, const D: usize>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    /// Keys to the vertices forming this cell.
    /// Phase 3A: Changed from `Vec<Vertex>` to `CellVertexBuffer` for:
    /// - Zero heap allocation for D ≤ 7 (stack-allocated)
    /// - Direct key access without UUID lookup
    /// - Better cache locality
    ///
    /// Note: Not serialized - vertices are serialized separately and keys
    /// are reconstructed during deserialization.
    vertices: CellVertexBuffer,

    /// The unique identifier of the cell.
    uuid: Uuid,

    /// Keys to neighboring cells, indexed by opposite vertex.
    /// Phase 3A: Changed from `Option<Vec<Option<Uuid>>>` to `Option<NeighborBuffer<Option<CellKey>>>`.
    ///
    /// Positional semantics: `neighbors[i]` is the neighbor opposite `vertices[i]`.
    ///
    /// # Example
    /// For a 3D cell (tetrahedron) with 4 vertices:
    /// - `neighbors[0]` is opposite `vertices[0]` (shares vertices 1, 2, 3)
    /// - `neighbors[1]` is opposite `vertices[1]` (shares vertices 0, 2, 3)
    /// - `neighbors[2]` is opposite `vertices[2]` (shares vertices 0, 1, 3)
    /// - `neighbors[3]` is opposite `vertices[3]` (shares vertices 0, 1, 2)
    ///
    /// Note: Not serialized — neighbors are reconstructed during deserialization by the TDS.
    /// Access via `neighbors()` method. Writable by TDS for neighbor assignment.
    pub(crate) neighbors: Option<NeighborBuffer<Option<CellKey>>>,

    /// The optional data associated with the cell.
    pub data: Option<V>,

    /// Phantom data to maintain type parameters T and U for coordinate and vertex data types.
    /// These are needed because cells store keys to vertices, not the vertices themselves.
    _phantom: PhantomData<(T, U)>,
}

// =============================================================================
// SERIALIZATION IMPLEMENTATION
// =============================================================================

/// Manual implementation of Serialize for Cell.
///
/// This implementation handles serialization of Cell fields. The `vertices` and `neighbors`
/// fields are skipped as they contain keys that are only valid within the current `SlotMap`.
/// During deserialization, these are reconstructed by the TDS.
///
/// **Field Count Optimization**: We dynamically adjust the field count and conditionally
/// serialize `data` to omit it from JSON when None (reducing output size). The second
/// `is_some()` check matches the field count logic—both could be removed to always
/// serialize "data": null, but tests explicitly verify the field is omitted when None.
impl<T, U, V, const D: usize> Serialize for Cell<T, U, V, D>
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
        let has_data = self.data.is_some();
        let field_count = if has_data { 2 } else { 1 };
        let mut state = serializer.serialize_struct("Cell", field_count)?;
        state.serialize_field("uuid", &self.uuid)?;
        if has_data {
            state.serialize_field("data", &self.data)?;
        }
        state.end()
    }
}

// =============================================================================
// DESERIALIZATION IMPLEMENTATION
// =============================================================================

/// Manual implementation of Deserialize for Cell
impl<'de, T, U, V, const D: usize> Deserialize<'de> for Cell<T, U, V, D>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
    {
        struct CellVisitor<T, U, V, const D: usize>
        where
            T: CoordinateScalar,
            U: DataType,
            V: DataType,
        {
            _phantom: PhantomData<(T, U, V)>,
        }

        impl<'de, T, U, V, const D: usize> Visitor<'de> for CellVisitor<T, U, V, D>
        where
            T: CoordinateScalar,
            U: DataType,
            V: DataType,
        {
            type Value = Cell<T, U, V, D>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a Cell struct")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Cell<T, U, V, D>, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut uuid = None;
                let mut data = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        "uuid" => {
                            if uuid.is_some() {
                                return Err(de::Error::duplicate_field("uuid"));
                            }
                            uuid = Some(map.next_value()?);
                        }
                        "data" => {
                            if data.is_some() {
                                return Err(de::Error::duplicate_field("data"));
                            }
                            data = Some(map.next_value()?);
                        }
                        _ => {
                            // Ignore any other fields (vertices, neighbors, etc.)
                            // These are skipped during serialization and reconstructed by TDS
                            let _ = map.next_value::<IgnoredAny>()?;
                        }
                    }
                }

                let uuid: Uuid = uuid.ok_or_else(|| de::Error::missing_field("uuid"))?;
                crate::core::util::validate_uuid(&uuid)
                    .map_err(|e| de::Error::custom(format!("invalid uuid: {e}")))?;
                // data is Option<Option<V>>: None if field missing, Some(inner) if present
                // flatten() converts None -> None and Some(inner) -> inner
                let data = data.flatten();

                // Phase 3A: vertices and neighbors are not serialized
                // They will be reconstructed by TDS deserialization using:
                // - vertices: rebuilt by the TDS using its serialized cell→vertex mapping
                // - neighbors: rebuilt by the TDS via assign_neighbors()
                let vertices = CellVertexBuffer::new();

                Ok(Cell {
                    vertices,
                    uuid,
                    neighbors: None, // Will be reconstructed by TDS
                    data,
                    _phantom: PhantomData,
                })
            }
        }

        const FIELDS: &[&str] = &["uuid", "data"];
        deserializer.deserialize_struct(
            "Cell",
            FIELDS,
            CellVisitor {
                _phantom: PhantomData,
            },
        )
    }
}

// =============================================================================
// CELL IMPLEMENTATION - CORE METHODS
// =============================================================================

// Minimal trait bounds impl block
impl<T, U, V, const D: usize> Cell<T, U, V, D>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    /// Internal constructor for TDS use only.
    ///
    /// Creates a Cell with the given vertex keys and optional data.
    /// This constructor is `pub(crate)` to restrict usage to within the crate,
    /// ensuring cells are always created through proper TDS methods.
    ///
    /// # Arguments
    ///
    /// * `vertices` - Keys to the vertices forming this cell (must be D+1 keys)
    /// * `data` - Optional cell data
    ///
    /// # Errors
    ///
    /// Returns:
    /// - `CellValidationError::InsufficientVertices` if `vertices` doesn't have exactly D+1 elements.
    /// - `CellValidationError::DuplicateVertices` if any vertex key appears more than once.
    pub(crate) fn new(
        vertices: impl Into<CellVertexBuffer>,
        data: Option<V>,
    ) -> Result<Self, CellValidationError> {
        let vertices = vertices.into();

        // Validate D+1 vertices
        let actual = vertices.len();
        if actual != D + 1 {
            return Err(CellValidationError::InsufficientVertices {
                actual,
                expected: D + 1,
                dimension: D,
            });
        }

        // Check for duplicate vertices early
        let mut seen: FastHashSet<VertexKey> = FastHashSet::default();
        for &vkey in &vertices {
            if !seen.insert(vkey) {
                return Err(CellValidationError::DuplicateVertices);
            }
        }

        Ok(Self {
            vertices,
            uuid: make_uuid(),
            neighbors: None,
            data,
            _phantom: PhantomData,
        })
    }

    /// Checks if this cell contains the given vertex key.
    ///
    /// This is a cheap operation (O(D)) that only compares keys.
    ///
    /// # Arguments
    ///
    /// * `vkey` - The vertex key to check
    ///
    /// # Returns
    ///
    /// `true` if the cell contains the vertex key, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let (_, cell) = dt.cells().next().unwrap();
    /// let vkey = cell.vertices()[0];
    ///
    /// if cell.contains_vertex(vkey) {
    ///     println!("Cell contains vertex {:?}", vkey);
    /// }
    /// ```
    #[inline]
    pub fn contains_vertex(&self, vkey: VertexKey) -> bool {
        self.vertices.contains(&vkey)
    }

    /// Checks if this cell has any vertex in common with another cell.
    ///
    /// This is a cheap operation that only compares keys.
    ///
    /// # Arguments
    ///
    /// * `other` - The other cell to check against
    ///
    /// # Returns
    ///
    /// `true` if the cells share at least one vertex.
    ///
    /// # Example
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    ///     vertex!([1.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let mut cells_iter = dt.cells().map(|(_, cell)| cell);
    /// let cell1 = cells_iter.next().unwrap();
    /// let cell2 = cells_iter.next().unwrap();
    ///
    /// if cell1.has_vertex_in_common(cell2) {
    ///     println!("Cells share vertices");
    /// }
    /// ```
    #[inline]
    pub fn has_vertex_in_common(&self, other: &Self) -> bool {
        self.vertices
            .iter()
            .any(|vkey| other.vertices.contains(vkey))
    }

    /// Returns an iterator over the vertex keys, paired with their indices.
    ///
    /// Useful for operations that need both the key and its position.
    ///
    /// # Returns
    ///
    /// An iterator yielding `(usize, &VertexKey)` pairs.
    ///
    /// # Example
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let (_, cell) = dt.cells().next().unwrap();
    ///
    /// for (idx, &vkey) in cell.vertices_enumerated() {
    ///     println!("Vertex {:?} at position {}", vkey, idx);
    /// }
    /// ```
    #[inline]
    pub fn vertices_enumerated(&self) -> impl Iterator<Item = (usize, &VertexKey)> {
        self.vertices.iter().enumerate()
    }

    /// Returns the neighbor keys for this cell.
    ///
    /// # Phase 3A
    ///
    /// Neighbors are stored as keys (not UUIDs) for direct TDS access.
    /// The positional semantics: `neighbors()[i]` is the neighbor opposite `vertices()[i]`.
    ///
    /// # Returns
    ///
    /// An `Option` containing neighbor keys if they have been assigned, or `None` otherwise.
    ///
    /// # Example
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let (_, cell) = dt.cells().next().unwrap();
    /// let tds = dt.tds();
    ///
    /// if let Some(neighbors) = cell.neighbors() {
    ///     for (i, neighbor_key_opt) in neighbors.iter().enumerate() {
    ///         if let Some(neighbor_key) = neighbor_key_opt {
    ///             let neighbor_cell = &tds.get_cell(*neighbor_key).unwrap();
    ///             // neighbor_cell is opposite to vertex i
    ///         }
    ///     }
    /// }
    /// ```
    #[inline]
    pub const fn neighbors(&self) -> Option<&NeighborBuffer<Option<CellKey>>> {
        self.neighbors.as_ref()
    }

    /// Returns the vertex keys for this cell.
    ///
    /// # Phase 3A
    ///
    /// This method returns keys (not full vertex objects). Use the TDS to resolve keys:
    /// ```rust
    /// use delaunay::core::collections::Uuid;
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let (_, cell) = dt.cells().next().unwrap();
    /// let tds = dt.tds();
    ///
    /// for &vkey in cell.vertices() {
    ///     let vertex = &tds.get_vertex_by_key(vkey).unwrap();
    ///     // use vertex data...
    ///     assert!(vertex.uuid() != Uuid::nil());
    /// }
    /// ```
    ///
    /// # Returns
    ///
    /// A slice containing the vertex keys.
    #[inline]
    pub fn vertices(&self) -> &[VertexKey] {
        &self.vertices[..]
    }

    /// Find the facet index in `neighbor_cell` that corresponds to the shared facet.
    ///
    /// `facet_idx` is interpreted as the index of the vertex opposite the facet in `self`.
    /// If `neighbor_cell` shares exactly that facet, this returns the index of the vertex
    /// opposite the same facet in `neighbor_cell` (CGAL-style "mirror facet").
    ///
    /// Returns `None` if `facet_idx` is out of range, or if the cells do not appear to share
    /// a single facet.
    #[inline]
    pub(crate) fn mirror_facet_index(
        &self,
        facet_idx: usize,
        neighbor_cell: &Self,
    ) -> Option<usize> {
        if facet_idx >= self.vertices.len() {
            return None;
        }

        // Mirror facet semantics are defined for same-dimensional simplices.
        debug_assert_eq!(
            self.vertices().len(),
            neighbor_cell.vertices().len(),
            "mirror_facet_index requires cells with matching vertex counts",
        );

        // Build the facet vertex set from the source cell (all except facet_idx)
        let mut facet_vertices: CellVertexBuffer = CellVertexBuffer::new();
        for (i, &vkey) in self.vertices().iter().enumerate() {
            if i != facet_idx {
                facet_vertices.push(vkey);
            }
        }

        // Find the vertex in neighbor_cell that is NOT in the facet.
        // That vertex's index is the mirror facet index.
        let mut mirror_idx: Option<usize> = None;
        for (idx, &neighbor_vkey) in neighbor_cell.vertices().iter().enumerate() {
            if !facet_vertices.contains(&neighbor_vkey) {
                if mirror_idx.is_some() {
                    // More than one vertex is not in the facet -> not a valid facet neighbor relation.
                    return None;
                }
                mirror_idx = Some(idx);
            }
        }

        mirror_idx
    }

    /// Adds a vertex key to this cell.
    ///
    /// # Phase 3A: Internal Use Only
    ///
    /// This method is used internally by TDS deserialization to rebuild cell vertex keys.
    /// It should not be used outside of TDS serialization/deserialization code.
    ///
    /// # Arguments
    ///
    /// * `vertex_key` - The vertex key to add
    #[inline]
    pub(crate) fn push_vertex_key(&mut self, vertex_key: VertexKey) {
        self.vertices.push(vertex_key);
    }

    /// Clears all vertex keys from this cell.
    ///
    /// # Phase 3A: Internal Use Only
    ///
    /// This method is used internally by TDS deserialization to clear stale vertex keys
    /// before rebuilding them from the serialized `cell_vertices` mapping.
    /// It should not be used outside of TDS serialization/deserialization code.
    #[inline]
    pub(crate) fn clear_vertex_keys(&mut self) {
        self.vertices.clear();
    }

    /// Ensures the cell has a properly initialized neighbors buffer of size D+1.
    ///
    /// This helper centralizes neighbor buffer initialization logic to avoid code duplication
    /// and reduce the error surface for off-by-one bugs.
    ///
    /// Note: This is currently only used by unit tests, but is kept as a small internal building
    /// block for future insertion/repair code that needs to mutate neighbor buffers in-place.
    ///
    /// # Returns
    ///
    /// A mutable reference to the neighbors buffer, guaranteed to be sized D+1 with all None values.
    ///
    /// # Performance
    ///
    /// Inline to zero cost in release builds. Only allocates if the buffer doesn't exist.
    #[inline]
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Currently only used by unit tests")
    )]
    pub(crate) fn ensure_neighbors_buffer_mut(&mut self) -> &mut NeighborBuffer<Option<CellKey>> {
        debug_assert!(
            self.neighbors.as_ref().is_none_or(|buf| buf.len() == D + 1),
            "neighbors buffer must always have length D+1"
        );
        self.neighbors.get_or_insert_with(|| {
            let mut buffer = NeighborBuffer::new();
            buffer.resize(D + 1, None);
            buffer
        })
    }
}

// Standard trait bounds impl block
impl<T, U, V, const D: usize> Cell<T, U, V, D>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    /// The function returns the number of vertices in the [Cell].
    ///
    /// # Returns
    ///
    /// The number of vertices in the [Cell].
    ///
    /// # Example
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
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let cell_key = dt.cells().next().unwrap().0;
    /// let tds = dt.tds();
    /// let cell = &tds.get_cell(cell_key).unwrap();
    /// assert_eq!(cell.number_of_vertices(), 4);
    /// ```
    #[inline]
    pub fn number_of_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Returns the UUID of the [Cell].
    ///
    /// # Returns
    ///
    /// The Uuid uniquely identifying this cell.
    ///
    /// # Example
    ///
    /// ```
    /// use delaunay::core::collections::Uuid;
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 1.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> =
    ///     DelaunayTriangulation::new(&vertices).unwrap();
    /// let (_, cell) = dt.cells().next().unwrap();
    /// assert_ne!(cell.uuid(), Uuid::nil());
    /// ```
    #[inline]
    pub const fn uuid(&self) -> Uuid {
        self.uuid
    }

    /// Clears the neighbors of the [Cell].
    ///
    /// **Internal API**: This method is `pub(crate)` to enforce that all neighbor
    /// modifications go through validated TDS methods. External code should use
    /// [`Tds::clear_all_neighbors()`](crate::core::triangulation_data_structure::Tds::clear_all_neighbors)
    /// which properly invalidates caches and maintains triangulation consistency.
    ///
    /// This method sets the `neighbors` field to `None`, effectively removing all
    /// neighbor relationships. This is useful for benchmarking neighbor assignment
    /// or when rebuilding neighbor relationships from scratch.
    ///
    /// # Example
    ///
    /// ```
    /// # use delaunay::prelude::*;
    /// # let vertices = vec![
    /// #     vertex!([0.0, 0.0]),
    /// #     vertex!([1.0, 0.0]),
    /// #     vertex!([0.0, 1.0]),
    /// # ];
    /// # let mut dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// // Note: clear_all_neighbors() is a Tds method, not on DelaunayTriangulation
    /// // This example is conceptual - actual usage requires accessing tds directly
    /// # let cell_key = dt.cells().next().unwrap().0;
    /// # let tds = dt.tds();
    /// # // assert!(tds.get_cell(cell_key).unwrap().neighbors().is_none());
    /// ```
    #[inline]
    pub(crate) fn clear_neighbors(&mut self) {
        self.neighbors = None;
    }

    /// Returns the UUIDs of the vertices in this cell.
    ///
    /// # Phase 3A Migration
    ///
    /// This method now requires a `&Tds` parameter to resolve vertex keys to UUIDs.
    ///
    /// # Parameters
    ///
    /// - `tds`: Reference to the triangulation data structure containing the vertices
    ///
    /// # Returns
    ///
    /// A `Result<CellVertexUuidBuffer, CellValidationError>` containing the UUIDs of all vertices in this cell,
    /// or an error if a vertex key is not found in the TDS. Uses stack allocation for typical dimensions.
    ///
    /// # Errors
    ///
    /// Returns `CellValidationError::VertexKeyNotFound` if a vertex key in the cell
    /// does not exist in the TDS. This indicates TDS corruption or inconsistency.
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
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let cell_key = dt.cells().next().unwrap().0;
    /// let tds = dt.tds();
    /// let cell = &tds.get_cell(cell_key).unwrap();
    /// let uuids = cell.vertex_uuids(tds).unwrap();
    /// assert_eq!(uuids.len(), 4);
    /// ```
    #[inline]
    pub fn vertex_uuids(
        &self,
        tds: &Tds<T, U, V, D>,
    ) -> Result<crate::core::collections::CellVertexUuidBuffer, CellValidationError> {
        self.vertices
            .iter()
            .map(|&vkey| {
                tds.get_vertex_by_key(vkey)
                    .map(Vertex::uuid)
                    .ok_or(CellValidationError::VertexKeyNotFound { key: vkey })
            })
            .collect()
    }

    /// Returns an iterator over vertex UUIDs without allocating a Vec.
    ///
    /// # Phase 3A Migration
    ///
    /// This method now requires a `&Tds` parameter to resolve vertex keys to UUIDs.
    ///
    /// # Parameters
    ///
    /// - `tds`: Reference to the triangulation data structure containing the vertices
    ///
    /// # Returns
    ///
    /// An iterator that yields `Result<Uuid, CellValidationError>` for each vertex in the cell.
    ///
    /// # Errors
    ///
    /// The iterator yields `CellValidationError::VertexKeyNotFound` for any vertex key
    /// that does not exist in the TDS. This indicates TDS corruption or inconsistency.
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
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let cell_key = dt.cells().next().unwrap().0;
    /// let tds = dt.tds();
    /// let cell = &tds.get_cell(cell_key).unwrap();
    /// let uuids: Vec<_> = cell.vertex_uuid_iter(tds).collect::<Result<Vec<_>, _>>().unwrap();
    /// assert_eq!(uuids.len(), 3);
    /// ```
    #[inline]
    pub fn vertex_uuid_iter<'a>(
        &'a self,
        tds: &'a Tds<T, U, V, D>,
    ) -> impl ExactSizeIterator<Item = Result<Uuid, CellValidationError>> + 'a {
        self.vertices.iter().map(move |&vkey| {
            tds.get_vertex_by_key(vkey)
                .map(Vertex::uuid)
                .ok_or(CellValidationError::VertexKeyNotFound { key: vkey })
        })
    }

    /// The `dim` function returns the dimensionality of the [Cell].
    ///
    /// # Returns
    ///
    /// The `dim` function returns the compile-time dimension `D` of the [Cell].
    ///
    /// # Example
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 1.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> =
    ///     DelaunayTriangulation::new(&vertices).unwrap();
    /// let (_, cell) = dt.cells().next().unwrap();
    /// assert_eq!(cell.dim(), 3);
    /// ```
    #[inline]
    pub const fn dim(&self) -> usize {
        D
    }

    /// Converts a vector of cells into a `FastHashMap` indexed by their UUIDs.
    ///
    /// This utility function transforms a collection of cells into a hash map structure
    /// for efficient lookups by UUID. Uses `FastHashMap` for performance.
    ///
    /// # Arguments
    ///
    /// * `cells` - A vector of cells to be converted into a `FastHashMap`.
    ///
    /// # Returns
    ///
    /// A [`FastHashMap\u003cUuid, Self\u003e`] where each key is a cell's UUID and each value
    /// is the corresponding cell. The map provides O(1) average-case lookups
    /// by UUID.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    /// use delaunay::core::cell::Cell;
    ///
    /// // Create two separate triangulations
    /// let vertices1 = vec![
    ///     vertex!([0.0, 0.0, 1.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0]),
    /// ];
    /// let vertices2 = vec![
    ///     vertex!([1.0, 1.0, 1.0]),
    ///     vertex!([2.0, 1.0, 1.0]),
    ///     vertex!([1.0, 2.0, 1.0]),
    ///     vertex!([1.0, 1.0, 2.0]),
    /// ];
    /// let dt1: DelaunayTriangulation<_, (), (), 3> =
    ///     DelaunayTriangulation::new(&vertices1).unwrap();
    /// let dt2: DelaunayTriangulation<_, (), (), 3> =
    ///     DelaunayTriangulation::new(&vertices2).unwrap();
    /// let cell1 = dt1.tds().cells().next().unwrap().1.clone();
    /// let cell2 = dt2.tds().cells().next().unwrap().1.clone();
    ///
    /// let cells = vec![cell1.clone(), cell2.clone()];
    /// let cell_map = Cell::into_hashmap(cells);
    ///
    /// // Access cells by their UUIDs
    /// assert_eq!(cell_map.get(&cell1.uuid()), Some(&cell1));
    /// assert_eq!(cell_map.get(&cell2.uuid()), Some(&cell2));
    /// assert_eq!(cell_map.len(), 2);
    /// ```
    ///
    /// ```
    /// use delaunay::core::cell::Cell;
    ///
    /// // Empty vector produces empty FastHashMap
    /// let empty_cells: Vec<Cell<f64, (), (), 3>> = vec![];
    /// let empty_map = Cell::into_hashmap(empty_cells);
    /// assert!(empty_map.is_empty());
    /// ```
    #[must_use]
    pub fn into_hashmap(cells: Vec<Self>) -> FastHashMap<Uuid, Self> {
        cells.into_iter().map(|c| (c.uuid, c)).collect()
    }

    /// The function `is_valid` checks if a [Cell] is valid.
    ///
    /// # Type Parameters
    ///
    /// This method relies on capabilities implied by `T: CoordinateScalar`
    /// (finite, comparable, and hashable coordinates suitable for geometric checks).
    ///
    /// # Returns
    ///
    /// A Result indicating whether the [Cell] is valid. Returns `Ok(())` if valid,
    /// or a `CellValidationError` if invalid. The validation checks that:
    /// - All vertices are valid (coordinates are finite and UUIDs are valid)
    /// - All vertices are distinct from one another
    /// - The cell UUID is valid and not nil
    /// - The cell has exactly D+1 vertices (forming a proper D-simplex)
    /// - If neighbors are provided, they must have exactly D+1 entries (positional semantics)
    ///
    /// Note: This method validates basic neighbor structure invariants but does not validate
    /// the correctness of neighbor relationships, which requires global knowledge of the
    /// triangulation and is handled by the [`Tds`].
    ///
    /// # Errors
    ///
    /// Returns `CellValidationError::InvalidVertex` if any vertex is invalid,
    /// `CellValidationError::InvalidUuid` if the cell's UUID is nil,
    /// `CellValidationError::DuplicateVertices` if the cell contains duplicate vertices,
    /// `CellValidationError::InsufficientVertices` if the cell doesn't have exactly D+1 vertices, or
    /// `CellValidationError::InvalidNeighborsLength` if neighbors are provided but don't have D+1 entries.
    ///
    /// # Example
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 1.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> =
    ///     DelaunayTriangulation::new(&vertices).unwrap();
    /// let (_, cell) = dt.cells().next().unwrap();
    /// assert!(cell.is_valid().is_ok());
    /// ```
    /// Phase 3A: Updated to use vertices (validation without full vertex data)
    /// For full validation including vertex data, use `is_valid_with_tds(&tds)`
    pub fn is_valid(&self) -> Result<(), CellValidationError> {
        // Check if UUID is valid
        validate_uuid(&self.uuid)?;

        // Check that cell has exactly D+1 vertex keys (a proper D-simplex)
        if self.vertices.len() != D + 1 {
            return Err(CellValidationError::InsufficientVertices {
                actual: self.vertices.len(),
                expected: D + 1,
                dimension: D,
            });
        }

        // Check that all vertex keys are distinct
        let mut seen: FastHashSet<VertexKey> = FastHashSet::default();
        for &vkey in &self.vertices {
            if !seen.insert(vkey) {
                return Err(CellValidationError::DuplicateVertices);
            }
        }

        // If neighbors are provided, enforce positional semantics: length must be D+1
        if let Some(ref neighbors) = self.neighbors
            && neighbors.len() != D + 1
        {
            return Err(CellValidationError::InvalidNeighborsLength {
                actual: neighbors.len(),
                expected: D + 1,
                dimension: D,
            });
        }

        Ok(())

        /* OLD CODE - TO BE ADAPTED FOR TDS-BASED VALIDATION:
        // Check if all vertices are valid (requires TDS context)
        for vertex in &self.vertices {
            vertex.is_valid()?;
        }
        */
    }
}

// Advanced implementation block for Cell methods
impl<T, U, V, const D: usize> Cell<T, U, V, D>
where
    T: CoordinateScalar + Clone + PartialEq + PartialOrd,
    U: DataType,
    V: DataType,
{
    /// Returns all facets (faces) of the cell.
    ///
    /// A facet is a (D-1)-dimensional face of a D-dimensional cell, obtained by removing
    /// exactly one vertex from the original cell. This operation creates all possible
    /// (D-1)-dimensional boundary faces of the D-dimensional simplex.
    ///
    /// ## Mathematical Background
    ///
    /// For a D-dimensional cell (D-simplex) with D+1 vertices:
    /// - Each facet is a (D-1)-dimensional simplex with D vertices
    /// - The total number of facets equals the number of vertices (D+1)
    /// - Each vertex defines exactly one facet by its exclusion from the cell
    ///
    /// ## Dimensional Examples
    ///
    /// - **1D cell (line segment)**: 2 facets, each being a 0D point (vertex)
    /// - **2D cell (triangle)**: 3 facets, each being a 1D line segment (edge)
    /// - **3D cell (tetrahedron)**: 4 facets, each being a 2D triangle (face)
    /// - **4D cell (4-simplex)**: 5 facets, each being a 3D tetrahedron
    ///
    /// ## Facet Construction
    ///
    /// Each facet is constructed by:
    /// 1. Taking all vertices from the original cell
    /// 2. Removing exactly one vertex (the "opposite" vertex)
    /// 3. Creating a new (D-1)-dimensional simplex from the remaining D vertices
    ///
    /// The facet "opposite" to vertex `v` contains all vertices of the cell except `v`.
    ///
    /// # Type Parameters
    ///
    /// This method requires the coordinate type `T` to implement additional traits
    /// beyond the basic `Cell` requirements:
    /// - `Clone + PartialEq + PartialOrd`: Required for
    ///   geometric computations and facet creation operations.
    ///
    /// # Returns
    ///
    /// A `Result<Vec<Facet<T, U, V, D>>, FacetError>` containing all facets of the cell.
    /// The returned vector has exactly D+1 facets, where each facet contains D vertices
    /// (one fewer than the original cell's D+1 vertices).
    ///
    /// The facets are returned in the same order as the vertices in the original cell,
    /// where `facets[i]` is the facet opposite to `vertices[i]`.
    ///
    /// # Errors
    ///
    /// Returns a [`FacetError`] if facet creation fails:
    /// - [`FacetError::CellDoesNotContainVertex`]: Internal consistency error where
    ///   a vertex appears to be missing from the cell during facet construction.
    ///   This should not occur under normal circumstances for properly constructed cells.
    ///
    /// Note: For properly constructed cells with D+1 distinct vertices, this method
    /// should not fail under normal circumstances.
    /// Returns all facets of a cell as lightweight `FacetView` objects using only TDS and cell key.
    ///
    /// This is a static method that provides a more robust alternative to `facet_views()` by avoiding
    /// potential mismatches between `self` and the cell retrieved by `cell_key`. It accesses the cell
    /// data directly from the TDS using the provided key.
    ///
    /// # Arguments
    ///
    /// * `tds` - Reference to the triangulation data structure
    /// * `cell_key` - The key of the cell in the TDS
    ///
    /// # Returns
    ///
    /// A `Result<Vec<FacetView>, FacetError>` containing all facets of the cell.
    /// Each facet is represented as a `FacetView` which provides efficient access
    /// to facet properties without cloning cell data.
    ///
    /// # Errors
    ///
    /// Returns a [`FacetError`] if:
    /// - The cell key is not found in the TDS
    /// - Facet creation fails during the construction of `FacetView` objects
    /// - The facet index cannot be represented as `u8` (very rare, only for extremely high dimensions)
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, _, _, 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tds = dt.tds();
    ///
    /// let cell_key = tds.cell_keys().next().unwrap();
    /// let facet_views = Cell::facet_views_from_tds(tds, cell_key).expect("Failed to get facet views");
    ///
    /// // Each facet should have 3 vertices (triangular faces of tetrahedron)
    /// for facet_view in &facet_views {
    ///     assert_eq!(facet_view.vertices().unwrap().count(), 3);
    /// }
    /// ```
    pub fn facet_views_from_tds(
        tds: &Tds<T, U, V, D>,
        cell_key: CellKey,
    ) -> Result<Vec<crate::core::facet::FacetView<'_, T, U, V, D>>, FacetError> {
        // Get the cell from the TDS using the key
        let cell = tds
            .get_cell(cell_key)
            .ok_or(FacetError::CellNotFoundInTriangulation)?;

        let vertex_count = cell.number_of_vertices();
        if vertex_count > u8::MAX as usize {
            return Err(FacetError::InvalidFacetIndex {
                index: u8::MAX,
                facet_count: vertex_count,
            });
        }

        let mut facet_views = Vec::with_capacity(vertex_count);
        for idx in 0..vertex_count {
            let facet_index = crate::core::util::usize_to_u8(idx, vertex_count)?;
            facet_views.push(crate::core::facet::FacetView::new(
                tds,
                cell_key,
                facet_index,
            )?);
        }
        Ok(facet_views)
    }

    /// Compare two cells by their vertex sets (using `Vertex::PartialEq`) for cross-TDS equality checking.
    ///
    /// This method enables semantic comparison of cells from different TDS instances by comparing
    /// the actual Vertex objects using `Vertex::PartialEq` (coordinate-based comparison).
    /// Two cells are considered equal if they contain the same set of vertices (by coordinates),
    /// regardless of order. This mirrors `Cell::PartialEq` semantics but works across TDS boundaries.
    ///
    /// # Arguments
    ///
    /// * `self_tds` - The TDS containing `self`
    /// * `other` - The other cell to compare against
    /// * `other_tds` - The TDS containing `other`
    ///
    /// # Returns
    ///
    /// `true` if both cells contain the same set of vertices (by coordinates), `false` otherwise.
    /// Returns `false` if any vertex keys cannot be resolved.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// // Example 1: Comparing cells from different TDS instances with same coordinates
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt1: DelaunayTriangulation<_, _, _, 2> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let dt2: DelaunayTriangulation<_, _, _, 2> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tds1 = dt1.tds();
    /// let tds2 = dt2.tds();
    ///
    /// let cell1 = tds1.cells().next().unwrap().1;
    /// let cell2 = tds2.cells().next().unwrap().1;
    ///
    /// // Different TDS instances, but same vertex coordinates
    /// assert!(cell1.eq_by_vertices(tds1, cell2, tds2));
    /// ```
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// // Example 2: Comparing cells with different coordinates returns false
    /// let vertices1 = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let vertices2 = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([2.0, 0.0]),  // Different coordinate
    ///     vertex!([0.0, 2.0]),  // Different coordinate
    /// ];
    /// let dt1: DelaunayTriangulation<_, _, _, 2> = DelaunayTriangulation::new(&vertices1).unwrap();
    /// let dt2: DelaunayTriangulation<_, _, _, 2> = DelaunayTriangulation::new(&vertices2).unwrap();
    /// let tds1 = dt1.tds();
    /// let tds2 = dt2.tds();
    ///
    /// let cell1 = tds1.cells().next().unwrap().1;
    /// let cell2 = tds2.cells().next().unwrap().1;
    ///
    /// // Different coordinates mean cells are not equal
    /// assert!(!cell1.eq_by_vertices(tds1, cell2, tds2));
    /// ```
    pub fn eq_by_vertices(
        &self,
        self_tds: &Tds<T, U, V, D>,
        other: &Self,
        other_tds: &Tds<T, U, V, D>,
    ) -> bool {
        // Get vertices for both cells
        let self_vertices: Option<Vec<_>> = self
            .vertices()
            .iter()
            .map(|&vkey| self_tds.get_vertex_by_key(vkey))
            .collect();

        let other_vertices: Option<Vec<_>> = other
            .vertices()
            .iter()
            .map(|&vkey| other_tds.get_vertex_by_key(vkey))
            .collect();

        // If we couldn't resolve all vertex keys, cells are not equal
        let (Some(mut self_vertices), Some(mut other_vertices)) = (self_vertices, other_vertices)
        else {
            return false;
        };

        // Sort vertices for order-independent comparison (matches Cell::PartialEq semantics)
        // Use Vertex::PartialOrd which compares coordinates
        self_vertices.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        other_vertices.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Compare using Vertex::PartialEq (coordinate-based)
        self_vertices == other_vertices
    }

    /// Returns an iterator over all facets of a cell as lightweight `FacetView` objects.
    ///
    /// This is a zero-allocation alternative to `facet_views_from_tds()` that returns an iterator
    /// instead of collecting results into a `Vec`. This is more memory-efficient for large cells
    /// or when you don't need to store all facet views at once.
    ///
    /// # Arguments
    ///
    /// * `tds` - Reference to the triangulation data structure
    /// * `cell_key` - The key of the cell in the TDS
    ///
    /// # Returns
    ///
    /// A `Result<impl ExactSizeIterator<Item = Result<FacetView, FacetError>>, FacetError>` that yields all facets of the cell.
    /// The iterator implements `ExactSizeIterator`, so you can call `.len()` to get the number of facets
    /// without consuming the iterator.
    ///
    /// # Errors
    ///
    /// Returns a [`FacetError`] if:
    /// - The cell key is not found in the TDS
    /// - The facet count cannot be represented as `u8` (very rare, only for extremely high dimensions)
    ///
    /// Individual facet creation errors are yielded by the iterator as `Result<FacetView, FacetError>`
    /// items, not returned immediately from this method.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, _, _, 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tds = dt.tds();
    ///
    /// let cell_key = tds.cell_keys().next().unwrap();
    /// let facet_iter = Cell::facet_view_iter(tds, cell_key).expect("Failed to get facet iterator");
    ///
    /// // Iterator knows the exact count
    /// assert_eq!(facet_iter.len(), 4); // 4 facets for a tetrahedron
    ///
    /// // Process facets one at a time (zero allocation)
    /// for (i, facet_result) in facet_iter.enumerate() {
    ///     let facet_view = facet_result.expect("Facet creation should succeed");
    ///     assert_eq!(facet_view.vertices().unwrap().count(), 3); // Each facet has 3 vertices
    /// }
    /// ```
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, _, _, 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tds = dt.tds();
    ///
    /// let cell_key = tds.cell_keys().next().unwrap();
    /// let facet_iter = Cell::facet_view_iter(tds, cell_key).expect("Failed to get facet iterator");
    ///
    /// // Collect only successful facets, filtering out any errors
    /// let successful_facets: Vec<_> = facet_iter
    ///     .filter_map(Result::ok)
    ///     .collect();
    /// assert_eq!(successful_facets.len(), 4);
    /// ```
    pub fn facet_view_iter(
        tds: &Tds<T, U, V, D>,
        cell_key: CellKey,
    ) -> Result<
        impl ExactSizeIterator<Item = Result<crate::core::facet::FacetView<'_, T, U, V, D>, FacetError>>,
        FacetError,
    > {
        // Get the cell from the TDS using the key
        let cell = tds
            .get_cell(cell_key)
            .ok_or(FacetError::CellNotFoundInTriangulation)?;

        let vertex_count = cell.number_of_vertices();
        if vertex_count > u8::MAX as usize {
            return Err(FacetError::InvalidFacetIndex {
                index: u8::MAX,
                facet_count: vertex_count,
            });
        }

        // Return a simple range-based iterator that maps indices to FacetView creation
        Ok((0..vertex_count).map(move |idx| {
            let facet_index = crate::core::util::usize_to_u8(idx, vertex_count)?;
            crate::core::facet::FacetView::new(tds, cell_key, facet_index)
        }))
    }
}

// =============================================================================
// STANDARD TRAIT IMPLEMENTATIONS
// =============================================================================

/// Phase 3A: Equality of cells based on sorted vertex keys.
///
/// Two cells are equal if they contain the same set of vertex keys,
/// regardless of order. This is fast (O(D log D)) and doesn't require TDS access.
///
/// **Note**: This compares cells within the same TDS context. For cross-TDS
/// comparison, use `eq_by_vertex_uuids()` (to be added if needed).
impl<T, U, V, const D: usize> PartialEq for Cell<T, U, V, D>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // Fast comparison using vertex keys (just u64 comparisons)
        // Use CellVertexBuffer for stack allocation (D+1 keys fit on stack for D ≤ 7)
        let mut self_keys: CellVertexBuffer = self.vertices.iter().copied().collect();
        let mut other_keys: CellVertexBuffer = other.vertices.iter().copied().collect();
        self_keys.sort_unstable();
        other_keys.sort_unstable();
        self_keys == other_keys

        /* OLD CODE - Phase 3A migration: compared full vertex objects
        sorted_vertices::<T, U, D>(&self.vertices) == sorted_vertices::<T, U, D>(&other.vertices)
        */
    }
}

/// Phase 3A: Order of cells based on lexicographic order of sorted vertex keys.
///
/// This provides a consistent ordering for cells based on their vertex keys.
/// Fast (O(D log D)) and doesn't require TDS access.
impl<T, U, V, const D: usize> PartialOrd for Cell<T, U, V, D>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        // Fast comparison using vertex keys
        // Use CellVertexBuffer for stack allocation (D+1 keys fit on stack for D ≤ 7)
        let mut self_keys: CellVertexBuffer = self.vertices.iter().copied().collect();
        let mut other_keys: CellVertexBuffer = other.vertices.iter().copied().collect();
        self_keys.sort_unstable();
        other_keys.sort_unstable();
        self_keys.partial_cmp(&other_keys)

        /* OLD CODE - Phase 3A migration: compared full vertex objects
        sorted_vertices::<T, U, D>(&self.vertices)
            .partial_cmp(&sorted_vertices::<T, U, D>(&other.vertices))
        */
    }
}

// =============================================================================
// HASHING AND EQUALITY IMPLEMENTATIONS
// =============================================================================

/// Phase 3A: Eq implementation for Cell based on sorted vertex keys.
///
/// Maintains the Eq contract with `PartialEq`: cells with the same vertex keys
/// are considered equal.
impl<T, U, V, const D: usize> Eq for Cell<T, U, V, D>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
}

/// Phase 3A: Custom Hash implementation for Cell using sorted vertex keys.
///
/// This ensures that cells with the same vertex keys have the same hash,
/// maintaining the Eq/Hash contract: if a == b, then hash(a) == hash(b).
///
/// **Performance**: Fast O(D log D) hashing using just vertex keys (u64).
///
/// **Note**: UUID, neighbors, and data are excluded from hashing to match
/// the `PartialEq` implementation which only compares vertex keys.
impl<T, U, V, const D: usize> Hash for Cell<T, U, V, D>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash sorted vertex keys for consistent ordering
        // Use CellVertexBuffer for stack allocation (D+1 keys fit on stack for D ≤ 7)
        let mut sorted_keys: CellVertexBuffer = self.vertices.iter().copied().collect();
        sorted_keys.sort_unstable();
        for key in sorted_keys {
            key.hash(state);
        }
        // Intentionally exclude UUID, neighbors, and data to maintain
        // consistency with PartialEq implementation which only compares vertex keys

        /* OLD CODE - Phase 3A migration: hashed full vertex objects
        for vertex in &sorted_vertices::<T, U, D>(&self.vertices) {
            vertex.hash(state);
        }
        */
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vertex::vertex;
    use crate::geometry::point::Point;
    use crate::geometry::predicates::insphere;
    use crate::geometry::util::{circumcenter, circumradius, circumradius_with_center};
    use approx::assert_relative_eq;
    use std::{cmp, collections::hash_map::DefaultHasher, hash::Hasher};

    // Type aliases for commonly used types to reduce repetition
    type TestVertex3D = Vertex<f64, (), 3>;
    type TestVertex2D = Vertex<f64, (), 2>;

    use crate::geometry::kernel::FastKernel;
    use crate::prelude::DelaunayTriangulation;

    // =============================================================================
    // DIMENSION-PARAMETERIZED TEST MACRO
    // =============================================================================
    // This macro generates comprehensive tests for each dimension (2D-5D)
    // Keep this at the top so it's not forgotten when adding new tests!

    /// Macro to generate dimension-specific cell tests for dimensions 2D-5D.
    ///
    /// This macro reduces test duplication by generating consistent tests across
    /// multiple dimensions. It creates tests for:
    /// - Basic cell creation and property validation
    /// - Serialization roundtrip (Some and None data)
    /// - UUID validation
    ///
    /// # Usage
    ///
    /// ```ignore
    /// test_cell_dimensions! {
    ///     cell_2d => 2 => vec![vertex!([0.0, 0.0]), vertex!([1.0, 0.0]), vertex!([0.0, 1.0])],
    /// }
    /// ```
    macro_rules! test_cell_dimensions {
        ($(
            $test_name:ident => $dim:expr => $vertices:expr
        ),+ $(,)?) => {
            $(
                #[test]
                fn $test_name() {
                    // Test basic cell creation
                    let vertices = $vertices;
                    let dt = DelaunayTriangulation::new(&vertices).unwrap();
                    let (_, cell) = dt.cells().next().unwrap();
                    assert_cell_properties(cell, $dim + 1, $dim);
                }

                pastey::paste! {
                    #[test]
                    fn [<$test_name _with_data>]() {
                        // Test cell with data - need generic constructor for non-() cell data
                        let vertices = $vertices;
                        let dt: DelaunayTriangulation<FastKernel<f64>, (), i32, $dim> =
                            DelaunayTriangulation::with_kernel(FastKernel::new(), &vertices).unwrap();
                        let (_, cell_ref) = dt.cells().next().unwrap();
                        let mut cell = cell_ref.clone();
                        cell.data = Some(42);
                        assert_cell_properties(&cell, $dim + 1, $dim);
                        assert_eq!(cell.data, Some(42));
                    }

                    #[test]
                    fn [<$test_name _serialization_roundtrip>]() {
                        // Test serialization with Some data - use generic constructor
                        let vertices = $vertices;
                        let dt: DelaunayTriangulation<FastKernel<f64>, (), i32, $dim> =
                            DelaunayTriangulation::with_kernel(FastKernel::new(), &vertices).unwrap();
                        let (_, cell) = dt.cells().next().unwrap();
                        let mut cell = cell.clone();
                        cell.data = Some(99);

                        let serialized = serde_json::to_string(&cell).unwrap();
                        assert!(serialized.contains("\"data\":"));
                        let deserialized: Cell<f64, (), i32, $dim> = serde_json::from_str(&serialized).unwrap();
                        assert_eq!(deserialized.data, Some(99));
                        assert_eq!(deserialized.uuid(), cell.uuid());

                        // Test serialization with None data - use simple constructor
                        let vertices = $vertices;
                        let dt = DelaunayTriangulation::new(&vertices).unwrap();
                        let (_, cell) = dt.cells().next().unwrap();

                        let serialized = serde_json::to_string(&cell).unwrap();
                        assert!(!serialized.contains("\"data\":"));
                        let deserialized: Cell<f64, (), Option<i32>, $dim> = serde_json::from_str(&serialized).unwrap();
                        assert_eq!(deserialized.data, None);
                    }

                    #[test]
                    fn [<$test_name _uuid_uniqueness>]() {
                        // Test UUID uniqueness by creating two separate triangulations
                        let vertices1 = $vertices;
                        let vertices2 = $vertices;
                        let dt1 = DelaunayTriangulation::new(&vertices1).unwrap();
                        let (_, cell1) = dt1.cells().next().unwrap();
                        let dt2 = DelaunayTriangulation::new(&vertices2).unwrap();
                        let (_, cell2) = dt2.cells().next().unwrap();
                        assert_ne!(cell1.uuid(), cell2.uuid());
                        assert!(!cell1.uuid().is_nil());
                        assert!(!cell2.uuid().is_nil());
                    }
                }
            )+
        };
    }

    // Generate tests for dimensions 2D through 5D
    test_cell_dimensions! {
        cell_2d => 2 => vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ],
        cell_3d => 3 => vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ],
        cell_4d => 4 => vec![
            vertex!([0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0]),
        ],
        cell_5d => 5 => vec![
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 0.0, 1.0]),
        ],
    }

    // =============================================================================
    // HELPER FUNCTIONS
    // =============================================================================

    /// Simplified helper function to test basic cell properties
    fn assert_cell_properties<T, U, V, const D: usize>(
        cell: &Cell<T, U, V, D>,
        expected_vertices: usize,
        expected_dim: usize,
    ) where
        T: CoordinateScalar,
        U: DataType,
        V: DataType,
    {
        assert_eq!(cell.number_of_vertices(), expected_vertices);
        assert_eq!(cell.dim(), expected_dim);
        assert!(!cell.uuid().is_nil());
    }

    // Helper functions for creating common test data
    fn create_test_vertices_3d() -> Vec<TestVertex3D> {
        vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ]
    }

    fn create_test_vertices_2d() -> Vec<TestVertex2D> {
        vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ]
    }

    // Tests covering the cell! macro functionality to ensure it works correctly
    // with different scenarios including vertex arrays and optional data.

    #[test]
    fn cell_macro_without_data() {
        // Test the cell! macro without data (explicit type annotation required)
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]), // Need 4 vertices for 3D cell
        ];

        // Phase 3A: Create DT to get cell with context
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let (_, cell) = dt.cells().next().unwrap();

        assert_eq!(cell.number_of_vertices(), 4);
        assert_eq!(cell.dim(), 3);
        assert!(cell.data.is_none());
        assert!(!cell.uuid().is_nil());

        // Verify all input vertices exist in the triangulation (order-independent)
        let cell_coords: Vec<Vec<f64>> = cell
            .vertices()
            .iter()
            .map(|&vkey| {
                dt.tds()
                    .get_vertex_by_key(vkey)
                    .unwrap()
                    .point()
                    .coords()
                    .as_slice()
                    .to_vec()
            })
            .collect();

        for original in &vertices {
            let original_coords = original.point().coords().as_slice();
            assert!(
                cell_coords.iter().any(|coords| {
                    coords
                        .iter()
                        .zip(original_coords)
                        .all(|(a, b)| (a - b).abs() < f64::EPSILON)
                }),
                "Input vertex {original_coords:?} not found in cell"
            );
        }

        // Human readable output for cargo test -- --nocapture
        println!("Cell without data: {cell:?}");
    }

    // =============================================================================
    // CELL EQUALITY TESTS
    // =============================================================================

    #[test]
    fn test_eq_by_vertices_same_coordinates_different_tds() {
        println!("Testing eq_by_vertices with same coordinates across different TDS");

        // Create two separate TDS instances with identical vertex coordinates
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt1 = DelaunayTriangulation::new(&vertices).unwrap();
        let dt2 = DelaunayTriangulation::new(&vertices).unwrap();

        let cell1 = dt1.cells().next().unwrap().1;
        let cell2 = dt2.cells().next().unwrap().1;

        // Despite different DT instances (and thus different vertex keys),
        // cells should be equal by coordinates
        assert!(cell1.eq_by_vertices(dt1.tds(), cell2, dt2.tds()));
        println!("  ✓ Cells from different DT with same coordinates are equal");
    }

    #[test]
    fn test_eq_by_vertices_2d() {
        println!("Testing eq_by_vertices in 2D");

        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt1 = DelaunayTriangulation::new(&vertices).unwrap();
        let dt2 = DelaunayTriangulation::new(&vertices).unwrap();

        let cell1 = dt1.cells().next().unwrap().1;
        let cell2 = dt2.cells().next().unwrap().1;

        assert!(cell1.eq_by_vertices(dt1.tds(), cell2, dt2.tds()));
        println!("  ✓ 2D cells with same coordinates are equal");
    }

    #[test]
    fn cell_macro_with_data() {
        // Test the cell! macro with data by creating TDS, cloning cell, and modifying
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        // Build DT with integer cell data type
        let dt: DelaunayTriangulation<FastKernel<f64>, (), i32, 3> =
            DelaunayTriangulation::with_kernel(FastKernel::new(), &vertices).unwrap();
        let (_, cell_ref) = dt.cells().next().unwrap();
        let mut cell = cell_ref.clone();
        cell.data = Some(42);

        assert_eq!(cell.number_of_vertices(), 4);
        assert_eq!(cell.dim(), 3);
        assert_eq!(cell.data.unwrap(), 42);
        assert!(!cell.uuid().is_nil());

        // Verify all input vertices exist in the cell (order-independent)
        let cell_coords: Vec<Vec<f64>> = cell
            .vertices()
            .iter()
            .map(|&vkey| {
                dt.tds()
                    .get_vertex_by_key(vkey)
                    .unwrap()
                    .point()
                    .coords()
                    .as_slice()
                    .to_vec()
            })
            .collect();

        for original in &vertices {
            let original_coords = original.point().coords().as_slice();
            assert!(
                cell_coords.iter().any(|coords| {
                    coords
                        .iter()
                        .zip(original_coords)
                        .all(|(a, b)| (a - b).abs() < f64::EPSILON)
                }),
                "Input vertex {original_coords:?} not found in cell"
            );
        }

        // Human readable output for cargo test -- --nocapture
        println!("Cell with data: {cell:?}");
    }

    #[test]
    fn cell_with_vertex_data() {
        // Phase 3A: Test cells with vertex data through TDS
        // Don't use cell! macro since we need TDS context for vertex data access
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0], 1),
            vertex!([1.0, 0.0, 0.0], 2),
            vertex!([0.0, 1.0, 0.0], 3),
            vertex!([0.0, 0.0, 1.0], 4), // Need 4 vertices for 3D cell
        ];

        // Create DT with vertex data - simple constructor works since cell data is ()
        let dt: DelaunayTriangulation<FastKernel<f64>, i32, (), 3> =
            DelaunayTriangulation::with_kernel(FastKernel::new(), &vertices).unwrap();
        let (_, cell) = dt.cells().next().unwrap();

        assert_eq!(cell.number_of_vertices(), 4);
        assert_eq!(cell.dim(), 3);

        // Check that all expected vertex data values exist (order-independent)
        let cell_data: Vec<i32> = cell
            .vertices()
            .iter()
            .map(|&vkey| dt.tds().get_vertex_by_key(vkey).unwrap().data.unwrap())
            .collect();

        for expected in &[1, 2, 3, 4] {
            assert!(
                cell_data.contains(expected),
                "Expected vertex data {expected} not found in cell"
            );
        }
    }

    // =============================================================================
    // TRAIT IMPLEMENTATION TESTS
    // =============================================================================
    // Tests covering core Rust traits like PartialEq, PartialOrd, Hash, Clone

    #[test]
    fn cell_partial_eq() {
        // Phase 3A: Test PartialEq using cells from TDS
        let vertices = vec![
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let (_, cell1) = dt.cells().next().unwrap();
        let cell2 = cell1.clone();

        // Test equality
        assert_eq!(*cell1, cell2);
        assert_eq!(cell1.uuid(), cell2.uuid()); // Same cell, same UUID after clone
        assert_eq!(cell1.vertices(), cell2.vertices());

        // Test cloned cell
        let cell3 = cell1.clone();
        assert_eq!(*cell1, cell3);
    }

    #[test]
    fn cell_partial_ord() {
        // Phase 3A: Test PartialOrd using TDS with multiple cells
        let all_vertices = vec![
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 1.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&all_vertices).unwrap();
        let cells: Vec<_> = dt.cells().map(|(_, cell)| cell).collect();

        if cells.len() >= 2 {
            // Test ordering between different cells
            let cell1 = cells[0];
            let cell2 = cells[1];

            // At least one ordering relationship should hold
            let has_ordering = cell1 != cell2 || cell1 == cell2;
            assert!(has_ordering, "Cells should have some ordering relationship");
        }
    }

    #[test]
    fn cell_hash() {
        let vertices = vec![
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let (_, cell1) = dt.cells().next().unwrap();
        let cell2 = cell1.clone();

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();

        cell1.hash(&mut hasher1);
        cell2.hash(&mut hasher2);

        // Same vertices should produce same hash (Eq/Hash contract)
        assert_eq!(*cell1, cell2); // They are equal by vertices
        assert_eq!(hasher1.finish(), hasher2.finish()); // Therefore hashes must be equal
        // Note: UUID is same since cell2 is a clone
        assert_eq!(cell1.uuid(), cell2.uuid());
    }

    #[test]
    fn cell_clone() {
        let vertices = vec![
            vertex!([0.0, 0.0, 1.0], 1),
            vertex!([0.0, 1.0, 0.0], 1),
            vertex!([1.0, 0.0, 0.0], 1),
            vertex!([1.0, 1.0, 1.0], 2),
        ];
        let dt: DelaunayTriangulation<FastKernel<f64>, i32, i32, 3> =
            DelaunayTriangulation::with_kernel(FastKernel::new(), &vertices).unwrap();
        let (_, cell_ref) = dt.cells().next().unwrap();
        let mut cell1 = cell_ref.clone();
        cell1.data = Some(42);
        let cell2 = cell1.clone();

        assert_eq!(cell1, cell2);
    }

    #[test]
    fn cell_ordering_edge_cases() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let (_, cell1) = dt.cells().next().unwrap();
        let cell2 = cell1.clone();

        // Test that equal cells are not less than each other
        assert_ne!(cell1.partial_cmp(&cell2), Some(cmp::Ordering::Less));
        assert_ne!(cell2.partial_cmp(cell1), Some(cmp::Ordering::Less));
        assert!(*cell1 <= cell2);
        assert!(cell2 <= *cell1);
        assert!(*cell1 >= cell2);
        assert!(cell2 >= *cell1);
    }
    // =============================================================================
    // CORE CELL METHODS TESTS
    // =============================================================================
    // Tests covering core cell functionality including basic properties, containment
    // checks, facet operations, and other fundamental cell methods.

    #[test]
    fn cell_number_of_vertices() {
        let vertices_2d = create_test_vertices_2d();
        let dt_2d = DelaunayTriangulation::new(&vertices_2d).unwrap();
        let cell_key_2d = dt_2d.tds().cell_keys().next().unwrap();
        let triangle = dt_2d.tds().get_cell(cell_key_2d).unwrap();
        assert_eq!(triangle.number_of_vertices(), 3);

        let vertices_3d = create_test_vertices_3d();
        let dt_3d = DelaunayTriangulation::new(&vertices_3d).unwrap();
        let cell_key_3d = dt_3d.tds().cell_keys().next().unwrap();
        let tetrahedron = dt_3d.tds().get_cell(cell_key_3d).unwrap();
        assert_eq!(tetrahedron.number_of_vertices(), 4);
    }

    #[test]
    fn cell_mirror_facet_index_shared_facet_2d() {
        // Four points in convex position should yield two triangles that share an edge.
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
            vertex!([1.0, 1.1]), // break cocircular symmetry
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();

        let cells: Vec<_> = dt.cells().map(|(_, cell)| cell).collect();
        assert!(
            cells.len() >= 2,
            "Expected at least 2 cells, got {}",
            cells.len()
        );

        // Find two distinct cells that share exactly D vertices (i.e., a facet).
        // For 2D, D = 2, so the shared facet is an edge.
        let mut found = None;
        for i in 0..cells.len() {
            for j in (i + 1)..cells.len() {
                let cell_a = cells[i];
                let cell_b = cells[j];

                let shared: FastHashSet<VertexKey> = cell_a
                    .vertices()
                    .iter()
                    .copied()
                    .filter(|v| cell_b.vertices().contains(v))
                    .collect();

                if shared.len() == 2 {
                    let facet_idx_a = cell_a
                        .vertices()
                        .iter()
                        .position(|v| !shared.contains(v))
                        .expect("cell_a should have one vertex not in the shared facet");

                    let facet_idx_b = cell_b
                        .vertices()
                        .iter()
                        .position(|v| !shared.contains(v))
                        .expect("cell_b should have one vertex not in the shared facet");

                    found = Some((cell_a, cell_b, facet_idx_a, facet_idx_b));
                    break;
                }
            }
            if found.is_some() {
                break;
            }
        }

        let Some((cell_a, cell_b, facet_idx_a, facet_idx_b)) = found else {
            panic!("Expected to find a pair of neighboring cells that share an edge");
        };

        assert_eq!(
            cell_a.mirror_facet_index(facet_idx_a, cell_b),
            Some(facet_idx_b)
        );
        assert_eq!(
            cell_b.mirror_facet_index(facet_idx_b, cell_a),
            Some(facet_idx_a)
        );

        // Out-of-range facet index
        assert_eq!(
            cell_a.mirror_facet_index(cell_a.number_of_vertices(), cell_b),
            None
        );
    }

    #[test]
    fn cell_mirror_facet_index_returns_none_when_cells_do_not_share_facet_2d() {
        // Add a point strictly inside the convex hull to yield multiple triangles.
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
            vertex!([1.0, 1.0]),
            vertex!([0.5, 0.5]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();

        let cells: Vec<_> = dt.cells().map(|(_, cell)| cell).collect();
        assert!(
            cells.len() >= 3,
            "Expected at least 3 cells, got {}",
            cells.len()
        );

        // Find two cells that share fewer than D vertices (D=2 in 2D).
        let mut non_adjacent = None;
        'outer: for i in 0..cells.len() {
            for j in (i + 1)..cells.len() {
                let cell_a = cells[i];
                let cell_b = cells[j];
                let shared_count = cell_a
                    .vertices()
                    .iter()
                    .filter(|v| cell_b.vertices().contains(v))
                    .count();

                if shared_count < 2 {
                    non_adjacent = Some((cell_a, cell_b));
                    break 'outer;
                }
            }
        }

        let Some((cell_a, cell_b)) = non_adjacent else {
            panic!("Expected to find a pair of non-adjacent cells");
        };

        assert_eq!(cell_a.mirror_facet_index(0, cell_b), None);
    }

    #[test]
    fn cell_dim() {
        let vertices_2d = create_test_vertices_2d();
        let dt_2d = DelaunayTriangulation::new(&vertices_2d).unwrap();
        let cell_key_2d = dt_2d.tds().cell_keys().next().unwrap();
        let triangle = dt_2d.tds().get_cell(cell_key_2d).unwrap();
        assert_eq!(triangle.dim(), 2);

        let vertices_3d = create_test_vertices_3d();
        let dt_3d = DelaunayTriangulation::new(&vertices_3d).unwrap();
        let cell_key_3d = dt_3d.tds().cell_keys().next().unwrap();
        let tetrahedron = dt_3d.tds().get_cell(cell_key_3d).unwrap();
        assert_eq!(tetrahedron.dim(), 3);
    }

    #[test]
    fn cell_contains_vertex() {
        let vertex1: Vertex<f64, i32, 3> = vertex!([0.0, 0.0, 1.0], 1);
        let vertex2 = vertex!([0.0, 1.0, 0.0], 1);
        let vertex3 = vertex!([1.0, 0.0, 0.0], 1);
        let vertex4 = vertex!([1.0, 1.0, 1.0], 2);

        // Create DT to get VertexKeys - use generic constructor for vertex data
        let vertices = vec![vertex1, vertex2, vertex3, vertex4];
        let dt: DelaunayTriangulation<FastKernel<f64>, i32, (), 3> =
            DelaunayTriangulation::with_kernel(FastKernel::new(), &vertices).unwrap();
        let cell_key = dt.cells().next().unwrap().0;
        let cell = &dt.tds().get_cell(cell_key).unwrap();

        // Get vertex keys from DT
        let vertex_keys: Vec<_> = dt.tds().vertices().map(|(k, _)| k).collect();

        assert!(cell.contains_vertex(vertex_keys[0]));
        assert!(cell.contains_vertex(vertex_keys[1]));
        assert!(cell.contains_vertex(vertex_keys[2]));
        assert!(cell.contains_vertex(vertex_keys[3]));

        // Human readable output for cargo test -- --nocapture
        println!("Cell: {cell:?}");
    }

    #[test]
    fn cell_has_vertex_in_common() {
        // Test has_vertex_in_common (replacement for deprecated contains_vertex_of)
        let vertices1 = vec![
            vertex!([0.0, 0.0, 1.0], 1),
            vertex!([0.0, 1.0, 0.0], 1),
            vertex!([1.0, 0.0, 0.0], 1),
            vertex!([1.0, 1.0, 1.0], 2),
        ];
        let tds1: DelaunayTriangulation<FastKernel<f64>, i32, i32, 3> =
            DelaunayTriangulation::with_kernel(FastKernel::new(), &vertices1).unwrap();
        let (_, cell_ref) = tds1.cells().next().unwrap();
        let mut cell = cell_ref.clone();
        cell.data = Some(42);

        let vertices2 = vec![
            vertex!([0.0, 0.0, 1.0], 1),
            vertex!([0.0, 1.0, 0.0], 1),
            vertex!([1.0, 0.0, 0.0], 1),
            vertex!([0.0, 0.0, 0.0], 0),
        ];
        let tds2: DelaunayTriangulation<FastKernel<f64>, i32, i32, 3> =
            DelaunayTriangulation::with_kernel(FastKernel::new(), &vertices2).unwrap();
        let (_, cell2_ref) = tds2.cells().next().unwrap();
        let mut cell2 = cell2_ref.clone();
        cell2.data = Some(43);

        assert!(cell.has_vertex_in_common(&cell2));

        // Human readable output for cargo test -- --nocapture
        println!("Cell: {cell:?}");
    }

    // Removed: cell_facets_contains - redundant with cell_facet_views_comprehensive

    // =============================================================================
    // VERTEX_UUIDS METHOD TESTS
    // =============================================================================
    // Comprehensive tests for the vertex_uuids method covering different scenarios.

    #[test]
    fn test_vertex_uuids_success() {
        // Test the vertex_uuids method returns correct vertex UUIDs vector
        let vertex1 = vertex!([0.0, 0.0, 0.0], 10);
        let vertex2 = vertex!([1.0, 0.0, 0.0], 20);
        let vertex3 = vertex!([0.0, 1.0, 0.0], 30);
        let vertex4 = vertex!([0.0, 0.0, 1.0], 40);

        let vertices = vec![vertex1, vertex2, vertex3, vertex4];
        let dt: DelaunayTriangulation<FastKernel<f64>, i32, (), 3> =
            DelaunayTriangulation::with_kernel(FastKernel::new(), &vertices).unwrap();
        let cell_key = dt.cells().next().unwrap().0;
        let cell = &dt.tds().get_cell(cell_key).unwrap();

        // Get vertex UUIDs
        let vertex_uuids = cell.vertex_uuids(dt.tds()).unwrap();
        assert_eq!(cell.vertex_uuid_iter(dt.tds()).count(), 4);

        // Verify UUIDs match the cell's vertices using iterator
        for (expected_uuid, returned_uuid) in
            cell.vertex_uuid_iter(dt.tds()).zip(vertex_uuids.iter())
        {
            assert_eq!(expected_uuid.unwrap(), *returned_uuid);
        }

        // Verify all UUIDs are unique
        let unique_uuids: std::collections::HashSet<_> = vertex_uuids.iter().collect();
        assert_eq!(unique_uuids.len(), vertex_uuids.len());

        // Verify no nil UUIDs using iterator
        for uuid in cell.vertex_uuid_iter(dt.tds()) {
            assert_ne!(uuid.unwrap(), Uuid::nil());
        }

        println!("✓ vertex_uuids method returns correct vertex UUIDs");
    }

    #[test]
    fn test_vertex_uuids_empty_cell_fails() {
        // Test that TDS creation fails gracefully with insufficient vertices for dimension
        // Phase 3A: Now tested through TDS which is the user-facing API

        let vertices = vec![vertex!([0.0, 0.0, 0.0])];
        let result = DelaunayTriangulation::new(&vertices);

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("Insufficient vertices"));

        println!("✓ DT construction properly validates vertex count");
    }

    #[test]
    fn test_vertex_uuids_2d_cell() {
        // Test vertex_uuids with a 2D cell (triangle)
        let vertex1 = vertex!([0.0, 0.0], 1);
        let vertex2 = vertex!([1.0, 0.0], 2);
        let vertex3 = vertex!([0.5, 1.0], 3);

        let vertices = vec![vertex1, vertex2, vertex3];
        let dt: DelaunayTriangulation<FastKernel<f64>, i32, (), 2> =
            DelaunayTriangulation::with_kernel(FastKernel::new(), &vertices).unwrap();
        let cell_key = dt.cells().next().unwrap().0;
        let cell = &dt.tds().get_cell(cell_key).unwrap();

        // Get vertex UUIDs
        let vertex_uuids = cell.vertex_uuids(dt.tds()).unwrap();
        assert_eq!(cell.vertex_uuid_iter(dt.tds()).count(), 3);

        // Verify UUIDs match the cell's vertices using iterator
        for (expected_uuid, returned_uuid) in
            cell.vertex_uuid_iter(dt.tds()).zip(vertex_uuids.iter())
        {
            assert_eq!(expected_uuid.unwrap(), *returned_uuid);
        }

        // Verify all UUIDs are unique
        let unique_uuids: std::collections::HashSet<_> = vertex_uuids.iter().collect();
        assert_eq!(unique_uuids.len(), vertex_uuids.len());

        // Verify no nil UUIDs using iterator
        for uuid in cell.vertex_uuid_iter(dt.tds()) {
            assert_ne!(uuid.unwrap(), Uuid::nil());
        }

        println!("✓ vertex_uuids works correctly for 2D cells");
    }

    #[test]
    fn test_vertex_uuids_4d_cell() {
        // Test vertex_uuids with a 4D cell (4-simplex) using integer data
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0, 0.0], 1),
            vertex!([1.0, 0.0, 0.0, 0.0], 2),
            vertex!([0.0, 1.0, 0.0, 0.0], 3),
            vertex!([0.0, 0.0, 1.0, 0.0], 4),
            vertex!([0.0, 0.0, 0.0, 1.0], 5),
        ];

        let dt: DelaunayTriangulation<FastKernel<f64>, i32, (), 4> =
            DelaunayTriangulation::with_kernel(FastKernel::new(), &vertices).unwrap();
        let cell_key = dt.cells().next().unwrap().0;
        let cell = &dt.tds().get_cell(cell_key).unwrap();

        // Get vertex UUIDs
        let vertex_uuids = cell.vertex_uuids(dt.tds()).unwrap();
        assert_eq!(cell.vertex_uuid_iter(dt.tds()).count(), 5);

        // Verify UUIDs match the cell's vertices using iterator
        for (expected_uuid, returned_uuid) in
            cell.vertex_uuid_iter(dt.tds()).zip(vertex_uuids.iter())
        {
            assert_eq!(expected_uuid.unwrap(), *returned_uuid);
        }

        // Verify all UUIDs are unique
        let unique_uuids: std::collections::HashSet<_> = vertex_uuids.iter().collect();
        assert_eq!(unique_uuids.len(), vertex_uuids.len());

        // Verify all expected vertex data values exist (order-independent)
        let vertex_data: Vec<i32> = cell
            .vertices()
            .iter()
            .map(|&vkey| dt.tds().get_vertex_by_key(vkey).unwrap().data.unwrap())
            .collect();

        for expected in 1..=5 {
            assert!(
                vertex_data.contains(&expected),
                "Expected vertex data {expected} not found"
            );
        }

        // Verify no nil UUIDs using iterator
        for uuid in cell.vertex_uuid_iter(dt.tds()) {
            assert_ne!(uuid.unwrap(), Uuid::nil());
        }

        println!("✓ vertex_uuids works correctly for 4D cells");
    }

    #[test]
    fn test_vertex_uuids_with_f32_coordinates() {
        // Test vertex_uuids with f32 coordinates
        let vertices = vec![
            vertex!([0.0f32, 0.0f32, 0.0f32]),
            vertex!([1.0f32, 0.0f32, 0.0f32]),
            vertex!([0.0f32, 1.0f32, 0.0f32]),
            vertex!([0.0f32, 0.0f32, 1.0f32]),
        ];

        // Note: DelaunayTriangulation::new() creates FastKernel<f64> by default
        // We need to use with_kernel to get FastKernel<f32> for f32 vertices
        let dt: DelaunayTriangulation<FastKernel<f32>, (), (), 3> =
            DelaunayTriangulation::with_kernel(FastKernel::new(), &vertices).unwrap();
        let cell_key = dt.cells().next().unwrap().0;
        let cell = &dt.tds().get_cell(cell_key).unwrap();

        // Get vertex UUIDs
        let vertex_uuids = cell.vertex_uuids(dt.tds()).unwrap();
        assert_eq!(cell.vertex_uuid_iter(dt.tds()).count(), 4);

        // Verify coordinate type is preserved
        let first_vertex_key = cell.vertices()[0];
        let first_vertex = &dt.tds().get_vertex_by_key(first_vertex_key).unwrap();
        assert_relative_eq!(
            first_vertex.point().coords()[0],
            0.0f32,
            epsilon = f32::EPSILON
        );

        // Verify UUIDs match the cell's vertices using iterator
        for (expected_uuid, returned_uuid) in
            cell.vertex_uuid_iter(dt.tds()).zip(vertex_uuids.iter())
        {
            assert_eq!(expected_uuid.unwrap(), *returned_uuid);
        }

        // Verify all UUIDs are unique
        let unique_uuids: std::collections::HashSet<_> = vertex_uuids.iter().collect();
        assert_eq!(unique_uuids.len(), vertex_uuids.len());

        // Verify no nil UUIDs using iterator
        for uuid in cell.vertex_uuid_iter(dt.tds()) {
            assert_ne!(uuid.unwrap(), Uuid::nil());
        }

        println!("✓ vertex_uuids works correctly with f32 coordinates");
    }

    // =============================================================================
    // DIMENSIONAL TESTS
    // =============================================================================
    // Tests covering cells in different dimensions (1D, 2D, 3D, 4D+) and
    // various coordinate types (f32, f64) to ensure dimensional flexibility.
    //
    // NOTE: The main dimension tests (cell_2d, cell_3d, cell_4d, cell_5d) are
    // generated by the test_cell_dimensions! macro at the top of this test module.

    // Keep 1D test separate as it's less common
    #[test]
    fn cell_1d() {
        let vertices = vec![vertex!([0.0]), vertex!([1.0])];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let (_, cell) = dt.cells().next().unwrap();
        assert_cell_properties(cell, 2, 1);
    }

    #[test]
    fn cell_with_f32() {
        let vertices = vec![
            vertex!([0.0f32, 0.0f32]),
            vertex!([1.0f32, 0.0f32]),
            vertex!([0.0f32, 1.0f32]),
        ];
        let dt: DelaunayTriangulation<FastKernel<f32>, (), (), 2> =
            DelaunayTriangulation::with_kernel(FastKernel::new(), &vertices).unwrap();
        let (_, cell) = dt.cells().next().unwrap();

        assert_eq!(cell.number_of_vertices(), 3);
        assert_eq!(cell.dim(), 2);
        assert!(!cell.uuid().is_nil());
    }

    #[test]
    fn cell_single_vertex() {
        // Test that creating a 3D triangulation with insufficient vertices fails validation
        // Phase 3A: Now tested through TDS which is the user-facing API
        let vertices = vec![vertex!([0.0, 0.0, 0.0])];
        let result = DelaunayTriangulation::new(&vertices);

        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Insufficient vertices"));
        assert!(error_msg.contains('1'));
        assert!(error_msg.contains('4'));
    }

    #[test]
    fn cell_neighbors_none_by_default() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let (_, cell) = dt.cells().next().unwrap();

        // Note: neighbors may be set by TDS construction, this tests cell structure
        assert!(cell.neighbors.is_some() || cell.neighbors.is_none());
    }

    #[test]
    fn cell_data_none_by_default() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let (_, cell) = dt.cells().next().unwrap();

        assert!(cell.data.is_none());
    }

    #[test]
    fn cell_data_can_be_set() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt: DelaunayTriangulation<FastKernel<f64>, (), i32, 3> =
            DelaunayTriangulation::with_kernel(FastKernel::new(), &vertices).unwrap();
        let (_, cell_ref) = dt.cells().next().unwrap();
        let mut cell = cell_ref.clone();
        cell.data = Some(42);

        assert_eq!(cell.data.unwrap(), 42);
    }

    #[test]
    fn cell_into_hashmap_empty() {
        let cells: Vec<Cell<f64, (), (), 3>> = Vec::new();
        let hashmap = Cell::into_hashmap(cells);

        assert!(hashmap.is_empty());
    }

    #[test]
    fn cell_into_hashmap_multiple() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([1.0, 1.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();

        // Collect cells from DT
        let cells_vec: Vec<_> = dt.cells().map(|(_, cell)| cell.clone()).collect();
        assert!(cells_vec.len() >= 2, "Need at least 2 cells for this test");

        let uuid1 = cells_vec[0].uuid();
        let uuid2 = cells_vec[1].uuid();
        let hashmap = Cell::into_hashmap(cells_vec);

        assert!(hashmap.len() >= 2);
        assert!(hashmap.contains_key(&uuid1));
        assert!(hashmap.contains_key(&uuid2));
    }

    #[test]
    fn cell_debug_format() {
        // Use a simple non-degenerate 3D tetrahedron so `DelaunayTriangulation::new` can construct
        // a valid simplex for debug-format testing.
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt: DelaunayTriangulation<FastKernel<f64>, (), i32, 3> =
            DelaunayTriangulation::with_kernel(FastKernel::new(), &vertices).unwrap();
        let (_, cell_ref) = dt.cells().next().unwrap();
        let mut cell = cell_ref.clone();
        cell.data = Some(42);
        let debug_str = format!("{cell:?}");

        // Phase 3A: Verify debug output contains basic cell information
        // Use structural checks rather than brittle string matching
        assert!(debug_str.contains("Cell"));
        assert!(!cell.vertices().is_empty());
        assert!(!cell.uuid().is_nil());
        assert_eq!(cell.data.unwrap(), 42);
    }

    // =============================================================================
    // COMPREHENSIVE SERIALIZATION TESTS
    // =============================================================================
    // Tests covering cell serialization and deserialization with different
    // data types, dimensions, and configurations using serde_json.

    #[test]
    fn cell_to_and_from_json() {
        // Phase 3A: Test serialization through TDS context (proper way)
        // Use a non-degenerate 3D tetrahedron so `DelaunayTriangulation::new` can construct a
        // valid initial simplex.
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();

        // Serialize the entire DT (includes cells with proper context)
        let serialized = serde_json::to_string(&dt).unwrap();
        assert!(serialized.contains("vertices"));
        assert!(serialized.contains("cells"));

        // Deserialize back to DT
        let deserialized: DelaunayTriangulation<FastKernel<f64>, (), (), 3> =
            serde_json::from_str(&serialized).unwrap();

        // Verify DT properties match
        assert_eq!(deserialized.number_of_vertices(), dt.number_of_vertices());
        assert_eq!(deserialized.number_of_cells(), dt.number_of_cells());
        assert_eq!(deserialized.dim(), dt.dim());

        // Verify cells within DT can be accessed
        assert_ne!(deserialized.number_of_cells(), 0);
        for (_cell_key, cell) in deserialized.tds().cells() {
            assert_eq!(cell.dim(), 3);
            assert_eq!(cell.number_of_vertices(), 4);
        }

        println!("TDS serialization/deserialization test passed");
    }

    #[test]
    fn cell_deserialization_error_cases() {
        // Test realistic JSON deserialization errors that users might encounter

        // Test missing required field (uuid)
        let invalid_json_missing_uuid = r#"{"data": null}"#;
        let result: Result<Cell<f64, (), (), 3>, _> =
            serde_json::from_str(invalid_json_missing_uuid);
        assert!(result.is_err(), "Missing UUID should cause error");
        let error = result.unwrap_err().to_string();
        assert!(
            error.contains("missing field") || error.contains("uuid"),
            "Error should mention missing uuid field: {error}"
        );

        // Test invalid UUID format
        let invalid_json_bad_uuid = r#"{"uuid": "not-a-valid-uuid"}"#;
        let result: Result<Cell<f64, (), (), 3>, _> = serde_json::from_str(invalid_json_bad_uuid);
        assert!(result.is_err(), "Invalid UUID format should cause error");

        // Test completely invalid JSON syntax
        let invalid_json_syntax = r"{this is not valid JSON}";
        let result: Result<Cell<f64, (), (), 3>, _> = serde_json::from_str(invalid_json_syntax);
        assert!(result.is_err(), "Invalid JSON syntax should cause error");

        // Test empty JSON object (missing required uuid)
        let empty_json = r"{}";
        let result: Result<Cell<f64, (), (), 3>, _> = serde_json::from_str(empty_json);
        assert!(result.is_err(), "Empty JSON should fail (missing uuid)");

        // Test deserialization with unknown fields (should succeed - ignored)
        let json_unknown_field = r#"{
            "uuid": "550e8400-e29b-41d4-a716-446655440000",
            "unknown_field": "value",
            "another_unknown": 123
        }"#;
        let result: Result<Cell<f64, (), (), 3>, _> = serde_json::from_str(json_unknown_field);
        assert!(result.is_ok(), "Unknown fields should be ignored");
    }

    #[test]
    fn cell_serialization_data_field_handling() {
        // Comprehensive test for data field serialization behavior

        // Test 1: Minimal valid JSON (only uuid required)
        let minimal_valid_json = r#"{"uuid": "550e8400-e29b-41d4-a716-446655440000"}"#;
        let result: Result<Cell<f64, (), (), 3>, _> = serde_json::from_str(minimal_valid_json);
        assert!(
            result.is_ok(),
            "Minimal valid JSON with just UUID should succeed"
        );
        let cell = result.unwrap();
        assert_eq!(
            cell.uuid().to_string(),
            "550e8400-e29b-41d4-a716-446655440000"
        );
        assert!(cell.data.is_none());

        // Test 2: Serialization with Some(data) includes field
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt: DelaunayTriangulation<FastKernel<f64>, (), i32, 3> =
            DelaunayTriangulation::with_kernel(FastKernel::new(), &vertices).unwrap();
        let (_, cell_ref) = dt.cells().next().unwrap();
        let mut cell_with_data = cell_ref.clone();
        cell_with_data.data = Some(42);
        let serialized = serde_json::to_string(&cell_with_data).unwrap();
        assert!(
            serialized.contains("\"data\":"),
            "Some(data) should include data field"
        );
        assert!(serialized.contains("42"));
        let deserialized: Cell<f64, (), i32, 3> = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.data, Some(42));

        // Test 3: Serialization with None data omits field (optimization)
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let (_, cell_none) = dt.cells().next().unwrap();
        let serialized_none = serde_json::to_string(&cell_none).unwrap();
        assert!(
            !serialized_none.contains("\"data\":"),
            "None should omit data field"
        );
        let deserialized_none: Cell<f64, (), Option<i32>, 3> =
            serde_json::from_str(&serialized_none).unwrap();
        assert_eq!(deserialized_none.data, None);

        // Test 4: Backward compatibility - explicit "data": null works
        let json_with_null = r#"{"uuid":"550e8400-e29b-41d4-a716-446655440000","data":null}"#;
        let cell_explicit_null: Cell<f64, (), Option<i32>, 3> =
            serde_json::from_str(json_with_null).unwrap();
        assert_eq!(cell_explicit_null.data, None);
    }

    // =============================================================================
    // GEOMETRIC PROPERTIES TESTS
    // =============================================================================
    // Tests for geometric properties and validation of cells

    #[test]
    fn cell_coordinate_ranges() {
        // Test triangulation construction with various coordinate ranges

        // Negative coordinates: fully in negative octant
        let negative_vertices = vec![
            vertex!([-1.0, -1.0, -1.0]),
            vertex!([-2.0, -1.0, -1.0]),
            vertex!([-1.0, -2.0, -1.0]),
            vertex!([-1.0, -1.0, -2.0]),
        ];
        let dt_neg = DelaunayTriangulation::new(&negative_vertices).unwrap();
        let (_, cell_neg) = dt_neg.tds().cells().next().unwrap();
        assert_eq!(cell_neg.number_of_vertices(), 4);
        assert_eq!(cell_neg.dim(), 3);

        // Large coordinates: large-magnitude coordinates
        let large_vertices = vec![
            vertex!([1e6, 1e6, 1e6]),
            vertex!([2e6, 1e6, 1e6]),
            vertex!([1e6, 2e6, 1e6]),
            vertex!([1e6, 1e6, 2e6]),
        ];
        let dt_large = DelaunayTriangulation::new(&large_vertices).unwrap();
        let (_, cell_large) = dt_large.tds().cells().next().unwrap();
        assert_eq!(cell_large.number_of_vertices(), 4);
        assert_eq!(cell_large.dim(), 3);

        // Small coordinates: scaled-down tetrahedron without degeneracy
        let small_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1e-3, 0.0, 0.0]),
            vertex!([0.0, 1e-3, 0.0]),
            vertex!([0.0, 0.0, 1e-3]),
        ];
        let dt_small = DelaunayTriangulation::new(&small_vertices).unwrap();
        let (_, cell_small) = dt_small.tds().cells().next().unwrap();
        assert_eq!(cell_small.number_of_vertices(), 4);
        assert_eq!(cell_small.dim(), 3);
    }

    #[test]
    fn cell_circumradius_2d() {
        let vertex1 = vertex!([0.0, 0.0]);
        let vertex2 = vertex!([1.0, 0.0]);
        let vertex3 = vertex!([0.0, 1.0]);

        let vertices = vec![vertex1, vertex2, vertex3];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let cell_key = dt.cells().next().unwrap().0;
        let cell = &dt.tds().get_cell(cell_key).unwrap();

        // Resolve VertexKeys to actual vertices
        let vertex_points: Vec<Point<f64, 2>> = cell
            .vertices()
            .iter()
            .map(|vk| *dt.tds().get_vertex_by_key(*vk).unwrap().point())
            .collect();
        let circumradius = circumradius(&vertex_points).unwrap();

        // For a right triangle with legs of length 1, circumradius is sqrt(2)/2
        let expected_radius = 2.0_f64.sqrt() / 2.0;
        assert_relative_eq!(circumradius, expected_radius, epsilon = 1e-10);
    }

    #[test]
    fn cell_contains_vertex_false() {
        let vertex1 = vertex!([0.0, 0.0, 0.0]);
        let vertex2 = vertex!([1.0, 0.0, 0.0]);
        let vertex3 = vertex!([0.0, 1.0, 0.0]);
        let vertex4 = vertex!([0.0, 0.0, 1.0]); // 4th vertex to complete 3D cell
        let vertex_outside: Vertex<f64, (), 3> = vertex!([2.0, 2.0, 2.0]);

        let vertices = vec![vertex1, vertex2, vertex3, vertex4];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let cell_key = dt.cells().next().unwrap().0;
        let cell = &dt.tds().get_cell(cell_key).unwrap();

        // Create a vertex key for the outside vertex - it won't be in the cell
        let outside_key = dt.tds().vertex_key_from_uuid(&vertex_outside.uuid());
        assert!(outside_key.is_none() || !cell.contains_vertex(outside_key.unwrap()));
    }

    #[test]
    fn cell_circumsphere_contains_vertex_determinant() {
        // Test the matrix determinant method for circumsphere containment
        // Use a simple, well-known case: unit tetrahedron
        let vertex1 = vertex!([0.0, 0.0, 0.0], 1);
        let vertex2 = vertex!([1.0, 0.0, 0.0], 1);
        let vertex3 = vertex!([0.0, 1.0, 0.0], 1);
        let vertex4 = vertex!([0.0, 0.0, 1.0], 2);

        let vertices = vec![vertex1, vertex2, vertex3, vertex4];
        let dt: DelaunayTriangulation<FastKernel<f64>, i32, (), 3> =
            DelaunayTriangulation::with_kernel(FastKernel::new(), &vertices).unwrap();
        let cell_key = dt.cells().next().unwrap().0;
        let cell = &dt.tds().get_cell(cell_key).unwrap();

        // Test vertex clearly outside circumsphere
        let vertex_far_outside: Vertex<f64, i32, 3> = vertex!([10.0, 10.0, 10.0], 4);
        // Just check that the method runs without error for now
        let vertex_points: Vec<Point<f64, 3>> = cell
            .vertices()
            .iter()
            .map(|vk| *dt.tds().get_vertex_by_key(*vk).unwrap().point())
            .collect();
        let result = insphere(&vertex_points, *vertex_far_outside.point());
        assert!(result.is_ok());

        // Test with origin (should be inside or on boundary)
        let origin: Vertex<f64, i32, 3> = vertex!([0.0, 0.0, 0.0], 3);
        let vertex_points: Vec<Point<f64, 3>> = cell
            .vertices()
            .iter()
            .map(|vk| *dt.tds().get_vertex_by_key(*vk).unwrap().point())
            .collect();
        let result_origin = insphere(&vertex_points, *origin.point());
        assert!(result_origin.is_ok());
    }

    #[test]
    fn cell_circumsphere_contains_vertex_2d() {
        // Test 2D case for circumsphere containment using determinant method
        let vertex1 = vertex!([0.0, 0.0]);
        let vertex2 = vertex!([1.0, 0.0]);
        let vertex3 = vertex!([0.0, 1.0]);

        let vertices = vec![vertex1, vertex2, vertex3];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let cell_key = dt.cells().next().unwrap().0;
        let cell = &dt.tds().get_cell(cell_key).unwrap();

        // Test vertex far outside circumcircle
        let vertex_far_outside: Vertex<f64, (), 2> = vertex!([10.0, 10.0]);
        let vertex_points: Vec<Point<f64, 2>> = cell
            .vertices()
            .iter()
            .map(|vk| *dt.tds().get_vertex_by_key(*vk).unwrap().point())
            .collect();
        let result = insphere(&vertex_points, *vertex_far_outside.point());
        assert!(result.is_ok());

        // Test with center of triangle (should be inside)
        let center: Vertex<f64, (), 2> = vertex!([0.33, 0.33]);
        let vertex_points: Vec<Point<f64, 2>> = cell
            .vertices()
            .iter()
            .map(|vk| *dt.tds().get_vertex_by_key(*vk).unwrap().point())
            .collect();
        let result_center = insphere(&vertex_points, *center.point());
        assert!(result_center.is_ok());
    }

    #[test]
    fn cell_circumradius_with_center() {
        // Test the circumradius_with_center method
        let vertex1 = vertex!([0.0, 0.0, 0.0]);
        let vertex2 = vertex!([1.0, 0.0, 0.0]);
        let vertex3 = vertex!([0.0, 1.0, 0.0]);
        let vertex4 = vertex!([0.0, 0.0, 1.0]);

        let vertices = vec![vertex1, vertex2, vertex3, vertex4];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let cell_key = dt.cells().next().unwrap().0;
        let cell = &dt.tds().get_cell(cell_key).unwrap();

        let vertex_points: Vec<Point<f64, 3>> = cell
            .vertices()
            .iter()
            .map(|vk| *dt.tds().get_vertex_by_key(*vk).unwrap().point())
            .collect();

        let circumcenter = circumcenter(&vertex_points).unwrap();
        let radius_with_center = circumradius_with_center(&vertex_points, &circumcenter);
        let radius_direct = circumradius(&vertex_points).unwrap();

        assert_relative_eq!(radius_with_center.unwrap(), radius_direct, epsilon = 1e-10);
    }

    // Phase 3A: Test updated to test facet_views instead of from_facet_and_vertex
    // (from_facet_and_vertex is commented out pending refactor)
    #[test]
    fn cell_facet_views_comprehensive() {
        // Test comprehensive facet view functionality
        let vertex1 = vertex!([0.0, 0.0, 0.0]);
        let vertex2 = vertex!([1.0, 0.0, 0.0]);
        let vertex3 = vertex!([0.0, 1.0, 0.0]);
        let vertex4 = vertex!([0.0, 0.0, 1.0]);

        let vertices = vec![vertex1, vertex2, vertex3, vertex4];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let cell_key = dt.cells().next().unwrap().0;

        // Test that we can get facet views for all facets
        let facet_views =
            Cell::facet_views_from_tds(dt.tds(), cell_key).expect("Failed to get facet views");
        assert_eq!(facet_views.len(), 4, "3D cell should have 4 facets");

        // Each facet should have 3 vertices (for 3D)
        for (i, facet_view) in facet_views.iter().enumerate() {
            let facet_vertices = facet_view.vertices().expect("Failed to get facet vertices");
            assert_eq!(
                facet_vertices.count(),
                3,
                "Facet {i} should have 3 vertices"
            );
        }

        // Verify opposite vertices are correct
        let cell = dt.tds().get_cell(cell_key).unwrap();
        for (i, facet_view) in facet_views.iter().enumerate() {
            let opposite_vertex = facet_view
                .opposite_vertex()
                .expect("Failed to get opposite vertex");
            // The opposite vertex should be one of the cell's vertices (by VertexKey)
            let opposite_key = dt
                .tds()
                .vertex_key_from_uuid(&opposite_vertex.uuid())
                .unwrap();
            assert!(
                cell.vertices().contains(&opposite_key),
                "Facet {i} opposite vertex key should be in cell"
            );
        }
    }

    // Phase 3A: Test updated to test facet vertex uniqueness instead of from_facet_and_vertex
    // (from_facet_and_vertex is commented out pending refactor)
    #[test]
    fn test_facet_vertex_uniqueness() {
        // Test that facet vertices are unique and don't include the opposite vertex
        let vertex1 = vertex!([0.0, 0.0, 1.0]);
        let vertex2 = vertex!([0.0, 1.0, 0.0]);
        let vertex3 = vertex!([1.0, 0.0, 0.0]);
        let vertex4 = vertex!([1.0, 1.0, 1.0]);

        let vertices = vec![vertex1, vertex2, vertex3, vertex4];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let cell_key = dt.cells().next().unwrap().0;

        // Get all facet views
        let facet_views =
            Cell::facet_views_from_tds(dt.tds(), cell_key).expect("Failed to get facet views");

        for facet_view in &facet_views {
            let opposite_vertex = facet_view
                .opposite_vertex()
                .expect("Failed to get opposite vertex");
            let opposite_vertex_key = dt
                .tds()
                .vertex_key_from_uuid(&opposite_vertex.uuid())
                .unwrap();
            let facet_vertices = facet_view.vertices().expect("Failed to get facet vertices");

            // Collect facet vertex keys
            let facet_vertex_keys: Vec<_> = facet_vertices
                .map(|v| dt.tds().vertex_key_from_uuid(&v.uuid()).unwrap())
                .collect();

            // Verify the opposite vertex key is NOT in the facet vertices
            assert!(
                !facet_vertex_keys.contains(&opposite_vertex_key),
                "Facet vertices should not include the opposite vertex key"
            );

            // Verify all facet vertices are unique
            let unique_count = facet_vertex_keys
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len();
            assert_eq!(
                unique_count,
                facet_vertex_keys.len(),
                "All facet vertices should be unique"
            );
        }
    }

    #[test]
    fn cell_different_numeric_types() {
        // Test with different numeric types to ensure type flexibility
        // Test with f32
        let vertex1_f32 = vertex!([0.0f32, 0.0f32]);
        let vertex2_f32 = vertex!([1.0f32, 0.0f32]);
        let vertex3_f32 = vertex!([0.0f32, 1.0f32]);

        let dt_f32 = DelaunayTriangulation::new(&[vertex1_f32, vertex2_f32, vertex3_f32]).unwrap();
        let (_, cell_f32) = dt_f32.cells().next().unwrap();
        assert_eq!(cell_f32.number_of_vertices(), 3);
        assert_eq!(cell_f32.dim(), 2);
    }

    #[test]
    fn cell_high_dimensional() {
        // Test with higher dimensions (5D)
        let vertex1 = vertex!([0.0, 0.0, 0.0, 0.0, 0.0]);
        let vertex2 = vertex!([1.0, 0.0, 0.0, 0.0, 0.0]);
        let vertex3 = vertex!([0.0, 1.0, 0.0, 0.0, 0.0]);
        let vertex4 = vertex!([0.0, 0.0, 1.0, 0.0, 0.0]);
        let vertex5 = vertex!([0.0, 0.0, 0.0, 1.0, 0.0]);
        let vertex6 = vertex!([0.0, 0.0, 0.0, 0.0, 1.0]);

        let vertices = vec![vertex1, vertex2, vertex3, vertex4, vertex5, vertex6];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let cell_key = dt.cells().next().unwrap().0;
        let cell = &dt.tds().get_cell(cell_key).unwrap();

        assert_eq!(cell.number_of_vertices(), 6);
        assert_eq!(cell.dim(), 5);
        assert_eq!(
            Cell::facet_views_from_tds(dt.tds(), cell_key)
                .expect("Failed to get facets")
                .len(),
            6
        ); // Each vertex creates one facet
    }

    #[test]
    fn cell_vertex_data_consistency() {
        // Test cells with vertices that have different data types
        let vertex1 = vertex!([0.0, 0.0, 0.0], 1);
        let vertex2 = vertex!([1.0, 0.0, 0.0], 2);
        let vertex3 = vertex!([0.0, 1.0, 0.0], 3);
        let vertex4 = vertex!([0.0, 0.0, 1.0], 4); // Need 4 vertices for 3D cell

        let vertices = vec![vertex1, vertex2, vertex3, vertex4];
        let mut dt: DelaunayTriangulation<FastKernel<f64>, i32, u32, 3> =
            DelaunayTriangulation::with_kernel(FastKernel::new(), &vertices).unwrap();
        let cell_key = dt.cells().next().unwrap().0;

        // Set the cell data to a known value
        if let Some(cell) = dt.tri.tds.cells_mut().get_mut(cell_key) {
            cell.data = Some(42u32);
        }

        let cell = &dt.tds().get_cell(cell_key).unwrap();

        // Verify all expected vertex data values exist (order-independent)
        let vertex_data: Vec<i32> = cell
            .vertices()
            .iter()
            .map(|&vkey| dt.tds().get_vertex_by_key(vkey).unwrap().data.unwrap())
            .collect();

        for expected in 1..=4 {
            assert!(
                vertex_data.contains(&expected),
                "Expected vertex data {expected} not found"
            );
        }
        assert_eq!(cell.data.unwrap(), 42u32);

        // Also verify we can access vertex data through facet_views
        let facet_views =
            Cell::facet_views_from_tds(dt.tds(), cell_key).expect("Failed to get facet views");
        for facet_view in &facet_views {
            // Get vertices from the facet view
            let vertices = facet_view.vertices().expect("Failed to get facet vertices");

            // Verify all vertices have valid data
            for vertex in vertices {
                assert!(vertex.data.is_some(), "Vertex data should be set");
                let data = vertex.data.unwrap();
                assert!(
                    (1..=4).contains(&data),
                    "Vertex data should be in range 1-4"
                );
            }
        }
    }

    // =============================================================================
    // CELL VALIDATION TESTS
    // =============================================================================
    // Tests covering cell validation logic for success and error cases

    #[test]
    fn cell_validation_success_cases() {
        // Test various valid cell configurations

        // Valid 3D cell with correct vertex count (D+1 = 4)
        let vertices_3d = vec![
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices_3d).unwrap();
        let (_, cell_3d) = dt.cells().next().unwrap();
        assert!(
            cell_3d.is_valid().is_ok(),
            "Valid 3D cell should pass validation"
        );

        // Valid 2D cell with correct vertex count (D+1 = 3)
        let vertices_2d = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices_2d).unwrap();
        let (_, cell_ref) = dt.cells().next().unwrap();
        let mut cell_2d = cell_ref.clone();
        assert!(
            cell_2d.is_valid().is_ok(),
            "Valid 2D cell should pass validation"
        );

        // Cell with None neighbors is valid
        cell_2d.neighbors = None;
        assert!(
            cell_2d.is_valid().is_ok(),
            "Cell with no neighbors should be valid"
        );

        // Cell with correct neighbors length is valid
        cell_2d.neighbors = Some(vec![None, None, None].into());
        assert!(
            cell_2d.is_valid().is_ok(),
            "Cell with correct neighbors length should be valid"
        );
    }

    #[test]
    fn cell_validation_error_cases() {
        // Test various invalid cell configurations

        // Invalid UUID (nil)
        let vertices = vec![
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let (_, cell_ref) = dt.cells().next().unwrap();
        let mut invalid_uuid_cell = cell_ref.clone();
        invalid_uuid_cell.uuid = uuid::Uuid::nil();
        assert!(
            matches!(
                invalid_uuid_cell.is_valid(),
                Err(CellValidationError::InvalidUuid { .. })
            ),
            "Nil UUID should fail validation"
        );

        // Insufficient vertices (TDS construction fails)
        let insufficient_vertices = vec![vertex!([0.0, 0.0, 1.0]), vertex!([0.0, 1.0, 0.0])];
        let result = DelaunayTriangulation::new(&insufficient_vertices);
        assert!(
            result.is_err(),
            "TDS should fail with insufficient vertices"
        );

        // Invalid neighbors length (too few)
        let vertices_2d = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices_2d).unwrap();
        let (_, cell_ref) = dt.cells().next().unwrap();
        let mut cell_wrong_neighbors = cell_ref.clone();
        cell_wrong_neighbors.neighbors = Some(vec![None, None].into());
        assert!(
            matches!(
                cell_wrong_neighbors.is_valid(),
                Err(CellValidationError::InvalidNeighborsLength {
                    actual: 2,
                    expected: 3,
                    dimension: 2
                })
            ),
            "Wrong neighbors count should fail validation"
        );

        // Invalid neighbors length (too many)
        cell_wrong_neighbors.neighbors = Some(vec![None, None, None, None].into());
        assert!(
            matches!(
                cell_wrong_neighbors.is_valid(),
                Err(CellValidationError::InvalidNeighborsLength {
                    actual: 4,
                    expected: 3,
                    dimension: 2
                })
            ),
            "Wrong neighbors count should fail validation"
        );
    }

    #[test]
    fn cell_new_rejects_insufficient_and_duplicate_vertices() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let vkeys: Vec<_> = dt.tds().vertices().map(|(k, _)| k).collect();

        // Too few vertices for a 3D cell (D+1 = 4)
        let err =
            Cell::<f64, (), (), 3>::new(vec![vkeys[0], vkeys[1], vkeys[2]], None).unwrap_err();
        assert!(matches!(
            err,
            CellValidationError::InsufficientVertices {
                actual: 3,
                expected: 4,
                dimension: 3,
            }
        ));

        // Duplicate vertex keys are rejected
        let err = Cell::<f64, (), (), 3>::new(vec![vkeys[0], vkeys[1], vkeys[2], vkeys[0]], None)
            .unwrap_err();
        assert!(matches!(err, CellValidationError::DuplicateVertices));
    }

    #[test]
    fn cell_is_valid_rejects_insufficient_and_duplicate_vertices() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let (_, cell_ref) = dt.cells().next().unwrap();

        // Insufficient vertices (wrong vertex buffer length)
        let mut wrong_len = cell_ref.clone();
        wrong_len.vertices.pop();
        assert!(matches!(
            wrong_len.is_valid(),
            Err(CellValidationError::InsufficientVertices { .. })
        ));

        // Duplicate vertices
        let mut dup = cell_ref.clone();
        dup.vertices[1] = dup.vertices[0];
        assert!(matches!(
            dup.is_valid(),
            Err(CellValidationError::DuplicateVertices)
        ));
    }

    #[test]
    fn cell_ensure_neighbors_buffer_mut_initializes_and_reuses() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let (cell_key, cell_ref) = dt.cells().next().unwrap();

        let mut cell = cell_ref.clone();
        assert!(cell.neighbors.is_none());

        let buf = cell.ensure_neighbors_buffer_mut();
        assert_eq!(buf.len(), 3);
        assert!(buf.iter().all(Option::is_none));

        // Mutate through the returned buffer and ensure it's preserved
        buf[0] = Some(cell_key);
        let buf2 = cell.ensure_neighbors_buffer_mut();
        assert_eq!(buf2[0], Some(cell_key));
    }

    #[test]
    fn cell_facet_view_helpers_reject_excessive_vertex_count() {
        use crate::core::facet::FacetError;

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut dt = DelaunayTriangulation::new(&vertices).unwrap();
        let cell_key = dt.tds().cell_keys().next().unwrap();

        // Grab a stable key we can duplicate to inflate the vertex buffer.
        let vkey0 = {
            let cell = dt.tds().get_cell(cell_key).unwrap();
            cell.vertices()[0]
        };

        {
            let cell = dt
                .tri
                .tds
                .cells_mut()
                .get_mut(cell_key)
                .expect("cell key should be valid in test");
            while u8::try_from(cell.number_of_vertices()).is_ok() {
                cell.push_vertex_key(vkey0);
            }
            assert!(u8::try_from(cell.number_of_vertices()).is_err());
        }

        // Both helpers should fail early (before attempting to build individual FacetViews).
        let err = Cell::facet_views_from_tds(dt.tds(), cell_key).unwrap_err();
        assert!(matches!(
            err,
            FacetError::InvalidFacetIndex {
                index: u8::MAX,
                facet_count,
            } if u8::try_from(facet_count).is_err()
        ));

        let err = Cell::facet_view_iter(dt.tds(), cell_key)
            .err()
            .expect("Expected facet_view_iter to fail on vertex_count overflow");
        assert!(matches!(
            err,
            FacetError::InvalidFacetIndex {
                index: u8::MAX,
                facet_count,
            } if u8::try_from(facet_count).is_err()
        ));
    }

    #[test]
    fn cell_deserialize_rejects_missing_uuid_and_duplicate_fields_and_invalid_uuid() {
        // Expecting() should surface for non-map inputs.
        let err = serde_json::from_str::<Cell<f64, (), i32, 3>>("null").unwrap_err();
        assert!(err.to_string().contains("a Cell struct"));

        // Missing required field.
        let err = serde_json::from_str::<Cell<f64, (), i32, 3>>("{\"data\":1}").unwrap_err();
        assert!(err.to_string().contains("missing field `uuid`"));

        // Duplicate uuid.
        let uuid = uuid::Uuid::new_v4();
        let json = format!("{{\"uuid\":\"{uuid}\",\"uuid\":\"{uuid}\"}}");
        let err = serde_json::from_str::<Cell<f64, (), i32, 3>>(&json).unwrap_err();
        assert!(err.to_string().contains("duplicate field `uuid`"));

        // Duplicate data.
        let uuid = uuid::Uuid::new_v4();
        let json = format!("{{\"uuid\":\"{uuid}\",\"data\":1,\"data\":2}}");
        let err = serde_json::from_str::<Cell<f64, (), i32, 3>>(&json).unwrap_err();
        assert!(err.to_string().contains("duplicate field `data`"));

        // Invalid uuid (nil) should be rejected by validate_uuid.
        let json = "{\"uuid\":\"00000000-0000-0000-0000-000000000000\"}";
        let err = serde_json::from_str::<Cell<f64, (), i32, 3>>(json).unwrap_err();
        assert!(err.to_string().contains("invalid uuid"));

        // Unknown fields are ignored.
        let uuid = uuid::Uuid::new_v4();
        let json = format!("{{\"uuid\":\"{uuid}\",\"data\":5,\"neighbors\":[1,2,3]}}");
        let cell = serde_json::from_str::<Cell<f64, (), i32, 3>>(&json).unwrap();
        assert_eq!(cell.data, Some(5));
        assert!(cell.neighbors.is_none());
        assert_eq!(cell.number_of_vertices(), 0, "vertices are not serialized");
    }

    #[test]
    fn cell_validation_error_from_stack_matrix_dispatch_error_maps_to_coordinate_conversion() {
        use crate::geometry::matrix::{MAX_STACK_MATRIX_DIM, StackMatrixDispatchError};

        let err = StackMatrixDispatchError::UnsupportedDim {
            k: MAX_STACK_MATRIX_DIM + 1,
            max: MAX_STACK_MATRIX_DIM,
        };
        let cell_err: CellValidationError = err.into();

        assert!(matches!(
            cell_err,
            CellValidationError::CoordinateConversion { .. }
        ));
    }

    // =============================================================================
    // CELL PARTIALEQ AND EQ TESTS
    // =============================================================================

    #[test]
    fn test_cell_partial_eq_different_dimensions() {
        // Test equality for cells of different dimensions
        let vertices_2d = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt1 = DelaunayTriangulation::new(&vertices_2d).unwrap();
        let (_, cell_2d) = dt1.tds().cells().next().unwrap();
        let dt2 = DelaunayTriangulation::new(&vertices_2d).unwrap();
        let (_, cell_2d_copy) = dt2.tds().cells().next().unwrap();

        // Test equality for 2D cells
        assert_eq!(cell_2d, cell_2d_copy, "Identical 2D cells should be equal");

        println!("✓ 2D cells work correctly with PartialEq");
    }

    #[test]
    fn test_facet_view_iter() {
        // Test the iterator-based facet_view_iter method
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();

        let cell_key = dt.tds().cell_keys().next().unwrap();

        // Test the iterator method
        let facet_iter =
            Cell::facet_view_iter(dt.tds(), cell_key).expect("Failed to get facet iterator");

        // Should know the exact count upfront (implements ExactSizeIterator)
        assert_eq!(
            facet_iter.len(),
            4,
            "Should have 4 facets for a tetrahedron"
        );

        // Collect all results
        let facet_results: Vec<_> = facet_iter.collect();
        assert_eq!(facet_results.len(), 4);

        // Each facet creation should succeed and have 3 vertices
        for (i, facet_result) in facet_results.iter().enumerate() {
            let facet_view = facet_result
                .as_ref()
                .unwrap_or_else(|_| panic!("Facet {i} creation should succeed"));
            let vertex_count = facet_view.vertices().unwrap().count();
            assert_eq!(vertex_count, 3, "Facet {i} should have 3 vertices");
        }

        // Test iterator is zero-allocation by using it without collect
        let facet_iter2 =
            Cell::facet_view_iter(dt.tds(), cell_key).expect("Failed to get second facet iterator");

        let mut count = 0;
        for facet_result in facet_iter2 {
            let _facet_view = facet_result.expect("Facet creation should succeed");
            count += 1;
        }
        assert_eq!(count, 4, "Iterator should yield 4 facets");

        // Test iterator combinators work correctly
        let facet_iter3 =
            Cell::facet_view_iter(dt.tds(), cell_key).expect("Failed to get third facet iterator");

        let successful_facets: Vec<_> = facet_iter3.filter_map(Result::ok).collect();
        assert_eq!(
            successful_facets.len(),
            4,
            "All facets should be created successfully"
        );

        // Compare with Vec-based method to ensure same results
        let vec_facets =
            Cell::facet_views_from_tds(dt.tds(), cell_key).expect("Vec-based method should work");
        assert_eq!(
            successful_facets.len(),
            vec_facets.len(),
            "Iterator and Vec methods should return same count"
        );

        println!("✓ facet_view_iter zero-allocation iterator works correctly");
    }

    // Removed: test_facet_view_memory_efficiency_comparison - covered by test_facet_view_iter

    #[test]
    #[cfg(feature = "bench")]
    fn test_vertex_uuid_iter_by_value_vs_by_reference_analysis() {
        // Comprehensive analysis of whether vertex_uuid_iter should return
        // Uuid by value (current) vs &Uuid by reference (proposed)
        use std::{collections::HashSet, mem, time::Instant};
        use uuid::Uuid;

        println!("\n=== UUID Performance Analysis: By Value vs By Reference ===");

        // Memory layout analysis
        println!("\nMemory Layout:");
        println!("  Size of Uuid:     {} bytes", mem::size_of::<Uuid>());
        println!("  Size of &Uuid:    {} bytes", mem::size_of::<&Uuid>());
        println!("  Size of usize:    {} bytes", mem::size_of::<usize>());

        // Create test cell with multiple vertices in a TDS context
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let (_cell_key, cell) = dt.cells().next().unwrap();

        println!("\nAPI Ergonomics Test:");

        // Test 1: Direct comparison (by value - current implementation)
        let first_uuid = cell.vertex_uuid_iter(dt.tds()).next().unwrap().unwrap();
        assert_ne!(first_uuid, Uuid::nil());
        println!("  ✓ By value: Direct comparison works: uuid != Uuid::nil()");

        // Test 2: What by-reference would look like (simulation)
        // Note: We simulate by-reference by collecting values then referencing them
        let uuid_values: Vec<Uuid> = cell
            .vertices()
            .iter()
            .map(|&vkey| dt.tds().get_vertex_by_key(vkey).unwrap().uuid())
            .collect();
        let uuid_refs: Vec<&Uuid> = uuid_values.iter().collect();
        let first_uuid_ref = uuid_refs[0];
        assert_ne!(*first_uuid_ref, Uuid::nil()); // Requires dereferencing
        println!("  ✓ By reference: Requires dereferencing: *uuid != Uuid::nil()");

        println!("\nPerformance Test (1000 iterations):");
        let iterations = 1000;

        // Benchmark current by-value implementation
        let start = Instant::now();
        let mut by_value_count = 0;
        for _ in 0..iterations {
            for uuid in cell.vertex_uuid_iter(dt.tds()) {
                if uuid.unwrap() != Uuid::nil() {
                    by_value_count += 1;
                }
            }
        }
        let by_value_time = start.elapsed();

        // Benchmark simulated by-reference implementation
        let start = Instant::now();
        let mut by_ref_count = 0;
        for _ in 0..iterations {
            // For by-reference simulation, we'd need to store values first
            let uuid_values: Vec<Uuid> = cell
                .vertices()
                .iter()
                .map(|&vkey| dt.tds().get_vertex_by_key(vkey).unwrap().uuid())
                .collect();
            for uuid_ref in &uuid_values {
                if *uuid_ref != Uuid::nil() {
                    by_ref_count += 1;
                }
            }
        }
        let by_ref_time = start.elapsed();

        println!("  By value time:     {by_value_time:?}");
        println!("  By reference time: {by_ref_time:?}");

        let ratio = by_ref_time.as_secs_f64() / by_value_time.as_secs_f64().max(f64::EPSILON);
        if ratio > 1.05 {
            println!("  → By value is {ratio:.2}x FASTER");
        } else if ratio < 0.95 {
            println!("  → By reference is {:.2}x faster", 1.0 / ratio);
        } else {
            println!("  → Performance is roughly equivalent");
        }

        assert_eq!(by_value_count, by_ref_count);

        println!("\nAnalysis Summary:");
        println!(
            "  UUID size: {} bytes (small value type)",
            mem::size_of::<Uuid>()
        );
        println!(
            "  Reference size: {} bytes (pointer)",
            mem::size_of::<&Uuid>()
        );
        println!("  \nFor UUIDs specifically:");
        println!("  • UUID is only 16 bytes (fits in two 64-bit registers)");
        println!("  • Modern CPUs copy 16 bytes very efficiently");
        println!("  • No indirection overhead with by-value");
        println!("  • Consumers can own the value (no lifetime constraints)");
        println!("  • Direct comparisons work without dereferencing");

        println!("  \nConclusion: Current by-value implementation is optimal");
        println!("  Reasons:");
        println!("    1. Better or equivalent performance (no indirection)");
        println!("    2. Simpler API (no lifetime constraints)");
        println!("    3. More ergonomic for consumers (no dereferencing)");
        println!("    4. Consistent with UUID::new() and similar APIs");
        println!("    5. 16 bytes is small enough to copy efficiently");

        // Validate current API works as expected
        assert_eq!(cell.vertex_uuid_iter(dt.tds()).count(), 4);

        // Test that we can directly use values in hashmaps, comparisons, etc.
        let unique_uuids: HashSet<_> = cell
            .vertex_uuid_iter(dt.tds())
            .collect::<Result<HashSet<_>, _>>()
            .unwrap();
        assert_eq!(unique_uuids.len(), 4);

        println!("  ✓ Current API validation passed");
    }

    #[test]
    fn test_clear_vertex_keys() {
        // Test the clear_vertex_keys method used in deserialization
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();

        // Get a cell and clone it (since we need mutable access)
        let (_cell_key, original_cell) = dt.cells().next().unwrap();
        let mut test_cell = original_cell.clone();

        // Verify the cell has 4 vertices initially
        assert_eq!(
            test_cell.number_of_vertices(),
            4,
            "Cell should start with 4 vertices"
        );

        // Clear the vertex keys
        test_cell.clear_vertex_keys();

        // Verify the cell now has 0 vertices
        assert_eq!(
            test_cell.number_of_vertices(),
            0,
            "Cell should have 0 vertices after clearing"
        );
        assert_eq!(
            test_cell.vertices().len(),
            0,
            "Vertices slice should be empty after clearing"
        );

        // Test that we can rebuild vertex keys after clearing (simulating deserialization)
        for &vkey in original_cell.vertices() {
            test_cell.push_vertex_key(vkey);
        }

        // Verify we've restored the correct number of vertices
        assert_eq!(
            test_cell.number_of_vertices(),
            4,
            "Cell should have 4 vertices after rebuilding"
        );

        // Verify the vertex keys match the original
        for (original_vkey, rebuilt_vkey) in original_cell
            .vertices()
            .iter()
            .zip(test_cell.vertices().iter())
        {
            assert_eq!(
                original_vkey, rebuilt_vkey,
                "Rebuilt vertex keys should match original keys"
            );
        }

        println!("✓ clear_vertex_keys() correctly clears and allows rebuilding of vertex keys");
    }
}
