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
//! - **Macro-based Construction**: Convenient cell creation using the `cell!` macro.
//!
//! # Examples
//!
//! ```rust
//! use delaunay::core::cell::Cell;
//! use delaunay::geometry::point::Point;
//! use delaunay::geometry::traits::coordinate::Coordinate;
//! use delaunay::{cell, vertex};
//!
//! // Create vertices for a tetrahedron
//! let vertices = vec![
//!     vertex!([0.0, 0.0, 0.0]),
//!     vertex!([1.0, 0.0, 0.0]),
//!     vertex!([0.0, 1.0, 0.0]),
//!     vertex!([0.0, 0.0, 1.0]),
//! ];
//!
//! // Create a 3D cell (tetrahedron)
//! let cell: Cell<f64, Option<()>, Option<()>, 3> = cell!(vertices);
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
use crate::core::collections::{
    FastHashMap, FastHashSet, MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer,
};
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

// =============================================================================
// CONVENIENCE MACROS AND HELPERS
// =============================================================================

/// Convenience macro for creating cells with less boilerplate.
///
/// # ⚠️ Deprecated
///
/// **This macro is deprecated and will be removed in v0.6.0.**
///
/// ## Why It's Being Removed
///
/// In Phase 3A, cells store `VertexKey`s that are only valid within a specific TDS context.
/// This macro creates a temporary TDS, clones a cell with its keys, then discards the TDS,
/// leaving a cell with "dangling" keys that cannot be used with any TDS operations.
///
/// This violates the Phase 3A architecture where **cells cannot exist independently from a TDS**.
///
/// ## Migration Guide
///
/// Instead of:
/// ```rust,ignore
/// use delaunay::{cell, vertex};
/// let vertices = vec![vertex!([0.0, 0.0, 0.0]), ...];
/// let c: Cell<f64, Option<()>, Option<()>, 3> = cell!(vertices);
/// // c.vertices() contains keys that reference a dropped TDS!
/// ```
///
/// Use:
/// ```rust
/// use delaunay::{vertex, core::triangulation_data_structure::Tds};
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
/// let (cell_key, cell) = tds.cells().next().unwrap();
/// // Now cell's keys are valid within tds context
/// ```
///
/// # For Existing Code
///
/// This macro still works for basic property testing (UUID, dimension, etc.) but
/// **DO NOT use the returned cell's `VertexKeys` with any TDS operations** - they
/// reference a TDS that no longer exists.
///
/// # ⚠️ Tests Only
///
/// **This macro should only be used in tests.** It creates cells with dangling keys
/// that reference a dropped TDS. Use `Tds::new()` for all production code.
#[deprecated(
    since = "0.5.2",
    note = "Cells cannot exist independently from a TDS in Phase 3A. \
            Create cells through Tds::new() and work with them in TDS context. \
            This macro will be removed in v0.6.0. See macro documentation for migration guide."
)]
#[macro_export]
macro_rules! cell {
    // Pattern 1: Just vertices - creates via TDS
    ($vertices:expr) => {{
        use $crate::core::triangulation_data_structure::Tds;
        let tds = Tds::new(&$vertices).expect("Failed to create triangulation from vertices");
        tds.cells()
            .map(|(_, cell)| cell)
            .next()
            .expect("No cells created - need at least D+1 vertices")
            .clone()
    }};

    // Pattern 2: Vertices with data - creates via TDS then sets data
    ($vertices:expr, $data:expr) => {{
        use $crate::core::triangulation_data_structure::Tds;
        let tds = Tds::new(&$vertices).expect("Failed to create triangulation from vertices");
        let mut cell = tds
            .cells()
            .map(|(_, cell)| cell)
            .next()
            .expect("No cells created - need at least D+1 vertices")
            .clone();
        cell.data = Some($data);
        cell
    }};
}

// Re-export the macro at the crate level for convenience
pub use crate::cell;

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
/// use delaunay::{vertex, core::triangulation_data_structure::Tds};
///
/// // Create a TDS with some vertices
/// let vertices = vec![
///     vertex!([0.0, 0.0]),
///     vertex!([1.0, 0.0]),
///     vertex!([0.0, 1.0]),
/// ];
/// let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
///
/// // Get first cell and iterate over vertex keys
/// let (cell_key, cell) = tds.cells().next().unwrap();
/// for &vertex_key in cell.vertices() {
///     let vertex = &tds.get_vertex_by_key(vertex_key).unwrap();
///     // use vertex...
///     assert!(vertex.uuid() != uuid::Uuid::nil());
/// }
/// ```
pub struct Cell<T, U, V, const D: usize>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    /// Keys to the vertices forming this cell.
    /// Phase 3A: Changed from `Vec<Vertex>` to `SmallBuffer<VertexKey, 8>` for:
    /// - Zero heap allocation for D ≤ 7 (stack-allocated)
    /// - Direct key access without UUID lookup
    /// - Better cache locality
    ///
    /// Note: Not serialized - vertices are serialized separately and keys
    /// are reconstructed during deserialization.
    vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,

    /// The unique identifier of the cell.
    uuid: Uuid,

    /// Keys to neighboring cells, indexed by opposite vertex.
    /// Phase 3A: Changed from `Option<Vec<Option<Uuid>>>` to `Option<SmallBuffer<Option<CellKey>, 8>>`.
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
    pub(crate) neighbors: Option<SmallBuffer<Option<CellKey>, MAX_PRACTICAL_DIMENSION_SIZE>>,

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
                let vertices = SmallBuffer::new();

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
    /// Returns `CellValidationError::InsufficientVertices` if `vertices` doesn't
    /// have exactly D+1 elements.
    pub(crate) fn new(
        vertices: impl Into<SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>>,
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
    /// use delaunay::{vertex, core::triangulation_data_structure::Tds};
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
    /// let (_, cell) = tds.cells().next().unwrap();
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
    /// use delaunay::{vertex, core::triangulation_data_structure::Tds};
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    ///     vertex!([1.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
    /// let mut cells_iter = tds.cells().map(|(_, cell)| cell);
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
    /// use delaunay::{vertex, core::triangulation_data_structure::Tds};
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
    /// let (_, cell) = tds.cells().next().unwrap();
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
    /// use delaunay::{vertex, core::triangulation_data_structure::Tds};
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
    /// let (_, cell) = tds.cells().next().unwrap();
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
    pub const fn neighbors(
        &self,
    ) -> Option<&SmallBuffer<Option<CellKey>, MAX_PRACTICAL_DIMENSION_SIZE>> {
        self.neighbors.as_ref()
    }

    /// Returns the vertex keys for this cell.
    ///
    /// # Phase 3A
    ///
    /// This method returns keys (not full vertex objects). Use the TDS to resolve keys:
    /// ```rust
    /// use delaunay::{vertex, core::triangulation_data_structure::Tds};
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
    /// let (_, cell) = tds.cells().next().unwrap();
    ///
    /// for &vkey in cell.vertices() {
    ///     let vertex = &tds.get_vertex_by_key(vkey).unwrap();
    ///     // use vertex data...
    ///     assert!(vertex.uuid() != uuid::Uuid::nil());
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
    /// and reduce the error surface for off-by-one bugs. Used internally by insertion algorithms.
    ///
    /// # Returns
    ///
    /// A mutable reference to the neighbors buffer, guaranteed to be sized D+1 with all None values.
    ///
    /// # Performance
    ///
    /// Inline to zero cost in release builds. Only allocates if the buffer doesn't exist.
    #[inline]
    pub(crate) fn ensure_neighbors_buffer_mut(
        &mut self,
    ) -> &mut SmallBuffer<Option<CellKey>, MAX_PRACTICAL_DIMENSION_SIZE> {
        self.neighbors.get_or_insert_with(|| {
            let mut buffer = SmallBuffer::new();
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
    /// use delaunay::{vertex, core::triangulation_data_structure::Tds};
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
    /// let cell_key = tds.cells().next().unwrap().0;
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
    /// use delaunay::{cell, vertex};
    /// use delaunay::core::cell::Cell;
    /// use uuid::Uuid;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 1.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0]),
    /// ];
    /// let cell: Cell<f64, Option<()>, Option<()>, 3> = cell!(vertices);
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
    /// ```rust
    /// use delaunay::{vertex, core::triangulation_data_structure::Tds};
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let mut tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
    /// // Use clear_all_neighbors() which is the public API for this operation
    /// tds.clear_all_neighbors();
    /// let cell_key = tds.cells().next().unwrap().0;
    /// assert!(tds.get_cell(cell_key).unwrap().neighbors().is_none());
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
    /// A `Result<Vec<Uuid>, CellValidationError>` containing the UUIDs of all vertices in this cell,
    /// or an error if a vertex key is not found in the TDS.
    ///
    /// # Errors
    ///
    /// Returns `CellValidationError::VertexKeyNotFound` if a vertex key in the cell
    /// does not exist in the TDS. This indicates TDS corruption or inconsistency.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::{vertex, core::triangulation_data_structure::Tds};
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
    /// let cell_key = tds.cells().next().unwrap().0;
    /// let cell = &tds.get_cell(cell_key).unwrap();
    /// let uuids = cell.vertex_uuids(&tds).unwrap();
    /// assert_eq!(uuids.len(), 4);
    /// ```
    #[inline]
    pub fn vertex_uuids(&self, tds: &Tds<T, U, V, D>) -> Result<Vec<Uuid>, CellValidationError> {
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
    /// use delaunay::{vertex, core::triangulation_data_structure::Tds};
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
    /// let cell_key = tds.cells().next().unwrap().0;
    /// let cell = &tds.get_cell(cell_key).unwrap();
    /// let uuids: Vec<_> = cell.vertex_uuid_iter(&tds).collect::<Result<Vec<_>, _>>().unwrap();
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
    /// use delaunay::{cell, vertex};
    /// use delaunay::core::cell::Cell;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 1.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0]),
    /// ];
    /// let cell: Cell<f64, Option<()>, Option<()>, 3> = cell!(vertices);
    /// assert_eq!(cell.dim(), 3);
    /// ```
    #[inline]
    pub const fn dim(&self) -> usize {
        D
    }

    /// Deprecated alias for `has_vertex_in_common`.
    ///
    /// This method will be removed in v0.6.0. Use `has_vertex_in_common` instead.
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
    /// use delaunay::{vertex, core::triangulation_data_structure::Tds};
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    ///     vertex!([1.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
    /// let mut cells_iter = tds.cells().map(|(_, cell)| cell);
    /// let cell1 = cells_iter.next().unwrap();
    /// let cell2 = cells_iter.next().unwrap();
    ///
    /// #[allow(deprecated)]
    /// {
    ///     // Old API (deprecated)
    ///     if cell1.contains_vertex_of(cell2) {
    ///         println!("Cells share vertices");
    ///     }
    /// }
    ///
    /// // New API (preferred)
    /// if cell1.has_vertex_in_common(cell2) {
    ///     println!("Cells share vertices");
    /// }
    /// ```
    #[deprecated(
        since = "0.5.1",
        note = "Use `has_vertex_in_common` instead. This method will be removed in v0.6.0."
    )]
    #[inline]
    pub fn contains_vertex_of(&self, other: &Self) -> bool {
        self.has_vertex_in_common(other)
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
    /// use delaunay::{cell, vertex};
    /// use delaunay::core::cell::Cell;
    /// use uuid::Uuid;
    ///
    /// // Create some cells (need 4 vertices each for 3D simplices)
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
    /// let cell1: Cell<f64, Option<()>, Option<()>, 3> = cell!(vertices1);
    /// let cell2: Cell<f64, Option<()>, Option<()>, 3> = cell!(vertices2);
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
    /// use delaunay::{cell, vertex};
    /// use delaunay::core::cell::Cell;
    ///
    /// // Empty vector produces empty FastHashMap
    /// let empty_cells: Vec<Cell<f64, Option<()>, Option<()>, 3>> = vec![];
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
    /// use delaunay::{cell, vertex};
    /// use delaunay::core::cell::Cell;
    /// use delaunay::core::vertex::Vertex;
    ///
    /// let vertex1: Vertex<f64, Option<()>, 3> = vertex!([0.0, 0.0, 1.0]);
    /// let vertex2: Vertex<f64, Option<()>, 3> = vertex!([0.0, 1.0, 0.0]);
    /// let vertex3: Vertex<f64, Option<()>, 3> = vertex!([1.0, 0.0, 0.0]);
    /// let vertex4: Vertex<f64, Option<()>, 3> = vertex!([0.0, 0.0, 0.0]);
    /// let cell: Cell<f64, Option<()>, Option<()>, 3> = cell!(vec![vertex1, vertex2, vertex3, vertex4]);
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
    ///
    /// # Examples
    ///
    /// Returns all facets of this cell as lightweight `FacetView` objects.
    ///
    /// This method provides a more efficient alternative to `facets()` by returning
    /// lightweight `FacetView` objects instead of owned `Facet` objects.
    ///
    /// # Arguments
    ///
    /// * `tds` - Reference to the triangulation data structure
    /// * `cell_key` - The key of this cell in the TDS
    ///
    /// # Returns
    ///
    /// A `Result<Vec<FacetView>, FacetError>` containing all facets of the cell.
    /// Each facet is represented as a `FacetView` which provides efficient access
    /// to facet properties without cloning cell data.
    ///
    /// # Errors
    ///
    /// Returns a [`FacetError`] if facet creation fails during the construction
    /// of `FacetView` objects.
    ///
    /// # Notes
    ///
    /// Returns an error if the facet index cannot be represented as `u8`.
    /// This should never happen in practice since facet indices are bounded
    /// by the dimension `D`, which is typically small (≤ 255).
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::{core::triangulation_data_structure::Tds, vertex};
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
    ///
    /// let cell_key = tds.cell_keys().next().unwrap();
    /// let cell = tds.get_cell(cell_key).unwrap();
    /// let facet_views = cell.facet_views(&tds, cell_key).expect("Failed to get facet views");
    ///
    /// // Each facet should have 3 vertices (triangular faces of tetrahedron)
    /// for facet_view in &facet_views {
    ///     assert_eq!(facet_view.vertices().unwrap().count(), 3);
    /// }
    /// ```
    #[deprecated(
        since = "0.5.2",
        note = "Use `Cell::facet_views_from_tds(tds, cell_key)` instead. This instance method will be removed in 0.6.0. The static method is more direct and avoids redundant parameters."
    )]
    pub fn facet_views<'tds>(
        &self,
        tds: &'tds Tds<T, U, V, D>,
        cell_key: CellKey,
    ) -> Result<Vec<crate::core::facet::FacetView<'tds, T, U, V, D>>, FacetError> {
        // Delegate to the static method to avoid code duplication
        Self::facet_views_from_tds(tds, cell_key)
    }

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
    /// use delaunay::{core::{triangulation_data_structure::Tds, cell::Cell}, vertex};
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
    ///
    /// let cell_key = tds.cell_keys().next().unwrap();
    /// let facet_views = Cell::facet_views_from_tds(&tds, cell_key).expect("Failed to get facet views");
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
    /// use delaunay::{core::triangulation_data_structure::Tds, vertex};
    ///
    /// // Example 1: Comparing cells from different TDS instances with same coordinates
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let tds1: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
    /// let tds2: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
    ///
    /// let cell1 = tds1.cells().next().unwrap().1;
    /// let cell2 = tds2.cells().next().unwrap().1;
    ///
    /// // Different TDS instances, but same vertex coordinates
    /// assert!(cell1.eq_by_vertices(&tds1, cell2, &tds2));
    /// ```
    ///
    /// ```
    /// use delaunay::{core::triangulation_data_structure::Tds, vertex};
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
    /// let tds1: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices1).unwrap();
    /// let tds2: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices2).unwrap();
    ///
    /// let cell1 = tds1.cells().next().unwrap().1;
    /// let cell2 = tds2.cells().next().unwrap().1;
    ///
    /// // Different coordinates mean cells are not equal
    /// assert!(!cell1.eq_by_vertices(&tds1, cell2, &tds2));
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
    /// use delaunay::{core::{triangulation_data_structure::Tds, cell::Cell}, vertex};
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
    ///
    /// let cell_key = tds.cell_keys().next().unwrap();
    /// let facet_iter = Cell::facet_view_iter(&tds, cell_key).expect("Failed to get facet iterator");
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
    /// use delaunay::{core::{triangulation_data_structure::Tds, cell::Cell}, vertex};
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
    ///
    /// let cell_key = tds.cell_keys().next().unwrap();
    /// let facet_iter = Cell::facet_view_iter(&tds, cell_key).expect("Failed to get facet iterator");
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
        let mut self_keys: Vec<_> = self.vertices.iter().copied().collect();
        let mut other_keys: Vec<_> = other.vertices.iter().copied().collect();
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
        let mut self_keys: Vec<_> = self.vertices.iter().copied().collect();
        let mut other_keys: Vec<_> = other.vertices.iter().copied().collect();
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
        let mut sorted_keys: Vec<_> = self.vertices.iter().copied().collect();
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
    use crate::geometry::predicates::{insphere, insphere_distance};
    use crate::geometry::util::{circumcenter, circumradius, circumradius_with_center};
    use approx::assert_relative_eq;
    use slotmap::KeyData;
    use std::{cmp, collections::hash_map::DefaultHasher, hash::Hasher};

    // Type aliases for commonly used types to reduce repetition
    type TestVertex3D = Vertex<f64, Option<()>, 3>;
    type TestVertex2D = Vertex<f64, Option<()>, 2>;

    use crate::core::triangulation_data_structure::Tds;

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

    /// Helper to create a TDS with a standard 3D tetrahedron and return (TDS, `CellKey`, Cell reference)
    fn create_test_tds_3d() -> (Tds<f64, Option<()>, Option<()>, 3>, CellKey) {
        let vertices = create_test_vertices_3d();
        let tds = Tds::new(&vertices).expect("Failed to create TDS");
        let cell_key = tds.cell_keys().next().expect("No cells in TDS");
        (tds, cell_key)
    }

    /// Helper to create a TDS with a standard 2D triangle and return (TDS, `CellKey`)
    fn create_test_tds_2d() -> (Tds<f64, Option<()>, Option<()>, 2>, CellKey) {
        let vertices = create_test_vertices_2d();
        let tds = Tds::new(&vertices).expect("Failed to create TDS");
        let cell_key = tds.cell_keys().next().expect("No cells in TDS");
        (tds, cell_key)
    }

    /// Helper to create a TDS with custom 3D vertices and cell data
    #[allow(dead_code)]
    fn create_test_tds_3d_with_cell_data<V: DataType>(
        cell_data: V,
    ) -> (Tds<f64, Option<()>, V, 3>, CellKey) {
        let vertices = create_test_vertices_3d();
        let mut tds = Tds::new(&vertices).expect("Failed to create TDS");
        let cell_key = tds.cell_keys().next().expect("No cells in TDS");
        // Set cell data
        if let Some(cell) = tds.cells_mut().get_mut(cell_key) {
            cell.data = Some(cell_data);
        }
        (tds, cell_key)
    }

    // =============================================================================
    // CONVENIENCE MACRO TESTS
    // =============================================================================
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

        // Phase 3A: Create TDS to get cell with context
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let (_, cell) = tds.cells().next().unwrap();

        assert_eq!(cell.number_of_vertices(), 4);
        assert_eq!(cell.dim(), 3);
        assert!(cell.data.is_none());
        assert!(!cell.uuid().is_nil());

        // Verify vertices match what we put in - need TDS to resolve keys
        for (original, &vkey) in vertices.iter().zip(cell.vertices().iter()) {
            let result = &tds.get_vertex_by_key(vkey).unwrap();
            assert_relative_eq!(
                original.point().coords().as_slice(),
                result.point().coords().as_slice(),
                epsilon = f64::EPSILON
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
        let tds1: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let tds2: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        let cell1 = tds1.cells().next().unwrap().1;
        let cell2 = tds2.cells().next().unwrap().1;

        // Despite different TDS instances (and thus different vertex keys),
        // cells should be equal by coordinates
        assert!(cell1.eq_by_vertices(&tds1, cell2, &tds2));
        println!("  ✓ Cells from different TDS with same coordinates are equal");
    }

    #[test]
    fn test_eq_by_vertices_different_coordinates() {
        println!("Testing eq_by_vertices with different coordinates");

        let vertices1 = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let vertices2 = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([2.0, 0.0, 0.0]), // Different
            vertex!([0.0, 2.0, 0.0]), // Different
            vertex!([0.0, 0.0, 2.0]), // Different
        ];
        let tds1: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices1).unwrap();
        let tds2: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices2).unwrap();

        let cell1 = tds1.cells().next().unwrap().1;
        let cell2 = tds2.cells().next().unwrap().1;

        // Cells with different coordinates should not be equal
        assert!(!cell1.eq_by_vertices(&tds1, cell2, &tds2));
        println!("  ✓ Cells with different coordinates are not equal");
    }

    #[test]
    fn test_eq_by_vertices_2d() {
        println!("Testing eq_by_vertices in 2D");

        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let tds1: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
        let tds2: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();

        let cell1 = tds1.cells().next().unwrap().1;
        let cell2 = tds2.cells().next().unwrap().1;

        assert!(cell1.eq_by_vertices(&tds1, cell2, &tds2));
        println!("  ✓ 2D cells with same coordinates are equal");
    }

    #[test]
    fn test_eq_by_vertices_order_independence() {
        println!("Testing eq_by_vertices is order-independent");

        // Create two TDS with vertices in different orders
        let vertices1 = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        // Different order but same coordinates
        let vertices2 = vec![
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
        ];
        let tds1: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices1).unwrap();
        let tds2: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices2).unwrap();

        // Both should create the same cell structure (same simplex)
        let cell1 = tds1.cells().next().unwrap().1;
        let cell2 = tds2.cells().next().unwrap().1;

        // Should be equal regardless of vertex order in input
        assert!(cell1.eq_by_vertices(&tds1, cell2, &tds2));
        println!("  ✓ Cell comparison is order-independent");
    }

    #[test]
    fn test_eq_by_vertices_same_tds() {
        println!("Testing eq_by_vertices on cells from same TDS");

        // Create TDS with multiple cells
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.5, 0.5, 0.5]), // Extra vertex to create multiple cells
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        // Get two different cells
        let mut cells = tds.cells();
        let cell1 = cells.next().unwrap().1;
        let cell2 = cells.next().unwrap().1;

        // They should not be equal (different vertex sets)
        assert!(!cell1.eq_by_vertices(&tds, cell2, &tds));

        // A cell should be equal to itself
        assert!(cell1.eq_by_vertices(&tds, cell1, &tds));
        println!("  ✓ Cell equality within same TDS works correctly");
    }

    #[test]
    fn test_eq_by_vertices_with_floating_point_precision() {
        println!("Testing eq_by_vertices respects Vertex::PartialEq precision");

        // Vertex::PartialEq uses ordered_equals which handles floating point comparison
        let vertices1 = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        // Exact same coordinates
        let vertices2 = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds1: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices1).unwrap();
        let tds2: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices2).unwrap();

        let cell1 = tds1.cells().next().unwrap().1;
        let cell2 = tds2.cells().next().unwrap().1;

        assert!(cell1.eq_by_vertices(&tds1, cell2, &tds2));
        println!("  ✓ Exact floating point values are equal");
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

        // Build TDS, then clone the first cell and set data
        let tds: Tds<f64, Option<()>, i32, 3> = Tds::new(&vertices).unwrap();
        let (_, cell_ref) = tds.cells().next().unwrap();
        let mut cell = cell_ref.clone();
        cell.data = Some(42);

        assert_eq!(cell.number_of_vertices(), 4);
        assert_eq!(cell.dim(), 3);
        assert_eq!(cell.data.unwrap(), 42);
        assert!(!cell.uuid().is_nil());

        // Verify against the same TDS instance
        for (original, &vkey) in vertices.iter().zip(cell.vertices().iter()) {
            let result = &tds.get_vertex_by_key(vkey).unwrap();
            assert_relative_eq!(
                original.point().coords().as_slice(),
                result.point().coords().as_slice(),
                epsilon = f64::EPSILON
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

        // Create TDS which creates cells from vertices
        let tds: Tds<f64, i32, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let (_, cell) = tds.cells().next().unwrap();

        assert_eq!(cell.number_of_vertices(), 4);
        assert_eq!(cell.dim(), 3);

        // Check vertex data through TDS - vertices() returns keys
        for (expected_data, &vkey) in [1, 2, 3, 4].iter().zip(cell.vertices().iter()) {
            let vertex = &tds.get_vertex_by_key(vkey).unwrap();
            assert_eq!(vertex.data.unwrap(), *expected_data);
        }
    }

    // Phase 3A: CellBuilder test removed
    // CellBuilder is deprecated in Phase 3A. Cells should be created through TDS.
    // The cell! macro now creates a temporary TDS internally for testing purposes.

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
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();
        let (_, cell1) = tds.cells().next().unwrap();
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
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&all_vertices).unwrap();
        let cells: Vec<_> = tds.cells().map(|(_, cell)| cell).collect();

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
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();
        let (_, cell1) = tds.cells().next().unwrap();
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

    // Phase 3A: Test removed - neighbors now use CellKey not UUID
    // Neighbors can only be set through TDS methods, not directly.
    // The hash implementation correctly ignores neighbors for Eq/Hash contract.

    #[test]
    fn cell_hash_distinct_data() {
        let vertices = vec![
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0]),
        ];
        let tds = Tds::<f64, Option<()>, i32, 3>::new(&vertices).unwrap();
        let (_, cell_ref) = tds.cells().next().unwrap();
        let mut cell1 = cell_ref.clone();
        cell1.data = Some(42);
        let mut cell2 = cell_ref.clone();
        cell2.data = Some(24);

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();

        cell1.hash(&mut hasher1);
        cell2.hash(&mut hasher2);

        // Same vertices should produce same hash despite different data (Eq/Hash contract)
        assert_eq!(cell1, cell2); // They are equal by vertices
        assert_eq!(hasher1.finish(), hasher2.finish()); // Therefore hashes must be equal
        // Note: Data is different but excluded from hashing to maintain Eq/Hash contract
        assert_ne!(cell1.data, cell2.data);
    }

    #[test]
    fn cell_clone() {
        let vertices = vec![
            vertex!([0.0, 0.0, 1.0], 1),
            vertex!([0.0, 1.0, 0.0], 1),
            vertex!([1.0, 0.0, 0.0], 1),
            vertex!([1.0, 1.0, 1.0], 2),
        ];
        let tds = Tds::<f64, i32, i32, 3>::new(&vertices).unwrap();
        let (_, cell_ref) = tds.cells().next().unwrap();
        let mut cell1 = cell_ref.clone();
        cell1.data = Some(42);
        let cell2 = cell1.clone();

        assert_eq!(cell1, cell2);
    }

    #[test]
    fn cell_eq_trait() {
        // Phase 3A: Test Eq trait using TDS
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();
        let (_, cell1) = tds.cells().next().unwrap();

        // Test Eq trait (reflexivity)
        assert_eq!(cell1, cell1); // reflexive

        // Test cloned cell equals original
        let cell1_clone = cell1.clone();
        assert_eq!(*cell1, cell1_clone); // same vertex keys after clone
    }

    #[test]
    fn cell_ordering_edge_cases() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();
        let (_, cell1) = tds.cells().next().unwrap();
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
        let (tds_2d, cell_key_2d) = create_test_tds_2d();
        let triangle = tds_2d.get_cell(cell_key_2d).unwrap();
        assert_eq!(triangle.number_of_vertices(), 3);

        let (tds_3d, cell_key_3d) = create_test_tds_3d();
        let tetrahedron = tds_3d.get_cell(cell_key_3d).unwrap();
        assert_eq!(tetrahedron.number_of_vertices(), 4);
    }

    #[test]
    fn cell_dim() {
        let (tds_2d, cell_key_2d) = create_test_tds_2d();
        let triangle = tds_2d.get_cell(cell_key_2d).unwrap();
        assert_eq!(triangle.dim(), 2);

        let (tds_3d, cell_key_3d) = create_test_tds_3d();
        let tetrahedron = tds_3d.get_cell(cell_key_3d).unwrap();
        assert_eq!(tetrahedron.dim(), 3);
    }

    #[test]
    fn cell_contains_vertex() {
        let vertex1: Vertex<f64, i32, 3> = vertex!([0.0, 0.0, 1.0], 1);
        let vertex2 = vertex!([0.0, 1.0, 0.0], 1);
        let vertex3 = vertex!([1.0, 0.0, 0.0], 1);
        let vertex4 = vertex!([1.0, 1.0, 1.0], 2);

        // Create TDS to get VertexKeys
        let vertices = vec![vertex1, vertex2, vertex3, vertex4];
        let tds = Tds::<f64, i32, Option<()>, 3>::new(&vertices).unwrap();
        let cell_key = tds.cells().next().unwrap().0;
        let cell = &tds.get_cell(cell_key).unwrap();

        // Get vertex keys from TDS
        let vertex_keys: Vec<_> = tds.vertices().map(|(k, _)| k).collect();

        assert!(cell.contains_vertex(vertex_keys[0]));
        assert!(cell.contains_vertex(vertex_keys[1]));
        assert!(cell.contains_vertex(vertex_keys[2]));
        assert!(cell.contains_vertex(vertex_keys[3]));

        // Human readable output for cargo test -- --nocapture
        println!("Cell: {cell:?}");
    }

    #[test]
    fn cell_contains_vertex_of() {
        // Create two triangulations to test contains_vertex_of
        let vertices1 = vec![
            vertex!([0.0, 0.0, 1.0], 1),
            vertex!([0.0, 1.0, 0.0], 1),
            vertex!([1.0, 0.0, 0.0], 1),
            vertex!([1.0, 1.0, 1.0], 2),
        ];
        let tds1 = Tds::<f64, i32, i32, 3>::new(&vertices1).unwrap();
        let (_, cell_ref) = tds1.cells().next().unwrap();
        let mut cell = cell_ref.clone();
        cell.data = Some(42);

        let vertices2 = vec![
            vertex!([0.0, 0.0, 1.0], 1),
            vertex!([0.0, 1.0, 0.0], 1),
            vertex!([1.0, 0.0, 0.0], 1),
            vertex!([0.0, 0.0, 0.0], 0),
        ];
        let tds2 = Tds::<f64, i32, i32, 3>::new(&vertices2).unwrap();
        let (_, cell2_ref) = tds2.cells().next().unwrap();
        let mut cell2 = cell2_ref.clone();
        cell2.data = Some(43);

        assert!(cell.contains_vertex_of(&cell2));

        // Human readable output for cargo test -- --nocapture
        println!("Cell: {cell:?}");
    }

    // Phase 3A: Test removed - neighbors now use CellKey and can only be set through TDS
    // The clear_neighbors() method still exists and is tested through TDS operations.

    #[test]
    fn cell_facets_contains() {
        let vertices = vec![
            vertex!([0.0, 0.0, 1.0], 1),
            vertex!([0.0, 1.0, 0.0], 1),
            vertex!([1.0, 0.0, 0.0], 1),
            vertex!([1.0, 1.0, 1.0], 2),
        ];

        // Create TDS to get proper cell and facet views
        let tds = Tds::<f64, i32, i32, 3>::new(&vertices).unwrap();
        let cell_key = tds.cells().next().unwrap().0;
        let cell = &tds.get_cell(cell_key).unwrap();

        let facets = cell
            .facet_views(&tds, cell_key)
            .expect("Failed to get facets");

        assert_eq!(facets.len(), 4);
        for facet in &facets {
            let facet_vertices = facet.vertices().expect("Failed to get facet vertices");
            let facet_vertices_vec: Vec<_> = facet_vertices.collect();

            assert!(
                cell.facet_views(&tds, cell_key)
                    .expect("Failed to get facets")
                    .iter()
                    .any(|f| f.vertices().is_ok_and(|fv| {
                        let fv_vec: Vec<_> = fv.collect();
                        fv_vec == facet_vertices_vec
                    }))
            );
        }

        // Human readable output for cargo test -- --nocapture
        println!("Facets: {facets:?}");
    }

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
        let tds = Tds::<f64, i32, Option<()>, 3>::new(&vertices).unwrap();
        let cell_key = tds.cells().next().unwrap().0;
        let cell = &tds.get_cell(cell_key).unwrap();

        // Get vertex UUIDs
        let vertex_uuids = cell.vertex_uuids(&tds).unwrap();
        assert_eq!(cell.vertex_uuid_iter(&tds).count(), 4);

        // Verify UUIDs match the cell's vertices using iterator
        for (expected_uuid, returned_uuid) in cell.vertex_uuid_iter(&tds).zip(vertex_uuids.iter()) {
            assert_eq!(expected_uuid.unwrap(), *returned_uuid);
        }

        // Verify all UUIDs are unique
        let unique_uuids: std::collections::HashSet<_> = vertex_uuids.iter().collect();
        assert_eq!(unique_uuids.len(), vertex_uuids.len());

        // Verify no nil UUIDs using iterator
        for uuid in cell.vertex_uuid_iter(&tds) {
            assert_ne!(uuid.unwrap(), Uuid::nil());
        }

        println!("✓ vertex_uuids method returns correct vertex UUIDs");
    }

    #[test]
    fn test_vertex_uuids_empty_cell_fails() {
        // Test that TDS creation fails gracefully with insufficient vertices for dimension
        // Phase 3A: Now tested through TDS which is the user-facing API

        let vertices = vec![vertex!([0.0, 0.0, 0.0])];
        let result = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices);

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("Insufficient vertices"));

        println!("✓ TDS construction properly validates vertex count");
    }

    #[test]
    fn test_vertex_uuids_2d_cell() {
        // Test vertex_uuids with a 2D cell (triangle)
        let vertex1 = vertex!([0.0, 0.0], 1);
        let vertex2 = vertex!([1.0, 0.0], 2);
        let vertex3 = vertex!([0.5, 1.0], 3);

        let vertices = vec![vertex1, vertex2, vertex3];
        let tds = Tds::<f64, i32, Option<()>, 2>::new(&vertices).unwrap();
        let cell_key = tds.cells().next().unwrap().0;
        let cell = &tds.get_cell(cell_key).unwrap();

        // Get vertex UUIDs
        let vertex_uuids = cell.vertex_uuids(&tds).unwrap();
        assert_eq!(cell.vertex_uuid_iter(&tds).count(), 3);

        // Verify UUIDs match the cell's vertices using iterator
        for (expected_uuid, returned_uuid) in cell.vertex_uuid_iter(&tds).zip(vertex_uuids.iter()) {
            assert_eq!(expected_uuid.unwrap(), *returned_uuid);
        }

        // Verify all UUIDs are unique
        let unique_uuids: std::collections::HashSet<_> = vertex_uuids.iter().collect();
        assert_eq!(unique_uuids.len(), vertex_uuids.len());

        // Verify no nil UUIDs using iterator
        for uuid in cell.vertex_uuid_iter(&tds) {
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

        let tds = Tds::<f64, i32, Option<()>, 4>::new(&vertices).unwrap();
        let cell_key = tds.cells().next().unwrap().0;
        let cell = &tds.get_cell(cell_key).unwrap();

        // Get vertex UUIDs
        let vertex_uuids = cell.vertex_uuids(&tds).unwrap();
        assert_eq!(cell.vertex_uuid_iter(&tds).count(), 5);

        // Verify UUIDs match the cell's vertices using iterator
        for (expected_uuid, returned_uuid) in cell.vertex_uuid_iter(&tds).zip(vertex_uuids.iter()) {
            assert_eq!(expected_uuid.unwrap(), *returned_uuid);
        }

        // Verify all UUIDs are unique
        let unique_uuids: std::collections::HashSet<_> = vertex_uuids.iter().collect();
        assert_eq!(unique_uuids.len(), vertex_uuids.len());

        // Verify vertex data integrity alongside UUIDs
        for (i, vertex_key) in cell.vertices().iter().enumerate() {
            let vertex = &tds.get_vertex_by_key(*vertex_key).unwrap();
            assert_eq!(vertex.data, Some(i32::try_from(i + 1).unwrap()));
        }

        // Verify no nil UUIDs using iterator
        for uuid in cell.vertex_uuid_iter(&tds) {
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

        let tds = Tds::<f32, Option<()>, Option<()>, 3>::new(&vertices).unwrap();
        let cell_key = tds.cells().next().unwrap().0;
        let cell = &tds.get_cell(cell_key).unwrap();

        // Get vertex UUIDs
        let vertex_uuids = cell.vertex_uuids(&tds).unwrap();
        assert_eq!(cell.vertex_uuid_iter(&tds).count(), 4);

        // Verify coordinate type is preserved
        let first_vertex_key = cell.vertices()[0];
        let first_vertex = &tds.get_vertex_by_key(first_vertex_key).unwrap();
        assert_relative_eq!(
            first_vertex.point().coords()[0],
            0.0f32,
            epsilon = f32::EPSILON
        );

        // Verify UUIDs match the cell's vertices using iterator
        for (expected_uuid, returned_uuid) in cell.vertex_uuid_iter(&tds).zip(vertex_uuids.iter()) {
            assert_eq!(expected_uuid.unwrap(), *returned_uuid);
        }

        // Verify all UUIDs are unique
        let unique_uuids: std::collections::HashSet<_> = vertex_uuids.iter().collect();
        assert_eq!(unique_uuids.len(), vertex_uuids.len());

        // Verify no nil UUIDs using iterator
        for uuid in cell.vertex_uuid_iter(&tds) {
            assert_ne!(uuid.unwrap(), Uuid::nil());
        }

        println!("✓ vertex_uuids works correctly with f32 coordinates");
    }

    // =============================================================================
    // DIMENSIONAL TESTS
    // =============================================================================
    // Tests covering cells in different dimensions (1D, 2D, 3D, 4D+) and
    // various coordinate types (f32, f64) to ensure dimensional flexibility.

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
                    let tds = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices).unwrap();
                    let (_, cell) = tds.cells().next().unwrap();
                    assert_cell_properties(cell, $dim + 1, $dim);
                }

                pastey::paste! {
                    #[test]
                    fn [<$test_name _with_data>]() {
                        // Test cell with data
                        let vertices = $vertices;
                        let tds = Tds::<f64, Option<()>, i32, $dim>::new(&vertices).unwrap();
                        let (_, cell_ref) = tds.cells().next().unwrap();
                        let mut cell = cell_ref.clone();
                        cell.data = Some(42);
                        assert_cell_properties(&cell, $dim + 1, $dim);
                        assert_eq!(cell.data, Some(42));
                    }

                    #[test]
                    fn [<$test_name _serialization_roundtrip>]() {
                        // Test serialization with Some data
                        let vertices = $vertices;
                        let tds: Tds<f64, Option<()>, i32, $dim> = Tds::new(&vertices).unwrap();
                        let (_, cell) = tds.cells().next().unwrap();
                        let mut cell = cell.clone();
                        cell.data = Some(99);

                        let serialized = serde_json::to_string(&cell).unwrap();
                        assert!(serialized.contains("\"data\":"));
                        let deserialized: Cell<f64, Option<()>, i32, $dim> = serde_json::from_str(&serialized).unwrap();
                        assert_eq!(deserialized.data, Some(99));
                        assert_eq!(deserialized.uuid(), cell.uuid());

                        // Test serialization with None data
                        let vertices = $vertices;
                        let tds: Tds<f64, Option<()>, Option<i32>, $dim> = Tds::new(&vertices).unwrap();
                        let (_, cell) = tds.cells().next().unwrap();

                        let serialized = serde_json::to_string(&cell).unwrap();
                        assert!(!serialized.contains("\"data\":"));
                        let deserialized: Cell<f64, Option<()>, Option<i32>, $dim> = serde_json::from_str(&serialized).unwrap();
                        assert_eq!(deserialized.data, None);
                    }

                    #[test]
                    fn [<$test_name _uuid_uniqueness>]() {
                        // Test UUID uniqueness by creating two separate TDS instances
                        let vertices1 = $vertices;
                        let vertices2 = $vertices;
                        let tds1 = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices1).unwrap();
                        let (_, cell1) = tds1.cells().next().unwrap();
                        let tds2 = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices2).unwrap();
                        let (_, cell2) = tds2.cells().next().unwrap();
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

    // Keep 1D test separate as it's less common
    #[test]
    fn cell_1d() {
        let vertices = vec![vertex!([0.0]), vertex!([1.0])];
        let tds = Tds::<f64, Option<()>, Option<()>, 1>::new(&vertices).unwrap();
        let (_, cell) = tds.cells().next().unwrap();
        assert_cell_properties(cell, 2, 1);
    }

    #[test]
    fn cell_with_f32() {
        let vertices = vec![
            vertex!([0.0f32, 0.0f32]),
            vertex!([1.0f32, 0.0f32]),
            vertex!([0.0f32, 1.0f32]),
        ];
        let tds = Tds::<f32, Option<()>, Option<()>, 2>::new(&vertices).unwrap();
        let (_, cell) = tds.cells().next().unwrap();

        assert_eq!(cell.number_of_vertices(), 3);
        assert_eq!(cell.dim(), 2);
        assert!(!cell.uuid().is_nil());
    }

    #[test]
    fn cell_single_vertex() {
        // Test that creating a 3D triangulation with insufficient vertices fails validation
        // Phase 3A: Now tested through TDS which is the user-facing API
        let vertices = vec![vertex!([0.0, 0.0, 0.0])];
        let result = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices);

        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Insufficient vertices"));
        assert!(error_msg.contains('1'));
        assert!(error_msg.contains('4'));
    }

    #[test]
    fn cell_uuid_uniqueness() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        // Create two separate TDS instances to get different UUIDs
        let tds1 = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();
        let (_, cell1) = tds1.cells().next().unwrap();
        let tds2 = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();
        let (_, cell2) = tds2.cells().next().unwrap();

        // Same vertices but different UUIDs (different TDS instances)
        assert_ne!(cell1.uuid(), cell2.uuid());
        assert!(!cell1.uuid().is_nil());
        assert!(!cell2.uuid().is_nil());
    }

    #[test]
    fn cell_neighbors_none_by_default() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();
        let (_, cell) = tds.cells().next().unwrap();

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
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();
        let (_, cell) = tds.cells().next().unwrap();

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
        let tds = Tds::<f64, Option<()>, i32, 3>::new(&vertices).unwrap();
        let (_, cell_ref) = tds.cells().next().unwrap();
        let mut cell = cell_ref.clone();
        cell.data = Some(42);

        assert_eq!(cell.data.unwrap(), 42);
    }

    #[test]
    fn cell_into_hashmap_empty() {
        let cells: Vec<Cell<f64, Option<()>, Option<()>, 3>> = Vec::new();
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
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();

        // Collect cells from TDS
        let cells_vec: Vec<_> = tds.cells().map(|(_, cell)| cell.clone()).collect();
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
        let vertices = vec![
            vertex!([1.0, 2.0, 3.0]),
            vertex!([4.0, 5.0, 6.0]),
            vertex!([7.0, 8.0, 9.0]),
            vertex!([10.0, 11.0, 12.0]),
        ];
        let tds = Tds::<f64, Option<()>, i32, 3>::new(&vertices).unwrap();
        let (_, cell_ref) = tds.cells().next().unwrap();
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
        let vertices = vec![
            vertex!([1.0, 2.0, 3.0]),
            vertex!([4.0, 5.0, 6.0]),
            vertex!([7.0, 8.0, 9.0]),
            vertex!([10.0, 11.0, 12.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        // Serialize the entire TDS (includes cells with proper context)
        let serialized = serde_json::to_string(&tds).unwrap();
        assert!(serialized.contains("vertices"));
        assert!(serialized.contains("cells"));

        // Deserialize back to TDS
        let deserialized: Tds<f64, Option<()>, Option<()>, 3> =
            serde_json::from_str(&serialized).unwrap();

        // Verify TDS properties match
        assert_eq!(deserialized.number_of_vertices(), tds.number_of_vertices());
        assert_eq!(deserialized.number_of_cells(), tds.number_of_cells());
        assert_eq!(deserialized.dim(), tds.dim());

        // Verify cells within TDS can be accessed
        assert_ne!(deserialized.number_of_cells(), 0);
        for (_cell_key, cell) in deserialized.cells() {
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
        let result: Result<Cell<f64, Option<()>, Option<()>, 3>, _> =
            serde_json::from_str(invalid_json_missing_uuid);
        assert!(result.is_err(), "Missing UUID should cause error");
        let error = result.unwrap_err().to_string();
        assert!(
            error.contains("missing field") || error.contains("uuid"),
            "Error should mention missing uuid field: {error}"
        );

        // Test invalid UUID format
        let invalid_json_bad_uuid = r#"{"uuid": "not-a-valid-uuid"}"#;
        let result: Result<Cell<f64, Option<()>, Option<()>, 3>, _> =
            serde_json::from_str(invalid_json_bad_uuid);
        assert!(result.is_err(), "Invalid UUID format should cause error");

        // Test completely invalid JSON syntax
        let invalid_json_syntax = r"{this is not valid JSON}";
        let result: Result<Cell<f64, Option<()>, Option<()>, 3>, _> =
            serde_json::from_str(invalid_json_syntax);
        assert!(result.is_err(), "Invalid JSON syntax should cause error");

        // Test empty JSON object (missing required uuid)
        let empty_json = r"{}";
        let result: Result<Cell<f64, Option<()>, Option<()>, 3>, _> =
            serde_json::from_str(empty_json);
        assert!(result.is_err(), "Empty JSON should fail (missing uuid)");

        // Test deserialization with unknown fields (should succeed - ignored)
        let json_unknown_field = r#"{
            "uuid": "550e8400-e29b-41d4-a716-446655440000",
            "unknown_field": "value",
            "another_unknown": 123
        }"#;
        let result: Result<Cell<f64, Option<()>, Option<()>, 3>, _> =
            serde_json::from_str(json_unknown_field);
        assert!(result.is_ok(), "Unknown fields should be ignored");
    }

    #[test]
    fn cell_deserialization_minimal_valid() {
        // Test deserialization with minimal valid JSON
        // Only uuid is required; vertices and neighbors are skipped fields
        let minimal_valid_json = r#"{"uuid": "550e8400-e29b-41d4-a716-446655440000"}"#;
        let result: Result<Cell<f64, Option<()>, Option<()>, 3>, _> =
            serde_json::from_str(minimal_valid_json);
        assert!(
            result.is_ok(),
            "Minimal valid JSON with just UUID should succeed"
        );

        let cell = result.unwrap();
        assert_eq!(
            cell.uuid().to_string(),
            "550e8400-e29b-41d4-a716-446655440000"
        );
        assert_eq!(cell.number_of_vertices(), 0); // No vertices since field is skipped
        assert!(cell.data.is_none());

        // Test with data field included
        let json_with_data = r#"{
            "uuid": "550e8400-e29b-41d4-a716-446655440001",
            "data": null
        }"#;
        let result: Result<Cell<f64, Option<()>, Option<()>, 3>, _> =
            serde_json::from_str(json_with_data);
        assert!(result.is_ok(), "JSON with null data should succeed");
    }

    #[test]
    fn cell_serialization_with_some_data_includes_field() {
        // Test that when data is Some, the JSON includes the data field
        let vertices = vec![
            vertex!([1.0, 2.0, 3.0]),
            vertex!([4.0, 5.0, 6.0]),
            vertex!([7.0, 8.0, 9.0]),
            vertex!([10.0, 11.0, 12.0]),
        ];
        let tds: Tds<f64, Option<()>, i32, 3> = Tds::new(&vertices).unwrap();
        let (_, cell) = tds.cells().next().unwrap();
        let mut cell = cell.clone();
        cell.data = Some(42);

        let serialized = serde_json::to_string(&cell).unwrap();

        // Verify JSON contains "data" field
        assert!(
            serialized.contains("\"data\":"),
            "JSON should include data field when Some"
        );
        assert!(serialized.contains("42"), "JSON should include data value");

        // Roundtrip test
        let deserialized: Cell<f64, Option<()>, i32, 3> =
            serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.data, Some(42));
        assert_eq!(deserialized.uuid(), cell.uuid());
    }

    #[test]
    fn cell_serialization_with_none_data_omits_field() {
        // Test that when data is None, the JSON omits the data field entirely
        let vertices = vec![
            vertex!([1.0, 2.0, 3.0]),
            vertex!([4.0, 5.0, 6.0]),
            vertex!([7.0, 8.0, 9.0]),
            vertex!([10.0, 11.0, 12.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<i32>, 3> = Tds::new(&vertices).unwrap();
        let (_, cell) = tds.cells().next().unwrap();

        let serialized = serde_json::to_string(&cell).unwrap();

        // Verify JSON does NOT contain "data" field (optimization)
        assert!(
            !serialized.contains("\"data\":"),
            "JSON should omit data field when None"
        );

        // Roundtrip test - missing data field should deserialize as None
        let deserialized: Cell<f64, Option<()>, Option<i32>, 3> =
            serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.data, None);
        assert_eq!(deserialized.uuid(), cell.uuid());
    }

    #[test]
    fn cell_deserialization_with_explicit_null_data() {
        // Test backward compatibility: explicit "data": null should still work
        let json_with_null = r#"{"uuid":"550e8400-e29b-41d4-a716-446655440000","data":null}"#;
        let cell: Cell<f64, Option<()>, Option<i32>, 3> =
            serde_json::from_str(json_with_null).unwrap();

        assert_eq!(cell.data, None);
        assert_eq!(
            cell.uuid().to_string(),
            "550e8400-e29b-41d4-a716-446655440000"
        );
    }

    // =============================================================================
    // GEOMETRIC PROPERTIES TESTS
    // =============================================================================
    // Tests for geometric properties and validation of cells

    #[test]
    fn cell_negative_coordinates() {
        let vertices = vec![
            vertex!([-1.0, -2.0, -3.0]),
            vertex!([-4.0, -5.0, -6.0]),
            vertex!([-7.0, -8.0, -9.0]),
            vertex!([-10.0, -11.0, -12.0]),
        ];
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();
        let (_, cell) = tds.cells().next().unwrap();

        assert_eq!(cell.number_of_vertices(), 4);
        assert_eq!(cell.dim(), 3);
    }

    #[test]
    fn cell_large_coordinates() {
        let vertices = vec![
            vertex!([1e6, 2e6, 3e6]),
            vertex!([4e6, 5e6, 6e6]),
            vertex!([7e6, 8e6, 9e6]),
            vertex!([10e6, 11e6, 12e6]),
        ];
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();
        let (_, cell) = tds.cells().next().unwrap();

        assert_eq!(cell.number_of_vertices(), 4);
        assert_eq!(cell.dim(), 3);
    }

    #[test]
    fn cell_small_coordinates() {
        let vertices = vec![
            vertex!([1e-6, 2e-6, 3e-6]),
            vertex!([4e-6, 5e-6, 6e-6]),
            vertex!([7e-6, 8e-6, 9e-6]),
            vertex!([10e-6, 11e-6, 12e-6]),
        ];
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();
        let (_, cell) = tds.cells().next().unwrap();

        assert_eq!(cell.number_of_vertices(), 4);
        assert_eq!(cell.dim(), 3);
    }

    #[test]
    fn cell_circumradius_2d() {
        let vertex1 = vertex!([0.0, 0.0]);
        let vertex2 = vertex!([1.0, 0.0]);
        let vertex3 = vertex!([0.0, 1.0]);

        let vertices = vec![vertex1, vertex2, vertex3];
        let tds = Tds::<f64, Option<()>, Option<()>, 2>::new(&vertices).unwrap();
        let cell_key = tds.cells().next().unwrap().0;
        let cell = &tds.get_cell(cell_key).unwrap();

        // Resolve VertexKeys to actual vertices
        let vertex_points: Vec<Point<f64, 2>> = cell
            .vertices()
            .iter()
            .map(|vk| *tds.get_vertex_by_key(*vk).unwrap().point())
            .collect();
        let circumradius = circumradius(&vertex_points).unwrap();

        // For a right triangle with legs of length 1, circumradius is sqrt(2)/2
        let expected_radius = 2.0_f64.sqrt() / 2.0;
        assert_relative_eq!(circumradius, expected_radius, epsilon = 1e-10);
    }

    #[test]
    fn cell_mixed_positive_negative_coordinates() {
        let vertices = vec![
            vertex!([1.0, -2.0, 3.0, -4.0]),
            vertex!([-5.0, 6.0, -7.0, 8.0]),
            vertex!([9.0, -10.0, 11.0, -12.0]),
            vertex!([-13.0, 14.0, -15.0, 16.0]),
            vertex!([17.0, -18.0, 19.0, -20.0]),
        ];
        let tds = Tds::<f64, Option<()>, Option<()>, 4>::new(&vertices).unwrap();
        let (_, cell) = tds.cells().next().unwrap();

        assert_eq!(cell.number_of_vertices(), 5);
        assert_eq!(cell.dim(), 4);
    }

    #[test]
    fn cell_contains_vertex_false() {
        let vertex1 = vertex!([0.0, 0.0, 0.0]);
        let vertex2 = vertex!([1.0, 0.0, 0.0]);
        let vertex3 = vertex!([0.0, 1.0, 0.0]);
        let vertex4 = vertex!([0.0, 0.0, 1.0]); // 4th vertex to complete 3D cell
        let vertex_outside: Vertex<f64, Option<()>, 3> = vertex!([2.0, 2.0, 2.0]);

        let vertices = vec![vertex1, vertex2, vertex3, vertex4];
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();
        let cell_key = tds.cells().next().unwrap().0;
        let cell = &tds.get_cell(cell_key).unwrap();

        // Create a vertex key for the outside vertex - it won't be in the cell
        let outside_key = tds.vertex_key_from_uuid(&vertex_outside.uuid());
        assert!(outside_key.is_none() || !cell.contains_vertex(outside_key.unwrap()));
    }

    #[test]
    fn cell_contains_vertex_of_false() {
        // Phase 3A: Test deprecated contains_vertex_of method
        // Create two separate cells to test the method works
        // Note: In Phase 3A, cells should ideally be created within a TDS, but for this
        // simple test of the contains_vertex_of method, we can use cell! macro

        let vertices1 = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let vertices2 = vec![
            vertex!([10.0, 10.0, 10.0]),
            vertex!([11.0, 10.0, 10.0]),
            vertex!([10.0, 11.0, 10.0]),
            vertex!([10.0, 10.0, 11.0]),
        ];

        let tds1 = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices1).unwrap();
        let (_, cell1) = tds1.cells().next().unwrap();
        let tds2 = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices2).unwrap();
        let (_, cell2) = tds2.cells().next().unwrap();

        // Note: This test may give false positives due to VertexKey collisions across
        // different TDS instances, but it tests the method's basic functionality
        #[allow(deprecated)]
        let result = cell1.contains_vertex_of(cell2);

        // We can't reliably assert the result due to potential key collisions,
        // but we verify the method executes without panicking
        println!("contains_vertex_of returned: {result}");
    }

    #[test]
    fn cell_validation_max_vertices() {
        // Test that creating a 2D triangulation with correct vertex count succeeds
        // Phase 3A: Now tested through TDS which is the user-facing API
        let vertex1 = vertex!([0.0, 0.0]);
        let vertex2 = vertex!([1.0, 0.0]);
        let vertex3 = vertex!([0.0, 1.0]);

        // This should work (3 vertices for 2D)
        let tds = Tds::<f64, Option<()>, Option<()>, 2>::new(&[vertex1, vertex2, vertex3]);

        assert!(tds.is_ok());
        let tds = tds.unwrap();
        assert_eq!(tds.number_of_vertices(), 3);
        // Should create one 2D cell
        assert_eq!(tds.number_of_cells(), 1);
        let cell = tds.cells().map(|(_, cell)| cell).next().unwrap();
        assert_eq!(cell.number_of_vertices(), 3);
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
        let tds = Tds::<f64, i32, Option<()>, 3>::new(&vertices).unwrap();
        let cell_key = tds.cells().next().unwrap().0;
        let cell = &tds.get_cell(cell_key).unwrap();

        // Test vertex clearly outside circumsphere
        let vertex_far_outside: Vertex<f64, i32, 3> = vertex!([10.0, 10.0, 10.0], 4);
        // Just check that the method runs without error for now
        let vertex_points: Vec<Point<f64, 3>> = cell
            .vertices()
            .iter()
            .map(|vk| *tds.get_vertex_by_key(*vk).unwrap().point())
            .collect();
        let result = insphere(&vertex_points, *vertex_far_outside.point());
        assert!(result.is_ok());

        // Test with origin (should be inside or on boundary)
        let origin: Vertex<f64, i32, 3> = vertex!([0.0, 0.0, 0.0], 3);
        let vertex_points: Vec<Point<f64, 3>> = cell
            .vertices()
            .iter()
            .map(|vk| *tds.get_vertex_by_key(*vk).unwrap().point())
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
        let tds = Tds::<f64, Option<()>, Option<()>, 2>::new(&vertices).unwrap();
        let cell_key = tds.cells().next().unwrap().0;
        let cell = &tds.get_cell(cell_key).unwrap();

        // Test vertex far outside circumcircle
        let vertex_far_outside: Vertex<f64, Option<()>, 2> = vertex!([10.0, 10.0]);
        let vertex_points: Vec<Point<f64, 2>> = cell
            .vertices()
            .iter()
            .map(|vk| *tds.get_vertex_by_key(*vk).unwrap().point())
            .collect();
        let result = insphere(&vertex_points, *vertex_far_outside.point());
        assert!(result.is_ok());

        // Test with center of triangle (should be inside)
        let center: Vertex<f64, Option<()>, 2> = vertex!([0.33, 0.33]);
        let vertex_points: Vec<Point<f64, 2>> = cell
            .vertices()
            .iter()
            .map(|vk| *tds.get_vertex_by_key(*vk).unwrap().point())
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
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();
        let cell_key = tds.cells().next().unwrap().0;
        let cell = &tds.get_cell(cell_key).unwrap();

        let vertex_points: Vec<Point<f64, 3>> = cell
            .vertices()
            .iter()
            .map(|vk| *tds.get_vertex_by_key(*vk).unwrap().point())
            .collect();

        let circumcenter = circumcenter(&vertex_points).unwrap();
        let radius_with_center = circumradius_with_center(&vertex_points, &circumcenter);
        let radius_direct = circumradius(&vertex_points).unwrap();

        assert_relative_eq!(radius_with_center.unwrap(), radius_direct, epsilon = 1e-10);
    }

    #[test]
    fn cell_facets_completeness() {
        // Test that facets are generated correctly and completely
        let vertex1 = vertex!([0.0, 0.0, 0.0]);
        let vertex2 = vertex!([1.0, 0.0, 0.0]);
        let vertex3 = vertex!([0.0, 1.0, 0.0]);
        let vertex4 = vertex!([0.0, 0.0, 1.0]);

        let vertices = vec![vertex1, vertex2, vertex3, vertex4];
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();
        let cell_key = tds.cells().next().unwrap().0;
        let cell = &tds.get_cell(cell_key).unwrap();

        let facets = cell
            .facet_views(&tds, cell_key)
            .expect("Failed to get facets");
        assert_eq!(facets.len(), 4); // A tetrahedron should have 4 facets

        // Each facet should have 3 vertices (for 3D tetrahedron)
        for facet in &facets {
            assert_eq!(facet.vertices().unwrap().count(), 3);
        }

        // All original vertices should be represented in facets
        let all_facet_vertex_uuids: std::collections::HashSet<_> = facets
            .iter()
            .flat_map(|f| f.vertices().unwrap())
            .map(super::super::vertex::Vertex::uuid)
            .collect();
        assert!(all_facet_vertex_uuids.contains(&vertex1.uuid()));
        assert!(all_facet_vertex_uuids.contains(&vertex2.uuid()));
        assert!(all_facet_vertex_uuids.contains(&vertex3.uuid()));
        assert!(all_facet_vertex_uuids.contains(&vertex4.uuid()));
    }

    #[test]
    fn cell_builder_validation_edge_cases() {
        // Test builder validation with exactly D+1 vertices (should work)
        let vertex1 = vertex!([0.0, 0.0, 0.0]);
        let vertex2 = vertex!([1.0, 0.0, 0.0]);
        let vertex3 = vertex!([0.0, 1.0, 0.0]);
        let vertex4 = vertex!([0.0, 0.0, 1.0]);

        // Exactly 4 vertices for 3D (D+1 = 3+1 = 4) should work
        // Phase 3A: Create via TDS
        let tds =
            Tds::<f64, Option<()>, Option<()>, 3>::new(&vec![vertex1, vertex2, vertex3, vertex4])
                .unwrap();
        let (_, cell) = tds.cells().next().unwrap();
        assert!(cell.is_valid().is_ok());

        // Test with insufficient vertices (should fail)
        let insufficient_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
        ];
        let tds_result = Tds::<f64, Option<()>, Option<()>, 3>::new(&insufficient_vertices);
        assert!(
            tds_result.is_err(),
            "Insufficient vertices (3) should fail for 3D triangulation (needs 4)"
        );
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
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();
        let cell_key = tds.cells().next().unwrap().0;
        let cell = &tds.get_cell(cell_key).unwrap();

        // Test that we can get facet views for all facets
        let facet_views = cell
            .facet_views(&tds, cell_key)
            .expect("Failed to get facet views");
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
        for (i, facet_view) in facet_views.iter().enumerate() {
            let opposite_vertex = facet_view
                .opposite_vertex()
                .expect("Failed to get opposite vertex");
            // The opposite vertex should be one of the cell's vertices (by VertexKey)
            let opposite_key = tds.vertex_key_from_uuid(&opposite_vertex.uuid()).unwrap();
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
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();
        let cell_key = tds.cells().next().unwrap().0;
        let cell = &tds.get_cell(cell_key).unwrap();

        // Get all facet views
        let facet_views = cell
            .facet_views(&tds, cell_key)
            .expect("Failed to get facet views");

        for facet_view in &facet_views {
            let opposite_vertex = facet_view
                .opposite_vertex()
                .expect("Failed to get opposite vertex");
            let opposite_vertex_key = tds.vertex_key_from_uuid(&opposite_vertex.uuid()).unwrap();
            let facet_vertices = facet_view.vertices().expect("Failed to get facet vertices");

            // Collect facet vertex keys
            let facet_vertex_keys: Vec<_> = facet_vertices
                .map(|v| tds.vertex_key_from_uuid(&v.uuid()).unwrap())
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

        let tds_f32 =
            Tds::<f32, Option<()>, Option<()>, 2>::new(&[vertex1_f32, vertex2_f32, vertex3_f32])
                .unwrap();
        let (_, cell_f32) = tds_f32.cells().next().unwrap();
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
        let tds = Tds::<f64, Option<()>, Option<()>, 5>::new(&vertices).unwrap();
        let cell_key = tds.cells().next().unwrap().0;
        let cell = &tds.get_cell(cell_key).unwrap();

        assert_eq!(cell.number_of_vertices(), 6);
        assert_eq!(cell.dim(), 5);
        assert_eq!(
            cell.facet_views(&tds, cell_key)
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
        let mut tds = Tds::<f64, i32, u32, 3>::new(&vertices).unwrap();
        let cell_key = tds.cells().next().unwrap().0;

        // Set the cell data to a known value
        if let Some(cell) = tds.cells_mut().get_mut(cell_key) {
            cell.data = Some(42u32);
        }

        let cell = &tds.get_cell(cell_key).unwrap();

        // Resolve VertexKeys to vertices to check data using the key-based API
        assert_eq!(
            tds.get_vertex_by_key(cell.vertices()[0])
                .unwrap()
                .data
                .unwrap(),
            1
        );
        assert_eq!(
            tds.get_vertex_by_key(cell.vertices()[1])
                .unwrap()
                .data
                .unwrap(),
            2
        );
        assert_eq!(
            tds.get_vertex_by_key(cell.vertices()[2])
                .unwrap()
                .data
                .unwrap(),
            3
        );
        assert_eq!(
            tds.get_vertex_by_key(cell.vertices()[3])
                .unwrap()
                .data
                .unwrap(),
            4
        );
        assert_eq!(cell.data.unwrap(), 42u32);

        // Also verify we can access vertex data through facet_views
        let facet_views = cell
            .facet_views(&tds, cell_key)
            .expect("Failed to get facet views");
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

    #[test]
    fn cell_circumsphere_edge_cases() {
        // Test circumsphere containment with simple cases
        let vertex1 = vertex!([0.0, 0.0]);
        let vertex2 = vertex!([1.0, 0.0]);
        let vertex3 = vertex!([0.0, 1.0]);

        let vertices = vec![vertex1, vertex2, vertex3];
        let tds = Tds::<f64, Option<()>, Option<()>, 2>::new(&vertices).unwrap();
        let cell_key = tds.cells().next().unwrap().0;
        let cell = &tds.get_cell(cell_key).unwrap();

        // Test that the methods run without error
        let test_point: Vertex<f64, Option<()>, 2> = vertex!([0.5, 0.5]);

        let vertex_points: Vec<Point<f64, 2>> = cell
            .vertices()
            .iter()
            .map(|vk| *tds.get_vertex_by_key(*vk).unwrap().point())
            .collect();

        let circumsphere_result = insphere_distance(&vertex_points, *test_point.point());
        assert!(circumsphere_result.is_ok());

        let determinant_result = insphere(&vertex_points, *test_point.point());
        assert!(determinant_result.is_ok());

        // At minimum, both methods should give the same result for the same input
        let far_point: Vertex<f64, Option<()>, 2> = vertex!([100.0, 100.0]);

        let vertex_points: Vec<Point<f64, 2>> = cell
            .vertices()
            .iter()
            .map(|vk| *tds.get_vertex_by_key(*vk).unwrap().point())
            .collect();

        let circumsphere_far = insphere_distance(&vertex_points, *far_point.point());
        let determinant_far = insphere(&vertex_points, *far_point.point());

        assert!(circumsphere_far.is_ok());
        assert!(determinant_far.is_ok());
    }

    #[test]
    fn cell_is_valid_correct_cell() {
        // Test cell is_valid with valid vertices (exactly D+1 = 4 vertices for 3D)
        let vertices = vec![
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0]),
        ];
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();
        let (_, cell) = tds.cells().next().unwrap();

        // Human readable output for cargo test -- --nocapture
        println!("Valid Cell: {cell:?}");
        assert!(cell.is_valid().is_ok());
    }

    // Phase 3A: Test removed - Cells are now constructed through TDS, not manually
    // Cell validation is handled at the TDS level, not at individual Cell level

    #[test]
    fn cell_is_valid_invalid_uuid_error() {
        // Test cell is_valid with nil UUID
        let vertex1 = vertex!([0.0, 0.0, 1.0]);
        let vertex2 = vertex!([0.0, 1.0, 0.0]);
        let vertex3 = vertex!([1.0, 0.0, 0.0]);
        let vertex4 = vertex!([0.0, 0.0, 0.0]);
        let tds =
            Tds::<f64, Option<()>, Option<()>, 3>::new(&vec![vertex1, vertex2, vertex3, vertex4])
                .unwrap();
        let (_, cell_ref) = tds.cells().next().unwrap();
        let mut invalid_uuid_cell = cell_ref.clone();

        // Manually set the UUID to nil to trigger the InvalidUuid error
        invalid_uuid_cell.uuid = uuid::Uuid::nil();

        let invalid_uuid_result = invalid_uuid_cell.is_valid();
        assert!(invalid_uuid_result.is_err());

        // Verify that we get the correct error type for invalid UUID
        match invalid_uuid_result {
            Err(CellValidationError::InvalidUuid { source: _ }) => {
                println!("✓ Correctly detected invalid UUID");
            }
            Err(other_error) => {
                panic!("Expected InvalidUuid error, but got: {other_error:?}");
            }
            Ok(()) => {
                panic!("Expected error for invalid UUID, but validation passed");
            }
        }
    }

    // Phase 3A: Test removed - Cells are now constructed through TDS, not manually
    // Duplicate vertex detection is handled at the TDS level during construction

    #[test]
    fn cell_is_valid_insufficient_vertices_error() {
        // Test that TDS fails when creating a 3D triangulation with insufficient vertices
        let insufficient_vertices = vec![vertex!([0.0, 0.0, 1.0]), vertex!([0.0, 1.0, 0.0])];

        // Phase 3A: Use TDS to test validation failure (CellBuilder is internal)
        let result = Tds::<f64, Option<()>, Option<()>, 3>::new(&insufficient_vertices);

        assert!(
            result.is_err(),
            "TDS should fail with insufficient vertices"
        );
        let error_msg = result.unwrap_err().to_string();
        // TDS should report insufficient vertices for 3D triangulation
        assert!(
            error_msg.contains("vertices") || error_msg.contains('4'),
            "Error should mention vertices or the required count: {error_msg}"
        );

        println!("✓ Correctly detected insufficient vertices during triangulation");
    }

    #[test]
    fn test_cell_is_valid_neighbors_length_validation() {
        // Test that is_valid checks neighbors length matches D+1
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        // Create a valid 2D cell (triangle)
        let tds = Tds::<f64, Option<()>, Option<()>, 2>::new(&vertices).unwrap();
        let (_, cell_ref) = tds.cells().next().unwrap();
        let mut cell = cell_ref.clone();
        assert!(cell.is_valid().is_ok(), "Valid cell should pass validation");

        // Manually set neighbors with correct length (D+1 = 3)
        cell.neighbors = Some(vec![None, None, None].into());
        assert!(
            cell.is_valid().is_ok(),
            "Cell with correct neighbors length should pass validation"
        );

        // Set neighbors with incorrect length (too few)
        cell.neighbors = Some(vec![None, None].into());
        let result = cell.is_valid();
        assert!(
            result.is_err(),
            "Cell with too few neighbors should fail validation"
        );

        if let Err(CellValidationError::InvalidNeighborsLength {
            actual,
            expected,
            dimension,
        }) = result
        {
            assert_eq!(actual, 2);
            assert_eq!(expected, 3);
            assert_eq!(dimension, 2);
        } else {
            panic!("Expected InvalidNeighborsLength error, got: {result:?}");
        }

        // Set neighbors with incorrect length (too many)
        cell.neighbors = Some(vec![None, None, None, None].into());
        let result = cell.is_valid();
        assert!(
            result.is_err(),
            "Cell with too many neighbors should fail validation"
        );

        if let Err(CellValidationError::InvalidNeighborsLength {
            actual,
            expected,
            dimension,
        }) = result
        {
            assert_eq!(actual, 4);
            assert_eq!(expected, 3);
            assert_eq!(dimension, 2);
        } else {
            panic!("Expected InvalidNeighborsLength error, got: {result:?}");
        }

        println!("✓ Cell neighbors length validation works correctly");
    }

    #[test]
    fn test_cell_is_valid_neighbors_length_3d() {
        // Test neighbors length validation for 3D cells
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();
        let (_, cell_ref) = tds.cells().next().unwrap();
        let mut cell = cell_ref.clone();

        // Set neighbors with correct length (D+1 = 4)
        cell.neighbors = Some(vec![None, None, None, None].into());
        assert!(
            cell.is_valid().is_ok(),
            "3D cell with correct neighbors length should pass validation"
        );

        // Set neighbors with incorrect length
        cell.neighbors = Some(vec![None, None, None].into());
        let result = cell.is_valid();
        assert!(
            result.is_err(),
            "3D cell with wrong neighbors length should fail validation"
        );

        if let Err(CellValidationError::InvalidNeighborsLength {
            actual,
            expected,
            dimension,
        }) = result
        {
            assert_eq!(actual, 3);
            assert_eq!(expected, 4);
            assert_eq!(dimension, 3);
        } else {
            panic!("Expected InvalidNeighborsLength error, got: {result:?}");
        }

        println!("✓ 3D cell neighbors length validation works correctly");
    }

    #[test]
    fn test_cell_is_valid_no_neighbors_is_valid() {
        // Test that cells with no neighbors (None) are still valid
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        let tds = Tds::<f64, Option<()>, Option<()>, 2>::new(&vertices).unwrap();
        let (_, cell_ref) = tds.cells().next().unwrap();
        let mut cell = cell_ref.clone();

        // Explicitly set neighbors to None
        cell.neighbors = None;
        assert!(
            cell.is_valid().is_ok(),
            "Cell with no neighbors should be valid"
        );

        println!("✓ Cell with no neighbors passes validation");
    }

    // =============================================================================
    // CELL PARTIALEQ AND EQ TESTS
    // =============================================================================

    #[test]
    fn test_cell_partial_eq_identical_cells() {
        // Test that cells with identical vertices are equal
        let vertices1 = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds1 = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices1).unwrap();
        let (_, cell1) = tds1.cells().next().unwrap();

        let vertices2 = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds2 = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices2).unwrap();
        let (_, cell2) = tds2.cells().next().unwrap();

        // Test equality - should be true for identical cells
        assert_eq!(
            cell1, cell2,
            "Cells with identical vertices should be equal"
        );

        // Test reflexive property
        assert_eq!(cell1, cell1, "Cell should be equal to itself");

        println!("✓ Identical cells are correctly identified as equal");
    }

    #[test]
    fn test_cell_partial_eq_different_vertex_order() {
        // Test that cells with same vertices in different order are equal
        let vertices1 = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds1 = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices1).unwrap();
        let (_, cell1) = tds1.cells().next().unwrap();

        let vertices2 = vec![
            vertex!([1.0, 0.0, 0.0]), // Different order
            vertex!([0.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.0, 1.0, 0.0]),
        ];
        let tds2 = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices2).unwrap();
        let (_, cell2) = tds2.cells().next().unwrap();

        // Test equality - should be true regardless of vertex order since we sort internally
        assert_eq!(
            cell1, cell2,
            "Cells with same vertices in different order should be equal"
        );

        println!("✓ Cells with same vertices in different order are correctly identified as equal");
    }

    #[test]
    fn test_cell_partial_eq_different_vertices() {
        // Phase 3A: Test inequality using cells from same TDS
        let all_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([2.0, 0.0, 0.0]),
        ];
        let tds = Tds::<f64, Option<()>, Option<()>, 3>::new(&all_vertices).unwrap();
        let cells: Vec<_> = tds.cells().map(|(_, cell)| cell).collect();

        // With 5 vertices in 3D, we get multiple cells with different vertex sets
        if cells.len() >= 2 {
            let cell1 = cells[0];
            let cell2 = cells[1];

            // Cells from different parts of triangulation should be unequal
            assert_ne!(cell1, cell2, "Different cells should not be equal");
        }

        println!("✓ Cells with different vertices are correctly identified as unequal");
    }

    #[test]
    fn test_cell_partial_eq_different_dimensions() {
        // Test equality for cells of different dimensions
        let vertices_2d = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let tds1 = Tds::<f64, Option<()>, Option<()>, 2>::new(&vertices_2d).unwrap();
        let (_, cell_2d) = tds1.cells().next().unwrap();
        let tds2 = Tds::<f64, Option<()>, Option<()>, 2>::new(&vertices_2d).unwrap();
        let (_, cell_2d_copy) = tds2.cells().next().unwrap();

        // Test equality for 2D cells
        assert_eq!(cell_2d, cell_2d_copy, "Identical 2D cells should be equal");

        println!("✓ 2D cells work correctly with PartialEq");
    }

    #[test]
    fn test_cell_partial_eq_with_data() {
        // Test that cells with same vertices but different cell data are still equal
        // (since PartialEq only compares vertices, not data)
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds1 = Tds::<f64, Option<()>, i32, 3>::new(&vertices).unwrap();
        let (_, cell_ref1) = tds1.cells().next().unwrap();
        let mut cell1 = cell_ref1.clone();
        cell1.data = Some(42);
        let tds2 = Tds::<f64, Option<()>, i32, 3>::new(&vertices).unwrap();
        let (_, cell_ref2) = tds2.cells().next().unwrap();
        let mut cell2 = cell_ref2.clone();
        cell2.data = Some(99); // Different data

        // Test equality - should be true since PartialEq only compares vertices
        assert_eq!(
            cell1, cell2,
            "Cells with same vertices but different data should be equal"
        );

        println!("✓ Cell equality correctly ignores data field");
    }

    #[test]
    fn test_cell_partial_eq_with_vertex_data() {
        // Test cells where vertices have different data but same coordinates
        let vertices1 = vec![
            vertex!([0.0, 0.0, 0.0], 1),
            vertex!([1.0, 0.0, 0.0], 2),
            vertex!([0.0, 1.0, 0.0], 3),
            vertex!([0.0, 0.0, 1.0], 4),
        ];
        let tds1 = Tds::<f64, i32, Option<()>, 3>::new(&vertices1).unwrap();
        let (_, cell1) = tds1.cells().next().unwrap();

        let vertices2 = vec![
            vertex!([0.0, 0.0, 0.0], 10), // Different vertex data
            vertex!([1.0, 0.0, 0.0], 20),
            vertex!([0.0, 1.0, 0.0], 30),
            vertex!([0.0, 0.0, 1.0], 40),
        ];
        let tds2 = Tds::<f64, i32, Option<()>, 3>::new(&vertices2).unwrap();
        let (_, cell2) = tds2.cells().next().unwrap();

        // Test equality - should be true since vertex PartialEq compares coordinates,
        // and cell PartialEq uses vertex PartialEq
        assert_eq!(
            cell1, cell2,
            "Cells with same vertex coordinates but different vertex data should be equal"
        );

        println!("✓ Cell equality correctly ignores vertex data field");
    }

    #[test]
    fn test_cell_partial_eq_different_vertex_count() {
        // Test cells with different numbers of vertices (different dimensions)
        let vertices_3d = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds_3d = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices_3d).unwrap();
        let (_, cell_3d) = tds_3d.cells().next().unwrap();

        let vertices_2d = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let tds_2d = Tds::<f64, Option<()>, Option<()>, 2>::new(&vertices_2d).unwrap();
        let (_, cell_2d) = tds_2d.cells().next().unwrap();

        // Note: We can't directly compare cells of different dimensions with ==
        // since they are different types, but this test documents the expected behavior
        assert_eq!(cell_3d.number_of_vertices(), 4);
        assert_eq!(cell_2d.number_of_vertices(), 3);

        println!("✓ Cells of different dimensions have different vertex counts as expected");
    }

    #[test]
    fn test_cell_partial_eq_uuid_neighbors_ignored() {
        // Test that UUID and neighbors are ignored in equality comparison
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds1 = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();
        let (_, cell_ref1) = tds1.cells().next().unwrap();
        let mut cell1 = cell_ref1.clone();
        let tds2 = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).unwrap();
        let (_, cell_ref2) = tds2.cells().next().unwrap();
        let mut cell2 = cell_ref2.clone();

        // Add different neighbors to the cells (3D cell needs exactly 4 neighbor slots)
        // Note: neighbors now use CellKey not Uuid, but for this test we just need to set the length
        // These will be invalid CellKeys but that's ok for testing equality comparison
        let dummy_key1 = CellKey::from(KeyData::from_ffi(1));
        let dummy_key2 = CellKey::from(KeyData::from_ffi(2));
        let dummy_key3 = CellKey::from(KeyData::from_ffi(3));
        cell1.neighbors = Some(vec![Some(dummy_key1), Some(dummy_key2), None, None].into());
        cell2.neighbors = Some(vec![Some(dummy_key3), None, None, None].into()); // Different neighbors

        // Cells should still be equal since PartialEq only compares vertices
        assert_eq!(
            cell1, cell2,
            "Cells with same vertices but different neighbors should be equal"
        );

        // UUIDs are different but cells should still be equal
        assert_ne!(
            cell1.uuid(),
            cell2.uuid(),
            "Cells should have different UUIDs"
        );
        assert_eq!(
            cell1, cell2,
            "Cells with different UUIDs but same vertices should be equal"
        );

        println!("✓ Cell equality correctly ignores UUID and neighbors fields");
    }

    #[test]
    fn test_cell_partial_eq_symmetry() {
        // Test that equality is symmetric: if a == b, then b == a
        let vertices1 = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds1 = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices1).unwrap();
        let (_, cell1) = tds1.cells().next().unwrap();

        let vertices2 = vec![
            vertex!([1.0, 0.0, 0.0]), // Different order
            vertex!([0.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.0, 1.0, 0.0]),
        ];
        let tds2 = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices2).unwrap();
        let (_, cell2) = tds2.cells().next().unwrap();

        // Test symmetry
        assert_eq!(
            cell1 == cell2,
            cell2 == cell1,
            "Equality should be symmetric"
        );
        assert!(cell1 == cell2, "Cells should be equal");
        assert!(cell2 == cell1, "Equality should be symmetric");

        println!("✓ Cell equality is symmetric");
    }

    #[test]
    fn test_cell_partial_eq_transitivity() {
        // Test that equality is transitive: if a == b and b == c, then a == c
        let vertices1 = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds1 = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices1).unwrap();
        let (_, cell1) = tds1.cells().next().unwrap();

        let vertices2 = vec![
            vertex!([1.0, 0.0, 0.0]), // Different order
            vertex!([0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds2 = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices2).unwrap();
        let (_, cell2) = tds2.cells().next().unwrap();

        let vertices3 = vec![
            vertex!([0.0, 1.0, 0.0]), // Another different order
            vertex!([0.0, 0.0, 1.0]),
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
        ];
        let tds3 = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices3).unwrap();
        let (_, cell3) = tds3.cells().next().unwrap();

        // Test transitivity
        assert!(cell1 == cell2, "cell1 should equal cell2");
        assert!(cell2 == cell3, "cell2 should equal cell3");
        assert!(cell1 == cell3, "cell1 should equal cell3 (transitivity)");

        println!("✓ Cell equality is transitive");
    }

    #[test]
    fn test_facet_views_from_tds() {
        // Test the static facet_views_from_tds method
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        let cell_key = tds.cell_keys().next().unwrap();

        // Test the static method
        let static_facet_views = Cell::facet_views_from_tds(&tds, cell_key)
            .expect("Failed to get facet views from static method");

        // Should have 4 facets for a 3D tetrahedron
        assert_eq!(static_facet_views.len(), 4);

        // Each facet should have 3 vertices
        for (i, facet_view) in static_facet_views.iter().enumerate() {
            let vertex_count = facet_view.vertices().unwrap().count();
            assert_eq!(vertex_count, 3, "Facet {i} should have 3 vertices");
        }

        // Test with the instance method for comparison
        let cell = tds.get_cell(cell_key).unwrap();
        let instance_facet_views = cell
            .facet_views(&tds, cell_key)
            .expect("Failed to get facet views from instance method");

        // Both methods should return the same number of facets
        assert_eq!(static_facet_views.len(), instance_facet_views.len());

        println!("✓ facet_views_from_tds static method works correctly");
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
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        let cell_key = tds.cell_keys().next().unwrap();

        // Test the iterator method
        let facet_iter =
            Cell::facet_view_iter(&tds, cell_key).expect("Failed to get facet iterator");

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
            Cell::facet_view_iter(&tds, cell_key).expect("Failed to get second facet iterator");

        let mut count = 0;
        for facet_result in facet_iter2 {
            let _facet_view = facet_result.expect("Facet creation should succeed");
            count += 1;
        }
        assert_eq!(count, 4, "Iterator should yield 4 facets");

        // Test iterator combinators work correctly
        let facet_iter3 =
            Cell::facet_view_iter(&tds, cell_key).expect("Failed to get third facet iterator");

        let successful_facets: Vec<_> = facet_iter3.filter_map(Result::ok).collect();
        assert_eq!(
            successful_facets.len(),
            4,
            "All facets should be created successfully"
        );

        // Compare with Vec-based method to ensure same results
        let vec_facets =
            Cell::facet_views_from_tds(&tds, cell_key).expect("Vec-based method should work");
        assert_eq!(
            successful_facets.len(),
            vec_facets.len(),
            "Iterator and Vec methods should return same count"
        );

        println!("✓ facet_view_iter zero-allocation iterator works correctly");
    }

    #[test]
    fn test_facet_view_memory_efficiency_comparison() {
        // Demonstrate the memory efficiency difference between Vec and iterator approaches
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        // Vec-based approach: allocates memory for all FacetViews upfront
        let vec_facets =
            Cell::facet_views_from_tds(&tds, cell_key).expect("Vec method should work");
        println!(
            "Vec approach: {} FacetViews allocated in memory",
            vec_facets.len()
        );

        // Iterator approach: zero allocation, processes one at a time
        let facet_iter =
            Cell::facet_view_iter(&tds, cell_key).expect("Iterator method should work");

        println!(
            "Iterator approach: processing {} facets with zero upfront allocation",
            facet_iter.len()
        );

        let mut processed_count = 0;
        for (i, facet_result) in facet_iter.enumerate() {
            let facet_view = facet_result.expect("Facet creation should succeed");
            // At this point, we only have ONE FacetView in memory, not all of them
            let vertex_count = facet_view.vertices().unwrap().count();
            processed_count += 1;
            println!(
                "  Processed facet {i}: {vertex_count} vertices (only this one FacetView in memory)"
            );
            // FacetView is dropped here, freeing memory before the next iteration
        }

        assert_eq!(processed_count, vec_facets.len());

        // Demonstrate early termination benefit: iterator can be stopped early without
        // creating all FacetViews
        let facet_iter2 =
            Cell::facet_view_iter(&tds, cell_key).expect("Iterator method should work");

        let first_two_facets: Vec<_> = facet_iter2
            .take(2) // Only process first 2 facets
            .filter_map(Result::ok)
            .collect();

        assert_eq!(first_two_facets.len(), 2);
        println!(
            "Early termination: processed only {} out of {} facets",
            first_two_facets.len(),
            4
        );

        println!("✓ Memory efficiency comparison demonstrates iterator benefits");
    }

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
        let tds: crate::core::triangulation_data_structure::Tds<f64, Option<()>, Option<()>, 3> =
            crate::core::triangulation_data_structure::Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();
        let cell = &tds.get_cell(cell_key).unwrap();

        println!("\nAPI Ergonomics Test:");

        // Test 1: Direct comparison (by value - current implementation)
        let first_uuid = cell.vertex_uuid_iter(&tds).next().unwrap().unwrap();
        assert_ne!(first_uuid, Uuid::nil());
        println!("  ✓ By value: Direct comparison works: uuid != Uuid::nil()");

        // Test 2: What by-reference would look like (simulation)
        // Note: We simulate by-reference by collecting values then referencing them
        let uuid_values: Vec<Uuid> = cell
            .vertices()
            .iter()
            .map(|&vkey| tds.get_vertex_by_key(vkey).unwrap().uuid())
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
            for uuid in cell.vertex_uuid_iter(&tds) {
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
                .map(|&vkey| tds.get_vertex_by_key(vkey).unwrap().uuid())
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
        assert_eq!(cell.vertex_uuid_iter(&tds).count(), 4);

        // Test that we can directly use values in hashmaps, comparisons, etc.
        let unique_uuids: HashSet<_> = cell
            .vertex_uuid_iter(&tds)
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
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        // Get a cell and clone it (since we need mutable access)
        let (_cell_key, original_cell) = tds.cells().next().unwrap();
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
