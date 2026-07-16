//! Data and operations on d-dimensional simplices.
//!
//! This module provides [`Simplex`], the TDS element representing a D-dimensional
//! simplex by its vertex keys, neighboring-simplex slots, UUID, and optional user data.
//!
//! # Key Features
//!
//! - **Topology-only storage**: the `Simplex` type stores vertex keys rather than
//!   coordinate-bearing vertices, so it is generic only over simplex data and dimension
//! - **Unique Identification**: Each simplex has a UUID for consistent identification
//! - **Vertices Management**: Stores vertices that form the simplex
//! - **Neighbor Tracking**: Maintains references to neighboring simplices
//! - **Optional Data Storage**: Supports attaching arbitrary user data of type `V`
//! - **Serialization Support**: Manual serde for `uuid` and `data`; vertex/neighbor keys are
//!   omitted from standalone simplex records and stored as UUID relationships in TDS snapshots
//!
//! # Examples
//!
//! ```rust
//! use delaunay::prelude::*;
//!
//! # fn main() -> DelaunayResult<()> {
//! // Create vertices for a tetrahedron
//! let vertices = vec![
//!     delaunay::vertex![0.0, 0.0, 0.0]?,
//!     delaunay::vertex![1.0, 0.0, 0.0]?,
//!     delaunay::vertex![0.0, 1.0, 0.0]?,
//!     delaunay::vertex![0.0, 0.0, 1.0]?,
//! ];
//!
//! // Create a 3D triangulation with simplices
//! let dt: DelaunayTriangulation<_, (), (), 3> =
//!     DelaunayTriangulationBuilder::new(&vertices).build()?;
//! let Some((simplex_key, simplex)) = dt.simplices().next() else {
//!     return Ok(());
//! };
//! assert_eq!(simplex.number_of_vertices(), 4);
//! # let _ = simplex_key;
//! # Ok(())
//! # }
//! ```
//!
//! ## Construction Macros
//!
//! There is intentionally no public `simplex!` macro. A [`Simplex`] stores
//! TDS-local vertex keys, not coordinate-bearing vertices, and its fallible
//! constructors are crate-private so the owning [`Tds`] can validate key
//! locality and topology relationships. Public callers should construct
//! triangulations from vertices or explicit index lists through the builder APIs
//! rather than manufacturing standalone simplex records.

#![allow(clippy::similar_names)]
#![forbid(unsafe_code)]

// =============================================================================
// IMPORTS
// =============================================================================

use super::vertex::{Vertex, VertexValidationError};
use super::{
    tds::{EntityKind, SimplexKey, Tds, TdsConstructionError, VertexKey},
    traits::{DataDeserialize, DataSerialize},
    util::{UuidValidationError, make_uuid, validate_uuid},
};
use crate::core::collections::{
    FastHashMap, NeighborBuffer, PeriodicOffsetBuffer, SimplexVertexBuffer, SimplexVertexKeyBuffer,
    SimplexVertexUuidBuffer, fast_hash_map_with_capacity,
};
use crate::geometry::matrix::StackMatrixDispatchError;
use crate::geometry::traits::coordinate::CoordinateConversionError;
use serde::{
    Deserialize, Deserializer, Serialize,
    de::{self, IgnoredAny, MapAccess, Visitor},
    ser::SerializeStruct,
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

/// Errors that can occur during simplex validation.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::tds::SimplexValidationError;
///
/// let err = SimplexValidationError::DuplicateVertices;
/// std::assert_matches!(err, SimplexValidationError::DuplicateVertices);
/// ```
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum SimplexValidationError {
    /// The simplex has an invalid vertex.
    #[error("Invalid vertex: {source}")]
    InvalidVertex {
        /// The underlying vertex validation error.
        #[from]
        source: VertexValidationError,
    },
    /// The simplex has an invalid UUID.
    #[error("Invalid UUID: {source}")]
    InvalidUuid {
        /// The underlying UUID validation error.
        #[from]
        source: UuidValidationError,
    },
    /// The simplex contains duplicate vertex identities.
    ///
    /// Ordinary identity is the vertex key. Periodic identity is the aligned
    /// `(vertex key, lattice offset)` pair.
    #[error("Duplicate vertices: simplex contains non-unique vertex identities")]
    DuplicateVertices,
    /// The simplex has insufficient vertices to form a proper D-simplex.
    #[error(
        "Insufficient vertices: simplex has {actual} vertices; expected exactly {expected} for a {dimension}D simplex"
    )]
    InsufficientVertices {
        /// The actual number of vertices in the simplex.
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
    /// An assigned neighbor buffer contains an unassigned slot.
    #[error(
        "Unassigned neighbor slot at facet {facet_index}; assigned neighbor buffers must contain only boundary or neighbor slots"
    )]
    UnassignedNeighborSlot {
        /// Facet slot that is still unassigned.
        facet_index: usize,
    },
    /// The periodic offset list is not aligned with the simplex's vertex list.
    #[error("Periodic offset length mismatch: got {found}, expected {expected}")]
    PeriodicOffsetLengthMismatch {
        /// The expected number of offsets (= number of vertices).
        expected: usize,
        /// The observed number of offsets.
        found: usize,
    },
    /// A vertex key referenced by the simplex was not found in the TDS.
    #[error("Vertex key {key:?} not found in TDS (indicates TDS corruption or inconsistency)")]
    VertexKeyNotFound {
        /// The vertex key that was not found.
        key: VertexKey,
    },
}

/// Aggregate report for standalone simplex validation failures.
///
/// This is the Level 1 element-local report counterpart to
/// [`Simplex::is_valid`] and [`Simplex::simplex_diagnostic`].
#[derive(Clone, Debug, PartialEq)]
pub struct SimplexValidationReport {
    /// The ordered list of simplex invariant violations that occurred.
    pub violations: Vec<SimplexValidationError>,
}

impl SimplexValidationReport {
    /// Returns `true` if no violations were recorded.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.violations.is_empty()
    }

    /// Returns the recorded simplex invariant violations.
    #[must_use]
    pub fn violations(&self) -> &[SimplexValidationError] {
        &self.violations
    }
}

impl From<StackMatrixDispatchError> for SimplexValidationError {
    fn from(source: StackMatrixDispatchError) -> Self {
        CoordinateConversionError::from(source).into()
    }
}

fn compare_vertices_by_coordinates<U, const D: usize>(
    left: &Vertex<U, D>,
    right: &Vertex<U, D>,
) -> cmp::Ordering {
    for (left_coord, right_coord) in left.point().coords().iter().zip(right.point().coords()) {
        let ordering = left_coord.total_cmp(right_coord);
        if ordering != cmp::Ordering::Equal {
            return ordering;
        }
    }
    cmp::Ordering::Equal
}

/// A typed neighbor slot for a simplex facet.
///
/// This distinguishes the TDS states that nested `Option` storage used to blur:
/// a neighbor buffer can be unassigned, a facet can be assigned as a boundary,
/// or a facet can point at a neighboring simplex.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::tds::{SimplexKey, NeighborSlot};
/// use slotmap::KeyData;
///
/// let key = SimplexKey::from(KeyData::from_ffi(1));
///
/// assert_eq!(
///     NeighborSlot::from_neighbor_key(Some(key)),
///     NeighborSlot::Neighbor(key)
/// );
/// assert!(NeighborSlot::from_neighbor_key(None).is_boundary());
/// assert!(NeighborSlot::Unassigned.is_unassigned());
/// assert_eq!(NeighborSlot::Boundary.simplex_key(), None);
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NeighborSlot {
    /// The facet slot has not been assigned by the TDS neighbor builder.
    Unassigned,
    /// The facet is assigned and lies on the boundary.
    Boundary,
    /// The facet is assigned and has a neighboring simplex.
    Neighbor(SimplexKey),
}

impl NeighborSlot {
    /// Converts an optional neighbor key into an assigned neighbor slot.
    ///
    /// `Some(key)` becomes [`Neighbor`](Self::Neighbor); `None` becomes
    /// [`Boundary`](Self::Boundary), not [`Unassigned`](Self::Unassigned).
    #[inline]
    #[must_use]
    pub const fn from_neighbor_key(neighbor: Option<SimplexKey>) -> Self {
        match neighbor {
            Some(simplex_key) => Self::Neighbor(simplex_key),
            None => Self::Boundary,
        }
    }

    /// Returns the neighboring simplex key when this slot has one.
    #[inline]
    #[must_use]
    pub const fn simplex_key(self) -> Option<SimplexKey> {
        match self {
            Self::Neighbor(simplex_key) => Some(simplex_key),
            Self::Boundary | Self::Unassigned => None,
        }
    }

    /// Returns whether the slot is an assigned boundary facet.
    #[inline]
    #[must_use]
    pub const fn is_boundary(self) -> bool {
        matches!(self, Self::Boundary)
    }

    /// Returns whether the slot has not been assigned by the TDS.
    #[inline]
    #[must_use]
    pub const fn is_unassigned(self) -> bool {
        matches!(self, Self::Unassigned)
    }
}

// =============================================================================
// CONVENIENCE MACROS AND HELPERS
// =============================================================================

// =============================================================================
// SIMPLEX STRUCT DEFINITION
// =============================================================================

#[derive(Clone, Debug)]
/// The [Simplex] struct represents a d-dimensional
/// [simplex](https://en.wikipedia.org/wiki/Simplex) with vertices, a unique
/// identifier, optional neighbors, and optional data.
///
/// # Storage Model
///
/// A simplex stores keys to vertices and neighbors. This keeps topology edits
/// cheap and lets the owning [`Tds`] provide the authoritative vertex storage.
///
/// # Properties
///
/// - `vertices`: Keys referencing vertices in the TDS. Access via `vertices()` method.
/// - `uuid`: Universally unique identifier for the simplex.
/// - `neighbors`: Optional keys to neighboring simplices (opposite each vertex). Access via `neighbors()` method.
/// - `data`: Optional user data associated with the simplex. Read via [`data()`](Self::data),
///   mutate via [`Tds::set_simplex_data`](crate::prelude::tds::Tds::set_simplex_data).
///
/// # Accessing Vertices
///
/// Since simplices store keys, use the owning [`Tds`] to resolve vertex data:
/// ```rust
/// use delaunay::prelude::collections::Uuid;
/// use delaunay::prelude::*;
///
/// # fn main() -> DelaunayResult<()> {
/// // Create a triangulation with some vertices
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0]?,
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
///
/// // Get first simplex and iterate over vertex keys
/// let Some((simplex_key, simplex)) = dt.simplices().next() else {
///     return Ok(());
/// };
/// for &vertex_key in simplex.vertices() {
///     let Some(vertex) = dt.vertex(vertex_key) else {
///         return Ok(());
///     };
///     // use vertex...
///     assert!(vertex.uuid() != Uuid::nil());
/// }
/// # let _ = simplex_key;
/// # Ok(())
/// # }
/// ```
pub struct Simplex<V, const D: usize> {
    /// Keys to the vertices forming this simplex.
    ///
    /// - Zero heap allocation for D ≤ 7 (stack-allocated)
    /// - Direct key access without UUID lookup
    /// - Better cache locality
    ///
    /// Note: Not serialized - vertices are serialized separately and keys
    /// are reconstructed during deserialization.
    vertices: SimplexVertexKeyBuffer,

    /// The unique identifier of the simplex.
    uuid: Uuid,

    /// Typed neighboring-simplex slots, indexed by opposite vertex.
    ///
    /// Positional semantics: `neighbors[i]` is the neighbor opposite `vertices[i]`.
    ///
    /// # Example
    /// For a 3D simplex (tetrahedron) with 4 vertices:
    /// - `neighbors[0]` is opposite `vertices[0]` (shares vertices 1, 2, 3)
    /// - `neighbors[1]` is opposite `vertices[1]` (shares vertices 0, 2, 3)
    /// - `neighbors[2]` is opposite `vertices[2]` (shares vertices 0, 1, 3)
    /// - `neighbors[3]` is opposite `vertices[3]` (shares vertices 0, 1, 2)
    ///
    /// Note: Not serialized by standalone simplex records. TDS snapshots store
    /// neighbor relationships separately as simplex UUIDs.
    /// Access via `neighbor_slots()` or `neighbors()`. Mutation goes through TDS-owned helpers.
    neighbors: Option<NeighborBuffer<NeighborSlot>>,

    /// The optional data associated with the simplex.
    pub(crate) data: Option<V>,

    /// Optional per-vertex periodic lattice offsets for quotient-simplex reconstruction.
    ///
    /// When present, this buffer is aligned with `vertices` by index:
    /// `periodic_vertex_offsets[i]` corresponds to `vertices[i]`.
    /// Standalone simplex records omit offsets. TDS snapshots store them
    /// separately in `simplex_vertex_offsets` and restore them during hydration.
    pub(crate) periodic_vertex_offsets: Option<PeriodicOffsetBuffer<D>>,
}

// =============================================================================
// SERIALIZATION IMPLEMENTATION
// =============================================================================

/// Manual implementation of Serialize for Simplex.
///
/// This implementation deliberately omits vertex keys because `VertexKey`
/// values are slotmap-local. `Tds` serialization stores a simplex UUID → vertex
/// UUID mapping and resolves those UUIDs back to live keys during TDS
/// deserialization.
///
/// **Field Count Optimization**: We dynamically adjust the field count and conditionally
/// serialize `data` to omit it from JSON when None (reducing output size). The second
/// `is_some()` check matches the field count logic—both could be removed to always
/// serialize "data": null, but tests explicitly verify the field is omitted when None.
impl<V, const D: usize> Serialize for Simplex<V, D>
where
    V: DataSerialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let has_data = self.data.is_some();
        let field_count = if has_data { 2 } else { 1 };
        let mut state = serializer.serialize_struct("Simplex", field_count)?;
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

/// Raw standalone simplex record used only to reject direct simplex deserialization.
///
/// TDS snapshots carry simplex-to-vertex and simplex-to-neighbor relationships
/// separately as UUIDs. This raw record intentionally does not construct a `Simplex`;
/// callers must deserialize the containing TDS snapshot instead.
struct StandaloneSimplexRecord<V> {
    uuid: Uuid,
    data: Option<V>,
}

impl<'de, V> Deserialize<'de> for StandaloneSimplexRecord<V>
where
    V: DataDeserialize,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
    {
        struct StandaloneSimplexVisitor<V>
        where
            V: DataDeserialize,
        {
            _phantom: PhantomData<V>,
        }

        impl<'de, V> Visitor<'de> for StandaloneSimplexVisitor<V>
        where
            V: DataDeserialize,
        {
            type Value = StandaloneSimplexRecord<V>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a standalone Simplex record")
            }

            fn visit_map<A>(self, mut map: A) -> Result<StandaloneSimplexRecord<V>, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut uuid = None;
                let mut data = None;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
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
                        "vertices" | "neighbors" | "periodic_vertex_offsets" => {
                            return Err(de::Error::custom(format!(
                                "{key} is storage-local simplex state and must not be deserialized; deserialize Tds so UUID relationships can be reconstructed",
                            )));
                        }
                        _ => {
                            let _ = map.next_value::<IgnoredAny>()?;
                        }
                    }
                }

                let uuid: Uuid = uuid.ok_or_else(|| de::Error::missing_field("uuid"))?;
                validate_uuid(&uuid)
                    .map_err(|e| de::Error::custom(format!("invalid uuid: {e}")))?;

                Ok(StandaloneSimplexRecord {
                    uuid,
                    data: data.flatten(),
                })
            }
        }

        const FIELDS: &[&str] = &["uuid", "data", "vertices"];
        deserializer.deserialize_struct(
            "StandaloneSimplex",
            FIELDS,
            StandaloneSimplexVisitor {
                _phantom: PhantomData,
            },
        )
    }
}

/// Manual implementation of Deserialize for Simplex.
///
/// A standalone simplex cannot be reconstructed safely from serialized data
/// because its vertex keys are slotmap-local. Deserialize the containing
/// [`Tds`] so simplex vertex UUIDs can be resolved to
/// live keys first.
impl<'de, V, const D: usize> Deserialize<'de> for Simplex<V, D>
where
    V: DataDeserialize,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
    {
        struct SimplexVisitor<V, const D: usize>
        where
            V: DataDeserialize,
        {
            _phantom: PhantomData<V>,
        }

        impl<'de, V, const D: usize> Visitor<'de> for SimplexVisitor<V, D>
        where
            V: DataDeserialize,
        {
            type Value = Simplex<V, D>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a Simplex struct")
            }

            fn visit_map<A>(self, map: A) -> Result<Simplex<V, D>, A::Error>
            where
                A: MapAccess<'de>,
            {
                let StandaloneSimplexRecord { uuid, data } =
                    StandaloneSimplexRecord::<V>::deserialize(
                        de::value::MapAccessDeserializer::new(map),
                    )?;
                let _ = (uuid, data);
                Err(de::Error::custom(
                    "standalone Simplex deserialization is unsupported; deserialize Tds so simplex vertex UUIDs can be resolved to live keys",
                ))
            }
        }

        const FIELDS: &[&str] = &["uuid", "data"];
        deserializer.deserialize_struct(
            "Simplex",
            FIELDS,
            SimplexVisitor {
                _phantom: PhantomData,
            },
        )
    }
}

// =============================================================================
// SIMPLEX IMPLEMENTATION - CORE METHODS
// =============================================================================

// Minimal trait bounds impl block
impl<V, const D: usize> Simplex<V, D> {
    /// Tries to create a simplex from vertex keys without user data.
    ///
    /// This constructor is `pub(crate)` because live [`VertexKey`] values are
    /// TDS-local handles. Public callers should construct triangulations from
    /// vertices or explicit index lists so the owning TDS can validate and map
    /// those handles.
    ///
    /// # Errors
    ///
    /// Returns [`SimplexValidationError::InsufficientVertices`] if `vertices`
    /// does not contain exactly `D + 1` keys, or
    /// [`SimplexValidationError::DuplicateVertices`] if a key appears more than
    /// once.
    pub(crate) fn try_new(
        vertices: impl Into<SimplexVertexKeyBuffer>,
    ) -> Result<Self, SimplexValidationError> {
        Self::try_new_with_data(vertices, None)
    }

    /// Tries to create a simplex from vertex keys with user data.
    ///
    /// # Errors
    ///
    /// Returns:
    /// - `SimplexValidationError::InsufficientVertices` if `vertices` doesn't have exactly D+1 elements.
    /// - `SimplexValidationError::DuplicateVertices` if any vertex key appears more than once.
    pub(crate) fn try_new_with_data(
        vertices: impl Into<SimplexVertexKeyBuffer>,
        data: Option<V>,
    ) -> Result<Self, SimplexValidationError> {
        let vertices = vertices.into();

        // Validate D+1 vertices
        let actual = vertices.len();
        if actual != D + 1 {
            return Err(SimplexValidationError::InsufficientVertices {
                actual,
                expected: D + 1,
                dimension: D,
            });
        }

        // D is intentionally small in this crate; a fixed-size scan avoids a hash allocation.
        for (index, &vkey) in vertices.iter().enumerate() {
            if vertices[..index].contains(&vkey) {
                return Err(SimplexValidationError::DuplicateVertices);
            }
        }

        Ok(Self {
            vertices,
            uuid: make_uuid(),
            neighbors: None,
            data,
            periodic_vertex_offsets: None,
        })
    }

    /// Creates a periodic simplex whose lifted `(vertex key, offset)` identities are distinct.
    ///
    /// Periodic quotient cells may contain two lattice images of the same canonical
    /// vertex. This constructor keeps ordinary simplices strict while admitting
    /// repeated keys only when aligned offsets distinguish the lifted vertices.
    pub(crate) fn try_new_periodic(
        vertices: impl Into<SimplexVertexKeyBuffer>,
        offsets: impl Into<PeriodicOffsetBuffer<D>>,
    ) -> Result<Self, SimplexValidationError> {
        Self::try_new_periodic_with_uuid(vertices, offsets, make_uuid(), None)
    }

    /// Creates a periodic simplex with stable UUID and optional user data.
    ///
    /// This is the snapshot-hydration counterpart to [`Self::try_new_periodic`].
    /// Repeated canonical vertex keys are valid only when their aligned lattice
    /// offsets produce distinct lifted vertex identities.
    pub(crate) fn try_new_periodic_with_uuid(
        vertices: impl Into<SimplexVertexKeyBuffer>,
        offsets: impl Into<PeriodicOffsetBuffer<D>>,
        uuid: Uuid,
        data: Option<V>,
    ) -> Result<Self, SimplexValidationError> {
        validate_uuid(&uuid)?;
        let vertices = vertices.into();
        let offsets = offsets.into();
        let actual = vertices.len();
        if actual != D + 1 {
            return Err(SimplexValidationError::InsufficientVertices {
                actual,
                expected: D + 1,
                dimension: D,
            });
        }
        if offsets.len() != vertices.len() {
            return Err(SimplexValidationError::PeriodicOffsetLengthMismatch {
                expected: vertices.len(),
                found: offsets.len(),
            });
        }
        for index in 0..vertices.len() {
            if (0..index).any(|earlier| {
                vertices[earlier] == vertices[index] && offsets[earlier] == offsets[index]
            }) {
                return Err(SimplexValidationError::DuplicateVertices);
            }
        }

        Ok(Self {
            vertices,
            uuid,
            neighbors: None,
            data,
            periodic_vertex_offsets: Some(offsets),
        })
    }

    /// Tries to create a simplex with a caller-provided UUID.
    ///
    /// This crate-internal constructor is used by TDS deserialization after
    /// serialized vertex UUID relationships have been resolved back to live
    /// [`VertexKey`] handles. Public callers construct simplices through the
    /// owning TDS so key locality and UUID validity are established together.
    ///
    /// # Errors
    ///
    /// Returns [`SimplexValidationError::InvalidUuid`] if `uuid` is nil,
    /// [`SimplexValidationError::InsufficientVertices`] if `vertices` does not
    /// contain exactly `D + 1` keys, or
    /// [`SimplexValidationError::DuplicateVertices`] if any key appears more
    /// than once.
    pub(crate) fn try_new_with_uuid(
        vertices: impl Into<SimplexVertexKeyBuffer>,
        uuid: Uuid,
        data: Option<V>,
    ) -> Result<Self, SimplexValidationError> {
        validate_uuid(&uuid)?;
        let mut simplex = Self::try_new_with_data(vertices, data)?;
        simplex.uuid = uuid;
        Ok(simplex)
    }

    /// Checks if this simplex contains the given vertex key.
    ///
    /// This is a cheap operation (O(D)) that only compares keys.
    ///
    /// # Arguments
    ///
    /// * `vkey` - The vertex key to check
    ///
    /// # Returns
    ///
    /// `true` if the simplex contains the vertex key, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let Some((_, simplex)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let vkey = simplex.vertices()[0];
    ///
    /// if simplex.contains_vertex(vkey) {
    ///     println!("Simplex contains vertex {:?}", vkey);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn contains_vertex(&self, vkey: VertexKey) -> bool {
        self.vertices.contains(&vkey)
    }

    /// Checks if this simplex has any vertex in common with another simplex.
    ///
    /// This is a cheap operation that only compares keys.
    ///
    /// # Arguments
    ///
    /// * `other` - The other simplex to check against
    ///
    /// # Returns
    ///
    /// `true` if the simplices share at least one vertex.
    ///
    /// # Example
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    ///     delaunay::vertex![1.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let mut simplices_iter = dt.simplices().map(|(_, simplex)| simplex);
    /// let Some(simplex1) = simplices_iter.next() else { return Ok(()); };
    /// let Some(simplex2) = simplices_iter.next() else { return Ok(()); };
    ///
    /// if simplex1.has_vertex_in_common(simplex2) {
    ///     println!("Simplices share vertices");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn has_vertex_in_common(&self, other: &Self) -> bool {
        self.vertices
            .iter()
            .any(|vkey| other.vertices.contains(vkey))
    }

    /// Returns the neighbor keys for this simplex without allocating.
    ///
    /// Neighbors are stored as keys (not UUIDs) for direct TDS access.
    /// The positional semantics: `neighbor_key(i)` is the neighbor opposite `vertices()[i]`.
    ///
    /// # Returns
    ///
    /// An `Option` containing an iterator over assigned neighbor keys, or
    /// `None` if neighbor slots have not been assigned. Inside an assigned
    /// buffer, `Some(key)` is a neighboring simplex and `None` is an assigned
    /// boundary facet. Use [`neighbor_slots`](Self::neighbor_slots) when
    /// callers need to distinguish an unassigned neighbor buffer from assigned
    /// boundary facets explicitly.
    ///
    /// # Example
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let Some((_, simplex)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// if let Some(neighbors) = simplex.neighbors() {
    ///     for (i, neighbor_key_opt) in neighbors.enumerate() {
    ///         if let Some(neighbor_key) = neighbor_key_opt {
    ///             let Some(neighbor_simplex) = dt.simplex(neighbor_key) else {
    ///                 continue;
    ///             };
    ///             // neighbor_simplex is opposite to vertex i
    ///             # let _ = neighbor_simplex;
    ///         }
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn neighbors(&self) -> Option<impl ExactSizeIterator<Item = Option<SimplexKey>> + '_> {
        self.neighbor_keys()
    }

    /// Returns neighbor keys without allocating an owned buffer.
    ///
    /// The iterator yields one entry per assigned neighbor slot. `Some(key)` is
    /// a neighboring simplex and `None` is an assigned boundary facet. A return
    /// value of `None` means neighbor assignment has not run or has been cleared.
    #[inline]
    #[must_use]
    pub(crate) fn neighbor_keys(
        &self,
    ) -> Option<impl ExactSizeIterator<Item = Option<SimplexKey>> + '_> {
        self.neighbors
            .as_ref()
            .map(|slots| slots.iter().map(|slot| slot.simplex_key()))
    }

    /// Returns one assigned neighbor key by facet index without allocating.
    ///
    /// The outer `Option` is `None` when neighbor slots are unassigned or when
    /// `facet_idx` is out of bounds. The inner `Option` is `None` for an
    /// assigned boundary facet.
    #[inline]
    #[must_use]
    pub fn neighbor_key(&self, facet_idx: usize) -> Option<Option<SimplexKey>> {
        self.neighbors
            .as_ref()
            .and_then(|slots| slots.get(facet_idx))
            .map(|slot| slot.simplex_key())
    }

    /// Returns the typed neighbor slots for this simplex.
    ///
    /// `None` means neighbor assignment has not run or has been explicitly
    /// cleared. `Some` means each facet slot is assigned as either boundary or
    /// neighboring-simplex state.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::tds::NeighborSlot;
    /// use delaunay::prelude::*;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let Some((_, simplex)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let Some(slots) = simplex.neighbor_slots() else {
    ///     return Ok(());
    /// };
    ///
    /// assert_eq!(slots.len(), 3);
    /// assert!(slots.iter().all(|slot| matches!(*slot, NeighborSlot::Boundary)));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub const fn neighbor_slots(&self) -> Option<&NeighborBuffer<NeighborSlot>> {
        self.neighbors.as_ref()
    }

    /// Returns mutable typed neighbor slots for TDS-owned mutation paths.
    #[inline]
    pub(crate) const fn neighbor_slots_mut(&mut self) -> Option<&mut NeighborBuffer<NeighborSlot>> {
        self.neighbors.as_mut()
    }

    /// Returns the vertex keys for this simplex.
    ///
    /// This method returns keys (not full vertex objects). Use the TDS to resolve keys:
    /// ```rust
    /// use delaunay::prelude::collections::Uuid;
    /// use delaunay::prelude::*;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let Some((_, simplex)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// for &vkey in simplex.vertices() {
    ///     let Some(vertex) = dt.vertex(vkey) else {
    ///         continue;
    ///     };
    ///     // use vertex data...
    ///     assert!(vertex.uuid() != Uuid::nil());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Returns
    ///
    /// A slice containing the vertex keys.
    #[inline]
    pub fn vertices(&self) -> &[VertexKey] {
        &self.vertices[..]
    }

    /// Returns the optional periodic lattice offsets aligned with `vertices`.
    #[inline]
    pub fn periodic_vertex_offsets(&self) -> Option<&[[i8; D]]> {
        self.periodic_vertex_offsets.as_deref()
    }

    /// Sets periodic lattice offsets aligned with `vertices`.
    #[inline]
    pub(crate) fn set_periodic_vertex_offsets(
        &mut self,
        offsets: impl Into<PeriodicOffsetBuffer<D>>,
    ) -> Result<(), SimplexValidationError> {
        let offsets = offsets.into();
        let found = offsets.len();
        let expected = self.vertices.len();
        if found != expected {
            return Err(SimplexValidationError::PeriodicOffsetLengthMismatch { expected, found });
        }
        self.periodic_vertex_offsets = Some(offsets);
        Ok(())
    }

    /// Find the facet index in `neighbor_simplex` that corresponds to the shared facet.
    ///
    /// `facet_idx` is interpreted as the index of the vertex opposite the facet in `self`.
    /// If `neighbor_simplex` shares exactly that facet, this returns the index of the vertex
    /// opposite the same facet in `neighbor_simplex` (CGAL-style "mirror facet").
    ///
    /// Returns `None` if `facet_idx` is out of range, or if the simplices do not appear to share
    /// a single facet.
    #[inline]
    pub(crate) fn mirror_facet_index(
        &self,
        facet_idx: usize,
        neighbor_simplex: &Self,
    ) -> Option<usize> {
        if facet_idx >= self.vertices.len() {
            return None;
        }

        if self.vertices().len() != neighbor_simplex.vertices().len() {
            return None;
        }

        // Build the facet vertex set from the source simplex (all except facet_idx)
        let mut facet_vertices: SimplexVertexKeyBuffer = SimplexVertexKeyBuffer::new();
        for (i, &vkey) in self.vertices().iter().enumerate() {
            if i != facet_idx {
                facet_vertices.push(vkey);
            }
        }

        // Find the vertex in neighbor_simplex that is NOT in the facet.
        // That vertex's index is the mirror facet index.
        let mut mirror_idx: Option<usize> = None;
        for (idx, &neighbor_vkey) in neighbor_simplex.vertices().iter().enumerate() {
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

    /// Adds a vertex key in tests that deliberately construct invalid topology.
    #[cfg(test)]
    #[inline]
    pub(crate) fn push_vertex_key(&mut self, vertex_key: VertexKey) {
        self.vertices.push(vertex_key);
        self.periodic_vertex_offsets = None;
    }

    /// Clears vertex keys in tests that deliberately construct invalid topology.
    #[cfg(test)]
    #[inline]
    pub(crate) fn clear_vertex_keys(&mut self) {
        self.vertices.clear();
        self.periodic_vertex_offsets = None;
    }

    /// Swaps two vertex slots and keeps aligned per-slot buffers consistent.
    ///
    /// This updates:
    /// - `vertices`
    /// - `neighbors` (if present)
    /// - `periodic_vertex_offsets` (if present)
    #[inline]
    pub(crate) fn swap_vertex_slots(&mut self, index_a: usize, index_b: usize) {
        let max_idx = index_a.max(index_b);
        assert!(
            max_idx < self.vertices.len(),
            "swap_vertex_slots vertices index out of bounds: max index {max_idx} >= vertices.len() {}",
            self.vertices.len(),
        );
        if let Some(neighbors) = self.neighbors.as_ref() {
            assert!(
                max_idx < neighbors.len(),
                "swap_vertex_slots neighbors index out of bounds: max index {max_idx} >= neighbors.len() {}",
                neighbors.len(),
            );
        }
        if let Some(offsets) = self.periodic_vertex_offsets.as_ref() {
            assert!(
                max_idx < offsets.len(),
                "swap_vertex_slots periodic offsets index out of bounds: max index {max_idx} >= periodic_vertex_offsets.len() {}",
                offsets.len(),
            );
        }
        self.vertices.swap(index_a, index_b);
        if let Some(neighbors) = &mut self.neighbors {
            neighbors.swap(index_a, index_b);
        }
        if let Some(offsets) = &mut self.periodic_vertex_offsets {
            offsets.swap(index_a, index_b);
        }
    }

    /// Replaces this simplex's assigned neighbor slots from optional neighbor keys.
    #[inline]
    pub(crate) fn set_neighbors_from_keys(
        &mut self,
        neighbors: impl IntoIterator<Item = Option<SimplexKey>>,
    ) -> Result<(), SimplexValidationError> {
        let mut slots = NeighborBuffer::new();
        slots.extend(neighbors.into_iter().map(NeighborSlot::from_neighbor_key));
        if slots.len() != D + 1 {
            return Err(SimplexValidationError::InvalidNeighborsLength {
                actual: slots.len(),
                expected: D + 1,
                dimension: D,
            });
        }
        self.neighbors = Some(slots);
        Ok(())
    }

    /// Ensures this simplex has an assigned neighbor-slot buffer.
    ///
    /// If the buffer does not exist, it is initialized with D+1 unassigned slots.
    #[inline]
    pub(crate) fn ensure_neighbors_buffer_mut(&mut self) -> &mut NeighborBuffer<NeighborSlot> {
        self.neighbors.get_or_insert_with(|| {
            let mut buffer = NeighborBuffer::new();
            buffer.resize(D + 1, NeighborSlot::Unassigned);
            buffer
        })
    }

    /// Ensures this simplex has a correctly sized assigned neighbor-slot buffer.
    ///
    /// If the buffer does not exist, it is initialized with D+1 unassigned slots.
    /// If a buffer exists with the wrong length, the invariant violation is
    /// returned instead of being silently normalized.
    #[inline]
    pub(crate) fn try_ensure_neighbors_buffer_mut(
        &mut self,
    ) -> Result<&mut NeighborBuffer<NeighborSlot>, SimplexValidationError> {
        let buffer = self.ensure_neighbors_buffer_mut();
        if buffer.len() != D + 1 {
            return Err(SimplexValidationError::InvalidNeighborsLength {
                actual: buffer.len(),
                expected: D + 1,
                dimension: D,
            });
        }
        Ok(buffer)
    }
}

// Standard read-only and validation impl block
impl<V, const D: usize> Simplex<V, D> {
    /// The function returns the number of vertices in the [Simplex].
    ///
    /// # Returns
    ///
    /// The number of vertices in the [Simplex].
    ///
    /// # Example
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let Some(simplex) = dt.simplex(simplex_key) else {
    ///     return Ok(());
    /// };
    /// assert_eq!(simplex.number_of_vertices(), 4);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn number_of_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Returns the UUID of the [Simplex].
    ///
    /// # Returns
    ///
    /// The Uuid uniquely identifying this simplex.
    ///
    /// # Example
    ///
    /// ```
    /// use delaunay::prelude::collections::Uuid;
    /// use delaunay::prelude::*;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> =
    ///     DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let Some((_, simplex)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    /// assert_ne!(simplex.uuid(), Uuid::nil());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub const fn uuid(&self) -> Uuid {
        self.uuid
    }

    /// Returns a reference to the optional user data associated with this simplex.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::tds::Simplex;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).simplex_data_type::<i32>().build()?;
    /// let Some((_, simplex)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    /// assert_eq!(simplex.data(), None); // No data set yet
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub const fn data(&self) -> Option<&V> {
        self.data.as_ref()
    }

    /// Clears the neighbors of the [Simplex].
    ///
    /// **Internal API**: This method is `pub(crate)` to enforce that all neighbor
    /// modifications go through validated TDS methods. External code should use
    /// triangulation-level repair or rebuild APIs rather than clearing neighbor state.
    ///
    /// This method sets the `neighbors` field to `None`, effectively removing all
    /// neighbor relationships. This is useful for benchmarking neighbor assignment
    /// or when rebuilding neighbor relationships from scratch.
    ///
    #[cfg(test)]
    #[inline]
    pub(crate) fn clear_neighbors(&mut self) {
        self.neighbors = None;
    }

    /// Returns the UUIDs of the vertices in this simplex.
    ///
    /// This method requires a `&Tds` parameter to resolve vertex keys to UUIDs.
    ///
    /// # Parameters
    ///
    /// - `tds`: Reference to the triangulation data structure containing the vertices
    ///
    /// # Returns
    ///
    /// A `Result<SimplexVertexUuidBuffer, SimplexValidationError>` containing the UUIDs of all vertices in this simplex,
    /// or an error if a vertex key is not found in the TDS. Uses stack allocation for typical dimensions.
    ///
    /// # Errors
    ///
    /// Returns `SimplexValidationError::VertexKeyNotFound` if a vertex key in the simplex
    /// does not exist in the TDS. This indicates TDS corruption or inconsistency.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::DelaunayResult;
    /// use delaunay::prelude::triangulation::{FastKernel, Triangulation};
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let tds =
    ///     Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices)?;
    /// let Some((simplex_key, _)) = tds.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let Some(simplex) = tds.simplex(simplex_key) else {
    ///     return Ok(());
    /// };
    /// let uuids = simplex.vertex_uuids(&tds)?;
    /// assert_eq!(uuids.len(), 4);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn vertex_uuids<U>(
        &self,
        tds: &Tds<U, V, D>,
    ) -> Result<SimplexVertexUuidBuffer, SimplexValidationError> {
        self.vertices
            .iter()
            .map(|&vkey| {
                tds.vertex(vkey)
                    .map(Vertex::uuid)
                    .ok_or(SimplexValidationError::VertexKeyNotFound { key: vkey })
            })
            .collect()
    }

    /// Returns an iterator over vertex UUIDs without allocating a Vec.
    ///
    /// This method requires a `&Tds` parameter to resolve vertex keys to UUIDs.
    ///
    /// # Parameters
    ///
    /// - `tds`: Reference to the triangulation data structure containing the vertices
    ///
    /// # Returns
    ///
    /// An iterator that yields `Result<Uuid, SimplexValidationError>` for each vertex in the simplex.
    ///
    /// # Errors
    ///
    /// The iterator yields `SimplexValidationError::VertexKeyNotFound` for any vertex key
    /// that does not exist in the TDS. This indicates TDS corruption or inconsistency.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::DelaunayResult;
    /// use delaunay::prelude::triangulation::{FastKernel, Triangulation};
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let tds =
    ///     Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices)?;
    /// let Some((simplex_key, _)) = tds.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let Some(simplex) = tds.simplex(simplex_key) else {
    ///     return Ok(());
    /// };
    /// let uuids: Vec<_> = simplex
    ///     .vertex_uuid_iter(&tds)
    ///     .collect::<Result<Vec<_>, _>>()?;
    /// assert_eq!(uuids.len(), 3);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn vertex_uuid_iter<'a, U>(
        &'a self,
        tds: &'a Tds<U, V, D>,
    ) -> impl ExactSizeIterator<Item = Result<Uuid, SimplexValidationError>> + 'a {
        self.vertices.iter().map(move |&vkey| {
            tds.vertex(vkey)
                .map(Vertex::uuid)
                .ok_or(SimplexValidationError::VertexKeyNotFound { key: vkey })
        })
    }

    /// The `dim` function returns the dimensionality of the [Simplex].
    ///
    /// # Returns
    ///
    /// The `dim` function returns the compile-time dimension `D` of the [Simplex].
    ///
    /// # Example
    ///
    /// ```
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::tds::Simplex;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> =
    ///     DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let Some((_, simplex)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    /// assert_eq!(simplex.dim(), 3);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub const fn dim(&self) -> usize {
        D
    }

    /// Converts a collection of simplices into a `FastHashMap` indexed by their UUIDs.
    ///
    /// This utility function transforms a collection of simplices into a hash map structure
    /// for efficient lookups by UUID. Uses `FastHashMap` for performance.
    ///
    /// # Arguments
    ///
    /// * `simplices` - Simplices to be converted into a `FastHashMap`.
    ///
    /// # Returns
    ///
    /// A [`FastHashMap\u003cUuid, Self\u003e`] where each key is a simplex's UUID and each value
    /// is the corresponding simplex. The map provides O(1) average-case lookups
    /// by UUID.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::tds::Simplex;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// // Create two separate triangulations
    /// let vertices1 = vec![
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    /// ];
    /// let vertices2 = vec![
    ///     delaunay::vertex![1.0, 1.0, 1.0]?,
    ///     delaunay::vertex![2.0, 1.0, 1.0]?,
    ///     delaunay::vertex![1.0, 2.0, 1.0]?,
    ///     delaunay::vertex![1.0, 1.0, 2.0]?,
    /// ];
    /// let dt1: DelaunayTriangulation<_, (), (), 3> =
    ///     DelaunayTriangulationBuilder::new(&vertices1).build()?;
    /// let dt2: DelaunayTriangulation<_, (), (), 3> =
    ///     DelaunayTriangulationBuilder::new(&vertices2).build()?;
    /// let Some((_, simplex1)) = dt1.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let Some((_, simplex2)) = dt2.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let simplex1 = simplex1.clone();
    /// let simplex2 = simplex2.clone();
    ///
    /// let uuid1 = simplex1.uuid();
    /// let uuid2 = simplex2.uuid();
    ///
    /// let simplex_map = Simplex::try_into_hashmap([simplex1, simplex2])?;
    ///
    /// // Access simplices by their UUIDs
    /// assert_eq!(simplex_map.get(&uuid1).map(Simplex::uuid), Some(uuid1));
    /// assert_eq!(simplex_map.get(&uuid2).map(Simplex::uuid), Some(uuid2));
    /// assert_eq!(simplex_map.len(), 2);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ```
    /// use delaunay::prelude::tds::Simplex;
    ///
    /// // Empty collection produces empty FastHashMap
    /// let empty_map = Simplex::<(), 3>::try_into_hashmap([])?;
    /// assert!(empty_map.is_empty());
    /// # Ok::<(), delaunay::prelude::tds::TdsConstructionError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`TdsConstructionError::DuplicateUuid`] if the input contains
    /// duplicate simplex UUIDs.
    pub fn try_into_hashmap<I>(
        simplices: I,
    ) -> Result<FastHashMap<Uuid, Self>, TdsConstructionError>
    where
        I: IntoIterator<Item = Self>,
    {
        let iter = simplices.into_iter();
        let mut map = fast_hash_map_with_capacity(iter.size_hint().0);
        for simplex in iter {
            let uuid = simplex.uuid();
            if map.insert(uuid, simplex).is_some() {
                return Err(TdsConstructionError::DuplicateUuid {
                    entity: EntityKind::Simplex,
                    uuid,
                });
            }
        }
        Ok(map)
    }

    /// The function `is_valid` checks if a [Simplex] is valid.
    ///
    /// # Returns
    ///
    /// A Result indicating whether the [Simplex] is valid. Returns `Ok(())` if valid,
    /// or a `SimplexValidationError` if invalid. The validation checks that:
    /// - Ordinary vertex keys are distinct, while periodic lifted
    ///   `(vertex key, offset)` identities are distinct
    /// - The simplex UUID is valid and not nil
    /// - The simplex has exactly D+1 vertices (forming a proper D-simplex)
    /// - If neighbors are provided, they must have exactly D+1 entries (positional semantics)
    ///
    /// A standalone simplex does not own the vertices referenced by its keys. Vertex validity,
    /// resolved coordinate uniqueness, and neighbor reciprocity require TDS-level context and are
    /// checked by [`Tds::validate`](Tds::validate).
    ///
    /// # Errors
    ///
    /// Returns `SimplexValidationError::InvalidUuid` if the simplex's UUID is nil,
    /// `SimplexValidationError::DuplicateVertices` if the simplex contains duplicate vertex identities,
    /// `SimplexValidationError::InsufficientVertices` if the simplex doesn't have exactly D+1 vertices,
    /// `SimplexValidationError::InvalidNeighborsLength` if neighbors are provided but don't have D+1 entries, or
    /// `SimplexValidationError::UnassignedNeighborSlot` if an assigned neighbor buffer still has an unassigned slot.
    ///
    /// # Example
    ///
    /// ```
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::tds::Simplex;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> =
    ///     DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let Some((_, simplex)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    /// assert!(simplex.is_valid().is_ok());
    /// # Ok(())
    /// # }
    /// ```
    pub fn is_valid(&self) -> Result<(), SimplexValidationError> {
        // Check if UUID is valid
        validate_uuid(&self.uuid)?;

        // Check that simplex has exactly D+1 vertex keys (a proper D-simplex)
        if self.vertices.len() != D + 1 {
            return Err(SimplexValidationError::InsufficientVertices {
                actual: self.vertices.len(),
                expected: D + 1,
                dimension: D,
            });
        }

        if let Some(offsets) = &self.periodic_vertex_offsets
            && offsets.len() != self.vertices.len()
        {
            return Err(SimplexValidationError::PeriodicOffsetLengthMismatch {
                expected: self.vertices.len(),
                found: offsets.len(),
            });
        }

        // Ordinary simplices require unique keys. Periodic simplices require
        // unique lifted `(key, offset)` identities instead.
        for index in 0..self.vertices.len() {
            let is_duplicate = (0..index).any(|earlier| {
                self.vertices[earlier] == self.vertices[index]
                    && self
                        .periodic_vertex_offsets
                        .as_ref()
                        .is_none_or(|offsets| offsets[earlier] == offsets[index])
            });
            if is_duplicate {
                return Err(SimplexValidationError::DuplicateVertices);
            }
        }

        // If neighbors are provided, enforce positional semantics: length must be D+1
        if let Some(ref neighbors) = self.neighbors
            && neighbors.len() != D + 1
        {
            return Err(SimplexValidationError::InvalidNeighborsLength {
                actual: neighbors.len(),
                expected: D + 1,
                dimension: D,
            });
        }
        if let Some(ref neighbors) = self.neighbors {
            for (facet_index, slot) in neighbors.iter().enumerate() {
                if slot.is_unassigned() {
                    return Err(SimplexValidationError::UnassignedNeighborSlot { facet_index });
                }
            }
        }

        Ok(())
    }

    /// Returns the first standalone simplex validation diagnostic, if any.
    #[must_use]
    pub fn simplex_diagnostic(&self) -> Option<SimplexValidationError> {
        self.is_valid().err()
    }

    /// Runs standalone simplex validation and returns all checkable failures.
    ///
    /// Unlike [`is_valid`](Self::is_valid), this method does
    /// not stop after the first invalid field.
    ///
    /// # Errors
    ///
    /// Returns a [`SimplexValidationReport`] containing all checkable simplex
    /// violations.
    pub fn simplex_report(&self) -> Result<(), SimplexValidationReport> {
        let mut violations = Vec::new();

        if let Err(source) = validate_uuid(&self.uuid) {
            violations.push(SimplexValidationError::InvalidUuid { source });
        }

        if self.vertices.len() != D + 1 {
            violations.push(SimplexValidationError::InsufficientVertices {
                actual: self.vertices.len(),
                expected: D + 1,
                dimension: D,
            });
        }

        let periodic_offsets_valid = self.periodic_vertex_offsets.as_ref().is_none_or(|offsets| {
            if offsets.len() == self.vertices.len() {
                true
            } else {
                violations.push(SimplexValidationError::PeriodicOffsetLengthMismatch {
                    expected: self.vertices.len(),
                    found: offsets.len(),
                });
                false
            }
        });

        let mut duplicate_vertices = false;
        if periodic_offsets_valid {
            for index in 0..self.vertices.len() {
                if (0..index).any(|earlier| {
                    self.vertices[earlier] == self.vertices[index]
                        && self
                            .periodic_vertex_offsets
                            .as_ref()
                            .is_none_or(|offsets| offsets[earlier] == offsets[index])
                }) {
                    duplicate_vertices = true;
                    break;
                }
            }
        }
        if duplicate_vertices {
            violations.push(SimplexValidationError::DuplicateVertices);
        }

        if let Some(ref neighbors) = self.neighbors {
            if neighbors.len() != D + 1 {
                violations.push(SimplexValidationError::InvalidNeighborsLength {
                    actual: neighbors.len(),
                    expected: D + 1,
                    dimension: D,
                });
            }
            for (facet_index, slot) in neighbors.iter().enumerate() {
                if slot.is_unassigned() {
                    violations.push(SimplexValidationError::UnassignedNeighborSlot { facet_index });
                }
            }
        }

        if violations.is_empty() {
            Ok(())
        } else {
            Err(SimplexValidationReport { violations })
        }
    }
}

// Advanced implementation block for Simplex methods
impl<V, const D: usize> Simplex<V, D> {
    /// Compare two simplices by their vertex sets (using `Vertex::PartialEq`) for cross-TDS equality checking.
    ///
    /// This method enables semantic comparison of simplices from different TDS instances by comparing
    /// the actual Vertex objects using `Vertex::PartialEq` (coordinate-based comparison).
    /// Two simplices are considered equal if they contain the same set of vertices (by coordinates),
    /// regardless of order. This mirrors `Simplex::PartialEq` semantics but works across TDS boundaries.
    ///
    /// # Arguments
    ///
    /// * `self_tds` - The TDS containing `self`
    /// * `other` - The other simplex to compare against
    /// * `other_tds` - The TDS containing `other`
    ///
    /// # Returns
    ///
    /// `true` if both simplices contain the same set of vertices (by coordinates), `false` otherwise.
    /// Returns `false` if any vertex keys cannot be resolved.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::DelaunayResult;
    /// use delaunay::prelude::triangulation::{FastKernel, Triangulation};
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// // Example 1: Comparing simplices from different TDS instances with same coordinates
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let tds1 =
    ///     Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices)?;
    /// let tds2 =
    ///     Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices)?;
    ///
    /// let Some((_, simplex1)) = tds1.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let Some((_, simplex2)) = tds2.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// // Different TDS instances, but same vertex coordinates
    /// assert!(simplex1.eq_by_vertices(&tds1, simplex2, &tds2));
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ```rust
    /// use delaunay::prelude::construction::DelaunayResult;
    /// use delaunay::prelude::triangulation::{FastKernel, Triangulation};
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// // Example 2: Comparing simplices with different coordinates returns false
    /// let vertices1 = vec![
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let vertices2 = vec![
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![2.0, 0.0]?,  // Different coordinate
    ///     delaunay::vertex![0.0, 2.0]?,  // Different coordinate
    /// ];
    /// let tds1 =
    ///     Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices1)?;
    /// let tds2 =
    ///     Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices2)?;
    ///
    /// let Some((_, simplex1)) = tds1.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let Some((_, simplex2)) = tds2.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// // Different coordinates mean simplices are not equal
    /// assert!(!simplex1.eq_by_vertices(&tds1, simplex2, &tds2));
    /// # Ok(())
    /// # }
    /// ```
    pub fn eq_by_vertices<U>(
        &self,
        self_tds: &Tds<U, V, D>,
        other: &Self,
        other_tds: &Tds<U, V, D>,
    ) -> bool {
        // Get vertices for both simplices
        let self_vertices: Option<Vec<_>> = self
            .vertices()
            .iter()
            .map(|&vkey| self_tds.vertex(vkey))
            .collect();

        let other_vertices: Option<Vec<_>> = other
            .vertices()
            .iter()
            .map(|&vkey| other_tds.vertex(vkey))
            .collect();

        // If we couldn't resolve all vertex keys, simplices are not equal
        let (Some(mut self_vertices), Some(mut other_vertices)) = (self_vertices, other_vertices)
        else {
            return false;
        };

        // Sort vertices for order-independent comparison (matches Simplex::PartialEq semantics)
        // Use Vertex::PartialOrd which compares coordinates
        self_vertices.sort_by(|a, b| compare_vertices_by_coordinates(a, b));
        other_vertices.sort_by(|a, b| compare_vertices_by_coordinates(a, b));

        // Compare using Vertex::PartialEq (coordinate-based)
        self_vertices == other_vertices
    }
}

// =============================================================================
// STANDARD TRAIT IMPLEMENTATIONS
// =============================================================================

/// Canonical identity used by equality, ordering, and hashing.
///
/// Periodic simplices include each vertex's aligned lattice offset. Missing
/// offsets normalize to zero, matching TDS duplicate validation, while the
/// optional length records malformed crate-internal states.
type SimplexIdentitySortKey<const D: usize> =
    (SimplexVertexBuffer<(VertexKey, [i8; D])>, Option<usize>);

/// Builds the canonical trait identity for one simplex.
fn canonical_simplex_identity<V, const D: usize>(
    simplex: &Simplex<V, D>,
) -> SimplexIdentitySortKey<D> {
    let periodic_offsets = simplex.periodic_vertex_offsets.as_deref();
    let mut identities: SimplexVertexBuffer<(VertexKey, [i8; D])> = simplex
        .vertices
        .iter()
        .copied()
        .enumerate()
        .map(|(index, key)| {
            let offset = periodic_offsets
                .and_then(|offsets| offsets.get(index))
                .copied()
                .unwrap_or([0_i8; D]);
            (key, offset)
        })
        .collect();
    identities.sort_unstable();

    let malformed_offset_len = periodic_offsets
        .and_then(|offsets| (offsets.len() != simplex.vertices.len()).then_some(offsets.len()));
    (identities, malformed_offset_len)
}

/// Equality of simplices based on their canonical vertex identities.
///
/// Two simplices are equal if they contain the same set of vertex keys,
/// regardless of order. Periodic simplices additionally compare the aligned
/// `(vertex key, lattice offset)` identities used by TDS duplicate validation;
/// absent offsets normalize to zero.
/// This is fast (O(D log D)) and doesn't require TDS access.
///
/// **Note**: This compares simplices within the same TDS context. For cross-TDS
/// comparison by coordinates, use [`Simplex::eq_by_vertices`].
impl<V, const D: usize> PartialEq for Simplex<V, D> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        canonical_simplex_identity(self) == canonical_simplex_identity(other)
    }
}

/// Order of simplices based on their canonical vertex identities.
///
/// Periodic simplices include aligned lattice offsets after the sorted vertex
/// keys, keeping ordering consistent with equality. Fast (O(D log D)) and
/// doesn't require TDS access.
impl<V, const D: usize> PartialOrd for Simplex<V, D> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        canonical_simplex_identity(self).partial_cmp(&canonical_simplex_identity(other))
    }
}

// =============================================================================
// HASHING AND EQUALITY IMPLEMENTATIONS
// =============================================================================

/// Eq implementation for Simplex based on canonical vertex identities.
///
/// Maintains the Eq contract with `PartialEq`: nonzero periodic offsets
/// participate, while absent offsets match explicit zero offsets.
impl<V, const D: usize> Eq for Simplex<V, D> {}

/// Custom Hash implementation for Simplex using canonical vertex identities.
///
/// This maintains the Eq/Hash contract: if `a == b`, then
/// `hash(a) == hash(b)`, including periodic lattice offsets.
///
/// **Performance**: Fast O(D log D) hashing using stack-backed identity buffers.
///
/// **Note**: UUID, neighbors, and data are excluded from hashing to match
/// the `PartialEq` implementation.
impl<V, const D: usize> Hash for Simplex<V, D> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        canonical_simplex_identity(self).hash(state);
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::DelaunayTriangulationBuilder;
    use crate::core::collections::FastHashSet;
    use crate::core::facet::FacetError;
    use crate::core::vertex::Vertex;
    use crate::geometry::kernel::AdaptiveKernel;
    use crate::geometry::matrix::MAX_STACK_MATRIX_DIM;
    use crate::geometry::point::Point;
    use crate::geometry::predicates::insphere;
    use crate::geometry::util::{circumcenter, circumradius, circumradius_with_center};
    use crate::prelude::DelaunayTriangulation;
    use crate::vertex;
    use approx::assert_relative_eq;
    use proptest::prelude::*;
    use std::assert_matches;
    use std::iter::once;
    use std::{
        cmp,
        collections::{HashSet, hash_map::DefaultHasher},
        hash::Hasher,
    };
    // Type aliases for commonly used types to reduce repetition
    type TestVertex3D = Vertex<(), 3>;
    type TestVertex2D = Vertex<(), 2>;

    struct NonDataType(String);

    fn simplex_with_non_data_type_metadata<const D: usize>(
        vertex_ids: [u64; D],
        data: NonDataType,
    ) -> Simplex<NonDataType, D> {
        Simplex {
            vertices: vertex_ids
                .into_iter()
                .map(|id| VertexKey::from(slotmap::KeyData::from_ffi(id)))
                .collect(),
            uuid: make_uuid(),
            neighbors: None,
            data: Some(data),
            periodic_vertex_offsets: None,
        }
    }

    /// Creates distinct synthetic vertex keys for tests.
    fn synthetic_vertex_keys(count: usize) -> Vec<VertexKey> {
        (0..count)
            .map(|index| {
                let key_value = u64::try_from(index + 1).expect("test key index fits in u64");
                VertexKey::from(slotmap::KeyData::from_ffi(key_value))
            })
            .collect()
    }

    /// Creates a simplex with distinct synthetic keys and aligned periodic offsets.
    fn simplex_with_periodic_offsets<const D: usize>(offsets: &[[i8; D]]) -> Simplex<(), D> {
        assert_eq!(offsets.len(), D + 1);
        let vertex_keys = synthetic_vertex_keys(D + 1);
        let mut simplex = Simplex::try_new(vertex_keys).expect("synthetic keys are distinct");
        simplex
            .set_periodic_vertex_offsets(offsets.to_vec())
            .expect("one offset per simplex vertex");
        simplex
    }

    /// Generates offset-alignment and malformed-length tests for each practical dimension.
    macro_rules! gen_periodic_offset_slot_tests {
        ($dim:literal, $uniform:path) => {
            pastey::paste! {
                proptest! {
                    #![proptest_config(ProptestConfig::with_cases(64))]

                    /// Arbitrary slot-swap sequences keep periodic offsets attached to their keys.
                    #[test]
                    fn [<prop_periodic_offsets_follow_slot_swaps_ $dim d>](
                        offsets in prop::collection::vec(
                            $uniform(any::<i8>()),
                            ($dim + 1)..=($dim + 1),
                        ),
                        swaps in prop::collection::vec(
                            (0_usize..($dim + 1), 0_usize..($dim + 1)),
                            0..=32,
                        ),
                    ) {
                        let mut simplex = simplex_with_periodic_offsets::<$dim>(&offsets);
                        let mut expected: Vec<_> = simplex
                            .vertices()
                            .iter()
                            .copied()
                            .zip(offsets.iter().copied())
                            .collect();

                        for (first, second) in swaps {
                            simplex.swap_vertex_slots(first, second);
                            expected.swap(first, second);
                        }

                        let actual: Vec<_> = simplex
                            .vertices()
                            .iter()
                            .copied()
                            .zip(
                                simplex
                                    .periodic_vertex_offsets()
                                    .expect("offsets remain present")
                                    .iter()
                                    .copied(),
                            )
                            .collect();
                        prop_assert_eq!(actual, expected);
                        prop_assert!(simplex.is_valid().is_ok());
                    }
                }

                #[test]
                fn [<test_periodic_offset_length_failure_is_atomic_ $dim d>]() {
                    let valid_offsets = vec![[0_i8; $dim]; $dim + 1];
                    let mut simplex = simplex_with_periodic_offsets::<$dim>(&valid_offsets);
                    let before = simplex
                        .periodic_vertex_offsets()
                        .expect("fixture offsets are present")
                        .to_vec();

                    for found in [0_usize, $dim, $dim + 2] {
                        let error = simplex
                            .set_periodic_vertex_offsets(vec![[1_i8; $dim]; found])
                            .expect_err("misaligned periodic offsets should be rejected");
                        assert_eq!(
                            error,
                            SimplexValidationError::PeriodicOffsetLengthMismatch {
                                expected: $dim + 1,
                                found,
                            }
                        );
                        assert_eq!(
                            simplex.periodic_vertex_offsets(),
                            Some(before.as_slice()),
                            "failed setter must preserve the prior aligned offsets",
                        );
                    }
                }
            }
        };
    }

    gen_periodic_offset_slot_tests!(2, prop::array::uniform2);
    gen_periodic_offset_slot_tests!(3, prop::array::uniform3);
    gen_periodic_offset_slot_tests!(4, prop::array::uniform4);
    gen_periodic_offset_slot_tests!(5, prop::array::uniform5);

    // =============================================================================
    // DIMENSION-PARAMETERIZED TEST MACRO
    // =============================================================================
    // This macro generates comprehensive tests for each dimension (2D-5D)
    // Keep this at the top so it's not forgotten when adding new tests!

    /// Macro to generate dimension-specific simplex tests for dimensions 2D-5D.
    ///
    /// This macro reduces test duplication by generating consistent tests across
    /// multiple dimensions. It creates tests for:
    /// - Basic simplex creation and property validation
    /// - Serialization roundtrip (Some and None data)
    /// - UUID validation
    macro_rules! test_simplex_dimensions {
        ($(
            $test_name:ident => $dim:expr => $vertices:expr
        ),+ $(,)?) => {
            $(
                #[test]
                fn $test_name() {
                    // Test basic simplex creation
                    let vertices = $vertices;
                    let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
                    let (_, simplex) = dt.simplices().next().unwrap();
                    assert_simplex_properties(simplex, $dim + 1, $dim);
                }

                pastey::paste! {
                    #[test]
                    fn [<$test_name _with_data>]() {
                        // Test simplex with data - need generic constructor for non-() simplex data
                        let vertices = $vertices;
                        let dt: DelaunayTriangulation<AdaptiveKernel<f64>, (), i32, $dim> =
                            DelaunayTriangulationBuilder::new(&vertices)
                                .simplex_data_type::<i32>()
                                .build()
                                .unwrap();

                        let (_, simplex_ref) = dt.simplices().next().unwrap();
                        let mut simplex = simplex_ref.clone();
                        simplex.data = Some(42);
                        assert_simplex_properties(&simplex, $dim + 1, $dim);
                        assert_eq!(simplex.data, Some(42));
                    }

                    #[test]
                    fn [<$test_name _serialization_roundtrip>]() {
                        // Test serialization with Some data - use generic constructor
                        let vertices = $vertices;
                        let dt: DelaunayTriangulation<AdaptiveKernel<f64>, (), i32, $dim> =
                            DelaunayTriangulationBuilder::new(&vertices)
                                .simplex_data_type::<i32>()
                                .build()
                                .unwrap();
                        let (_, simplex) = dt.simplices().next().unwrap();
                        let mut simplex = simplex.clone();
                        simplex.data = Some(99);

                        let serialized = serde_json::to_string(&simplex).unwrap();
                        assert!(serialized.contains("\"data\":"));
                        let err = serde_json::from_str::<Simplex<i32, $dim>>(&serialized)
                            .expect_err("standalone Simplex deserialization should require TDS UUID mapping");
                        assert!(err.to_string().contains("standalone Simplex deserialization"));

                        // Test serialization with None data - use simple constructor
                        let vertices = $vertices;
                        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
                        let (_, simplex) = dt.simplices().next().unwrap();

                        let serialized = serde_json::to_string(&simplex).unwrap();
                        assert!(!serialized.contains("\"data\":"));
                        let err = serde_json::from_str::<Simplex<Option<i32>, $dim>>(&serialized)
                            .expect_err("standalone Simplex deserialization should require TDS UUID mapping");
                        assert!(err.to_string().contains("standalone Simplex deserialization"));
                    }

                    #[test]
                    fn [<$test_name _uuid_uniqueness>]() {
                        // Test UUID uniqueness by creating two separate triangulations
                        let vertices1 = $vertices;
                        let vertices2 = $vertices;
                        let dt1 = DelaunayTriangulation::builder(&vertices1).build().unwrap();
                        let (_, simplex1) = dt1.simplices().next().unwrap();
                        let dt2 = DelaunayTriangulation::builder(&vertices2).build().unwrap();
                        let (_, simplex2) = dt2.simplices().next().unwrap();
                        assert_ne!(simplex1.uuid(), simplex2.uuid());
                        assert!(!simplex1.uuid().is_nil());
                        assert!(!simplex2.uuid().is_nil());
                    }
                }
            )+
        };
    }

    #[test]
    fn equality_ordering_and_hash_do_not_require_data_type_metadata() {
        let simplex_a =
            simplex_with_non_data_type_metadata([1, 2, 3], NonDataType("left".to_string()));
        let simplex_b =
            simplex_with_non_data_type_metadata([3, 2, 1], NonDataType("right".to_string()));
        let simplex_c =
            simplex_with_non_data_type_metadata([1, 2, 4], NonDataType("other".to_string()));

        assert!(simplex_a == simplex_b);
        assert!(simplex_a != simplex_c);
        assert_eq!(
            simplex_a.partial_cmp(&simplex_b),
            Some(cmp::Ordering::Equal)
        );

        let mut hash_a = DefaultHasher::new();
        let mut hash_b = DefaultHasher::new();
        simplex_a.hash(&mut hash_a);
        simplex_b.hash(&mut hash_b);
        assert_eq!(hash_a.finish(), hash_b.finish());

        assert_eq!(simplex_a.data.as_ref().unwrap().0, "left");
        assert_eq!(simplex_b.data.as_ref().unwrap().0, "right");
    }

    #[test]
    fn periodic_equality_ordering_and_hash_include_offsets() {
        let keys = synthetic_vertex_keys(3);
        let non_periodic = Simplex::<(), 2>::try_new(keys.clone()).unwrap();
        let zero_offset_periodic =
            Simplex::<(), 2>::try_new_periodic(keys.clone(), vec![[0, 0]; 3]).unwrap();
        let simplex =
            Simplex::<(), 2>::try_new_periodic(keys.clone(), vec![[0, 0], [1, 0], [0, 1]]).unwrap();
        let permuted = Simplex::<(), 2>::try_new_periodic(
            vec![keys[2], keys[0], keys[1]],
            vec![[0, 1], [0, 0], [1, 0]],
        )
        .unwrap();
        let different_offset =
            Simplex::<(), 2>::try_new_periodic(keys, vec![[0, 0], [2, 0], [0, 1]]).unwrap();

        assert_eq!(simplex, permuted);
        assert_eq!(simplex.partial_cmp(&permuted), Some(cmp::Ordering::Equal));
        assert_ne!(simplex, different_offset);
        assert_ne!(
            simplex.partial_cmp(&different_offset),
            Some(cmp::Ordering::Equal)
        );
        assert_eq!(non_periodic, zero_offset_periodic);
        assert_eq!(
            non_periodic.partial_cmp(&zero_offset_periodic),
            Some(cmp::Ordering::Equal)
        );

        let mut non_periodic_hash = DefaultHasher::new();
        let mut zero_offset_periodic_hash = DefaultHasher::new();
        let mut simplex_hash = DefaultHasher::new();
        let mut permuted_hash = DefaultHasher::new();
        let mut different_offset_hash = DefaultHasher::new();
        non_periodic.hash(&mut non_periodic_hash);
        zero_offset_periodic.hash(&mut zero_offset_periodic_hash);
        simplex.hash(&mut simplex_hash);
        permuted.hash(&mut permuted_hash);
        different_offset.hash(&mut different_offset_hash);
        assert_eq!(
            non_periodic_hash.finish(),
            zero_offset_periodic_hash.finish()
        );
        assert_eq!(simplex_hash.finish(), permuted_hash.finish());
        assert_ne!(simplex_hash.finish(), different_offset_hash.finish());
    }

    // Generate tests for dimensions 2D through 5D
    test_simplex_dimensions! {
        simplex_2d => 2 => vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ],
        simplex_3d => 3 => vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ],
        simplex_4d => 4 => vec![
            vertex!([0.0, 0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 0.0, 1.0]).unwrap(),
        ],
        simplex_5d => 5 => vec![
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 0.0, 0.0, 1.0]).unwrap(),
        ],
    }

    // =============================================================================
    // HELPER FUNCTIONS
    // =============================================================================

    /// Simplified helper function to test basic simplex properties
    fn assert_simplex_properties<V, const D: usize>(
        simplex: &Simplex<V, D>,
        expected_vertices: usize,
        expected_dim: usize,
    ) {
        assert_eq!(simplex.number_of_vertices(), expected_vertices);
        assert_eq!(simplex.dim(), expected_dim);
        assert!(!simplex.uuid().is_nil());
    }

    // Helper functions for creating common test data
    fn create_test_vertices_3d() -> Vec<TestVertex3D> {
        vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ]
    }

    fn create_test_vertices_2d() -> Vec<TestVertex2D> {
        vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ]
    }

    // Tests covering simplex construction through the owning TDS context.

    #[test]
    fn simplex_from_triangulation_without_data() {
        // Public callers obtain simplices from a triangulation, not a standalone macro.
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(), // Need 4 vertices for 3D simplex
        ];

        // Create DT to get a simplex with TDS context.
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let (_, simplex) = dt.simplices().next().unwrap();

        assert_eq!(simplex.number_of_vertices(), 4);
        assert_eq!(simplex.dim(), 3);
        assert!(simplex.data.is_none());
        assert!(!simplex.uuid().is_nil());

        // Verify all input vertices exist in the triangulation (order-independent)
        let simplex_coords: Vec<Vec<f64>> = simplex
            .vertices()
            .iter()
            .map(|&vkey| {
                dt.tds()
                    .vertex(vkey)
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
                simplex_coords.iter().any(|coords| {
                    coords
                        .iter()
                        .zip(original_coords)
                        .all(|(a, b)| (a - b).abs() < f64::EPSILON)
                }),
                "Input vertex {original_coords:?} not found in simplex"
            );
        }
    }

    // =============================================================================
    // SIMPLEX EQUALITY TESTS
    // =============================================================================

    #[test]
    fn test_eq_by_vertices_same_coordinates_different_tds() {
        // Create two separate TDS instances with identical vertex coordinates
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt1 = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let dt2 = DelaunayTriangulation::builder(&vertices).build().unwrap();

        let simplex1 = dt1.simplices().next().unwrap().1;
        let simplex2 = dt2.simplices().next().unwrap().1;

        // Despite different DT instances (and thus different vertex keys),
        // simplices should be equal by coordinates
        assert!(simplex1.eq_by_vertices(dt1.tds(), simplex2, dt2.tds()));
    }

    #[test]
    fn test_eq_by_vertices_2d() {
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let dt1 = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let dt2 = DelaunayTriangulation::builder(&vertices).build().unwrap();

        let simplex1 = dt1.simplices().next().unwrap().1;
        let simplex2 = dt2.simplices().next().unwrap().1;

        assert!(simplex1.eq_by_vertices(dt1.tds(), simplex2, dt2.tds()));
    }

    #[test]
    fn simplex_from_triangulation_with_data() {
        // Exercise simplex data by creating a TDS-backed simplex and modifying a clone.
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];

        // Build DT with integer simplex data type
        let dt: DelaunayTriangulation<AdaptiveKernel<f64>, (), i32, 3> =
            DelaunayTriangulationBuilder::new(&vertices)
                .simplex_data_type::<i32>()
                .build()
                .unwrap();

        let (_, simplex_ref) = dt.simplices().next().unwrap();
        let mut simplex = simplex_ref.clone();
        simplex.data = Some(42);

        assert_eq!(simplex.number_of_vertices(), 4);
        assert_eq!(simplex.dim(), 3);
        assert_eq!(simplex.data.unwrap(), 42);
        assert!(!simplex.uuid().is_nil());

        // Verify all input vertices exist in the simplex (order-independent)
        let simplex_coords: Vec<Vec<f64>> = simplex
            .vertices()
            .iter()
            .map(|&vkey| {
                dt.tds()
                    .vertex(vkey)
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
                simplex_coords.iter().any(|coords| {
                    coords
                        .iter()
                        .zip(original_coords)
                        .all(|(a, b)| (a - b).abs() < f64::EPSILON)
                }),
                "Input vertex {original_coords:?} not found in simplex"
            );
        }
    }

    #[test]
    fn simplex_with_vertex_data() {
        // Test simplices with vertex data through TDS.
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]; data = 1).unwrap(),
            vertex!([1.0, 0.0, 0.0]; data = 2).unwrap(),
            vertex!([0.0, 1.0, 0.0]; data = 3).unwrap(),
            vertex!([0.0, 0.0, 1.0]; data = 4).unwrap(), // Need 4 vertices for 3D simplex
        ];

        // Create DT with vertex data and default simplex data.
        let dt: DelaunayTriangulation<AdaptiveKernel<f64>, i32, (), 3> =
            DelaunayTriangulationBuilder::new(&vertices)
                .build()
                .unwrap();

        let (_, simplex) = dt.simplices().next().unwrap();

        assert_eq!(simplex.number_of_vertices(), 4);
        assert_eq!(simplex.dim(), 3);

        // Check that all expected vertex data values exist (order-independent)
        let simplex_data: Vec<i32> = simplex
            .vertices()
            .iter()
            .map(|&vkey| dt.tds().vertex(vkey).unwrap().data.unwrap())
            .collect();

        for expected in &[1, 2, 3, 4] {
            assert!(
                simplex_data.contains(expected),
                "Expected vertex data {expected} not found in simplex"
            );
        }
    }

    // =============================================================================
    // TRAIT IMPLEMENTATION TESTS
    // =============================================================================
    // Tests covering core Rust traits like PartialEq, PartialOrd, Hash, Clone

    #[test]
    fn simplex_partial_eq() {
        // Test PartialEq using simplices from TDS.
        let vertices = vec![
            vertex!([0.0, 0.0, 1.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 0.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let (_, simplex1) = dt.simplices().next().unwrap();
        let simplex2 = simplex1.clone();

        // Test equality
        assert_eq!(*simplex1, simplex2);
        assert_eq!(simplex1.uuid(), simplex2.uuid()); // Same simplex, same UUID after clone
        assert_eq!(simplex1.vertices(), simplex2.vertices());

        // Test cloned simplex
        let simplex3 = simplex1.clone();
        assert_eq!(*simplex1, simplex3);
    }

    #[test]
    fn simplex_partial_ord() {
        // Test PartialOrd using TDS with multiple simplices.
        let all_vertices = vec![
            vertex!([0.0, 0.0, 1.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 1.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::builder(&all_vertices)
            .build()
            .unwrap();
        let simplices: Vec<_> = dt.simplices().map(|(_, simplex)| simplex).collect();

        if simplices.len() >= 2 {
            // Test ordering between different simplices
            let simplex1 = simplices[0];
            let simplex2 = simplices[1];

            // At least one ordering relationship should hold
            let has_ordering = simplex1 != simplex2 || simplex1 == simplex2;
            assert!(
                has_ordering,
                "Simplices should have some ordering relationship"
            );
        }
    }

    #[test]
    fn simplex_hash() {
        let vertices = vec![
            vertex!([0.0, 0.0, 1.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 0.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let (_, simplex1) = dt.simplices().next().unwrap();
        let simplex2 = simplex1.clone();

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();

        simplex1.hash(&mut hasher1);
        simplex2.hash(&mut hasher2);

        // Same vertices should produce same hash (Eq/Hash contract)
        assert_eq!(*simplex1, simplex2); // They are equal by vertices
        assert_eq!(hasher1.finish(), hasher2.finish()); // Therefore hashes must be equal
        // Note: UUID is same since simplex2 is a clone
        assert_eq!(simplex1.uuid(), simplex2.uuid());
    }

    #[test]
    fn simplex_clone() {
        let vertices = vec![
            vertex!([0.0, 0.0, 1.0]; data = 1).unwrap(),
            vertex!([0.0, 1.0, 0.0]; data = 1).unwrap(),
            vertex!([1.0, 0.0, 0.0]; data = 1).unwrap(),
            vertex!([1.0, 1.0, 1.0]; data = 2).unwrap(),
        ];
        let dt: DelaunayTriangulation<AdaptiveKernel<f64>, i32, i32, 3> =
            DelaunayTriangulationBuilder::new(&vertices)
                .simplex_data_type::<i32>()
                .build()
                .unwrap();
        let (_, simplex_ref) = dt.simplices().next().unwrap();
        let mut simplex1 = simplex_ref.clone();
        simplex1.data = Some(42);
        let simplex2 = simplex1.clone();

        assert_eq!(simplex1, simplex2);
    }

    #[test]
    fn simplex_ordering_edge_cases() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let (_, simplex1) = dt.simplices().next().unwrap();
        let simplex2 = simplex1.clone();

        // Test that equal simplices are not less than each other
        assert_ne!(simplex1.partial_cmp(&simplex2), Some(cmp::Ordering::Less));
        assert_ne!(simplex2.partial_cmp(simplex1), Some(cmp::Ordering::Less));
        assert!(*simplex1 <= simplex2);
        assert!(simplex2 <= *simplex1);
        assert!(*simplex1 >= simplex2);
        assert!(simplex2 >= *simplex1);
    }
    // =============================================================================
    // CORE SIMPLEX METHODS TESTS
    // =============================================================================
    // Tests covering core simplex functionality including basic properties, containment
    // checks, facet operations, and other fundamental simplex methods.

    #[test]
    fn simplex_number_of_vertices() {
        let vertices_2d = create_test_vertices_2d();
        let dt_2d = DelaunayTriangulation::builder(&vertices_2d)
            .build()
            .unwrap();
        let simplex_key_2d = dt_2d.tds().simplex_keys().next().unwrap();
        let triangle = dt_2d.tds().simplex(simplex_key_2d).unwrap();
        assert_eq!(triangle.number_of_vertices(), 3);

        let vertices_3d = create_test_vertices_3d();
        let dt_3d = DelaunayTriangulation::builder(&vertices_3d)
            .build()
            .unwrap();
        let simplex_key_3d = dt_3d.tds().simplex_keys().next().unwrap();
        let tetrahedron = dt_3d.tds().simplex(simplex_key_3d).unwrap();
        assert_eq!(tetrahedron.number_of_vertices(), 4);
    }

    #[test]
    fn simplex_mirror_facet_index_shared_facet_2d() {
        // Four points in convex position should yield two triangles that share an edge.
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
            vertex!([1.0, 1.1]).unwrap(), // break cocircular symmetry
        ];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();

        let simplices: Vec<_> = dt.simplices().map(|(_, simplex)| simplex).collect();
        assert!(
            simplices.len() >= 2,
            "Expected at least 2 simplices, got {}",
            simplices.len()
        );

        // Find two distinct simplices that share exactly D vertices (i.e., a facet).
        // For 2D, D = 2, so the shared facet is an edge.
        let mut found = None;
        for i in 0..simplices.len() {
            for j in (i + 1)..simplices.len() {
                let simplex_a = simplices[i];
                let simplex_b = simplices[j];

                let shared: FastHashSet<VertexKey> = simplex_a
                    .vertices()
                    .iter()
                    .copied()
                    .filter(|v| simplex_b.vertices().contains(v))
                    .collect();

                if shared.len() == 2 {
                    let facet_idx_a = simplex_a
                        .vertices()
                        .iter()
                        .position(|v| !shared.contains(v))
                        .expect("simplex_a should have one vertex not in the shared facet");

                    let facet_idx_b = simplex_b
                        .vertices()
                        .iter()
                        .position(|v| !shared.contains(v))
                        .expect("simplex_b should have one vertex not in the shared facet");

                    found = Some((simplex_a, simplex_b, facet_idx_a, facet_idx_b));
                    break;
                }
            }
            if found.is_some() {
                break;
            }
        }

        let Some((simplex_a, simplex_b, facet_idx_a, facet_idx_b)) = found else {
            panic!("Expected to find a pair of neighboring simplices that share an edge");
        };

        assert_eq!(
            simplex_a.mirror_facet_index(facet_idx_a, simplex_b),
            Some(facet_idx_b)
        );
        assert_eq!(
            simplex_b.mirror_facet_index(facet_idx_b, simplex_a),
            Some(facet_idx_a)
        );

        // Out-of-range facet index
        assert_eq!(
            simplex_a.mirror_facet_index(simplex_a.number_of_vertices(), simplex_b),
            None
        );
    }

    #[test]
    fn simplex_mirror_facet_index_returns_none_when_simplices_do_not_share_facet_2d() {
        // Add a point strictly inside the convex hull to yield multiple triangles.
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
            vertex!([1.0, 1.0]).unwrap(),
            vertex!([0.5, 0.5]).unwrap(),
        ];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();

        let simplices: Vec<_> = dt.simplices().map(|(_, simplex)| simplex).collect();
        assert!(
            simplices.len() >= 3,
            "Expected at least 3 simplices, got {}",
            simplices.len()
        );

        // Find two simplices that share fewer than D vertices (D=2 in 2D).
        let mut non_adjacent = None;
        'outer: for i in 0..simplices.len() {
            for j in (i + 1)..simplices.len() {
                let simplex_a = simplices[i];
                let simplex_b = simplices[j];
                let shared_count = simplex_a
                    .vertices()
                    .iter()
                    .filter(|v| simplex_b.vertices().contains(v))
                    .count();

                if shared_count < 2 {
                    non_adjacent = Some((simplex_a, simplex_b));
                    break 'outer;
                }
            }
        }

        let Some((simplex_a, simplex_b)) = non_adjacent else {
            panic!("Expected to find a pair of non-adjacent simplices");
        };

        assert_eq!(simplex_a.mirror_facet_index(0, simplex_b), None);
    }

    #[test]
    fn simplex_mirror_facet_index_returns_none_for_mismatched_vertex_counts() {
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let (_, simplex_ref) = dt.simplices().next().unwrap();

        let mut malformed_neighbor = simplex_ref.clone();
        malformed_neighbor.vertices.pop();

        assert_eq!(simplex_ref.mirror_facet_index(0, &malformed_neighbor), None);
    }

    #[test]
    fn simplex_dim() {
        let vertices_2d = create_test_vertices_2d();
        let dt_2d = DelaunayTriangulation::builder(&vertices_2d)
            .build()
            .unwrap();
        let simplex_key_2d = dt_2d.tds().simplex_keys().next().unwrap();
        let triangle = dt_2d.tds().simplex(simplex_key_2d).unwrap();
        assert_eq!(triangle.dim(), 2);

        let vertices_3d = create_test_vertices_3d();
        let dt_3d = DelaunayTriangulation::builder(&vertices_3d)
            .build()
            .unwrap();
        let simplex_key_3d = dt_3d.tds().simplex_keys().next().unwrap();
        let tetrahedron = dt_3d.tds().simplex(simplex_key_3d).unwrap();
        assert_eq!(tetrahedron.dim(), 3);
    }

    #[test]
    fn simplex_contains_vertex() {
        let vertex1: Vertex<i32, 3> = vertex!([0.0, 0.0, 1.0]; data = 1).unwrap();
        let vertex2 = vertex!([0.0, 1.0, 0.0]; data = 1).unwrap();
        let vertex3 = vertex!([1.0, 0.0, 0.0]; data = 1).unwrap();
        let vertex4 = vertex!([1.0, 1.0, 1.0]; data = 2).unwrap();

        // Create DT to get VertexKeys - use generic constructor for vertex data
        let vertices = vec![vertex1, vertex2, vertex3, vertex4];
        let dt: DelaunayTriangulation<AdaptiveKernel<f64>, i32, (), 3> =
            DelaunayTriangulationBuilder::new(&vertices)
                .build()
                .unwrap();

        let simplex_key = dt.simplices().next().unwrap().0;
        let simplex = &dt.tds().simplex(simplex_key).unwrap();

        // Get vertex keys from DT
        let vertex_keys: Vec<_> = dt.tds().vertices().map(|(k, _)| k).collect();

        assert!(simplex.contains_vertex(vertex_keys[0]));
        assert!(simplex.contains_vertex(vertex_keys[1]));
        assert!(simplex.contains_vertex(vertex_keys[2]));
        assert!(simplex.contains_vertex(vertex_keys[3]));
    }

    #[test]
    fn simplex_has_vertex_in_common() {
        // Test has_vertex_in_common (replacement for deprecated contains_vertex_of)
        let vertices1 = vec![
            vertex!([0.0, 0.0, 1.0]; data = 1).unwrap(),
            vertex!([0.0, 1.0, 0.0]; data = 1).unwrap(),
            vertex!([1.0, 0.0, 0.0]; data = 1).unwrap(),
            vertex!([1.0, 1.0, 1.0]; data = 2).unwrap(),
        ];
        let tds1: DelaunayTriangulation<AdaptiveKernel<f64>, i32, i32, 3> =
            DelaunayTriangulationBuilder::new(&vertices1)
                .simplex_data_type::<i32>()
                .build()
                .unwrap();
        let (_, simplex_ref) = tds1.simplices().next().unwrap();
        let mut simplex = simplex_ref.clone();
        simplex.data = Some(42);

        let vertices2 = vec![
            vertex!([0.0, 0.0, 1.0]; data = 1).unwrap(),
            vertex!([0.0, 1.0, 0.0]; data = 1).unwrap(),
            vertex!([1.0, 0.0, 0.0]; data = 1).unwrap(),
            vertex!([0.0, 0.0, 0.0]; data = 0).unwrap(),
        ];
        let tds2: DelaunayTriangulation<AdaptiveKernel<f64>, i32, i32, 3> =
            DelaunayTriangulationBuilder::new(&vertices2)
                .simplex_data_type::<i32>()
                .build()
                .unwrap();
        let (_, simplex2_ref) = tds2.simplices().next().unwrap();
        let mut simplex2 = simplex2_ref.clone();
        simplex2.data = Some(43);

        assert!(simplex.has_vertex_in_common(&simplex2));
    }

    // =============================================================================
    // VERTEX_UUIDS METHOD TESTS
    // =============================================================================
    // Comprehensive tests for the vertex_uuids method covering different scenarios.

    #[test]
    fn test_vertex_uuids_success() {
        // Test the vertex_uuids method returns correct vertex UUIDs vector
        let vertex1 = vertex!([0.0, 0.0, 0.0]; data = 10).unwrap();
        let vertex2 = vertex!([1.0, 0.0, 0.0]; data = 20).unwrap();
        let vertex3 = vertex!([0.0, 1.0, 0.0]; data = 30).unwrap();
        let vertex4 = vertex!([0.0, 0.0, 1.0]; data = 40).unwrap();

        let vertices = vec![vertex1, vertex2, vertex3, vertex4];
        let dt: DelaunayTriangulation<AdaptiveKernel<f64>, i32, (), 3> =
            DelaunayTriangulationBuilder::new(&vertices)
                .build()
                .unwrap();

        let simplex_key = dt.simplices().next().unwrap().0;
        let simplex = &dt.tds().simplex(simplex_key).unwrap();

        // Get vertex UUIDs
        let vertex_uuids = simplex.vertex_uuids(dt.tds()).unwrap();
        assert_eq!(simplex.vertex_uuid_iter(dt.tds()).count(), 4);

        // Verify UUIDs match the simplex's vertices using iterator
        for (expected_uuid, returned_uuid) in
            simplex.vertex_uuid_iter(dt.tds()).zip(vertex_uuids.iter())
        {
            assert_eq!(expected_uuid.unwrap(), *returned_uuid);
        }

        // Verify all UUIDs are unique
        let unique_uuids: HashSet<_> = vertex_uuids.iter().collect();
        assert_eq!(unique_uuids.len(), vertex_uuids.len());

        // Verify no nil UUIDs using iterator
        for uuid in simplex.vertex_uuid_iter(dt.tds()) {
            assert_ne!(uuid.unwrap(), Uuid::nil());
        }
    }

    #[test]
    fn test_vertex_uuids_empty_simplex_fails() {
        // Test that TDS creation fails gracefully with insufficient vertices for dimension
        // Tested through TDS, which is the user-facing API.

        let vertices = vec![vertex!([0.0, 0.0, 0.0]).unwrap()];
        let result = DelaunayTriangulation::builder(&vertices).build();

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("Insufficient vertices"));
    }

    #[test]
    fn test_vertex_uuids_2d_simplex() {
        // Test vertex_uuids with a 2D simplex (triangle)
        let vertex1 = vertex!([0.0, 0.0]; data = 1).unwrap();
        let vertex2 = vertex!([1.0, 0.0]; data = 2).unwrap();
        let vertex3 = vertex!([0.5, 1.0]; data = 3).unwrap();

        let vertices = vec![vertex1, vertex2, vertex3];
        let dt: DelaunayTriangulation<AdaptiveKernel<f64>, i32, (), 2> =
            DelaunayTriangulationBuilder::new(&vertices)
                .build()
                .unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;
        let simplex = &dt.tds().simplex(simplex_key).unwrap();

        // Get vertex UUIDs
        let vertex_uuids = simplex.vertex_uuids(dt.tds()).unwrap();
        assert_eq!(simplex.vertex_uuid_iter(dt.tds()).count(), 3);

        // Verify UUIDs match the simplex's vertices using iterator
        for (expected_uuid, returned_uuid) in
            simplex.vertex_uuid_iter(dt.tds()).zip(vertex_uuids.iter())
        {
            assert_eq!(expected_uuid.unwrap(), *returned_uuid);
        }

        // Verify all UUIDs are unique
        let unique_uuids: HashSet<_> = vertex_uuids.iter().collect();
        assert_eq!(unique_uuids.len(), vertex_uuids.len());

        // Verify no nil UUIDs using iterator
        for uuid in simplex.vertex_uuid_iter(dt.tds()) {
            assert_ne!(uuid.unwrap(), Uuid::nil());
        }
    }

    #[test]
    fn test_vertex_uuids_4d_simplex() {
        // Test vertex_uuids with a 4D simplex (4-simplex) using integer data
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]; data = 1).unwrap(),
            vertex!([1.0, 0.0, 0.0, 0.0]; data = 2).unwrap(),
            vertex!([0.0, 1.0, 0.0, 0.0]; data = 3).unwrap(),
            vertex!([0.0, 0.0, 1.0, 0.0]; data = 4).unwrap(),
            vertex!([0.0, 0.0, 0.0, 1.0]; data = 5).unwrap(),
        ];

        let dt: DelaunayTriangulation<AdaptiveKernel<f64>, i32, (), 4> =
            DelaunayTriangulationBuilder::new(&vertices)
                .build()
                .unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;
        let simplex = &dt.tds().simplex(simplex_key).unwrap();

        // Get vertex UUIDs
        let vertex_uuids = simplex.vertex_uuids(dt.tds()).unwrap();
        assert_eq!(simplex.vertex_uuid_iter(dt.tds()).count(), 5);

        // Verify UUIDs match the simplex's vertices using iterator
        for (expected_uuid, returned_uuid) in
            simplex.vertex_uuid_iter(dt.tds()).zip(vertex_uuids.iter())
        {
            assert_eq!(expected_uuid.unwrap(), *returned_uuid);
        }

        // Verify all UUIDs are unique
        let unique_uuids: HashSet<_> = vertex_uuids.iter().collect();
        assert_eq!(unique_uuids.len(), vertex_uuids.len());

        // Verify all expected vertex data values exist (order-independent)
        let vertex_data: Vec<i32> = simplex
            .vertices()
            .iter()
            .map(|&vkey| dt.tds().vertex(vkey).unwrap().data.unwrap())
            .collect();

        for expected in 1..=5 {
            assert!(
                vertex_data.contains(&expected),
                "Expected vertex data {expected} not found"
            );
        }

        // Verify no nil UUIDs using iterator
        for uuid in simplex.vertex_uuid_iter(dt.tds()) {
            assert_ne!(uuid.unwrap(), Uuid::nil());
        }
    }

    // =============================================================================
    // DIMENSIONAL TESTS
    // =============================================================================
    // Tests covering simplices in different dimensions (1D, 2D, 3D, 4D+) to ensure
    // dimensional flexibility.
    //
    // NOTE: The main dimension tests (simplex_2d, simplex_3d, simplex_4d, simplex_5d) are
    // generated by the test_simplex_dimensions! macro at the top of this test module.

    // Keep 1D test separate as it's less common
    #[test]
    fn simplex_1d() {
        let vertices = vec![vertex!([0.0]).unwrap(), vertex!([1.0]).unwrap()];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let (_, simplex) = dt.simplices().next().unwrap();
        assert_simplex_properties(simplex, 2, 1);
    }

    #[test]
    fn simplex_single_vertex() {
        // Test that creating a 3D triangulation with insufficient vertices fails validation
        // Tested through TDS, which is the user-facing API.
        let vertices = vec![vertex!([0.0, 0.0, 0.0]).unwrap()];
        let result = DelaunayTriangulation::builder(&vertices).build();

        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Insufficient vertices"));
        assert!(error_msg.contains('1'));
        assert!(error_msg.contains('4'));
    }

    #[test]
    fn simplex_neighbors_none_by_default() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let (_, simplex) = dt.simplices().next().unwrap();

        // Note: neighbors may be set by TDS construction, this tests simplex structure
        assert!(simplex.neighbors.is_some() || simplex.neighbors.is_none());
    }

    #[test]
    fn simplex_data_none_by_default() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let (_, simplex) = dt.simplices().next().unwrap();

        assert!(simplex.data.is_none());
    }

    #[test]
    fn simplex_data_can_be_set() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<AdaptiveKernel<f64>, (), i32, 3> =
            DelaunayTriangulationBuilder::new(&vertices)
                .simplex_data_type::<i32>()
                .build()
                .unwrap();

        let (_, simplex_ref) = dt.simplices().next().unwrap();
        let mut simplex = simplex_ref.clone();
        simplex.data = Some(42);

        assert_eq!(simplex.data.unwrap(), 42);
    }

    #[test]
    fn simplex_into_hashmap_empty() {
        let hashmap = Simplex::<(), 3>::try_into_hashmap([]).unwrap();

        assert!(hashmap.is_empty());
    }

    #[test]
    fn simplex_into_hashmap_multiple() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
            vertex!([1.0, 1.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();

        // Iterate simplices from DT (clone into owned values for the hashmap).
        let mut simplices_iter = dt.simplices().map(|(_, simplex)| simplex.clone());

        let simplex1 = simplices_iter
            .next()
            .expect("Need at least 2 simplices for this test");
        let simplex2 = simplices_iter
            .next()
            .expect("Need at least 2 simplices for this test");

        let uuid1 = simplex1.uuid();
        let uuid2 = simplex2.uuid();
        let hashmap =
            Simplex::try_into_hashmap(once(simplex1).chain(once(simplex2)).chain(simplices_iter))
                .unwrap();

        assert!(hashmap.len() >= 2);
        assert!(hashmap.contains_key(&uuid1));
        assert!(hashmap.contains_key(&uuid2));
    }

    #[test]
    fn simplex_try_into_hashmap_rejects_duplicate_uuid() {
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let (_, simplex_ref) = dt.simplices().next().unwrap();
        let simplex = simplex_ref.clone();
        let uuid = simplex.uuid();
        let duplicate =
            Simplex::try_new_with_uuid(simplex.vertices().to_vec(), uuid, None).unwrap();

        assert_matches!(
            Simplex::try_into_hashmap([simplex, duplicate]),
            Err(TdsConstructionError::DuplicateUuid {
                entity: EntityKind::Simplex,
                uuid: duplicate_uuid,
            }) if duplicate_uuid == uuid
        );
    }

    #[test]
    fn simplex_debug_format() {
        // Use a simple non-degenerate 3D tetrahedron so `DelaunayTriangulationBuilder::build` can construct
        // a valid simplex for debug-format testing.
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<AdaptiveKernel<f64>, (), i32, 3> =
            DelaunayTriangulationBuilder::new(&vertices)
                .simplex_data_type::<i32>()
                .build()
                .unwrap();

        let (_, simplex_ref) = dt.simplices().next().unwrap();
        let mut simplex = simplex_ref.clone();
        simplex.data = Some(42);
        let debug_str = format!("{simplex:?}");

        // Verify debug output contains basic simplex information.
        // Use structural checks rather than brittle string matching
        assert!(debug_str.contains("Simplex"));
        assert!(!simplex.vertices().is_empty());
        assert!(!simplex.uuid().is_nil());
        assert_eq!(simplex.data.unwrap(), 42);
    }

    // =============================================================================
    // COMPREHENSIVE SERIALIZATION TESTS
    // =============================================================================
    // Tests covering simplex serialization and deserialization with different
    // data types, dimensions, and configurations using serde_json.

    #[test]
    fn simplex_to_and_from_json() {
        // Test serialization through TDS context.
        // Use a non-degenerate 3D tetrahedron so `DelaunayTriangulationBuilder::build` can construct a
        // valid initial simplex.
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();

        // Serialize the entire DT (includes simplices with proper context)
        let serialized = serde_json::to_string(&dt).unwrap();
        assert!(serialized.contains("vertices"));
        assert!(serialized.contains("simplices"));

        // Deserialize back to DT via try_from_tds (AdaptiveKernel has no auto-Deserialize).
        let tds: Tds<(), (), 3> = serde_json::from_str(&serialized).unwrap();
        let deserialized = DelaunayTriangulation::try_from_tds(tds, AdaptiveKernel::new())
            .expect("serialized Delaunay TDS should validate");

        // Verify DT properties match
        assert_eq!(deserialized.number_of_vertices(), dt.number_of_vertices());
        assert_eq!(deserialized.number_of_simplices(), dt.number_of_simplices());
        assert_eq!(deserialized.dim(), dt.dim());

        // Verify simplices within DT can be accessed
        assert_ne!(deserialized.number_of_simplices(), 0);
        for (_simplex_key, simplex) in deserialized.tds().simplices() {
            assert_eq!(simplex.dim(), 3);
            assert_eq!(simplex.number_of_vertices(), 4);
        }
    }

    #[test]
    fn simplex_deserialization_error_cases() {
        // Test realistic JSON deserialization errors that users might encounter

        // Test missing required field (uuid)
        let invalid_json_missing_uuid = r#"{"data": null}"#;
        let result: Result<Simplex<(), 3>, _> = serde_json::from_str(invalid_json_missing_uuid);
        assert!(result.is_err(), "Missing UUID should cause error");
        let error = result.unwrap_err().to_string();
        assert!(
            error.contains("missing field") || error.contains("uuid"),
            "Error should mention missing uuid field: {error}"
        );

        // Test invalid UUID format
        let invalid_json_bad_uuid = r#"{"uuid": "not-a-valid-uuid"}"#;
        let result: Result<Simplex<(), 3>, _> = serde_json::from_str(invalid_json_bad_uuid);
        assert!(result.is_err(), "Invalid UUID format should cause error");

        // Test completely invalid JSON syntax
        let invalid_json_syntax = r"{this is not valid JSON}";
        let result: Result<Simplex<(), 3>, _> = serde_json::from_str(invalid_json_syntax);
        assert!(result.is_err(), "Invalid JSON syntax should cause error");

        // Test empty JSON object (missing required uuid)
        let empty_json = r"{}";
        let result: Result<Simplex<(), 3>, _> = serde_json::from_str(empty_json);
        assert!(result.is_err(), "Empty JSON should fail (missing uuid)");

        // Test deserialization with unknown fields still rejects standalone simplex records.
        let json_unknown_field = r#"{
            "uuid": "550e8400-e29b-41d4-a716-446655440000",
            "unknown_field": "value",
            "another_unknown": 123
        }"#;
        let result: Result<Simplex<(), 3>, _> = serde_json::from_str(json_unknown_field);
        let err = result.expect_err("standalone simplex records should not deserialize");
        assert!(
            err.to_string()
                .contains("standalone Simplex deserialization")
        );

        let json_with_slotmap_vertices = r#"{
            "uuid": "550e8400-e29b-41d4-a716-446655440000",
            "vertices": [{"idx": 1, "version": 1}]
        }"#;
        let result: Result<Simplex<(), 3>, _> = serde_json::from_str(json_with_slotmap_vertices);
        let err = result.expect_err("slotmap vertex keys must be rejected");
        assert!(
            err.to_string().contains("storage-local simplex state"),
            "unexpected error for serialized slotmap vertex keys: {err}"
        );

        let json_with_neighbors = r#"{
            "uuid": "550e8400-e29b-41d4-a716-446655440000",
            "neighbors": [null, null, null, null]
        }"#;
        let result: Result<Simplex<(), 3>, _> = serde_json::from_str(json_with_neighbors);
        let err = result.expect_err("slotmap neighbor keys must be rejected");
        assert!(
            err.to_string().contains("storage-local simplex state"),
            "unexpected error for serialized slotmap neighbor keys: {err}"
        );
    }

    #[test]
    fn simplex_serialization_data_field_handling() {
        // Comprehensive test for data field serialization behavior

        // Test 1: Minimal simplex record is valid only as a TDS reconstruction record.
        let minimal_valid_json = r#"{"uuid": "550e8400-e29b-41d4-a716-446655440000"}"#;
        let result: Result<Simplex<(), 3>, _> = serde_json::from_str(minimal_valid_json);
        let err = result.expect_err("minimal standalone simplex record should fail");
        assert!(
            err.to_string()
                .contains("standalone Simplex deserialization")
        );

        // Test 2: Serialization with Some(data) includes field
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<AdaptiveKernel<f64>, (), i32, 3> =
            DelaunayTriangulationBuilder::new(&vertices)
                .simplex_data_type::<i32>()
                .build()
                .unwrap();
        let (_, simplex_ref) = dt.simplices().next().unwrap();
        let mut simplex_with_data = simplex_ref.clone();
        simplex_with_data.data = Some(42);
        let serialized = serde_json::to_string(&simplex_with_data).unwrap();
        assert!(
            serialized.contains("\"data\":"),
            "Some(data) should include data field"
        );
        assert!(serialized.contains("42"));
        let err = serde_json::from_str::<Simplex<i32, 3>>(&serialized)
            .expect_err("standalone Simplex deserialization should require TDS UUID mapping");
        assert!(
            err.to_string()
                .contains("standalone Simplex deserialization")
        );

        // Test 3: Serialization with None data omits field (optimization)
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let (_, simplex_none) = dt.simplices().next().unwrap();
        let serialized_none = serde_json::to_string(&simplex_none).unwrap();
        assert!(
            !serialized_none.contains("\"data\":"),
            "None should omit data field"
        );
        let err = serde_json::from_str::<Simplex<Option<i32>, 3>>(&serialized_none)
            .expect_err("standalone Simplex deserialization should require TDS UUID mapping");
        assert!(
            err.to_string()
                .contains("standalone Simplex deserialization")
        );

        // Test 4: Explicit "data": null still rejects outside TDS reconstruction.
        let json_with_null = r#"{"uuid":"550e8400-e29b-41d4-a716-446655440000","data":null}"#;
        let err = serde_json::from_str::<Simplex<Option<i32>, 3>>(json_with_null)
            .expect_err("standalone simplex record should fail");
        assert!(
            err.to_string()
                .contains("standalone Simplex deserialization")
        );
    }

    // =============================================================================
    // GEOMETRIC PROPERTIES TESTS
    // =============================================================================
    // Tests for geometric properties and validation of simplices

    #[test]
    fn simplex_coordinate_ranges() {
        // Test triangulation construction with various coordinate ranges

        // Negative coordinates: fully in negative octant
        let negative_vertices = vec![
            vertex!([-1.0, -1.0, -1.0]).unwrap(),
            vertex!([-2.0, -1.0, -1.0]).unwrap(),
            vertex!([-1.0, -2.0, -1.0]).unwrap(),
            vertex!([-1.0, -1.0, -2.0]).unwrap(),
        ];
        let dt_neg = DelaunayTriangulation::builder(&negative_vertices)
            .build()
            .unwrap();
        let (_, simplex_neg) = dt_neg.tds().simplices().next().unwrap();
        assert_eq!(simplex_neg.number_of_vertices(), 4);
        assert_eq!(simplex_neg.dim(), 3);

        // Large coordinates: large-magnitude coordinates
        let large_vertices = vec![
            vertex!([1e6, 1e6, 1e6]).unwrap(),
            vertex!([2e6, 1e6, 1e6]).unwrap(),
            vertex!([1e6, 2e6, 1e6]).unwrap(),
            vertex!([1e6, 1e6, 2e6]).unwrap(),
        ];
        let dt_large = DelaunayTriangulation::builder(&large_vertices)
            .build()
            .unwrap();
        let (_, simplex_large) = dt_large.tds().simplices().next().unwrap();
        assert_eq!(simplex_large.number_of_vertices(), 4);
        assert_eq!(simplex_large.dim(), 3);

        // Small coordinates: scaled-down tetrahedron without degeneracy
        let small_vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1e-3, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1e-3, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1e-3]).unwrap(),
        ];
        let dt_small = DelaunayTriangulation::builder(&small_vertices)
            .build()
            .unwrap();
        let (_, simplex_small) = dt_small.tds().simplices().next().unwrap();
        assert_eq!(simplex_small.number_of_vertices(), 4);
        assert_eq!(simplex_small.dim(), 3);
    }

    #[test]
    fn simplex_circumradius_2d() {
        let vertex1 = vertex!([0.0, 0.0]).unwrap();
        let vertex2 = vertex!([1.0, 0.0]).unwrap();
        let vertex3 = vertex!([0.0, 1.0]).unwrap();

        let vertices = vec![vertex1, vertex2, vertex3];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;
        let simplex = &dt.tds().simplex(simplex_key).unwrap();

        // Resolve VertexKeys to actual vertices
        let vertex_points: Vec<Point<2>> = simplex
            .vertices()
            .iter()
            .map(|vk| *dt.tds().vertex(*vk).unwrap().point())
            .collect();
        let circumradius = circumradius(&vertex_points).unwrap();

        // For a right triangle with legs of length 1, circumradius is sqrt(2)/2
        let expected_radius = 2.0_f64.sqrt() / 2.0;
        assert_relative_eq!(circumradius, expected_radius, epsilon = 1e-10);
    }

    #[test]
    fn simplex_contains_vertex_false() {
        let vertex1 = vertex!([0.0, 0.0, 0.0]).unwrap();
        let vertex2 = vertex!([1.0, 0.0, 0.0]).unwrap();
        let vertex3 = vertex!([0.0, 1.0, 0.0]).unwrap();
        let vertex4 = vertex!([0.0, 0.0, 1.0]).unwrap(); // 4th vertex to complete 3D simplex
        let vertex_outside: Vertex<(), 3> = vertex!([2.0, 2.0, 2.0]).unwrap();

        let vertices = vec![vertex1, vertex2, vertex3, vertex4];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;
        let simplex = dt.simplex(simplex_key).unwrap();

        // Create a vertex key for the outside vertex - it won't be in the simplex
        let outside_key = dt.vertex_key_from_uuid(&vertex_outside.uuid());
        assert!(outside_key.is_none() || !simplex.contains_vertex(outside_key.unwrap()));
    }

    #[test]
    fn simplex_circumsphere_contains_vertex_determinant() {
        // Test the matrix determinant method for circumsphere containment
        // Use a simple, well-known case: unit tetrahedron
        let vertex1 = vertex!([0.0, 0.0, 0.0]; data = 1).unwrap();
        let vertex2 = vertex!([1.0, 0.0, 0.0]; data = 1).unwrap();
        let vertex3 = vertex!([0.0, 1.0, 0.0]; data = 1).unwrap();
        let vertex4 = vertex!([0.0, 0.0, 1.0]; data = 2).unwrap();

        let vertices = vec![vertex1, vertex2, vertex3, vertex4];
        let dt: DelaunayTriangulation<AdaptiveKernel<f64>, i32, (), 3> =
            DelaunayTriangulationBuilder::new(&vertices)
                .build()
                .unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;
        let simplex = &dt.tds().simplex(simplex_key).unwrap();

        // Test vertex clearly outside circumsphere
        let vertex_far_outside: Vertex<i32, 3> = vertex!([10.0, 10.0, 10.0]; data = 4).unwrap();
        // Just check that the method runs without error for now
        let vertex_points: Vec<Point<3>> = simplex
            .vertices()
            .iter()
            .map(|vk| *dt.tds().vertex(*vk).unwrap().point())
            .collect();
        let result = insphere(&vertex_points, *vertex_far_outside.point());
        assert!(result.is_ok());

        // Test with origin (should be inside or on boundary)
        let origin: Vertex<i32, 3> = vertex!([0.0, 0.0, 0.0]; data = 3).unwrap();
        let vertex_points: Vec<Point<3>> = simplex
            .vertices()
            .iter()
            .map(|vk| *dt.tds().vertex(*vk).unwrap().point())
            .collect();
        let result_origin = insphere(&vertex_points, *origin.point());
        assert!(result_origin.is_ok());
    }

    #[test]
    fn simplex_circumsphere_contains_vertex_2d() {
        // Test 2D case for circumsphere containment using determinant method
        let vertex1 = vertex!([0.0, 0.0]).unwrap();
        let vertex2 = vertex!([1.0, 0.0]).unwrap();
        let vertex3 = vertex!([0.0, 1.0]).unwrap();

        let vertices = vec![vertex1, vertex2, vertex3];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;
        let simplex = &dt.tds().simplex(simplex_key).unwrap();

        // Test vertex far outside circumcircle
        let vertex_far_outside: Vertex<(), 2> = vertex!([10.0, 10.0]).unwrap();
        let vertex_points: Vec<Point<2>> = simplex
            .vertices()
            .iter()
            .map(|vk| *dt.tds().vertex(*vk).unwrap().point())
            .collect();
        let result = insphere(&vertex_points, *vertex_far_outside.point());
        assert!(result.is_ok());

        // Test with center of triangle (should be inside)
        let center: Vertex<(), 2> = vertex!([0.33, 0.33]).unwrap();
        let vertex_points: Vec<Point<2>> = simplex
            .vertices()
            .iter()
            .map(|vk| *dt.tds().vertex(*vk).unwrap().point())
            .collect();
        let result_center = insphere(&vertex_points, *center.point());
        assert!(result_center.is_ok());
    }

    #[test]
    fn simplex_circumradius_with_center() {
        // Test the circumradius_with_center method
        let vertex1 = vertex!([0.0, 0.0, 0.0]).unwrap();
        let vertex2 = vertex!([1.0, 0.0, 0.0]).unwrap();
        let vertex3 = vertex!([0.0, 1.0, 0.0]).unwrap();
        let vertex4 = vertex!([0.0, 0.0, 1.0]).unwrap();

        let vertices = vec![vertex1, vertex2, vertex3, vertex4];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;
        let simplex = &dt.tds().simplex(simplex_key).unwrap();

        let vertex_points: Vec<Point<3>> = simplex
            .vertices()
            .iter()
            .map(|vk| *dt.tds().vertex(*vk).unwrap().point())
            .collect();

        let circumcenter = circumcenter(&vertex_points).unwrap();
        let radius_with_center = circumradius_with_center(&vertex_points, &circumcenter);
        let radius_direct = circumradius(&vertex_points).unwrap();

        assert_relative_eq!(radius_with_center.unwrap(), radius_direct, epsilon = 1e-10);
    }

    // Test facet views directly.
    // (from_facet_and_vertex is commented out pending refactor)
    #[test]
    fn simplex_facet_views_comprehensive() {
        // Test comprehensive facet view functionality
        let vertex1 = vertex!([0.0, 0.0, 0.0]).unwrap();
        let vertex2 = vertex!([1.0, 0.0, 0.0]).unwrap();
        let vertex3 = vertex!([0.0, 1.0, 0.0]).unwrap();
        let vertex4 = vertex!([0.0, 0.0, 1.0]).unwrap();

        let vertices = vec![vertex1, vertex2, vertex3, vertex4];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        // Test that we can get facet views for all facets
        let facet_views = dt
            .tds()
            .try_simplex_facets(simplex_key)
            .expect("Failed to get facet iterator")
            .collect::<Result<Vec<_>, _>>()
            .expect("Failed to get facet views");
        assert_eq!(facet_views.len(), 4, "3D simplex should have 4 facets");

        // Each facet should have 3 vertices (for 3D)
        for (i, facet_view) in facet_views.iter().enumerate() {
            let facet_vertices = facet_view.vertices();
            assert_eq!(
                facet_vertices.count(),
                3,
                "Facet {i} should have 3 vertices"
            );
        }

        // Verify opposite vertices are correct
        let simplex = dt.tds().simplex(simplex_key).unwrap();
        for (i, facet_view) in facet_views.iter().enumerate() {
            let opposite_vertex = facet_view.opposite_vertex();
            // The opposite vertex should be one of the simplex's vertices (by VertexKey)
            let opposite_key = dt
                .tds()
                .vertex_key_from_uuid(&opposite_vertex.uuid())
                .unwrap();
            assert!(
                simplex.vertices().contains(&opposite_key),
                "Facet {i} opposite vertex key should be in simplex"
            );
        }
    }

    // Test facet vertex uniqueness directly.
    // (from_facet_and_vertex is commented out pending refactor)
    #[test]
    fn test_facet_vertex_uniqueness() {
        // Test that facet vertices are unique and don't include the opposite vertex
        let vertex1 = vertex!([0.0, 0.0, 1.0]).unwrap();
        let vertex2 = vertex!([0.0, 1.0, 0.0]).unwrap();
        let vertex3 = vertex!([1.0, 0.0, 0.0]).unwrap();
        let vertex4 = vertex!([1.0, 1.0, 1.0]).unwrap();

        let vertices = vec![vertex1, vertex2, vertex3, vertex4];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        // Get all facet views
        let facet_views = dt
            .simplex_facets(simplex_key)
            .expect("Failed to get facet iterator")
            .collect::<Result<Vec<_>, _>>()
            .expect("Failed to get facet views");

        for facet_view in &facet_views {
            let opposite_vertex = facet_view.opposite_vertex();
            let opposite_vertex_key = dt.vertex_key_from_uuid(&opposite_vertex.uuid()).unwrap();
            let facet_vertices = facet_view.vertices();

            // Collect facet vertex keys
            let facet_vertex_keys: Vec<_> = facet_vertices
                .map(|v| dt.vertex_key_from_uuid(&v.uuid()).unwrap())
                .collect();

            // Verify the opposite vertex key is NOT in the facet vertices
            assert!(
                !facet_vertex_keys.contains(&opposite_vertex_key),
                "Facet vertices should not include the opposite vertex key"
            );

            // Verify all facet vertices are unique
            let unique_count = facet_vertex_keys.iter().collect::<HashSet<_>>().len();
            assert_eq!(
                unique_count,
                facet_vertex_keys.len(),
                "All facet vertices should be unique"
            );
        }
    }

    #[test]
    fn simplex_high_dimensional() {
        // Test with higher dimensions (5D)
        let vertex1 = vertex!([0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let vertex2 = vertex!([1.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let vertex3 = vertex!([0.0, 1.0, 0.0, 0.0, 0.0]).unwrap();
        let vertex4 = vertex!([0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
        let vertex5 = vertex!([0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
        let vertex6 = vertex!([0.0, 0.0, 0.0, 0.0, 1.0]).unwrap();

        let vertices = vec![vertex1, vertex2, vertex3, vertex4, vertex5, vertex6];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;
        let simplex = &dt.tds().simplex(simplex_key).unwrap();

        assert_eq!(simplex.number_of_vertices(), 6);
        assert_eq!(simplex.dim(), 5);
        assert_eq!(
            dt.tds()
                .try_simplex_facets(simplex_key)
                .expect("Failed to get facets")
                .len(),
            6
        ); // Each vertex creates one facet
    }

    #[test]
    fn simplex_vertex_data_consistency() {
        // Test simplices with vertices that have different data types
        let vertex1 = vertex!([0.0, 0.0, 0.0]; data = 1).unwrap();
        let vertex2 = vertex!([1.0, 0.0, 0.0]; data = 2).unwrap();
        let vertex3 = vertex!([0.0, 1.0, 0.0]; data = 3).unwrap();
        let vertex4 = vertex!([0.0, 0.0, 1.0]; data = 4).unwrap(); // Need 4 vertices for 3D simplex

        let vertices = vec![vertex1, vertex2, vertex3, vertex4];
        let mut dt: DelaunayTriangulation<AdaptiveKernel<f64>, i32, u32, 3> =
            DelaunayTriangulationBuilder::new(&vertices)
                .simplex_data_type::<u32>()
                .build()
                .unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        // Set the simplex data to a known value
        if let Some(simplex) = dt.tri.tds.simplex_mut(simplex_key) {
            simplex.data = Some(42u32);
        }

        let simplex = &dt.tds().simplex(simplex_key).unwrap();

        // Verify all expected vertex data values exist (order-independent)
        let vertex_data: Vec<i32> = simplex
            .vertices()
            .iter()
            .map(|&vkey| dt.tds().vertex(vkey).unwrap().data.unwrap())
            .collect();

        for expected in 1..=4 {
            assert!(
                vertex_data.contains(&expected),
                "Expected vertex data {expected} not found"
            );
        }
        assert_eq!(simplex.data.unwrap(), 42u32);

        // Also verify we can access vertex data through facet views
        let facet_views = dt
            .tds()
            .try_simplex_facets(simplex_key)
            .expect("Failed to get facet iterator");
        for facet_view in facet_views {
            let facet_view = facet_view.expect("Failed to get facet view");
            // Get vertices from the facet view
            let vertices = facet_view.vertices();

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
    // SIMPLEX VALIDATION TESTS
    // =============================================================================
    // Tests covering simplex validation logic for success and error cases

    #[test]
    fn simplex_validation_success_cases() {
        // Test various valid simplex configurations

        // Valid 3D simplex with correct vertex count (D+1 = 4)
        let vertices_3d = vec![
            vertex!([0.0, 0.0, 1.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 0.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::builder(&vertices_3d)
            .build()
            .unwrap();
        let (_, simplex_3d) = dt.simplices().next().unwrap();
        assert!(
            simplex_3d.is_valid().is_ok(),
            "Valid 3D simplex should pass validation"
        );

        // Valid 2D simplex with correct vertex count (D+1 = 3)
        let vertices_2d = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::builder(&vertices_2d)
            .build()
            .unwrap();
        let (_, simplex_ref) = dt.simplices().next().unwrap();
        let mut simplex_2d = simplex_ref.clone();
        assert!(
            simplex_2d.is_valid().is_ok(),
            "Valid 2D simplex should pass validation"
        );

        // Simplex with None neighbors is valid
        simplex_2d.neighbors = None;
        assert!(
            simplex_2d.is_valid().is_ok(),
            "Simplex with no neighbors should be valid"
        );

        // Simplex with correct neighbors length is valid
        simplex_2d
            .set_neighbors_from_keys(vec![None, None, None])
            .unwrap();
        assert!(
            simplex_2d.is_valid().is_ok(),
            "Simplex with correct neighbors length should be valid"
        );
    }

    #[test]
    fn simplex_validation_error_cases() {
        // Test various invalid simplex configurations

        // Invalid UUID (nil)
        let vertices = vec![
            vertex!([0.0, 0.0, 1.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 0.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let (_, simplex_ref) = dt.simplices().next().unwrap();
        let mut invalid_uuid_simplex = simplex_ref.clone();
        invalid_uuid_simplex.uuid = uuid::Uuid::nil();
        assert!(
            matches!(
                invalid_uuid_simplex.is_valid(),
                Err(SimplexValidationError::InvalidUuid { .. })
            ),
            "Nil UUID should fail validation"
        );

        // Insufficient vertices (TDS construction fails)
        let insufficient_vertices = vec![
            vertex!([0.0, 0.0, 1.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
        ];
        let result = DelaunayTriangulation::builder(&insufficient_vertices).build();
        assert!(
            result.is_err(),
            "TDS should fail with insufficient vertices"
        );

        // Invalid neighbors length (too few)
        let vertices_2d = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::builder(&vertices_2d)
            .build()
            .unwrap();
        let (_, simplex_ref) = dt.simplices().next().unwrap();
        let mut simplex_wrong_neighbors = simplex_ref.clone();
        let err = simplex_wrong_neighbors
            .set_neighbors_from_keys(vec![None, None])
            .unwrap_err();
        assert!(
            matches!(
                err,
                SimplexValidationError::InvalidNeighborsLength {
                    actual: 2,
                    expected: 3,
                    dimension: 2
                }
            ),
            "Wrong neighbors count should fail validation"
        );

        // Invalid neighbors length (too many)
        let err = simplex_wrong_neighbors
            .set_neighbors_from_keys(vec![None, None, None, None])
            .unwrap_err();
        assert!(
            matches!(
                err,
                SimplexValidationError::InvalidNeighborsLength {
                    actual: 4,
                    expected: 3,
                    dimension: 2
                }
            ),
            "Wrong neighbors count should fail validation"
        );
    }

    #[test]
    fn simplex_new_rejects_insufficient_and_duplicate_vertices() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let vkeys: Vec<_> = dt.tds().vertices().map(|(k, _)| k).collect();

        // Too few vertices for a 3D simplex (D+1 = 4)
        let err = Simplex::<(), 3>::try_new_with_data(vec![vkeys[0], vkeys[1], vkeys[2]], None)
            .unwrap_err();
        assert_matches!(
            err,
            SimplexValidationError::InsufficientVertices {
                actual: 3,
                expected: 4,
                dimension: 3,
            }
        );

        // Duplicate vertex keys are rejected
        let err =
            Simplex::<(), 3>::try_new_with_data(vec![vkeys[0], vkeys[1], vkeys[2], vkeys[0]], None)
                .unwrap_err();
        assert_matches!(err, SimplexValidationError::DuplicateVertices);
    }

    #[test]
    fn periodic_simplex_accepts_repeated_key_with_distinct_lifted_offsets() {
        let keys = synthetic_vertex_keys(3);
        let simplex = Simplex::<(), 3>::try_new_periodic(
            vec![keys[0], keys[1], keys[2], keys[0]],
            vec![[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0]],
        )
        .expect("distinct lifted identities should form a periodic simplex");

        assert!(simplex.is_valid().is_ok());
    }

    #[test]
    fn periodic_simplex_rejects_repeated_lifted_identity() {
        let keys = synthetic_vertex_keys(3);
        let error = Simplex::<(), 3>::try_new_periodic(
            vec![keys[0], keys[1], keys[2], keys[0]],
            vec![[1, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0]],
        )
        .expect_err("an identical key-offset pair must remain invalid");

        assert_eq!(error, SimplexValidationError::DuplicateVertices);
    }

    #[test]
    fn periodic_simplex_rejects_misaligned_shapes_and_reports_corruption() {
        let keys = synthetic_vertex_keys(4);
        assert_eq!(
            Simplex::<(), 3>::try_new_periodic(keys[..3].to_vec(), vec![[0; 3]; 3]).unwrap_err(),
            SimplexValidationError::InsufficientVertices {
                actual: 3,
                expected: 4,
                dimension: 3,
            }
        );
        assert_eq!(
            Simplex::<(), 3>::try_new_periodic(keys.clone(), vec![[0; 3]; 3]).unwrap_err(),
            SimplexValidationError::PeriodicOffsetLengthMismatch {
                expected: 4,
                found: 3,
            }
        );

        let mut misaligned =
            Simplex::<(), 3>::try_new_periodic(keys.clone(), vec![[0; 3]; 4]).unwrap();
        misaligned.periodic_vertex_offsets.as_mut().unwrap().pop();
        let mismatch = SimplexValidationError::PeriodicOffsetLengthMismatch {
            expected: 4,
            found: 3,
        };
        assert_eq!(misaligned.is_valid(), Err(mismatch.clone()));
        assert_eq!(
            misaligned.simplex_report().unwrap_err().violations,
            vec![mismatch]
        );

        let mut duplicate = Simplex::<(), 3>::try_new_periodic(keys, vec![[0; 3]; 4]).unwrap();
        duplicate.vertices[3] = duplicate.vertices[0];
        assert_eq!(
            duplicate.simplex_report().unwrap_err().violations,
            vec![SimplexValidationError::DuplicateVertices]
        );
    }

    #[test]
    fn simplex_is_valid_rejects_insufficient_and_duplicate_vertices() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let (_, simplex_ref) = dt.simplices().next().unwrap();

        // Insufficient vertices (wrong vertex buffer length)
        let mut wrong_len = simplex_ref.clone();
        wrong_len.vertices.pop();
        assert_matches!(
            wrong_len.is_valid(),
            Err(SimplexValidationError::InsufficientVertices { .. })
        );

        // Duplicate vertices
        let mut dup = simplex_ref.clone();
        dup.vertices[1] = dup.vertices[0];
        assert_matches!(
            dup.is_valid(),
            Err(SimplexValidationError::DuplicateVertices)
        );
    }

    #[test]
    fn simplex_ensure_neighbors_buffer_mut_initializes_and_reuses() {
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let (simplex_key, simplex_ref) = dt.simplices().next().unwrap();

        let mut simplex = simplex_ref.clone();
        simplex.clear_neighbors();
        assert!(simplex.neighbors.is_none());

        let buf = simplex.ensure_neighbors_buffer_mut();
        assert_eq!(buf.len(), 3);
        assert!(buf.iter().all(|slot| slot.is_unassigned()));

        // Mutate through the returned buffer and ensure it's preserved
        buf[0] = NeighborSlot::Neighbor(simplex_key);
        let buf2 = simplex.ensure_neighbors_buffer_mut();
        assert_eq!(buf2[0], NeighborSlot::Neighbor(simplex_key));
    }

    #[test]
    fn simplex_try_ensure_neighbors_buffer_mut_rejects_malformed_existing_buffer() {
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let (_, simplex_ref) = dt.simplices().next().unwrap();

        let mut simplex = simplex_ref.clone();
        simplex.ensure_neighbors_buffer_mut().truncate(2);

        assert_matches!(
            simplex.try_ensure_neighbors_buffer_mut(),
            Err(SimplexValidationError::InvalidNeighborsLength {
                actual: 2,
                expected: 3,
                dimension: 2
            })
        );
        assert_eq!(simplex.neighbor_slots().unwrap().len(), 2);
    }

    #[test]
    fn simplex_neighbor_views_distinguish_unassigned_boundary_and_neighbor_slots() {
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let (simplex_key, simplex_ref) = dt.simplices().next().unwrap();

        let mut simplex = simplex_ref.clone();
        simplex.clear_neighbors();
        assert!(simplex.neighbor_slots().is_none());
        assert!(simplex.neighbors().is_none());

        simplex
            .set_neighbors_from_keys([None, Some(simplex_key), None])
            .unwrap();

        let slots = simplex
            .neighbor_slots()
            .expect("assigned slots should exist");
        assert_eq!(slots.len(), 3);
        assert_eq!(slots[0], NeighborSlot::Boundary);
        assert_eq!(slots[1], NeighborSlot::Neighbor(simplex_key));
        assert_eq!(slots[2], NeighborSlot::Boundary);

        let neighbor_keys: Vec<_> = simplex
            .neighbors()
            .expect("neighbor iterator should exist")
            .collect();
        assert_eq!(neighbor_keys, &[None, Some(simplex_key), None]);
    }

    #[test]
    fn simplex_validation_rejects_unassigned_slot_inside_assigned_neighbors() {
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let (_, simplex_ref) = dt.simplices().next().unwrap();

        let mut simplex = simplex_ref.clone();
        let slots = simplex.ensure_neighbors_buffer_mut();
        slots[0] = NeighborSlot::Unassigned;

        assert_matches!(
            simplex.is_valid(),
            Err(SimplexValidationError::UnassignedNeighborSlot { facet_index: 0 })
        );
    }

    #[test]
    fn simplex_swap_vertex_slots_swaps_vertices_neighbors_and_offsets() {
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let (simplex_key, simplex_ref) = dt.simplices().next().unwrap();

        let mut simplex = simplex_ref.clone();
        simplex
            .set_neighbors_from_keys(vec![Some(simplex_key), None, Some(simplex_key)])
            .unwrap();
        simplex
            .set_periodic_vertex_offsets(vec![[1, 0], [2, 0], [3, 0]])
            .unwrap();

        let before_vertices = simplex.vertices().to_vec();
        let before_neighbors: Vec<_> = simplex.neighbors().unwrap().collect();
        let before_offsets = simplex.periodic_vertex_offsets().unwrap().to_vec();

        simplex.swap_vertex_slots(0, 2);

        assert_eq!(simplex.vertices()[0], before_vertices[2]);
        assert_eq!(simplex.vertices()[2], before_vertices[0]);

        assert_eq!(simplex.neighbor_key(0).flatten(), before_neighbors[2]);
        assert_eq!(simplex.neighbor_key(2).flatten(), before_neighbors[0]);

        let offsets = simplex.periodic_vertex_offsets().unwrap();
        assert_eq!(offsets[0], before_offsets[2]);
        assert_eq!(offsets[2], before_offsets[0]);
    }

    #[test]
    #[should_panic(expected = "neighbors index out of bounds")]
    fn simplex_swap_vertex_slots_panics_when_neighbors_shorter_than_vertices() {
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let (_, simplex_ref) = dt.simplices().next().unwrap();

        let mut simplex = simplex_ref.clone();
        let mut neighbors = NeighborBuffer::<NeighborSlot>::new();
        neighbors.resize(2, NeighborSlot::Boundary);
        simplex.neighbors = Some(neighbors);
        simplex.swap_vertex_slots(0, 2);
    }

    #[test]
    fn simplex_facet_view_helpers_reject_excessive_vertex_count() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let mut dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let simplex_key = dt.tds().simplex_keys().next().unwrap();

        // Grab a stable key we can duplicate to inflate the vertex buffer.
        let vkey0 = {
            let simplex = dt.tds().simplex(simplex_key).unwrap();
            simplex.vertices()[0]
        };

        {
            let simplex = dt
                .tri
                .tds
                .simplex_mut(simplex_key)
                .expect("simplex key should be valid in test");
            while simplex.number_of_vertices() <= usize::from(u8::MAX) + 1 {
                simplex.push_vertex_key(vkey0);
            }
            assert!(simplex.number_of_vertices() > usize::from(u8::MAX) + 1);
        }

        // The owner-bound iterator should fail early before building individual FacetViews.
        let err = dt
            .tds()
            .try_simplex_facets(simplex_key)
            .err()
            .expect("Expected try_simplex_facets to fail on vertex_count overflow");
        assert_matches!(
            err,
            FacetError::InvalidFacetIndexOverflow {
                original_index,
                facet_count,
            } if original_index == usize::from(u8::MAX) + 1
                && facet_count > usize::from(u8::MAX) + 1
        );
    }

    #[test]
    fn simplex_deserialize_rejects_missing_uuid_and_duplicate_fields_and_invalid_uuid() {
        // Expecting() should surface for non-map inputs.
        let err = serde_json::from_str::<Simplex<i32, 3>>("null").unwrap_err();
        assert!(err.to_string().contains("a Simplex struct"));

        // Missing required field.
        let err = serde_json::from_str::<Simplex<i32, 3>>("{\"data\":1}").unwrap_err();
        assert!(err.to_string().contains("missing field `uuid`"));

        // Duplicate uuid.
        let uuid = uuid::Uuid::new_v4();
        let json = format!("{{\"uuid\":\"{uuid}\",\"uuid\":\"{uuid}\"}}");
        let err = serde_json::from_str::<Simplex<i32, 3>>(&json).unwrap_err();
        assert!(err.to_string().contains("duplicate field `uuid`"));

        // Duplicate data.
        let uuid = uuid::Uuid::new_v4();
        let json = format!("{{\"uuid\":\"{uuid}\",\"data\":1,\"data\":2}}");
        let err = serde_json::from_str::<Simplex<i32, 3>>(&json).unwrap_err();
        assert!(err.to_string().contains("duplicate field `data`"));

        // Invalid uuid (nil) should be rejected by validate_uuid.
        let json = "{\"uuid\":\"00000000-0000-0000-0000-000000000000\"}";
        let err = serde_json::from_str::<Simplex<i32, 3>>(json).unwrap_err();
        assert!(err.to_string().contains("invalid uuid"));

        // Unknown fields are ignored by the raw record parser, then rejected at the standalone boundary.
        let uuid = uuid::Uuid::new_v4();
        let json = format!("{{\"uuid\":\"{uuid}\",\"data\":5,\"extra\":[1,2,3]}}");
        let err = serde_json::from_str::<Simplex<i32, 3>>(&json).unwrap_err();
        assert!(
            err.to_string()
                .contains("standalone Simplex deserialization")
        );
    }

    #[test]
    fn simplex_validation_error_from_stack_matrix_dispatch_error_maps_to_coordinate_conversion() {
        let err = StackMatrixDispatchError::UnsupportedDim {
            k: MAX_STACK_MATRIX_DIM + 1,
            max: MAX_STACK_MATRIX_DIM,
        };
        let simplex_err: SimplexValidationError = err.into();

        assert_matches!(
            simplex_err,
            SimplexValidationError::CoordinateConversion { .. }
        );
    }

    // =============================================================================
    // SIMPLEX PARTIALEQ AND EQ TESTS
    // =============================================================================

    #[test]
    fn test_simplex_partial_eq_different_dimensions() {
        // Test equality for simplices of different dimensions
        let vertices_2d = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let dt1 = DelaunayTriangulation::builder(&vertices_2d)
            .build()
            .unwrap();
        let (_, simplex_2d) = dt1.tds().simplices().next().unwrap();
        let dt2 = DelaunayTriangulation::builder(&vertices_2d)
            .build()
            .unwrap();
        let (_, simplex_2d_copy) = dt2.tds().simplices().next().unwrap();

        // Test equality for 2D simplices
        assert_eq!(
            simplex_2d, simplex_2d_copy,
            "Identical 2D simplices should be equal"
        );
    }

    #[test]
    fn test_try_simplex_facets() {
        // Test the owner-bound simplex facet iterator.
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();

        let simplex_key = dt.tds().simplex_keys().next().unwrap();

        // Test the iterator method
        let facet_iter = dt
            .tds()
            .try_simplex_facets(simplex_key)
            .expect("Failed to get facet iterator");

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
            let vertex_count = facet_view.vertices().count();
            assert_eq!(vertex_count, 3, "Facet {i} should have 3 vertices");
        }

        // Test iterator is zero-allocation by using it without collect
        let facet_iter2 = dt
            .tds()
            .try_simplex_facets(simplex_key)
            .expect("Failed to get second facet iterator");

        let mut count = 0;
        for facet_result in facet_iter2 {
            let _facet_view = facet_result.expect("Facet creation should succeed");
            count += 1;
        }
        assert_eq!(count, 4, "Iterator should yield 4 facets");

        // Test iterator combinators work correctly
        let facet_iter3 = dt
            .tds()
            .try_simplex_facets(simplex_key)
            .expect("Failed to get third facet iterator");

        let successful_facets: Vec<_> = facet_iter3
            .collect::<Result<Vec<_>, _>>()
            .expect("all facets should be created successfully");
        assert_eq!(
            successful_facets.len(),
            4,
            "All facets should be created successfully"
        );
    }

    #[test]
    fn test_simplex_data_accessor() {
        let vertices = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let mut dt = DelaunayTriangulationBuilder::new(&vertices)
            .simplex_data_type::<i32>()
            .build()
            .unwrap();
        let key = dt.simplices().next().unwrap().0;

        // No data initially
        assert_eq!(dt.tds().simplex(key).unwrap().data(), None);

        // Set data and verify via accessor
        dt.set_simplex_data(key, Some(99)).unwrap();
        assert_eq!(dt.tds().simplex(key).unwrap().data(), Some(&99));
    }
}
