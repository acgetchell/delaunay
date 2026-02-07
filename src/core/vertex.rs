//! Data and operations on d-dimensional vertices.
//!
//! This module provides the `Vertex` struct which represents a geometric vertex
//! in D-dimensional space with associated metadata including unique identification,
//! incident cell references, and optional user data.
//!
//! # Key Features
//!
//! - **Generic Coordinate Support**: Works with any floating-point type (`f32`, `f64`, etc.)
//!   that implements the `CoordinateScalar` trait
//! - **Unique Identification**: Each vertex has a UUID for consistent identification
//! - **Optional Data Storage**: Supports attaching user data of any type `U` that implements [`DataType`], or use `()` for no data
//! - **Incident Cell Tracking**: Maintains references to containing cells
//! - **Serialization Support**: Serde support for persistence (`incident_cell` is reconstructed by TDS)
//! - **Builder Pattern**: Convenient vertex construction using `VertexBuilder`
//!
//! # Examples
//!
//! ```rust
//! use delaunay::core::vertex::Vertex;
//! use delaunay::vertex;
//!
//! // Create a simple vertex
//! let vertex: Vertex<f64, (), 3> = vertex!([1.0, 2.0, 3.0]);
//!
//! // Create vertex with data
//! let vertex_with_data: Vertex<f64, i32, 2> = vertex!([1.0, 2.0], 42);
//! ```

#![forbid(unsafe_code)]

use super::{
    traits::DataType,
    triangulation_data_structure::CellKey,
    util::{UuidValidationError, make_uuid, validate_uuid},
};
use crate::geometry::{
    point::Point,
    traits::coordinate::{Coordinate, CoordinateScalar, CoordinateValidationError},
};
use serde::{
    Deserialize, Serialize,
    de::{self, IgnoredAny, MapAccess, Visitor},
};
use std::{
    cmp::Ordering,
    collections::HashMap,
    fmt::{self, Debug},
    hash::{Hash, Hasher},
    marker::PhantomData,
};
use thiserror::Error;
use uuid::Uuid;

// =============================================================================
// ERROR TYPES
// =============================================================================

/// Errors that can occur during vertex validation.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::util::UuidValidationError;
/// use delaunay::core::vertex::VertexValidationError;
///
/// let err = VertexValidationError::InvalidUuid {
///     source: UuidValidationError::NilUuid,
/// };
/// assert!(matches!(err, VertexValidationError::InvalidUuid { .. }));
/// ```
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum VertexValidationError {
    /// The vertex has an invalid point.
    #[error("Invalid point: {source}")]
    InvalidPoint {
        /// The underlying point validation error.
        #[from]
        source: CoordinateValidationError,
    },
    /// The vertex has an invalid UUID.
    #[error("Invalid UUID: {source}")]
    InvalidUuid {
        /// The underlying UUID validation error.
        #[from]
        source: UuidValidationError,
    },
}

// =============================================================================
// CONVENIENCE MACROS AND HELPERS
// =============================================================================

/// Convenience macro for creating vertices with less boilerplate.
///
/// This macro simplifies vertex creation by using the `VertexBuilder` pattern internally
/// and automatically unwrapping the result for convenience. It takes coordinate arrays
/// and optional data, returning a `Vertex` directly.
///
/// # Returns
///
/// Returns `Vertex<T, U, D>` where:
/// - `T` is the coordinate scalar type
/// - `U` is the user data type (use `()` for no data)
/// - `D` is the spatial dimension
///
/// # Panics
///
/// Panics if the `VertexBuilder` fails to construct a valid vertex, which should
/// not happen under normal circumstances with valid input data.
///
/// # Usage
///
/// ```rust
/// use delaunay::vertex;
/// use delaunay::core::vertex::Vertex;
///
/// // Create a vertex without data
/// let v1: Vertex<f64, (), 3> = vertex!([1.0, 2.0, 3.0]);
///
/// // Create a vertex with data
/// let v2: Vertex<f64, i32, 2> = vertex!([0.0, 1.0], 42);
/// ```
#[macro_export]
macro_rules! vertex {
    // Pattern 1: Just coordinates - no data (defaults to ())
    ($coords:expr) => {
        $crate::core::vertex::VertexBuilder::<_, (), _>::default()
            .point($crate::geometry::point::Point::try_from($coords)
                .expect("Failed to convert coordinates to Point: invalid or out-of-range values"))
            .build()
            .expect("Failed to build vertex: invalid coordinates or builder configuration")
    };

    // Pattern 2: Coordinates with data
    ($coords:expr, $data:expr) => {
        $crate::core::vertex::VertexBuilder::default()
            .point($crate::geometry::point::Point::try_from($coords)
                .expect("Failed to convert coordinates to Point: invalid or out-of-range values"))
            .data($data)
            .build()
            .expect("Failed to build vertex with data: invalid coordinates, data, or builder configuration")
    };
}

// Re-export the macro at the crate level for convenience
pub use crate::vertex;

// =============================================================================
// VERTEX STRUCT DEFINITION
// =============================================================================

#[derive(Builder, Clone, Copy, Debug)]
/// The `Vertex` struct represents a vertex in a triangulation with geometric
/// coordinates, unique identification, and optional metadata.
///
/// # Generic Parameters
///
/// * `T` - The scalar coordinate type (typically `f32` or `f64`)
/// * `U` - User data type that implements `DataType` (use `()` for no data)
/// * `D` - The spatial dimension (compile-time constant)
///
/// # Properties
///
/// - **`point`**: A `Point<T, D>` representing the geometric coordinates of the vertex
/// - **`uuid`**: A universally unique identifier for the vertex (auto-generated)
/// - **`incident_cell`**: Optional reference to a containing cell (managed by TDS)
/// - **`data`**: Optional user-defined data associated with the vertex
///
/// # Constraints
///
/// - `T` must implement `CoordinateScalar` (floating-point operations, validation, etc.)
/// - `U` must implement `DataType` (serialization, equality, hashing, etc.)
///
/// # Usage
///
/// Vertices are typically created using the builder pattern for convenience:
///
/// ```rust
/// use delaunay::core::vertex::Vertex;
/// use delaunay::vertex;
///
/// let vertex: Vertex<f64, i32, 3> = vertex!([1.0, 2.0, 3.0], 42);
/// ```
pub struct Vertex<T, U, const D: usize>
where
    T: CoordinateScalar,
    U: DataType,
{
    /// The coordinates of the vertex as a D-dimensional Point.
    point: Point<T, D>,
    /// A universally unique identifier for the vertex.
    #[builder(setter(skip), default = "make_uuid()")]
    uuid: Uuid,
    /// The `CellKey` of the cell that the vertex is incident to.
    /// Phase 3: Changed from UUID to direct key reference for performance.
    ///
    /// Note: This field is not serialized because `CellKey` is only valid within
    /// the current `SlotMap` instance. During deserialization, the TDS automatically
    /// reconstructs `incident_cell` mappings via `assign_incident_cells()`.
    #[builder(setter(skip), default = "None")]
    pub incident_cell: Option<CellKey>,
    /// Optional data associated with the vertex.
    #[builder(setter(into, strip_option), default)]
    pub data: Option<U>,
}

// =============================================================================
// SERIALIZATION IMPLEMENTATION
// =============================================================================

/// Manual implementation of Serialize for Vertex.
///
/// This implementation handles serialization of all vertex fields. The `incident_cell`
/// field is skipped as it's a runtime-only reference that gets reconstructed during
/// deserialization.
impl<T, U, const D: usize> Serialize for Vertex<T, U, D>
where
    T: CoordinateScalar,
    U: DataType,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let field_count = if self.data.is_some() { 3 } else { 2 };
        let mut state = serializer.serialize_struct("Vertex", field_count)?;
        state.serialize_field("point", &self.point)?;
        state.serialize_field("uuid", &self.uuid)?;
        if self.data.is_some() {
            state.serialize_field("data", &self.data)?;
        }
        state.end()
    }
}

// =============================================================================
// DESERIALIZATION IMPLEMENTATION
// =============================================================================

/// Manual implementation of Deserialize for Vertex.
///
/// This custom implementation ensures proper handling of all vertex fields
/// during deserialization, including validation of required fields.
impl<'de, T, U, const D: usize> Deserialize<'de> for Vertex<T, U, D>
where
    T: CoordinateScalar,
    U: DataType,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: serde::Deserializer<'de>,
    {
        struct VertexVisitor<T, U, const D: usize>
        where
            T: CoordinateScalar,
            U: DataType,
        {
            _phantom: PhantomData<(T, U)>,
        }

        impl<'de, T, U, const D: usize> Visitor<'de> for VertexVisitor<T, U, D>
        where
            T: CoordinateScalar,
            U: DataType,
        {
            type Value = Vertex<T, U, D>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a Vertex struct")
            }

            fn visit_map<V>(self, mut map: V) -> Result<Vertex<T, U, D>, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut point: Option<Point<T, D>> = None;
                let mut uuid = None;
                let mut incident_cell = None;
                let mut data = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        "point" => {
                            if point.is_some() {
                                return Err(de::Error::duplicate_field("point"));
                            }
                            point = Some(map.next_value()?);
                        }
                        "uuid" => {
                            if uuid.is_some() {
                                return Err(de::Error::duplicate_field("uuid"));
                            }
                            uuid = Some(map.next_value()?);
                        }
                        "incident_cell" => {
                            if incident_cell.is_some() {
                                return Err(de::Error::duplicate_field("incident_cell"));
                            }
                            // Phase 3: Ignore payload to accept both legacy UUID and new CellKey formats.
                            // TDS reconstructs incident_cell mappings via assign_incident_cells().
                            let _ = map.next_value::<IgnoredAny>()?;
                            incident_cell = Some(None);
                        }
                        "data" => {
                            if data.is_some() {
                                return Err(de::Error::duplicate_field("data"));
                            }
                            data = Some(map.next_value()?);
                        }
                        _ => {
                            let _ = map.next_value::<IgnoredAny>()?;
                        }
                    }
                }

                let point = point.ok_or_else(|| de::Error::missing_field("point"))?;
                let uuid: Uuid = uuid.ok_or_else(|| de::Error::missing_field("uuid"))?;
                validate_uuid(&uuid)
                    .map_err(|e| de::Error::custom(format!("invalid uuid: {e}")))?;
                let incident_cell = incident_cell.unwrap_or(None);
                let data = data.unwrap_or(None);

                // Validate point before constructing
                point.validate().map_err(|e| {
                    de::Error::custom(format!("Invalid point during deserialization: {e}"))
                })?;

                Ok(Vertex {
                    point,
                    uuid,
                    incident_cell,
                    data,
                })
            }
        }

        const FIELDS: &[&str] = &["point", "uuid", "incident_cell", "data"];
        deserializer.deserialize_struct(
            "Vertex",
            FIELDS,
            VertexVisitor {
                _phantom: PhantomData,
            },
        )
    }
}

// =============================================================================
// VERTEX IMPLEMENTATION - CORE METHODS
// =============================================================================

impl<T, U, const D: usize> Vertex<T, U, D>
where
    T: CoordinateScalar,
    U: DataType,
{
    /// Creates an empty vertex at the origin with nil UUID and default data.
    ///
    /// This method creates a vertex with coordinates all set to `T::default()` (typically zero),
    /// a nil UUID, and default data. This is useful for creating placeholder vertices or
    /// for testing purposes.
    ///
    /// Note: A vertex created with `empty()` will fail validation due to the nil UUID.
    /// Use the `vertex!` macro or `new()` constructor for creating valid vertices.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::core::vertex::Vertex;
    /// use delaunay::geometry::traits::coordinate::Coordinate;
    /// use approx::assert_relative_eq;
    ///
    /// let empty_vertex: Vertex<f64, (), 3> = Vertex::empty();
    /// assert_relative_eq!(
    ///     empty_vertex.point().coords().as_slice(),
    ///     [0.0, 0.0, 0.0].as_slice(),
    ///     epsilon = 1e-9
    /// );
    /// assert!(empty_vertex.uuid().is_nil());
    /// assert!(empty_vertex.data.is_none());
    /// ```
    #[must_use]
    pub fn empty() -> Self
    where
        T: Default,
    {
        Self {
            point: Point::default(),
            uuid: Uuid::nil(),
            incident_cell: None,
            data: None,
        }
    }

    /// The function `from_points` takes a slice of points and returns a
    /// vector of vertices, using the `new` method.
    ///
    /// # Arguments
    ///
    /// * `points`: `points` is a slice of [Point] objects.
    ///
    /// # Returns
    ///
    /// The function `from_points` returns a `Vec<Vertex<T, U, D>>`, where `T`
    /// is the type of the coordinates of the [Vertex], `U` is the type of the
    /// optional data associated with the [Vertex], and `D` is the
    /// dimensionality of the [Vertex].
    ///
    /// # Panics
    ///
    /// Panics if the `VertexBuilder` fails to build a vertex from any point.
    /// This should not happen under normal circumstances with valid point data.
    ///
    /// # Example
    ///
    /// ```
    /// use delaunay::core::vertex::Vertex;
    /// use delaunay::geometry::point::Point;
    /// use delaunay::geometry::traits::coordinate::Coordinate;
    /// let points = [Point::new([1.0, 2.0, 3.0])];
    /// let vertices: Vec<Vertex<f64, (), 3>> = Vertex::from_points(&points);
    /// assert_eq!(vertices.len(), 1);
    /// assert_eq!(vertices[0].point().coords(), &[1.0, 2.0, 3.0]);
    /// ```
    #[inline]
    #[must_use]
    pub fn from_points(points: &[Point<T, D>]) -> Vec<Self> {
        points
            .iter()
            .map(|p| VertexBuilder::default().point(*p).build().unwrap())
            .collect()
    }

    /// The function `into_hashmap` converts a collection of vertices into a
    /// [`HashMap`], using the vertices [Uuid] as the key.
    ///
    /// # Arguments
    ///
    /// * `vertices`: Vertices to be converted into a `HashMap`.
    ///
    /// # Returns
    ///
    /// The function `into_hashmap` returns a [`HashMap`] with the key type
    /// [Uuid] and the value type [Vertex], i.e. `std::collections::HashMap<Uuid, Vertex<T, U, D>>`.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use delaunay::core::vertex::Vertex;
    /// use delaunay::geometry::point::Point;
    /// use delaunay::geometry::traits::coordinate::Coordinate;
    /// let points = vec![Point::new([1.0, 2.0]), Point::new([3.0, 4.0])];
    /// let vertices = Vertex::<f64, (), 2>::from_points(&points);
    /// let map: HashMap<_, _> = Vertex::into_hashmap(vertices);
    /// assert_eq!(map.len(), 2);
    /// assert!(map.values().all(|v| v.dim() == 2));
    /// ```
    #[inline]
    #[must_use]
    pub fn into_hashmap<I>(vertices: I) -> HashMap<Uuid, Self>
    where
        I: IntoIterator<Item = Self>,
    {
        vertices.into_iter().map(|v| (v.uuid(), v)).collect()
    }

    /// Returns the point coordinates of the vertex.
    ///
    /// # Returns
    ///
    /// A reference to the Point representing the vertex's coordinates.
    ///
    /// # Example
    ///
    /// ```
    /// use delaunay::core::vertex::Vertex;
    /// use delaunay::vertex;
    /// use delaunay::geometry::traits::coordinate::Coordinate;
    ///
    /// let vertex: Vertex<f64, (), 3> = vertex!([1.0, 2.0, 3.0]);
    /// let retrieved_point = vertex.point();
    /// assert_eq!(retrieved_point.coords(), &[1.0, 2.0, 3.0]);
    /// ```
    #[inline]
    pub const fn point(&self) -> &Point<T, D> {
        &self.point
    }

    /// Returns the UUID of the vertex.
    ///
    /// # Returns
    ///
    /// The Uuid uniquely identifying this vertex.
    ///
    /// # Example
    ///
    /// ```
    /// use delaunay::core::vertex::Vertex;
    /// use delaunay::vertex;
    /// use uuid::Uuid;
    ///
    /// let vertex: Vertex<f64, (), 3> = vertex!([1.0, 2.0, 3.0]);
    /// let vertex_uuid = vertex.uuid();
    /// // UUID should be valid and unique
    /// assert_ne!(vertex_uuid, Uuid::nil());
    ///
    /// // Creating another vertex should have a different UUID
    /// let another_vertex: Vertex<f64, (), 3> = vertex!([1.0, 2.0, 3.0]);
    /// assert_ne!(vertex.uuid(), another_vertex.uuid());
    /// ```
    #[inline]
    pub const fn uuid(&self) -> Uuid {
        self.uuid
    }

    /// Sets the vertex UUID with validation.
    ///
    /// This is a test-only utility for creating vertices with specific UUIDs
    /// to test error handling (e.g., duplicate UUID detection).
    ///
    /// # Arguments
    ///
    /// * `uuid` - The new UUID to set for this vertex
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the UUID is valid and was set successfully,
    /// otherwise returns a `VertexValidationError::InvalidUuid` if the UUID
    /// is nil or has an invalid version.
    ///
    /// # Errors
    ///
    /// Returns `VertexValidationError::InvalidUuid` if the UUID is nil or invalid.
    #[cfg(test)]
    pub(crate) fn set_uuid(&mut self, uuid: Uuid) -> Result<(), VertexValidationError> {
        // Validate the UUID before setting it
        validate_uuid(&uuid)?;

        self.uuid = uuid;
        Ok(())
    }

    /// The `dim` function returns the dimensionality of the [Vertex].
    ///
    /// # Returns
    ///
    /// The `dim` function is returning the value of `D`, which the number of
    /// coordinates.
    ///
    /// # Example
    /// ```
    /// use delaunay::core::vertex::Vertex;
    /// use delaunay::vertex;
    ///
    /// let vertex: Vertex<f64, (), 4> = vertex!([1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(vertex.dim(), 4);
    /// ```
    #[inline]
    pub const fn dim(&self) -> usize {
        D
    }

    /// The function `is_valid` checks if a [Vertex] is valid.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the [Vertex] is valid, otherwise returns a
    /// `VertexValidationError` indicating the specific validation failure.
    /// A valid vertex has:
    /// - A valid [Point] with finite coordinates (no NaN or infinite values)
    /// - A valid [Uuid] that is not nil
    ///
    /// # Errors
    ///
    /// Returns `VertexValidationError::InvalidPoint` if the point has invalid coordinates,
    /// or `VertexValidationError::InvalidUuid` if the UUID is nil.
    ///
    /// # Example
    ///
    /// ```
    /// use delaunay::core::vertex::{Vertex, VertexValidationError};
    /// use delaunay::vertex;
    /// use uuid::Uuid;
    ///
    /// let vertex: Vertex<f64, (), 3> = vertex!([1.0, 2.0, 3.0]);
    /// assert!(vertex.is_valid().is_ok());
    ///
    /// // Test with empty vertex (which has nil UUID) to show validation
    /// let default_vertex: Vertex<f64, (), 3> = Vertex::empty();
    /// match default_vertex.is_valid() {
    ///     Err(VertexValidationError::InvalidUuid { .. }) => (), // Expected - nil UUID
    ///     other => panic!("Expected InvalidUuid error, got: {:?}", other),
    /// }
    /// ```
    pub fn is_valid(self) -> Result<(), VertexValidationError>
    where
        Point<T, D>: Coordinate<T, D>,
    {
        // Check if the point is valid using the Coordinate trait validation
        self.point
            .validate()
            .map_err(|source| VertexValidationError::InvalidPoint { source })?;

        // Check if UUID is valid using centralized validation
        validate_uuid(&self.uuid())?;

        Ok(())
        // Note: incident_cell validation is handled at the TDS level via:
        // - Tds::assign_incident_cells() ensures proper cell assignment
        // - Tds::is_valid() validates cell mappings and references
        // Individual vertices cannot validate incident_cell without TDS context.
        // User data validation (if U: DataType requires it) could be added here.
    }

    /// Create a vertex for testing with a specific UUID.
    ///
    /// # ⚠️ WARNING: Internal Test Helper - Do Not Use
    ///
    /// This function is **only for internal testing** and bypasses:
    /// - UUID auto-generation
    /// - UUID validation
    /// - All safety checks
    ///
    /// Using this function in production code will likely lead to:
    /// - Duplicate UUID collisions
    /// - Invalid triangulation state
    /// - Data corruption
    ///
    /// **Always use** the `vertex!` macro or builder pattern in production code.
    ///
    /// # Arguments
    ///
    /// * `point` - The point coordinates for the vertex
    /// * `uuid` - The UUID to assign to the vertex (must be unique!)
    /// * `data` - Optional user data for the vertex
    ///
    /// # Returns
    ///
    /// A new `Vertex` with the specified UUID and data (unchecked).
    #[doc(hidden)]
    pub const fn new_with_uuid(point: Point<T, D>, uuid: Uuid, data: Option<U>) -> Self {
        Self {
            point,
            uuid,
            incident_cell: None,
            data,
        }
    }
}

// =============================================================================
// STANDARD TRAIT IMPLEMENTATIONS
// =============================================================================
impl<T, U, const D: usize> PartialEq for Vertex<T, U, D>
where
    T: CoordinateScalar,
    U: DataType,
{
    /// Equality of vertices is based on ordered equality of coordinates using the Coordinate trait.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.point.ordered_equals(&other.point)
        // && self.uuid == other.uuid
        // && self.incident_cell == other.incident_cell
        // && self.data == other.data
    }
}

impl<T, U, const D: usize> PartialOrd for Vertex<T, U, D>
where
    T: CoordinateScalar,
    U: DataType,
{
    /// Order of vertices is based on lexicographic order of coordinates using Point's `partial_cmp`.
    /// This ensures consistent ordering with special floating-point values (NaN, infinity)
    /// through `OrderedFloat` semantics.
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.point.partial_cmp(&other.point)
    }
}

/// Enable implicit conversion from Vertex to coordinate array
/// This allows using `Into` to convert from `Vertex` to `[T; D]`
impl<T, U, const D: usize> From<Vertex<T, U, D>> for [T; D]
where
    T: CoordinateScalar,
    U: DataType,
{
    #[inline]
    fn from(vertex: Vertex<T, U, D>) -> [T; D] {
        *vertex.point().coords()
    }
}

/// Enable implicit conversion from Vertex reference to coordinate array
/// This allows `&vertex` to be implicitly converted to `[T; D]` for coordinate access
impl<T, U, const D: usize> From<&Vertex<T, U, D>> for [T; D]
where
    T: CoordinateScalar,
    U: DataType,
{
    #[inline]
    fn from(vertex: &Vertex<T, U, D>) -> [T; D] {
        *vertex.point().coords()
    }
}

/// Enable implicit conversion from Vertex reference to Point
/// This allows `&vertex` to be implicitly converted to `Point<T, D>`
impl<T, U, const D: usize> From<&Vertex<T, U, D>> for Point<T, D>
where
    T: CoordinateScalar,
    U: DataType,
{
    #[inline]
    fn from(vertex: &Vertex<T, U, D>) -> Self {
        *vertex.point()
    }
}

// =============================================================================
// HASHING AND EQUALITY IMPLEMENTATIONS
// =============================================================================
impl<T, U, const D: usize> Eq for Vertex<T, U, D>
where
    T: CoordinateScalar,
    U: DataType,
{
    // Generic Eq implementation for Vertex based on point equality
}

impl<T, U, const D: usize> Hash for Vertex<T, U, D>
where
    T: CoordinateScalar,
    U: DataType,
    Point<T, D>: Hash,
{
    /// Hash implementation for Vertex using only coordinates for consistency with `PartialEq`.
    ///
    /// This ensures that vertices with the same coordinates have the same hash,
    /// maintaining the Eq/Hash contract: if a == b, then hash(a) == hash(b).
    ///
    /// Note: UUID, `incident_cell`, and data are excluded from hashing to match
    /// the `PartialEq` implementation which only compares coordinates.
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.point.hash_coordinate(state);
        // Intentionally exclude UUID, incident_cell, and data to maintain
        // consistency with PartialEq implementation
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::collections::{FastHashMap, FastHashSet};
    use crate::core::triangulation_data_structure::CellKey;
    use crate::core::util::{UuidValidationError, make_uuid, usize_to_u8};
    use crate::geometry::point::Point;
    use crate::geometry::traits::coordinate::Coordinate;
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use serde::{Deserialize, Serialize};
    use slotmap::KeyData;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    // Test enum for demonstrating vertex data usage
    #[repr(u8)]
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    enum PointType {
        Origin = 0,
        Boundary = 1,
        Interior = 2,
        Corner = 3,
    }

    impl From<PointType> for u8 {
        fn from(point_type: PointType) -> Self {
            point_type as Self
        }
    }

    impl Serialize for PointType {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            serializer.serialize_u8(u8::from(*self))
        }
    }

    impl<'de> Deserialize<'de> for PointType {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            let value = u8::deserialize(deserializer)?;
            match value {
                0 => Ok(Self::Origin),
                1 => Ok(Self::Boundary),
                2 => Ok(Self::Interior),
                3 => Ok(Self::Corner),
                _ => Err(serde::de::Error::custom(format!(
                    "Invalid PointType: {value}"
                ))),
            }
        }
    }

    // =============================================================================
    // HELPER FUNCTIONS
    // =============================================================================

    /// Simplified helper function to test basic vertex properties
    fn assert_vertex_properties<T, U, const D: usize>(
        vertex: &Vertex<T, U, D>,
        expected_coords: [T; D],
    ) where
        T: CoordinateScalar,
        U: DataType,
    {
        assert_eq!(vertex.point().coords(), &expected_coords);
        assert_eq!(vertex.dim(), D);
        assert!(!vertex.uuid().is_nil());
        assert!(vertex.incident_cell.is_none());
    }

    // =============================================================================
    // CONVENIENCE MACRO AND HELPER TESTS
    // =============================================================================

    #[test]
    fn test_vertex_macro() {
        // Test new macro syntax without data - no None required!
        let v1: Vertex<f64, (), 3> = vertex!([1.0, 2.0, 3.0]);
        assert_relative_eq!(
            v1.point().coords().as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(v1.dim(), 3);
        assert!(!v1.uuid().is_nil());
        assert!(v1.data.is_none());

        // Test new macro syntax with data - no Some() required!
        let v2: Vertex<f64, i32, 2> = vertex!([0.0, 1.0], 99);
        assert_relative_eq!(
            v2.point().coords().as_slice(),
            [0.0, 1.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(v2.dim(), 2);
        assert!(!v2.uuid().is_nil());
        assert_eq!(v2.data.unwrap(), 99);

        // Test macro with different data type (using Copy type)
        let v3: Vertex<f64, u32, 4> = vertex!([1.0, 2.0, 3.0, 4.0], 42u32);
        assert_relative_eq!(
            v3.point().coords().as_slice(),
            [1.0f64, 2.0f64, 3.0f64, 4.0f64].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(v3.dim(), 4);
        assert_eq!(v3.data.unwrap(), 42u32);
    }

    // =============================================================================
    // POINTTYPE CONVERSION TESTS
    // =============================================================================

    #[test]
    fn test_pointtype_u8_conversion() {
        // Test u8::from conversion for each PointType variant
        assert_eq!(u8::from(PointType::Origin), 0);
        assert_eq!(u8::from(PointType::Boundary), 1);
        assert_eq!(u8::from(PointType::Interior), 2);
        assert_eq!(u8::from(PointType::Corner), 3);

        // Test serialization uses the From trait correctly
        let origin_json = serde_json::to_string(&PointType::Origin).unwrap();
        assert_eq!(origin_json, "0");

        let corner_json = serde_json::to_string(&PointType::Corner).unwrap();
        assert_eq!(corner_json, "3");

        // Test round-trip serialization/deserialization
        let original = PointType::Boundary;
        let serialized = serde_json::to_string(&original).unwrap();
        let deserialized: PointType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(original, deserialized);

        println!("✓ PointType u8::from conversion works correctly");
    }

    // =============================================================================
    // BASIC VERTEX FUNCTIONALITY
    // =============================================================================

    #[test]
    fn test_vertex_basic_operations() {
        // Test Vertex::empty()
        let empty_vertex: Vertex<f64, (), 3> = Vertex::empty();
        assert_relative_eq!(
            empty_vertex.point().coords().as_slice(),
            [0.0, 0.0, 0.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(empty_vertex.dim(), 3);
        assert!(empty_vertex.uuid().is_nil());
        assert!(empty_vertex.incident_cell.is_none());
        assert!(empty_vertex.data.is_none());

        // Test vertex copying
        let vertex: Vertex<f64, u8, 4> = vertex!([1.0, 2.0, 3.0, 4.0], 4u8);
        let vertex_copy = vertex;
        assert_eq!(vertex, vertex_copy);
        assert_relative_eq!(
            vertex_copy.point().coords().as_slice(),
            [1.0, 2.0, 3.0, 4.0].as_slice(),
            epsilon = 1e-9
        );

        // Test Vertex::from_points() with multiple points
        let points = vec![
            Point::new([1.0, 2.0, 3.0]),
            Point::new([4.0, 5.0, 6.0]),
            Point::new([7.0, 8.0, 9.0]),
        ];
        let vertices: Vec<Vertex<f64, (), 3>> = Vertex::from_points(&points);

        assert_eq!(vertices.len(), 3);
        assert_relative_eq!(
            vertices[0].point().coords().as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(vertices[0].dim(), 3);
        assert_relative_eq!(
            vertices[1].point().coords().as_slice(),
            [4.0, 5.0, 6.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(vertices[1].dim(), 3);
        assert_relative_eq!(
            vertices[2].point().coords().as_slice(),
            [7.0, 8.0, 9.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(vertices[2].dim(), 3);

        // Test Vertex::from_points() with empty slice
        let empty_points: Vec<Point<f64, 3>> = Vec::new();
        let empty_vertices: Vec<Vertex<f64, (), 3>> = Vertex::from_points(&empty_points);
        assert!(empty_vertices.is_empty());

        // Test Vertex::from_points() with single point
        let single_point = vec![Point::new([1.0, 2.0, 3.0])];
        let single_vertices: Vec<Vertex<f64, (), 3>> = Vertex::from_points(&single_point);
        assert_eq!(single_vertices.len(), 1);
        assert_relative_eq!(
            single_vertices[0].point().coords().as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(single_vertices[0].dim(), 3);
        assert!(!single_vertices[0].uuid().is_nil());

        // Test Vertex::into_hashmap() with multiple vertices
        let mut vertices_clone: Vec<Vertex<f64, (), 3>> = Vertex::from_points(&points);
        let hashmap = Vertex::into_hashmap(vertices_clone.clone());
        let mut values: Vec<Vertex<f64, (), 3>> = hashmap.into_values().collect();

        assert_eq!(values.len(), 3);

        values.sort_by_key(super::Vertex::uuid);
        vertices_clone.sort_by_key(super::Vertex::uuid);

        assert_eq!(values, vertices_clone);

        // Test Vertex::into_hashmap() with empty vector
        let empty_vertices_vec: Vec<Vertex<f64, (), 3>> = Vec::new();
        let empty_hashmap = Vertex::into_hashmap(empty_vertices_vec);
        assert!(empty_hashmap.is_empty());

        // Test Vertex::into_hashmap() with single vertex
        let single_vertex: Vertex<f64, (), 3> = vertex!([1.0, 2.0, 3.0]);
        let uuid = single_vertex.uuid();
        let single_vertices_vec = vec![single_vertex];
        let single_hashmap = Vertex::into_hashmap(single_vertices_vec);

        assert_eq!(single_hashmap.len(), 1);
        assert!(single_hashmap.contains_key(&uuid));
        assert_relative_eq!(
            single_hashmap
                .get(&uuid)
                .unwrap()
                .point()
                .coords()
                .as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = 1e-9
        );

        println!("{empty_vertex:?}");
        println!("{vertices:?}");
        println!("values = {values:?}");
        println!("vertices = {vertices_clone:?}");
    }

    #[test]
    fn test_vertex_serialization_roundtrip() {
        // Test basic serialization/deserialization roundtrip
        let vertex: Vertex<f64, (), 3> = vertex!([1.0, 2.0, 3.0]);
        let serialized = serde_json::to_string(&vertex).unwrap();

        assert!(serialized.contains("point"));
        assert!(serialized.contains("[1.0,2.0,3.0]"));

        let deserialized: Vertex<f64, (), 3> = serde_json::from_str(&serialized).unwrap();

        // Check that deserialized vertex has same point coordinates using approx equality
        assert_relative_eq!(
            deserialized.point().coords().as_slice(),
            vertex.point().coords().as_slice(),
            epsilon = f64::EPSILON
        );
        assert_eq!(deserialized.dim(), vertex.dim());
        assert_eq!(deserialized.incident_cell, vertex.incident_cell);
        assert_eq!(deserialized.data, vertex.data);
        assert_eq!(deserialized.uuid(), vertex.uuid());

        // Test serialization with data
        let vertex_with_data: Vertex<f64, i32, 3> = vertex!([1.0, 2.0, 3.0], 42);
        let serialized_with_data = serde_json::to_string(&vertex_with_data).unwrap();
        assert!(serialized_with_data.contains("\"data\":"));
        assert!(serialized_with_data.contains("42"));

        let deserialized_with_data: Vertex<f64, i32, 3> =
            serde_json::from_str(&serialized_with_data).unwrap();
        assert_eq!(deserialized_with_data.data, Some(42));
        assert_relative_eq!(
            deserialized_with_data.point().coords().as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = f64::EPSILON
        );

        // Test serialization with None data (should omit data field)
        let vertex_no_data: Vertex<f64, (), 3> = vertex!([1.0, 2.0, 3.0]);
        let serialized_no_data = serde_json::to_string(&vertex_no_data).unwrap();
        assert!(!serialized_no_data.contains("\"data\":"));

        let deserialized_no_data: Vertex<f64, (), 3> =
            serde_json::from_str(&serialized_no_data).unwrap();
        assert_eq!(deserialized_no_data.data, None);

        // Test backward compatibility: explicit "data": null should still work
        let json_with_null =
            r#"{"point":[1.0,2.0,3.0],"uuid":"550e8400-e29b-41d4-a716-446655440000","data":null}"#;
        let vertex_null_data: Vertex<f64, (), 3> = serde_json::from_str(json_with_null).unwrap();
        assert_eq!(vertex_null_data.data, None);

        // Test with different data types
        let vertex_char: Vertex<f64, char, 4> = vertex!([1.0, 2.0, 3.0, 4.0], 'A');
        let serialized_char = serde_json::to_string(&vertex_char).unwrap();
        let deserialized_char: Vertex<f64, char, 4> =
            serde_json::from_str(&serialized_char).unwrap();
        assert_eq!(deserialized_char.data, Some('A'));
        assert_relative_eq!(
            deserialized_char.point().coords().as_slice(),
            [1.0, 2.0, 3.0, 4.0].as_slice(),
            epsilon = f64::EPSILON
        );

        println!("Serialized: {serialized:?}");
    }

    // =============================================================================
    // EQUALITY AND HASHING TESTS
    // =============================================================================

    /// Comprehensive tests for Vertex equality (`PartialEq`, `Eq`) and hashing (`Hash`)
    /// These tests ensure the Hash/Eq contract is properly maintained:
    /// - If a == b, then hash(a) == hash(b)
    /// - Equality is based only on vertex coordinates (not UUID or metadata)
    /// - Hash is based only on vertex coordinates (consistent with equality)

    #[test]
    fn test_vertex_equality_and_hashing() {
        // Test basic equality behavior
        let v1: Vertex<f64, (), 3> = vertex!([1.0, 2.0, 3.0]);
        let v2: Vertex<f64, (), 3> = vertex!([1.0, 2.0, 3.0]);
        let v3: Vertex<f64, (), 3> = vertex!([1.0, 2.0, 4.0]);

        // Same coordinates should be equal
        assert_eq!(v1, v2);
        assert!(v1.eq(&v2));
        assert!(v2.eq(&v1));

        // Different coordinates should not be equal
        assert_ne!(v1, v3);
        assert_ne!(v2, v3);
        assert!(!v1.eq(&v3));
        assert!(!v2.eq(&v3));

        // Test reflexivity
        assert_eq!(v1, v1);
        assert!(v1.eq(&v1));

        // Test that equality ignores UUID, incident_cell, and data
        let v4: Vertex<f64, i32, 2> = vertex!([1.0, 2.0], 42);
        let v5: Vertex<f64, i32, 2> = vertex!([1.0, 2.0], 99); // Different data

        // Different UUIDs and data but same coordinates
        assert_ne!(v4.uuid(), v5.uuid());
        assert_ne!(v4.data, v5.data);

        // Should still be equal because coordinates match
        assert_eq!(v4, v5);

        // Test with None data
        let v6: Vertex<f64, (), 2> = vertex!([1.0, 2.0]);
        let v7: Vertex<f64, (), 2> = vertex!([1.0, 2.0]);
        assert_eq!(v6, v7);

        // Test hash consistency for same vertex
        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();

        v1.hash(&mut hasher1);
        v1.hash(&mut hasher2);

        let hash1 = hasher1.finish();
        let hash2 = hasher2.finish();
        assert_eq!(hash1, hash2);

        // Test hash consistency for equal vertices (Hash/Eq contract)
        let mut hasher3 = DefaultHasher::new();
        v2.hash(&mut hasher3);
        let hash3 = hasher3.finish();

        assert_eq!(v1, v2); // Vertices are equal
        assert_eq!(hash1, hash3); // Therefore hashes must be equal

        // Different coordinates should produce different hashes
        let mut hasher4 = DefaultHasher::new();
        v3.hash(&mut hasher4);
        let hash4 = hasher4.finish();

        assert_ne!(v1, v3);
        assert_ne!(hash1, hash4);

        // Test that hash ignores UUID, incident_cell, and data (consistent with equality)
        let mut hasher5 = DefaultHasher::new();
        let mut hasher6 = DefaultHasher::new();

        v4.hash(&mut hasher5);
        v5.hash(&mut hasher6);

        let hash5 = hasher5.finish();
        let hash6 = hasher6.finish();

        // Same coordinates should produce same hash despite different metadata
        assert_eq!(v4, v5); // Equal by coordinates
        assert_eq!(hash5, hash6); // Therefore hashes must be equal
        assert_ne!(v4.uuid(), v5.uuid()); // But UUIDs are different
        assert_ne!(v4.data, v5.data); // And data is different

        // Comprehensive test of the Hash/Eq contract: if a == b, then hash(a) == hash(b)
        let test_cases: Vec<([f64; 2], [f64; 2])> = vec![
            ([0.0, 0.0], [0.0, 0.0]),
            ([1.0, 2.0], [1.0, 2.0]),
            ([-1.0, -2.0], [-1.0, -2.0]),
        ];

        for (coords1, coords2) in test_cases {
            let v_a: Vertex<f64, (), 2> = vertex!(coords1);
            let v_b: Vertex<f64, (), 2> = vertex!(coords2);

            // Verify equality
            assert_eq!(v_a, v_b);

            // Verify hash equality
            let mut hasher_a = DefaultHasher::new();
            let mut hasher_b = DefaultHasher::new();

            v_a.hash(&mut hasher_a);
            v_b.hash(&mut hasher_b);

            assert_eq!(hasher_a.finish(), hasher_b.finish());
        }
    }

    #[test]
    fn test_vertex_collections() {
        // Test vertices in collections to verify Hash/Eq contract in practice
        let mut set: FastHashSet<Vertex<f64, (), 2>> = FastHashSet::default();

        let v1: Vertex<f64, (), 2> = vertex!([1.0, 2.0]);
        let v2: Vertex<f64, (), 2> = vertex!([3.0, 4.0]);
        let v3: Vertex<f64, (), 2> = vertex!([1.0, 2.0]); // Same coordinates as v1

        // Insert vertices
        assert!(set.insert(v1)); // First insert should succeed
        assert!(set.insert(v2)); // Different coordinates, should succeed
        assert!(!set.insert(v3)); // Same coordinates as v1, should fail

        assert_eq!(set.len(), 2); // Only 2 unique vertices by coordinates

        // Check containment
        assert!(set.contains(&v1));
        assert!(set.contains(&v2));
        assert!(set.contains(&v3)); // v3 is "found" because it equals v1

        // Verify we can look up by coordinates
        let v4: Vertex<f64, (), 2> = vertex!([1.0, 2.0]);
        assert!(set.contains(&v4)); // Should find it even with different UUID

        // Test vertices as FastHashMap keys
        let mut map: FastHashMap<Vertex<f64, (), 2>, i32> = FastHashMap::default();

        let v5: Vertex<f64, (), 2> = vertex!([1.0, 2.0]);
        let v6: Vertex<f64, (), 2> = vertex!([3.0, 4.0]);

        map.insert(v5, 10);
        map.insert(v6, 20);

        // Verify lookups work
        assert_eq!(map.get(&v5), Some(&10));
        assert_eq!(map.get(&v6), Some(&20));
        assert_eq!(map.len(), 2);

        // Test lookup with equivalent vertex (same coordinates, different UUID)
        let v7: Vertex<f64, (), 2> = vertex!([1.0, 2.0]);
        assert_eq!(map.get(&v7), Some(&10)); // Should find v5's value

        // Test overwrite with equivalent vertex
        let old_value = map.insert(v7, 30);
        assert_eq!(old_value, Some(10)); // Should return v5's old value
        assert_eq!(map.len(), 2); // Size shouldn't change
        assert_eq!(map.get(&v5), Some(&30)); // v5 now maps to new value

        // Test that vertices with different data types but same coordinates work in collections
        let v8: Vertex<f64, u16, 2> = vertex!([1.0, 2.0], 999u16);
        let v9: Vertex<f64, i32, 2> = vertex!([3.0, 4.0], -42i32);

        let mut map1: FastHashMap<Vertex<f64, u16, 2>, &str> = FastHashMap::default();
        map1.insert(v8, "first");
        assert_eq!(map1.len(), 1);

        let mut map2: FastHashMap<Vertex<f64, i32, 2>, bool> = FastHashMap::default();
        map2.insert(v9, true);
        assert_eq!(map2.len(), 1);
    }

    // =============================================================================
    // DIMENSION-SPECIFIC TESTS
    // =============================================================================

    /// Macro to generate dimension-specific vertex tests for dimensions 2D-5D.
    ///
    /// This macro reduces test duplication by generating consistent tests across
    /// multiple dimensions. It creates tests for:
    /// - Basic vertex creation and property validation
    /// - Serialization roundtrip (Some and None data)
    /// - UUID validation
    ///
    /// # Usage
    ///
    /// ```ignore
    /// test_vertex_dimensions! {
    ///     vertex_2d => 2 => [1.0, 2.0],
    ///     vertex_3d => 3 => [1.0, 2.0, 3.0],
    /// }
    /// ```
    macro_rules! test_vertex_dimensions {
        ($(
            $test_name:ident => $dim:expr => [$($coord:expr),+ $(,)?]
        ),+ $(,)?) => {
            $(
                #[test]
                fn $test_name() {
                    // Test basic vertex creation
                    let vertex: Vertex<f64, (), $dim> = vertex!([$($coord),+]);
                    assert_vertex_properties(&vertex, [$($coord),+]);
                    assert!(vertex.data.is_none());
                }

                pastey::paste! {
                    #[test]
                    fn [<$test_name _with_data>]() {
                        // Test vertex with data
                        let vertex: Vertex<f64, i32, $dim> = vertex!([$($coord),+], 42);
                        assert_vertex_properties(&vertex, [$($coord),+]);
                        assert_eq!(vertex.data, Some(42));
                    }

                    #[test]
                    fn [<$test_name _serialization_roundtrip>]() {
                        // Test serialization with Some data
                        let vertex_with_data: Vertex<f64, i32, $dim> = vertex!([$($coord),+], 99);
                        let serialized = serde_json::to_string(&vertex_with_data).unwrap();
                        assert!(serialized.contains("\"data\":"));
                        let deserialized: Vertex<f64, i32, $dim> = serde_json::from_str(&serialized).unwrap();
                        assert_eq!(deserialized.data, Some(99));
                        assert_vertex_properties(&deserialized, [$($coord),+]);

                        // Test serialization with None data
                        let vertex_no_data: Vertex<f64, (), $dim> = vertex!([$($coord),+]);
                        let serialized = serde_json::to_string(&vertex_no_data).unwrap();
                        assert!(!serialized.contains("\"data\":"));
                        let deserialized: Vertex<f64, (), $dim> = serde_json::from_str(&serialized).unwrap();
                        assert_eq!(deserialized.data, None);
                    }

                    #[test]
                    fn [<$test_name _uuid_uniqueness>]() {
                        // Test UUID uniqueness for same coordinates
                        let v1: Vertex<f64, (), $dim> = vertex!([$($coord),+]);
                        let v2: Vertex<f64, (), $dim> = vertex!([$($coord),+]);
                        assert_ne!(v1.uuid(), v2.uuid());
                        assert!(!v1.uuid().is_nil());
                        assert!(!v2.uuid().is_nil());
                    }
                }
            )+
        };
    }

    // Generate tests for dimensions 2D through 5D
    test_vertex_dimensions! {
        vertex_2d => 2 => [1.0, 2.0],
        vertex_3d => 3 => [1.0, 2.0, 3.0],
        vertex_4d => 4 => [1.0, 2.0, 3.0, 4.0],
        vertex_5d => 5 => [1.0, 2.0, 3.0, 4.0, 5.0],
    }

    // Keep 1D test separate as it's less common
    #[test]
    fn vertex_1d() {
        let vertex: Vertex<f64, (), 1> = vertex!([42.0]);
        assert_vertex_properties(&vertex, [42.0]);
        assert!(vertex.data.is_none());
    }

    // =============================================================================
    // DATA TYPE TESTS
    // =============================================================================

    #[test]
    fn test_vertex_data_types_and_ordering() {
        // Test vertex with tuple data
        let vertex_tuple: Vertex<f64, (i32, i32), 2> = vertex!([1.0, 2.0], (42, 84));
        assert_vertex_properties(&vertex_tuple, [1.0, 2.0]);
        assert_eq!(vertex_tuple.data.unwrap(), (42, 84));

        // Test debug formatting
        let vertex_debug: Vertex<f64, i32, 3> = vertex!([1.0, 2.0, 3.0], 42);
        let debug_str = format!("{vertex_debug:?}");

        assert!(debug_str.contains("Vertex"));
        assert!(debug_str.contains("point"));
        assert!(debug_str.contains("uuid"));
        assert!(debug_str.contains("1.0"));
        assert!(debug_str.contains("2.0"));
        assert!(debug_str.contains("3.0"));

        // Test ordering edge cases
        let vertex1: Vertex<f64, (), 2> = vertex!([1.0, 2.0]);
        let vertex2: Vertex<f64, (), 2> = vertex!([1.0, 2.0]);

        // Test that equal points result in equal ordering
        assert!(vertex1.partial_cmp(&vertex2) != Some(Ordering::Less));
        assert!(vertex2.partial_cmp(&vertex1) != Some(Ordering::Less));
        assert!(matches!(
            vertex1.partial_cmp(&vertex2),
            Some(Ordering::Less | Ordering::Equal)
        ));
        assert!(matches!(
            vertex2.partial_cmp(&vertex1),
            Some(Ordering::Less | Ordering::Equal)
        ));
        assert!(matches!(
            vertex1.partial_cmp(&vertex2),
            Some(Ordering::Greater | Ordering::Equal)
        ));
        assert!(matches!(
            vertex2.partial_cmp(&vertex1),
            Some(Ordering::Greater | Ordering::Equal)
        ));
    }

    #[test]
    fn test_vertex_coordinate_values() {
        // Test negative coordinates
        let vertex_neg: Vertex<f64, (), 3> = vertex!([-1.0, -2.0, -3.0]);
        assert_relative_eq!(
            vertex_neg.point().coords().as_slice(),
            [-1.0, -2.0, -3.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(vertex_neg.dim(), 3);

        // Test zero coordinates
        let vertex_zero: Vertex<f64, (), 3> = vertex!([0.0, 0.0, 0.0]);
        let origin_vertex: Vertex<f64, (), 3> = vertex!([0.0, 0.0, 0.0]);
        assert_eq!(vertex_zero.point(), origin_vertex.point());

        // Test large coordinates
        let vertex_large: Vertex<f64, (), 3> = vertex!([1e6, 2e6, 3e6]);
        assert_relative_eq!(
            vertex_large.point().coords().as_slice(),
            [1_000_000.0, 2_000_000.0, 3_000_000.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(vertex_large.dim(), 3);

        // Test small coordinates
        let vertex_small: Vertex<f64, (), 3> = vertex!([1e-6, 2e-6, 3e-6]);
        assert_relative_eq!(
            vertex_small.point().coords().as_slice(),
            [0.000_001, 0.000_002, 0.000_003].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(vertex_small.dim(), 3);

        // Test mixed positive/negative coordinates
        let vertex_mixed: Vertex<f64, (), 4> = vertex!([1.0, -2.0, 3.0, -4.0]);
        assert_relative_eq!(
            vertex_mixed.point().coords().as_slice(),
            [1.0, -2.0, 3.0, -4.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(vertex_mixed.dim(), 4);
    }

    #[test]
    fn test_vertex_properties() {
        // Test UUID uniqueness
        let vertex1: Vertex<f64, (), 3> = vertex!([1.0, 2.0, 3.0]);
        let vertex2: Vertex<f64, (), 3> = vertex!([1.0, 2.0, 3.0]);

        // Same points but different UUIDs
        assert_ne!(vertex1.uuid(), vertex2.uuid());
        assert!(!vertex1.uuid().is_nil());
        assert!(!vertex2.uuid().is_nil());
    }

    #[test]
    fn test_vertex_type_conversions() {
        // Test implicit conversion from owned vertex to coordinates
        let vertex_coords: Vertex<f64, (), 3> = vertex!([1.0, 2.0, 3.0]);
        let coords_owned: [f64; 3] = vertex_coords.into();
        assert_relative_eq!(
            coords_owned.as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = 1e-9
        );

        // Test implicit conversion from vertex reference to coordinates
        let vertex_ref_coords: Vertex<f64, (), 3> = vertex!([4.0, 5.0, 6.0]);
        let coords_ref: [f64; 3] = (&vertex_ref_coords).into();
        assert_relative_eq!(
            coords_ref.as_slice(),
            [4.0, 5.0, 6.0].as_slice(),
            epsilon = 1e-9
        );

        // Verify the original vertex is still available after reference conversion
        assert_relative_eq!(
            vertex_ref_coords.point().coords().as_slice(),
            [4.0, 5.0, 6.0].as_slice(),
            epsilon = 1e-9
        );

        // Test implicit conversion from vertex reference to Point
        let vertex_point: Vertex<f64, (), 3> = vertex!([1.0, 2.0, 3.0]);
        let point_from_vertex: Point<f64, 3> = (&vertex_point).into();
        assert_relative_eq!(
            point_from_vertex.coords().as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = 1e-9
        );

        // Test that the converted point is equal to the original point
        assert_eq!(point_from_vertex, *vertex_point.point());

        // Verify the original vertex is still available after conversion
        assert_relative_eq!(
            vertex_point.point().coords().as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = 1e-9
        );

        // Test with different dimensions
        let vertex_2d: Vertex<f64, (), 2> = vertex!([10.5, -5.3]);
        let point_2d: Point<f64, 2> = (&vertex_2d).into();
        assert_relative_eq!(
            point_2d.coords().as_slice(),
            [10.5, -5.3].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(point_2d, *vertex_2d.point());
    }

    // =============================================================================
    // VALIDATION TESTS
    // =============================================================================

    #[test]
    #[expect(clippy::too_many_lines, reason = "Comprehensive validation test")]
    fn test_vertex_validation() {
        // Test valid vertices with various coordinate types and dimensions
        let valid_f64: Vertex<f64, (), 3> = vertex!([1.0, 2.0, 3.0]);
        assert!(valid_f64.is_valid().is_ok());

        let valid_f32: Vertex<f32, (), 2> = vertex!([1.5f32, 2.5f32]);
        assert!(valid_f32.is_valid().is_ok());

        let valid_negative: Vertex<f64, (), 3> = vertex!([-1.0, -2.0, -3.0]);
        assert!(valid_negative.is_valid().is_ok());

        let valid_zero: Vertex<f64, (), 3> = vertex!([0.0, 0.0, 0.0]);
        assert!(valid_zero.is_valid().is_ok());

        // Test different dimensions
        let valid_1d: Vertex<f64, (), 1> = vertex!([42.0]);
        assert!(valid_1d.is_valid().is_ok());

        let valid_5d: Vertex<f64, (), 5> = vertex!([1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(valid_5d.is_valid().is_ok());

        // Test invalid vertices with NaN coordinates
        let invalid_nan_f64: Vertex<f64, (), 3> = Vertex {
            point: Point::new([1.0, f64::NAN, 3.0]),
            uuid: make_uuid(),
            incident_cell: None,
            data: None,
        };
        assert!(invalid_nan_f64.is_valid().is_err());

        let invalid_all_nan: Vertex<f64, (), 3> = Vertex {
            point: Point::new([f64::NAN, f64::NAN, f64::NAN]),
            uuid: make_uuid(),
            incident_cell: None,
            data: None,
        };
        assert!(invalid_all_nan.is_valid().is_err());

        let invalid_nan_f32: Vertex<f32, (), 2> = Vertex {
            point: Point::new([1.0f32, f32::NAN]),
            uuid: make_uuid(),
            incident_cell: None,
            data: None,
        };
        assert!(invalid_nan_f32.is_valid().is_err());

        let invalid_1d_nan: Vertex<f64, (), 1> = Vertex {
            point: Point::new([f64::NAN]),
            uuid: make_uuid(),
            incident_cell: None,
            data: None,
        };
        assert!(invalid_1d_nan.is_valid().is_err());

        let invalid_5d_nan: Vertex<f64, (), 5> = Vertex {
            point: Point::new([1.0, 2.0, f64::NAN, 4.0, 5.0]),
            uuid: make_uuid(),
            incident_cell: None,
            data: None,
        };
        assert!(invalid_5d_nan.is_valid().is_err());

        // Test invalid vertices with infinity coordinates
        let invalid_pos_inf: Vertex<f64, (), 3> = Vertex {
            point: Point::new([1.0, f64::INFINITY, 3.0]),
            uuid: make_uuid(),
            incident_cell: None,
            data: None,
        };
        assert!(invalid_pos_inf.is_valid().is_err());

        let invalid_neg_inf: Vertex<f64, (), 3> = Vertex {
            point: Point::new([1.0, f64::NEG_INFINITY, 3.0]),
            uuid: make_uuid(),
            incident_cell: None,
            data: None,
        };
        assert!(invalid_neg_inf.is_valid().is_err());

        let invalid_inf_f32: Vertex<f32, (), 2> = Vertex {
            point: Point::new([f32::INFINITY, 2.0f32]),
            uuid: make_uuid(),
            incident_cell: None,
            data: None,
        };
        assert!(invalid_inf_f32.is_valid().is_err());

        let invalid_mixed: Vertex<f64, (), 3> = Vertex {
            point: Point::new([f64::NAN, f64::INFINITY, 1.0]),
            uuid: make_uuid(),
            incident_cell: None,
            data: None,
        };
        assert!(invalid_mixed.is_valid().is_err());

        // Test UUID validation
        let valid_vertex: Vertex<f64, (), 3> = vertex!([1.0, 2.0, 3.0]);
        assert!(valid_vertex.is_valid().is_ok());
        assert!(!valid_vertex.uuid().is_nil());

        // Test that empty vertex (which has nil UUID) is invalid
        let default_vertex: Vertex<f64, (), 3> = Vertex::empty();
        match default_vertex.is_valid() {
            Err(VertexValidationError::InvalidUuid { source: _ }) => (), // Expected
            other => panic!("Expected InvalidUuid error, got: {other:?}"),
        }
        assert!(default_vertex.uuid().is_nil());
        assert!(default_vertex.point().validate().is_ok());

        // Create a vertex with valid point but manually set nil UUID to test UUID validation
        let invalid_uuid_vertex: Vertex<f64, (), 3> = Vertex {
            point: Point::new([1.0, 2.0, 3.0]),
            uuid: uuid::Uuid::nil(),
            incident_cell: None,
            data: None,
        };
        match invalid_uuid_vertex.is_valid() {
            Err(VertexValidationError::InvalidUuid { source: _ }) => (), // Expected
            other => panic!("Expected InvalidUuid error, got: {other:?}"),
        }
        assert!(invalid_uuid_vertex.point().validate().is_ok());
        assert!(invalid_uuid_vertex.uuid().is_nil()); // UUID is nil

        // Test vertex with both invalid point and nil UUID (should return point error first)
        let invalid_both: Vertex<f64, (), 3> = Vertex {
            point: Point::new([f64::NAN, 2.0, 3.0]),
            uuid: uuid::Uuid::nil(),
            incident_cell: None,
            data: None,
        };
        match invalid_both.is_valid() {
            Err(VertexValidationError::InvalidPoint { .. }) => (), // Expected - point checked first
            other => panic!("Expected InvalidPoint error, got: {other:?}"),
        }
        assert!(invalid_both.point().validate().is_err());
        assert!(invalid_both.uuid().is_nil()); // UUID is nil
    }

    // =============================================================================
    // ADVANCED DATA TESTS
    // =============================================================================

    #[test]
    fn vertex_string_data_usage_examples() {
        // This test demonstrates what works and what doesn't work with string data in vertices.
        // Note: String data has limitations due to the DataType trait requirements and lifetime complexities.

        // =====================================================================
        // DEMONSTRATE THE FUNDAMENTAL ISSUE
        // =====================================================================

        // The following would NOT compile because String doesn't implement Copy:
        // let vertex_string: Vertex<f64, String, 2> = vertex!([1.0, 2.0], "test".to_string());
        // Error: String doesn't implement Copy trait required by DataType

        // The following would also cause lifetime issues in real usage:
        // let vertex_str: Vertex<f64, &str, 2> = vertex!([1.0, 2.0], "test");
        // While this compiles, it has severe lifetime limitations in practice

        // =====================================================================
        // PRACTICAL ALTERNATIVE: Use numeric IDs with external lookup
        // =====================================================================

        // Create a lookup table for string labels - this is the recommended approach
        let mut label_lookup: FastHashMap<u32, String> = FastHashMap::default();
        label_lookup.insert(0, "center".to_string());
        label_lookup.insert(1, "corner".to_string());
        label_lookup.insert(2, "edge_midpoint".to_string());
        label_lookup.insert(3, "boundary_point".to_string());

        // Use numeric IDs in vertices - this works perfectly and is efficient
        let vertices_with_ids: Vec<Vertex<f64, u32, 2>> = vec![
            vertex!([0.5, 0.5], 0u32), // center
            vertex!([1.0, 1.0], 1u32), // corner
            vertex!([0.5, 1.0], 2u32), // edge_midpoint
            vertex!([0.0, 0.5], 3u32), // boundary_point
        ];

        // Verify we can retrieve the labels
        for (i, v) in vertices_with_ids.iter().enumerate() {
            let label_id = v.data.unwrap();
            let label = label_lookup.get(&label_id).unwrap();
            match i {
                0 => assert_eq!(label, "center"),
                1 => assert_eq!(label, "corner"),
                2 => assert_eq!(label, "edge_midpoint"),
                3 => assert_eq!(label, "boundary_point"),
                _ => unreachable!(),
            }
        }

        // Test that these vertices work with all normal operations
        assert_eq!(vertices_with_ids.len(), 4);
        let coords = vertices_with_ids[0].point().coords();
        assert_abs_diff_eq!(coords[0], 0.5, epsilon = f64::EPSILON);
        assert_abs_diff_eq!(coords[1], 0.5, epsilon = f64::EPSILON);
        assert_eq!(vertices_with_ids[1].data.unwrap(), 1u32);

        // Test hashing and equality (works because u32 implements all required traits)
        let vertex_set: FastHashSet<Vertex<f64, u32, 2>> =
            vertices_with_ids.iter().copied().collect();
        assert_eq!(vertex_set.len(), 4);

        // =====================================================================
        // OTHER COPY-ABLE ALTERNATIVES FOR LABELS
        // =====================================================================

        // Alternative 1: Use character codes
        let vertices_with_chars: Vec<Vertex<f64, char, 2>> = vec![
            vertex!([0.0, 0.0], 'A'),
            vertex!([1.0, 0.0], 'B'),
            vertex!([0.0, 1.0], 'C'),
        ];

        for (i, v) in vertices_with_chars.iter().enumerate() {
            let expected_char =
                char::from(b'A' + usize_to_u8(i, 26).expect("Index should fit in u8"));
            assert_eq!(v.data.unwrap(), expected_char);
        }

        // Alternative 2: Use small integer codes with enum mapping

        let vertices_with_enums: Vec<Vertex<f64, PointType, 2>> = vec![
            vertex!([0.0, 0.0], PointType::Origin),
            vertex!([1.0, 0.0], PointType::Corner),
            vertex!([0.5, 0.5], PointType::Interior),
        ];

        assert_eq!(vertices_with_enums[0].data.unwrap(), PointType::Origin);
        assert_eq!(vertices_with_enums[1].data.unwrap(), PointType::Corner);
        assert_eq!(vertices_with_enums[2].data.unwrap(), PointType::Interior);

        // =====================================================================
        // SUMMARY OF STRING DATA LIMITATIONS
        // =====================================================================

        // 1. String doesn't work because it doesn't implement Copy
        // 2. &str has complex lifetime issues that make it impractical
        // 3. &'static str could work but only for compile-time constants
        // 4. Recommended alternatives:
        //    - Numeric IDs with external lookup (most flexible)
        //    - Character codes (for single characters)
        //    - Custom Copy enums (for predefined categories)
        //    - Small fixed-size byte arrays (for very short strings)
    }

    #[test]
    fn vertex_hash_with_copy_data() {
        // Test hashing with Copy data
        let vertex1: Vertex<f64, u16, 2> = vertex!([1.0, 2.0], 999u16);

        let vertex2: Vertex<f64, i32, 2> = vertex!([3.0, 4.0], 42);

        // Test that vertices with Copy data can be used as HashMap keys
        let mut map: FastHashMap<Vertex<f64, u16, 2>, i32> = FastHashMap::default();
        map.insert(vertex1, 100);

        let mut map2: FastHashMap<Vertex<f64, i32, 2>, u8> = FastHashMap::default();
        map2.insert(vertex2, 255u8);

        assert_eq!(map.len(), 1);
        assert_eq!(map2.len(), 1);
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "Comprehensive deserialization edge-case test"
    )]
    fn test_vertex_deserialization_edge_cases() {
        // Test deserialization with minimal required fields
        let json_minimal = r#"{
            "point": [10.0, 20.0],
            "uuid": "550e8400-e29b-41d4-a716-446655440000"
        }"#;

        let result: Result<Vertex<f64, (), 2>, _> = serde_json::from_str(json_minimal);
        assert!(result.is_ok());
        let vertex = result.unwrap();

        assert_relative_eq!(
            vertex.point().coords().as_slice(),
            [10.0, 20.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(
            vertex.uuid().to_string(),
            "550e8400-e29b-41d4-a716-446655440000"
        );
        assert!(vertex.incident_cell.is_none());
        assert!(vertex.data.is_none());

        // Test deserialization with all fields including CellKey
        let point = Point::new([1.5, 2.5, 3.5]);
        let uuid_str = "550e8400-e29b-41d4-a716-446655440000";
        let uuid = uuid::Uuid::parse_str(uuid_str).unwrap();
        let cell_key = CellKey::from(KeyData::from_ffi(42u64));

        let vertex_with_all = Vertex {
            point,
            uuid,
            incident_cell: Some(cell_key),
            data: Some(123i32),
        };

        assert_relative_eq!(
            vertex_with_all.point().coords().as_slice(),
            [1.5, 2.5, 3.5].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(vertex_with_all.uuid().to_string(), uuid_str);
        assert!(vertex_with_all.incident_cell.is_some());
        assert_eq!(vertex_with_all.incident_cell.unwrap(), cell_key);
        assert_eq!(vertex_with_all.data.unwrap(), 123);

        // Test unknown field handling (should be ignored)
        let json_with_unknown = r#"{
            "point": [1.0, 2.0, 3.0],
            "uuid": "550e8400-e29b-41d4-a716-446655440000",
            "incident_cell": null,
            "data": null,
            "unknown_field": "this should be ignored"
        }"#;

        let result: Result<Vertex<f64, (), 3>, _> = serde_json::from_str(json_with_unknown);
        assert!(result.is_ok());
        let vertex = result.unwrap();
        assert_relative_eq!(
            vertex.point().coords().as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = 1e-9
        );

        // Test deserialization error cases
        let test_cases = vec![
            (
                r#"{"point": [1.0, 2.0, 3.0], "point": [4.0, 5.0, 6.0], "uuid": "550e8400-e29b-41d4-a716-446655440000"}"#,
                "duplicate point",
            ),
            (
                r#"{"point": [1.0, 2.0, 3.0], "uuid": "550e8400-e29b-41d4-a716-446655440000", "uuid": "550e8400-e29b-41d4-a716-446655440001"}"#,
                "duplicate uuid",
            ),
            (
                r#"{"point": [1.0, 2.0, 3.0], "uuid": "550e8400-e29b-41d4-a716-446655440000", "incident_cell": null, "incident_cell": "550e8400-e29b-41d4-a716-446655440001"}"#,
                "duplicate incident_cell",
            ),
            (
                r#"{"point": [1.0, 2.0, 3.0], "uuid": "550e8400-e29b-41d4-a716-446655440000", "data": null, "data": null}"#,
                "duplicate data",
            ),
            (
                r#"{"uuid": "550e8400-e29b-41d4-a716-446655440000"}"#,
                "missing point",
            ),
            (r#"{"point": [1.0, 2.0, 3.0]}"#, "missing uuid"),
        ];

        for (json, description) in test_cases {
            let result: Result<Vertex<f64, (), 3>, _> = serde_json::from_str(json);
            assert!(
                result.is_err(),
                "Expected error for {description}, but got success"
            );
            let error_message = result.unwrap_err().to_string();
            // Check that error message contains relevant keywords
            let has_relevant_error = error_message.contains("duplicate")
                || error_message.contains("missing")
                || error_message.contains("point")
                || error_message.contains("uuid")
                || error_message.contains("data")
                || error_message.contains("incident_cell");
            assert!(
                has_relevant_error,
                "Error message for {description} doesn't contain expected keywords: {error_message}"
            );
        }

        // Test invalid JSON structure
        let invalid_json = r#"["not", "a", "vertex", "object"]"#;
        let result: Result<Vertex<f64, (), 3>, _> = serde_json::from_str(invalid_json);
        assert!(result.is_err());
        let error_message = result.unwrap_err().to_string();
        assert!(
            error_message.contains("Vertex") || error_message.to_lowercase().contains("struct"),
            "Error message should mention Vertex struct: {error_message}"
        );

        // Test validation error for nil UUID
        let vertex_with_nil_uuid = Vertex {
            point: Point::new([1.0, 2.0, 3.0]),
            uuid: uuid::Uuid::nil(),
            incident_cell: None,
            data: None::<()>,
        };

        let validation_result = vertex_with_nil_uuid.is_valid();
        assert!(validation_result.is_err());
        match validation_result.unwrap_err() {
            VertexValidationError::InvalidUuid { source: _ } => (), // Expected
            other @ VertexValidationError::InvalidPoint { .. } => {
                panic!("Expected InvalidUuid error, got: {other:?}")
            }
        }
    }

    // =============================================================================
    // ERROR HANDLING EDGE CASES
    // =============================================================================

    #[test]
    fn test_vertex_validation_error_display() {
        // Test error display formatting
        let point_error =
            crate::geometry::traits::coordinate::CoordinateValidationError::InvalidCoordinate {
                coordinate_index: 1,
                coordinate_value: "NaN".to_string(),
                dimension: 3,
            };
        let vertex_error = VertexValidationError::InvalidPoint {
            source: point_error,
        };
        let error_string = format!("{vertex_error}");
        assert!(error_string.contains("Invalid point"));

        let uuid_error = VertexValidationError::InvalidUuid {
            source: UuidValidationError::NilUuid,
        };
        let uuid_error_string = format!("{uuid_error}");
        assert!(uuid_error_string.contains("Invalid UUID"));
    }

    #[test]
    fn test_vertex_validation_error_equality() {
        // Test PartialEq for VertexValidationError
        let error1 = VertexValidationError::InvalidUuid {
            source: UuidValidationError::NilUuid,
        };
        let error2 = VertexValidationError::InvalidUuid {
            source: UuidValidationError::NilUuid,
        };
        assert_eq!(error1, error2);

        let point_error =
            crate::geometry::traits::coordinate::CoordinateValidationError::InvalidCoordinate {
                coordinate_index: 1,
                coordinate_value: "NaN".to_string(),
                dimension: 3,
            };
        let error3 = VertexValidationError::InvalidPoint {
            source: point_error.clone(),
        };
        let error4 = VertexValidationError::InvalidPoint {
            source: point_error,
        };
        assert_eq!(error3, error4);

        assert_ne!(error1, error3);
    }

    // =============================================================================
    // SET_UUID METHOD TESTS
    // =============================================================================

    #[test]
    fn test_set_uuid_valid() {
        let mut vertex: Vertex<f64, (), 3> = vertex!([1.0, 2.0, 3.0]);
        let original_uuid = vertex.uuid();
        let new_uuid = make_uuid();

        // Test setting a valid UUID
        let result = vertex.set_uuid(new_uuid);
        assert!(result.is_ok());
        assert_eq!(vertex.uuid(), new_uuid);
        assert_ne!(vertex.uuid(), original_uuid);
    }

    #[test]
    fn test_set_uuid_nil_uuid() {
        let mut vertex: Vertex<f64, (), 3> = vertex!([1.0, 2.0, 3.0]);
        let original_uuid = vertex.uuid();

        // Test setting a nil UUID (should fail)
        let result = vertex.set_uuid(uuid::Uuid::nil());
        assert!(result.is_err());

        // Verify the UUID wasn't changed
        assert_eq!(vertex.uuid(), original_uuid);

        // Verify the error type
        match result.unwrap_err() {
            VertexValidationError::InvalidUuid {
                source: UuidValidationError::NilUuid,
            } => (), // Expected
            other => panic!("Expected InvalidUuid with NilUuid source, got: {other:?}"),
        }
    }

    #[test]
    fn test_set_uuid_invalid_version() {
        let mut vertex: Vertex<f64, (), 3> = vertex!([1.0, 2.0, 3.0]);
        let original_uuid = vertex.uuid();

        // Create a UUID with an invalid version (version 0)
        // Note: This is tricky since the uuid crate doesn't easily allow creating invalid UUIDs
        // We'll use the fact that version 3 UUIDs exist but our validation might reject them
        // depending on implementation
        let uuid_bytes = [0u8; 16]; // This creates a nil-like UUID
        let invalid_uuid = uuid::Uuid::from_bytes(uuid_bytes);

        // Test setting an invalid UUID
        let result = vertex.set_uuid(invalid_uuid);
        assert!(result.is_err());

        // Verify the UUID wasn't changed
        assert_eq!(vertex.uuid(), original_uuid);

        // Verify it's a UUID validation error
        match result.unwrap_err() {
            VertexValidationError::InvalidUuid { source: _ } => (), // Expected some UUID error
            other @ VertexValidationError::InvalidPoint { .. } => {
                panic!("Expected InvalidUuid error, got: {other:?}")
            }
        }
    }

    // =============================================================================
    // SERIALIZATION ROUNDTRIP TESTS
    // =============================================================================

    #[test]
    fn test_serialization_deserialization_roundtrip() {
        // Test that serialization -> deserialization preserves all data
        let original_vertex: Vertex<f64, char, 4> = vertex!([1.0, 2.0, 3.0, 4.0], 'A');

        // Serialize
        let serialized = serde_json::to_string(&original_vertex).unwrap();

        // Deserialize
        let deserialized_vertex: Vertex<f64, char, 4> = serde_json::from_str(&serialized).unwrap();

        // Verify all fields match
        assert_relative_eq!(
            original_vertex.point().coords().as_slice(),
            deserialized_vertex.point().coords().as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(original_vertex.uuid(), deserialized_vertex.uuid());
        assert_eq!(
            original_vertex.incident_cell,
            deserialized_vertex.incident_cell
        );
        assert_eq!(original_vertex.data, deserialized_vertex.data);
    }

    #[test]
    fn test_serialization_with_some_data_includes_field() {
        // Test that when data is Some, the JSON includes the data field
        let vertex: Vertex<f64, i32, 3> = vertex!([1.0, 2.0, 3.0], 42);
        let serialized = serde_json::to_string(&vertex).unwrap();

        // Verify JSON contains "data" field
        assert!(
            serialized.contains("\"data\":"),
            "JSON should include data field when Some"
        );
        assert!(serialized.contains("42"), "JSON should include data value");

        // Roundtrip test
        let deserialized: Vertex<f64, i32, 3> = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.data, Some(42));
        assert_relative_eq!(
            vertex.point().coords().as_slice(),
            deserialized.point().coords().as_slice(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn test_serialization_with_none_data_omits_field() {
        // Test that when data is None, the JSON omits the data field entirely
        let vertex: Vertex<f64, (), 3> = vertex!([1.0, 2.0, 3.0]);
        let serialized = serde_json::to_string(&vertex).unwrap();

        // Verify JSON does NOT contain "data" field (optimization)
        assert!(
            !serialized.contains("\"data\":"),
            "JSON should omit data field when None"
        );

        // Roundtrip test - missing data field should deserialize as None
        let deserialized: Vertex<f64, (), 3> = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.data, None);
        assert_relative_eq!(
            vertex.point().coords().as_slice(),
            deserialized.point().coords().as_slice(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn test_deserialization_with_explicit_null_data() {
        // Test backward compatibility: explicit "data": null should still work
        let json_with_null =
            r#"{"point":[1.0,2.0,3.0],"uuid":"550e8400-e29b-41d4-a716-446655440000","data":null}"#;
        let vertex: Vertex<f64, (), 3> = serde_json::from_str(json_with_null).unwrap();

        assert_eq!(vertex.data, None);
        assert_relative_eq!(
            vertex.point().coords().as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = 1e-9
        );
    }
}
