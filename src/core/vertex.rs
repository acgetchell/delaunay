//! Data and operations on d-dimensional vertices.
//!
//! This module provides the `Vertex` struct which represents a geometric vertex
//! in D-dimensional space with associated metadata including unique identification,
//! incident simplex references, and optional user data.
//!
//! # Key Features
//!
//! - **f64 coordinate storage**: vertices store `Point<D>` coordinates, matching
//!   the crate's supported caller-visible coordinate scalar
//! - **Unique Identification**: Each vertex has a UUID for consistent identification
//! - **Optional Data Storage**: [`Vertex`] supports arbitrary user data `U`;
//!   serialization adds [`DataSerialize`] / [`DataDeserialize`] bounds when needed
//! - **Incident Simplex Tracking**: Maintains references to containing simplices
//! - **Serialization Support**: Serde support for persistence (`incident_simplex` is reconstructed by TDS)
//!
//! # Examples
//!
//! ```rust
//! use delaunay::prelude::construction::{Vertex, vertex};
//!
//! // Create a simple vertex
//! let vertex: Vertex<(), 3> = vertex![1.0, 2.0, 3.0]?;
//!
//! // Create vertex with data
//! let vertex_with_data: Vertex<i32, 2> = vertex![1.0, 2.0; data = 42]?;
//! # Ok::<(), delaunay::prelude::geometry::CoordinateConversionError>(())
//! ```

#![forbid(unsafe_code)]

use super::{
    tds::{EntityKind, SimplexKey, TdsConstructionError},
    traits::{DataDeserialize, DataSerialize},
    util::{UuidValidationError, make_uuid, validate_uuid},
};
use crate::geometry::{
    point::Point,
    traits::coordinate::{Coordinate, CoordinateConversionError, CoordinateValidationError},
};
use serde::{
    Deserialize, Serialize,
    de::{self, IgnoredAny, MapAccess, Visitor},
    ser::SerializeStruct,
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
/// use delaunay::prelude::tds::UuidValidationError;
/// use delaunay::prelude::VertexValidationError;
///
/// let err = VertexValidationError::InvalidUuid {
///     source: UuidValidationError::NilUuid,
/// };
/// std::assert_matches!(err, VertexValidationError::InvalidUuid { .. });
/// ```
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
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

/// Aggregate report for standalone vertex validation failures.
///
/// This is the Level 1 element-local report counterpart to
/// [`Vertex::is_valid`] and [`Vertex::vertex_diagnostic`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VertexValidationReport {
    /// The ordered list of vertex invariant violations that occurred.
    pub violations: Vec<VertexValidationError>,
}

impl VertexValidationReport {
    /// Returns `true` if no violations were recorded.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.violations.is_empty()
    }

    /// Returns the recorded vertex invariant violations.
    #[must_use]
    pub fn violations(&self) -> &[VertexValidationError] {
        &self.violations
    }
}

// =============================================================================
// CONVENIENCE MACROS AND HELPERS
// =============================================================================

/// Creates a [`Vertex`] through the existing fallible smart constructors.
///
/// The macro is intentionally thin: it does not unwrap, allocate hidden
/// topology state, or bypass coordinate validation. Callers still handle the
/// same [`CoordinateConversionError`] returned by [`Vertex::try_new`] and
/// [`Vertex::try_new_with_data`].
///
/// When using `; data = ...`, the data expression is stored as the exact
/// vertex payload type inferred for `U`; convert owned payloads before invoking
/// the macro when needed.
///
/// # Errors
///
/// Expands to [`Vertex::try_new`] or [`Vertex::try_new_with_data`], so it
/// returns [`CoordinateConversionError`] when any coordinate cannot be converted
/// to a finite `f64`.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::construction::{Vertex, vertex};
/// use delaunay::prelude::geometry::CoordinateConversionError;
///
/// # fn main() -> Result<(), CoordinateConversionError> {
/// let vertex: Vertex<(), 2> = vertex![0.0, 1.0]?;
/// let bracketed: Vertex<(), 2> = vertex!([1.0, 0.0])?;
///
/// assert_eq!(vertex.point().coords(), &[0.0, 1.0]);
/// assert_eq!(bracketed.point().coords(), &[1.0, 0.0]);
/// # Ok(())
/// # }
/// ```
///
/// ```rust
/// use delaunay::prelude::construction::{Vertex, vertex};
/// use delaunay::prelude::geometry::CoordinateConversionError;
///
/// # fn main() -> Result<(), CoordinateConversionError> {
/// let vertex: Vertex<&str, 2> = vertex![0.0, 1.0; data = "boundary"]?;
///
/// assert_eq!(vertex.data(), Some(&"boundary"));
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! vertex {
    ($coords:expr; data = $data:expr $(,)?) => {
        $crate::tds::Vertex::<_, _>::try_new_with_data($coords, $data)
    };
    ($($coord:expr),+; data = $data:expr $(,)?) => {
        $crate::tds::Vertex::<_, _>::try_new_with_data([$($coord),+], $data)
    };
    ($coords:expr $(,)?) => {
        $crate::tds::Vertex::<(), _>::try_new($coords)
    };
    ($($coord:expr),+ $(,)?) => {
        $crate::tds::Vertex::<(), _>::try_new([$($coord),+])
    };
}

// =============================================================================
// VERTEX STRUCT DEFINITION
// =============================================================================

#[derive(Clone, Copy, Debug)]
/// The `Vertex` struct represents a vertex in a triangulation with geometric
/// coordinates, unique identification, and optional metadata.
///
/// # Generic Parameters
///
/// * `U` - User data type stored with the vertex (use `()` for no data)
/// * `D` - The spatial dimension (compile-time constant)
///
/// # Properties
///
/// - **`point`**: A `Point<D>` representing the geometric coordinates of the vertex
/// - **`uuid`**: A universally unique identifier for the vertex (auto-generated)
/// - **`incident_simplex`**: Optional reference to a containing simplex (managed by TDS)
/// - **`data`**: Optional user-defined data associated with the vertex. Read via [`data()`](Self::data),
///   mutate via [`Tds::set_vertex_data`](crate::prelude::tds::Tds::set_vertex_data)
///
/// # Constraints
///
/// - `U` has no bound for standalone [`Vertex`] construction or access.
///   Serialization uses [`DataSerialize`] / [`DataDeserialize`]; TDS and
///   triangulation algorithms add [`DataType`](crate::prelude::DataType)
///   only where they need copy/debug/serde metadata behavior.
///
/// # Usage
///
/// Vertices are typically created from raw coordinates with [`vertex!`],
/// [`Vertex::try_new`], or [`Vertex::try_new_with_data`].
///
/// ```rust
/// use delaunay::prelude::Vertex;
///
/// # fn main() -> Result<(), delaunay::prelude::geometry::CoordinateConversionError> {
/// let vertex = Vertex::<i32, 3>::try_new_with_data([1.0, 2.0, 3.0], 42)?;
/// assert_eq!(vertex.data(), Some(&42));
/// # Ok(())
/// # }
/// ```
pub struct Vertex<U, const D: usize> {
    /// The coordinates of the vertex as a D-dimensional Point.
    point: Point<D>,
    /// A universally unique identifier for the vertex.
    uuid: Uuid,
    /// The `SimplexKey` of the simplex that the vertex is incident to.
    ///
    /// Note: This field is not serialized because `SimplexKey` is only valid within
    /// the current `SlotMap` instance. During deserialization, the TDS automatically
    /// reconstructs `incident_simplex` mappings via `assign_incident_simplices()`.
    pub(crate) incident_simplex: Option<SimplexKey>,
    /// Optional data associated with the vertex.
    pub(crate) data: Option<U>,
}

impl<U, const D: usize> Vertex<U, D> {
    /// Returns the UUID of the vertex.
    #[inline]
    pub const fn uuid(&self) -> Uuid {
        self.uuid
    }

    /// Returns the spatial dimension of the vertex.
    #[inline]
    pub const fn dim(&self) -> usize {
        D
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
    /// use delaunay::prelude::Vertex;
    /// use delaunay::prelude::geometry::Coordinate;
    ///
    /// # fn main() -> Result<(), delaunay::prelude::geometry::CoordinateConversionError> {
    /// let vertex: Vertex<(), 3> = delaunay::prelude::Vertex::<(), _>::try_new([1.0, 2.0, 3.0])?;
    /// let retrieved_point = vertex.point();
    /// assert_eq!(retrieved_point.coords(), &[1.0, 2.0, 3.0]);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub const fn point(&self) -> &Point<D> {
        &self.point
    }

    /// Returns the TDS-managed incident simplex pointer for this vertex.
    ///
    /// The pointer is maintained by topology mutation and repair operations,
    /// so callers can inspect it but cannot assign arbitrary simplex keys.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Coordinates(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0])?,
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0])?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let Some((_, vertex)) = dt.vertices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// assert!(vertex.incident_simplex().is_some());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub const fn incident_simplex(&self) -> Option<SimplexKey> {
        self.incident_simplex
    }

    /// Returns a reference to the optional user data associated with this vertex.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::Vertex;
    ///
    /// # fn main() -> Result<(), delaunay::prelude::geometry::CoordinateConversionError> {
    /// let v: Vertex<i32, 2> = delaunay::prelude::Vertex::<_, _>::try_new_with_data([1.0, 2.0], 42)?;
    /// assert_eq!(v.data(), Some(&42));
    ///
    /// let v_no_data: Vertex<(), 2> = delaunay::prelude::Vertex::<(), _>::try_new([1.0, 2.0])?;
    /// assert_eq!(v_no_data.data(), None);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub const fn data(&self) -> Option<&U> {
        self.data.as_ref()
    }

    /// Updates the TDS-managed incident simplex pointer.
    #[inline]
    pub(crate) const fn set_incident_simplex(&mut self, incident_simplex: Option<SimplexKey>) {
        self.incident_simplex = incident_simplex;
    }
}

// =============================================================================
// SERIALIZATION IMPLEMENTATION
// =============================================================================

/// Manual implementation of Serialize for Vertex.
///
/// This implementation handles serialization of all vertex fields. The `incident_simplex`
/// field is skipped as it's a runtime-only reference that gets reconstructed during
/// deserialization.
impl<U, const D: usize> Serialize for Vertex<U, D>
where
    U: DataSerialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
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
impl<'de, U, const D: usize> Deserialize<'de> for Vertex<U, D>
where
    U: DataDeserialize,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: serde::Deserializer<'de>,
    {
        struct VertexVisitor<U, const D: usize>
        where
            U: DataDeserialize,
        {
            _phantom: PhantomData<U>,
        }

        impl<'de, U, const D: usize> Visitor<'de> for VertexVisitor<U, D>
        where
            U: DataDeserialize,
        {
            type Value = Vertex<U, D>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a Vertex struct")
            }

            fn visit_map<V>(self, mut map: V) -> Result<Vertex<U, D>, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut point: Option<Point<D>> = None;
                let mut uuid = None;
                let mut data = None;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
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
                        "incident_simplex" => {
                            return Err(de::Error::custom(
                                "incident_simplex is a storage-local slotmap key and must not be deserialized; deserialize Tds so incident mappings can be reconstructed",
                            ));
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
                let data = data.unwrap_or(None);

                Ok(Vertex {
                    point,
                    uuid,
                    incident_simplex: None,
                    data,
                })
            }
        }

        const FIELDS: &[&str] = &["point", "uuid", "data"];
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

impl<U, const D: usize> Vertex<U, D> {
    /// Creates a vertex from already-validated point coordinates.
    #[inline]
    #[must_use]
    pub(crate) fn from_validated_point(point: Point<D>, data: Option<U>) -> Self {
        Self {
            point,
            uuid: make_uuid(),
            incident_simplex: None,
            data,
        }
    }

    /// Tries to create a vertex from raw coordinate values with a fresh UUID.
    ///
    /// # Errors
    ///
    /// Returns [`CoordinateConversionError`] when any coordinate cannot be
    /// converted exactly to finite `f64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::Vertex;
    /// use delaunay::prelude::geometry::CoordinateConversionError;
    ///
    /// let vertex = Vertex::<(), 2>::try_new([0.0, 1.0])?;
    /// assert_eq!(vertex.point().coords(), &[0.0, 1.0]);
    /// # Ok::<(), CoordinateConversionError>(())
    /// ```
    ///
    /// ```rust
    /// use delaunay::prelude::construction::Vertex;
    /// use delaunay::prelude::geometry::CoordinateConversionError;
    ///
    /// std::assert_matches!(
    ///     Vertex::<(), 2>::try_new([f64::INFINITY, 1.0]),
    ///     Err(CoordinateConversionError::NonFiniteValue { coordinate_index: 0, .. })
    /// );
    /// ```
    #[inline]
    pub fn try_new<T>(coords: [T; D]) -> Result<Self, CoordinateConversionError>
    where
        T: num_traits::cast::NumCast + Copy + fmt::Debug + PartialEq,
    {
        Point::try_from(coords).map(|point| Self::from_validated_point(point, None))
    }

    /// Tries to create a vertex with user data from raw coordinate values.
    ///
    /// The `data` argument is stored as the exact vertex payload type `U`.
    /// Convert owned payloads before calling when needed; coordinate parsing is
    /// the only fallible conversion performed by this constructor.
    ///
    /// # Errors
    ///
    /// Returns [`CoordinateConversionError`] when any coordinate cannot be
    /// converted exactly to finite `f64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::Vertex;
    /// use delaunay::prelude::geometry::CoordinateConversionError;
    ///
    /// let vertex = Vertex::<_, 2>::try_new_with_data([0.0, 1.0], "boundary")?;
    /// assert_eq!(vertex.data(), Some(&"boundary"));
    ///
    /// let owned: Vertex<String, 2> =
    ///     Vertex::try_new_with_data([1.0, 0.0], String::from("owned-label"))?;
    /// assert_eq!(owned.data().map(String::as_str), Some("owned-label"));
    /// # Ok::<(), CoordinateConversionError>(())
    /// ```
    ///
    /// ```rust
    /// use delaunay::prelude::construction::Vertex;
    /// use delaunay::prelude::geometry::CoordinateConversionError;
    ///
    /// std::assert_matches!(
    ///     Vertex::<&str, 2>::try_new_with_data([f64::NAN, 1.0], "bad"),
    ///     Err(CoordinateConversionError::NonFiniteValue { coordinate_index: 0, .. })
    /// );
    /// ```
    #[inline]
    pub fn try_new_with_data<T>(coords: [T; D], data: U) -> Result<Self, CoordinateConversionError>
    where
        T: num_traits::cast::NumCast + Copy + fmt::Debug + PartialEq,
    {
        Point::try_from(coords).map(|point| Self::from_validated_point(point, Some(data)))
    }

    /// Converts vertices into a [`HashMap`] keyed by stable vertex [`Uuid`].
    ///
    /// # Arguments
    ///
    /// * `vertices`: Vertices to be converted into a `HashMap`.
    ///
    /// # Returns
    ///
    /// Returns a [`HashMap`] whose keys are the vertices' stable UUIDs.
    ///
    /// # Errors
    ///
    /// Returns [`TdsConstructionError::DuplicateUuid`] if the input contains
    /// duplicate vertex UUIDs.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use delaunay::prelude::Vertex;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// #     #[error(transparent)]
    /// #     Tds(#[from] delaunay::prelude::tds::TdsConstructionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let v1: Vertex<(), 2> = delaunay::prelude::Vertex::<(), _>::try_new([1.0, 2.0])?;
    /// let v2: Vertex<(), 2> = delaunay::prelude::Vertex::<(), _>::try_new([3.0, 4.0])?;
    ///
    /// let map: HashMap<_, _> = Vertex::try_into_hashmap([v1, v2])?;
    /// assert_eq!(map.len(), 2);
    /// assert!(map.values().all(|v| v.dim() == 2));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn try_into_hashmap<I>(vertices: I) -> Result<HashMap<Uuid, Self>, TdsConstructionError>
    where
        I: IntoIterator<Item = Self>,
    {
        let iter = vertices.into_iter();
        let mut map = HashMap::with_capacity(iter.size_hint().0);
        for vertex in iter {
            let uuid = vertex.uuid();
            if map.insert(uuid, vertex).is_some() {
                return Err(TdsConstructionError::DuplicateUuid {
                    entity: EntityKind::Vertex,
                    uuid,
                });
            }
        }
        Ok(map)
    }

    /// Checks whether this vertex satisfies its standalone invariants.
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
    /// use delaunay::prelude::{Point, Vertex, VertexValidationError};
    /// use uuid::Uuid;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] VertexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertex: Vertex<(), 3> = Vertex::<(), _>::try_new([1.0, 2.0, 3.0])?;
    /// assert!(vertex.is_valid().is_ok());
    ///
    /// match Vertex::<(), 3>::try_new_with_uuid(Point::default(), Uuid::nil(), None) {
    ///     Err(VertexValidationError::InvalidUuid { .. }) => (), // Expected - nil UUID
    ///     other => panic!("Expected InvalidUuid error, got: {:?}", other),
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn is_valid(&self) -> Result<(), VertexValidationError> {
        // Check if the point is valid using the Coordinate trait validation
        self.point
            .validate()
            .map_err(|source| VertexValidationError::InvalidPoint { source })?;

        // Check if UUID is valid using centralized validation
        validate_uuid(&self.uuid())?;

        Ok(())
        // Note: incident_simplex validation is handled at the TDS level via:
        // - Tds::assign_incident_simplices() ensures proper simplex assignment
        // - Tds::is_valid() validates simplex mappings and references
        // Individual vertices cannot validate incident_simplex without TDS context.
        // User data validation (if U: DataType requires it) could be added here.
    }

    /// Returns the first standalone vertex validation diagnostic, if any.
    #[must_use]
    pub fn vertex_diagnostic(&self) -> Option<VertexValidationError> {
        self.is_valid().err()
    }

    /// Runs standalone vertex validation and returns all checkable failures.
    ///
    /// Unlike [`is_valid`](Self::is_valid), this method does not
    /// stop after the first invalid field.
    ///
    /// # Errors
    ///
    /// Returns a [`VertexValidationReport`] containing all checkable vertex
    /// violations.
    pub fn vertex_report(&self) -> Result<(), VertexValidationReport> {
        let mut violations = Vec::new();

        if let Err(source) = self.point.validate() {
            violations.push(VertexValidationError::InvalidPoint { source });
        }

        if let Err(source) = validate_uuid(&self.uuid()) {
            violations.push(VertexValidationError::InvalidUuid { source });
        }

        if violations.is_empty() {
            Ok(())
        } else {
            Err(VertexValidationReport { violations })
        }
    }

    /// Creates a vertex with a caller-provided UUID after validating it.
    ///
    /// This constructor is intended for serialization round-trips and other
    /// boundary code that must preserve an existing stable vertex identity.
    /// Ordinary construction should prefer [`Self::try_new`] or
    /// [`Self::try_new_with_data`], which generate a fresh UUID automatically.
    ///
    /// # Errors
    ///
    /// Returns [`VertexValidationError::InvalidUuid`] if `uuid` is nil or
    /// otherwise fails UUID validation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use delaunay::prelude::{Point, Vertex};
    /// use uuid::Uuid;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] delaunay::prelude::VertexValidationError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let point = Point::default();
    /// let uuid = Uuid::from_u128(0x67e5504410b1426f9247bb680e5fe0c8);
    /// let vertex = Vertex::<(), 3>::try_new_with_uuid(point, uuid, None)?;
    ///
    /// assert_eq!(vertex.uuid(), uuid);
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_new_with_uuid(
        point: Point<D>,
        uuid: Uuid,
        data: Option<U>,
    ) -> Result<Self, VertexValidationError> {
        validate_uuid(&uuid)?;

        Ok(Self::from_validated_point_with_uuid(point, uuid, data))
    }

    /// Creates a vertex with a UUID already known to be valid.
    ///
    /// This crate-internal constructor preserves stable vertex identity while
    /// transforming already-valid vertices or rebuilding topology that has
    /// validated UUIDs at the boundary. Public callers with raw UUIDs must use
    /// [`Self::try_new_with_uuid`] so the UUID proof is established before the
    /// value is stored.
    ///
    /// # Arguments
    ///
    /// * `point` - The point coordinates for the vertex
    /// * `uuid` - The UUID to assign to the vertex (must be unique!)
    /// * `data` - Optional user data for the vertex
    ///
    /// # Returns
    ///
    /// A new `Vertex` with the specified UUID and data.
    pub(crate) const fn from_validated_point_with_uuid(
        point: Point<D>,
        uuid: Uuid,
        data: Option<U>,
    ) -> Self {
        Self {
            point,
            uuid,
            incident_simplex: None,
            data,
        }
    }
}

// =============================================================================
// STANDARD TRAIT IMPLEMENTATIONS
// =============================================================================
impl<U, const D: usize> PartialEq for Vertex<U, D> {
    /// Equality of vertices is based on ordered equality of coordinates using the Coordinate trait.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.point.ordered_equals(&other.point)
        // && self.uuid == other.uuid
        // && self.incident_simplex == other.incident_simplex
        // && self.data == other.data
    }
}

impl<U, const D: usize> PartialOrd for Vertex<U, D> {
    /// Order of vertices is based on lexicographic order of their validated finite coordinates.
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.point.partial_cmp(&other.point)
    }
}

/// Enable implicit conversion from Vertex to coordinate array
/// This allows using `Into` to convert from `Vertex` to `[f64; D]`
impl<U, const D: usize> From<Vertex<U, D>> for [f64; D] {
    #[inline]
    fn from(vertex: Vertex<U, D>) -> [f64; D] {
        vertex.point.into()
    }
}

/// Enable implicit conversion from Vertex reference to coordinate array
/// This allows `&vertex` to be implicitly converted to `[f64; D]` for coordinate access
impl<U, const D: usize> From<&Vertex<U, D>> for [f64; D] {
    #[inline]
    fn from(vertex: &Vertex<U, D>) -> [f64; D] {
        vertex.point().into()
    }
}

/// Enable implicit conversion from Vertex reference to Point
/// This allows `&vertex` to be implicitly converted to `Point<D>`
impl<U, const D: usize> From<&Vertex<U, D>> for Point<D> {
    #[inline]
    fn from(vertex: &Vertex<U, D>) -> Self {
        *vertex.point()
    }
}

// =============================================================================
// HASHING AND EQUALITY IMPLEMENTATIONS
// =============================================================================
impl<U, const D: usize> Eq for Vertex<U, D> {
    // Generic Eq implementation for Vertex based on point equality
}

impl<U, const D: usize> Hash for Vertex<U, D> {
    /// Hash implementation for Vertex using only coordinates for consistency with `PartialEq`.
    ///
    /// This ensures that vertices with the same coordinates have the same hash,
    /// maintaining the Eq/Hash contract: if a == b, then hash(a) == hash(b).
    ///
    /// Note: UUID, `incident_simplex`, and data are excluded from hashing to match
    /// the `PartialEq` implementation which only compares coordinates.
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.point.hash_coordinate(state);
        // Intentionally exclude UUID, incident_simplex, and data to maintain
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
    use crate::core::tds::SimplexKey;
    use crate::core::traits::DataType;
    use crate::core::util::{UuidValidationError, make_uuid, usize_to_u8};
    use crate::core::vertex::Vertex;
    use crate::geometry::point::Point;
    use crate::geometry::traits::coordinate::{
        Coordinate, CoordinateValidationError, InvalidCoordinateValue,
    };
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use serde::{Deserialize, Serialize};
    use slotmap::KeyData;
    use std::assert_matches;
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
    fn assert_vertex_properties<U, const D: usize>(vertex: &Vertex<U, D>, expected_coords: [f64; D])
    where
        U: DataType,
    {
        assert_abs_diff_eq!(
            vertex.point().coords().as_slice(),
            expected_coords.as_slice()
        );
        assert_eq!(vertex.dim(), D);
        assert!(!vertex.uuid().is_nil());
        assert!(vertex.incident_simplex.is_none());
    }

    // =============================================================================
    // VERTEX CONSTRUCTOR TESTS
    // =============================================================================

    #[test]
    fn test_vertex_try_new() {
        let v: Vertex<(), 3> = Vertex::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");
        assert_relative_eq!(
            v.point().coords().as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = 1e-9
        );
        assert!(!v.uuid().is_nil());
        assert!(v.incident_simplex.is_none());
        assert!(v.data.is_none());
    }

    #[test]
    fn test_vertex_try_new_with_data() {
        let v: Vertex<i32, 2> =
            Vertex::try_new_with_data([0.0, 1.0], 42).expect("finite point coordinates");
        assert_relative_eq!(
            v.point().coords().as_slice(),
            [0.0, 1.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(v.data, Some(42));
    }

    #[test]
    fn test_vertex_try_new_with_data_stores_exact_metadata_type() {
        let v: Vertex<String, 2> =
            Vertex::try_new_with_data([0.0, 1.0], String::from("standalone-label"))
                .expect("finite point coordinates");

        assert_eq!(v.data().map(String::as_str), Some("standalone-label"));
    }

    #[test]
    fn test_vertex_try_new_rejects_invalid_coordinates() {
        let result = Vertex::<(), 3>::try_new([1.0, f64::NAN, 3.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_vertex_data_accessor() {
        let v_with: Vertex<i32, 2> = Vertex::<_, _>::try_new_with_data([1.0, 2.0], 42).unwrap();
        assert_eq!(v_with.data(), Some(&42));

        let v_without: Vertex<(), 2> = Vertex::<(), _>::try_new([1.0, 2.0]).unwrap();
        assert_eq!(v_without.data(), None);
    }

    // =============================================================================
    // SMART CONSTRUCTOR TESTS
    // =============================================================================

    #[test]
    fn test_vertex_try_new_constructor_variants() {
        let v1: Vertex<(), 3> = Vertex::<(), _>::try_new([1.0, 2.0, 3.0]).unwrap();
        assert_relative_eq!(
            v1.point().coords().as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(v1.dim(), 3);
        assert!(!v1.uuid().is_nil());
        assert!(v1.data.is_none());

        let v2: Vertex<i32, 2> = Vertex::<_, _>::try_new_with_data([0.0, 1.0], 99).unwrap();
        assert_relative_eq!(
            v2.point().coords().as_slice(),
            [0.0, 1.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(v2.dim(), 2);
        assert!(!v2.uuid().is_nil());
        assert_eq!(v2.data.unwrap(), 99);

        let v3: Vertex<u32, 4> =
            Vertex::<_, _>::try_new_with_data([1.0, 2.0, 3.0, 4.0], 42u32).unwrap();
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
    }

    // =============================================================================
    // BASIC VERTEX FUNCTIONALITY
    // =============================================================================

    #[test]
    fn test_vertex_basic_operations() {
        // Test vertex copying
        let vertex: Vertex<u8, 4> =
            Vertex::<_, _>::try_new_with_data([1.0, 2.0, 3.0, 4.0], 4u8).unwrap();
        let vertex_copy = vertex;
        assert_eq!(vertex, vertex_copy);
        assert_relative_eq!(
            vertex_copy.point().coords().as_slice(),
            [1.0, 2.0, 3.0, 4.0].as_slice(),
            epsilon = 1e-9
        );

        // Test batch construction from already-validated test points.
        let points = [
            Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates"),
            Point::try_new([4.0, 5.0, 6.0]).expect("finite point coordinates"),
            Point::try_new([7.0, 8.0, 9.0]).expect("finite point coordinates"),
        ];
        let mut vertices: Vec<Vertex<(), 3>> = points
            .iter()
            .copied()
            .map(|point| Vertex::from_validated_point(point, None))
            .collect();

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

        // Test Vertex::try_into_hashmap() with multiple vertices
        let hashmap = Vertex::try_into_hashmap(vertices.iter().copied()).unwrap();
        let mut values: Vec<Vertex<(), 3>> = hashmap.into_values().collect();

        assert_eq!(values.len(), 3);

        values.sort_by_key(Vertex::uuid);
        vertices.sort_by_key(Vertex::uuid);

        assert_eq!(values, vertices);

        // Test Vertex::try_into_hashmap() with empty input
        let empty_hashmap = Vertex::<(), 3>::try_into_hashmap([]).unwrap();
        assert!(empty_hashmap.is_empty());

        // Test Vertex::try_into_hashmap() with single vertex
        let single_vertex: Vertex<(), 3> = Vertex::<(), _>::try_new([1.0, 2.0, 3.0]).unwrap();
        let uuid = single_vertex.uuid();
        let single_hashmap = Vertex::try_into_hashmap([single_vertex]).unwrap();

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
    }

    #[test]
    fn test_try_into_hashmap_rejects_duplicate_uuid() {
        let uuid = make_uuid();
        let first: Vertex<(), 2> = Vertex::from_validated_point_with_uuid(
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            uuid,
            None,
        );
        let second: Vertex<(), 2> = Vertex::from_validated_point_with_uuid(
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            uuid,
            None,
        );

        assert_matches!(
            Vertex::try_into_hashmap([first, second]),
            Err(TdsConstructionError::DuplicateUuid {
                entity: EntityKind::Vertex,
                uuid: duplicate_uuid,
            }) if duplicate_uuid == uuid
        );
    }

    #[test]
    fn test_try_new_with_uuid_rejects_nil_uuid() {
        let point = Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");

        let result = Vertex::<(), 3>::try_new_with_uuid(point, uuid::Uuid::nil(), None);

        assert_eq!(
            result.unwrap_err(),
            VertexValidationError::InvalidUuid {
                source: UuidValidationError::NilUuid,
            }
        );
    }

    #[test]
    fn test_try_new_with_uuid_preserves_valid_uuid() {
        let point = Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");
        let uuid = make_uuid();

        let vertex = Vertex::<u8, 3>::try_new_with_uuid(point, uuid, Some(7)).unwrap();

        assert_eq!(vertex.uuid(), uuid);
        assert_eq!(vertex.point(), &point);
        assert_eq!(vertex.data(), Some(&7));
    }

    #[test]
    fn test_vertex_serialization_roundtrip() {
        // Test basic serialization/deserialization roundtrip
        let vertex: Vertex<(), 3> = Vertex::<(), _>::try_new([1.0, 2.0, 3.0]).unwrap();
        let serialized = serde_json::to_string(&vertex).unwrap();

        assert!(serialized.contains("point"));
        assert!(serialized.contains("[1.0,2.0,3.0]"));

        let deserialized: Vertex<(), 3> = serde_json::from_str(&serialized).unwrap();

        // Check that deserialized vertex has same point coordinates using approx equality
        assert_relative_eq!(
            deserialized.point().coords().as_slice(),
            vertex.point().coords().as_slice(),
            epsilon = f64::EPSILON
        );
        assert_eq!(deserialized.dim(), vertex.dim());
        assert_eq!(deserialized.incident_simplex, vertex.incident_simplex);
        assert_eq!(deserialized.data, vertex.data);
        assert_eq!(deserialized.uuid(), vertex.uuid());

        // Test serialization with data
        let vertex_with_data: Vertex<i32, 3> =
            Vertex::<_, _>::try_new_with_data([1.0, 2.0, 3.0], 42).unwrap();
        let serialized_with_data = serde_json::to_string(&vertex_with_data).unwrap();
        assert!(serialized_with_data.contains("\"data\":"));
        assert!(serialized_with_data.contains("42"));

        let deserialized_with_data: Vertex<i32, 3> =
            serde_json::from_str(&serialized_with_data).unwrap();
        assert_eq!(deserialized_with_data.data, Some(42));
        assert_relative_eq!(
            deserialized_with_data.point().coords().as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = f64::EPSILON
        );

        // Test serialization with None data (should omit data field)
        let vertex_no_data: Vertex<(), 3> = Vertex::<(), _>::try_new([1.0, 2.0, 3.0]).unwrap();
        let serialized_no_data = serde_json::to_string(&vertex_no_data).unwrap();
        assert!(!serialized_no_data.contains("\"data\":"));

        let deserialized_no_data: Vertex<(), 3> =
            serde_json::from_str(&serialized_no_data).unwrap();
        assert_eq!(deserialized_no_data.data, None);

        let json_with_incident_simplex = r#"{"point":[1.0,2.0,3.0],"uuid":"550e8400-e29b-41d4-a716-446655440000","incident_simplex":{"idx":1,"version":1}}"#;
        let incident_simplex_error =
            serde_json::from_str::<Vertex<(), 3>>(json_with_incident_simplex).unwrap_err();
        assert!(
            incident_simplex_error
                .to_string()
                .contains("incident_simplex is a storage-local slotmap key")
        );

        // Test backward compatibility: explicit "data": null should still work
        let json_with_null =
            r#"{"point":[1.0,2.0,3.0],"uuid":"550e8400-e29b-41d4-a716-446655440000","data":null}"#;
        let vertex_null_data: Vertex<(), 3> = serde_json::from_str(json_with_null).unwrap();
        assert_eq!(vertex_null_data.data, None);

        // Test with different data types
        let vertex_char: Vertex<char, 4> =
            Vertex::<_, _>::try_new_with_data([1.0, 2.0, 3.0, 4.0], 'A').unwrap();
        let serialized_char = serde_json::to_string(&vertex_char).unwrap();
        let deserialized_char: Vertex<char, 4> = serde_json::from_str(&serialized_char).unwrap();
        assert_eq!(deserialized_char.data, Some('A'));
        assert_relative_eq!(
            deserialized_char.point().coords().as_slice(),
            [1.0, 2.0, 3.0, 4.0].as_slice(),
            epsilon = f64::EPSILON
        );
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
        let v1: Vertex<(), 3> = Vertex::<(), _>::try_new([1.0, 2.0, 3.0]).unwrap();
        let v2: Vertex<(), 3> = Vertex::<(), _>::try_new([1.0, 2.0, 3.0]).unwrap();
        let v3: Vertex<(), 3> = Vertex::<(), _>::try_new([1.0, 2.0, 4.0]).unwrap();

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

        // Test that equality ignores UUID, incident_simplex, and data
        let v4: Vertex<i32, 2> = Vertex::<_, _>::try_new_with_data([1.0, 2.0], 42).unwrap();
        let v5: Vertex<i32, 2> = Vertex::<_, _>::try_new_with_data([1.0, 2.0], 99).unwrap(); // Different data

        // Different UUIDs and data but same coordinates
        assert_ne!(v4.uuid(), v5.uuid());
        assert_ne!(v4.data, v5.data);

        // Should still be equal because coordinates match
        assert_eq!(v4, v5);

        // Test with None data
        let v6: Vertex<(), 2> = Vertex::<(), _>::try_new([1.0, 2.0]).unwrap();
        let v7: Vertex<(), 2> = Vertex::<(), _>::try_new([1.0, 2.0]).unwrap();
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

        // Test that hash ignores UUID, incident_simplex, and data (consistent with equality)
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
            let v_a: Vertex<(), 2> = Vertex::<(), _>::try_new(coords1).unwrap();
            let v_b: Vertex<(), 2> = Vertex::<(), _>::try_new(coords2).unwrap();

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
    fn test_vertex_signed_zero_eq_hash_and_order_are_consistent() {
        let positive_zero: Vertex<(), 2> = Vertex::<(), _>::try_new([0.0, -0.0]).unwrap();
        let negative_zero: Vertex<(), 2> = Vertex::<(), _>::try_new([-0.0, 0.0]).unwrap();

        assert_eq!(positive_zero, negative_zero);
        assert_eq!(
            positive_zero.partial_cmp(&negative_zero),
            Some(Ordering::Equal)
        );

        let mut positive_hash = DefaultHasher::new();
        let mut negative_hash = DefaultHasher::new();
        positive_zero.hash(&mut positive_hash);
        negative_zero.hash(&mut negative_hash);
        assert_eq!(positive_hash.finish(), negative_hash.finish());
    }

    #[test]
    fn test_vertex_collections() {
        // Test vertices in collections to verify Hash/Eq contract in practice
        let mut set: FastHashSet<Vertex<(), 2>> = FastHashSet::default();

        let v1: Vertex<(), 2> = Vertex::<(), _>::try_new([1.0, 2.0]).unwrap();
        let v2: Vertex<(), 2> = Vertex::<(), _>::try_new([3.0, 4.0]).unwrap();
        let v3: Vertex<(), 2> = Vertex::<(), _>::try_new([1.0, 2.0]).unwrap(); // Same coordinates as v1

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
        let v4: Vertex<(), 2> = Vertex::<(), _>::try_new([1.0, 2.0]).unwrap();
        assert!(set.contains(&v4)); // Should find it even with different UUID

        // Test vertices as FastHashMap keys
        let mut map: FastHashMap<Vertex<(), 2>, i32> = FastHashMap::default();

        let v5: Vertex<(), 2> = Vertex::<(), _>::try_new([1.0, 2.0]).unwrap();
        let v6: Vertex<(), 2> = Vertex::<(), _>::try_new([3.0, 4.0]).unwrap();

        map.insert(v5, 10);
        map.insert(v6, 20);

        // Verify lookups work
        assert_eq!(map.get(&v5), Some(&10));
        assert_eq!(map.get(&v6), Some(&20));
        assert_eq!(map.len(), 2);

        // Test lookup with equivalent vertex (same coordinates, different UUID)
        let v7: Vertex<(), 2> = Vertex::<(), _>::try_new([1.0, 2.0]).unwrap();
        assert_eq!(map.get(&v7), Some(&10)); // Should find v5's value

        // Test overwrite with equivalent vertex
        let old_value = map.insert(v7, 30);
        assert_eq!(old_value, Some(10)); // Should return v5's old value
        assert_eq!(map.len(), 2); // Size shouldn't change
        assert_eq!(map.get(&v5), Some(&30)); // v5 now maps to new value

        // Test that vertices with different data types but same coordinates work in collections
        let v8: Vertex<u16, 2> = Vertex::<_, _>::try_new_with_data([1.0, 2.0], 999u16).unwrap();
        let v9: Vertex<i32, 2> = Vertex::<_, _>::try_new_with_data([3.0, 4.0], -42i32).unwrap();

        let mut map1: FastHashMap<Vertex<u16, 2>, &str> = FastHashMap::default();
        map1.insert(v8, "first");
        assert_eq!(map1.len(), 1);

        let mut map2: FastHashMap<Vertex<i32, 2>, bool> = FastHashMap::default();
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
                        let vertex: Vertex<(), $dim> = Vertex::<(), _>::try_new([$($coord),+]).unwrap();
                    assert_vertex_properties(&vertex, [$($coord),+]);
                        assert!(vertex.data.is_none());
                    }

                pastey::paste! {
                    #[test]
                    fn [<$test_name _with_data>]() {
                        // Test vertex with data
                        let vertex: Vertex<i32, $dim> = Vertex::<_, _>::try_new_with_data([$($coord),+], 42).unwrap();
                        assert_vertex_properties(&vertex, [$($coord),+]);
                        assert_eq!(vertex.data, Some(42));
                    }

                    #[test]
                    fn [<$test_name _serialization_roundtrip>]() {
                        // Test serialization with Some data
                        let vertex_with_data: Vertex<i32, $dim> = Vertex::<_, _>::try_new_with_data([$($coord),+], 99).unwrap();
                        let serialized = serde_json::to_string(&vertex_with_data).unwrap();
                        assert!(serialized.contains("\"data\":"));
                        let deserialized: Vertex<i32, $dim> = serde_json::from_str(&serialized).unwrap();
                        assert_eq!(deserialized.data, Some(99));
                        assert_vertex_properties(&deserialized, [$($coord),+]);

                        // Test serialization with None data
                        let vertex_no_data: Vertex<(), $dim> = Vertex::<(), _>::try_new([$($coord),+]).unwrap();
                        let serialized = serde_json::to_string(&vertex_no_data).unwrap();
                        assert!(!serialized.contains("\"data\":"));
                        let deserialized: Vertex<(), $dim> = serde_json::from_str(&serialized).unwrap();
                        assert_eq!(deserialized.data, None);
                    }

                    #[test]
                    fn [<$test_name _uuid_uniqueness>]() {
                        // Test UUID uniqueness for same coordinates
                        let v1: Vertex<(), $dim> = Vertex::<(), _>::try_new([$($coord),+]).unwrap();
                        let v2: Vertex<(), $dim> = Vertex::<(), _>::try_new([$($coord),+]).unwrap();
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
        let vertex: Vertex<(), 1> = Vertex::<(), _>::try_new([42.0]).unwrap();
        assert_vertex_properties(&vertex, [42.0]);
        assert!(vertex.data.is_none());
    }

    // =============================================================================
    // DATA TYPE TESTS
    // =============================================================================

    #[test]
    fn test_vertex_data_types_and_ordering() {
        // Test vertex with tuple data
        let vertex_tuple: Vertex<(i32, i32), 2> =
            Vertex::<_, _>::try_new_with_data([1.0, 2.0], (42, 84)).unwrap();
        assert_vertex_properties(&vertex_tuple, [1.0, 2.0]);
        assert_eq!(vertex_tuple.data.unwrap(), (42, 84));

        // Test debug formatting
        let vertex_debug: Vertex<i32, 3> =
            Vertex::<_, _>::try_new_with_data([1.0, 2.0, 3.0], 42).unwrap();
        let debug_str = format!("{vertex_debug:?}");

        assert!(debug_str.contains("Vertex"));
        assert!(debug_str.contains("point"));
        assert!(debug_str.contains("uuid"));
        assert!(debug_str.contains("1.0"));
        assert!(debug_str.contains("2.0"));
        assert!(debug_str.contains("3.0"));

        // Test ordering edge cases
        let vertex1: Vertex<(), 2> = Vertex::<(), _>::try_new([1.0, 2.0]).unwrap();
        let vertex2: Vertex<(), 2> = Vertex::<(), _>::try_new([1.0, 2.0]).unwrap();

        // Test that equal points result in equal ordering
        assert!(vertex1.partial_cmp(&vertex2) != Some(Ordering::Less));
        assert!(vertex2.partial_cmp(&vertex1) != Some(Ordering::Less));
        assert_matches!(
            vertex1.partial_cmp(&vertex2),
            Some(Ordering::Less | Ordering::Equal)
        );
        assert_matches!(
            vertex2.partial_cmp(&vertex1),
            Some(Ordering::Less | Ordering::Equal)
        );
        assert_matches!(
            vertex1.partial_cmp(&vertex2),
            Some(Ordering::Greater | Ordering::Equal)
        );
        assert_matches!(
            vertex2.partial_cmp(&vertex1),
            Some(Ordering::Greater | Ordering::Equal)
        );
    }

    #[test]
    fn test_vertex_coordinate_values() {
        // Test negative coordinates
        let vertex_neg: Vertex<(), 3> = Vertex::<(), _>::try_new([-1.0, -2.0, -3.0]).unwrap();
        assert_relative_eq!(
            vertex_neg.point().coords().as_slice(),
            [-1.0, -2.0, -3.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(vertex_neg.dim(), 3);

        // Test zero coordinates
        let vertex_zero: Vertex<(), 3> = Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap();
        let origin_vertex: Vertex<(), 3> = Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap();
        assert_eq!(vertex_zero.point(), origin_vertex.point());

        // Test large coordinates
        let vertex_large: Vertex<(), 3> = Vertex::<(), _>::try_new([1e6, 2e6, 3e6]).unwrap();
        assert_relative_eq!(
            vertex_large.point().coords().as_slice(),
            [1_000_000.0, 2_000_000.0, 3_000_000.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(vertex_large.dim(), 3);

        // Test small coordinates
        let vertex_small: Vertex<(), 3> = Vertex::<(), _>::try_new([1e-6, 2e-6, 3e-6]).unwrap();
        assert_relative_eq!(
            vertex_small.point().coords().as_slice(),
            [0.000_001, 0.000_002, 0.000_003].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(vertex_small.dim(), 3);

        // Test mixed positive/negative coordinates
        let vertex_mixed: Vertex<(), 4> = Vertex::<(), _>::try_new([1.0, -2.0, 3.0, -4.0]).unwrap();
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
        let vertex1: Vertex<(), 3> = Vertex::<(), _>::try_new([1.0, 2.0, 3.0]).unwrap();
        let vertex2: Vertex<(), 3> = Vertex::<(), _>::try_new([1.0, 2.0, 3.0]).unwrap();

        // Same points but different UUIDs
        assert_ne!(vertex1.uuid(), vertex2.uuid());
        assert!(!vertex1.uuid().is_nil());
        assert!(!vertex2.uuid().is_nil());
    }

    #[test]
    fn test_vertex_type_conversions() {
        // Test implicit conversion from owned vertex to coordinates
        let vertex_coords: Vertex<(), 3> = Vertex::<(), _>::try_new([1.0, 2.0, 3.0]).unwrap();
        let coords_owned: [f64; 3] = vertex_coords.into();
        assert_relative_eq!(
            coords_owned.as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = 1e-9
        );

        // Test implicit conversion from vertex reference to coordinates
        let vertex_ref_coords: Vertex<(), 3> = Vertex::<(), _>::try_new([4.0, 5.0, 6.0]).unwrap();
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
        let vertex_point: Vertex<(), 3> = Vertex::<(), _>::try_new([1.0, 2.0, 3.0]).unwrap();
        let point_from_vertex: Point<3> = (&vertex_point).into();
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
        let vertex_2d: Vertex<(), 2> = Vertex::<(), _>::try_new([10.5, -5.3]).unwrap();
        let point_2d: Point<2> = (&vertex_2d).into();
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
    fn test_vertex_validation() {
        // Test valid vertices with f64 coordinates and various dimensions
        let valid_f64: Vertex<(), 3> = Vertex::<(), _>::try_new([1.0, 2.0, 3.0]).unwrap();
        assert!(valid_f64.is_valid().is_ok());

        let valid_negative: Vertex<(), 3> = Vertex::<(), _>::try_new([-1.0, -2.0, -3.0]).unwrap();
        assert!(valid_negative.is_valid().is_ok());

        let valid_zero: Vertex<(), 3> = Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap();
        assert!(valid_zero.is_valid().is_ok());

        // Test different dimensions
        let valid_1d: Vertex<(), 1> = Vertex::<(), _>::try_new([42.0]).unwrap();
        assert!(valid_1d.is_valid().is_ok());

        let valid_5d: Vertex<(), 5> = Vertex::<(), _>::try_new([1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert!(valid_5d.is_valid().is_ok());

        assert!(Point::<3>::try_new([1.0, f64::NAN, 3.0]).is_err());
        assert!(Point::<3>::try_new([f64::NAN, f64::NAN, f64::NAN]).is_err());
        assert!(Point::<1>::try_new([f64::NAN]).is_err());
        assert!(Point::<5>::try_new([1.0, 2.0, f64::NAN, 4.0, 5.0]).is_err());
        assert!(Point::<3>::try_new([1.0, f64::INFINITY, 3.0]).is_err());
        assert!(Point::<3>::try_new([1.0, f64::NEG_INFINITY, 3.0]).is_err());
        assert!(Point::<3>::try_new([f64::NAN, f64::INFINITY, 1.0]).is_err());

        // Test UUID validation
        let valid_vertex: Vertex<(), 3> = Vertex::<(), _>::try_new([1.0, 2.0, 3.0]).unwrap();
        assert!(valid_vertex.is_valid().is_ok());
        assert!(!valid_vertex.uuid().is_nil());

        // Create a vertex with valid point but manually set nil UUID to test UUID validation
        let invalid_uuid_vertex: Vertex<(), 3> = Vertex {
            point: Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates"),
            uuid: uuid::Uuid::nil(),
            incident_simplex: None,
            data: None,
        };
        match invalid_uuid_vertex.is_valid() {
            Err(VertexValidationError::InvalidUuid { source: _ }) => (), // Expected
            other => panic!("Expected InvalidUuid error, got: {other:?}"),
        }
        assert!(invalid_uuid_vertex.point().validate().is_ok());
        assert!(invalid_uuid_vertex.uuid().is_nil()); // UUID is nil

        assert!(Point::<3>::try_new([f64::NAN, 2.0, 3.0]).is_err());
    }

    // =============================================================================
    // ADVANCED DATA TESTS
    // =============================================================================

    #[test]
    fn vertex_string_data_usage_examples() {
        // String metadata works with `Vertex` constructors.

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
        let vertices_with_ids: Vec<Vertex<u32, 2>> = vec![
            Vertex::<_, _>::try_new_with_data([0.5, 0.5], 0u32).unwrap(), // center
            Vertex::<_, _>::try_new_with_data([1.0, 1.0], 1u32).unwrap(), // corner
            Vertex::<_, _>::try_new_with_data([0.5, 1.0], 2u32).unwrap(), // edge_midpoint
            Vertex::<_, _>::try_new_with_data([0.0, 0.5], 3u32).unwrap(), // boundary_point
        ];

        // Verify we can retrieve the labels
        for (v, expected_label) in
            vertices_with_ids
                .iter()
                .zip(["center", "corner", "edge_midpoint", "boundary_point"])
        {
            let label_id = v.data.unwrap();
            let label = label_lookup.get(&label_id).unwrap();
            assert_eq!(label, expected_label);
        }

        // Test that these vertices work with all normal operations
        assert_eq!(vertices_with_ids.len(), 4);
        let coords = vertices_with_ids[0].point().coords();
        assert_abs_diff_eq!(coords[0], 0.5, epsilon = f64::EPSILON);
        assert_abs_diff_eq!(coords[1], 0.5, epsilon = f64::EPSILON);
        assert_eq!(vertices_with_ids[1].data.unwrap(), 1u32);

        // Test hashing and equality (works because u32 implements all required traits)
        let vertex_set: FastHashSet<Vertex<u32, 2>> = vertices_with_ids.iter().copied().collect();
        assert_eq!(vertex_set.len(), 4);

        // =====================================================================
        // OTHER COPY-ABLE ALTERNATIVES FOR LABELS
        // =====================================================================

        // Alternative 1: Use character codes
        let vertices_with_chars: Vec<Vertex<char, 2>> = vec![
            Vertex::<_, _>::try_new_with_data([0.0, 0.0], 'A').unwrap(),
            Vertex::<_, _>::try_new_with_data([1.0, 0.0], 'B').unwrap(),
            Vertex::<_, _>::try_new_with_data([0.0, 1.0], 'C').unwrap(),
        ];

        for (i, v) in vertices_with_chars.iter().enumerate() {
            let expected_char =
                char::from(b'A' + usize_to_u8(i, 26).expect("Index should fit in u8"));
            assert_eq!(v.data.unwrap(), expected_char);
        }

        // Alternative 2: Use small integer codes with enum mapping

        let vertices_with_enums: Vec<Vertex<PointType, 2>> = vec![
            Vertex::<_, _>::try_new_with_data([0.0, 0.0], PointType::Origin).unwrap(),
            Vertex::<_, _>::try_new_with_data([1.0, 0.0], PointType::Corner).unwrap(),
            Vertex::<_, _>::try_new_with_data([0.5, 0.5], PointType::Interior).unwrap(),
        ];

        assert_eq!(vertices_with_enums[0].data.unwrap(), PointType::Origin);
        assert_eq!(vertices_with_enums[1].data.unwrap(), PointType::Corner);
        assert_eq!(vertices_with_enums[2].data.unwrap(), PointType::Interior);

        // =====================================================================
        // SUMMARY OF STRING DATA LIMITATIONS
        // =====================================================================

        // 1. Owned String metadata is supported with Vertex constructors,
        //    but it is not Copy and cannot be used in Copy-only contexts.
        // 2. &str has complex lifetime issues that make it impractical
        // 3. &'static str could work but only for compile-time constants
        // 4. Recommended Copy alternatives:
        //    - Numeric IDs with external lookup (most flexible)
        //    - Character codes (for single characters)
        //    - Custom Copy enums (for predefined categories)
        //    - Small fixed-size byte arrays (for very short strings)
    }

    #[test]
    fn vertex_hash_with_copy_data() {
        // Test hashing with Copy data
        let vertex1: Vertex<u16, 2> =
            Vertex::<_, _>::try_new_with_data([1.0, 2.0], 999u16).unwrap();

        let vertex2: Vertex<i32, 2> = Vertex::<_, _>::try_new_with_data([3.0, 4.0], 42).unwrap();

        // Test that vertices with Copy data can be used as HashMap keys
        let mut map: FastHashMap<Vertex<u16, 2>, i32> = FastHashMap::default();
        map.insert(vertex1, 100);

        let mut map2: FastHashMap<Vertex<i32, 2>, u8> = FastHashMap::default();
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

        let result: Result<Vertex<(), 2>, _> = serde_json::from_str(json_minimal);
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
        assert!(vertex.incident_simplex.is_none());
        assert!(vertex.data.is_none());

        // Test deserialization with all fields including SimplexKey
        let point = Point::try_new([1.5, 2.5, 3.5]).expect("finite point coordinates");
        let uuid_str = "550e8400-e29b-41d4-a716-446655440000";
        let uuid = uuid::Uuid::parse_str(uuid_str).unwrap();
        let simplex_key = SimplexKey::from(KeyData::from_ffi(42u64));

        let vertex_with_all = Vertex {
            point,
            uuid,
            incident_simplex: Some(simplex_key),
            data: Some(123i32),
        };

        assert_relative_eq!(
            vertex_with_all.point().coords().as_slice(),
            [1.5, 2.5, 3.5].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(vertex_with_all.uuid().to_string(), uuid_str);
        assert!(vertex_with_all.incident_simplex.is_some());
        assert_eq!(vertex_with_all.incident_simplex.unwrap(), simplex_key);
        assert_eq!(vertex_with_all.data.unwrap(), 123);

        // Test unknown field handling (should be ignored)
        let json_with_unknown = r#"{
            "point": [1.0, 2.0, 3.0],
            "uuid": "550e8400-e29b-41d4-a716-446655440000",
            "data": null,
            "unknown_field": "this should be ignored"
        }"#;

        let result: Result<Vertex<(), 3>, _> = serde_json::from_str(json_with_unknown);
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
                "duplicate field `point`",
            ),
            (
                r#"{"point": [1.0, 2.0, 3.0], "uuid": "550e8400-e29b-41d4-a716-446655440000", "uuid": "550e8400-e29b-41d4-a716-446655440001"}"#,
                "duplicate uuid",
                "duplicate field `uuid`",
            ),
            (
                r#"{"point": [1.0, 2.0, 3.0], "uuid": "550e8400-e29b-41d4-a716-446655440000", "incident_simplex": null}"#,
                "forbidden incident_simplex",
                "storage-local slotmap key",
            ),
            (
                r#"{"point": [1.0, 2.0, 3.0], "uuid": "550e8400-e29b-41d4-a716-446655440000", "data": null, "data": null}"#,
                "duplicate data",
                "duplicate field `data`",
            ),
            (
                r#"{"uuid": "550e8400-e29b-41d4-a716-446655440000"}"#,
                "missing point",
                "missing field `point`",
            ),
            (
                r#"{"point": [1.0, 2.0, 3.0]}"#,
                "missing uuid",
                "missing field `uuid`",
            ),
        ];

        for (json, description, expected_fragment) in test_cases {
            let result: Result<Vertex<(), 3>, _> = serde_json::from_str(json);
            assert!(
                result.is_err(),
                "Expected error for {description}, but got success"
            );
            let error_message = result.unwrap_err().to_string();
            assert!(
                error_message.contains(expected_fragment),
                "Error message for {description} should contain {expected_fragment:?}: {error_message}"
            );
        }

        // Test invalid JSON structure
        let invalid_json = r#"["not", "a", "vertex", "object"]"#;
        let result: Result<Vertex<(), 3>, _> = serde_json::from_str(invalid_json);
        assert!(result.is_err());
        let error_message = result.unwrap_err().to_string();
        assert!(
            error_message.contains("Vertex") || error_message.to_lowercase().contains("struct"),
            "Error message should mention Vertex struct: {error_message}"
        );

        // Test validation error for nil UUID
        let vertex_with_nil_uuid = Vertex {
            point: Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates"),
            uuid: uuid::Uuid::nil(),
            incident_simplex: None,
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
        let point_error = CoordinateValidationError::InvalidCoordinate {
            coordinate_index: 1,
            coordinate_value: InvalidCoordinateValue::Nan,
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

        let point_error = CoordinateValidationError::InvalidCoordinate {
            coordinate_index: 1,
            coordinate_value: InvalidCoordinateValue::Nan,
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
    // SERIALIZATION ROUNDTRIP TESTS
    // =============================================================================

    #[test]
    fn test_serialization_deserialization_roundtrip() {
        // Test that serialization -> deserialization preserves all data
        let original_vertex: Vertex<char, 4> =
            Vertex::<_, _>::try_new_with_data([1.0, 2.0, 3.0, 4.0], 'A').unwrap();

        // Serialize
        let serialized = serde_json::to_string(&original_vertex).unwrap();

        // Deserialize
        let deserialized_vertex: Vertex<char, 4> = serde_json::from_str(&serialized).unwrap();

        // Verify all fields match
        assert_relative_eq!(
            original_vertex.point().coords().as_slice(),
            deserialized_vertex.point().coords().as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(original_vertex.uuid(), deserialized_vertex.uuid());
        assert_eq!(
            original_vertex.incident_simplex,
            deserialized_vertex.incident_simplex
        );
        assert_eq!(original_vertex.data, deserialized_vertex.data);
    }

    #[test]
    fn test_serialization_with_some_data_includes_field() {
        // Test that when data is Some, the JSON includes the data field
        let vertex: Vertex<i32, 3> =
            Vertex::<_, _>::try_new_with_data([1.0, 2.0, 3.0], 42).unwrap();
        let serialized = serde_json::to_string(&vertex).unwrap();

        // Verify JSON contains "data" field
        assert!(
            serialized.contains("\"data\":"),
            "JSON should include data field when Some"
        );
        assert!(serialized.contains("42"), "JSON should include data value");

        // Roundtrip test
        let deserialized: Vertex<i32, 3> = serde_json::from_str(&serialized).unwrap();
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
        let vertex: Vertex<(), 3> = Vertex::<(), _>::try_new([1.0, 2.0, 3.0]).unwrap();
        let serialized = serde_json::to_string(&vertex).unwrap();

        // Verify JSON does NOT contain "data" field (optimization)
        assert!(
            !serialized.contains("\"data\":"),
            "JSON should omit data field when None"
        );

        // Roundtrip test - missing data field should deserialize as None
        let deserialized: Vertex<(), 3> = serde_json::from_str(&serialized).unwrap();
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
        let vertex: Vertex<(), 3> = serde_json::from_str(json_with_null).unwrap();

        assert_eq!(vertex.data, None);
        assert_relative_eq!(
            vertex.point().coords().as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = 1e-9
        );
    }
}
