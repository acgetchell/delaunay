//! D-dimensional Facets Representation
//!
//! This module provides the `Facet` struct which represents a facet of a d-dimensional simplex
//! (d-1 sub-simplex) within a triangulation. Each facet is defined in terms of a cell and the
//! vertex opposite to it, similar to [CGAL](https://doc.cgal.org/latest/TDS_3/index.html#title3).
//!
//! # Key Features
//!
//! - **Dimensional Simplicity**: Represents co-dimension 1 sub-simplexes of d-dimensional simplexes
//! - **Cell Association**: Each facet resides within a specific cell and is described by its opposite vertex
//! - **Support for Delaunay Triangulations**: Facilitates operations fundamental to the
//!   [Bowyer-Watson algorithm](https://en.wikipedia.org/wiki/Bowyer–Watson_algorithm)
//! - **On-demand Creation**: Facets are generated dynamically as needed rather than stored persistently in the TDS
//! - **Serialization Support**: Full serde support for persistence and interoperability
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
//! use delaunay::core::facet::Facet;
//! use delaunay::core::cell::Cell;
//! use delaunay::core::vertex::Vertex;
//! use delaunay::{cell, vertex};
//! use delaunay::geometry::point::Point;
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
//! let cell: Cell<f64, Option<()>, Option<()>, 3> = cell!(vertices.clone());
//!
//! // Create a facet with vertex as opposite
//! let facet = Facet::new(cell, vertices[0]).unwrap();
//! assert_eq!(facet.vertices().len(), 3);  // Facet (triangle) in 3D has 3 vertices
//! ```

// =============================================================================
// IMPORTS
// =============================================================================

use super::traits::data_type::DataType;
use super::util::stable_hash_u64_slice;
use super::{cell::Cell, triangulation_data_structure::VertexKey, vertex::Vertex};
use crate::geometry::traits::coordinate::CoordinateScalar;
use serde::{Serialize, de::DeserializeOwned};
use slotmap::Key;
use std::hash::{Hash, Hasher};
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
    /// A vertex UUID was not found in the vertex bimap.
    #[error("Vertex UUID not found in bimap: {uuid}")]
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
}

// =============================================================================
// FACET STRUCT DEFINITION
// =============================================================================

#[derive(Clone, Debug, Default, PartialEq, PartialOrd, Serialize)]
/// The [Facet] struct represents a facet of a d-dimensional simplex.
/// Passing in a [Vertex] and a [Cell] containing that vertex to the
/// constructor will create a [Facet] struct.
///
/// # Properties
///
/// - `cell` - The [Cell] that contains this facet.
/// - `vertex` - The [Vertex] in the [Cell] opposite to this [Facet].
///
/// Note that `D` is the dimensionality of the [Cell] and [Vertex];
/// the [Facet] is one dimension less than the [Cell] (co-dimension 1).
pub struct Facet<T, U, V, const D: usize>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// The [Cell] that contains this facet.
    cell: Cell<T, U, V, D>,

    /// The [Vertex] opposite to this facet.
    vertex: Vertex<T, U, D>,
}

// =============================================================================
// DESERIALIZATION IMPLEMENTATION
// =============================================================================

/// Manual implementation of Deserialize for Facet
impl<'de, T, U, V, const D: usize> serde::Deserialize<'de> for Facet<T, U, V, D>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        struct FacetVisitor<T, U, V, const D: usize>
        where
            T: CoordinateScalar,
            U: DataType,
            V: DataType,
            [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
        {
            _phantom: std::marker::PhantomData<(T, U, V)>,
        }

        impl<'de, T, U, V, const D: usize> Visitor<'de> for FacetVisitor<T, U, V, D>
        where
            T: CoordinateScalar,
            U: DataType,
            V: DataType,
            [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
        {
            type Value = Facet<T, U, V, D>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a Facet struct")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Facet<T, U, V, D>, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut cell = None;
                let mut vertex = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        "cell" => {
                            if cell.is_some() {
                                return Err(de::Error::duplicate_field("cell"));
                            }
                            cell = Some(map.next_value()?);
                        }
                        "vertex" => {
                            if vertex.is_some() {
                                return Err(de::Error::duplicate_field("vertex"));
                            }
                            vertex = Some(map.next_value()?);
                        }
                        _ => {
                            let _ = map.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }

                let cell = cell.ok_or_else(|| de::Error::missing_field("cell"))?;
                let vertex = vertex.ok_or_else(|| de::Error::missing_field("vertex"))?;

                Facet::new(cell, vertex).map_err(|e| {
                    de::Error::custom(format!("Failed to create Facet from cell and vertex: {e}"))
                })
            }
        }

        const FIELDS: &[&str] = &["cell", "vertex"];
        deserializer.deserialize_struct(
            "Facet",
            FIELDS,
            FacetVisitor {
                _phantom: std::marker::PhantomData,
            },
        )
    }
}

// =============================================================================
// FACET IMPLEMENTATION
// =============================================================================

impl<T, U, V, const D: usize> Facet<T, U, V, D>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// The `new` function is a constructor for the [Facet]. It takes
    /// in a [Cell] and a [Vertex] as arguments and returns a [Result]
    /// containing a [Facet] or an error message.
    ///
    /// # Arguments
    ///
    /// - `cell`: The [Cell] that contains the [Facet].
    /// - `vertex`: The [Vertex] opposite to the [Facet].
    ///
    /// # Returns
    ///
    /// A [Result] containing a [Facet] or a [`FacetError`] as to why
    /// the [Facet] could not be created.
    ///
    /// # Errors
    ///
    /// Returns a [`FacetError`] if:
    /// - The cell does not contain the specified vertex ([`FacetError::CellDoesNotContainVertex`])
    ///
    /// # Example
    ///
    /// ```
    /// use delaunay::{cell, vertex};
    /// use delaunay::core::cell::Cell;
    /// use delaunay::core::facet::Facet;
    /// use delaunay::core::vertex::Vertex;
    /// let vertex1: Vertex<f64, Option<()>, 3> = vertex!([0.0, 0.0, 0.0]);
    /// let vertex2: Vertex<f64, Option<()>, 3> = vertex!([1.0, 0.0, 0.0]);
    /// let vertex3: Vertex<f64, Option<()>, 3> = vertex!([0.0, 1.0, 0.0]);
    /// let vertex4: Vertex<f64, Option<()>, 3> = vertex!([0.0, 0.0, 1.0]);
    /// let cell: Cell<f64, Option<()>, Option<()>, 3> = cell!(vec![vertex1, vertex2, vertex3, vertex4]);
    /// let facet = Facet::new(cell.clone(), vertex1).unwrap();
    /// assert_eq!(facet.cell(), &cell);
    /// ```
    pub fn new(cell: Cell<T, U, V, D>, vertex: Vertex<T, U, D>) -> Result<Self, FacetError> {
        if !cell.contains_vertex(vertex) {
            return Err(FacetError::CellDoesNotContainVertex);
        }

        Ok(Self { cell, vertex })
    }

    /// Returns a reference to the [Cell] that contains this facet.
    ///
    /// # Returns
    ///
    /// A reference to the [Cell] that defines this facet.
    ///
    /// # Example
    ///
    /// ```
    /// use delaunay::{cell, vertex};
    /// use delaunay::core::cell::Cell;
    /// use delaunay::core::facet::Facet;
    /// use delaunay::geometry::point::Point;
    /// use delaunay::geometry::traits::coordinate::Coordinate;
    ///
    /// let vertex1 = vertex!([0.0, 0.0, 0.0]);
    /// let vertex2 = vertex!([1.0, 0.0, 0.0]);
    /// let vertex3 = vertex!([0.0, 1.0, 0.0]);
    /// let vertex4 = vertex!([0.0, 0.0, 1.0]);
    ///
    /// let cell: Cell<f64, Option<()>, Option<()>, 3> = cell!(vec![vertex1, vertex2, vertex3, vertex4]);
    ///
    /// let facet = Facet::new(cell.clone(), vertex1).unwrap();
    ///
    /// // Access the cell through the getter
    /// let facet_cell = facet.cell();
    /// assert_eq!(facet_cell.vertices().len(), 4);
    /// assert_eq!(facet_cell.uuid(), cell.uuid());
    /// ```
    #[inline]
    pub const fn cell(&self) -> &Cell<T, U, V, D> {
        &self.cell
    }

    /// Returns a reference to the [Vertex] opposite to this facet.
    ///
    /// The opposite vertex is the vertex in the cell that is not part of the facet.
    /// In a d-dimensional simplex, the facet is a (d-1)-dimensional sub-simplex,
    /// and the opposite vertex is the one vertex that, when removed, leaves the facet.
    ///
    /// # Returns
    ///
    /// A reference to the [Vertex] opposite to this facet.
    ///
    /// # Example
    ///
    /// ```
    /// use delaunay::{cell, vertex};
    /// use delaunay::core::cell::Cell;
    /// use delaunay::core::facet::Facet;
    /// use delaunay::geometry::point::Point;
    /// use delaunay::geometry::traits::coordinate::Coordinate;
    ///
    /// let vertex1 = vertex!([0.0, 0.0, 0.0]);
    /// let vertex2 = vertex!([1.0, 0.0, 0.0]);
    /// let vertex3 = vertex!([0.0, 1.0, 0.0]);
    /// let vertex4 = vertex!([0.0, 0.0, 1.0]);
    ///
    /// let cell: Cell<f64, Option<()>, Option<()>, 3> = cell!(vec![vertex1, vertex2, vertex3, vertex4]);
    ///
    /// let facet = Facet::new(cell.clone(), vertex1).unwrap();
    ///
    /// // Access the opposite vertex through the getter
    /// let opposite_vertex = facet.vertex();
    /// assert_eq!(opposite_vertex.uuid(), vertex1.uuid());
    ///
    /// // The facet's vertices should be all vertices except the opposite one
    /// let facet_vertices = facet.vertices();
    /// assert_eq!(facet_vertices.len(), 3);
    /// assert!(!facet_vertices.contains(&vertex1)); // opposite vertex not in facet
    /// assert!(facet_vertices.contains(&vertex2));
    /// assert!(facet_vertices.contains(&vertex3));
    /// assert!(facet_vertices.contains(&vertex4));
    /// ```
    #[inline]
    pub const fn vertex(&self) -> &Vertex<T, U, D> {
        &self.vertex
    }

    /// Returns the vertices that make up this facet.
    ///
    /// In a d-dimensional simplex, a facet is a (d-1)-dimensional sub-simplex.
    /// This method returns all vertices of the cell except the opposite vertex.
    /// For example, in a 3D tetrahedron (4 vertices), each facet is a triangle (3 vertices).
    ///
    /// # Returns
    ///
    /// A `Vec<Vertex<T, U, D>>` containing all vertices that form this facet,
    /// which are all the cell's vertices excluding the opposite vertex.
    ///
    /// # Example
    ///
    /// ```
    /// use delaunay::{cell, vertex};
    /// use delaunay::core::cell::Cell;
    /// use delaunay::core::facet::Facet;
    /// use delaunay::geometry::point::Point;
    /// use delaunay::geometry::traits::coordinate::Coordinate;
    ///
    /// // Create a 3D tetrahedron with 4 vertices
    /// let vertex1 = vertex!([0.0, 0.0, 0.0]); // origin
    /// let vertex2 = vertex!([1.0, 0.0, 0.0]); // x-axis
    /// let vertex3 = vertex!([0.0, 1.0, 0.0]); // y-axis
    /// let vertex4 = vertex!([0.0, 0.0, 1.0]); // z-axis
    ///
    /// let cell: Cell<f64, Option<()>, Option<()>, 3> = cell!(vec![vertex1, vertex2, vertex3, vertex4]);
    ///
    /// // Create a facet with vertex1 as the opposite vertex
    /// let facet = Facet::new(cell.clone(), vertex1).unwrap();
    ///
    /// // Get the vertices that make up this facet
    /// let facet_vertices = facet.vertices();
    ///
    /// // The facet should contain 3 vertices (it's a triangle in 3D)
    /// assert_eq!(facet_vertices.len(), 3);
    ///
    /// // The facet should NOT contain the opposite vertex (vertex1)
    /// assert!(!facet_vertices.contains(&vertex1));
    ///
    /// // The facet should contain all other vertices
    /// assert!(facet_vertices.contains(&vertex2));
    /// assert!(facet_vertices.contains(&vertex3));
    /// assert!(facet_vertices.contains(&vertex4));
    ///
    /// // Verify we have exactly the expected vertices
    /// let mut expected_vertices = vec![vertex2, vertex3, vertex4];
    /// expected_vertices.sort_by_key(|v| v.uuid());
    /// let mut actual_vertices = facet_vertices;
    /// actual_vertices.sort_by_key(|v| v.uuid());
    /// assert_eq!(actual_vertices, expected_vertices);
    ///
    /// // Test with a different opposite vertex
    /// let facet2 = Facet::new(cell.clone(), vertex2).unwrap();
    /// let facet2_vertices = facet2.vertices();
    ///
    /// // This facet should exclude vertex2 and include vertex1, vertex3, vertex4
    /// assert_eq!(facet2_vertices.len(), 3);
    /// assert!(!facet2_vertices.contains(&vertex2)); // opposite vertex excluded
    /// assert!(facet2_vertices.contains(&vertex1));
    /// assert!(facet2_vertices.contains(&vertex3));
    /// assert!(facet2_vertices.contains(&vertex4));
    /// ```
    pub fn vertices(&self) -> Vec<Vertex<T, U, D>> {
        self.cell
            .vertices()
            .iter()
            .filter(|v| **v != self.vertex)
            .copied()
            .collect()
    }

    /// Returns a canonical key for the facet.
    ///
    /// This key is a stable hash of the vertex UUIDs after sorting the vertices by UUID,
    /// ensuring any two facets sharing the same vertices have the same key, regardless of input order.
    /// Uses the same deterministic hash algorithm as `facet_key_from_vertex_keys`.
    ///
    /// # Returns
    ///
    /// A `u64` hash value representing the canonical key of the facet.
    pub fn key(&self) -> u64 {
        let mut vertices = self.vertices();
        vertices.sort_by_key(Vertex::uuid);

        // Convert UUIDs to u64 values for hashing
        let mut uuid_values = Vec::with_capacity(vertices.len() * 2);
        for vertex in vertices {
            let uuid_bytes = vertex.uuid().as_u128();
            // Intentionally truncate to u64 to get low bits, then high bits
            #[allow(clippy::cast_possible_truncation)]
            {
                uuid_values.push(uuid_bytes as u64);
                uuid_values.push((uuid_bytes >> 64) as u64);
            }
        }

        // Use the shared stable hash function
        stable_hash_u64_slice(&uuid_values)
    }
}

// =============================================================================
// TRAIT IMPLEMENTATIONS
// =============================================================================

// Consolidated trait implementations for Facet
impl<T, U, V, const D: usize> Eq for Facet<T, U, V, D>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    Vertex<T, U, D>: Hash,
    Cell<T, U, V, D>: Hash,
{
}

impl<T, U, V, const D: usize> Hash for Facet<T, U, V, D>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    Vertex<T, U, D>: Hash,
    Cell<T, U, V, D>: Hash,
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.cell.hash(state);
        self.vertex.hash(state);
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
/// * `vertex_keys` - A slice of `VertexKeys` representing the vertices of the facet
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
/// use delaunay::core::facet::facet_key_from_vertex_keys;
/// use delaunay::core::triangulation_data_structure::VertexKey;
/// use slotmap::Key;
///
/// // Create some vertex keys (normally these would come from a TDS)
/// let vertex_keys = vec![
///     VertexKey::from(slotmap::KeyData::from_ffi(1u64)),
///     VertexKey::from(slotmap::KeyData::from_ffi(2u64)),
///     VertexKey::from(slotmap::KeyData::from_ffi(3u64)),
/// ];
///
/// // Generate facet key from vertex keys
/// let facet_key = facet_key_from_vertex_keys(&vertex_keys);
///
/// // The same vertices in different order should produce the same key
/// let mut reversed_keys = vertex_keys.clone();
/// reversed_keys.reverse();
/// let facet_key_reversed = facet_key_from_vertex_keys(&reversed_keys);
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
pub fn facet_key_from_vertex_keys(vertex_keys: &[VertexKey]) -> u64 {
    // Handle empty case
    if vertex_keys.is_empty() {
        return 0;
    }

    // Convert VertexKeys to u64 and sort for deterministic ordering
    let mut key_values: Vec<u64> = vertex_keys.iter().map(|key| key.data().as_ffi()).collect();
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
    use crate::core::triangulation_data_structure::VertexKey;
    use crate::{cell, vertex};
    use approx::assert_relative_eq;

    // =============================================================================
    // TYPE ALIASES AND HELPERS
    // =============================================================================

    type TestVertex3D = Vertex<f64, Option<()>, 3>;
    type TestCell3D = Cell<f64, Option<()>, Option<()>, 3>;

    // Helper function to create a standard tetrahedron (3D cell with 4 vertices)
    fn create_tetrahedron() -> (TestCell3D, Vec<TestVertex3D>) {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let cell = cell!(vertices.clone());
        (cell, vertices)
    }

    type TestVertex2D = Vertex<f64, Option<()>, 2>;
    type TestCell2D = Cell<f64, Option<()>, Option<()>, 2>;

    // Helper function to create a triangle (2D cell with 3 vertices)
    fn create_triangle() -> (TestCell2D, Vec<TestVertex2D>) {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.5, 1.0]),
        ];
        let cell = cell!(vertices.clone());
        (cell, vertices)
    }

    // =============================================================================
    // FACET CREATION TESTS
    // =============================================================================

    #[test]
    fn test_facet_error_handling() {
        // Test cell does not contain vertex error
        let vertex1: Vertex<f64, Option<()>, 1> = vertex!([0.0]);
        let vertex2: Vertex<f64, Option<()>, 1> = vertex!([1.0]);
        let cell_1d: Cell<f64, Option<()>, Option<()>, 1> = cell!(vec![vertex1, vertex2]);

        // Test cell does not contain vertex error using the valid 2-vertex cell
        let vertex3: Vertex<f64, Option<()>, 1> = vertex!([2.0]);
        assert!(matches!(
            Facet::new(cell_1d, vertex3),
            Err(FacetError::CellDoesNotContainVertex)
        ));
    }

    #[test]
    fn facet_new() {
        let (cell, vertices) = create_tetrahedron();
        let facet = Facet::new(cell.clone(), vertices[0]).unwrap();

        assert_eq!(facet.cell(), &cell);

        // Human readable output for cargo test -- --nocapture
        println!("Facet: {facet:?}");
    }

    #[test]
    fn test_facet_new_success_coverage() {
        // Test 2D case: Create a triangle (2D cell with 3 vertices)
        let (cell_2d, vertices_2d) = create_triangle();
        let result_2d = Facet::new(cell_2d, vertices_2d[0]);

        // Assert that the result is Ok, ensuring the Ok(Self { ... }) line is covered
        assert!(result_2d.is_ok());
        let facet_2d = result_2d.unwrap();
        assert_eq!(facet_2d.vertices().len(), 2); // 2D facet should have 2 vertices

        // Test 1D case: Create an edge (1D cell with 2 vertices)
        let vertex1 = vertex!([0.0]);
        let vertex2 = vertex!([1.0]);
        let cell_1d: Cell<f64, Option<()>, Option<()>, 1> = cell!(vec![vertex1, vertex2]);
        let result_1d = Facet::new(cell_1d, vertex1);

        // Assert that the result is Ok, ensuring the Ok(Self { ... }) line is covered
        assert!(result_1d.is_ok());
        let facet_1d = result_1d.unwrap();
        assert_eq!(facet_1d.vertices().len(), 1); // 1D facet should have 1 vertex
    }

    #[test]
    fn facet_new_with_incorrect_vertex() {
        let vertex1 = vertex!([0.0, 0.0, 0.0]);
        let vertex2 = vertex!([1.0, 0.0, 0.0]);
        let vertex3 = vertex!([0.0, 1.0, 0.0]);
        let vertex4 = vertex!([0.0, 0.0, 1.0]);
        let cell: Cell<f64, Option<()>, Option<()>, 3> =
            cell!(vec![vertex1, vertex2, vertex3, vertex4]);
        let vertex5 = vertex!([1.0, 1.0, 1.0]);

        assert!(Facet::new(cell, vertex5).is_err());
    }

    #[test]
    fn facet_vertices() {
        let (cell, vertices) = create_tetrahedron();
        let facet = Facet::new(cell, vertices[0]).unwrap();
        let facet_vertices = facet.vertices();

        assert_eq!(facet_vertices.len(), 3);
        assert_eq!(facet_vertices[0], vertices[1]);
        assert_eq!(facet_vertices[1], vertices[2]);
        assert_eq!(facet_vertices[2], vertices[3]);

        // Human readable output for cargo test -- --nocapture
        println!("Facet: {facet:?}");
    }

    // =============================================================================
    // SERIALIZATION TESTS
    // =============================================================================

    /// Helper function that constructs a simple 2D facet (triangle) and serializes it to JSON,
    /// then splits the JSON string into separate `cell` and `vertex` components for reuse
    /// in custom JSON inputs for error-path tests.
    fn create_facet_json_components() -> (String, String) {
        // Create a simple 2D triangle facet
        let (cell, vertices) = create_triangle();
        let facet = Facet::new(cell, vertices[0]).unwrap();

        // Serialize the entire facet to JSON
        let serialized = serde_json::to_string(&facet).unwrap();

        // Parse the JSON to extract cell and vertex components
        let json: serde_json::Value = serde_json::from_str(&serialized).unwrap();
        let cell_json = serde_json::to_string(&json["cell"]).unwrap();
        let vertex_json = serde_json::to_string(&json["vertex"]).unwrap();

        (cell_json, vertex_json)
    }

    #[test]
    fn facet_deserialization_with_extra_field() {
        // Create a Facet JSON snippet with additional "extra" field
        let (cell_json, vertex_json) = create_facet_json_components();
        let json_with_extra = format!(
            "{{ \"cell\": {cell_json}, \"vertex\": {vertex_json}, \"extra\": \"ignored\" }}"
        );

        // Deserialize and test that it succeeds, ignoring the extra field
        let result: Result<Facet<f64, Option<()>, Option<()>, 2>, _> =
            serde_json::from_str(&json_with_extra);

        // Should succeed - extra fields are ignored
        assert!(result.is_ok());
    }

    #[test]
    fn test_facet_json_components() {
        let (cell_json, vertex_json) = create_facet_json_components();

        // Verify that we can extract valid JSON components
        println!("Cell JSON: {cell_json}");
        println!("Vertex JSON: {vertex_json}");

        // Basic validation that the components contain expected data
        assert!(cell_json.contains("vertices"));
        assert!(cell_json.contains("uuid"));
        assert!(vertex_json.contains("[0.0,0.0]")); // The opposite vertex coordinates

        // Verify the components can be parsed back as JSON
        let _: serde_json::Value = serde_json::from_str(&cell_json).unwrap();
        let _: serde_json::Value = serde_json::from_str(&vertex_json).unwrap();
    }

    #[test]
    fn facet_to_json() {
        let vertex1 = vertex!([0.0, 0.0, 0.0]);
        let vertex2 = vertex!([1.0, 0.0, 0.0]);
        let vertex3 = vertex!([0.0, 1.0, 0.0]);
        let vertex4 = vertex!([0.0, 0.0, 1.0]);
        let cell: Cell<f64, Option<()>, Option<()>, 3> =
            cell!(vec![vertex1, vertex2, vertex3, vertex4]);
        let facet = Facet::new(cell, vertex1).unwrap();
        let serialized = serde_json::to_string(&facet).unwrap();

        assert!(serialized.contains("[1.0,0.0,0.0]"));
        assert!(serialized.contains("[0.0,1.0,0.0]"));
        assert!(serialized.contains("[0.0,0.0,1.0]"));

        // Note: Deserialization test removed since we use DeserializeOwned trait bound
        // instead of the derive macro to avoid conflicts with serde trait bounds

        // Human readable output for cargo test -- --nocapture
        println!("Serialized = {serialized:?}");
    }

    #[test]
    fn test_facet_deserialization_duplicate_vertex_field() {
        let (cell_json, vertex_json) = create_facet_json_components();
        let json_with_duplicate_vertex = format!(
            "{{ \"cell\": {cell_json}, \"vertex\": {vertex_json}, \"vertex\": {vertex_json} }}"
        );

        let result: Result<Facet<f64, Option<()>, Option<()>, 2>, _> =
            serde_json::from_str(&json_with_duplicate_vertex);
        assert!(result.is_err());
        let error_message = result.unwrap_err().to_string();
        assert!(error_message.contains("duplicate field `vertex`"));
    }

    #[test]
    fn test_facet_deserialization_duplicate_cell_field_v2() {
        let (cell_json, vertex_json) = create_facet_json_components();
        let json_with_duplicate_cell = format!(
            "{{ \"cell\": {cell_json}, \"vertex\": {vertex_json}, \"cell\": {cell_json} }}"
        );

        let result: Result<Facet<f64, Option<()>, Option<()>, 2>, _> =
            serde_json::from_str(&json_with_duplicate_cell);
        assert!(result.is_err());
        let error_message = result.unwrap_err().to_string();
        assert!(error_message.contains("duplicate field `cell`"));
    }

    #[test]
    fn test_missing_cell_field_error() {
        // Step 5: Test missing `cell` field error
        // Construct JSON with only the `vertex` field: `{\"vertex\":<v>}`
        let (_, vertex_json) = create_facet_json_components();
        let json_with_only_vertex = format!("{{ \"vertex\": {vertex_json} }}");

        // Attempt to deserialize and assert the error message contains `missing field \"cell\"`
        let result: Result<Facet<f64, Option<()>, Option<()>, 2>, _> =
            serde_json::from_str(&json_with_only_vertex);
        assert!(result.is_err());
        let error_message = result.unwrap_err().to_string();
        assert!(error_message.contains("missing field `cell`"));
    }

    #[test]
    fn test_empty_json_object_missing_cell_field() {
        // Step 7: Test empty JSON object error
        // Attempt to deserialize the string `{}` into `Facet<...>` and assert it errors out
        // with `missing field "cell"`, covering the very first missing-field path.
        let empty_json = "{}";

        let result: Result<Facet<f64, Option<()>, Option<()>, 3>, _> =
            serde_json::from_str(empty_json);
        assert!(result.is_err());

        let error_message = result.unwrap_err().to_string();
        assert!(error_message.contains("missing field `cell`"));

        // Human readable output for cargo test -- --nocapture
        println!("Empty JSON deserialization error: {error_message}");
    }

    #[test]
    fn test_facet_deserialization_expecting_formatter() {
        // Test the expecting formatter method
        let invalid_json = r#"["not", "a", "facet", "object"]"#;
        let result: Result<Facet<f64, Option<()>, Option<()>, 3>, _> =
            serde_json::from_str(invalid_json);
        assert!(result.is_err());
        let error_message = result.unwrap_err().to_string();
        // The error should mention that it expected a Facet struct
        assert!(error_message.contains("Facet") || error_message.to_lowercase().contains("struct"));
    }

    // =============================================================================
    // EQUALITY AND ORDERING TESTS
    // =============================================================================

    #[test]
    fn facet_partial_eq() {
        let vertex1 = vertex!([0.0, 0.0, 0.0]);
        let vertex2 = vertex!([1.0, 0.0, 0.0]);
        let vertex3 = vertex!([0.0, 1.0, 0.0]);
        let vertex4 = vertex!([0.0, 0.0, 1.0]);
        let cell: Cell<f64, Option<()>, Option<()>, 3> =
            cell!(vec![vertex1, vertex2, vertex3, vertex4]);
        let facet1 = Facet::new(cell.clone(), vertex1).unwrap();
        let facet2 = Facet::new(cell.clone(), vertex1).unwrap();
        let facet3 = Facet::new(cell, vertex2).unwrap();

        assert_eq!(facet1, facet2);
        assert_ne!(facet1, facet3);
    }

    #[test]
    fn facet_partial_ord() {
        let vertex1 = vertex!([0.0, 0.0, 0.0]);
        let vertex2 = vertex!([1.0, 0.0, 0.0]);
        let vertex3 = vertex!([0.0, 1.0, 0.0]);
        let vertex4 = vertex!([0.0, 0.0, 1.0]);
        let cell: Cell<f64, Option<()>, Option<()>, 3> =
            cell!(vec![vertex1, vertex2, vertex3, vertex4]);
        let facet1 = Facet::new(cell.clone(), vertex1).unwrap();
        let facet2 = Facet::new(cell.clone(), vertex1).unwrap();
        let facet3 = Facet::new(cell.clone(), vertex2).unwrap();
        let facet4 = Facet::new(cell, vertex3).unwrap();

        assert!(facet1 < facet3);
        assert!(facet2 < facet3);
        assert!(facet3 > facet1);
        assert!(facet3 > facet2);
        assert!(facet3 > facet4);
    }

    #[test]
    fn facet_clone() {
        let vertex1 = vertex!([0.0, 0.0, 0.0]);
        let vertex2 = vertex!([1.0, 0.0, 0.0]);
        let vertex3 = vertex!([0.0, 1.0, 0.0]);
        let vertex4 = vertex!([0.0, 0.0, 1.0]);
        let cell: Cell<f64, Option<()>, Option<()>, 3> =
            cell!(vec![vertex1, vertex2, vertex3, vertex4]);
        let facet = Facet::new(cell, vertex1).unwrap();
        let cloned_facet = facet.clone();

        assert_eq!(facet, cloned_facet);
        assert_eq!(facet.cell().uuid(), cloned_facet.cell().uuid());
        assert_eq!(facet.vertex().uuid(), cloned_facet.vertex().uuid());
    }

    #[test]
    fn facet_default() {
        let facet: Facet<f64, Option<()>, Option<()>, 3> = Facet::default();

        // Default facet should have empty cell and default vertex
        assert_eq!(facet.cell().vertices().len(), 0);
        let default_coords: [f64; 3] = facet.vertex().into();
        assert_relative_eq!(
            default_coords.as_slice(),
            [0.0, 0.0, 0.0].as_slice(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn facet_debug() {
        let vertex1 = vertex!([1.0, 2.0, 3.0]);
        let vertex2 = vertex!([4.0, 5.0, 6.0]);
        let vertex3 = vertex!([7.0, 8.0, 9.0]);
        let vertex4 = vertex!([10.0, 11.0, 12.0]);
        let cell: Cell<f64, Option<()>, Option<()>, 3> =
            cell!(vec![vertex1, vertex2, vertex3, vertex4]);
        let facet = Facet::new(cell, vertex1).unwrap();
        let debug_str = format!("{facet:?}");

        assert!(debug_str.contains("Facet"));
        assert!(debug_str.contains("cell"));
        assert!(debug_str.contains("vertex"));
    }

    // =============================================================================
    // DIMENSIONAL AND GEOMETRIC TESTS
    // =============================================================================

    #[test]
    fn facet_with_typed_data() {
        let vertex1: Vertex<f64, i32, 3> = vertex!([0.0, 0.0, 0.0], 1);
        let vertex2: Vertex<f64, i32, 3> = vertex!([1.0, 0.0, 0.0], 2);
        let vertex3: Vertex<f64, i32, 3> = vertex!([0.0, 1.0, 0.0], 3);
        let vertex4: Vertex<f64, i32, 3> = vertex!([0.0, 0.0, 1.0], 4);
        let cell: Cell<f64, i32, i32, 3> = cell!(vec![vertex1, vertex2, vertex3, vertex4], 3);
        let facet = Facet::new(cell, vertex1).unwrap();

        assert_eq!(facet.cell().data, Some(3));
        assert_eq!(facet.vertex().data, Some(1));

        let vertices = facet.vertices();
        assert_eq!(vertices.len(), 3); // 3D facet should have 3 vertices (D)
        assert!(vertices.iter().any(|v| v.data == Some(2)));
        assert!(vertices.iter().any(|v| v.data == Some(3)));
        assert!(vertices.iter().any(|v| v.data == Some(4)));
    }

    #[test]
    fn facet_2d_triangle() {
        let (cell, vertices) = create_triangle();
        let facet = Facet::new(cell, vertices[0]).unwrap();

        // Facet of 2D triangle is an edge (1D)
        let facet_vertices = facet.vertices();
        assert_eq!(facet_vertices.len(), 2);
        assert_eq!(facet_vertices[0], vertices[1]);
        assert_eq!(facet_vertices[1], vertices[2]);
    }

    #[test]
    fn facet_1d_edge() {
        let vertex1 = vertex!([0.0]);
        let vertex2 = vertex!([1.0]);
        let cell: Cell<f64, Option<()>, Option<()>, 1> = cell!(vec![vertex1, vertex2]);
        let facet = Facet::new(cell, vertex1).unwrap();

        // Facet of 1D edge is a point (0D)
        let vertices = facet.vertices();
        assert_eq!(vertices.len(), 1);
        assert_eq!(vertices[0], vertex2);
    }

    #[test]
    fn facet_4d_simplex() {
        let vertex1 = vertex!([0.0, 0.0, 0.0, 0.0]);
        let vertex2 = vertex!([1.0, 0.0, 0.0, 0.0]);
        let vertex3 = vertex!([0.0, 1.0, 0.0, 0.0]);
        let vertex4 = vertex!([0.0, 0.0, 1.0, 0.0]);
        let vertex5 = vertex!([0.0, 0.0, 0.0, 1.0]);
        let cell: Cell<f64, Option<()>, Option<()>, 4> =
            cell!(vec![vertex1, vertex2, vertex3, vertex4, vertex5]);
        let facet = Facet::new(cell, vertex1).unwrap();

        // Facet of 4D simplex is a 3D tetrahedron
        let vertices = facet.vertices();
        assert_eq!(vertices.len(), 4);
        assert!(vertices.contains(&vertex2));
        assert!(vertices.contains(&vertex3));
        assert!(vertices.contains(&vertex4));
        assert!(vertices.contains(&vertex5));
        assert!(!vertices.contains(&vertex1));
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
        let (cell, vertices) = create_tetrahedron();

        // Create a facet with vertices[0] as opposite vertex
        // This facet contains vertices[1], vertices[2], vertices[3]
        let facet1 = Facet::new(cell.clone(), vertices[0]).unwrap();

        // Create another cell with the same vertices but in different order
        let reversed_vertices = {
            let mut a = vertices.clone();
            a.reverse();
            a
        };
        let cell2: TestCell3D = cell!(reversed_vertices.clone());

        // Create a facet from the reversed cell with reversed_vertices[3] (which is vertices[0]) as opposite
        // This facet should contain the same vertices as facet1: vertices[1], vertices[2], vertices[3]
        let facet2 = Facet::new(cell2, reversed_vertices[3]).unwrap();

        // Both facets should have the same vertices (just in different order), so same key
        assert_eq!(
            facet1.key(),
            facet2.key(),
            "Keys should be consistent for facets with same vertices regardless of vertex order"
        );

        // Create a different facet from the original cell with vertices[1] as opposite
        // This facet contains vertices[0], vertices[2], vertices[3] - different from facet1
        let facet3 = Facet::new(cell, vertices[1]).unwrap();

        // This should have a different key since it has different vertices
        assert_ne!(
            facet1.key(),
            facet3.key(),
            "Keys should be different for facets with different vertices"
        );
    }

    #[test]
    fn facet_vertices_empty_cell() {
        // This tests the edge case of a facet with a minimal cell
        // We'll use a 1D cell (2 vertices) to test filtering behavior
        let vertex1 = vertex!([0.0]);
        let vertex2 = vertex!([1.0]);
        let minimal_cell: Cell<f64, Option<()>, Option<()>, 1> = cell!(vec![vertex1, vertex2]);

        // Create facet with vertex1 as opposite - should have only vertex2 in facet
        let facet = Facet::new(minimal_cell, vertex1).unwrap();

        let vertices = facet.vertices();
        assert_eq!(vertices.len(), 1);
        assert_eq!(vertices[0], vertex2);

        // Test the opposite case - vertex2 as opposite should have only vertex1 in facet
        let minimal_cell2: Cell<f64, Option<()>, Option<()>, 1> = cell!(vec![vertex1, vertex2]);
        let facet2 = Facet::new(minimal_cell2, vertex2).unwrap();
        let vertices2 = facet2.vertices();
        assert_eq!(vertices2.len(), 1);
        assert_eq!(vertices2[0], vertex1);
    }

    #[test]
    fn facet_vertices_ordering() {
        // Test that vertices are returned in the same order as in the cell
        let vertex1 = vertex!([0.0, 0.0, 0.0]);
        let vertex2 = vertex!([1.0, 0.0, 0.0]);
        let vertex3 = vertex!([0.0, 1.0, 0.0]);
        let vertex4 = vertex!([0.0, 0.0, 1.0]);

        // Create 3D cell with exactly 4 vertices (3+1)
        let cell: Cell<f64, Option<()>, Option<()>, 3> =
            cell!(vec![vertex1, vertex2, vertex3, vertex4]);

        let facet = Facet::new(cell, vertex3).unwrap();
        let vertices = facet.vertices();

        // Should have all vertices except vertex3
        assert_eq!(vertices.len(), 3);
        assert!(vertices.contains(&vertex1));
        assert!(vertices.contains(&vertex2));
        assert!(vertices.contains(&vertex4));
        assert!(!vertices.contains(&vertex3));

        // Check ordering is preserved (vertices should appear in same order as in cell)
        assert_eq!(vertices[0], vertex1);
        assert_eq!(vertices[1], vertex2);
        assert_eq!(vertices[2], vertex4);
    }

    #[test]
    fn facet_to_and_from_json() {
        let vertex1: Vertex<f32, u8, 2> = vertex!([0.0f32, 0.0f32], 1u8);
        let vertex2: Vertex<f32, u8, 2> = vertex!([1.0f32, 0.0f32], 2u8);
        let vertex3: Vertex<f32, u8, 2> = vertex!([0.5f32, 1.0f32], 3u8);

        let cell: Cell<f32, u8, u16, 2> = cell!(vec![vertex1, vertex2, vertex3], 100u16);

        let facet = Facet::new(cell, vertex1).unwrap();

        // Test serialization
        let serialized = serde_json::to_string(&facet).unwrap();
        assert!(serialized.contains("1.0"));
        assert!(serialized.contains("0.5"));

        // Test deserialization using manual Deserialize implementation
        let deserialized: Facet<f32, u8, u16, 2> = serde_json::from_str(&serialized).unwrap();

        // Verify the deserialized facet has the same properties
        assert_eq!(facet.cell().uuid(), deserialized.cell().uuid());
        assert_eq!(facet.vertex().uuid(), deserialized.vertex().uuid());
        assert_eq!(
            facet.cell().vertices().len(),
            deserialized.cell().vertices().len()
        );
        assert_eq!(facet.cell().data, deserialized.cell().data);
        assert_eq!(facet.vertex().data, deserialized.vertex().data);

        // Verify vertex coordinates using approximate equality for floats
        let original_coords: [f32; 2] = facet.vertex().into();
        let deserialized_coords: [f32; 2] = deserialized.vertex().into();
        assert_relative_eq!(
            original_coords.as_slice(),
            deserialized_coords.as_slice(),
            epsilon = f32::EPSILON
        );

        // Verify cell vertices coordinates and data
        for (orig_v, deserialized_v) in facet
            .cell()
            .vertices()
            .iter()
            .zip(deserialized.cell().vertices().iter())
        {
            let orig_coords: [f32; 2] = orig_v.into();
            let deserialized_coords: [f32; 2] = deserialized_v.into();
            assert_relative_eq!(
                orig_coords.as_slice(),
                deserialized_coords.as_slice(),
                epsilon = f32::EPSILON
            );
            assert_eq!(orig_v.data, deserialized_v.data);
            assert_eq!(orig_v.uuid(), deserialized_v.uuid());
        }

        // Human readable output for cargo test -- --nocapture
        println!("Original facet: {facet:?}");
        println!("Serialized: {serialized}");
        println!("Deserialized facet: {deserialized:?}");
    }

    #[test]
    fn facet_eq_different_vertices() {
        let vertex1 = vertex!([0.0, 0.0, 0.0]);
        let vertex2 = vertex!([1.0, 0.0, 0.0]);
        let vertex3 = vertex!([0.0, 1.0, 0.0]);
        let vertex4 = vertex!([0.0, 0.0, 1.0]);

        let cell: Cell<f64, Option<()>, Option<()>, 3> =
            cell!(vec![vertex1, vertex2, vertex3, vertex4]);

        let facet1 = Facet::new(cell.clone(), vertex1).unwrap();
        let facet2 = Facet::new(cell.clone(), vertex2).unwrap();
        let facet3 = Facet::new(cell.clone(), vertex3).unwrap();
        let facet4 = Facet::new(cell, vertex4).unwrap();

        // All facets should be different because they have different opposite vertices
        assert_ne!(facet1, facet2);
        assert_ne!(facet1, facet3);
        assert_ne!(facet1, facet4);
        assert_ne!(facet2, facet3);
        assert_ne!(facet2, facet4);
        assert_ne!(facet3, facet4);
    }

    #[test]
    fn facet_hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Helper function to get hash value
        fn get_hash<T: Hash>(value: &T) -> u64 {
            let mut hasher = DefaultHasher::new();
            value.hash(&mut hasher);
            hasher.finish()
        }

        // Create a cell with some vertices
        let vertex1 = vertex!([0.0, 0.0, 0.0]);
        let vertex2 = vertex!([1.0, 0.0, 0.0]);
        let vertex3 = vertex!([0.0, 1.0, 0.0]);
        let vertex4 = vertex!([0.0, 0.0, 1.0]);

        let cell: Cell<f64, Option<()>, Option<()>, 3> =
            cell!(vec![vertex1, vertex2, vertex3, vertex4]);

        // Create two facets that should be equal and hash to the same value
        let facet1 = Facet::new(cell.clone(), vertex1).unwrap();
        let facet2 = Facet::new(cell.clone(), vertex1).unwrap();

        // Create a different facet that should hash to a different value
        let facet3 = Facet::new(cell, vertex2).unwrap();

        // Test that equal facets hash to the same value
        assert_eq!(get_hash(&facet1), get_hash(&facet2));

        // Test that different facets hash to different values
        assert_ne!(get_hash(&facet1), get_hash(&facet3));
    }

    #[test]
    fn facet_missing_vertex_field_error() {
        // Test deserialization with missing "vertex" field
        // Construct JSON with only the "cell" field
        let json_with_only_cell = r#"{
            "cell": {
                "vertices": [
                    {
                        "point": [0.0, 0.0, 0.0],
                        "uuid": "550e8400-e29b-41d4-a716-446655440000",
                        "data": null
                    },
                    {
                        "point": [1.0, 0.0, 0.0],
                        "uuid": "550e8400-e29b-41d4-a716-446655440001",
                        "data": null
                    },
                    {
                        "point": [0.0, 1.0, 0.0],  
                        "uuid": "550e8400-e29b-41d4-a716-446655440002",
                        "data": null
                    }
                ],
                "uuid": "550e8400-e29b-41d4-a716-446655440003",
                "neighbors": null,
                "data": null
            }
        }"#;

        // Attempt to deserialize - this should fail with missing field "vertex" error
        let result: Result<Facet<f64, Option<()>, Option<()>, 3>, _> =
            serde_json::from_str(json_with_only_cell);

        assert!(result.is_err());
        let error_message = result.unwrap_err().to_string();

        // Assert the error message contains `missing field "vertex"`
        assert!(error_message.contains("missing field"));
        assert!(error_message.contains("vertex"));

        println!("✓ Correctly detected missing vertex field: {error_message}");
    }

    // =============================================================================
    // FACET KEY GENERATION TESTS
    // =============================================================================

    #[test]
    fn test_facet_key_from_vertex_keys() {
        // Create a temporary SlotMap to generate valid VertexKeys
        use slotmap::SlotMap;
        let mut temp_vertices: SlotMap<VertexKey, ()> = SlotMap::with_key();
        let vertex_keys = vec![
            temp_vertices.insert(()),
            temp_vertices.insert(()),
            temp_vertices.insert(()),
        ];
        let key1 = facet_key_from_vertex_keys(&vertex_keys);

        let mut reversed_keys = vertex_keys;
        reversed_keys.reverse();
        let key2 = facet_key_from_vertex_keys(&reversed_keys);

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
        let key3 = facet_key_from_vertex_keys(&different_keys);

        assert_ne!(
            key1, key3,
            "Different vertices should produce different keys"
        );

        // Test empty case
        let empty_keys: Vec<VertexKey> = vec![];
        let key_empty = facet_key_from_vertex_keys(&empty_keys);
        assert_eq!(key_empty, 0, "Empty vertex keys should produce key 0");
    }
}
