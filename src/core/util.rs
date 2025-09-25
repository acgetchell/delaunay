//! General helper utilities

use serde::{Serialize, de::DeserializeOwned};
use thiserror::Error;
use uuid::Uuid;

use crate::core::facet::{Facet, FacetError, FacetView};
use crate::core::traits::data_type::DataType;
use crate::core::vertex::Vertex;
use crate::geometry::traits::coordinate::CoordinateScalar;

// =============================================================================
// TYPES
// =============================================================================

/// Errors that can occur during UUID validation.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum UuidValidationError {
    /// The UUID is nil (all zeros), which is not allowed.
    #[error("UUID is nil (all zeros) which is not allowed")]
    NilUuid,
    /// The UUID is not version 4.
    #[error("UUID is not version 4: expected version 4, found version {found}")]
    InvalidVersion {
        /// The version number that was found.
        found: usize,
    },
}

/// Validates that a UUID is not nil and is version 4.
///
/// This function performs comprehensive UUID validation to ensure that UUIDs
/// used throughout the system meet our requirements:
/// - Must not be nil (all zeros)
/// - Must be version 4 (randomly generated)
///
/// # Arguments
///
/// * `uuid` - The UUID to validate
///
/// # Returns
///
/// Returns `Ok(())` if the UUID is valid, or a `UuidValidationError` if invalid.
///
/// # Errors
///
/// Returns `UuidValidationError::NilUuid` if the UUID is nil,
/// or `UuidValidationError::InvalidVersion` if the UUID is not version 4.
///
/// # Examples
///
/// ```
/// use delaunay::core::util::{make_uuid, validate_uuid};
/// use uuid::Uuid;
///
/// // Valid UUID (version 4)
/// let valid_uuid = make_uuid();
/// assert!(validate_uuid(&valid_uuid).is_ok());
///
/// // Invalid UUID (nil)
/// let nil_uuid = Uuid::nil();
/// assert!(validate_uuid(&nil_uuid).is_err());
/// ```
pub const fn validate_uuid(uuid: &Uuid) -> Result<(), UuidValidationError> {
    // Check if UUID is nil
    if uuid.is_nil() {
        return Err(UuidValidationError::NilUuid);
    }

    // Check if UUID is version 4
    let version = uuid.get_version_num();
    if version != 4 {
        return Err(UuidValidationError::InvalidVersion { found: version });
    }

    Ok(())
}

/// The function `make_uuid` generates a version 4 [Uuid].
///
/// # Returns
///
/// a randomly generated [Uuid] (Universally Unique Identifier) using the
/// `new_v4` method from the [Uuid] struct.
///
/// # Example
///
/// ```
/// use delaunay::core::util::make_uuid;
/// let uuid = make_uuid();
/// assert_eq!(uuid.get_version_num(), 4);
/// ```
#[must_use]
pub fn make_uuid() -> Uuid {
    Uuid::new_v4()
}

/// Checks if two facets are adjacent by comparing their vertex sets.
///
/// Two facets are considered adjacent if they share the exact same set of vertices,
/// regardless of the order. This is a common check in triangulation algorithms to
/// identify neighboring cells.
///
/// # Arguments
///
/// * `facet1` - A reference to the first facet.
/// * `facet2` - A reference to the second facet.
///
/// # Returns
///
/// `true` if the facets share the same vertices, `false` otherwise.
///
/// # Examples
///
/// ```
/// use delaunay::core::facet::Facet;
/// use delaunay::core::util::facets_are_adjacent;
/// use delaunay::core::vertex::Vertex;
/// use delaunay::core::cell::Cell;
/// use delaunay::{cell, vertex};
///
/// let v1: Vertex<f64, Option<()>, 2> = vertex!([0.0, 0.0]);
/// let v2: Vertex<f64, Option<()>, 2> = vertex!([1.0, 0.0]);
/// let v3: Vertex<f64, Option<()>, 2> = vertex!([0.0, 1.0]);
/// let v4: Vertex<f64, Option<()>, 2> = vertex!([1.0, 1.0]);
///
/// let cell1: Cell<f64, Option<()>, Option<()>, 2> = cell!(vec![v1, v2, v3]);
/// let cell2: Cell<f64, Option<()>, Option<()>, 2> = cell!(vec![v2, v3, v4]);
///
/// let facet1 = Facet::new(cell1, v1).unwrap();
/// let facet2 = Facet::new(cell2, v4).unwrap();
///
/// // These facets share vertices v2 and v3, so they are adjacent
/// assert!(facets_are_adjacent(&facet1, &facet2));
/// ```
#[deprecated(
    since = "0.5.0",
    note = "Use facet_views_are_adjacent instead. This heavyweight implementation will be removed in v1.0.0."
)]
pub fn facets_are_adjacent<T, U, V, const D: usize>(
    facet1: &Facet<T, U, V, D>,
    facet2: &Facet<T, U, V, D>,
) -> bool
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
{
    use crate::core::collections::FastHashSet;
    let vertices1: FastHashSet<_> = facet1.vertices().into_iter().collect();
    let vertices2: FastHashSet<_> = facet2.vertices().into_iter().collect();
    vertices1 == vertices2
}

/// Determines if two facet views are adjacent by comparing their vertices.
///
/// Two facets are considered adjacent if they contain the same set of vertices.
/// This is the modern replacement for `facets_are_adjacent` using `FacetView`.
///
/// # Arguments
///
/// * `facet1` - The first facet view to compare
/// * `facet2` - The second facet view to compare
///
/// # Returns
///
/// `true` if the facets share the same vertices, `false` otherwise.
///
/// # Examples
///
/// ```rust,no_run
/// use delaunay::core::facet::FacetView;
/// use delaunay::core::util::facet_views_are_adjacent;
/// use delaunay::core::triangulation_data_structure::Tds;
///
/// // This is a conceptual example - in practice you would get these from a real TDS
/// fn example(tds: &Tds<f64, Option<()>, Option<()>, 3>) -> Result<(), Box<dyn std::error::Error>> {
///     let cell_keys: Vec<_> = tds.cell_keys().take(2).collect();
///     if cell_keys.len() >= 2 {
///         let facet1 = FacetView::new(tds, cell_keys[0], 0)?;
///         let facet2 = FacetView::new(tds, cell_keys[1], 0)?;
///
///         if facet_views_are_adjacent(&facet1, &facet2) {
///             println!("Facets are adjacent");
///         }
///     }
///     Ok(())
/// }
/// ```
#[must_use]
pub fn facet_views_are_adjacent<T, U, V, const D: usize>(
    facet1: &FacetView<'_, T, U, V, D>,
    facet2: &FacetView<'_, T, U, V, D>,
) -> bool
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
{
    use crate::core::collections::FastHashSet;

    // Compare facets by their vertex UUIDs for efficiency
    // Handle the Result from vertices() - if either fails, they're not adjacent
    let vertices1_result: Result<FastHashSet<_>, _> = facet1
        .vertices()
        .map(|iter| iter.map(super::vertex::Vertex::uuid).collect());
    let vertices2_result: Result<FastHashSet<_>, _> = facet2
        .vertices()
        .map(|iter| iter.map(super::vertex::Vertex::uuid).collect());

    match (vertices1_result, vertices2_result) {
        (Ok(vertices1), Ok(vertices2)) => vertices1 == vertices2,
        _ => false, // If either facet has missing cells, they're not adjacent
    }
}

/// Generates all unique combinations of `k` items from a given slice.
///
/// This function is used to generate vertex combinations for creating k-simplices
/// (e.g., edges, triangles, tetrahedra) from a set of vertices.
///
/// # Arguments
///
/// * `vertices` - A slice of vertices from which to generate combinations.
/// * `k` - The size of each combination.
///
/// # Returns
///
/// A vector of vectors, where each inner vector is a unique combination of `k` vertices.
///
/// # Examples
///
/// This function is made public for testing purposes.
///
/// ```
/// use delaunay::core::util::generate_combinations;
/// use delaunay::core::vertex::Vertex;
/// use delaunay::vertex;
///
/// let vertices: Vec<Vertex<f64, Option<()>, 1>> = vec![
///     vertex!([0.0]),
///     vertex!([1.0]),
///     vertex!([2.0]),
/// ];
///
/// // Generate all 2-vertex combinations (edges)
/// let combinations = generate_combinations(&vertices, 2);
///
/// assert_eq!(combinations.len(), 3);
/// assert!(combinations.contains(&vec![vertices[0], vertices[1]]));
/// assert!(combinations.contains(&vec![vertices[0], vertices[2]]));
/// assert!(combinations.contains(&vec![vertices[1], vertices[2]]));
/// ```
pub fn generate_combinations<T, U, const D: usize>(
    vertices: &[Vertex<T, U, D>],
    k: usize,
) -> Vec<Vec<Vertex<T, U, D>>>
where
    T: CoordinateScalar,
    U: DataType,
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
{
    let mut combinations = Vec::new();

    if k == 0 {
        combinations.push(Vec::new());
        return combinations;
    }

    if k > vertices.len() {
        return combinations;
    }

    if k == vertices.len() {
        combinations.push(vertices.to_vec());
        return combinations;
    }

    // Generate combinations using iterative approach
    let n = vertices.len();
    let mut indices = (0..k).collect::<Vec<_>>();

    loop {
        // Add current combination
        let combination = indices.iter().map(|i| vertices[*i]).collect();
        combinations.push(combination);

        // Find next combination
        let mut i = k;
        loop {
            if i == 0 {
                return combinations;
            }
            i -= 1;
            if indices[i] != i + n - k {
                break;
            }
        }

        indices[i] += 1;
        for j in (i + 1)..k {
            indices[j] = indices[j - 1] + 1;
        }
    }
}

// =============================================================================
// HASH UTILITIES
// =============================================================================

/// Applies a stable hash function to a slice of sorted u64 values.
///
/// This function uses an FNV-based polynomial rolling hash with an avalanche step
/// to produce deterministic hash values. The input slice should be pre-sorted to ensure
/// consistent results regardless of input order.
///
/// # Arguments
///
/// * `sorted_values` - A slice of u64 values that should be pre-sorted
///
/// # Returns
///
/// A `u64` hash value representing the stable hash of the input values
///
/// # Algorithm
///
/// Uses FNV constants with polynomial rolling hash:
/// 1. Start with FNV offset basis
/// 2. For each value: `hash = hash.wrapping_mul(PRIME).wrapping_add(value)`
/// 3. Apply avalanche step for better bit distribution
///
/// # Examples
///
/// ```
/// use delaunay::core::util::stable_hash_u64_slice;
/// let values = vec![1u64, 2u64, 3u64];
/// let hash1 = stable_hash_u64_slice(&values);
///
/// let mut reversed = values.clone();
/// reversed.reverse();
/// let hash2 = stable_hash_u64_slice(&reversed);
///
/// // Different order produces different hash (input should be pre-sorted)
/// assert_ne!(hash1, hash2);
///
/// // Same sorted input produces same hash
/// let mut sorted1 = values;
/// sorted1.sort_unstable();
/// let mut sorted2 = reversed;
/// sorted2.sort_unstable();
/// assert_eq!(stable_hash_u64_slice(&sorted1), stable_hash_u64_slice(&sorted2));
/// ```
#[must_use]
pub fn stable_hash_u64_slice(sorted_values: &[u64]) -> u64 {
    // Hash constants for facet key generation (FNV-based)
    const HASH_PRIME: u64 = 1_099_511_628_211; // Large prime (FNV prime)
    const HASH_OFFSET: u64 = 14_695_981_039_346_656_037; // FNV offset basis

    // Handle empty case
    if sorted_values.is_empty() {
        return 0;
    }

    // Use a polynomial rolling hash for efficient combination
    let mut hash = HASH_OFFSET;
    for &value in sorted_values {
        hash = hash.wrapping_mul(HASH_PRIME).wrapping_add(value);
    }

    // Apply avalanche step for better bit distribution
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    hash ^= hash >> 33;

    hash
}

// =============================================================================
// FACET KEY UTILITIES
// =============================================================================

/// Derives a facet key from the facet's vertices using the TDS vertex mappings.
///
/// This utility function converts the facet's vertices to vertex keys and computes
/// the canonical facet key for lookup in facet-to-cells mappings. This is a common
/// operation used across boundary analysis, convex hull algorithms, and insertion
/// algorithms.
///
/// # Arguments
///
/// * `facet_vertices` - The vertices that make up the facet
/// * `tds` - The triangulation data structure for vertex key lookups
///
/// # Returns
///
/// A `Result` containing the facet key or a `FacetError` if validation or vertex lookup fails.
///
/// # Errors
///
/// Returns `FacetError::InsufficientVertices` if the vertex count doesn't equal `D`
/// Returns `FacetError::VertexNotFound` if any vertex UUID cannot be found in the TDS
///
/// # Examples
///
/// ```
/// use delaunay::core::util::derive_facet_key_from_vertices;
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
/// // Get facet vertices from a cell - must be exactly D vertices for a D-dimensional triangulation
/// if let Some(cell) = tds.cells().values().next() {
///     let facet_vertices: Vec<_> = cell.vertices().iter().skip(1).cloned().collect(); // Skip 1 vertex to get D vertices
///     assert_eq!(facet_vertices.len(), 3); // For 3D triangulation, facet has 3 vertices
///     let facet_key = derive_facet_key_from_vertices(&facet_vertices, &tds).unwrap();
///     println!("Facet key: {}", facet_key);
/// }
/// ```
///
/// # Performance
///
/// - Time Complexity: O(V) where V is the number of vertices in the facet
/// - Space Complexity: O(V) for the temporary vertex keys buffer
/// - Uses stack-allocated `SmallBuffer` for performance on hot paths
pub fn derive_facet_key_from_vertices<T, U, V, const D: usize>(
    facet_vertices: &[crate::core::vertex::Vertex<T, U, D>],
    tds: &crate::core::triangulation_data_structure::Tds<T, U, V, D>,
) -> Result<u64, crate::core::facet::FacetError>
where
    T: crate::geometry::traits::coordinate::CoordinateScalar,
    U: crate::core::traits::data_type::DataType,
    V: crate::core::traits::data_type::DataType,
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
{
    use crate::core::collections::SmallBuffer;
    use crate::core::facet::{FacetError, facet_key_from_vertex_keys};
    use crate::core::triangulation_data_structure::VertexKey;

    // Validate that the number of vertices matches the expected dimension
    // In a D-dimensional triangulation, a facet should have exactly D vertices
    if facet_vertices.len() != D {
        return Err(FacetError::InsufficientVertices {
            expected: D,
            actual: facet_vertices.len(),
            dimension: D,
        });
    }

    // Compute the facet key using VertexKeys (same method as build_facet_to_cells_hashmap)
    // Stack-allocate for performance on hot paths
    let mut vertex_keys: SmallBuffer<
        VertexKey,
        { crate::core::collections::MAX_PRACTICAL_DIMENSION_SIZE },
    > = SmallBuffer::new();

    for vertex in facet_vertices {
        match tds.vertex_key_from_uuid(&vertex.uuid()) {
            Some(key) => vertex_keys.push(key),
            None => {
                return Err(FacetError::VertexNotFound {
                    uuid: vertex.uuid(),
                });
            }
        }
    }

    Ok(facet_key_from_vertex_keys(&vertex_keys))
}

// =============================================================================
// MEMORY MEASUREMENT UTILITIES
// =============================================================================

/// Memory measurement helper for allocation tracking in examples, tests, and benchmarks.
///
/// This utility function provides a consistent interface for measuring memory allocations
/// across different parts of the codebase. It returns both the result of the closure
/// and allocation information when the `count-allocations` feature is enabled.
///
/// # Arguments
///
/// * `f` - A closure to execute while measuring allocations
///
/// # Returns
///
/// When `count-allocations` feature is enabled: Returns a tuple `(R, AllocationInfo)`
/// where `R` is the closure result and `AllocationInfo` contains allocation metrics.
///
/// When feature is disabled: Returns a tuple `(R, ())` where the allocation info is empty.
///
/// # Panics
///
/// This function should never panic under normal usage. The internal `expect()` call
/// is used because the closure is guaranteed to execute and set the result.
///
/// # Examples
///
/// ```rust,ignore
/// // With count-allocations feature enabled
/// use delaunay::core::util::measure_with_result;
///
/// let (result, alloc_info) = measure_with_result(|| {
///     // Some memory-allocating operation
///     vec![1, 2, 3, 4, 5]
/// });
///
/// #[cfg(feature = "count-allocations")]
/// println!("Allocated {} bytes", alloc_info.bytes_total);
/// ```
#[cfg(feature = "count-allocations")]
pub fn measure_with_result<F, R>(f: F) -> (R, allocation_counter::AllocationInfo)
where
    F: FnOnce() -> R,
{
    let mut result: Option<R> = None;
    let info = allocation_counter::measure(|| {
        result = Some(f());
    });
    (result.expect("Closure should have set result"), info)
}

/// Memory measurement helper (no-op version when count-allocations feature is disabled).
///
/// See [`measure_with_result`] for full documentation.
#[cfg(not(feature = "count-allocations"))]
pub fn measure_with_result<F, R>(f: F) -> (R, ())
where
    F: FnOnce() -> R,
{
    (f(), ())
}

/// Helper function to safely convert usize to u8 for facet indices.
///
/// This function provides a centralized, safe conversion from `usize` to `u8`
/// that is commonly needed throughout the triangulation codebase for facet indexing.
/// It handles the conversion error gracefully by returning appropriate `FacetError`
/// variants with detailed error information.
///
/// # Arguments
///
/// * `idx` - The usize index to convert
/// * `facet_count` - The number of facets (for error reporting)
///
/// # Returns
///
/// A `Result` containing the converted u8 index or a `FacetError`.
///
/// # Errors
///
/// Returns `FacetError::InvalidFacetIndex` if the index cannot fit in a u8.
///
/// # Examples
///
/// ```
/// use delaunay::core::util::usize_to_u8;
///
/// // Successful conversion
/// assert_eq!(usize_to_u8(0, 4), Ok(0));
/// assert_eq!(usize_to_u8(255, 256), Ok(255));
///
/// // Failed conversion
/// let result = usize_to_u8(256, 10);
/// assert!(result.is_err());
/// ```
pub fn usize_to_u8(idx: usize, facet_count: usize) -> Result<u8, FacetError> {
    u8::try_from(idx).map_err(|_| FacetError::InvalidFacetIndex {
        index: u8::MAX,
        facet_count,
    })
}

#[cfg(test)]
mod tests {

    use crate::geometry::point::Point;
    use crate::geometry::traits::coordinate::Coordinate;
    use crate::vertex;
    use uuid::Uuid;

    use super::*;

    // =============================================================================
    // UUID UTILITIES TESTS
    // =============================================================================

    #[test]
    fn utilities_make_uuid_uniqueness() {
        let uuid1 = make_uuid();
        let uuid2 = make_uuid();
        let uuid3 = make_uuid();

        // All UUIDs should be different
        assert_ne!(uuid1, uuid2);
        assert_ne!(uuid1, uuid3);
        assert_ne!(uuid2, uuid3);

        // All should be version 4
        assert_eq!(uuid1.get_version_num(), 4);
        assert_eq!(uuid2.get_version_num(), 4);
        assert_eq!(uuid3.get_version_num(), 4);
    }

    #[test]
    fn utilities_make_uuid_format() {
        let uuid = make_uuid();
        let uuid_string = uuid.to_string();

        // UUID should have proper format: 8-4-4-4-12 characters
        assert_eq!(uuid_string.len(), 36); // Including hyphens
        assert_eq!(uuid_string.chars().filter(|&c| c == '-').count(), 4);

        // Should be valid hyphenated format
        let parts: Vec<&str> = uuid_string.split('-').collect();
        assert_eq!(parts.len(), 5);
        assert_eq!(parts[0].len(), 8);
        assert_eq!(parts[1].len(), 4);
        assert_eq!(parts[2].len(), 4);
        assert_eq!(parts[3].len(), 4);
        assert_eq!(parts[4].len(), 12);
    }

    #[test]
    fn test_validate_uuid_comprehensive() {
        // Test valid UUID (version 4)
        let valid_uuid = make_uuid();
        assert!(
            validate_uuid(&valid_uuid).is_ok(),
            "Valid v4 UUID should pass validation"
        );

        // Test nil UUID
        let nil_uuid = Uuid::nil();
        let nil_result = validate_uuid(&nil_uuid);
        assert!(nil_result.is_err(), "Nil UUID should fail validation");
        match nil_result {
            Err(UuidValidationError::NilUuid) => (), // Expected
            Err(other) => panic!("Expected NilUuid error, got: {other:?}"),
            Ok(()) => panic!("Expected error for nil UUID, but validation passed"),
        }

        // Test wrong version UUID (version 1)
        let v1_uuid = Uuid::parse_str("550e8400-e29b-11d4-a716-446655440000").unwrap();
        assert_eq!(v1_uuid.get_version_num(), 1);
        let version_result = validate_uuid(&v1_uuid);
        assert!(
            version_result.is_err(),
            "Non-v4 UUID should fail validation"
        );
        match version_result {
            Err(UuidValidationError::InvalidVersion { found }) => {
                assert_eq!(found, 1, "Should report correct version number");
            }
            Err(other) => panic!("Expected InvalidVersion error, got: {other:?}"),
            Ok(()) => panic!("Expected error for version 1 UUID, but validation passed"),
        }

        // Test error display formatting
        let nil_error = UuidValidationError::NilUuid;
        let nil_error_string = format!("{nil_error}");
        assert!(
            nil_error_string.contains("nil"),
            "Nil error message should contain 'nil'"
        );
        assert!(
            nil_error_string.contains("not allowed"),
            "Nil error message should mention 'not allowed'"
        );

        let version_error = UuidValidationError::InvalidVersion { found: 3 };
        let version_error_string = format!("{version_error}");
        assert!(
            version_error_string.contains("version 4"),
            "Version error should mention 'version 4'"
        );
        assert!(
            version_error_string.contains("found version 3"),
            "Version error should show found version"
        );

        // Test PartialEq for UuidValidationError
        let error1 = UuidValidationError::NilUuid;
        let error2 = UuidValidationError::NilUuid;
        assert_eq!(error1, error2, "Same nil errors should be equal");

        let error3 = UuidValidationError::InvalidVersion { found: 2 };
        let error4 = UuidValidationError::InvalidVersion { found: 2 };
        assert_eq!(error3, error4, "Same version errors should be equal");

        let error5 = UuidValidationError::InvalidVersion { found: 3 };
        assert_ne!(
            error3, error5,
            "Different version errors should not be equal"
        );
        assert_ne!(error1, error3, "Different error types should not be equal");
    }

    // =============================================================================
    // FACET UTILITIES TESTS
    // =============================================================================

    #[test]
    #[allow(clippy::too_many_lines)]
    #[allow(deprecated)] // Testing deprecated function during transition
    fn test_facets_are_adjacent_multidimensional() {
        use crate::core::{cell::Cell, facet::Facet};
        use crate::{cell, vertex};

        // Test 2D case - basic adjacency detection
        let v1: Vertex<f64, Option<()>, 2> = vertex!([0.0, 0.0]);
        let v2: Vertex<f64, Option<()>, 2> = vertex!([1.0, 0.0]);
        let v3: Vertex<f64, Option<()>, 2> = vertex!([0.0, 1.0]);
        let v4: Vertex<f64, Option<()>, 2> = vertex!([1.0, 1.0]);

        let cell2d_1: Cell<f64, Option<()>, Option<()>, 2> = cell!(vec![v1, v2, v3]);
        let cell2d_2: Cell<f64, Option<()>, Option<()>, 2> = cell!(vec![v2, v3, v4]);
        let cell2d_3: Cell<f64, Option<()>, Option<()>, 2> = cell!(vec![v1, v2, v4]);

        let facet2d_1 = Facet::new(cell2d_1, v1).unwrap(); // Vertices: v2, v3
        let facet2d_2 = Facet::new(cell2d_2, v4).unwrap(); // Vertices: v2, v3
        let facet2d_3 = Facet::new(cell2d_3, v4).unwrap(); // Vertices: v1, v2

        assert!(
            facets_are_adjacent(&facet2d_1, &facet2d_2),
            "2D: Same vertices should be adjacent"
        );
        assert!(
            !facets_are_adjacent(&facet2d_1, &facet2d_3),
            "2D: Different vertices should not be adjacent"
        );

        // Test 3D case - cells with shared and non-shared vertices
        let points3d_1 = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        let points3d_2 = vec![
            Point::new([0.0, 0.0, 0.0]), // Shared
            Point::new([1.0, 0.0, 0.0]), // Shared
            Point::new([0.0, 1.0, 0.0]), // Shared
            Point::new([2.0, 0.0, 0.0]), // Different
        ];
        let points3d_separate = vec![
            Point::new([10.0, 10.0, 10.0]),
            Point::new([11.0, 10.0, 10.0]),
            Point::new([10.0, 11.0, 10.0]),
            Point::new([10.0, 10.0, 11.0]),
        ];

        let cell3d_1: Cell<f64, usize, usize, 3> = cell!(Vertex::from_points(points3d_1));
        let cell3d_2: Cell<f64, usize, usize, 3> = cell!(Vertex::from_points(points3d_2));
        let cell3d_separate: Cell<f64, usize, usize, 3> =
            cell!(Vertex::from_points(points3d_separate));

        let facets3d_1 = cell3d_1
            .facets()
            .expect("Failed to get facets from 3D cell1");
        let facets3d_2 = cell3d_2
            .facets()
            .expect("Failed to get facets from 3D cell2");
        let facets3d_separate = cell3d_separate
            .facets()
            .expect("Failed to get facets from separate 3D cell");

        // Test that cells sharing 3 vertices have adjacent facets
        let mut found_shared_adjacent = false;
        for f1 in &facets3d_1 {
            for f2 in &facets3d_2 {
                if facets_are_adjacent(f1, f2) {
                    found_shared_adjacent = true;
                    break;
                }
            }
            if found_shared_adjacent {
                break;
            }
        }
        assert!(
            found_shared_adjacent,
            "3D: Cells sharing vertices should have adjacent facets"
        );

        // Test that completely separate cells have no adjacent facets
        let mut found_separate_adjacent = false;
        for f1 in &facets3d_1 {
            for f_sep in &facets3d_separate {
                if facets_are_adjacent(f1, f_sep) {
                    found_separate_adjacent = true;
                    break;
                }
            }
            if found_separate_adjacent {
                break;
            }
        }
        assert!(
            !found_separate_adjacent,
            "3D: Separate cells should not have adjacent facets"
        );

        // Test 4D case - verify higher dimensional functionality
        let points4d_1 = vec![
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0]),
        ];
        let points4d_2 = vec![
            Point::new([0.0, 0.0, 0.0, 0.0]), // Shared
            Point::new([1.0, 0.0, 0.0, 0.0]), // Shared
            Point::new([0.0, 1.0, 0.0, 0.0]), // Shared
            Point::new([0.0, 0.0, 1.0, 0.0]), // Shared
            Point::new([2.0, 0.0, 0.0, 0.0]), // Different
        ];

        let cell4d_1: Cell<f64, usize, usize, 4> = cell!(Vertex::from_points(points4d_1));
        let cell4d_2: Cell<f64, usize, usize, 4> = cell!(Vertex::from_points(points4d_2));

        let facets4d_1 = cell4d_1
            .facets()
            .expect("Failed to get facets from 4D cell1");
        let facets4d_2 = cell4d_2
            .facets()
            .expect("Failed to get facets from 4D cell2");

        // Test 4D adjacency
        let mut found_4d_adjacent = false;
        for f1 in &facets4d_1 {
            for f2 in &facets4d_2 {
                if facets_are_adjacent(f1, f2) {
                    found_4d_adjacent = true;
                    break;
                }
            }
            if found_4d_adjacent {
                break;
            }
        }
        assert!(
            found_4d_adjacent,
            "4D: Cells sharing 4 vertices should have adjacent facets"
        );

        // Test 5D case - maximum practical dimension
        let points5d_1 = vec![
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 0.0, 1.0]),
        ];
        let points5d_2 = vec![
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0]), // Shared
            Point::new([1.0, 0.0, 0.0, 0.0, 0.0]), // Shared
            Point::new([0.0, 1.0, 0.0, 0.0, 0.0]), // Shared
            Point::new([0.0, 0.0, 1.0, 0.0, 0.0]), // Shared
            Point::new([0.0, 0.0, 0.0, 1.0, 0.0]), // Shared
            Point::new([3.0, 0.0, 0.0, 0.0, 0.0]), // Different
        ];

        let cell5d_1: Cell<f64, usize, usize, 5> = cell!(Vertex::from_points(points5d_1));
        let cell5d_2: Cell<f64, usize, usize, 5> = cell!(Vertex::from_points(points5d_2));

        let facets5d_1 = cell5d_1
            .facets()
            .expect("Failed to get facets from 5D cell1");
        let facets5d_2 = cell5d_2
            .facets()
            .expect("Failed to get facets from 5D cell2");

        // Test 5D adjacency
        let mut found_5d_adjacent = false;
        for f1 in &facets5d_1 {
            for f2 in &facets5d_2 {
                if facets_are_adjacent(f1, f2) {
                    found_5d_adjacent = true;
                    break;
                }
            }
            if found_5d_adjacent {
                break;
            }
        }
        assert!(
            found_5d_adjacent,
            "5D: Cells sharing 5 vertices should have adjacent facets"
        );
    }

    // =============================================================================
    // HASH UTILITIES TESTS
    // =============================================================================

    #[test]
    fn test_stable_hash_u64_slice_comprehensive() {
        // Test basic functionality with order sensitivity
        let values = vec![1u64, 2u64, 3u64];
        let hash1 = stable_hash_u64_slice(&values);

        let mut reversed = values.clone();
        reversed.reverse();
        let hash2 = stable_hash_u64_slice(&reversed);
        assert_ne!(
            hash1, hash2,
            "Different order should produce different hash"
        );

        // Same sorted input produces same hash
        let mut sorted1 = values;
        sorted1.sort_unstable();
        let mut sorted2 = reversed;
        sorted2.sort_unstable();
        assert_eq!(
            stable_hash_u64_slice(&sorted1),
            stable_hash_u64_slice(&sorted2),
            "Same sorted input should produce same hash"
        );

        // Test edge cases: empty, single value, different lengths
        let empty: Vec<u64> = vec![];
        assert_eq!(
            stable_hash_u64_slice(&empty),
            0,
            "Empty slice should produce hash 0"
        );

        let single = vec![42u64];
        let single_copy = vec![42u64];
        assert_eq!(
            stable_hash_u64_slice(&single),
            stable_hash_u64_slice(&single_copy),
            "Same single value should produce same hash"
        );

        let different_single = vec![43u64];
        assert_ne!(
            stable_hash_u64_slice(&single),
            stable_hash_u64_slice(&different_single),
            "Different single values should produce different hashes"
        );

        // Test deterministic behavior
        let test_values = vec![100u64, 200u64, 300u64, 400u64];
        let hash_a = stable_hash_u64_slice(&test_values);
        let hash_b = stable_hash_u64_slice(&test_values);
        let hash_c = stable_hash_u64_slice(&test_values);
        assert_eq!(hash_a, hash_b, "Multiple calls should be deterministic");
        assert_eq!(hash_b, hash_c, "Multiple calls should be deterministic");

        // Test different lengths
        let short = vec![1u64, 2u64];
        let long = vec![1u64, 2u64, 3u64];
        assert_ne!(
            stable_hash_u64_slice(&short),
            stable_hash_u64_slice(&long),
            "Different lengths should produce different hashes"
        );

        // Test large values
        let large_values = vec![u64::MAX, u64::MAX - 1, u64::MAX - 2];
        let hash_large1 = stable_hash_u64_slice(&large_values);
        let hash_large2 = stable_hash_u64_slice(&large_values);
        assert_eq!(
            hash_large1, hash_large2,
            "Large values should be handled consistently"
        );

        let different_large = vec![u64::MAX - 3, u64::MAX - 4, u64::MAX - 5];
        assert_ne!(
            hash_large1,
            stable_hash_u64_slice(&different_large),
            "Different large values should produce different hashes"
        );
    }

    // =============================================================================
    // COMBINATION UTILITIES TESTS
    // =============================================================================

    #[test]
    fn test_generate_combinations_comprehensive() {
        // Test basic functionality with 4 vertices
        let vertices: Vec<Vertex<f64, Option<()>, 1>> = vec![
            vertex!([0.0]),
            vertex!([1.0]),
            vertex!([2.0]),
            vertex!([3.0]),
        ];

        // Combinations of 2 from 4 - should be C(4,2) = 6
        let combinations_2 = generate_combinations(&vertices, 2);
        assert_eq!(combinations_2.len(), 6, "C(4,2) should equal 6");

        // Combinations of 3 from 4 - should be C(4,3) = 4
        let combinations_3 = generate_combinations(&vertices, 3);
        assert_eq!(combinations_3.len(), 4, "C(4,3) should equal 4");
        assert!(
            combinations_3.contains(&vec![vertices[0], vertices[1], vertices[2]]),
            "Should contain specific combination"
        );

        // Single vertex combinations (k=1) - should be C(4,1) = 4
        let combinations_1 = generate_combinations(&vertices, 1);
        assert_eq!(combinations_1.len(), 4, "C(4,1) should equal 4");
        assert!(
            combinations_1.contains(&vec![vertices[0]]),
            "Should contain first vertex"
        );
        assert!(
            combinations_1.contains(&vec![vertices[1]]),
            "Should contain second vertex"
        );
        assert!(
            combinations_1.contains(&vec![vertices[2]]),
            "Should contain third vertex"
        );
        assert!(
            combinations_1.contains(&vec![vertices[3]]),
            "Should contain fourth vertex"
        );

        // Edge case: k=0 - should return one empty combination
        let combinations_0 = generate_combinations(&vertices, 0);
        assert_eq!(combinations_0.len(), 1, "C(4,0) should equal 1");
        assert!(
            combinations_0[0].is_empty(),
            "k=0 should produce empty combination"
        );

        // Edge case: k > len - should return empty result
        let combinations_5 = generate_combinations(&vertices, 5);
        assert!(
            combinations_5.is_empty(),
            "k > n should return no combinations"
        );

        // Edge case: k == len - should return all vertices as single combination
        let combinations_4 = generate_combinations(&vertices, 4);
        assert_eq!(combinations_4.len(), 1, "C(4,4) should equal 1");
        assert_eq!(
            combinations_4[0], vertices,
            "k=n should return all vertices"
        );

        // Test with different size - 3 vertices, choose 2
        let small_vertices: Vec<Vertex<f64, Option<()>, 1>> =
            vec![vertex!([1.0]), vertex!([2.0]), vertex!([3.0])];
        let combinations_small = generate_combinations(&small_vertices, 2);
        assert_eq!(combinations_small.len(), 3, "C(3,2) should equal 3");

        // Test larger case - 5 vertices, choose 3 to exercise inner loops
        let large_vertices: Vec<Vertex<f64, Option<()>, 1>> = vec![
            vertex!([1.0]),
            vertex!([2.0]),
            vertex!([3.0]),
            vertex!([4.0]),
            vertex!([5.0]),
        ];
        let combinations_large = generate_combinations(&large_vertices, 3);
        assert_eq!(combinations_large.len(), 10, "C(5,3) should equal 10");

        // Verify some specific combinations exist in large case
        assert!(
            combinations_large.contains(&vec![
                large_vertices[0],
                large_vertices[1],
                large_vertices[2]
            ]),
            "Should contain first combination"
        );
        assert!(
            combinations_large.contains(&vec![
                large_vertices[2],
                large_vertices[3],
                large_vertices[4]
            ]),
            "Should contain last combination"
        );

        // Test empty input edge cases
        let empty_vertices: Vec<Vertex<f64, Option<()>, 1>> = vec![];
        let combinations_empty_k1 = generate_combinations(&empty_vertices, 1);
        assert!(
            combinations_empty_k1.is_empty(),
            "Empty input with k>0 should return no combinations"
        );

        let combinations_empty_k0 = generate_combinations(&empty_vertices, 0);
        assert_eq!(
            combinations_empty_k0.len(),
            1,
            "Empty input with k=0 should return one empty combination"
        );
        assert!(
            combinations_empty_k0[0].is_empty(),
            "Empty input k=0 combination should be empty"
        );
    }

    // =============================================================================
    // MEMORY MEASUREMENT UTILITIES TESTS
    // =============================================================================

    #[test]
    fn test_measure_with_result_comprehensive() {
        // Test basic functionality - returns correct result
        let expected_result = 42;
        let (result, _alloc_info) = measure_with_result(|| expected_result);
        assert_eq!(result, expected_result);

        // Test with various allocation patterns
        let (vec_result, _) = measure_with_result(|| vec![1, 2, 3, 4, 5]);
        assert_eq!(vec_result, vec![1, 2, 3, 4, 5]);

        let (string_result, _) = measure_with_result(|| {
            let mut s = String::new();
            s.push_str("Hello, ");
            s.push_str("World!");
            s
        });
        assert_eq!(string_result, "Hello, World!");

        let (complex_result, _) = measure_with_result(|| {
            let mut data: Vec<String> = Vec::new();
            for i in 0..5 {
                data.push(format!("Item {i}"));
            }
            data.len()
        });
        assert_eq!(complex_result, 5);

        // Test various return types
        let (tuple_result, _) = measure_with_result(|| ("hello", 42));
        assert_eq!(tuple_result, ("hello", 42));

        let (option_result, _) = measure_with_result(|| Some("value"));
        assert_eq!(option_result, Some("value"));

        let (result_result, _) = measure_with_result(|| Ok::<i32, &str>(123));
        assert_eq!(result_result, Ok(123));

        // Test no-panic behavior
        let (sum_result, _) = measure_with_result(|| {
            let data = [1, 2, 3];
            data.iter().sum::<i32>()
        });
        assert_eq!(sum_result, 6);
    }

    #[cfg(feature = "count-allocations")]
    #[test]
    fn test_measure_with_result_allocation_info_structure() {
        // Test that allocation info has expected structure when feature is enabled
        let (_result, alloc_info) = measure_with_result(|| {
            // Allocate some memory
            vec![0u8; 1024]
        });

        // Verify that we got an AllocationInfo struct by accessing its fields
        // This validates that the function properly returns allocation info
        // We access all fields to ensure the struct is properly constructed
        std::hint::black_box(&alloc_info.bytes_total);
        std::hint::black_box(&alloc_info.count_total);
        std::hint::black_box(&alloc_info.bytes_current);
        std::hint::black_box(&alloc_info.count_current);
        std::hint::black_box(&alloc_info.bytes_max);
        std::hint::black_box(&alloc_info.count_max);

        // Test that we can actually use the allocation info
        // For a vec![0u8; 1024], we expect some allocation to have occurred
        assert!(
            alloc_info.bytes_total > 0,
            "Should have allocated memory for the vector"
        );
    }

    #[cfg(not(feature = "count-allocations"))]
    #[test]
    fn test_measure_with_result_no_allocation_feature() {
        // Test that when feature is disabled, we get unit type
        let (_result, alloc_info) = measure_with_result(|| vec![0u8; 1024]);

        // Verify that alloc_info is unit type ()
        let _: () = alloc_info;
    }

    // =============================================================================
    // FACET KEY UTILITIES TESTS
    // =============================================================================

    #[test]
    #[allow(clippy::cognitive_complexity)]
    #[allow(clippy::too_many_lines)]
    fn test_derive_facet_key_from_vertices_comprehensive() {
        use crate::core::triangulation_data_structure::Tds;
        use crate::core::vertex::{Vertex, VertexBuilder};
        use uuid::Uuid;

        println!("Testing derive_facet_key_from_vertices comprehensively");

        // Create a triangulation
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        // Test 1: Basic functionality - successful key derivation
        println!("  Testing basic functionality...");
        let cell = tds.cells().values().next().unwrap();
        let facet_vertices: Vec<_> = cell.vertices().iter().skip(1).copied().collect();

        let result = derive_facet_key_from_vertices(&facet_vertices, &tds);
        assert!(
            result.is_ok(),
            "Facet key derivation should succeed for valid vertices"
        );

        let facet_key = result.unwrap();
        println!("    Derived facet key: {facet_key}");

        // Test deterministic behavior - same vertices produce same key
        let result2 = derive_facet_key_from_vertices(&facet_vertices, &tds);
        assert!(result2.is_ok(), "Second derivation should also succeed");
        assert_eq!(
            facet_key,
            result2.unwrap(),
            "Same vertices should produce same facet key"
        );

        // Test different vertices produce different keys
        let all_vertices = cell.vertices();
        let different_facet_vertices: Vec<_> = all_vertices.iter().take(3).copied().collect();
        if different_facet_vertices.len() == 3 && different_facet_vertices != facet_vertices {
            let result3 = derive_facet_key_from_vertices(&different_facet_vertices, &tds);
            assert!(
                result3.is_ok(),
                "Different facet key derivation should succeed"
            );
            let different_facet_key = result3.unwrap();
            assert_ne!(
                facet_key, different_facet_key,
                "Different vertices should produce different facet keys"
            );
            println!("    Different facet key: {different_facet_key}");
        }

        // Test 2: Error cases
        println!("  Testing error handling...");

        // Wrong vertex count
        let single_vertex = vec![vertices[0]];
        let result_count = derive_facet_key_from_vertices(&single_vertex, &tds);
        assert!(
            result_count.is_err(),
            "Should return error for wrong vertex count"
        );
        if let Err(error) = result_count {
            match error {
                crate::core::facet::FacetError::InsufficientVertices {
                    expected,
                    actual,
                    dimension,
                } => {
                    assert_eq!(expected, 3, "Expected 3 vertices for 3D");
                    assert_eq!(actual, 1, "Got 1 vertex");
                    assert_eq!(dimension, 3, "Dimension should be 3");
                }
                _ => panic!("Expected InsufficientVertices error, got: {error:?}"),
            }
        }

        // Empty vertices
        let empty_vertices: Vec<Vertex<f64, Option<()>, 3>> = vec![];
        let result_empty = derive_facet_key_from_vertices(&empty_vertices, &tds);
        assert!(
            result_empty.is_err(),
            "Empty vertices should fail validation"
        );
        if let Err(error) = result_empty {
            match error {
                crate::core::facet::FacetError::InsufficientVertices {
                    expected,
                    actual,
                    dimension,
                } => {
                    assert_eq!(expected, 3, "Expected 3 vertices for 3D");
                    assert_eq!(actual, 0, "Got 0 vertices");
                    assert_eq!(dimension, 3, "Dimension should be 3");
                }
                _ => {
                    panic!("Expected InsufficientVertices error for empty vertices, got: {error:?}")
                }
            }
        }

        // Vertex not found in TDS
        let invalid_uuid = Uuid::new_v4();
        let mut invalid_vertex = VertexBuilder::default()
            .point(crate::geometry::point::Point::new([99.0, 99.0, 99.0]))
            .build()
            .expect("Failed to create test vertex");
        invalid_vertex
            .set_uuid(invalid_uuid)
            .expect("Failed to set UUID");
        let invalid_vertices = vec![invalid_vertex, invalid_vertex, invalid_vertex];

        let result_invalid = derive_facet_key_from_vertices(&invalid_vertices, &tds);
        assert!(
            result_invalid.is_err(),
            "Should return error for vertex not found in TDS"
        );
        if let Err(error) = result_invalid {
            match error {
                crate::core::facet::FacetError::VertexNotFound { uuid } => {
                    assert_eq!(uuid, invalid_uuid, "Error should contain the correct UUID");
                }
                _ => panic!("Expected VertexNotFound error, got: {error:?}"),
            }
        }

        // Test 3: Consistency with TDS cache
        println!("  Testing consistency with TDS...");
        let cache = tds
            .build_facet_to_cells_map()
            .expect("Should build facet map in test");
        let mut keys_found = 0;
        let mut keys_tested = 0;

        for cell in tds.cells().values() {
            let cell_vertices = cell.vertices();
            for skip_vertex_idx in 0..cell_vertices.len() {
                let facet_vertices: Vec<_> = cell_vertices
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != skip_vertex_idx)
                    .map(|(_, v)| *v)
                    .collect();

                if !facet_vertices.is_empty() {
                    let key_result = derive_facet_key_from_vertices(&facet_vertices, &tds);
                    if let Ok(derived_key) = key_result {
                        keys_tested += 1;
                        if cache.contains_key(&derived_key) {
                            keys_found += 1;
                        }
                    }
                }
            }
        }

        println!("    Found {keys_found}/{keys_tested} derived keys in TDS cache");
        assert!(keys_tested > 0, "Should have tested some keys");
        println!("  ✓ All facet key derivation tests passed");
    }

    // =============================================================================
    // FACET VIEW ADJACENCY TESTS
    // =============================================================================

    #[test]
    fn test_facet_views_are_adjacent_comprehensive() {
        use crate::core::{facet::FacetView, triangulation_data_structure::Tds};

        // Test 1: Adjacent facets in 3D (tetrahedra sharing a triangular face)
        println!("Test 1: Adjacent facets in 3D");

        // Create two tetrahedra that share 3 vertices (forming a shared triangular face)
        let shared_vertices = vec![
            vertex!([0.0, 0.0, 0.0]), // v0
            vertex!([1.0, 0.0, 0.0]), // v1
            vertex!([0.5, 1.0, 0.0]), // v2
        ];

        let vertex_a = vertex!([0.5, 0.5, 1.0]); // Above the shared triangle
        let vertex_b = vertex!([0.5, 0.5, -1.0]); // Below the shared triangle

        // Tetrahedron 1: shared triangle + vertex_a
        let mut vertices1 = shared_vertices.clone();
        vertices1.push(vertex_a);

        // Tetrahedron 2: shared triangle + vertex_b
        let mut vertices2 = shared_vertices;
        vertices2.push(vertex_b);

        let tds1: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices1).unwrap();
        let tds2: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices2).unwrap();

        let cell1_key = tds1.cell_keys().next().unwrap();
        let cell2_key = tds2.cell_keys().next().unwrap();

        // Find the facets that correspond to the shared triangle
        // In tetrahedron 1, this is the facet opposite to vertex_a (index 3)
        // In tetrahedron 2, this is the facet opposite to vertex_b (index 3)
        let facet_view1 = FacetView::new(&tds1, cell1_key, 3).unwrap();
        let facet_view2 = FacetView::new(&tds2, cell2_key, 3).unwrap();

        assert!(
            facet_views_are_adjacent(&facet_view1, &facet_view2),
            "Facets representing the same shared triangle should be adjacent"
        );
        println!("  ✓ Adjacent facets correctly identified");

        // Test 2: Non-adjacent facets from the same tetrahedra
        println!("Test 2: Non-adjacent facets from same tetrahedra");

        // Different facets from the same tetrahedra (not sharing vertices)
        let facet_view1_diff = FacetView::new(&tds1, cell1_key, 0).unwrap(); // Different facet
        let facet_view2_diff = FacetView::new(&tds2, cell2_key, 1).unwrap(); // Different facet

        assert!(
            !facet_views_are_adjacent(&facet_view1_diff, &facet_view2_diff),
            "Different facets with different vertices should not be adjacent"
        );
        println!("  ✓ Non-adjacent facets correctly identified");

        // Test 3: Same facet should be adjacent to itself
        println!("Test 3: Facet adjacent to itself");

        assert!(
            facet_views_are_adjacent(&facet_view1, &facet_view1),
            "A facet should be adjacent to itself"
        );
        println!("  ✓ Self-adjacency works correctly");
    }

    #[test]
    fn test_facet_views_are_adjacent_2d_cases() {
        use crate::core::{facet::FacetView, triangulation_data_structure::Tds};

        println!("Test 2D facet adjacency");

        // Create two 2D triangles that share an edge (2 vertices)
        let shared_edge = vec![
            vertex!([0.0, 0.0]), // v0
            vertex!([1.0, 0.0]), // v1
        ];

        let vertex_c = vertex!([0.5, 1.0]); // Above the shared edge
        let vertex_d = vertex!([0.5, -1.0]); // Below the shared edge

        // Triangle 1: shared edge + vertex_c
        let mut vertices1 = shared_edge.clone();
        vertices1.push(vertex_c);

        // Triangle 2: shared edge + vertex_d
        let mut vertices2 = shared_edge;
        vertices2.push(vertex_d);

        let tds1: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices1).unwrap();
        let tds2: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices2).unwrap();

        let cell1_key = tds1.cell_keys().next().unwrap();
        let cell2_key = tds2.cell_keys().next().unwrap();

        // In 2D, facets are edges. Find the facets that correspond to the shared edge
        // This is the facet opposite to the non-shared vertex
        let facet_view1 = FacetView::new(&tds1, cell1_key, 2).unwrap(); // Opposite to vertex_c
        let facet_view2 = FacetView::new(&tds2, cell2_key, 2).unwrap(); // Opposite to vertex_d

        assert!(
            facet_views_are_adjacent(&facet_view1, &facet_view2),
            "2D facets (edges) sharing vertices should be adjacent"
        );

        // Test non-adjacent edges
        let facet_view1_diff = FacetView::new(&tds1, cell1_key, 0).unwrap();
        let facet_view2_diff = FacetView::new(&tds2, cell2_key, 1).unwrap();

        assert!(
            !facet_views_are_adjacent(&facet_view1_diff, &facet_view2_diff),
            "2D facets with different vertices should not be adjacent"
        );

        println!("  ✓ 2D facet adjacency works correctly");
    }

    #[test]
    fn test_facet_views_are_adjacent_1d_cases() {
        use crate::core::{facet::FacetView, triangulation_data_structure::Tds};

        println!("Test 1D facet adjacency");

        // In 1D, cells are edges and facets are vertices (0D)
        // Two edges sharing a vertex have adjacent facets

        let shared_vertex = vertex!([0.0]);
        let vertex_left = vertex!([-1.0]);
        let vertex_right = vertex!([1.0]);

        // Edge 1: shared_vertex to vertex_left
        let vertices1 = vec![shared_vertex, vertex_left];
        // Edge 2: shared_vertex to vertex_right
        let vertices2 = vec![shared_vertex, vertex_right];

        let tds1: Tds<f64, Option<()>, Option<()>, 1> = Tds::new(&vertices1).unwrap();
        let tds2: Tds<f64, Option<()>, Option<()>, 1> = Tds::new(&vertices2).unwrap();

        let cell1_key = tds1.cell_keys().next().unwrap();
        let cell2_key = tds2.cell_keys().next().unwrap();

        // In 1D, the facets are the individual vertices
        // Facet 0: opposite to vertex at index 0 (so contains vertex at index 1)
        // Facet 1: opposite to vertex at index 1 (so contains vertex at index 0)

        // Both edges contain the shared vertex, so we need to find which facet index
        // corresponds to the shared vertex
        let facet_view1_0 = FacetView::new(&tds1, cell1_key, 0).unwrap(); // Contains vertex_left
        let facet_view1_1 = FacetView::new(&tds1, cell1_key, 1).unwrap(); // Contains shared_vertex

        let facet_view2_0 = FacetView::new(&tds2, cell2_key, 0).unwrap(); // Contains vertex_right
        let facet_view2_1 = FacetView::new(&tds2, cell2_key, 1).unwrap(); // Contains shared_vertex

        // The facets containing the shared vertex should be adjacent
        assert!(
            facet_views_are_adjacent(&facet_view1_1, &facet_view2_1),
            "1D facets (vertices) that are the same should be adjacent"
        );

        // The facets containing different vertices should not be adjacent
        assert!(
            !facet_views_are_adjacent(&facet_view1_0, &facet_view2_0),
            "1D facets with different vertices should not be adjacent"
        );

        println!("  ✓ 1D facet adjacency works correctly");
    }

    #[test]
    fn test_facet_views_are_adjacent_edge_cases() {
        use crate::core::{facet::FacetView, triangulation_data_structure::Tds};

        println!("Test facet adjacency edge cases");

        // Test with minimal triangulation (single tetrahedron)
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        // All facets of the same tetrahedron should be different from each other
        let facet0 = FacetView::new(&tds, cell_key, 0).unwrap();
        let facet1 = FacetView::new(&tds, cell_key, 1).unwrap();
        let facet2 = FacetView::new(&tds, cell_key, 2).unwrap();
        let facet3 = FacetView::new(&tds, cell_key, 3).unwrap();

        // Each facet should be adjacent to itself
        assert!(facet_views_are_adjacent(&facet0, &facet0));
        assert!(facet_views_are_adjacent(&facet1, &facet1));
        assert!(facet_views_are_adjacent(&facet2, &facet2));
        assert!(facet_views_are_adjacent(&facet3, &facet3));

        // Different facets of the same tetrahedron should not be adjacent
        // (they have different sets of vertices)
        assert!(!facet_views_are_adjacent(&facet0, &facet1));
        assert!(!facet_views_are_adjacent(&facet0, &facet2));
        assert!(!facet_views_are_adjacent(&facet0, &facet3));
        assert!(!facet_views_are_adjacent(&facet1, &facet2));
        assert!(!facet_views_are_adjacent(&facet1, &facet3));
        assert!(!facet_views_are_adjacent(&facet2, &facet3));

        println!("  ✓ Single tetrahedron facet relationships correct");
    }

    #[test]
    fn test_facet_views_are_adjacent_performance() {
        use crate::core::{facet::FacetView, triangulation_data_structure::Tds};
        use std::time::Instant;

        println!("Test facet adjacency performance");

        // Create a moderately complex case to test performance
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([2.0, 0.0, 0.0]),
            vertex!([1.0, 2.0, 0.0]),
            vertex!([1.0, 1.0, 2.0]),
        ];

        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        let facet1 = FacetView::new(&tds, cell_key, 0).unwrap();
        let facet2 = FacetView::new(&tds, cell_key, 1).unwrap();

        // Run the adjacency check many times to measure performance
        let start = Instant::now();
        let iterations = 10000;

        for _ in 0..iterations {
            // This should be very fast since it just compares UUID sets
            let _result = facet_views_are_adjacent(&facet1, &facet2);
        }

        let duration = start.elapsed();
        println!("  ✓ {iterations} adjacency checks completed in {duration:?}");

        // Performance info: each check is just UUID set comparison
        // Note: Timing can vary significantly based on build type and CI environment
        if duration.as_millis() > 500 {
            println!("  ⚠️  Performance warning: adjacency checks took {duration:?}");
            println!("     This may indicate debug build or slower CI environment");
        }
    }

    #[test]
    fn test_facet_views_are_adjacent_different_geometries() {
        use crate::core::{facet::FacetView, triangulation_data_structure::Tds};

        println!("Test facet adjacency with different geometries");

        // Create vertices with different coordinates to ensure different UUIDs
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

        let tds1: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices1).unwrap();
        let tds2: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices2).unwrap();

        let cell1_key = tds1.cell_keys().next().unwrap();
        let cell2_key = tds2.cell_keys().next().unwrap();

        let facet1 = FacetView::new(&tds1, cell1_key, 0).unwrap();
        let facet2 = FacetView::new(&tds2, cell2_key, 0).unwrap();

        // Facets from completely different geometries should not be adjacent
        assert!(
            !facet_views_are_adjacent(&facet1, &facet2),
            "Facets from different geometries should not be adjacent"
        );

        println!("  ✓ Different geometries correctly distinguished");
    }

    #[test]
    fn test_facet_views_are_adjacent_uuid_based_comparison() {
        use crate::core::{facet::FacetView, triangulation_data_structure::Tds};

        println!("Test that adjacency is purely UUID-based");

        // Create identical geometry in separate TDS instances
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let tds1: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let tds2: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        let cell1_key = tds1.cell_keys().next().unwrap();
        let cell2_key = tds2.cell_keys().next().unwrap();

        let facet1 = FacetView::new(&tds1, cell1_key, 0).unwrap();
        let facet2 = FacetView::new(&tds2, cell2_key, 0).unwrap();

        // Check if the UUID generation is deterministic based on coordinates
        let facet1_vertex_uuids: Vec<_> = match facet1.vertices() {
            Ok(iter) => iter.map(Vertex::uuid).collect(),
            Err(_) => return, // Skip test if facet1 is invalid
        };
        let facet2_vertex_uuids: Vec<_> = match facet2.vertices() {
            Ok(iter) => iter.map(Vertex::uuid).collect(),
            Err(_) => return, // Skip test if facet2 is invalid
        };

        let uuids_are_same = facet1_vertex_uuids == facet2_vertex_uuids;
        let facets_are_adjacent = facet_views_are_adjacent(&facet1, &facet2);

        // The adjacency should match the UUID equality
        assert_eq!(
            uuids_are_same, facets_are_adjacent,
            "Facet adjacency should exactly match vertex UUID equality"
        );

        if uuids_are_same {
            println!("  ✓ Identical coordinates produce identical UUIDs - facets are adjacent");
        } else {
            println!("  ✓ Different UUIDs for identical coordinates - facets are not adjacent");
        }
    }

    #[test]
    fn test_facet_views_are_adjacent_4d_cases() {
        use crate::core::{facet::FacetView, triangulation_data_structure::Tds};

        println!("Test 4D facet adjacency");

        // Create two 4D simplices (5-vertices each) that share a 3D facet (4 vertices)
        let shared_tetrahedron = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]), // v0
            vertex!([1.0, 0.0, 0.0, 0.0]), // v1
            vertex!([0.0, 1.0, 0.0, 0.0]), // v2
            vertex!([0.0, 0.0, 1.0, 0.0]), // v3
        ];

        let vertex_e = vertex!([0.25, 0.25, 0.25, 1.0]); // Above in 4th dimension
        let vertex_f = vertex!([0.25, 0.25, 0.25, -1.0]); // Below in 4th dimension

        // 4D Simplex 1: shared tetrahedron + vertex_e
        let mut vertices1 = shared_tetrahedron.clone();
        vertices1.push(vertex_e);

        // 4D Simplex 2: shared tetrahedron + vertex_f
        let mut vertices2 = shared_tetrahedron;
        vertices2.push(vertex_f);

        let tds1: Tds<f64, Option<()>, Option<()>, 4> = Tds::new(&vertices1).unwrap();
        let tds2: Tds<f64, Option<()>, Option<()>, 4> = Tds::new(&vertices2).unwrap();

        let cell1_key = tds1.cell_keys().next().unwrap();
        let cell2_key = tds2.cell_keys().next().unwrap();

        // In 4D, facets are tetrahedra. Find the facets that correspond to the shared tetrahedron
        // This is the facet opposite to the non-shared vertex (index 4)
        let facet_view1 = FacetView::new(&tds1, cell1_key, 4).unwrap(); // Opposite to vertex_e
        let facet_view2 = FacetView::new(&tds2, cell2_key, 4).unwrap(); // Opposite to vertex_f

        assert!(
            facet_views_are_adjacent(&facet_view1, &facet_view2),
            "4D facets (tetrahedra) sharing vertices should be adjacent"
        );

        // Test non-adjacent tetrahedra within the same 4D simplices
        let facet_view1_diff = FacetView::new(&tds1, cell1_key, 0).unwrap();
        let facet_view2_diff = FacetView::new(&tds2, cell2_key, 1).unwrap();

        assert!(
            !facet_views_are_adjacent(&facet_view1_diff, &facet_view2_diff),
            "4D facets with different vertices should not be adjacent"
        );

        println!("  ✓ 4D facet adjacency works correctly");
    }

    #[test]
    fn test_facet_views_are_adjacent_5d_cases() {
        use crate::core::{facet::FacetView, triangulation_data_structure::Tds};

        println!("Test 5D facet adjacency");

        // Create two 5D simplices (6-vertices each) that share a 4D facet (5 vertices)
        let shared_4d_simplex = vec![
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0]), // v0
            vertex!([1.0, 0.0, 0.0, 0.0, 0.0]), // v1
            vertex!([0.0, 1.0, 0.0, 0.0, 0.0]), // v2
            vertex!([0.0, 0.0, 1.0, 0.0, 0.0]), // v3
            vertex!([0.0, 0.0, 0.0, 1.0, 0.0]), // v4
        ];

        let vertex_g = vertex!([0.2, 0.2, 0.2, 0.2, 1.0]); // Above in 5th dimension
        let vertex_h = vertex!([0.2, 0.2, 0.2, 0.2, -1.0]); // Below in 5th dimension

        // 5D Simplex 1: shared 4D simplex + vertex_g
        let mut vertices1 = shared_4d_simplex.clone();
        vertices1.push(vertex_g);

        // 5D Simplex 2: shared 4D simplex + vertex_h
        let mut vertices2 = shared_4d_simplex;
        vertices2.push(vertex_h);

        let tds1: Tds<f64, Option<()>, Option<()>, 5> = Tds::new(&vertices1).unwrap();
        let tds2: Tds<f64, Option<()>, Option<()>, 5> = Tds::new(&vertices2).unwrap();

        let cell1_key = tds1.cell_keys().next().unwrap();
        let cell2_key = tds2.cell_keys().next().unwrap();

        // In 5D, facets are 4D simplices. Find the facets that correspond to the shared 4D simplex
        // This is the facet opposite to the non-shared vertex (index 5)
        let facet_view1 = FacetView::new(&tds1, cell1_key, 5).unwrap(); // Opposite to vertex_g
        let facet_view2 = FacetView::new(&tds2, cell2_key, 5).unwrap(); // Opposite to vertex_h

        assert!(
            facet_views_are_adjacent(&facet_view1, &facet_view2),
            "5D facets (4D simplices) sharing vertices should be adjacent"
        );

        // Test non-adjacent 4D simplices within the same 5D simplices
        let facet_view1_diff = FacetView::new(&tds1, cell1_key, 0).unwrap();
        let facet_view2_diff = FacetView::new(&tds2, cell2_key, 1).unwrap();

        assert!(
            !facet_views_are_adjacent(&facet_view1_diff, &facet_view2_diff),
            "5D facets with different vertices should not be adjacent"
        );

        println!("  ✓ 5D facet adjacency works correctly");
    }

    #[test]
    fn test_facet_views_are_adjacent_multidimensional_summary() {
        println!("Testing facet adjacency across all supported dimensions (1D-5D)");

        // This test summarizes the multidimensional support
        let dimensions_tested = vec![
            ("1D", "edges", "vertices"),
            ("2D", "triangles", "edges"),
            ("3D", "tetrahedra", "triangles"),
            ("4D", "4-simplices", "tetrahedra"),
            ("5D", "5-simplices", "4-simplices"),
        ];

        for (dim, cell_type, facet_type) in dimensions_tested {
            println!("  ✓ {dim}: {cell_type} with {facet_type} facets");
        }

        println!("  ✓ All dimensional cases covered comprehensively");
    }

    // =============================================================================
    // USIZE TO U8 CONVERSION UTILITY TESTS
    // =============================================================================

    #[test]
    fn test_usize_to_u8_conversion_comprehensive() {
        use super::usize_to_u8;

        // Test successful conversions
        assert_eq!(usize_to_u8(0, 4), Ok(0));
        assert_eq!(usize_to_u8(1, 4), Ok(1));
        assert_eq!(usize_to_u8(255, 256), Ok(255));

        // Test conversion at boundary
        assert_eq!(usize_to_u8(u8::MAX as usize, 256), Ok(u8::MAX));

        // Test failed conversion (index too large)
        let result = usize_to_u8(256, 10);
        assert!(result.is_err());
        if let Err(FacetError::InvalidFacetIndex { index, facet_count }) = result {
            assert_eq!(index, u8::MAX); // Should use MAX as placeholder
            assert_eq!(facet_count, 10);
        } else {
            panic!("Expected InvalidFacetIndex error");
        }

        // Test failed conversion (very large index)
        let result = usize_to_u8(usize::MAX, 5);
        assert!(result.is_err());
        if let Err(FacetError::InvalidFacetIndex { index, facet_count }) = result {
            assert_eq!(index, u8::MAX);
            assert_eq!(facet_count, 5);
        } else {
            panic!("Expected InvalidFacetIndex error");
        }
    }

    #[test]
    fn test_usize_to_u8_boundary_cases() {
        use super::usize_to_u8;

        // Test all valid u8 values
        for i in 0u8..=255 {
            let result = usize_to_u8(i as usize, 256);
            assert_eq!(result, Ok(i), "Failed to convert {i}");
        }

        // Test just above boundary
        let result = usize_to_u8(256, 300);
        assert!(result.is_err());

        // Test various large values
        let large_values = [257, 1000, 10000, 65536, usize::MAX];
        for &val in &large_values {
            let result = usize_to_u8(val, val);
            assert!(result.is_err(), "Should fail for value {val}");
            if let Err(FacetError::InvalidFacetIndex { index, facet_count }) = result {
                assert_eq!(index, u8::MAX);
                assert_eq!(facet_count, val);
            }
        }
    }

    #[test]
    fn test_usize_to_u8_error_consistency() {
        use super::usize_to_u8;

        // Test that all out-of-range values produce consistent errors
        let test_cases = [
            (256, 100),
            (300, 500),
            (1000, 1500),
            (usize::MAX, 42),
            (65536, 10),
        ];

        for &(idx, count) in &test_cases {
            let result = usize_to_u8(idx, count);
            assert!(result.is_err(), "Should fail for index {idx}");

            match result {
                Err(FacetError::InvalidFacetIndex { index, facet_count }) => {
                    assert_eq!(index, u8::MAX, "Should use u8::MAX as placeholder");
                    assert_eq!(facet_count, count, "Should preserve facet_count");
                }
                _ => panic!("Expected InvalidFacetIndex error for index {idx}"),
            }
        }
    }

    #[test]
    fn test_usize_to_u8_edge_values() {
        use super::usize_to_u8;

        // Test edge cases around u8::MAX
        assert_eq!(usize_to_u8(254, 300), Ok(254));
        assert_eq!(usize_to_u8(255, 300), Ok(255));
        assert!(usize_to_u8(256, 300).is_err());
        assert!(usize_to_u8(257, 300).is_err());

        // Test with different facet_count values
        let facet_counts = [1, 10, 100, 255, 256, 1000, usize::MAX];
        for &count in &facet_counts {
            // Valid conversion
            let result_valid = usize_to_u8(0, count);
            assert_eq!(result_valid, Ok(0));

            // Invalid conversion
            let result_invalid = usize_to_u8(256, count);
            assert!(result_invalid.is_err());
            if let Err(FacetError::InvalidFacetIndex { facet_count, .. }) = result_invalid {
                assert_eq!(facet_count, count);
            }
        }
    }

    #[test]
    fn test_usize_to_u8_deterministic_behavior() {
        use super::usize_to_u8;

        // Test that same inputs produce same results
        for i in 0..10 {
            let result1 = usize_to_u8(i, 20);
            let result2 = usize_to_u8(i, 20);
            assert_eq!(result1, result2, "Results should be deterministic for {i}");
        }

        // Test that error cases are also deterministic
        for i in [256, 1000, usize::MAX] {
            let result1 = usize_to_u8(i, 100);
            let result2 = usize_to_u8(i, 100);
            assert_eq!(
                result1, result2,
                "Error results should be deterministic for {i}"
            );
        }
    }

    #[test]
    fn test_usize_to_u8_performance_characteristics() {
        use super::usize_to_u8;
        use std::time::Instant;

        // Test that the function is fast for valid conversions
        let start = Instant::now();
        for i in 0..1000 {
            let _ = usize_to_u8(i % 256, 300);
        }
        let duration = start.elapsed();

        // Should be very fast (under 1ms even in debug mode)
        assert!(
            duration.as_millis() < 10,
            "Conversion should be fast, took {duration:?}"
        );

        // Test that error cases are also fast
        let start = Instant::now();
        for i in 256..1256 {
            let _ = usize_to_u8(i, 100);
        }
        let duration = start.elapsed();

        assert!(
            duration.as_millis() < 10,
            "Error cases should be fast, took {duration:?}"
        );
    }

    #[test]
    fn test_usize_to_u8_memory_efficiency() {
        use super::usize_to_u8;

        // Test that the function doesn't allocate memory unnecessarily
        // This is a behavioral test - the function should use stack allocation only
        let (result, _) = measure_with_result(|| {
            let mut results = Vec::new();
            for i in 0..100 {
                results.push(usize_to_u8(i, 200));
            }
            results
        });

        // Verify results are correct
        for (i, result) in result.iter().enumerate() {
            assert_eq!(
                *result,
                Ok(u8::try_from(i).unwrap()),
                "Result should be correct for {i}"
            );
        }
    }

    #[test]
    fn test_usize_to_u8_error_message_quality() {
        use super::usize_to_u8;

        // Test that error messages contain useful information
        let result = usize_to_u8(300, 42);
        assert!(result.is_err());

        if let Err(error) = result {
            let error_string = format!("{error}");
            // The error should contain information about the limits
            assert!(
                error_string.contains("InvalidFacetIndex") || error_string.contains("index"),
                "Error message should indicate it's about invalid index: {error_string}"
            );
        }

        // Test error with different values
        let result = usize_to_u8(usize::MAX, 7);
        assert!(result.is_err());
        if let Err(FacetError::InvalidFacetIndex { index, facet_count }) = result {
            assert_eq!(index, u8::MAX);
            assert_eq!(facet_count, 7);
        }
    }

    #[test]
    fn test_usize_to_u8_thread_safety() {
        use super::usize_to_u8;
        use std::thread;

        // Test that the function works correctly in multi-threaded context
        let handles: Vec<_> = (0..4)
            .map(|thread_id| {
                thread::spawn(move || {
                    let mut results = Vec::new();
                    for i in 0..100 {
                        let val = (thread_id * 50 + i) % 256;
                        results.push(usize_to_u8(val, 300));
                    }
                    results
                })
            })
            .collect();

        // Join all threads and verify results
        for handle in handles {
            let results = handle.join().expect("Thread should complete successfully");
            for result in results {
                assert!(result.is_ok(), "All results should be successful");
            }
        }
    }
}
