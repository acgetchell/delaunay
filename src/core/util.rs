//! General helper utilities

use serde::{Serialize, de::DeserializeOwned};
use thiserror::Error;
use uuid::Uuid;

use crate::core::facet::Facet;
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
pub fn facets_are_adjacent<T, U, V, const D: usize>(
    facet1: &Facet<T, U, V, D>,
    facet2: &Facet<T, U, V, D>,
) -> bool
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    use crate::core::collections::FastHashSet;
    let vertices1: FastHashSet<_> = facet1.vertices().into_iter().collect();
    let vertices2: FastHashSet<_> = facet2.vertices().into_iter().collect();
    vertices1 == vertices2
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
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
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
    [T; D]: Copy + Default + serde::de::DeserializeOwned + serde::Serialize + Sized,
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
}
