//! Facet key utilities.

#![forbid(unsafe_code)]

use crate::core::facet::FacetError;
use crate::core::traits::data_type::DataType;
use crate::core::triangulation_data_structure::{CellKey, Tds, VertexKey};
use crate::geometry::traits::coordinate::CoordinateScalar;

/// Derives a facet key directly from vertex keys.
///
/// Computes the canonical facet key for lookup in facet-to-cells mappings. This is useful
/// in hot paths like visibility checking in convex hull algorithms and boundary analysis.
///
/// If you have `Vertex` instances instead of `VertexKey`s, obtain the keys via the TDS
/// using the vertex's UUID (e.g., `tds.vertex_key_from_uuid(vertex.uuid())`).
///
/// # Arguments
///
/// * `facet_vertex_keys` - The vertex keys that make up the facet
///
/// # Returns
///
/// A `Result` containing the facet key or a `FacetError` if validation fails.
///
/// # Errors
///
/// Returns `FacetError::InsufficientVertices` if the vertex count doesn't equal `D`
///
/// # Examples
///
/// ```
/// use delaunay::prelude::*;
/// use delaunay::core::util::checked_facet_key_from_vertex_keys;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let dt = DelaunayTriangulation::new(&vertices).unwrap();
/// let tds = dt.tds();
///
/// // Get facet vertex keys from a cell - no need to materialize Vertex objects
/// if let Some(cell) = tds.cells().map(|(_, cell)| cell).next() {
///     let facet_vertex_keys: Vec<_> = cell.vertices().iter().skip(1).copied().collect(); // Skip 1 vertex to get D vertices
///     assert_eq!(facet_vertex_keys.len(), 3); // For 3D triangulation, facet has 3 vertices
///     let facet_key = checked_facet_key_from_vertex_keys::<3>(&facet_vertex_keys).unwrap();
///     println!("Facet key: {}", facet_key);
/// }
/// ```
///
/// # Performance
///
/// - Time Complexity: O(D log D) where D is the facet dimension (for sorting vertex keys)
/// - Space Complexity: O(D) for the temporary sorted buffer (stack-allocated via `SmallBuffer`)
/// - Improves cache locality by working only with compact `VertexKey` types
///
/// # See Also
///
/// - [`crate::core::facet::facet_key_from_vertices`] - Low-level function that computes the hash from keys
pub fn checked_facet_key_from_vertex_keys<const D: usize>(
    facet_vertex_keys: &[VertexKey],
) -> Result<u64, FacetError> {
    use crate::core::facet::facet_key_from_vertices;

    // Validate that the number of vertex keys matches the expected dimension
    // In a D-dimensional triangulation, a facet should have exactly D vertices
    if facet_vertex_keys.len() != D {
        return Err(FacetError::InsufficientVertices {
            expected: D,
            actual: facet_vertex_keys.len(),
            dimension: D,
        });
    }

    // Directly compute the facet key from vertex keys
    // facet_key_from_vertices handles the sorting internally
    Ok(facet_key_from_vertices(facet_vertex_keys))
}

/// Verifies facet index consistency between two neighboring cells.
///
/// This function checks that a shared facet computed from both cells' perspectives
/// produces the same facet key, which is critical for catching subtle neighbor
/// assignment errors in triangulation algorithms.
///
/// # Arguments
///
/// * `tds` - The triangulation data structure
/// * `cell1_key` - Key of the first cell
/// * `cell2_key` - Key of the second (neighboring) cell
/// * `facet_idx` - Index of the facet in cell1 that should match a facet in cell2
///
/// # Returns
///
/// `Ok(true)` if a matching facet is found in cell2 with the same facet key.
/// `Ok(false)` if no matching facet is found.
/// `Err(FacetError)` if there's an error accessing cell or facet data.
///
/// # Errors
///
/// Returns `FacetError` if:
/// - Either cell cannot be found in the TDS
/// - Facet views cannot be created from the cells
/// - Facet vertices cannot be accessed
/// - The facet index is out of bounds
///
/// # Examples
///
/// ```rust,no_run
/// use delaunay::core::util::verify_facet_index_consistency;
/// use delaunay::core::triangulation_data_structure::Tds;
///
/// fn validate_neighbor_consistency(
///     tds: &Tds<f64, (), (), 3>,
/// ) -> Result<(), String> {
///     // Get two neighboring cell keys
///     let cell_keys: Vec<_> = tds.cell_keys().take(2).collect();
///     if cell_keys.len() >= 2 {
///         // Check if facet 0 of cell1 matches a facet in cell2
///         let consistent = verify_facet_index_consistency(
///             tds,
///             cell_keys[0],
///             cell_keys[1],
///             0,
///         )
///         .map_err(|e| format!("Facet error: {}", e))?;
///
///         if consistent {
///             println!("Facet indices are consistent");
///             Ok(())
///         } else {
///             Err("No matching facet found in neighbor".to_string())
///         }
///     } else {
///         Ok(())
///     }
/// }
/// ```
///
/// # Performance
///
/// - Time Complexity: O(D²) where D is the dimension (iterates over facets and vertices)
/// - Space Complexity: O(D) for temporary vertex buffers
pub fn verify_facet_index_consistency<const D: usize>(
    tds: &Tds<impl CoordinateScalar, impl DataType, impl DataType, D>,
    cell1_key: CellKey,
    cell2_key: CellKey,
    facet_idx: usize,
) -> Result<bool, FacetError> {
    // Get facet views from both cells (validates cells exist)
    let cell1_facets = crate::core::cell::Cell::facet_views_from_tds(tds, cell1_key)?;
    let cell2_facets = crate::core::cell::Cell::facet_views_from_tds(tds, cell2_key)?;

    // Check facet index bounds
    if facet_idx >= cell1_facets.len() {
        // Saturate to u8::MAX for error reporting if index overflows u8
        let idx_u8 = u8::try_from(facet_idx).unwrap_or(u8::MAX);
        return Err(FacetError::InvalidFacetIndex {
            index: idx_u8,
            facet_count: cell1_facets.len(),
        });
    }

    // Get the facet from cell1 and compute its key
    let cell1_facet = &cell1_facets[facet_idx];
    let cell1_key_value = cell1_facet.key()?;

    // Find matching facet in cell2
    for cell2_facet in &cell2_facets {
        if cell1_key_value == cell2_facet.key()? {
            return Ok(true);
        }
    }

    Ok(false) // No matching facet found
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
/// Returns `FacetError::InvalidFacetIndexOverflow` if the index cannot fit in a u8.
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
    u8::try_from(idx).map_err(|_| FacetError::InvalidFacetIndexOverflow {
        original_index: idx,
        facet_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::core::util::measure_with_result;
    use crate::vertex;

    use std::thread;
    use std::time::Instant;

    #[test]
    #[expect(clippy::too_many_lines)]
    fn test_checked_facet_key_from_vertex_keys_comprehensive() {
        println!("Testing checked_facet_key_from_vertex_keys comprehensively");

        // Create a triangulation
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices).unwrap();
        let tds = &dt.as_triangulation().tds;

        // Test 1: Basic functionality - successful key derivation
        println!("  Testing basic functionality...");
        let cell = tds.cells().map(|(_, cell)| cell).next().unwrap();
        let facet_vertex_keys: Vec<_> = cell.vertices().iter().skip(1).copied().collect();

        let result = checked_facet_key_from_vertex_keys::<3>(&facet_vertex_keys);
        assert!(
            result.is_ok(),
            "Facet key derivation should succeed for valid vertex keys"
        );

        let facet_key = result.unwrap();
        println!("    Derived facet key: {facet_key}");

        // Test deterministic behavior - same vertex keys produce same key
        let result2 = checked_facet_key_from_vertex_keys::<3>(&facet_vertex_keys);
        assert!(result2.is_ok(), "Second derivation should also succeed");
        assert_eq!(
            facet_key,
            result2.unwrap(),
            "Same vertex keys should produce same facet key"
        );

        // Test different vertex keys produce different keys
        let all_vertex_keys = cell.vertices();
        let different_facet_vertex_keys: Vec<_> = all_vertex_keys.iter().take(3).copied().collect();
        if different_facet_vertex_keys.len() == 3
            && different_facet_vertex_keys != facet_vertex_keys
        {
            let result3 = checked_facet_key_from_vertex_keys::<3>(&different_facet_vertex_keys);
            assert!(
                result3.is_ok(),
                "Different facet key derivation should succeed"
            );
            let different_facet_key = result3.unwrap();
            assert_ne!(
                facet_key, different_facet_key,
                "Different vertex keys should produce different facet keys"
            );
            println!("    Different facet key: {different_facet_key}");
        }

        // Test 2: Error cases
        println!("  Testing error handling...");

        // Wrong vertex key count
        let single_key: Vec<VertexKey> = vec![facet_vertex_keys[0]];
        let result_count = checked_facet_key_from_vertex_keys::<3>(&single_key);
        assert!(
            result_count.is_err(),
            "Should return error for wrong vertex key count"
        );
        if let Err(error) = result_count {
            match error {
                FacetError::InsufficientVertices {
                    expected,
                    actual,
                    dimension,
                } => {
                    assert_eq!(expected, 3, "Expected 3 vertex keys for 3D");
                    assert_eq!(actual, 1, "Got 1 vertex key");
                    assert_eq!(dimension, 3, "Dimension should be 3");
                }
                _ => panic!("Expected InsufficientVertices error, got: {error:?}"),
            }
        }

        // Empty vertex keys
        let empty_keys: Vec<VertexKey> = vec![];
        let result_empty = checked_facet_key_from_vertex_keys::<3>(&empty_keys);
        assert!(
            result_empty.is_err(),
            "Empty vertex keys should fail validation"
        );
        if let Err(error) = result_empty {
            match error {
                FacetError::InsufficientVertices {
                    expected,
                    actual,
                    dimension,
                } => {
                    assert_eq!(expected, 3, "Expected 3 vertex keys for 3D");
                    assert_eq!(actual, 0, "Got 0 vertex keys");
                    assert_eq!(dimension, 3, "Dimension should be 3");
                }
                _ => {
                    panic!(
                        "Expected InsufficientVertices error for empty vertex keys, got: {error:?}"
                    )
                }
            }
        }

        // Test 3: Consistency with TDS cache
        println!("  Testing consistency with TDS...");
        let cache = tds
            .build_facet_to_cells_map()
            .expect("Should build facet map in test");
        let mut keys_found = 0;
        let mut keys_tested = 0;

        for cell in tds.cells().map(|(_, cell)| cell) {
            let cell_vertex_keys = cell.vertices();
            for skip_vertex_idx in 0..cell_vertex_keys.len() {
                let facet_vertex_keys: Vec<_> = cell_vertex_keys
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != skip_vertex_idx)
                    .map(|(_, &vk)| vk)
                    .collect();

                if !facet_vertex_keys.is_empty() {
                    let key_result = checked_facet_key_from_vertex_keys::<3>(&facet_vertex_keys);
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

    #[test]
    fn test_verify_facet_index_consistency_true_false_and_error_cases() {
        // True case: comparing a cell to itself.
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices).unwrap();
        let tds = &dt.as_triangulation().tds;
        let cell_key = tds.cell_keys().next().unwrap();
        assert!(verify_facet_index_consistency(tds, cell_key, cell_key, 0).unwrap());

        // Error case: facet index out of bounds.
        let err = verify_facet_index_consistency(tds, cell_key, cell_key, 99).unwrap_err();
        assert!(matches!(err, FacetError::InvalidFacetIndex { .. }));

        // Logging: demonstrate behavior for large out-of-bounds facet index
        let err_large = verify_facet_index_consistency(tds, cell_key, cell_key, 300).unwrap_err();
        println!("    Large facet_idx=300 error: {err_large:?}");
        assert!(matches!(err_large, FacetError::InvalidFacetIndex { .. }));

        // False case: two disjoint triangles in the same TDS share no facet keys.
        let mut tds2: Tds<f64, (), (), 2> = Tds::empty();
        let v_a = tds2
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]))
            .unwrap();
        let v_b = tds2
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]))
            .unwrap();
        let v_c = tds2
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]))
            .unwrap();
        let v_d = tds2
            .insert_vertex_with_mapping(vertex!([10.0, 10.0]))
            .unwrap();
        let v_e = tds2
            .insert_vertex_with_mapping(vertex!([11.0, 10.0]))
            .unwrap();
        let v_f = tds2
            .insert_vertex_with_mapping(vertex!([10.0, 11.0]))
            .unwrap();

        let c1 = tds2
            .insert_cell_with_mapping(
                crate::core::cell::Cell::new(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let c2 = tds2
            .insert_cell_with_mapping(
                crate::core::cell::Cell::new(vec![v_d, v_e, v_f], None).unwrap(),
            )
            .unwrap();

        assert!(!verify_facet_index_consistency(&tds2, c1, c2, 0).unwrap());
    }

    #[test]
    fn test_usize_to_u8_conversion_comprehensive() {
        // Sub-test: Successful conversions
        assert_eq!(usize_to_u8(0, 4), Ok(0));
        assert_eq!(usize_to_u8(1, 4), Ok(1));
        assert_eq!(usize_to_u8(255, 256), Ok(255));
        assert_eq!(usize_to_u8(u8::MAX as usize, 256), Ok(u8::MAX));

        // Sub-test: All valid u8 values (0..=255)
        for i in 0u8..=255 {
            let result = usize_to_u8(i as usize, 256);
            assert_eq!(result, Ok(i), "Failed to convert {i}");
        }

        // Sub-test: Edge cases around u8::MAX boundary
        assert_eq!(usize_to_u8(254, 300), Ok(254));
        assert_eq!(usize_to_u8(255, 300), Ok(255));
        assert!(usize_to_u8(256, 300).is_err());
        assert!(usize_to_u8(257, 300).is_err());

        // Sub-test: Different facet_count values
        let facet_counts = [1, 10, 100, 255, 256, 1000, usize::MAX];
        for &count in &facet_counts {
            assert_eq!(
                usize_to_u8(0, count),
                Ok(0),
                "Valid conversion should succeed"
            );
            let result_invalid = usize_to_u8(256, count);
            assert!(result_invalid.is_err(), "Invalid conversion should fail");
            if let Err(FacetError::InvalidFacetIndexOverflow { facet_count, .. }) = result_invalid {
                assert_eq!(facet_count, count);
            }
        }

        // Sub-test: Deterministic behavior
        for i in 0..10 {
            assert_eq!(
                usize_to_u8(i, 20),
                usize_to_u8(i, 20),
                "Should be deterministic"
            );
        }
        for i in [256, 1000, usize::MAX] {
            assert_eq!(
                usize_to_u8(i, 100),
                usize_to_u8(i, 100),
                "Error cases should be deterministic"
            );
        }
    }

    #[test]
    fn test_usize_to_u8_error_handling() {
        // Sub-test: Basic error cases
        let result = usize_to_u8(256, 10);
        assert!(result.is_err());
        if let Err(FacetError::InvalidFacetIndexOverflow {
            original_index,
            facet_count,
        }) = result
        {
            assert_eq!(original_index, 256);
            assert_eq!(facet_count, 10);
        } else {
            panic!("Expected InvalidFacetIndexOverflow error");
        }

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

        // Sub-test: Error consistency across various inputs
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
                Err(FacetError::InvalidFacetIndexOverflow {
                    original_index,
                    facet_count,
                }) => {
                    assert_eq!(original_index, idx, "Should preserve original index");
                    assert_eq!(facet_count, count, "Should preserve facet_count");
                }
                _ => panic!("Expected InvalidFacetIndex error for index {idx}"),
            }
        }

        // Sub-test: Large values
        let large_values = [257, 1000, 10000, 65536, usize::MAX];
        for &val in &large_values {
            let result = usize_to_u8(val, val);
            assert!(result.is_err(), "Should fail for value {val}");
            if let Err(FacetError::InvalidFacetIndexOverflow {
                original_index,
                facet_count,
            }) = result
            {
                assert_eq!(original_index, val);
                assert_eq!(facet_count, val);
            }
        }

        // Sub-test: Error message quality
        let result = usize_to_u8(300, 42);
        assert!(result.is_err());
        if let Err(error) = result {
            let error_string = format!("{error}");
            assert!(
                error_string.contains("InvalidFacetIndex") || error_string.contains("index"),
                "Error message should indicate invalid index: {error_string}"
            );
        }

        let result = usize_to_u8(usize::MAX, 7);
        if let Err(FacetError::InvalidFacetIndexOverflow {
            original_index,
            facet_count,
        }) = result
        {
            assert_eq!(original_index, usize::MAX);
            assert_eq!(facet_count, 7);
        } else {
            panic!("Expected InvalidFacetIndexOverflow error");
        }
    }

    #[test]
    fn test_usize_to_u8_performance_and_threading() {
        // Sub-test: Performance characteristics (informational only)
        let start = Instant::now();
        for i in 0..1000 {
            let _ = usize_to_u8(i % 256, 300);
        }
        let duration = start.elapsed();
        eprintln!("usize_to_u8 valid conversions: 1000 iters in {duration:?}");

        let start = Instant::now();
        for i in 256..1256 {
            let _ = usize_to_u8(i, 100);
        }
        let duration = start.elapsed();
        eprintln!("usize_to_u8 error conversions: 1000 iters in {duration:?}");

        // Sub-test: Memory efficiency (stack allocation only)
        let (result, _alloc_info) = measure_with_result(|| {
            let mut results = Vec::new();
            for i in 0..100 {
                results.push(usize_to_u8(i, 200));
            }
            results
        });
        for (i, result) in result.iter().enumerate() {
            assert_eq!(
                *result,
                Ok(u8::try_from(i).unwrap()),
                "Result should be correct for {i}"
            );
        }

        // Sub-test: Thread safety
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
        for handle in handles {
            let results = handle.join().expect("Thread should complete successfully");
            for result in results {
                assert!(result.is_ok(), "All results should be successful");
            }
        }
    }
}
