//! Property-based tests for Euler characteristic computation.
//!
//! This module uses proptest to verify Euler characteristic calculation
//! across random triangulations in multiple dimensions (2D-5D).
//!
//! ## Test Properties
//!
//! 1. **Euler Formula Consistency**: Computed χ matches expected value for classification
//! 2. **Simplex Count Validity**: Vertex and cell counts match Tds counts
//! 3. **Classification Consistency**: Expected χ for classification matches computed χ  
//!
//! ## Notes
//!
//! Random triangulations are expected to satisfy Euler characteristic validation; any
//! mismatch indicates a bug in construction or validation.
//!
//! For deterministic tests with known configurations, see `euler_characteristic.rs`.

use delaunay::geometry::util::generate_random_triangulation_with_topology_guarantee;
use delaunay::prelude::triangulation::*;
use delaunay::topology::characteristics::{euler, validation};
use proptest::prelude::*;

// =============================================================================
// TEST CONFIGURATION
// =============================================================================

/// Strategy for generating finite f64 coordinates
fn finite_coordinate() -> impl Strategy<Value = f64> {
    (-100.0..100.0).prop_filter("must be finite", |x: &f64| x.is_finite())
}

// =============================================================================
// PROPERTY-BASED TESTS - RANDOM TRIANGULATIONS
// =============================================================================

/// Macro to generate Euler characteristic property tests for a given dimension.
///
/// These tests verify that randomly generated valid Delaunay triangulations
/// have Euler characteristics that match their topological classification.
///
/// # Test Properties
///
/// 1. **Euler Formula Consistency**: Computed χ matches expected value
/// 2. **Simplex Count Validity**: All simplex counts are consistent
/// 3. **Classification Consistency**: Classification χ matches computed χ
///
/// # Randomness Strategy
///
/// Uses property-based testing (proptest) with:
/// - Random point coordinates in [-100, 100]
/// - Variable number of vertices per dimension
/// - Filters for finite coordinates
/// - Automatic shrinking on failure
macro_rules! test_euler_properties {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal $(, #[$attr:meta])*) => {
        pastey::paste! {
            proptest! {
                /// Property: Euler characteristic matches topological classification
                $(#[$attr])*
                #[test]
                fn [<prop_euler_matches_classification_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| vertex!(coords)),
                        $min_vertices..$max_vertices
                    )
                ) {
                    // Attempt to build triangulation
                    if let Ok(dt) = DelaunayTriangulation::new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        // Validate Euler characteristic
                        let result = validation::validate_triangulation_euler(dt.tds())?;


                        // Core property: χ must match expected value for the topology
                        // The validation checks this internally via is_valid()
                        prop_assert!(result.is_valid(),
                            "{}D triangulation Euler characteristic doesn't match classification: \
                            χ={}, expected={:?}, classification={:?}, V={}, cells={}",
                            $dim,
                            result.chi,
                            result.expected,
                            result.classification,
                            result.counts.count(0),
                            result.counts.count($dim)
                        );
                    }
                }

                /// Property: Simplex counts are internally consistent
                $(#[$attr])*
                #[test]
                fn [<prop_simplex_counts_consistent_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| vertex!(coords)),
                        $min_vertices..$max_vertices
                    )
                ) {
                    if let Ok(dt) = DelaunayTriangulation::new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        let counts = euler::count_simplices(dt.tds())?;

                        // Basic sanity checks
                        prop_assert_eq!(
                            counts.count(0),
                            dt.number_of_vertices(),
                            "{}D: Vertex count mismatch",
                            $dim
                        );

                        prop_assert_eq!(
                            counts.count($dim),
                            dt.number_of_cells(),
                            "{}D: Cell count mismatch",
                            $dim
                        );

                        // All dimensions should be represented in counts
                        prop_assert_eq!(
                            counts.dimension(),
                            $dim,
                            "{}D: Dimension mismatch in simplex counts",
                            $dim
                        );
                    }
                }

                /// Property: Classification and expected χ are consistent
                $(#[$attr])*
                #[test]
                fn [<prop_classification_chi_consistent_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| vertex!(coords)),
                        $min_vertices..$max_vertices
                    )
                ) {
                    if let Ok(dt) = DelaunayTriangulation::new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        let result = validation::validate_triangulation_euler(dt.tds())?;

                        // If we have an expected χ, computed χ must match
                        if let Some(expected_chi) = result.expected {
                            prop_assert_eq!(
                                result.chi,
                                expected_chi,
                                "{}D: Computed χ={} doesn't match expected χ={} for {:?}",
                                $dim,
                                result.chi,
                                expected_chi,
                                result.classification
                            );
                        }
                    }
                }
            }
        }
    };
}

#[test]
fn test_seeded_random_generator_euler_consistent() {
    let dt_2d = generate_random_triangulation_with_topology_guarantee::<f64, (), (), 2>(
        15,
        (0.0, 10.0),
        None,
        Some(555),
        TopologyGuarantee::PLManifold,
    )
    .unwrap();
    let result_2d = validation::validate_triangulation_euler(dt_2d.tds()).unwrap();
    assert!(
        result_2d.is_valid(),
        "2D seeded random triangulation Euler mismatch: χ={}, expected={:?}, classification={:?}, V={}, cells={}",
        result_2d.chi,
        result_2d.expected,
        result_2d.classification,
        result_2d.counts.count(0),
        result_2d.counts.count(2),
    );

    let dt_3d = generate_random_triangulation_with_topology_guarantee::<f64, (), (), 3>(
        20,
        (-3.0, 3.0),
        None,
        Some(666),
        TopologyGuarantee::PLManifold,
    )
    .unwrap();
    let result_3d = validation::validate_triangulation_euler(dt_3d.tds()).unwrap();
    assert!(
        result_3d.is_valid(),
        "3D seeded random triangulation Euler mismatch: χ={}, expected={:?}, classification={:?}, V={}, cells={}",
        result_3d.chi,
        result_3d.expected,
        result_3d.classification,
        result_3d.counts.count(0),
        result_3d.counts.count(3),
    );
}
// Generate property tests for dimensions 2-5
// Parameters: dimension, min_vertices, max_vertices
//
// Vertex ranges chosen to:
// - Ensure D+1 minimum for valid simplex
// - Balance test execution time with coverage
// - Match patterns in other proptest files
test_euler_properties!(2, 4, 15);
test_euler_properties!(3, 5, 20);
test_euler_properties!(4, 6, 25, #[ignore = "Slow (>60s) in test-integration"]);
test_euler_properties!(5, 7, 30, #[ignore = "Slow (>60s) in test-integration"]);
