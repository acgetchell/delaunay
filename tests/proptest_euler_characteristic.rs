//! Property-based tests for Euler characteristic computation.
//!
//! This module uses proptest to verify Euler characteristic calculation
//! across random triangulations in multiple dimensions (2D-5D).
//!
//! ## Test Properties
//!
//! 1. **Euler Formula Consistency**: Computed χ matches expected value for classification
//! 2. **Simplex Count Validity**: Vertex and simplex counts match owner counts
//! 3. **Classification Consistency**: Expected χ for classification matches computed χ  
//!
//! ## Notes
//!
//! Random triangulations are expected to satisfy Euler characteristic validation; any
//! mismatch indicates a bug in construction or validation.
//!
//! For deterministic tests with known configurations, see `euler_characteristic.rs`.

#[path = "common/full_dimensional_vertices.rs"]
mod full_dimensional_vertices;

#[macro_use]
#[path = "common/proptest_config.rs"]
mod proptest_config;

use delaunay::prelude::construction::{DelaunayTriangulation, TopologyGuarantee};
use delaunay::prelude::generators::try_generate_random_triangulation_with_topology;
use full_dimensional_vertices::full_dimensional_vertices;
use proptest::prelude::*;
use proptest::test_runner::TestCaseError;
use std::num::NonZeroUsize;

// =============================================================================
// TEST CONFIGURATION
// =============================================================================

/// Builds non-zero point-count literals for deterministic generator checks.
const fn nonzero(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("test point count must be non-zero")
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
/// - A fixed full-dimensional simplex plus random coordinates in [-100, 100]
/// - Variable number of vertices per dimension
/// - Coordinate-only duplicate rejection before construction
/// - Automatic shrinking on failure
macro_rules! test_euler_properties {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal $(, #[$attr:meta])*) => {
        pastey::paste! {
            repo_proptest! {
                /// Property: Euler characteristic matches topological classification
                $(#[$attr])*
                #[test]
                fn [<prop_euler_matches_classification_ $dim d>](
                    vertices in full_dimensional_vertices::<$dim>($min_vertices, $max_vertices - 1)
                ) {
                    let dt = DelaunayTriangulation::builder(&vertices)
                        .topology_guarantee(TopologyGuarantee::PLManifold)
                        .build()
                        .map_err(|error| TestCaseError::fail(format!(
                            "{}D construction failed for admitted vertices {vertices:?}: {error:?}", $dim
                        )))?;
                        // Validate Euler characteristic
                        let result = dt.euler_check()?;


                        // Core property: χ must match expected value for the topology
                        // The validation checks this internally via is_valid()
                        prop_assert!(result.is_valid(),
                            "{}D triangulation Euler characteristic doesn't match classification: \
                            χ={}, expected={:?}, classification={:?}, V={}, simplices={}",
                            $dim,
                            result.chi,
                            result.expected,
                            result.classification,
                            result.counts.count(0),
                            result.counts.count($dim)
                        );
                }

                /// Property: Simplex counts are internally consistent
                $(#[$attr])*
                #[test]
                fn [<prop_simplex_counts_consistent_ $dim d>](
                    vertices in full_dimensional_vertices::<$dim>($min_vertices, $max_vertices - 1)
                ) {
                    let dt = DelaunayTriangulation::builder(&vertices)
                        .topology_guarantee(TopologyGuarantee::PLManifold)
                        .build()
                        .map_err(|error| TestCaseError::fail(format!(
                            "{}D construction failed for admitted vertices {vertices:?}: {error:?}", $dim
                        )))?;
                        let counts = dt.simplex_counts()?;

                        // Basic sanity checks
                        prop_assert_eq!(
                            counts.count(0),
                            dt.number_of_vertices(),
                            "{}D: Vertex count mismatch",
                            $dim
                        );

                        prop_assert_eq!(
                            counts.count($dim),
                            dt.number_of_simplices(),
                            "{}D: Simplex count mismatch",
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

                /// Property: Classification and expected χ are consistent
                $(#[$attr])*
                #[test]
                fn [<prop_classification_chi_consistent_ $dim d>](
                    vertices in full_dimensional_vertices::<$dim>($min_vertices, $max_vertices - 1)
                ) {
                    let dt = DelaunayTriangulation::builder(&vertices)
                        .topology_guarantee(TopologyGuarantee::PLManifold)
                        .build()
                        .map_err(|error| TestCaseError::fail(format!(
                            "{}D construction failed for admitted vertices {vertices:?}: {error:?}", $dim
                        )))?;
                    let result = dt.euler_check()?;

                    // A full-dimensional convex Euclidean cloud is a ball, so
                    // this oracle does not depend on production classification.
                    prop_assert_eq!(result.expected, Some(1), "{}D Euclidean ball classification", $dim);
                    prop_assert_eq!(result.chi, 1, "{}D Euclidean ball Euler characteristic", $dim);
                }
            }
        }
    };
}

#[test]
fn test_seeded_random_generator_euler_consistent() {
    let dt_2d = try_generate_random_triangulation_with_topology::<(), (), 2>(
        nonzero(15),
        (0.0, 10.0),
        None,
        Some(555),
        TopologyGuarantee::PLManifold,
    )
    .unwrap();
    let result_2d = dt_2d.euler_check().unwrap();
    assert!(
        result_2d.is_valid(),
        "2D seeded random triangulation Euler mismatch: χ={}, expected={:?}, classification={:?}, V={}, simplices={}",
        result_2d.chi,
        result_2d.expected,
        result_2d.classification,
        result_2d.counts.count(0),
        result_2d.counts.count(2),
    );

    let dt_3d = try_generate_random_triangulation_with_topology::<(), (), 3>(
        nonzero(20),
        (-3.0, 3.0),
        None,
        Some(666),
        TopologyGuarantee::PLManifold,
    )
    .unwrap();
    let result_3d = dt_3d.euler_check().unwrap();
    assert!(
        result_3d.is_valid(),
        "3D seeded random triangulation Euler mismatch: χ={}, expected={:?}, classification={:?}, V={}, simplices={}",
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
test_euler_properties!(4, 6, 25, #[cfg(feature = "slow-tests")]);
test_euler_properties!(5, 7, 16, #[cfg(feature = "slow-tests")]);
