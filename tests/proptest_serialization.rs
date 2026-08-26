//! Property-based tests for serialization/deserialization roundtrips.
//!
//! This module uses proptest to verify that serialization and deserialization
//! preserve all important properties of triangulation structures, including:
//! - Triangulation structure preservation (vertices, simplices, neighbors)
//! - Simplex and vertex data preservation
//! - Triangulation validity after roundtrip
//!
//! Tests are generated for dimensions 2D-5D using macros to reduce duplication.

#[macro_use]
#[path = "common/proptest_config.rs"]
mod proptest_config;

use delaunay::prelude::construction::{DelaunayTriangulation, TopologyGuarantee, Vertex};
use delaunay::prelude::geometry::RobustKernel;
use delaunay::prelude::query::*;
use delaunay::try_vertices_from_points;
use proptest::prelude::*;
use proptest_config::with_default_cases;
use std::collections::HashSet;
use uuid::Uuid;

/// Type alias for the serde round-trip target.
type DefaultDt<const D: usize> = DelaunayTriangulation<RobustKernel<f64>, (), (), D>;

// =============================================================================
// TEST CONFIGURATION
// =============================================================================

/// Strategy for generating finite f64 coordinates
fn finite_coordinate() -> impl Strategy<Value = f64> {
    (-100.0..100.0).prop_filter("must be finite", |x: &f64| x.is_finite())
}

/// Counts coordinate-distinct inputs before construction so round-trip
/// properties reject only insufficient generator data.
fn unique_vertex_count<const D: usize>(vertices: &[Vertex<(), D>]) -> usize {
    vertices
        .iter()
        .map(|vertex| vertex.point().coords().map(f64::to_bits))
        .collect::<HashSet<_>>()
        .len()
}

/// Returns simplex UUIDs and their neighbor UUID slots in stable owner-independent order.
fn neighbor_signatures<K, const D: usize>(
    triangulation: &DelaunayTriangulation<K, (), (), D>,
) -> Vec<(Uuid, Vec<Option<Uuid>>)> {
    let mut signatures = triangulation
        .simplices()
        .map(|(_, simplex)| {
            let neighbors = simplex
                .neighbors()
                .map(|neighbors| {
                    neighbors
                        .map(|neighbor| {
                            neighbor.and_then(|key| triangulation.simplex_uuid_from_key(key))
                        })
                        .collect()
                })
                .unwrap_or_default();
            (simplex.uuid(), neighbors)
        })
        .collect::<Vec<_>>();
    signatures.sort_unstable_by_key(|(uuid, _)| *uuid);
    signatures
}

// =============================================================================
// DIMENSIONAL TEST GENERATION MACROS
// =============================================================================

/// Macro to generate serialization property tests for a given dimension
macro_rules! test_serialization_properties {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal $(, cases = $cases:literal)? $(, #[$attr:meta])*) => {
        pastey::paste! {
            repo_proptest! {
                $(
                    #![proptest_config(with_default_cases($cases))]
                )?

                /// Property: Triangulation structure preserved after JSON roundtrip
                $(#[$attr])*
                #[test]
                fn [<prop_triangulation_json_roundtrip_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| try_vertices_from_points(&v).expect("finite point coordinates"))
                ) {
                    prop_assume!(unique_vertex_count(&vertices) > $dim);
                    let dt = DelaunayTriangulation::builder(&vertices)
                        .topology_guarantee(TopologyGuarantee::PLManifold)
                        .build()
                        .map_err(|error| TestCaseError::fail(format!(
                            "{}D JSON-roundtrip fixture construction failed: {error:?}",
                            $dim,
                        )))?;
                    let expected_manifest = dt.checkpoint_manifest().map_err(|error| {
                        TestCaseError::fail(format!(
                            "{}D source checkpoint manifest construction failed: {error:?}",
                            $dim,
                        ))
                    })?;
                    // Serialize to JSON
                    let json = serde_json::to_string(&dt).map_err(|error| {
                        TestCaseError::fail(format!("{}D serialization failed: {error:?}", $dim))
                    })?;

                    let deserialized: DefaultDt<$dim> = serde_json::from_str(&json).map_err(|error| {
                        TestCaseError::fail(format!("{}D deserialization failed: {error:?}", $dim))
                    })?;

                        // Verify structure preservation
                        prop_assert_eq!(
                            deserialized.number_of_vertices(),
                            dt.number_of_vertices(),
                            "{}D vertex count should be preserved",
                            $dim
                        );
                        prop_assert_eq!(
                            deserialized.number_of_simplices(),
                            dt.number_of_simplices(),
                            "{}D simplex count should be preserved",
                            $dim
                        );
                        prop_assert_eq!(
                            deserialized.dim(),
                            dt.dim(),
                            "{}D dimension should be preserved",
                            $dim
                        );
                        prop_assert_eq!(
                            deserialized.checkpoint_manifest().map_err(|error| {
                                TestCaseError::fail(format!(
                                    "{}D restored checkpoint manifest construction failed: {error:?}",
                                    $dim,
                                ))
                            })?,
                            expected_manifest,
                            "{}D checkpoint manifest should be preserved",
                            $dim
                        );
                }

                /// Property: Deserialized triangulation remains valid
                $(#[$attr])*
                #[test]
                fn [<prop_deserialized_triangulation_valid_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| try_vertices_from_points(&v).expect("finite point coordinates"))
                ) {
                    prop_assume!(unique_vertex_count(&vertices) > $dim);
                    let dt = DelaunayTriangulation::builder(&vertices)
                        .topology_guarantee(TopologyGuarantee::PLManifold)
                        .build()
                        .map_err(|error| TestCaseError::fail(format!(
                            "{}D deserialization-validity fixture construction failed: {error:?}",
                            $dim,
                        )))?;
                    // Serialize and deserialize through the owner checkpoint.
                    let json = serde_json::to_string(&dt).map_err(|error| {
                        TestCaseError::fail(format!("{}D serialization failed: {error:?}", $dim))
                    })?;
                    let deserialized: DefaultDt<$dim> = serde_json::from_str(&json).map_err(|error| {
                        TestCaseError::fail(format!("{}D deserialization failed: {error:?}", $dim))
                    })?;

                        // Check the cumulative Levels 1-5 contract restored by schema v2.
                        let validation = deserialized.validate();
                        prop_assert!(
                            validation.is_ok(),
                            "{}D deserialized triangulation should pass Levels 1-5: {:?}",
                            $dim,
                            validation.err()
                        );
                }

                /// Property: Vertex coordinates preserved after roundtrip
                $(#[$attr])*
                #[test]
                fn [<prop_vertex_coordinates_preserved_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| try_vertices_from_points(&v).expect("finite point coordinates"))
                ) {
                    // Need more than a minimal simplex for meaningful membership coverage.
                    prop_assume!(unique_vertex_count(&vertices) > $dim + 1);
                    let dt = DelaunayTriangulation::builder(&vertices)
                        .topology_guarantee(TopologyGuarantee::PLManifold)
                        .build()
                        .map_err(|error| TestCaseError::fail(format!(
                            "{}D coordinate-roundtrip fixture construction failed: {error:?}",
                            $dim,
                        )))?;

                        // Collect original vertex points
                        let original_points: Vec<_> = dt.vertices()
                            .map(|(_, v)| *v.point())
                            .collect();

                        // Serialize and deserialize through the owner checkpoint.
                        let json = serde_json::to_string(&dt).map_err(|error| {
                            TestCaseError::fail(format!("{}D serialization failed: {error:?}", $dim))
                        })?;
                        let deserialized: DefaultDt<$dim> = serde_json::from_str(&json).map_err(|error| {
                            TestCaseError::fail(format!("{}D deserialization failed: {error:?}", $dim))
                        })?;

                        // Collect deserialized vertex points
                        let deserialized_points: Vec<_> = deserialized.vertices()
                            .map(|(_, v)| *v.point())
                            .collect();

                        // Compare counts
                        prop_assert_eq!(
                            deserialized_points.len(),
                            original_points.len(),
                            "{}D vertex count mismatch",
                            $dim
                        );

                        // Check exact point membership. Schema v2 carries the TDS as bytes,
                        // so the outer JSON codec must preserve every f64 bit.
                        for orig_point in &original_points {
                            let found = deserialized_points.contains(orig_point);
                            prop_assert!(
                                found,
                                "{}D vertex point {:?} not found exactly after roundtrip",
                                $dim,
                                orig_point
                            );
                        }
                }

                /// Property: Neighbor relationships preserved after roundtrip
                $(#[$attr])*
                #[test]
                fn [<prop_neighbor_relationships_preserved_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| try_vertices_from_points(&v).expect("finite point coordinates"))
                ) {
                    prop_assume!(unique_vertex_count(&vertices) > $dim);
                    let dt = DelaunayTriangulation::builder(&vertices)
                        .topology_guarantee(TopologyGuarantee::PLManifold)
                        .build()
                        .map_err(|error| TestCaseError::fail(format!(
                            "{}D neighbor-roundtrip fixture construction failed: {error:?}",
                            $dim,
                        )))?;
                    let original_neighbors = neighbor_signatures(&dt);

                        // Serialize and deserialize through the owner checkpoint.
                    let json = serde_json::to_string(&dt).map_err(|error| {
                        TestCaseError::fail(format!("{}D serialization failed: {error:?}", $dim))
                    })?;
                    let deserialized: DefaultDt<$dim> = serde_json::from_str(&json).map_err(|error| {
                        TestCaseError::fail(format!("{}D deserialization failed: {error:?}", $dim))
                    })?;

                        // Exact UUID-based adjacency should match independently of slotmap keys.
                        prop_assert_eq!(
                            neighbor_signatures(&deserialized),
                            original_neighbors,
                            "{}D neighbor UUID slots should be preserved",
                            $dim
                        );
                }
            }
        }
    };
}

// Generate tests for dimensions 2-5
// Parameters: dimension, min_vertices, max_vertices
test_serialization_properties!(2, 4, 10);
test_serialization_properties!(3, 5, 9, cases = 4);
test_serialization_properties!(4, 6, 14, #[cfg(feature = "slow-tests")]);
test_serialization_properties!(5, 7, 16, #[cfg(feature = "slow-tests")]);
