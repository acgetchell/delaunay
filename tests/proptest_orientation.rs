//! Property-based tests for coherent orientation invariants.
//!
//! This module adds focused orientation coverage for:
//! - successful construction (`tds.is_coherently_oriented() == true`)
//! - orientation tamper detection (`OrientationViolation` or earlier neighbor-consistency rejection)
//! - incremental insertion coherence after each successful insertion
//!
//! Tests are generated for dimensions 2D-5D (with 4D/5D ignored in regular
//! test-integration runs due to runtime).

#![forbid(unsafe_code)]

#[macro_use]
#[path = "common/proptest_config.rs"]
mod proptest_config;

use delaunay::prelude::construction::{
    DelaunayIncrementalBuilder, DelaunayTriangulation, TopologyGuarantee, Vertex,
};
use delaunay::prelude::geometry::*;
use delaunay::prelude::insertion::InsertionOutcome;
use delaunay::prelude::tds::{Tds, TdsError};
use delaunay::try_vertices_from_points;
use proptest::prelude::*;
use std::collections::HashSet;

/// Strategy for generating finite `f64` coordinates in a reasonable range.
fn finite_coordinate() -> impl Strategy<Value = f64> {
    (-100.0..100.0).prop_filter("must be finite", |x: &f64| x.is_finite())
}

/// Counts coordinate-distinct inputs before construction so tamper-only
/// properties reject only generator-domain insufficiency.
fn unique_vertex_count<const D: usize>(vertices: &[Vertex<(), D>]) -> usize {
    vertices
        .iter()
        .map(|vertex| vertex.point().coords().map(f64::to_bits))
        .collect::<HashSet<_>>()
        .len()
}

macro_rules! gen_orientation_construction_and_tamper_props {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal $(, #[$attr:meta])*) => {
        pastey::paste! {
            repo_proptest! {
                /// Property: every successfully constructed triangulation is coherently oriented.
                $(#[$attr])*
                #[test]
                fn [<prop_orientation_coherent_after_construction_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                        $min_vertices..=$max_vertices
                    ).prop_map(|points| {
                        try_vertices_from_points(&points).expect("finite point coordinates")
                    })
                ) {
                    let dt = DelaunayTriangulation::builder(&vertices)
                        .topology_guarantee(TopologyGuarantee::PLManifold)
                        .build()
                        .map_err(|error| TestCaseError::fail(format!(
                            "{}D coherent-orientation fixture construction failed: {error:?}",
                            $dim,
                        )))?;
                    prop_assert!(
                        dt.is_coherently_oriented(),
                        "{}D: constructed triangulation must be coherently oriented",
                        $dim
                    );
                }

                /// Property: swapping one simplex's vertex order should violate TDS structure.
                ///
                /// Tampering is done through serialized `simplex_vertices` so deserialization rebuilds
                /// incidence while preserving the corrupted UUID topology. Depending on the generated
                /// triangulation, preserving serialized neighbor UUIDs may expose neighbor inconsistency
                /// before the orientation check runs.
                $(#[$attr])*
                #[test]
                fn [<prop_orientation_tamper_detected_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                        $min_vertices..=$max_vertices
                    ).prop_map(|points| {
                        try_vertices_from_points(&points).expect("finite point coordinates")
                    })
                ) {
                    prop_assume!(unique_vertex_count(&vertices) > $dim + 1);
                    let dt = DelaunayTriangulation::builder(&vertices)
                        .topology_guarantee(TopologyGuarantee::PLManifold)
                        .build()
                        .map_err(|error| TestCaseError::fail(format!(
                            "{}D orientation-tamper fixture construction failed: {error:?}",
                            $dim,
                        )))?;
                        prop_assert!(dt.number_of_simplices() >= 2);
                        prop_assert!(dt.is_coherently_oriented());

                        let tds = dt.into_triangulation().into_tds();
                        let mut serialized = serde_json::to_value(tds).map_err(|error| {
                            TestCaseError::fail(format!(
                                "{}D orientation-tamper serialization failed: {error:?}",
                                $dim,
                            ))
                        })?;
                        let simplex_vertices_map = serialized
                            .get_mut("simplex_vertices")
                            .and_then(serde_json::Value::as_object_mut)
                            .unwrap();
                        let first_simplex_vertices = simplex_vertices_map
                            .values_mut()
                            .next()
                            .and_then(serde_json::Value::as_array_mut)
                            .unwrap();
                        prop_assert!(first_simplex_vertices.len() >= 2);
                        first_simplex_vertices.swap(0, 1);

                        let tampered_json = serde_json::to_string(&serialized).map_err(|error| {
                            TestCaseError::fail(format!(
                                "{}D orientation-tamper JSON encoding failed: {error:?}",
                                $dim,
                            ))
                        })?;
                        match serde_json::from_str::<Tds<(), (), $dim>>(&tampered_json) {
                            Ok(tampered_tds) => {
                                prop_assert!(
                                    !tampered_tds.is_coherently_oriented(),
                                    "{}D: tampered triangulation should not remain coherently oriented",
                                    $dim
                                );
                                prop_assert!(
                                    matches!(
                                        tampered_tds.is_valid(),
                                        Err(
                                            TdsError::OrientationViolation { .. }
                                                | TdsError::InvalidNeighbors { .. }
                                        )
                                    ),
                                    "{}D: tampered triangulation should fail structural TDS validation",
                                    $dim
                                );
                            }
                            Err(error) => {
                                let message = error.to_string();
                                prop_assert!(
                                    message.contains("Orientation invariant violated")
                                        || message.contains("Invalid neighbor relationships"),
                                    "{}D: tampered triangulation should be rejected by structural TDS validation, got {error}",
                                    $dim
                                );
                            }
                        }
                }
            }
        }
    };
}

macro_rules! gen_orientation_incremental_props {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal $(, #[$attr:meta])*) => {
        pastey::paste! {
            repo_proptest! {
                /// Property: after each successful insertion, orientation remains coherent.
                $(#[$attr])*
                #[test]
                fn [<prop_orientation_coherent_after_each_successful_insert_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                        $min_vertices..=$max_vertices
                    ).prop_map(|points| {
                        try_vertices_from_points(&points).expect("finite point coordinates")
                    })
                ) {
                    let mut dt: DelaunayIncrementalBuilder<_, (), (), $dim> =
                        DelaunayIncrementalBuilder::with_topology_guarantee(
                            TopologyGuarantee::PLManifold,
                        );

                    for vertex in vertices {
                        let result = dt.insert_best_effort_with_statistics(vertex);
                        match result {
                            Ok((InsertionOutcome::Inserted { .. }, _stats)) => {
                                prop_assert!(
                                    dt.is_coherently_oriented(),
                                    "{}D: orientation must remain coherent after successful insertion",
                                    $dim
                                );
                            }
                            Ok((InsertionOutcome::Skipped { .. }, _stats)) => {}
                            Err(error) => {
                                return Err(TestCaseError::fail(format!(
                                    "{}D incremental orientation insertion failed: {error:?}",
                                    $dim,
                                )));
                            }
                        }
                    }
                }
            }
        }
    };
}

gen_orientation_construction_and_tamper_props!(2, 4, 10);
gen_orientation_construction_and_tamper_props!(3, 5, 12);
gen_orientation_construction_and_tamper_props!(4, 6, 14);
gen_orientation_construction_and_tamper_props!(
    5,
    7,
    16,
    #[cfg(feature = "slow-tests")]
);

gen_orientation_incremental_props!(2, 4, 10);
gen_orientation_incremental_props!(3, 5, 12);
gen_orientation_incremental_props!(4, 6, 14);
gen_orientation_incremental_props!(5, 7, 16, #[cfg(feature = "slow-tests")]);
