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

use delaunay::prelude::construction::{DelaunayTriangulation, TopologyGuarantee};
use delaunay::prelude::geometry::*;
use delaunay::prelude::insertion::InsertionOutcome;
use delaunay::prelude::tds::{Tds, TdsError};
use delaunay::try_vertices_from_points;
use proptest::prelude::*;

/// Strategy for generating finite `f64` coordinates in a reasonable range.
fn finite_coordinate() -> impl Strategy<Value = f64> {
    (-100.0..100.0).prop_filter("must be finite", |x: &f64| x.is_finite())
}

macro_rules! gen_orientation_construction_and_tamper_props {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal $(, #[$attr:meta])*) => {
        pastey::paste! {
            proptest! {
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
                    if let Ok(dt) = DelaunayTriangulation::builder(&vertices).topology_guarantee(TopologyGuarantee::PLManifold).build() {
                        prop_assert!(
                            dt.is_coherently_oriented(),
                            "{}D: constructed triangulation must be coherently oriented",
                            $dim
                        );
                    }
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
                    if let Ok(dt) = DelaunayTriangulation::builder(&vertices).topology_guarantee(TopologyGuarantee::PLManifold).build() {
                        prop_assume!(dt.number_of_simplices() >= 2);
                        prop_assert!(dt.is_coherently_oriented());

                        let mut serialized = serde_json::to_value(&dt).unwrap();
                        let simplex_vertices_map = serialized
                            .get_mut("simplex_vertices")
                            .and_then(serde_json::Value::as_object_mut)
                            .unwrap();
                        let first_simplex_vertices = simplex_vertices_map
                            .values_mut()
                            .next()
                            .and_then(serde_json::Value::as_array_mut)
                            .unwrap();
                        prop_assume!(first_simplex_vertices.len() >= 2);
                        first_simplex_vertices.swap(0, 1);

                        let tampered_json = serde_json::to_string(&serialized).unwrap();
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
        }
    };
}

macro_rules! gen_orientation_incremental_props {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal $(, #[$attr:meta])*) => {
        pastey::paste! {
            proptest! {
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
                    let mut dt: DelaunayTriangulation<_, (), (), $dim> =
                        DelaunayTriangulation::empty_with_topology_guarantee(
                            TopologyGuarantee::PLManifold,
                        );

                    for vertex in vertices {
                        let result = dt.insert_best_effort_with_statistics(vertex);
                        if let Ok((InsertionOutcome::Inserted { .. }, _stats)) = result {
                            prop_assert!(
                                dt.is_coherently_oriented(),
                                "{}D: orientation must remain coherent after successful insertion",
                                $dim
                            );
                        }
                    }
                }
            }
        }
    };
}

gen_orientation_construction_and_tamper_props!(2, 4, 10);
gen_orientation_construction_and_tamper_props!(3, 5, 12);
gen_orientation_construction_and_tamper_props!(
    4,
    6,
    14,
    #[cfg(feature = "slow-tests")]
);
gen_orientation_construction_and_tamper_props!(
    5,
    7,
    16,
    #[cfg(feature = "slow-tests")]
);

gen_orientation_incremental_props!(2, 4, 10);
gen_orientation_incremental_props!(3, 5, 12);
gen_orientation_incremental_props!(4, 6, 14, #[cfg(feature = "slow-tests")]);
gen_orientation_incremental_props!(5, 7, 16, #[cfg(feature = "slow-tests")]);
