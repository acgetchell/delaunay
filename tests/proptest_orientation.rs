//! Property-based tests for coherent orientation invariants.
//!
//! This module adds focused orientation coverage for:
//! - successful construction (`tds.is_coherently_oriented() == true`)
//! - orientation tamper detection (`OrientationViolation`)
//! - incremental insertion coherence after each successful insertion
//!
//! Tests are generated for dimensions 2D-5D (with 4D/5D ignored in regular
//! test-integration runs due to runtime).

#![forbid(unsafe_code)]

use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
use delaunay::core::triangulation_data_structure::{Tds, TdsValidationError};
use delaunay::prelude::geometry::*;
use delaunay::prelude::triangulation::*;
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
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|points| Vertex::from_points(&points))
                ) {
                    if let Ok(dt) = DelaunayTriangulation::<_, (), (), $dim>::new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        prop_assert!(
                            dt.tds().is_coherently_oriented(),
                            "{}D: constructed triangulation must be coherently oriented",
                            $dim
                        );
                    }
                }

                /// Property: swapping one cell's vertex order should violate coherent orientation.
                ///
                /// Tampering is done through serialized `cell_vertices` so deserialization rebuilds
                /// neighbors/incidence normally while preserving the orientation corruption.
                $(#[$attr])*
                #[test]
                fn [<prop_orientation_tamper_detected_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|points| Vertex::from_points(&points))
                ) {
                    if let Ok(dt) = DelaunayTriangulation::<_, (), (), $dim>::new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        prop_assume!(dt.tds().number_of_cells() >= 2);
                        prop_assert!(dt.tds().is_coherently_oriented());

                        let mut serialized = serde_json::to_value(dt.tds()).unwrap();
                        let cell_vertices_map = serialized
                            .get_mut("cell_vertices")
                            .and_then(serde_json::Value::as_object_mut)
                            .unwrap();
                        let first_cell_vertices = cell_vertices_map
                            .values_mut()
                            .next()
                            .and_then(serde_json::Value::as_array_mut)
                            .unwrap();
                        prop_assume!(first_cell_vertices.len() >= 2);
                        first_cell_vertices.swap(0, 1);

                        let tampered_json = serde_json::to_string(&serialized).unwrap();
                        let tampered_tds: Tds<f64, (), (), $dim> =
                            serde_json::from_str(&tampered_json).unwrap();

                        prop_assert!(
                            !tampered_tds.is_coherently_oriented(),
                            "{}D: tampered triangulation should not remain coherently oriented",
                            $dim
                        );
                        prop_assert!(
                            matches!(
                                tampered_tds.is_valid(),
                                Err(TdsValidationError::OrientationViolation { .. })
                            ),
                            "{}D: tampered triangulation should fail with OrientationViolation",
                            $dim
                        );
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
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|points| Vertex::from_points(&points))
                ) {
                    let mut dt: DelaunayTriangulation<_, (), (), $dim> =
                        DelaunayTriangulation::empty_with_topology_guarantee(
                            TopologyGuarantee::PLManifold,
                        );

                    for vertex in vertices {
                        let result = dt.insert_with_statistics(vertex);
                        if let Ok((InsertionOutcome::Inserted { .. }, _stats)) = result {
                            prop_assert!(
                                dt.tds().is_coherently_oriented(),
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
    #[ignore = "Slow (>60s) in test-integration"]
);
gen_orientation_construction_and_tamper_props!(
    5,
    7,
    16,
    #[ignore = "Slow (>60s) in test-integration"]
);

gen_orientation_incremental_props!(2, 4, 10);
gen_orientation_incremental_props!(3, 5, 12);
gen_orientation_incremental_props!(4, 6, 14, #[ignore = "Slow (>60s) in test-integration"]);
gen_orientation_incremental_props!(5, 7, 16, #[ignore = "Slow (>60s) in test-integration"]);
