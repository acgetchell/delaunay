//! Property-based tests for documented invariants (src/lib.rs).
//! - Duplicate coordinate rejection
//! - Cell vertex count (D+1)

#![expect(deprecated)] // Tests use deprecated Tds::new() and tds.add() until migration to DelaunayTriangulation

use delaunay::core::triangulation_data_structure::{Tds, TriangulationConstructionError};
use delaunay::core::vertex::Vertex;
use delaunay::geometry::point::Point;
use delaunay::geometry::traits::coordinate::Coordinate;
use proptest::prelude::*;

// Strategy: finite coordinate range
fn finite_coordinate() -> impl Strategy<Value = f64> {
    (-100.0..100.0).prop_filter("must be finite", |x: &f64| x.is_finite())
}

// Macro: duplicate coordinate rejection for dimension $dim
macro_rules! gen_duplicate_coords_test {
    ($dim:literal, $min:literal, $max:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_duplicate_coordinates_rejected_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min..=$max
                    ).prop_map(|v| Vertex::from_points(&v))
                ) {
                    if let Ok(mut tds) = Tds::<f64, (), (), $dim>::new(&vertices) {
                        // Select a vertex that is actually present in the triangulation.
                        // `Tds::new` may skip some input vertices (e.g., due to degeneracy),
                        // so we must use stored vertices to test duplicate rejection.
                        let (_, existing_vertex) = tds
                            .vertices()
                            .next()
                            .expect("Tds::new returned Ok but has no vertices");
                        let p = *existing_vertex.point();
                        let dup = Vertex::from_points(&[p])[0];
                        let result = tds.add(dup);
                        prop_assert!(
                            matches!(result, Err(TriangulationConstructionError::DuplicateCoordinates { .. })),
                            "expected DuplicateCoordinates error, got {result:?}"
                        );
                    }
                }
            }
        }
    };
}

// Macro: cell vertex count is D+1 for dimension $dim
macro_rules! gen_cell_vertex_count_test {
    ($dim:literal, $min:literal, $max:literal, $expected:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_cell_vertex_count_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min..=$max
                    ).prop_map(|v| Vertex::from_points(&v))
                ) {
                    if let Ok(tds) = Tds::<f64, (), (), $dim>::new(&vertices) {
                        for (_, c) in tds.cells() {
                            prop_assert_eq!(c.number_of_vertices(), $expected);
                        }
                    }
                }
            }
        }
    };
}

// Instantiate tests for 2D–5D
// Duplicate coordinate rejection
gen_duplicate_coords_test!(2, 3, 10);
gen_duplicate_coords_test!(3, 4, 12);
gen_duplicate_coords_test!(4, 5, 14);
gen_duplicate_coords_test!(5, 6, 16);

// Cell vertex count (D+1)
gen_cell_vertex_count_test!(2, 3, 10, 3);
gen_cell_vertex_count_test!(3, 4, 12, 4);
gen_cell_vertex_count_test!(4, 5, 14, 5);
gen_cell_vertex_count_test!(5, 6, 16, 6);
