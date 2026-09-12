//! Coordinate-admitted inputs for Euclidean hull and Euler properties.

use delaunay::prelude::construction::Vertex;
use delaunay::vertex;
use proptest::prelude::*;

/// Keeps an exact full-dimensional basis while varying the surrounding cloud.
///
/// The origin and the scaled coordinate axes prove full dimensionality without
/// consulting construction or predicate results. Duplicate coordinates are
/// rejected before constructing vertices, including equivalent signed zeros.
pub fn full_dimensional_vertices<const D: usize>(
    min_vertices: usize,
    max_vertices: usize,
) -> impl Strategy<Value = Vec<Vertex<(), D>>> {
    assert!(min_vertices > D + 1);
    assert!(max_vertices >= min_vertices);

    prop::collection::vec(
        prop::collection::vec(-100.0_f64..100.0, D),
        (min_vertices - D - 1)..=(max_vertices - D - 1),
    )
    .prop_map(|extra_points| {
        let mut points = vec![[0.0; D]];
        for axis in 0..D {
            let mut point = [0.0; D];
            point[axis] = 100.0;
            points.push(point);
        }
        points.extend(
            extra_points
                .into_iter()
                .map(|coords| std::array::from_fn(|axis| coords[axis])),
        );
        points
    })
    .prop_filter("coordinate-distinct points", |points| {
        points
            .iter()
            .enumerate()
            .all(|(index, point)| !points[..index].contains(point))
    })
    .prop_map(|points| {
        points
            .into_iter()
            .map(|coords| vertex!(coords).expect("generated coordinates are finite"))
            .collect()
    })
}
