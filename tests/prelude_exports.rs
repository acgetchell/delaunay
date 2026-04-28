//! Public prelude smoke tests.
//!
//! These tests intentionally use focused preludes instead of module-internal
//! paths so doctests, integration tests, examples, and benchmarks have a small
//! import contract to copy from.

use delaunay::prelude::generators::generate_random_points_seeded;
use delaunay::prelude::geometry::{AdaptiveKernel, Point};
use delaunay::prelude::query::ConvexHull;
use delaunay::prelude::triangulation::flips::{BistellarFlips, TopologyGuarantee};
use delaunay::prelude::triangulation::{
    ConstructionOptions, DelaunayTriangulation, InsertionOrderStrategy, Vertex,
};
use delaunay::vertex;

const fn assert_bistellar_flips(_: &impl BistellarFlips<AdaptiveKernel<f64>, (), (), 3>) {}

#[test]
fn preludes_cover_bench_apis() {
    let _generated_points: Vec<Point<f64, 2>> =
        generate_random_points_seeded(3, (0.0, 1.0), 42).unwrap();

    let vertices: Vec<Vertex<f64, (), 3>> = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];
    let options =
        ConstructionOptions::default().with_insertion_order(InsertionOrderStrategy::Input);
    let dt = DelaunayTriangulation::new_with_options(&vertices, options).unwrap();

    assert_eq!(dt.topology_guarantee(), TopologyGuarantee::PLManifold);
    assert!(dt.boundary_facets().count() > 0);
    assert!(ConvexHull::from_triangulation(dt.as_triangulation()).is_ok());
    assert!(dt.validate().is_ok());
    assert_bistellar_flips(&dt);
}
