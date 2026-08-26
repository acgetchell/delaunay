#![forbid(unsafe_code)]

use delaunay::prelude::construction::{DelaunayTriangulation, DelaunayTriangulationBuilder, vertex};
use delaunay::prelude::geometry::RobustKernel;
use serde_json::{from_slice, to_vec};

fn main() {
    let vertices = [
        vertex!([0.0, 0.0]).unwrap(),
        vertex!([0.0, -22.546_422_723_221_383]).unwrap(),
        vertex!([12.252_739_760_228_783, 0.0]).unwrap(),
        vertex!([1.572_332_120_964_092_1, 2.883_274_933_268_964]).unwrap(),
    ];
    let triangulation: DelaunayTriangulation<RobustKernel<f64>, (), (), 2> =
        DelaunayTriangulationBuilder::new(&vertices)
            .build_with_kernel(&RobustKernel::new())
            .unwrap();
    let mut expected = triangulation
        .vertices()
        .map(|(_, vertex)| vertex.point().coords().map(f64::to_bits))
        .collect::<Vec<_>>();
    expected.sort_unstable();

    let json = to_vec(&triangulation).unwrap();
    let restored: DelaunayTriangulation<RobustKernel<f64>, (), (), 2> =
        from_slice(&json).unwrap();
    let mut actual = restored
        .vertices()
        .map(|(_, vertex)| vertex.point().coords().map(f64::to_bits))
        .collect::<Vec<_>>();
    actual.sort_unstable();
    assert_eq!(actual, expected);
    restored.validate().unwrap();
}
