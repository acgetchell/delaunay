//! Edge case and regression tests for `DelaunayTriangulation`.
//!
//! These tests cover:
//! - Known regression configurations (previously failing cases)
//! - Degenerate configurations (collinear, coplanar points)
//! - Extreme coordinate values
//! - Edge cases from legacy Tds tests
//!
//! Converted from legacy `Tds::new()` tests to use the new `DelaunayTriangulation` API.

use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
use delaunay::vertex;

// =========================================================================
// Regression Tests - Known Failing Configurations
// =========================================================================
// These configurations previously caused Delaunay validation failures
// and must remain valid across releases.

/// Macro to generate regression tests across dimensions using `DelaunayTriangulation`.
macro_rules! test_regression_config {
    ($name:ident, $dim:expr, $vertices:expr) => {
        pastey::paste! {
            #[test]
            fn [<test_ $name>]() {
                let vertices = $vertices;

                let dt: DelaunayTriangulation<_, (), (), $dim> =
                    DelaunayTriangulation::new(&vertices)
                        .unwrap_or_else(|err| {
                            panic!(
                                "{}D regression configuration failed to construct: {err}",
                                $dim
                            )
                        });

                // Verify basic properties
                assert_eq!(dt.number_of_vertices(), vertices.len());
                assert!(dt.number_of_cells() > 0);
            }
        }
    };
    ($name:ident, $dim:expr, $vertices:expr, ignore = $reason:expr) => {
        pastey::paste! {
            #[test]
            #[ignore = $reason]
            fn [<test_ $name>]() {
                let vertices = $vertices;

                let dt: DelaunayTriangulation<_, (), (), $dim> =
                    DelaunayTriangulation::new(&vertices)
                        .unwrap_or_else(|err| {
                            panic!(
                                "{}D regression configuration failed to construct: {err}",
                                $dim
                            )
                        });

                // Verify basic properties
                assert_eq!(dt.number_of_vertices(), vertices.len());
                assert!(dt.number_of_cells() > 0);
            }
        }
    };
}

// 2D regression: base triangle with interior and exterior point
// NOTE: Ignored - requires hull extension (exterior point)
test_regression_config!(
    regression_2d_canonical,
    2,
    vec![
        vertex!([0.0, 0.0]),
        vertex!([2.0, 0.0]),
        vertex!([0.0, 2.0]),
        vertex!([0.8, 0.7]),   // interior
        vertex!([-0.5, -0.4]), // exterior
    ],
    ignore = "Requires hull extension for exterior points"
);

// 3D regression: tetrahedron with interior point
test_regression_config!(
    regression_3d_canonical,
    3,
    vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([2.0, 0.0, 0.0]),
        vertex!([0.0, 2.0, 0.0]),
        vertex!([0.0, 0.0, 2.0]),
        vertex!([0.4, 0.4, 0.3]), // interior
    ]
);

// 4D regression: 4-simplex with interior point
test_regression_config!(
    regression_4d_canonical,
    4,
    vec![
        vertex!([0.0, 0.0, 0.0, 0.0]),
        vertex!([2.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 2.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 2.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 2.0]),
        vertex!([0.4, 0.3, 0.3, 0.3]), // interior
    ]
);

// 5D regression: known configuration that previously failed
// NOTE: Ignored - requires hull extension
test_regression_config!(
    regression_5d_known_config,
    5,
    vec![
        vertex!([
            61.994_906_139_357_86,
            66.880_064_158_234_8,
            62.542_871_273_730_91,
            -27.857_784_980_103_375,
            -78.369_282_526_711_23,
        ]),
        vertex!([
            -31.430_765_957_270_268,
            50.418_208_939_604_746,
            88.657_219_404_750_96,
            47.248_786_623_931_88,
            -81.163_199_600_681_14,
        ]),
        vertex!([
            -89.902_834_998_758_96,
            93.719_989_121_636_87,
            64.524_277_928_893_98,
            40.001_314_184_454_05,
            14.196_053_554_411_321,
        ]),
        vertex!([
            2.625_958_385_925_883,
            48.251_155_688_054_36,
            3.491_542_746_106_750_5,
            97.241_732_043_079_37,
            -27.107_939_334_194_757,
        ]),
        vertex!([
            62.628_856_831_188_11,
            -18.181_728_263_486_345,
            -32.153_141_689_537_584,
            25.692_809_519_458_7,
            26.369_541_091_117_114,
        ]),
        vertex!([
            -41.886_149_523_644_406,
            -54.537_563_736_672_65,
            -54.555_379_092_740_964,
            75.499_924_758_912_23,
            16.127_546_041_675_355,
        ]),
        vertex!([
            -77.161_459_173_963_2,
            -59.065_517_574_769_37,
            -19.652_689_679_369_03,
            -51.622_382_706_243_18,
            -26.000_263_271_298_543,
        ]),
    ],
    ignore = "Requires hull extension - some points outside initial simplex"
);

// =========================================================================
// Edge Cases - Minimal Configurations
// =========================================================================

#[test]
fn test_exact_minimum_vertices_2d() {
    // Exactly D+1 = 3 vertices for 2D
    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
    ];

    let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&vertices).unwrap();

    assert_eq!(dt.number_of_vertices(), 3);
    assert_eq!(dt.number_of_cells(), 1);
}

#[test]
fn test_exact_minimum_vertices_3d() {
    // Exactly D+1 = 4 vertices for 3D
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];

    let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();

    assert_eq!(dt.number_of_vertices(), 4);
    assert_eq!(dt.number_of_cells(), 1);
}

#[test]
fn test_exact_minimum_vertices_4d() {
    // Exactly D+1 = 5 vertices for 4D
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0]),
    ];

    let dt: DelaunayTriangulation<_, (), (), 4> = DelaunayTriangulation::new(&vertices).unwrap();

    assert_eq!(dt.number_of_vertices(), 5);
    assert_eq!(dt.number_of_cells(), 1);
}

// =========================================================================
// Edge Cases - Interior/Exterior Points
// =========================================================================

#[test]
fn test_multiple_interior_points_2d() {
    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([4.0, 0.0]),
        vertex!([2.0, 4.0]),
        vertex!([1.5, 1.5]), // interior
        vertex!([2.0, 1.5]), // interior
        vertex!([2.5, 1.5]), // interior
    ];

    let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&vertices).unwrap();

    assert_eq!(dt.number_of_vertices(), 6);
    assert!(dt.number_of_cells() >= 4);
}

#[test]
fn test_multiple_interior_points_3d() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([2.0, 0.0, 0.0]),
        vertex!([0.0, 2.0, 0.0]),
        vertex!([0.0, 0.0, 2.0]),
        vertex!([0.5, 0.5, 0.5]), // interior
        vertex!([0.6, 0.5, 0.5]), // interior
        vertex!([0.5, 0.6, 0.5]), // interior
    ];

    let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();

    assert_eq!(dt.number_of_vertices(), 7);
    assert!(dt.number_of_cells() >= 4);
}

// =========================================================================
// Edge Cases - Symmetric Configurations
// =========================================================================

#[test]
#[ignore = "Requires hull extension - square vertices may be outside initial simplex"]
fn test_square_with_center_2d() {
    // Square vertices with center point
    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([2.0, 0.0]),
        vertex!([2.0, 2.0]),
        vertex!([0.0, 2.0]),
        vertex!([1.0, 1.0]), // center
    ];

    let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&vertices).unwrap();

    assert_eq!(dt.number_of_vertices(), 5);
    assert!(dt.number_of_cells() >= 4);
}

#[test]
#[ignore = "Geometric degeneracy - cube corners are coplanar in sets of 4"]
fn test_cube_vertices_3d() {
    // 8 corners of a unit cube
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([1.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
        vertex!([1.0, 0.0, 1.0]),
        vertex!([0.0, 1.0, 1.0]),
        vertex!([1.0, 1.0, 1.0]),
    ];

    let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();

    assert_eq!(dt.number_of_vertices(), 8);
    assert!(dt.number_of_cells() >= 5); // At least 5 tetrahedra
}

// =========================================================================
// Edge Cases - Varying Scales
// =========================================================================

#[test]
fn test_large_coordinates_2d() {
    let vertices = vec![
        vertex!([1000.0, 1000.0]),
        vertex!([2000.0, 1000.0]),
        vertex!([1500.0, 2000.0]),
        vertex!([1500.0, 1500.0]),
    ];

    let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&vertices).unwrap();

    assert_eq!(dt.number_of_vertices(), 4);
    assert!(dt.number_of_cells() > 0);
}

#[test]
#[ignore = "Requires hull extension - interior point outside initial simplex"]
fn test_small_coordinates_3d() {
    let vertices = vec![
        vertex!([0.001, 0.001, 0.001]),
        vertex!([0.002, 0.001, 0.001]),
        vertex!([0.001, 0.002, 0.001]),
        vertex!([0.001, 0.001, 0.002]),
        vertex!([0.0015, 0.0015, 0.0015]),
    ];

    let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();

    assert_eq!(dt.number_of_vertices(), 5);
    assert!(dt.number_of_cells() > 0);
}

#[test]
fn test_negative_coordinates_2d() {
    let vertices = vec![
        vertex!([-1.0, -1.0]),
        vertex!([1.0, -1.0]),
        vertex!([0.0, 1.0]),
        vertex!([0.0, 0.0]),
    ];

    let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&vertices).unwrap();

    assert_eq!(dt.number_of_vertices(), 4);
    assert!(dt.number_of_cells() > 0);
}

// =========================================================================
// Edge Cases - Different Kernels
// =========================================================================

#[test]
fn test_robust_kernel_with_edge_case() {
    use delaunay::geometry::kernel::RobustKernel;

    // Configuration that might benefit from robust predicates
    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.5, 0.866_025_403_784_438_7]), // ~sqrt(3)/2 - near-equilateral
        vertex!([0.5, 0.5]),
    ];

    let dt: DelaunayTriangulation<RobustKernel<f64>, (), (), 2> =
        DelaunayTriangulation::with_kernel(RobustKernel::new(), &vertices).unwrap();

    assert_eq!(dt.number_of_vertices(), 4);
    assert!(dt.number_of_cells() > 0);
}

// =========================================================================
// Edge Cases - Higher Dimensions
// =========================================================================

#[test]
fn test_5d_simplex_plus_interior() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 0.0, 1.0]),
        vertex!([0.2, 0.2, 0.2, 0.2, 0.2]), // interior
    ];

    let dt: DelaunayTriangulation<_, (), (), 5> = DelaunayTriangulation::new(&vertices).unwrap();

    assert_eq!(dt.number_of_vertices(), 7);
    assert!(dt.number_of_cells() > 1);
}
