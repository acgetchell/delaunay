//! Edge case and regression tests for `DelaunayTriangulation`.
//!
//! These tests cover:
//! - Known regression configurations (previously failing cases)
//! - Degenerate configurations (collinear, coplanar points)
//! - Extreme coordinate values
//! - Edge cases from legacy Tds tests
//!
//! Converted from legacy `Tds::new()` tests to use the new `DelaunayTriangulation` API.

use delaunay::geometry::util::generate_random_triangulation;
use delaunay::prelude::*;

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
test_regression_config!(
    regression_2d_canonical,
    2,
    vec![
        vertex!([0.0, 0.0]),
        vertex!([2.0, 0.0]),
        vertex!([0.0, 2.0]),
        vertex!([0.8, 0.7]),   // interior
        vertex!([-0.5, -0.4]), // exterior
    ]
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
    ]
);

// =========================================================================
// Regression Tests - Non-Manifold Topology During Incremental Insertion
// =========================================================================

/// Regression test for CI benchmark failure with non-manifold topology.
///
/// This test reproduces a specific failure discovered in the CI benchmark suite
/// where incremental insertion created temporary non-manifold topology (facet
/// shared by 4 cells). The fix made `wire_cavity_neighbors` tolerant of this
/// condition, allowing the localized repair mechanism to handle it.
///
/// ## Historical Context
///
/// - **When**: Discovered in CI run 19874972833 (2025-12-02)
/// - **Where**: `benches/ci_performance_suite.rs:85` (3D, 50 points, seed 123)
/// - **Error**: "Non-manifold topology detected: facet 11030659497163937569
///   shared by 4 cells (expected ≤2)"
/// - **Fix**: Modified `wire_cavity_neighbors` to skip wiring over-shared facets
///   and rely on `repair_local_facet_issues` to fix them post-insertion
///
/// ## Configuration
///
/// Matches CI benchmark configuration exactly:
/// - Dimension: 3D
/// - Point count: 50 (matches `COUNTS` in `ci_performance_suite.rs`)
/// - Seed: 123 (from `benchmark_tds_new_dimension!(3, benchmark_tds_new_3d, 123)`)
/// - Bounds: (-100.0, 100.0)
#[test]
fn test_regression_non_manifold_3d_seed123_50pts() {
    // Exact configuration from CI failure (matches ci_performance_suite.rs)
    let result = generate_random_triangulation::<f64, (), (), 3>(
        50,              // Point count from CI benchmark
        (-100.0, 100.0), // Bounds from benchmark
        None,            // No vertex data
        Some(123),       // Seed from benchmark (line 85 in ci_performance_suite.rs)
    );

    // Should succeed now that wire_cavity_neighbors is tolerant of non-manifold topology
    assert!(
        result.is_ok(),
        "Failed to generate 3D triangulation with 50 points (seed 123): {:?}",
        result.err()
    );

    let dt = result.unwrap();

    // Verify basic properties
    // Note: Some vertices may be skipped due to geometric degeneracy
    let num_vertices = dt.number_of_vertices();
    assert!(
        num_vertices <= 50,
        "Should have ≤50 vertices, got {num_vertices}"
    );
    assert!(
        num_vertices >= 20,
        "Should have ≥20 vertices (extremely degenerate cases can skip 60%+), got {num_vertices}"
    );
    assert!(dt.number_of_cells() > 0);

    // Most importantly: validate topology (Levels 1–3: elements + structure + manifold)
    let validation = dt.triangulation().validate();
    assert!(
        validation.is_ok(),
        "Triangulation has topology violations: {:?}",
        validation.err()
    );
}

/// Stress test: Multiple seeds around the problematic configuration.
///
/// Tests seeds near 123 to ensure robustness across similar random configurations.
/// Some may also trigger temporary non-manifold conditions during insertion.
#[test]
fn test_regression_non_manifold_nearby_seeds() {
    let test_seeds = [120, 121, 122, 123, 124, 125, 126];

    for seed in test_seeds {
        let result =
            generate_random_triangulation::<f64, (), (), 3>(50, (-100.0, 100.0), None, Some(seed));

        assert!(
            result.is_ok(),
            "Failed with seed {}: {:?}",
            seed,
            result.err()
        );

        let dt = result.unwrap();
        let num_vertices = dt.number_of_vertices();
        assert!(
            num_vertices <= 50,
            "Seed {seed}: too many vertices ({num_vertices})"
        );
        assert!(
            num_vertices >= 20,
            "Seed {seed}: too few vertices ({num_vertices}), degenerate cases can skip 60%+"
        );
        let validation = dt.triangulation().validate();
        assert!(
            validation.is_ok(),
            "Seed {}: topology violations: {:?}",
            seed,
            validation.err()
        );
    }
}

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
// Edge Cases - Degenerate Configurations
// =========================================================================

#[test]
fn test_collinear_points_2d() {
    // All points lie on a line in 2D: no non-degenerate simplex exists.
    let collinear = vec![
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([2.0, 0.0]),
        vertex!([3.0, 0.0]),
    ];

    let result: Result<DelaunayTriangulation<_, (), (), 2>, _> =
        DelaunayTriangulation::new(&collinear);

    // Verify it fails with GeometricDegeneracy due to collinear simplex
    assert!(
        matches!(
            result,
            Err(DelaunayTriangulationConstructionError::Triangulation(
                TriangulationConstructionError::GeometricDegeneracy { .. },
            ))
        ),
        "Expected GeometricDegeneracy error for collinear points, got: {result:?}"
    );
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
