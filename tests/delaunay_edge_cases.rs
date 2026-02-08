//! Edge case and regression tests for `DelaunayTriangulation`.
//!
//! These tests cover:
//! - Known regression configurations (previously failing cases)
//! - Degenerate configurations (collinear, coplanar points)
//! - Extreme coordinate values
//! - Edge cases from legacy Tds tests
//!
//! Converted from legacy `Tds::new()` tests to use the new `DelaunayTriangulation` API.

use delaunay::geometry::kernel::RobustKernel;
use delaunay::prelude::triangulation::*;
use rand::SeedableRng;
use rand::seq::SliceRandom;
fn init_tracing() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_test_writer()
            .try_init();
    });
}

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
                    DelaunayTriangulation::new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    )
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
                    DelaunayTriangulation::new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    )
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

#[test]
#[expect(clippy::collapsible_if)]
#[expect(clippy::too_many_lines)]
#[expect(clippy::unreadable_literal)]
fn debug_issue_120_empty_circumsphere_5d() {
    init_tracing();
    let vertices = vec![
        vertex!([
            18.781125710207355,
            85.19556603270544,
            -35.577425948458725,
            -78.4710254162274,
            25.721771703577573
        ]),
        vertex!([
            -21.95447633622051,
            83.05734190480365,
            96.97214006048821,
            -48.80161083192332,
            -21.250997394474208
        ]),
        vertex!([
            50.96615929339812,
            -12.888014856181814,
            64.35842847516192,
            81.20742517692801,
            -67.85330902948604
        ]),
        vertex!([
            -74.19363960080797,
            -87.39864134220277,
            -31.002590635557322,
            -73.43717909637807,
            38.224369898650814
        ]),
        vertex!([
            -15.305719099426401,
            37.44773385928881,
            -31.57846617007415,
            -11.413274473891796,
            32.32927241254111
        ]),
        vertex!([
            -58.19271902837987,
            -50.763430349360824,
            72.37200252994022,
            66.28041332398725,
            53.51398806010464
        ]),
        vertex!([
            96.64514856949884,
            69.99880120063219,
            29.117126126375382,
            88.1850558085571,
            95.34623469752856
        ]),
    ];

    let mut dt: DelaunayTriangulation<_, (), (), 5> =
        DelaunayTriangulation::new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap_or_else(|err| panic!("5D debug configuration failed to construct: {err}"));
    match dt.repair_delaunay_with_flips() {
        Ok(stats) => {
            eprintln!(
                "[Issue #120 debug] repair_delaunay_with_flips stats: checked={}, flips={}, max_queue={}",
                stats.facets_checked, stats.flips_performed, stats.max_queue_len
            );
        }
        Err(err) => {
            eprintln!("[Issue #120 debug] repair_delaunay_with_flips error: {err}");
        }
    }
    let mut dt_robust: DelaunayTriangulation<RobustKernel<f64>, (), (), 5> =
        DelaunayTriangulation::from_tds_with_topology_guarantee(
            dt.tds().clone(),
            RobustKernel::new(),
            TopologyGuarantee::PLManifold,
        );
    match dt_robust.repair_delaunay_with_flips() {
        Ok(stats) => {
            eprintln!(
                "[Issue #120 debug] robust repair stats: checked={}, flips={}, max_queue={}",
                stats.facets_checked, stats.flips_performed, stats.max_queue_len
            );
        }
        Err(err) => {
            eprintln!("[Issue #120 debug] robust repair error: {err}");
        }
    }
    if let Err(err) = dt_robust.is_valid() {
        eprintln!("[Issue #120 debug] robust triangulation still invalid: {err:?}");
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x1200_5eed);
    for attempt in 0..20 {
        let mut shuffled = vertices.clone();
        shuffled.shuffle(&mut rng);
        if let Ok(dt_alt) = DelaunayTriangulation::<_, (), (), 5>::new_with_topology_guarantee(
            &shuffled,
            TopologyGuarantee::PLManifold,
        ) {
            if dt_alt.is_valid().is_ok() {
                eprintln!(
                    "[Issue #120 debug] found valid triangulation after shuffle attempt {}",
                    attempt + 1
                );
                break;
            }
        }
        if attempt == 19 {
            eprintln!("[Issue #120 debug] no valid triangulation found in 20 shuffles");
        }
    }
    for (cell_key, cell) in dt.cells() {
        eprintln!("[Issue #120 debug] cell {cell_key:?}:");
        for &vkey in cell.vertices() {
            let vertex = dt
                .tds()
                .get_vertex_by_key(vkey)
                .expect("vertex key should exist");
            eprintln!(
                "  vkey={vkey:?}, uuid={}, point={:?}",
                vertex.uuid(),
                vertex.point()
            );
        }
    }

    if let Err(err) = dt.is_valid() {
        #[cfg(any(test, debug_assertions))]
        {
            delaunay::core::util::debug_print_first_delaunay_violation(dt.tds(), None);
        }
        panic!("5D debug configuration violates Delaunay property: {err:?}");
    }
}

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

/// Regression: 4D construction must not produce a manifold-with-boundary "shell" (χ=0).
///
/// This configuration was found by `proptest_delaunay_triangulation::prop_insertion_order_robustness_4d`
/// and produced an Euler characteristic mismatch:
/// - expected Ball(4) χ = 1
/// - computed χ = 0
///
/// The triangulation must either:
/// - construct a valid manifold ball (Levels 1–3 pass), or
/// - reject/skip degeneracies without leaving the structure topologically invalid.
#[test]
#[ignore = "Regression test for 4D Euler characteristic mismatch"]
fn test_regression_proptest_insertion_order_4d_euler_mismatch() {
    let vertices = vec![
        vertex!([
            -65.070_532_013_377_94,
            72.223_880_592_145_24,
            -97.837_333_303_337_39,
            35.700_988_360_396_63,
        ]),
        vertex!([
            27.011_682_298_030_294,
            52.054_594_213_988_53,
            -4.067_357_689_604_181,
            59.253_713_038_817_09,
        ]),
        vertex!([
            64.848_601_083_080_47,
            84.907_367_805_317_42,
            -98.659_828_418_664_1,
            70.056_498_821_543_85,
        ]),
        vertex!([
            -23.543_852_823_069_876,
            96.741_963_200_207_22,
            -50.539_503_136_092_634,
            -49.616_262_314_856_67,
        ]),
        vertex!([
            24.886_830_567_772_98,
            -81.708_725_314_824,
            50.775_700_870_880_27,
            20.281_603_779_436_033,
        ]),
        vertex!([
            78.030_479_318_165_25,
            -82.763_788_627_520_88,
            94.075_337_487_756_27,
            44.637_774_779_142_73,
        ]),
        vertex!([
            -5.175_491_150_708_228,
            97.527_582_084_288_54,
            95.344_552_027_220_42,
            84.908_292_808_161_85,
        ]),
        vertex!([
            71.994_788_686_588_11,
            4.833_973_465_666_131,
            -80.802_685_728_835_39,
            64.010_634_775_159_37,
        ]),
        vertex!([
            23.390_079_186_814_17,
            -2.157_395_632_040_824_7,
            -6.601_766_119_999_574,
            32.062_796_044_560_2,
        ]),
    ];

    let dt: DelaunayTriangulation<_, (), (), 4> =
        DelaunayTriangulation::new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap_or_else(|err| panic!("4D regression configuration failed to construct: {err}"));

    assert!(
        dt.number_of_vertices() >= 5,
        "Should have at least 5 vertices for a 4D triangulation"
    );
    assert!(dt.number_of_vertices() <= vertices.len());
    assert!(dt.number_of_cells() > 0);

    let validation = dt.as_triangulation().validate();
    assert!(
        validation.is_ok(),
        "Triangulation has topology violations: {:?}",
        validation.err()
    );
}

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
#[ignore = "Regression test for specific CI benchmark failure"]
fn test_regression_non_manifold_3d_seed123_50pts() {
    // Exact configuration from CI failure (matches ci_performance_suite.rs)
    let n_points = 50;
    let result = delaunay::geometry::util::generate_random_triangulation_with_topology_guarantee::<
        f64,
        (),
        (),
        3,
    >(
        n_points,        // Point count from CI benchmark
        (-100.0, 100.0), // Bounds from benchmark
        None,            // No vertex data
        Some(123),       // Seed from benchmark (line 85 in ci_performance_suite.rs)
        TopologyGuarantee::PLManifold,
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
    let min_vertices = (n_points / 6).max(4);
    assert!(
        num_vertices <= n_points,
        "Should have ≤{n_points} vertices, got {num_vertices}"
    );
    assert!(
        num_vertices >= min_vertices,
        "Should have ≥{min_vertices} vertices (extremely degenerate cases can skip 80%+), got {num_vertices}"
    );
    assert!(dt.number_of_cells() > 0);

    // Most importantly: validate topology with strict PL-manifold checks (Levels 1–3)
    let validation = dt.as_triangulation().validate();
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
#[ignore = "Stress test for non-manifold topology with nearby seeds"]
fn test_regression_non_manifold_nearby_seeds() {
    let test_seeds = [120, 121, 122, 123, 124, 125, 126];
    let n_points = 50;
    let min_vertices = (n_points / 6).max(4);

    for seed in test_seeds {
        let result =
            delaunay::geometry::util::generate_random_triangulation_with_topology_guarantee::<
                f64,
                (),
                (),
                3,
            >(
                n_points,
                (-100.0, 100.0),
                None,
                Some(seed),
                TopologyGuarantee::PLManifold,
            );

        assert!(
            result.is_ok(),
            "Failed with seed {}: {:?}",
            seed,
            result.err()
        );

        let dt = result.unwrap();

        let num_vertices = dt.number_of_vertices();
        assert!(
            num_vertices <= n_points,
            "Seed {seed}: too many vertices ({num_vertices})"
        );
        assert!(
            num_vertices >= min_vertices,
            "Seed {seed}: too few vertices ({num_vertices}), degenerate cases can skip 80%+"
        );
        let validation = dt.as_triangulation().validate();
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

    let dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

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

    let dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

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

    let dt: DelaunayTriangulation<_, (), (), 4> =
        DelaunayTriangulation::new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

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

    let dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

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

    let dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

    assert_eq!(dt.number_of_vertices(), 7);
    assert!(dt.number_of_cells() >= 4);
}

// =========================================================================
// Edge Cases - Symmetric Configurations
// =========================================================================

#[test]
fn test_square_with_center_2d() {
    init_tracing();
    // Square vertices with center point
    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([2.0, 0.0]),
        vertex!([2.0, 2.0]),
        vertex!([0.0, 2.0]),
        vertex!([1.0, 1.0]), // center
    ];

    let dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

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

    let dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

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

    let dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

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

    let dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

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

    let dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

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
        DelaunayTriangulation::with_topology_guarantee(
            &RobustKernel::new(),
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

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
        DelaunayTriangulation::new_with_topology_guarantee(
            &collinear,
            TopologyGuarantee::PLManifold,
        );

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

    let dt: DelaunayTriangulation<_, (), (), 5> =
        DelaunayTriangulation::new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

    assert_eq!(dt.number_of_vertices(), 7);
    assert!(dt.number_of_cells() > 1);
}
