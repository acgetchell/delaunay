//! Edge case and regression tests for `DelaunayTriangulation`.
//!
//! These tests cover:
//! - Known regression configurations (previously failing cases)
//! - Degenerate configurations (collinear, coplanar points)
//! - Extreme coordinate values
//! - Edge cases from legacy Tds tests
//!
//! Converted from legacy `Tds::new()` tests to use the new `DelaunayTriangulation` API.

use delaunay::construction::DelaunayConstructionRetryFailure;
use delaunay::prelude::construction::{
    DelaunayConstructionFailure, DelaunayTriangulation, DelaunayTriangulationConstructionError,
    TopologyGuarantee, Vertex,
};
#[cfg(feature = "diagnostics")]
use delaunay::prelude::diagnostics::debug_print_first_delaunay_violation;
use delaunay::prelude::generators::{
    generate_random_points_in_ball_seeded,
    try_generate_random_triangulation_with_topology_guarantee,
};
use delaunay::prelude::geometry::RobustKernel;
use delaunay::prelude::validation::{
    DelaunayTriangulationValidationError, TriangulationEmbeddingValidationError,
};
use delaunay::vertex;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use std::num::NonZeroUsize;

const fn nonzero(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("test point count must be non-zero")
}

fn init_tracing() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let default_filter = if cfg!(feature = "diagnostics") {
            "info"
        } else {
            "warn"
        };
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default_filter));
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_test_writer()
            .try_init();
    });
}

/// Return whether construction failed with a direct geometric degeneracy error.
const fn is_geometric_degeneracy_error(error: &DelaunayTriangulationConstructionError) -> bool {
    matches!(
        error,
        DelaunayTriangulationConstructionError::Triangulation(
            DelaunayConstructionFailure::GeometricDegeneracy { .. }
        )
    )
}

/// Accept direct degeneracy failures and retry exhaustion wrapping that same typed source.
fn is_geometric_degeneracy_or_retry_exhausted(
    error: &DelaunayTriangulationConstructionError,
) -> bool {
    match error {
        DelaunayTriangulationConstructionError::Triangulation(
            DelaunayConstructionFailure::ShuffledRetryExhausted { source, .. },
        ) => matches!(
            source.as_ref(),
            DelaunayConstructionRetryFailure::Construction { source }
                if is_geometric_degeneracy_error(source)
        ),
        other => is_geometric_degeneracy_error(other),
    }
}

fn validation_error_is_degenerate_simplex(error: &DelaunayTriangulationValidationError) -> bool {
    matches!(
        error,
        DelaunayTriangulationValidationError::Embedding(source)
            if matches!(
                source.as_ref(),
                TriangulationEmbeddingValidationError::DegenerateSimplex { .. }
            )
    )
}

fn construction_error_is_degenerate_simplex(
    error: &DelaunayTriangulationConstructionError,
) -> bool {
    match error {
        DelaunayTriangulationConstructionError::Triangulation(
            DelaunayConstructionFailure::FinalDelaunayValidation { source, .. },
        ) => validation_error_is_degenerate_simplex(source),
        DelaunayTriangulationConstructionError::Triangulation(
            DelaunayConstructionFailure::InsertionEmbeddingValidation { source },
        ) => matches!(
            source,
            TriangulationEmbeddingValidationError::DegenerateSimplex { .. }
        ),
        DelaunayTriangulationConstructionError::Triangulation(
            DelaunayConstructionFailure::ShuffledRetryExhausted { source, .. },
        ) => match source.as_ref() {
            DelaunayConstructionRetryFailure::Construction { source } => {
                construction_error_is_degenerate_simplex(source)
            }
            _ => false,
        },
        _ => false,
    }
}

macro_rules! test_debug_info {
    ($($arg:tt)*) => {{
        #[cfg(feature = "diagnostics")]
        {
            init_tracing();
            tracing::info!($($arg)*);
        }
        #[cfg(not(feature = "diagnostics"))]
        {
            let _ = format_args!($($arg)*);
        }
    }};
}

macro_rules! test_debug_warn {
    ($($arg:tt)*) => {{
        #[cfg(feature = "diagnostics")]
        {
            init_tracing();
            tracing::warn!($($arg)*);
        }
        #[cfg(not(feature = "diagnostics"))]
        {
            let _ = format_args!($($arg)*);
        }
    }};
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
                    DelaunayTriangulation::try_new_with_topology_guarantee(
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
                assert!(dt.number_of_simplices() > 0);
            }
        }
    };
}

// 2D regression: base triangle with interior and exterior point
test_regression_config!(
    regression_2d_canonical,
    2,
    vec![
        vertex!([0.0, 0.0]).unwrap(),
        vertex!([2.0, 0.0]).unwrap(),
        vertex!([0.0, 2.0]).unwrap(),
        vertex!([0.8, 0.7]).unwrap(),   // interior
        vertex!([-0.5, -0.4]).unwrap(), // exterior
    ]
);

#[test]
#[expect(
    clippy::collapsible_if,
    reason = "test keeps nested invariant checks aligned with diagnostic messages"
)]
#[expect(
    clippy::too_many_lines,
    reason = "edge-case regression test keeps construction and validation assertions together"
)]
#[expect(
    clippy::unreadable_literal,
    reason = "large literal documents the exact stress-case coordinate"
)]
fn debug_issue_120_empty_circumsphere_5d() {
    init_tracing();
    let vertices = vec![
        vertex!([
            18.781125710207355,
            85.19556603270544,
            -35.577425948458725,
            -78.4710254162274,
            25.721771703577573,
        ])
        .unwrap(),
        vertex!([
            -21.95447633622051,
            83.05734190480365,
            96.97214006048821,
            -48.80161083192332,
            -21.250997394474208,
        ])
        .unwrap(),
        vertex!([
            50.96615929339812,
            -12.888014856181814,
            64.35842847516192,
            81.20742517692801,
            -67.85330902948604,
        ])
        .unwrap(),
        vertex!([
            -74.19363960080797,
            -87.39864134220277,
            -31.002590635557322,
            -73.43717909637807,
            38.224369898650814,
        ])
        .unwrap(),
        vertex!([
            -15.305719099426401,
            37.44773385928881,
            -31.57846617007415,
            -11.413274473891796,
            32.32927241254111,
        ])
        .unwrap(),
        vertex!([
            -58.19271902837987,
            -50.763430349360824,
            72.37200252994022,
            66.28041332398725,
            53.51398806010464,
        ])
        .unwrap(),
        vertex!([
            96.64514856949884,
            69.99880120063219,
            29.117126126375382,
            88.1850558085571,
            95.34623469752856,
        ])
        .unwrap(),
    ];

    let mut dt: DelaunayTriangulation<_, (), (), 5> =
        DelaunayTriangulation::try_new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap_or_else(|err| panic!("5D debug configuration failed to construct: {err}"));
    match dt.repair_delaunay_with_flips() {
        Ok(stats) => {
            test_debug_info!(
                "[Issue #120 debug] repair_delaunay_with_flips stats: checked={}, flips={}, max_queue={}",
                stats.facets_checked,
                stats.flips_performed,
                stats.max_queue_len
            );
        }
        Err(err) => {
            test_debug_warn!("[Issue #120 debug] repair_delaunay_with_flips error: {err}");
        }
    }
    let mut dt_robust: DelaunayTriangulation<RobustKernel<f64>, (), (), 5> =
        DelaunayTriangulation::try_from_tds_with_topology_guarantee(
            dt.tds().clone(),
            RobustKernel::new(),
            TopologyGuarantee::PLManifold,
        )
        .unwrap_or_else(|err| panic!("5D robust TDS should validate: {err}"));
    match dt_robust.repair_delaunay_with_flips() {
        Ok(stats) => {
            test_debug_info!(
                "[Issue #120 debug] robust repair stats: checked={}, flips={}, max_queue={}",
                stats.facets_checked,
                stats.flips_performed,
                stats.max_queue_len
            );
        }
        Err(err) => {
            test_debug_warn!("[Issue #120 debug] robust repair error: {err}");
        }
    }
    if let Err(err) = dt_robust.is_valid_delaunay() {
        test_debug_warn!("[Issue #120 debug] robust triangulation still invalid: {err:?}");
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x1200_5eed);
    for attempt in 0..20 {
        let mut shuffled = vertices.clone();
        shuffled.shuffle(&mut rng);
        if let Ok(dt_alt) = DelaunayTriangulation::<_, (), (), 5>::try_new_with_topology_guarantee(
            &shuffled,
            TopologyGuarantee::PLManifold,
        ) {
            if dt_alt.is_valid_delaunay().is_ok() {
                test_debug_info!(
                    "[Issue #120 debug] found valid triangulation after shuffle attempt {}",
                    attempt + 1
                );
                break;
            }
        }
        if attempt == 19 {
            test_debug_warn!("[Issue #120 debug] no valid triangulation found in 20 shuffles");
        }
    }
    for (simplex_key, simplex) in dt.simplices() {
        test_debug_info!("[Issue #120 debug] simplex {simplex_key:?}:");
        for &vkey in simplex.vertices() {
            let vertex = dt.tds().vertex(vkey).expect("vertex key should exist");
            test_debug_info!(
                "  vkey={vkey:?}, uuid={}, point={:?}",
                vertex.uuid(),
                vertex.point()
            );
        }
    }

    if let Err(err) = dt.is_valid_delaunay() {
        #[cfg(feature = "diagnostics")]
        {
            debug_print_first_delaunay_violation(dt.tds(), None);
        }
        panic!("5D debug configuration violates Delaunay property: {err:?}");
    }
}

// 3D regression: tetrahedron with interior point
test_regression_config!(
    regression_3d_canonical,
    3,
    vec![
        vertex!([0.0, 0.0, 0.0]).unwrap(),
        vertex!([2.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 2.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 2.0]).unwrap(),
        vertex!([0.4, 0.4, 0.3]).unwrap(), // interior
    ]
);

// 4D regression: 4-simplex with interior point
test_regression_config!(
    regression_4d_canonical,
    4,
    vec![
        vertex!([0.0, 0.0, 0.0, 0.0]).unwrap(),
        vertex!([2.0, 0.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 2.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 2.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 0.0, 2.0]).unwrap(),
        vertex!([0.4, 0.3, 0.3, 0.3]).unwrap(), // interior
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
fn test_regression_proptest_insertion_order_4d_euler_mismatch() {
    let vertices = vec![
        vertex!([
            -65.070_532_013_377_94,
            72.223_880_592_145_24,
            -97.837_333_303_337_39,
            35.700_988_360_396_63,
        ])
        .unwrap(),
        vertex!([
            27.011_682_298_030_294,
            52.054_594_213_988_53,
            -4.067_357_689_604_181,
            59.253_713_038_817_09,
        ])
        .unwrap(),
        vertex!([
            64.848_601_083_080_47,
            84.907_367_805_317_42,
            -98.659_828_418_664_1,
            70.056_498_821_543_85,
        ])
        .unwrap(),
        vertex!([
            -23.543_852_823_069_876,
            96.741_963_200_207_22,
            -50.539_503_136_092_634,
            -49.616_262_314_856_67,
        ])
        .unwrap(),
        vertex!([
            24.886_830_567_772_98,
            -81.708_725_314_824,
            50.775_700_870_880_27,
            20.281_603_779_436_033,
        ])
        .unwrap(),
        vertex!([
            78.030_479_318_165_25,
            -82.763_788_627_520_88,
            94.075_337_487_756_27,
            44.637_774_779_142_73,
        ])
        .unwrap(),
        vertex!([
            -5.175_491_150_708_228,
            97.527_582_084_288_54,
            95.344_552_027_220_42,
            84.908_292_808_161_85,
        ])
        .unwrap(),
        vertex!([
            71.994_788_686_588_11,
            4.833_973_465_666_131,
            -80.802_685_728_835_39,
            64.010_634_775_159_37,
        ])
        .unwrap(),
        vertex!([
            23.390_079_186_814_17,
            -2.157_395_632_040_824_7,
            -6.601_766_119_999_574,
            32.062_796_044_560_2,
        ])
        .unwrap(),
    ];

    let dt: DelaunayTriangulation<_, (), (), 4> =
        DelaunayTriangulation::try_new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap_or_else(|err| panic!("4D regression configuration failed to construct: {err}"));

    assert!(
        dt.number_of_vertices() >= 5,
        "Should have at least 5 vertices for a 4D triangulation"
    );
    assert!(dt.number_of_vertices() <= vertices.len());
    assert!(dt.number_of_simplices() > 0);

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
        ])
        .unwrap(),
        vertex!([
            -31.430_765_957_270_268,
            50.418_208_939_604_746,
            88.657_219_404_750_96,
            47.248_786_623_931_88,
            -81.163_199_600_681_14,
        ])
        .unwrap(),
        vertex!([
            -89.902_834_998_758_96,
            93.719_989_121_636_87,
            64.524_277_928_893_98,
            40.001_314_184_454_05,
            14.196_053_554_411_321,
        ])
        .unwrap(),
        vertex!([
            2.625_958_385_925_883,
            48.251_155_688_054_36,
            3.491_542_746_106_750_5,
            97.241_732_043_079_37,
            -27.107_939_334_194_757,
        ])
        .unwrap(),
        vertex!([
            62.628_856_831_188_11,
            -18.181_728_263_486_345,
            -32.153_141_689_537_584,
            25.692_809_519_458_7,
            26.369_541_091_117_114,
        ])
        .unwrap(),
        vertex!([
            -41.886_149_523_644_406,
            -54.537_563_736_672_65,
            -54.555_379_092_740_964,
            75.499_924_758_912_23,
            16.127_546_041_675_355,
        ])
        .unwrap(),
        vertex!([
            -77.161_459_173_963_2,
            -59.065_517_574_769_37,
            -19.652_689_679_369_03,
            -51.622_382_706_243_18,
            -26.000_263_271_298_543,
        ])
        .unwrap(),
    ]
);

// =========================================================================
// Regression Tests - Non-Manifold Topology During Incremental Insertion
// =========================================================================

/// Regression test for CI benchmark failure with non-manifold topology.
///
/// This test reproduces a specific failure discovered in the CI benchmark suite
/// where incremental insertion created temporary non-manifold topology (facet
/// shared by 4 simplices). The fix made `wire_cavity_neighbors` tolerant of this
/// condition, allowing the localized repair mechanism to handle it.
///
/// ## Historical Context
///
/// - **When**: Discovered in CI run 19874972833 (2025-12-02)
/// - **Where**: `benches/ci_performance_suite.rs:85` (3D, 50 points, seed 123)
/// - **Error**: "Non-manifold topology detected: facet 11030659497163937569
///   shared by 4 simplices (expected ≤2)"
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
    let n_points = nonzero(50);
    let raw_n_points = n_points.get();
    let result = try_generate_random_triangulation_with_topology_guarantee::<(), (), 3>(
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
    let min_vertices = (raw_n_points / 6).max(4);
    assert!(
        num_vertices <= raw_n_points,
        "Should have ≤{raw_n_points} vertices, got {num_vertices}"
    );
    assert!(
        num_vertices >= min_vertices,
        "Should have ≥{min_vertices} vertices (extremely degenerate cases can skip 80%+), got {num_vertices}"
    );
    assert!(dt.number_of_simplices() > 0);

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
fn test_regression_non_manifold_nearby_seeds() {
    let test_seeds = [120, 121, 122, 123, 124, 125, 126];
    let n_points = nonzero(50);
    let raw_n_points = n_points.get();
    let min_vertices = (raw_n_points / 6).max(4);

    for seed in test_seeds {
        let result = try_generate_random_triangulation_with_topology_guarantee::<(), (), 3>(
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
            num_vertices <= raw_n_points,
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
        vertex!([0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
    ];

    let dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::try_new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

    assert_eq!(dt.number_of_vertices(), 3);
    assert_eq!(dt.number_of_simplices(), 1);
}

#[test]
fn test_exact_minimum_vertices_3d() {
    // Exactly D+1 = 4 vertices for 3D
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 1.0]).unwrap(),
    ];

    let dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::try_new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

    assert_eq!(dt.number_of_vertices(), 4);
    assert_eq!(dt.number_of_simplices(), 1);
}

#[test]
fn test_exact_minimum_vertices_4d() {
    // Exactly D+1 = 5 vertices for 4D
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 1.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 0.0, 1.0]).unwrap(),
    ];

    let dt: DelaunayTriangulation<_, (), (), 4> =
        DelaunayTriangulation::try_new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

    assert_eq!(dt.number_of_vertices(), 5);
    assert_eq!(dt.number_of_simplices(), 1);
}

// =========================================================================
// Edge Cases - Interior/Exterior Points
// =========================================================================

#[test]
fn test_multiple_interior_points_2d() {
    let vertices = vec![
        vertex!([0.0, 0.0]).unwrap(),
        vertex!([4.0, 0.0]).unwrap(),
        vertex!([2.0, 4.0]).unwrap(),
        vertex!([1.5, 1.5]).unwrap(), // interior
        vertex!([2.0, 1.5]).unwrap(), // interior
        vertex!([2.5, 1.5]).unwrap(), // interior
    ];

    let dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::try_new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

    assert_eq!(dt.number_of_vertices(), 6);
    assert!(dt.number_of_simplices() >= 4);
}

#[test]
fn test_multiple_interior_points_3d() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]).unwrap(),
        vertex!([2.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 2.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 2.0]).unwrap(),
        vertex!([0.5, 0.5, 0.5]).unwrap(), // interior
        vertex!([0.6, 0.5, 0.5]).unwrap(), // interior
        vertex!([0.5, 0.6, 0.5]).unwrap(), // interior
    ];

    let dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::try_new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

    assert_eq!(dt.number_of_vertices(), 7);
    assert!(dt.number_of_simplices() >= 4);
}

// =========================================================================
// Edge Cases - Symmetric Configurations
// =========================================================================

#[test]
fn test_square_with_center_2d() {
    init_tracing();
    // Square vertices with center point
    let vertices = vec![
        vertex!([0.0, 0.0]).unwrap(),
        vertex!([2.0, 0.0]).unwrap(),
        vertex!([2.0, 2.0]).unwrap(),
        vertex!([0.0, 2.0]).unwrap(),
        vertex!([1.0, 1.0]).unwrap(), // center
    ];

    let dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::try_new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

    assert_eq!(dt.number_of_vertices(), 5);
    assert!(dt.number_of_simplices() >= 4);
}

#[test]
fn test_cube_vertices_3d() {
    // 8 corners of a unit cube
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0, 0.0]).unwrap(),
        vertex!([1.0, 1.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 1.0]).unwrap(),
        vertex!([1.0, 0.0, 1.0]).unwrap(),
        vertex!([0.0, 1.0, 1.0]).unwrap(),
        vertex!([1.0, 1.0, 1.0]).unwrap(),
    ];

    let err = DelaunayTriangulation::<_, (), (), 3>::try_new_with_topology_guarantee(
        &vertices,
        TopologyGuarantee::PLManifold,
    )
    .expect_err("exact cube corners should fail before storing a zero-volume simplex");

    assert!(
        construction_error_is_degenerate_simplex(&err),
        "cube-corner failure should preserve the embedding degeneracy source: {err:?}"
    );
}

// =========================================================================
// Edge Cases - Varying Scales
// =========================================================================

#[test]
fn test_large_coordinates_2d() {
    let vertices = vec![
        vertex!([1000.0, 1000.0]).unwrap(),
        vertex!([2000.0, 1000.0]).unwrap(),
        vertex!([1500.0, 2000.0]).unwrap(),
        vertex!([1500.0, 1500.0]).unwrap(),
    ];

    let dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::try_new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

    assert_eq!(dt.number_of_vertices(), 4);
    assert!(dt.number_of_simplices() > 0);
}

#[test]
fn test_small_coordinates_3d() {
    let vertices = vec![
        vertex!([0.001, 0.001, 0.001]).unwrap(),
        vertex!([0.002, 0.001, 0.001]).unwrap(),
        vertex!([0.001, 0.002, 0.001]).unwrap(),
        vertex!([0.001, 0.001, 0.002]).unwrap(),
        vertex!([0.0015, 0.0015, 0.0015]).unwrap(),
    ];

    let dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::try_new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

    assert_eq!(dt.number_of_vertices(), 5);
    assert!(dt.number_of_simplices() > 0);
}

#[test]
fn test_negative_coordinates_2d() {
    let vertices = vec![
        vertex!([-1.0, -1.0]).unwrap(),
        vertex!([1.0, -1.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
        vertex!([0.0, 0.0]).unwrap(),
    ];

    let dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::try_new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

    assert_eq!(dt.number_of_vertices(), 4);
    assert!(dt.number_of_simplices() > 0);
}

// =========================================================================
// Edge Cases - Different Kernels
// =========================================================================

#[test]
fn test_robust_kernel_with_edge_case() {
    // Configuration that might benefit from robust predicates
    let vertices = vec![
        vertex!([0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.5, 0.866_025_403_784_438_7]).unwrap(), // ~sqrt(3)/2 - near-equilateral
        vertex!([0.5, 0.5]).unwrap(),
    ];

    let dt: DelaunayTriangulation<RobustKernel<f64>, (), (), 2> =
        DelaunayTriangulation::try_with_topology_guarantee(
            &RobustKernel::new(),
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

    assert_eq!(dt.number_of_vertices(), 4);
    assert!(dt.number_of_simplices() > 0);
}

// =========================================================================
// Edge Cases - Degenerate Configurations
// =========================================================================

#[test]
fn test_collinear_points_2d() {
    // All points lie on a line in 2D: no non-degenerate simplex exists.
    // AdaptiveKernel uses exact orientation (no SoS), so collinear points
    // are correctly detected as degenerate.
    let collinear = vec![
        vertex!([0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([2.0, 0.0]).unwrap(),
        vertex!([3.0, 0.0]).unwrap(),
    ];

    let result: Result<DelaunayTriangulation<_, (), (), 2>, _> =
        DelaunayTriangulation::try_new_with_topology_guarantee(
            &collinear,
            TopologyGuarantee::PLManifold,
        );

    // Verify it fails with GeometricDegeneracy due to collinear simplex.
    // The default retry policy may wrap the final typed degeneracy once all
    // shuffled retries are exhausted.
    assert!(
        result
            .as_ref()
            .is_err_and(is_geometric_degeneracy_or_retry_exhausted),
        "Expected GeometricDegeneracy error for collinear points, got: {result:?}"
    );
}

// =========================================================================
// Regression: #228 exact-predicate paths (fast variant)
// =========================================================================

/// Fast regression test for the exact-predicate code paths changed in #228.
///
/// Constructs a 3D triangulation from 16 random ball-distributed points using
/// `AdaptiveKernel` (the default; exact+SoS predicates) and verifies the
/// Delaunay property. This exercises:
/// - `det_errbound()` fast filter in orientation/insphere predicates
/// - Unified kernel predicates in flip repair
/// - `solve_exact_rounded_f64` circumcenter fallback for near-singular simplices
///
/// Unlike the slow-tests gated 1000-point test in `large_scale_debug.rs`, this
/// runs in normal CI (seconds in debug, sub-second in release).
#[test]
fn regression_issue_228_exact_predicate_paths_3d_fast() {
    let seed: u64 = 0x0228_FA53_0003;
    let points = generate_random_points_in_ball_seeded::<3>(16, 100.0, seed)
        .expect("point generation should succeed");
    let vertices: Vec<Vertex<(), 3>> = points
        .into_iter()
        .map(|p| vertex!(p.into()).unwrap())
        .collect();

    let dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::try_new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::Pseudomanifold,
        )
        .expect("3D 16-point construction must not fail (#228 fast regression)");

    assert!(
        dt.is_delaunay_via_flips().is_ok(),
        "Delaunay property must hold (#228 fast regression, seed=0x{seed:X})"
    );
    assert!(dt.number_of_vertices() > 0);
    assert!(dt.number_of_simplices() > 0);
}

// =========================================================================
// Edge Cases - Higher Dimensions
// =========================================================================

#[test]
fn test_5d_simplex_plus_interior() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0, 0.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 1.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 0.0, 1.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 0.0, 0.0, 1.0]).unwrap(),
        vertex!([0.2, 0.2, 0.2, 0.2, 0.2]).unwrap(), // interior
    ];

    let dt: DelaunayTriangulation<_, (), (), 5> =
        DelaunayTriangulation::try_new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

    assert_eq!(dt.number_of_vertices(), 7);
    assert!(dt.number_of_simplices() > 1);
}
