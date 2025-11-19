//! Deterministic regression tests for canonical Delaunay configurations.
//!
//! Each test encodes a small, fixed point set that either (a) previously
//! violated the global Delaunay condition or (b) is a canonical baseline
//! configuration we want to keep Delaunay-consistent across releases.

use delaunay::core::triangulation_data_structure::{Tds, TriangulationConstructionError};
use delaunay::vertex;

/// Macro to define a regression test for a canonical D-dimensional configuration
/// that must successfully construct a Delaunay triangulation.
///
/// The test:
/// - builds a fixed vertex set,
/// - calls `Tds::new` for the given dimension,
/// - asserts that construction succeeds without Delaunay-related errors.
macro_rules! regression_delaunay_violation_test {
    (
        $name:ident,
        $dim:literal,
        $build_vertices:block
    ) => {
        pastey::paste! {
            #[test]
            fn $name() {
                let vertices = $build_vertices;

                let result: Result<
                    Tds<f64, Option<()>, Option<()>, $dim>,
                    TriangulationConstructionError,
                > = Tds::new(&vertices);

                let tds = result.unwrap_or_else(|err| {
                    panic!(
                        "{}D canonical regression configuration failed to construct: {err}",
                        $dim,
                    )
                });

                assert!(
                    tds.validate_delaunay().is_ok(),
                    "{}D canonical regression configuration should remain globally Delaunay",
                    $dim,
                );
            }
        }
    };
}

/// Helper macro to register multiple canonical configurations across dimensions.
macro_rules! regression_delaunay_known_configs_all_dims {
    ( $( $name:ident, $dim:literal => $build_vertices:block ),+ $(,)? ) => {
        $(
            regression_delaunay_violation_test!($name, $dim, $build_vertices);
        )+
    };
}

// =============================================================================
// CANONICAL CONFIGURATIONS (2D–5D)
// =============================================================================
//
// 2D–4D cases are simple, non-degenerate point sets that exercise basic
// interior/exterior behavior. The 5D case below is reconstructed from a
// historically failing `proptest_delaunay_condition` configuration and
// must remain Delaunay under `Tds::new`.
regression_delaunay_known_configs_all_dims! {
    regression_delaunay_violation_2d_canonical, 2 => {
        // Simple 2D set: base triangle with one interior and one exterior point.
        vec![
            vertex!([0.0, 0.0]),
            vertex!([2.0, 0.0]),
            vertex!([0.0, 2.0]),
            vertex!([0.8, 0.7]),   // interior-ish
            vertex!([-0.5, -0.4]), // exterior
        ]
    },
    regression_delaunay_violation_3d_canonical, 3 => {
        // 3D tetrahedron plus a point strictly inside the tetrahedron.
        vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([2.0, 0.0, 0.0]),
            vertex!([0.0, 2.0, 0.0]),
            vertex!([0.0, 0.0, 2.0]),
            vertex!([0.4, 0.4, 0.3]), // interior
        ]
    },
    regression_delaunay_violation_4d_canonical, 4 => {
        // 4D simplex (5 vertices) plus one interior point.
        vec![
            vertex!([0.0, 0.0, 0.0, 0.0]),
            vertex!([2.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 2.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 2.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 2.0]),
            vertex!([0.4, 0.3, 0.3, 0.3]), // interior
        ]
    },
    regression_delaunay_violation_5d_known_config, 5 => {
        // 5D CANONICAL CONFIGURATION (FROM DEBUG LOGS)
        //
        // This configuration matches the 7-point 5D vertex set reconstructed from
        // Delaunay debug output and used in:
        // - `tests/debug_delaunay_violation_5d.rs`
        // - `src/core/algorithms/unified_insertion_pipeline.rs` (stepwise 5D test)
        //
        // It was previously known to trigger a global Delaunay validation failure
        // after construction. This regression test ensures it now remains Delaunay.
        //
        // NOTE: The 5D vertex coordinates below are intentionally duplicated in
        // `tests/debug_delaunay_violation_5d.rs`. Keeping both tests self-contained
        // avoids a shared test-only helper module and makes it easier to inspect
        // each regression in isolation when debugging.
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
    },
}
