//! Deterministic regression tests for previously failing Delaunay configurations.
//!
//! Each test encodes a small, fixed point set that either used to violate the
//! global Delaunay condition or is known to be numerically challenging. These
//! tests ensure that once the algorithm is corrected, the configurations remain
//! Delaunay-consistent.

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
        $ignore_reason:expr,
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

// =============================================================================
// 5D CANONICAL CONFIGURATION (FROM DEBUG LOGS)
// =============================================================================
//
// This configuration matches the 7-point 5D vertex set reconstructed from
// Delaunay debug output and used in:
// - `tests/debug_delaunay_violation_5d.rs`
// - `src/core/algorithms/unified_insertion_pipeline.rs` (stepwise 5D test)
//
// It was previously known to trigger a global Delaunay validation failure
// after construction. This regression test ensures it now remains Delaunay.
regression_delaunay_violation_test!(
    regression_delaunay_violation_5d_known_config,
    5,
    "Known 5D Delaunay violation; see docs/fix-delaunay.md (Captured seeds) and debug_delaunay_violation_5d.rs",
    {
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
    }
);
