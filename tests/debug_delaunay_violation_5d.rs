//! Regression test for a 5D configuration that previously violated the
//! Delaunay condition.
//!
//! This configuration was originally captured from Delaunay debug output where
//! `Tds::new` succeeded structurally but global Delaunay validation reported
//! violations. The unified insertion pipeline and robust repair logic have
//! since been updated so that this configuration now produces a fully
//! Delaunay-consistent triangulation.

use delaunay::core::triangulation_data_structure::Tds;
use delaunay::vertex;

/// Reproduces a 5D configuration observed in earlier Delaunay debug output
/// where `validate_delaunay` used to report violations. This now serves as a
/// regression test that the configuration is handled correctly.
#[test]
fn debug_known_5d_delaunay_violation_case() {
    // 5D point set reconstructed from prior Delaunay debug logs.
    // Each point is represented as vertex!([...]) for convenience.
    let vertices = vec![
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
    ];

    // This configuration should now triangulate successfully and satisfy the
    // global Delaunay property.
    let result: Result<Tds<f64, Option<()>, Option<()>, 5>, _> = Tds::new(&vertices);
    let tds = result
        .expect("5D canonical configuration should now construct without Delaunay-related errors");

    assert!(
        tds.validate_delaunay().is_ok(),
        "5D canonical configuration should be globally Delaunay after construction",
    );
}

// A separate stepwise debug harness for this configuration lives in the
// `src/core/algorithms/unified_insertion_pipeline.rs` module tests, where it can
// access the internal `UnifiedInsertionPipeline` type without exposing it as
// part of the public API.
