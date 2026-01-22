//! Regression tests for 2D Delaunay triangulation edge cases.
//!
//! This module tests circumsphere and flip repair behavior in minimal 2D configurations.

use delaunay::core::util::debug_print_first_delaunay_violation;
use delaunay::prelude::*;

#[test]
fn regression_empty_circumsphere_2d_minimal_case() {
    let vertices = vec![
        vertex!([48.564_246_621_452_234, 23.481_505_128_710_488]),
        vertex!([-9.807_184_344_740_996, -36.451_902_443_093_33]),
        vertex!([75.784_620_110_257_45, 25.382_048_382_678_306]),
        vertex!([50.330_335_525_698_53, 25.294_356_716_784_847]),
        vertex!([77.411_339_748_608_4, -86.531_849_594_875_54]),
        vertex!([-93.661_180_847_043, 1.562_430_007_326_195_9]),
    ];

    let mut dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

    if dt.is_valid().is_err() {
        debug_print_first_delaunay_violation(dt.tds(), None);
    }

    let stats = dt.repair_delaunay_with_flips().unwrap();
    eprintln!("[regression] flip repair stats: {stats:?}");

    assert!(
        dt.is_valid().is_ok(),
        "2D triangulation should be a valid PL-manifold after global flip repair"
    );
}

#[test]
fn regression_issue_120_minimal_failing_input_2d() {
    // From docs/issue_120_investigation.md (Example Failure Case (2D)).
    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([-54.687, 0.0]),
        vertex!([-85.026, 36.185]),
        vertex!([0.0, 38.424]),
    ];

    let dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

    if let Err(err) = dt.validate() {
        debug_print_first_delaunay_violation(dt.tds(), None);
        panic!("Issue #120 2D regression must validate Levels 1-4: {err}");
    }
}
