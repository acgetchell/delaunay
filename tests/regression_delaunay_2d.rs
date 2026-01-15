#![allow(missing_docs)]

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
        DelaunayTriangulation::new(&vertices).unwrap();

    if dt.is_valid().is_err() {
        #[cfg(any(test, debug_assertions))]
        debug_print_first_delaunay_violation(dt.tds(), None);
    }

    let stats = dt.repair_delaunay_with_flips().unwrap();
    eprintln!("[regression] flip repair stats: {stats:?}");

    assert!(
        dt.is_valid().is_ok(),
        "2D triangulation should satisfy Delaunay property after global flip repair"
    );
}
