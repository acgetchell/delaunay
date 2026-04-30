//! Regression tests for fixed Delaunay triangulation bugs.
//!
//! Add new fixed-bug regression cases here rather than creating issue-specific
//! integration test crates, unless the case needs separate crate-level setup,
//! feature flags, or profile isolation.

#[cfg(feature = "diagnostics")]
use delaunay::core::util::debug_print_first_delaunay_violation;
use delaunay::geometry::kernel::RobustKernel;
use delaunay::geometry::point::Point;
use delaunay::geometry::util::generate_random_points_in_ball_seeded;
use delaunay::prelude::ordering::{hilbert_indices_prequantized, hilbert_quantize};
use delaunay::prelude::triangulation::*;
use delaunay::triangulation::delaunay::{ConstructionOptions, InsertionOrderStrategy, RetryPolicy};

/// Replays a full Hilbert ordering while keeping only the prefix that first
/// exposed issue #307, so the regression stays fast and deterministic.
fn hilbert_ordered_prefix<const D: usize>(
    points: Vec<Point<f64, D>>,
    prefix_len: usize,
) -> Vec<Vertex<f64, (), D>> {
    let (min, max) = coordinate_bounds(&points);
    let bits_per_coord = 31;
    let quantized: Vec<[u32; D]> = points
        .iter()
        .map(|point| {
            hilbert_quantize(point.coords(), (min, max), bits_per_coord)
                .expect("finite generated points should quantize")
        })
        .collect();
    let indices = hilbert_indices_prequantized(&quantized, bits_per_coord)
        .expect("4D Hilbert indices should fit in u128");

    let mut keyed: Vec<(u128, [u32; D], Point<f64, D>, usize)> = points
        .into_iter()
        .enumerate()
        .map(|(input_index, point)| {
            (
                indices[input_index],
                quantized[input_index],
                point,
                input_index,
            )
        })
        .collect();

    keyed.sort_by(|(a_idx, a_q, a_point, a_in), (b_idx, b_q, b_point, b_in)| {
        a_idx
            .cmp(b_idx)
            .then_with(|| a_q.cmp(b_q))
            .then_with(|| {
                a_point.partial_cmp(b_point).unwrap_or_else(|| {
                    panic!(
                        "non-finite point in regression Hilbert sort: left={a_point:?}, right={b_point:?}"
                    )
                })
            })
            .then_with(|| a_in.cmp(b_in))
    });

    keyed
        .into_iter()
        .take(prefix_len)
        .map(|(_, _, point, _)| vertex!(point))
        .collect()
}

/// Computes the scalar range used by batch Hilbert ordering so regression
/// prefixes match the original full construction order.
fn coordinate_bounds<const D: usize>(points: &[Point<f64, D>]) -> (f64, f64) {
    points
        .iter()
        .flat_map(Point::coords)
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &coord| {
            (min.min(coord), max.max(coord))
        })
}

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
        #[cfg(feature = "diagnostics")]
        debug_print_first_delaunay_violation(dt.tds(), None);
    }

    dt.repair_delaunay_with_flips().unwrap();

    assert!(
        dt.is_valid().is_ok(),
        "2D triangulation should be a valid PL-manifold after global flip repair"
    );
}

#[test]
fn regression_issue_120_minimal_failing_input_2d() {
    // From docs/archive/issue_120_investigation.md (Example Failure Case (2D)).
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
        #[cfg(feature = "diagnostics")]
        debug_print_first_delaunay_violation(dt.tds(), None);
        panic!("Issue #120 2D regression must validate Levels 1-4: {err}");
    }
}

/// The 35-vertex 3D seed `0xE30C78582376677C` produces a Hilbert-ordered
/// insertion sequence where vertex 23 triggers flip-repair cycling on
/// co-spherical configurations.
///
/// With the former release-mode `MAX_REPEAT_SIGNATURE = 32` and
/// `RetryPolicy::Disabled`, construction failed deterministically. The fix
/// (#306) unified these constants so the repair has sufficient patience and
/// shuffled retries are always available.
///
/// Run with `cargo test --release --test regressions` to exercise the release
/// profile.
#[test]
fn regression_issue_306_3d_construction_succeeds() {
    let seed: u64 = 0xE30C_7858_2376_677C;
    let points = generate_random_points_in_ball_seeded::<f64, 3>(35, 100.0, seed)
        .expect("point generation should succeed");
    let vertices: Vec<Vertex<f64, (), 3>> = points.into_iter().map(|p| vertex!(p)).collect();

    let dt: Result<DelaunayTriangulation<_, (), (), 3>, _> = DelaunayTriangulation::new(&vertices);
    assert!(
        dt.is_ok(),
        "35-vertex 3D construction with seed 0x{seed:X} should succeed \
         (requires unified repair constants); got: {}",
        dt.unwrap_err()
    );
}

/// The first 14 vertices from the 100-point 4D seed used to leave one negative
/// cell after bulk local repair, causing the next insertion to be skipped.
#[test]
fn regression_issue_307_4d_bulk_repair_keeps_positive_orientation() {
    let seed: u64 = 0x9B77_86C9_99C5_6A16;
    let points = generate_random_points_in_ball_seeded::<f64, 4>(100, 100.0, seed)
        .expect("point generation should succeed");
    let vertices = hilbert_ordered_prefix(points, 14);

    let kernel = RobustKernel::<f64>::new();
    let options = ConstructionOptions::default()
        .with_insertion_order(InsertionOrderStrategy::Input)
        .with_retry_policy(RetryPolicy::Disabled);
    let (dt, stats) = DelaunayTriangulation::<RobustKernel<f64>, (), (), 4>::with_topology_guarantee_and_options_with_construction_statistics(
        &kernel,
        &vertices,
        TopologyGuarantee::PLManifoldStrict,
        options,
    )
    .expect("4D bulk construction should not fail after repair orientation cleanup");

    assert_eq!(
        stats.inserted,
        vertices.len(),
        "all prefix vertices should insert without orientation-related skips",
    );
    assert_eq!(stats.total_skipped(), 0);
    assert!(
        dt.as_triangulation().is_valid().is_ok(),
        "bulk repair must leave all cells in positive geometric orientation",
    );
    assert!(
        dt.as_triangulation().validate().is_ok(),
        "bulk repair must leave the triangulation structurally and topologically valid",
    );
}

/// The 4D 500-point seed `0xD225B8A07E274AE6` (ball radius 100) exhausted all
/// shuffled retries before #204: every attempt finished with skip-heavy output
/// (`inserted≈266–300`, `skipped≈200–234`) and the construction ultimately
/// failed with `Cell violates Delaunay property: cell contains vertex that is
/// inside circumsphere`. The dominant failure mode was a cascade of
/// `Ridge fan detected: 4 facets share ridge with 3 vertices` skips driven by
/// a per-insertion local-repair flip budget that was too tight for D≥4
/// (50-flip ceiling vs. observed `max_queue` p95 = 312).
///
/// Fix 2 of the #204 plan (see `docs/archive/issue_204_investigation.md`)
/// raised the D≥4 budget factor/floor (`LOCAL_REPAIR_FLIP_BUDGET_FACTOR_D_GE_4`
/// = 12, `LOCAL_REPAIR_FLIP_BUDGET_FLOOR_D_GE_4` = 96) and added one
/// escalation pass with a 4× budget and the full TDS as seed set before the
/// soft-fail path accepts a non-convergent repair. Post-fix, the same seed
/// inserts 500/500 vertices with zero skips and passes full Level 1–4
/// validation.
///
/// Gated behind `slow-tests` because batch insertion currently takes ~4 min
/// wall time in release mode (still well below the previous ~10 min retry
/// exhaustion); run with:
///
/// ```bash
/// cargo test --release --test regressions --features slow-tests \
///     regression_issue_204_4d_500_local_repair_budget -- --nocapture
/// ```
#[cfg(feature = "slow-tests")]
#[test]
fn regression_issue_204_4d_500_local_repair_budget() {
    let seed: u64 = 0xD225_B8A0_7E27_4AE6;
    let ball_radius = 100.0;
    let n_points: usize = 500;

    let points = generate_random_points_in_ball_seeded::<f64, 4>(n_points, ball_radius, seed)
        .expect("point generation should succeed");
    let vertices: Vec<Vertex<f64, (), 4>> = points.into_iter().map(|p| vertex!(p)).collect();

    let (dt, stats) =
        DelaunayTriangulation::<_, (), (), 4>::new_with_construction_statistics(&vertices)
            .unwrap_or_else(|e| {
                panic!(
                    "#204 regression: 4D {n_points}-point construction with seed 0x{seed:X} \
             (ball radius {ball_radius}) must succeed after Fix 2; got: {}",
                    e.error
                )
            });

    assert_eq!(
        stats.inserted, n_points,
        "#204 regression: all {n_points} vertices should insert with the raised \
         D≥4 local-repair budget (seed 0x{seed:X})",
    );
    assert_eq!(
        stats.total_skipped(),
        0,
        "#204 regression: no vertex should be skipped (seed 0x{seed:X})",
    );
    assert!(
        dt.as_triangulation().validate().is_ok(),
        "#204 regression: triangulation must pass Levels 1–4 validation \
         (seed 0x{seed:X})",
    );
}
