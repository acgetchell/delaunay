//! Regression test for issue #307: bulk flip repair must not leave negative
//! geometric orientation behind in 4D construction.

use delaunay::core::triangulation::TopologyGuarantee;
use delaunay::core::util::{hilbert_indices_prequantized, hilbert_quantize};
use delaunay::geometry::kernel::RobustKernel;
use delaunay::geometry::point::Point;
use delaunay::geometry::util::generate_random_points_in_ball_seeded;
use delaunay::prelude::triangulation::*;
use delaunay::triangulation::delaunay::{ConstructionOptions, InsertionOrderStrategy, RetryPolicy};
use std::cmp::Ordering;

/// Replays the full 100-point Hilbert ordering but keeps only the prefix that
/// first exposed the negative-orientation cell, keeping the regression fast.
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
            .then_with(|| a_point.partial_cmp(b_point).unwrap_or(Ordering::Equal))
            .then_with(|| a_in.cmp(b_in))
    });

    keyed
        .into_iter()
        .take(prefix_len)
        .map(|(_, _, point, _)| vertex!(point))
        .collect()
}

/// Computes the scalar range used by batch Hilbert ordering so the test prefix
/// matches the original 100-point construction order.
fn coordinate_bounds<const D: usize>(points: &[Point<f64, D>]) -> (f64, f64) {
    points
        .iter()
        .flat_map(Point::coords)
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &coord| {
            (min.min(coord), max.max(coord))
        })
}

/// The first 13 vertices from the 100-point 4D seed used to leave one negative
/// cell after bulk local repair, causing every later insertion to be skipped.
#[test]
fn regression_issue_307_4d_bulk_repair_keeps_positive_orientation() {
    let seed: u64 = 0x9B77_86C9_99C5_6A16;
    let points = generate_random_points_in_ball_seeded::<f64, 4>(100, 100.0, seed)
        .expect("point generation should succeed");
    let vertices = hilbert_ordered_prefix(points, 13);

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
        dt.as_triangulation().validate().is_ok(),
        "bulk repair must leave all cells in positive geometric orientation",
    );
}
