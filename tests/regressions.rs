//! Regression tests for fixed Delaunay triangulation bugs.
//!
//! Add new fixed-bug regression cases here rather than creating issue-specific
//! integration test crates, unless the case needs separate crate-level setup,
//! feature flags, or profile isolation.

use delaunay::prelude::construction::{
    ConstructionOptions, ConstructionStatistics, DelaunayRepairPolicy, DelaunayTriangulation,
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError,
    ExplicitConstructionError, InsertionOrderStrategy, RetryPolicy, TopologyGuarantee, Vertex,
};
#[cfg(feature = "diagnostics")]
use delaunay::prelude::diagnostics::debug_print_first_delaunay_violation;
use delaunay::prelude::generators::generate_random_points_in_ball_seeded;
use delaunay::prelude::geometry::{CoordinateRange, Point, RobustKernel};
use delaunay::prelude::insertion::{HullExtensionReason, InsertionError};
use delaunay::prelude::ordering::{
    HilbertBitDepth, hilbert_indices_prequantized, hilbert_quantize_batch_in_range,
    hilbert_quantize_in_range,
};
use delaunay::vertex;
use std::num::NonZeroUsize;

/// Replays a full Hilbert ordering while keeping only the prefix that first
/// exposed issue #307, so the regression stays fast and deterministic.
fn hilbert_ordered_prefix<const D: usize>(
    points: Vec<Point<D>>,
    prefix_len: usize,
) -> Vec<Vertex<(), D>> {
    let bounds = coordinate_bounds(&points);
    let bits_per_coord = HilbertBitDepth::try_new(31).expect("test bit depth must be valid");
    let quantized: Vec<[u32; D]> = points
        .iter()
        .map(|point| {
            hilbert_quantize_in_range(point.coords(), bounds, bits_per_coord)
                .expect("finite generated points should quantize")
        })
        .collect();
    let indices = hilbert_indices_prequantized(&quantized, bits_per_coord)
        .expect("4D Hilbert indices should fit in u128");

    let mut keyed: Vec<(u128, [u32; D], Point<D>, usize)> = points
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
        .map(|(_, _, point, _)| vertex!(point.into()).unwrap())
        .collect()
}

/// Computes the scalar range used by batch Hilbert ordering so regression
/// prefixes match the original full construction order.
fn coordinate_bounds<const D: usize>(points: &[Point<D>]) -> CoordinateRange<f64> {
    let (min, max) = points
        .iter()
        .flat_map(Point::coords)
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &coord| {
            (min.min(coord), max.max(coord))
        });
    CoordinateRange::try_new(min, max)
        .expect("generated regression points should span a finite non-empty range")
}

fn open_cdt_strip_vertex(
    slice: u32,
    index: u32,
    vertices_per_slice: u32,
    slice_count: u32,
    vertical_jitter: f64,
) -> ([f64; 2], u32) {
    let min_spacing = 1.0_f64 / f64::from(vertices_per_slice - 1);
    let side_jitter = min_spacing / 4.0;
    let interior_jitter = min_spacing / (16.0 * f64::from(slice_count));
    let spacing = 1.0_f64 / f64::from(vertices_per_slice - 1);
    let temporal_index = f64::from(slice);
    let temporal_span = f64::from(slice_count - 1);
    let side_arc = if temporal_span.abs() < f64::EPSILON {
        0.0
    } else {
        side_jitter * temporal_index * (temporal_span - temporal_index) / temporal_span.powi(2)
    };
    let x = if index == 0 || index == vertices_per_slice - 1 {
        let boundary = f64::from(index).mul_add(spacing, side_jitter);
        if index == 0 {
            boundary - side_arc
        } else {
            boundary + side_arc
        }
    } else {
        let sign = if (index + slice).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        f64::from(index).mul_add(spacing, side_jitter) + sign * interior_jitter
    };
    let spatial_index = f64::from(index);
    let arc = vertical_jitter * spatial_index * f64::from(vertices_per_slice - 1 - index)
        / f64::from((vertices_per_slice - 1).pow(2));
    let base_y = f64::from(slice);
    let y = if slice == 0 {
        base_y - arc
    } else if slice + 1 == slice_count {
        base_y + arc
    } else {
        let sign = if (index + slice).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        (sign * arc).mul_add(0.5, base_y)
    };
    ([x, y], slice)
}

fn exact_open_cdt_strip_vertices(vertices_per_slice: u32, slice_count: u32) -> Vec<Vertex<u32, 2>> {
    let total_vertices = usize::try_from(vertices_per_slice)
        .expect("test vertices per slice fits usize")
        .saturating_mul(usize::try_from(slice_count).expect("test slice count fits usize"));
    let mut vertices = Vec::with_capacity(total_vertices);
    for slice in 0..slice_count {
        for index in 0..vertices_per_slice {
            let ([x, y], label) =
                open_cdt_strip_vertex(slice, index, vertices_per_slice, slice_count, 0.0);
            vertices.push(vertex![x, y; data = label].expect("finite layered strip vertex"));
        }
    }
    vertices
}

fn exact_open_cdt_strip_simplices(vertices_per_slice: u32, slice_count: u32) -> Vec<Vec<usize>> {
    let vertices_per_slice =
        usize::try_from(vertices_per_slice).expect("test vertices per slice fits usize");
    let slice_count = usize::try_from(slice_count).expect("test slice count fits usize");
    let mut simplices = Vec::with_capacity(
        2 * vertices_per_slice
            .saturating_sub(1)
            .saturating_mul(slice_count.saturating_sub(1)),
    );

    for slice in 0..slice_count.saturating_sub(1) {
        for index in 0..vertices_per_slice.saturating_sub(1) {
            let lower_left = slice * vertices_per_slice + index;
            let lower_right = lower_left + 1;
            let upper_left = (slice + 1) * vertices_per_slice + index;
            let upper_right = upper_left + 1;
            simplices.push(vec![lower_left, lower_right, upper_right]);
            simplices.push(vec![lower_left, upper_right, upper_left]);
        }
    }

    simplices
}

fn sorted_vertex_signatures(vertices: &[Vertex<u32, 2>]) -> Vec<(u64, u64, u32)> {
    let mut signatures: Vec<_> = vertices
        .iter()
        .map(|vertex| {
            let coords = vertex.point().coords();
            (
                coords[0].to_bits(),
                coords[1].to_bits(),
                *vertex.data().expect("strip vertices are labeled"),
            )
        })
        .collect();
    signatures.sort_unstable();
    signatures
}

fn sorted_triangulation_vertex_signatures(
    dt: &DelaunayTriangulation<RobustKernel<f64>, u32, i32, 2>,
) -> Vec<(u64, u64, u32)> {
    let mut signatures: Vec<_> = dt
        .vertices()
        .map(|(_, vertex)| {
            let coords = vertex.point().coords();
            (
                coords[0].to_bits(),
                coords[1].to_bits(),
                *vertex.data().expect("strip vertices are labeled"),
            )
        })
        .collect();
    signatures.sort_unstable();
    signatures
}

fn assert_strip_vertices_use_exact_time_labels(vertices: &[Vertex<u32, 2>]) {
    for vertex in vertices {
        let coords = vertex.point().coords();
        let label = *vertex.data().expect("strip vertices are labeled");
        assert_eq!(
            coords[1].to_bits(),
            f64::from(label).to_bits(),
            "strip vertex y coordinate should exactly encode its time label: vertex={vertex:?}",
        );
    }
}

fn assert_triangulation_vertices_use_exact_time_labels(
    dt: &DelaunayTriangulation<RobustKernel<f64>, u32, i32, 2>,
) {
    for (_, vertex) in dt.vertices() {
        let coords = vertex.point().coords();
        let label = *vertex.data().expect("strip vertices are labeled");
        assert_eq!(
            coords[1].to_bits(),
            f64::from(label).to_bits(),
            "constructed strip vertex y coordinate should exactly encode its time label: vertex={vertex:?}",
        );
    }
}

fn assert_exact_strip_construction_result(
    case: &str,
    dt: &DelaunayTriangulation<RobustKernel<f64>, u32, i32, 2>,
    stats: &ConstructionStatistics,
    input_signatures: &[(u64, u64, u32)],
) {
    assert_eq!(
        dt.number_of_vertices(),
        input_signatures.len(),
        "{case} construction should preserve all distinct strip vertices; stats={stats:?}",
    );
    assert_eq!(
        stats.total_skipped(),
        0,
        "{case} construction should not skip collinear strip vertices; stats={stats:?}",
    );
    assert_eq!(
        stats.used_perturbation, 0,
        "{case} construction should not physically perturb strip vertices; stats={stats:?}",
    );
    assert_eq!(
        sorted_triangulation_vertex_signatures(dt).as_slice(),
        input_signatures,
        "{case} construction should preserve exact strip coordinate bits and labels; stats={stats:?}",
    );
    assert_triangulation_vertices_use_exact_time_labels(dt);
    dt.as_triangulation()
        .validate()
        .expect("exact degenerate strip should satisfy Levels 1-4");
}

#[test]
fn regression_issue_447_exact_layered_strip_preserves_collinear_boundary_vertices() {
    let vertices = exact_open_cdt_strip_vertices(5, 3);
    let kernel = RobustKernel::<f64>::new();
    let input_signatures = sorted_vertex_signatures(&vertices);
    assert_strip_vertices_use_exact_time_labels(&vertices);
    let exact_degenerate_options =
        ConstructionOptions::default().without_final_delaunay_enforcement();
    assert!(
        !exact_degenerate_options.enforces_final_delaunay(),
        "exact degenerate construction mode should document that Level 5 enforcement is disabled",
    );

    let (default_dt, default_stats) =
        DelaunayTriangulation::<_, u32, i32, 2>::try_with_options_and_statistics(
            &kernel,
            &vertices,
            TopologyGuarantee::DEFAULT,
            exact_degenerate_options,
        )
        .expect("exact layered CDT strip point construction should succeed");

    assert_exact_strip_construction_result(
        "exact degenerate",
        &default_dt,
        &default_stats,
        &input_signatures,
    );

    let input_options = ConstructionOptions::default()
        .with_insertion_order(InsertionOrderStrategy::Input)
        .with_retry_policy(RetryPolicy::Disabled)
        .without_final_delaunay_enforcement();
    let (input_dt, input_stats) =
        DelaunayTriangulation::<_, u32, i32, 2>::try_with_options_and_statistics(
            &kernel,
            &vertices,
            TopologyGuarantee::Pseudomanifold,
            input_options,
        )
        .expect("input-order exact layered CDT strip construction should succeed");

    assert_exact_strip_construction_result(
        "input-order",
        &input_dt,
        &input_stats,
        &input_signatures,
    );

    let no_stats_dt =
        DelaunayTriangulation::<_, u32, i32, 2>::try_with_topology_guarantee_and_options(
            &kernel,
            &vertices,
            TopologyGuarantee::Pseudomanifold,
            ConstructionOptions::default()
                .with_insertion_order(InsertionOrderStrategy::Input)
                .with_retry_policy(RetryPolicy::Disabled)
                .without_final_delaunay_enforcement(),
        )
        .expect("non-stat exact layered CDT strip construction should honor non-enforcing policy");

    assert_eq!(
        no_stats_dt.number_of_vertices(),
        vertices.len(),
        "non-stat construction should preserve all exact strip vertices",
    );
    assert_eq!(
        sorted_triangulation_vertex_signatures(&no_stats_dt),
        input_signatures,
        "non-stat construction should preserve exact strip coordinate bits and labels",
    );
    assert_triangulation_vertices_use_exact_time_labels(&no_stats_dt);
    no_stats_dt
        .as_triangulation()
        .validate()
        .expect("non-stat exact strip should satisfy Levels 1-4");
}

#[test]
fn regression_issue_447_explicit_exact_strip_default_strict_level5_fails() {
    let vertices = exact_open_cdt_strip_vertices(5, 3);
    let simplices = exact_open_cdt_strip_simplices(5, 3);

    let err = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
        .expect("exact CDT strip explicit simplex specs should validate")
        .build::<i32>()
        .expect_err("strict explicit construction should reject the non-Delaunay CDT strip");

    assert!(
        matches!(
            err,
            DelaunayTriangulationConstructionError::ExplicitConstruction(
                ExplicitConstructionError::DelaunayValidation { .. }
            )
        ),
        "strict explicit construction should fail at Level 5 Delaunay validation, got: {err:?}",
    );
}

#[test]
fn regression_issue_447_explicit_exact_strip_preserves_vertices_without_level5_enforcement() {
    let vertices = exact_open_cdt_strip_vertices(5, 3);
    let simplices = exact_open_cdt_strip_simplices(5, 3);
    let kernel = RobustKernel::<f64>::new();
    let input_signatures = sorted_vertex_signatures(&vertices);
    assert_strip_vertices_use_exact_time_labels(&vertices);

    let dt = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
        .expect("exact CDT strip explicit simplex specs should validate")
        .construction_options(
            ConstructionOptions::default()
                .without_final_delaunay_enforcement()
                .with_batch_repair_policy(DelaunayRepairPolicy::EveryN(
                    NonZeroUsize::new(2).unwrap(),
                )),
        )
        .build_with_kernel::<_, i32>(&kernel)
        .expect("explicit exact CDT strip should import under the non-enforcing policy");

    assert_eq!(
        dt.number_of_vertices(),
        vertices.len(),
        "explicit construction should preserve every exact strip vertex",
    );
    assert_eq!(
        dt.number_of_simplices(),
        simplices.len(),
        "explicit construction should preserve the supplied strip connectivity",
    );
    assert_eq!(
        sorted_triangulation_vertex_signatures(&dt),
        input_signatures,
        "explicit construction should preserve exact strip coordinate bits and labels",
    );
    assert_triangulation_vertices_use_exact_time_labels(&dt);
    dt.as_triangulation()
        .validate()
        .expect("explicit exact CDT strip should satisfy Levels 1-4");
}

/// Locks the equivalence between the single-pass proof-carrying batch quantizer
/// used by Hilbert construction ordering and the original two-step
/// `quantize` + `hilbert_indices_prequantized` path.
///
/// `order_vertices_hilbert` switched to `hilbert_quantize_batch_in_range` to
/// drop a redundant per-coordinate range rescan (and a per-point quantization
/// scale recompute). This regression guards that the change does not alter the
/// quantized cells or Hilbert indices — and therefore the deterministic
/// insertion order — across representative dimensions and adversarial inputs.
#[test]
fn regression_hilbert_batch_quantize_matches_two_step_path() {
    fn assert_paths_match<const D: usize>(points: &[Point<D>]) {
        let bounds = coordinate_bounds(points);
        // Mirror `order_vertices_hilbert`'s per-dimension precision so the
        // `D * bits <= 128` index-width invariant holds for every D.
        let bits_per_coord = (128_u32 / u32::try_from(D).expect("dimension fits in u32")).min(31);
        let bits = HilbertBitDepth::try_new(bits_per_coord).expect("test bit depth must be valid");

        // Original two-step path: per-point quantize, then bulk index.
        let two_step_quantized: Vec<[u32; D]> = points
            .iter()
            .map(|point| {
                hilbert_quantize_in_range(point.coords(), bounds, bits)
                    .expect("finite points should quantize")
            })
            .collect();
        let two_step_indices = hilbert_indices_prequantized(&two_step_quantized, bits)
            .expect("indices should fit in u128");

        // New single-pass proof-carrying batch path.
        let batch = hilbert_quantize_batch_in_range(points, bounds, bits, |point| *point.coords())
            .expect("finite points should quantize");
        let (batch_indices, batch_quantized) = batch.into_indices_and_coordinates();

        assert_eq!(
            batch_quantized, two_step_quantized,
            "batch quantizer must produce identical quantized cells in {D}D"
        );
        assert_eq!(
            batch_indices, two_step_indices,
            "batch quantizer must produce identical Hilbert indices in {D}D"
        );
    }

    // Adversarial mixes: negative/asymmetric ranges, clamping at both ends,
    // duplicate cells, and exact endpoints.
    assert_paths_match::<2>(&[
        Point::try_new([-2.0, -1.0]).expect("finite point coordinates"),
        Point::try_new([-1.5, 0.25]).expect("finite point coordinates"),
        Point::try_new([0.1, -0.7]).expect("finite point coordinates"),
        Point::try_new([3.0, 3.0]).expect("finite point coordinates"),
        Point::try_new([3.0, 3.0]).expect("finite point coordinates"),
    ]);
    assert_paths_match::<3>(&[
        Point::try_new([-2.0, -1.0, 0.0]).expect("finite point coordinates"),
        Point::try_new([-1.5, 0.25, 1.75]).expect("finite point coordinates"),
        Point::try_new([0.1, -0.7, 2.2]).expect("finite point coordinates"),
        Point::try_new([3.0, 3.0, -2.0]).expect("finite point coordinates"),
    ]);
    assert_paths_match::<5>(&[
        Point::try_new([-2.0, -1.0, 0.0, 1.0, 2.0]).expect("finite point coordinates"),
        Point::try_new([-1.5, 0.25, 1.75, 2.5, -0.5]).expect("finite point coordinates"),
        Point::try_new([0.1, -0.7, 2.2, -1.8, 1.4]).expect("finite point coordinates"),
        Point::try_new([3.0, 3.0, -2.0, -2.0, 0.5]).expect("finite point coordinates"),
    ]);
}

#[test]
fn regression_empty_circumsphere_2d_minimal_case() {
    let vertices = vec![
        vertex!([48.564_246_621_452_234, 23.481_505_128_710_488]).unwrap(),
        vertex!([-9.807_184_344_740_996, -36.451_902_443_093_33]).unwrap(),
        vertex!([75.784_620_110_257_45, 25.382_048_382_678_306]).unwrap(),
        vertex!([50.330_335_525_698_53, 25.294_356_716_784_847]).unwrap(),
        vertex!([77.411_339_748_608_4, -86.531_849_594_875_54]).unwrap(),
        vertex!([-93.661_180_847_043, 1.562_430_007_326_195_9]).unwrap(),
    ];

    let mut dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::try_new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

    if dt.is_valid_delaunay().is_err() {
        #[cfg(feature = "diagnostics")]
        debug_print_first_delaunay_violation(dt.tds(), None);
    }

    dt.repair_delaunay_with_flips().unwrap();

    dt.as_triangulation()
        .validate_embedding()
        .expect("2D triangulation should preserve lower-layer invariants after global flip repair");
    assert!(
        dt.is_valid_delaunay().is_ok(),
        "2D triangulation should be a valid PL-manifold after global flip repair"
    );
}

#[test]
fn regression_issue_120_minimal_failing_input_2d() {
    // From docs/archive/issue_120_investigation.md (Example Failure Case (2D)).
    let vertices = vec![
        vertex!([0.0, 0.0]).unwrap(),
        vertex!([-54.687, 0.0]).unwrap(),
        vertex!([-85.026, 36.185]).unwrap(),
        vertex!([0.0, 38.424]).unwrap(),
    ];

    let dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::try_new_with_topology_guarantee(
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

#[test]
fn regression_insertion_error_preserves_top_level_retryability() {
    let source = InsertionError::HullExtension {
        reason: HullExtensionReason::NoVisibleFacets,
    };
    assert!(source.is_retryable());
}

#[test]
fn regression_periodic_neighbor_validation_uses_lifted_vertex_offsets() {
    let vertices: Vec<Vertex<(), 2>> = (0..7)
        .map(|index| {
            let index_f64 = f64::from(u32::try_from(index).expect("test index fits in u32"));
            vertex!([
                0.9_f64.mul_add(((index_f64 + 1.0) * 0.618_033_988_749_894_8).fract(), 0.05),
                0.9_f64.mul_add(((index_f64 + 1.0) * 0.414_213_562_373_095_03).fract(), 0.05),
            ])
            .unwrap()
        })
        .collect();
    let kernel = RobustKernel::<f64>::new();

    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .try_toroidal([1.0_f64; 2])
        .unwrap()
        .build_with_kernel::<_, ()>(&kernel)
        .expect("periodic 2D build should succeed");

    assert!(
        dt.simplices()
            .any(|(_, simplex)| simplex.periodic_vertex_offsets().is_some()),
        "periodic image-point construction should populate lifted per-simplex offsets"
    );
    assert!(
        dt.tds().is_valid().is_ok(),
        "neighbor validation must compare lifted (offset) identities"
    );
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
    let points = generate_random_points_in_ball_seeded::<3>(35, 100.0, seed)
        .expect("point generation should succeed");
    let vertices: Vec<Vertex<(), 3>> = points
        .into_iter()
        .map(|p| vertex!(p.into()).unwrap())
        .collect();

    let dt: Result<DelaunayTriangulation<_, (), (), 3>, _> =
        DelaunayTriangulation::try_new(&vertices);
    assert!(
        dt.is_ok(),
        "35-vertex 3D construction with seed 0x{seed:X} should succeed \
         (requires unified repair constants); got: {}",
        dt.unwrap_err()
    );
}

/// The first 14 vertices from the 100-point 4D seed used to leave one negative
/// simplex after bulk local repair, causing the next insertion to be skipped.
#[test]
fn regression_issue_307_4d_bulk_repair_keeps_positive_orientation() {
    let seed: u64 = 0x9B77_86C9_99C5_6A16;
    let points = generate_random_points_in_ball_seeded::<4>(100, 100.0, seed)
        .expect("point generation should succeed");
    let vertices = hilbert_ordered_prefix(points, 14);

    let kernel = RobustKernel::<f64>::new();
    let options = ConstructionOptions::default()
        .with_insertion_order(InsertionOrderStrategy::Input)
        .with_retry_policy(RetryPolicy::Disabled);
    let (dt, stats) =
        DelaunayTriangulation::<RobustKernel<f64>, (), (), 4>::try_with_options_and_statistics(
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
        dt.as_triangulation().is_valid_topology().is_ok(),
        "bulk repair must leave all simplices in positive geometric orientation",
    );
    assert!(
        dt.as_triangulation().validate().is_ok(),
        "bulk repair must leave the triangulation structurally and topologically valid",
    );
}

/// The 4D 500-point seed `0xD225B8A07E274AE6` (ball radius 100) exhausted all
/// shuffled retries before #204: every attempt finished with skip-heavy output
/// (`inserted≈266–300`, `skipped≈200–234`) and the construction ultimately
/// failed with `Simplex violates Delaunay property: simplex contains vertex that is
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

    let points = generate_random_points_in_ball_seeded::<4>(n_points, ball_radius, seed)
        .expect("point generation should succeed");
    let vertices: Vec<Vertex<(), 4>> = points
        .into_iter()
        .map(|p| vertex!(p.into()).unwrap())
        .collect();

    let (dt, stats) =
        DelaunayTriangulation::<_, (), (), 4>::try_new_with_construction_statistics(&vertices)
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
