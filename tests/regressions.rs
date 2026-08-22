//! Regression tests for fixed Delaunay triangulation bugs.
//!
//! Add new fixed-bug regression cases here rather than creating issue-specific
//! integration test crates, unless the case needs separate crate-level setup,
//! feature flags, or profile isolation.

use delaunay::flips::{BistellarFlips, FlipError, SimplexKey};
use delaunay::prelude::construction::{
    ConstructionOptions, ConstructionStatistics, DelaunayIncrementalBuilder, DelaunayTriangulation,
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError,
    ExplicitConstructionError, InsertionOrderStrategy, RetryPolicy, TopologyGuarantee, Vertex,
};
use delaunay::prelude::delaunayize::{DelaunayRefinementBuilder, DelaunayizeError};
use delaunay::prelude::generators::generate_random_points_in_ball_seeded;
use delaunay::prelude::geometry::{
    CoordinateRange, InSphere, Point, RobustKernel, insphere, insphere_lifted,
};
use delaunay::prelude::insertion::{HullExtensionReason, InsertionError};
use delaunay::prelude::ordering::{
    HilbertBitDepth, hilbert_indices_prequantized, hilbert_quantize_batch_in_range,
    hilbert_quantize_in_range,
};
use delaunay::prelude::pachner::{PachnerMove, PachnerMoves};
use delaunay::prelude::repair::DelaunayRepairError;
use delaunay::prelude::tds::{InvariantError, Tds};
use delaunay::prelude::topology::spaces::{GlobalTopology, TopologyKind};
use delaunay::prelude::triangulation::{
    Triangulation, TriangulationBuilder, TriangulationBuilderError,
};
use delaunay::prelude::validation::{
    DelaunayTriangulationValidationError, TriangulationRealizationValidationError,
    TriangulationValidationError, ValidationPolicy,
};
use delaunay::vertex;
use uuid::Uuid;

#[test]
fn regression_exact_insphere_methods_agree_on_clean_2d_boundary_and_interior() {
    let simplex = [
        Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
        Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
        Point::try_new([0.0, 1.0]).expect("finite point coordinates"),
    ];
    let edge_midpoint = Point::try_new([0.5, 0.0]).expect("finite point coordinates");
    let opposite_boundary = Point::try_new([1.0, 1.0]).expect("finite point coordinates");

    assert_eq!(
        insphere(&simplex, edge_midpoint).expect("absolute predicate should succeed"),
        InSphere::INSIDE
    );
    assert_eq!(
        insphere_lifted(&simplex, edge_midpoint).expect("relative predicate should succeed"),
        InSphere::INSIDE
    );
    assert_eq!(
        insphere(&simplex, opposite_boundary).expect("absolute predicate should succeed"),
        InSphere::BOUNDARY
    );
    assert_eq!(
        insphere_lifted(&simplex, opposite_boundary).expect("relative predicate should succeed"),
        InSphere::BOUNDARY
    );
}

#[test]
fn regression_issue_557_builder_validation_policy_is_order_independent() {
    let vertices = [
        vertex!([0.0_f64, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
    ];
    let options =
        ConstructionOptions::default().with_insertion_order(InsertionOrderStrategy::Input);

    let policy_then_options = DelaunayTriangulationBuilder::new(&vertices)
        .validation_policy(ValidationPolicy::Always)
        .construction_options(options)
        .build()
        .expect("construction options must not overwrite the builder validation policy");
    let options_then_policy = DelaunayTriangulationBuilder::new(&vertices)
        .construction_options(options)
        .validation_policy(ValidationPolicy::Always)
        .build()
        .expect("builder setter order must not change the resulting policy");

    assert_eq!(
        policy_then_options.validation_policy(),
        ValidationPolicy::Always
    );
    assert_eq!(
        options_then_policy.validation_policy(),
        ValidationPolicy::Always
    );
    policy_then_options
        .validate()
        .expect("policy-first construction must preserve the Level 5 proof");
    options_then_policy
        .validate()
        .expect("options-first construction must preserve the Level 5 proof");
}

#[test]
fn regression_issue_557_delaunay_checkpoint_preserves_proof_context() {
    let vertices = [
        vertex!([0.0_f64, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
    ];
    let original: DelaunayTriangulation<RobustKernel<f64>, (), (), 2> =
        DelaunayTriangulationBuilder::new(&vertices)
            .topology_guarantee(TopologyGuarantee::Pseudomanifold)
            .validation_policy(ValidationPolicy::Never)
            .build_with_kernel(&RobustKernel::new())
            .expect("fixture construction should succeed");

    let checkpoint = serde_json::to_string(&original).expect("checkpoint should serialize");
    let restored: DelaunayTriangulation<RobustKernel<f64>, (), (), 2> =
        serde_json::from_str(&checkpoint).expect("checkpoint should re-prove Levels 1-5");

    assert_eq!(
        restored.topology_guarantee(),
        TopologyGuarantee::Pseudomanifold
    );
    assert_eq!(restored.global_topology(), GlobalTopology::Euclidean);
    assert_eq!(restored.validation_policy(), ValidationPolicy::Never);
    restored
        .validate()
        .expect("restored checkpoint must retain the cumulative Level 5 proof");

    let legacy_tds_json = serde_json::to_string(&original.into_triangulation().into_tds())
        .expect("legacy fixture should serialize");
    serde_json::from_str::<DelaunayTriangulation<RobustKernel<f64>, (), (), 2>>(&legacy_tds_json)
        .expect_err("a TDS-only payload must not silently acquire default owner context");
}

#[test]
fn regression_issue_557_delaunay_checkpoint_rejects_incompatible_policy() {
    let vertices = [
        vertex!([0.0_f64, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
    ];
    let original: DelaunayTriangulation<RobustKernel<f64>, (), (), 2> =
        DelaunayTriangulationBuilder::new(&vertices)
            .topology_guarantee(TopologyGuarantee::Pseudomanifold)
            .validation_policy(ValidationPolicy::Never)
            .build_with_kernel(&RobustKernel::new())
            .expect("fixture construction should succeed");

    let mut checkpoint = serde_json::to_value(&original).expect("checkpoint should serialize");
    checkpoint["topology_guarantee"] = serde_json::json!("pl_manifold");

    let error =
        serde_json::from_value::<DelaunayTriangulation<RobustKernel<f64>, (), (), 2>>(checkpoint)
            .expect_err("PL-manifold checkpoints must reject ValidationPolicy::Never");
    let message = error.to_string();
    assert!(message.contains("incompatible"));
    assert!(message.contains("PLManifold"));
    assert!(message.contains("Never"));
}

#[test]
fn regression_issue_557_triangulation_topology_setter_preserves_levels_one_through_four() {
    let mut tri: Triangulation<RobustKernel<f64>, (), (), 2> =
        TriangulationBuilder::new(Tds::empty(), RobustKernel::new())
            .build()
            .unwrap();
    let previous_topology = tri.global_topology();

    let error = tri
        .try_set_global_topology(GlobalTopology::Hyperbolic)
        .expect_err("unsupported Level 4 topology metadata must be rejected");

    assert!(matches!(
        error,
        InvariantError::Realization { source }
            if matches!(
                source,
                TriangulationRealizationValidationError::UnsupportedTopology {
                    topology: TopologyKind::Hyperbolic,
                    dimension: 2,
                }
            )
    ));
    assert_eq!(tri.global_topology(), previous_topology);
    tri.validate_realization()
        .expect("failed topology metadata update must preserve the prior Levels 1-4 value");
}

#[test]
fn regression_issue_557_delaunay_topology_setter_preserves_levels_one_through_five() {
    let mut dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayIncrementalBuilder::new().finish().unwrap();
    let previous_topology = dt.global_topology();

    let error = dt
        .try_set_global_topology(GlobalTopology::Hyperbolic)
        .expect_err("unsupported Level 4 topology metadata must be rejected");

    match error {
        DelaunayTriangulationValidationError::Realization { source } => {
            assert!(matches!(
                source.as_ref(),
                TriangulationRealizationValidationError::UnsupportedTopology {
                    topology: TopologyKind::Hyperbolic,
                    dimension: 2,
                }
            ));
        }
        other => panic!("expected a Level 4 realization error, got {other:?}"),
    }
    assert_eq!(dt.global_topology(), previous_topology);
    dt.validate()
        .expect("failed topology metadata update must preserve the prior Level 5 value");
}

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
    dt: &Triangulation<RobustKernel<f64>, u32, i32, 2>,
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
    dt: &Triangulation<RobustKernel<f64>, u32, i32, 2>,
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
    dt: &Triangulation<RobustKernel<f64>, u32, i32, 2>,
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
    dt.validate_realization()
        .expect("exact degenerate strip should satisfy Levels 1-4");
}

#[test]
fn regression_issue_447_exact_layered_strip_preserves_collinear_boundary_vertices() {
    let vertices = exact_open_cdt_strip_vertices(5, 3);
    let kernel = RobustKernel::<f64>::new();
    let input_signatures = sorted_vertex_signatures(&vertices);
    assert_strip_vertices_use_exact_time_labels(&vertices);
    let (default_dt, default_stats) = DelaunayTriangulationBuilder::new(&vertices)
        .simplex_data_type::<i32>()
        .build_triangulation_with_kernel_and_statistics(&kernel)
        .expect("exact layered CDT strip point construction should succeed");

    assert_exact_strip_construction_result(
        "exact degenerate",
        &default_dt,
        &default_stats,
        &input_signatures,
    );

    let input_options = ConstructionOptions::default()
        .with_insertion_order(InsertionOrderStrategy::Input)
        .with_retry_policy(RetryPolicy::Disabled);
    let (input_dt, input_stats) = DelaunayTriangulationBuilder::new(&vertices)
        .simplex_data_type::<i32>()
        .topology_guarantee(TopologyGuarantee::Pseudomanifold)
        .construction_options(input_options)
        .build_triangulation_with_kernel_and_statistics(&kernel)
        .expect("input-order exact layered CDT strip construction should succeed");

    assert_exact_strip_construction_result(
        "input-order",
        &input_dt,
        &input_stats,
        &input_signatures,
    );

    let no_stats_dt = DelaunayTriangulationBuilder::new(&vertices)
        .simplex_data_type::<i32>()
        .topology_guarantee(TopologyGuarantee::Pseudomanifold)
        .construction_options(
            ConstructionOptions::default()
                .with_insertion_order(InsertionOrderStrategy::Input)
                .with_retry_policy(RetryPolicy::Disabled),
        )
        .build_triangulation_with_kernel(&kernel)
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
        .validate_realization()
        .expect("non-stat exact strip should satisfy Levels 1-4");
}

#[test]
fn regression_issue_447_explicit_exact_strip_attempts_repair_before_failing() {
    let vertices = exact_open_cdt_strip_vertices(5, 3);
    let simplices = exact_open_cdt_strip_simplices(5, 3);

    let err = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
        .expect("exact CDT strip explicit simplex specs should validate")
        .simplex_data_type::<i32>()
        .build()
        .expect_err("bounded repair should report non-convergence for the constrained CDT strip");

    let DelaunayTriangulationConstructionError::ExplicitConstruction {
        source: ExplicitConstructionError::DelaunayRepair { source },
    } = err
    else {
        panic!("strict explicit construction should preserve the repair failure, got: {err:?}");
    };
    assert!(
        matches!(
            source.as_ref(),
            DelaunayRepairError::NonConvergent { diagnostics, .. }
                if diagnostics.flips_performed > 0
        ),
        "strict construction must attempt at least one flip before reporting non-convergence: {source:?}",
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
        .simplex_data_type::<i32>()
        .build_triangulation_with_kernel(&kernel)
        .expect("explicit exact CDT strip should import through the Levels 1-4 terminal");

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
    dt.validate_realization()
        .expect("explicit exact CDT strip should satisfy Levels 1-4");
    assert!(
        DelaunayRefinementBuilder::new(dt).build().is_err(),
        "the imported strip must not cross the strict Level 5 boundary",
    );
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

    let tri = DelaunayTriangulation::builder(&vertices)
        .topology_guarantee(TopologyGuarantee::PLManifold)
        .build_triangulation()
        .unwrap();

    let dt = DelaunayRefinementBuilder::new(tri)
        .repair_by_flips()
        .build()
        .expect("Levels 1–4 fixture should convert through bounded Delaunay repair")
        .triangulation;

    dt.as_triangulation()
        .validate_realization()
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

    let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::builder(&vertices)
        .topology_guarantee(TopologyGuarantee::PLManifold)
        .build()
        .unwrap();

    if let Err(err) = dt.validate() {
        #[cfg(feature = "diagnostics")]
        dt.debug_print_first_delaunay_violation(None);
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

/// Builds the deterministic periodic T² fixture shared by regressions #536 and #551.
fn periodic_regression_fixture_t2() -> DelaunayTriangulation<RobustKernel<f64>, (), (), 2> {
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

    DelaunayTriangulationBuilder::new(&vertices)
        .try_toroidal([1.0_f64; 2])
        .unwrap()
        .build_with_kernel(&kernel)
        .expect("periodic T^2 build should succeed")
}

/// Finds a periodic simplex by exact coordinate bits and lattice offsets.
fn periodic_simplex_key(
    dt: &Triangulation<RobustKernel<f64>, (), (), 2>,
    expected: [([f64; 2], [i8; 2]); 3],
) -> SimplexKey {
    let mut expected: Vec<_> = expected
        .into_iter()
        .map(|(coords, offset)| (coords.map(f64::to_bits), offset))
        .collect();
    expected.sort_unstable();

    dt.simplices()
        .find_map(|(simplex_key, simplex)| {
            let offsets = simplex.periodic_vertex_offsets()?;
            let mut actual: Vec<_> = simplex
                .vertices()
                .iter()
                .copied()
                .zip(offsets.iter().copied())
                .map(|(vertex_key, offset)| {
                    let coords = *dt
                        .vertex(vertex_key)
                        .expect("simplex should reference a live vertex")
                        .point()
                        .coords();
                    (coords.map(f64::to_bits), offset)
                })
                .collect();
            actual.sort_unstable();
            (actual == expected).then_some(simplex_key)
        })
        .expect("periodic fixture should contain the requested lifted simplex")
}

type PeriodicSimplexSnapshot = (Uuid, Vec<(Uuid, [i8; 2])>, Vec<Option<Uuid>>);

#[derive(Debug, PartialEq, Eq)]
struct PeriodicTopologySnapshot {
    vertices: Vec<(Uuid, [u64; 2])>,
    simplices: Vec<PeriodicSimplexSnapshot>,
}

/// Captures canonical periodic topology and realization state through public views.
fn snapshot_periodic_topology(
    dt: &Triangulation<RobustKernel<f64>, (), (), 2>,
) -> PeriodicTopologySnapshot {
    let mut vertices: Vec<_> = dt
        .vertices()
        .map(|(_, vertex)| (vertex.uuid(), vertex.point().coords().map(f64::to_bits)))
        .collect();
    vertices.sort_unstable_by_key(|(uuid, _)| *uuid);

    let mut simplices: Vec<_> = dt
        .simplices()
        .map(|(_, simplex)| {
            let offsets = simplex
                .periodic_vertex_offsets()
                .expect("periodic simplex should carry lifted offsets");
            let vertices = simplex
                .vertices()
                .iter()
                .copied()
                .zip(offsets.iter().copied())
                .map(|(vertex_key, offset)| {
                    let uuid = dt
                        .vertex(vertex_key)
                        .expect("simplex should reference a live vertex")
                        .uuid();
                    (uuid, offset)
                })
                .collect();
            let neighbors = simplex
                .neighbors()
                .map(|keys| {
                    keys.map(|neighbor_key| {
                        neighbor_key.map(|key| {
                            dt.simplex(key)
                                .expect("neighbor should reference a live simplex")
                                .uuid()
                        })
                    })
                    .collect()
                })
                .unwrap_or_default();
            (simplex.uuid(), vertices, neighbors)
        })
        .collect();
    simplices.sort_unstable_by_key(|(uuid, _, _)| *uuid);

    PeriodicTopologySnapshot {
        vertices,
        simplices,
    }
}

/// Builds a payload-bearing periodic T^2 fixture for realized-state reconstruction.
fn periodic_payload_fixture_t2() -> DelaunayTriangulation<RobustKernel<f64>, u32, u32, 2> {
    let vertices: Vec<Vertex<u32, 2>> = (0_u32..7)
        .map(|index| {
            let index_f64 = f64::from(index);
            vertex!(
                [
                    0.9_f64.mul_add(
                        ((index_f64 + 1.0) * 0.618_033_988_749_894_8).fract(),
                        0.05,
                    ),
                    0.9_f64.mul_add(
                        ((index_f64 + 1.0) * 0.414_213_562_373_095_03).fract(),
                        0.05,
                    ),
                ];
                data = index
            )
            .unwrap()
        })
        .collect();
    let kernel = RobustKernel::<f64>::new();
    let mut dt = DelaunayTriangulationBuilder::new(&vertices)
        .simplex_data_type::<u32>()
        .try_toroidal([1.0_f64; 2])
        .unwrap()
        .build_with_kernel(&kernel)
        .expect("periodic payload-bearing T^2 build should succeed");
    let mut next_simplex_payload = 100_u32;
    dt.fill_simplex_data(|_, _| {
        let payload = next_simplex_payload;
        next_simplex_payload += 1;
        payload
    });
    dt
}

/// Applies a realized-geometry-preserving move that intentionally breaks Level 5.
fn evolve_periodic_fixture_without_delaunay(
    dt: &DelaunayTriangulation<RobustKernel<f64>, u32, u32, 2>,
) -> Triangulation<RobustKernel<f64>, u32, u32, 2> {
    let triangulation = dt.clone().into_triangulation();
    let facets: Vec<_> = triangulation
        .facets()
        .map(|facet| {
            facet
                .expect("periodic fixture facets should reborrow")
                .handle()
        })
        .collect();

    for facet in facets {
        let mut trial = triangulation.clone();
        let Ok(proposal) = trial.propose_pachner(PachnerMove::K2 { facet }) else {
            continue;
        };
        if proposal.attempt_on(&mut trial).is_ok()
            && DelaunayRefinementBuilder::new(trial.clone())
                .build()
                .is_err()
            && DelaunayRefinementBuilder::new(trial.clone())
                .repair_by_flips()
                .build()
                .is_ok()
        {
            return trial;
        }
    }

    panic!(
        "periodic fixture should contain a realized k=2 move that breaks Level 5 and is repairable"
    );
}

fn assert_composed_level_five_failure_returns_triangulation(
    evolved_tds: Tds<u32, u32, 2>,
    expected_snapshot: &serde_json::Value,
    topology_guarantee: TopologyGuarantee,
    global_topology: GlobalTopology<2>,
) {
    let triangulation = TriangulationBuilder::new(evolved_tds, RobustKernel::new())
        .topology_guarantee(topology_guarantee)
        .global_topology(global_topology)
        .build()
        .expect("Levels 3-4 should succeed before Level 5 rejection");
    let failure = DelaunayRefinementBuilder::new(triangulation)
        .build()
        .expect_err("strict reconstruction must continue to reject the non-Delaunay state");
    assert_eq!(
        serde_json::to_value(failure.owner().clone().into_tds())
            .expect("composed recovery owner should serialize exactly"),
        *expected_snapshot,
        "strict Level 5 certification must return the unchanged Levels 1-4 owner"
    );
    failure
        .owner()
        .validate_realization()
        .expect("composed Level 5 rejection must retain a valid Levels 1-4 owner");
}

fn assert_level_three_failure_returns_tds(
    evolved_tds: Tds<u32, u32, 2>,
    topology_guarantee: TopologyGuarantee,
    global_topology: GlobalTopology<2>,
) {
    let mut invalid_snapshot = serde_json::to_value(evolved_tds)
        .expect("valid evolved TDS should serialize for invalid-topology fixture setup");
    let isolated_vertex = vertex!([0.42_f64, 0.42]; data = 999_u32)
        .expect("isolated regression vertex should be valid by itself");
    invalid_snapshot
        .get_mut("vertices")
        .and_then(serde_json::Value::as_array_mut)
        .expect("serialized TDS should contain vertex records")
        .push(
            serde_json::to_value(isolated_vertex)
                .expect("isolated regression vertex should serialize"),
        );
    let invalid_tds: Tds<u32, u32, 2> = serde_json::from_value(invalid_snapshot)
        .expect("isolated vertex should preserve Levels 1-2 snapshot validity");
    let expected_invalid_tds = serde_json::to_value(&invalid_tds)
        .expect("invalid-topology Levels 1-2 owner should serialize exactly");
    let invalid_error = TriangulationBuilder::new(invalid_tds, RobustKernel::new())
        .topology_guarantee(topology_guarantee)
        .global_topology(global_topology)
        .build()
        .expect_err("Levels 1-4 reconstruction must reject invalid toroidal topology");
    assert!(
        matches!(
            invalid_error.reason(),
            TriangulationBuilderError::TopologyValidation { source }
                if matches!(
                    source.as_ref(),
                    InvariantError::Triangulation {
                        source: TriangulationValidationError::IsolatedVertex { .. }
                    }
                )
        ),
        "invalid toroidal topology should report an isolated vertex: {invalid_error:?}"
    );
    invalid_error
        .owner()
        .validate()
        .expect("failed Levels 3-4 reconstruction must return the valid Levels 1-2 owner");
    assert_eq!(
        serde_json::to_value(invalid_error.owner())
            .expect("recovered Levels 1-2 owner should serialize exactly"),
        expected_invalid_tds,
        "failed Levels 3-4 reconstruction must return the unchanged TDS"
    );
}

#[test]
fn regression_issue_557_initial_simplex_publishes_a_valid_constructed_tds() {
    let vertices = [
        vertex!([0.0, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 1.0]).unwrap(),
    ];

    let tds = Triangulation::<RobustKernel<f64>, (), (), 3>::build_initial_simplex(&vertices)
        .expect("the 3D bootstrap should publish a complete TDS");

    tds.validate()
        .expect("the published bootstrap TDS must carry the Levels 1–2 proof");
}

#[test]
fn regression_issue_557_restores_evolved_toroidal_state_through_level_4() {
    let fresh = periodic_payload_fixture_t2();
    fresh
        .validate()
        .expect("fresh T^2 state must pass Levels 1-5");
    assert!(fresh.global_topology().is_toroidal());

    let mut evolved = evolve_periodic_fixture_without_delaunay(&fresh);
    let simplex_keys: Vec<_> = evolved.simplices().map(|(key, _)| key).collect();
    for (next_simplex_payload, simplex_key) in (1_000_u32..).zip(simplex_keys) {
        evolved
            .set_simplex_data(simplex_key, Some(next_simplex_payload))
            .expect("evolved simplex payload assignment should preserve topology");
    }
    evolved
        .validate_realization()
        .expect("evolved T^2 state must preserve Levels 1-4");
    assert!(
        DelaunayRefinementBuilder::new(evolved.clone())
            .build()
            .is_err()
    );

    let topology_guarantee = evolved.topology_guarantee();
    let global_topology = evolved.global_topology();
    let serialized = serde_json::to_string(&evolved.into_tds())
        .expect("evolved toroidal triangulation should serialize as TDS data");
    let evolved_tds: Tds<u32, u32, 2> = serde_json::from_str(&serialized)
        .expect("serialized evolved state should hydrate into validated TDS storage");
    let expected_snapshot = serde_json::to_value(&evolved_tds)
        .expect("validated evolved TDS should retain a durable snapshot");

    let restored = TriangulationBuilder::new(evolved_tds.clone(), RobustKernel::new())
        .topology_guarantee(topology_guarantee)
        .global_topology(global_topology)
        .build()
        .expect("Levels 1-4 reconstruction should accept evolved toroidal state");

    assert_eq!(restored.topology_guarantee(), topology_guarantee);
    assert_eq!(restored.global_topology(), global_topology);
    assert_eq!(
        serde_json::to_value(restored.clone().into_tds())
            .expect("restored triangulation storage should serialize exactly"),
        expected_snapshot,
        "reconstruction must preserve connectivity, periodic offsets, UUIDs, and payloads"
    );
    restored
        .validate_realization()
        .expect("restored T^2 state must preserve Levels 1-4");

    let strict_failure = DelaunayRefinementBuilder::new(restored)
        .build()
        .expect_err("strict certification must reject the non-Delaunay triangulation");
    let (restored, strict_reason) = strict_failure.into_parts();
    assert!(matches!(
        strict_reason,
        DelaunayTriangulationValidationError::VerificationFailed { .. }
    ));
    assert_eq!(
        serde_json::to_value(restored.clone().into_tds())
            .expect("recovered triangulation storage should serialize exactly"),
        expected_snapshot,
        "failed Level 5 certification must return the unchanged Levels 1-4 owner"
    );
    restored
        .validate_realization()
        .expect("failed Level 5 certification must retain a valid Levels 1-4 owner");

    let converted = DelaunayRefinementBuilder::new(restored)
        .repair_by_flips()
        .build()
        .expect("bounded toroidal flip repair should convert the realized triangulation");
    assert_eq!(converted.triangulation.global_topology(), global_topology);
    assert_eq!(
        converted.triangulation.topology_guarantee(),
        topology_guarantee
    );
    converted
        .triangulation
        .validate()
        .expect("delaunayize must publish only a cumulative Levels 1-5 value");

    assert_composed_level_five_failure_returns_triangulation(
        evolved_tds.clone(),
        &expected_snapshot,
        topology_guarantee,
        global_topology,
    );
    assert_level_three_failure_returns_tds(evolved_tds, topology_guarantee, global_topology);
}

#[test]
fn regression_issue_557_failed_delaunayize_returns_original_triangulation() {
    let vertices = [vertex!([0.0]).unwrap(), vertex!([1.0]).unwrap()];
    let triangulation = DelaunayTriangulationBuilder::new(&vertices)
        .build()
        .expect("one-dimensional fixture should construct")
        .into_triangulation();
    let expected_snapshot = serde_json::to_value(triangulation.clone().into_tds())
        .expect("original Levels 1-4 owner should serialize exactly");

    let failure = DelaunayRefinementBuilder::new(triangulation)
        .repair_by_flips()
        .build()
        .expect_err("flip repair is unsupported in one dimension without fallback");
    let (triangulation, reason) = failure.into_parts();

    assert!(matches!(
        reason,
        DelaunayizeError::DelaunayRepairFailed {
            source: DelaunayRepairError::Flip { source },
        } if matches!(source.as_ref(), FlipError::UnsupportedDimension { dimension: 1 })
    ));
    assert_eq!(
        serde_json::to_value(triangulation.clone().into_tds())
            .expect("recovered Levels 1-4 owner should serialize exactly"),
        expected_snapshot,
        "failed repairing refinement must return the unchanged triangulation"
    );
    triangulation
        .validate_realization()
        .expect("failed repairing refinement must retain a valid Levels 1-4 owner");
}

#[test]
fn regression_periodic_neighbor_validation_uses_lifted_vertex_offsets() {
    let dt = periodic_regression_fixture_t2().into_triangulation();

    assert!(
        dt.simplices()
            .any(|(_, simplex)| simplex.periodic_vertex_offsets().is_some()),
        "periodic image-point construction should populate lifted per-simplex offsets"
    );
    assert!(
        dt.is_valid_structure().is_ok(),
        "neighbor validation must compare lifted (offset) identities"
    );
}

#[test]
fn regression_issue_551_periodic_k1_preflight_rejects_orientation_repair_failure() {
    let dt = periodic_regression_fixture_t2().into_triangulation();
    let simplex_key = periodic_simplex_key(
        &dt,
        [
            ([0.131_152_949_166_187_8, 0.113_961_030_586_907_43], [0, 0]),
            ([0.262_461_179_900_496_8, 0.795_584_412_305_938_3], [0, -1]),
            ([0.343_614_129_000_127_8, 0.859_545_442_942_608_2], [0, -1]),
        ],
    );
    let candidate = vertex!([0.05, 0.35]).expect("finite regression point");
    let candidate_uuid = candidate.uuid();
    let before = snapshot_periodic_topology(&dt);

    let preflight_error = dt
        .can_flip_k1_insert(simplex_key, &candidate)
        .expect_err("lifted-chart exterior point must fail immutable feasibility");
    assert!(matches!(
        preflight_error,
        FlipError::K1InsertionOutsideSimplex {
            simplex_key: rejected,
            ..
        } if rejected == simplex_key
    ));
    assert_eq!(
        snapshot_periodic_topology(&dt),
        before,
        "immutable preflight must not alter the TDS"
    );
    assert!(dt.vertex_key_from_uuid(&candidate_uuid).is_none());

    let mut trial = dt;
    let before_commit = snapshot_periodic_topology(&trial);
    let commit_error = trial
        .flip_k1_insert(simplex_key, candidate)
        .expect_err("commit must share the deterministic k=1 preflight");
    assert!(matches!(
        commit_error,
        FlipError::K1InsertionOutsideSimplex {
            simplex_key: rejected,
            ..
        } if rejected == simplex_key
    ));
    assert_eq!(
        snapshot_periodic_topology(&trial),
        before_commit,
        "rejected commit must not mutate"
    );
    trial
        .validate_realization()
        .expect("rejected insertion must preserve the original realization");
}

#[test]
fn regression_issue_551_periodic_k1_preflight_success_matches_commit() {
    let dt = periodic_regression_fixture_t2().into_triangulation();
    let simplex_key = periodic_simplex_key(
        &dt,
        [
            ([0.131_152_949_166_187_8, 0.113_961_030_586_907_43], [0, 0]),
            ([0.343_614_129_000_127_8, 0.859_545_442_942_608_2], [0, -1]),
            ([0.606_230_589_834_825_3, 0.422_792_206_212_024_2], [0, 0]),
        ],
    );
    let candidate = vertex!([0.360_332_556_000_380_3, 0.132_099_559_913_846_6])
        .expect("finite periodic interior point");
    let before = snapshot_periodic_topology(&dt);

    let feasibility = dt
        .can_flip_k1_insert(simplex_key, &candidate)
        .expect("strictly interior periodic point should pass preflight");
    assert_eq!(
        snapshot_periodic_topology(&dt),
        before,
        "successful preflight must not alter the TDS"
    );

    let mut trial = dt;
    let committed = trial
        .flip_k1_insert(simplex_key, candidate)
        .expect("successful periodic preflight should be executable");
    assert_eq!(feasibility.kind, committed.kind);
    assert_eq!(feasibility.direction, committed.direction);
    assert_eq!(feasibility.removed_simplices, committed.removed_simplices);
    assert_eq!(
        feasibility.removed_face_vertices,
        committed.removed_face_vertices
    );
    trial
        .validate_realization()
        .expect("successful periodic k=1 insertion must preserve Level 4 realization");
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
        DelaunayTriangulation::builder(&vertices).build();
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
    let (dt, stats) = DelaunayTriangulationBuilder::new(&vertices)
        .topology_guarantee(TopologyGuarantee::PLManifold)
        .validation_policy(ValidationPolicy::Always)
        .construction_options(options)
        .build_with_kernel_and_statistics(&kernel)
        .expect("4D bulk construction should not fail after repair orientation cleanup");

    assert_eq!(
        stats.inserted,
        vertices.len(),
        "all prefix vertices should insert without orientation-related skips",
    );
    assert_eq!(stats.total_skipped(), 0);
    assert_eq!(dt.validation_policy(), ValidationPolicy::Always);
    assert!(
        dt.as_triangulation().is_valid_topology().is_ok(),
        "bulk repair must leave all simplices in positive geometric orientation",
    );
    dt.as_triangulation()
        .validate_realization()
        .expect("bulk repair must leave the triangulation valid through Level 4");
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
/// Gated behind `slow-tests` because the deterministic 500-point 4D repair
/// workload exceeds the default-suite 10-second budget in release mode; run
/// with:
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

    let (dt, stats) = DelaunayTriangulation::builder(&vertices)
        .build_with_statistics()
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
    dt.as_triangulation()
        .validate_realization()
        .expect("#204 regression triangulation must pass Levels 1-4 validation");
}
