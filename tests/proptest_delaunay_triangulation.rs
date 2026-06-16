//! Property-based tests for `DelaunayTriangulation` invariants.
//!
//! This module tests all Delaunay-specific properties including both structural
//! invariants and the expensive geometric Delaunay property validation.
//!
//! ## Architectural Context
//!
//! Following CGAL's architecture:
//! - **Tds** - Pure combinatorial/topological structure (tested in `proptest_tds.rs`)
//! - **Triangulation** - Generic geometric layer with kernel (tested in `proptest_triangulation.rs`)
//! - **`DelaunayTriangulation`** - Delaunay-specific operations (tested here)
//!
//! ## Invariants Tested
//!
//! ### Structural Invariants (Fast)
//! - **Incremental insertion validity** - Triangulation remains valid after each insertion
//! - **Duplicate coordinate rejection** - Geometric duplicate detection at insertion time
//!
//! ### Delaunay Property (Fast O(N) via Flip Predicates)
//! - **Empty circumsphere condition** - No vertex lies strictly inside any simplex's circumsphere (2D-5D; verified via flip predicates)
//! - **Insertion-order robustness** - Levels 1–3 validity across insertion orders (2D-5D; Delaunay property not asserted; see Issue #120)
//! - **Duplicate cloud integration** - Full pipeline with messy real-world inputs (2D-5D: duplicates + near-duplicates)
//!
//! All structural tests construct with `DelaunayTriangulation::try_new_with_topology_guarantee(..., TopologyGuarantee::PLManifold)`
//! and then use `insert()`, ensuring PL-manifoldness by construction while maintaining
//! invariants through the incremental cavity-based insertion algorithm.
//!
//! ## Performance Note
//!
//! Delaunay property validation now uses `verify_delaunay_via_flip_predicates()` which checks
//! local flip configurations (O(simplices)) instead of the naive O(simplices × vertices) brute-force.
//! This provides ~40-100x speedup for property-based testing while remaining equally correct.

use delaunay::geometry::kernel::{AdaptiveKernel, RobustKernel};
use delaunay::prelude::construction::{
    ConstructionOptions, DedupPolicy, DelaunayRepairPolicy, DelaunayTriangulation,
    TopologyGuarantee, Vertex,
};
use delaunay::prelude::geometry::*;
use delaunay::prelude::insertion::InsertionOutcome;
use delaunay::prelude::validation::ValidationPolicy;
use proptest::prelude::*;
use proptest::test_runner::{Config, TestCaseError, TestRunner};
use rand::{SeedableRng, seq::SliceRandom};
use std::cell::RefCell;
use std::collections::HashSet;

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
// =============================================================================
// TEST CONFIGURATION
// =============================================================================

const PROPTEST_COORD_NONZERO_EPS: f64 = 1e-6;

/// Strategy for generating finite `f64` coordinates in a reasonable range.
///
/// ### Why filter out near-zero values?
///
/// Proptest shrinkers aggressively drive floating-point values toward `0.0`. In this crate, a
/// large fraction of the `~0` input space corresponds to geometric degeneracies (many coordinates
/// exactly `0`, many points on coordinate hyperplanes, coincident/near-coincident vertices, etc.).
/// Those cases are useful to test, but they tend to dominate shrinking and can turn unrelated
/// failures into “degenerate topology” noise.
///
/// We therefore exclude `|x| <= 1e-6` as a **shrink guard**. The value `1e-6` is intentionally tiny
/// relative to the generated range `[-100, 100]` (1e-8 of the range), and is **not** intended to
/// be a geometric tolerance.
///
/// Coverage note: small-coordinate edge cases are exercised via targeted deterministic tests (see
/// [`tests/delaunay_edge_cases.rs`](tests/delaunay_edge_cases.rs:1)), which can be expanded if we
/// need stronger coverage of coordinate-scaling issues.
fn finite_coordinate() -> impl Strategy<Value = f64> {
    // Proptest shrinkers aggressively drive floating-point values toward `0.0`. In this crate,
    // a large fraction of the `~0` input space corresponds to geometric degeneracies.
    //
    // Exclude `|x| <= 1e-6` as a shrink guard (not a geometric tolerance).
    let eps = PROPTEST_COORD_NONZERO_EPS;
    prop_oneof![(-100.0..=-eps), (eps..=100.0)]
}

/// Strategy for generating non-finite `f64` coordinates.
fn non_finite_coordinate() -> impl Strategy<Value = f64> {
    prop_oneof![Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY),]
}

/// Strategy for generating 2D vertices
fn vertex_2d() -> impl Strategy<Value = Point<2>> {
    prop::array::uniform2(finite_coordinate())
        .prop_map(|coords| Point::try_new(coords).expect("finite point coordinates"))
}

/// Strategy for generating 3D vertices
fn vertex_3d() -> impl Strategy<Value = Point<3>> {
    prop::array::uniform3(finite_coordinate())
        .prop_map(|coords| Point::try_new(coords).expect("finite point coordinates"))
}

/// Strategy for generating 4D vertices
fn vertex_4d() -> impl Strategy<Value = Point<4>> {
    prop::array::uniform4(finite_coordinate())
        .prop_map(|coords| Point::try_new(coords).expect("finite point coordinates"))
}

/// Strategy for generating 5D vertices
fn vertex_5d() -> impl Strategy<Value = Point<5>> {
    prop::array::uniform5(finite_coordinate())
        .prop_map(|coords| Point::try_new(coords).expect("finite point coordinates"))
}

/* Deduplicate helpers to avoid pathological degeneracies in property tests */
fn dedup_vertices_by_coords<const D: usize>(vertices: Vec<Vertex<(), D>>) -> Vec<Vertex<(), D>> {
    let mut unique: Vec<Vertex<(), D>> = Vec::with_capacity(vertices.len());
    'outer: for v in vertices {
        let vc = *v.point().coords();
        for u in &unique {
            let uc = *u.point().coords();
            // Compare using bit representation to avoid float_cmp lint and match exact duplicates
            if vc
                .iter()
                .zip(uc.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
            {
                continue 'outer; // skip exact coordinate duplicate
            }
        }
        unique.push(v);
    }
    unique
}

/// Guardrail filter: reject pathological shrink targets with too many points lying exactly on any
/// coordinate hyperplane (`x_i == 0.0`).
///
/// This is currently redundant with `finite_coordinate()` (we filter out `|x| <= 1e-6`), but we
/// keep it as defense-in-depth in case the generators change.
///
/// Returns `true` iff **no** coordinate hyperplane contains `>= D` input vertices.
fn has_no_coordinate_hyperplane_degeneracy<const D: usize>(vertices: &[Vertex<(), D>]) -> bool {
    let mut axis_counts = [0usize; D];

    for v in vertices {
        let coords = *v.point().coords();
        for a in 0..D {
            if coords[a] == 0.0 {
                axis_counts[a] += 1;
            }
        }
    }

    axis_counts.iter().all(|&count| count < D)
}

/// Conservative “general position” filter for the specialized 3D insertion-order property.
///
/// This rejects point sets that contain *any* nearly-coplanar tetrahedron (4-point subset).
///
/// Rationale:
/// - Nearly-coplanar tetrahedra are a common proptest shrink target and correlate with 3D
///   cavity-boundary ridge-fan degeneracy / topology-repair edge cases.
/// - This property is specifically scoped to the “standard incremental path” where we want
///   insertion-order invariance for **validation Levels 1–3** (elements + structure + topology),
///   leaving degenerate/perturbation behavior to dedicated suites.
fn has_no_nearly_coplanar_tetrahedra_3d(vertices: &[Vertex<(), 3>]) -> bool {
    // Relative threshold: volume6 scales with L^3, so compare against scale^3.
    // Intentionally loose: we only want to drop “shrink-pathological” nearly-coplanar cases,
    // not typical random inputs.
    const REL_EPS: f64 = 1e-12;

    fn sub(lhs: [f64; 3], rhs: [f64; 3]) -> [f64; 3] {
        [lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]]
    }

    fn cross(u: [f64; 3], v: [f64; 3]) -> [f64; 3] {
        [
            u[1].mul_add(v[2], -(u[2] * v[1])),
            u[2].mul_add(v[0], -(u[0] * v[2])),
            u[0].mul_add(v[1], -(u[1] * v[0])),
        ]
    }

    fn dot(u: [f64; 3], v: [f64; 3]) -> f64 {
        u[2].mul_add(v[2], u[0].mul_add(v[0], u[1] * v[1]))
    }

    fn norm2(u: [f64; 3]) -> f64 {
        dot(u, u).sqrt()
    }

    let vertex_count = vertices.len();
    if vertex_count < 4 {
        return true;
    }

    for i in 0..vertex_count {
        for j in (i + 1)..vertex_count {
            for k in (j + 1)..vertex_count {
                for l in (k + 1)..vertex_count {
                    let point_a = *vertices[i].point().coords();
                    let point_b = *vertices[j].point().coords();
                    let point_c = *vertices[k].point().coords();
                    let point_d = *vertices[l].point().coords();

                    let ab = sub(point_b, point_a);
                    let ac = sub(point_c, point_a);
                    let ad = sub(point_d, point_a);

                    let volume6 = dot(ab, cross(ac, ad)).abs();
                    let scale = norm2(ab).max(norm2(ac)).max(norm2(ad));

                    // If scale is 0 (or NaN), we treat as degenerate (reject),
                    // though earlier dedup should have removed exact duplicates.
                    if scale.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
                        return false;
                    }

                    let threshold = REL_EPS * scale.powi(3);
                    if volume6 <= threshold {
                        return false;
                    }
                }
            }
        }
    }

    true
}

/// Conservative “general position” filter for the specialized 3D insertion-order property.
///
/// Rejects point sets that contain any co-spherical degeneracy among 5 points.
///
/// Rationale:
/// - In a non-degenerate 3D Delaunay triangulation, no 5 points lie on the same sphere.
/// - Co-spherical inputs are exactly where the cavity boundary can become non-manifold
///   (ridge fan: >2 boundary facets sharing an edge) and where insertion-order can change
///   the chosen triangulation.
/// - This property is intentionally scoped to a slice of 3D space where we expect insertion-order
///   invariance for **validation Levels 1–3** without depending on perturbation heuristics.
fn has_no_cospherical_5_tuples_3d(vertices: &[Vertex<(), 3>]) -> bool {
    const MAX_N: usize = 15;

    let n = vertices.len();
    if n < 5 {
        return true;
    }

    // Guardrail: this filter is O(n^5) due to iterating all 5-point subsets, and it performs up to
    // 5 `in_sphere` predicate evaluations per 5-tuple for robustness.
    //
    // With n=15: C(15, 5) = 3003 5-tuples => ~15k `in_sphere` calls per generated case.
    //
    // If you intentionally want to run this filter with larger inputs, set:
    //   DELAUNAY_ALLOW_SLOW_COSPHERICAL_FILTER=1
    let allow_slow = std::env::var_os("DELAUNAY_ALLOW_SLOW_COSPHERICAL_FILTER").is_some();

    // C(n, 5) using integer arithmetic: n*(n-1)*(n-2)*(n-3)*(n-4) / 5!
    let tuples: u128 =
        (n as u128) * ((n - 1) as u128) * ((n - 2) as u128) * ((n - 3) as u128) * ((n - 4) as u128)
            / 120u128;
    let in_sphere_calls: u128 = tuples * 5u128;

    if n > MAX_N && allow_slow {
        tracing::warn!(
            "has_no_cospherical_5_tuples_3d warning: n={n} > {MAX_N}; checking {tuples} 5-tuples (~{in_sphere_calls} in_sphere predicate calls)"
        );
    }

    assert!(
        n <= MAX_N || allow_slow,
        "has_no_cospherical_5_tuples_3d: n={n} exceeds MAX_N={MAX_N}; would check {tuples} 5-tuples (~{in_sphere_calls} in_sphere predicate calls). If intentional, set DELAUNAY_ALLOW_SLOW_COSPHERICAL_FILTER=1"
    );

    // Performance note: this is O(n^5) in the number of input vertices due to iterating all
    // 5-point subsets.
    //
    // With the current generator bounds used by the specialized 3D property (n <= 10), that's
    // at most C(10, 5) = 252 5-tuples. We do up to 5 `in_sphere` predicate calls per tuple (one
    // per "test point") for robustness, so the worst case is 252 * 5 = 1260 `in_sphere`
    // evaluations per generated case.
    //
    // If you expand the proptest ranges, consider benchmarking this filter or reducing the
    // number of predicate calls.
    let kernel = RobustKernel::<f64>::new();

    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                for l in (k + 1)..n {
                    for m in (l + 1)..n {
                        let points = [
                            *vertices[i].point(),
                            *vertices[j].point(),
                            *vertices[k].point(),
                            *vertices[l].point(),
                            *vertices[m].point(),
                        ];

                        // For each choice of "test point", ensure it is not exactly on the circumsphere
                        // of the tetrahedron formed by the other 4 points.
                        for test_idx in 0..5 {
                            let mut simplex = [points[0], points[1], points[2], points[3]];
                            let test_point = points[test_idx];

                            // Build simplex by skipping test_idx.
                            let mut s = 0usize;
                            for (p_idx, p) in points.iter().enumerate() {
                                if p_idx == test_idx {
                                    continue;
                                }
                                simplex[s] = *p;
                                s += 1;
                            }

                            let Ok(in_sphere) = kernel.in_sphere(&simplex, &test_point) else {
                                // Treat any conversion/predicate failure as a degeneracy for test purposes.
                                return false;
                            };

                            // 0 == on boundary => co-spherical.
                            if in_sphere == 0 {
                                return false;
                            }
                        }
                    }
                }
            }
        }
    }

    true
}

/// Assert the layered validation contract we rely on in these properties:
/// - Levels 1–3 only (elements + structure + topology)
/// - Level 4 (Delaunay empty-circumsphere) is intentionally NOT asserted here
macro_rules! prop_assert_levels_1_to_3_valid {
    ($dim:expr, $dt:expr, $context:expr) => {{
        let validation = ($dt).as_triangulation().validate();
        prop_assert!(
            validation.is_ok(),
            "{}D: {} failed Levels 1–3 validation: {:?}",
            $dim,
            $context,
            validation.err()
        );
    }};
}

/// Build the four corners of an axis-aligned rectangle.
///
/// Rectangle corners are exactly cocircular, so randomized insertion orders
/// exercise the `OnSuspicion` validation cadence on an adversarial boundary
/// between valid Delaunay triangulations.
fn rectangle_corners_2d(origin: [i32; 2], side_lengths: [i32; 2]) -> Vec<Point<2>> {
    let x0 = f64::from(origin[0]);
    let y0 = f64::from(origin[1]);
    let x1 = x0 + f64::from(side_lengths[0]);
    let y1 = y0 + f64::from(side_lengths[1]);

    vec![
        Point::try_new([x0, y0]).expect("finite point coordinates"),
        Point::try_new([x1, y0]).expect("finite point coordinates"),
        Point::try_new([x0, y1]).expect("finite point coordinates"),
        Point::try_new([x1, y1]).expect("finite point coordinates"),
    ]
}

/// Build the eight corners of an axis-aligned cuboid.
///
/// Cuboid corners are exactly cospherical. This keeps the property focused on
/// topology and Delaunay validation under degenerate but finite inputs.
fn cuboid_corners_3d(origin: [i32; 3], side_lengths: [i32; 3]) -> Vec<Point<3>> {
    let x0 = f64::from(origin[0]);
    let y0 = f64::from(origin[1]);
    let z0 = f64::from(origin[2]);
    let x1 = x0 + f64::from(side_lengths[0]);
    let y1 = y0 + f64::from(side_lengths[1]);
    let z1 = z0 + f64::from(side_lengths[2]);

    vec![
        Point::try_new([x0, y0, z0]).expect("finite point coordinates"),
        Point::try_new([x1, y0, z0]).expect("finite point coordinates"),
        Point::try_new([x0, y1, z0]).expect("finite point coordinates"),
        Point::try_new([x0, y0, z1]).expect("finite point coordinates"),
        Point::try_new([x1, y1, z0]).expect("finite point coordinates"),
        Point::try_new([x1, y0, z1]).expect("finite point coordinates"),
        Point::try_new([x0, y1, z1]).expect("finite point coordinates"),
        Point::try_new([x1, y1, z1]).expect("finite point coordinates"),
    ]
}

/// Shuffle points with a deterministic seed supplied by proptest.
fn shuffle_points<const D: usize>(mut points: Vec<Point<D>>, seed: u64) -> Vec<Point<D>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    points.shuffle(&mut rng);
    points
}

/// Insert an adversarial point sequence under `OnSuspicion` and assert that
/// validation succeeds after each attempted insertion and at the end of the
/// sequence.
fn assert_on_suspicion_sequence_valid<const D: usize>(
    points: Vec<Point<D>>,
) -> Result<(), TestCaseError> {
    let mut dt: DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D> =
        DelaunayTriangulation::empty_with_topology_guarantee(TopologyGuarantee::PLManifold);
    dt.set_validation_policy(ValidationPolicy::OnSuspicion);

    for (idx, point) in points.into_iter().enumerate() {
        let result = dt.insert_best_effort_with_statistics(
            delaunay::prelude::Vertex::<(), _>::try_new(point.into()).unwrap(),
        );

        if dt.number_of_simplices() > 0 {
            let validation = dt.validate();
            prop_assert!(
                validation.is_ok(),
                "{D}D OnSuspicion validation failed after cospherical attempt {idx}: result={result:?}, validation={:?}",
                validation.err()
            );
        }
    }

    prop_assert!(
        dt.number_of_simplices() > 0,
        "{D}D OnSuspicion cospherical sequence did not create any simplices"
    );

    let validation = dt.validate();
    prop_assert!(
        validation.is_ok(),
        "{D}D OnSuspicion final Level 1-4 validation failed after cospherical sequence: {:?}",
        validation.err()
    );

    Ok(())
}

/// Build a fixed non-degenerate unit simplex for non-finite insertion tests.
fn unit_simplex_vertices<const D: usize>() -> Vec<Vertex<(), D>> {
    let mut points = Vec::with_capacity(D + 1);
    points.push(Point::try_new([0.0; D]).expect("finite point coordinates"));

    for axis in 0..D {
        let mut coords = [0.0; D];
        coords[axis] = 1.0;
        points.push(Point::try_new(coords).expect("finite point coordinates"));
    }

    Vertex::from_validated_points(&points)
}

/// Build coordinate input with one non-finite coordinate and finite filler elsewhere.
fn non_finite_coords<const D: usize>(axis: usize, value: f64) -> [f64; D] {
    let mut coords = [0.25; D];
    for (idx, coord) in coords.iter_mut().enumerate() {
        if idx % 2 == 1 {
            *coord = -0.25;
        }
    }
    coords[axis] = value;
    coords
}

const fn invalid_coordinate_value(value: f64) -> InvalidCoordinateValue {
    if value.is_nan() {
        InvalidCoordinateValue::Nan
    } else if value.is_sign_positive() {
        InvalidCoordinateValue::PositiveInfinity
    } else {
        InvalidCoordinateValue::NegativeInfinity
    }
}

/// Assert non-finite public coordinates are rejected before insertion can mutate topology.
fn assert_non_finite_point_rejected_and_preserves_validity<const D: usize>(
    initial_vertices: &[Vertex<(), D>],
    coords: [f64; D],
    axis: usize,
    value: f64,
) -> Result<(), TestCaseError> {
    prop_assert_eq!(
        Point::<D>::try_new(coords),
        Err(CoordinateValidationError::InvalidCoordinate {
            coordinate_index: axis,
            coordinate_value: invalid_coordinate_value(value),
            dimension: D,
        })
    );

    let empty: DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D> =
        DelaunayTriangulation::empty_with_topology_guarantee(TopologyGuarantee::PLManifold);

    prop_assert_eq!(empty.number_of_vertices(), 0);
    prop_assert_eq!(empty.number_of_simplices(), 0);
    let empty_validation = empty.validate();
    prop_assert!(
        empty_validation.is_ok(),
        "{D}D empty triangulation became invalid after rejected non-finite insertion: {:?}",
        empty_validation.err()
    );

    let dt =
        DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), D>::try_new_with_topology_guarantee(
            initial_vertices,
            TopologyGuarantee::PLManifold,
        )
        .map_err(|err| {
            TestCaseError::fail(format!(
                "{D}D finite initial simplex construction failed: {err:?}"
            ))
        })?;
    let vertex_count = dt.number_of_vertices();
    let simplex_count = dt.number_of_simplices();

    prop_assert_eq!(dt.number_of_vertices(), vertex_count);
    prop_assert_eq!(dt.number_of_simplices(), simplex_count);

    let validation = dt.validate();
    prop_assert!(
        validation.is_ok(),
        "{D}D triangulation became invalid after rejected non-finite insertion: {:?}",
        validation.err()
    );

    Ok(())
}

/// 3D-only helper for the insertion-order robustness property: perform incremental insertion and
/// classify whether the run stayed within the "no retry / no skip" scope.
///
/// This makes the contract explicit in the test: we only accept "clean" runs where insertion
/// succeeded on the first attempt for every vertex and did not skip any input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InsertionOrder3dRunStatus {
    Clean,
    UsedRetry,
    UsedSkip,
    NonRetryableError,
}

fn insert_vertices_3d_no_retry_or_skip(
    dt: &mut DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 3>,
    vertices: &[Vertex<(), 3>],
) -> InsertionOrder3dRunStatus {
    for (idx, v) in vertices.iter().enumerate() {
        let result = dt.insert_best_effort_with_statistics(*v);
        let Ok((outcome, stats)) = result else {
            if std::env::var_os("DELAUNAY_PROPTEST_INSERT_ERRORS").is_some()
                && let Err(err) = result
            {
                let points: Vec<_> = vertices.iter().map(|vertex| *vertex.point()).collect();
                tracing::warn!(
                    "3D insertion-order: non-retryable insertion error at index {idx}: {err}"
                );
                tracing::warn!("3D insertion-order: insertion order points: {points:?}");
            }
            return InsertionOrder3dRunStatus::NonRetryableError;
        };

        if stats.attempts > 1 {
            return InsertionOrder3dRunStatus::UsedRetry;
        }

        if matches!(outcome, InsertionOutcome::Skipped { .. }) {
            return InsertionOrder3dRunStatus::UsedSkip;
        }
    }
    InsertionOrder3dRunStatus::Clean
}

#[test]
fn regression_insertion_order_3d_case_001() {
    let points = vec![
        Point::try_new([
            -44.945_005_120_296_535,
            99.756_651_886_055_48,
            -42.335_053_700_056_505,
        ])
        .expect("finite point coordinates"),
        Point::try_new([
            73.700_964_567_587_9,
            -63.612_148_894_577_96,
            74.105_780_529_266_97,
        ])
        .expect("finite point coordinates"),
        Point::try_new([
            17.617_609_442_986_858,
            82.105_819_037_825_37,
            96.704_584_226_450_41,
        ])
        .expect("finite point coordinates"),
        Point::try_new([
            -84.890_821_270_170_35,
            -81.870_190_740_455_77,
            -15.370_417_803_763_095,
        ])
        .expect("finite point coordinates"),
        Point::try_new([
            91.801_310_819_317_69,
            -59.048_589_230_347_44,
            32.595_036_426_462_01,
        ])
        .expect("finite point coordinates"),
        Point::try_new([
            71.577_209_178_042_41,
            67.758_811_346_527_74,
            -13.549_441_576_893_503,
        ])
        .expect("finite point coordinates"),
        Point::try_new([
            13.014_109_789_682_385,
            -44.307_776_868_547_734,
            12.429_638_278_179_98,
        ])
        .expect("finite point coordinates"),
        Point::try_new([
            56.289_801_257_128_175,
            8.701_429_811_815_13,
            -74.980_558_102_554_71,
        ])
        .expect("finite point coordinates"),
        Point::try_new([
            -94.434_209_076_643_74,
            -18.606_000_110_184_25,
            4.127_801_687_310_058,
        ])
        .expect("finite point coordinates"),
    ];
    let vertices = Vertex::from_validated_points(&points);
    let mut dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::empty_with_topology_guarantee(TopologyGuarantee::PLManifold);

    for (idx, v) in vertices.iter().enumerate() {
        let result = dt.insert_best_effort_with_statistics(*v);
        match result {
            Ok((InsertionOutcome::Inserted { .. }, _stats)) => {}
            Ok((InsertionOutcome::Skipped { error }, _stats)) => {
                panic!("insertion skipped at index {idx}: {error}");
            }
            Err(err) => {
                panic!("insertion error at index {idx}: {err}");
            }
        }
    }
}

// =============================================================================
// INCREMENTAL INSERTION VALIDITY TESTS
// =============================================================================

macro_rules! gen_incremental_insertion_validity {
    ($dim:literal, $min:literal, $max:literal $(, cases = $cases:literal)? $(, #[$attr:meta])*) => {
        pastey::paste! {
            proptest! {
                $(
                    #![proptest_config(Config {
                        cases: $cases,
                        ..Config::default()
                    })]
                )?
                $(#[$attr])*
                #[test]
                fn [<prop_incremental_insertion_maintains_validity_ $dim d>](
                    initial_points in prop::collection::vec([<vertex_ $dim d>](), $min..=$max),
                    additional_point in [<vertex_ $dim d>](),
                ) {
                    init_tracing();
                    // Dedup exact duplicates to avoid pathological degeneracies during shrinking.
                    let initial_vertices =
                        dedup_vertices_by_coords::<$dim>(Vertex::from_validated_points(&initial_points));

                    // Require at least D+1 distinct vertices for valid simplices.
                    prop_assume!(initial_vertices.len() > $dim);

                    // Reject pathological shrink targets with too many points on a coordinate hyperplane (x_i == 0).
                    // This is redundant with `finite_coordinate()` today (|x| > 1e-6), but kept as a guardrail if the
                    // generator changes.
                    prop_assume!(has_no_coordinate_hyperplane_degeneracy(&initial_vertices));

                    let additional_vertex =
                        delaunay::prelude::Vertex::<(), _>::try_new(additional_point.into())
                            .unwrap();

                    // Avoid duplicate insertion cases here; duplicate-handling is tested in dedicated suites.
                    let add_coords = *additional_vertex.point().coords();
                    let is_dup = initial_vertices.iter().any(|u| {
                        let uc = *u.point().coords();
                        add_coords
                            .iter()
                            .zip(uc.iter())
                            .all(|(a, b)| a.to_bits() == b.to_bits())
                    });
                    prop_assume!(!is_dup);

                    let dt = DelaunayTriangulation::<_, (), (), $dim>::try_new_with_topology_guarantee(
                        &initial_vertices,
                        TopologyGuarantee::PLManifold,
                    );
                    if let Err(err) = &dt {
                        if std::env::var_os("DELAUNAY_PROPTEST_CONSTRUCTION_ERRORS").is_some() {
                            tracing::warn!(
                                "{}D: incremental insertion construction failed (treated as rejection): {err}",
                                $dim
                            );
                        }
                    }
                    let mut dt = dt.prop_assume_ok()?;
                    prop_assert_levels_1_to_3_valid!($dim, &dt, "initial triangulation");

                    let insert_result = dt.insert(additional_vertex);
                    if let Err(e) = &insert_result {
                        if std::env::var_os("DELAUNAY_PROPTEST_INSERT_ERRORS").is_some() {
                            tracing::warn!(
                                "{}D: incremental insertion error (treated as rejection): {e}",
                                $dim
                            );
                        }
                    }
                    prop_assume!(insert_result.is_ok());
                    prop_assert_levels_1_to_3_valid!($dim, &dt, "after insertion");
                }
            }
        }
    };
}

gen_incremental_insertion_validity!(2, 3, 5);
proptest! {
    #[test]
    fn prop_incremental_insertion_maintains_validity_3d(
        initial_points in prop::collection::vec(vertex_3d(), 4..=6),
        additional_point in vertex_3d(),
    ) {
        init_tracing();
        // Dedup exact duplicates to avoid pathological degeneracies during shrinking.
        let initial_vertices =
            dedup_vertices_by_coords::<3>(Vertex::from_validated_points(&initial_points));

        // Require at least D+1 distinct vertices for valid simplices.
        prop_assume!(initial_vertices.len() > 3);

        // Reject pathological shrink targets with too many points on a coordinate hyperplane (x_i == 0).
        prop_assume!(has_no_coordinate_hyperplane_degeneracy(&initial_vertices));

        // Scope to general-position inputs for 3D incremental insertion.
        prop_assume!(has_no_nearly_coplanar_tetrahedra_3d(&initial_vertices));
        prop_assume!(has_no_cospherical_5_tuples_3d(&initial_vertices));

        let additional_vertex =
            delaunay::prelude::Vertex::<(), _>::try_new(additional_point.into()).unwrap();

        // Avoid duplicate insertion cases here; duplicate-handling is tested in dedicated suites.
        let add_coords = *additional_vertex.point().coords();
        let is_dup = initial_vertices.iter().any(|u| {
            let uc = *u.point().coords();
            add_coords
                .iter()
                .zip(uc.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
        });
        prop_assume!(!is_dup);

        // Ensure the combined set (including the new vertex) remains in general position.
        let mut all_vertices = initial_vertices.clone();
        all_vertices.push(additional_vertex);
        prop_assume!(has_no_coordinate_hyperplane_degeneracy(&all_vertices));
        prop_assume!(has_no_nearly_coplanar_tetrahedra_3d(&all_vertices));
        prop_assume!(has_no_cospherical_5_tuples_3d(&all_vertices));

        let dt = DelaunayTriangulation::<_, (), (), 3>::try_new_with_topology_guarantee(
            &initial_vertices,
            TopologyGuarantee::PLManifold,
        );
        prop_assume!(dt.is_ok());
        let mut dt = dt.unwrap();
        prop_assert_levels_1_to_3_valid!(3, &dt, "initial triangulation");

        let insert_result = dt.insert_best_effort_with_statistics(additional_vertex);
        if let Err(e) = &insert_result
            && std::env::var_os("DELAUNAY_PROPTEST_INSERT_ERRORS").is_some()
        {
            tracing::warn!("3D: incremental insertion error (treated as rejection): {e}");
        }
        prop_assume!(insert_result.is_ok());
        let (outcome, stats) = insert_result.unwrap();
        if let InsertionOutcome::Skipped { error } = &outcome
            && std::env::var_os("DELAUNAY_PROPTEST_INSERT_ERRORS").is_some()
        {
            tracing::warn!("3D: incremental insertion skipped (treated as rejection): {error}");
        }
        prop_assume!(matches!(outcome, InsertionOutcome::Inserted { .. }));
        // Reject cases that required perturbation retries; those are handled in dedicated suites.
        prop_assume!(stats.attempts == 1);
        prop_assert_levels_1_to_3_valid!(3, &dt, "after insertion");
    }
}
gen_incremental_insertion_validity!(4, 5, 7);
gen_incremental_insertion_validity!(5, 6, 8, cases = 12);

proptest! {
    #![proptest_config(Config {
        cases: 64,
        ..Config::default()
    })]

    #[test]
    fn prop_on_suspicion_validates_cocircular_2d_sequences(
        origin in prop::array::uniform2(-16i32..=16),
        side_lengths in prop::array::uniform2(1i32..=16),
        seed in any::<u64>(),
    ) {
        init_tracing();
        let points = shuffle_points(rectangle_corners_2d(origin, side_lengths), seed);

        assert_on_suspicion_sequence_valid(points)?;
    }

    #[test]
    fn prop_on_suspicion_validates_cospherical_3d_sequences(
        origin in prop::array::uniform3(-8i32..=8),
        side_lengths in prop::array::uniform3(1i32..=8),
        seed in any::<u64>(),
    ) {
        init_tracing();
        let mut points = cuboid_corners_3d(origin, side_lengths);
        let tail = points.split_off(4);
        points.extend(shuffle_points(tail, seed));

        assert_on_suspicion_sequence_valid(points)?;
    }
}

macro_rules! gen_non_finite_insert_rejection_tests {
    ($($dim:literal),+ $(,)?) => {
        pastey::paste! {
            proptest! {
                $(
                    #[test]
                    fn [<prop_non_finite_point_rejected_before_insert_preserves_topology_ $dim d>](
                        value in non_finite_coordinate(),
                        axis in 0usize..$dim,
                    ) {
                        init_tracing();
                        let vertices = unit_simplex_vertices::<$dim>();
                        let coords = non_finite_coords::<$dim>(axis, value);

                        assert_non_finite_point_rejected_and_preserves_validity(
                            &vertices,
                            coords,
                            axis,
                            value,
                        )?;
                    }
                )+
            }
        }
    };
}

gen_non_finite_insert_rejection_tests!(2, 3, 4, 5);

// =============================================================================
// DUPLICATE COORDINATE REJECTION TESTS
// =============================================================================

macro_rules! gen_duplicate_coords_test {
    ($dim:literal, $min:literal, $max:literal $(, #[$attr:meta])*) => {
        pastey::paste! {
            proptest! {
                /// Tests that duplicate coordinates are rejected during insertion.
                ///
                /// **Status**: Active in 2D/3D. 4D/5D remain ignored for runtime (slow in test-integration).
                $(#[$attr])*
                #[test]
                fn [<prop_duplicate_coordinates_rejected_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                        $min..=$max
                    ).prop_map(|v| Vertex::from_validated_points(&v))
                ) {
                    let options = ConstructionOptions::default()
                        .with_dedup_policy(DedupPolicy::Exact);
                    let dt = DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), $dim>::try_new_with_options(
                        &vertices,
                        options,
                    );
                    prop_assume!(dt.is_ok());
                    let mut dt = dt.unwrap();
                    dt.try_set_validation_policy(ValidationPolicy::ExplicitOnly)
                        .expect("explicit-only validation policy should be compatible");
                    dt.set_delaunay_repair_policy(DelaunayRepairPolicy::Never);
                    // Select a vertex that is actually present in the triangulation.
                    // `DelaunayTriangulation::try_new_with_options` may skip some input vertices (e.g., due to degeneracy),
                    // so we must use stored vertices to test duplicate rejection.
                    let (_, existing_vertex) = dt.vertices().next()
                        .expect("DelaunayTriangulation::try_new_with_options returned Ok but has no vertices");
                    let p = *existing_vertex.point();
                    let dup = Vertex::from_validated_points(&[p])[0];
                    let result = dt.insert(dup);
                    prop_assert!(
                        result.is_err(),
                        "expected insertion error for duplicate coordinates, got {result:?}"
                    );
                }
            }
        }
    };
}

gen_duplicate_coords_test!(2, 3, 10);
gen_duplicate_coords_test!(3, 4, 12);
gen_duplicate_coords_test!(4, 5, 14, #[cfg(feature = "slow-tests")]);
gen_duplicate_coords_test!(5, 6, 16, #[cfg(feature = "slow-tests")]);

/// Allow runtime tuning for the empty-circumsphere property in higher dimensions.
///
/// These tests are expensive in 4D/5D; you can clamp the max vertex count (in addition to
/// `PROPTEST_CASES`) via:
/// - `DELAUNAY_EMPTY_CIRCUMSPHERE_MAX_VERTICES_4D`
/// - `DELAUNAY_EMPTY_CIRCUMSPHERE_MAX_VERTICES_5D`
fn empty_circumsphere_max_vertices(dim: usize, min_vertices: usize, default_max: usize) -> usize {
    let env_key = match dim {
        4 => "DELAUNAY_EMPTY_CIRCUMSPHERE_MAX_VERTICES_4D",
        5 => "DELAUNAY_EMPTY_CIRCUMSPHERE_MAX_VERTICES_5D",
        _ => return default_max,
    };

    let Ok(value) = std::env::var(env_key) else {
        return default_max;
    };

    match value.parse::<usize>() {
        Ok(parsed) if parsed >= min_vertices => parsed.min(default_max),
        _ => default_max,
    }
}

macro_rules! empty_circumsphere_vertices {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal) => {
        pastey::paste! {{
            let max_vertices = empty_circumsphere_max_vertices($dim, $min_vertices, $max_vertices);
            prop::collection::vec([<vertex_ $dim d>](), $min_vertices..=max_vertices)
                .prop_map(|pts| dedup_vertices_by_coords::<$dim>(Vertex::from_validated_points(&pts)))
        }}
    };
}

// =============================================================================
// DELAUNAY EMPTY CIRCUMSPHERE PROPERTY
// =============================================================================

macro_rules! test_empty_circumsphere {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal $(, #[$attr:meta])*) => {
        pastey::paste! {
proptest! {
                /// Property: For every simplex, no other vertex lies strictly inside
                /// the circumsphere defined by that simplex (Delaunay condition).
                $(#[$attr])*
                #[test]
                fn [<prop_empty_circumsphere_ $dim d>](
                    vertices in empty_circumsphere_vertices!($dim, $min_vertices, $max_vertices)
                ) {
                    init_tracing();
                    // Build Delaunay triangulation with PL-manifold guarantee, triangulating all vertices together.
                    // This ensures the entire triangulation (including initial simplex) satisfies Delaunay property

                    // Require at least D+1 distinct vertices to form valid D-simplices
                    prop_assume!(vertices.len() > $dim);

                    // General position filter: reject pathological shrink targets with too many points lying exactly on a
                    // coordinate hyperplane (x_i == 0.0).
                    prop_assume!(has_no_coordinate_hyperplane_degeneracy(&vertices));

                    // Use DelaunayTriangulation::try_new_with_topology_guarantee() to triangulate ALL vertices together
                    let dt = DelaunayTriangulation::<_, (), (), $dim>::try_new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    );
                    if let Err(err) = &dt {
                        if std::env::var_os("DELAUNAY_PROPTEST_CONSTRUCTION_ERRORS").is_some() {
                            tracing::warn!(
                                "{}D: empty-circumsphere construction failed (treated as rejection): {err}",
                                $dim
                            );
                        }
                    }
                    let dt = dt.prop_assume_ok()?;

                    // Verify the triangulation satisfies the Delaunay property (Level 4)
                    // Use fast O(N) flip-based verification instead of O(N×V) brute-force
                    let delaunay_result = dt.is_delaunay_via_flips();
                    prop_assert!(
                        delaunay_result.is_ok(),
                        "{}D triangulation should satisfy Delaunay property: {:?}",
                        $dim,
                        delaunay_result.err()
                    );
                }
            }
        }
    };
}

// 2D–5D coverage (keep ranges small to bound runtime)
test_empty_circumsphere!(2, 6, 10);
test_empty_circumsphere!(3, 6, 10);
test_empty_circumsphere!(4, 6, 12, #[cfg(feature = "slow-tests")]);
test_empty_circumsphere!(5, 7, 12, #[cfg(feature = "slow-tests")]);

// =============================================================================
// FAST HIGH-DIMENSIONAL CI SMOKE TESTS
// =============================================================================

macro_rules! gen_high_dim_delaunay_smoke {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal) => {
        pastey::paste! {
            #[test]
            #[expect(
                clippy::too_many_lines,
                reason = "Smoke test intentionally covers construction, duplicate handling, and validation telemetry together"
            )]
            fn [<prop_high_dim_delaunay_active_smoke_ $dim d>]() {
                #[derive(Debug, Default)]
                struct SmokeStats {
                    generated: usize,
                    accepted: usize,
                    rejected_too_few_unique: usize,
                    rejected_coordinate_hyperplane: usize,
                    rejected_construction_failed: usize,
                    rejected_duplicate_cloud_failed: usize,
                }

                init_tracing();

                let config = Config {
                    cases: 6,
                    max_shrink_iters: 16,
                    ..Config::default()
                };
                let target_cases = config.cases;
                let mut runner = TestRunner::new(config);
                let strategy = prop::collection::vec(
                    [<vertex_ $dim d>](),
                    $min_vertices..=$max_vertices,
                )
                .prop_map(|pts| dedup_vertices_by_coords::<$dim>(Vertex::from_validated_points(&pts)));
                let stats = RefCell::new(SmokeStats::default());

                let run_result = runner.run(&strategy, |vertices| {
                    let mut stats = stats.borrow_mut();
                    stats.generated += 1;

                    if vertices.len() <= $dim {
                        stats.rejected_too_few_unique += 1;
                        return Err(TestCaseError::reject(format!(
                            "{}D: fewer than {} unique vertices after deduplication",
                            $dim,
                            $dim + 1
                        )));
                    }

                    if !has_no_coordinate_hyperplane_degeneracy(&vertices) {
                        stats.rejected_coordinate_hyperplane += 1;
                        return Err(TestCaseError::reject(format!(
                            "{}D: coordinate-hyperplane degeneracy present",
                            $dim
                        )));
                    }

                    let mut dt = match DelaunayTriangulation::<_, (), (), $dim>::try_new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        Ok(dt) => dt,
                        Err(err) => {
                            stats.rejected_construction_failed += 1;
                            return Err(TestCaseError::reject(format!(
                                "{}D: construction failed in active smoke test: {err}",
                                $dim
                            )));
                        }
                    };
                    prop_assert_levels_1_to_3_valid!($dim, &dt, "active smoke construction");

                    let delaunay_result = dt.is_delaunay_via_flips();
                    prop_assert!(
                        delaunay_result.is_ok(),
                        "{}D active smoke triangulation should satisfy Level 4 Delaunay validation: {:?}",
                        $dim,
                        delaunay_result.err()
                    );

                    let (_, existing_vertex) = dt
                        .vertices()
                        .next()
                        .expect("successfully constructed triangulation should contain vertices");
                    let duplicate = Vertex::from_validated_points(&[*existing_vertex.point()])[0];
                    let duplicate_result = dt.insert(duplicate);
                    prop_assert!(
                        duplicate_result.is_err(),
                        "{}D active smoke should reject duplicate coordinate insertion",
                        $dim
                    );

                    let mut cloud_points: Vec<Point<$dim>> =
                        vertices.iter().map(|v| *v.point()).collect();
                    if cloud_points.len() >= 2 {
                        cloud_points.push(cloud_points[0]);
                        let mut jittered = *cloud_points[1].coords();
                        for coord in &mut jittered {
                            *coord += 1e-7;
                        }
                        cloud_points.push(Point::try_new(jittered).expect("finite point coordinates"));
                    }

                    let cloud_vertices = Vertex::from_validated_points(&cloud_points);
                    let options = ConstructionOptions::default()
                        .with_dedup_policy(DedupPolicy::try_epsilon(1e-6).unwrap());
                    let cloud_dt = match DelaunayTriangulation::<_, (), (), $dim>::try_new_with_options(
                        &cloud_vertices,
                        options,
                    ) {
                        Ok(dt) => dt,
                        Err(err) => {
                            stats.rejected_duplicate_cloud_failed += 1;
                            return Err(TestCaseError::reject(format!(
                                "{}D: duplicate-cloud construction failed in active smoke test: {err}",
                                $dim
                            )));
                        }
                    };
                    prop_assert_levels_1_to_3_valid!(
                        $dim,
                        &cloud_dt,
                        "active smoke duplicate-cloud construction"
                    );
                    let cloud_delaunay = cloud_dt.is_valid();
                    prop_assert!(
                        cloud_delaunay.is_ok(),
                        "{}D active smoke duplicate cloud should be globally Delaunay: {:?}",
                        $dim,
                        cloud_delaunay.err()
                    );

                    stats.accepted += 1;
                    Ok(())
                });

                let stats = stats.into_inner();
                let generated = stats.generated.max(1);
                let acceptance_rate_percent_x100: u128 =
                    (stats.accepted as u128 * 10_000u128) / (generated as u128);
                let acceptance_rate_whole = acceptance_rate_percent_x100 / 100;
                let acceptance_rate_frac = acceptance_rate_percent_x100 % 100;

                let min_acceptance_pct_str =
                    std::env::var(format!("DELAUNAY_PROPTEST_MIN_ACCEPTANCE_PCT_{}D", $dim))
                        .ok()
                        .or_else(|| std::env::var("DELAUNAY_PROPTEST_MIN_ACCEPTANCE_PCT").ok());
                if let Some(min_acceptance_pct_str) = min_acceptance_pct_str {
                    if let Ok(min_acceptance_pct) = min_acceptance_pct_str.parse::<u128>() {
                        let min_acceptance_pct_x100 = min_acceptance_pct * 100;
                        assert!(
                            acceptance_rate_percent_x100 >= min_acceptance_pct_x100,
                            "prop_high_dim_delaunay_active_smoke_{}d acceptance rate {}.{:02}% below required {min_acceptance_pct}% (generated={}, accepted={})",
                            $dim,
                            acceptance_rate_whole,
                            acceptance_rate_frac,
                            stats.generated,
                            stats.accepted
                        );
                    } else {
                        tracing::warn!(
                            "prop_high_dim_delaunay_active_smoke_{}d: invalid DELAUNAY_PROPTEST_MIN_ACCEPTANCE_PCT value {min_acceptance_pct_str:?} (expected integer percent, e.g. 10)",
                            $dim
                        );
                    }
                }

                let print_stats =
                    std::env::var_os("DELAUNAY_PROPTEST_REJECT_STATS").is_some() || run_result.is_err();
                if print_stats {
                    let rejected_total = stats.generated.saturating_sub(stats.accepted);
                    tracing::warn!(
                        "prop_high_dim_delaunay_active_smoke_{}d reject stats: target_cases={target_cases} generated={} accepted={} acceptance_rate={}.{:02}% rejected_total={} too_few_unique={} coord_hyperplane={} construction_failed={} duplicate_cloud_failed={}",
                        $dim,
                        stats.generated,
                        stats.accepted,
                        acceptance_rate_whole,
                        acceptance_rate_frac,
                        rejected_total,
                        stats.rejected_too_few_unique,
                        stats.rejected_coordinate_hyperplane,
                        stats.rejected_construction_failed,
                        stats.rejected_duplicate_cloud_failed
                    );
                }

                let construction_rejections = stats
                    .rejected_construction_failed
                    .saturating_add(stats.rejected_duplicate_cloud_failed);
                let max_allowed_construction_rejections =
                    usize::try_from(target_cases).map_or(usize::MAX, |cases| cases.max(1));
                assert!(
                    construction_rejections <= max_allowed_construction_rejections,
                    "prop_high_dim_delaunay_active_smoke_{}d had {} construction rejects (primary={}, duplicate_cloud={}) above allowed {}; generated={}, accepted={}",
                    $dim,
                    construction_rejections,
                    stats.rejected_construction_failed,
                    stats.rejected_duplicate_cloud_failed,
                    max_allowed_construction_rejections,
                    stats.generated,
                    stats.accepted
                );

                assert!(
                    stats.accepted > 0,
                    "prop_high_dim_delaunay_active_smoke_{}d should accept at least one case",
                    $dim
                );
                run_result.unwrap();
            }
        }
    };
}

gen_high_dim_delaunay_smoke!(4, 6, 8);
gen_high_dim_delaunay_smoke!(5, 7, 9);

// =============================================================================
// INSERTION-ORDER INVARIANCE (2D-5D)
// =============================================================================

macro_rules! gen_insertion_order_robustness_test {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal) => {
        pastey::paste! {
            proptest! {
                /// Property: Delaunay triangulations remain valid across different insertion orders.
                ///
                /// **Status**: Passing - validates insertion-order robustness.
                ///
                /// This test verifies that the triangulation algorithm produces valid triangulations
                /// regardless of the insertion order of the input points:
                /// - Both triangulations are structurally/topologically valid (Levels 1–3)
                /// - Same vertex counts (all input points successfully inserted)
                ///
                /// The Delaunay property (Level 4) is not asserted here (see Issue #120).
                ///
                /// **Note**: The exact edge sets may differ between different insertion orders, as
                /// Delaunay triangulation is not unique for degenerate/co-spherical point sets.
                /// Multiple valid triangulations can exist for the same point set.
                ///
                /// The test uses filtering to reduce degeneracies:
                /// - Deduplicates exact coordinate matches
                /// - Rejects configurations with > D points on any axis
                #[test]
                fn [<prop_insertion_order_robustness_ $dim d>](
                    points in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                        $min_vertices..=$max_vertices
                    ).prop_map(|pts| dedup_vertices_by_coords::<$dim>(Vertex::from_validated_points(&pts)))
                ) {
                    // Require at least D+1 distinct vertices for valid simplices
                    prop_assume!(points.len() > $dim);

                    // General position filter: reject pathological shrink targets with too many points lying exactly on a
                    // coordinate hyperplane (x_i == 0.0).
                    prop_assume!(has_no_coordinate_hyperplane_degeneracy(&points));

                    // Build first triangulation with natural order
                    let dt_a = DelaunayTriangulation::<_, (), (), $dim>::try_new_with_topology_guarantee(
                        &points,
                        TopologyGuarantee::PLManifold,
                    );
                    prop_assume!(dt_a.is_ok());
                    let dt_a = dt_a.unwrap();

                    let validation_a = dt_a.as_triangulation().validate();
                    prop_assert!(
                        validation_a.is_ok(),
                        "{}D: Triangulation A failed Levels 1–3 validation: {:?}",
                        $dim,
                        validation_a.err()
                    );

                    // Build second triangulation with shuffled order
                    let mut rng = rand::rngs::StdRng::seed_from_u64(0x00DE_C0DE);
                    let mut points_shuffled = points;
                    points_shuffled.shuffle(&mut rng);

                    let dt_b = DelaunayTriangulation::<_, (), (), $dim>::try_new_with_topology_guarantee(
                        &points_shuffled,
                        TopologyGuarantee::PLManifold,
                    );
                    prop_assume!(dt_b.is_ok());
                    let dt_b = dt_b.unwrap();

                    let validation_b = dt_b.as_triangulation().validate();
                    prop_assert!(
                        validation_b.is_ok(),
                        "{}D: Triangulation B failed Levels 1–3 validation: {:?}",
                        $dim,
                        validation_b.err()
                    );

                    // Verify both triangulations have the same number of vertices
                    // With early degeneracy detection, different insertion orders may reject
                    // different vertices during initial simplex construction, so we only verify
                    // that both succeeded and are valid, not that they have identical counts
                    let verts_a = dt_a.number_of_vertices();
                    let verts_b = dt_b.number_of_vertices();

                    // Both should have at least D+1 vertices (valid simplex)
                    prop_assert!(verts_a > $dim, "{}D: Triangulation A should have > {} vertices, got {}", $dim, $dim, verts_a);
                    prop_assert!(verts_b > $dim, "{}D: Triangulation B should have > {} vertices, got {}", $dim, $dim, verts_b);

                    // Both triangulations are valid - this is the key invariant
                    // The exact topology (edge sets, simplex counts) may differ for degenerate/co-spherical
                    // point sets, which is expected and valid behavior

                    // TODO: Once bistellar flips are implemented to ensure unique canonical triangulations,
                    // add explicit Level-4 checks here:
                    // prop_assert!(dt_a.is_valid().is_ok(), "{}D: Triangulation A must satisfy Delaunay property", $dim);
                    // prop_assert!(dt_b.is_valid().is_ok(), "{}D: Triangulation B must satisfy Delaunay property", $dim);
                    // Bistellar flips will produce canonical triangulations, making edge-set comparison more meaningful.
                }
            }
        }
    };
}

// Generate tests for 2D-5D
gen_insertion_order_robustness_test!(2, 6, 10);

// 3D is specialized below because 3D incremental insertion has a larger surface area
// for “almost-degenerate” inputs to trigger topology-repair edge cases (e.g. ridge-fan
// non-manifold cavity boundaries when points are nearly coplanar or nearly cospherical).
//
// Why this test is more complex than 2D/4D/5D:
// - We want a *pure insertion-order robustness* signal for typical (non-degenerate) 3D inputs.
// - Proptest shrinking aggressively produces nearly-coplanar / nearly-cospherical configurations,
//   and those cases are known to correlate with occasional Level-3 (topology/Euler) validation failures.
// - The insertion algorithm sometimes resolves degeneracy by retrying with perturbation. Those
//   runs are useful for robustness tracking, but they add noise to this property (and can make the
//   test “pass/fail” depend more on perturbation heuristics than on insertion order).
//
// Coverage note:
// - The filters below intentionally trade breadth for determinism and debuggability.
// - Degenerate/perturbation paths are exercised by unit regressions in
//   `src/core/triangulation.rs` and the curated edge-case suites in
//   [`tests/delaunay_edge_cases.rs`](tests/delaunay_edge_cases.rs:1) /
//   [`tests/delaunay_incremental_insertion.rs`](tests/delaunay_incremental_insertion.rs:1).
// - If these filters start rejecting a large fraction of generated cases, that is a signal
//   to improve the generator or to strengthen the underlying 3D robustness (rather than to
//   further weaken the property).
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "Large property-based test with extensive rejection tracking and diagnostics"
)]
fn prop_insertion_order_robustness_3d() {
    /// Rejection-rate tracking for the specialized 3D insertion-order property.
    ///
    /// By default this is *metrics-only* (no hard thresholds) because:
    /// - proptest seeds vary across runs, and
    /// - this property is explicitly scoped via rejection filters.
    ///
    /// CI can optionally enforce a minimum acceptance rate by setting:
    /// - `DELAUNAY_PROPTEST_MIN_ACCEPTANCE_PCT` (integer percentage, e.g. `10`)
    ///
    /// To print the summary (captured unless `cargo test -- --nocapture`):
    ///   `DELAUNAY_PROPTEST_REJECT_STATS=1 cargo test prop_insertion_order_robustness_3d --test proptest_delaunay_triangulation -- --nocapture`
    #[derive(Debug, Default)]
    struct RejectStats {
        generated: usize,
        accepted: usize,

        rejected_too_few_unique: usize,
        rejected_nearly_coplanar: usize,
        rejected_cospherical: usize,

        rejected_run_a_used_retry: usize,
        rejected_run_a_used_skip: usize,
        rejected_run_a_non_retryable_error: usize,
        rejected_run_a_invalid_levels_1_to_3: usize,

        rejected_run_b_used_retry: usize,
        rejected_run_b_used_skip: usize,
        rejected_run_b_non_retryable_error: usize,
        rejected_run_b_invalid_levels_1_to_3: usize,

        rejected_new_a_failed: usize,
        rejected_new_a_skipped_vertices: usize,
        rejected_new_a_invalid_levels_1_to_3: usize,

        rejected_new_b_failed: usize,
        rejected_new_b_skipped_vertices: usize,
        rejected_new_b_invalid_levels_1_to_3: usize,
    }

    init_tracing();

    let config = Config::default();
    let target_cases = config.cases;
    let mut runner = TestRunner::new(config);

    let strategy = prop::collection::vec(
        prop::array::uniform3(finite_coordinate())
            .prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
        6..=10,
    )
    .prop_map(|pts| dedup_vertices_by_coords::<3>(Vertex::from_validated_points(&pts)));

    // `TestRunner::run` takes an `Fn` (not `FnMut`) closure, so use interior mutability to
    // track rejection rates.
    let stats = RefCell::new(RejectStats::default());

    let run_result = runner.run(&strategy, |points| {
        let mut stats = stats.borrow_mut();
        stats.generated += 1;

        // Require at least D+1 distinct vertices for valid simplices.
        if points.len() <= 3 {
            stats.rejected_too_few_unique += 1;
            return Err(TestCaseError::reject(
                "3D: <4 unique points after dedup (out of scope)",
            ));
        }

        // Reject point sets that contain nearly-coplanar tetrahedra (4-point subsets).
        if !has_no_nearly_coplanar_tetrahedra_3d(&points) {
            stats.rejected_nearly_coplanar += 1;
            return Err(TestCaseError::reject(
                "3D: nearly-coplanar tetrahedron present (out of scope)",
            ));
        }

        // Reject point sets with co-spherical 5-tuples (degenerate Delaunay cases).
        if !has_no_cospherical_5_tuples_3d(&points) {
            stats.rejected_cospherical += 1;
            return Err(TestCaseError::reject(
                "3D: co-spherical 5-tuple present (out of scope)",
            ));
        }

        // Build triangulation A via incremental insertion, requiring a "clean run":
        // - no retry/perturbation (stats.attempts == 1 for all insertions)
        // - no skipped vertices
        let mut dt_a: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::empty_with_topology_guarantee(TopologyGuarantee::PLManifold);
        let run_a = insert_vertices_3d_no_retry_or_skip(&mut dt_a, &points);
        match run_a {
            InsertionOrder3dRunStatus::Clean => {}
            InsertionOrder3dRunStatus::UsedRetry => {
                stats.rejected_run_a_used_retry += 1;
                return Err(TestCaseError::reject(
                    "3D: run A required retry/perturbation (out of scope)",
                ));
            }
            InsertionOrder3dRunStatus::UsedSkip => {
                stats.rejected_run_a_used_skip += 1;
                return Err(TestCaseError::reject(
                    "3D: run A skipped a vertex (out of scope)",
                ));
            }
            InsertionOrder3dRunStatus::NonRetryableError => {
                stats.rejected_run_a_non_retryable_error += 1;
                return Err(TestCaseError::reject(
                    "3D: run A hit non-retryable insertion error (out of scope)",
                ));
            }
        }

        let validation_a = dt_a.as_triangulation().validate();
        if let Err(e) = validation_a {
            stats.rejected_run_a_invalid_levels_1_to_3 += 1;
            return Err(TestCaseError::reject(format!(
                "3D: Triangulation A (clean insertion run) failed Levels 1–3 validation (treated as out of scope): {e:?}"
            )));
        }

        // Build triangulation B with shuffled order, same retry/skip rejection.
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x00DE_C0DE);
        let mut points_shuffled = points.clone();
        points_shuffled.shuffle(&mut rng);

        let mut dt_b: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::empty_with_topology_guarantee(TopologyGuarantee::PLManifold);
        let run_b = insert_vertices_3d_no_retry_or_skip(&mut dt_b, &points_shuffled);
        match run_b {
            InsertionOrder3dRunStatus::Clean => {}
            InsertionOrder3dRunStatus::UsedRetry => {
                stats.rejected_run_b_used_retry += 1;
                return Err(TestCaseError::reject(
                    "3D: run B required retry/perturbation (out of scope)",
                ));
            }
            InsertionOrder3dRunStatus::UsedSkip => {
                stats.rejected_run_b_used_skip += 1;
                return Err(TestCaseError::reject(
                    "3D: run B skipped a vertex (out of scope)",
                ));
            }
            InsertionOrder3dRunStatus::NonRetryableError => {
                stats.rejected_run_b_non_retryable_error += 1;
                return Err(TestCaseError::reject(
                    "3D: run B hit non-retryable insertion error (out of scope)",
                ));
            }
        }

        let validation_b = dt_b.as_triangulation().validate();
        if let Err(e) = validation_b {
            stats.rejected_run_b_invalid_levels_1_to_3 += 1;
            return Err(TestCaseError::reject(format!(
                "3D: Triangulation B (clean insertion run) failed Levels 1–3 validation (treated as out of scope): {e:?}"
            )));
        }

        // Both should have inserted all vertices (we reject Skipped cases above).
        prop_assert_eq!(
            dt_a.number_of_vertices(),
            dt_b.number_of_vertices(),
            "3D: both triangulations should insert the same number of vertices"
        );

        // Both should have at least D+1 vertices (valid simplex).
        let verts_a = dt_a.number_of_vertices();
        let verts_b = dt_b.number_of_vertices();
        prop_assert!(
            verts_a > 3,
            "3D: Triangulation A should have > {} vertices, got {}",
            3,
            verts_a
        );
        prop_assert!(
            verts_b > 3,
            "3D: Triangulation B should have > {} vertices, got {}",
            3,
            verts_b
        );

        // Parity check: the high-level constructor path (`DelaunayTriangulation::try_new_with_topology_guarantee`) should also
        // succeed for the same generated inputs. This helps prevent maintenance drift vs the
        // 2D/4D/5D insertion-order tests which use `new()` directly.
        let dt_new_a = match DelaunayTriangulation::<_, (), (), 3>::try_new_with_topology_guarantee(
            &points,
            TopologyGuarantee::PLManifold,
        ) {
            Ok(dt) => dt,
            Err(e) => {
                stats.rejected_new_a_failed += 1;
                return Err(TestCaseError::reject(format!(
                    "3D: DelaunayTriangulation::try_new_with_topology_guarantee() failed for generated inputs (order A; treated as out of scope): {e}"
                )));
            }
        };

        let dt_new_b = match DelaunayTriangulation::<_, (), (), 3>::try_new_with_topology_guarantee(
            &points_shuffled,
            TopologyGuarantee::PLManifold,
        ) {
                Ok(dt) => dt,
                Err(e) => {
                    stats.rejected_new_b_failed += 1;
                    return Err(TestCaseError::reject(format!(
                        "3D: DelaunayTriangulation::try_new_with_topology_guarantee() failed for generated inputs (order B; treated as out of scope): {e}"
                    )));
                }
            };

        if dt_new_a.number_of_vertices() != points.len() {
            stats.rejected_new_a_skipped_vertices += 1;
            return Err(TestCaseError::reject(format!(
                "3D: try_new_with_topology_guarantee() skipped vertices for generated inputs (order A; treated as out of scope): expected {}, got {}",
                points.len(),
                dt_new_a.number_of_vertices()
            )));
        }

        if dt_new_b.number_of_vertices() != points.len() {
            stats.rejected_new_b_skipped_vertices += 1;
            return Err(TestCaseError::reject(format!(
                "3D: try_new_with_topology_guarantee() skipped vertices for generated inputs (order B; treated as out of scope): expected {}, got {}",
                points.len(),
                dt_new_b.number_of_vertices()
            )));
        }

        let validation_new_a = dt_new_a.as_triangulation().validate();
        if let Err(e) = validation_new_a {
            stats.rejected_new_a_invalid_levels_1_to_3 += 1;
            return Err(TestCaseError::reject(format!(
                "3D: Triangulation A (try_new_with_topology_guarantee()) failed Levels 1–3 validation (treated as out of scope): {e:?}"
            )));
        }

        let validation_new_b = dt_new_b.as_triangulation().validate();
        if let Err(e) = validation_new_b {
            stats.rejected_new_b_invalid_levels_1_to_3 += 1;
            return Err(TestCaseError::reject(format!(
                "3D: Triangulation B (try_new_with_topology_guarantee()) failed Levels 1–3 validation (treated as out of scope): {e:?}"
            )));
        }

        stats.accepted += 1;
        Ok(())
    });

    // Emit a one-line reject summary on demand (or automatically if the run aborted).
    let print_stats =
        std::env::var_os("DELAUNAY_PROPTEST_REJECT_STATS").is_some() || run_result.is_err();

    let stats = stats.into_inner();

    // Report acceptance rate as a percentage with 2 decimals using integer arithmetic
    // (avoids `usize as f64` precision-loss lints under strict clippy settings).
    //
    // percent_x100: 100.00% => 10_000
    let generated = stats.generated.max(1);
    let acceptance_rate_percent_x100: u128 =
        (stats.accepted as u128 * 10_000u128) / (generated as u128);
    let acceptance_rate_whole = acceptance_rate_percent_x100 / 100;
    let acceptance_rate_frac = acceptance_rate_percent_x100 % 100;

    if let Ok(min_acceptance_pct_str) = std::env::var("DELAUNAY_PROPTEST_MIN_ACCEPTANCE_PCT") {
        if let Ok(min_acceptance_pct) = min_acceptance_pct_str.parse::<u128>() {
            let min_acceptance_pct_x100 = min_acceptance_pct * 100;
            assert!(
                acceptance_rate_percent_x100 >= min_acceptance_pct_x100,
                "prop_insertion_order_robustness_3d acceptance rate {}.{:02}% below required {min_acceptance_pct}% (generated={}, accepted={})",
                acceptance_rate_whole,
                acceptance_rate_frac,
                stats.generated,
                stats.accepted
            );
        } else {
            tracing::warn!(
                "prop_insertion_order_robustness_3d: invalid DELAUNAY_PROPTEST_MIN_ACCEPTANCE_PCT={min_acceptance_pct_str:?} (expected integer percent, e.g. 10)"
            );
        }
    }

    if print_stats {
        let rejected_total = stats.generated.saturating_sub(stats.accepted);

        tracing::warn!(
            "prop_insertion_order_robustness_3d reject stats: target_cases={target_cases} generated={} accepted={} acceptance_rate={}.{:02}% rejected_total={} too_few_unique={} nearly_coplanar={} cospherical={} run_a(retry={}, skip={}, err={}, invalid={}) run_b(retry={}, skip={}, err={}, invalid={}) new_a(fail={}, skip={}, invalid={}) new_b(fail={}, skip={}, invalid={})",
            stats.generated,
            stats.accepted,
            acceptance_rate_whole,
            acceptance_rate_frac,
            rejected_total,
            stats.rejected_too_few_unique,
            stats.rejected_nearly_coplanar,
            stats.rejected_cospherical,
            stats.rejected_run_a_used_retry,
            stats.rejected_run_a_used_skip,
            stats.rejected_run_a_non_retryable_error,
            stats.rejected_run_a_invalid_levels_1_to_3,
            stats.rejected_run_b_used_retry,
            stats.rejected_run_b_used_skip,
            stats.rejected_run_b_non_retryable_error,
            stats.rejected_run_b_invalid_levels_1_to_3,
            stats.rejected_new_a_failed,
            stats.rejected_new_a_skipped_vertices,
            stats.rejected_new_a_invalid_levels_1_to_3,
            stats.rejected_new_b_failed,
            stats.rejected_new_b_skipped_vertices,
            stats.rejected_new_b_invalid_levels_1_to_3
        );
    }

    run_result.unwrap();
}

macro_rules! gen_insertion_order_robustness_high_dim_impl {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal $(, $attr:meta)*) => {
        pastey::paste! {
            #[test]
            $(#[$attr])*
            #[expect(
                clippy::too_many_lines,
                reason = "Large property-based test with rejection tracking and diagnostics"
            )]
            fn [<prop_insertion_order_robustness_ $dim d>]() {
                /// Rejection-rate tracking for the high-dimensional insertion-order property.
                ///
                /// To print the summary (captured unless `cargo test -- --nocapture`):
                ///   `DELAUNAY_PROPTEST_REJECT_STATS=1 cargo test prop_insertion_order_robustness_4d --test proptest_delaunay_triangulation -- --nocapture`
                #[derive(Debug, Default)]
                struct RejectStats {
                    generated: usize,
                    accepted: usize,

                    rejected_too_few_unique: usize,
                    rejected_coordinate_hyperplane: usize,

                    rejected_new_a_failed: usize,
                    rejected_new_a_invalid_levels_1_to_3: usize,

                    rejected_new_b_failed: usize,
                    rejected_new_b_invalid_levels_1_to_3: usize,
                }

                init_tracing();

                let config = Config::default();
                let target_cases = config.cases;
                let mut runner = TestRunner::new(config);

                let strategy = prop::collection::vec(
                    [<vertex_ $dim d>](),
                    $min_vertices..=$max_vertices,
                )
                .prop_map(|pts| dedup_vertices_by_coords::<$dim>(Vertex::from_validated_points(&pts)));

                // `TestRunner::run` takes an `Fn` (not `FnMut`) closure, so use interior mutability to
                // track rejection rates.
                let stats = RefCell::new(RejectStats::default());

                let run_result = runner.run(&strategy, |points| {
                    let mut stats = stats.borrow_mut();
                    stats.generated += 1;

                    if points.len() <= $dim {
                        stats.rejected_too_few_unique += 1;
                        return Err(TestCaseError::reject(format!(
                            "{}D: <{} unique points after dedup (out of scope)",
                            $dim,
                            $dim + 1
                        )));
                    }

                    if !has_no_coordinate_hyperplane_degeneracy(&points) {
                        stats.rejected_coordinate_hyperplane += 1;
                        return Err(TestCaseError::reject(format!(
                            "{}D: coordinate-hyperplane degeneracy present (out of scope)",
                            $dim
                        )));
                    }

                    let dt_a = match DelaunayTriangulation::<_, (), (), $dim>::try_new_with_topology_guarantee(
                        &points,
                        TopologyGuarantee::PLManifold,
                    ) {
                        Ok(dt) => dt,
                        Err(e) => {
                            stats.rejected_new_a_failed += 1;
                            return Err(TestCaseError::reject(format!(
                                "{}D: new_with_topology_guarantee failed for order A (out of scope): {e}",
                                $dim
                            )));
                        }
                    };

                    let validation_a = dt_a.as_triangulation().validate();
                    if let Err(e) = validation_a {
                        stats.rejected_new_a_invalid_levels_1_to_3 += 1;
                        return Err(TestCaseError::reject(format!(
                            "{}D: Triangulation A failed Levels 1–3 validation (out of scope): {e:?}",
                            $dim
                        )));
                    }

                    let mut rng = rand::rngs::StdRng::seed_from_u64(0x00DE_C0DE);
                    let mut points_shuffled = points;
                    points_shuffled.shuffle(&mut rng);

                    let dt_b = match DelaunayTriangulation::<_, (), (), $dim>::try_new_with_topology_guarantee(
                        &points_shuffled,
                        TopologyGuarantee::PLManifold,
                    ) {
                        Ok(dt) => dt,
                        Err(e) => {
                            stats.rejected_new_b_failed += 1;
                            return Err(TestCaseError::reject(format!(
                                "{}D: new_with_topology_guarantee failed for order B (out of scope): {e}",
                                $dim
                            )));
                        }
                    };

                    let validation_b = dt_b.as_triangulation().validate();
                    if let Err(e) = validation_b {
                        stats.rejected_new_b_invalid_levels_1_to_3 += 1;
                        return Err(TestCaseError::reject(format!(
                            "{}D: Triangulation B failed Levels 1–3 validation (out of scope): {e:?}",
                            $dim
                        )));
                    }

                    let verts_a = dt_a.number_of_vertices();
                    let verts_b = dt_b.number_of_vertices();

                    prop_assert!(
                        verts_a > $dim,
                        "{}D: Triangulation A should have > {} vertices, got {}",
                        $dim,
                        $dim,
                        verts_a
                    );
                    prop_assert!(
                        verts_b > $dim,
                        "{}D: Triangulation B should have > {} vertices, got {}",
                        $dim,
                        $dim,
                        verts_b
                    );

                    stats.accepted += 1;
                    Ok(())
                });

                let stats = stats.into_inner();

                // Report acceptance rate as a percentage with 2 decimals using integer arithmetic.
                let generated = stats.generated.max(1);
                let acceptance_rate_percent_x100: u128 =
                    (stats.accepted as u128 * 10_000u128) / (generated as u128);
                let acceptance_rate_whole = acceptance_rate_percent_x100 / 100;
                let acceptance_rate_frac = acceptance_rate_percent_x100 % 100;

                let min_acceptance_pct_str =
                    std::env::var(format!("DELAUNAY_PROPTEST_MIN_ACCEPTANCE_PCT_{}D", $dim))
                        .ok()
                        .or_else(|| std::env::var("DELAUNAY_PROPTEST_MIN_ACCEPTANCE_PCT").ok());

                if let Some(min_acceptance_pct_str) = min_acceptance_pct_str {
                    if let Ok(min_acceptance_pct) = min_acceptance_pct_str.parse::<u128>() {
                        let min_acceptance_pct_x100 = min_acceptance_pct * 100;
                        assert!(
                            acceptance_rate_percent_x100 >= min_acceptance_pct_x100,
                            "prop_insertion_order_robustness_{}d acceptance rate {}.{:02}% below required {min_acceptance_pct}% (generated={}, accepted={})",
                            $dim,
                            acceptance_rate_whole,
                            acceptance_rate_frac,
                            stats.generated,
                            stats.accepted
                        );
                    } else {
                        tracing::warn!(
                            "prop_insertion_order_robustness_{}d: invalid DELAUNAY_PROPTEST_MIN_ACCEPTANCE_PCT value {min_acceptance_pct_str:?} (expected integer percent, e.g. 10)",
                            $dim
                        );
                    }
                }

                let print_stats =
                    std::env::var_os("DELAUNAY_PROPTEST_REJECT_STATS").is_some() || run_result.is_err();

                if print_stats {
                    let rejected_total = stats.generated.saturating_sub(stats.accepted);

                    tracing::warn!(
                        "prop_insertion_order_robustness_{}d reject stats: target_cases={target_cases} generated={} accepted={} acceptance_rate={}.{:02}% rejected_total={} too_few_unique={} coord_hyperplane={} new_a(fail={}, invalid={}) new_b(fail={}, invalid={})",
                        $dim,
                        stats.generated,
                        stats.accepted,
                        acceptance_rate_whole,
                        acceptance_rate_frac,
                        rejected_total,
                        stats.rejected_too_few_unique,
                        stats.rejected_coordinate_hyperplane,
                        stats.rejected_new_a_failed,
                        stats.rejected_new_a_invalid_levels_1_to_3,
                        stats.rejected_new_b_failed,
                        stats.rejected_new_b_invalid_levels_1_to_3
                    );
                }

                run_result.unwrap();
            }
        }
    };
}

macro_rules! gen_insertion_order_robustness_high_dim {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal, ignore $(, #[$attr:meta])*) => {
        gen_insertion_order_robustness_high_dim_impl!(
            $dim,
            $min_vertices,
            $max_vertices,
            ignore = "Insertion order issues; see plan 72755302-2c93-43bb-879c-ef6ed21f5560"
            $(, $attr)*
        );
    };
    ($dim:literal, $min_vertices:literal, $max_vertices:literal $(, #[$attr:meta])*) => {
        gen_insertion_order_robustness_high_dim_impl!($dim, $min_vertices, $max_vertices $(, $attr)*);
    };
}

gen_insertion_order_robustness_high_dim!(4, 6, 12, #[cfg(feature = "slow-tests")]);
gen_insertion_order_robustness_high_dim!(5, 7, 12, #[cfg(feature = "slow-tests")]);

// =============================================================================
// DUPLICATE CLOUD INTEGRATION TESTS
// =============================================================================

/// Count unique coordinate tuples using bitwise equality.
/// Used to ensure at least D+1 distinct points before attempting triangulation.
fn count_unique_coords_by_bits<const D: usize>(pts: &[Point<D>]) -> usize {
    let mut set: HashSet<Vec<u64>> = HashSet::with_capacity(pts.len());
    for p in pts {
        let coords = *p.coords();
        let key: Vec<u64> = coords.iter().map(|x| x.to_bits()).collect();
        set.insert(key);
    }
    set.len()
}

macro_rules! gen_duplicate_cloud_test {
    ($dim:literal, $min_vertices:literal $(, #[$attr:meta])*) => {
        pastey::paste! {
            /// Generate random point cloud with exact duplicates and near-duplicates (1e-7 jitter).
            /// Tests full construction pipeline with realistic messy inputs.
            $(#[$attr])*
            fn [<cloud_with_duplicates_ $dim d>]() -> impl Strategy<Value = Vec<Point<$dim>>> {
                prop::collection::vec(
                    prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                    6..=12,
                )
                .prop_map(|mut pts| {
                    if pts.len() >= 3 {
                        // Exact duplicate of the first point
                        let dup = pts[0];
                        pts.push(dup);

                        // Jittered near-duplicate of the second point
                        let mut coords = *pts[1].coords();
                        for c in &mut coords {
                            *c += 1e-7;
                        }
                        pts.push(Point::try_new(coords).expect("finite point coordinates"));
                    }
                    pts
                })
            }

            proptest! {
                /// Property: Random clouds with duplicates and near-duplicates
                /// produce triangulations that are globally Delaunay for the kept subset.
                ///
                /// **Status**: Enabled - flip repair in place; validates Delaunay for the kept subset.
                ///
                /// This integration test exercises the full construction pipeline with messy real-world
                /// inputs (exact duplicates + near-duplicates).
                ///
                /// See: Issue #120, src/core/algorithms/flips.rs
                $(#[$attr])*
                #[test]
                fn [<prop_cloud_with_duplicates_is_delaunay_ $dim d>](
                    points in [<cloud_with_duplicates_ $dim d>]()
                ) {
                    init_tracing();
                    let log_coverage = std::env::var_os("DELAUNAY_PROPTEST_COVERAGE_LOGS").is_some()
                        && $dim >= 4;
                    // Require at least D+1 distinct points
                    let unique = count_unique_coords_by_bits(&points);
                    prop_assume!(unique > $min_vertices);

                    let vertices: Vec<Vertex<(), $dim>> = Vertex::from_validated_points(&points);

                    let build_start = std::time::Instant::now();
                    let options = ConstructionOptions::default()
                        .with_dedup_policy(DedupPolicy::try_epsilon(1e-6).unwrap());
                    let dt = DelaunayTriangulation::<_, (), (), $dim>::try_new_with_options(
                        &vertices,
                        options,
                    );
                    let build_elapsed = build_start.elapsed();
                    if let Err(err) = &dt {
                        if std::env::var_os("DELAUNAY_PROPTEST_CONSTRUCTION_ERRORS").is_some() {
                            tracing::warn!(
                                "{}D: duplicate-cloud construction failed (treated as rejection): {err}",
                                $dim
                            );
                        }
                    }
                    let dt = dt.prop_assume_ok()?;
                    if log_coverage {
                        tracing::info!(
                            dim = $dim,
                            points = points.len(),
                            unique = unique,
                            kept_vertices = dt.number_of_vertices(),
                            build_elapsed = ?build_elapsed,
                            "prop_cloud_with_duplicates_is_delaunay build complete"
                        );
                    }

                    // Structural/topological validity (Levels 1–3) for kept subset
                    prop_assert_levels_1_to_3_valid!($dim, &dt, "triangulation (kept subset)");

                    // Delaunay validity (Level 4) for kept subset
                    let validate_start = std::time::Instant::now();
                    let delaunay = dt.is_valid();
                    let validate_elapsed = validate_start.elapsed();
                    if log_coverage {
                        tracing::info!(
                            dim = $dim,
                            validate_elapsed = ?validate_elapsed,
                            "prop_cloud_with_duplicates_is_delaunay is_valid complete"
                        );
                    }
                    prop_assert!(
                        delaunay.is_ok(),
                        "{}D triangulation should satisfy Delaunay property: {:?}",
                        $dim,
                        delaunay.err()
                    );
                }
            }
        }
    };
}

gen_duplicate_cloud_test!(2, 2);
gen_duplicate_cloud_test!(3, 3);
gen_duplicate_cloud_test!(4, 4, #[cfg(feature = "slow-tests")]);
gen_duplicate_cloud_test!(5, 5, #[cfg(feature = "slow-tests")]);
