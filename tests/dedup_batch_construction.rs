//! Integration tests for batch construction with duplicate vertices.
//!
//! These tests verify that:
//! - Explicit batch dedup removes exact duplicates during batch construction
//! - Explicit Hilbert-sort dedup collapses quantization-resolution collisions
//! - The resulting triangulations are valid (Levels 1–5)
//! - Simplex-level coordinate uniqueness validation catches no violations post-dedup
//! - Explicit `DedupPolicy::Exact` works for non-Hilbert orderings
//!
//! Dimension coverage: 2D–5D via `gen_dedup_batch_tests!`.

use delaunay::construction::DelaunayConstructionRetryFailure;
use delaunay::prelude::construction::{
    ConstructionOptions, DedupPolicy, DelaunayConstructionFailure, DelaunayTriangulation,
    DelaunayTriangulationConstructionError, InsertionOrderStrategy, Vertex,
};
use delaunay::vertex;

// =============================================================================
// HELPERS
// =============================================================================

fn init_tracing() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_test_writer()
            .try_init();
    });
}

/// Return whether construction failed with a direct geometric degeneracy error.
const fn is_geometric_degeneracy_error(error: &DelaunayTriangulationConstructionError) -> bool {
    matches!(
        error,
        DelaunayTriangulationConstructionError::Triangulation(
            DelaunayConstructionFailure::GeometricDegeneracy { .. }
        )
    )
}

/// Accept direct degeneracy failures and retry exhaustion wrapping that same typed source.
fn is_geometric_degeneracy_or_retry_exhausted(
    error: &DelaunayTriangulationConstructionError,
) -> bool {
    match error {
        DelaunayTriangulationConstructionError::Triangulation(
            DelaunayConstructionFailure::ShuffledRetryExhausted { source, .. },
        ) => matches!(
            source.as_ref(),
            DelaunayConstructionRetryFailure::Construction { source }
                if is_geometric_degeneracy_error(source)
        ),
        other => is_geometric_degeneracy_error(other),
    }
}

/// Build D+1 standard simplex vertices: origin + D unit vectors.
fn simplex_vertices<const D: usize>() -> Vec<Vertex<(), D>> {
    let mut verts = Vec::with_capacity(D + 1);
    verts.push(vertex!([0.0; D]).unwrap());
    for i in 0..D {
        let mut coords = [0.0; D];
        coords[i] = 1.0;
        verts.push(vertex!(coords).unwrap());
    }
    verts
}

/// Build simplex vertices + an interior point + exact duplicates of the origin
/// and interior point. Returns `(vertices, distinct_count)`.
#[expect(
    clippy::cast_precision_loss,
    reason = "D ≤ 5 in practice; no precision loss"
)]
fn simplex_with_interior_and_duplicates<const D: usize>() -> (Vec<Vertex<(), D>>, usize) {
    let mut verts = simplex_vertices::<D>();
    // Interior point
    let interior = [0.25 / (D as f64); D];
    verts.push(vertex!(interior).unwrap());
    let distinct = verts.len(); // D+2

    // Duplicates: origin + interior again
    verts.push(vertex!([0.0; D]).unwrap());
    verts.push(vertex!(interior).unwrap());
    (verts, distinct)
}

/// Build `count` copies of the same all-identical vertex.
fn all_identical_vertices<const D: usize>(count: usize) -> Vec<Vertex<(), D>> {
    (0..count).map(|_| vertex!([1.0; D]).unwrap()).collect()
}

/// Select exact preprocessing dedup so tests opt into duplicate collapse explicitly.
fn exact_dedup_options() -> ConstructionOptions {
    ConstructionOptions::default().with_dedup_policy(DedupPolicy::Exact)
}

/// Build simplex vertices plus an interior point, then one duplicate of the
/// origin. Returns `(vertices, distinct_count)`.
#[expect(
    clippy::cast_precision_loss,
    reason = "D ≤ 5 in practice; no precision loss"
)]
fn simplex_with_one_duplicate<const D: usize>() -> (Vec<Vertex<(), D>>, usize) {
    let mut verts = simplex_vertices::<D>();
    // Extra non-vertex interior point to make the triangulation interesting
    let interior = [0.5 / (D as f64); D];
    verts.push(vertex!(interior).unwrap());
    let distinct = verts.len();
    // One duplicate of the origin
    verts.push(vertex!([0.0; D]).unwrap());
    (verts, distinct)
}

// =============================================================================
// MACRO — BATCH DEDUP TESTS (2D–5D)
// =============================================================================

/// Generate batch construction dedup tests for a given dimension:
///
/// - Explicit Hilbert dedup removes exact duplicates
/// - All-identical input fails gracefully, with explicit dedup reporting too few
///   unique vertices and default construction reporting geometric degeneracy
/// - `DedupPolicy::Exact` with `Input` ordering removes duplicates
/// - Many-duplicate stress test collapses correctly
macro_rules! gen_dedup_batch_tests {
    ($dim:literal) => {
        pastey::paste! {
            #[test]
            fn [<test_batch_construction_deduplicates_exact_duplicates_ $dim d>]() {
                init_tracing();
                let (vertices, distinct) = simplex_with_interior_and_duplicates::<$dim>();
                assert!(vertices.len() > distinct);

                let opts = exact_dedup_options();
                let dt: DelaunayTriangulation<_, (), (), $dim> =
                    DelaunayTriangulation::builder(&vertices)
                    .construction_options(opts)
                    .build()
                    .expect(concat!(
                        stringify!($dim), "D construction with duplicates should succeed"
                    ));

                assert_eq!(
                    dt.number_of_vertices(),
                    distinct,
                    "{}D: duplicates should be removed by explicit dedup",
                    $dim
                );
                assert!(dt.number_of_simplices() > 0);

                // Full validation (Levels 1–5) including coordinate uniqueness
                let validation = dt.validate();
                assert!(
                    validation.is_ok(),
                    "{}D: triangulation should pass validation: {validation:?}",
                    $dim
                );
            }

            #[test]
            fn [<test_batch_construction_all_duplicates_fails_without_hidden_dedup_ $dim d>]() {
                init_tracing();
                let vertices = all_identical_vertices::<$dim>($dim + 2);

                let result: Result<DelaunayTriangulation<_, (), (), $dim>, _> =
                    DelaunayTriangulation::builder(&vertices).build();

                assert!(
                    result
                        .as_ref()
                        .is_err_and(is_geometric_degeneracy_or_retry_exhausted),
                    "{}D: all-duplicate input without explicit dedup should fail as degenerate",
                    $dim
                );
            }

            #[test]
            fn [<test_batch_construction_all_duplicates_with_exact_dedup_fails_ $dim d>]() {
                init_tracing();
                // D+2 identical vertices → collapses to 1 → insufficient for simplex
                let vertices = all_identical_vertices::<$dim>($dim + 2);

                let result: Result<DelaunayTriangulation<_, (), (), $dim>, _> =
                    DelaunayTriangulation::builder(&vertices)
                        .construction_options(exact_dedup_options())
                        .build();

                assert!(
                    matches!(
                        result,
                        Err(DelaunayTriangulationConstructionError::Triangulation(
                            DelaunayConstructionFailure::InsufficientVertices { .. }
                        ))
                    ),
                    "{}D: all-duplicate input should fail with InsufficientVertices",
                    $dim
                );
            }

            #[test]
            fn [<test_explicit_dedup_exact_input_order_ $dim d>]() {
                init_tracing();
                let (vertices, distinct) = simplex_with_one_duplicate::<$dim>();

                let opts = ConstructionOptions::default()
                    .with_insertion_order(InsertionOrderStrategy::Input)
                    .with_dedup_policy(DedupPolicy::Exact);

                let dt: DelaunayTriangulation<_, (), (), $dim> =
                    DelaunayTriangulation::builder(&vertices)
                        .construction_options(opts)
                        .build()
                        .expect(concat!(
                            stringify!($dim),
                            "D: DedupPolicy::Exact + Input should succeed"
                        ));

                assert_eq!(
                    dt.number_of_vertices(),
                    distinct,
                    "{}D: DedupPolicy::Exact should remove duplicate",
                    $dim
                );
                assert!(dt.validate().is_ok());
            }

            #[test]
            fn [<test_batch_construction_many_duplicates_ $dim d>]() {
                init_tracing();
                // D+2 distinct vertices, each repeated 5× = 5(D+2) total
                let (base, distinct_count_raw) = simplex_with_interior_and_duplicates::<$dim>();
                // Take only the distinct portion using the helper's reported count.
                let distinct: Vec<Vertex<(), $dim>> =
                    base.into_iter().take(distinct_count_raw).collect();
                let distinct_count = distinct.len();

                let vertices: Vec<Vertex<(), $dim>> = distinct
                    .iter()
                    .cycle()
                    .take(distinct_count * 5)
                    .copied()
                    .collect();

                assert_eq!(vertices.len(), distinct_count * 5);

                let dt: DelaunayTriangulation<_, (), (), $dim> =
                    DelaunayTriangulation::builder(&vertices)
                        .construction_options(exact_dedup_options())
                        .build()
                        .expect(concat!(
                            stringify!($dim),
                            "D: many-duplicate construction should succeed"
                        ));

                assert_eq!(
                    dt.number_of_vertices(),
                    distinct_count,
                    "{}D: should collapse {} inputs to {} distinct vertices",
                    $dim,
                    distinct_count * 5,
                    distinct_count
                );
                assert!(dt.validate().is_ok());
            }
        }
    };
}

gen_dedup_batch_tests!(2);
gen_dedup_batch_tests!(3);
gen_dedup_batch_tests!(4);
gen_dedup_batch_tests!(5);

/// Verify that explicit Hilbert-sort dedup collapses geometrically distinct points
/// that quantize to the same Hilbert grid cell.
///
/// For D=2, bits=31, grid spacing ≈ 1/(2³¹ − 1) ≈ 4.66×10⁻¹⁰.
/// Two points differing by ~1×10⁻¹⁰ will collide at quantization resolution.
#[test]
fn test_hilbert_dedup_quantized_collision_2d() {
    init_tracing();
    let mut vertices = simplex_vertices::<2>();
    vertices.push(vertex!([0.5, 0.5]).unwrap());
    vertices.push(vertex!([0.5 + 1e-10, 0.5]).unwrap()); // quantizes to same simplex
    let total = vertices.len();

    let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::builder(&vertices)
        .construction_options(exact_dedup_options())
        .build()
        .expect("2D construction with quantized-collision should succeed");

    // Hilbert dedup should collapse the near-coincident pair.
    assert_eq!(
        dt.number_of_vertices(),
        total - 1,
        "expected Hilbert dedup to collapse exactly the quantized-collision pair"
    );
    assert!(dt.validate().is_ok());
}
