//! Integration tests for batch construction with duplicate vertices.
//!
//! These tests verify that:
//! - Hilbert-sort dedup silently removes exact duplicates during batch construction
//! - The resulting triangulations are valid (Levels 1–4)
//! - Cell-level coordinate uniqueness validation catches no violations post-dedup
//! - Explicit `DedupPolicy::Exact` works for non-Hilbert orderings
//!
//! Dimension coverage: 2D–5D via `gen_dedup_batch_tests!`.

use delaunay::core::delaunay_triangulation::{
    ConstructionOptions, DedupPolicy, InsertionOrderStrategy,
};
use delaunay::prelude::triangulation::*;

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

/// Build D+1 standard simplex vertices: origin + D unit vectors.
fn simplex_vertices<const D: usize>() -> Vec<Vertex<f64, (), D>> {
    let mut verts = Vec::with_capacity(D + 1);
    verts.push(vertex!([0.0; D]));
    for i in 0..D {
        let mut coords = [0.0; D];
        coords[i] = 1.0;
        verts.push(vertex!(coords));
    }
    verts
}

/// Build simplex vertices + an interior point + exact duplicates of the origin
/// and interior point. Returns `(vertices, distinct_count)`.
#[expect(
    clippy::cast_precision_loss,
    reason = "D ≤ 5 in practice; no precision loss"
)]
fn simplex_with_interior_and_duplicates<const D: usize>() -> (Vec<Vertex<f64, (), D>>, usize) {
    let mut verts = simplex_vertices::<D>();
    // Interior point
    let interior = [0.25 / (D as f64); D];
    verts.push(vertex!(interior));
    let distinct = verts.len(); // D+2

    // Duplicates: origin + interior again
    verts.push(vertex!([0.0; D]));
    verts.push(vertex!(interior));
    (verts, distinct)
}

/// Build `count` copies of the same all-identical vertex.
fn all_identical_vertices<const D: usize>(count: usize) -> Vec<Vertex<f64, (), D>> {
    (0..count).map(|_| vertex!([1.0; D])).collect()
}

/// Build simplex vertices plus an interior point, then one duplicate of the
/// origin. Returns `(vertices, distinct_count)`.
#[expect(
    clippy::cast_precision_loss,
    reason = "D ≤ 5 in practice; no precision loss"
)]
fn simplex_with_one_duplicate<const D: usize>() -> (Vec<Vertex<f64, (), D>>, usize) {
    let mut verts = simplex_vertices::<D>();
    // Extra non-vertex interior point to make the triangulation interesting
    let interior = [0.5 / (D as f64); D];
    verts.push(vertex!(interior));
    let distinct = verts.len();
    // One duplicate of the origin
    verts.push(vertex!([0.0; D]));
    (verts, distinct)
}

// =============================================================================
// MACRO — BATCH DEDUP TESTS (2D–5D)
// =============================================================================

/// Generate batch construction dedup tests for a given dimension:
///
/// - Hilbert dedup removes exact duplicates
/// - All-identical input fails gracefully
/// - `DedupPolicy::Exact` with `Input` ordering removes duplicates
/// - `DedupPolicy::Exact` with `Lexicographic` ordering removes duplicates
/// - Many-duplicate stress test collapses correctly
macro_rules! gen_dedup_batch_tests {
    ($dim:literal) => {
        pastey::paste! {
            #[test]
            fn [<test_batch_construction_deduplicates_exact_duplicates_ $dim d>]() {
                init_tracing();
                let (vertices, distinct) = simplex_with_interior_and_duplicates::<$dim>();
                assert!(vertices.len() > distinct);

                let dt: DelaunayTriangulation<_, (), (), $dim> =
                    DelaunayTriangulation::new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    )
                    .expect(concat!(
                        stringify!($dim), "D construction with duplicates should succeed"
                    ));

                assert_eq!(
                    dt.number_of_vertices(),
                    distinct,
                    "{}D: duplicates should be removed by Hilbert-sort dedup",
                    $dim
                );
                assert!(dt.number_of_cells() > 0);

                // Full validation (Levels 1–4) including coordinate uniqueness
                let validation = dt.as_triangulation().validate();
                assert!(
                    validation.is_ok(),
                    "{}D: triangulation should pass validation: {validation:?}",
                    $dim
                );
            }

            #[test]
            fn [<test_batch_construction_all_duplicates_fails_ $dim d>]() {
                init_tracing();
                // D+2 identical vertices → collapses to 1 → insufficient for simplex
                let vertices = all_identical_vertices::<$dim>($dim + 2);

                let result: Result<DelaunayTriangulation<_, (), (), $dim>, _> =
                    DelaunayTriangulation::new(&vertices);

                assert!(
                    result.is_err(),
                    "{}D: all-duplicate input should fail",
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
                    DelaunayTriangulation::new_with_options(&vertices, opts)
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
            fn [<test_explicit_dedup_exact_lexicographic_order_ $dim d>]() {
                init_tracing();
                let (vertices, distinct) = simplex_with_one_duplicate::<$dim>();

                let opts = ConstructionOptions::default()
                    .with_insertion_order(InsertionOrderStrategy::Lexicographic)
                    .with_dedup_policy(DedupPolicy::Exact);

                let dt: DelaunayTriangulation<_, (), (), $dim> =
                    DelaunayTriangulation::new_with_options(&vertices, opts)
                        .expect(concat!(
                            stringify!($dim),
                            "D: DedupPolicy::Exact + Lexicographic should succeed"
                        ));

                assert_eq!(
                    dt.number_of_vertices(),
                    distinct,
                    "{}D: DedupPolicy::Exact + Lexicographic should remove duplicate",
                    $dim
                );
                assert!(dt.validate().is_ok());
            }

            #[test]
            fn [<test_batch_construction_many_duplicates_ $dim d>]() {
                init_tracing();
                // D+2 distinct vertices, each repeated 5× = 5(D+2) total
                let base = simplex_with_interior_and_duplicates::<$dim>().0;
                // Take only the distinct portion (first D+2 entries)
                let distinct: Vec<Vertex<f64, (), $dim>> =
                    base.into_iter().take($dim + 2).collect();
                let distinct_count = distinct.len();

                let vertices: Vec<Vertex<f64, (), $dim>> = distinct
                    .iter()
                    .cycle()
                    .take(distinct_count * 5)
                    .copied()
                    .collect();

                assert_eq!(vertices.len(), distinct_count * 5);

                let dt: DelaunayTriangulation<_, (), (), $dim> =
                    DelaunayTriangulation::new(&vertices)
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
