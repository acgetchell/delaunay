//! Comprehensive tests for `insert_with_statistics` methods.
//!
//! This module tests:
//! - `DelaunayTriangulation::insert_with_statistics`
//! - `DelaunayTriangulation::insert_best_effort_with_statistics`
//!
//! Triangulation-layer insertion tests live in `src/triangulation/insertion.rs`.
//!
//! Coverage includes:
//! - Basic insertion and statistics tracking
//! - Hint caching behavior
//! - Perturbation retry mechanism
//! - Skipped vertex handling
//! - Non-retryable errors
//! - Bootstrap phase (< D+1 vertices)
//! - Post-bootstrap phase (≥ D+1 vertices)

use delaunay::vertex;
use std::assert_matches;

use delaunay::prelude::construction::{
    DelaunayIncrementalBuilder, DelaunayIncrementalBuilderError, TopologyGuarantee, Vertex,
};
use delaunay::prelude::geometry::CoordinateValues;
use delaunay::prelude::insertion::{InsertionError, InsertionOutcome};

fn unit_simplex_vertices<const D: usize>() -> Vec<Vertex<(), D>> {
    let mut vertices = vec![vertex!([0.0; D]).unwrap()];
    for axis in 0..D {
        let mut coordinates = [0.0; D];
        coordinates[axis] = 1.0;
        vertices.push(vertex!(coordinates).unwrap());
    }
    vertices
}

fn duplicate_coordinates(error: DelaunayIncrementalBuilderError) -> CoordinateValues {
    match error {
        DelaunayIncrementalBuilderError::Insertion { source } => match *source {
            InsertionError::DuplicateCoordinates { coordinates } => coordinates,
            other => panic!("expected DuplicateCoordinates, got {other:?}"),
        },
        other => panic!("expected insertion error, got {other:?}"),
    }
}

macro_rules! test_near_duplicate_stage_parity {
    ($($dim:literal),+ $(,)?) => {
        pastey::paste! {
            $(
                #[test]
                fn [<near_duplicate_policy_matches_before_and_after_publication_ $dim d>]() {
                    let simplex = unit_simplex_vertices::<$dim>();
                    let candidate = [5.0e-11; $dim];
                    let expected_coordinates = CoordinateValues::from(candidate);

                    let mut bootstrap: DelaunayIncrementalBuilder<_, (), (), $dim> =
                        DelaunayIncrementalBuilder::new();
                    for vertex in simplex.iter().take($dim) {
                        bootstrap.insert_vertex(*vertex).unwrap();
                    }
                    assert_eq!(bootstrap.number_of_vertices(), $dim);
                    assert_eq!(bootstrap.number_of_simplices(), 0);

                    let bootstrap_error = bootstrap
                        .insert_vertex(vertex!(candidate).unwrap())
                        .expect_err("unit-scale bootstrap should reject a near-duplicate");
                    assert_eq!(duplicate_coordinates(bootstrap_error), expected_coordinates);
                    assert_eq!(bootstrap.number_of_vertices(), $dim);
                    assert_eq!(bootstrap.number_of_simplices(), 0);
                    bootstrap.validate_structure().unwrap();

                    let mut published: DelaunayIncrementalBuilder<_, (), (), $dim> =
                        DelaunayIncrementalBuilder::new();
                    for vertex in simplex {
                        published.insert_vertex(vertex).unwrap();
                    }
                    let published_vertex_count = published.number_of_vertices();
                    let published_simplex_count = published.number_of_simplices();
                    assert_eq!(published_vertex_count, $dim + 1);
                    assert!(published_simplex_count > 0);

                    let published_error = published
                        .insert_vertex(vertex!(candidate).unwrap())
                        .expect_err("published owner should reject the same near-duplicate");
                    assert_eq!(duplicate_coordinates(published_error), expected_coordinates);
                    assert_eq!(published.number_of_vertices(), published_vertex_count);
                    assert_eq!(published.number_of_simplices(), published_simplex_count);
                    published.finish().unwrap().validate().unwrap();
                }
            )+
        }
    };
}

test_near_duplicate_stage_parity!(2, 3, 4, 5);

// =============================================================================
// DELAUNAY TRIANGULATION TESTS
// =============================================================================

#[test]
fn delaunay_insert_with_statistics_basic_2d() {
    let mut dt: DelaunayIncrementalBuilder<_, (), (), 2> =
        DelaunayIncrementalBuilder::with_topology_guarantee(TopologyGuarantee::PLManifold);

    // Insert first vertex
    let (outcome, stats) = dt
        .insert_with_statistics(vertex!([0.0, 0.0]).unwrap())
        .expect("insertion should succeed");

    assert_matches!(outcome, InsertionOutcome::Inserted { .. });
    assert_eq!(stats.attempts, 1);
    assert!(!stats.used_perturbation());
    assert!(!stats.skipped());
    assert!(stats.success());
    assert_eq!(stats.simplices_removed_during_repair, 0);
    assert_eq!(dt.number_of_vertices(), 1);
    assert_eq!(dt.number_of_simplices(), 0);

    // Insert second vertex
    let (outcome, stats) = dt
        .insert_with_statistics(vertex!([1.0, 0.0]).unwrap())
        .expect("insertion should succeed");

    assert_matches!(outcome, InsertionOutcome::Inserted { .. });
    assert_eq!(stats.attempts, 1);
    assert_eq!(dt.number_of_vertices(), 2);

    // Insert third vertex (completes simplex)
    let (outcome, stats) = dt
        .insert_with_statistics(vertex!([0.5, 1.0]).unwrap())
        .expect("insertion should succeed");

    assert_matches!(outcome, InsertionOutcome::Inserted { hint, .. } if hint.is_some());
    assert_eq!(stats.attempts, 1);
    assert!(stats.success());
    assert_eq!(dt.number_of_vertices(), 3);
    assert_eq!(dt.number_of_simplices(), 1);
}

#[test]
fn delaunay_insert_with_statistics_hint_caching_3d() {
    let mut dt: DelaunayIncrementalBuilder<_, (), (), 3> =
        DelaunayIncrementalBuilder::with_topology_guarantee(TopologyGuarantee::PLManifold);

    // Build initial simplex
    dt.insert_with_statistics(vertex!([0.0, 0.0, 0.0]).unwrap())
        .unwrap();
    dt.insert_with_statistics(vertex!([1.0, 0.0, 0.0]).unwrap())
        .unwrap();
    dt.insert_with_statistics(vertex!([0.0, 1.0, 0.0]).unwrap())
        .unwrap();
    let (outcome, _) = dt
        .insert_with_statistics(vertex!([0.0, 0.0, 1.0]).unwrap())
        .unwrap();

    // After simplex creation, hint should be available
    assert_matches!(outcome, InsertionOutcome::Inserted { hint: Some(_), .. });

    // Insert interior point - should benefit from hint
    let (outcome, stats) = dt
        .insert_with_statistics(vertex!([0.25, 0.25, 0.25]).unwrap())
        .unwrap();

    assert_matches!(outcome, InsertionOutcome::Inserted { hint: Some(_), .. });
    assert_eq!(stats.attempts, 1);
    assert!(stats.success());
    assert_eq!(dt.number_of_vertices(), 5);
    assert!(dt.number_of_simplices() > 1);
}

#[test]
fn delaunay_insert_with_statistics_multiple_vertices_4d() {
    let mut dt: DelaunayIncrementalBuilder<_, (), (), 4> =
        DelaunayIncrementalBuilder::with_topology_guarantee(TopologyGuarantee::PLManifold);

    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 1.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 0.0, 1.0]).unwrap(),
        vertex!([0.2, 0.2, 0.2, 0.2]).unwrap(),
        vertex!([0.8, 0.1, 0.1, 0.1]).unwrap(),
    ];

    let input_count = vertices.len();

    let mut total_attempts = 0;
    let mut successful_insertions = 0;
    let mut skipped = 0;

    for v in vertices {
        match dt.insert_best_effort_with_statistics(v) {
            Ok((InsertionOutcome::Inserted { .. }, stats)) => {
                total_attempts += stats.attempts;
                successful_insertions += 1;
                assert!(stats.success());
                assert!(!stats.skipped());
            }
            Ok((InsertionOutcome::Skipped { .. }, stats)) => {
                total_attempts += stats.attempts;
                skipped += 1;
                assert!(stats.skipped());
                assert!(!stats.success());
            }
            Err(e) => panic!("unexpected non-retryable error: {e}"),
        }
    }

    assert_eq!(successful_insertions + skipped, input_count);
    assert_eq!(dt.number_of_vertices(), successful_insertions);
    assert!(total_attempts >= input_count); // At least 1 attempt per vertex
}

#[test]
fn delaunay_insert_with_statistics_handles_degenerate_k2_flips_4d() {
    let mut dt: DelaunayIncrementalBuilder<_, (), (), 4> =
        DelaunayIncrementalBuilder::with_topology_guarantee(TopologyGuarantee::PLManifold);

    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 1.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 0.0, 1.0]).unwrap(),
        vertex!([0.2, 0.2, 0.2, 0.2]).unwrap(),
        vertex!([0.8, 0.1, 0.1, 0.1]).unwrap(),
    ];

    for v in vertices {
        let result = dt.insert_with_statistics(v);
        assert!(result.is_ok(), "4D insertion failed: {result:?}");
    }

    assert_eq!(dt.number_of_vertices(), 7);
    assert!(dt.validate_structure().is_ok());
    assert!(dt.finish().is_ok());
}

#[test]
fn delaunay_insert_with_statistics_duplicate_coordinates_2d() {
    let mut dt: DelaunayIncrementalBuilder<_, (), (), 2> =
        DelaunayIncrementalBuilder::with_topology_guarantee(TopologyGuarantee::PLManifold);

    // Insert first vertex
    dt.insert_with_statistics(vertex!([1.0, 2.0]).unwrap())
        .expect("first insertion should succeed");

    // The strict statistics API reports skipped insertions as errors so callers
    // using `?` cannot miss them.
    let result = dt.insert_with_statistics(vertex!([1.0, 2.0]).unwrap());
    let duplicate_coordinates = CoordinateValues::from([1.0, 2.0]);
    assert!(
        matches!(
            result,
            Err(DelaunayIncrementalBuilderError::Insertion { ref source })
                if matches!(
                    source.as_ref(),
                    InsertionError::DuplicateCoordinates { coordinates }
                        if coordinates == &duplicate_coordinates
                )
        ),
        "expected duplicate coordinate error, got: {result:?}"
    );

    // The explicitly best-effort API preserves the skipped outcome plus telemetry.
    let result = dt.insert_best_effort_with_statistics(vertex!([1.0, 2.0]).unwrap());

    match result {
        Ok((
            InsertionOutcome::Skipped {
                error: InsertionError::DuplicateCoordinates { coordinates },
            },
            stats,
        )) => {
            assert_eq!(coordinates, duplicate_coordinates);
            assert!(stats.skipped_duplicate());
            assert_eq!(stats.attempts, 1);
        }
        other => {
            panic!("expected best-effort Ok(Skipped) with DuplicateCoordinates, got: {other:?}")
        }
    }

    // Still in bootstrap (no simplices yet), so validate only Levels 1–2 (elements + structure).
    assert!(dt.validate_structure().is_ok());
    assert_eq!(dt.number_of_vertices(), 1);
}

#[test]
fn delaunay_insert_with_statistics_bootstrap_happy_path_3d() {
    // Happy path: inserting D+1 well-separated vertices should succeed without retries.
    let mut dt: DelaunayIncrementalBuilder<_, (), (), 3> =
        DelaunayIncrementalBuilder::with_topology_guarantee(TopologyGuarantee::PLManifold);

    // Build simplex with well-separated points
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 1.0]).unwrap(),
    ];

    for v in vertices {
        let (outcome, stats) = dt.insert_best_effort_with_statistics(v).unwrap();
        assert_matches!(outcome, InsertionOutcome::Inserted { .. });
        assert_eq!(stats.attempts, 1);
    }

    assert_eq!(dt.number_of_vertices(), 4);
}

#[test]
fn delaunay_insert_with_statistics_statistics_fields_3d() {
    let mut dt: DelaunayIncrementalBuilder<_, (), (), 3> =
        DelaunayIncrementalBuilder::with_topology_guarantee(TopologyGuarantee::PLManifold);

    // Bootstrap phase
    for i in 0..4 {
        let mut coords = [0.0; 3];
        if i > 0 {
            coords[i - 1] = 1.0;
        }

        let (outcome, stats) = dt.insert_with_statistics(vertex!(coords).unwrap()).unwrap();

        // Verify all statistics fields
        assert_matches!(outcome, InsertionOutcome::Inserted { .. });
        assert!(stats.attempts >= 1);
        assert!(!stats.skipped());
        assert!(stats.success());
        assert_eq!(stats.simplices_removed_during_repair, 0);

        if i < 3 {
            assert!(!stats.used_perturbation());
        }
    }

    assert_eq!(dt.number_of_vertices(), 4);
    assert_eq!(dt.number_of_simplices(), 1);
}
// =============================================================================
// PROPERTY TESTS (STATISTICS INVARIANTS)
// =============================================================================

#[test]
fn statistics_invariants() {
    let mut dt: DelaunayIncrementalBuilder<_, (), (), 3> =
        DelaunayIncrementalBuilder::with_topology_guarantee(TopologyGuarantee::PLManifold);

    // Build simplex
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 1.0]).unwrap(),
        vertex!([0.25, 0.25, 0.25]).unwrap(),
    ];

    for v in vertices {
        let (outcome, stats) = dt.insert_with_statistics(v).unwrap();

        match outcome {
            InsertionOutcome::Inserted { .. } => {
                // Invariant: success => attempts >= 1
                assert!(
                    stats.attempts >= 1,
                    "successful insertion must have ≥1 attempt"
                );

                // Invariant: success => success flag set
                assert!(
                    stats.success(),
                    "successful insertion must set success flag"
                );

                // Invariant: success => not skipped
                assert!(!stats.skipped(), "successful insertion must not be skipped");

                // Invariant: used_perturbation => attempts > 1
                if stats.used_perturbation() {
                    assert!(stats.attempts > 1, "perturbation implies multiple attempts");
                }
            }
            InsertionOutcome::Skipped { .. } => {
                // Invariant: skipped => skipped flag set
                assert!(stats.skipped(), "skipped outcome must set skipped flag");

                // Invariant: skipped => not success
                assert!(!stats.success(), "skipped insertion must not be successful");

                // Invariant: skipped => attempts >= 1
                assert!(
                    stats.attempts >= 1,
                    "skipped insertion must have ≥1 attempt"
                );
            }
        }
    }
}
// =============================================================================
// DIMENSIONAL COVERAGE
// =============================================================================

#[test]
fn insert_with_statistics_2d_coverage() {
    let mut dt: DelaunayIncrementalBuilder<_, (), (), 2> =
        DelaunayIncrementalBuilder::with_topology_guarantee(TopologyGuarantee::PLManifold);

    let vertices = vec![
        vertex!([0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
        vertex!([0.5, 0.5]).unwrap(),
    ];

    for v in vertices {
        let result = dt.insert_with_statistics(v);
        assert!(result.is_ok(), "2D insertion failed: {result:?}");
    }

    assert_eq!(dt.number_of_vertices(), 4);
}

#[test]
fn insert_with_statistics_3d_coverage() {
    let mut dt: DelaunayIncrementalBuilder<_, (), (), 3> =
        DelaunayIncrementalBuilder::with_topology_guarantee(TopologyGuarantee::PLManifold);

    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 1.0]).unwrap(),
        vertex!([0.5, 0.5, 0.5]).unwrap(),
    ];

    for v in vertices {
        let result = dt.insert_with_statistics(v);
        assert!(result.is_ok(), "3D insertion failed: {result:?}");
    }

    assert_eq!(dt.number_of_vertices(), 5);
}

#[test]
fn insert_with_statistics_4d_coverage() {
    let mut dt: DelaunayIncrementalBuilder<_, (), (), 4> =
        DelaunayIncrementalBuilder::with_topology_guarantee(TopologyGuarantee::PLManifold);

    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 1.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 0.0, 1.0]).unwrap(),
        vertex!([0.2, 0.2, 0.2, 0.2]).unwrap(),
    ];

    for v in vertices {
        let result = dt.insert_with_statistics(v);
        assert!(result.is_ok(), "4D insertion failed: {result:?}");
    }

    assert_eq!(dt.number_of_vertices(), 6);
}

#[test]
fn insert_with_statistics_5d_coverage() {
    let mut dt: DelaunayIncrementalBuilder<_, (), (), 5> =
        DelaunayIncrementalBuilder::with_topology_guarantee(TopologyGuarantee::PLManifold);

    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0, 0.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 1.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 0.0, 1.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 0.0, 0.0, 1.0]).unwrap(),
    ];

    for v in vertices {
        let result = dt.insert_with_statistics(v);
        assert!(result.is_ok(), "5D insertion failed: {result:?}");
    }

    assert_eq!(dt.number_of_vertices(), 6);
}
