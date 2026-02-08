//! Comprehensive tests for `insert_with_statistics` methods.
//!
//! This module tests:
//! - `DelaunayTriangulation::insert_with_statistics`
//!
//! Triangulation-layer insertion tests live in `core::triangulation` unit tests.
//!
//! Coverage includes:
//! - Basic insertion and statistics tracking
//! - Hint caching behavior
//! - Perturbation retry mechanism
//! - Skipped vertex handling
//! - Non-retryable errors
//! - Bootstrap phase (< D+1 vertices)
//! - Post-bootstrap phase (≥ D+1 vertices)

use delaunay::prelude::triangulation::*;

// =============================================================================
// DELAUNAY TRIANGULATION TESTS
// =============================================================================

#[test]
fn delaunay_insert_with_statistics_basic_2d() {
    let mut dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::empty_with_topology_guarantee(TopologyGuarantee::PLManifold);

    // Insert first vertex
    let (outcome, stats) = dt
        .insert_with_statistics(vertex!([0.0, 0.0]))
        .expect("insertion should succeed");

    assert!(matches!(outcome, InsertionOutcome::Inserted { .. }));
    assert_eq!(stats.attempts, 1);
    assert!(!stats.used_perturbation());
    assert!(!stats.skipped());
    assert!(stats.success());
    assert_eq!(stats.cells_removed_during_repair, 0);
    assert_eq!(dt.number_of_vertices(), 1);
    assert_eq!(dt.number_of_cells(), 0);

    // Insert second vertex
    let (outcome, stats) = dt
        .insert_with_statistics(vertex!([1.0, 0.0]))
        .expect("insertion should succeed");

    assert!(matches!(outcome, InsertionOutcome::Inserted { .. }));
    assert_eq!(stats.attempts, 1);
    assert_eq!(dt.number_of_vertices(), 2);

    // Insert third vertex (completes simplex)
    let (outcome, stats) = dt
        .insert_with_statistics(vertex!([0.5, 1.0]))
        .expect("insertion should succeed");

    assert!(matches!(outcome, InsertionOutcome::Inserted { hint, .. } if hint.is_some()));
    assert_eq!(stats.attempts, 1);
    assert!(stats.success());
    assert_eq!(dt.number_of_vertices(), 3);
    assert_eq!(dt.number_of_cells(), 1);
}

#[test]
fn delaunay_insert_with_statistics_hint_caching_3d() {
    let mut dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::empty_with_topology_guarantee(TopologyGuarantee::PLManifold);

    // Build initial simplex
    dt.insert_with_statistics(vertex!([0.0, 0.0, 0.0])).unwrap();
    dt.insert_with_statistics(vertex!([1.0, 0.0, 0.0])).unwrap();
    dt.insert_with_statistics(vertex!([0.0, 1.0, 0.0])).unwrap();
    let (outcome, _) = dt.insert_with_statistics(vertex!([0.0, 0.0, 1.0])).unwrap();

    // After simplex creation, hint should be available
    assert!(matches!(
        outcome,
        InsertionOutcome::Inserted { hint: Some(_), .. }
    ));

    // Insert interior point - should benefit from hint
    let (outcome, stats) = dt
        .insert_with_statistics(vertex!([0.25, 0.25, 0.25]))
        .unwrap();

    assert!(matches!(
        outcome,
        InsertionOutcome::Inserted { hint: Some(_), .. }
    ));
    assert_eq!(stats.attempts, 1);
    assert!(stats.success());
    assert_eq!(dt.number_of_vertices(), 5);
    assert!(dt.number_of_cells() > 1);
}

#[test]
fn delaunay_insert_with_statistics_multiple_vertices_4d() {
    let mut dt: DelaunayTriangulation<_, (), (), 4> =
        DelaunayTriangulation::empty_with_topology_guarantee(TopologyGuarantee::PLManifold);

    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0]),
        vertex!([0.2, 0.2, 0.2, 0.2]),
        vertex!([0.8, 0.1, 0.1, 0.1]),
    ];

    let input_count = vertices.len();

    let mut total_attempts = 0;
    let mut successful_insertions = 0;
    let mut skipped = 0;

    for v in vertices {
        match dt.insert_with_statistics(v) {
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
    let mut dt: DelaunayTriangulation<_, (), (), 4> =
        DelaunayTriangulation::empty_with_topology_guarantee(TopologyGuarantee::PLManifold);

    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0]),
        vertex!([0.2, 0.2, 0.2, 0.2]),
        vertex!([0.8, 0.1, 0.1, 0.1]),
    ];

    for v in vertices {
        let result = dt.insert_with_statistics(v);
        assert!(result.is_ok(), "4D insertion failed: {result:?}");
    }

    assert_eq!(dt.number_of_vertices(), 7);
    assert!(dt.tds().validate().is_ok());
}

#[test]
fn delaunay_insert_with_statistics_duplicate_coordinates_2d() {
    let mut dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::empty_with_topology_guarantee(TopologyGuarantee::PLManifold);

    // Insert first vertex
    dt.insert_with_statistics(vertex!([1.0, 2.0]))
        .expect("first insertion should succeed");

    // Try to insert vertex with same coordinates - should be skipped
    let result = dt.insert_with_statistics(vertex!([1.0, 2.0]));

    match result {
        Ok((
            InsertionOutcome::Skipped {
                error: InsertionError::DuplicateCoordinates { coordinates },
            },
            stats,
        )) => {
            assert!(coordinates.contains('1'));
            assert!(coordinates.contains('2'));
            assert!(stats.skipped_duplicate());
            assert_eq!(stats.attempts, 1);
        }
        other => panic!("expected Ok(Skipped) with DuplicateCoordinates, got: {other:?}"),
    }

    // Still in bootstrap (no cells yet), so validate only Levels 1–2 (elements + structure).
    assert!(dt.tds().validate().is_ok());
    assert_eq!(dt.number_of_vertices(), 1);
}

#[test]
fn delaunay_insert_with_statistics_bootstrap_happy_path_3d() {
    // Happy path: inserting D+1 well-separated vertices should succeed without retries.
    let mut dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::empty_with_topology_guarantee(TopologyGuarantee::PLManifold);

    // Build simplex with well-separated points
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];

    for v in vertices {
        let (outcome, stats) = dt.insert_with_statistics(v).unwrap();
        assert!(matches!(outcome, InsertionOutcome::Inserted { .. }));
        assert_eq!(stats.attempts, 1);
    }

    assert_eq!(dt.number_of_vertices(), 4);
}

#[test]
fn delaunay_insert_with_statistics_statistics_fields_3d() {
    let mut dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::empty_with_topology_guarantee(TopologyGuarantee::PLManifold);

    // Bootstrap phase
    for i in 0..4 {
        let mut coords = [0.0; 3];
        if i > 0 {
            coords[i - 1] = 1.0;
        }

        let (outcome, stats) = dt.insert_with_statistics(vertex!(coords)).unwrap();

        // Verify all statistics fields
        assert!(matches!(outcome, InsertionOutcome::Inserted { .. }));
        assert!(stats.attempts >= 1);
        assert!(!stats.skipped());
        assert!(stats.success());
        assert_eq!(stats.cells_removed_during_repair, 0);

        if i < 3 {
            assert!(!stats.used_perturbation());
        }
    }

    assert_eq!(dt.number_of_vertices(), 4);
    assert_eq!(dt.number_of_cells(), 1);
}
// =============================================================================
// PROPERTY TESTS (STATISTICS INVARIANTS)
// =============================================================================

#[test]
fn statistics_invariants() {
    let mut dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::empty_with_topology_guarantee(TopologyGuarantee::PLManifold);

    // Build simplex
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
        vertex!([0.25, 0.25, 0.25]),
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
    let mut dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::empty_with_topology_guarantee(TopologyGuarantee::PLManifold);

    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
        vertex!([0.5, 0.5]),
    ];

    for v in vertices {
        let result = dt.insert_with_statistics(v);
        assert!(result.is_ok(), "2D insertion failed: {result:?}");
    }

    assert_eq!(dt.number_of_vertices(), 4);
}

#[test]
fn insert_with_statistics_3d_coverage() {
    let mut dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::empty_with_topology_guarantee(TopologyGuarantee::PLManifold);

    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
        vertex!([0.5, 0.5, 0.5]),
    ];

    for v in vertices {
        let result = dt.insert_with_statistics(v);
        assert!(result.is_ok(), "3D insertion failed: {result:?}");
    }

    assert_eq!(dt.number_of_vertices(), 5);
}

#[test]
fn insert_with_statistics_4d_coverage() {
    let mut dt: DelaunayTriangulation<_, (), (), 4> =
        DelaunayTriangulation::empty_with_topology_guarantee(TopologyGuarantee::PLManifold);

    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0]),
        vertex!([0.2, 0.2, 0.2, 0.2]),
    ];

    for v in vertices {
        let result = dt.insert_with_statistics(v);
        assert!(result.is_ok(), "4D insertion failed: {result:?}");
    }

    assert_eq!(dt.number_of_vertices(), 6);
}

#[test]
fn insert_with_statistics_5d_coverage() {
    let mut dt: DelaunayTriangulation<_, (), (), 5> =
        DelaunayTriangulation::empty_with_topology_guarantee(TopologyGuarantee::PLManifold);

    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 0.0, 1.0]),
    ];

    for v in vertices {
        let result = dt.insert_with_statistics(v);
        assert!(result.is_ok(), "5D insertion failed: {result:?}");
    }

    assert_eq!(dt.number_of_vertices(), 6);
}
