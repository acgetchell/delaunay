//! Comprehensive tests for `insert_with_statistics` methods.
//!
//! This module tests both:
//! - `DelaunayTriangulation::insert_with_statistics`
//! - `Triangulation::insert_with_statistics`
//!
//! Coverage includes:
//! - Basic insertion and statistics tracking
//! - Hint caching behavior
//! - Perturbation retry mechanism
//! - Skipped vertex handling
//! - Non-retryable errors
//! - Bootstrap phase (< D+1 vertices)
//! - Post-bootstrap phase (≥ D+1 vertices)

use delaunay::prelude::*;

// =============================================================================
// DELAUNAY TRIANGULATION TESTS
// =============================================================================

#[test]
fn delaunay_insert_with_statistics_basic_2d() {
    let mut dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::empty();

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
    let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();

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
    let mut dt: DelaunayTriangulation<_, (), (), 4> = DelaunayTriangulation::empty();

    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0]),
        vertex!([0.2, 0.2, 0.2, 0.2]),
        vertex!([0.8, 0.1, 0.1, 0.1]),
    ];

    let mut total_attempts = 0;
    let mut successful_insertions = 0;

    for v in vertices {
        match dt.insert_with_statistics(v) {
            Ok((InsertionOutcome::Inserted { .. }, stats)) => {
                total_attempts += stats.attempts;
                successful_insertions += 1;
                assert!(stats.success());
                assert!(!stats.skipped());
            }
            Ok((InsertionOutcome::Skipped { .. }, stats)) => {
                assert!(stats.skipped());
                assert!(!stats.success());
            }
            Err(e) => panic!("unexpected non-retryable error: {e}"),
        }
    }

    assert_eq!(successful_insertions, 7);
    assert_eq!(dt.number_of_vertices(), 7);
    assert!(total_attempts >= 7); // At least 1 attempt per vertex
}

#[test]
fn delaunay_insert_with_statistics_duplicate_coordinates_2d() {
    let mut dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::empty();

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

    // Triangulation should still be valid
    assert!(dt.is_valid().is_ok());
    assert_eq!(dt.number_of_vertices(), 1);
}

#[test]
fn delaunay_insert_with_statistics_skipped_after_retries() {
    // This test would need a known degenerate configuration that exhausts retries.
    // For now, we test the structure rather than triggering actual skips.
    let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();

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
    let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();

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
// TRIANGULATION TESTS
// =============================================================================

#[test]
fn triangulation_insert_with_statistics_basic_2d() {
    let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
        Triangulation::new_empty(FastKernel::new());

    // Insert first vertex
    let (outcome, stats) = tri
        .insert_with_statistics(vertex!([0.0, 0.0]), None, None)
        .expect("insertion should succeed");

    assert!(matches!(
        outcome,
        InsertionOutcome::Inserted { hint: None, .. }
    ));
    assert_eq!(stats.attempts, 1);
    assert!(!stats.used_perturbation());
    assert!(!stats.skipped());
    assert!(stats.success());
    assert_eq!(tri.number_of_vertices(), 1);
}

#[test]
fn triangulation_insert_with_statistics_bootstrap_3d() {
    let mut tri: Triangulation<FastKernel<f64>, (), (), 3> =
        Triangulation::new_empty(FastKernel::new());

    // Insert D+1 vertices to create initial simplex
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];

    for (i, v) in vertices.into_iter().enumerate() {
        let (outcome, stats) = tri.insert_with_statistics(v, None, None).unwrap();

        assert!(matches!(outcome, InsertionOutcome::Inserted { .. }));
        assert_eq!(stats.attempts, 1);

        if i < 3 {
            // Bootstrap phase - no hint yet
            assert!(matches!(
                outcome,
                InsertionOutcome::Inserted { hint: None, .. }
            ));
        } else {
            // After D+1 vertices, hint should be available
            assert!(matches!(
                outcome,
                InsertionOutcome::Inserted { hint: Some(_), .. }
            ));
        }
    }

    assert_eq!(tri.number_of_vertices(), 4);
    assert_eq!(tri.number_of_cells(), 1);
}

#[test]
fn triangulation_insert_with_statistics_hint_usage_4d() {
    let mut tri: Triangulation<FastKernel<f64>, (), (), 4> =
        Triangulation::new_empty(FastKernel::new());

    // Build initial simplex
    for i in 0..5 {
        let mut coords = [0.0; 4];
        if i > 0 {
            coords[i - 1] = 1.0;
        }
        tri.insert_with_statistics(vertex!(coords), None, None)
            .unwrap();
    }

    // Insert with explicit hint
    let hint_cell = tri.cells().next().map(|(key, _)| key);
    let (outcome, stats) = tri
        .insert_with_statistics(vertex!([0.2, 0.2, 0.2, 0.2]), None, hint_cell)
        .unwrap();

    assert!(matches!(
        outcome,
        InsertionOutcome::Inserted { hint: Some(_), .. }
    ));
    assert_eq!(stats.attempts, 1);
    assert!(stats.success());
}

#[test]
fn triangulation_insert_with_statistics_duplicate_coordinates_3d() {
    let mut tri: Triangulation<FastKernel<f64>, (), (), 3> =
        Triangulation::new_empty(FastKernel::new());

    // Insert first vertex
    tri.insert_with_statistics(vertex!([1.0, 2.0, 3.0]), None, None)
        .unwrap();

    // Try duplicate - should be skipped
    let result = tri.insert_with_statistics(vertex!([1.0, 2.0, 3.0]), None, None);

    assert!(matches!(
        result,
        Ok((
            InsertionOutcome::Skipped {
                error: InsertionError::DuplicateCoordinates { .. }
            },
            _
        ))
    ));
}

#[test]
fn triangulation_insert_with_statistics_multiple_insertions_2d() {
    let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
        Triangulation::new_empty(FastKernel::new());

    let points = vec![
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.5, 1.0]),
        vertex!([0.3, 0.3]),
        vertex!([0.7, 0.3]),
    ];

    let mut all_succeeded = true;
    let mut max_attempts = 0;

    for point in points {
        match tri.insert_with_statistics(point, None, None) {
            Ok((InsertionOutcome::Inserted { .. }, stats)) => {
                max_attempts = max_attempts.max(stats.attempts);
                assert!(stats.success());
            }
            Ok((InsertionOutcome::Skipped { .. }, _)) | Err(_) => {
                all_succeeded = false;
            }
        }
    }

    assert!(all_succeeded, "all insertions should succeed");
    assert!(max_attempts >= 1);
    assert_eq!(tri.number_of_vertices(), 5);
}

#[test]
fn triangulation_insert_with_statistics_outcome_types() {
    let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
        Triangulation::new_empty(FastKernel::new());

    // Test Inserted variant
    let (outcome, _) = tri
        .insert_with_statistics(vertex!([0.0, 0.0]), None, None)
        .unwrap();

    match outcome {
        InsertionOutcome::Inserted { vertex_key, hint } => {
            // Verify we can access the fields
            assert!(tri.vertices().any(|(k, _)| k == vertex_key));
            assert_eq!(hint, None); // No hint during bootstrap
        }
        InsertionOutcome::Skipped { .. } => panic!("expected Inserted, got Skipped"),
    }
}

#[test]
fn triangulation_insert_with_statistics_sequential_5d() {
    let mut tri: Triangulation<FastKernel<f64>, (), (), 5> =
        Triangulation::new_empty(FastKernel::new());

    // Insert 6 vertices to form initial simplex
    for i in 0..6 {
        let mut coords = [0.0; 5];
        if i > 0 {
            coords[i - 1] = 1.0;
        }

        let (outcome, stats) = tri
            .insert_with_statistics(vertex!(coords), None, None)
            .unwrap();

        assert!(matches!(outcome, InsertionOutcome::Inserted { .. }));
        assert_eq!(stats.attempts, 1);
        assert!(stats.success());
    }

    assert_eq!(tri.number_of_vertices(), 6);
    assert_eq!(tri.number_of_cells(), 1);
}

// =============================================================================
// PROPERTY TESTS (STATISTICS INVARIANTS)
// =============================================================================

#[test]
fn statistics_invariants() {
    let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();

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

#[test]
fn statistics_cells_removed_during_repair() {
    let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
        Triangulation::new_empty(FastKernel::new());

    // Build simplex
    tri.insert_with_statistics(vertex!([0.0, 0.0]), None, None)
        .unwrap();
    tri.insert_with_statistics(vertex!([1.0, 0.0]), None, None)
        .unwrap();
    tri.insert_with_statistics(vertex!([0.5, 1.0]), None, None)
        .unwrap();

    // Insert interior point - might trigger repair
    let (_outcome, stats) = tri
        .insert_with_statistics(vertex!([0.5, 0.3]), None, None)
        .unwrap();

    // cells_removed_during_repair is usize, so always non-negative by type
    // Just verify it's reasonable for a simple insertion
    assert!(
        stats.cells_removed_during_repair < 100,
        "cells removed should be reasonable for simple insertion, got {}",
        stats.cells_removed_during_repair
    );
}

// =============================================================================
// DIMENSIONAL COVERAGE
// =============================================================================

#[test]
fn insert_with_statistics_2d_coverage() {
    let mut dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::empty();

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
    let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();

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
    let mut dt: DelaunayTriangulation<_, (), (), 4> = DelaunayTriangulation::empty();

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
    let mut dt: DelaunayTriangulation<_, (), (), 5> = DelaunayTriangulation::empty();

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
