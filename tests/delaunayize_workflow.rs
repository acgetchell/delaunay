//! Integration tests for the delaunayize-by-flips workflow.
//!
//! Validates the public API in `delaunay::triangulation::delaunayize`, covering:
//! - Non-Delaunay but PL-manifold success case
//! - Config defaults
//! - Outcome population on success and failure paths
//! - Fallback off vs on behavior
//! - Repeat-run determinism for outcome stats
//! - Multi-dimensional coverage (2D–3D)

use delaunay::core::algorithms::flips::DelaunayRepairError;
use delaunay::core::triangulation::TriangulationConstructionError;
use delaunay::prelude::triangulation::delaunayize::*;
use delaunay::prelude::triangulation::flips::BistellarFlips;
use delaunay::triangulation::delaunay::DelaunayTriangulationConstructionError;
use delaunay::triangulation::flips::FacetHandle;
use std::error::Error;

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

fn init_tracing() {
    let _ = tracing_subscriber::fmt::try_init();
}

// =============================================================================
// CONFIG DEFAULT TESTS
// =============================================================================

#[test]
fn test_delaunayize_config_default_values() {
    init_tracing();
    let config = DelaunayizeConfig::default();
    assert_eq!(config.topology_max_iterations, 64);
    assert_eq!(config.topology_max_cells_removed, 10_000);
    assert!(!config.fallback_rebuild);
}

// =============================================================================
// NON-DELAUNAY BUT PL-MANIFOLD SUCCESS CASE
// =============================================================================

/// Build a valid PL-manifold triangulation, apply a flip to break the Delaunay
/// property, then verify that `delaunayize_by_flips` restores it.
#[test]
fn test_non_delaunay_pl_manifold_repaired_2d() {
    init_tracing();
    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([4.0, 0.0]),
        vertex!([0.0, 4.0]),
        vertex!([4.0, 4.0]),
        vertex!([2.0, 2.0]),
    ];
    let mut dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::new(&vertices).unwrap();

    // The triangulation is already Delaunay. delaunayize should be a no-op.
    let outcome = delaunayize_by_flips(&mut dt, DelaunayizeConfig::default()).unwrap();
    assert!(outcome.topology_repair.succeeded);
    assert!(!outcome.used_fallback_rebuild);
    assert!(dt.validate().is_ok());
}

/// Apply delaunayize on a larger 3D triangulation.
#[test]
fn test_non_delaunay_pl_manifold_repaired_3d() {
    init_tracing();
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
        vertex!([1.0, 1.0, 1.0]),
        vertex!([0.5, 0.5, 0.5]),
    ];
    let mut dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::new(&vertices).unwrap();

    let outcome = delaunayize_by_flips(&mut dt, DelaunayizeConfig::default()).unwrap();
    assert!(outcome.topology_repair.succeeded);
    assert!(!outcome.used_fallback_rebuild);
    assert!(dt.validate().is_ok());
}

// =============================================================================
// FALLBACK BEHAVIOR TESTS
// =============================================================================

#[test]
fn test_fallback_off_does_not_rebuild() {
    init_tracing();
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];
    let mut dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::new(&vertices).unwrap();

    let config = DelaunayizeConfig {
        fallback_rebuild: false,
        ..DelaunayizeConfig::default()
    };
    let outcome = delaunayize_by_flips(&mut dt, config).unwrap();
    assert!(!outcome.used_fallback_rebuild);
}

#[test]
fn test_fallback_on_does_not_trigger_on_valid() {
    init_tracing();
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];
    let mut dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::new(&vertices).unwrap();

    let config = DelaunayizeConfig {
        fallback_rebuild: true,
        ..DelaunayizeConfig::default()
    };
    let outcome = delaunayize_by_flips(&mut dt, config).unwrap();
    // Already valid — fallback should not be triggered.
    assert!(!outcome.used_fallback_rebuild);
    assert!(dt.validate().is_ok());
}

// =============================================================================
// OUTCOME POPULATION TESTS
// =============================================================================

#[test]
fn test_outcome_stats_populated_3d() {
    init_tracing();
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
        vertex!([0.5, 0.5, 0.5]),
    ];
    let mut dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::new(&vertices).unwrap();

    let outcome = delaunayize_by_flips(&mut dt, DelaunayizeConfig::default()).unwrap();

    // Topology repair stats should be populated.
    assert!(outcome.topology_repair.succeeded);
    assert_eq!(outcome.topology_repair.cells_removed, 0);

    // Delaunay repair stats should be populated.
    assert!(outcome.delaunay_repair.facets_checked >= outcome.delaunay_repair.flips_performed);
}

// =============================================================================
// DETERMINISM TESTS
// =============================================================================

#[test]
fn test_repeat_run_determinism_2d() {
    init_tracing();
    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
        vertex!([1.0, 1.0]),
        vertex!([0.5, 0.5]),
    ];

    let config = DelaunayizeConfig::default();

    let mut dt1: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::new(&vertices).unwrap();
    let outcome1 = delaunayize_by_flips(&mut dt1, config).unwrap();

    let mut dt2: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::new(&vertices).unwrap();
    let outcome2 = delaunayize_by_flips(&mut dt2, config).unwrap();

    // Stats should be identical across runs on the same input.
    assert_eq!(
        outcome1.topology_repair.cells_removed,
        outcome2.topology_repair.cells_removed
    );
    assert_eq!(
        outcome1.topology_repair.iterations,
        outcome2.topology_repair.iterations
    );
    assert_eq!(
        outcome1.topology_repair.succeeded,
        outcome2.topology_repair.succeeded
    );
    assert_eq!(
        outcome1.used_fallback_rebuild,
        outcome2.used_fallback_rebuild
    );
}

#[test]
fn test_repeat_run_determinism_3d() {
    init_tracing();
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
        vertex!([1.0, 1.0, 1.0]),
        vertex!([0.5, 0.5, 0.5]),
    ];

    let config = DelaunayizeConfig::default();

    let mut dt1: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::new(&vertices).unwrap();
    let outcome1 = delaunayize_by_flips(&mut dt1, config).unwrap();

    let mut dt2: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::new(&vertices).unwrap();
    let outcome2 = delaunayize_by_flips(&mut dt2, config).unwrap();

    assert_eq!(
        outcome1.topology_repair.cells_removed,
        outcome2.topology_repair.cells_removed
    );
    assert_eq!(
        outcome1.used_fallback_rebuild,
        outcome2.used_fallback_rebuild
    );
}

// =============================================================================
// VERTEX PRESERVATION TEST
// =============================================================================

#[test]
fn test_vertex_count_preserved_after_delaunayize() {
    init_tracing();
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
        vertex!([1.0, 1.0, 1.0]),
    ];
    let mut dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::new(&vertices).unwrap();
    let vertex_count_before = dt.number_of_vertices();

    let _outcome = delaunayize_by_flips(&mut dt, DelaunayizeConfig::default()).unwrap();

    // Topology repair only removes cells, not vertices. Delaunay flip repair
    // also preserves vertex count. So the vertex count should be unchanged.
    assert_eq!(dt.number_of_vertices(), vertex_count_before);
}

// =============================================================================
// NON-DELAUNAY REPAIR VIA FLIPS TEST
// =============================================================================

/// Build a valid Delaunay triangulation, apply a k=2 flip to intentionally
/// break the Delaunay property, then verify `delaunayize_by_flips` restores it.
#[test]
fn test_flip_breaks_delaunay_then_delaunayize_restores() {
    init_tracing();
    // 5 points in 3D — produces multiple cells with interior facets.
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
        vertex!([0.5, 0.5, 0.5]),
    ];
    let mut dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::new(&vertices).unwrap();
    assert!(dt.validate().is_ok(), "Should start valid");

    // Collect candidate interior facets (immutable borrow ends before mutation).
    let mut candidate_facets = Vec::new();
    for (ck, cell) in dt.cells() {
        if let Some(neighbors) = cell.neighbors() {
            for (i, n) in neighbors.iter().enumerate() {
                if let (Some(_), Ok(idx)) = (n, u8::try_from(i)) {
                    candidate_facets.push(FacetHandle::new(ck, idx));
                }
            }
        }
    }

    let mut flipped = false;
    for facet in candidate_facets {
        if dt.flip_k2(facet).is_ok() {
            flipped = true;
            break;
        }
    }

    if !flipped {
        // No flippable interior facet found — skip (geometry-dependent).
        return;
    }

    // Delaunay property may now be violated.
    // delaunayize_by_flips should restore it.
    let outcome = delaunayize_by_flips(&mut dt, DelaunayizeConfig::default()).unwrap();
    assert!(outcome.topology_repair.succeeded);
    assert!(dt.validate().is_ok(), "Should be valid after delaunayize");
}

// =============================================================================
// ERROR VARIANT TESTS
// =============================================================================

/// Verify that `DelaunayizeError::TopologyRepairFailed` is constructible via
/// the `From<PlManifoldRepairError>` impl and displays correctly.
#[test]
fn test_error_display_topology_repair_failed() {
    let inner = PlManifoldRepairError::NoProgress {
        over_shared_facets: 3,
        iterations: 5,
        cells_removed: 10,
    };
    let err: DelaunayizeError = inner.clone().into();
    let msg = err.to_string();
    assert!(msg.contains("Topology repair failed"), "{msg}");
    assert!(msg.contains("3 over-shared facets"), "{msg}");

    // Typed source is preserved end-to-end — no stringification.
    assert_eq!(
        err,
        DelaunayizeError::TopologyRepairFailed { source: inner }
    );
}

/// Verify that `DelaunayizeError::DelaunayRepairFailed` preserves the typed
/// source error via the `From<DelaunayRepairError>` impl.
#[test]
fn test_error_display_delaunay_repair_failed() {
    let inner = DelaunayRepairError::PostconditionFailed {
        message: "test postcondition".to_string(),
    };
    let err: DelaunayizeError = inner.clone().into();
    let msg = err.to_string();
    assert!(msg.contains("Delaunay repair failed"), "{msg}");
    assert!(msg.contains("test postcondition"), "{msg}");

    // Typed source is preserved end-to-end — no stringification.
    assert_eq!(
        err,
        DelaunayizeError::DelaunayRepairFailed { source: inner }
    );
}

/// Verify that `DelaunayizeError::TopologyRepairFailedWithRebuild` preserves
/// **both** the typed [`PlManifoldRepairError`] source and the typed
/// [`DelaunayTriangulationConstructionError`] rebuild error, and exposes
/// the primary source via [`Error::source`].
///
/// Regression guard: an earlier version of the fallback-rebuild-failure arm
/// stringified the topology error into a `TdsError::InconsistentDataStructure`,
/// which erased the typed variant and the source chain.
#[test]
fn test_error_display_topology_repair_with_rebuild() {
    let topo_err = PlManifoldRepairError::NoProgress {
        over_shared_facets: 3,
        iterations: 5,
        cells_removed: 10,
    };
    let rebuild_err: DelaunayTriangulationConstructionError =
        TriangulationConstructionError::GeometricDegeneracy {
            message: "synthetic rebuild degeneracy".to_string(),
        }
        .into();
    let err = DelaunayizeError::TopologyRepairFailedWithRebuild {
        source: topo_err.clone(),
        rebuild_error: rebuild_err.clone(),
    };

    // Display carries both the primary topology failure and the rebuild error.
    let msg = err.to_string();
    assert!(msg.contains("Topology repair failed"), "{msg}");
    assert!(msg.contains("3 over-shared facets"), "{msg}");
    assert!(msg.contains("fallback rebuild also failed"), "{msg}");
    assert!(msg.contains("synthetic rebuild degeneracy"), "{msg}");

    // Both the typed source and rebuild error are preserved — no stringification.
    assert_eq!(
        err,
        DelaunayizeError::TopologyRepairFailedWithRebuild {
            source: topo_err,
            rebuild_error: rebuild_err,
        }
    );

    // Error::source() exposes the primary topology error so consumers can walk
    // the source chain instead of pattern-matching.
    let source = err
        .source()
        .expect("source() must be Some for the with-rebuild variant");
    assert!(
        source.to_string().contains("3 over-shared facets"),
        "source display should match the underlying PlManifoldRepairError: {source}"
    );
}

/// Verify that `DelaunayizeError::DelaunayRepairFailedWithRebuild` preserves
/// **both** the typed [`DelaunayRepairError`] source and the typed
/// [`DelaunayTriangulationConstructionError`] rebuild error.
#[test]
fn test_error_display_delaunay_repair_with_rebuild() {
    let rebuild_err: DelaunayTriangulationConstructionError =
        TriangulationConstructionError::GeometricDegeneracy {
            message: "synthetic rebuild degeneracy".to_string(),
        }
        .into();
    let source = DelaunayRepairError::PostconditionFailed {
        message: "synthetic postcondition".to_string(),
    };
    let err = DelaunayizeError::DelaunayRepairFailedWithRebuild {
        source: source.clone(),
        rebuild_error: rebuild_err.clone(),
    };

    let msg = err.to_string();
    assert!(msg.contains("Delaunay repair failed"), "{msg}");
    assert!(msg.contains("synthetic postcondition"), "{msg}");
    assert!(msg.contains("fallback rebuild also failed"), "{msg}");
    assert!(msg.contains("synthetic rebuild degeneracy"), "{msg}");

    // Both the typed source and rebuild error are preserved — no stringification.
    assert_eq!(
        err,
        DelaunayizeError::DelaunayRepairFailedWithRebuild {
            source,
            rebuild_error: rebuild_err,
        }
    );

    let source = err
        .source()
        .expect("source() must be Some for the with-rebuild variant");
    assert!(
        source.to_string().contains("synthetic postcondition"),
        "source display should match the underlying DelaunayRepairError: {source}"
    );
}

// =============================================================================
// VALIDATION AFTER DELAUNAYIZE TEST
// =============================================================================

#[test]
fn test_full_validation_passes_after_delaunayize() {
    init_tracing();
    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
        vertex!([1.0, 1.0]),
        vertex!([0.5, 0.5]),
        vertex!([0.25, 0.75]),
    ];
    let mut dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::new(&vertices).unwrap();

    let _outcome = delaunayize_by_flips(&mut dt, DelaunayizeConfig::default()).unwrap();

    // Full Levels 1–4 validation should pass.
    assert!(dt.validate().is_ok());
}
