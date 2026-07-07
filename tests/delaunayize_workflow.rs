//! Integration tests for the delaunayize-by-flips workflow.
//!
//! Validates the public API in `delaunay::delaunayize`, covering:
//! - Public workflow behavior with explicit flip budgets and fallback config
//! - Outcome population on public success and failure paths
//! - Repeat-run determinism for outcome stats
//! - Cross-crate prelude exports and typed error payloads

use delaunay::prelude::construction::{
    DelaunayTriangulation, TriangulationConstructionError, Vertex,
};
use delaunay::prelude::delaunayize::*;
use delaunay::prelude::geometry::AdaptiveKernel;
use delaunay::prelude::pachner::{PachnerMove, PachnerMoves};
use delaunay::vertex;
use std::{error::Error, mem::size_of};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

fn init_tracing() {
    let _ = tracing_subscriber::fmt::try_init();
}

type StableDelaunay3 = DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 3>;

fn stable_3d_flip_vertices() -> Vec<Vertex<(), 3>> {
    vec![
        vertex!([0.0, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 1.0]).unwrap(),
        vertex!([0.20, 0.20, 0.20]).unwrap(),
        vertex!([0.75, 0.15, 0.30]).unwrap(),
        vertex!([0.20, 0.70, 0.35]).unwrap(),
        vertex!([0.30, 0.25, 0.80]).unwrap(),
        vertex!([0.65, 0.60, 0.55]).unwrap(),
    ]
}

fn apply_first_k2_flip(dt: &mut StableDelaunay3) -> bool {
    let mut candidate_facets = Vec::new();
    for facet in dt.facets() {
        let facet = facet.expect("facet iterator should resolve valid facets");
        if facet
            .simplex()
            .neighbor_key(usize::from(facet.facet_index()))
            .flatten()
            .is_some()
        {
            candidate_facets.push(facet.handle());
        }
    }

    for facet in candidate_facets {
        let Ok(proposal) = dt.propose_pachner(PachnerMove::K2 { facet }) else {
            continue;
        };
        if proposal.attempt_on(dt).is_ok() {
            return true;
        }
    }
    false
}

// =============================================================================
// DETERMINISM TESTS
// =============================================================================

#[test]
fn test_repeat_run_determinism_2d() {
    init_tracing();
    let vertices = vec![
        vertex!([0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
        vertex!([1.0, 1.0]).unwrap(),
        vertex!([0.5, 0.5]).unwrap(),
    ];

    let config = DelaunayizeConfig::default();

    let mut dt1: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::builder(&vertices).build().unwrap();
    let outcome1 = delaunayize_by_flips(&mut dt1, config).unwrap();

    let mut dt2: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::builder(&vertices).build().unwrap();
    let outcome2 = delaunayize_by_flips(&mut dt2, config).unwrap();

    // Stats should be identical across runs on the same input.
    assert_eq!(
        outcome1.topology_repair.simplices_removed,
        outcome2.topology_repair.simplices_removed
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

// =============================================================================
// VERTEX PRESERVATION TEST
// =============================================================================

#[test]
fn test_vertex_count_preserved_after_delaunayize() {
    init_tracing();
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 1.0]).unwrap(),
        vertex!([1.0, 1.0, 1.0]).unwrap(),
    ];
    let mut dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::builder(&vertices).build().unwrap();
    let vertex_count_before = dt.number_of_vertices();

    let _outcome = delaunayize_by_flips(&mut dt, DelaunayizeConfig::default()).unwrap();

    // Topology repair only removes simplices, not vertices. Delaunay flip repair
    // also preserves vertex count. So the vertex count should be unchanged.
    assert_eq!(dt.number_of_vertices(), vertex_count_before);
}

// =============================================================================
// NON-DELAUNAY REPAIR VIA FLIPS TEST
// =============================================================================

/// Build a valid Delaunay triangulation, apply a k=2 Pachner move to
/// intentionally break the Delaunay property, then verify
/// `delaunayize_by_flips` restores it.
#[test]
fn test_flip_breaks_delaunay_then_delaunayize_restores() {
    init_tracing();
    let vertices = stable_3d_flip_vertices();
    let mut dt: StableDelaunay3 = DelaunayTriangulation::builder(&vertices).build().unwrap();
    assert!(dt.validate().is_ok(), "Should start valid");

    assert!(
        apply_first_k2_flip(&mut dt),
        "3D delaunayize fixture should provide an accepted k=2 Pachner move"
    );

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
        simplices_removed: 10,
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
        reason: Box::new(DelaunayRepairPostconditionFailure::Disconnected { simplex_count: 1 }),
    };
    let err: DelaunayizeError = inner.clone().into();
    let msg = err.to_string();
    assert!(msg.contains("Delaunay repair failed"), "{msg}");
    assert!(msg.contains("disconnected the triangulation"), "{msg}");

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
        simplices_removed: 10,
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
        reason: Box::new(DelaunayRepairPostconditionFailure::Disconnected { simplex_count: 1 }),
    };
    let err = DelaunayizeError::DelaunayRepairFailedWithRebuild {
        source: source.clone(),
        rebuild_error: rebuild_err.clone(),
    };

    let msg = err.to_string();
    assert!(msg.contains("Delaunay repair failed"), "{msg}");
    assert!(msg.contains("disconnected the triangulation"), "{msg}");
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
        source
            .to_string()
            .contains("disconnected the triangulation"),
        "source display should match the underlying DelaunayRepairError: {source}"
    );
}

/// Verify that the focused delaunayize prelude is sufficient for naming the
/// workflow's public typed error payloads.
#[test]
fn test_prelude_exports_error_payloads() {
    const _: usize = size_of::<DelaunayRepairError>();
    const _: usize = size_of::<DelaunayRepairStats>();
    const _: usize = size_of::<PlManifoldRepairError>();
    const _: usize = size_of::<PlManifoldRepairStats<(), (), 2>>();
    const _: usize = size_of::<SimplexValidationError>();
    const _: usize = size_of::<DelaunayTriangulationConstructionError>();
}

// =============================================================================
// EXPLICIT FLIP BUDGET TESTS
// =============================================================================

/// Verify that `delaunayize_by_flips` works with an explicit `delaunay_max_flips`
/// budget, which routes through `repair_delaunay_with_flips_advanced` instead
/// of `repair_delaunay_with_flips`.
#[test]
fn test_delaunayize_with_explicit_flip_budget_3d() {
    init_tracing();
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 1.0]).unwrap(),
        vertex!([0.5, 0.5, 0.5]).unwrap(),
    ];
    let mut dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::builder(&vertices).build().unwrap();

    let config = DelaunayizeConfig::default().with_delaunay_max_flips(1000);
    let outcome = delaunayize_by_flips(&mut dt, config).unwrap();
    assert!(outcome.topology_repair.succeeded);
    assert!(!outcome.used_fallback_rebuild);
    assert!(dt.validate().is_ok());
}

/// Verify that `delaunayize_by_flips` handles both `delaunay_max_flips` and
/// `fallback_rebuild` together on valid input.
#[test]
fn test_delaunayize_with_flip_budget_and_fallback_2d() {
    init_tracing();
    let vertices = vec![
        vertex!([0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
        vertex!([1.0, 1.0]).unwrap(),
        vertex!([0.5, 0.5]).unwrap(),
    ];
    let mut dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::builder(&vertices).build().unwrap();

    let config = DelaunayizeConfig::default()
        .with_delaunay_max_flips(500)
        .with_fallback_rebuild(true);
    let outcome = delaunayize_by_flips(&mut dt, config).unwrap();
    assert!(outcome.topology_repair.succeeded);
    // Already valid — fallback should not be triggered.
    assert!(!outcome.used_fallback_rebuild);
    assert!(dt.validate().is_ok());
}

/// Apply a k=2 Pachner move to break the Delaunay property, then verify
/// `delaunayize_by_flips` with an explicit flip budget restores it.
#[test]
fn test_flip_breaks_then_delaunayize_with_budget_restores_3d() {
    init_tracing();
    let vertices = stable_3d_flip_vertices();
    let mut dt: StableDelaunay3 = DelaunayTriangulation::builder(&vertices).build().unwrap();
    assert!(dt.validate().is_ok());

    assert!(
        apply_first_k2_flip(&mut dt),
        "3D delaunayize budget fixture should provide an accepted k=2 Pachner move"
    );

    let config = DelaunayizeConfig::default().with_delaunay_max_flips(1000);
    let outcome = delaunayize_by_flips(&mut dt, config).unwrap();
    assert!(outcome.topology_repair.succeeded);
    assert!(dt.validate().is_ok());
}

// =============================================================================
// VALIDATION AFTER DELAUNAYIZE TEST
// =============================================================================

#[test]
fn test_full_validation_passes_after_delaunayize() {
    init_tracing();
    let vertices = vec![
        vertex!([0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
        vertex!([1.0, 1.0]).unwrap(),
        vertex!([0.5, 0.5]).unwrap(),
        vertex!([0.25, 0.75]).unwrap(),
    ];
    let mut dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::builder(&vertices).build().unwrap();

    let _outcome = delaunayize_by_flips(&mut dt, DelaunayizeConfig::default()).unwrap();

    // Full Levels 1–4 validation should pass.
    assert!(dt.validate().is_ok());
}
