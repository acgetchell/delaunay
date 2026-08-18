//! Integration tests for the consuming `delaunayize` workflow.
//!
//! Validates the public API in `delaunay::delaunayize`, covering:
//! - Public workflow behavior with explicit flip budgets and fallback config
//! - Outcome population on public success and failure paths
//! - Repeat-run determinism for outcome stats
//! - Cross-crate prelude exports and typed error payloads

use delaunay::prelude::construction::{
    DelaunayTriangulation, TopologyGuarantee, TriangulationConstructionError, Vertex,
};
use delaunay::prelude::delaunayize::*;
use delaunay::prelude::geometry::AdaptiveKernel;
use delaunay::prelude::pachner::{PachnerMove, PachnerMoves};
use delaunay::prelude::triangulation::Triangulation;
use delaunay::vertex;
use std::{error::Error, mem::size_of};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

fn init_tracing() {
    let _ = tracing_subscriber::fmt::try_init();
}

type StableTriangulation3 = Triangulation<AdaptiveKernel<f64>, (), (), 3>;

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

fn apply_first_delaunay_breaking_k2_flip(dt: &mut StableTriangulation3) -> bool {
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
        let mut trial = dt.clone();
        let Ok(proposal) = trial.propose_pachner(PachnerMove::K2 { facet }) else {
            continue;
        };
        if proposal.attempt_on(&mut trial).is_ok()
            && trial
                .delaunay_violation_report(None)
                .is_ok_and(|report| !report.is_valid())
        {
            *dt = trial;
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

    let dt1 = DelaunayTriangulation::builder(&vertices).build().unwrap();
    let outcome1 = delaunayize(dt1.into_triangulation(), config)
        .unwrap()
        .outcome;

    let dt2 = DelaunayTriangulation::builder(&vertices).build().unwrap();
    let outcome2 = delaunayize(dt2.into_triangulation(), config)
        .unwrap()
        .outcome;

    // Stats should be identical across runs on the same input.
    assert_eq!(
        outcome1.delaunay_repair.facets_checked,
        outcome2.delaunay_repair.facets_checked
    );
    assert_eq!(
        outcome1.delaunay_repair.flips_performed,
        outcome2.delaunay_repair.flips_performed
    );
    assert_eq!(
        outcome1.delaunay_repair.max_queue_len,
        outcome2.delaunay_repair.max_queue_len
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
    let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
    let vertex_count_before = dt.number_of_vertices();
    let mut vertex_uuids_before = dt
        .vertices()
        .map(|(_, vertex)| vertex.uuid())
        .collect::<Vec<_>>();
    vertex_uuids_before.sort_unstable();

    let converted = delaunayize(dt.into_triangulation(), DelaunayizeConfig::default()).unwrap();
    let mut vertex_uuids_after = converted
        .triangulation
        .vertices()
        .map(|(_, vertex)| vertex.uuid())
        .collect::<Vec<_>>();
    vertex_uuids_after.sort_unstable();

    // Delaunay flips preserve the vertex set.
    assert_eq!(
        converted.triangulation.number_of_vertices(),
        vertex_count_before
    );
    assert_eq!(vertex_uuids_after, vertex_uuids_before);
}

// =============================================================================
// NON-DELAUNAY REPAIR VIA FLIPS TEST
// =============================================================================

/// Build a valid Delaunay triangulation, apply a k=2 Pachner move to
/// intentionally break the Delaunay property, then verify
/// `delaunayize` restores it.
#[test]
fn test_flip_breaks_delaunay_then_delaunayize_restores() {
    init_tracing();
    let vertices = stable_3d_flip_vertices();
    let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
    assert!(dt.validate().is_ok(), "Should start valid");
    let mut tri = dt.into_triangulation();

    assert!(
        apply_first_delaunay_breaking_k2_flip(&mut tri),
        "3D delaunayize fixture should provide a k=2 move with a proven Level 5 violation"
    );
    assert!(
        tri.delaunay_violation_report(None)
            .is_ok_and(|report| !report.is_valid()),
        "fixture must be non-Delaunay before conversion"
    );

    // Delaunay property may now be violated.
    // `delaunayize` should restore it.
    let converted = delaunayize(tri, DelaunayizeConfig::default()).unwrap();
    assert!(
        converted.triangulation.validate().is_ok(),
        "Should be valid after delaunayize"
    );
}

// =============================================================================
// ERROR VARIANT TESTS
// =============================================================================

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

/// Fallback reconstruction must not bypass the PL-manifold proof required by
/// the flip conversion workflow.
#[test]
fn fallback_does_not_bypass_flip_topology_precondition() {
    let vertices = [
        vertex!([0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
    ];
    let triangulation = DelaunayTriangulation::builder(&vertices)
        .topology_guarantee(TopologyGuarantee::Pseudomanifold)
        .build()
        .unwrap()
        .into_triangulation();

    let error = delaunayize(
        triangulation,
        DelaunayizeConfig::default().with_fallback_rebuild(true),
    )
    .expect_err("a weaker topology guarantee must not enter flip repair or fallback");

    std::assert_matches!(
        error,
        DelaunayizeError::FlipTopologyNotAdmissible {
            found: TopologyGuarantee::Pseudomanifold,
            ..
        }
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
    const _: usize = size_of::<SimplexValidationError>();
    const _: usize = size_of::<DelaunayTriangulationConstructionError>();
}

// =============================================================================
// EXPLICIT FLIP BUDGET TESTS
// =============================================================================

/// Verify that `delaunayize` works with an explicit `delaunay_max_flips`
/// budget, which is forwarded to the low-level flip repair engine.
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
    let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();

    let config = DelaunayizeConfig::default().with_delaunay_max_flips(1000);
    let converted = delaunayize(dt.into_triangulation(), config).unwrap();
    assert!(!converted.outcome.used_fallback_rebuild);
    assert!(converted.triangulation.validate().is_ok());
}

/// Verify that `delaunayize` handles both `delaunay_max_flips` and
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
    let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();

    let config = DelaunayizeConfig::default()
        .with_delaunay_max_flips(500)
        .with_fallback_rebuild(true);
    let converted = delaunayize(dt.into_triangulation(), config).unwrap();
    // Already valid — fallback should not be triggered.
    assert!(!converted.outcome.used_fallback_rebuild);
    assert!(converted.triangulation.validate().is_ok());
}

/// Apply a k=2 Pachner move to break the Delaunay property, then verify
/// `delaunayize` with an explicit flip budget restores it.
#[test]
fn test_flip_breaks_then_delaunayize_with_budget_restores_3d() {
    init_tracing();
    let vertices = stable_3d_flip_vertices();
    let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
    assert!(dt.validate().is_ok());
    let mut tri = dt.into_triangulation();

    assert!(
        apply_first_delaunay_breaking_k2_flip(&mut tri),
        "3D delaunayize budget fixture should provide a k=2 move with a proven Level 5 violation"
    );
    assert!(
        tri.delaunay_violation_report(None)
            .is_ok_and(|report| !report.is_valid()),
        "fixture must be non-Delaunay before conversion"
    );

    let config = DelaunayizeConfig::default().with_delaunay_max_flips(1000);
    let converted = delaunayize(tri, config).unwrap();
    assert!(converted.triangulation.validate().is_ok());
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
    let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();

    let converted = delaunayize(dt.into_triangulation(), DelaunayizeConfig::default()).unwrap();

    // Full Levels 1–4 validation should pass.
    assert!(converted.triangulation.validate().is_ok());
}
