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

    let dt1 = DelaunayTriangulation::builder(&vertices).build().unwrap();
    let outcome1 = DelaunayRefinementBuilder::new(dt1.into_triangulation())
        .repair_by_flips()
        .build()
        .unwrap()
        .outcome;

    let dt2 = DelaunayTriangulation::builder(&vertices).build().unwrap();
    let outcome2 = DelaunayRefinementBuilder::new(dt2.into_triangulation())
        .repair_by_flips()
        .build()
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

    let converted = DelaunayRefinementBuilder::new(dt.into_triangulation())
        .repair_by_flips()
        .build()
        .unwrap();
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
    let converted = DelaunayRefinementBuilder::new(tri)
        .repair_by_flips()
        .build()
        .unwrap();
    assert!(
        converted.outcome.delaunay_repair.flips_performed > 0,
        "known violation should require at least one repair flip"
    );
    assert!(!converted.outcome.used_fallback_rebuild);
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

    let error = DelaunayRefinementBuilder::new(triangulation)
        .repair_by_flips()
        .fallback_rebuild(true)
        .build()
        .expect_err("a weaker topology guarantee must not enter flip repair or fallback");

    let (triangulation, reason) = error.into_parts();
    std::assert_matches!(
        reason,
        DelaunayizeError::FlipTopologyNotAdmissible {
            found: TopologyGuarantee::Pseudomanifold,
            ..
        }
    );
    triangulation
        .validate_realization()
        .expect("rejected repair precondition must return the valid Levels 1-4 owner");
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
// EXPLICIT FLIP BUDGET TEST
// =============================================================================

/// A zero flip budget must reach the repair engine, reject a known violation,
/// and return the unchanged Levels 1–4 owner.
#[test]
fn zero_flip_budget_is_forwarded_and_rolls_back() {
    init_tracing();
    let vertices = stable_3d_flip_vertices();
    let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
    let mut triangulation = dt.into_triangulation();

    assert!(
        apply_first_delaunay_breaking_k2_flip(&mut triangulation),
        "3D budget fixture should provide a k=2 move with a proven Level 5 violation"
    );
    let vertex_count_before = triangulation.number_of_vertices();
    let simplex_count_before = triangulation.number_of_simplices();
    let generation_before = triangulation.topology_generation();

    let failure = DelaunayRefinementBuilder::new(triangulation)
        .repair_by_flips()
        .max_flips(0)
        .build()
        .expect_err("zero budget must reject a non-Delaunay triangulation");
    let (triangulation, reason) = failure.into_parts();

    std::assert_matches!(
        reason,
        DelaunayizeError::DelaunayRepairFailed {
            source: DelaunayRepairError::NonConvergent {
                max_flips: 0,
                diagnostics,
            },
        } if diagnostics.flips_performed == 0
    );
    assert_eq!(triangulation.number_of_vertices(), vertex_count_before);
    assert_eq!(triangulation.number_of_simplices(), simplex_count_before);
    assert_eq!(triangulation.topology_generation(), generation_before);
    assert!(
        triangulation
            .delaunay_violation_report(None)
            .is_ok_and(|report| !report.is_valid()),
        "failed repair must return the original non-Delaunay owner"
    );
    triangulation
        .validate_realization()
        .expect("rollback must preserve the Levels 1–4 proof");
}
