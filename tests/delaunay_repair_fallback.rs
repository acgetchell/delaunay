//! Integration tests for Delaunay repair fallback behavior.
//!
//! This module tests that when flip-based repair fails to converge or leaves
//! Delaunay violations, the deterministic rebuild heuristic is triggered and
//! successfully produces a valid Delaunay triangulation.

use delaunay::prelude::construction::{DelaunayTriangulation, TopologyGuarantee};
use delaunay::prelude::delaunayize::DelaunayRefinementBuilder;
use delaunay::prelude::pachner::{PachnerMove, PachnerMoves};
use delaunay::vertex;

#[cfg(feature = "diagnostics")]
fn init_tracing() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_test_writer()
            .try_init();
    });
}

#[cfg(not(feature = "diagnostics"))]
const fn init_tracing() {}

macro_rules! test_debug_info {
    ($($arg:tt)*) => {{
        #[cfg(feature = "diagnostics")]
        {
            init_tracing();
            tracing::info!($($arg)*);
        }
        #[cfg(not(feature = "diagnostics"))]
        {
            let _ = format_args!($($arg)*);
        }
    }};
}

/// Test that the public advanced repair API exercises heuristic rebuild fallback.
#[test]
fn repair_fallback_produces_valid_triangulation() {
    init_tracing();
    let vertices = vec![
        vertex!([0.0, 0.0]).unwrap(),
        vertex!([4.0, 0.0]).unwrap(),
        vertex!([4.0, 2.0]).unwrap(),
        vertex!([1.0, 2.0]).unwrap(),
    ];
    let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::builder(&vertices)
        .topology_guarantee(TopologyGuarantee::PLManifold)
        .build()
        .expect("fixture construction should succeed");
    let mut tri = dt.into_triangulation();

    let mut candidate_facets = Vec::new();
    for facet in tri.facets() {
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

    let mut flipped = false;
    for facet in candidate_facets {
        let mut trial = tri.clone();
        let Ok(proposal) = trial.propose_pachner(PachnerMove::K2 { facet }) else {
            continue;
        };
        if proposal.attempt_on(&mut trial).is_ok()
            && trial
                .delaunay_violation_report(None)
                .is_ok_and(|report| !report.is_valid())
        {
            tri = trial;
            flipped = true;
            break;
        }
    }
    assert!(
        flipped,
        "fixture should contain a realized flip that violates Level 5"
    );

    let converted = DelaunayRefinementBuilder::new(tri)
        .repair_by_flips()
        .max_flips(0)
        .fallback_rebuild(true)
        .build()
        .expect("heuristic rebuild fallback should repair the non-Delaunay fixture");
    test_debug_info!("bounded conversion outcome: {:?}", converted.outcome);
    assert!(
        converted.outcome.used_fallback_rebuild,
        "zero flip budget should force heuristic rebuild fallback"
    );

    let dt = converted.triangulation;
    dt.validate()
        .expect("Triangulation should be fully valid after heuristic fallback");
    assert_eq!(dt.number_of_vertices(), vertices.len());
    assert_eq!(dt.dim(), 2, "Should be a full 2D triangulation");
    assert!(
        dt.number_of_simplices() > 0,
        "Should have at least one simplex"
    );
}
