//! Integration tests for Delaunay repair fallback behavior.
//!
//! This module tests that when flip-based repair fails to converge or leaves
//! Delaunay violations, the deterministic rebuild heuristic is triggered and
//! successfully produces a valid Delaunay triangulation.

use delaunay::prelude::construction::{
    DelaunayRepairPolicy, DelaunayTriangulation, TopologyGuarantee,
};
use delaunay::prelude::pachner::{PachnerMove, PachnerMoves};
use delaunay::prelude::repair::DelaunayRepairHeuristicConfig;
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
    let mut dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::builder(&vertices)
        .topology_guarantee(TopologyGuarantee::PLManifold)
        .build()
        .expect("fixture construction should succeed");

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

    let mut flipped = false;
    for facet in candidate_facets {
        let Ok(proposal) = dt.propose_pachner(PachnerMove::K2 { facet }) else {
            continue;
        };
        if proposal.attempt_on(&mut dt).is_ok() {
            flipped = true;
            break;
        }
    }
    assert!(flipped, "fixture should contain a flippable interior facet");

    let config = DelaunayRepairHeuristicConfig::default().with_delaunay_max_flips(0);
    let outcome = dt
        .repair_delaunay_with_flips_advanced(config)
        .expect("heuristic rebuild fallback should repair the non-Delaunay fixture");
    assert!(
        outcome.used_heuristic(),
        "zero flip budget should force heuristic rebuild fallback"
    );

    dt.validate()
        .expect("Triangulation should be fully valid after heuristic fallback");
    assert_eq!(dt.number_of_vertices(), vertices.len());
    assert_eq!(dt.dim(), 2, "Should be a full 2D triangulation");
    assert!(
        dt.number_of_simplices() > 0,
        "Should have at least one simplex"
    );
}

/// Test incremental insertion with repair fallback.
///
/// Verifies that even when inserting vertices one-by-one triggers repair failures,
/// the fallback mechanism maintains validity throughout.
#[test]
fn incremental_insertion_with_repair_fallback() {
    init_tracing();
    let mut dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::empty_with_topology_guarantee(TopologyGuarantee::PLManifold);

    // Default repair policy should be enabled
    assert_ne!(
        dt.delaunay_repair_policy(),
        DelaunayRepairPolicy::Never,
        "Repair should be enabled by default"
    );

    // Insert vertices that might trigger repair challenges
    let test_vertices = vec![
        vertex!([0.0, 0.0, 0.0]).unwrap(),
        vertex!([2.0, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 2.0, 0.0]).unwrap(),
        vertex!([1.0, 0.5, 1.5]).unwrap(),
        vertex!([1.0, 0.5, 0.5]).unwrap(), // Interior point
        vertex!([0.8, 0.8, 0.8]).unwrap(), // Another interior point
        vertex!([1.2, 0.6, 0.7]).unwrap(), // Close to existing
    ];

    for (i, vertex) in test_vertices.into_iter().enumerate() {
        let result = dt.insert_vertex(vertex);

        match result {
            Ok(_) => {
                // Skip validation during bootstrap (vertices exist but no simplices yet)
                // Level 3 topology validation is not meaningful until simplices exist
                if dt.number_of_simplices() > 0 {
                    dt.validate().unwrap_or_else(|e| {
                        panic!("Triangulation invalid after insertion {}: {}", i + 1, e)
                    });
                }
            }
            Err(e) => {
                // Some insertions may be skipped (duplicates, degeneracies), which is fine
                test_debug_info!("Vertex {} skipped: {}", i + 1, e);
            }
        }
    }

    // Final validation
    dt.validate()
        .expect("Final triangulation should be valid after all insertions");

    assert!(
        dt.number_of_vertices() >= 5,
        "Should have inserted most vertices"
    );
}

/// Test that repair fallback works in 2D as well.
#[test]
fn repair_fallback_2d() {
    // Use non-collinear points to avoid degeneracy
    let vertices = vec![
        vertex!([0.0, 0.0]).unwrap(),
        vertex!([4.0, 0.0]).unwrap(),
        vertex!([2.0, 3.5]).unwrap(), // Non-collinear with first two
        vertex!([0.5, 2.0]).unwrap(),
        vertex!([3.5, 2.0]).unwrap(),
        vertex!([1.5, 1.0]).unwrap(),
        vertex!([2.5, 2.5]).unwrap(),
        vertex!([1.0, 3.0]).unwrap(),
    ];

    let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::builder(&vertices)
        .topology_guarantee(TopologyGuarantee::PLManifold)
        .build()
        .expect("2D construction should succeed with fallback if needed");

    dt.validate()
        .expect("2D triangulation should be valid after construction");

    assert_eq!(dt.number_of_vertices(), vertices.len());
    assert_eq!(dt.dim(), 2);
}

/// Test that explicit repair call works and validates properly.
#[test]
fn explicit_repair_call_validates_result() {
    init_tracing();
    // Build a triangulation
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 1.0]).unwrap(),
        vertex!([0.5, 0.5, 0.5]).unwrap(),
    ];

    let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::builder(&vertices)
        .topology_guarantee(TopologyGuarantee::PLManifold)
        .build()
        .expect("Construction should succeed");

    // Call explicit repair (should be a no-op if already valid, or fix any issues)
    let stats = dt
        .repair_delaunay_with_flips()
        .expect("Explicit repair should succeed");

    #[cfg(feature = "diagnostics")]
    test_debug_info!("Explicit repair stats: {stats:?}");
    #[cfg(not(feature = "diagnostics"))]
    let _ = &stats;

    // Verify triangulation is valid after explicit repair
    dt.validate()
        .expect("Triangulation should be valid after explicit repair");

    // Verify Delaunay property specifically
    dt.is_valid_delaunay()
        .expect("Should satisfy Delaunay property after repair");
}

/// Test that repair policy can be configured and fallback still works.
#[test]
fn repair_policy_configuration_with_fallback() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 1.0]).unwrap(),
    ];

    // Test with different repair policies
    for policy in [
        DelaunayRepairPolicy::EveryInsertion,
        DelaunayRepairPolicy::EveryN(std::num::NonZeroUsize::new(2).unwrap()),
    ] {
        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::empty_with_topology_guarantee(TopologyGuarantee::PLManifold);
        dt.set_delaunay_repair_policy(policy);

        for vertex in &vertices {
            dt.insert_vertex(*vertex)
                .expect("Insertion should succeed with any repair policy");
        }

        dt.validate()
            .unwrap_or_else(|e| panic!("Triangulation invalid with policy {policy:?}: {e}"));
    }
}
