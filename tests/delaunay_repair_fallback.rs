//! Integration tests for Delaunay repair fallback behavior.
//!
//! This module tests that when flip-based repair fails to converge or leaves
//! Delaunay violations, the deterministic rebuild heuristic is triggered and
//! successfully produces a valid Delaunay triangulation.

use delaunay::prelude::triangulation::construction::{
    DelaunayRepairPolicy, DelaunayTriangulation, TopologyGuarantee, vertex,
};
use delaunay::prelude::triangulation::flips::BistellarFlips;
use delaunay::prelude::triangulation::repair::DelaunayRepairHeuristicConfig;
use delaunay::triangulation::flips::FacetHandle;

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
        vertex!([0.0, 0.0]),
        vertex!([4.0, 0.0]),
        vertex!([4.0, 2.0]),
        vertex!([1.0, 2.0]),
    ];
    let mut dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .expect("fixture construction should succeed");

    let mut candidate_facets = Vec::new();
    for (cell_key, cell) in dt.cells() {
        if let Some(neighbors) = cell.neighbors() {
            for (index, neighbor) in neighbors.iter().enumerate() {
                if neighbor.is_some() {
                    let facet_index = u8::try_from(index).expect("2D facet index fits in u8");
                    candidate_facets.push(FacetHandle::new(cell_key, facet_index));
                }
            }
        }
    }

    let flipped = candidate_facets
        .into_iter()
        .any(|facet| dt.flip_k2(facet).is_ok());
    assert!(flipped, "fixture should contain a flippable interior facet");

    let mut config = DelaunayRepairHeuristicConfig::default();
    config.max_flips = Some(0);
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
    assert!(dt.number_of_cells() > 0, "Should have at least one cell");
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
        vertex!([0.0, 0.0, 0.0]),
        vertex!([2.0, 0.0, 0.0]),
        vertex!([1.0, 2.0, 0.0]),
        vertex!([1.0, 0.5, 1.5]),
        vertex!([1.0, 0.5, 0.5]), // Interior point
        vertex!([0.8, 0.8, 0.8]), // Another interior point
        vertex!([1.2, 0.6, 0.7]), // Close to existing
    ];

    for (i, vertex) in test_vertices.into_iter().enumerate() {
        let result = dt.insert(vertex);

        match result {
            Ok(_) => {
                // Skip validation during bootstrap (vertices exist but no cells yet)
                // Level 3 topology validation is not meaningful until cells exist
                if dt.number_of_cells() > 0 {
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
        vertex!([0.0, 0.0]),
        vertex!([4.0, 0.0]),
        vertex!([2.0, 3.5]), // Non-collinear with first two
        vertex!([0.5, 2.0]),
        vertex!([3.5, 2.0]),
        vertex!([1.5, 1.0]),
        vertex!([2.5, 2.5]),
        vertex!([1.0, 3.0]),
    ];

    let dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
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
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
        vertex!([0.5, 0.5, 0.5]),
    ];

    let mut dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
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
    dt.is_valid()
        .expect("Should satisfy Delaunay property after repair");
}

/// Test that repair policy can be configured and fallback still works.
#[test]
fn repair_policy_configuration_with_fallback() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
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
            dt.insert(*vertex)
                .expect("Insertion should succeed with any repair policy");
        }

        dt.validate()
            .unwrap_or_else(|e| panic!("Triangulation invalid with policy {policy:?}: {e}"));
    }
}
