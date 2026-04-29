//! Public prelude smoke tests.
//!
//! These tests intentionally use focused preludes instead of module-internal
//! paths so doctests, integration tests, examples, and benchmarks have a small
//! import contract to copy from.

use delaunay::prelude::generators::generate_random_points_seeded;
use delaunay::prelude::geometry::{AdaptiveKernel, Point};
use delaunay::prelude::query::ConvexHull;
use delaunay::prelude::triangulation::delaunayize::{
    DelaunayizeConfig, DelaunayizeError, DelaunayizeOutcome, delaunayize_by_flips,
};
use delaunay::prelude::triangulation::flips::{BistellarFlips, TopologyGuarantee};
use delaunay::prelude::triangulation::repair::{
    DelaunayCheckPolicy, DelaunayRepairDiagnostics, DelaunayRepairError, DelaunayRepairOutcome,
    DelaunayRepairPolicy, DelaunayRepairStats, FlipError, RepairQueueOrder,
    verify_delaunay_for_triangulation,
};
use delaunay::prelude::triangulation::{
    ConstructionOptions, DelaunayRepairOperation, DelaunayTriangulation,
    DelaunayTriangulationValidationError, InsertionOrderStrategy, Vertex,
};
use delaunay::vertex;

const fn assert_bistellar_flips(_: &impl BistellarFlips<AdaptiveKernel<f64>, (), (), 3>) {}

#[test]
fn preludes_cover_bench_apis() {
    let _generated_points: Vec<Point<f64, 2>> =
        generate_random_points_seeded(3, (0.0, 1.0), 42).unwrap();

    let vertices: Vec<Vertex<f64, (), 3>> = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];
    let options =
        ConstructionOptions::default().with_insertion_order(InsertionOrderStrategy::Input);
    let dt = DelaunayTriangulation::new_with_options(&vertices, options).unwrap();

    assert_eq!(dt.topology_guarantee(), TopologyGuarantee::PLManifold);
    assert!(dt.boundary_facets().count() > 0);
    assert!(ConvexHull::from_triangulation(dt.as_triangulation()).is_ok());
    assert!(dt.validate().is_ok());
    assert_bistellar_flips(&dt);
}

#[test]
fn diagnostic_preludes_cover_repair_apis() {
    let vertices: Vec<Vertex<f64, (), 3>> = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];
    let mut dt = DelaunayTriangulation::new(&vertices).unwrap();

    let repair_stats = DelaunayRepairStats::default();
    let repair_outcome = DelaunayRepairOutcome {
        stats: repair_stats,
        heuristic: None,
    };
    assert!(!repair_outcome.used_heuristic());
    assert_eq!(
        DelaunayRepairPolicy::default(),
        DelaunayRepairPolicy::EveryInsertion
    );
    assert!(!DelaunayCheckPolicy::default().should_check(1));
    assert_eq!(RepairQueueOrder::Fifo, RepairQueueOrder::Fifo);
    let diagnostics = DelaunayRepairDiagnostics {
        facets_checked: 0,
        flips_performed: 0,
        max_queue_len: 0,
        ambiguous_predicates: 0,
        ambiguous_predicate_samples: Vec::new(),
        predicate_failures: 0,
        cycle_detections: 0,
        cycle_signature_samples: Vec::new(),
        attempt: 1,
        queue_order: RepairQueueOrder::Fifo,
    };
    assert!(diagnostics.to_string().contains("checked"));
    assert!(matches!(
        DelaunayRepairError::Flip(FlipError::DegenerateCell),
        DelaunayRepairError::Flip(_)
    ));
    let validation_error = DelaunayTriangulationValidationError::RepairOperationFailed {
        operation: DelaunayRepairOperation::VertexRemoval,
        source: DelaunayRepairError::Flip(FlipError::DegenerateCell),
    };
    assert!(validation_error.to_string().contains("vertex removal"));

    verify_delaunay_for_triangulation(dt.as_triangulation()).unwrap();

    let outcome = delaunayize_by_flips(&mut dt, DelaunayizeConfig::default()).unwrap();
    assert!(!outcome.used_fallback_rebuild);
    let _typed_outcome: DelaunayizeOutcome<f64, (), (), 3> = outcome;
    let _typed_error: Option<DelaunayizeError> = None;
}
