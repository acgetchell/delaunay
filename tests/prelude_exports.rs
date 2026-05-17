//! Public prelude smoke tests.
//!
//! These tests intentionally use focused preludes instead of module-internal
//! paths so doctests, integration tests, examples, and benchmarks have a small
//! import contract to copy from.

#![forbid(unsafe_code)]
#![expect(
    clippy::result_large_err,
    reason = "tests preserve typed construction, repair, and delaunayize errors"
)]

use delaunay::prelude::DelaunayValidationError;
use delaunay::prelude::algorithms::LocateResult;
#[cfg(feature = "diagnostics")]
use delaunay::prelude::collections::SimplexKeyBuffer;
use delaunay::prelude::collections::{
    SecureHashMap as ScopedSecureHashMap, SecureHashSet as ScopedSecureHashSet,
};
#[cfg(feature = "diagnostics")]
use delaunay::prelude::diagnostics::{
    DelaunayViolationDetail, DelaunayViolationReport, NeighborSlot as DiagnosticNeighborSlot,
    debug_print_first_delaunay_violation, delaunay_violation_report,
    verify_conflict_region_completeness,
};
use delaunay::prelude::generators::{RandomPointGenerationError, generate_random_points_seeded};
#[cfg(feature = "diagnostics")]
use delaunay::prelude::geometry::Coordinate;
use delaunay::prelude::geometry::{
    AdaptiveKernel, CoordinateConversionError, DegenerateSimplexReason, MatrixError, Point,
};
use delaunay::prelude::ordering::{
    HilbertError, hilbert_index, hilbert_indices_prequantized, hilbert_quantize,
    hilbert_sort_by_stable, hilbert_sort_by_unstable, hilbert_sorted_indices,
};
use delaunay::prelude::query::ConvexHull;
#[cfg(feature = "diagnostics")]
use delaunay::prelude::tds::Tds;
use delaunay::prelude::tds::{InvariantErrorSummaryDetail, NeighborSlot, TdsErrorKind};
use delaunay::prelude::triangulation::construction::{
    CavityFillingError, CavityRepairStage, ConstructionOptions, ConstructionSkipSample,
    ConstructionSlowInsertionSample, DelaunayConstructionFailure, DelaunayRepairPolicy,
    DelaunayTriangulation, DelaunayTriangulationConstructionError, ExplicitConstructionError,
    ExplicitDelaunayValidationError, ExplicitDelaunayValidationErrorKind,
    ExplicitDelaunayValidationSourceKind, ExplicitInsertionError, ExplicitInsertionErrorKind,
    ExplicitInvariantError, ExplicitInvariantErrorKind, ExplicitTdsError, ExplicitTdsErrorKind,
    InsertionOrderStrategy, SimplexValidationError, TopologyGuarantee, Vertex, vertex,
};
use delaunay::prelude::triangulation::delaunayize::{
    DelaunayizeConfig, DelaunayizeError, DelaunayizeOutcome, delaunayize_by_flips,
};
use delaunay::prelude::triangulation::diagnostics::ConstructionTelemetry;
use delaunay::prelude::triangulation::flips::BistellarFlips;
use delaunay::prelude::triangulation::insertion::{
    InsertionError, NeighborRebuildError, Tds as InsertionTds, TdsMutationError,
    repair_neighbor_pointers_local,
};
use delaunay::prelude::triangulation::repair::{
    DelaunayCheckPolicy, DelaunayRepairDiagnostics, DelaunayRepairError, DelaunayRepairOperation,
    DelaunayRepairOutcome, DelaunayRepairStats, DelaunayRepairVerificationContext,
    DelaunayTriangulationValidationError, FlipEdgeAdjacencyError, FlipError,
    FlipTriangleAdjacencyError, FlipVertexAdjacencyError, RepairQueueOrder,
    verify_delaunay_for_triangulation,
};
use delaunay::prelude::triangulation::validation::ValidationCadence;
use delaunay::prelude::{SecureHashMap, SecureHashSet};

#[derive(Debug, thiserror::Error)]
enum PreludeExportTestError {
    #[error(transparent)]
    RandomPointGeneration(#[from] RandomPointGenerationError),
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    DelaunayValidation(#[from] DelaunayValidationError),
    #[error(transparent)]
    DelaunayRepair(#[from] DelaunayRepairError),
    #[error(transparent)]
    Delaunayize(#[from] DelaunayizeError),
    #[error(transparent)]
    Insertion(#[from] InsertionError),
}

/// Proves the focused flips prelude exports the trait bound expected by benchmarks.
const fn assert_bistellar_flips(_: &impl BistellarFlips<AdaptiveKernel<f64>, (), 3>) {}

const fn assert_send_sync_unpin<T: Send + Sync + Unpin>() {}

#[test]
fn preludes_cover_bench_apis() -> Result<(), PreludeExportTestError> {
    let _generated_points: Vec<Point<f64, 2>> = generate_random_points_seeded(3, (0.0, 1.0), 42)?;

    let vertices: Vec<Vertex<f64, (), 3>> = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];
    let options =
        ConstructionOptions::default().with_insertion_order(InsertionOrderStrategy::Input);
    assert!(matches!(
        options.batch_repair_policy(),
        DelaunayRepairPolicy::EveryInsertion
    ));
    let dt = DelaunayTriangulation::new_with_options(&vertices, options)?;

    assert_eq!(dt.topology_guarantee(), TopologyGuarantee::PLManifold);
    assert!(dt.boundary_facets().count() > 0);
    assert!(ConvexHull::from_triangulation(dt.as_triangulation()).is_ok());
    assert!(dt.validate().is_ok());
    assert_bistellar_flips(&dt);

    let mut empty_tds: InsertionTds<f64, (), (), 2> = InsertionTds::empty();
    assert_eq!(
        repair_neighbor_pointers_local(&mut empty_tds, &[], None)?,
        0
    );
    assert!(matches!(
        DelaunayConstructionFailure::GeometricDegeneracy {
            message: "synthetic".to_string(),
        },
        DelaunayConstructionFailure::GeometricDegeneracy { .. }
    ));
    let cavity_failure = DelaunayConstructionFailure::InsertionCavityFilling {
        source: CavityFillingError::EmptyFanTriangulation,
    };
    let DelaunayConstructionFailure::InsertionCavityFilling { source } = cavity_failure else {
        unreachable!("constructed cavity-filling failure should match its own variant");
    };
    assert_eq!(source, CavityFillingError::EmptyFanTriangulation);
    assert_eq!(
        CavityRepairStage::PrimaryInsertion.to_string(),
        "primary insertion"
    );
    assert!(matches!(LocateResult::Outside, LocateResult::Outside));
    assert!(matches!(
        ValidationCadence::from_optional_every(Some(128)),
        ValidationCadence::EveryN(every) if every.get() == 128
    ));
    assert_send_sync_unpin::<TdsMutationError>();
    assert_send_sync_unpin::<NeighborRebuildError>();
    assert_send_sync_unpin::<ConstructionSkipSample>();
    assert_send_sync_unpin::<ConstructionSlowInsertionSample>();
    assert_send_sync_unpin::<CoordinateConversionError>();
    assert_send_sync_unpin::<DegenerateSimplexReason>();
    assert_send_sync_unpin::<MatrixError>();
    assert!(NeighborSlot::Boundary.is_boundary());
    assert_eq!(
        DegenerateSimplexReason::ZeroOrientation.to_string(),
        "zero orientation"
    );
    assert!(matches!(
        MatrixError::OutOfBounds {
            row: 1,
            column: 2,
            dimension: 3
        },
        MatrixError::OutOfBounds { .. }
    ));
    let mut root_secure_map: SecureHashMap<[u64; 2], usize> = SecureHashMap::default();
    root_secure_map.insert([1, 2], 3);
    assert_eq!(root_secure_map.get(&[1, 2]), Some(&3));

    let mut root_secure_set: SecureHashSet<[u64; 2]> = SecureHashSet::default();
    root_secure_set.insert([1, 2]);
    assert!(root_secure_set.contains(&[1, 2]));

    let mut scoped_secure_map: ScopedSecureHashMap<[u64; 2], usize> =
        ScopedSecureHashMap::default();
    scoped_secure_map.insert([3, 4], 5);
    assert_eq!(scoped_secure_map.get(&[3, 4]), Some(&5));

    let mut scoped_secure_set: ScopedSecureHashSet<[u64; 2]> = ScopedSecureHashSet::default();
    scoped_secure_set.insert([3, 4]);
    assert!(scoped_secure_set.contains(&[3, 4]));

    let telemetry = ConstructionTelemetry::default();
    assert!(!telemetry.has_data());
    Ok(())
}

#[test]
fn construction_prelude_covers_explicit_error_summaries() {
    let explicit_tds = ExplicitTdsError {
        kind: ExplicitTdsErrorKind::FacetSharingViolation,
        message: "facet sharing violation".to_string(),
    };
    let explicit_construction = ExplicitConstructionError::StructuralValidation {
        source: explicit_tds,
    };
    let ExplicitConstructionError::StructuralValidation { source } = explicit_construction else {
        unreachable!("constructed structural validation variant should match");
    };
    assert_eq!(source.kind, ExplicitTdsErrorKind::FacetSharingViolation);

    let simplex_creation = ExplicitConstructionError::SimplexCreation {
        simplex_index: 3,
        source: SimplexValidationError::DuplicateVertices,
    };
    let ExplicitConstructionError::SimplexCreation {
        simplex_index,
        source,
    } = simplex_creation
    else {
        unreachable!("constructed simplex creation variant should match");
    };
    assert_eq!(simplex_index, 3);
    assert_eq!(source, SimplexValidationError::DuplicateVertices);

    let explicit_insertion = ExplicitInsertionError {
        kind: ExplicitInsertionErrorKind::TopologyValidation,
        source_kind: None,
        message: "topology validation failed".to_string(),
    };
    assert_eq!(
        explicit_insertion.kind,
        ExplicitInsertionErrorKind::TopologyValidation
    );

    let explicit_invariant = ExplicitInvariantError {
        kind: ExplicitInvariantErrorKind::Tds,
        detail: InvariantErrorSummaryDetail::Tds(TdsErrorKind::FacetSharingViolation),
        message: "tds validation failed".to_string(),
    };
    assert_eq!(explicit_invariant.kind, ExplicitInvariantErrorKind::Tds);

    let explicit_delaunay_validation = ExplicitDelaunayValidationError {
        kind: ExplicitDelaunayValidationErrorKind::Tds,
        source_kind: Some(ExplicitDelaunayValidationSourceKind::Tds(
            TdsErrorKind::FacetSharingViolation,
        )),
        message: "delaunay validation failed".to_string(),
    };
    assert_eq!(
        explicit_delaunay_validation.kind,
        ExplicitDelaunayValidationErrorKind::Tds
    );
}

#[test]
fn diagnostic_preludes_cover_repair_apis() -> Result<(), PreludeExportTestError> {
    let vertices: Vec<Vertex<f64, (), 3>> = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];
    let mut dt = DelaunayTriangulation::new(&vertices)?;

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
        DelaunayRepairError::Flip(FlipError::DegenerateSimplex),
        DelaunayRepairError::Flip(_)
    ));
    assert_send_sync_unpin::<FlipEdgeAdjacencyError>();
    assert_send_sync_unpin::<FlipTriangleAdjacencyError>();
    assert_send_sync_unpin::<FlipVertexAdjacencyError>();
    let validation_error = DelaunayTriangulationValidationError::RepairOperationFailed {
        operation: DelaunayRepairOperation::VertexRemoval,
        source: Box::new(DelaunayRepairError::VerificationFailed {
            context: DelaunayRepairVerificationContext::StrictValidation,
            source: Box::new(FlipError::DegenerateSimplex),
        }),
    };
    assert!(validation_error.to_string().contains("vertex removal"));

    verify_delaunay_for_triangulation(dt.as_triangulation())?;

    let outcome = delaunayize_by_flips(&mut dt, DelaunayizeConfig::default())?;
    assert!(!outcome.used_fallback_rebuild);
    let _typed_outcome: DelaunayizeOutcome<f64, (), (), 3> = outcome;
    let _typed_error: Option<DelaunayizeError> = None;
    Ok(())
}

#[cfg(feature = "diagnostics")]
#[test]
fn diagnostics_prelude_covers_opt_in_helpers() -> Result<(), PreludeExportTestError> {
    let tds: Tds<f64, (), (), 2> = Tds::empty();
    debug_print_first_delaunay_violation(&tds, None);
    let report = delaunay_violation_report(&tds, None)?;
    let _typed_report: DelaunayViolationReport = report;
    let _typed_detail: Option<DelaunayViolationDetail> = None;
    assert!(DiagnosticNeighborSlot::Boundary.is_boundary());

    let kernel = AdaptiveKernel::new();
    let point = Point::new([0.0, 0.0]);
    let conflict_simplices = SimplexKeyBuffer::new();
    assert_eq!(
        verify_conflict_region_completeness(&tds, &kernel, &point, &conflict_simplices),
        0
    );
    Ok(())
}

#[test]
fn ordering_prelude_covers_hilbert_apis() -> Result<(), HilbertError> {
    let coords = [[0.9_f64, 0.9], [0.1, 0.1], [0.5, 0.5]];
    let order = hilbert_sorted_indices(&coords, (0.0, 1.0), 8)?;
    assert_eq!(order.len(), coords.len());

    let quantized: Vec<[u32; 2]> = coords
        .iter()
        .map(|coord| hilbert_quantize(coord, (0.0, 1.0), 8))
        .collect::<Result<_, _>>()?;
    let indices = hilbert_indices_prequantized(&quantized, 8)?;
    assert_eq!(indices.len(), coords.len());

    let index = hilbert_index(&coords[0], (0.0, 1.0), 8)?;
    assert_eq!(index, indices[0]);

    let mut stable_payload = vec![0_usize, 1, 2];
    hilbert_sort_by_stable(&mut stable_payload, (0.0, 1.0), 8, |&i| coords[i])?;
    assert_eq!(stable_payload, order);

    let mut unstable_payload = vec![0_usize, 1, 2];
    hilbert_sort_by_unstable(&mut unstable_payload, (0.0, 1.0), 8, |&i| coords[i])?;
    assert_eq!(unstable_payload, order);

    Ok(())
}
