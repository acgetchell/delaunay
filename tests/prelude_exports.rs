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

use std::assert_matches;

use delaunay::prelude::DelaunayValidationError;
use delaunay::prelude::algorithms::LocateResult;
#[cfg(feature = "diagnostics")]
use delaunay::prelude::collections::SimplexKeyBuffer;
use delaunay::prelude::collections::{
    SecureHashMap as ScopedSecureHashMap, SecureHashSet as ScopedSecureHashSet,
};
use delaunay::prelude::construction::{
    CavityFillingError, CavityRepairStage, ConstructionOptions, ConstructionSkipSample,
    ConstructionSlowInsertionSample, DelaunayConstructionFailure, DelaunayRepairPolicy,
    DelaunayTriangulation, DelaunayTriangulationConstructionError, ExplicitConstructionError,
    ExplicitDelaunayValidationError, ExplicitDelaunayValidationErrorKind,
    ExplicitDelaunayValidationSourceKind, ExplicitInsertionError, ExplicitInsertionErrorKind,
    ExplicitInvariantError, ExplicitInvariantErrorKind, ExplicitTdsError, ExplicitTdsErrorKind,
    InsertionOrderStrategy, SimplexValidationError, TopologyGuarantee, Vertex, vertex,
};
use delaunay::prelude::delaunayize::{
    DelaunayTriangulationBuilder as DelaunayizeDelaunayTriangulationBuilder, DelaunayizeConfig,
    DelaunayizeError, DelaunayizeOutcome, delaunayize_by_flips,
};
use delaunay::prelude::diagnostics::ConstructionTelemetry;
#[cfg(feature = "diagnostics")]
use delaunay::prelude::diagnostics::{
    DelaunayViolationDetail, DelaunayViolationReport, NeighborSlot as DiagnosticNeighborSlot,
    debug_print_first_delaunay_violation, delaunay_violation_report,
    verify_conflict_region_completeness,
};
use delaunay::prelude::flips::BistellarFlips;
use delaunay::prelude::generators::{RandomPointGenerationError, generate_random_points_seeded};
#[cfg(feature = "diagnostics")]
use delaunay::prelude::geometry::{AdaptiveKernel, Coordinate};
use delaunay::prelude::geometry::{
    CoordinateConversionError, DegenerateSimplexReason, LaError, MatrixError, Point,
};
use delaunay::prelude::insertion::{
    InsertionError, NeighborRebuildError, Tds as InsertionTds, TdsMutationError,
    repair_neighbor_pointers_local,
};
use delaunay::prelude::ordering::{
    HilbertBitDepth, HilbertError, MAX_HILBERT_BITS, hilbert_index, hilbert_indices_prequantized,
    hilbert_quantize, hilbert_sort_by_stable, hilbert_sort_by_unstable, hilbert_sorted_indices,
};
use delaunay::prelude::query::{ConvexHull, QueryError};
use delaunay::prelude::repair::{
    DelaunayCheckPolicy, DelaunayRepairDiagnostics, DelaunayRepairError, DelaunayRepairOperation,
    DelaunayRepairOutcome, DelaunayRepairStats, DelaunayRepairVerificationContext,
    DelaunayTriangulationValidationError, FlipEdgeAdjacencyError, FlipError,
    FlipTriangleAdjacencyError, FlipVertexAdjacencyError, RepairQueueOrder,
    verify_delaunay_for_triangulation,
};
#[cfg(feature = "diagnostics")]
use delaunay::prelude::tds::Tds;
use delaunay::prelude::tds::{InvariantErrorSummaryDetail, NeighborSlot, TdsErrorKind, VertexKey};
use delaunay::prelude::topology::validation::{
    ManifoldError, RidgeVertices, RidgeVerticesError, ridge_star_simplices,
};
use delaunay::prelude::triangulation::{
    FacetIssuesMap as TriangulationFacetIssuesMap, FastKernel as TriangulationFastKernel,
    InsertionError as TriangulationInsertionError, QueryError as TriangulationQueryError,
    TdsError as TriangulationTdsError, TopologyGuarantee as TriangulationTopologyGuarantee,
    Triangulation as GenericTriangulation,
    TriangulationConstructionError as GenericTriangulationConstructionError,
    ValidationConfigurationError as TriangulationValidationConfigurationError,
    ValidationPolicy as TriangulationValidationPolicy, vertex as triangulation_vertex,
};
use delaunay::prelude::validation::{
    TopologyGuarantee as FocusedValidationTopologyGuarantee, ValidationCadence,
    ValidationConfigurationError as FocusedValidationConfigurationError,
    ValidationPolicy as FocusedValidationPolicy,
};
use delaunay::prelude::{
    SecureHashMap, SecureHashSet, ValidationConfigurationError as RootValidationConfigurationError,
};
#[derive(Debug, thiserror::Error)]
enum RootApiExportTestError {
    #[error(transparent)]
    Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    #[error(transparent)]
    Validation(#[from] delaunay::DelaunayTriangulationValidationError),
    #[error(transparent)]
    DelaunayRepair(#[from] delaunay::flips::DelaunayRepairError),
    #[error(transparent)]
    Delaunayize(#[from] delaunay::delaunayize::DelaunayizeError),
}

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
    #[error(transparent)]
    Manifold(#[from] ManifoldError),
    #[error(transparent)]
    Query(#[from] QueryError),
    #[error(transparent)]
    RidgeVertices(#[from] RidgeVerticesError),
}

/// Proves the focused flips prelude exports the trait bound expected by benchmarks.
const fn assert_bistellar_flips(_: &impl BistellarFlips<3, Scalar = f64, VertexData = ()>) {}

/// Proves the root flips module exports the same public trait bound.
const fn assert_root_bistellar_flips(
    _: &impl delaunay::flips::BistellarFlips<3, Scalar = f64, VertexData = ()>,
) {
}

const fn assert_send_sync_unpin<T: Send + Sync + Unpin>() {}

#[test]
fn root_exports_cover_flattened_public_api() -> Result<(), RootApiExportTestError> {
    use delaunay::builder::DelaunayTriangulationBuilder as BuilderModuleBuilder;
    use delaunay::construction::{
        ConstructionOptions as ConstructionModuleOptions, InsertionOrderStrategy,
    };
    use delaunay::delaunayize::{
        DelaunayizeConfig as DelaunayizeModuleConfig, delaunayize_by_flips,
    };
    use delaunay::repair::{DelaunayCheckPolicy, DelaunayRepairPolicy};
    use delaunay::validation::{DelaunayTriangulationValidationError, ValidationCadence};
    use delaunay::{
        ConstructionOptions, DelaunayTriangulation, DelaunayTriangulationBuilder,
        TopologyGuarantee, ValidationPolicy,
    };

    let vertices = vec![
        delaunay::vertex!([0.0, 0.0, 0.0]),
        delaunay::vertex!([1.0, 0.0, 0.0]),
        delaunay::vertex!([0.0, 1.0, 0.0]),
        delaunay::vertex!([0.0, 0.0, 1.0]),
    ];

    let options: ConstructionOptions =
        ConstructionModuleOptions::default().with_insertion_order(InsertionOrderStrategy::Input);
    let builder: BuilderModuleBuilder<'_, f64, (), 3> =
        DelaunayTriangulationBuilder::new(&vertices).construction_options(options);
    let mut dt: DelaunayTriangulation<_, (), (), 3> = builder.build::<()>()?;

    assert_eq!(dt.topology_guarantee(), TopologyGuarantee::PLManifold);
    assert_eq!(dt.validation_policy(), ValidationPolicy::ExplicitOnly);
    assert_matches!(
        ValidationCadence::from_optional_every(Some(2)),
        ValidationCadence::EveryN(every) if every.get() == 2
    );
    assert_eq!(
        DelaunayRepairPolicy::default(),
        DelaunayRepairPolicy::EveryInsertion
    );
    assert!(!DelaunayCheckPolicy::default().should_check(1));

    let validation_result: Result<(), DelaunayTriangulationValidationError> = dt.validate();
    validation_result?;
    assert_bistellar_flips(&dt);
    assert_root_bistellar_flips(&dt);

    let outcome = delaunayize_by_flips(&mut dt, DelaunayizeModuleConfig::default())?;
    assert!(!outcome.used_fallback_rebuild);
    assert!(outcome.topology_repair.succeeded);
    Ok(())
}

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
    assert_matches!(
        options.batch_repair_policy(),
        DelaunayRepairPolicy::EveryInsertion
    );
    let dt = DelaunayTriangulation::new_with_options(&vertices, options)?;

    assert_eq!(dt.topology_guarantee(), TopologyGuarantee::PLManifold);
    assert!(dt.boundary_facets()?.count() > 0);
    assert!(ConvexHull::from_triangulation(dt.as_triangulation()).is_ok());
    assert!(dt.validate().is_ok());
    assert_bistellar_flips(&dt);

    let mut empty_tds: InsertionTds<f64, (), (), 2> = InsertionTds::empty();
    assert_eq!(
        repair_neighbor_pointers_local(&mut empty_tds, &[], None)?,
        0
    );
    assert_matches!(
        DelaunayConstructionFailure::GeometricDegeneracy {
            message: "synthetic".to_string(),
        },
        DelaunayConstructionFailure::GeometricDegeneracy { .. }
    );
    let unsupported_periodic_dimension =
        DelaunayConstructionFailure::UnsupportedPeriodicDimension {
            dimension: 4,
            max_validated_dimension: 3,
            tracking_issue: 416,
        };
    let DelaunayConstructionFailure::UnsupportedPeriodicDimension {
        dimension,
        max_validated_dimension,
        tracking_issue,
    } = unsupported_periodic_dimension
    else {
        unreachable!("constructed unsupported periodic dimension variant should match");
    };
    assert_eq!(dimension, 4);
    assert_eq!(max_validated_dimension, 3);
    assert_eq!(tracking_issue, 416);
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
    assert_matches!(LocateResult::Outside, LocateResult::Outside);
    assert_matches!(
        ValidationCadence::from_optional_every(Some(128)),
        ValidationCadence::EveryN(every) if every.get() == 128
    );
    assert_send_sync_unpin::<TdsMutationError>();
    assert_send_sync_unpin::<NeighborRebuildError>();
    assert_send_sync_unpin::<ConstructionSkipSample>();
    assert_send_sync_unpin::<ConstructionSlowInsertionSample>();
    assert_send_sync_unpin::<CoordinateConversionError>();
    assert_send_sync_unpin::<DegenerateSimplexReason>();
    assert_send_sync_unpin::<LaError>();
    assert_send_sync_unpin::<MatrixError>();
    assert!(NeighborSlot::Boundary.is_boundary());
    assert_eq!(
        DegenerateSimplexReason::ZeroOrientation.to_string(),
        "zero orientation"
    );
    assert_matches!(
        MatrixError::OutOfBounds {
            row: 1,
            column: 2,
            dimension: 3
        },
        MatrixError::OutOfBounds { .. }
    );
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
fn validation_prelude_covers_configuration_error() {
    let focused_error =
        FocusedValidationConfigurationError::IncompatibleTopologyAndValidationPolicy {
            topology_guarantee: FocusedValidationTopologyGuarantee::PLManifold,
            validation_policy: FocusedValidationPolicy::Never,
        };
    let root_error: RootValidationConfigurationError = focused_error;
    assert_matches!(
        root_error,
        RootValidationConfigurationError::IncompatibleTopologyAndValidationPolicy {
            topology_guarantee: FocusedValidationTopologyGuarantee::PLManifold,
            validation_policy: FocusedValidationPolicy::Never,
        }
    );

    let triangulation_error =
        TriangulationValidationConfigurationError::IncompatibleTopologyAndValidationPolicy {
            topology_guarantee: TriangulationTopologyGuarantee::PLManifoldStrict,
            validation_policy: TriangulationValidationPolicy::Never,
        };
    assert_matches!(
        triangulation_error,
        TriangulationValidationConfigurationError::IncompatibleTopologyAndValidationPolicy {
            topology_guarantee: TriangulationTopologyGuarantee::PLManifoldStrict,
            validation_policy: TriangulationValidationPolicy::Never,
        }
    );
}

fn simplex_prelude_vertices<const D: usize>(origin: f64, scale: f64) -> Vec<Vertex<f64, (), D>> {
    let mut vertices = Vec::with_capacity(D + 1);
    vertices.push(vertex!([origin; D]));

    for axis in 0..D {
        let mut coords = [origin; D];
        coords[axis] = origin + scale;
        vertices.push(vertex!(coords));
    }

    vertices
}

fn cospherical_prelude_vertices<const D: usize>() -> Vec<Vertex<f64, (), D>> {
    let mut vertices = Vec::with_capacity(D + 2);

    for axis in 0..D {
        let mut coords = [0.0; D];
        coords[axis] = 1.0;
        vertices.push(vertex!(coords));
    }

    let mut negative_first_axis = [0.0; D];
    negative_first_axis[0] = -1.0;
    vertices.push(vertex!(negative_first_axis));

    let mut negative_second_axis = [0.0; D];
    negative_second_axis[1] = -1.0;
    vertices.push(vertex!(negative_second_axis));

    vertices
}

fn degenerate_prelude_vertices<const D: usize>() -> Vec<Vertex<f64, (), D>> {
    let mut vertices = Vec::with_capacity(D + 1);
    let mut coordinate = 0.0;
    for _ in 0..=D {
        let mut coords = [0.0; D];
        coords[0] = coordinate;
        vertices.push(vertex!(coords));
        coordinate += 1.0;
    }
    vertices
}

fn assert_single_simplex_ridge_star<const D: usize>(
    vertices: &[Vertex<f64, (), D>],
) -> Result<(), PreludeExportTestError> {
    let dt = DelaunayTriangulation::new(vertices)?;
    let ridge = RidgeVertices::<D>::try_from_vertices(dt.tds().vertex_keys().take(D - 1))?;
    let star = ridge_star_simplices(dt.tds(), &ridge)?;

    assert_eq!(star.len(), 1);
    Ok(())
}

fn assert_cospherical_ridge_star<const D: usize>() -> Result<(), PreludeExportTestError> {
    let vertices = cospherical_prelude_vertices::<D>();
    let dt = DelaunayTriangulation::new(&vertices)?;
    let ridge = RidgeVertices::<D>::try_from_vertices(dt.tds().vertex_keys().take(D - 1))?;
    let star = ridge_star_simplices(dt.tds(), &ridge)?;

    assert!(!star.is_empty());
    Ok(())
}

fn assert_ridge_vertices_reject_adversarial_keys<const D: usize>(keys: &[VertexKey]) {
    assert_matches!(
        RidgeVertices::<D>::try_from_vertices(keys.iter().take(D.saturating_sub(2)).copied()),
        Err(RidgeVerticesError::WrongArity {
            expected,
            actual,
            ..
        }) if expected == D - 1 && actual == D.saturating_sub(2)
    );

    if D >= 3 {
        assert_matches!(
            RidgeVertices::<D>::try_from_vertices(std::iter::repeat_n(keys[0], D - 1)),
            Err(RidgeVerticesError::DuplicateVertex { vertex_key }) if vertex_key == keys[0]
        );
    }
}

fn assert_topology_prelude_dimension<const D: usize>() -> Result<(), PreludeExportTestError> {
    let simplex_vertices = simplex_prelude_vertices::<D>(0.0, 1.0);
    assert_single_simplex_ridge_star(&simplex_vertices)?;

    let near_boundary_vertices = simplex_prelude_vertices::<D>(f64::EPSILON, 1.0);
    assert_single_simplex_ridge_star(&near_boundary_vertices)?;

    let dt = DelaunayTriangulation::new(&simplex_vertices)?;
    let keys = dt.tds().vertex_keys().collect::<Vec<_>>();
    assert_ridge_vertices_reject_adversarial_keys::<D>(&keys);

    assert_cospherical_ridge_star::<D>()?;
    assert_matches!(
        DelaunayTriangulation::new(&degenerate_prelude_vertices::<D>()),
        Err(DelaunayTriangulationConstructionError::Triangulation(
            DelaunayConstructionFailure::GeometricDegeneracy { .. }
        ))
    );
    Ok(())
}

#[test]
fn topology_validation_prelude_covers_ridge_star_api() -> Result<(), PreludeExportTestError> {
    assert_topology_prelude_dimension::<2>()?;
    assert_topology_prelude_dimension::<3>()?;
    assert_topology_prelude_dimension::<4>()?;
    assert_topology_prelude_dimension::<5>()?;
    Ok(())
}

#[test]
fn triangulation_prelude_covers_generic_layer() -> Result<(), GenericTriangulationConstructionError>
{
    let vertices = vec![
        triangulation_vertex!([0.0, 0.0]),
        triangulation_vertex!([1.0, 0.0]),
        triangulation_vertex!([0.0, 1.0]),
    ];
    let tds =
        GenericTriangulation::<TriangulationFastKernel<f64>, (), (), 2>::build_initial_simplex(
            &vertices,
        )?;
    assert_eq!(tds.number_of_vertices(), 3);
    assert_eq!(tds.number_of_simplices(), 1);

    let mut tri: GenericTriangulation<TriangulationFastKernel<f64>, (), (), 2> =
        GenericTriangulation::new_empty(TriangulationFastKernel::new());
    tri.set_topology_guarantee(TriangulationTopologyGuarantee::Pseudomanifold);
    tri.set_validation_policy(TriangulationValidationPolicy::Never);
    assert!(tri.validate().is_ok());

    let empty_issues = TriangulationFacetIssuesMap::default();
    let removed = tri
        .repair_local_facet_issues(&empty_issues, 0)
        .expect("empty issue set should not fail generic local repair");
    assert_eq!(removed, 0);
    assert_eq!(tri.boundary_facets().unwrap().count(), 0);

    assert_send_sync_unpin::<TriangulationInsertionError>();
    assert_send_sync_unpin::<TriangulationQueryError>();
    assert_send_sync_unpin::<TriangulationTdsError>();
    Ok(())
}

#[test]
fn diagnostic_preludes_cover_repair_apis() -> Result<(), PreludeExportTestError> {
    let vertices: Vec<Vertex<f64, (), 3>> = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];
    let mut dt = DelaunayizeDelaunayTriangulationBuilder::new(&vertices).build::<()>()?;

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
    assert_matches!(
        DelaunayRepairError::from(FlipError::DegenerateSimplex),
        DelaunayRepairError::Flip { .. }
    );
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
    assert_eq!(MAX_HILBERT_BITS, 31);
    let bits = HilbertBitDepth::try_new(8)?;
    let order = hilbert_sorted_indices(&coords, (0.0, 1.0), bits)?;
    assert_eq!(order.len(), coords.len());

    let quantized: Vec<[u32; 2]> = coords
        .iter()
        .map(|coord| hilbert_quantize(coord, (0.0, 1.0), bits))
        .collect::<Result<_, _>>()?;
    let indices = hilbert_indices_prequantized(&quantized, bits)?;
    assert_eq!(indices.len(), coords.len());

    let index = hilbert_index(&coords[0], (0.0, 1.0), bits)?;
    assert_eq!(index, indices[0]);

    let mut stable_payload = vec![0_usize, 1, 2];
    hilbert_sort_by_stable(&mut stable_payload, (0.0, 1.0), bits, |&i| coords[i])?;
    assert_eq!(stable_payload, order);

    let mut unstable_payload = vec![0_usize, 1, 2];
    hilbert_sort_by_unstable(&mut unstable_payload, (0.0, 1.0), bits, |&i| coords[i])?;
    assert_eq!(unstable_payload, order);

    Ok(())
}
