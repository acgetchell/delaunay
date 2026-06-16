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

use std::{assert_matches, num::NonZeroUsize};

use approx::assert_relative_eq;
use delaunay::geometry::CoordinateRange as GeometryCoordinateRange;
use delaunay::prelude::DelaunayValidationError;
use delaunay::prelude::algorithms::LocateResult;
#[cfg(feature = "diagnostics")]
use delaunay::prelude::collections::SimplexKeyBuffer;
use delaunay::prelude::collections::{
    SecureHashMap as ScopedSecureHashMap, SecureHashSet as ScopedSecureHashSet,
};
use delaunay::prelude::construction::{
    CavityFillingError, CavityRepairStage, ConstructionOptions, ConstructionSkipSample,
    ConstructionSlowInsertionSample, CoordinateRangeError as ConstructionCoordinateRangeError,
    CoordinateRangeOrdering as ConstructionCoordinateRangeOrdering,
    CoordinateValidationError as ConstructionCoordinateValidationError, DedupPolicy,
    DedupTolerance, DeduplicationError, DelaunayConstructionFailure, DelaunayRepairPolicy,
    DelaunayTriangulation, DelaunayTriangulationConstructionError,
    DelaunayTriangulationValidationError as ConstructionDelaunayTriangulationValidationError,
    DelaunayVerificationError as ConstructionDelaunayVerificationError,
    DelaunayVerificationErrorKind as ConstructionDelaunayVerificationErrorKind,
    ExplicitConstructionError, ExplicitDelaunayValidationError,
    ExplicitDelaunayValidationErrorKind, ExplicitDelaunayValidationSourceKind,
    ExplicitInsertionError, ExplicitInsertionErrorKind, ExplicitInvariantError,
    ExplicitInvariantErrorKind, ExplicitTdsError, ExplicitTdsErrorKind,
    GlobalTopologyModelError as ConstructionGlobalTopologyModelError, InsertionOrderStrategy,
    InvalidCoordinateValue as ConstructionInvalidCoordinateValue,
    InvalidPositiveScalar as ConstructionInvalidPositiveScalar, RandomPointGenerationError,
    SimplexValidationError,
    SpatialIndexConstructionFailure as ConstructionSpatialIndexConstructionFailure,
    TopologyGuarantee, ToroidalDomain as ConstructionToroidalDomain, Vertex, VertexValidationError,
    try_vertices_from_points as construction_try_vertices_from_points,
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
use delaunay::prelude::flips::{
    BistellarFlips, FlipOrientationCheckStage as FocusedFlipOrientationCheckStage,
};
use delaunay::prelude::generators::{
    CoordinateRange, CoordinateRangeError, InvalidPositiveScalar,
    RandomPointGenerationError as GeneratorRandomPointGenerationError, RandomTriangulationBuilder,
    generate_grid_points, generate_random_points_in_range_seeded,
    try_generate_random_points_seeded,
};
#[cfg(feature = "diagnostics")]
use delaunay::prelude::geometry::AdaptiveKernel;
use delaunay::prelude::geometry::{
    ArrayConversionFailureReason, CircumcenterError, CircumcenterFailureReason,
    CoordinateConversionError, CoordinateConversionValue, CoordinateValidationError,
    DegenerateGeometry, DegenerateMeasure, DegenerateSimplexReason, FiniteCoordinateValue,
    InvalidCoordinateValue, LaError, MatrixError, Point, QualitySimplexVerticesError,
    SurfaceMeasureError, ValueConversionError, ValueConversionFailureReason,
};
use delaunay::prelude::insertion::{
    InitialSimplexConstructionError, InitialSimplexUnexpectedInsertionStage, InsertionError,
    NeighborRebuildError, Tds as InsertionTds, TdsMutationError, repair_neighbor_pointers_local,
};
use delaunay::prelude::ordering::{
    HilbertBitDepth, HilbertError, HilbertQuantizedBatch, MAX_HILBERT_BITS, hilbert_index_in_range,
    hilbert_indices_for_quantized_batch, hilbert_indices_prequantized, hilbert_quantize_in_range,
    hilbert_sort_by_stable_in_range, hilbert_sort_by_unstable_in_range,
    hilbert_sorted_indices_in_range, try_hilbert_index, try_hilbert_quantize,
    try_hilbert_sort_by_stable, try_hilbert_sort_by_unstable, try_hilbert_sorted_indices,
};
use delaunay::prelude::query::{
    AllFacetsIter as QueryAllFacetsIter, BoundaryFacetsIter as QueryBoundaryFacetsIter, ConvexHull,
    QueryError,
};
use delaunay::prelude::repair::{
    DelaunayCheckPolicy, DelaunayRepairDiagnostics, DelaunayRepairError, DelaunayRepairOperation,
    DelaunayRepairOutcome, DelaunayRepairStats, DelaunayRepairVerificationContext,
    DelaunayTriangulationValidationError, FlipEdgeAdjacencyError, FlipError,
    FlipOrientationCheckStage as RepairFlipOrientationCheckStage, FlipTriangleAdjacencyError,
    FlipVertexAdjacencyError, RepairQueueOrder, verify_delaunay_for_triangulation,
};
#[cfg(feature = "diagnostics")]
use delaunay::prelude::tds::Tds;
use delaunay::prelude::tds::{
    AllFacetsIter as TdsAllFacetsIter, BoundaryFacetsIter as TdsBoundaryFacetsIter, FacetError,
    FacetHandle, FacetView, InvariantErrorSummaryDetail, NeighborSlot, TdsError, TdsErrorKind,
    VertexKey,
};
use delaunay::prelude::topology::spaces::{
    GlobalTopology, GlobalTopologyModelError, TopologyKind, ToroidalConstructionMode,
    ToroidalDomain, ToroidalDomainError,
};
use delaunay::prelude::topology::validation::{
    GlobalTopology as TopologyValidationGlobalTopology, ManifoldError, RidgeVertices,
    RidgeVerticesError, ridge_star_simplices,
};
use delaunay::prelude::triangulation::{
    AllFacetsIter as TriangulationAllFacetsIter,
    BoundaryFacetsIter as TriangulationBoundaryFacetsIter,
    FacetIssuesMap as TriangulationFacetIssuesMap, FastKernel as TriangulationFastKernel,
    InsertionError as TriangulationInsertionError, QueryError as TriangulationQueryError,
    SpatialIndexConstructionFailure as GenericSpatialIndexConstructionFailure,
    TdsError as TriangulationTdsError, TopologyGuarantee as TriangulationTopologyGuarantee,
    Triangulation as GenericTriangulation,
    TriangulationConstructionError as GenericTriangulationConstructionError,
    ValidationConfigurationError as TriangulationValidationConfigurationError,
    ValidationPolicy as TriangulationValidationPolicy,
};
use delaunay::prelude::try_vertices_from_points as prelude_try_vertices_from_points;
use delaunay::prelude::validation::{
    TopologyGuarantee as FocusedValidationTopologyGuarantee, ValidationCadence,
    ValidationConfigurationError as FocusedValidationConfigurationError,
    ValidationPolicy as FocusedValidationPolicy,
};
use delaunay::prelude::{
    CoordinateRange as RootCoordinateRange,
    FlipOrientationCheckStage as RootFlipOrientationCheckStage, SecureHashMap, SecureHashSet,
    ValidationConfigurationError as RootValidationConfigurationError,
};
use delaunay::query::{
    AllFacetsIter as QueryFacadeAllFacetsIter, BoundaryFacetsIter as QueryFacadeBoundaryFacetsIter,
};
#[derive(Debug, thiserror::Error)]
enum RootApiExportTestError {
    #[error(transparent)]
    Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    #[error(transparent)]
    CoordinateConversion(#[from] delaunay::geometry::CoordinateConversionError),
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
    CoordinateRange(#[from] CoordinateRangeError),
    #[error(transparent)]
    CoordinateConversion(#[from] CoordinateConversionError),
    #[error(transparent)]
    CoordinateValidation(#[from] CoordinateValidationError),
    #[error(transparent)]
    VertexValidation(#[from] VertexValidationError),
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
    Facet(#[from] FacetError),
    #[error(transparent)]
    RidgeVertices(#[from] RidgeVerticesError),
    #[error(transparent)]
    ToroidalDomain(#[from] ToroidalDomainError),
    #[error(transparent)]
    GenericTriangulationConstruction(#[from] GenericTriangulationConstructionError),
}

/// Proves the focused flips prelude exports the trait bound expected by benchmarks.
const fn assert_bistellar_flips(_: &impl BistellarFlips<3, VertexData = ()>) {}

/// Proves the root flips module exports the same public trait bound.
const fn assert_root_bistellar_flips(_: &impl delaunay::flips::BistellarFlips<3, VertexData = ()>) {
}

const fn assert_send_sync_unpin<T: Send + Sync + Unpin>() {}

#[test]
fn construction_prelude_covers_dedup_policy() {
    assert_eq!(
        DedupTolerance::try_new(-1.0),
        Err(DeduplicationError::NegativeEpsilon)
    );
    let dedup_policy = DedupPolicy::epsilon(DedupTolerance::try_new(0.0).unwrap());
    assert_matches!(dedup_policy, DedupPolicy::Epsilon { .. });
}

#[test]
fn construction_prelude_covers_typed_construction_failure_variants() {
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
    assert_matches!(
        unsupported_periodic_dimension,
        DelaunayConstructionFailure::UnsupportedPeriodicDimension {
            dimension: 4,
            max_validated_dimension: 3,
            tracking_issue: 416,
        }
    );
    let topology_model_failure = DelaunayConstructionFailure::TopologyModelConfiguration {
        source: ConstructionGlobalTopologyModelError::PeriodicOffsetsUnsupported {
            kind: TopologyKind::Euclidean,
        },
    };
    assert_matches!(
        topology_model_failure,
        DelaunayConstructionFailure::TopologyModelConfiguration {
            source: ConstructionGlobalTopologyModelError::PeriodicOffsetsUnsupported {
                kind: TopologyKind::Euclidean,
            },
        }
    );
    let vertex_canonicalization_failure = DelaunayConstructionFailure::VertexCanonicalization {
        vertex_index: 2,
        source: GlobalTopologyModelError::NonFiniteCoordinate {
            axis: 1,
            value: f64::INFINITY,
        },
    };
    assert_matches!(
        vertex_canonicalization_failure,
        DelaunayConstructionFailure::VertexCanonicalization {
            vertex_index: 2,
            source: GlobalTopologyModelError::NonFiniteCoordinate {
                axis: 1,
                value,
            },
        } if value.is_infinite()
    );
    let canonicalized_point_failure = DelaunayConstructionFailure::CanonicalizedPointValidation {
        vertex_index: 3,
        source: ConstructionCoordinateValidationError::InvalidCoordinate {
            dimension: 2,
            coordinate_index: 0,
            coordinate_value: ConstructionInvalidCoordinateValue::Nan,
        },
    };
    assert_matches!(
        canonicalized_point_failure,
        DelaunayConstructionFailure::CanonicalizedPointValidation {
            vertex_index: 3,
            source: ConstructionCoordinateValidationError::InvalidCoordinate {
                dimension: 2,
                coordinate_index: 0,
                coordinate_value: ConstructionInvalidCoordinateValue::Nan,
            },
        }
    );
    let cavity_failure = DelaunayConstructionFailure::InsertionCavityFilling {
        source: CavityFillingError::EmptyFanTriangulation,
    };
    assert_matches!(
        cavity_failure,
        DelaunayConstructionFailure::InsertionCavityFilling {
            source: CavityFillingError::EmptyFanTriangulation,
        }
    );
}

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
        InitialSimplexUnexpectedInsertionStage as RootInitialSimplexUnexpectedInsertionStage,
        TopologyGuarantee, ValidationPolicy,
    };

    let vertices = vec![
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    ];

    let options: ConstructionOptions =
        ConstructionModuleOptions::default().with_insertion_order(InsertionOrderStrategy::Input);
    let builder: BuilderModuleBuilder<'_, (), 3> =
        DelaunayTriangulationBuilder::new(&vertices).construction_options(options);
    let mut dt: DelaunayTriangulation<_, (), (), 3> = builder.build::<()>()?;

    assert_eq!(dt.topology_guarantee(), TopologyGuarantee::PLManifold);
    assert_eq!(dt.validation_policy(), ValidationPolicy::ExplicitOnly);
    assert_matches!(
        RootInitialSimplexUnexpectedInsertionStage::NonManifoldTopology {
            facet_hash: 0x00C0_FFEE,
            simplex_count: 3,
        },
        RootInitialSimplexUnexpectedInsertionStage::NonManifoldTopology {
            facet_hash: 0x00C0_FFEE,
            simplex_count: 3
        }
    );
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
fn flip_preludes_cover_orientation_check_stage() {
    assert_matches!(
        FocusedFlipOrientationCheckStage::BeforeMutation,
        FocusedFlipOrientationCheckStage::BeforeMutation
    );
    assert_matches!(
        RepairFlipOrientationCheckStage::AfterTrialMutation,
        RepairFlipOrientationCheckStage::AfterTrialMutation
    );
    assert_matches!(
        RootFlipOrientationCheckStage::AfterTrialMutation,
        RootFlipOrientationCheckStage::AfterTrialMutation
    );
    assert_matches!(
        delaunay::flips::FlipOrientationCheckStage::BeforeMutation,
        delaunay::flips::FlipOrientationCheckStage::BeforeMutation
    );
}

#[test]
fn preludes_cover_bench_apis() -> Result<(), PreludeExportTestError> {
    let _generated_points: Vec<Point<2>> = try_generate_random_points_seeded(3, (0.0, 1.0), 42)?;

    let vertices: Vec<Vertex<(), 3>> = vec![
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    ];
    let options =
        ConstructionOptions::default().with_insertion_order(InsertionOrderStrategy::Input);
    assert_matches!(
        options.batch_repair_policy(),
        DelaunayRepairPolicy::EveryInsertion
    );
    let dt = DelaunayTriangulation::try_new_with_options(&vertices, options)?;

    assert_eq!(dt.topology_guarantee(), TopologyGuarantee::PLManifold);
    let _query_facade_all_facets: QueryFacadeAllFacetsIter<'_, (), (), 3> = dt.facets()?;
    let _query_facade_boundary_facets: QueryFacadeBoundaryFacetsIter<'_, (), (), 3> =
        dt.boundary_facets()?;
    let _query_prelude_all_facets: QueryAllFacetsIter<'_, (), (), 3> = dt.facets()?;
    let _query_prelude_boundary_facets: QueryBoundaryFacetsIter<'_, (), (), 3> =
        dt.boundary_facets()?;
    let (simplex_key, _simplex) = dt
        .simplices()
        .next()
        .expect("constructed tetrahedron should contain a simplex");
    let facet_handle = FacetHandle::try_new(dt.tds(), simplex_key, 0)?;
    let facet_view: FacetView<'_, (), (), 3> = facet_handle.view(dt.tds())?;
    assert_eq!(facet_view.handle(), facet_handle);
    let boundary_facet_count = dt.boundary_facets()?.try_fold(0_usize, |count, facet| {
        facet
            .map(|_| count + 1)
            .map_err(|source| QueryError::TriangulationCorrupted {
                source: source.into(),
            })
    })?;
    assert!(boundary_facet_count > 0);
    let _hull = ConvexHull::try_from_triangulation(dt.as_triangulation()).unwrap();
    dt.validate().unwrap();
    assert_bistellar_flips(&dt);

    let mut empty_tds: InsertionTds<(), (), 2> = InsertionTds::empty();
    let _tds_all_facets: TdsAllFacetsIter<'_, (), (), 2> = empty_tds.facets().unwrap();
    let _tds_boundary_facets: Option<TdsBoundaryFacetsIter<'static, (), (), 2>> = None;
    assert_eq!(
        repair_neighbor_pointers_local(&mut empty_tds, &[], None)?,
        0
    );
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
fn insertion_prelude_covers_initial_simplex_stage_errors() {
    let stage = InitialSimplexUnexpectedInsertionStage::NonManifoldTopology {
        facet_hash: 0xABCD,
        simplex_count: 3,
    };
    let error = InitialSimplexConstructionError::UnexpectedInsertionStage {
        reason: Box::new(stage),
    };

    assert_matches!(
        error,
        InitialSimplexConstructionError::UnexpectedInsertionStage { reason }
            if matches!(
                *reason,
                InitialSimplexUnexpectedInsertionStage::NonManifoldTopology {
                    facet_hash: 0xABCD,
                    simplex_count: 3,
                }
            )
    );
}

#[test]
fn geometry_prelude_covers_typed_error_variants() {
    let finite = FiniteCoordinateValue::try_new(1.5).expect("1.5 is finite");
    assert_relative_eq!(finite.get(), 1.5, epsilon = f64::EPSILON);
    assert_eq!(
        CoordinateConversionValue::from_f64(f64::NAN),
        CoordinateConversionValue::NonFinite(InvalidCoordinateValue::Nan)
    );
    assert_eq!(
        CoordinateConversionValue::from_usize(7),
        CoordinateConversionValue::UnsignedInteger(7)
    );
    assert_eq!(DegenerateMeasure::SurfaceArea.to_string(), "surface area");
    assert_eq!(
        DegenerateGeometry::CollinearOrCoplanarPoints.to_string(),
        "collinear or coplanar points"
    );

    let circumcenter_error = CircumcenterError::MatrixInversionFailed {
        reason: CircumcenterFailureReason::DegenerateSimplex {
            measure: DegenerateMeasure::Volume,
            degeneracy: DegenerateGeometry::CoplanarPoints,
        },
    };
    assert_matches!(
        circumcenter_error,
        CircumcenterError::MatrixInversionFailed {
            reason: CircumcenterFailureReason::DegenerateSimplex {
                measure: DegenerateMeasure::Volume,
                degeneracy: DegenerateGeometry::CoplanarPoints,
            },
        }
    );

    let conversion_error = ValueConversionError::ConversionFailed {
        value: CoordinateConversionValue::Scalar(finite),
        from_type: "f64",
        to_type: "usize",
        reason: ValueConversionFailureReason::TargetTypeRejected,
    };
    assert_matches!(
        conversion_error,
        ValueConversionError::ConversionFailed {
            reason: ValueConversionFailureReason::TargetTypeRejected,
            ..
        }
    );

    assert_matches!(
        ArrayConversionFailureReason::LengthMismatch,
        ArrayConversionFailureReason::LengthMismatch
    );

    let quality_error = QualitySimplexVerticesError::UnexpectedTdsFailure {
        source: Box::new(TdsError::DuplicateSimplices {
            message: "same vertex set appears twice".to_string(),
        }),
    };
    assert_matches!(
        quality_error,
        QualitySimplexVerticesError::UnexpectedTdsFailure { source }
            if matches!(
                *source,
                TdsError::DuplicateSimplices { ref message }
                    if message == "same vertex set appears twice"
            )
    );

    let surface_error = SurfaceMeasureError::FacetVertices {
        source: FacetError::SimplexNotFoundInTriangulation,
    };
    assert_matches!(
        surface_error,
        SurfaceMeasureError::FacetVertices {
            source: FacetError::SimplexNotFoundInTriangulation
        }
    );
}

#[test]
fn generator_prelude_covers_validated_coordinate_ranges() -> Result<(), PreludeExportTestError> {
    let generated_range = CoordinateRange::try_new(0.0_f64, 1.0)?;
    let range_points: Vec<Point<2>> =
        generate_random_points_in_range_seeded(3, generated_range, 42)?;
    let grid_points: Vec<Point<2>> = generate_grid_points(NonZeroUsize::MIN, 1.0, [0.0, 0.0])?;
    let builder = RandomTriangulationBuilder::try_new(NonZeroUsize::MIN, (-1.0_f64, 1.0))?
        .seed(7)
        .insertion_order(InsertionOrderStrategy::Input);
    let _ = builder;

    let root_range = RootCoordinateRange::try_new(-1.0_f64, 1.0)?;
    assert_eq!(root_range.bounds(), (-1.0, 1.0));
    let geometry_range = GeometryCoordinateRange::try_new(-2.0_f64, -1.0)?;
    assert_eq!(geometry_range.bounds(), (-2.0, -1.0));
    assert_eq!(range_points.len(), 3);
    assert_eq!(grid_points.len(), 1);
    assert_matches!(
        InvalidPositiveScalar::NonPositive { value: 0.0_f64 },
        InvalidPositiveScalar::NonPositive { value: 0.0 }
    );
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
    assert_matches!(
        explicit_construction,
        ExplicitConstructionError::StructuralValidation {
            source: ExplicitTdsError {
                kind: ExplicitTdsErrorKind::FacetSharingViolation,
                ..
            }
        }
    );

    let simplex_creation = ExplicitConstructionError::SimplexCreation {
        simplex_index: 3,
        source: SimplexValidationError::DuplicateVertices,
    };
    assert_matches!(
        simplex_creation,
        ExplicitConstructionError::SimplexCreation {
            simplex_index: 3,
            source: SimplexValidationError::DuplicateVertices,
        }
    );

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

fn simplex_prelude_vertices<const D: usize>(
    origin: f64,
    scale: f64,
) -> Result<Vec<Vertex<(), D>>, PreludeExportTestError> {
    let mut vertices = Vec::with_capacity(D + 1);
    vertices.push(delaunay::prelude::Vertex::<(), _>::try_new([origin; D])?);

    for axis in 0..D {
        let mut coords = [origin; D];
        coords[axis] = origin + scale;
        vertices.push(delaunay::prelude::Vertex::<(), _>::try_new(coords)?);
    }

    Ok(vertices)
}

fn cospherical_prelude_vertices<const D: usize>()
-> Result<Vec<Vertex<(), D>>, PreludeExportTestError> {
    let mut vertices = Vec::with_capacity(D + 2);

    for axis in 0..D {
        let mut coords = [0.0; D];
        coords[axis] = 1.0;
        vertices.push(delaunay::prelude::Vertex::<(), _>::try_new(coords)?);
    }

    let mut negative_first_axis = [0.0; D];
    negative_first_axis[0] = -1.0;
    vertices.push(delaunay::prelude::Vertex::<(), _>::try_new(
        negative_first_axis,
    )?);

    let mut negative_second_axis = [0.0; D];
    negative_second_axis[1] = -1.0;
    vertices.push(delaunay::prelude::Vertex::<(), _>::try_new(
        negative_second_axis,
    )?);

    Ok(vertices)
}

fn degenerate_prelude_vertices<const D: usize>()
-> Result<Vec<Vertex<(), D>>, PreludeExportTestError> {
    let mut vertices = Vec::with_capacity(D + 1);
    let mut coordinate = 0.0;
    for _ in 0..=D {
        let mut coords = [0.0; D];
        coords[0] = coordinate;
        vertices.push(delaunay::prelude::Vertex::<(), _>::try_new(coords)?);
        coordinate += 1.0;
    }
    Ok(vertices)
}

fn assert_single_simplex_ridge_star<const D: usize>(
    vertices: &[Vertex<(), D>],
) -> Result<(), PreludeExportTestError> {
    let dt = DelaunayTriangulation::try_new(vertices)?;
    let ridge = RidgeVertices::<D>::try_from_vertices(dt.tds().vertex_keys().take(D - 1))?;
    let star = ridge_star_simplices(dt.tds(), &ridge)?;

    assert_eq!(star.len(), 1);
    Ok(())
}

fn assert_cospherical_ridge_star<const D: usize>() -> Result<(), PreludeExportTestError> {
    let vertices = cospherical_prelude_vertices::<D>()?;
    let dt = DelaunayTriangulation::try_new(&vertices)?;
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
    let simplex_vertices = simplex_prelude_vertices::<D>(0.0, 1.0)?;
    assert_single_simplex_ridge_star(&simplex_vertices)?;

    let near_boundary_vertices = simplex_prelude_vertices::<D>(f64::EPSILON, 1.0)?;
    assert_single_simplex_ridge_star(&near_boundary_vertices)?;

    let dt = DelaunayTriangulation::try_new(&simplex_vertices)?;
    let keys = dt.tds().vertex_keys().collect::<Vec<_>>();
    assert_ridge_vertices_reject_adversarial_keys::<D>(&keys);

    assert_cospherical_ridge_star::<D>()?;
    assert_matches!(
        DelaunayTriangulation::try_new(&degenerate_prelude_vertices::<D>()?),
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
fn topology_spaces_prelude_covers_toroidal_domain_api() -> Result<(), PreludeExportTestError> {
    let domain = ToroidalDomain::<3>::try_new([1.0, 2.0, 3.0])?;
    assert_relative_eq!(domain.periods()[0], 1.0);
    assert_relative_eq!(domain.periods()[1], 2.0);
    assert_relative_eq!(domain.periods()[2], 3.0);
    assert_eq!(domain.period(1), Some(2.0));

    let construction_domain = ConstructionToroidalDomain::<3>::try_new([1.0, 2.0, 3.0])?;
    assert_relative_eq!(construction_domain.periods()[2], 3.0);

    let validation_topology = TopologyValidationGlobalTopology::<3>::default();
    assert!(validation_topology.is_euclidean());

    let topology = GlobalTopology::try_toroidal(
        [1.0, 2.0, 3.0],
        ToroidalConstructionMode::PeriodicImagePoint,
    )?;
    assert!(topology.is_toroidal());
    assert_send_sync_unpin::<ToroidalDomainError>();
    Ok(())
}

#[test]
fn triangulation_prelude_covers_generic_layer() -> Result<(), PreludeExportTestError> {
    let vertices = vec![
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0])?,
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
    tri.validate().unwrap();
    let _triangulation_all_facets: TriangulationAllFacetsIter<'_, (), (), 2> =
        tri.facets().unwrap();
    let _triangulation_boundary_facets: TriangulationBoundaryFacetsIter<'_, (), (), 2> =
        tri.boundary_facets().unwrap();

    let empty_issues = TriangulationFacetIssuesMap::default();
    let removed = tri
        .repair_local_facet_issues(&empty_issues, 0)
        .expect("empty issue set should not fail generic local repair");
    assert_eq!(removed, 0);
    assert_eq!(
        tri.boundary_facets()
            .unwrap()
            .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))
            .unwrap(),
        0
    );

    assert_send_sync_unpin::<TriangulationInsertionError>();
    assert_send_sync_unpin::<TriangulationQueryError>();
    assert_send_sync_unpin::<TriangulationTdsError>();
    let generic_spatial_failure = GenericTriangulationConstructionError::SpatialIndexConstruction {
        reason: GenericSpatialIndexConstructionFailure::NonFiniteCellSize {
            value: ConstructionInvalidCoordinateValue::Nan,
        },
    };
    assert_matches!(
        generic_spatial_failure,
        GenericTriangulationConstructionError::SpatialIndexConstruction {
            reason: GenericSpatialIndexConstructionFailure::NonFiniteCellSize {
                value: ConstructionInvalidCoordinateValue::Nan,
            },
        }
    );
    Ok(())
}

#[test]
fn construction_prelude_covers_random_point_generation_failure_variant()
-> Result<(), PreludeExportTestError> {
    let points = [Point::try_new([0.0, 0.0])?, Point::try_new([1.0, 0.0])?];
    let root_prelude_vertices = prelude_try_vertices_from_points(&points)?;
    let construction_prelude_vertices = construction_try_vertices_from_points(&points)?;
    assert_eq!(root_prelude_vertices.len(), 2);
    assert_eq!(construction_prelude_vertices.len(), 2);

    let width_overflow = GeneratorRandomPointGenerationError::CoordinateRangeWidthOverflow {
        min: -f64::MAX,
        max: f64::MAX,
    };
    assert_matches!(
        width_overflow,
        GeneratorRandomPointGenerationError::CoordinateRangeWidthOverflow { min, max }
            if min.to_bits() == (-f64::MAX).to_bits() && max.to_bits() == f64::MAX.to_bits()
    );

    assert_matches!(
        DelaunayConstructionFailure::RandomPointGeneration {
            source: RandomPointGenerationError::InvalidCoordinateRange {
                source: ConstructionCoordinateRangeError::NonIncreasing {
                    ordering: ConstructionCoordinateRangeOrdering::Decreasing,
                    min: 1.0,
                    max: 0.0,
                },
            },
        },
        DelaunayConstructionFailure::RandomPointGeneration { .. }
    );

    assert_matches!(
        DelaunayConstructionFailure::RandomPointGeneration {
            source: RandomPointGenerationError::InvalidBallRadius {
                reason: ConstructionInvalidPositiveScalar::NonFinite {
                    value: ConstructionInvalidCoordinateValue::Nan,
                },
            },
        },
        DelaunayConstructionFailure::RandomPointGeneration { .. }
    );

    let spatial_failure = DelaunayConstructionFailure::SpatialIndexConstruction {
        reason: ConstructionSpatialIndexConstructionFailure::NonPositiveCellSize {
            value: CoordinateConversionValue::from_f64(0.0),
        },
    };
    assert_matches!(
        spatial_failure,
        DelaunayConstructionFailure::SpatialIndexConstruction {
            reason: ConstructionSpatialIndexConstructionFailure::NonPositiveCellSize { value },
        } if value == CoordinateConversionValue::from_f64(0.0)
    );

    assert_matches!(
        DelaunayConstructionFailure::FinalDelaunayValidation {
            source: ConstructionDelaunayTriangulationValidationError::VerificationFailed {
                source: ConstructionDelaunayVerificationError::from(
                    DelaunayRepairError::PostconditionFailed {
                        message: "synthetic final Level 4 failure".to_string(),
                    },
                )
                .into(),
            },
        },
        DelaunayConstructionFailure::FinalDelaunayValidation {
            source: ConstructionDelaunayTriangulationValidationError::VerificationFailed {
                source,
            },
        } if source.to_string().contains("synthetic final Level 4 failure")
    );

    let validation_summary = ExplicitDelaunayValidationError::from(
        ConstructionDelaunayTriangulationValidationError::VerificationFailed {
            source: ConstructionDelaunayVerificationError::from(
                DelaunayRepairError::PostconditionFailed {
                    message: "synthetic Level 4 summary failure".to_string(),
                },
            )
            .into(),
        },
    );
    assert_eq!(
        validation_summary.source_kind,
        Some(ExplicitDelaunayValidationSourceKind::Verification(
            ConstructionDelaunayVerificationErrorKind::FlipPredicates,
        ))
    );

    Ok(())
}

#[test]
fn diagnostic_preludes_cover_repair_apis() -> Result<(), PreludeExportTestError> {
    let vertices: Vec<Vertex<(), 3>> = vec![
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
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
    let _typed_outcome: DelaunayizeOutcome<(), (), 3> = outcome;
    let _typed_error: Option<DelaunayizeError> = None;
    Ok(())
}

#[cfg(feature = "diagnostics")]
#[test]
fn diagnostics_prelude_covers_opt_in_helpers() -> Result<(), PreludeExportTestError> {
    let tds: Tds<(), (), 2> = Tds::empty();
    debug_print_first_delaunay_violation(&tds, None);
    let report = delaunay_violation_report(&tds, None)?;
    let _typed_report: DelaunayViolationReport = report;
    let _typed_detail: Option<DelaunayViolationDetail> = None;
    assert!(DiagnosticNeighborSlot::Boundary.is_boundary());

    let kernel = AdaptiveKernel::new();
    let point = Point::try_new([0.0, 0.0])?;
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
    let order = try_hilbert_sorted_indices(&coords, (0.0, 1.0), bits)?;
    assert_eq!(order.len(), coords.len());
    let bounds = CoordinateRange::try_new(0.0_f64, 1.0).expect("test bounds should be valid");
    let range_order = hilbert_sorted_indices_in_range(&coords, bounds, bits)?;
    assert_eq!(range_order, order);

    let quantized: Vec<[u32; 2]> = coords
        .iter()
        .map(|coord| try_hilbert_quantize(coord, (0.0, 1.0), bits))
        .collect::<Result<_, _>>()?;
    let range_quantized = hilbert_quantize_in_range(&coords[0], bounds, bits)?;
    assert_eq!(range_quantized, quantized[0]);
    let indices = hilbert_indices_prequantized(&quantized, bits)?;
    assert_eq!(indices.len(), coords.len());
    let quantized_batch = HilbertQuantizedBatch::try_new(&quantized, bits)?;
    assert_eq!(quantized_batch.coordinates(), quantized.as_slice());
    assert_eq!(quantized_batch.bits(), bits);
    assert_eq!(quantized_batch.indices(), indices);
    assert_eq!(
        hilbert_indices_for_quantized_batch(quantized_batch),
        indices
    );

    let index = try_hilbert_index(&coords[0], (0.0, 1.0), bits)?;
    assert_eq!(index, indices[0]);
    let range_index = hilbert_index_in_range(&coords[0], bounds, bits)?;
    assert_eq!(range_index, index);

    let mut stable_payload = vec![0_usize, 1, 2];
    try_hilbert_sort_by_stable(&mut stable_payload, (0.0, 1.0), bits, |&i| coords[i])?;
    assert_eq!(stable_payload, order);
    let mut range_stable_payload = vec![0_usize, 1, 2];
    hilbert_sort_by_stable_in_range(&mut range_stable_payload, bounds, bits, |&i| coords[i])?;
    assert_eq!(range_stable_payload, order);

    let mut unstable_payload = vec![0_usize, 1, 2];
    try_hilbert_sort_by_unstable(&mut unstable_payload, (0.0, 1.0), bits, |&i| coords[i])?;
    assert_eq!(unstable_payload, order);
    let mut range_unstable_payload = vec![0_usize, 1, 2];
    hilbert_sort_by_unstable_in_range(&mut range_unstable_payload, bounds, bits, |&i| coords[i])?;
    assert_eq!(range_unstable_payload, order);

    Ok(())
}
