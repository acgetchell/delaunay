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

use std::{assert_matches, error::Error, mem::size_of, num::NonZeroUsize};

use approx::{abs_diff_eq, assert_relative_eq};
use slotmap::KeyData;

use delaunay::builder::DelaunayTriangulationBuilder as BuilderModuleBuilder;
use delaunay::construction::{
    ConstructionOptions as ConstructionModuleOptions,
    InsertionOrderStrategy as ConstructionModuleInsertionOrderStrategy,
};
use delaunay::delaunayize::{
    DelaunayizeConfig as DelaunayizeModuleConfig, DelaunayizeError as DelaunayizeModuleError,
    delaunayize_by_flips as module_delaunayize_by_flips,
};
use delaunay::flips::{
    BistellarFlips, DelaunayRepairError as DirectDelaunayRepairError,
    FlipFailureKind as DirectFlipFailureKind,
    FlipOrientationCheckStage as DirectFlipOrientationCheckStage,
};
use delaunay::geometry::{
    CoordinateConversionError as GeometryModuleCoordinateConversionError,
    CoordinateRange as GeometryCoordinateRange,
    LabeledSimplexEmbedding as GeometryModuleLabeledSimplexEmbedding,
    validate_simplex_embeddings_intersect_only_in_shared_faces as geometry_module_validate_simplex_embeddings_intersect_only_in_shared_faces,
};
use delaunay::pachner::PachnerMoves as DirectPachnerMoves;
use delaunay::prelude::DelaunayValidationError;
use delaunay::prelude::algorithms::LocateResult;
#[cfg(feature = "diagnostics")]
use delaunay::prelude::collections::SimplexKeyBuffer;
use delaunay::prelude::collections::{
    SecureHashMap as ScopedSecureHashMap, SecureHashSet as ScopedSecureHashSet, Uuid,
};
use delaunay::prelude::construction::{
    CavityFillingError, CavityRepairStage, ConstructionOptions, ConstructionSkipSample,
    ConstructionSlowInsertionSample, CoordinateRangeError as ConstructionCoordinateRangeError,
    CoordinateRangeOrdering as ConstructionCoordinateRangeOrdering,
    CoordinateValidationError as ConstructionCoordinateValidationError, DedupPolicy,
    DedupTolerance, DeduplicationError, DelaunayConstructionFailure,
    DelaunayConstructionRetryFailure, DelaunayError, DelaunayRepairPolicy, DelaunayResult,
    DelaunayTriangulation, DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError,
    DelaunayTriangulationValidationError as ConstructionDelaunayTriangulationValidationError,
    DelaunayVerificationError as ConstructionDelaunayVerificationError, DeleteVertexError,
    ExplicitConstructionError, FinalDelaunayValidationContext, FinalTopologyValidationContext,
    GlobalTopologyModelError as ConstructionGlobalTopologyModelError, InsertionOrderStrategy,
    InvalidCoordinateValue as ConstructionInvalidCoordinateValue,
    InvalidPositiveScalar as ConstructionInvalidPositiveScalar, RandomPointGenerationError,
    SimplexValidationError,
    SpatialIndexConstructionFailure as ConstructionSpatialIndexConstructionFailure,
    TopologyGuarantee, ToroidalDomain as ConstructionToroidalDomain, Vertex, VertexValidationError,
    try_vertices_from_points as construction_try_vertices_from_points, vertex,
};
use delaunay::prelude::delaunayize::{
    DelaunayTriangulationBuilder as DelaunayizeDelaunayTriangulationBuilder, DelaunayizeConfig,
    DelaunayizeError, DelaunayizeOutcome, PlManifoldRepairError, PlManifoldRepairStage,
    PlManifoldRepairStats, SimplexDataRestoreError, delaunayize_by_flips,
};
use delaunay::prelude::deletion::{
    DeleteVertexError as FocusedDeleteVertexError, VertexKey as DeletionVertexKey,
};
use delaunay::prelude::diagnostics::ConstructionTelemetry;
#[cfg(feature = "diagnostics")]
use delaunay::prelude::diagnostics::{
    DelaunayViolationDetail, DelaunayViolationReport, NeighborSlot as DiagnosticNeighborSlot,
    debug_print_first_delaunay_violation, delaunay_violation_report,
    verify_conflict_region_completeness,
};
use delaunay::prelude::export::{
    InvalidCoordinateValue as ExportPreludeInvalidCoordinateValue,
    MESH_EXPORT_SCHEMA as ExportPreludeMeshExportSchema, MeshExport as ExportPreludeMeshExport,
    MeshExportError as ExportPreludeMeshExportError,
    MeshExportValidationError as ExportPreludeMeshExportValidationError,
    ValidatedMeshExport as ExportPreludeValidatedMeshExport,
    ValidatedVisualizationData as ExportPreludeValidatedVisualizationData,
    VisualizationData as ExportPreludeVisualizationData,
    VisualizationTopologyGuarantee as ExportPreludeVisualizationTopologyGuarantee,
    VisualizationTopologyKind as ExportPreludeVisualizationTopologyKind,
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
    CoordinateValues, DegenerateGeometry, DegenerateMeasure, DegenerateSimplexReason,
    FiniteCoordinateValue, InvalidCoordinateValue, LaError, LabeledSimplexEmbedding,
    LabeledSimplexEmbeddingError, MatrixError, PeriodicSimplexSpan, PeriodicSimplexSpanError,
    Point, QualitySimplexVerticesError, SimplexEmbeddingBuffer, SimplexIntersectionFailure,
    SimplexIntersectionWitness, SurfaceMeasureError, ValueConversionError,
    ValueConversionFailureReason, axis_aligned_bounding_boxes_overlap, coordinate_range_for_axis,
    try_periodic_simplex_span, validate_simplex_embeddings_intersect_only_in_shared_faces,
};
use delaunay::prelude::insertion::{
    InitialSimplexConstructionError, InitialSimplexUnexpectedInsertionStage, InsertionError,
    InsertionErrorKind as FocusedInsertionErrorKind, InsertionTopologyValidationContext,
    NeighborRebuildError, Tds as InsertionTds, TdsMutationError, repair_neighbor_pointers_local,
};
use delaunay::prelude::ordering::{
    HilbertBitDepth, HilbertError, HilbertQuantizedBatch, MAX_HILBERT_BITS, hilbert_index_in_range,
    hilbert_indices_for_quantized_batch, hilbert_indices_prequantized, hilbert_quantize_in_range,
    hilbert_sort_by_stable_in_range, hilbert_sort_by_unstable_in_range,
    hilbert_sorted_indices_in_range, try_hilbert_index, try_hilbert_quantize,
    try_hilbert_sort_by_stable, try_hilbert_sort_by_unstable, try_hilbert_sorted_indices,
};
use delaunay::prelude::pachner::{
    BistellarFlipKind as PachnerBistellarFlipKind, EdgeKey as PachnerEdgeKey,
    EdgeKeyError as PachnerEdgeKeyError, FacetError as PachnerFacetError,
    FacetHandle as PachnerFacetHandle, FlipDirection as PachnerFlipDirection,
    FlipError as PachnerFlipError, PachnerMove, PachnerMoveFeasibility, PachnerMoveResult,
    PachnerMoves, RidgeHandle as PachnerRidgeHandle, SimplexKey as PachnerSimplexKey,
    TriangleHandle as PachnerTriangleHandle, TriangleHandleError as PachnerTriangleHandleError,
    Vertex as PachnerVertex, VertexKey as PachnerVertexKey, vertex as pachner_vertex,
};
use delaunay::prelude::query::{
    AllFacetsIter as QueryAllFacetsIter, BoundaryFacetsIter as QueryBoundaryFacetsIter, ConvexHull,
    ConvexHullConstructionError, EdgeIndex as QueryEdgeIndex, EdgeKey as QueryEdgeKey,
    EdgeView as QueryEdgeView, FacetHandle as QueryFacetHandle,
    FacetIncidenceAnalysis as QueryFacetIncidenceAnalysis,
    FacetIncidenceView as QueryFacetIncidenceView, IncidenceView as QueryIncidenceView,
    OneSidedFacetsIter as QueryOneSidedFacetsIter, QueryError,
    SimplexFacetsIter as QuerySimplexFacetsIter, SimplexNeighborIndex as QuerySimplexNeighborIndex,
    TopologyIndexBuildError, TriangulationAdjacency as QueryTriangulationAdjacency,
};
use delaunay::prelude::repair::{
    DelaunayCheckPolicy, DelaunayRepairDiagnostics, DelaunayRepairError,
    DelaunayRepairHeuristicRebuildFailure, DelaunayRepairHeuristicRebuildFailureKind,
    DelaunayRepairHeuristicVertexContext, DelaunayRepairOperation,
    DelaunayRepairOrientationCanonicalizationFailure,
    DelaunayRepairOrientationCanonicalizationFailureKind, DelaunayRepairOutcome,
    DelaunayRepairPostconditionFailure, DelaunayRepairStats, DelaunayRepairVerificationContext,
    DelaunayTriangulationValidationError, FlipEdgeAdjacencyError, FlipError, FlipFailureKind,
    FlipOrientationCheckStage as RepairFlipOrientationCheckStage, FlipTriangleAdjacencyError,
    FlipVertexAdjacencyError, RepairQueueOrder, verify_delaunay_for_triangulation,
};
use delaunay::prelude::tds::{
    AllFacetsIter as TdsAllFacetsIter, BoundaryFacetsIter as TdsBoundaryFacetsIter, EdgeKey,
    EdgeKeyError, EdgeView, FacetError, FacetHandle, FacetIncidenceView as TdsFacetIncidenceView,
    FacetView, InvariantError, NeighborSlot, OneSidedFacetsIter as TdsOneSidedFacetsIter,
    SimplexFacetsIter as TdsSimplexFacetsIter, SimplexKey, Tds, TdsConstructionError, TdsError,
    VertexKey,
};
use delaunay::prelude::topology::spaces::{
    GlobalTopology, GlobalTopologyModelError, LiftedLinkEdge, LiftedVertexId, TopologyKind,
    ToroidalConstructionMode, ToroidalDomain, ToroidalDomainError,
};
use delaunay::prelude::topology::validation::{
    GlobalTopology as TopologyValidationGlobalTopology, ManifoldError, RidgeCandidate,
    RidgeCandidateError, RidgeLinkView, RidgeQuery, RidgeView, ridge_star_simplices,
};
use delaunay::prelude::triangulation::{
    AllFacetsIter as TriangulationAllFacetsIter,
    BoundaryFacetsIter as TriangulationBoundaryFacetsIter, EdgeIndex as GenericEdgeIndex,
    FacetIssuesMap as TriangulationFacetIssuesMap, FastKernel as TriangulationFastKernel,
    IncidenceView as GenericIncidenceView, InsertionError as TriangulationInsertionError,
    OneSidedFacetsIter as TriangulationOneSidedFacetsIter, QueryError as TriangulationQueryError,
    SimplexFacetsIter as GenericSimplexFacetsIter,
    SimplexNeighborIndex as GenericSimplexNeighborIndex,
    SpatialIndexConstructionFailure as GenericSpatialIndexConstructionFailure,
    TdsError as TriangulationTdsError, TopologyGuarantee as TriangulationTopologyGuarantee,
    Triangulation as GenericTriangulation, TriangulationAdjacency as GenericTriangulationAdjacency,
    TriangulationConstructionError as GenericTriangulationConstructionError,
    ValidationConfigurationError as TriangulationValidationConfigurationError,
    ValidationPolicy as TriangulationValidationPolicy, vertex as triangulation_vertex,
};
use delaunay::prelude::try_vertices_from_points as prelude_try_vertices_from_points;
use delaunay::prelude::validation::{
    DelaunayValidationError as FocusedDelaunayValidationError,
    DelaunayViolationDetail as FocusedDelaunayViolationDetail,
    DelaunayViolationReport as FocusedDelaunayViolationReport,
    PeriodicDomainPeriodError as FocusedPeriodicDomainPeriodError,
    TopologyGuarantee as FocusedValidationTopologyGuarantee,
    TriangulationValidationReport as FocusedValidationReport, ValidationCadence,
    ValidationConfigurationError as FocusedValidationConfigurationError,
    ValidationPolicy as FocusedValidationPolicy,
    delaunay_violation_report as focused_delaunay_violation_report,
    find_delaunay_violations as focused_find_delaunay_violations,
};
use delaunay::prelude::{
    CoordinateRange as RootCoordinateRange, DelaunayError as RootDelaunayError,
    DelaunayResult as RootDelaunayResult, DelaunayTriangulation as RootDelaunayTriangulation,
    DelaunayTriangulationBuilder as RootDelaunayTriangulationBuilder,
    DelaunayViolationDetail as RootDelaunayViolationDetail,
    DelaunayViolationReport as RootDelaunayViolationReport, EdgeIndex as RootEdgeIndex,
    FacetIncidenceView as RootFacetIncidenceView, FlipFailureKind as RootFlipFailureKind,
    FlipOrientationCheckStage as RootFlipOrientationCheckStage,
    GlobalTopology as RootGlobalTopology, GlobalTopologyModelError as RootGlobalTopologyModelError,
    IncidenceView as RootIncidenceView,
    InitialSimplexUnexpectedInsertionStage as RootInitialSimplexUnexpectedInsertionStage,
    PeriodicDomainPeriodError as RootPeriodicDomainPeriodError,
    PlManifoldRepairStage as RootPreludePlManifoldRepairStage, SecureHashMap, SecureHashSet,
    SimplexNeighborIndex as RootSimplexNeighborIndex, TopologyError as RootTopologyError,
    TopologyGuarantee as RootTopologyGuarantee, TopologyKind as RootTopologyKind,
    ToroidalConstructionMode as RootToroidalConstructionMode, ToroidalDomain as RootToroidalDomain,
    ToroidalDomainError as RootToroidalDomainError,
    TriangulationAdjacency as RootTriangulationAdjacency,
    TriangulationValidationReport as RootTriangulationValidationReport,
    ValidationConfigurationError as RootValidationConfigurationError,
    ValidationPolicy as RootValidationPolicy,
    delaunay_violation_report as root_delaunay_violation_report, vertex as root_vertex,
};
use delaunay::query::{
    AllFacetsIter as QueryFacadeAllFacetsIter, BoundaryFacetsIter as QueryFacadeBoundaryFacetsIter,
    EdgeIndex as QueryFacadeEdgeIndex, FacetHandle as QueryFacadeFacetHandle,
    IncidenceView as QueryFacadeIncidenceView, OneSidedFacetsIter as QueryFacadeOneSidedFacetsIter,
    SimplexFacetsIter as QueryFacadeSimplexFacetsIter,
    SimplexNeighborIndex as QueryFacadeSimplexNeighborIndex,
    TriangulationAdjacency as QueryFacadeTriangulationAdjacency,
};
use delaunay::repair::{
    DelaunayCheckPolicy as RepairModuleDelaunayCheckPolicy,
    DelaunayRepairPolicy as RepairModuleDelaunayRepairPolicy,
};
use delaunay::topology::{
    BoundaryFacetClassification as TopologyBoundaryFacetClassification,
    classify_boundary_facet as topology_classify_boundary_facet,
};
use delaunay::validation::{
    DelaunayTriangulationValidationError as ValidationModuleDelaunayTriangulationValidationError,
    ValidationCadence as ValidationModuleCadence,
};
use delaunay::{
    ConstructionOptions as RootConstructionOptions,
    DelaunayConstructionRetryFailure as RootConstructionRetryFailure,
    DelaunayTriangulationConstructionError as RootDelaunayTriangulationConstructionError,
    DelaunayTriangulationValidationError as RootDelaunayTriangulationValidationError,
    MESH_EXPORT_SCHEMA as RootMeshExportSchema, MeshExport as RootMeshExport,
    MeshExportError as RootMeshExportError,
    MeshExportValidationError as RootMeshExportValidationError,
    PlManifoldRepairStage as RootPlManifoldRepairStage,
    ValidatedMeshExport as RootValidatedMeshExport,
    ValidatedVisualizationData as RootValidatedVisualizationData,
    VisualizationTopologyGuarantee as RootVisualizationTopologyGuarantee,
    VisualizationTopologyKind as RootVisualizationTopologyKind,
};
#[derive(Debug, thiserror::Error)]
enum RootApiExportTestError {
    #[error(transparent)]
    Construction(#[from] RootDelaunayTriangulationConstructionError),
    #[error(transparent)]
    CoordinateConversion(#[from] GeometryModuleCoordinateConversionError),
    #[error(transparent)]
    Validation(#[from] RootDelaunayTriangulationValidationError),
    #[error(transparent)]
    DelaunayRepair(#[from] DirectDelaunayRepairError),
    #[error(transparent)]
    Delaunayize(#[from] DelaunayizeModuleError),
    #[error(transparent)]
    MeshExport(#[from] RootMeshExportError),
    #[error(transparent)]
    MeshExportValidation(#[from] RootMeshExportValidationError),
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
    MeshExport(#[from] ExportPreludeMeshExportError),
    #[error(transparent)]
    MeshExportValidation(#[from] ExportPreludeMeshExportValidationError),
    #[error(transparent)]
    Insertion(#[from] InsertionError),
    #[error(transparent)]
    Manifold(#[from] ManifoldError),
    #[error(transparent)]
    Query(#[from] QueryError),
    #[error(transparent)]
    ConvexHull(#[from] ConvexHullConstructionError),
    #[error(transparent)]
    TopologyIndex(#[from] TopologyIndexBuildError),
    #[error(transparent)]
    Facet(#[from] FacetError),
    #[error(transparent)]
    Tds(#[from] TdsError),
    #[error(transparent)]
    Edge(#[from] EdgeKeyError),
    #[error(transparent)]
    PachnerFlip(#[from] PachnerFlipError),
    #[error(transparent)]
    RidgeCandidate(#[from] RidgeCandidateError),
    #[error(transparent)]
    ToroidalDomain(#[from] ToroidalDomainError),
    #[error(transparent)]
    GenericTriangulationConstruction(#[from] GenericTriangulationConstructionError),
}

/// Proves the direct flips module keeps the expert trait bound available.
const fn assert_bistellar_flips(_: &impl BistellarFlips<3, VertexData = ()>) {}

/// Proves the root flips module exports the same public trait bound.
const fn assert_root_bistellar_flips(_: &impl BistellarFlips<3, VertexData = ()>) {}

struct NonKernelMarker;

/// Proves explicit topology edits do not require kernel-backed predicates.
const fn assert_bistellar_flips_without_kernel<T: BistellarFlips<2, VertexData = ()>>() {}

/// Proves the focused Pachner prelude exports the unified workflow trait.
const fn assert_pachner_moves(_: &impl PachnerMoves<3, VertexData = ()>) {}

/// Proves the root Pachner module exports the same unified workflow trait.
const fn assert_root_pachner_moves(_: &impl DirectPachnerMoves<3, VertexData = ()>) {}

/// Proves unified Pachner dispatch inherits the kernel-free explicit flip contract.
const fn assert_pachner_moves_without_kernel<T: PachnerMoves<2, VertexData = ()>>() {}

/// Proves unified Pachner dispatch is available through unsized flip trait objects.
const fn assert_pachner_moves_for_unsized_flip_trait_objects<
    T: PachnerMoves<2, VertexData = ()> + ?Sized,
>() {
}

const fn assert_pachner_prelude_type_exports(dt: &impl PachnerMoves<3, VertexData = ()>) {
    assert_pachner_moves(dt);
    assert_root_pachner_moves(dt);
    let _pachner_kind_size = size_of::<PachnerBistellarFlipKind>();
    let _pachner_direction_size = size_of::<PachnerFlipDirection>();
    let _pachner_error_size = size_of::<PachnerFlipError>();
    let _pachner_move_size = size_of::<PachnerMove<(), 3>>();
    let _pachner_move_feasibility_size = size_of::<PachnerMoveFeasibility<3>>();
    let _pachner_result_size = size_of::<PachnerMoveResult<3>>();
    let _pachner_edge_size = size_of::<PachnerEdgeKey>();
    let _pachner_edge_error_size = size_of::<PachnerEdgeKeyError>();
    let _pachner_facet_error_size = size_of::<PachnerFacetError>();
    let _pachner_facet_size = size_of::<PachnerFacetHandle>();
    let _pachner_ridge_size = size_of::<PachnerRidgeHandle>();
    let _pachner_simplex_size = size_of::<PachnerSimplexKey>();
    let _pachner_triangle_size = size_of::<PachnerTriangleHandle>();
    let _pachner_triangle_error_size = size_of::<PachnerTriangleHandleError>();
    let _pachner_vertex_size = size_of::<PachnerVertex<(), 3>>();
    let _pachner_vertex_key_size = size_of::<PachnerVertexKey>();
}

fn assert_pachner_prelude_exports(
    dt: &impl PachnerMoves<3, VertexData = ()>,
    simplex_key: PachnerSimplexKey,
) -> Result<(), PreludeExportTestError> {
    assert_pachner_prelude_type_exports(dt);
    let pachner_vertex: PachnerVertex<(), 3> = pachner_vertex![0.25, 0.25, 0.25]?;
    let pachner_move = PachnerMove::K1Insert {
        simplex_key,
        vertex: pachner_vertex,
    };
    assert_matches!(
        pachner_move,
        PachnerMove::K1Insert { simplex_key: key, .. } if key == simplex_key
    );
    let feasibility = dt.can_attempt_pachner(&pachner_move)?;
    assert_eq!(feasibility.kind, PachnerBistellarFlipKind::k1(3));
    assert!(feasibility.inserted_face_vertices.is_none());
    Ok(())
}

fn assert_delaunayize_prelude_repair_exports() {
    let _typed_repair_stats: PlManifoldRepairStats<(), (), 3> = PlManifoldRepairStats::default();
    let repair_stage = PlManifoldRepairStage::RidgeLink;
    assert_eq!(repair_stage, PlManifoldRepairStage::RidgeLink);
    let repair_error = PlManifoldRepairError::TargetedPostconditionValidation {
        source: Box::new(ManifoldError::ManifoldFacetMultiplicity {
            facet_key: 0x1234,
            simplex_count: 3,
        }),
    };
    assert!(repair_error.to_string().contains("postcondition"));
}

const fn assert_send_sync_unpin<T: Send + Sync + Unpin>() {}

const fn assert_error<T: Error>() {}

const fn assert_query_facet_incidence_trait_export<T>(_: &T)
where
    T: QueryFacetIncidenceAnalysis<(), (), 3> + ?Sized,
{
}

fn assert_construction_prelude_unsupported_topology_variants() {
    let unsupported_euclidean_topology =
        DelaunayConstructionFailure::EuclideanUnsupportedGlobalTopology {
            topology: TopologyKind::Spherical,
        };
    assert_matches!(
        unsupported_euclidean_topology,
        DelaunayConstructionFailure::EuclideanUnsupportedGlobalTopology {
            topology: TopologyKind::Spherical,
        }
    );
    let unsupported_canonicalized_topology =
        DelaunayConstructionFailure::CanonicalizedUnsupportedGlobalTopology {
            topology: TopologyKind::Toroidal,
        };
    assert_matches!(
        unsupported_canonicalized_topology,
        DelaunayConstructionFailure::CanonicalizedUnsupportedGlobalTopology {
            topology: TopologyKind::Toroidal,
        }
    );
    let conflicting_periodic_topology =
        DelaunayConstructionFailure::PeriodicImageConflictingGlobalTopology {
            requested_topology: TopologyKind::Euclidean,
            requested_mode: None,
            requested_periods: None,
            expected_mode: ToroidalConstructionMode::PeriodicImagePoint,
            expected_periods: vec![1.0, 1.0],
        };
    assert_matches!(
        conflicting_periodic_topology,
        DelaunayConstructionFailure::PeriodicImageConflictingGlobalTopology {
            requested_topology: TopologyKind::Euclidean,
            requested_mode: None,
            requested_periods: None,
            expected_mode: ToroidalConstructionMode::PeriodicImagePoint,
            expected_periods,
        }
        if expected_periods.as_slice() == [1.0, 1.0]
    );
}

#[test]
fn construction_prelude_exports_common_delaunay_error_aliases() {
    let source = CoordinateConversionError::InvalidSimplexPointCount {
        actual: 2,
        expected: 3,
        dimension: 2,
    };

    let focused_error = DelaunayError::from(source.clone());
    assert_matches!(
        focused_error,
        DelaunayError::CoordinateConversion { source: err } if err.as_ref() == &source
    );

    let root_error = RootDelaunayError::from(source.clone());
    assert_matches!(
        root_error,
        RootDelaunayError::CoordinateConversion { source: err } if err.as_ref() == &source
    );

    let no_vertices: [Vertex<(), 2>; 0] = [];
    let construction = DelaunayTriangulationBuilder::new(&no_vertices)
        .build::<()>()
        .expect_err("empty Delaunay construction should fail");
    assert_matches!(
        DelaunayError::from(construction),
        DelaunayError::Construction { source }
            if matches!(
                source.as_ref(),
                DelaunayTriangulationConstructionError::Triangulation(
                    DelaunayConstructionFailure::InsufficientVertices { dimension: 2, .. }
                )
            )
    );

    let insertion = InsertionError::DuplicateCoordinates {
        coordinates: CoordinateValues::from([0.0, 0.0]),
    };
    assert_matches!(
        DelaunayError::from(insertion.clone()),
        DelaunayError::Insertion { source: err } if err.as_ref() == &insertion
    );

    let delete_vertex = DeleteVertexError::VertexNotFound {
        vertex_key: VertexKey::from(KeyData::from_ffi(2)),
    };
    assert_matches!(
        DelaunayError::from(delete_vertex.clone()),
        DelaunayError::DeleteVertex { source: err } if err.as_ref() == &delete_vertex
    );

    let flip = FlipError::DegenerateSimplex;
    assert_matches!(
        DelaunayError::from(flip.clone()),
        DelaunayError::Flip { source: err } if err.as_ref() == &flip
    );

    let tds_mutation = TdsMutationError::from(TdsError::SimplexNotFound {
        simplex_key: SimplexKey::from(KeyData::from_ffi(3)),
        context: "prelude smoke test".to_owned(),
    });
    assert_matches!(
        DelaunayError::from(tds_mutation.clone()),
        DelaunayError::TdsMutation { source: err } if err.as_ref() == &tds_mutation
    );
}

#[test]
fn construction_prelude_exports_validation_and_result_aliases() {
    let configuration =
        FocusedValidationConfigurationError::IncompatibleTopologyAndValidationPolicy {
            topology_guarantee: TopologyGuarantee::PLManifold,
            validation_policy: FocusedValidationPolicy::Never,
        };
    let expected_configuration =
        FocusedValidationConfigurationError::IncompatibleTopologyAndValidationPolicy {
            topology_guarantee: TopologyGuarantee::PLManifold,
            validation_policy: FocusedValidationPolicy::Never,
        };
    assert_matches!(
        DelaunayError::from(configuration),
        DelaunayError::ValidationConfiguration { source: err }
            if err.as_ref() == &expected_configuration
    );

    let toroidal_domain = ToroidalDomainError::InvalidPeriod {
        axis: 0,
        period: 0.0,
    };
    assert_matches!(
        DelaunayError::from(toroidal_domain),
        DelaunayError::ToroidalDomain { source }
            if matches!(
                source.as_ref(),
                ToroidalDomainError::InvalidPeriod { axis: 0, period }
                    if period.to_bits() == 0.0_f64.to_bits()
            )
    );

    let simplex_key = SimplexKey::from(KeyData::from_ffi(1));
    let validation = ConstructionDelaunayTriangulationValidationError::VerificationFailed {
        source: Box::new(ConstructionDelaunayVerificationError::from(
            DelaunayValidationError::DelaunayViolation {
                simplex_key,
                simplex_vertices: Box::default(),
                offending_vertex: None,
                neighbor_simplices: Box::default(),
            },
        )),
    };
    assert_matches!(
        DelaunayError::from(validation.clone()),
        DelaunayError::Validation { source: err } if err.as_ref() == &validation
    );

    let focused_result: DelaunayResult<()> = Ok(());
    let root_result: RootDelaunayResult<()> = focused_result;
    assert!(root_result.is_ok());
}

#[test]
fn deletion_prelude_exports_delete_vertex_error_and_key() {
    let err = FocusedDeleteVertexError::VertexNotFound {
        vertex_key: DeletionVertexKey::from(KeyData::from_ffi(9)),
    };

    assert_matches!(err, FocusedDeleteVertexError::VertexNotFound { .. });
}

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
fn construction_prelude_covers_vertex_macro() -> Result<(), PreludeExportTestError> {
    let vertex: Vertex<(), 2> = vertex![0.0, 0.0]?;
    let bracketed: Vertex<(), 2> = vertex!([1.0, 0.0])?;
    let labeled: Vertex<&str, 2> = vertex![0.0, 1.0; data = "boundary"]?;
    let bracketed_labeled: Vertex<String, 2> = vertex!([1.0, 1.0]; data = String::from("corner"),)?;
    let root: Vertex<(), 2> = root_vertex![0.25, 0.75]?;
    let crate_root: Vertex<(), 2> = delaunay::vertex![0.5, 0.5]?;

    assert_relative_eq!(vertex.point().coords().as_slice(), [0.0, 0.0].as_slice());
    assert_relative_eq!(bracketed.point().coords().as_slice(), [1.0, 0.0].as_slice());
    assert_eq!(labeled.data(), Some(&"boundary"));
    assert_relative_eq!(
        bracketed_labeled.point().coords().as_slice(),
        [1.0, 1.0].as_slice()
    );
    assert_eq!(bracketed_labeled.data().map(String::as_str), Some("corner"));
    assert_relative_eq!(root.point().coords().as_slice(), [0.25, 0.75].as_slice());
    assert_relative_eq!(
        crate_root.point().coords().as_slice(),
        [0.5, 0.5].as_slice()
    );
    std::assert_matches!(
        vertex![f64::NAN, 0.0],
        Err(CoordinateConversionError::NonFiniteValue {
            coordinate_index: 0,
            ..
        })
    );
    Ok(())
}

#[test]
fn construction_prelude_covers_typed_construction_failure_variants() {
    assert_eq!(
        FinalTopologyValidationContext::ConstructionFinalize.to_string(),
        "topology validation failed after construction"
    );
    assert_eq!(
        FinalDelaunayValidationContext::PeriodicQuotientDelaunay.to_string(),
        "periodic quotient failed final Level 5 Delaunay validation"
    );
    assert_eq!(
        InsertionTopologyValidationContext::PostInsertion.to_string(),
        "post-insertion topology validation failed"
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
    assert_matches!(
        unsupported_periodic_dimension,
        DelaunayConstructionFailure::UnsupportedPeriodicDimension {
            dimension: 4,
            max_validated_dimension: 3,
            tracking_issue: 416,
        }
    );
    assert_construction_prelude_unsupported_topology_variants();
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
fn construction_prelude_covers_retry_exhaustion_source() {
    let retry_failure = DelaunayConstructionFailure::ShuffledRetryExhausted {
        attempt_count: 7,
        source: Box::new(DelaunayConstructionRetryFailure::Construction {
            source: Box::new(DelaunayTriangulationConstructionError::Triangulation(
                DelaunayConstructionFailure::GeometricDegeneracy {
                    message: "collinear".to_string(),
                },
            )),
        }),
    };

    assert_matches!(
        &retry_failure,
        DelaunayConstructionFailure::ShuffledRetryExhausted {
            attempt_count: 7,
            source,
        } if matches!(
            source.as_ref(),
            DelaunayConstructionRetryFailure::Construction { source }
                if matches!(
                    source.as_ref(),
                    DelaunayTriangulationConstructionError::Triangulation(
                        DelaunayConstructionFailure::GeometricDegeneracy { message }
                    ) if message == "collinear"
                )
        )
    );
    assert!(
        retry_failure
            .to_string()
            .contains("7 construction attempts, including the initial input order")
    );
    assert!(
        !retry_failure
            .to_string()
            .contains("shuffled reconstruction attempts")
    );
}

#[test]
fn root_exports_cover_flattened_public_api() -> Result<(), RootApiExportTestError> {
    let vertices = vec![
        vertex![0.0, 0.0, 0.0]?,
        vertex![1.0, 0.0, 0.0]?,
        vertex![0.0, 1.0, 0.0]?,
        vertex![0.0, 0.0, 1.0]?,
    ];

    let options: RootConstructionOptions = ConstructionModuleOptions::default()
        .with_insertion_order(ConstructionModuleInsertionOrderStrategy::Input);
    let builder: BuilderModuleBuilder<'_, (), 3> =
        RootDelaunayTriangulationBuilder::new(&vertices).construction_options(options);
    let mut dt: RootDelaunayTriangulation<_, (), (), 3> = builder.build::<()>()?;

    assert_eq!(dt.topology_guarantee(), RootTopologyGuarantee::PLManifold);
    assert_eq!(dt.validation_policy(), RootValidationPolicy::ExplicitOnly);
    assert_eq!(
        RootPreludePlManifoldRepairStage::RidgeLink,
        RootPlManifoldRepairStage::RidgeLink
    );
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
        RootConstructionRetryFailure::Construction {
            source: Box::new(DelaunayTriangulationConstructionError::Triangulation(
                DelaunayConstructionFailure::GeometricDegeneracy {
                    message: "synthetic".to_string(),
                },
            )),
        },
        RootConstructionRetryFailure::Construction { .. }
    );
    assert_matches!(
        ValidationModuleCadence::from_optional_every(Some(2)),
        ValidationModuleCadence::EveryN(every) if every.get() == 2
    );
    assert_eq!(
        RepairModuleDelaunayRepairPolicy::default(),
        RepairModuleDelaunayRepairPolicy::EveryInsertion
    );
    assert!(!RepairModuleDelaunayCheckPolicy::default().should_check(1));

    let validation_result: Result<(), ValidationModuleDelaunayTriangulationValidationError> =
        dt.validate();
    validation_result?;
    let root_mesh_export: RootMeshExport<3> = dt.to_mesh_export()?;
    assert_eq!(root_mesh_export.metadata.schema, RootMeshExportSchema);
    assert_eq!(
        root_mesh_export.metadata.topology_kind,
        RootVisualizationTopologyKind::Euclidean
    );
    assert_eq!(
        root_mesh_export.metadata.topology_guarantee,
        RootVisualizationTopologyGuarantee::PLManifold
    );
    let root_validated_export: RootValidatedMeshExport<3> = root_mesh_export.into_validated()?;
    assert_eq!(
        root_validated_export.metadata().schema,
        RootMeshExportSchema
    );
    let root_validated_data: RootValidatedVisualizationData<3> =
        dt.to_visualization_data()?.into_validated()?;
    assert_eq!(root_validated_data.metadata().schema, RootMeshExportSchema);
    assert_bistellar_flips(&dt);
    assert_root_bistellar_flips(&dt);

    let outcome = module_delaunayize_by_flips(&mut dt, DelaunayizeModuleConfig::default())?;
    assert!(!outcome.used_fallback_rebuild);
    assert!(outcome.topology_repair.succeeded);
    Ok(())
}

#[test]
fn flip_exports_cover_orientation_check_stage() {
    assert_bistellar_flips_without_kernel::<GenericTriangulation<NonKernelMarker, (), (), 2>>();
    assert_pachner_moves_without_kernel::<GenericTriangulation<NonKernelMarker, (), (), 2>>();
    assert_pachner_moves_for_unsized_flip_trait_objects::<dyn BistellarFlips<2, VertexData = ()>>();
    assert_matches!(
        DirectFlipOrientationCheckStage::BeforeMutation,
        DirectFlipOrientationCheckStage::BeforeMutation
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
        DirectFlipOrientationCheckStage::BeforeMutation,
        DirectFlipOrientationCheckStage::BeforeMutation
    );
}

fn assert_edge_view_exports(
    tds: &Tds<(), (), 3>,
    a: VertexKey,
    b: VertexKey,
) -> Result<(), PreludeExportTestError> {
    let edge_key = EdgeKey::try_new(tds, a, b)?;
    let query_edge_key: QueryEdgeKey = edge_key;
    let edge_view: EdgeView<'_, (), (), 3> = edge_key.view(tds)?;
    let query_edge_view: QueryEdgeView<'_, (), (), 3> = edge_key.view(tds)?;

    assert_eq!(query_edge_key, edge_key);
    assert_eq!(edge_view.key(), edge_key);
    assert_eq!(query_edge_view.key(), edge_key);
    assert!(!edge_view.incident_simplices().is_empty());
    Ok(())
}

fn assert_simplex_facet_iter_exports(
    tds: &Tds<(), (), 3>,
    simplex_key: SimplexKey,
) -> Result<(), FacetError> {
    let _query_facade_simplex_facets: QueryFacadeSimplexFacetsIter<'_, (), (), 3> =
        tds.try_simplex_facets(simplex_key)?;
    let _query_simplex_facets: QuerySimplexFacetsIter<'_, (), (), 3> =
        tds.try_simplex_facets(simplex_key)?;
    let _tds_simplex_facets: TdsSimplexFacetsIter<'_, (), (), 3> =
        tds.try_simplex_facets(simplex_key)?;
    let _triangulation_simplex_facets: GenericSimplexFacetsIter<'_, (), (), 3> =
        tds.try_simplex_facets(simplex_key)?;
    Ok(())
}

fn assert_facet_incidence_exports(
    tds: &Tds<(), (), 3>,
    simplex_key: SimplexKey,
) -> Result<(), PreludeExportTestError> {
    let facet_handle = FacetHandle::try_new(tds, simplex_key, 0)?;
    let query_facet_handle: QueryFacetHandle = facet_handle;
    let query_facade_facet_handle: QueryFacadeFacetHandle = facet_handle;
    let facet_view: FacetView<'_, (), (), 3> = facet_handle.view(tds)?;
    assert_eq!(facet_view.handle(), facet_handle);
    assert_eq!(query_facet_handle, facet_handle);
    assert_eq!(query_facade_facet_handle, facet_handle);

    let facet_index = tds.build_facet_to_simplices_index()?;
    let incidence = facet_index
        .get(&facet_view.key())
        .expect("fresh index should contain the facet view key");
    let root_incidence: RootFacetIncidenceView<'_, '_, (), (), 3> = incidence;
    let query_incidence: QueryFacetIncidenceView<'_, '_, (), (), 3> = incidence;
    let tds_incidence: TdsFacetIncidenceView<'_, '_, (), (), 3> = incidence;

    assert_eq!(root_incidence.facet_key(), query_incidence.facet_key());
    assert_eq!(query_incidence.facet_key(), tds_incidence.facet_key());
    assert!(tds_incidence.is_one_sided());
    assert_query_facet_incidence_trait_export(tds);

    let query_facade_one_sided_facets: Option<QueryFacadeOneSidedFacetsIter<'_, (), (), 3>> = None;
    let query_one_sided_facets: Option<QueryOneSidedFacetsIter<'_, (), (), 3>> = None;
    let tds_one_sided_facets: Option<TdsOneSidedFacetsIter<'_, (), (), 3>> = None;
    let triangulation_one_sided_facets: Option<TriangulationOneSidedFacetsIter<'_, (), (), 3>> =
        None;
    assert!(query_facade_one_sided_facets.is_none());
    assert!(query_one_sided_facets.is_none());
    assert!(tds_one_sided_facets.is_none());
    assert!(triangulation_one_sided_facets.is_none());
    let one_sided_count = tds
        .one_sided_facets()?
        .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))?;
    assert!(one_sided_count > 0);

    let topology_classification =
        topology_classify_boundary_facet(incidence, GlobalTopology::Euclidean)?;
    assert_matches!(
        topology_classification,
        TopologyBoundaryFacetClassification::Boundary(_)
    );
    Ok(())
}

fn assert_export_prelude_exports<K, U, V, const D: usize>(
    dt: &DelaunayTriangulation<K, U, V, D>,
) -> Result<(), PreludeExportTestError> {
    let focused_mesh_export: ExportPreludeMeshExport<D> = dt.to_mesh_export()?;
    let focused_visualization_data: ExportPreludeVisualizationData<D> =
        dt.to_visualization_data()?;
    assert_eq!(
        focused_mesh_export.metadata.schema,
        ExportPreludeMeshExportSchema
    );
    assert_eq!(
        focused_visualization_data.metadata.schema,
        ExportPreludeMeshExportSchema
    );
    assert_eq!(
        focused_visualization_data.metadata.topology_kind,
        ExportPreludeVisualizationTopologyKind::Euclidean
    );
    assert_eq!(
        focused_visualization_data.metadata.topology_guarantee,
        ExportPreludeVisualizationTopologyGuarantee::PLManifold
    );
    let focused_validated_mesh_export: ExportPreludeValidatedMeshExport<D> =
        focused_mesh_export.into_validated()?;
    let focused_validated_visualization_data: ExportPreludeValidatedVisualizationData<D> =
        focused_visualization_data.into_validated()?;
    assert_eq!(
        focused_validated_mesh_export.metadata().schema,
        ExportPreludeMeshExportSchema
    );
    assert_eq!(
        focused_validated_visualization_data.metadata().schema,
        ExportPreludeMeshExportSchema
    );
    assert_eq!(ExportPreludeInvalidCoordinateValue::Nan.to_string(), "NaN");
    Ok(())
}

fn assert_insertion_prelude_empty_tds_exports() -> Result<(), PreludeExportTestError> {
    let mut empty_tds: InsertionTds<(), (), 2> = InsertionTds::empty();
    let _tds_all_facets: TdsAllFacetsIter<'_, (), (), 2> = empty_tds.facets();
    let tds_boundary_facets: Option<TdsBoundaryFacetsIter<'static, (), (), 2>> = None;
    assert!(tds_boundary_facets.is_none());
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
    Ok(())
}

#[test]
fn preludes_cover_bench_apis() -> Result<(), PreludeExportTestError> {
    let _generated_points: Vec<Point<2>> = try_generate_random_points_seeded(3, (0.0, 1.0), 42)?;

    let vertices: Vec<Vertex<(), 3>> = vec![
        vertex![0.0, 0.0, 0.0]?,
        vertex![1.0, 0.0, 0.0]?,
        vertex![0.0, 1.0, 0.0]?,
        vertex![0.0, 0.0, 1.0]?,
    ];
    let options =
        ConstructionOptions::default().with_insertion_order(InsertionOrderStrategy::Input);
    assert_matches!(
        options.batch_repair_policy(),
        DelaunayRepairPolicy::EveryInsertion
    );
    let dt = DelaunayTriangulation::try_new_with_options(&vertices, options)?;

    assert_eq!(dt.topology_guarantee(), TopologyGuarantee::PLManifold);
    assert_export_prelude_exports(&dt)?;
    let _query_facade_all_facets: QueryFacadeAllFacetsIter<'_, (), (), 3> = dt.facets();
    let _query_facade_boundary_facets: QueryFacadeBoundaryFacetsIter<'_, (), (), 3> =
        dt.boundary_facets()?;
    let _query_prelude_all_facets: QueryAllFacetsIter<'_, (), (), 3> = dt.facets();
    let _query_prelude_boundary_facets: QueryBoundaryFacetsIter<'_, (), (), 3> =
        dt.boundary_facets()?;
    let (simplex_key, simplex) = dt
        .simplices()
        .next()
        .expect("constructed tetrahedron should contain a simplex");
    assert_edge_view_exports(dt.tds(), simplex.vertices()[0], simplex.vertices()[1])?;
    assert_simplex_facet_iter_exports(dt.tds(), simplex_key)?;
    assert_facet_incidence_exports(dt.tds(), simplex_key)?;
    let boundary_facet_count = dt.boundary_facets()?.try_fold(0_usize, |count, facet| {
        facet
            .map(|_| count + 1)
            .map_err(|source| QueryError::TriangulationCorrupted {
                source: Box::new(source.into()),
            })
    })?;
    assert!(boundary_facet_count > 0);
    let hull = ConvexHull::try_from_triangulation(dt.as_triangulation())?;
    assert_eq!(hull.facet_handles().count(), boundary_facet_count);
    let hull_facet_view_count = hull
        .try_facets(dt.as_triangulation())?
        .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))?;
    assert_eq!(hull_facet_view_count, boundary_facet_count);
    dt.validate().unwrap();
    assert_bistellar_flips(&dt);
    assert_pachner_prelude_exports(&dt, simplex_key)?;

    assert_insertion_prelude_empty_tds_exports()?;
    assert_send_sync_unpin::<TdsMutationError>();
    assert_send_sync_unpin::<NeighborRebuildError>();
    assert_send_sync_unpin::<ConstructionSkipSample>();
    assert_send_sync_unpin::<ConstructionSlowInsertionSample>();
    assert_send_sync_unpin::<DelaunayError>();
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
fn query_preludes_cover_borrowed_adjacency_view() -> Result<(), PreludeExportTestError> {
    let vertices: Vec<Vertex<(), 3>> = vec![
        vertex![0.0, 0.0, 0.0]?,
        vertex![1.0, 0.0, 0.0]?,
        vertex![0.0, 1.0, 0.0]?,
        vertex![0.0, 0.0, 1.0]?,
    ];
    let dt = DelaunayTriangulation::try_new(&vertices)?;

    let query_adjacency: QueryTriangulationAdjacency<'_> = dt.adjacency()?;
    let generic_adjacency: GenericTriangulationAdjacency<'_> = dt.as_triangulation().adjacency()?;
    let facade_adjacency: QueryFacadeTriangulationAdjacency<'_> = dt.adjacency()?;
    let root_adjacency: RootTriangulationAdjacency<'_> = dt.adjacency()?;
    let query_incidence: QueryIncidenceView<'_> = dt.incidence()?;
    let generic_incidence: GenericIncidenceView<'_> = dt.as_triangulation().incidence()?;
    let facade_incidence: QueryFacadeIncidenceView<'_> = dt.incidence()?;
    let root_incidence: RootIncidenceView<'_> = dt.incidence()?;
    let query_edges: QueryEdgeIndex<'_> = dt.build_edge_index()?;
    let generic_edges: GenericEdgeIndex<'_> = dt.as_triangulation().build_edge_index()?;
    let facade_edges: QueryFacadeEdgeIndex<'_> = dt.build_edge_index()?;
    let root_edges: RootEdgeIndex<'_> = dt.build_edge_index()?;
    let query_neighbors: QuerySimplexNeighborIndex<'_> = dt.build_simplex_neighbor_index()?;
    let generic_neighbors: GenericSimplexNeighborIndex<'_> =
        dt.as_triangulation().build_simplex_neighbor_index()?;
    let facade_neighbors: QueryFacadeSimplexNeighborIndex<'_> =
        dt.build_simplex_neighbor_index()?;
    let root_neighbors: RootSimplexNeighborIndex<'_> = dt.build_simplex_neighbor_index()?;
    let Some((vertex_key, _)) = dt.vertices().next() else {
        return Ok(());
    };
    let Some((simplex_key, _)) = dt.simplices().next() else {
        return Ok(());
    };

    assert_eq!(query_adjacency.number_of_edges(), 6);
    assert_eq!(
        generic_adjacency.number_of_edges(),
        query_adjacency.number_of_edges()
    );
    assert_eq!(
        facade_adjacency.number_of_edges(),
        query_adjacency.number_of_edges()
    );
    assert_eq!(
        root_adjacency.number_of_edges(),
        query_adjacency.number_of_edges()
    );
    assert_eq!(query_incidence.number_of_adjacent_simplices(vertex_key), 1);
    assert_eq!(
        generic_incidence.number_of_adjacent_simplices(vertex_key),
        query_incidence.number_of_adjacent_simplices(vertex_key)
    );
    assert_eq!(
        facade_incidence.number_of_adjacent_simplices(vertex_key),
        query_incidence.number_of_adjacent_simplices(vertex_key)
    );
    assert_eq!(
        root_incidence.number_of_adjacent_simplices(vertex_key),
        query_incidence.number_of_adjacent_simplices(vertex_key)
    );
    assert_eq!(query_edges.number_of_edges(), 6);
    assert_eq!(
        generic_edges.number_of_edges(),
        query_edges.number_of_edges()
    );
    assert_eq!(
        facade_edges.number_of_edges(),
        query_edges.number_of_edges()
    );
    assert_eq!(root_edges.number_of_edges(), query_edges.number_of_edges());
    assert_eq!(query_neighbors.number_of_simplex_neighbors(simplex_key), 0);
    assert_eq!(
        generic_neighbors.number_of_simplex_neighbors(simplex_key),
        query_neighbors.number_of_simplex_neighbors(simplex_key)
    );
    assert_eq!(
        facade_neighbors.number_of_simplex_neighbors(simplex_key),
        query_neighbors.number_of_simplex_neighbors(simplex_key)
    );
    assert_eq!(
        root_neighbors.number_of_simplex_neighbors(simplex_key),
        query_neighbors.number_of_simplex_neighbors(simplex_key)
    );
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
fn geometry_prelude_covers_simplex_embedding_validation() {
    let first =
        LabeledSimplexEmbedding::try_new([0_usize, 1, 2], [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
            .expect("valid labeled simplex embedding");
    let second =
        LabeledSimplexEmbedding::try_new([0_usize, 1, 3], [[0.0, 0.0], [1.0, 0.0], [0.25, 0.25]])
            .expect("valid labeled simplex embedding");

    assert_matches!(
        coordinate_range_for_axis(&first, 0),
        Some((min, max))
            if abs_diff_eq!(min, 0.0, epsilon = f64::EPSILON)
                && abs_diff_eq!(max, 1.0, epsilon = f64::EPSILON)
    );
    assert!(axis_aligned_bounding_boxes_overlap(&first, &second));
    assert_matches!(
        validate_simplex_embeddings_intersect_only_in_shared_faces(&first, &second),
        Err(SimplexIntersectionFailure::IntersectionOutsideSharedFace {
            witness: SimplexIntersectionWitness {
                shared,
                first_only_witness,
                second_only_witness,
            },
            ..
        }) if shared.as_slice() == [0, 1]
            && first_only_witness.as_slice() == [2]
            && second_only_witness.as_slice() == [3]
    );
    let module_root_simplex = GeometryModuleLabeledSimplexEmbedding::try_new(
        [10_usize, 11, 12],
        [[2.0, 2.0], [3.0, 2.0], [2.0, 3.0]],
    )
    .expect("geometry module root re-exports labeled simplex embedding");
    geometry_module_validate_simplex_embeddings_intersect_only_in_shared_faces(
        &first,
        &module_root_simplex,
    )
    .expect("geometry module root re-exports simplex-intersection validation");

    let spanning_simplex =
        LabeledSimplexEmbedding::try_new([4_usize, 5, 6], [[0.0, 0.0], [1.0, 0.0], [0.0, 0.25]])
            .expect("valid labeled simplex embedding");
    assert_matches!(
        try_periodic_simplex_span(&spanning_simplex, &[1.0, 2.0]),
        Ok(Some(PeriodicSimplexSpan { axis: 0, span, period }))
            if abs_diff_eq!(span, 1.0, epsilon = f64::EPSILON)
                && abs_diff_eq!(period, 1.0, epsilon = f64::EPSILON)
    );

    let duplicate = LabeledSimplexEmbedding::<_, 2>::try_new(
        [7_usize, 7, 8],
        [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]],
    );
    assert_matches!(
        duplicate,
        Err(LabeledSimplexEmbeddingError::DuplicateLabel {
            first_index: 0,
            duplicate_index: 1
        })
    );
    assert_matches!(
        try_periodic_simplex_span(&first, &[0.0, 1.0]),
        Err(PeriodicSimplexSpanError::NonPositivePeriod {
            axis: 0,
            period: 0.0
        })
    );

    let _labels: SimplexEmbeddingBuffer<usize> = [0, 1].into_iter().collect();
    assert_send_sync_unpin::<LabeledSimplexEmbeddingError>();
    assert_send_sync_unpin::<PeriodicSimplexSpanError>();
    assert_send_sync_unpin::<SimplexIntersectionFailure<usize>>();
    assert_error::<SimplexIntersectionFailure<usize>>();
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
    assert_relative_eq!(root_range.bounds().0, -1.0, epsilon = f64::EPSILON);
    assert_relative_eq!(root_range.bounds().1, 1.0, epsilon = f64::EPSILON);
    let geometry_range = GeometryCoordinateRange::try_new(-2.0_f64, -1.0)?;
    assert_relative_eq!(geometry_range.bounds().0, -2.0, epsilon = f64::EPSILON);
    assert_relative_eq!(geometry_range.bounds().1, -1.0, epsilon = f64::EPSILON);
    assert_eq!(range_points.len(), 3);
    assert_eq!(grid_points.len(), 1);
    assert_matches!(
        InvalidPositiveScalar::NonPositive { value: 0.0_f64 },
        InvalidPositiveScalar::NonPositive { value: 0.0 }
    );
    Ok(())
}

#[test]
fn construction_prelude_covers_typed_explicit_errors() {
    let explicit_construction = ExplicitConstructionError::StructuralValidation {
        source: Box::new(TdsError::FacetSharingViolation {
            facet_key: 42,
            existing_incident_count: 2,
            attempted_incident_count: 3,
            max_incident_count: 2,
            candidate_simplex_uuid: Uuid::default(),
            candidate_facet_index: 0,
        }),
    };
    assert_matches!(
        explicit_construction,
        ExplicitConstructionError::StructuralValidation {
            source
        } if matches!(
            source.as_ref(),
            TdsError::FacetSharingViolation {
                attempted_incident_count: 3,
                ..
            }
        )
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

    let explicit_insertion = ExplicitConstructionError::OrientationNormalization {
        source: Box::new(InsertionError::TopologyValidation(
            TdsError::InconsistentDataStructure {
                message: "topology validation failed".to_string(),
            },
        )),
    };
    assert_matches!(
        explicit_insertion,
        ExplicitConstructionError::OrientationNormalization { source }
            if matches!(
                source.as_ref(),
                InsertionError::TopologyValidation(TdsError::InconsistentDataStructure { .. })
            )
    );

    let explicit_invariant = ExplicitConstructionError::TopologyValidation {
        source: Box::new(InvariantError::Tds(TdsError::FacetSharingViolation {
            facet_key: 42,
            existing_incident_count: 2,
            attempted_incident_count: 3,
            max_incident_count: 2,
            candidate_simplex_uuid: Uuid::default(),
            candidate_facet_index: 0,
        })),
    };
    assert_matches!(
        explicit_invariant,
        ExplicitConstructionError::TopologyValidation { source }
            if matches!(source.as_ref(), InvariantError::Tds(TdsError::FacetSharingViolation { .. }))
    );

    let explicit_embedding = ExplicitConstructionError::EmbeddingValidation {
        source: Box::new(ConstructionDelaunayTriangulationValidationError::Tds(
            Box::new(TdsError::InconsistentDataStructure {
                message: "embedding validation failed".to_string(),
            }),
        )),
    };
    assert_matches!(
        explicit_embedding,
        ExplicitConstructionError::EmbeddingValidation { source }
            if matches!(
                source.as_ref(),
                ConstructionDelaunayTriangulationValidationError::Tds(tds)
                    if matches!(tds.as_ref(), TdsError::InconsistentDataStructure { .. })
            )
    );

    let explicit_tds_construction = ExplicitConstructionError::TdsAssembly {
        source: Box::new(TdsConstructionError::ValidationError(
            TdsError::FacetSharingViolation {
                facet_key: 42,
                existing_incident_count: 2,
                attempted_incident_count: 3,
                max_incident_count: 2,
                candidate_simplex_uuid: Uuid::default(),
                candidate_facet_index: 0,
            },
        )),
    };
    assert_matches!(
        explicit_tds_construction,
        ExplicitConstructionError::TdsAssembly { source }
            if matches!(
                source.as_ref(),
                TdsConstructionError::ValidationError(TdsError::FacetSharingViolation { .. })
            )
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

    let focused_report = FocusedValidationReport {
        violations: Vec::new(),
    };
    let root_report: RootTriangulationValidationReport = focused_report;
    assert!(root_report.is_empty());

    let focused_period_error = FocusedPeriodicDomainPeriodError::NonPositivePeriod {
        axis: 0,
        period: 0.0,
    };
    assert_matches!(
        focused_period_error,
        FocusedPeriodicDomainPeriodError::NonPositivePeriod { axis: 0, .. }
    );

    let root_period_error = RootPeriodicDomainPeriodError::NonFinitePeriod {
        axis: 1,
        period: InvalidCoordinateValue::PositiveInfinity,
    };
    assert_matches!(
        root_period_error,
        RootPeriodicDomainPeriodError::NonFinitePeriod { axis: 1, .. }
    );
}

#[test]
fn validation_prelude_covers_delaunay_property_diagnostics() -> Result<(), PreludeExportTestError> {
    let tds: Tds<(), (), 2> = Tds::empty();

    let violations = focused_find_delaunay_violations(&tds, None)?;
    assert!(violations.is_empty());

    let report = focused_delaunay_violation_report(&tds, None)?;
    let focused_report: FocusedDelaunayViolationReport = report;
    let _focused_detail: Option<FocusedDelaunayViolationDetail> = None;
    assert!(focused_report.is_valid());

    let simplex_key = SimplexKey::from(KeyData::from_ffi(11));
    let focused_error = FocusedDelaunayValidationError::DelaunayViolation {
        simplex_key,
        simplex_vertices: Box::default(),
        offending_vertex: None,
        neighbor_simplices: Box::default(),
    };
    assert_matches!(
        focused_error,
        FocusedDelaunayValidationError::DelaunayViolation {
            simplex_key: key,
            offending_vertex: None,
            ..
        } if key == simplex_key
    );

    let root_report = root_delaunay_violation_report(&tds, None)?;
    let root_typed_report: RootDelaunayViolationReport = root_report;
    let _root_typed_detail: Option<RootDelaunayViolationDetail> = None;
    assert!(root_typed_report.is_valid());

    Ok(())
}

fn simplex_prelude_vertices<const D: usize>(
    origin: f64,
    scale: f64,
) -> Result<Vec<Vertex<(), D>>, PreludeExportTestError> {
    let mut vertices = Vec::with_capacity(D + 1);
    vertices.push(vertex!([origin; D])?);

    for axis in 0..D {
        let mut coords = [origin; D];
        coords[axis] = origin + scale;
        vertices.push(vertex!(coords)?);
    }

    Ok(vertices)
}

fn cospherical_prelude_vertices<const D: usize>()
-> Result<Vec<Vertex<(), D>>, PreludeExportTestError> {
    let mut vertices = Vec::with_capacity(D + 2);

    for axis in 0..D {
        let mut coords = [0.0; D];
        coords[axis] = 1.0;
        vertices.push(vertex!(coords)?);
    }

    let mut negative_first_axis = [0.0; D];
    negative_first_axis[0] = -1.0;
    vertices.push(vertex!(negative_first_axis)?);

    let mut negative_second_axis = [0.0; D];
    negative_second_axis[1] = -1.0;
    vertices.push(vertex!(negative_second_axis)?);

    Ok(vertices)
}

fn degenerate_prelude_vertices<const D: usize>()
-> Result<Vec<Vertex<(), D>>, PreludeExportTestError> {
    let mut vertices = Vec::with_capacity(D + 1);
    let mut coordinate = 0.0;
    for _ in 0..=D {
        let mut coords = [0.0; D];
        coords[0] = coordinate;
        vertices.push(vertex!(coords)?);
        coordinate += 1.0;
    }
    Ok(vertices)
}

fn assert_single_simplex_ridge_star<const D: usize>(
    vertices: &[Vertex<(), D>],
) -> Result<(), PreludeExportTestError> {
    let dt = DelaunayTriangulation::try_new(vertices)?;
    let ridge = RidgeCandidate::<D>::try_from_vertices(dt.tds().vertex_keys().take(D - 1))?;
    let star = ridge_star_simplices(dt.tds(), &ridge)?;
    let ridge_query: RidgeQuery<'_, (), (), D> = ridge.query(dt.tds())?;
    let query_star = ridge_query.incident_simplices();
    let ridge_view: RidgeView<'_, (), (), D> = ridge.view(dt.tds())?;
    let view_star = ridge_view.incident_simplices();
    let ridge_vertices = ridge_view.vertices();
    let ridge_links = ridge_view.links()?;
    let ridge_link: &RidgeLinkView<'_, (), (), D> = ridge_links
        .first()
        .expect("simplex ridge should have a link");
    let link_edges = ridge_link.edges();
    let link_edge: &LiftedLinkEdge = link_edges
        .first()
        .expect("single simplex ridge link should have an edge");
    let (first_endpoint, _second_endpoint): (&LiftedVertexId, &LiftedVertexId) =
        link_edge.endpoints();

    assert_eq!(star.len(), 1);
    assert_eq!(query_star.len(), star.len());
    assert_eq!(view_star.len(), star.len());
    assert_eq!(ridge_vertices.len(), D - 1);
    assert_eq!(ridge_links.len(), 1);
    assert_eq!(ridge_link.incident_simplices().len(), star.len());
    assert_eq!(first_endpoint.vertex_key(), link_edge.vertex_keys().0);
    assert_eq!(link_edges.len(), star.len());
    Ok(())
}

fn assert_cospherical_ridge_star<const D: usize>() -> Result<(), PreludeExportTestError> {
    let vertices = cospherical_prelude_vertices::<D>()?;
    let dt = DelaunayTriangulation::try_new(&vertices)?;
    let ridge = RidgeCandidate::<D>::try_from_vertices(dt.tds().vertex_keys().take(D - 1))?;
    let star = ridge_star_simplices(dt.tds(), &ridge)?;

    assert!(!star.is_empty());
    Ok(())
}

fn assert_ridge_candidate_reject_adversarial_keys<const D: usize>(keys: &[VertexKey]) {
    assert_matches!(
        RidgeCandidate::<D>::try_from_vertices(keys.iter().take(D.saturating_sub(2)).copied()),
        Err(RidgeCandidateError::WrongArity {
            expected,
            actual,
            ..
        }) if expected == D - 1 && actual == D.saturating_sub(2)
    );

    if D >= 3 {
        assert_matches!(
            RidgeCandidate::<D>::try_from_vertices(std::iter::repeat_n(keys[0], D - 1)),
            Err(RidgeCandidateError::DuplicateVertex { vertex_key }) if vertex_key == keys[0]
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
    assert_ridge_candidate_reject_adversarial_keys::<D>(&keys);

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

    let root_domain = RootToroidalDomain::<3>::try_new([1.0, 2.0, 3.0])?;
    assert_relative_eq!(
        root_domain.periods().as_slice(),
        domain.periods().as_slice()
    );
    let root_topology = RootGlobalTopology::try_toroidal(
        [1.0, 2.0, 3.0],
        RootToroidalConstructionMode::PeriodicImagePoint,
    )?;
    assert_eq!(root_topology.kind(), RootTopologyKind::Toroidal);
    let root_topology_error: Option<RootTopologyError> = None;
    let root_topology_model_error: Option<RootGlobalTopologyModelError> = None;
    assert!(root_topology_error.is_none());
    assert!(root_topology_model_error.is_none());
    assert_send_sync_unpin::<ToroidalDomainError>();
    assert_send_sync_unpin::<RootToroidalDomainError>();
    Ok(())
}

#[test]
fn triangulation_prelude_covers_generic_layer() -> Result<(), PreludeExportTestError> {
    let vertices = vec![
        triangulation_vertex![0.0, 0.0]?,
        triangulation_vertex![1.0, 0.0]?,
        triangulation_vertex![0.0, 1.0]?,
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
    let _triangulation_all_facets: TriangulationAllFacetsIter<'_, (), (), 2> = tri.facets();
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
            context: FinalDelaunayValidationContext::ConstructionFinalize,
            source: ConstructionDelaunayTriangulationValidationError::VerificationFailed {
                source: ConstructionDelaunayVerificationError::from(
                    DelaunayRepairError::PostconditionFailed {
                        reason: Box::new(DelaunayRepairPostconditionFailure::Disconnected {
                            simplex_count: 1,
                        }),
                    },
                )
                .into(),
            },
        },
        DelaunayConstructionFailure::FinalDelaunayValidation {
            context: FinalDelaunayValidationContext::ConstructionFinalize,
            source: ConstructionDelaunayTriangulationValidationError::VerificationFailed {
                source,
            },
        } if source.to_string().contains("disconnected the triangulation")
    );

    let validation_error = ConstructionDelaunayTriangulationValidationError::VerificationFailed {
        source: ConstructionDelaunayVerificationError::from(
            DelaunayRepairError::PostconditionFailed {
                reason: Box::new(DelaunayRepairPostconditionFailure::Disconnected {
                    simplex_count: 1,
                }),
            },
        )
        .into(),
    };
    assert_matches!(
        validation_error,
        ConstructionDelaunayTriangulationValidationError::VerificationFailed { source }
            if source.to_string().contains("disconnected the triangulation")
    );

    Ok(())
}

#[test]
fn diagnostic_preludes_cover_repair_apis() -> Result<(), PreludeExportTestError> {
    let vertices: Vec<Vertex<(), 3>> = vec![
        vertex![0.0, 0.0, 0.0]?,
        vertex![1.0, 0.0, 0.0]?,
        vertex![0.0, 1.0, 0.0]?,
        vertex![0.0, 0.0, 1.0]?,
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
    assert_eq!(
        FlipFailureKind::from(&FlipError::DegenerateSimplex),
        FlipFailureKind::DegenerateSimplex
    );
    let dangling_vertex_incidence = FlipError::DanglingVertexIncidence {
        vertex_key: VertexKey::from(KeyData::from_ffi(1)),
        simplex_key: SimplexKey::from(KeyData::from_ffi(2)),
    };
    assert_eq!(
        RootFlipFailureKind::from(&dangling_vertex_incidence),
        RootFlipFailureKind::DanglingVertexIncidence
    );
    assert_eq!(
        DirectFlipFailureKind::from(&dangling_vertex_incidence),
        DirectFlipFailureKind::DanglingVertexIncidence
    );
    let orientation_reason = DelaunayRepairOrientationCanonicalizationFailure::AfterFlipRepair {
        source: Box::new(InsertionError::DuplicateCoordinates {
            coordinates: CoordinateValues::from([0.0, 0.0, 0.0]),
        }),
    };
    assert!(orientation_reason.to_string().contains("after flip repair"));
    let orientation_kind = DelaunayRepairOrientationCanonicalizationFailureKind::AfterFlipRepair {
        source_kind: FocusedInsertionErrorKind::DuplicateCoordinates,
    };
    assert_matches!(
        orientation_kind,
        DelaunayRepairOrientationCanonicalizationFailureKind::AfterFlipRepair { .. }
    );
    let heuristic_vertex: Option<DelaunayRepairHeuristicVertexContext> = None;
    assert!(heuristic_vertex.is_none());
    let heuristic_reason =
        DelaunayRepairHeuristicRebuildFailure::RecursionDepthExceeded { max_depth: 1 };
    assert!(
        heuristic_reason
            .to_string()
            .contains("recursion depth exceeded")
    );
    let heuristic_kind = DelaunayRepairHeuristicRebuildFailureKind::RecursionDepthExceeded;
    assert_eq!(
        heuristic_kind,
        DelaunayRepairHeuristicRebuildFailureKind::RecursionDepthExceeded
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
    assert_delaunayize_prelude_repair_exports();
    assert_send_sync_unpin::<SimplexDataRestoreError>();
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
