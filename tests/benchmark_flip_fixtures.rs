#![forbid(unsafe_code)]

//! Tests for benchmark-owned public bistellar flip fixtures.
//!
//! The roundtrip assertions are n=1 ergodicity checks for the public
//! Pachner/bistellar move API. For each selected roundtrip fixture move, one
//! admissible flip followed immediately by its inverse must recover the same
//! valid triangulation: the same vertex UUID set and the same simplex-to-vertex
//! UUID incidence. Exact equality remains the pass/fail condition; failed
//! roundtrips report Jaccard similarity for the vertex and simplex-incidence
//! sets to make near misses debuggable. Forward-only checks keep the default
//! fixture suite focused on one selected admissible move. Pachner's connectedness
//! theorem motivates the broader ergodicity story; see `REFERENCES.md` under
//! "Bistellar (Pachner) Moves and Delaunay Repair".

// Reuse the benchmark-owned fixture catalog so this integration test certifies
// the same public flip workflows that the Criterion harnesses measure.
#[path = "../benches/common/flip_fixtures.rs"]
mod flip_fixtures;
#[path = "../benches/common/flip_workflows.rs"]
mod flip_workflows;

use std::assert_matches;

use delaunay::flips::FlipError;
use delaunay::prelude::construction::{
    DelaunayConstructionFailure, DelaunayConstructionRetryFailure,
    DelaunayTriangulationConstructionError,
};
use delaunay::prelude::tds::{FacetError, SimplexKey};
use delaunay::prelude::validation::{
    DelaunayTriangulationValidationError, TriangulationEmbeddingValidationError,
};
use slotmap::KeyData;

use flip_fixtures::{
    ADVERSARIAL_POINTS_2D, ADVERSARIAL_POINTS_3D, DEGENERATE_POINTS_3D, STABLE_POINTS_2D,
    STABLE_POINTS_3D,
};
#[cfg(feature = "slow-tests")]
use flip_fixtures::{
    ADVERSARIAL_POINTS_4D, ADVERSARIAL_POINTS_5D, STABLE_POINTS_4D, STABLE_POINTS_5D,
};
#[cfg(feature = "slow-tests")]
use flip_workflows::verify_k3_roundtrip;
use flip_workflows::{
    CandidateFilter, FlipMoveKind, FlipTriangulation, FlipWorkflowContext, FlipWorkflowError,
    assert_same_topology, build_flip_dt, facet_support_touches_adversarial_feature,
    flippable_k2_facet, flippable_k3_ridge, forward_k2, largest_volume_simplex,
    ridge_support_touches_adversarial_feature, roundtrip_k1, simplex_touches_adversarial_feature,
    snapshot_topology, verify_k1_roundtrip, verify_k2_forward, verify_k2_roundtrip,
    verify_k3_forward,
};

#[cfg(feature = "slow-tests")]
#[derive(Clone, Copy, Debug)]
enum RoundtripMove {
    K1,
    K2,
    K3,
}

const MINIMAL_K2_ROUNDTRIP_POINTS_3D: &[[f64; 3]] = &[
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.20, 0.20, 0.85],
    [0.20, 0.20, -0.85],
];

/// Verifies the stable and adversarial 2D public flip fixture workflows.
#[test]
fn flip_fixtures_cover_2d_workflows() {
    verify_2d_fixture(STABLE_POINTS_2D, CandidateFilter::Any);
    verify_2d_fixture(
        ADVERSARIAL_POINTS_2D,
        CandidateFilter::TouchesAdversarialFeature,
    );
}

/// Verifies the stable 3D public k=1 fixture workflow.
#[test]
fn flip_fixtures_cover_stable_3d_k1_roundtrip() {
    verify_3d_fixture_move(STABLE_POINTS_3D, CandidateFilter::Any, FlipMoveKind::K1);
}

/// Verifies the stable 3D public k=2 fixture workflow.
#[test]
fn flip_fixtures_cover_stable_3d_k2_forward() {
    verify_3d_fixture_move(STABLE_POINTS_3D, CandidateFilter::Any, FlipMoveKind::K2);
}

/// Verifies the public 3D k=2 inverse workflow on a minimal local support.
#[test]
fn flip_fixtures_cover_minimal_3d_k2_roundtrip() {
    let base_dt =
        build_flip_dt(MINIMAL_K2_ROUNDTRIP_POINTS_3D).expect("minimal 3D fixture should build");
    assert_topology_and_delaunay_valid(&base_dt, "minimal 3D k=2 fixture");
    let facet = flippable_k2_facet(&base_dt, true, CandidateFilter::Any)
        .expect("minimal 3D fixture should provide a roundtrip-capable k=2 facet");
    verify_k2_roundtrip(&base_dt, facet)
        .expect("minimal 3D k=2 roundtrip should recover the same triangulation");
}

/// Verifies the stable 3D public k=3 fixture workflow.
#[test]
fn flip_fixtures_cover_stable_3d_k3_forward() {
    verify_3d_fixture_move(STABLE_POINTS_3D, CandidateFilter::Any, FlipMoveKind::K3);
}

/// Verifies the adversarial 3D public k=1 fixture workflow.
#[test]
fn flip_fixtures_cover_adversarial_3d_k1_roundtrip() {
    verify_3d_fixture_move(
        ADVERSARIAL_POINTS_3D,
        CandidateFilter::TouchesAdversarialFeature,
        FlipMoveKind::K1,
    );
}

/// Verifies the adversarial 3D public k=2 fixture workflow.
#[test]
fn flip_fixtures_cover_adversarial_3d_k2_forward() {
    verify_3d_fixture_move(
        ADVERSARIAL_POINTS_3D,
        CandidateFilter::TouchesAdversarialFeature,
        FlipMoveKind::K2,
    );
}

/// Verifies the adversarial 3D public k=3 fixture workflow.
#[test]
fn flip_fixtures_cover_adversarial_3d_k3_forward() {
    verify_3d_fixture_move(
        ADVERSARIAL_POINTS_3D,
        CandidateFilter::TouchesAdversarialFeature,
        FlipMoveKind::K3,
    );
}

/// Verifies the stable and adversarial 4D public flip fixture workflows.
#[cfg(feature = "slow-tests")]
#[test]
fn flip_fixtures_cover_stable_4d_k1_workflow() {
    verify_roundtrip_fixture_move(STABLE_POINTS_4D, CandidateFilter::Any, RoundtripMove::K1);
}

#[cfg(feature = "slow-tests")]
#[test]
fn flip_fixtures_cover_stable_4d_k2_workflow() {
    verify_roundtrip_fixture_move(STABLE_POINTS_4D, CandidateFilter::Any, RoundtripMove::K2);
}

#[cfg(feature = "slow-tests")]
#[test]
fn flip_fixtures_cover_stable_4d_k3_workflow() {
    verify_roundtrip_fixture_move(STABLE_POINTS_4D, CandidateFilter::Any, RoundtripMove::K3);
}

#[cfg(feature = "slow-tests")]
#[test]
fn flip_fixtures_cover_adversarial_4d_k1_workflow() {
    verify_roundtrip_fixture_move(
        ADVERSARIAL_POINTS_4D,
        CandidateFilter::TouchesAdversarialFeature,
        RoundtripMove::K1,
    );
}

#[cfg(feature = "slow-tests")]
#[test]
fn flip_fixtures_cover_adversarial_4d_k2_workflow() {
    verify_roundtrip_fixture_move(
        ADVERSARIAL_POINTS_4D,
        CandidateFilter::TouchesAdversarialFeature,
        RoundtripMove::K2,
    );
}

#[cfg(feature = "slow-tests")]
#[test]
fn flip_fixtures_cover_adversarial_4d_k3_workflow() {
    verify_roundtrip_fixture_move(
        ADVERSARIAL_POINTS_4D,
        CandidateFilter::TouchesAdversarialFeature,
        RoundtripMove::K3,
    );
}

/// Verifies the stable and adversarial 5D public flip fixture workflows.
#[cfg(feature = "slow-tests")]
#[test]
fn flip_fixtures_cover_stable_5d_k1_workflow() {
    verify_roundtrip_fixture_move(STABLE_POINTS_5D, CandidateFilter::Any, RoundtripMove::K1);
}

#[cfg(feature = "slow-tests")]
#[test]
fn flip_fixtures_cover_stable_5d_k2_workflow() {
    verify_roundtrip_fixture_move(STABLE_POINTS_5D, CandidateFilter::Any, RoundtripMove::K2);
}

#[cfg(feature = "slow-tests")]
#[test]
fn flip_fixtures_cover_stable_5d_k3_workflow() {
    verify_roundtrip_fixture_move(STABLE_POINTS_5D, CandidateFilter::Any, RoundtripMove::K3);
}

#[cfg(feature = "slow-tests")]
#[test]
fn flip_fixtures_cover_adversarial_5d_k1_workflow() {
    verify_roundtrip_fixture_move(
        ADVERSARIAL_POINTS_5D,
        CandidateFilter::TouchesAdversarialFeature,
        RoundtripMove::K1,
    );
}

#[cfg(feature = "slow-tests")]
#[test]
fn flip_fixtures_cover_adversarial_5d_k2_workflow() {
    verify_roundtrip_fixture_move(
        ADVERSARIAL_POINTS_5D,
        CandidateFilter::TouchesAdversarialFeature,
        RoundtripMove::K2,
    );
}

#[cfg(feature = "slow-tests")]
#[test]
fn flip_fixtures_cover_adversarial_5d_k3_workflow() {
    verify_roundtrip_fixture_move(
        ADVERSARIAL_POINTS_5D,
        CandidateFilter::TouchesAdversarialFeature,
        RoundtripMove::K3,
    );
}

/// Verifies that construction errors remain recoverable for malformed fixtures.
#[test]
fn empty_flip_fixture_returns_construction_error() {
    let err = build_flip_dt::<2>(&[]).expect_err("empty fixture should not build");
    match err {
        FlipWorkflowError::Construction { dimension, source } => {
            assert_eq!(dimension, 2);
            match *source {
                DelaunayTriangulationConstructionError::Triangulation(
                    DelaunayConstructionFailure::InsufficientVertices { dimension, .. },
                ) => assert_eq!(dimension, 2),
                other => panic!("unexpected construction source: {other}"),
            }
        }
        other => panic!("unexpected construction error: {other}"),
    }
}

/// Verifies malformed flip fixtures fail with a typed degeneracy source.
#[test]
fn degenerate_flip_fixture_is_rejected_instead_of_sanitized() {
    let err = build_flip_dt(DEGENERATE_POINTS_3D).expect_err("degenerate fixture should not build");

    match err {
        FlipWorkflowError::Construction { dimension, source } => {
            assert_eq!(dimension, 3);
            assert!(
                construction_error_is_degenerate(&source),
                "degenerate fixture should fail with a typed degeneracy source: {source:#?}"
            );
        }
        other => panic!("unexpected degenerate fixture error: {other}"),
    }
}

/// Verifies that adversarial filtering does not silently accept stable supports.
#[test]
fn adversarial_filter_rejects_stable_fixture_supports() {
    let base_dt = build_flip_dt(STABLE_POINTS_2D).expect("stable 2D fixture should build");

    let err = largest_volume_simplex(&base_dt, CandidateFilter::TouchesAdversarialFeature)
        .expect_err("stable fixture should not provide an adversarial k=1 support");
    match err {
        FlipWorkflowError::NoNondegenerateSimplex { dimension, filter } => {
            assert_eq!(dimension, 2);
            assert_eq!(filter, CandidateFilter::TouchesAdversarialFeature);
        }
        other => panic!("unexpected k=1 filter error: {other}"),
    }

    let err = flippable_k2_facet(&base_dt, false, CandidateFilter::TouchesAdversarialFeature)
        .expect_err("stable fixture should not provide an adversarial k=2 support");
    match err {
        FlipWorkflowError::NoFlippableFacet {
            dimension,
            filter,
            last_error,
        } => {
            assert_eq!(dimension, 2);
            assert_eq!(filter, CandidateFilter::TouchesAdversarialFeature);
            assert!(
                last_error.is_none(),
                "filter-only rejection should not report a failed flip candidate"
            );
        }
        other => panic!("unexpected k=2 filter error: {other}"),
    }
}

/// Verifies that missing simplex keys are reported as typed workflow errors.
#[test]
fn missing_simplex_support_returns_typed_error() {
    let base_dt = build_flip_dt(STABLE_POINTS_2D).expect("stable 2D fixture should build");
    let missing_simplex = missing_simplex_key();

    let err = simplex_touches_adversarial_feature(&base_dt, missing_simplex)
        .expect_err("missing simplex support should be reported as an error");
    match err {
        FlipWorkflowError::MissingSimplex { simplex_key } => {
            assert_eq!(simplex_key, missing_simplex);
        }
        other => panic!("unexpected missing simplex support error: {other}"),
    }

    let mut roundtrip_dt = base_dt;
    let err = roundtrip_k1(&mut roundtrip_dt, missing_simplex)
        .expect_err("missing k=1 support should be reported as an error");
    match err {
        FlipWorkflowError::MissingSimplex { simplex_key } => {
            assert_eq!(simplex_key, missing_simplex);
        }
        other => panic!("unexpected missing simplex roundtrip error: {other}"),
    }
}

/// Verifies that invalid facet support inspection reports a specific error.
#[test]
fn invalid_facet_support_returns_specific_error() {
    let base_dt = build_flip_dt(STABLE_POINTS_2D).expect("stable 2D fixture should build");
    let (simplex_key, _) = base_dt
        .simplices()
        .next()
        .expect("stable 2D fixture should contain a simplex");
    let err = base_dt
        .facet_handle(simplex_key, u8::MAX)
        .expect_err("invalid facet index should be rejected at construction");
    assert_eq!(
        err,
        FacetError::InvalidFacetIndex {
            index: u8::MAX,
            facet_count: 3,
        }
    );
}

/// Verifies that invalid ridge support inspection reports a specific error.
#[test]
fn invalid_ridge_support_returns_specific_error() {
    let base_dt = build_flip_dt(STABLE_POINTS_3D).expect("stable 3D fixture should build");
    let (simplex_key, _) = base_dt
        .simplices()
        .next()
        .expect("stable 3D fixture should contain a simplex");
    let err = base_dt
        .ridge_handle(simplex_key, u8::MAX, u8::MAX)
        .expect_err("invalid ridge indices should be rejected at construction");
    assert_matches!(
        err,
        FlipError::InvalidRidgeIndex {
            simplex_key: observed_simplex,
            omit_a: u8::MAX,
            omit_b: u8::MAX,
            vertex_count: 4,
        } if observed_simplex == simplex_key
    );

    let err = base_dt
        .ridge_handle(simplex_key, 0, 0)
        .expect_err("duplicate ridge indices should be rejected at construction");
    assert_matches!(
        err,
        FlipError::InvalidRidgeIndex {
            simplex_key: observed_simplex,
            omit_a: 0,
            omit_b: 0,
            vertex_count: 4,
        } if observed_simplex == simplex_key
    );
}

/// Verifies that failed public flips preserve the typed [`FlipError`] source.
#[test]
fn forward_flip_failure_preserves_typed_source() {
    let base_dt = build_flip_dt(STABLE_POINTS_2D).expect("stable 2D fixture should build");
    let (simplex_key, _) = base_dt
        .simplices()
        .next()
        .expect("stable 2D fixture should contain a simplex");
    let err = base_dt
        .facet_handle(simplex_key, u8::MAX)
        .expect_err("invalid k=2 facet should fail before flip execution");
    assert_eq!(
        err,
        FacetError::InvalidFacetIndex {
            index: u8::MAX,
            facet_count: 3,
        }
    );
}

/// Verifies that topology mismatches include Jaccard diagnostics in the fail path.
#[test]
fn topology_mismatch_reports_jaccard_diagnostics() {
    let base_dt = build_flip_dt(STABLE_POINTS_2D).expect("stable 2D fixture should build");
    let before = snapshot_topology(&base_dt).expect("stable 2D topology snapshot should succeed");
    let facet = flippable_k2_facet(&base_dt, false, CandidateFilter::Any)
        .expect("stable 2D fixture should provide a k=2 facet");
    let mut flipped = base_dt;
    forward_k2(&mut flipped, facet).expect("2D k=2 forward flip should succeed");

    let err = assert_same_topology(
        &flipped,
        &before,
        FlipWorkflowContext::forward_only::<2>(FlipMoveKind::K2),
    )
    .expect_err("forward-only k=2 flip should not match the original topology");
    match err {
        FlipWorkflowError::TopologyMismatch {
            context,
            vertex_report,
            simplex_report,
            ..
        } => {
            assert_eq!(
                context,
                FlipWorkflowContext::forward_only::<2>(FlipMoveKind::K2)
            );
            assert!(
                vertex_report.contains("Jaccard Similarity Report"),
                "missing vertex Jaccard report in topology mismatch diagnostics: {vertex_report}"
            );
            assert!(
                simplex_report.contains("expected simplex UUID incidence"),
                "missing simplex-incidence label in topology mismatch diagnostics: {simplex_report}"
            );
        }
        other => panic!("unexpected topology mismatch error: {other}"),
    }
}

/// Verifies all selected 2D public flip workflows for one fixture.
fn verify_2d_fixture(points: &[[f64; 2]], filter: CandidateFilter) {
    let base_dt = build_flip_dt(points).expect("2D benchmark flip fixture should build");
    assert_topology_and_delaunay_valid(&base_dt, "2D benchmark flip fixture");

    let simplex_key = largest_volume_simplex(&base_dt, filter)
        .expect("2D benchmark fixture should provide a selected k=1 simplex");
    if filter == CandidateFilter::TouchesAdversarialFeature {
        assert!(
            simplex_touches_adversarial_feature(&base_dt, simplex_key)
                .expect("2D k=1 support should be inspectable"),
            "2D adversarial k=1 support should touch an adversarial fixture feature"
        );
    }
    verify_k1_roundtrip(&base_dt, simplex_key)
        .expect("2D k=1 roundtrip should recover the same triangulation");

    let facet = flippable_k2_facet(&base_dt, false, filter)
        .expect("2D benchmark fixture should provide a selected k=2 facet");
    if filter == CandidateFilter::TouchesAdversarialFeature {
        assert!(
            facet_support_touches_adversarial_feature(&base_dt, facet)
                .expect("2D k=2 support should be inspectable"),
            "2D adversarial k=2 support should touch an adversarial fixture feature"
        );
    }
    verify_k2_forward(&base_dt, facet)
        .expect("2D benchmark k=2 forward flip should preserve topology");
}

/// Verifies one selected 3D public flip workflow for one fixture.
fn verify_3d_fixture_move(points: &[[f64; 3]], filter: CandidateFilter, move_kind: FlipMoveKind) {
    let base_dt = build_flip_dt(points).expect("3D benchmark flip fixture should build");
    assert_topology_and_delaunay_valid(&base_dt, "3D benchmark flip fixture");

    match move_kind {
        FlipMoveKind::K1 => {
            let simplex_key = largest_volume_simplex(&base_dt, filter)
                .expect("3D benchmark fixture should provide a selected k=1 simplex");
            if filter == CandidateFilter::TouchesAdversarialFeature {
                assert!(
                    simplex_touches_adversarial_feature(&base_dt, simplex_key)
                        .expect("3D k=1 support should be inspectable"),
                    "3D adversarial k=1 support should touch an adversarial fixture feature"
                );
            }
            verify_k1_roundtrip(&base_dt, simplex_key)
                .expect("3D k=1 roundtrip should recover the same triangulation");
        }
        FlipMoveKind::K2 => {
            let facet = flippable_k2_facet(&base_dt, false, filter)
                .expect("3D benchmark fixture should provide a selected k=2 facet");
            if filter == CandidateFilter::TouchesAdversarialFeature {
                assert!(
                    facet_support_touches_adversarial_feature(&base_dt, facet)
                        .expect("3D k=2 support should be inspectable"),
                    "3D adversarial k=2 support should touch an adversarial fixture feature"
                );
            }
            verify_k2_forward(&base_dt, facet)
                .expect("3D benchmark k=2 forward flip should preserve topology");
        }
        FlipMoveKind::K3 => {
            let ridge = flippable_k3_ridge(&base_dt, false, filter)
                .expect("3D benchmark fixture should provide a selected k=3 ridge");
            if filter == CandidateFilter::TouchesAdversarialFeature {
                assert!(
                    ridge_support_touches_adversarial_feature(&base_dt, ridge)
                        .expect("3D k=3 support should be inspectable"),
                    "3D adversarial k=3 support should touch an adversarial fixture feature"
                );
            }
            verify_k3_forward(&base_dt, ridge)
                .expect("3D benchmark k=3 forward flip should preserve topology");
        }
    }
}

/// Verifies one selected roundtrip-capable public flip workflow for one dimension.
#[cfg(feature = "slow-tests")]
fn verify_roundtrip_fixture_move<const D: usize>(
    points: &[[f64; D]],
    filter: CandidateFilter,
    roundtrip_move: RoundtripMove,
) {
    let base_dt = build_flip_dt(points).expect("benchmark flip fixture should build");
    assert_topology_and_delaunay_valid(&base_dt, "benchmark flip fixture");

    match roundtrip_move {
        RoundtripMove::K1 => {
            let simplex_key = largest_volume_simplex(&base_dt, filter)
                .expect("benchmark fixture should provide a selected k=1 simplex");
            if filter == CandidateFilter::TouchesAdversarialFeature {
                assert!(
                    simplex_touches_adversarial_feature(&base_dt, simplex_key)
                        .expect("k=1 support should be inspectable"),
                    "adversarial k=1 support should touch an adversarial fixture feature"
                );
            }
            verify_k1_roundtrip(&base_dt, simplex_key)
                .expect("k=1 roundtrip should recover the same triangulation");
        }
        RoundtripMove::K2 => {
            let facet = flippable_k2_facet(&base_dt, true, filter)
                .expect("benchmark fixture should provide a selected k=2 facet");
            if filter == CandidateFilter::TouchesAdversarialFeature {
                assert!(
                    facet_support_touches_adversarial_feature(&base_dt, facet)
                        .expect("k=2 support should be inspectable"),
                    "adversarial k=2 support should touch an adversarial fixture feature"
                );
            }
            verify_k2_roundtrip(&base_dt, facet)
                .expect("k=2 roundtrip should recover the same triangulation");
        }
        RoundtripMove::K3 => {
            let ridge = flippable_k3_ridge(&base_dt, true, filter)
                .expect("benchmark fixture should provide a selected k=3 ridge");
            if filter == CandidateFilter::TouchesAdversarialFeature {
                assert!(
                    ridge_support_touches_adversarial_feature(&base_dt, ridge)
                        .expect("k=3 support should be inspectable"),
                    "adversarial k=3 support should touch an adversarial fixture feature"
                );
            }
            verify_k3_roundtrip(&base_dt, ridge)
                .expect("k=3 roundtrip should recover the same triangulation");
        }
    }
}

fn assert_topology_and_delaunay_valid<const D: usize>(dt: &FlipTriangulation<D>, context: &str) {
    dt.as_triangulation()
        .validate()
        .unwrap_or_else(|err| panic!("{context} should pass Levels 1-3: {err}"));
    dt.as_triangulation()
        .is_valid_embedding()
        .unwrap_or_else(|err| panic!("{context} should pass Level 4 embedding: {err}"));
    dt.is_valid_delaunay()
        .unwrap_or_else(|err| panic!("{context} should pass Level 5: {err}"));
}

fn construction_error_is_degenerate(error: &DelaunayTriangulationConstructionError) -> bool {
    match error {
        DelaunayTriangulationConstructionError::Triangulation(
            DelaunayConstructionFailure::GeometricDegeneracy { .. },
        ) => true,
        DelaunayTriangulationConstructionError::Triangulation(
            DelaunayConstructionFailure::FinalDelaunayValidation { source, .. },
        ) => validation_error_is_degenerate(source),
        DelaunayTriangulationConstructionError::Triangulation(
            DelaunayConstructionFailure::InsertionEmbeddingValidation { source },
        ) => matches!(
            source,
            TriangulationEmbeddingValidationError::DegenerateSimplex { .. }
        ),
        DelaunayTriangulationConstructionError::Triangulation(
            DelaunayConstructionFailure::ShuffledRetryExhausted { source, .. },
        ) => match source.as_ref() {
            DelaunayConstructionRetryFailure::Construction { source } => {
                construction_error_is_degenerate(source)
            }
            _ => false,
        },
        _ => false,
    }
}

fn validation_error_is_degenerate(error: &DelaunayTriangulationValidationError) -> bool {
    matches!(
        error,
        DelaunayTriangulationValidationError::Embedding(source)
            if matches!(
                source.as_ref(),
                TriangulationEmbeddingValidationError::DegenerateSimplex { .. }
            )
    )
}

/// Creates a synthetic simplex key that cannot be live in fixture triangulations.
fn missing_simplex_key() -> SimplexKey {
    SimplexKey::from(KeyData::from_ffi(u64::MAX))
}
