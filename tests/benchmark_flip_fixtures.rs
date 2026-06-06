#![forbid(unsafe_code)]

//! Tests for benchmark-owned public bistellar flip fixtures.
//!
//! The roundtrip assertions are n=1 ergodicity checks for the public
//! Pachner/bistellar move API. For each selected fixture move, one admissible
//! flip followed immediately by its inverse must recover the same valid
//! triangulation: the same vertex UUID set and the same simplex-to-vertex UUID
//! incidence. Exact equality remains the pass/fail condition; failed roundtrips
//! report Jaccard similarity for the vertex and simplex-incidence sets to make
//! near misses debuggable. Pachner's connectedness theorem motivates the broader
//! ergodicity story; see `REFERENCES.md` under "Bistellar (Pachner) Moves and
//! Delaunay Repair".

#[path = "../benches/common/flip_fixtures.rs"]
mod flip_fixtures;
#[path = "../benches/common/flip_workflows.rs"]
mod flip_workflows;

use delaunay::prelude::construction::{
    DelaunayConstructionFailure, DelaunayTriangulationConstructionError,
};
use delaunay::prelude::flips::{FacetHandle, FlipError, RidgeHandle};
use delaunay::prelude::tds::SimplexKey;
use slotmap::KeyData;

use flip_fixtures::{
    ADVERSARIAL_POINTS_2D, ADVERSARIAL_POINTS_3D, ADVERSARIAL_POINTS_4D, ADVERSARIAL_POINTS_5D,
    STABLE_POINTS_2D, STABLE_POINTS_3D, STABLE_POINTS_4D, STABLE_POINTS_5D,
};
use flip_workflows::{
    CandidateFilter, FlipMoveKind, FlipWorkflowError, assert_same_topology, build_flip_dt,
    facet_support_touches_adversarial_feature, flippable_k2_facet, flippable_k3_ridge, forward_k2,
    forward_k3, largest_volume_simplex, ridge_support_touches_adversarial_feature, roundtrip_k1,
    simplex_touches_adversarial_feature, snapshot_topology, verify_k1_roundtrip,
    verify_k2_roundtrip, verify_k3_roundtrip,
};

/// Verifies the stable and adversarial 2D public flip fixture workflows.
#[test]
fn flip_fixtures_cover_2d_workflows() {
    verify_2d_fixture(STABLE_POINTS_2D, CandidateFilter::Any);
    verify_2d_fixture(
        ADVERSARIAL_POINTS_2D,
        CandidateFilter::TouchesAdversarialFeature,
    );
}

/// Verifies the stable and adversarial 3D public flip fixture workflows.
#[test]
fn flip_fixtures_cover_3d_workflows() {
    verify_3d_fixture(STABLE_POINTS_3D, CandidateFilter::Any);
    verify_3d_fixture(
        ADVERSARIAL_POINTS_3D,
        CandidateFilter::TouchesAdversarialFeature,
    );
}

/// Verifies the stable and adversarial 4D public flip fixture workflows.
#[test]
fn flip_fixtures_cover_4d_workflows() {
    verify_roundtrip_fixture(STABLE_POINTS_4D, CandidateFilter::Any);
    verify_roundtrip_fixture(
        ADVERSARIAL_POINTS_4D,
        CandidateFilter::TouchesAdversarialFeature,
    );
}

/// Verifies the stable and adversarial 5D public flip fixture workflows.
#[test]
fn flip_fixtures_cover_5d_workflows() {
    verify_roundtrip_fixture(STABLE_POINTS_5D, CandidateFilter::Any);
    verify_roundtrip_fixture(
        ADVERSARIAL_POINTS_5D,
        CandidateFilter::TouchesAdversarialFeature,
    );
}

/// Verifies that construction errors remain recoverable for malformed fixtures.
#[test]
fn empty_flip_fixture_returns_construction_error() {
    let err = build_flip_dt::<2>(&[]).expect_err("empty fixture should not build");
    match err {
        FlipWorkflowError::Construction { dimension, source } => {
            assert_eq!(dimension, 2);
            match source {
                DelaunayTriangulationConstructionError::Triangulation(
                    DelaunayConstructionFailure::InsufficientVertices { dimension, .. },
                ) => assert_eq!(dimension, 2),
                other => panic!("unexpected construction source: {other}"),
            }
        }
        other => panic!("unexpected construction error: {other}"),
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
    let invalid_facet = FacetHandle::new(simplex_key, u8::MAX);

    let err = facet_support_touches_adversarial_feature(&base_dt, invalid_facet)
        .expect_err("invalid facet support should be reported as an error");
    match err {
        FlipWorkflowError::InvalidFacetSupportIndex {
            facet,
            facet_index,
            vertex_count,
            simplex_key: observed_simplex,
        } => {
            assert_eq!(facet, invalid_facet);
            assert_eq!(facet_index, u8::MAX);
            assert_eq!(vertex_count, 3);
            assert_eq!(observed_simplex, simplex_key);
        }
        other => panic!("unexpected invalid facet error: {other}"),
    }
}

/// Verifies that invalid ridge support inspection reports a specific error.
#[test]
fn invalid_ridge_support_returns_specific_error() {
    let base_dt = build_flip_dt(STABLE_POINTS_3D).expect("stable 3D fixture should build");
    let (simplex_key, _) = base_dt
        .simplices()
        .next()
        .expect("stable 3D fixture should contain a simplex");
    let invalid_ridge = RidgeHandle::new(simplex_key, u8::MAX, u8::MAX);

    let err = ridge_support_touches_adversarial_feature(&base_dt, invalid_ridge)
        .expect_err("invalid ridge support should be reported as an error");
    match err {
        FlipWorkflowError::InvalidRidgeSupportIndex {
            ridge,
            omit_a,
            omit_b,
            vertex_count,
            simplex_key: observed_simplex,
        } => {
            assert_eq!(ridge, invalid_ridge);
            assert_eq!(omit_a, u8::MAX);
            assert_eq!(omit_b, u8::MAX);
            assert_eq!(vertex_count, 4);
            assert_eq!(observed_simplex, simplex_key);
        }
        other => panic!("unexpected invalid ridge error: {other}"),
    }

    let duplicate_ridge = RidgeHandle::new(simplex_key, 0, 0);
    let err = ridge_support_touches_adversarial_feature(&base_dt, duplicate_ridge)
        .expect_err("duplicate ridge support indices should be reported as an error");
    match err {
        FlipWorkflowError::DuplicateRidgeSupportIndex {
            ridge,
            omit_a,
            omit_b,
            vertex_count,
            simplex_key: observed_simplex,
        } => {
            assert_eq!(ridge, duplicate_ridge);
            assert_eq!(omit_a, 0);
            assert_eq!(omit_b, 0);
            assert_eq!(vertex_count, 4);
            assert_eq!(observed_simplex, simplex_key);
        }
        other => panic!("unexpected duplicate ridge error: {other}"),
    }
}

/// Verifies that failed public flips preserve the typed [`FlipError`] source.
#[test]
fn forward_flip_failure_preserves_typed_source() {
    let mut base_dt = build_flip_dt(STABLE_POINTS_2D).expect("stable 2D fixture should build");
    let (simplex_key, _) = base_dt
        .simplices()
        .next()
        .expect("stable 2D fixture should contain a simplex");
    let invalid_facet = FacetHandle::new(simplex_key, u8::MAX);

    let err = forward_k2(&mut base_dt, invalid_facet)
        .expect_err("invalid k=2 facet should be reported as a flip failure");
    match err {
        FlipWorkflowError::FlipFailed {
            dimension,
            move_kind,
            source,
        } => {
            assert_eq!(dimension, 2);
            assert_eq!(move_kind, FlipMoveKind::K2);
            match *source {
                FlipError::InvalidFacetIndex {
                    simplex_key: observed_simplex,
                    facet_index,
                    ..
                } => {
                    assert_eq!(observed_simplex, simplex_key);
                    assert_eq!(facet_index, u8::MAX);
                }
                other => panic!("unexpected flip source: {other}"),
            }
        }
        other => panic!("unexpected forward flip error: {other}"),
    }
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

    let err = assert_same_topology(&flipped, &before, "2D k=2 forward mismatch")
        .expect_err("forward-only k=2 flip should not match the original topology");
    match err {
        FlipWorkflowError::TopologyMismatch {
            context,
            vertex_report,
            simplex_report,
            ..
        } => {
            assert_eq!(context, "2D k=2 forward mismatch");
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
    base_dt
        .validate()
        .expect("2D benchmark flip fixture should validate");

    let simplex_key = largest_volume_simplex(&base_dt, filter)
        .expect("2D benchmark fixture should provide a selected k=1 simplex");
    if filter == CandidateFilter::TouchesAdversarialFeature {
        assert!(
            simplex_touches_adversarial_feature(&base_dt, simplex_key)
                .expect("2D k=1 support should be inspectable"),
            "2D adversarial k=1 support should touch an adversarial fixture feature"
        );
    }
    verify_k1_roundtrip(&base_dt, simplex_key, "2D k=1 n=1 ergodicity roundtrip")
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
    let mut k2 = base_dt;
    forward_k2(&mut k2, facet).expect("2D benchmark k=2 forward flip should succeed");
    k2.as_triangulation()
        .validate()
        .expect("2D benchmark k=2 forward flip should preserve topology");
}

/// Verifies all selected 3D public flip workflows for one fixture.
fn verify_3d_fixture(points: &[[f64; 3]], filter: CandidateFilter) {
    let base_dt = build_flip_dt(points).expect("3D benchmark flip fixture should build");
    base_dt
        .validate()
        .expect("3D benchmark flip fixture should validate");

    let simplex_key = largest_volume_simplex(&base_dt, filter)
        .expect("3D benchmark fixture should provide a selected k=1 simplex");
    if filter == CandidateFilter::TouchesAdversarialFeature {
        assert!(
            simplex_touches_adversarial_feature(&base_dt, simplex_key)
                .expect("3D k=1 support should be inspectable"),
            "3D adversarial k=1 support should touch an adversarial fixture feature"
        );
    }
    verify_k1_roundtrip(&base_dt, simplex_key, "3D k=1 n=1 ergodicity roundtrip")
        .expect("3D k=1 roundtrip should recover the same triangulation");

    let facet = flippable_k2_facet(&base_dt, true, filter)
        .expect("3D benchmark fixture should provide a selected k=2 facet");
    if filter == CandidateFilter::TouchesAdversarialFeature {
        assert!(
            facet_support_touches_adversarial_feature(&base_dt, facet)
                .expect("3D k=2 support should be inspectable"),
            "3D adversarial k=2 support should touch an adversarial fixture feature"
        );
    }
    verify_k2_roundtrip(&base_dt, facet, "3D k=2 n=1 ergodicity roundtrip")
        .expect("3D k=2 roundtrip should recover the same triangulation");

    let ridge = flippable_k3_ridge(&base_dt, false, filter)
        .expect("3D benchmark fixture should provide a selected k=3 ridge");
    if filter == CandidateFilter::TouchesAdversarialFeature {
        assert!(
            ridge_support_touches_adversarial_feature(&base_dt, ridge)
                .expect("3D k=3 support should be inspectable"),
            "3D adversarial k=3 support should touch an adversarial fixture feature"
        );
    }
    let mut k3 = base_dt;
    forward_k3(&mut k3, ridge).expect("3D benchmark k=3 forward flip should succeed");
    k3.as_triangulation()
        .validate()
        .expect("3D benchmark k=3 forward flip should preserve topology");
}

/// Verifies all selected roundtrip-capable public flip workflows for one dimension.
fn verify_roundtrip_fixture<const D: usize>(points: &[[f64; D]], filter: CandidateFilter) {
    let base_dt = build_flip_dt(points).expect("benchmark flip fixture should build");
    base_dt
        .validate()
        .expect("benchmark flip fixture should validate");

    let simplex_key = largest_volume_simplex(&base_dt, filter)
        .expect("benchmark fixture should provide a selected k=1 simplex");
    if filter == CandidateFilter::TouchesAdversarialFeature {
        assert!(
            simplex_touches_adversarial_feature(&base_dt, simplex_key)
                .expect("k=1 support should be inspectable"),
            "adversarial k=1 support should touch an adversarial fixture feature"
        );
    }
    verify_k1_roundtrip(&base_dt, simplex_key, "k=1 n=1 ergodicity roundtrip")
        .expect("k=1 roundtrip should recover the same triangulation");

    let facet = flippable_k2_facet(&base_dt, true, filter)
        .expect("benchmark fixture should provide a selected k=2 facet");
    if filter == CandidateFilter::TouchesAdversarialFeature {
        assert!(
            facet_support_touches_adversarial_feature(&base_dt, facet)
                .expect("k=2 support should be inspectable"),
            "adversarial k=2 support should touch an adversarial fixture feature"
        );
    }
    verify_k2_roundtrip(&base_dt, facet, "k=2 n=1 ergodicity roundtrip")
        .expect("k=2 roundtrip should recover the same triangulation");

    let ridge = flippable_k3_ridge(&base_dt, true, filter)
        .expect("benchmark fixture should provide a selected k=3 ridge");
    if filter == CandidateFilter::TouchesAdversarialFeature {
        assert!(
            ridge_support_touches_adversarial_feature(&base_dt, ridge)
                .expect("k=3 support should be inspectable"),
            "adversarial k=3 support should touch an adversarial fixture feature"
        );
    }
    verify_k3_roundtrip(&base_dt, ridge, "k=3 n=1 ergodicity roundtrip")
        .expect("k=3 roundtrip should recover the same triangulation");
}

/// Creates a synthetic simplex key that cannot be live in fixture triangulations.
fn missing_simplex_key() -> SimplexKey {
    SimplexKey::from(KeyData::from_ffi(u64::MAX))
}
