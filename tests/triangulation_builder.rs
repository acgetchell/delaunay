//! Integration tests for [`DelaunayTriangulationBuilder`].
//!
//! These tests exercise the public API from the outside, using only items exposed
//! through `delaunay::prelude` and `delaunay::builder`.

#![forbid(unsafe_code)]

use std::assert_matches;
use std::collections::HashMap;
use std::f64::consts::TAU;

use delaunay::prelude::construction::{
    ConstructionOptions, DelaunayConstructionFailure, DelaunayResult, DelaunayTriangulation,
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError,
    ExplicitConstructionError, InsertionOrderStrategy, TopologyGuarantee, Vertex,
};
use delaunay::prelude::geometry::RobustKernel;
use delaunay::prelude::insertion::InsertionError;
use delaunay::prelude::tds::{InvariantError, TdsConstructionError, TdsError, VertexKey};
use delaunay::prelude::topology::spaces::{GlobalTopology, TopologyKind, ToroidalConstructionMode};
use delaunay::prelude::topology::validation::{TopologyClassification, euler_characteristic};
use delaunay::prelude::validation::{
    DelaunayTriangulationValidationError, TriangulationValidationError, ValidationPolicy,
};
use delaunay::vertex;

// =============================================================================
// Euclidean path
// =============================================================================

/// The associated builder alias should match the explicit builder path.
#[test]
fn test_builder_euclidean_matches_new_2d() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
    ];

    let dt_new = DelaunayTriangulation::builder(&vertices)
        .build()
        .expect("associated builder should succeed");
    let dt_builder = DelaunayTriangulationBuilder::new(&vertices)
        .build()
        .expect("builder should succeed");

    assert_eq!(dt_new.number_of_vertices(), dt_builder.number_of_vertices());
    assert_eq!(
        dt_new.number_of_simplices(),
        dt_builder.number_of_simplices()
    );
    assert_eq!(dt_new.dim(), dt_builder.dim());
}

/// Builder euclidean path — 3D sanity check.
#[test]
fn test_builder_euclidean_3d() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 1.0]).unwrap(),
    ];
    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .build()
        .expect("3D build should succeed");
    assert_eq!(dt.number_of_vertices(), 4);
    assert_eq!(dt.dim(), 3);
    assert!(dt.validate().is_ok(), "Level 1-4 validation should pass");
}

/// `TopologyGuarantee` set on the builder is propagated to the triangulation.
#[test]
fn test_builder_topology_guarantee_propagated() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
    ];
    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .topology_guarantee(TopologyGuarantee::Pseudomanifold)
        .build()
        .expect("build should succeed");
    assert_eq!(dt.topology_guarantee(), TopologyGuarantee::Pseudomanifold);
}

/// Builder derives validation policy from topology guarantee instead of exposing
/// a separate construction-time policy axis.
#[test]
fn test_builder_validation_policy_derived_from_topology_guarantee() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
    ];

    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .topology_guarantee(TopologyGuarantee::PLManifoldStrict)
        .build()
        .expect("build should succeed");

    assert_eq!(dt.topology_guarantee(), TopologyGuarantee::PLManifoldStrict);
    assert_eq!(dt.validation_policy(), ValidationPolicy::Always);
}

#[test]
fn test_builder_constructs_typed_simplex_storage_for_follow_on_fill() -> DelaunayResult<()> {
    let vertices = vec![
        vertex!([0.0_f64, 0.0])?,
        vertex!([1.0, 0.0])?,
        vertex!([0.0, 1.0])?,
    ];
    let mut dt = DelaunayTriangulationBuilder::new(&vertices)
        .simplex_data_type::<usize>()
        .build()?;

    dt.fill_simplex_data(|_, simplex| simplex.number_of_vertices());

    for (_, simplex) in dt.simplices() {
        assert_eq!(simplex.data(), Some(&3));
    }
    dt.validate()?;
    Ok(())
}

/// Custom `ConstructionOptions` are accepted and the triangulation is valid.
#[test]
fn test_builder_custom_construction_options() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
    ];
    let opts = ConstructionOptions::default().with_insertion_order(InsertionOrderStrategy::Input);
    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .construction_options(opts)
        .build()
        .expect("build should succeed");
    assert_eq!(dt.number_of_vertices(), 3);
    assert!(dt.validate().is_ok());
}

/// The fluent statistics terminal exposes insertion stats and total timing for Euclidean builds.
#[test]
fn test_builder_build_with_statistics_euclidean_records_telemetry() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
    ];
    let (dt, stats) = DelaunayTriangulationBuilder::new(&vertices)
        .build_with_statistics()
        .expect("Euclidean statistics build should succeed");

    assert_eq!(dt.number_of_vertices(), stats.inserted);
    assert_eq!(stats.total_skipped(), 0);
    assert!(stats.total_attempts >= stats.inserted);
    assert!(stats.telemetry.has_data());
    assert!(stats.telemetry.construction_total_nanos > 0);
    assert!(dt.validate().is_ok());
}

/// Canonicalized toroidal construction has the same fluent statistics terminal.
#[test]
fn test_builder_build_with_statistics_canonicalized_toroidal_records_telemetry() {
    let vertices = vec![
        vertex!([0.2_f64, 0.3]).unwrap(),
        vertex!([1.8, 0.1]).unwrap(),
        vertex!([0.5, 0.7]).unwrap(),
        vertex!([-0.4, 0.9]).unwrap(),
    ];
    let (dt, stats) = DelaunayTriangulationBuilder::new(&vertices)
        .try_canonicalized_toroidal([1.0, 1.0])
        .unwrap()
        .build_with_statistics()
        .expect("canonicalized toroidal statistics build should succeed");

    assert_eq!(dt.number_of_vertices(), stats.inserted);
    assert_eq!(stats.total_skipped(), 0);
    assert!(stats.total_attempts >= stats.inserted);
    assert!(stats.telemetry.has_data());
    assert!(stats.telemetry.construction_total_nanos > 0);
    assert!(dt.validate().is_ok());
}

// =============================================================================
// Convenience entry point
// =============================================================================

/// `DelaunayTriangulation::builder()` convenience method compiles and works.
#[test]
fn test_builder_convenience() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
    ];
    let dt = DelaunayTriangulation::builder(&vertices)
        .build()
        .expect("builder() convenience method should succeed");
    assert_eq!(dt.number_of_vertices(), 3);
    assert!(dt.validate().is_ok());
}

/// `DelaunayTriangulation::builder()` with canonicalized toroidal wrapping.
#[test]
fn test_builder_canonicalized_toroidal_convenience() {
    let vertices = vec![
        vertex!([0.2_f64, 0.3]).unwrap(),
        vertex!([1.8, 0.1]).unwrap(), // → (0.8, 0.1)
        vertex!([0.5, 0.7]).unwrap(),
        vertex!([-0.4, 0.9]).unwrap(), // → (0.6, 0.9)
    ];
    let dt = DelaunayTriangulation::builder(&vertices)
        .try_canonicalized_toroidal([1.0, 1.0])
        .unwrap()
        .build()
        .expect("canonicalized toroidal builder should succeed");
    assert_eq!(dt.number_of_vertices(), 4);
    assert!(dt.as_triangulation().validate().is_ok());
}

// =============================================================================
// Canonicalized toroidal path
// =============================================================================

/// Out-of-domain coordinates are canonicalized: the resulting triangulation
/// is geometrically identical to building directly from the wrapped coordinates.
#[test]
fn test_builder_canonicalized_toroidal_canonicalizes_coordinates() {
    // Canonical (already-wrapped) coordinates
    let canonical_vertices = vec![
        vertex!([0.2_f64, 0.3]).unwrap(),
        vertex!([0.8, 0.1]).unwrap(),
        vertex!([0.5, 0.7]).unwrap(),
        vertex!([0.6, 0.9]).unwrap(),
    ];
    // Out-of-domain equivalents
    let shifted_vertices = vec![
        vertex!([2.2_f64, 3.3]).unwrap(), // → (0.2, 0.3)
        vertex!([-0.2, 1.1]).unwrap(),    // → (0.8, 0.1)
        vertex!([1.5, 0.7]).unwrap(),     // → (0.5, 0.7)
        vertex!([-0.4, 2.9]).unwrap(),    // → (0.6, 0.9)
    ];

    let dt_canonical = DelaunayTriangulationBuilder::new(&canonical_vertices)
        .try_canonicalized_toroidal([1.0, 1.0])
        .unwrap()
        .build()
        .expect("canonical build should succeed");

    let dt_shifted = DelaunayTriangulationBuilder::new(&shifted_vertices)
        .try_canonicalized_toroidal([1.0, 1.0])
        .unwrap()
        .build()
        .expect("shifted build should succeed");

    assert_eq!(
        dt_canonical.number_of_vertices(),
        dt_shifted.number_of_vertices(),
        "Both inputs should produce the same number of vertices after canonicalization"
    );
    assert_eq!(
        dt_canonical.number_of_simplices(),
        dt_shifted.number_of_simplices(),
        "Both inputs should produce the same number of simplices after canonicalization"
    );
}

/// Full Levels 1-3 validation passes after canonicalizing inputs into a toroidal domain.
#[test]
fn test_builder_canonicalized_toroidal_validates_2d() {
    let vertices = vec![
        vertex!([0.2_f64, 0.3]).unwrap(),
        vertex!([0.8, 0.1]).unwrap(),
        vertex!([0.5, 0.7]).unwrap(),
        vertex!([0.1, 0.9]).unwrap(),
    ];
    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .try_canonicalized_toroidal([1.0, 1.0])
        .unwrap()
        .build()
        .expect("canonicalized toroidal build should succeed");

    assert!(
        dt.as_triangulation().validate().is_ok(),
        "Levels 1-3 validation should pass after toroidal-domain input canonicalization"
    );
}

/// Level 5 (Delaunay property) validation passes after toroidal-domain input canonicalization.
#[test]
fn test_builder_canonicalized_toroidal_delaunay_property_valid_2d() {
    let vertices = vec![
        vertex!([0.2_f64, 0.3]).unwrap(),
        vertex!([0.8, 0.1]).unwrap(),
        vertex!([0.5, 0.7]).unwrap(),
        vertex!([0.1, 0.9]).unwrap(),
    ];
    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .try_canonicalized_toroidal([1.0, 1.0])
        .unwrap()
        .build()
        .expect("canonicalized toroidal build should succeed");

    assert!(
        dt.validate().is_ok(),
        "Full Levels 1-4 validation should pass after toroidal-domain input canonicalization"
    );
}

/// `dim()` is 2 for four non-degenerate 2D points.
#[test]
fn test_builder_canonicalized_toroidal_2d_euler_dimension() {
    let vertices = vec![
        vertex!([0.2_f64, 0.3]).unwrap(),
        vertex!([0.8, 0.1]).unwrap(),
        vertex!([0.5, 0.7]).unwrap(),
        vertex!([0.1, 0.9]).unwrap(),
    ];
    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .try_canonicalized_toroidal([1.0, 1.0])
        .unwrap()
        .build()
        .expect("build should succeed");

    assert_eq!(
        dt.dim(),
        2,
        "Four 2D points should produce a 2-dimensional triangulation"
    );
}

/// Building with canonicalized toroidal wrapping on already-canonical input matches Euclidean construction.
#[test]
fn test_builder_canonicalized_toroidal_matches_euclidean_on_canonical_input() {
    let vertices = vec![
        vertex!([0.1_f64, 0.2]).unwrap(),
        vertex!([0.8, 0.3]).unwrap(),
        vertex!([0.4, 0.9]).unwrap(),
    ];
    let dt_euclidean = DelaunayTriangulationBuilder::new(&vertices)
        .build()
        .expect("euclidean build should succeed");
    let dt_toroidal = DelaunayTriangulationBuilder::new(&vertices)
        .try_canonicalized_toroidal([1.0, 1.0])
        .unwrap()
        .build()
        .expect("canonicalized toroidal build should succeed");

    assert_eq!(
        dt_euclidean.number_of_vertices(),
        dt_toroidal.number_of_vertices()
    );
    assert_eq!(
        dt_euclidean.number_of_simplices(),
        dt_toroidal.number_of_simplices()
    );
}

/// Larger 2D canonicalized toroidal point set — full Level 1–4 validation.
///
/// Uses eight hand-picked, well-separated points to stay in general position
/// with the default `FastKernel`, exercising the canonicalized toroidal build path
/// beyond the minimal 3–4 vertex cases covered by other tests.
#[test]
fn test_builder_canonicalized_toroidal_larger_point_set_2d() {
    let vertices = vec![
        vertex!([0.1_f64, 0.2]).unwrap(),
        vertex!([0.6, 0.1]).unwrap(),
        vertex!([0.9, 0.4]).unwrap(),
        vertex!([0.7, 0.8]).unwrap(),
        vertex!([0.3, 0.7]).unwrap(),
        vertex!([0.48, 0.52]).unwrap(),
        vertex!([0.15, 0.85]).unwrap(),
        vertex!([0.8, 0.65]).unwrap(),
    ];
    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .try_canonicalized_toroidal([1.0, 1.0])
        .unwrap()
        .build()
        .expect("larger canonicalized toroidal build should succeed");

    assert_eq!(dt.number_of_vertices(), 8);
    assert!(dt.validate().is_ok(), "Full validation should pass");
}

// =============================================================================
// Custom kernel
// =============================================================================

/// `build_with_kernel` with `RobustKernel` works for a 3D canonicalized toroidal case.
#[test]
fn test_builder_canonicalized_toroidal_robust_kernel_3d() {
    let vertices = vec![
        vertex!([0.2_f64, 0.3, 0.4]).unwrap(),
        vertex!([0.8, 0.1, 0.2]).unwrap(),
        vertex!([0.5, 0.7, 0.6]).unwrap(),
        vertex!([0.1, 0.9, 0.3]).unwrap(),
        vertex!([0.6, 0.4, 0.8]).unwrap(),
    ];
    let kernel = RobustKernel::new();
    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .try_canonicalized_toroidal([1.0, 1.0, 1.0])
        .unwrap()
        .build_with_kernel(&kernel)
        .expect("canonicalized toroidal robust kernel 3D build should succeed");

    assert_eq!(dt.number_of_vertices(), 5);
    assert!(dt.validate().is_ok());
}

// =============================================================================
// Periodic (image-point method) path
// =============================================================================

fn toroidal_vertices<const D: usize>() -> Vec<Vertex<(), D>> {
    assert!((2..=5).contains(&D));

    let multipliers = [
        0.618_033_988_749_894_8,
        0.414_213_562_373_095_03,
        0.732_050_807_568_877_2,
        0.236_067_977_499_789_8,
        0.324_717_957_244_746,
    ];
    (0..(2 * D + 3))
        .map(|index| {
            let mut coords = [0.0_f64; D];
            let index_f64 = f64::from(u32::try_from(index).expect("test index fits in u32"));
            for axis in 0..D {
                let axis_f64 = f64::from(u32::try_from(axis).expect("test axis fits in u32"));
                let stride = 0.037_f64.mul_add(axis_f64 + 1.0, multipliers[axis]);
                let phase = (index_f64 + 1.0) * stride;
                coords[axis] = 0.9_f64.mul_add(phase.fract(), 0.05);
            }
            vertex!(coords).unwrap()
        })
        .collect()
}

fn build_toroidal_triangulation<const D: usize>()
-> DelaunayTriangulation<RobustKernel<f64>, (), (), D> {
    let vertices = toroidal_vertices::<D>();
    let expected_vertices = vertices.len();
    let kernel = RobustKernel::new();
    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .try_toroidal([1.0_f64; D])
        .unwrap()
        .build_with_kernel(&kernel)
        .expect("periodic build should succeed");

    assert_eq!(dt.number_of_vertices(), expected_vertices);
    assert!(
        dt.global_topology().is_toroidal(),
        "global_topology should be Toroidal"
    );
    assert!(
        dt.global_topology().is_periodic(),
        "global_topology should use periodic image-point construction"
    );
    dt
}

fn count_boundary_facets<K, U, V, const D: usize>(dt: &DelaunayTriangulation<K, U, V, D>) -> usize {
    dt.boundary_facets()
        .unwrap()
        .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))
        .unwrap()
}

/// `toroidal` builds a valid 2D periodic triangulation with χ = 0.
///
/// Verifies structural validity and χ = 0 directly.
/// See the macro-generated toroidal validation tests for the full
/// `PLManifold` and Level 4 `validate()` paths.
#[test]
fn test_builder_toroidal_chi_zero_2d() {
    let dt = build_toroidal_triangulation::<2>();

    assert!(
        dt.is_valid_structure().is_ok(),
        "structural validity should pass for periodic triangulation"
    );
    let counts = dt.simplex_counts().unwrap();
    let chi = euler_characteristic(&counts);
    assert_eq!(
        chi, 0,
        "Euler characteristic of periodic 2D triangulation must be 0 (torus)"
    );

    let semantic_result = dt.euler_check().unwrap();
    assert_eq!(
        semantic_result.classification,
        TopologyClassification::ClosedToroid(2),
        "topology-aware Euler validation must classify the periodic quotient as closed toroidal",
    );
    assert_eq!(semantic_result.chi, 0);
    assert_eq!(semantic_result.expected, Some(0));
    assert!(semantic_result.is_valid());
}

/// The same input points have Euclidean hull boundary, but the periodic quotient is closed.
#[test]
fn test_builder_toroidal_boundary_query_is_topology_aware() {
    let vertices = toroidal_vertices::<2>();
    let kernel = RobustKernel::new();

    let euclidean = DelaunayTriangulationBuilder::new(&vertices)
        .build_with_kernel(&kernel)
        .expect("Euclidean build should succeed for toroidal fixture points");
    let toroidal = DelaunayTriangulationBuilder::new(&vertices)
        .try_toroidal([1.0_f64; 2])
        .unwrap()
        .build_with_kernel(&kernel)
        .expect("periodic toroidal builder should succeed");

    assert_eq!(
        euclidean.number_of_vertices(),
        toroidal.number_of_vertices()
    );
    assert!(
        count_boundary_facets(&euclidean) > 0,
        "Euclidean triangulation of finite points should expose hull boundary facets",
    );
    assert_eq!(
        count_boundary_facets(&toroidal),
        0,
        "periodic toroidal quotient should be closed with no boundary facets",
    );
}

/// `DelaunayTriangulation::builder()` uses `.try_toroidal()` for the periodic quotient path.
#[test]
fn test_builder_toroidal_convenience() {
    let vertices = toroidal_vertices::<2>();
    let kernel = RobustKernel::new();
    let dt = DelaunayTriangulation::builder(&vertices)
        .try_toroidal([1.0_f64; 2])
        .unwrap()
        .build_with_kernel(&kernel)
        .expect("periodic toroidal builder should succeed");

    assert!(dt.global_topology().is_periodic());
    assert!(dt.is_valid_structure().is_ok());
}

/// Periodic quotient construction exposes the same fluent statistics terminal.
#[test]
fn test_builder_build_with_statistics_periodic_toroidal_records_telemetry() {
    let vertices = toroidal_vertices::<2>();
    let kernel = RobustKernel::new();
    let (dt, stats) = DelaunayTriangulationBuilder::new(&vertices)
        .try_toroidal([1.0_f64; 2])
        .unwrap()
        .build_with_kernel_and_statistics(&kernel)
        .expect("periodic toroidal statistics build should succeed");

    assert_eq!(dt.number_of_vertices(), stats.inserted);
    assert_eq!(stats.total_skipped(), 0);
    assert_eq!(stats.total_attempts, 0);
    assert_eq!(stats.telemetry.insertion_wall_time_calls, 0);
    assert!(stats.telemetry.has_data());
    assert!(stats.telemetry.construction_total_nanos > 0);
    assert!(dt.global_topology().is_periodic());
    assert!(dt.validate().is_ok());
}

macro_rules! gen_toroidal_validation_test {
    ($dim:literal, $label:ident, $run_level4:expr $(, #[$attr:meta])?) => {
        pastey::paste! {
            /// `toroidal` validation for this dimension.
            ///
            /// Periodic-aware ridge and vertex link validation correctly handles reused
            /// vertex keys via lifted vertex identity. When `$run_level4` is true, this
            /// also exercises Level 4 periodic lifted Delaunay predicates.
            #[test]
            $(#[$attr])?
            fn [<test_builder_toroidal_validate_ $label _ $dim d>]() {
                let dt = build_toroidal_triangulation::<$dim>();

                let topology_result = dt.as_triangulation().validate();
                assert!(
                    topology_result.is_ok(),
                    "PLManifold Levels 1-3 validate() should pass for toroidal: {:?}",
                    topology_result.err()
                );

                if $run_level4 {
                    let level4_result = dt.validate();
                    assert!(
                        level4_result.is_ok(),
                        "Level 1-4 validate() should pass for toroidal: {:?}",
                        level4_result.err()
                    );
                }
            }
        }
    };
}

gen_toroidal_validation_test!(2, levels_1_to_4, true);
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-mode guardrail; debug/coverage quotient search is intentionally skipped"
)]
fn test_builder_toroidal_3d_fails_fast_until_scalable_quotient() {
    let vertices = vec![
        vertex!([0.2_f64, 0.3, 0.4]).unwrap(),
        vertex!([0.8, 0.1, 0.2]).unwrap(),
        vertex!([0.5, 0.7, 0.6]).unwrap(),
        vertex!([0.1, 0.9, 0.3]).unwrap(),
        vertex!([0.6, 0.4, 0.8]).unwrap(),
        vertex!([0.3, 0.5, 0.9]).unwrap(),
        vertex!([0.9, 0.2, 0.6]).unwrap(),
    ];
    let kernel = RobustKernel::new();
    let err = DelaunayTriangulationBuilder::new(&vertices)
        .try_toroidal([1.0_f64; 3])
        .unwrap()
        .build_with_kernel(&kernel)
        .expect_err("compact periodic 3D quotient remains pending scalable selection");

    match err {
        DelaunayTriangulationConstructionError::Triangulation(
            DelaunayConstructionFailure::PeriodicQuotientSelectionIncompleteCoverage {
                dimension,
                covered_vertex_count,
                canonical_vertex_count,
            },
        ) => {
            assert_eq!(dimension, 3);
            assert!(covered_vertex_count < canonical_vertex_count);
        }
        other => panic!("expected 3D periodic quotient coverage guardrail, got {other:?}"),
    }
}

macro_rules! gen_toroidal_high_dim_guardrail_test {
    ($dim:literal) => {
        pastey::paste! {
            #[test]
            fn [<test_builder_toroidal_ $dim d_fails_fast_until_scalable_quotient>]() {
                let vertices = toroidal_vertices::<$dim>();
                let kernel = RobustKernel::new();
                let err = DelaunayTriangulationBuilder::new(&vertices)
                    .try_toroidal([1.0_f64; $dim])
                    .unwrap()
                    .build_with_kernel(&kernel)
                    .expect_err(concat!(
                        stringify!($dim),
                        "D periodic quotient construction should fail fast pending #416"
                    ));

                match err {
                    DelaunayTriangulationConstructionError::Triangulation(
                        DelaunayConstructionFailure::UnsupportedPeriodicDimension {
                            dimension,
                            max_validated_dimension,
                            tracking_issue,
                        },
                    ) => {
                        assert_eq!(dimension, $dim);
                        assert_eq!(max_validated_dimension, 3);
                        assert_eq!(tracking_issue, 416);
                    }
                    other => panic!("expected high-dimensional periodic guardrail, got {other:?}"),
                }
            }
        }
    };
}

gen_toroidal_high_dim_guardrail_test!(4);
gen_toroidal_high_dim_guardrail_test!(5);

#[test]
fn test_builder_toroidal_large_dimension_fails_before_expansion_math() {
    let vertices: Vec<Vertex<(), 64>> = Vec::new();
    let kernel = RobustKernel::new();
    let err = DelaunayTriangulationBuilder::new(&vertices)
        .try_toroidal([1.0_f64; 64])
        .unwrap()
        .build_with_kernel(&kernel)
        .expect_err("64D periodic quotient should fail before computing 3^D image count");

    match err {
        DelaunayTriangulationConstructionError::Triangulation(
            DelaunayConstructionFailure::UnsupportedPeriodicDimension {
                dimension,
                max_validated_dimension,
                tracking_issue,
            },
        ) => {
            assert_eq!(dimension, 64);
            assert_eq!(max_validated_dimension, 3);
            assert_eq!(tracking_issue, 416);
        }
        other => panic!("expected high-dimensional periodic guardrail, got {other:?}"),
    }
}

/// Explicit 7-vertex torus (Heawood triangulation) with `GlobalTopology::Toroidal`
/// is rejected until explicit non-Euclidean construction has quotient realization validation.
///
/// The 14-triangle closed mesh has χ = 0 (torus), but explicit quotient
/// connectivity cannot yet be validated against a faithful quotient realization.
#[test]
fn test_explicit_toroidal_heawood_torus_rejected() {
    // Regular heptagon: 7 well-separated points, no 3 collinear.
    let vertices: Vec<_> = (0..7)
        .map(|i| {
            let angle = TAU * f64::from(i) / 7.0;
            vertex!([angle.cos(), angle.sin()]).unwrap()
        })
        .collect();

    // Heawood triangulation: two families of 7 triangles each, 14 total.
    // Family 1: {i, i+1, i+3} mod 7
    // Family 2: {i, i+2, i+3} mod 7
    let mut simplices: Vec<Vec<usize>> = Vec::with_capacity(14);
    for i in 0..7 {
        simplices.push(vec![i, (i + 1) % 7, (i + 3) % 7]);
        simplices.push(vec![i, (i + 2) % 7, (i + 3) % 7]);
    }

    let topology =
        GlobalTopology::try_toroidal([2.0, 2.0], ToroidalConstructionMode::Explicit).unwrap();
    let err = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
        .unwrap()
        .global_topology(topology)
        .build()
        .expect_err("explicit toroidal connectivity requires a quotient realization validator");

    match err {
        DelaunayTriangulationConstructionError::ExplicitConstruction(
            ExplicitConstructionError::UnsupportedExplicitTopology { topology },
        ) => assert_eq!(topology, TopologyKind::Toroidal),
        other => panic!("expected explicit construction validation failure, got {other:?}"),
    }
}

/// Explicit 7-vertex torus with Euclidean `global_topology` fails Euler validation.
///
/// Same mesh as above but without the Toroidal override. The heuristic classifies
/// the closed mesh as `ClosedSphere(2)` (expected χ = 2), but actual χ = 0.
#[test]
fn test_explicit_toroidal_torus_euler_mismatch_without_override() {
    let vertices: Vec<_> = (0..7)
        .map(|i| {
            let angle = TAU * f64::from(i) / 7.0;
            vertex!([angle.cos(), angle.sin()]).unwrap()
        })
        .collect();

    let mut simplices: Vec<Vec<usize>> = Vec::with_capacity(14);
    for i in 0..7 {
        simplices.push(vec![i, (i + 1) % 7, (i + 3) % 7]);
        simplices.push(vec![i, (i + 2) % 7, (i + 3) % 7]);
    }

    // Build with default Euclidean topology — should fail at Euler validation.
    let err = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
        .unwrap()
        .build()
        .expect_err("explicit torus without Toroidal metadata should fail Euler validation");

    match err {
        DelaunayTriangulationConstructionError::ExplicitConstruction(
            ExplicitConstructionError::TopologyValidation { source },
        ) => {
            assert_matches!(
                source.as_ref(),
                InvariantError::Triangulation(
                    TriangulationValidationError::EulerCharacteristicMismatch { .. }
                )
            );
        }
        DelaunayTriangulationConstructionError::ExplicitConstruction(
            ExplicitConstructionError::OrientationNormalization { source },
        ) => {
            assert_matches!(
                source.as_ref(),
                InsertionError::TopologyValidationFailed {
                    source: TriangulationValidationError::OrientationPromotionNonConvergence { .. },
                    ..
                }
            );
        }
        other => {
            panic!("expected explicit topology or orientation-normalization failure, got {other:?}")
        }
    }
}

// =============================================================================
// Explicit construction (try_from_vertices_and_simplices)
// =============================================================================

fn explicit_builder_parse_error<U, const D: usize>(
    vertices: &[Vertex<U, D>],
    simplices: &[Vec<usize>],
) -> ExplicitConstructionError {
    match DelaunayTriangulationBuilder::try_from_vertices_and_simplices(vertices, simplices) {
        Ok(_) => panic!("explicit simplex specs should be rejected before builder storage"),
        Err(err) => err,
    }
}

/// Builds the origin and unit-basis facet plus an equal-coordinate apex.
/// An apex sum above one places it across the shared facet from the origin.
fn shared_facet_vertices<const D: usize>(apex_coordinate_sum: f64) -> Vec<Vertex<(), D>> {
    let dim = u32::try_from(D).expect("test dimension fits in u32");
    let high_apex_coord = apex_coordinate_sum / f64::from(dim);
    let mut vertices = Vec::with_capacity(D + 2);
    vertices.push(Vertex::try_new([0.0; D]).expect("origin vertex should be valid"));

    for axis in 0..D {
        let mut coords = [0.0; D];
        coords[axis] = 1.0;
        vertices.push(Vertex::try_new(coords).expect("basis vertex should be valid"));
    }

    vertices
        .push(Vertex::try_new([high_apex_coord; D]).expect("opposite apex vertex should be valid"));
    vertices
}

/// Connects the origin and apex to the unit-basis facet as two D-simplices,
/// exercising a valid realization that can fail the Delaunay predicate.
fn shared_facet_simplices<const D: usize>() -> Vec<Vec<usize>> {
    let low_simplex = (0..=D).collect();
    let mut high_simplex: Vec<usize> = (1..=D).collect();
    high_simplex.push(D + 1);
    vec![low_simplex, high_simplex]
}

/// Embeds a crossing 2D triangle strip in the first two coordinates and adds
/// one unit cone vertex along every additional coordinate axis.
fn crossing_cone_vertices<const D: usize>() -> Vec<Vertex<(), D>> {
    let base_coords = [
        [2.0, 0.0],
        [3.0, -2.0],
        [-2.0, -1.0],
        [0.0, -2.0],
        [3.0, -4.0],
        [0.0, 3.0],
    ];
    let mut vertices = Vec::with_capacity(D + 4);

    for [x, y] in base_coords {
        let mut coords = [0.0; D];
        coords[0] = x;
        coords[1] = y;
        vertices.push(Vertex::try_new(coords).expect("base crossing vertex should be valid"));
    }

    for axis in 2..D {
        let mut coords = [0.0; D];
        coords[axis] = 1.0;
        vertices.push(Vertex::try_new(coords).expect("cone vertex should be valid"));
    }

    vertices
}

/// Extends each crossing-strip triangle by all cone-axis vertices, preserving
/// its unintended intersection as a D-dimensional realization failure.
fn crossing_cone_simplices<const D: usize>() -> Vec<Vec<usize>> {
    let base_simplices = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]];
    let cone_vertices: Vec<usize> = (6..D + 4).collect();

    base_simplices
        .into_iter()
        .map(|base| {
            let mut simplex = base.to_vec();
            simplex.extend(cone_vertices.iter().copied());
            simplex
        })
        .collect()
}

fn assert_relaxed_explicit_non_delaunay_succeeds<const D: usize>() {
    let vertices = shared_facet_vertices::<D>(1.1);
    let simplices = shared_facet_simplices::<D>();

    let dt = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
        .expect("explicit simplex specs should validate")
        .construction_options(ConstructionOptions::default().without_final_delaunay_enforcement())
        .build()
        .expect("relaxed explicit construction should accept a realized non-Delaunay mesh");

    assert_eq!(dt.number_of_vertices(), D + 2);
    assert_eq!(dt.number_of_simplices(), 2);
    dt.as_triangulation()
        .validate_realization()
        .expect("relaxed explicit mesh should pass Levels 1-4");
    assert!(
        dt.is_valid_delaunay().is_err(),
        "fixture should still violate Level 5 Delaunay predicates",
    );
}

fn assert_relaxed_explicit_invalid_realization_fails<const D: usize>() {
    let vertices = crossing_cone_vertices::<D>();
    let simplices = crossing_cone_simplices::<D>();

    let err = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
        .expect("explicit simplex specs should validate")
        .construction_options(ConstructionOptions::default().without_final_delaunay_enforcement())
        .build()
        .expect_err("relaxed explicit construction should still reject invalid realizations");

    match err {
        DelaunayTriangulationConstructionError::ExplicitConstruction(
            ExplicitConstructionError::RealizationValidation { source },
        ) => assert_matches!(
            source.as_ref(),
            DelaunayTriangulationValidationError::Realization(_)
        ),
        other => panic!("expected relaxed explicit realization-validation failure, got {other:?}"),
    }
}

macro_rules! gen_relaxed_explicit_validation_tests {
    ($dim:literal) => {
        pastey::paste! {
            #[test]
            fn [<test_relaxed_explicit_non_delaunay_mesh_succeeds_ $dim d>]() {
                assert_relaxed_explicit_non_delaunay_succeeds::<$dim>();
            }

            #[test]
            fn [<test_relaxed_explicit_invalid_realization_fails_ $dim d>]() {
                assert_relaxed_explicit_invalid_realization_fails::<$dim>();
            }
        }
    };
}

gen_relaxed_explicit_validation_tests!(2);
gen_relaxed_explicit_validation_tests!(3);
gen_relaxed_explicit_validation_tests!(4);
gen_relaxed_explicit_validation_tests!(5);

/// 2D: Build two triangles forming a quad from explicit vertices and simplices.
#[test]
fn test_explicit_2d_two_triangle_quad() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([1.0, 1.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
    ];
    let simplices = vec![vec![0, 1, 2], vec![0, 2, 3]];

    let dt = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
        .unwrap()
        .build()
        .expect("explicit 2D build should succeed");

    assert_eq!(dt.number_of_vertices(), 4);
    assert_eq!(dt.number_of_simplices(), 2);
    assert!(
        dt.is_valid_structure().is_ok(),
        "triangulation should be structurally valid"
    );

    // Verify neighbor pointers: the two triangles share the edge (0,2) so
    // each should have the other as a neighbor.
    let mut neighbor_count = 0;
    for (_, simplex) in dt.simplices() {
        if let Some(neighbors) = simplex.neighbors() {
            for n in neighbors {
                if n.is_some() {
                    neighbor_count += 1;
                }
            }
        }
    }
    // Two simplices sharing one facet → 2 neighbor slots filled (one in each simplex).
    assert!(
        neighbor_count >= 2,
        "Shared facet should produce neighbor pointers"
    );
}

/// Explicit connectivity exposes the same fluent statistics terminal.
#[test]
fn test_builder_build_with_statistics_explicit_records_telemetry() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([1.0, 1.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
    ];
    let simplices = vec![vec![0, 1, 2], vec![0, 2, 3]];

    let (dt, stats) =
        DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
            .unwrap()
            .build_with_statistics()
            .expect("explicit statistics build should succeed");

    assert_eq!(dt.number_of_vertices(), stats.inserted);
    assert_eq!(stats.total_skipped(), 0);
    assert_eq!(stats.total_attempts, 0);
    assert_eq!(stats.telemetry.insertion_wall_time_calls, 0);
    assert!(stats.telemetry.has_data());
    assert!(stats.telemetry.construction_total_nanos > 0);
    assert!(dt.validate().is_ok());
}

/// Explicit construction normalizes incoherent local simplex orderings.
///
/// Swapping two vertices in one simplex flips its local orientation relative to an
/// otherwise valid two-triangle mesh. The builder should repair that internal
/// ordering detail and still produce a structurally valid TDS.
#[test]
fn test_explicit_normalizes_incoherent_simplex_order() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([1.0, 1.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
    ];
    let mut simplices = vec![vec![0, 1, 2], vec![0, 2, 3]];
    simplices[1].swap(0, 1);

    let dt = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
        .unwrap()
        .build()
        .expect("explicit build should normalize incoherent simplex ordering");

    assert!(
        dt.is_valid_structure().is_ok(),
        "builder should canonicalize incoherent simplex orderings into a valid structure"
    );
}

/// 3D: Build two tetrahedra sharing a face.
#[test]
fn test_explicit_3d_two_tetrahedra() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 1.0]).unwrap(),
        vertex!([1.0, 1.0, -1.0]).unwrap(),
    ];
    // Two tetrahedra sharing face (0, 1, 2)
    let simplices = vec![vec![0, 1, 2, 3], vec![0, 1, 2, 4]];

    let dt = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
        .unwrap()
        .build()
        .expect("explicit 3D build should succeed");

    assert_eq!(dt.number_of_vertices(), 5);
    assert_eq!(dt.number_of_simplices(), 2);
    assert!(
        dt.is_valid_structure().is_ok(),
        "triangulation should be structurally valid"
    );
}

/// Round-trip 3D: Build via Delaunay → extract → reconstruct via explicit → same structure.
#[test]
fn test_explicit_round_trip_3d() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 1.0]).unwrap(),
        vertex!([1.0, 1.0, 1.0]).unwrap(),
    ];

    let dt_original = DelaunayTriangulation::builder(&vertices)
        .build()
        .expect("Delaunay build should succeed");
    let original_vertex_count = dt_original.number_of_vertices();
    let original_simplex_count = dt_original.number_of_simplices();

    let vertex_keys: Vec<_> = dt_original.vertices().map(|(key, _)| key).collect();
    let key_to_index: HashMap<_, _> = vertex_keys
        .iter()
        .enumerate()
        .map(|(idx, &vk)| (vk, idx))
        .collect();

    let extracted_vertices: Vec<_> = vertex_keys
        .iter()
        .map(|&vk| *dt_original.vertex(vk).unwrap())
        .collect();

    let mut simplex_specs: Vec<Vec<usize>> = Vec::new();
    for (_, simplex) in dt_original.simplices() {
        let spec: Vec<usize> = simplex
            .vertices()
            .iter()
            .map(|vk| key_to_index[vk])
            .collect();
        simplex_specs.push(spec);
    }

    let dt_reconstructed = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(
        &extracted_vertices,
        &simplex_specs,
    )
    .unwrap()
    .build()
    .expect("explicit 3D reconstruction should succeed");

    assert_eq!(dt_reconstructed.number_of_vertices(), original_vertex_count);
    assert_eq!(
        dt_reconstructed.number_of_simplices(),
        original_simplex_count
    );
    assert!(
        dt_reconstructed.is_valid_structure().is_ok(),
        "reconstructed 3D triangulation should be structurally valid"
    );
}

/// Round-trip: Build via Delaunay → extract → reconstruct via explicit → same structure.
#[test]
fn test_explicit_round_trip_2d() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
        vertex!([1.0, 1.0]).unwrap(),
    ];

    // Build via standard Delaunay.
    let dt_original = DelaunayTriangulation::builder(&vertices)
        .build()
        .expect("Delaunay build should succeed");
    let original_vertex_count = dt_original.number_of_vertices();
    let original_simplex_count = dt_original.number_of_simplices();

    // Extract vertex keys → index mapping and simplex specifications.
    let vertex_keys: Vec<_> = dt_original.vertices().map(|(key, _)| key).collect();
    let key_to_index: HashMap<_, _> = vertex_keys
        .iter()
        .enumerate()
        .map(|(idx, &vk)| (vk, idx))
        .collect();

    let extracted_vertices: Vec<_> = vertex_keys
        .iter()
        .map(|&vk| *dt_original.vertex(vk).unwrap())
        .collect();

    let mut simplex_specs: Vec<Vec<usize>> = Vec::new();
    for (_, simplex) in dt_original.simplices() {
        let spec: Vec<usize> = simplex
            .vertices()
            .iter()
            .map(|vk| key_to_index[vk])
            .collect();
        simplex_specs.push(spec);
    }

    // Reconstruct via explicit.
    let dt_reconstructed = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(
        &extracted_vertices,
        &simplex_specs,
    )
    .unwrap()
    .build()
    .expect("explicit reconstruction should succeed");

    assert_eq!(dt_reconstructed.number_of_vertices(), original_vertex_count);
    assert_eq!(
        dt_reconstructed.number_of_simplices(),
        original_simplex_count
    );
    assert!(
        dt_reconstructed.is_valid_structure().is_ok(),
        "reconstructed triangulation should be structurally valid"
    );
}

/// Error: empty simplices should fail.
#[test]
fn test_explicit_error_empty_simplices() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
    ];
    let simplices: Vec<Vec<usize>> = vec![];

    let result =
        DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices);

    assert_matches!(
        result.err(),
        Some(ExplicitConstructionError::EmptySimplices),
        "Empty simplices should produce an error"
    );
}

/// Error: wrong simplex arity.
#[test]
fn test_explicit_error_wrong_arity() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
    ];
    // 2D expects 3 vertices per simplex, but we provide 2.
    let simplices = vec![vec![0, 1]];

    let result =
        DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices);

    assert_matches!(
        result.err(),
        Some(ExplicitConstructionError::InvalidSimplexArity {
            simplex_index: 0,
            actual: 2,
            expected: 3,
        }),
        "Wrong arity should produce an error"
    );
}

/// Error: out-of-bounds vertex index.
#[test]
fn test_explicit_error_index_out_of_bounds() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
    ];
    let simplices = vec![vec![0, 1, 99]]; // 99 is out of bounds

    let result =
        DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices);

    assert_matches!(
        result.err(),
        Some(ExplicitConstructionError::IndexOutOfBounds {
            simplex_index: 0,
            vertex_index: 99,
            bound: 3,
        }),
        "Out-of-bounds index should produce an error"
    );
}

/// Error: duplicate vertex in simplex.
#[test]
fn test_explicit_error_duplicate_vertex_in_simplex() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
    ];
    let simplices = vec![vec![0, 1, 1]]; // Duplicate vertex 1

    let result =
        DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices);

    assert_matches!(
        result.err(),
        Some(ExplicitConstructionError::DuplicateVertexInSimplex {
            simplex_index: 0,
            vertex_index: 1,
        }),
        "Duplicate vertex should produce an error"
    );
}

/// Minimal case: a single triangle in 2D.
#[test]
fn test_explicit_2d_single_triangle() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
    ];
    let simplices = vec![vec![0, 1, 2]];

    let dt = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
        .unwrap()
        .build()
        .expect("single triangle should succeed");

    assert_eq!(dt.number_of_vertices(), 3);
    assert_eq!(dt.number_of_simplices(), 1);
    assert!(dt.is_valid_structure().is_ok());
}

/// Minimal case: a single tetrahedron in 3D.
#[test]
fn test_explicit_3d_single_tetrahedron() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 1.0]).unwrap(),
    ];
    let simplices = vec![vec![0, 1, 2, 3]];

    let dt = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
        .unwrap()
        .build()
        .expect("single tetrahedron should succeed");

    assert_eq!(dt.number_of_vertices(), 4);
    assert_eq!(dt.number_of_simplices(), 1);
    assert!(dt.is_valid_structure().is_ok());
}

/// Non-Delaunay mesh: prescribed connectivity that violates the empty-circumsphere
/// property. Because the builder returns `DelaunayTriangulation`, Level 4
/// validation must reject this connectivity before construction succeeds.
///
/// Geometry: A=(0,0), B=(4,0), C=(4,2), D=(1,2). The circumcircle of ABC has
/// center (2,1) and radius √5. Point D=(1,2) is at distance √2 < √5 from the
/// center, so D lies strictly inside the circumcircle of ABC. Using diagonal AC
/// (simplices [0,1,2] and [0,2,3]) is therefore non-Delaunay. The Delaunay
/// triangulation would use diagonal BD instead.
#[test]
fn test_explicit_non_delaunay_mesh() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]).unwrap(),
        vertex!([4.0, 0.0]).unwrap(),
        vertex!([4.0, 2.0]).unwrap(),
        vertex!([1.0, 2.0]).unwrap(),
    ];
    // Diagonal AC = (0,0)-(4,2): non-Delaunay because D=(1,2) is inside
    // the circumcircle of triangle ABC.
    let simplices = vec![vec![0, 1, 2], vec![0, 2, 3]];

    let err = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
        .unwrap()
        .build()
        .expect_err("non-Delaunay mesh must not construct a DelaunayTriangulation");

    assert!(
        matches!(
            err,
            DelaunayTriangulationConstructionError::ExplicitConstruction(
                ExplicitConstructionError::DelaunayValidation { .. }
            )
        ),
        "expected explicit validation failure, got {err:?}"
    );
    assert!(
        err.to_string().contains("Delaunay validation failed"),
        "error should identify the Level 5 validation failure: {err}"
    );
}

/// Topology guarantee is propagated through explicit construction.
#[test]
fn test_explicit_topology_guarantee_propagated() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
    ];
    let simplices = vec![vec![0, 1, 2]];

    let dt = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
        .unwrap()
        .topology_guarantee(TopologyGuarantee::Pseudomanifold)
        .build()
        .expect("build should succeed");

    assert_eq!(dt.topology_guarantee(), TopologyGuarantee::Pseudomanifold);
}

/// Explicit construction preserves vertex data (U ≠ ()).
#[test]
fn test_explicit_preserves_vertex_data() {
    let vertices: Vec<Vertex<i32, 2>> = vec![
        vertex!([0.0, 0.0]; data = 10_i32).unwrap(),
        vertex!([1.0, 0.0]; data = 20_i32).unwrap(),
        vertex!([0.0, 1.0]; data = 30_i32).unwrap(),
    ];
    let simplices = vec![vec![0, 1, 2]];

    let dt = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
        .unwrap()
        .build()
        .expect("explicit build with vertex data should succeed");

    let mut data: Vec<i32> = dt
        .vertices()
        .filter_map(|(_, v)| v.data().copied())
        .collect();
    data.sort_unstable();
    assert_eq!(
        data,
        vec![10, 20, 30],
        "Vertex data must survive explicit construction"
    );
}

/// Full `validate()` (Levels 1–5) on a Delaunay-compatible explicit mesh.
#[test]
fn test_explicit_validate_delaunay_mesh() {
    // Use a known Delaunay configuration: the standard simplex.
    let vertices = vec![
        vertex!([0.0_f64, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.5, 0.866_025_403_784_438_6]).unwrap(),
    ];
    let simplices = vec![vec![0, 1, 2]];

    let dt = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
        .unwrap()
        .build()
        .expect("build should succeed");

    assert!(
        dt.validate().is_ok(),
        "Full Levels 1-4 validation should pass for Delaunay-compatible explicit mesh"
    );
}

/// Unreferenced vertices fail Level 3 topology validation (isolated vertex).
#[test]
fn test_explicit_unreferenced_vertices_rejected() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
        vertex!([5.0, 5.0]).unwrap(), // Not referenced by any simplex
    ];
    let simplices = vec![vec![0, 1, 2]];

    let err = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
        .unwrap()
        .build()
        .unwrap_err();

    assert!(
        matches!(
            err,
            DelaunayTriangulationConstructionError::ExplicitConstruction(
                ExplicitConstructionError::TopologyValidation { .. }
            )
        ),
        "Unreferenced vertices should produce TopologyValidation, got: {err}"
    );
}

/// Error variant: empty simplices returns ExplicitConstruction(EmptySimplices).
#[test]
fn test_explicit_error_variant_empty_simplices() {
    let vertices = vec![vertex!([0.0_f64, 0.0]).unwrap()];
    let simplices: Vec<Vec<usize>> = vec![];

    let err = explicit_builder_parse_error(&vertices, &simplices);

    assert!(
        matches!(err, ExplicitConstructionError::EmptySimplices),
        "Expected ExplicitConstruction(EmptySimplices), got: {err}"
    );
}

/// Error variant: wrong arity returns ExplicitConstruction(InvalidSimplexArity { .. }).
#[test]
fn test_explicit_error_variant_wrong_arity() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
    ];
    let simplices = vec![vec![0, 1]]; // 2D expects 3 vertices

    let err = explicit_builder_parse_error(&vertices, &simplices);

    assert!(
        matches!(
            err,
            ExplicitConstructionError::InvalidSimplexArity {
                simplex_index: 0,
                actual: 2,
                expected: 3
            }
        ),
        "Expected InvalidSimplexArity, got: {err}"
    );
}

/// Error variant: non-manifold facet sharing is rejected during TDS insertion.
#[test]
fn test_explicit_error_variant_non_manifold_facet() {
    // Three triangles sharing the same edge (0,1) — facet shared by 3 simplices
    // violates the 2-manifold property.
    let vertices = vec![
        vertex!([0.0_f64, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
        vertex!([1.0, 1.0]).unwrap(),
        vertex!([0.5, -1.0]).unwrap(),
    ];
    let simplices = vec![vec![0, 1, 2], vec![0, 1, 3], vec![0, 1, 4]];

    let err = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
        .unwrap()
        .build()
        .unwrap_err();

    let DelaunayTriangulationConstructionError::ExplicitConstruction(
        ExplicitConstructionError::TdsAssembly { source },
    ) = &err
    else {
        panic!("Expected explicit TDS assembly failure, got: {err}");
    };

    assert_matches!(
        source.as_ref(),
        TdsConstructionError::ValidationError(TdsError::ExplicitFacetSharingViolation {
            facet_vertex_indices,
            existing_incident_count: 2,
            attempted_incident_count: 3,
            max_incident_count: 2,
            candidate_simplex_index: 2,
            candidate_facet_index: 2,
            ..
        }) if facet_vertex_indices == &[0, 1]
    );
}

/// Error variant: duplicate vertex returns ExplicitConstruction(DuplicateVertexInSimplex { .. }).
#[test]
fn test_explicit_error_variant_duplicate_vertex_in_simplex() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
    ];
    let simplices = vec![vec![0, 1, 1]]; // Duplicate vertex 1

    let err = explicit_builder_parse_error(&vertices, &simplices);

    assert!(
        matches!(
            err,
            ExplicitConstructionError::DuplicateVertexInSimplex {
                simplex_index: 0,
                vertex_index: 1,
            }
        ),
        "Expected DuplicateVertexInSimplex, got: {err}"
    );
}

/// Error variant: toroidal + explicit returns ExplicitConstruction(IncompatibleTopology).
#[test]
fn test_explicit_error_variant_incompatible_topology() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
    ];
    let simplices = vec![vec![0, 1, 2]];

    let err = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
        .unwrap()
        .try_canonicalized_toroidal([1.0, 1.0])
        .unwrap()
        .build()
        .unwrap_err();

    assert!(
        matches!(
            err,
            DelaunayTriangulationConstructionError::ExplicitConstruction(
                ExplicitConstructionError::IncompatibleTopology
            )
        ),
        "Expected IncompatibleTopology, got: {err}"
    );
}

/// Error variant: explicit connectivity rejects point-insertion-only construction options.
#[test]
fn test_explicit_error_variant_unsupported_construction_options() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
    ];
    let simplices = vec![vec![0, 1, 2]];

    let err = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
        .unwrap()
        .construction_options(
            ConstructionOptions::default().with_insertion_order(InsertionOrderStrategy::Input),
        )
        .build()
        .unwrap_err();

    assert!(
        matches!(
            err,
            DelaunayTriangulationConstructionError::ExplicitConstruction(
                ExplicitConstructionError::UnsupportedConstructionOptions
            )
        ),
        "Expected UnsupportedConstructionOptions, got: {err}"
    );

    let mixed_err =
        DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
            .unwrap()
            .construction_options(
                ConstructionOptions::default()
                    .without_final_delaunay_enforcement()
                    .with_insertion_order(InsertionOrderStrategy::Input),
            )
            .build()
            .unwrap_err();

    assert!(
        matches!(
            mixed_err,
            DelaunayTriangulationConstructionError::ExplicitConstruction(
                ExplicitConstructionError::UnsupportedConstructionOptions
            )
        ),
        "Expected mixed non-enforcing point-insertion options to be rejected, got: {mixed_err}",
    );
}

/// Error variant: duplicate maximal simplices are rejected during TDS insertion.
#[test]
fn test_explicit_error_variant_duplicate_simplices_structural_validation() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
    ];
    let simplices = vec![vec![0, 1, 2], vec![0, 2, 1]];

    let err = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
        .unwrap()
        .build()
        .unwrap_err();

    let DelaunayTriangulationConstructionError::ExplicitConstruction(
        ExplicitConstructionError::TdsAssembly { source },
    ) = &err
    else {
        panic!("expected explicit TDS assembly failure, got {err:?}");
    };

    assert_matches!(
        source.as_ref(),
        TdsConstructionError::ValidationError(TdsError::DuplicateExplicitSimplices {
            existing_simplex_index: 0,
            duplicate_simplex_index: 1,
            vertex_indices,
            ..
        }) if vertex_indices == &[0, 1, 2]
    );
}

/// Error variant: degenerate explicit simplices fail geometric nondegeneracy validation.
#[test]
fn test_explicit_error_variant_geometric_nondegeneracy() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([2.0, 0.0]).unwrap(),
    ];
    let simplices = vec![vec![0, 1, 2]];

    let err = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
        .unwrap()
        .build()
        .unwrap_err();

    match err {
        DelaunayTriangulationConstructionError::ExplicitConstruction(
            ExplicitConstructionError::GeometricNondegeneracy { source },
        ) => assert_matches!(source.as_ref(), TdsError::Geometric(_)),
        other => panic!("expected explicit geometric nondegeneracy failure, got {other:?}"),
    }
}

/// Error variant: completion-time PL validation preserves vertex-link summaries.
#[test]
fn test_explicit_error_variant_completion_validation_summary() {
    let err = ExplicitConstructionError::CompletionValidation {
        source: Box::new(InvariantError::Triangulation(
            TriangulationValidationError::VertexLinkNotManifold {
                vertex_key: VertexKey::default(),
                link_vertex_count: 0,
                link_simplex_count: 0,
                boundary_facet_count: 0,
                max_degree: 0,
                connected: false,
                interior_vertex: false,
            },
        )),
    };

    match err {
        ExplicitConstructionError::CompletionValidation { source } => {
            assert_matches!(
                source.as_ref(),
                InvariantError::Triangulation(
                    TriangulationValidationError::VertexLinkNotManifold { .. }
                )
            );
        }
        other => panic!("expected explicit completion validation failure, got {other:?}"),
    }
}

/// Error variant: orientation-normalization failures preserve typed insertion summaries.
#[test]
fn test_explicit_error_variant_orientation_normalization_summary() {
    let source = ExplicitConstructionError::OrientationNormalization {
        source: Box::new(InsertionError::TopologyValidation(
            TdsError::InconsistentDataStructure {
                message: "orientation normalization could not establish coherent simplices"
                    .to_string(),
            },
        )),
    };

    match source {
        ExplicitConstructionError::OrientationNormalization { source } => {
            assert_matches!(
                source.as_ref(),
                InsertionError::TopologyValidation(TdsError::InconsistentDataStructure { .. })
            );
            assert!(source.to_string().contains("coherent simplices"));
        }
        other => panic!("expected orientation-normalization variant, got {other:?}"),
    }
}

/// Error variant: out-of-bounds returns ExplicitConstruction(IndexOutOfBounds { .. }).
#[test]
fn test_explicit_error_variant_index_out_of_bounds() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
    ];
    let simplices = vec![vec![0, 1, 99]];

    let err = explicit_builder_parse_error(&vertices, &simplices);

    assert!(
        matches!(
            err,
            ExplicitConstructionError::IndexOutOfBounds {
                simplex_index: 0,
                vertex_index: 99,
                bound: 3,
            }
        ),
        "Expected IndexOutOfBounds, got: {err}"
    );
}
