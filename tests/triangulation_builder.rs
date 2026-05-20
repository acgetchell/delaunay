//! Integration tests for [`DelaunayTriangulationBuilder`].
//!
//! These tests exercise the public API from the outside, using only items exposed
//! through `delaunay::prelude` and `delaunay::builder`.

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::f64::consts::TAU;

use delaunay::prelude::construction::{
    ConstructionOptions, DelaunayTriangulation, DelaunayTriangulationBuilder,
    DelaunayTriangulationConstructionError, ExplicitConstructionError, ExplicitInsertionError,
    ExplicitInsertionErrorKind, ExplicitInvariantError, ExplicitInvariantErrorKind,
    ExplicitTdsErrorKind, InsertionOrderStrategy, TopologyGuarantee, Vertex, VertexBuilder, vertex,
};
use delaunay::prelude::geometry::{Coordinate, Point, RobustKernel};
use delaunay::prelude::insertion::InsertionErrorSourceKind;
use delaunay::prelude::repair::DelaunayRepairError;
use delaunay::prelude::tds::{InvariantErrorSummaryDetail, TriangulationValidationErrorKind};
use delaunay::prelude::topology::spaces::{GlobalTopology, TopologyKind, ToroidalConstructionMode};
use delaunay::prelude::topology::validation::{count_simplices, euler_characteristic};
use delaunay::prelude::validation::ValidationPolicy;

// =============================================================================
// Euclidean path
// =============================================================================

/// Builder with no options set should produce the same triangulation as `new()`.
#[test]
fn test_builder_euclidean_matches_new_2d() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
    ];

    let dt_new = DelaunayTriangulation::new(&vertices).expect("new() should succeed");
    let dt_builder = DelaunayTriangulationBuilder::new(&vertices)
        .build::<()>()
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
        vertex!([0.0_f64, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];
    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .build::<()>()
        .expect("3D build should succeed");
    assert_eq!(dt.number_of_vertices(), 4);
    assert_eq!(dt.dim(), 3);
    assert!(dt.validate().is_ok(), "Level 1-4 validation should pass");
}

/// `TopologyGuarantee` set on the builder is propagated to the triangulation.
#[test]
fn test_builder_topology_guarantee_propagated() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
    ];
    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .topology_guarantee(TopologyGuarantee::Pseudomanifold)
        .build::<()>()
        .expect("build should succeed");
    assert_eq!(dt.topology_guarantee(), TopologyGuarantee::Pseudomanifold);
}

/// Builder derives validation policy from topology guarantee instead of exposing
/// a separate construction-time policy axis.
#[test]
fn test_builder_validation_policy_derived_from_topology_guarantee() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
    ];

    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .topology_guarantee(TopologyGuarantee::PLManifoldStrict)
        .build::<()>()
        .expect("build should succeed");

    assert_eq!(dt.topology_guarantee(), TopologyGuarantee::PLManifoldStrict);
    assert_eq!(dt.validation_policy(), ValidationPolicy::Always);
}

/// Custom `ConstructionOptions` are accepted and the triangulation is valid.
#[test]
fn test_builder_custom_construction_options() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
    ];
    let opts = ConstructionOptions::default().with_insertion_order(InsertionOrderStrategy::Input);
    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .construction_options(opts)
        .build::<()>()
        .expect("build should succeed");
    assert_eq!(dt.number_of_vertices(), 3);
    assert!(dt.validate().is_ok());
}

// =============================================================================
// Convenience entry point
// =============================================================================

/// `DelaunayTriangulation::builder()` convenience method compiles and works.
#[test]
fn test_builder_convenience() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
    ];
    let dt = DelaunayTriangulation::builder(&vertices)
        .build::<()>()
        .expect("builder() convenience method should succeed");
    assert_eq!(dt.number_of_vertices(), 3);
    assert!(dt.validate().is_ok());
}

/// `DelaunayTriangulation::builder()` with toroidal wrapping.
#[test]
fn test_builder_toroidal_convenience() {
    let vertices = vec![
        vertex!([0.2_f64, 0.3]),
        vertex!([1.8, 0.1]), // → (0.8, 0.1)
        vertex!([0.5, 0.7]),
        vertex!([-0.4, 0.9]), // → (0.6, 0.9)
    ];
    let dt = DelaunayTriangulation::builder(&vertices)
        .toroidal([1.0, 1.0])
        .build::<()>()
        .expect("toroidal builder should succeed");
    assert_eq!(dt.number_of_vertices(), 4);
    assert!(dt.as_triangulation().validate().is_ok());
}

// =============================================================================
// Toroidal path
// =============================================================================

/// Out-of-domain coordinates are canonicalized: the resulting triangulation
/// is geometrically identical to building directly from the wrapped coordinates.
#[test]
fn test_builder_toroidal_canonicalizes_coordinates() {
    // Canonical (already-wrapped) coordinates
    let canonical_vertices = vec![
        vertex!([0.2_f64, 0.3]),
        vertex!([0.8, 0.1]),
        vertex!([0.5, 0.7]),
        vertex!([0.6, 0.9]),
    ];
    // Out-of-domain equivalents
    let shifted_vertices = vec![
        vertex!([2.2_f64, 3.3]), // → (0.2, 0.3)
        vertex!([-0.2, 1.1]),    // → (0.8, 0.1)
        vertex!([1.5, 0.7]),     // → (0.5, 0.7)
        vertex!([-0.4, 2.9]),    // → (0.6, 0.9)
    ];

    let dt_canonical = DelaunayTriangulationBuilder::new(&canonical_vertices)
        .toroidal([1.0, 1.0])
        .build::<()>()
        .expect("canonical build should succeed");

    let dt_shifted = DelaunayTriangulationBuilder::new(&shifted_vertices)
        .toroidal([1.0, 1.0])
        .build::<()>()
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

/// Full Levels 1–3 validation passes on a toroidally-built triangulation.
#[test]
fn test_builder_toroidal_validates_2d() {
    let vertices = vec![
        vertex!([0.2_f64, 0.3]),
        vertex!([0.8, 0.1]),
        vertex!([0.5, 0.7]),
        vertex!([0.1, 0.9]),
    ];
    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .toroidal([1.0, 1.0])
        .build::<()>()
        .expect("toroidal build should succeed");

    assert!(
        dt.as_triangulation().validate().is_ok(),
        "Levels 1-3 validation should pass for toroidally-built triangulation"
    );
}

/// Level 4 (Delaunay property) validation passes on a toroidally-built triangulation.
#[test]
fn test_builder_toroidal_delaunay_property_valid_2d() {
    let vertices = vec![
        vertex!([0.2_f64, 0.3]),
        vertex!([0.8, 0.1]),
        vertex!([0.5, 0.7]),
        vertex!([0.1, 0.9]),
    ];
    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .toroidal([1.0, 1.0])
        .build::<()>()
        .expect("toroidal build should succeed");

    assert!(
        dt.validate().is_ok(),
        "Full Levels 1-4 validation should pass for toroidally-built triangulation"
    );
}

/// `dim()` is 2 for four non-degenerate 2D points.
#[test]
fn test_builder_toroidal_2d_euler_dimension() {
    let vertices = vec![
        vertex!([0.2_f64, 0.3]),
        vertex!([0.8, 0.1]),
        vertex!([0.5, 0.7]),
        vertex!([0.1, 0.9]),
    ];
    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .toroidal([1.0, 1.0])
        .build::<()>()
        .expect("build should succeed");

    assert_eq!(
        dt.dim(),
        2,
        "Four 2D points should produce a 2-dimensional triangulation"
    );
}

/// Building with toroidal wrapping on already-canonical input matches Euclidean construction.
#[test]
fn test_builder_toroidal_matches_euclidean_on_canonical_input() {
    let vertices = vec![
        vertex!([0.1_f64, 0.2]),
        vertex!([0.8, 0.3]),
        vertex!([0.4, 0.9]),
    ];
    let dt_euclidean = DelaunayTriangulationBuilder::new(&vertices)
        .build::<()>()
        .expect("euclidean build should succeed");
    let dt_toroidal = DelaunayTriangulationBuilder::new(&vertices)
        .toroidal([1.0, 1.0])
        .build::<()>()
        .expect("toroidal build should succeed");

    assert_eq!(
        dt_euclidean.number_of_vertices(),
        dt_toroidal.number_of_vertices()
    );
    assert_eq!(
        dt_euclidean.number_of_simplices(),
        dt_toroidal.number_of_simplices()
    );
}

/// Larger 2D toroidal point set — full Level 1–4 validation.
///
/// Uses eight hand-picked, well-separated points to stay in general position
/// with the default `FastKernel`, exercising the full toroidal build pipeline
/// beyond the minimal 3–4 vertex cases covered by other tests.
#[test]
fn test_builder_toroidal_larger_point_set_2d() {
    let vertices = vec![
        vertex!([0.1_f64, 0.2]),
        vertex!([0.6, 0.1]),
        vertex!([0.9, 0.4]),
        vertex!([0.7, 0.8]),
        vertex!([0.3, 0.7]),
        vertex!([0.48, 0.52]),
        vertex!([0.15, 0.85]),
        vertex!([0.8, 0.65]),
    ];
    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .toroidal([1.0, 1.0])
        .build::<()>()
        .expect("larger toroidal build should succeed");

    assert_eq!(dt.number_of_vertices(), 8);
    assert!(dt.validate().is_ok(), "Full validation should pass");
}

// =============================================================================
// Custom kernel
// =============================================================================

/// `build_with_kernel` with `RobustKernel` works for a 3D toroidal case.
#[test]
fn test_builder_toroidal_robust_kernel_3d() {
    let vertices = vec![
        vertex!([0.2_f64, 0.3, 0.4]),
        vertex!([0.8, 0.1, 0.2]),
        vertex!([0.5, 0.7, 0.6]),
        vertex!([0.1, 0.9, 0.3]),
        vertex!([0.6, 0.4, 0.8]),
    ];
    let kernel = RobustKernel::new();
    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .toroidal([1.0, 1.0, 1.0])
        .build_with_kernel::<_, ()>(&kernel)
        .expect("toroidal robust kernel 3D build should succeed");

    assert_eq!(dt.number_of_vertices(), 5);
    assert!(dt.validate().is_ok());
}

// =============================================================================
// Periodic (image-point method) path
// =============================================================================

fn toroidal_periodic_vertices<const D: usize>() -> Vec<Vertex<f64, (), D>> {
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
            vertex!(coords)
        })
        .collect()
}

fn build_toroidal_periodic_triangulation<const D: usize>()
-> DelaunayTriangulation<RobustKernel<f64>, (), (), D> {
    let vertices = toroidal_periodic_vertices::<D>();
    let expected_vertices = vertices.len();
    let kernel = RobustKernel::new();
    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .toroidal_periodic([1.0_f64; D])
        .build_with_kernel::<_, ()>(&kernel)
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

/// `toroidal_periodic` builds a valid 2D periodic triangulation with χ = 0.
///
/// Verifies TDS structural validity and χ = 0 directly.
/// See the macro-generated toroidal periodic validation tests for the full
/// `PLManifold` and Level 4 `validate()` paths.
#[test]
fn test_builder_toroidal_periodic_chi_zero_2d() {
    let dt = build_toroidal_periodic_triangulation::<2>();

    assert!(
        dt.tds().is_valid().is_ok(),
        "TDS structural validity should pass for periodic triangulation"
    );
    let counts = count_simplices(dt.tds()).unwrap();
    let chi = euler_characteristic(&counts);
    assert_eq!(
        chi, 0,
        "Euler characteristic of periodic 2D triangulation must be 0 (torus)"
    );
}

macro_rules! gen_toroidal_periodic_validation_test {
    ($dim:literal, $label:ident, $run_level4:expr $(, #[$attr:meta])?) => {
        pastey::paste! {
            /// `toroidal_periodic` validation for this dimension.
            ///
            /// Periodic-aware ridge and vertex link validation correctly handles reused
            /// vertex keys via lifted vertex identity. When `$run_level4` is true, this
            /// also exercises Level 4 periodic lifted Delaunay predicates.
            #[test]
            $(#[$attr])?
            fn [<test_builder_toroidal_periodic_validate_ $label _ $dim d>]() {
                let dt = build_toroidal_periodic_triangulation::<$dim>();

                let topology_result = dt.as_triangulation().validate();
                assert!(
                    topology_result.is_ok(),
                    "PLManifold Levels 1-3 validate() should pass for toroidal_periodic: {:?}",
                    topology_result.err()
                );

                if $run_level4 {
                    let level4_result = dt.validate();
                    assert!(
                        level4_result.is_ok(),
                        "Level 1-4 validate() should pass for toroidal_periodic: {:?}",
                        level4_result.err()
                    );
                }
            }
        }
    };
}

gen_toroidal_periodic_validation_test!(2, levels_1_to_4, true);
#[test]
fn test_builder_periodic_topology_level4_smoke_3d() {
    let vertices = vec![
        vertex!([0.2_f64, 0.3, 0.4]),
        vertex!([0.8, 0.1, 0.2]),
        vertex!([0.5, 0.7, 0.6]),
        vertex!([0.1, 0.9, 0.3]),
        vertex!([0.6, 0.4, 0.8]),
        vertex!([0.3, 0.5, 0.9]),
        vertex!([0.9, 0.2, 0.6]),
    ];
    let kernel = RobustKernel::new();
    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .toroidal_periodic([1.0_f64; 3])
        .build_with_kernel::<_, ()>(&kernel)
        .expect("compact periodic 3D build should succeed");

    assert_eq!(dt.number_of_vertices(), vertices.len());
    assert!(
        dt.tds().is_valid().is_ok(),
        "TDS structural validity should pass for compact periodic 3D"
    );
    assert!(
        dt.global_topology().is_periodic(),
        "global_topology should use periodic image-point construction"
    );
    assert!(
        dt.number_of_simplices() > 0,
        "periodic image-point construction should produce at least one simplex"
    );
    assert!(
        dt.simplices().all(|(_, simplex)| {
            simplex
                .periodic_vertex_offsets()
                .is_some_and(|offsets| offsets.len() == simplex.number_of_vertices())
        }),
        "periodic image-point construction should populate per-simplex periodic offsets"
    );
    // The compact 3D quotient is a smoke fixture for lifted predicate evaluation,
    // not a full periodic-Delaunay quality fixture. A local violation is a valid
    // Level 4 result; malformed periodic-offset plumbing is not.
    match dt.is_delaunay_via_flips() {
        Ok(()) => {}
        Err(DelaunayRepairError::PostconditionFailed { message }) => {
            assert!(
                !message.contains("predicate failed in strict mode")
                    && !message.contains("periodic offset")
                    && !message.contains("cannot align periodic vertex"),
                "periodic Level 4 should evaluate lifted predicates with populated offsets: {message}"
            );
        }
        Err(err) => panic!("periodic Level 4 validation returned an unexpected error: {err:?}"),
    }
}
gen_toroidal_periodic_validation_test!(
    3,
    levels_1_to_4,
    true,
    #[ignore = "Slow (>60s): periodic 3D expands to 3^D image points; run with --ignored"]
);
gen_toroidal_periodic_validation_test!(
    4,
    levels_1_to_3,
    false,
    #[ignore = "Slow: periodic 4D expands to 3^D image points; run with --ignored"]
);
gen_toroidal_periodic_validation_test!(
    5,
    levels_1_to_3,
    false,
    #[ignore = "Slow: periodic 5D expands to 3^D image points; run with --ignored"]
);

/// Explicit 7-vertex torus (Heawood triangulation) with `GlobalTopology::Toroidal`
/// is rejected until explicit non-Euclidean construction has Level 4 validation.
///
/// The 14-triangle closed mesh has χ = 0 (torus), but explicit quotient
/// connectivity cannot yet be validated against the Level 4 Delaunay property.
#[test]
fn test_explicit_toroidal_heawood_torus_rejected() {
    // Regular heptagon: 7 well-separated points, no 3 collinear.
    let vertices: Vec<_> = (0..7)
        .map(|i| {
            let angle = TAU * f64::from(i) / 7.0;
            vertex!([angle.cos(), angle.sin()])
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

    let err = DelaunayTriangulationBuilder::from_vertices_and_simplices(&vertices, &simplices)
        .global_topology(GlobalTopology::Toroidal {
            domain: [2.0, 2.0],
            mode: ToroidalConstructionMode::Explicit,
        })
        .build::<()>()
        .expect_err("explicit toroidal connectivity requires a Level 4 quotient validator");

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
            vertex!([angle.cos(), angle.sin()])
        })
        .collect();

    let mut simplices: Vec<Vec<usize>> = Vec::with_capacity(14);
    for i in 0..7 {
        simplices.push(vec![i, (i + 1) % 7, (i + 3) % 7]);
        simplices.push(vec![i, (i + 2) % 7, (i + 3) % 7]);
    }

    // Build with default Euclidean topology — should fail at Euler validation.
    let err = DelaunayTriangulationBuilder::from_vertices_and_simplices(&vertices, &simplices)
        .build::<()>()
        .expect_err("explicit torus without Toroidal metadata should fail Euler validation");

    match err {
        DelaunayTriangulationConstructionError::ExplicitConstruction(
            ExplicitConstructionError::TopologyValidation { source },
        ) => {
            assert_eq!(source.kind, ExplicitInvariantErrorKind::Triangulation);
            assert_eq!(
                source.detail,
                InvariantErrorSummaryDetail::Triangulation(
                    TriangulationValidationErrorKind::EulerCharacteristicMismatch,
                ),
            );
        }
        DelaunayTriangulationConstructionError::ExplicitConstruction(
            ExplicitConstructionError::OrientationNormalization { source },
        ) => {
            assert_eq!(
                source.kind,
                ExplicitInsertionErrorKind::TopologyValidationFailed
            );
            assert_eq!(
                source.source_kind,
                Some(InsertionErrorSourceKind::Triangulation(
                    TriangulationValidationErrorKind::OrientationPromotionNonConvergence,
                )),
            );
        }
        other => {
            panic!("expected explicit topology or orientation-normalization failure, got {other:?}")
        }
    }
}

/// `toroidal_periodic` in 3D builds a valid periodic triangulation.
///
/// For a 3D periodic triangulation on the 3-torus the Euler characteristic is
/// also 0, so we verify TDS structural validity rather than the full `validate()`
/// check (which would expect χ = 2 for a sphere).
#[test]
#[ignore = "Slow (>60s): periodic 3D expands to 3^D image points; run with --ignored"]
fn test_builder_toroidal_periodic_3d_success() {
    // Keep this
    // The periodic 3D pipeline expands to 3^D image points internally, so runtime grows quickly
    // with input size and can become flaky under CI load. We keep a compact, well-separated set
    // above the algorithm minimum (2*D + 1 = 7 points for D=3).

    let vertices = vec![
        vertex!([0.1_f64, 0.2, 0.3]),
        vertex!([0.4, 0.7, 0.1]),
        vertex!([0.7, 0.3, 0.8]),
        vertex!([0.2, 0.9, 0.5]),
        vertex!([0.8, 0.6, 0.2]),
        vertex!([0.5, 0.1, 0.7]),
        vertex!([0.3, 0.5, 0.9]),
        vertex!([0.6, 0.8, 0.4]),
        vertex!([0.9, 0.2, 0.6]),
    ];
    let n = vertices.len();
    let kernel = RobustKernel::new();
    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .toroidal_periodic([1.0_f64, 1.0, 1.0])
        .build_with_kernel::<_, ()>(&kernel)
        .expect("periodic 3D build should succeed");
    assert_eq!(dt.number_of_vertices(), n);
    assert!(
        dt.tds().is_valid().is_ok(),
        "TDS structural validity should pass for 3D periodic triangulation"
    );
}

// =============================================================================
// Non-f64 scalar: from_vertices()
// =============================================================================

/// `from_vertices()` with f32 scalars constructs a valid triangulation.
#[test]
fn test_builder_from_vertices_f32() {
    let vertices: Vec<Vertex<f32, (), 2>> = vec![
        VertexBuilder::default()
            .point(Point::new([0.0_f32, 0.0]))
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([1.0_f32, 0.0]))
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.0_f32, 1.0]))
            .build()
            .unwrap(),
    ];

    let dt = DelaunayTriangulationBuilder::from_vertices(&vertices)
        .build::<()>()
        .expect("f32 from_vertices build should succeed");

    assert_eq!(dt.number_of_vertices(), 3);
    assert_eq!(dt.number_of_simplices(), 1);
}

// =============================================================================
// Explicit construction (from_vertices_and_simplices)
// =============================================================================

/// 2D: Build two triangles forming a quad from explicit vertices and simplices.
#[test]
fn test_explicit_2d_two_triangle_quad() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([1.0, 1.0]),
        vertex!([0.0, 1.0]),
    ];
    let simplices = vec![vec![0, 1, 2], vec![0, 2, 3]];

    let dt = DelaunayTriangulationBuilder::from_vertices_and_simplices(&vertices, &simplices)
        .build::<()>()
        .expect("explicit 2D build should succeed");

    assert_eq!(dt.number_of_vertices(), 4);
    assert_eq!(dt.number_of_simplices(), 2);
    assert!(
        dt.tds().is_valid().is_ok(),
        "TDS should be structurally valid"
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

/// Explicit construction normalizes incoherent local simplex orderings.
///
/// Swapping two vertices in one simplex flips its local orientation relative to an
/// otherwise valid two-triangle mesh. The builder should repair that internal
/// ordering detail and still produce a structurally valid TDS.
#[test]
fn test_explicit_normalizes_incoherent_simplex_order() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([1.0, 1.0]),
        vertex!([0.0, 1.0]),
    ];
    let mut simplices = vec![vec![0, 1, 2], vec![0, 2, 3]];
    simplices[1].swap(0, 1);

    let dt = DelaunayTriangulationBuilder::from_vertices_and_simplices(&vertices, &simplices)
        .build::<()>()
        .expect("explicit build should normalize incoherent simplex ordering");

    assert!(
        dt.tds().is_valid().is_ok(),
        "builder should canonicalize incoherent simplex orderings into a valid TDS"
    );
}

/// 3D: Build two tetrahedra sharing a face.
#[test]
fn test_explicit_3d_two_tetrahedra() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
        vertex!([1.0, 1.0, -1.0]),
    ];
    // Two tetrahedra sharing face (0, 1, 2)
    let simplices = vec![vec![0, 1, 2, 3], vec![0, 1, 2, 4]];

    let dt = DelaunayTriangulationBuilder::from_vertices_and_simplices(&vertices, &simplices)
        .build::<()>()
        .expect("explicit 3D build should succeed");

    assert_eq!(dt.number_of_vertices(), 5);
    assert_eq!(dt.number_of_simplices(), 2);
    assert!(
        dt.tds().is_valid().is_ok(),
        "TDS should be structurally valid"
    );
}

/// Round-trip 3D: Build via Delaunay → extract → reconstruct via explicit → same structure.
#[test]
fn test_explicit_round_trip_3d() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
        vertex!([1.0, 1.0, 1.0]),
    ];

    let dt_original = DelaunayTriangulation::new(&vertices).expect("Delaunay build should succeed");
    let original_vertex_count = dt_original.number_of_vertices();
    let original_simplex_count = dt_original.number_of_simplices();

    let tds = dt_original.tds();
    let vertex_keys: Vec<_> = tds.vertex_keys().collect();
    let key_to_index: HashMap<_, _> = vertex_keys
        .iter()
        .enumerate()
        .map(|(idx, &vk)| (vk, idx))
        .collect();

    let extracted_vertices: Vec<_> = vertex_keys
        .iter()
        .map(|&vk| *tds.vertex(vk).unwrap())
        .collect();

    let mut simplex_specs: Vec<Vec<usize>> = Vec::new();
    for (_, simplex) in tds.simplices() {
        let spec: Vec<usize> = simplex
            .vertices()
            .iter()
            .map(|vk| key_to_index[vk])
            .collect();
        simplex_specs.push(spec);
    }

    let dt_reconstructed = DelaunayTriangulationBuilder::from_vertices_and_simplices(
        &extracted_vertices,
        &simplex_specs,
    )
    .build::<()>()
    .expect("explicit 3D reconstruction should succeed");

    assert_eq!(dt_reconstructed.number_of_vertices(), original_vertex_count);
    assert_eq!(
        dt_reconstructed.number_of_simplices(),
        original_simplex_count
    );
    assert!(
        dt_reconstructed.tds().is_valid().is_ok(),
        "Reconstructed 3D TDS should be structurally valid"
    );
}

/// Round-trip: Build via Delaunay → extract → reconstruct via explicit → same structure.
#[test]
fn test_explicit_round_trip_2d() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
        vertex!([1.0, 1.0]),
    ];

    // Build via standard Delaunay.
    let dt_original = DelaunayTriangulation::new(&vertices).expect("Delaunay build should succeed");
    let original_vertex_count = dt_original.number_of_vertices();
    let original_simplex_count = dt_original.number_of_simplices();

    // Extract vertex keys → index mapping and simplex specifications.
    let tds = dt_original.tds();
    let vertex_keys: Vec<_> = tds.vertex_keys().collect();
    let key_to_index: HashMap<_, _> = vertex_keys
        .iter()
        .enumerate()
        .map(|(idx, &vk)| (vk, idx))
        .collect();

    let extracted_vertices: Vec<_> = vertex_keys
        .iter()
        .map(|&vk| *tds.vertex(vk).unwrap())
        .collect();

    let mut simplex_specs: Vec<Vec<usize>> = Vec::new();
    for (_, simplex) in tds.simplices() {
        let spec: Vec<usize> = simplex
            .vertices()
            .iter()
            .map(|vk| key_to_index[vk])
            .collect();
        simplex_specs.push(spec);
    }

    // Reconstruct via explicit.
    let dt_reconstructed = DelaunayTriangulationBuilder::from_vertices_and_simplices(
        &extracted_vertices,
        &simplex_specs,
    )
    .build::<()>()
    .expect("explicit reconstruction should succeed");

    assert_eq!(dt_reconstructed.number_of_vertices(), original_vertex_count);
    assert_eq!(
        dt_reconstructed.number_of_simplices(),
        original_simplex_count
    );
    assert!(
        dt_reconstructed.tds().is_valid().is_ok(),
        "Reconstructed TDS should be structurally valid"
    );
}

/// Error: empty simplices should fail.
#[test]
fn test_explicit_error_empty_simplices() {
    let vertices = vec![vertex!([0.0_f64, 0.0]), vertex!([1.0, 0.0])];
    let simplices: Vec<Vec<usize>> = vec![];

    let result = DelaunayTriangulationBuilder::from_vertices_and_simplices(&vertices, &simplices)
        .build::<()>();

    assert!(result.is_err(), "Empty simplices should produce an error");
}

/// Error: wrong simplex arity.
#[test]
fn test_explicit_error_wrong_arity() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
    ];
    // 2D expects 3 vertices per simplex, but we provide 2.
    let simplices = vec![vec![0, 1]];

    let result = DelaunayTriangulationBuilder::from_vertices_and_simplices(&vertices, &simplices)
        .build::<()>();

    assert!(result.is_err(), "Wrong arity should produce an error");
}

/// Error: out-of-bounds vertex index.
#[test]
fn test_explicit_error_index_out_of_bounds() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
    ];
    let simplices = vec![vec![0, 1, 99]]; // 99 is out of bounds

    let result = DelaunayTriangulationBuilder::from_vertices_and_simplices(&vertices, &simplices)
        .build::<()>();

    assert!(
        result.is_err(),
        "Out-of-bounds index should produce an error"
    );
}

/// Error: duplicate vertex in simplex.
#[test]
fn test_explicit_error_duplicate_vertex_in_simplex() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
    ];
    let simplices = vec![vec![0, 1, 1]]; // Duplicate vertex 1

    let result = DelaunayTriangulationBuilder::from_vertices_and_simplices(&vertices, &simplices)
        .build::<()>();

    assert!(result.is_err(), "Duplicate vertex should produce an error");
}

/// Error: toroidal + explicit simplices is incompatible.
#[test]
fn test_explicit_error_toroidal_incompatible() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
    ];
    let simplices = vec![vec![0, 1, 2]];

    // Construct with explicit simplices then add toroidal.
    let result = DelaunayTriangulationBuilder::from_vertices_and_simplices(&vertices, &simplices)
        .toroidal([1.0, 1.0])
        .build::<()>();

    assert!(
        result.is_err(),
        "Toroidal + explicit simplices should produce an error"
    );
}

/// Minimal case: a single triangle in 2D.
#[test]
fn test_explicit_2d_single_triangle() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
    ];
    let simplices = vec![vec![0, 1, 2]];

    let dt = DelaunayTriangulationBuilder::from_vertices_and_simplices(&vertices, &simplices)
        .build::<()>()
        .expect("single triangle should succeed");

    assert_eq!(dt.number_of_vertices(), 3);
    assert_eq!(dt.number_of_simplices(), 1);
    assert!(dt.tds().is_valid().is_ok());
}

/// Minimal case: a single tetrahedron in 3D.
#[test]
fn test_explicit_3d_single_tetrahedron() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];
    let simplices = vec![vec![0, 1, 2, 3]];

    let dt = DelaunayTriangulationBuilder::from_vertices_and_simplices(&vertices, &simplices)
        .build::<()>()
        .expect("single tetrahedron should succeed");

    assert_eq!(dt.number_of_vertices(), 4);
    assert_eq!(dt.number_of_simplices(), 1);
    assert!(dt.tds().is_valid().is_ok());
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
        vertex!([0.0_f64, 0.0]),
        vertex!([4.0, 0.0]),
        vertex!([4.0, 2.0]),
        vertex!([1.0, 2.0]),
    ];
    // Diagonal AC = (0,0)-(4,2): non-Delaunay because D=(1,2) is inside
    // the circumcircle of triangle ABC.
    let simplices = vec![vec![0, 1, 2], vec![0, 2, 3]];

    let err = DelaunayTriangulationBuilder::from_vertices_and_simplices(&vertices, &simplices)
        .build::<()>()
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
        "error should identify the Level 4 validation failure: {err}"
    );
}

/// Topology guarantee is propagated through explicit construction.
#[test]
fn test_explicit_topology_guarantee_propagated() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
    ];
    let simplices = vec![vec![0, 1, 2]];

    let dt = DelaunayTriangulationBuilder::from_vertices_and_simplices(&vertices, &simplices)
        .topology_guarantee(TopologyGuarantee::Pseudomanifold)
        .build::<()>()
        .expect("build should succeed");

    assert_eq!(dt.topology_guarantee(), TopologyGuarantee::Pseudomanifold);
}

/// Explicit construction preserves vertex data (U ≠ ()).
#[test]
fn test_explicit_preserves_vertex_data() {
    let vertices: Vec<Vertex<f64, i32, 2>> = vec![
        VertexBuilder::default()
            .point(Point::new([0.0, 0.0]))
            .data(10_i32)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([1.0, 0.0]))
            .data(20_i32)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.0, 1.0]))
            .data(30_i32)
            .build()
            .unwrap(),
    ];
    let simplices = vec![vec![0, 1, 2]];

    let dt = DelaunayTriangulationBuilder::from_vertices_and_simplices(&vertices, &simplices)
        .build::<()>()
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

/// Full `validate()` (Levels 1–4) on a Delaunay-compatible explicit mesh.
#[test]
fn test_explicit_validate_delaunay_mesh() {
    // Use a known Delaunay configuration: the standard simplex.
    let vertices = vec![
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.5, 0.866_025_403_784_438_6]),
    ];
    let simplices = vec![vec![0, 1, 2]];

    let dt = DelaunayTriangulationBuilder::from_vertices_and_simplices(&vertices, &simplices)
        .build::<()>()
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
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
        vertex!([5.0, 5.0]), // Not referenced by any simplex
    ];
    let simplices = vec![vec![0, 1, 2]];

    let err = DelaunayTriangulationBuilder::from_vertices_and_simplices(&vertices, &simplices)
        .build::<()>()
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
    let vertices = vec![vertex!([0.0_f64, 0.0])];
    let simplices: Vec<Vec<usize>> = vec![];

    let err = DelaunayTriangulationBuilder::from_vertices_and_simplices(&vertices, &simplices)
        .build::<()>()
        .unwrap_err();

    assert!(
        matches!(
            err,
            DelaunayTriangulationConstructionError::ExplicitConstruction(
                ExplicitConstructionError::EmptySimplices
            )
        ),
        "Expected ExplicitConstruction(EmptySimplices), got: {err}"
    );
}

/// Error variant: wrong arity returns ExplicitConstruction(InvalidSimplexArity { .. }).
#[test]
fn test_explicit_error_variant_wrong_arity() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
    ];
    let simplices = vec![vec![0, 1]]; // 2D expects 3 vertices

    let err = DelaunayTriangulationBuilder::from_vertices_and_simplices(&vertices, &simplices)
        .build::<()>()
        .unwrap_err();

    assert!(
        matches!(
            err,
            DelaunayTriangulationConstructionError::ExplicitConstruction(
                ExplicitConstructionError::InvalidSimplexArity {
                    simplex_index: 0,
                    actual: 2,
                    expected: 3
                }
            )
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
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
        vertex!([1.0, 1.0]),
        vertex!([0.5, -1.0]),
    ];
    let simplices = vec![vec![0, 1, 2], vec![0, 1, 3], vec![0, 1, 4]];

    let err = DelaunayTriangulationBuilder::from_vertices_and_simplices(&vertices, &simplices)
        .build::<()>()
        .unwrap_err();

    let DelaunayTriangulationConstructionError::ExplicitConstruction(
        ExplicitConstructionError::TdsAssembly { source },
    ) = &err
    else {
        panic!("Expected explicit TDS assembly failure, got: {err}");
    };

    assert_eq!(source.kind, ExplicitTdsErrorKind::FacetSharingViolation);
    assert!(source.message.contains("observed 3 incident simplices"));
    assert!(source.message.contains("max 2"));
}

/// Error variant: duplicate vertex returns ExplicitConstruction(DuplicateVertexInSimplex { .. }).
#[test]
fn test_explicit_error_variant_duplicate_vertex_in_simplex() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
    ];
    let simplices = vec![vec![0, 1, 1]]; // Duplicate vertex 1

    let err = DelaunayTriangulationBuilder::from_vertices_and_simplices(&vertices, &simplices)
        .build::<()>()
        .unwrap_err();

    assert!(
        matches!(
            err,
            DelaunayTriangulationConstructionError::ExplicitConstruction(
                ExplicitConstructionError::DuplicateVertexInSimplex { simplex_index: 0 }
            )
        ),
        "Expected DuplicateVertexInSimplex, got: {err}"
    );
}

/// Error variant: toroidal + explicit returns ExplicitConstruction(IncompatibleTopology).
#[test]
fn test_explicit_error_variant_incompatible_topology() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
    ];
    let simplices = vec![vec![0, 1, 2]];

    let err = DelaunayTriangulationBuilder::from_vertices_and_simplices(&vertices, &simplices)
        .toroidal([1.0, 1.0])
        .build::<()>()
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
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
    ];
    let simplices = vec![vec![0, 1, 2]];

    let err = DelaunayTriangulationBuilder::from_vertices_and_simplices(&vertices, &simplices)
        .construction_options(
            ConstructionOptions::default().with_insertion_order(InsertionOrderStrategy::Input),
        )
        .build::<()>()
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
}

/// Error variant: duplicate maximal simplices are rejected during TDS insertion.
#[test]
fn test_explicit_error_variant_duplicate_simplices_structural_validation() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
    ];
    let simplices = vec![vec![0, 1, 2], vec![0, 1, 2]];

    let err = DelaunayTriangulationBuilder::from_vertices_and_simplices(&vertices, &simplices)
        .build::<()>()
        .unwrap_err();

    let DelaunayTriangulationConstructionError::ExplicitConstruction(
        ExplicitConstructionError::TdsAssembly { source },
    ) = &err
    else {
        panic!("expected explicit TDS assembly failure, got {err:?}");
    };

    assert_eq!(source.kind, ExplicitTdsErrorKind::DuplicateSimplices);
}

/// Error variant: degenerate explicit simplices fail geometric nondegeneracy validation.
#[test]
fn test_explicit_error_variant_geometric_nondegeneracy() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([2.0, 0.0]),
    ];
    let simplices = vec![vec![0, 1, 2]];

    let err = DelaunayTriangulationBuilder::from_vertices_and_simplices(&vertices, &simplices)
        .build::<()>()
        .unwrap_err();

    match err {
        DelaunayTriangulationConstructionError::ExplicitConstruction(
            ExplicitConstructionError::GeometricNondegeneracy { source },
        ) => assert_eq!(source.kind, ExplicitTdsErrorKind::Geometric),
        other => panic!("expected explicit geometric nondegeneracy failure, got {other:?}"),
    }
}

/// Error variant: completion-time PL validation preserves vertex-link summaries.
#[test]
fn test_explicit_error_variant_completion_validation_summary() {
    let err = ExplicitConstructionError::CompletionValidation {
        source: ExplicitInvariantError {
            kind: ExplicitInvariantErrorKind::Triangulation,
            detail: InvariantErrorSummaryDetail::Triangulation(
                TriangulationValidationErrorKind::VertexLinkNotManifold,
            ),
            message: "vertex link is disconnected".to_string(),
        },
    };

    match err {
        ExplicitConstructionError::CompletionValidation { source } => {
            assert_eq!(source.kind, ExplicitInvariantErrorKind::Triangulation);
            assert_eq!(
                source.detail,
                InvariantErrorSummaryDetail::Triangulation(
                    TriangulationValidationErrorKind::VertexLinkNotManifold,
                ),
            );
        }
        other => panic!("expected explicit completion validation failure, got {other:?}"),
    }
}

/// Error variant: orientation-normalization failures preserve typed insertion summaries.
#[test]
fn test_explicit_error_variant_orientation_normalization_summary() {
    let source = ExplicitConstructionError::OrientationNormalization {
        source: ExplicitInsertionError {
            kind: ExplicitInsertionErrorKind::TopologyValidation,
            source_kind: None,
            message: "orientation normalization could not establish coherent simplices".to_string(),
        },
    };

    match source {
        ExplicitConstructionError::OrientationNormalization { source } => {
            assert_eq!(source.kind, ExplicitInsertionErrorKind::TopologyValidation);
            assert!(source.to_string().contains("coherent simplices"));
        }
        other => panic!("expected orientation-normalization variant, got {other:?}"),
    }
}

/// Error variant: out-of-bounds returns ExplicitConstruction(IndexOutOfBounds { .. }).
#[test]
fn test_explicit_error_variant_index_out_of_bounds() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
    ];
    let simplices = vec![vec![0, 1, 99]];

    let err = DelaunayTriangulationBuilder::from_vertices_and_simplices(&vertices, &simplices)
        .build::<()>()
        .unwrap_err();

    assert!(
        matches!(
            err,
            DelaunayTriangulationConstructionError::ExplicitConstruction(
                ExplicitConstructionError::IndexOutOfBounds {
                    simplex_index: 0,
                    vertex_index: 99,
                    bound: 3,
                }
            )
        ),
        "Expected IndexOutOfBounds, got: {err}"
    );
}
