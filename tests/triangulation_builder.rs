//! Integration tests for [`DelaunayTriangulationBuilder`].
//!
//! These tests exercise the public API from the outside, using only items exposed
//! through `delaunay::prelude::triangulation` and `delaunay::core::builder`.

use delaunay::core::builder::DelaunayTriangulationBuilder;
use delaunay::core::delaunay_triangulation::{
    ConstructionOptions, DelaunayTriangulation, InsertionOrderStrategy,
};
use delaunay::core::triangulation::TopologyGuarantee;
use delaunay::vertex;

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
    assert_eq!(dt_new.number_of_cells(), dt_builder.number_of_cells());
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
fn test_delaunay_triangulation_builder_convenience_method() {
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
fn test_delaunay_triangulation_builder_toroidal_convenience() {
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
        dt_canonical.number_of_cells(),
        dt_shifted.number_of_cells(),
        "Both inputs should produce the same number of cells after canonicalization"
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

/// Building with toroidal wrapping on already-canonical input is idempotent.
#[test]
fn test_builder_toroidal_idempotent_on_canonical_input() {
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
        dt_euclidean.number_of_cells(),
        dt_toroidal.number_of_cells()
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
    use delaunay::geometry::kernel::RobustKernel;

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
