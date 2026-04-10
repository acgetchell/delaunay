//! Integration tests for [`DelaunayTriangulationBuilder`].
//!
//! These tests exercise the public API from the outside, using only items exposed
//! through `delaunay::prelude::triangulation` and `delaunay::core::builder`.

use std::collections::HashMap;
use std::f64::consts::TAU;

use delaunay::core::builder::{DelaunayTriangulationBuilder, ExplicitConstructionError};
use delaunay::core::delaunay_triangulation::{
    ConstructionOptions, DelaunayTriangulation, DelaunayTriangulationConstructionError,
    InsertionOrderStrategy,
};
use delaunay::core::triangulation::TopologyGuarantee;
use delaunay::core::vertex::{Vertex, VertexBuilder};
use delaunay::geometry::kernel::RobustKernel;
use delaunay::geometry::point::Point;
use delaunay::geometry::traits::coordinate::Coordinate;
use delaunay::topology::characteristics::euler::{count_simplices, euler_characteristic};
use delaunay::topology::traits::topological_space::{GlobalTopology, ToroidalConstructionMode};
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

/// `toroidal_periodic` builds a valid 2D periodic triangulation with χ = 0.
///
/// Verifies TDS structural validity and χ = 0 directly.
/// See `test_builder_toroidal_periodic_full_validate_2d` for the full
/// `PLManifold` `validate()` path.
#[test]
fn test_builder_toroidal_periodic_chi_zero_2d() {
    let vertices = vec![
        vertex!([0.1_f64, 0.2]),
        vertex!([0.4, 0.7]),
        vertex!([0.7, 0.3]),
        vertex!([0.2, 0.9]),
        vertex!([0.8, 0.6]),
        vertex!([0.5, 0.1]),
        vertex!([0.3, 0.5]),
    ];
    let kernel = RobustKernel::new();
    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .toroidal_periodic([1.0_f64, 1.0])
        .build_with_kernel::<_, ()>(&kernel)
        .expect("periodic 2D build should succeed");

    assert_eq!(dt.number_of_vertices(), 7);
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

/// `toroidal_periodic` 2D with full `PLManifold` Levels 1–3 validation.
///
/// Periodic-aware ridge and vertex link validation correctly handles reused
/// vertex keys via lifted vertex identity.  This exercises the
/// `GlobalTopology::Toroidal` Euler override AND the lifted-vertex-identity
/// logic in `validate_ridge_links` / `validate_vertex_links`.
///
/// Level 4 (Delaunay property) is not checked because the quotient mesh
/// may contain local violations that are valid under periodic identification.
#[test]
fn test_builder_toroidal_periodic_full_validate_2d() {
    let vertices = vec![
        vertex!([0.1_f64, 0.2]),
        vertex!([0.4, 0.7]),
        vertex!([0.7, 0.3]),
        vertex!([0.2, 0.9]),
        vertex!([0.8, 0.6]),
        vertex!([0.5, 0.1]),
        vertex!([0.3, 0.5]),
    ];
    let kernel = RobustKernel::new();
    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .toroidal_periodic([1.0_f64, 1.0])
        .build_with_kernel::<_, ()>(&kernel)
        .expect("periodic 2D build should succeed");

    assert_eq!(dt.number_of_vertices(), 7);
    assert!(
        dt.global_topology().is_toroidal(),
        "global_topology should be Toroidal"
    );
    // Levels 1–3 (TDS structure + topology including PLManifold ridge/vertex links).
    let result = dt.as_triangulation().validate();
    assert!(
        result.is_ok(),
        "PLManifold Levels 1-3 validate() should pass for toroidal_periodic: {:?}",
        result.err()
    );
}

/// Explicit 7-vertex torus (Heawood triangulation) with `GlobalTopology::Toroidal`
/// builds successfully under `PLManifold` guarantee.
///
/// The 14-triangle closed mesh has χ = 0 (torus). Without the Toroidal override
/// in `validate_topology_core()`, the heuristic would classify it as
/// `ClosedSphere(2)` and expect χ = 2, failing at build time. Setting
/// `global_topology = Toroidal` overrides to `ClosedToroid(2)` (χ = 0).
///
/// Uses explicit construction with unique vertex keys, so ridge and vertex link
/// checks are correctly evaluated under `PLManifold` guarantee.
///
/// Note: `validate_geometric_cell_orientation()` fails for this planar embedding
/// of a torus (self-intersection makes some cells negative-oriented), so we
/// verify TDS structural validity (Level 1–2) rather than full `validate()`.
#[test]
fn test_explicit_toroidal_heawood_torus_validates() {
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
    let mut cells: Vec<Vec<usize>> = Vec::with_capacity(14);
    for i in 0..7 {
        cells.push(vec![i, (i + 1) % 7, (i + 3) % 7]);
        cells.push(vec![i, (i + 2) % 7, (i + 3) % 7]);
    }

    // Build succeeds: the Toroidal override in validate_topology_core() accepts
    // χ = 0 for the ClosedToroid classification. Without global_topology = Toroidal,
    // this build would fail (see test_explicit_toroidal_torus_euler_mismatch_without_override).
    let dt = DelaunayTriangulationBuilder::from_vertices_and_cells(&vertices, &cells)
        .global_topology(GlobalTopology::Toroidal {
            domain: [2.0, 2.0],
            mode: ToroidalConstructionMode::Explicit,
        })
        .build::<()>()
        .expect("explicit toroidal torus build should succeed");

    assert_eq!(dt.number_of_vertices(), 7);
    assert_eq!(dt.number_of_cells(), 14);
    assert!(
        dt.global_topology().is_toroidal(),
        "global_topology should be Toroidal"
    );
    assert!(
        dt.tds().is_valid().is_ok(),
        "TDS structural validity (Level 1-2) should pass"
    );
    let counts = count_simplices(dt.tds()).unwrap();
    let chi = euler_characteristic(&counts);
    assert_eq!(chi, 0, "Euler characteristic of explicit torus must be 0");
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

    let mut cells: Vec<Vec<usize>> = Vec::with_capacity(14);
    for i in 0..7 {
        cells.push(vec![i, (i + 1) % 7, (i + 3) % 7]);
        cells.push(vec![i, (i + 2) % 7, (i + 3) % 7]);
    }

    // Build with default Euclidean topology — should fail at Euler validation.
    let err = DelaunayTriangulationBuilder::from_vertices_and_cells(&vertices, &cells)
        .build::<()>()
        .expect_err("explicit torus without Toroidal metadata should fail Euler validation");
    let msg = err.to_string();
    assert!(
        msg.contains("Euler characteristic") || msg.contains("topology validation failed"),
        "Error should mention topology/Euler failure: {msg}"
    );
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
    assert_eq!(dt.number_of_cells(), 1);
}

// =============================================================================
// Explicit construction (from_vertices_and_cells)
// =============================================================================

/// 2D: Build two triangles forming a quad from explicit vertices and cells.
#[test]
fn test_explicit_2d_two_triangle_quad() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([1.0, 1.0]),
        vertex!([0.0, 1.0]),
    ];
    let cells = vec![vec![0, 1, 2], vec![0, 2, 3]];

    let dt = DelaunayTriangulationBuilder::from_vertices_and_cells(&vertices, &cells)
        .build::<()>()
        .expect("explicit 2D build should succeed");

    assert_eq!(dt.number_of_vertices(), 4);
    assert_eq!(dt.number_of_cells(), 2);
    assert!(
        dt.tds().is_valid().is_ok(),
        "TDS should be structurally valid"
    );

    // Verify neighbor pointers: the two triangles share the edge (0,2) so
    // each should have the other as a neighbor.
    let mut neighbor_count = 0;
    for (_, cell) in dt.cells() {
        if let Some(neighbors) = cell.neighbors() {
            for n in neighbors {
                if n.is_some() {
                    neighbor_count += 1;
                }
            }
        }
    }
    // Two cells sharing one facet → 2 neighbor slots filled (one in each cell).
    assert!(
        neighbor_count >= 2,
        "Shared facet should produce neighbor pointers"
    );
}

/// Explicit construction normalizes incoherent local cell orderings.
///
/// Swapping two vertices in one cell flips its local orientation relative to an
/// otherwise valid two-triangle mesh. The builder should repair that internal
/// ordering detail and still produce a structurally valid TDS.
#[test]
fn test_explicit_normalizes_incoherent_cell_order() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([1.0, 1.0]),
        vertex!([0.0, 1.0]),
    ];
    let mut cells = vec![vec![0, 1, 2], vec![0, 2, 3]];
    cells[1].swap(0, 1);

    let dt = DelaunayTriangulationBuilder::from_vertices_and_cells(&vertices, &cells)
        .build::<()>()
        .expect("explicit build should normalize incoherent cell ordering");

    assert!(
        dt.tds().is_valid().is_ok(),
        "builder should canonicalize incoherent cell orderings into a valid TDS"
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
        vertex!([1.0, 1.0, 1.0]),
    ];
    // Two tetrahedra sharing face (0, 1, 2)
    let cells = vec![vec![0, 1, 2, 3], vec![0, 1, 2, 4]];

    let dt = DelaunayTriangulationBuilder::from_vertices_and_cells(&vertices, &cells)
        .build::<()>()
        .expect("explicit 3D build should succeed");

    assert_eq!(dt.number_of_vertices(), 5);
    assert_eq!(dt.number_of_cells(), 2);
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
    let original_cell_count = dt_original.number_of_cells();

    let tds = dt_original.tds();
    let vertex_keys: Vec<_> = tds.vertex_keys().collect();
    let key_to_index: HashMap<_, _> = vertex_keys
        .iter()
        .enumerate()
        .map(|(idx, &vk)| (vk, idx))
        .collect();

    let extracted_vertices: Vec<_> = vertex_keys
        .iter()
        .map(|&vk| *tds.get_vertex_by_key(vk).unwrap())
        .collect();

    let mut cell_specs: Vec<Vec<usize>> = Vec::new();
    for (_, cell) in tds.cells() {
        let spec: Vec<usize> = cell.vertices().iter().map(|vk| key_to_index[vk]).collect();
        cell_specs.push(spec);
    }

    let dt_reconstructed =
        DelaunayTriangulationBuilder::from_vertices_and_cells(&extracted_vertices, &cell_specs)
            .build::<()>()
            .expect("explicit 3D reconstruction should succeed");

    assert_eq!(dt_reconstructed.number_of_vertices(), original_vertex_count);
    assert_eq!(dt_reconstructed.number_of_cells(), original_cell_count);
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
    let original_cell_count = dt_original.number_of_cells();

    // Extract vertex keys → index mapping and cell specifications.
    let tds = dt_original.tds();
    let vertex_keys: Vec<_> = tds.vertex_keys().collect();
    let key_to_index: HashMap<_, _> = vertex_keys
        .iter()
        .enumerate()
        .map(|(idx, &vk)| (vk, idx))
        .collect();

    let extracted_vertices: Vec<_> = vertex_keys
        .iter()
        .map(|&vk| *tds.get_vertex_by_key(vk).unwrap())
        .collect();

    let mut cell_specs: Vec<Vec<usize>> = Vec::new();
    for (_, cell) in tds.cells() {
        let spec: Vec<usize> = cell.vertices().iter().map(|vk| key_to_index[vk]).collect();
        cell_specs.push(spec);
    }

    // Reconstruct via explicit.
    let dt_reconstructed =
        DelaunayTriangulationBuilder::from_vertices_and_cells(&extracted_vertices, &cell_specs)
            .build::<()>()
            .expect("explicit reconstruction should succeed");

    assert_eq!(dt_reconstructed.number_of_vertices(), original_vertex_count);
    assert_eq!(dt_reconstructed.number_of_cells(), original_cell_count);
    assert!(
        dt_reconstructed.tds().is_valid().is_ok(),
        "Reconstructed TDS should be structurally valid"
    );
}

/// Error: empty cells should fail.
#[test]
fn test_explicit_error_empty_cells() {
    let vertices = vec![vertex!([0.0_f64, 0.0]), vertex!([1.0, 0.0])];
    let cells: Vec<Vec<usize>> = vec![];

    let result =
        DelaunayTriangulationBuilder::from_vertices_and_cells(&vertices, &cells).build::<()>();

    assert!(result.is_err(), "Empty cells should produce an error");
}

/// Error: wrong cell arity.
#[test]
fn test_explicit_error_wrong_arity() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
    ];
    // 2D expects 3 vertices per cell, but we provide 2.
    let cells = vec![vec![0, 1]];

    let result =
        DelaunayTriangulationBuilder::from_vertices_and_cells(&vertices, &cells).build::<()>();

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
    let cells = vec![vec![0, 1, 99]]; // 99 is out of bounds

    let result =
        DelaunayTriangulationBuilder::from_vertices_and_cells(&vertices, &cells).build::<()>();

    assert!(
        result.is_err(),
        "Out-of-bounds index should produce an error"
    );
}

/// Error: duplicate vertex in cell.
#[test]
fn test_explicit_error_duplicate_vertex_in_cell() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
    ];
    let cells = vec![vec![0, 1, 1]]; // Duplicate vertex 1

    let result =
        DelaunayTriangulationBuilder::from_vertices_and_cells(&vertices, &cells).build::<()>();

    assert!(result.is_err(), "Duplicate vertex should produce an error");
}

/// Error: toroidal + explicit cells is incompatible.
#[test]
fn test_explicit_error_toroidal_incompatible() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
    ];
    let cells = vec![vec![0, 1, 2]];

    // Construct with explicit cells then add toroidal.
    let result = DelaunayTriangulationBuilder::from_vertices_and_cells(&vertices, &cells)
        .toroidal([1.0, 1.0])
        .build::<()>();

    assert!(
        result.is_err(),
        "Toroidal + explicit cells should produce an error"
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
    let cells = vec![vec![0, 1, 2]];

    let dt = DelaunayTriangulationBuilder::from_vertices_and_cells(&vertices, &cells)
        .build::<()>()
        .expect("single triangle should succeed");

    assert_eq!(dt.number_of_vertices(), 3);
    assert_eq!(dt.number_of_cells(), 1);
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
    let cells = vec![vec![0, 1, 2, 3]];

    let dt = DelaunayTriangulationBuilder::from_vertices_and_cells(&vertices, &cells)
        .build::<()>()
        .expect("single tetrahedron should succeed");

    assert_eq!(dt.number_of_vertices(), 4);
    assert_eq!(dt.number_of_cells(), 1);
    assert!(dt.tds().is_valid().is_ok());
}

/// Non-Delaunay mesh: prescribed connectivity that violates the empty-circumsphere
/// property. Levels 1–3 should pass; Level 4 (Delaunay) should fail.
///
/// Geometry: A=(0,0), B=(4,0), C=(4,2), D=(1,2). The circumcircle of ABC has
/// center (2,1) and radius √5. Point D=(1,2) is at distance √2 < √5 from the
/// center, so D lies strictly inside the circumcircle of ABC. Using diagonal AC
/// (cells [0,1,2] and [0,2,3]) is therefore non-Delaunay. The Delaunay
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
    let cells = vec![vec![0, 1, 2], vec![0, 2, 3]];

    let dt = DelaunayTriangulationBuilder::from_vertices_and_cells(&vertices, &cells)
        .build::<()>()
        .expect("non-Delaunay mesh should build successfully (Levels 1-3)");

    assert_eq!(dt.number_of_vertices(), 4);
    assert_eq!(dt.number_of_cells(), 2);
    assert!(
        dt.tds().is_valid().is_ok(),
        "TDS structural validity (Levels 1-3) should pass"
    );
    assert!(
        dt.is_valid().is_err(),
        "Delaunay property (Level 4) should fail for non-Delaunay connectivity"
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
    let cells = vec![vec![0, 1, 2]];

    let dt = DelaunayTriangulationBuilder::from_vertices_and_cells(&vertices, &cells)
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
    let cells = vec![vec![0, 1, 2]];

    let dt = DelaunayTriangulationBuilder::from_vertices_and_cells(&vertices, &cells)
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
    let cells = vec![vec![0, 1, 2]];

    let dt = DelaunayTriangulationBuilder::from_vertices_and_cells(&vertices, &cells)
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
        vertex!([5.0, 5.0]), // Not referenced by any cell
    ];
    let cells = vec![vec![0, 1, 2]];

    let err = DelaunayTriangulationBuilder::from_vertices_and_cells(&vertices, &cells)
        .build::<()>()
        .unwrap_err();

    assert!(
        matches!(
            err,
            DelaunayTriangulationConstructionError::ExplicitConstruction(
                ExplicitConstructionError::ValidationFailed { .. }
            )
        ),
        "Unreferenced vertices should produce ValidationFailed, got: {err}"
    );
}

/// Error variant: empty cells returns ExplicitConstruction(EmptyCells).
#[test]
fn test_explicit_error_variant_empty_cells() {
    let vertices = vec![vertex!([0.0_f64, 0.0])];
    let cells: Vec<Vec<usize>> = vec![];

    let err = DelaunayTriangulationBuilder::from_vertices_and_cells(&vertices, &cells)
        .build::<()>()
        .unwrap_err();

    assert!(
        matches!(
            err,
            DelaunayTriangulationConstructionError::ExplicitConstruction(
                ExplicitConstructionError::EmptyCells
            )
        ),
        "Expected ExplicitConstruction(EmptyCells), got: {err}"
    );
}

/// Error variant: wrong arity returns ExplicitConstruction(InvalidCellArity { .. }).
#[test]
fn test_explicit_error_variant_wrong_arity() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
    ];
    let cells = vec![vec![0, 1]]; // 2D expects 3 vertices

    let err = DelaunayTriangulationBuilder::from_vertices_and_cells(&vertices, &cells)
        .build::<()>()
        .unwrap_err();

    assert!(
        matches!(
            err,
            DelaunayTriangulationConstructionError::ExplicitConstruction(
                ExplicitConstructionError::InvalidCellArity {
                    cell_index: 0,
                    actual: 2,
                    expected: 3
                }
            )
        ),
        "Expected InvalidCellArity, got: {err}"
    );
}

/// Error variant: non-manifold facet sharing returns ExplicitConstruction(ValidationFailed { .. }).
#[test]
fn test_explicit_error_variant_non_manifold_facet() {
    // Three triangles sharing the same edge (0,1) — facet shared by 3 cells
    // violates the 2-manifold property.
    let vertices = vec![
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
        vertex!([1.0, 1.0]),
        vertex!([0.5, -1.0]),
    ];
    let cells = vec![vec![0, 1, 2], vec![0, 1, 3], vec![0, 1, 4]];

    let err = DelaunayTriangulationBuilder::from_vertices_and_cells(&vertices, &cells)
        .build::<()>()
        .unwrap_err();

    assert!(
        matches!(
            err,
            DelaunayTriangulationConstructionError::ExplicitConstruction(
                ExplicitConstructionError::ValidationFailed { .. }
            )
        ),
        "Expected ExplicitConstruction(ValidationFailed), got: {err}"
    );
}

/// Error variant: duplicate vertex returns ExplicitConstruction(DuplicateVertexInCell { .. }).
#[test]
fn test_explicit_error_variant_duplicate_vertex_in_cell() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
    ];
    let cells = vec![vec![0, 1, 1]]; // Duplicate vertex 1

    let err = DelaunayTriangulationBuilder::from_vertices_and_cells(&vertices, &cells)
        .build::<()>()
        .unwrap_err();

    assert!(
        matches!(
            err,
            DelaunayTriangulationConstructionError::ExplicitConstruction(
                ExplicitConstructionError::DuplicateVertexInCell { cell_index: 0 }
            )
        ),
        "Expected DuplicateVertexInCell, got: {err}"
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
    let cells = vec![vec![0, 1, 2]];

    let err = DelaunayTriangulationBuilder::from_vertices_and_cells(&vertices, &cells)
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

/// Error variant: out-of-bounds returns ExplicitConstruction(IndexOutOfBounds { .. }).
#[test]
fn test_explicit_error_variant_index_out_of_bounds() {
    let vertices = vec![
        vertex!([0.0_f64, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
    ];
    let cells = vec![vec![0, 1, 99]];

    let err = DelaunayTriangulationBuilder::from_vertices_and_cells(&vertices, &cells)
        .build::<()>()
        .unwrap_err();

    assert!(
        matches!(
            err,
            DelaunayTriangulationConstructionError::ExplicitConstruction(
                ExplicitConstructionError::IndexOutOfBounds {
                    cell_index: 0,
                    vertex_index: 99,
                    bound: 3,
                }
            )
        ),
        "Expected IndexOutOfBounds, got: {err}"
    );
}
