#![forbid(unsafe_code)]

//! Deterministic integration tests for Euler characteristic computation.
//!
//! This module tests the topology module's Euler characteristic calculation
//! using known geometric configurations across multiple dimensions.
//!
//! ## Test Coverage
//!
//! - Empty triangulations (χ = 0)
//! - Single simplices (χ = 1)
//! - Multiple simplices with interior vertices
//! - Known 2D, 3D, 4D, and 5D configurations
//!
//! For property-based tests with random triangulations, see `proptest_euler_characteristic.rs`.

use delaunay::vertex;
use std::assert_matches;

use delaunay::builder::DelaunayTriangulationBuilder;
use delaunay::prelude::construction::{
    DelaunayTriangulation, DelaunayTriangulationConstructionError, ExplicitConstructionError,
    TopologyGuarantee,
};
use delaunay::prelude::geometry::AdaptiveKernel;
use delaunay::prelude::query::FacetIncidenceAnalysis;
use delaunay::prelude::tds::Tds;
use delaunay::prelude::topology::validation::ManifoldError;
use delaunay::topology::characteristics::{euler, validation};
use delaunay::topology::traits::topological_space::{
    GlobalTopology, TopologyError, TopologyKind, ToroidalConstructionMode,
};

// =============================================================================
// DETERMINISTIC TESTS - KNOWN CONFIGURATIONS
// =============================================================================

#[test]
fn test_empty_triangulation_euler() {
    // Empty triangulation should have χ = 0
    let tds: Tds<(), (), 3> = Tds::empty();

    let counts = euler::count_simplices(&tds).unwrap();
    assert_eq!(counts.count(0), 0); // No vertices
    assert_eq!(counts.count(3), 0); // No simplices

    let chi = euler::euler_characteristic(&counts);
    assert_eq!(chi, 0, "Empty triangulation should have χ = 0");

    let classification = euler::classify_triangulation(&tds, GlobalTopology::Euclidean).unwrap();
    assert_eq!(classification, euler::TopologyClassification::Empty);

    let expected = euler::expected_chi_for(&classification);
    assert_eq!(expected, Some(0));
}

#[test]
fn test_2d_single_triangle() {
    // Single triangle: V=3, E=3, F=1 → χ = 3-3+1 = 1
    let vertices = vec![
        vertex!([0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.5, 1.0]).unwrap(),
    ];

    let dt = DelaunayTriangulation::try_new_with_topology_guarantee(
        &vertices,
        TopologyGuarantee::PLManifold,
    )
    .unwrap();
    let result = validation::validate_triangulation_euler(dt.tds(), dt.global_topology()).unwrap();

    assert_eq!(result.counts.count(0), 3, "Should have 3 vertices");
    assert_eq!(result.counts.count(1), 3, "Should have 3 edges");
    assert_eq!(result.counts.count(2), 1, "Should have 1 face");
    assert_eq!(result.chi, 1, "Single triangle should have χ = 1");
    assert!(result.is_valid(), "Topology should be valid");
    assert_eq!(
        result.classification,
        euler::TopologyClassification::SingleSimplex(2)
    );
}

#[test]
fn test_euler_rejects_open_single_simplex_in_closed_topology() {
    let vertices = vec![
        vertex!([0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.5, 1.0]).unwrap(),
    ];

    let dt = DelaunayTriangulation::try_new_with_topology_guarantee(
        &vertices,
        TopologyGuarantee::PLManifold,
    )
    .unwrap();

    let classify_err =
        euler::classify_triangulation(dt.tds(), GlobalTopology::Spherical).unwrap_err();
    assert_matches!(
        classify_err,
        TopologyError::BoundaryClassification { source }
            if matches!(
                source.as_ref(),
                ManifoldError::BoundaryFacetInClosedTopology {
                    topology: TopologyKind::Spherical,
                    ..
                }
            )
    );

    let validation_err =
        validation::validate_triangulation_euler(dt.tds(), GlobalTopology::Spherical).unwrap_err();
    assert_matches!(
        validation_err,
        TopologyError::BoundaryClassification { source }
            if matches!(
                source.as_ref(),
                ManifoldError::BoundaryFacetInClosedTopology {
                    topology: TopologyKind::Spherical,
                    ..
                }
            )
    );
}

#[test]
fn test_2d_multiple_triangles() {
    // Four points forming multiple triangles
    let vertices = vec![
        vertex!([0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.5, 1.0]).unwrap(),
        vertex!([0.5, 0.3]).unwrap(), // Interior point
    ];

    let dt = DelaunayTriangulation::try_new_with_topology_guarantee(
        &vertices,
        TopologyGuarantee::PLManifold,
    )
    .unwrap();
    let result = validation::validate_triangulation_euler(dt.tds(), dt.global_topology()).unwrap();

    assert_eq!(result.counts.count(0), 4, "Should have 4 vertices");
    assert_eq!(
        result.chi, 1,
        "2D triangulation with boundary should have χ = 1"
    );
    assert!(result.is_valid(), "Topology should be valid");
    assert_eq!(
        result.classification,
        euler::TopologyClassification::Ball(2)
    );
}

#[test]
fn test_3d_single_tetrahedron() {
    // Single tetrahedron: V=4, E=6, F=4, C=1 → χ = 4-6+4-1 = 1
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 1.0]).unwrap(),
    ];

    let dt = DelaunayTriangulation::try_new_with_topology_guarantee(
        &vertices,
        TopologyGuarantee::PLManifold,
    )
    .unwrap();
    let result = validation::validate_triangulation_euler(dt.tds(), dt.global_topology()).unwrap();

    assert_eq!(result.counts.count(0), 4, "Should have 4 vertices");
    assert_eq!(result.counts.count(1), 6, "Should have 6 edges");
    assert_eq!(result.counts.count(2), 4, "Should have 4 faces");
    assert_eq!(result.counts.count(3), 1, "Should have 1 simplex");
    assert_eq!(result.chi, 1, "Single tetrahedron should have χ = 1");
    assert!(result.is_valid(), "Topology should be valid");
    assert_eq!(
        result.classification,
        euler::TopologyClassification::SingleSimplex(3)
    );
}

#[test]
fn test_3d_with_interior_vertex() {
    // Tetrahedron with one interior vertex
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 1.0]).unwrap(),
        vertex!([0.25, 0.25, 0.25]).unwrap(), // Interior point
    ];

    let dt = DelaunayTriangulation::try_new_with_topology_guarantee(
        &vertices,
        TopologyGuarantee::PLManifold,
    )
    .unwrap();
    let result = validation::validate_triangulation_euler(dt.tds(), dt.global_topology()).unwrap();

    assert_eq!(result.counts.count(0), 5, "Should have 5 vertices");
    assert_eq!(
        result.chi, 1,
        "3D triangulation with boundary should have χ = 1"
    );
    assert!(result.is_valid(), "Topology should be valid");
    assert_eq!(
        result.classification,
        euler::TopologyClassification::Ball(3)
    );
}

#[test]
fn test_4d_single_simplex() {
    // Single 4-simplex: V=5, E=10, F=10, T=5, C=1
    // χ = 5 - 10 + 10 - 5 + 1 = 1
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 1.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 0.0, 1.0]).unwrap(),
    ];

    let dt = DelaunayTriangulation::try_new_with_topology_guarantee(
        &vertices,
        TopologyGuarantee::PLManifold,
    )
    .unwrap();
    let result = validation::validate_triangulation_euler(dt.tds(), dt.global_topology()).unwrap();

    assert_eq!(result.counts.count(0), 5, "Should have 5 vertices");
    assert_eq!(result.counts.count(1), 10, "Should have 10 edges");
    assert_eq!(result.counts.count(2), 10, "Should have 10 faces");
    assert_eq!(result.counts.count(3), 5, "Should have 5 tetrahedra");
    assert_eq!(result.counts.count(4), 1, "Should have 1 4-simplex");
    assert_eq!(result.chi, 1, "Single 4-simplex should have χ = 1");
    assert!(result.is_valid(), "Topology should be valid");
    assert_eq!(
        result.classification,
        euler::TopologyClassification::SingleSimplex(4)
    );
}

#[test]
fn test_5d_single_simplex() {
    // Single 5-simplex: χ = 1
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0, 0.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 1.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 0.0, 1.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 0.0, 0.0, 1.0]).unwrap(),
    ];

    let dt = DelaunayTriangulation::try_new_with_topology_guarantee(
        &vertices,
        TopologyGuarantee::PLManifold,
    )
    .unwrap();
    let result = validation::validate_triangulation_euler(dt.tds(), dt.global_topology()).unwrap();

    assert_eq!(result.counts.count(0), 6, "Should have 6 vertices");
    assert_eq!(result.chi, 1, "Single 5-simplex should have χ = 1");
    assert!(result.is_valid(), "Topology should be valid");
    assert_eq!(
        result.classification,
        euler::TopologyClassification::SingleSimplex(5)
    );
}

// =============================================================================
// TOROIDAL EXPLICIT CONSTRUCTION TESTS
// =============================================================================

#[test]
fn test_2d_toroidal_explicit_construction_rejected() {
    // 3×3 periodic grid torus (T²): 9 vertices, 18 triangles.
    // Each quad (i,j) is split into two triangles with periodic wrapping.
    // Vertex layout: v(row, col) = row * 3 + col.
    //
    // Row 0: (0.0, 0.0)  (0.333, 0.0)  (0.667, 0.0)
    // Row 1: (0.0, 0.333) (0.333, 0.333) (0.667, 0.333)
    // Row 2: (0.0, 0.667) (0.333, 0.667) (0.667, 0.667)
    const N: usize = 3;
    let coords: &[[f64; 2]] = &[
        [0.0, 0.0],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.0, 0.333],
        [0.333, 0.333],
        [0.667, 0.333],
        [0.0, 0.667],
        [0.333, 0.667],
        [0.667, 0.667],
    ];
    let vertices: Vec<_> = coords
        .iter()
        .map(|c| vertex!([c[0], c[1]]).unwrap())
        .collect();
    let v = |i: usize, j: usize| -> usize { (i % N) * N + (j % N) };
    let mut simplices = Vec::with_capacity(2 * N * N);
    for i in 0..N {
        for j in 0..N {
            // Up triangle: [v(i,j), v(i+1,j), v(i,j+1)]
            simplices.push(vec![v(i, j), v(i + 1, j), v(i, j + 1)]);
            // Down triangle: [v(i+1,j), v(i+1,j+1), v(i,j+1)]
            simplices.push(vec![v(i + 1, j), v(i + 1, j + 1), v(i, j + 1)]);
        }
    }

    let topology =
        GlobalTopology::try_toroidal([1.0, 1.0], ToroidalConstructionMode::Explicit).unwrap();
    let err = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
        .unwrap()
        .global_topology(topology)
        .topology_guarantee(TopologyGuarantee::Pseudomanifold)
        .build::<()>()
        .expect_err("explicit toroidal connectivity requires a quotient embedding validator");

    match err {
        DelaunayTriangulationConstructionError::ExplicitConstruction(
            ExplicitConstructionError::UnsupportedExplicitTopology { topology },
        ) => assert_eq!(topology, TopologyKind::Toroidal),
        other => panic!("expected explicit construction validation failure, got {other:?}"),
    }
}

#[test]
fn test_3d_toroidal_explicit_construction_rejected() {
    // 3×3×3 periodic Freudenthal triangulation of T³.
    //
    // The Freudenthal (Kuhn) triangulation decomposes each unit cube into D! = 6
    // tetrahedra, one per permutation σ of the coordinate axes. For permutation σ,
    // the tetrahedron walks from the cube corner along the permuted axes:
    //   v₀ = corner, v₁ = v₀+e_{σ(1)}, v₂ = v₁+e_{σ(2)}, v₃ = v₂+e_{σ(3)}
    //
    // References:
    //   Freudenthal, H. "Simplizialzerlegungen von beschränkter Flachheit."
    //     Annals of Mathematics 43(3), 1942, pp. 580–582.
    //   Munkres, J. R. Elements of Algebraic Topology. Addison–Wesley, 1984.
    //   See also REFERENCES.md § "Cube Triangulation (Freudenthal / Kuhn)".
    //
    // Expected f-vector for N=3:
    //   V = 27, E = 189, F = 324, C = 162
    //   χ = 27 − 189 + 324 − 162 = 0  ✓
    const N: usize = 3;

    // Build 27 vertices on [0,1)³.
    let grid: [f64; N] = [0.0, 1.0 / 3.0, 2.0 / 3.0];
    let mut vertices = Vec::with_capacity(N * N * N);
    for &x in &grid {
        for &y in &grid {
            for &z in &grid {
                vertices.push(vertex!([x, y, z]).unwrap());
            }
        }
    }

    let v = |i: usize, j: usize, k: usize| -> usize { ((i % N) * N + (j % N)) * N + (k % N) };

    // The 6 permutations of (0,1,2) give the 6 Freudenthal tetrahedra per cube.
    let permutations: &[[usize; 3]] = &[
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];

    let mut simplices = Vec::with_capacity(6 * N * N * N);
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                let corner = [i, j, k];
                for perm in permutations {
                    // Build the 4 vertices of this tetrahedron by walking
                    // along the permuted axes from the cube corner.
                    let mut cur = corner;
                    let v0 = v(cur[0], cur[1], cur[2]);
                    cur[perm[0]] += 1;
                    let v1 = v(cur[0], cur[1], cur[2]);
                    cur[perm[1]] += 1;
                    let v2 = v(cur[0], cur[1], cur[2]);
                    cur[perm[2]] += 1;
                    let v3 = v(cur[0], cur[1], cur[2]);
                    simplices.push(vec![v0, v1, v2, v3]);
                }
            }
        }
    }

    let topology =
        GlobalTopology::try_toroidal([1.0, 1.0, 1.0], ToroidalConstructionMode::Explicit).unwrap();
    let err = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
        .unwrap()
        .global_topology(topology)
        .topology_guarantee(TopologyGuarantee::Pseudomanifold)
        .build::<()>()
        .expect_err("explicit toroidal connectivity requires a quotient embedding validator");

    match err {
        DelaunayTriangulationConstructionError::ExplicitConstruction(
            ExplicitConstructionError::UnsupportedExplicitTopology { topology },
        ) => assert_eq!(topology, TopologyKind::Toroidal),
        other => panic!("expected explicit construction validation failure, got {other:?}"),
    }
}

// =============================================================================
// FULL COMPLEX VS BOUNDARY TESTS
// =============================================================================
// These tests verify that:
// 1. The full simplicial complex has χ = 1 (D-ball)
// 2. The boundary (convex hull) has χ matching (D-1)-sphere
// This serves as a cross-check between simplex counting, boundary detection,
// and convex hull extraction.
//
// Each test creates a triangulation with at least one interior point to ensure
// we have a proper D-ball (not just a single simplex).

macro_rules! test_complex_with_interior {
    ($test_name:ident, $dim:expr, $vertices:expr, $expected_boundary_chi:expr) => {
        #[test]
        fn $test_name() {
            type DT = DelaunayTriangulation<AdaptiveKernel<f64>, (), (), $dim>;
            let dt = DT::try_new_with_topology_guarantee($vertices, TopologyGuarantee::PLManifold)
                .unwrap();

            // Full complex should have χ = 1 (D-ball)
            let full_result =
                validation::validate_triangulation_euler(dt.tds(), dt.global_topology()).unwrap();
            assert_eq!(
                full_result.chi, 1,
                "Full {}-dimensional complex should have χ = 1 (D-ball)",
                $dim
            );
            assert!(
                full_result.is_valid(),
                "Full complex should be valid for dimension {}",
                $dim
            );

            // Verify we have boundary facets
            let boundary_facet_count = dt.tds().number_of_one_sided_facets().unwrap();
            assert!(
                boundary_facet_count > 0,
                "Should have boundary facets in dimension {}",
                $dim
            );

            // Verify we have more than one simplex (ensuring interior point)
            let simplex_count = dt.tds().number_of_simplices();
            assert!(
                simplex_count > 1,
                "Should have multiple simplices (>1) to ensure interior point in dimension {}",
                $dim
            );

            // Compute boundary Euler characteristic
            // Expected boundary χ values:
            // - 2D: boundary is S¹ (circle) → χ = 0
            // - 3D: boundary is S² (sphere) → χ = 2
            // - 4D: boundary is S³ (3-sphere) → χ = 0
            // - 5D: boundary is S⁴ (4-sphere) → χ = 2
            // Generally: χ(S^k) = 1 + (-1)^k
            let boundary_counts =
                euler::count_boundary_simplices(dt.tds(), dt.global_topology()).unwrap();
            let boundary_chi = euler::euler_characteristic(&boundary_counts);

            let expected_boundary_chi = $expected_boundary_chi;
            assert_eq!(
                boundary_chi,
                expected_boundary_chi,
                "Boundary should have χ = {} for {}-sphere in dimension {}",
                expected_boundary_chi,
                $dim - 1,
                $dim
            );
        }
    };
}

// 2D test: triangle with interior point
test_complex_with_interior!(
    test_2d_complex_with_interior,
    2,
    &[
        vertex!([0.0, 0.0]).unwrap(),
        vertex!([2.0, 0.0]).unwrap(),
        vertex!([1.0, 2.0]).unwrap(),
        vertex!([1.0, 0.5]).unwrap(), // Interior point
    ],
    0 // S¹ has χ = 0
);

// 3D test: tetrahedron with interior point
test_complex_with_interior!(
    test_3d_complex_with_interior,
    3,
    &[
        vertex!([0.0, 0.0, 0.0]).unwrap(),
        vertex!([2.0, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 2.0, 0.0]).unwrap(),
        vertex!([1.0, 0.5, 2.0]).unwrap(),
        vertex!([1.0, 0.5, 0.5]).unwrap(), // Interior point
    ],
    2 // S² has χ = 2
);

// 4D test: 4-simplex with interior point
test_complex_with_interior!(
    test_4d_complex_with_interior,
    4,
    &[
        vertex!([0.0, 0.0, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 1.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 0.0, 1.0]).unwrap(),
        vertex!([0.21, 0.17, 0.13, 0.11]).unwrap(), // Interior point
    ],
    0 // S³ has χ = 0
);

// 5D test: 5-simplex with interior point
test_complex_with_interior!(
    test_5d_complex_with_interior,
    5,
    &[
        vertex!([0.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0, 0.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 1.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 0.0, 1.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 0.0, 0.0, 1.0]).unwrap(),
        vertex!([0.2, 0.2, 0.2, 0.2, 0.2]).unwrap(), // Interior point
    ],
    2 // S⁴ has χ = 2
);
