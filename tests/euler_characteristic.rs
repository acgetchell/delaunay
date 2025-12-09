//! Deterministic integration tests for Euler characteristic computation.
//!
//! This module tests the topology module's Euler characteristic calculation
//! using known geometric configurations across multiple dimensions.
//!
//! ## Test Coverage
//!
//! - Empty triangulations (χ = 0)
//! - Single simplices (χ = 1)
//! - Multiple cells with interior vertices
//! - Known 2D, 3D, 4D, and 5D configurations
//!
//! For property-based tests with random triangulations, see `proptest_euler_characteristic.rs`.

use delaunay::prelude::*;
use delaunay::topology::characteristics::{euler, validation};

// =============================================================================
// DETERMINISTIC TESTS - KNOWN CONFIGURATIONS
// =============================================================================

#[test]
fn test_empty_triangulation_euler() {
    // Empty triangulation should have χ = 0
    let tds: Tds<f64, (), (), 3> = Tds::empty();

    let counts = euler::count_simplices(&tds).unwrap();
    assert_eq!(counts.count(0), 0); // No vertices
    assert_eq!(counts.count(3), 0); // No cells

    let chi = euler::euler_characteristic(&counts);
    assert_eq!(chi, 0, "Empty triangulation should have χ = 0");

    let classification = euler::classify_triangulation(&tds).unwrap();
    assert_eq!(classification, euler::TopologyClassification::Empty);

    let expected = euler::expected_chi_for(&classification);
    assert_eq!(expected, Some(0));
}

#[test]
fn test_2d_single_triangle() {
    // Single triangle: V=3, E=3, F=1 → χ = 3-3+1 = 1
    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.5, 1.0]),
    ];

    let dt = DelaunayTriangulation::new(&vertices).unwrap();
    let result = validation::validate_triangulation_euler(dt.tds()).unwrap();

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
fn test_2d_multiple_triangles() {
    // Four points forming multiple triangles
    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.5, 1.0]),
        vertex!([0.5, 0.3]), // Interior point
    ];

    let dt = DelaunayTriangulation::new(&vertices).unwrap();
    let result = validation::validate_triangulation_euler(dt.tds()).unwrap();

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
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];

    let dt = DelaunayTriangulation::new(&vertices).unwrap();
    let result = validation::validate_triangulation_euler(dt.tds()).unwrap();

    assert_eq!(result.counts.count(0), 4, "Should have 4 vertices");
    assert_eq!(result.counts.count(1), 6, "Should have 6 edges");
    assert_eq!(result.counts.count(2), 4, "Should have 4 faces");
    assert_eq!(result.counts.count(3), 1, "Should have 1 cell");
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
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
        vertex!([0.25, 0.25, 0.25]), // Interior point
    ];

    let dt = DelaunayTriangulation::new(&vertices).unwrap();
    let result = validation::validate_triangulation_euler(dt.tds()).unwrap();

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
        vertex!([0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0]),
    ];

    let dt = DelaunayTriangulation::new(&vertices).unwrap();
    let result = validation::validate_triangulation_euler(dt.tds()).unwrap();

    assert_eq!(result.counts.count(0), 5, "Should have 5 vertices");
    assert_eq!(result.counts.count(1), 10, "Should have 10 edges");
    assert_eq!(result.counts.count(2), 10, "Should have 10 faces");
    assert_eq!(result.counts.count(3), 5, "Should have 5 tetrahedra");
    assert_eq!(result.counts.count(4), 1, "Should have 1 4-cell");
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
        vertex!([0.0, 0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 0.0, 1.0]),
    ];

    let dt = DelaunayTriangulation::new(&vertices).unwrap();
    let result = validation::validate_triangulation_euler(dt.tds()).unwrap();

    assert_eq!(result.counts.count(0), 6, "Should have 6 vertices");
    assert_eq!(result.chi, 1, "Single 5-simplex should have χ = 1");
    assert!(result.is_valid(), "Topology should be valid");
    assert_eq!(
        result.classification,
        euler::TopologyClassification::SingleSimplex(5)
    );
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
            use delaunay::geometry::kernel::FastKernel;
            type DT = DelaunayTriangulation<FastKernel<f64>, (), (), $dim>;
            let dt = DT::new($vertices).unwrap();

            // Full complex should have χ = 1 (D-ball)
            let full_result = validation::validate_triangulation_euler(dt.tds()).unwrap();
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
            let boundary_facet_count = dt.tds().number_of_boundary_facets().unwrap();
            assert!(
                boundary_facet_count > 0,
                "Should have boundary facets in dimension {}",
                $dim
            );

            // Verify we have more than one cell (ensuring interior point)
            let cell_count = dt.tds().number_of_cells();
            assert!(
                cell_count > 1,
                "Should have multiple cells (>1) to ensure interior point in dimension {}",
                $dim
            );

            // Compute boundary Euler characteristic
            // Expected boundary χ values:
            // - 2D: boundary is S¹ (circle) → χ = 0
            // - 3D: boundary is S² (sphere) → χ = 2
            // - 4D: boundary is S³ (3-sphere) → χ = 0
            // - 5D: boundary is S⁴ (4-sphere) → χ = 2
            // Generally: χ(S^k) = 1 + (-1)^k
            let boundary_counts = euler::count_boundary_simplices(dt.tds()).unwrap();
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
        vertex!([0.0, 0.0]),
        vertex!([2.0, 0.0]),
        vertex!([1.0, 2.0]),
        vertex!([1.0, 0.5]), // Interior point
    ],
    0 // S¹ has χ = 0
);

// 3D test: tetrahedron with interior point
test_complex_with_interior!(
    test_3d_complex_with_interior,
    3,
    &[
        vertex!([0.0, 0.0, 0.0]),
        vertex!([2.0, 0.0, 0.0]),
        vertex!([1.0, 2.0, 0.0]),
        vertex!([1.0, 0.5, 2.0]),
        vertex!([1.0, 0.5, 0.5]), // Interior point
    ],
    2 // S² has χ = 2
);

// 4D test: 4-simplex with interior point
test_complex_with_interior!(
    test_4d_complex_with_interior,
    4,
    &[
        vertex!([0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0]),
        vertex!([0.25, 0.25, 0.25, 0.25]), // Interior point
    ],
    0 // S³ has χ = 0
);

// 5D test: 5-simplex with interior point
test_complex_with_interior!(
    test_5d_complex_with_interior,
    5,
    &[
        vertex!([0.0, 0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 0.0, 1.0]),
        vertex!([0.2, 0.2, 0.2, 0.2, 0.2]), // Interior point
    ],
    2 // S⁴ has χ = 2
);
