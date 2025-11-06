//! Integration tests for geometry utility functions
//!
//! This module tests error paths, edge cases, and multi-dimensional behavior
//! for geometric utility functions that are not fully covered by doctests.
//!
//! Focus areas:
//! - Error handling for invalid inputs (ranges, simplex points, conversions)
//! - Multi-dimensional testing (2D-5D) for functions with dimension-dependent behavior
//! - Edge cases (zero, negative values, extreme magnitudes)

use approx::assert_relative_eq;
use delaunay::geometry::point::Point;
use delaunay::geometry::traits::coordinate::Coordinate;
use delaunay::geometry::util::{
    CircumcenterError, RandomPointGenerationError, circumcenter, circumradius,
    circumradius_with_center, facet_measure, generate_grid_points, generate_poisson_points,
    generate_random_points, generate_random_points_seeded, hypot, inradius, simplex_volume,
    squared_norm,
};

// ============================================================================
// Random Point Generation - Error Path Tests
// ============================================================================

#[test]
fn test_generate_random_points_invalid_range() {
    // min >= max should error
    let result = generate_random_points::<f64, 2>(100, (10.0, -10.0));
    assert!(result.is_err(), "Should error when min >= max");

    match result {
        Err(RandomPointGenerationError::InvalidRange { min, max }) => {
            assert_eq!(min, "10.0");
            assert_eq!(max, "-10.0");
        }
        _ => panic!("Expected InvalidRange error"),
    }
}

#[test]
fn test_generate_random_points_seeded_invalid_range() {
    let result = generate_random_points_seeded::<f64, 3>(50, (5.0, 5.0), 42);
    assert!(result.is_err(), "Should error when min == max");
}

#[test]
fn test_generate_random_points_seeded_reproducibility() {
    // Same seed should produce identical results
    let points1 = generate_random_points_seeded::<f64, 2>(10, (0.0, 1.0), 42).unwrap();
    let points2 = generate_random_points_seeded::<f64, 2>(10, (0.0, 1.0), 42).unwrap();
    assert_eq!(
        points1, points2,
        "Same seed should produce identical points"
    );

    // Different seeds should produce different results
    let points3 = generate_random_points_seeded::<f64, 2>(10, (0.0, 1.0), 123).unwrap();
    assert_ne!(
        points1, points3,
        "Different seeds should produce different points"
    );
}

#[test]
fn test_generate_grid_points_zero_points_per_dim() {
    let result = generate_grid_points::<f64, 3>(0, 1.0, [0.0, 0.0, 0.0]);
    assert!(result.is_err(), "Should error with zero points_per_dim");

    match result {
        Err(RandomPointGenerationError::InvalidPointCount { n_points }) => {
            assert_eq!(n_points, 0);
        }
        _ => panic!("Expected InvalidPointCount error"),
    }
}

#[test]
fn test_generate_poisson_points_invalid_range() {
    let result = generate_poisson_points::<f64, 2>(50, (1.0, -1.0), 0.1, 42);
    assert!(result.is_err(), "Should error with invalid range");
}

#[test]
fn test_generate_poisson_points_zero_min_distance() {
    // Zero or negative min_distance should fall back to simple random generation
    let result = generate_poisson_points::<f64, 2>(10, (0.0, 1.0), 0.0, 42);
    assert!(result.is_ok(), "Should succeed with zero min_distance");
    assert_eq!(
        result.unwrap().len(),
        10,
        "Should generate exact count with zero spacing"
    );
}

#[test]
fn test_generate_poisson_points_impossible_spacing() {
    // min_distance too large for bounds should generate fewer points or error
    let result = generate_poisson_points::<f64, 2>(100, (0.0, 1.0), 10.0, 42);
    // Should either error or return very few points
    if let Ok(points) = result {
        assert!(
            points.len() < 100,
            "Should generate fewer points with impossible spacing"
        );
    }
}

// ============================================================================
// Circumcenter / Circumradius - Error Path Tests
// ============================================================================

#[test]
fn test_circumcenter_empty_points() {
    let points: Vec<Point<f64, 3>> = vec![];
    let result = circumcenter(&points);
    assert!(result.is_err(), "Should error with empty point set");
    assert!(
        result == Err(CircumcenterError::EmptyPointSet),
        "Expected EmptyPointSet error"
    );
}

#[test]
fn test_circumcenter_invalid_simplex_wrong_count() {
    // 3D requires exactly 4 points, provide 3
    let points = vec![
        Point::new([0.0, 0.0, 0.0]),
        Point::new([1.0, 0.0, 0.0]),
        Point::new([0.0, 1.0, 0.0]),
    ];
    let result = circumcenter(&points);
    assert!(result.is_err(), "Should error with wrong number of points");

    match result {
        Err(CircumcenterError::InvalidSimplex {
            actual,
            expected,
            dimension,
        }) => {
            assert_eq!(actual, 3);
            assert_eq!(expected, 4);
            assert_eq!(dimension, 3);
        }
        _ => panic!("Expected InvalidSimplex error"),
    }
}

#[test]
fn test_circumcenter_degenerate_2d_collinear() {
    // Collinear points in 2D should fail (degenerate triangle)
    let points = vec![
        Point::new([0.0, 0.0]),
        Point::new([1.0, 0.0]),
        Point::new([2.0, 0.0]),
    ];
    let result = circumcenter(&points);
    // Should fail due to matrix inversion (singular matrix)
    assert!(result.is_err(), "Should error with collinear points");
}

#[test]
fn test_circumradius_with_center_empty_points() {
    let points: Vec<Point<f64, 3>> = vec![];
    let center = Point::new([0.0, 0.0, 0.0]);
    let result = circumradius_with_center(&points, &center);
    assert!(result.is_err(), "Should error with empty point set");
}

// ============================================================================
// Simplex Volume - Multi-dimensional and Error Tests
// ============================================================================

#[test]
fn test_simplex_volume_invalid_simplex_2d() {
    // 2D requires 3 points, provide 2
    let points = vec![Point::new([0.0, 0.0]), Point::new([1.0, 0.0])];
    let result = simplex_volume(&points);
    assert!(result.is_err(), "Should error with wrong point count");
}

#[test]
fn test_simplex_volume_degenerate_1d_coincident() {
    // 1D: coincident points (zero length)
    let points = vec![Point::new([5.0]), Point::new([5.0])];
    let result = simplex_volume(&points);
    assert!(result.is_err(), "Should error with coincident points in 1D");
}

#[test]
fn test_simplex_volume_degenerate_2d_collinear() {
    // 2D: collinear points (zero area)
    let points = vec![
        Point::new([0.0, 0.0]),
        Point::new([1.0, 0.0]),
        Point::new([2.0, 0.0]),
    ];
    let result = simplex_volume(&points);
    assert!(result.is_err(), "Should error with collinear points in 2D");
}

#[test]
fn test_simplex_volume_degenerate_3d_coplanar() {
    // 3D: coplanar points (zero volume)
    let points = vec![
        Point::new([0.0, 0.0, 0.0]),
        Point::new([1.0, 0.0, 0.0]),
        Point::new([0.0, 1.0, 0.0]),
        Point::new([0.5, 0.5, 0.0]), // In same plane
    ];
    let result = simplex_volume(&points);
    assert!(result.is_err(), "Should error with coplanar points in 3D");
}

// Macro to generate tests for regular D-simplexes across multiple dimensions
macro_rules! test_regular_simplex_volume {
    ($dim:expr, $expected:expr) => {
        pastey::paste! {
            #[test]
            fn [<test_simplex_volume_ $dim d>]() {
                // Generate unit simplex vertices for dimension D
                let mut points = vec![Point::new([0.0; $dim])];
                for i in 0..$dim {
                    let mut coords = [0.0; $dim];
                    coords[i] = 1.0;
                    points.push(Point::new(coords));
                }
                let volume = simplex_volume(&points).unwrap();
                assert_relative_eq!(volume, $expected, epsilon = 1e-10);
            }
        }
    };
}

test_regular_simplex_volume!(4, 1.0 / 24.0); // 4D: 1/4! = 1/24
test_regular_simplex_volume!(5, 1.0 / 120.0); // 5D: 1/5! = 1/120

// ============================================================================
// Facet Measure - Multi-dimensional Tests
// ============================================================================

#[test]
fn test_facet_measure_invalid_count_2d() {
    // 2D requires 2 points, provide 3
    let points = vec![
        Point::new([0.0, 0.0]),
        Point::new([1.0, 0.0]),
        Point::new([0.0, 1.0]),
    ];
    let result = facet_measure(&points);
    assert!(result.is_err(), "Should error with wrong point count");
}

#[test]
fn test_facet_measure_1d_point() {
    // 1D: single point has measure 0
    let points = vec![Point::new([5.0])];
    let measure = facet_measure(&points).unwrap();
    assert_relative_eq!(measure, 0.0, epsilon = 1e-10);
}

#[test]
fn test_facet_measure_4d_facet() {
    // 4D: facet is 3-dimensional (tetrahedron)
    let points = vec![
        Point::new([0.0, 0.0, 0.0, 0.0]),
        Point::new([1.0, 0.0, 0.0, 0.0]),
        Point::new([0.0, 1.0, 0.0, 0.0]),
        Point::new([0.0, 0.0, 1.0, 0.0]),
    ];
    let measure = facet_measure(&points).unwrap();
    // 3D tetrahedron with unit edges: volume = 1/6
    assert_relative_eq!(measure, 1.0 / 6.0, epsilon = 1e-10);
}

// ============================================================================
// Inradius - Error and Multi-dimensional Tests
// ============================================================================

#[test]
fn test_inradius_invalid_simplex() {
    // 3D requires 4 points, provide 3
    let points = vec![
        Point::new([0.0, 0.0, 0.0]),
        Point::new([1.0, 0.0, 0.0]),
        Point::new([0.0, 1.0, 0.0]),
    ];
    let result = inradius(&points);
    assert!(result.is_err(), "Should error with wrong point count");
}

#[test]
fn test_inradius_degenerate_simplex() {
    // Degenerate 2D triangle (collinear points)
    let points = vec![
        Point::new([0.0, 0.0]),
        Point::new([1.0, 0.0]),
        Point::new([2.0, 0.0]),
    ];
    let result = inradius(&points);
    assert!(result.is_err(), "Should error with degenerate simplex");
}

#[test]
fn test_inradius_1d_segment() {
    // 1D: segment inradius is half the length
    let points = vec![Point::new([0.0]), Point::new([10.0])];
    let r_in = inradius(&points).unwrap();
    assert_relative_eq!(r_in, 5.0, epsilon = 1e-10); // length/2 = 10/2 = 5
}

#[test]
fn test_inradius_3d_tetrahedron() {
    // 3D: Right-angled tetrahedron with vertices at origin and unit axes
    // This is the standard 3-simplex: vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    let points = vec![
        Point::new([0.0, 0.0, 0.0]),
        Point::new([1.0, 0.0, 0.0]),
        Point::new([0.0, 1.0, 0.0]),
        Point::new([0.0, 0.0, 1.0]),
    ];

    // First verify the volume is correct
    // Volume formula: V = (1/6)|det([v1-v0, v2-v0, v3-v0])| = 1/6 for unit simplex
    let volume = simplex_volume(&points).unwrap();
    assert_relative_eq!(volume, 1.0 / 6.0, epsilon = 1e-10);

    // Now verify the inradius using the formula: r = D×V / S
    // where D=3 (dimension), V=volume, S=surface area
    //
    // Mathematical derivation for this specific tetrahedron:
    // 1. Volume: V = 1/6 (verified above)
    //
    // 2. Surface area calculation (4 triangular faces):
    //    Face 1 (xy-plane): vertices (0,0,0), (1,0,0), (0,1,0)
    //      → Right triangle with legs 1,1: area = (1×1)/2 = 1/2
    //    Face 2 (xz-plane): vertices (0,0,0), (1,0,0), (0,0,1)
    //      → Right triangle with legs 1,1: area = (1×1)/2 = 1/2
    //    Face 3 (yz-plane): vertices (0,0,0), (0,1,0), (0,0,1)
    //      → Right triangle with legs 1,1: area = (1×1)/2 = 1/2
    //    Face 4 (slanted): vertices (1,0,0), (0,1,0), (0,0,1)
    //      → Equilateral triangle with side √2: area = (√3/4)×2 = √3/2
    //
    //    Total surface area: S = 3×(1/2) + √3/2 = (3 + √3)/2
    //
    // 3. Inradius formula: r = 3V/S = 3×(1/6) / ((3 + √3)/2)
    //                        = (1/2) / ((3 + √3)/2)
    //                        = 1 / (3 + √3)
    //                        ≈ 0.2113248654 (numerical value)
    let r_in = inradius(&points).unwrap();
    let expected_inradius = 1.0 / (3.0 + 3.0_f64.sqrt());
    assert_relative_eq!(r_in, expected_inradius, epsilon = 1e-10);
}

// ============================================================================
// Hypot and Squared Norm - Multi-dimensional Tests
// ============================================================================

// Macro to test hypot across dimensions with unit vectors
macro_rules! test_hypot_unit_vector {
    ($dim:expr) => {
        pastey::paste! {
            #[test]
            fn [<test_hypot_ $dim d>]() {
                let distance = hypot([1.0; $dim]);
                let expected = f64::from($dim).sqrt();
                assert_relative_eq!(distance, expected, epsilon = 1e-10);
            }
        }
    };
}

#[test]
fn test_hypot_0d() {
    let distance = hypot::<f64, 0>([]);
    assert_relative_eq!(distance, 0.0, epsilon = 1e-10);
}

#[test]
fn test_hypot_1d_negative() {
    let distance = hypot([-10.0]);
    assert_relative_eq!(distance, 10.0, epsilon = 1e-10);
}

test_hypot_unit_vector!(2);
test_hypot_unit_vector!(3);
test_hypot_unit_vector!(4);
test_hypot_unit_vector!(5);

#[test]
fn test_hypot_large_values() {
    // Test numerical stability with large values
    let distance: f64 = hypot([1e100, 1e100]);
    assert!(
        distance.is_finite(),
        "Should handle large values without overflow"
    );
    assert!(
        distance > 1e100,
        "Distance should be greater than individual components"
    );
}

#[test]
fn test_squared_norm_2d() {
    let norm_sq = squared_norm([3.0, 4.0]);
    assert_relative_eq!(norm_sq, 25.0, epsilon = 1e-10);
}

#[test]
fn test_squared_norm_5d() {
    let norm_sq = squared_norm([1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_relative_eq!(norm_sq, 55.0, epsilon = 1e-10); // 1+4+9+16+25=55
}

#[test]
fn test_squared_norm_zero() {
    let norm_sq = squared_norm([0.0, 0.0, 0.0]);
    assert_relative_eq!(norm_sq, 0.0, epsilon = 1e-10);
}

// ============================================================================
// Circumcenter / Circumradius - Multi-dimensional Tests
// ============================================================================

// Macro to test circumcenter/circumradius computation across dimensions
macro_rules! test_circumradius_unit_simplex {
    ($dim:expr) => {
        pastey::paste! {
            #[test]
            fn [<test_circumradius_ $dim d>]() {
                // Generate unit simplex vertices
                let mut points = vec![Point::new([0.0; $dim])];
                for i in 0..$dim {
                    let mut coords = [0.0; $dim];
                    coords[i] = 1.0;
                    points.push(Point::new(coords));
                }
                let radius = circumradius(&points);
                assert!(radius.is_ok(), concat!("Should compute ", stringify!($dim), "D circumradius"));
                assert!(radius.unwrap() > 0.0, "Circumradius should be positive");
            }
        }
    };
}

test_circumradius_unit_simplex!(4);
test_circumradius_unit_simplex!(5);

#[test]
fn test_circumradius_2d_equilateral_triangle() {
    // 2D: equilateral triangle with side length 2
    // Vertices of equilateral triangle
    let s = 2.0_f64.sqrt();
    let points = vec![
        Point::new([0.0, 0.0]),
        Point::new([2.0, 0.0]),
        Point::new([1.0, s * 1.5_f64.sqrt()]),
    ];
    let radius = circumradius(&points).unwrap();
    // For equilateral triangle with side a: R = a/sqrt(3)
    let expected = 2.0 / 3.0_f64.sqrt();
    assert_relative_eq!(radius, expected, epsilon = 1e-6);
}
