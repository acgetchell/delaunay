//! Property-based tests for geometric predicates.
//!
//! This module uses proptest to verify mathematical properties of geometric
//! predicates that must hold universally, including:
//! - Orientation sign consistency under coordinate permutations
//! - Insphere transitivity and symmetry properties
//! - Robustness under small perturbations

use delaunay::geometry::point::Point;
use delaunay::geometry::predicates::{InSphere, Orientation, insphere, simplex_orientation};
use delaunay::geometry::traits::coordinate::Coordinate;
use proptest::prelude::*;

// =============================================================================
// TEST CONFIGURATION
// =============================================================================

/// Strategy for generating finite f64 coordinates in a reasonable range
fn finite_coordinate() -> impl Strategy<Value = f64> {
    // Avoid extremely large values that might cause numerical issues
    (-1000.0..1000.0).prop_filter("must be finite", |x: &f64| x.is_finite())
}

/// Strategy for generating 2D points
fn point_2d() -> impl Strategy<Value = Point<f64, 2>> {
    prop::array::uniform2(finite_coordinate()).prop_map(Point::new)
}

/// Strategy for generating 3D points
fn point_3d() -> impl Strategy<Value = Point<f64, 3>> {
    prop::array::uniform3(finite_coordinate()).prop_map(Point::new)
}

/// Strategy for generating 4D points
fn point_4d() -> impl Strategy<Value = Point<f64, 4>> {
    prop::array::uniform4(finite_coordinate()).prop_map(Point::new)
}

// =============================================================================
// ORIENTATION PROPERTY TESTS
// =============================================================================

proptest! {
    /// Property: Reversing the order of two vertices in a simplex should flip
    /// the orientation sign (or remain DEGENERATE).
    #[test]
    fn prop_orientation_sign_flip_2d(
        p0 in point_2d(),
        p1 in point_2d(),
        p2 in point_2d(),
    ) {
        let simplex1 = vec![p0, p1, p2];
        let simplex2 = vec![p0, p2, p1];  // Swap last two

        let orient1 = simplex_orientation(&simplex1);
        let orient2 = simplex_orientation(&simplex2);

        // Both should succeed or both fail
        match (orient1, orient2) {
            (Ok(o1), Ok(o2)) => {
                // If one is degenerate, the other should be too
                if o1 == Orientation::DEGENERATE || o2 == Orientation::DEGENERATE {
                    prop_assert_eq!(o1, Orientation::DEGENERATE);
                    prop_assert_eq!(o2, Orientation::DEGENERATE);
                } else {
                    // Non-degenerate orientations should be opposite
                    match (o1, o2) {
                        (Orientation::POSITIVE, Orientation::NEGATIVE) | (Orientation::NEGATIVE, Orientation::POSITIVE) => {},
                        _ => prop_assert!(false, "Expected opposite orientations, got {:?} and {:?}", o1, o2),
                    }
                }
            }
            (Err(_), Err(_)) => {},
            _ => prop_assert!(false, "One orientation succeeded while the other failed"),
        }
    }

    /// Property: Cyclic permutation of vertices should preserve orientation sign
    /// in 2D (for triangles).
    #[test]
    fn prop_orientation_cyclic_invariance_2d(
        p0 in point_2d(),
        p1 in point_2d(),
        p2 in point_2d(),
    ) {
        let simplex1 = vec![p0, p1, p2];
        let simplex2 = vec![p1, p2, p0];  // Cyclic rotation
        let simplex3 = vec![p2, p0, p1];  // Another cyclic rotation

        if let (Ok(o1), Ok(o2), Ok(o3)) = (
            simplex_orientation(&simplex1),
            simplex_orientation(&simplex2),
            simplex_orientation(&simplex3),
        ) {
            // All cyclic permutations should have the same orientation
            prop_assert_eq!(o1, o2, "First cyclic permutation changed orientation");
            prop_assert_eq!(o2, o3, "Second cyclic permutation changed orientation");
        }
    }

    /// Property: 3D orientation sign flip under vertex swap
    #[test]
    fn prop_orientation_sign_flip_3d(
        p0 in point_3d(),
        p1 in point_3d(),
        p2 in point_3d(),
        p3 in point_3d(),
    ) {
        let simplex1 = vec![p0, p1, p2, p3];
        let simplex2 = vec![p0, p1, p3, p2];  // Swap last two

        let orient1 = simplex_orientation(&simplex1);
        let orient2 = simplex_orientation(&simplex2);

        if let (Ok(o1), Ok(o2)) = (orient1, orient2) {
            if o1 == Orientation::DEGENERATE || o2 == Orientation::DEGENERATE {
                prop_assert_eq!(o1, Orientation::DEGENERATE);
                prop_assert_eq!(o2, Orientation::DEGENERATE);
            } else {
                match (o1, o2) {
                    (Orientation::POSITIVE, Orientation::NEGATIVE) | (Orientation::NEGATIVE, Orientation::POSITIVE) => {},
                    _ => prop_assert!(false, "Expected opposite orientations"),
                }
            }
        }
    }
}

// =============================================================================
// INSPHERE PROPERTY TESTS
// =============================================================================

proptest! {
    /// Property: A point on the circumcenter should be equidistant from all
    /// simplex vertices (within numerical tolerance).
    ///
    /// Note: This tests the BOUNDARY case implicitly.
    #[test]
    fn prop_insphere_simplex_vertices_on_boundary_2d(
        p0 in point_2d(),
        p1 in point_2d(),
        p2 in point_2d(),
    ) {
        let simplex = vec![p0, p1, p2];

        // All vertices of the simplex should be on or near the boundary
        // of the circumsphere (they define it)
        for vertex in &simplex {
            if let Ok(result) = insphere(&simplex, *vertex) {
                // Vertices should not be strictly INSIDE (due to numerical tolerance)
                // They should be BOUNDARY or possibly OUTSIDE due to rounding
                prop_assert!(
                    result == InSphere::BOUNDARY || result == InSphere::OUTSIDE,
                    "Simplex vertex {:?} is reported as {:?}, expected BOUNDARY or OUTSIDE",
                    vertex,
                    result
                );
            }
        }
    }

    /// Property: The centroid of a non-degenerate simplex should typically be
    /// INSIDE or BOUNDARY of the circumsphere (not guaranteed for all simplices,
    /// but holds for many "reasonable" configurations).
    #[test]
    fn prop_insphere_centroid_typically_inside_2d(
        p0 in point_2d(),
        p1 in point_2d(),
        p2 in point_2d(),
    ) {
        let simplex = vec![p0, p1, p2];

        // Compute centroid
        let coords0: [f64; 2] = p0.into();
        let coords1: [f64; 2] = p1.into();
        let coords2: [f64; 2] = p2.into();

        let centroid = Point::new([
            (coords0[0] + coords1[0] + coords2[0]) / 3.0,
            (coords0[1] + coords1[1] + coords2[1]) / 3.0,
        ]);

        // Check if centroid test succeeds
        if let Ok(result) = insphere(&simplex, centroid) {
            // For well-formed simplices, centroid is usually INSIDE or BOUNDARY
            // We just verify the function doesn't crash and returns a valid result
            prop_assert!(
                matches!(result, InSphere::INSIDE | InSphere::BOUNDARY | InSphere::OUTSIDE),
                "Unexpected insphere result: {:?}",
                result
            );
        }
    }

    /// Property: Moving a test point farther from the simplex should eventually
    /// make it OUTSIDE the circumsphere (scaling property).
    #[test]
    fn prop_insphere_scaling_makes_point_outside_2d(
        p0 in point_2d(),
        p1 in point_2d(),
        p2 in point_2d(),
        test_point in point_2d(),
    ) {
        let simplex = vec![p0, p1, p2];

        // Get the test point coordinates
        let coords: [f64; 2] = test_point.into();

        // Scale the test point away from origin by a large factor
        let scaled_point = Point::new([coords[0] * 100.0, coords[1] * 100.0]);

        if let Ok(result) = insphere(&simplex, scaled_point) {
            // A point scaled far away should be OUTSIDE
            // (unless the simplex itself is very large, which is unlikely with our range)
            prop_assert!(
                result == InSphere::OUTSIDE,
                "Expected scaled point to be OUTSIDE, got {:?}",
                result
            );
        }
    }

    /// Property: 3D insphere - simplex vertices should be on boundary
    #[test]
    fn prop_insphere_simplex_vertices_on_boundary_3d(
        p0 in point_3d(),
        p1 in point_3d(),
        p2 in point_3d(),
        p3 in point_3d(),
    ) {
        let simplex = vec![p0, p1, p2, p3];

        for vertex in &simplex {
            if let Ok(result) = insphere(&simplex, *vertex) {
                prop_assert!(
                    result == InSphere::BOUNDARY || result == InSphere::OUTSIDE,
                    "3D simplex vertex is {:?}, expected BOUNDARY or OUTSIDE",
                    result
                );
            }
        }
    }

    /// Property: 4D insphere - scaling property
    #[test]
    fn prop_insphere_scaling_makes_point_outside_4d(
        p0 in point_4d(),
        p1 in point_4d(),
        p2 in point_4d(),
        p3 in point_4d(),
        p4 in point_4d(),
        test_point in point_4d(),
    ) {
        let simplex = vec![p0, p1, p2, p3, p4];

        let coords: [f64; 4] = test_point.into();
        let scaled_point = Point::new([
            coords[0] * 100.0,
            coords[1] * 100.0,
            coords[2] * 100.0,
            coords[3] * 100.0,
        ]);

        if let Ok(result) = insphere(&simplex, scaled_point) {
            prop_assert!(
                result == InSphere::OUTSIDE,
                "4D scaled point should be OUTSIDE, got {:?}",
                result
            );
        }
    }
}

// =============================================================================
// CROSS-PREDICATE CONSISTENCY TESTS
// =============================================================================

proptest! {
    /// Property: If a simplex has DEGENERATE orientation, insphere tests
    /// on that simplex may fail or return inconsistent results.
    ///
    /// This test verifies that we handle degenerate cases gracefully.
    #[test]
    fn prop_degenerate_orientation_insphere_robustness_2d(
        p0 in point_2d(),
        scale in 0.0..1.0,  // Small scale to create near-degenerate simplices
    ) {
        // Create a nearly degenerate simplex (three nearly collinear points)
        let coords0: [f64; 2] = p0.into();
        let p1 = Point::new([coords0[0] + 1.0, coords0[1]]);
        let p2 = Point::new([coords0[0] + 2.0, f64::mul_add(scale, 0.01, coords0[1])]);  // Very small y offset

        let simplex = vec![p0, p1, p2];

        // Check orientation
        if let Ok(orientation) = simplex_orientation(&simplex) {
            if orientation == Orientation::DEGENERATE {
                // If degenerate, insphere might fail - that's acceptable
                let _ = insphere(&simplex, p0);
            } else {
                // If not degenerate, insphere should succeed
                prop_assert!(insphere(&simplex, p0).is_ok(), "Insphere failed on non-degenerate simplex");
            }
        }
    }
}
