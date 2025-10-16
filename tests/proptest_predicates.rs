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
use delaunay::geometry::util::circumcenter;
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

/// Strategy for generating 5D points
fn point_5d() -> impl Strategy<Value = Point<f64, 5>> {
    prop::array::uniform5(finite_coordinate()).prop_map(Point::new)
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

    /// Property: 4D orientation sign flip under vertex swap
    #[test]
    fn prop_orientation_sign_flip_4d(
        p0 in point_4d(),
        p1 in point_4d(),
        p2 in point_4d(),
        p3 in point_4d(),
        p4 in point_4d(),
    ) {
        let simplex1 = vec![p0, p1, p2, p3, p4];
        let simplex2 = vec![p0, p1, p2, p4, p3];  // Swap last two

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

    /// Property: 5D orientation sign flip under vertex swap
    #[test]
    fn prop_orientation_sign_flip_5d(
        p0 in point_5d(),
        p1 in point_5d(),
        p2 in point_5d(),
        p3 in point_5d(),
        p4 in point_5d(),
        p5 in point_5d(),
    ) {
        let simplex1 = vec![p0, p1, p2, p3, p4, p5];
        let simplex2 = vec![p0, p1, p2, p3, p5, p4];  // Swap last two

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

    /// Property: A point scaled inward from circumcenter toward centroid should eventually be INSIDE.
    /// This validates the fundamental geometric property that points close to the simplex interior
    /// are inside the circumsphere.
    #[test]
    fn prop_insphere_inward_scaling_makes_point_inside_2d(
        p0 in point_2d(),
        p1 in point_2d(),
        p2 in point_2d(),
    ) {
        let simplex = vec![p0, p1, p2];

        // Compute circumcenter and centroid
        if let Ok(circumcenter_point) = circumcenter(&simplex) {
            let center_coords: [f64; 2] = circumcenter_point.into();

            // Compute centroid
            let coords0: [f64; 2] = p0.into();
            let coords1: [f64; 2] = p1.into();
            let coords2: [f64; 2] = p2.into();
            let centroid = [
                (coords0[0] + coords1[0] + coords2[0]) / 3.0,
                (coords0[1] + coords1[1] + coords2[1]) / 3.0,
            ];

            // Scale inward from circumcenter toward centroid
            // A point very close to the centroid should be INSIDE for well-conditioned simplices
            let scale = 0.01;  // Very close to centroid
            let interior_point = Point::new([
                center_coords[0].mul_add(1.0 - scale, centroid[0] * scale),
                center_coords[1].mul_add(1.0 - scale, centroid[1] * scale),
            ]);

            if let Ok(result) = insphere(&simplex, interior_point) {
                // Skip if circumcenter and centroid are too close (degenerate case)
                let dx = center_coords[0] - centroid[0];
                let dy = center_coords[1] - centroid[1];
                let dist_sq = dx.mul_add(dx, dy * dy);

                if dist_sq > 1e-6 {
                    prop_assert!(
                        result == InSphere::INSIDE || result == InSphere::BOUNDARY,
                        "Point scaled toward centroid should be INSIDE or BOUNDARY, got {:?}",
                        result
                    );
                }
            }
        }
    }

    /// Property: A point scaled far from the simplex centroid should be OUTSIDE.
    /// This tests that the insphere predicate correctly identifies distant points.
    #[test]
    fn prop_insphere_distant_point_is_outside_2d(
        p0 in point_2d(),
        p1 in point_2d(),
        p2 in point_2d(),
    ) {
        let simplex = vec![p0, p1, p2];

        // Compute simplex centroid
        let coords0: [f64; 2] = p0.into();
        let coords1: [f64; 2] = p1.into();
        let coords2: [f64; 2] = p2.into();

        let centroid = [
            (coords0[0] + coords1[0] + coords2[0]) / 3.0,
            (coords0[1] + coords1[1] + coords2[1]) / 3.0,
        ];

        // Create a point far from the centroid by scaling the centroid direction
        // Use a large scaling factor to ensure it's outside
        let scale = 10000.0;
        let far_point = Point::new([
            centroid[0] * scale,
            centroid[1] * scale,
        ]);

        if let Ok(result) = insphere(&simplex, far_point) {
            // The scaled point should be OUTSIDE (unless centroid is near zero)
            if centroid[0].abs() > 0.01 || centroid[1].abs() > 0.01 {
                prop_assert!(
                    result == InSphere::OUTSIDE,
                    "Expected scaled point to be OUTSIDE, got {:?}",
                    result
                );
            }
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

    /// Property: 4D insphere - simplex vertices should be on boundary
    #[test]
    fn prop_insphere_simplex_vertices_on_boundary_4d(
        p0 in point_4d(),
        p1 in point_4d(),
        p2 in point_4d(),
        p3 in point_4d(),
        p4 in point_4d(),
    ) {
        let simplex = vec![p0, p1, p2, p3, p4];

        for vertex in &simplex {
            if let Ok(result) = insphere(&simplex, *vertex) {
                prop_assert!(
                    result == InSphere::BOUNDARY || result == InSphere::OUTSIDE,
                    "4D simplex vertex is {:?}, expected BOUNDARY or OUTSIDE",
                    result
                );
            }
        }
    }

    /// Property: 4D insphere - a point scaled far away from circumcenter is OUTSIDE
    #[test]
    fn prop_insphere_scaling_makes_point_outside_4d(
        p0 in point_4d(),
        p1 in point_4d(),
        p2 in point_4d(),
        p3 in point_4d(),
        p4 in point_4d(),
    ) {
        let simplex_points = vec![p0, p1, p2, p3, p4];

        // Compute circumcenter
        if let Ok(center) = circumcenter(&simplex_points) {
            let center_coords: [f64; 4] = center.into();

            // Create a point far from the circumcenter by scaling away from it
            let scale = 10000.0;
            let far_point = Point::new([
                center_coords[0] * scale,
                center_coords[1] * scale,
                center_coords[2] * scale,
                center_coords[3] * scale,
            ]);

            if let Ok(result) = insphere(&simplex_points, far_point) {
                // Point scaled away from circumcenter should be OUTSIDE
                // (unless circumcenter is at origin)
                if center_coords.iter().any(|&c| c.abs() > 0.01) {
                    prop_assert!(
                        result == InSphere::OUTSIDE,
                        "4D point scaled away from circumcenter should be OUTSIDE, got {:?}",
                        result
                    );
                }
            }
        }
    }

    /// Property: 5D insphere - simplex vertices should be on boundary
    #[test]
    fn prop_insphere_simplex_vertices_on_boundary_5d(
        p0 in point_5d(),
        p1 in point_5d(),
        p2 in point_5d(),
        p3 in point_5d(),
        p4 in point_5d(),
        p5 in point_5d(),
    ) {
        let simplex = vec![p0, p1, p2, p3, p4, p5];

        for vertex in &simplex {
            if let Ok(result) = insphere(&simplex, *vertex) {
                prop_assert!(
                    result == InSphere::BOUNDARY || result == InSphere::OUTSIDE,
                    "5D simplex vertex is {:?}, expected BOUNDARY or OUTSIDE",
                    result
                );
            }
        }
    }

    /// Property: 5D insphere - a point scaled far away from circumcenter is OUTSIDE
    #[test]
    fn prop_insphere_scaling_makes_point_outside_5d(
        p0 in point_5d(),
        p1 in point_5d(),
        p2 in point_5d(),
        p3 in point_5d(),
        p4 in point_5d(),
        p5 in point_5d(),
    ) {
        let simplex_points = vec![p0, p1, p2, p3, p4, p5];

        // Compute circumcenter
        if let Ok(center) = circumcenter(&simplex_points) {
            let center_coords: [f64; 5] = center.into();

            // Create a point far from the circumcenter by scaling away from it
            let scale = 10000.0;
            let far_point = Point::new([
                center_coords[0] * scale,
                center_coords[1] * scale,
                center_coords[2] * scale,
                center_coords[3] * scale,
                center_coords[4] * scale,
            ]);

            if let Ok(result) = insphere(&simplex_points, far_point) {
                // Point scaled away from circumcenter should be OUTSIDE
                // (unless circumcenter is at origin)
                if center_coords.iter().any(|&c| c.abs() > 0.01) {
                    prop_assert!(
                        result == InSphere::OUTSIDE,
                        "5D point scaled away from circumcenter should be OUTSIDE, got {:?}",
                        result
                    );
                }
            }
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
