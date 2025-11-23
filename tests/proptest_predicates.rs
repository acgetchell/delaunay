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
use delaunay::geometry::util::{circumcenter, circumradius};
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
// ORIENTATION AND INSPHERE MACROS
// =============================================================================

// Generates orientation sign-flip property across dimensions
macro_rules! gen_orientation_sign_flip {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_orientation_sign_flip_ $dim d>](
                    simplex in prop::collection::vec([<point_ $dim d>](), $dim + 1)
                ) {
                    let simplex1 = simplex.clone();
                    let mut simplex2 = simplex.clone();
                    let len = simplex2.len();
                    simplex2.swap(len - 1, len - 2);

                    let orient1 = simplex_orientation(&simplex1);
                    let orient2 = simplex_orientation(&simplex2);

                    match (orient1, orient2) {
                        (Ok(o1), Ok(o2)) => {
                            if o1 == Orientation::DEGENERATE || o2 == Orientation::DEGENERATE {
                                prop_assert_eq!(o1, Orientation::DEGENERATE);
                                prop_assert_eq!(o2, Orientation::DEGENERATE);
                            } else {
                                match (o1, o2) {
                                    (Orientation::POSITIVE, Orientation::NEGATIVE)
                                    | (Orientation::NEGATIVE, Orientation::POSITIVE) => {}
                                    _ => prop_assert!(false, "Expected opposite orientations, got {:?} and {:?}", o1, o2),
                                }
                            }
                        }
                        (Err(_), Err(_)) => {}
                        _ => prop_assert!(false, "One orientation succeeded while the other failed"),
                    }
                }
            }
        }
    };
}

// 2D-only: cyclic permutation preserves orientation (3-cycle is even)
macro_rules! gen_orientation_cyclic_invariance {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_orientation_cyclic_invariance_ $dim d>](
                    simplex in prop::collection::vec([<point_ $dim d>](), $dim + 1)
                ) {
                    // For 2D: simplex has 3 points
                    let simplex1 = simplex.clone();
                    let simplex2 = [simplex1[1], simplex1[2], simplex1[0]];
                    let simplex3 = [simplex1[2], simplex1[0], simplex1[1]];

                    if let (Ok(o1), Ok(o2), Ok(o3)) = (
                        simplex_orientation(&simplex1),
                        simplex_orientation(&simplex2),
                        simplex_orientation(&simplex3),
                    ) {
                        prop_assert_eq!(o1, o2, "First cyclic permutation changed orientation");
                        prop_assert_eq!(o2, o3, "Second cyclic permutation changed orientation");
                    }
                }
            }
        }
    };
}

// Vertices of a simplex lie on or outside the circumsphere boundary
macro_rules! gen_insphere_simplex_vertices_on_boundary {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_insphere_simplex_vertices_on_boundary_ $dim d>](
                    simplex in prop::collection::vec([<point_ $dim d>](), $dim + 1)
                ) {
                    for vertex in &simplex {
                        if let Ok(result) = insphere(&simplex, *vertex) {
                            prop_assert!(
                                result == InSphere::BOUNDARY || result == InSphere::OUTSIDE,
                                "Simplex vertex is {:?}, expected BOUNDARY or OUTSIDE",
                                result
                            );
                        }
                    }
                }
            }
        }
    };
}

// 2D: scale from circumcenter toward centroid -> INSIDE or BOUNDARY
macro_rules! gen_insphere_inward_scaling_inside {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_insphere_inward_scaling_makes_point_inside_ $dim d>](
                    simplex in prop::collection::vec([<point_ $dim d>](), $dim + 1)
                ) {
                    if let Ok(center) = circumcenter(&simplex) {
                        let center_coords: [f64; $dim] = center.into();

                        // Compute centroid
                        let mut centroid = [0.0_f64; $dim];
                        for p in &simplex {
                            let c: [f64; $dim] = (*p).into();
                            for i in 0..$dim { centroid[i] += c[i]; }
                        }
                        let inv_len: f64 = 1.0 / f64::from($dim + 1);
                        for i in 0..$dim { centroid[i] *= inv_len; }

                        // Use centroid, which is guaranteed to lie inside the simplex
                        let interior_point = Point::new(centroid);

                        if let Ok(result) = insphere(&simplex, interior_point) {
                            // Check separation to avoid degenerate case
                            let mut dist_sq = 0.0;
                            for i in 0..$dim { let d = center_coords[i] - centroid[i]; dist_sq += d * d; }
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
            }
        }
    };
}

// D-dimensional: place point at 10x circumradius along a direction -> OUTSIDE
macro_rules! gen_insphere_distant_point_outside_by_radius {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_insphere_distant_point_is_outside_ $dim d>](
                    simplex in prop::collection::vec([<point_ $dim d>](), $dim + 1)
                ) {
                    if let (Ok(center), Ok(radius)) = (circumcenter(&simplex), circumradius(&simplex)) {
                        if radius < 1e-6 || !radius.is_finite() {
                            return Ok(());
                        }
                        let center_coords: [f64; $dim] = center.into();
                        // Build a direction: normalize center if nonzero; else use e0
                        let mut norm_sq = 0.0;
                        for i in 0..$dim { norm_sq += center_coords[i] * center_coords[i]; }
                        let mut dir = [0.0_f64; $dim];
                        if norm_sq > 1e-8 {
                            let inv_norm = norm_sq.sqrt().recip();
                            for i in 0..$dim { dir[i] = center_coords[i] * inv_norm; }
                        } else {
                            dir[0] = 1.0;
                        }
                        let distance = radius * 10.0;
                        let mut far = [0.0_f64; $dim];
                        for i in 0..$dim {
                            far[i] = dir[i].mul_add(distance, center_coords[i]);
                        }
                        let far_point = Point::new(far);
                        if let Ok(result) = insphere(&simplex, far_point) {
                            prop_assert!(result == InSphere::OUTSIDE, "Point at 10x circumradius from center should be OUTSIDE, got {:?}", result);
                        }
                    }
                }
            }
        }
    };
}

// 4D/5D: scale circumcenter coordinates far away -> OUTSIDE (if not near origin)
macro_rules! gen_insphere_scale_center_outside {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_insphere_scaling_makes_point_outside_ $dim d>](
                    simplex in prop::collection::vec([<point_ $dim d>](), $dim + 1)
                ) {
                    if let Ok(center) = circumcenter(&simplex) {
                        let center_coords: [f64; $dim] = center.into();
                        let scale = 10000.0;
                        let mut far = [0.0_f64; $dim];
                        for i in 0..$dim { far[i] = center_coords[i] * scale; }
                        let far_point = Point::new(far);
                        if let Ok(result) = insphere(&simplex, far_point) {
                            if center_coords.iter().any(|&c| c.abs() > 0.01) {
                                prop_assert!(result == InSphere::OUTSIDE, "Point scaled away from circumcenter should be OUTSIDE, got {:?}", result);
                            }
                        }
                    }
                }
            }
        }
    };
}

// =============================================================================
// ORIENTATION PROPERTY TESTS
// =============================================================================

gen_orientation_sign_flip!(2);
gen_orientation_sign_flip!(3);
gen_orientation_sign_flip!(4);
gen_orientation_sign_flip!(5);

gen_orientation_cyclic_invariance!(2);

// =============================================================================
// INSPHERE PROPERTY TESTS
// =============================================================================

gen_insphere_simplex_vertices_on_boundary!(2);
gen_insphere_simplex_vertices_on_boundary!(3);
gen_insphere_simplex_vertices_on_boundary!(4);
gen_insphere_simplex_vertices_on_boundary!(5);

gen_insphere_inward_scaling_inside!(2);

gen_insphere_distant_point_outside_by_radius!(2);
gen_insphere_distant_point_outside_by_radius!(3);
gen_insphere_distant_point_outside_by_radius!(4);
gen_insphere_distant_point_outside_by_radius!(5);

gen_insphere_scale_center_outside!(4);
gen_insphere_scale_center_outside!(5);

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

        let simplex = [p0, p1, p2];

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
