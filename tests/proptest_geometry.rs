//! Property-based tests for geometric utility functions.
//!
//! This module uses proptest to verify fundamental geometric properties including:
//! - Circumcenter equidistance (all simplex vertices equidistant from circumcenter)
//! - Circumradius consistency (distance from circumcenter to any vertex)
//! - Circumradius agreement between direct and precomputed-center code paths
//! - Volume positivity for non-degenerate simplices
//! - Distance and norm properties (triangle inequality, scaling, non-negativity)
//! - Inradius positivity for non-degenerate simplices
//!
//! Tests are generated for dimensions 2D-5D using macros to reduce duplication.

use delaunay::prelude::geometry::*;
use proptest::prelude::*;

// =============================================================================
// TEST CONFIGURATION
// =============================================================================

/// Strategy for generating finite f64 coordinates
fn finite_coordinate() -> impl Strategy<Value = f64> {
    (-100.0..100.0).prop_filter("must be finite", |x: &f64| x.is_finite())
}

fn well_conditioned_edge_length() -> impl Strategy<Value = f64> {
    (0.25_f64..50.0).prop_filter("must be finite and positive", |x: &f64| {
        x.is_finite() && *x > 0.0
    })
}

fn nonzero_scale() -> impl Strategy<Value = f64> {
    (-90.0_f64..90.0).prop_map(|exponent| 10.0_f64.powf(exponent))
}

fn axis_aligned_simplex<const D: usize>(base: [f64; D], side_lengths: [f64; D]) -> Vec<Point<D>> {
    let mut simplex = Vec::with_capacity(D + 1);
    simplex.push(Point::try_new(base).expect("finite point coordinates"));

    for (axis, side_length) in side_lengths.iter().copied().enumerate() {
        let mut coords = base;
        coords[axis] += side_length;
        simplex.push(Point::try_new(coords).expect("finite point coordinates"));
    }

    simplex
}

fn prop_assert_relative_close(actual: f64, expected: f64) -> Result<(), TestCaseError> {
    let tolerance = 1e-10 * expected.abs().max(1.0e-300);
    prop_assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected:e}, got {actual:e}"
    );
    Ok(())
}

// =============================================================================
// DIMENSIONAL TEST GENERATION MACROS
// =============================================================================

/// Macro to generate geometric property tests for a given dimension
macro_rules! test_geometry_properties {
    ($dim:literal, $num_points:literal) => {
        pastey::paste! {
            proptest! {
                /// Property: All simplex vertices are equidistant from the circumcenter
                #[test]
                fn [<prop_circumcenter_equidistance_ $dim d>](
                    simplex_points in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                        $num_points
                    )
                ) {
                    if let Ok(center) = circumcenter(&simplex_points) {
                        let center_coords = *center.coords();
                        let mut distances = Vec::new();
                        for point in &simplex_points {
                            let point_coords = *point.coords();
                            let mut diff = [0.0; $dim];
                            for i in 0..$dim {
                                diff[i] = point_coords[i] - center_coords[i];
                            }
                            distances.push(hypot(&diff));
                        }
                        if let Some(&first_dist) = distances.first() {
                            for &dist in &distances[1..] {
                                prop_assert!(
                                    (dist - first_dist).abs() < 1e-6 * first_dist.max(1.0)
                                );
                            }
                        }
                    }
                }

                /// Property: Circumradius equals distance from circumcenter to any vertex
                #[test]
                fn [<prop_circumradius_matches_distance_ $dim d>](
                    simplex_points in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                        $num_points
                    )
                ) {
                    if let (Ok(center), Ok(radius)) = (circumcenter(&simplex_points), circumradius(&simplex_points)) {
                        let center_coords = *center.coords();
                        if let Some(first_point) = simplex_points.first() {
                            let point_coords = *first_point.coords();
                            let mut diff = [0.0; $dim];
                            for i in 0..$dim {
                                diff[i] = point_coords[i] - center_coords[i];
                            }
                            let dist = hypot(&diff);
                            prop_assert!((dist - radius).abs() < 1e-6 * radius.max(1.0));
                        }
                    }
                }

                /// Property: Circumradius agrees between direct and precomputed-center paths
                #[test]
                fn [<prop_circumradius_agrees_with_precomputed_center_ $dim d>](
                    base in prop::array::[<uniform $dim>](finite_coordinate()),
                    side_lengths in prop::array::[<uniform $dim>](well_conditioned_edge_length())
                ) {
                    let simplex = axis_aligned_simplex::<$dim>(base, side_lengths);
                    let center = circumcenter(&simplex).map_err(|err| {
                        TestCaseError::fail(format!(
                            "{dim}D axis-aligned simplex should have a circumcenter: {err:?}",
                            dim = $dim,
                        ))
                    })?;
                    let direct = circumradius(&simplex).map_err(|err| {
                        TestCaseError::fail(format!(
                            "{dim}D axis-aligned simplex should have a direct circumradius: {err:?}",
                            dim = $dim,
                        ))
                    })?;
                    let with_center = circumradius_with_center(&simplex, &center).map_err(|err| {
                        TestCaseError::fail(format!(
                            "{dim}D axis-aligned simplex should have a circumradius from center: {err:?}",
                            dim = $dim,
                        ))
                    })?;

                    prop_assert!(direct.is_finite(), "direct radius must be finite");
                    prop_assert!(with_center.is_finite(), "precomputed-center radius must be finite");
                    prop_assert!(direct > 0.0, "direct radius must be positive");
                    prop_assert!(with_center > 0.0, "precomputed-center radius must be positive");
                    prop_assert_relative_close(with_center, direct)?;
                }

                /// Property: Volume of non-degenerate simplex is positive
                #[test]
                fn [<prop_volume_positivity_ $dim d>](
                    simplex_points in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                        $num_points
                    )
                ) {
                    if let Ok(volume) = simplex_volume(&simplex_points) {
                        prop_assert!(volume >= 0.0);
                        if let Ok(radius) = circumradius(&simplex_points) {
                            if radius > 1e-6 {
                                prop_assert!(volume > 0.0);
                            }
                        }
                    }
                }

                /// Property: Inradius of non-degenerate simplex is positive
                #[test]
                fn [<prop_inradius_positivity_ $dim d>](
                    simplex_points in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                        $num_points
                    )
                ) {
                    if let Ok(inrad) = inradius(&simplex_points) {
                        prop_assert!(inrad >= 0.0);
                        if let Ok(volume) = simplex_volume(&simplex_points) {
                            if volume > 1e-6 {
                                prop_assert!(inrad > 0.0);
                            }
                        }
                    }
                }

                /// Property: Hypot (Euclidean norm) satisfies basic properties
                #[test]
                fn [<prop_hypot_properties_ $dim d>](
                    coords in prop::array::[<uniform $dim>](finite_coordinate())
                ) {
                    let norm: f64 = hypot(&coords);
                    prop_assert!(norm >= 0.0);

                    let zero_coords = [0.0f64; $dim];
                    let zero_norm: f64 = hypot(&zero_coords);
                    prop_assert!(zero_norm.abs() < 1e-12);

                    let scale = 2.5f64;
                    let mut scaled_coords = [0.0f64; $dim];
                    for i in 0..$dim {
                        scaled_coords[i] = coords[i] * scale;
                    }
                    let scaled_norm: f64 = hypot(&scaled_coords);
                    let expected = scale.mul_add(norm, -scaled_norm);
                    prop_assert!(expected.abs() < 1e-6 * scaled_norm.max(1.0));
                }

                /// Property: Squared norm equals norm squared
                #[test]
                fn [<prop_squared_norm_consistency_ $dim d>](
                    coords in prop::array::[<uniform $dim>](finite_coordinate())
                ) {
                    let norm = hypot(&coords);
                    let sq_norm = squared_norm(&coords);
                    let diff = norm.mul_add(norm, -sq_norm);
                    prop_assert!(diff.abs() < 1e-6 * sq_norm.max(1.0));
                    prop_assert!(sq_norm >= 0.0);
                }

                /// Property: Triangle inequality for norm: ||a + b|| <= ||a|| + ||b||
                #[test]
                fn [<prop_triangle_inequality_ $dim d>](
                    coords_a in prop::array::[<uniform $dim>](finite_coordinate()),
                    coords_b in prop::array::[<uniform $dim>](finite_coordinate())
                ) {
                    let norm_a = hypot(&coords_a);
                    let norm_b = hypot(&coords_b);
                    let mut coords_sum = [0.0; $dim];
                    for i in 0..$dim {
                        coords_sum[i] = coords_a[i] + coords_b[i];
                    }
                    let norm_sum = hypot(&coords_sum);
                    prop_assert!(norm_sum <= norm_a + norm_b + 1e-10);
                }
            }
        }
    };
}

// Generate tests for dimensions 2-5
// Parameters: dimension, number_of_points (D+1 for D-simplex)
test_geometry_properties!(2, 3);
test_geometry_properties!(3, 4);
test_geometry_properties!(4, 5);
test_geometry_properties!(5, 6);

proptest! {
    /// Property: Low-dimensional simplex volume remains scale-aware across many orders
    /// of magnitude. This guards against fixed absolute degeneracy thresholds.
    #[test]
    fn prop_low_dimensional_simplex_volume_accepts_scaled_valid_simplices(scale in nonzero_scale()) {
        let segment = vec![Point::try_new([0.0]).expect("finite point coordinates"), Point::try_new([scale]).expect("finite point coordinates")];
        let length = simplex_volume(&segment).unwrap();
        prop_assert_relative_close(length, scale)?;

        let triangle = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([scale, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, scale]).expect("finite point coordinates"),
        ];
        let area = simplex_volume(&triangle).unwrap();
        prop_assert_relative_close(area, scale * scale / 2.0)?;

        let tetrahedron = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([scale, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, scale, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, scale]).expect("finite point coordinates"),
        ];
        let volume = simplex_volume(&tetrahedron).unwrap();
        prop_assert_relative_close(volume, scale * scale * scale / 6.0)?;
    }

    /// Property: Low-dimensional facet measure remains scale-aware for valid facets.
    #[test]
    fn prop_low_dimensional_facet_measure_accepts_scaled_valid_facets(scale in nonzero_scale()) {
        let segment = vec![Point::try_new([0.0, 0.0]).expect("finite point coordinates"), Point::try_new([scale, 0.0]).expect("finite point coordinates")];
        let length = facet_measure(&segment).unwrap();
        prop_assert_relative_close(length, scale)?;

        let triangle = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([scale, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, scale, 0.0]).expect("finite point coordinates"),
        ];
        let area = facet_measure(&triangle).unwrap();
        prop_assert_relative_close(area, scale * scale / 2.0)?;
    }
}
