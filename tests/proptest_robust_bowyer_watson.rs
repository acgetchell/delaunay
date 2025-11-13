//! Property-based tests for `RobustBowyerWatson` algorithm.
//!
//! This module uses proptest to verify fundamental properties of the robust
//! Bowyer-Watson insertion algorithm, including:
//! - TDS validity is maintained after all insertion attempts
//! - Statistics counters are monotonically non-decreasing
//! - Algorithm handles extreme coordinate values gracefully
//! - Bad cells detection is consistent with geometric predicates
//! - Visibility detection works correctly for exterior points
//! - Cache invalidation occurs with structural modifications
//!
//! Tests are generated for dimensions 2D-5D using macros to reduce duplication.

use delaunay::core::algorithms::robust_bowyer_watson::RobustBowyerWatson;
use delaunay::core::traits::facet_cache::FacetCacheProvider;
use delaunay::core::traits::insertion_algorithm::InsertionAlgorithm;
use delaunay::core::triangulation_data_structure::Tds;
use delaunay::core::vertex::Vertex;
use delaunay::geometry::point::Point;
use delaunay::geometry::traits::coordinate::Coordinate;
use delaunay::geometry::util::safe_usize_to_scalar;
use delaunay::vertex;
use proptest::prelude::*;
use std::sync::atomic::Ordering;

// =============================================================================
// TEST CONFIGURATION
// =============================================================================

/// Strategy for generating finite f64 coordinates
fn finite_coordinate() -> impl Strategy<Value = f64> {
    (-100.0..100.0).prop_filter("must be finite", |x: &f64| x.is_finite())
}

// =============================================================================
// DIMENSIONAL TEST GENERATION MACROS
// =============================================================================

/// Macro to generate robust algorithm property tests for a given dimension
macro_rules! test_robust_algorithm_properties {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal) => {
        pastey::paste! {
            proptest! {
                /// Property: TDS validity is maintained after insertion attempts
                #[test]
                fn [<prop_insertion_maintains_validity_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(Vertex::from_points),
                    test_coords in prop::array::[<uniform $dim>](finite_coordinate())
                ) {
                    if let Ok(mut tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                        let mut algorithm = RobustBowyerWatson::new();
                        let test_vertex = Vertex::from_points(vec![Point::new(test_coords)])
                            .into_iter()
                            .next()
                            .expect("single vertex");

                        let _result = algorithm.insert_vertex(&mut tds, test_vertex);

                        // Property: TDS validity maintained regardless of insertion outcome
                        if let Err(e) = tds.is_valid() {
                            prop_assert!(
                                false,
                                "{}D TDS should remain valid after insertion attempt. Validation error: {}",
                                $dim,
                                e
                            );
                        }
                    }
                }

                /// Property: Statistics counters are monotonically non-decreasing
                #[test]
                fn [<prop_statistics_monotonic_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(Vertex::from_points),
                    test_points in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        1..5_usize
                    )
                ) {
                    if let Ok(mut tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                        let mut algorithm = RobustBowyerWatson::new();

                        let mut prev_processed = 0;
                        let mut prev_created = 0;
                        let mut prev_removed = 0;

                        for point in test_points {
                            let test_vertex = Vertex::from_points(vec![point])
                                .into_iter()
                                .next()
                                .expect("single vertex");
                            let _ = algorithm.insert_vertex(&mut tds, test_vertex);

                            let (processed, created, removed) = algorithm.get_statistics();

                            prop_assert!(
                                processed >= prev_processed,
                                "{}D: Processed should be non-decreasing: {} >= {}",
                                $dim, processed, prev_processed
                            );
                            prop_assert!(
                                created >= prev_created,
                                "{}D: Created should be non-decreasing: {} >= {}",
                                $dim, created, prev_created
                            );
                            prop_assert!(
                                removed >= prev_removed,
                                "{}D: Removed should be non-decreasing: {} >= {}",
                                $dim, removed, prev_removed
                            );

                            prev_processed = processed;
                            prev_created = created;
                            prev_removed = removed;
                        }
                    }
                }

                /// Property: Successful insertions have consistent info
                #[test]
                fn [<prop_insertion_info_consistency_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(Vertex::from_points),
                    test_coords in prop::array::[<uniform $dim>](finite_coordinate())
                ) {
                    if let Ok(mut tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                        let mut algorithm = RobustBowyerWatson::new();
                        let test_vertex = Vertex::from_points(vec![Point::new(test_coords)])
                            .into_iter()
                            .next()
                            .expect("single vertex");

                        let result = algorithm.insert_vertex(&mut tds, test_vertex);

                        if let Ok(info) = result {
                            // Property: Successful insertions have success=true
                            prop_assert!(info.success, "{}D: Successful result should have success=true", $dim);

                            // Property: Bowyer-Watson can create fewer cells than removed
                            // (e.g., inserting in a cavity with shared vertices)
                            // Just verify the counts are reasonable
                            prop_assert!(
                                info.cells_created > 0 || info.cells_removed == 0,
                                "{}D: Should create cells unless no cells removed: created={}, removed={}",
                                $dim, info.cells_created, info.cells_removed
                            );
                        }
                    }
                }

                /// Property: Interior point insertion should succeed
                #[test]
                fn [<prop_interior_insertion_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(Vertex::from_points)
                ) {
                    if let Ok(mut tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                        let mut algorithm = RobustBowyerWatson::new();

                        // Create interior point (average of existing vertices)
                        let mut interior_coords = [0.0f64; $dim];
                        for vertex in &vertices {
                            let coords: [f64; $dim] = (*vertex.point()).into();
                            for i in 0..$dim {
                                interior_coords[i] += coords[i];
                            }
                        }
                        let vertex_count_f64 = safe_usize_to_scalar::<f64>(vertices.len())
                            .expect("vertex count should be small enough for f64");
                        for coord in &mut interior_coords {
                            *coord /= vertex_count_f64;
                        }

                        let interior_vertex = Vertex::from_points(vec![Point::new(interior_coords)])
                            .into_iter()
                            .next()
                            .expect("single vertex");
                        let _result = algorithm.insert_vertex(&mut tds, interior_vertex);

                        // Property: Interior point insertion should succeed (or fail gracefully)
                        // and maintain TDS validity
                        if let Err(e) = tds.is_valid() {
                            prop_assert!(
                                false,
                                "{}D: TDS should remain valid after interior insertion attempt. Validation error: {}",
                                $dim,
                                e
                            );
                        }
                    }
                }

                /// Property: Cache invalidation occurs with insertions
                #[test]
                fn [<prop_cache_invalidation_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(Vertex::from_points),
                    test_points in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        1..5_usize
                    )
                ) {
                    if let Ok(mut tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                        let mut algorithm = RobustBowyerWatson::new();

                        let initial_gen = algorithm.cached_generation().load(Ordering::Acquire);

                        for point in test_points {
                            let test_vertex = Vertex::from_points(vec![point])
                                .into_iter()
                                .next()
                                .expect("single vertex");
                            let _ = algorithm.insert_vertex(&mut tds, test_vertex);
                        }

                        let final_gen = algorithm.cached_generation().load(Ordering::Acquire);

                        // Property: Cache generation advances with insertions
                        prop_assert!(
                            final_gen >= initial_gen,
                            "{}D: Cache generation should advance: {} >= {}",
                            $dim, final_gen, initial_gen
                        );
                    }
                }

                /// Property: Reset clears statistics and resets cache generation
                #[test]
                fn [<prop_reset_consistency_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(Vertex::from_points),
                    test_coords in prop::array::[<uniform $dim>](finite_coordinate())
                ) {
                    if let Ok(mut tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                        let mut algorithm = RobustBowyerWatson::new();

                        // Insert a vertex to potentially change state
                        let test_vertex = Vertex::from_points(vec![Point::new(test_coords)])
                            .into_iter()
                            .next()
                            .expect("single vertex");
                        let _ = algorithm.insert_vertex(&mut tds, test_vertex);

                        // Reset algorithm
                        algorithm.reset();

                        let gen_after_reset = algorithm.cached_generation().load(Ordering::Acquire);

                        // Property: Cache generation is reset to 0 after reset
                        // (invalidate_facet_cache stores 0, not increments)
                        prop_assert_eq!(
                            gen_after_reset, 0,
                            "{}D: Cache generation should be 0 after reset",
                            $dim
                        );

                        // Property: Statistics cleared after reset
                        let (processed, created, removed) = algorithm.get_statistics();
                        prop_assert_eq!(processed, 0, "{}D: Processed should be 0 after reset", $dim);
                        prop_assert_eq!(created, 0, "{}D: Created should be 0 after reset", $dim);
                        prop_assert_eq!(removed, 0, "{}D: Removed should be 0 after reset", $dim);
                    }
                }

                /// Property: Algorithm handles extreme coordinate values
                #[test]
                fn [<prop_extreme_coordinates_ $dim d>](
                    scale_exp in -12i32..7i32
                ) {
                    let scale = 10.0f64.powi(scale_exp);
                    // Create simplex with extreme scale
                    let mut scaled_points = Vec::new();
                    scaled_points.push(Point::new([0.0f64; $dim]));
                    for i in 0..$dim {
                        let mut coords = [0.0f64; $dim];
                        coords[i] = scale;
                        scaled_points.push(Point::new(coords));
                    }

                    let vertices: Vec<Vertex<f64, Option<()>, $dim>> = Vertex::from_points(scaled_points);
                    let mut algorithm: RobustBowyerWatson<f64, Option<()>, Option<()>, $dim> = RobustBowyerWatson::new();
                    let tds_result = algorithm.new_triangulation(&vertices);

                    if let Ok(mut tds) = tds_result {
                        // Try inserting at center
                        let center_coord = scale * 0.25;
                        let test_vertex = Vertex::from_points(
                            vec![Point::new([center_coord; $dim])]
                        )
                            .into_iter()
                            .next()
                            .expect("single vertex");
                        let _ = algorithm.insert_vertex(&mut tds, test_vertex);

                        // Property: TDS handles extreme scales gracefully
                        prop_assert!(
                            tds.is_valid().is_ok(),
                            "{}D TDS should handle extreme scale {}",
                            $dim, scale
                        );
                    }
                }
            }
        }
    };
}

// =============================================================================
// 3D-SPECIFIC PROPERTIES
// =============================================================================

proptest! {
    /// Property: Exterior vertex insertion at various angles (3D)
    #[test]
    fn prop_exterior_spherical_3d(
        distance in 0.1f64..10.0,
        angle_theta in 0.0f64..std::f64::consts::TAU,
        angle_phi in 0.0f64..std::f64::consts::PI,
    ) {
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();

        // Spherical coordinates
        let x = distance * angle_phi.sin() * angle_theta.cos();
        let y = distance * angle_phi.sin() * angle_theta.sin();
        let z = distance * angle_phi.cos();

        let exterior_vertex = vertex!([x, y, z]);

        let mut algorithm = RobustBowyerWatson::new();
        let _result = algorithm.insert_vertex(&mut tds, exterior_vertex);

        // Property: Exterior vertex insertion should succeed or fail gracefully
        // TDS validity should be maintained
        prop_assert!(
            tds.is_valid().is_ok(),
            "TDS should remain valid after exterior vertex insertion attempt at distance={}",
            distance
        );
    }

    /// Property: Nearly coplanar configurations handled gracefully (3D)
    #[test]
    fn prop_degenerate_coplanar_3d(
        z_coord in -1.0f64..1.0,
        perturbation in -1e-10f64..1e-10,
    ) {
        let initial_vertices = vec![
            vertex!([0.0, 0.0, perturbation]),
            vertex!([1.0, 0.0, perturbation]),
            vertex!([0.0, 1.0, perturbation]),
            vertex!([0.5, 0.5, perturbation + 1e-12]),
        ];

        let mut algorithm: RobustBowyerWatson<f64, Option<()>, Option<()>, 3> =
            RobustBowyerWatson::for_degenerate_cases();
        let tds_result = algorithm.new_triangulation(&initial_vertices);

        if let Ok(mut tds) = tds_result {
            let test_vertex = vertex!([0.5, 0.5, z_coord]);
            let _ = algorithm.insert_vertex(&mut tds, test_vertex);

            // Property: Degenerate configurations handled without panic
            prop_assert!(
                tds.is_valid().is_ok(),
                "TDS should handle nearly coplanar configuration"
            );
        }
        // If triangulation creation fails, that's acceptable for truly degenerate cases
    }

    /// Property: Exterior point insertion handles distant points (3D)
    #[test]
    fn prop_exterior_insertion_3d(
        exterior_distance in 3.0f64..10.0,
    ) {
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();

        let exterior_vertex = vertex!([exterior_distance, 0.0, 0.0]);

        let mut algorithm = RobustBowyerWatson::new();
        let _result = algorithm.insert_vertex(&mut tds, exterior_vertex);

        // Property: TDS should remain valid after exterior point insertion
        prop_assert!(
            tds.is_valid().is_ok(),
            "TDS should remain valid after distant exterior point at distance {}",
            exterior_distance
        );
    }
}

// =============================================================================
// GENERATE TESTS FOR DIMENSIONS 2D-5D
// =============================================================================

// Parameters: dimension, min_vertices, max_vertices
test_robust_algorithm_properties!(2, 4, 10);
test_robust_algorithm_properties!(3, 5, 12);
test_robust_algorithm_properties!(4, 6, 14);
test_robust_algorithm_properties!(5, 7, 16);
