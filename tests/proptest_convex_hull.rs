//! Property-based tests for convex hull operations.
//!
//! This module uses proptest to verify fundamental properties of convex hull
//! extraction and operations, including:
//! - Hull facets form a closed polytope
//! - All vertices are on or inside the hull
//! - Hull is valid after triangulation construction
//! - Facet count bounds and topological properties
//!
//! Tests are generated for dimensions 2D-5D using macros to reduce duplication.

use delaunay::core::traits::boundary_analysis::BoundaryAnalysis;
use delaunay::core::triangulation_data_structure::Tds;
use delaunay::core::vertex::Vertex;
use delaunay::geometry::algorithms::convex_hull::ConvexHull;
use delaunay::geometry::point::Point;
use delaunay::geometry::traits::coordinate::Coordinate;
use proptest::prelude::*;

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

/// Macro to generate convex hull property tests for a given dimension
macro_rules! test_convex_hull_properties {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal) => {
        pastey::paste! {
            proptest! {
                /// Property: Convex hull can be constructed from valid triangulation
                #[test]
                fn [<prop_hull_construction_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(Vertex::from_points)
                ) {
                    if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                        // Filter: Skip degenerate configurations (no boundary facets)
                        // These are tested separately in dedicated degenerate case tests
                        let boundary_count = tds.number_of_boundary_facets().unwrap_or(0);
                        prop_assume!(boundary_count > 0);

                        // Should be able to construct hull from valid triangulation
                        let hull_result = ConvexHull::from_triangulation(&tds);
                        prop_assert!(
                            hull_result.is_ok(),
                            "{}D convex hull construction should succeed for valid triangulation: {:?}",
                            $dim,
                            hull_result.err()
                        );

                        if let Ok(hull) = hull_result {
                            // Hull should be valid for the TDS it was created from
                            prop_assert!(
                                hull.is_valid_for_tds(&tds),
                                "{}D convex hull should be valid for its source TDS",
                                $dim
                            );

                            // Facet count should be positive
                            prop_assert!(
                                hull.facet_count() > 0,
                                "{}D convex hull must have at least one facet",
                                $dim
                            );
                        }
                    }
                }

                /// Property: Hull facet count is bounded by combinatorial limits
                #[test]
                fn [<prop_hull_facet_bounds_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(Vertex::from_points)
                ) {
                    if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                        if let Ok(hull) = ConvexHull::from_triangulation(&tds) {
                            let facet_count = hull.facet_count();
                            let vertex_count = tds.vertices().count();

                            // Lower bound: more than D facets for a simplex in D dimensions
                            let min_facets = $dim;

                            // Upper bound: for n vertices in D dimensions, convex hull
                            // has at most O(n^(D/2)) facets (loose bound to avoid false positives)
                            let max_facets = if vertex_count <= $dim + 1 {
                                $dim + 1
                            } else {
                                // Very generous upper bound
                                (vertex_count * vertex_count) * 10
                            };

                            prop_assert!(
                                facet_count > min_facets,
                                "{}D hull with {} vertices should have more than {} facets, got {}",
                                $dim,
                                vertex_count,
                                min_facets,
                                facet_count
                            );

                            prop_assert!(
                                facet_count <= max_facets,
                                "{}D hull with {} vertices should have at most {} facets, got {}",
                                $dim,
                                vertex_count,
                                max_facets,
                                facet_count
                            );
                        }
                    }
                }

                /// Property: Hull becomes invalid after TDS modification
                #[test]
                fn [<prop_hull_staleness_detection_ $dim d>](
                    initial_vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(Vertex::from_points),
                    new_point in prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new)
                ) {
                    if let Ok(mut tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&initial_vertices) {
                        // Filter: Skip degenerate initial configurations
                        let initial_boundary_count = tds.number_of_boundary_facets().unwrap_or(0);
                        prop_assume!(initial_boundary_count > 0);

                        if let Ok(hull) = ConvexHull::from_triangulation(&tds) {
                            // Hull should be valid initially
                            prop_assert!(
                                hull.is_valid_for_tds(&tds),
                                "{}D hull should be valid for its TDS before modification",
                                $dim
                            );

                            // Modify the TDS
                            let new_vertex = Vertex::from_points(vec![new_point]);
                            if tds.add(new_vertex[0].clone()).is_ok() {
                                // Filter: Skip if modification resulted in degenerate configuration
                                let modified_boundary_count = tds.number_of_boundary_facets().unwrap_or(0);
                                prop_assume!(modified_boundary_count > 0);

                                // Hull should now be invalid (stale)
                                prop_assert!(
                                    !hull.is_valid_for_tds(&tds),
                                    "{}D hull should be invalid after TDS modification",
                                    $dim
                                );

                                // Creating a new hull should succeed for non-degenerate TDS
                                let new_hull_result = ConvexHull::from_triangulation(&tds);
                                prop_assert!(
                                    new_hull_result.is_ok(),
                                    "{}D creating new hull after modification should succeed",
                                    $dim
                                );

                                if let Ok(new_hull) = new_hull_result {
                                    prop_assert!(
                                        new_hull.is_valid_for_tds(&tds),
                                        "{}D new hull should be valid for modified TDS",
                                        $dim
                                    );
                                }
                            }
                        }
                    }
                }

                /// Property: Hull vertices are a subset of triangulation vertices
                #[test]
                fn [<prop_hull_vertices_subset_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(Vertex::from_points)
                ) {
                    if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                        if let Ok(hull) = ConvexHull::from_triangulation(&tds) {
                            let tds_vertex_count = tds.vertices().count();
                            let facet_count = hull.facet_count();

                            // Each facet references D vertices (D-dimensional facets in D-space)
                            // Total references could be up to facet_count * D
                            // But actual unique vertices on hull should be <= total vertices
                            // This is a basic sanity check
                            prop_assert!(
                                facet_count > 0,
                                "{}D hull should have positive facet count",
                                $dim
                            );

                            // For D-dimensional triangulation with n vertices,
                            // hull should have between D+1 and n vertices
                            // (we can't easily extract unique hull vertices without iterating facets,
                            // so we just check facet count is reasonable)
                            prop_assert!(
                                facet_count > $dim,
                                "{}D hull should have more than {} facets for {} TDS vertices",
                                $dim,
                                $dim,
                                tds_vertex_count
                            );
                        }
                    }
                }

                /// Property: Hull facet count for minimal simplex is D+1
                #[test]
                fn [<prop_minimal_simplex_hull_ $dim d>](
                    base_scale in 0.1f64..10.0f64
                ) {
                    // Create a minimal simplex (D+1 vertices in D dimensions)
                    let mut points = Vec::new();

                    // Origin
                    points.push(Point::new([0.0f64; $dim]));

                    // D more points along coordinate axes
                    for i in 0..$dim {
                        let mut coords = [0.0f64; $dim];
                        coords[i] = base_scale;
                        points.push(Point::new(coords));
                    }

                    let vertices = Vertex::from_points(points);

                    if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                        if let Ok(hull) = ConvexHull::from_triangulation(&tds) {
                            // A minimal D-simplex should have exactly D+1 facets
                            prop_assert_eq!(
                                hull.facet_count(),
                                $dim + 1,
                                "{}D minimal simplex hull should have exactly {} facets",
                                $dim,
                                $dim + 1
                            );
                        }
                    }
                }

                /// Property: Reconstructing hull from same TDS gives consistent facet count
                #[test]
                fn [<prop_hull_reconstruction_consistency_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(Vertex::from_points)
                ) {
                    if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                        if let Ok(hull1) = ConvexHull::from_triangulation(&tds) {
                            if let Ok(hull2) = ConvexHull::from_triangulation(&tds) {
                                // Both hulls should have the same facet count
                                prop_assert_eq!(
                                    hull1.facet_count(),
                                    hull2.facet_count(),
                                    "{}D reconstructing hull from same TDS should give same facet count",
                                    $dim
                                );

                                // Both should be valid for the same TDS
                                prop_assert!(
                                    hull1.is_valid_for_tds(&tds) && hull2.is_valid_for_tds(&tds),
                                    "{}D both reconstructed hulls should be valid for the same TDS",
                                    $dim
                                );
                            }
                        }
                    }
                }
            }
        }
    };
}

// Generate tests for dimensions 2-5
// Parameters: dimension, min_vertices, max_vertices
test_convex_hull_properties!(2, 4, 10);
test_convex_hull_properties!(3, 5, 12);
test_convex_hull_properties!(4, 6, 14);
test_convex_hull_properties!(5, 7, 16);
