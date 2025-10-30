//! Property-based tests for geometric quality metrics.
//!
//! This module uses proptest to verify fundamental properties of quality
//! metrics for simplicial cells, including:
//! - Radius ratio bounds (R/r ≥ D for D-dimensional simplex)
//! - Radius ratio scaling invariance
//! - Quality metrics are non-negative
//! - Well-shaped simplices have better quality than degenerate ones
//!
//! Tests are generated for dimensions 2D-5D using macros to reduce duplication.

use delaunay::core::triangulation_data_structure::Tds;
use delaunay::core::vertex::Vertex;
use delaunay::geometry::point::Point;
use delaunay::geometry::quality::radius_ratio;
use delaunay::geometry::traits::coordinate::Coordinate;
use delaunay::geometry::util::{circumradius, inradius};
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

/// Macro to generate quality metric property tests for a given dimension
macro_rules! test_quality_properties {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal, $num_points:literal) => {
        pastey::paste! {
            proptest! {
                /// Property: Radius ratio R/r ≥ D for non-degenerate D-simplices
                #[test]
                fn [<prop_radius_ratio_lower_bound_ $dim d>](
                    simplex_points in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $num_points
                    )
                ) {
                    if let (Ok(r_outer), Ok(r_inner)) = (circumradius(&simplex_points), inradius(&simplex_points)) {
                        // Skip degenerate cases
                        if r_outer > 1e-6 && r_inner > 1e-9 {
                            let ratio = r_outer / r_inner;
                            let dim_f64 = f64::from($dim);
                            prop_assert!(
                                ratio >= dim_f64 - 0.1,  // Small tolerance for numerical errors
                                "{}D radius ratio {} should be >= {}",
                                $dim,
                                ratio,
                                $dim
                            );
                        }
                    }
                }

                /// Property: Radius ratio is scale-invariant
                #[test]
                fn [<prop_radius_ratio_scale_invariant_ $dim d>](
                    simplex_points in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $num_points
                    ),
                    scale in 0.1f64..10.0f64
                ) {
                    // Compute original radius ratio
                    if let (Ok(r1), Ok(r_inner_1)) = (circumradius(&simplex_points), inradius(&simplex_points)) {
                        if r1 > 1e-6 && r_inner_1 > 1e-9 {
                            let ratio1 = r1 / r_inner_1;

                            // Scale all points
                            let scaled_points: Vec<Point<f64, $dim>> = simplex_points
                                .iter()
                                .map(|p| {
                                    let coords: [f64; $dim] = (*p).into();
                                    let mut scaled = [0.0f64; $dim];
                                    for i in 0..$dim {
                                        scaled[i] = coords[i] * scale;
                                    }
                                    Point::new(scaled)
                                })
                                .collect();

                            // Compute scaled radius ratio
                            if let (Ok(r2), Ok(r_inner_2)) = (circumradius(&scaled_points), inradius(&scaled_points)) {
                                if r2 > 1e-6 && r_inner_2 > 1e-9 {
                                    let ratio2 = r2 / r_inner_2;
                                    prop_assert!(
                                        (ratio1 - ratio2).abs() < 0.01 * ratio1.max(1.0),
                                        "{}D radius ratio should be scale-invariant: {} vs {}",
                                        $dim,
                                        ratio1,
                                        ratio2
                                    );
                                }
                            }
                        }
                    }
                }

                /// Property: Radius ratio is always positive for valid simplices
                #[test]
                fn [<prop_radius_ratio_positive_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(Vertex::from_points)
                ) {
                    if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                        for cell_key in tds.cell_keys() {
                            if let Ok(ratio) = radius_ratio(&tds, cell_key) {
                                prop_assert!(
                                    ratio > 0.0,
                                    "{}D radius ratio should be positive, got {}",
                                    $dim,
                                    ratio
                                );
                            }
                        }
                    }
                }

                /// Property: Regular simplex has better (lower) radius ratio than degenerate
                #[test]
                fn [<prop_regular_simplex_quality_ $dim d>](
                    base_scale in 0.1f64..10.0f64
                ) {
                    // Create a regular simplex (vertices at unit distance from origin)
                    let mut regular_points = Vec::new();

                    // First vertex at origin (scaled)
                    let origin = [0.0f64; $dim];
                    regular_points.push(Point::new(origin));

                    // D more vertices with one coordinate = base_scale
                    for i in 0..$dim {
                        let mut coords = [0.0f64; $dim];
                        coords[i] = base_scale;
                        regular_points.push(Point::new(coords));
                    }

                    // Create a flatter simplex (still valid but lower quality)
                    // Make it elongated in one direction with small extent in others
                    let mut degenerate_points = Vec::new();
                    for i in 0..=($dim) {
                        let mut coords = [0.0f64; $dim];
                        coords[0] = f64::from(i) * base_scale;  // Long in first dimension
                        // Add small but non-zero spread in other dimensions
                        for j in 1..$dim {
                            #[allow(clippy::cast_precision_loss)]
                            let j_factor = j as f64;
                            coords[j] = f64::from(i) * 0.05 * base_scale * (1.0 / (j_factor + 1.0));
                        }
                        degenerate_points.push(Point::new(coords));
                    }

                    // Try to compute quality metrics, but skip if degenerate
                    // (higher dimensions with nearly collinear points can cause numerical issues)
                    let regular_quality = circumradius(&regular_points)
                        .and_then(|r| inradius(&regular_points).map(|r_inner| (r, r_inner)));
                    let degenerate_quality = circumradius(&degenerate_points)
                        .and_then(|r| inradius(&degenerate_points).map(|r_inner| (r, r_inner)));

                    if let (Ok((r_reg, r_inner_reg)), Ok((r_deg, r_inner_deg))) = (regular_quality, degenerate_quality) {
                        if r_reg > 1e-6 && r_inner_reg > 1e-9 && r_deg > 1e-6 && r_inner_deg > 1e-9 && r_inner_deg.is_finite() {
                            let ratio_reg = r_reg / r_inner_reg;
                            let ratio_deg = r_deg / r_inner_deg;

                            // Only assert if both ratios are finite
                            if ratio_reg.is_finite() && ratio_deg.is_finite() {
                                prop_assert!(
                                    ratio_reg < ratio_deg * 0.9,  // Regular should be significantly better
                                    "{}D regular simplex quality ({}) should be better than degenerate ({})",
                                    $dim,
                                    ratio_reg,
                                    ratio_deg
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
// Parameters: dimension, min_vertices, max_vertices, num_points (D+1)
test_quality_properties!(2, 4, 10, 3);
test_quality_properties!(3, 5, 12, 4);
test_quality_properties!(4, 6, 14, 5);
test_quality_properties!(5, 7, 16, 6);
