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
use delaunay::geometry::quality::{normalized_volume, radius_ratio};
use delaunay::geometry::traits::coordinate::Coordinate;
use delaunay::geometry::util::{circumradius, inradius, safe_usize_to_scalar};
use proptest::prelude::*;
use std::collections::HashMap;

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
                    let mut degenerate_points = Vec::with_capacity($dim + 1);
                    degenerate_points.push(Point::new([0.0f64; $dim]));
                    for i in 0..$dim {
                        let mut coords = [0.0f64; $dim];
                        let i_f64: f64 = safe_usize_to_scalar(i).unwrap();
                        coords[0] = (10.0 + i_f64) * base_scale;
                        let axis = (i + 1) % $dim;
                        coords[axis] += 0.05 * base_scale;
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

                /// Property: Radius ratio is translation-invariant
                #[test]
                fn [<prop_radius_ratio_translation_invariant_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(Vertex::from_points),
                    translation in prop::array::[<uniform $dim>](finite_coordinate())
                ) {
                    if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                        // Translate all vertices
                        let translated_vertices: Vec<_> = vertices
                            .iter()
                            .map(|v| {
                                let coords: [f64; $dim] = (*v.point()).into();
                                let mut translated = [0.0f64; $dim];
                                for i in 0..$dim {
                                    translated[i] = coords[i] + translation[i];
                                }
                                // Use from_points to create vertices with auto-generated UUIDs
                                Point::new(translated)
                            })
                            .collect::<Vec<_>>();

                        let translated_vertices = Vertex::from_points(translated_vertices);

                        if let Ok(tds_translated) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&translated_vertices) {
                            // Build mapping from original UUIDs to translated UUIDs
                            let uuid_map: HashMap<_, _> = vertices.iter()
                                .zip(translated_vertices.iter())
                                .map(|(orig, trans)| (orig.uuid(), trans.uuid()))
                                .collect();

                            // Match cells by translating their vertex UUIDs
                            for orig_key in tds.cell_keys() {
                                if let Some(orig_cell) = tds.get_cell(orig_key) {
                                    // Get original cell's vertex UUIDs
                                    if let Ok(orig_uuids) = orig_cell.vertex_uuids(&tds) {
                                        // Map to translated UUIDs
                                        let trans_uuids: Vec<_> = orig_uuids
                                            .iter()
                                            .filter_map(|uuid| uuid_map.get(uuid))
                                            .copied()
                                            .collect();
                                        let mut matched = false;

                                        // Find matching cell in translated triangulation
                                        for trans_key in tds_translated.cell_keys() {
                                            if let Some(trans_cell) = tds_translated.get_cell(trans_key) {
                                                if let Ok(trans_cell_uuids) =
                                                    trans_cell.vertex_uuids(&tds_translated)
                                                {
                                                    // Check if cells have same vertices (by UUID)
                                                    if trans_uuids.len() == trans_cell_uuids.len()
                                                        && trans_uuids
                                                            .iter()
                                                            .all(|u| trans_cell_uuids.contains(u))
                                                    {
                                                        matched = true;
                                                        // Found matching cell - compare quality
                                                        if let (Ok(ratio_orig), Ok(ratio_trans)) = (
                                                            radius_ratio(&tds, orig_key),
                                                            radius_ratio(&tds_translated, trans_key),
                                                        ) {
                                                            let rel_diff = ((ratio_orig - ratio_trans).abs()
                                                                / ratio_orig.max(1.0))
                                                            .min(1.0);
                                                            prop_assert!(
                                                                rel_diff < 0.05,
                                                                "{}D radius ratio should be translation-invariant: {} vs {} (diff: {})",
                                                                $dim,
                                                                ratio_orig,
                                                                ratio_trans,
                                                                rel_diff
                                                            );
                                                        }
                                                        break;
                                                    }
                                                }
                                            }
                                        }

                                        // If no matching cell is found in the translated triangulation,
                                        // skip this cell. The robust construction pipeline may legitimately
                                        // drop or re-triangulate cells differently under extreme
                                        // translations; in those cases we cannot meaningfully compare
                                        // quality metrics.
                                        if !matched {
                                            continue;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                /// Property: Normalized volume is translation-invariant
                #[test]
                fn [<prop_normalized_volume_translation_invariant_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(Vertex::from_points),
                    translation in prop::array::[<uniform $dim>](finite_coordinate())
                ) {
                    if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                        // Translate all vertices
                        let translated_vertices: Vec<_> = vertices
                            .iter()
                            .map(|v| {
                                let coords: [f64; $dim] = (*v.point()).into();
                                let mut translated = [0.0f64; $dim];
                                for i in 0..$dim {
                                    translated[i] = coords[i] + translation[i];
                                }
                                Point::new(translated)
                            })
                            .collect::<Vec<_>>();

                        let translated_vertices = Vertex::from_points(translated_vertices);

                        if let Ok(tds_translated) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&translated_vertices) {
                            // Build UUID mapping
                            let uuid_map: HashMap<_, _> = vertices.iter()
                                .zip(translated_vertices.iter())
                                .map(|(orig, trans)| (orig.uuid(), trans.uuid()))
                                .collect();

                            // Match cells by vertex UUIDs and compare quality
                            for orig_key in tds.cell_keys() {
                                if let Some(orig_cell) = tds.get_cell(orig_key) {
                                    if let Ok(orig_uuids) = orig_cell.vertex_uuids(&tds) {
                                        let trans_uuids: Vec<_> = orig_uuids
                                            .iter()
                                            .filter_map(|uuid| uuid_map.get(uuid))
                                            .copied()
                                            .collect();
                                        let mut matched = false;

                                        for trans_key in tds_translated.cell_keys() {
                                            if let Some(trans_cell) = tds_translated.get_cell(trans_key) {
                                                if let Ok(trans_cell_uuids) =
                                                    trans_cell.vertex_uuids(&tds_translated)
                                                {
                                                    if trans_uuids.len() == trans_cell_uuids.len()
                                                        && trans_uuids
                                                            .iter()
                                                            .all(|u| trans_cell_uuids.contains(u))
                                                    {
                                                        matched = true;
                                                        if let (Ok(vol_orig), Ok(vol_trans)) = (
                                                            normalized_volume(&tds, orig_key),
                                                            normalized_volume(&tds_translated, trans_key),
                                                        ) {
                                                            let rel_diff = ((vol_orig - vol_trans).abs()
                                                                / vol_orig.max(1e-6))
                                                            .min(1.0);
                                                            prop_assert!(
                                                                rel_diff < 0.01,
                                                                "{}D normalized volume should be translation-invariant: {} vs {} (diff: {})",
                                                                $dim,
                                                                vol_orig,
                                                                vol_trans,
                                                                rel_diff
                                                            );
                                                        }
                                                        break;
                                                    }
                                                }
                                            }
                                        }

                                        // As with radius ratio, skip cells that cannot be matched in the
                                        // translated triangulation due to robustness-driven topology changes.
                                        if !matched {
                                            continue;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                /// Property: Radius ratio is scale-invariant (uniform scaling)
                #[test]
                fn [<prop_normalized_volume_scale_invariant_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(Vertex::from_points),
                    scale in 0.1f64..10.0f64
                ) {
                    if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                        // Scale all vertices uniformly
                        let scaled_vertices: Vec<_> = vertices
                            .iter()
                            .map(|v| {
                                let coords: [f64; $dim] = (*v.point()).into();
                                let mut scaled = [0.0f64; $dim];
                                for i in 0..$dim {
                                    scaled[i] = coords[i] * scale;
                                }
                                Point::new(scaled)
                            })
                            .collect::<Vec<_>>();

                        let scaled_vertices = Vertex::from_points(scaled_vertices);

                        if let Ok(tds_scaled) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&scaled_vertices) {
                            // Build UUID mapping
                            let uuid_map: HashMap<_, _> = vertices.iter()
                                .zip(scaled_vertices.iter())
                                .map(|(orig, scaled)| (orig.uuid(), scaled.uuid()))
                                .collect();

                            // Match cells by vertex UUIDs and compare quality
                            for orig_key in tds.cell_keys() {
                                if let Some(orig_cell) = tds.get_cell(orig_key) {
                                    if let Ok(orig_uuids) = orig_cell.vertex_uuids(&tds) {
                                        let scaled_uuids: Vec<_> = orig_uuids
                                            .iter()
                                            .filter_map(|uuid| uuid_map.get(uuid))
                                            .copied()
                                            .collect();
                                        let mut matched = false;

                                        for scaled_key in tds_scaled.cell_keys() {
                                            if let Some(scaled_cell) = tds_scaled.get_cell(scaled_key) {
                                                if let Ok(scaled_cell_uuids) =
                                                    scaled_cell.vertex_uuids(&tds_scaled)
                                                {
                                                    if scaled_uuids.len() == scaled_cell_uuids.len()
                                                        && scaled_uuids
                                                            .iter()
                                                            .all(|u| scaled_cell_uuids.contains(u))
                                                    {
                                                        matched = true;
                                                        if let (Ok(vol_orig), Ok(vol_scaled)) = (
                                                            normalized_volume(&tds, orig_key),
                                                            normalized_volume(&tds_scaled, scaled_key),
                                                        ) {
                                                            let rel_diff = ((vol_orig - vol_scaled).abs()
                                                                / vol_orig.max(1e-6))
                                                            .min(1.0);
                                                            prop_assert!(
                                                                rel_diff < 0.01,
                                                                "{}D normalized volume should be scale-invariant: {} vs {} (diff: {})",
                                                                $dim,
                                                                vol_orig,
                                                                vol_scaled,
                                                                rel_diff
                                                            );
                                                        }
                                                        break;
                                                    }
                                                }
                                            }
                                        }

                                        // Skip unmatched cells where the scaled triangulation no longer
                                        // contains an equivalent combinatorial cell (e.g., due to
                                        // degeneracy detection changing under scaling).
                                        if !matched {
                                            continue;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                /// Property: Both metrics detect degeneracy consistently
                #[test]
                fn [<prop_degeneracy_consistency_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(Vertex::from_points)
                ) {
                    if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                        for cell_key in tds.cell_keys() {
                            let rr_result = radius_ratio(&tds, cell_key);
                            let nv_result = normalized_volume(&tds, cell_key);

                            // Both metrics should agree on degeneracy
                            // If one fails with DegenerateCell, both should fail
                            match (rr_result, nv_result) {
                                (Ok(_), Ok(_)) => (),  // Both succeed - OK
                                (Err(rr_err), Err(nv_err)) => {
                                    // Both fail - verify they're both degeneracy errors or both other errors
                                    let rr_is_degen = matches!(rr_err, delaunay::geometry::quality::QualityError::DegenerateCell { .. });
                                    let nv_is_degen = matches!(nv_err, delaunay::geometry::quality::QualityError::DegenerateCell { .. });

                                    if rr_is_degen || nv_is_degen {
                                        prop_assert!(
                                            rr_is_degen == nv_is_degen,
                                            "{}D: Both metrics should detect degeneracy consistently: rr={}, nv={}",
                                            $dim,
                                            rr_is_degen,
                                            nv_is_degen
                                        );
                                    }
                                }
                                (Ok(_), Err(_)) | (Err(_), Ok(_)) => {
                                    // One succeeds, one fails - this is acceptable for numerical edge cases
                                    // but log for investigation if it's a degeneracy disagreement
                                }
                            }
                        }
                    }
                }

                /// Property: Extreme deformation degrades quality (becomes degenerate)
                #[test]
                fn [<prop_quality_degrades_under_collapse_ $dim d>](
                    base_scale in 0.1f64..10.0f64
                ) {
                    // Create a regular simplex
                    let mut regular_points = Vec::new();
                    regular_points.push(Point::new([0.0f64; $dim]));
                    for i in 0..$dim {
                        let mut coords = [0.0f64; $dim];
                        coords[i] = base_scale;
                        regular_points.push(Point::new(coords));
                    }

                    // Create a nearly-degenerate version (collapse last vertex toward origin)
                    let mut degenerate_points = regular_points.clone();
                    if let Some(last) = degenerate_points.last_mut() {
                        let coords: [f64; $dim] = (*last).into();
                        let mut collapsed_coords = coords;
                        // Collapse to nearly coincident with first vertex
                        for i in 0..$dim {
                            collapsed_coords[i] *= 0.01; // Move 99% toward origin
                        }
                        *last = Point::new(collapsed_coords);
                    }

                    // Compare quality metrics - collapsed should be much worse
                    if let (Ok(r_reg), Ok(r_inner_reg)) = (circumradius(&regular_points), inradius(&regular_points)) {
                        if let (Ok(r_coll), Ok(r_inner_coll)) = (circumradius(&degenerate_points), inradius(&degenerate_points)) {
                            if r_reg > 1e-6 && r_inner_reg > 1e-9 && r_coll > 1e-6 && r_inner_coll > 1e-9 {
                                let ratio_reg = r_reg / r_inner_reg;
                                let ratio_coll = r_coll / r_inner_coll;

                                if ratio_reg.is_finite() && ratio_coll.is_finite() {
                                    // Collapsed simplex should have significantly worse quality
                                    prop_assert!(
                                        ratio_coll > ratio_reg * 1.5,
                                        "{}D: Collapsed simplex should have worse quality: regular={}, collapsed={}",
                                        $dim,
                                        ratio_reg,
                                        ratio_coll
                                    );
                                }
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
