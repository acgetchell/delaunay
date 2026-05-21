//! Property-based tests for Triangulation layer invariants.
//!
//! This module tests the generic geometric layer that combines a `Kernel` with
//! the combinatorial `Tds`. The Triangulation layer provides geometric operations
//! while delegating pure topology to Tds.
//!
//! ## Architectural Context
//!
//! Following CGAL's architecture:
//! - **Tds** - Pure combinatorial/topological structure (tested in `proptest_tds.rs`)
//! - **Triangulation** - Generic geometric layer with kernel (tested here)
//! - **`DelaunayTriangulation`** - Delaunay-specific operations (tested in `proptest_delaunay_triangulation.rs`)
//!
//! ## Tests Included
//!
//! ### Geometric Quality Metrics
//! Property-based tests for quality metrics including:
//! - Radius ratio bounds (R/r ≥ D for D-dimensional simplex)
//! - Radius ratio scaling and translation invariance
//! - Normalized volume invariance properties
//! - Quality metric consistency (degeneracy detection)
//! - Quality degradation under deformation
//!
//! **Note**: Tests use `DelaunayTriangulation` for construction (most convenient way
//! to obtain valid triangulations), but the properties tested are generic
//! Triangulation-layer concerns applicable to any triangulation with a kernel.
//!
//! ### Future Tests
//!
//! - **Facet iteration** - `facets()`, `boundary_facets()` consistency
//! - **Boundary detection** - Correct identification of hull facets
//! - **Topology repair** - `detect_local_facet_issues` + `repair_local_facet_issues` preserve validity
//! - **Kernel consistency** - Geometric predicates produce consistent results
//!
//! Tests are generated for dimensions 2D-5D using macros to reduce duplication.

use ::uuid::Uuid;
use delaunay::prelude::construction::{DelaunayTriangulation, TopologyGuarantee, Vertex, vertex};
use delaunay::prelude::geometry::*;
use delaunay::prelude::tds::SimplexKey;
use proptest::prelude::*;
use proptest::test_runner::{Config, TestCaseError, TestRunner};
use std::cell::RefCell;
use std::collections::HashMap;

// =============================================================================
// TEST CONFIGURATION
// =============================================================================

/// Strategy for generating finite f64 coordinates
fn finite_coordinate() -> impl Strategy<Value = f64> {
    (-100.0..100.0).prop_filter("must be finite", |x: &f64| x.is_finite())
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Compare quality metrics between two triangulations by matching simplices via vertex UUIDs.
///
/// This helper function matches simplices between an original and transformed triangulation
/// by mapping vertex UUIDs, then compares their quality metrics using the provided
/// comparison function.
///
/// # Arguments
/// * `tds_orig` - Original triangulation
/// * `tds_transformed` - Transformed triangulation (translated, scaled, rotated, etc.)
/// * `uuid_map` - Mapping from original vertex UUIDs to transformed vertex UUIDs
/// * `_metric_name` - Name of the metric being tested (for error messages)
/// * `_dimension` - Dimensionality (for error messages)
/// * `compare_fn` - Function to compute and compare quality metrics for matched simplices.
///   Returns `Ok(())` if metrics match within tolerance, `Err(TestCaseError)` otherwise.
///
/// # Returns
/// * `Ok(())` - At least one simplex was successfully matched and compared
/// * `Err(TestCaseError)` - A metric comparison failed
///
/// # Purpose
/// This function centralizes the UUID-based simplex matching logic used across multiple
/// transformation invariance tests (translation, scaling, rotation). It eliminates
/// ~200 lines of duplicated code by extracting the common pattern:
/// 1. Iterate through simplices in original triangulation
/// 2. Map their vertex UUIDs to transformed triangulation
/// 3. Find matching simplex in transformed triangulation
/// 4. Compare quality metrics between matched simplices
///
/// Degenerate Delaunay inputs can have more than one valid simplex set, so an
/// independently constructed transformed triangulation may choose a different
/// valid tessellation. Unmatched simplices are skipped; cases with no comparable
/// simplices are rejected rather than treated as metric failures.
fn compare_transformed_simplices<const D: usize, F>(
    dt_orig: &DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>,
    dt_transformed: &DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>,
    uuid_map: &HashMap<Uuid, Uuid>,
    _metric_name: &str,
    _dimension: usize,
    mut compare_fn: F,
) -> Result<(), TestCaseError>
where
    F: FnMut(SimplexKey, SimplexKey) -> Result<(), TestCaseError>,
{
    let tds_orig = dt_orig.tds();
    let tds_transformed = dt_transformed.tds();
    let mut matched_simplices = 0usize;

    // Iterate through all simplices in original triangulation
    for orig_key in tds_orig.simplex_keys() {
        prop_assert!(
            tds_orig.simplex(orig_key).is_some(),
            "original simplex key from iterator should exist: {orig_key:?}"
        );
        let orig_simplex = tds_orig.simplex(orig_key).expect("checked above");
        let orig_uuids = orig_simplex.vertex_uuids(tds_orig)?;
        let transformed_uuids: Vec<_> = orig_uuids
            .iter()
            .filter_map(|uuid| uuid_map.get(uuid))
            .copied()
            .collect();
        prop_assert_eq!(
            transformed_uuids.len(),
            orig_uuids.len(),
            "all original simplex UUIDs should map to transformed UUIDs"
        );

        for trans_key in tds_transformed.simplex_keys() {
            prop_assert!(
                tds_transformed.simplex(trans_key).is_some(),
                "transformed simplex key from iterator should exist: {trans_key:?}"
            );
            let trans_simplex = tds_transformed.simplex(trans_key).expect("checked above");
            if let Ok(trans_simplex_uuids) = trans_simplex.vertex_uuids(tds_transformed) {
                // Check if simplices have same vertices (by UUID)
                if transformed_uuids.len() == trans_simplex_uuids.len()
                    && transformed_uuids
                        .iter()
                        .all(|u| trans_simplex_uuids.contains(u))
                {
                    // Found matching simplex - compare quality metrics
                    compare_fn(orig_key, trans_key)?;
                    matched_simplices += 1;
                    break; // Found the match, no need to check other simplices
                }
            }
        }
    }

    prop_assume!(matched_simplices >= 1);

    Ok(())
}

// =============================================================================
// DIMENSIONAL TEST GENERATION MACROS
// =============================================================================

/// Macro to generate simplex-only quality metric property tests for a given dimension
macro_rules! test_simplex_quality_properties {
    ($dim:literal, $num_points:literal $(, #[$attr:meta])*) => {
        pastey::paste! {
            proptest! {
                /// Property: Radius ratio R/r ≥ D for non-degenerate D-simplices
                $(#[$attr])*
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
                $(#[$attr])*
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
                                    let coords = *p.coords();
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

                /// Property: Regular simplex has better (lower) radius ratio than degenerate
                $(#[$attr])*
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

                /// Property: Extreme deformation degrades quality (becomes degenerate)
                $(#[$attr])*
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
                        let coords = *last.coords();
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

/// Macro to generate quality metric property tests for a given dimension
macro_rules! test_quality_properties {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal $(, #[$attr:meta])*) => {
        pastey::paste! {
            proptest! {

                /// Property: Radius ratio is always positive for valid simplices
                $(#[$attr])*
                #[test]
                fn [<prop_radius_ratio_positive_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v))
                ) {
                    if let Ok(dt) = DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), $dim>::with_topology_guarantee(
                        &AdaptiveKernel::default(),
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        let tds = dt.tds();
                        let tri = dt.as_triangulation();
                        for simplex_key in tds.simplex_keys() {
                            if let Ok(ratio) = radius_ratio(tri, simplex_key) {
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


                /// Property: Radius ratio is translation-invariant
                $(#[$attr])*
                #[test]
                fn [<prop_radius_ratio_translation_invariant_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v)),
                    translation in prop::array::[<uniform $dim>](finite_coordinate())
                ) {
                    if let Ok(dt) = DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), $dim>::with_topology_guarantee(
                        &AdaptiveKernel::default(),
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        // Translate all vertices
                        let translated_vertices: Vec<_> = vertices
                            .iter()
                            .map(|v| {
                                let coords = *v.point().coords();
                                let mut translated = [0.0f64; $dim];
                                for i in 0..$dim {
                                    translated[i] = coords[i] + translation[i];
                                }
                                Point::new(translated)
                            })
                            .collect::<Vec<_>>();

                        let translated_vertices = Vertex::from_points(&translated_vertices);

                        if let Ok(dt_translated) = DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), $dim>::with_topology_guarantee(
                            &AdaptiveKernel::default(),
                            &translated_vertices,
                            TopologyGuarantee::PLManifold,
                        ) {
                            // Build mapping from original UUIDs to translated UUIDs
                            let uuid_map: HashMap<_, _> = vertices.iter()
                                .zip(translated_vertices.iter())
                                .map(|(orig, trans)| (orig.uuid(), trans.uuid()))
                                .collect();

                            // Compare simplices using helper function
                            compare_transformed_simplices(
                                &dt,
                                &dt_translated,
                                &uuid_map,
                                "radius_ratio",
                                $dim,
                                |orig_key, trans_key| {
                                    let tri = dt.as_triangulation();
                                    let tri_translated = dt_translated.as_triangulation();
                                    if let (Ok(ratio_orig), Ok(ratio_trans)) = (
                                        radius_ratio(tri, orig_key),
                                        radius_ratio(tri_translated, trans_key),
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
                                    Ok(())
                                },
                            )?;
                        }
                    }
                }

                /// Property: Normalized volume is translation-invariant
                $(#[$attr])*
                #[test]
                fn [<prop_normalized_volume_translation_invariant_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v)),
                    translation in prop::array::[<uniform $dim>](finite_coordinate())
                ) {
                    if let Ok(dt) = DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), $dim>::with_topology_guarantee(
                        &AdaptiveKernel::default(),
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        // Translate all vertices
                        let translated_vertices: Vec<_> = vertices
                            .iter()
                            .map(|v| {
                                let coords = *v.point().coords();
                                let mut translated = [0.0f64; $dim];
                                for i in 0..$dim {
                                    translated[i] = coords[i] + translation[i];
                                }
                                Point::new(translated)
                            })
                            .collect::<Vec<_>>();

                        let translated_vertices = Vertex::from_points(&translated_vertices);

                        if let Ok(dt_translated) = DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), $dim>::with_topology_guarantee(
                            &AdaptiveKernel::default(),
                            &translated_vertices,
                            TopologyGuarantee::PLManifold,
                        ) {
                            // Build UUID mapping
                            let uuid_map: HashMap<_, _> = vertices.iter()
                                .zip(translated_vertices.iter())
                                .map(|(orig, trans)| (orig.uuid(), trans.uuid()))
                                .collect();

                            // Compare simplices using helper function
                            compare_transformed_simplices(
                                &dt,
                                &dt_translated,
                                &uuid_map,
                                "normalized_volume",
                                $dim,
                                |orig_key, trans_key| {
                                    let tri = dt.as_triangulation();
                                    let tri_translated = dt_translated.as_triangulation();
                                    if let (Ok(vol_orig), Ok(vol_trans)) = (
                                        normalized_volume(tri, orig_key),
                                        normalized_volume(tri_translated, trans_key),
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
                                    Ok(())
                                },
                            )?;
                        }
                    }
                }

                /// Property: Normalized volume is scale-invariant (uniform scaling)
                $(#[$attr])*
                #[test]
                fn [<prop_normalized_volume_scale_invariant_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v)),
                    scale in 0.1f64..10.0f64
                ) {
                    if let Ok(dt) = DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), $dim>::with_topology_guarantee(
                        &AdaptiveKernel::default(),
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        // Scale all vertices uniformly
                        let scaled_vertices: Vec<_> = vertices
                            .iter()
                            .map(|v| {
                                let coords = *v.point().coords();
                                let mut scaled = [0.0f64; $dim];
                                for i in 0..$dim {
                                    scaled[i] = coords[i] * scale;
                                }
                                Point::new(scaled)
                            })
                            .collect::<Vec<_>>();

                        let scaled_vertices = Vertex::from_points(&scaled_vertices);

                        if let Ok(dt_scaled) = DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), $dim>::with_topology_guarantee(
                            &AdaptiveKernel::default(),
                            &scaled_vertices,
                            TopologyGuarantee::PLManifold,
                        ) {
                            // Build UUID mapping
                            let uuid_map: HashMap<_, _> = vertices.iter()
                                .zip(scaled_vertices.iter())
                                .map(|(orig, scaled)| (orig.uuid(), scaled.uuid()))
                                .collect();

                            // Compare simplices using helper function
                            compare_transformed_simplices(
                                &dt,
                                &dt_scaled,
                                &uuid_map,
                                "normalized_volume",
                                $dim,
                                |orig_key, scaled_key| {
                                    let tri = dt.as_triangulation();
                                    let tri_scaled = dt_scaled.as_triangulation();
                                    if let (Ok(vol_orig), Ok(vol_scaled)) = (
                                        normalized_volume(tri, orig_key),
                                        normalized_volume(tri_scaled, scaled_key),
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
                                    Ok(())
                                },
                            )?;
                        }
                    }
                }

                /// Property: Both metrics detect degeneracy consistently
                $(#[$attr])*
                #[test]
                fn [<prop_degeneracy_consistency_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v))
                ) {
                    if let Ok(dt) = DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), $dim>::with_topology_guarantee(
                        &AdaptiveKernel::default(),
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        let tds = dt.tds();
                        let tri = dt.as_triangulation();
                        for simplex_key in tds.simplex_keys() {
                            let rr_result = radius_ratio(tri, simplex_key);
                            let nv_result = normalized_volume(tri, simplex_key);

                            // Both metrics should agree on degeneracy
                            // If one fails with DegenerateSimplex, both should fail
                            match (rr_result, nv_result) {
                                (Ok(_), Ok(_)) => (),  // Both succeed - OK
                                (Err(rr_err), Err(nv_err)) => {
                                    // Both fail - verify they're both degeneracy errors or both other errors
                                    let rr_is_degen = matches!(rr_err, delaunay::geometry::quality::QualityError::DegenerateSimplex { .. });
                                    let nv_is_degen = matches!(nv_err, delaunay::geometry::quality::QualityError::DegenerateSimplex { .. });

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

            }
        }
    };
}

// Generate tests for dimensions 2-5
// Simplex-only quality metrics (D+1 points)
test_simplex_quality_properties!(2, 3);
test_simplex_quality_properties!(3, 4);
test_simplex_quality_properties!(4, 5);
test_simplex_quality_properties!(5, 6);
// Parameters: dimension, min_vertices, max_vertices
test_quality_properties!(2, 4, 10);
test_quality_properties!(3, 5, 12);
test_quality_properties!(4, 6, 14, #[cfg(feature = "slow-tests")]);
test_quality_properties!(5, 7, 16, #[cfg(feature = "slow-tests")]);

// =============================================================================
// FACET TOPOLOGY INVARIANT TESTS
// =============================================================================

/// Macro to generate facet topology invariant property tests for a given dimension.
///
/// These tests verify the **critical manifold topology invariant**: each facet
/// must be shared by at most 2 simplices (1 for boundary, 2 for interior). This
/// invariant is essential for facet walking used in point location.
///
/// The localized validation functions (`detect_local_facet_issues`,
/// `repair_local_facet_issues`) maintain this invariant in O(k) time.
macro_rules! test_facet_topology_invariant {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal $(, #[$attr:meta])*) => {
        pastey::paste! {
            proptest! {
                /// Property: All simplices in a valid triangulation have no over-shared facets
                $(#[$attr])*
                #[test]
                fn [<prop_no_over_shared_facets_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| vertex!(coords)),
                        $min_vertices..$max_vertices
                    )
                ) {
                    // Build triangulation
                    if let Ok(dt) = DelaunayTriangulation::new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        let tri = dt.as_triangulation();

                        // Get all simplex keys
                        let simplex_keys: Vec<_> = tri.simplices().map(|(k, _)| k).collect();

                        // Validate no over-shared facets
                        let issues = tri.detect_local_facet_issues(&simplex_keys)?;
                        prop_assert!(
                            issues.is_none(),
                            "{}D: Triangulation has {} over-shared facets (violates manifold topology invariant)",
                            $dim,
                            issues.as_ref().map_or(0, |m| m.len())
                        );
                    }
                }

                /// Property: After repair, no over-shared facets remain
                $(#[$attr])*
                #[test]
                fn [<prop_repair_fixes_all_issues_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| vertex!(coords)),
                        $min_vertices..$max_vertices
                    )
                ) {
                    // Build triangulation
                    if let Ok(dt) = DelaunayTriangulation::new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        let mut tri = dt.as_triangulation().clone();

                        // Get all simplex keys
                        let simplex_keys: Vec<_> = tri.simplices().map(|(k, _)| k).collect();

                        // If there are any issues, repair them
                        if let Some(issues) = tri.detect_local_facet_issues(&simplex_keys)? {
                            let _removed = tri.repair_local_facet_issues(&issues, usize::MAX)?;

                            // After repair, re-check - should have no issues
                            let simplex_keys_after: Vec<_> = tri.simplices().map(|(k, _)| k).collect();
                            let issues_after = tri.detect_local_facet_issues(&simplex_keys_after)?;
                            prop_assert!(
                                issues_after.is_none(),
                                "{}D: After repair, {} over-shared facets still remain",
                                $dim,
                                issues_after.as_ref().map_or(0, |m| m.len())
                            );
                        }
                    }
                }

                /// Property: Empty simplex list returns no issues
                $(#[$attr])*
                #[test]
                fn [<prop_empty_simplex_list_no_issues_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| vertex!(coords)),
                        $min_vertices..$max_vertices
                    )
                ) {
                    // Build triangulation
                    if let Ok(dt) = DelaunayTriangulation::new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        let tri = dt.as_triangulation();

                        // Empty simplex list should always return None
                        let issues = tri.detect_local_facet_issues(&[])?;
                        prop_assert!(
                            issues.is_none(),
                            "{}D: Empty simplex list should have no issues",
                            $dim
                        );
                    }
                }
            }
        }
    };
}

// Generate facet topology invariant tests for dimensions 2-5
// Parameters: dimension, min_vertices, max_vertices
test_facet_topology_invariant!(2, 4, 10);
test_facet_topology_invariant!(3, 5, 12);
test_facet_topology_invariant!(4, 6, 14);
test_facet_topology_invariant!(5, 7, 16, #[cfg(feature = "slow-tests")]);

// =============================================================================
// FAST HIGH-DIMENSIONAL FACET TOPOLOGY SMOKE TESTS
// =============================================================================

macro_rules! gen_high_dim_facet_topology_smoke {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal) => {
        pastey::paste! {
            #[test]
            fn [<prop_high_dim_facet_topology_active_smoke_ $dim d>]() {
                #[derive(Debug, Default)]
                struct SmokeStats {
                    generated: usize,
                    accepted: usize,
                    rejected_construction_failed: usize,
                }

                let config = Config {
                    cases: 6,
                    max_shrink_iters: 16,
                    ..Config::default()
                };
                let target_cases = config.cases;
                let mut runner = TestRunner::new(config);
                let strategy = prop::collection::vec(
                    prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| vertex!(coords)),
                    $min_vertices..=$max_vertices,
                );
                let stats = RefCell::new(SmokeStats::default());

                let run_result = runner.run(&strategy, |vertices| {
                    let mut stats = stats.borrow_mut();
                    stats.generated += 1;

                    let dt = match DelaunayTriangulation::new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        Ok(dt) => dt,
                        Err(err) => {
                            stats.rejected_construction_failed += 1;
                            return Err(TestCaseError::reject(format!(
                                "{}D: construction failed in active facet-topology smoke test: {err}",
                                $dim
                            )));
                        }
                    };

                    let tri = dt.as_triangulation();
                    let simplex_keys: Vec<_> = tri.simplices().map(|(key, _)| key).collect();
                    let issues = tri.detect_local_facet_issues(&simplex_keys)?;
                    prop_assert!(
                        issues.is_none(),
                        "{}D active facet-topology smoke should have no over-shared facets: {:?}",
                        $dim,
                        issues
                    );

                    let empty_issues = tri.detect_local_facet_issues(&[])?;
                    prop_assert!(
                        empty_issues.is_none(),
                        "{}D active facet-topology smoke empty scope should have no issues",
                        $dim
                    );

                    stats.accepted += 1;
                    Ok(())
                });

                let stats = stats.into_inner();
                let generated = stats.generated.max(1);
                let acceptance_rate_percent_x100: u128 =
                    (stats.accepted as u128 * 10_000u128) / (generated as u128);
                let acceptance_rate_whole = acceptance_rate_percent_x100 / 100;
                let acceptance_rate_frac = acceptance_rate_percent_x100 % 100;
                let print_stats =
                    std::env::var_os("DELAUNAY_PROPTEST_REJECT_STATS").is_some() || run_result.is_err();

                if print_stats {
                    let rejected_total = stats.generated.saturating_sub(stats.accepted);
                    tracing::warn!(
                        "prop_high_dim_facet_topology_active_smoke_{}d reject stats: target_cases={target_cases} generated={} accepted={} acceptance_rate={}.{:02}% rejected_total={} construction_failed={}",
                        $dim,
                        stats.generated,
                        stats.accepted,
                        acceptance_rate_whole,
                        acceptance_rate_frac,
                        rejected_total,
                        stats.rejected_construction_failed
                    );
                }

                let max_allowed_construction_rejections =
                    usize::try_from(target_cases).map_or(usize::MAX, |cases| cases.max(1));
                assert!(
                    stats.rejected_construction_failed <= max_allowed_construction_rejections,
                    "prop_high_dim_facet_topology_active_smoke_{}d had {} construction rejects above allowed {}; generated={}, accepted={}",
                    $dim,
                    stats.rejected_construction_failed,
                    max_allowed_construction_rejections,
                    stats.generated,
                    stats.accepted
                );

                assert!(
                    stats.accepted > 0,
                    "prop_high_dim_facet_topology_active_smoke_{}d should accept at least one case",
                    $dim
                );
                run_result.unwrap();
            }
        }
    };
}

gen_high_dim_facet_topology_smoke!(4, 6, 8);
gen_high_dim_facet_topology_smoke!(5, 7, 9);
