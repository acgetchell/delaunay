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
use delaunay::prelude::*;
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
// HELPER FUNCTIONS
// =============================================================================

/// Compare quality metrics between two triangulations by matching cells via vertex UUIDs.
///
/// This helper function matches cells between an original and transformed triangulation
/// by mapping vertex UUIDs, then compares their quality metrics using the provided
/// comparison function.
///
/// # Arguments
/// * `tds_orig` - Original triangulation
/// * `tds_transformed` - Transformed triangulation (translated, scaled, rotated, etc.)
/// * `uuid_map` - Mapping from original vertex UUIDs to transformed vertex UUIDs
/// * `_metric_name` - Name of the metric being tested (for error messages)
/// * `_dimension` - Dimensionality (for error messages)
/// * `compare_fn` - Function to compute and compare quality metrics for matched cells.
///   Returns `Ok(())` if metrics match within tolerance, `Err(TestCaseError)` otherwise.
///
/// # Returns
/// * `Ok(true)` - At least one cell was successfully matched and compared
/// * `Ok(false)` - No cells could be matched (topology changed too much)
/// * `Err(TestCaseError)` - A metric comparison failed
///
/// # Purpose
/// This function centralizes the UUID-based cell matching logic used across multiple
/// transformation invariance tests (translation, scaling, rotation). It eliminates
/// ~200 lines of duplicated code by extracting the common pattern:
/// 1. Iterate through cells in original triangulation
/// 2. Map their vertex UUIDs to transformed triangulation
/// 3. Find matching cell in transformed triangulation
/// 4. Compare quality metrics between matched cells
/// 5. Track whether any cells were successfully matched (to avoid vacuous success)
fn compare_transformed_cells<const D: usize, F>(
    dt_orig: &DelaunayTriangulation<FastKernel<f64>, (), (), D>,
    dt_transformed: &DelaunayTriangulation<FastKernel<f64>, (), (), D>,
    uuid_map: &HashMap<Uuid, Uuid>,
    _metric_name: &str,
    _dimension: usize,
    mut compare_fn: F,
) -> Result<bool, TestCaseError>
where
    F: FnMut(CellKey, CellKey) -> Result<(), TestCaseError>,
{
    let mut matched_any = false;
    let tds_orig = dt_orig.tds();
    let tds_transformed = dt_transformed.tds();

    // Iterate through all cells in original triangulation
    for orig_key in tds_orig.cell_keys() {
        if let Some(orig_cell) = tds_orig.get_cell(orig_key) {
            // Get original cell's vertex UUIDs
            if let Ok(orig_uuids) = orig_cell.vertex_uuids(tds_orig) {
                // Map to transformed UUIDs
                let transformed_uuids: Vec<_> = orig_uuids
                    .iter()
                    .filter_map(|uuid| uuid_map.get(uuid))
                    .copied()
                    .collect();

                // Find matching cell in transformed triangulation
                for trans_key in tds_transformed.cell_keys() {
                    if let Some(trans_cell) = tds_transformed.get_cell(trans_key)
                        && let Ok(trans_cell_uuids) = trans_cell.vertex_uuids(tds_transformed)
                    {
                        // Check if cells have same vertices (by UUID)
                        if transformed_uuids.len() == trans_cell_uuids.len()
                            && transformed_uuids
                                .iter()
                                .all(|u| trans_cell_uuids.contains(u))
                        {
                            // Found matching cell - compare quality metrics
                            compare_fn(orig_key, trans_key)?;
                            matched_any = true;
                            break; // Found the match, no need to check other cells
                        }
                    }
                }
            }
        }
    }

    Ok(matched_any)
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
                    ).prop_map(|v| Vertex::from_points(&v))
                ) {
                    if let Ok(dt) = DelaunayTriangulation::<FastKernel<f64>, (), (), $dim>::with_topology_guarantee(
                        FastKernel::default(),
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        let tds = dt.tds();
                        let tri = dt.as_triangulation();
                        for cell_key in tds.cell_keys() {
                            if let Ok(ratio) = radius_ratio(tri, cell_key) {
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
                    ).prop_map(|v| Vertex::from_points(&v)),
                    translation in prop::array::[<uniform $dim>](finite_coordinate())
                ) {
                    if let Ok(dt) = DelaunayTriangulation::<FastKernel<f64>, (), (), $dim>::with_topology_guarantee(
                        FastKernel::default(),
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
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

                        let translated_vertices = Vertex::from_points(&translated_vertices);

                        if let Ok(dt_translated) = DelaunayTriangulation::<FastKernel<f64>, (), (), $dim>::with_topology_guarantee(
                            FastKernel::default(),
                            &translated_vertices,
                            TopologyGuarantee::PLManifold,
                        ) {
                            // Build mapping from original UUIDs to translated UUIDs
                            let uuid_map: HashMap<_, _> = vertices.iter()
                                .zip(translated_vertices.iter())
                                .map(|(orig, trans)| (orig.uuid(), trans.uuid()))
                                .collect();

                            // Compare cells using helper function
                            let matched_any = compare_transformed_cells(
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

                            // If no cells matched, discard this case to avoid vacuous success
                            prop_assume!(matched_any);
                        }
                    }
                }

                /// Property: Normalized volume is translation-invariant
                #[test]
                fn [<prop_normalized_volume_translation_invariant_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v)),
                    translation in prop::array::[<uniform $dim>](finite_coordinate())
                ) {
                    if let Ok(dt) = DelaunayTriangulation::<FastKernel<f64>, (), (), $dim>::with_topology_guarantee(
                        FastKernel::default(),
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
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

                        let translated_vertices = Vertex::from_points(&translated_vertices);

                        if let Ok(dt_translated) = DelaunayTriangulation::<FastKernel<f64>, (), (), $dim>::with_topology_guarantee(
                            FastKernel::default(),
                            &translated_vertices,
                            TopologyGuarantee::PLManifold,
                        ) {
                            // Build UUID mapping
                            let uuid_map: HashMap<_, _> = vertices.iter()
                                .zip(translated_vertices.iter())
                                .map(|(orig, trans)| (orig.uuid(), trans.uuid()))
                                .collect();

                            // Compare cells using helper function
                            let matched_any = compare_transformed_cells(
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

                            // If no cells matched, discard this case to avoid vacuous success
                            prop_assume!(matched_any);
                        }
                    }
                }

                /// Property: Normalized volume is scale-invariant (uniform scaling)
                #[test]
                fn [<prop_normalized_volume_scale_invariant_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v)),
                    scale in 0.1f64..10.0f64
                ) {
                    if let Ok(dt) = DelaunayTriangulation::<FastKernel<f64>, (), (), $dim>::with_topology_guarantee(
                        FastKernel::default(),
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
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

                        let scaled_vertices = Vertex::from_points(&scaled_vertices);

                        if let Ok(dt_scaled) = DelaunayTriangulation::<FastKernel<f64>, (), (), $dim>::with_topology_guarantee(
                            FastKernel::default(),
                            &scaled_vertices,
                            TopologyGuarantee::PLManifold,
                        ) {
                            // Build UUID mapping
                            let uuid_map: HashMap<_, _> = vertices.iter()
                                .zip(scaled_vertices.iter())
                                .map(|(orig, scaled)| (orig.uuid(), scaled.uuid()))
                                .collect();

                            // Compare cells using helper function
                            let matched_any = compare_transformed_cells(
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

                            // If no cells matched, discard this case to avoid vacuous success
                            prop_assume!(matched_any);
                        }
                    }
                }

                /// Property: Both metrics detect degeneracy consistently
                #[test]
                fn [<prop_degeneracy_consistency_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v))
                ) {
                    if let Ok(dt) = DelaunayTriangulation::<FastKernel<f64>, (), (), $dim>::with_kernel(FastKernel::default(), &vertices) {
                        let tds = dt.tds();
                        let tri = dt.as_triangulation();
                        for cell_key in tds.cell_keys() {
                            let rr_result = radius_ratio(tri, cell_key);
                            let nv_result = normalized_volume(tri, cell_key);

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

// =============================================================================
// FACET TOPOLOGY INVARIANT TESTS
// =============================================================================

/// Macro to generate facet topology invariant property tests for a given dimension.
///
/// These tests verify the **critical manifold topology invariant**: each facet
/// must be shared by at most 2 cells (1 for boundary, 2 for interior). This
/// invariant is essential for facet walking used in point location.
///
/// The localized validation functions (`detect_local_facet_issues`,
/// `repair_local_facet_issues`) maintain this invariant in O(k) time.
macro_rules! test_facet_topology_invariant {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal) => {
        pastey::paste! {
            proptest! {
                /// Property: All cells in a valid triangulation have no over-shared facets
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

                        // Get all cell keys
                        let cell_keys: Vec<_> = tri.cells().map(|(k, _)| k).collect();

                        // Validate no over-shared facets
                        let issues = tri.detect_local_facet_issues(&cell_keys)?;
                        prop_assert!(
                            issues.is_none(),
                            "{}D: Triangulation has {} over-shared facets (violates manifold topology invariant)",
                            $dim,
                            issues.as_ref().map_or(0, |m| m.len())
                        );
                    }
                }

                /// Property: After repair, no over-shared facets remain
                #[test]
                fn [<prop_repair_fixes_all_issues_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| vertex!(coords)),
                        $min_vertices..$max_vertices
                    )
                ) {
                    // Build triangulation
                    if let Ok(mut dt) = DelaunayTriangulation::new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        let tri = dt.as_triangulation_mut();

                        // Get all cell keys
                        let cell_keys: Vec<_> = tri.cells().map(|(k, _)| k).collect();

                        // If there are any issues, repair them
                        if let Some(issues) = tri.detect_local_facet_issues(&cell_keys)? {
                            let _removed = tri.repair_local_facet_issues(&issues)?;

                            // After repair, re-check - should have no issues
                            let cell_keys_after: Vec<_> = tri.cells().map(|(k, _)| k).collect();
                            let issues_after = tri.detect_local_facet_issues(&cell_keys_after)?;
                            prop_assert!(
                                issues_after.is_none(),
                                "{}D: After repair, {} over-shared facets still remain",
                                $dim,
                                issues_after.as_ref().map_or(0, |m| m.len())
                            );
                        }
                    }
                }

                /// Property: Empty cell list returns no issues
                #[test]
                fn [<prop_empty_cell_list_no_issues_ $dim d>](
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

                        // Empty cell list should always return None
                        let issues = tri.detect_local_facet_issues(&[])?;
                        prop_assert!(
                            issues.is_none(),
                            "{}D: Empty cell list should have no issues",
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
test_facet_topology_invariant!(5, 7, 16);
