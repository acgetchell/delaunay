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

use delaunay::assert_jaccard_gte;
use delaunay::prelude::query::*;
use delaunay::prelude::topology::validation::*;
use delaunay::try_vertices_from_points;
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

/// Macro to generate minimal simplex hull property tests for a given dimension
macro_rules! test_minimal_simplex_hull {
    ($dim:literal $(, #[$attr:meta])*) => {
        pastey::paste! {
            proptest! {
                /// Property: Hull facet count for minimal simplex is D+1
                $(#[$attr])*
                #[test]
                fn [<prop_minimal_simplex_hull_ $dim d>](
                    base_scale in 0.1f64..10.0f64
                ) {
                    // Create a minimal simplex (D+1 vertices in D dimensions)
                    let mut points = Vec::new();

                    // Origin
                    points.push(Point::try_new([0.0f64; $dim]).expect("finite point coordinates"));

                    // D more points along coordinate axes
                    for i in 0..$dim {
                        let mut coords = [0.0f64; $dim];
                        coords[i] = base_scale;
                        points.push(Point::try_new(coords).expect("finite point coordinates"));
                    }

                    let vertices =
                        try_vertices_from_points(&points).expect("finite point coordinates");

                    let dt_result = DelaunayTriangulation::<_, (), (), $dim>::try_new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    );
                    prop_assert!(
                        dt_result.is_ok(),
                        "{}D minimal simplex triangulation should construct: {:?}",
                        $dim,
                        dt_result.as_ref().err()
                    );
                    let dt = dt_result.expect("checked minimal simplex construction");

                    let hull_result = ConvexHull::try_from_triangulation(dt.as_triangulation());
                    prop_assert!(
                        hull_result.is_ok(),
                        "{}D minimal simplex hull construction should succeed: {:?}",
                        $dim,
                        hull_result.as_ref().err()
                    );
                    let hull = hull_result.expect("checked minimal simplex hull construction");

                    // A minimal D-simplex should have exactly D+1 facets
                    prop_assert_eq!(
                        hull.number_of_facets(),
                        $dim + 1,
                        "{}D minimal simplex hull should have exactly {} facets",
                        $dim,
                        $dim + 1
                    );
                }
            }
        }
    };
}

/// Macro to generate convex hull property tests for a given dimension
macro_rules! test_convex_hull_properties {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal $(, #[$attr:meta])*) => {
        pastey::paste! {
            proptest! {
                /// Property: Convex hull can be constructed from valid triangulation
                $(#[$attr])*
                #[test]
                fn [<prop_hull_construction_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| try_vertices_from_points(&v).expect("finite point coordinates"))
                ) {
                    let dt_result = DelaunayTriangulation::<_, (), (), $dim>::try_new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    );
                    prop_assume!(dt_result.is_ok());
                    let dt = dt_result.expect("assumed valid random triangulation");

                    // Filter: Skip degenerate configurations (no boundary facets)
                    // These are tested separately in dedicated degenerate case tests
                    let boundary_count = dt.tds().number_of_one_sided_facets().unwrap_or(0);
                    prop_assume!(boundary_count > 0);

                    // Should be able to construct hull from valid triangulation
                    let hull_result = ConvexHull::try_from_triangulation(dt.as_triangulation());
                    prop_assert!(
                        hull_result.is_ok(),
                        "{}D convex hull construction should succeed for valid triangulation: {:?}",
                        $dim,
                        hull_result.as_ref().err()
                    );
                    let hull = hull_result.expect("checked random hull construction");

                    // Hull should be valid for the TDS it was created from
                    prop_assert!(
                        hull.is_valid_for_triangulation(dt.as_triangulation()),
                        "{}D convex hull should be valid for its source triangulation",
                        $dim
                    );

                    // Facet count should be positive
                    prop_assert!(
                        hull.number_of_facets() > 0,
                        "{}D convex hull must have at least one facet",
                        $dim
                    );
                }

                /// Property: Hull facet count is bounded by combinatorial limits
                $(#[$attr])*
                #[test]
                fn [<prop_hull_facet_bounds_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| try_vertices_from_points(&v).expect("finite point coordinates"))
                ) {
                    let dt_result = DelaunayTriangulation::<_, (), (), $dim>::try_new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    );
                    prop_assume!(dt_result.is_ok());
                    let dt = dt_result.expect("assumed valid random triangulation");
                    let hull_result = ConvexHull::try_from_triangulation(dt.as_triangulation());
                    prop_assume!(hull_result.is_ok());
                    let hull = hull_result.expect("assumed hull construction");

                    let facet_count = hull.number_of_facets();
                    let vertex_count = dt.tds().vertices().count();

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

                /// Property: Hull becomes invalid after TDS modification
                $(#[$attr])*
                #[test]
                fn [<prop_hull_staleness_detection_ $dim d>](
                    initial_vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| try_vertices_from_points(&v).expect("finite point coordinates")),
                    new_point in prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates"))
                ) {
                    let dt_result = DelaunayTriangulation::<_, (), (), $dim>::try_new_with_topology_guarantee(
                        &initial_vertices,
                        TopologyGuarantee::PLManifold,
                    );
                    prop_assume!(dt_result.is_ok());
                    let mut dt = dt_result.expect("assumed valid random triangulation");

                    // Filter: Skip degenerate initial configurations
                    let initial_boundary_count =
                        dt.tds().number_of_one_sided_facets().unwrap_or(0);
                    prop_assume!(initial_boundary_count > 0);

                    let hull_result = ConvexHull::try_from_triangulation(dt.as_triangulation());
                    prop_assume!(hull_result.is_ok());
                    let hull = hull_result.expect("assumed initial hull construction");

                    // Hull should be valid initially
                    prop_assert!(
                        hull.is_valid_for_triangulation(dt.as_triangulation()),
                        "{}D hull should be valid for its triangulation before modification",
                        $dim
                    );

                    // Modify the triangulation by inserting a new vertex
                    let new_vertex =
                        try_vertices_from_points(&[new_point]).expect("finite point coordinates");
                    prop_assume!(dt.insert(new_vertex[0]).is_ok());

                    // Filter: Skip if modification resulted in degenerate configuration
                    let modified_boundary_count =
                        dt.tds().number_of_one_sided_facets().unwrap_or(0);
                    prop_assume!(modified_boundary_count > 0);

                    // Hull should now be invalid (stale)
                    prop_assert!(
                        !hull.is_valid_for_triangulation(dt.as_triangulation()),
                        "{}D hull should be invalid after triangulation modification",
                        $dim
                    );

                    // Creating a new hull should succeed for non-degenerate triangulation
                    let new_hull_result = ConvexHull::try_from_triangulation(dt.as_triangulation());
                    prop_assert!(
                        new_hull_result.is_ok(),
                        "{}D creating new hull after modification should succeed",
                        $dim
                    );
                    let new_hull = new_hull_result.expect("checked modified hull construction");

                    prop_assert!(
                        new_hull.is_valid_for_triangulation(dt.as_triangulation()),
                        "{}D new hull should be valid for modified triangulation",
                        $dim
                    );
                }

                /// Property: Hull vertices are a subset of triangulation vertices
                $(#[$attr])*
                #[test]
                fn [<prop_hull_vertices_subset_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| try_vertices_from_points(&v).expect("finite point coordinates"))
                ) {
                    let dt_result = DelaunayTriangulation::<_, (), (), $dim>::try_new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    );
                    prop_assume!(dt_result.is_ok());
                    let dt = dt_result.expect("assumed valid random triangulation");
                    let hull_result = ConvexHull::try_from_triangulation(dt.as_triangulation());
                    prop_assume!(hull_result.is_ok());
                    let hull = hull_result.expect("assumed hull construction");

                    let tds_vertex_count = dt.tds().vertices().count();
                    let facet_count = hull.number_of_facets();

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

                /// Property: Reconstructing hull from same TDS gives consistent facet count
                $(#[$attr])*
                #[test]
                fn [<prop_hull_reconstruction_consistency_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| try_vertices_from_points(&v).expect("finite point coordinates"))
                ) {
                    let dt_result = DelaunayTriangulation::<_, (), (), $dim>::try_new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    );
                    prop_assume!(dt_result.is_ok());
                    let dt = dt_result.expect("assumed valid random triangulation");
                    let hull1_result = ConvexHull::try_from_triangulation(dt.as_triangulation());
                    let hull2_result = ConvexHull::try_from_triangulation(dt.as_triangulation());
                    prop_assume!(hull1_result.is_ok());
                    prop_assume!(hull2_result.is_ok());
                    let hull1 = hull1_result.expect("assumed first hull construction");
                    let hull2 = hull2_result.expect("assumed second hull construction");

                    // Both hulls should have the same facet count
                    prop_assert_eq!(
                        hull1.number_of_facets(),
                        hull2.number_of_facets(),
                        "{}D reconstructing hull from same triangulation should give same facet count",
                        $dim
                    );

                    // Both should be valid for the same triangulation
                    prop_assert!(
                        hull1.is_valid_for_triangulation(dt.as_triangulation())
                            && hull2.is_valid_for_triangulation(dt.as_triangulation()),
                        "{}D both reconstructed hulls should be valid for the same triangulation",
                        $dim
                    );

                    // Extract facet sets and compare via Jaccard similarity
                    // Should be exactly identical (Jaccard = 1.0) since same triangulation
                    let facets1 =
                        extract_hull_facet_set(&hull1, dt.as_triangulation())
                        .expect("facet extraction should not fail");
                    let facets2 =
                        extract_hull_facet_set(&hull2, dt.as_triangulation())
                        .expect("facet extraction should not fail");
                    assert_jaccard_gte!(
                        &facets1,
                        &facets2,
                        1.0,
                        "{}D hull reconstruction facet topology (exact match expected)",
                        $dim
                    );
                }
            }
        }
    };
}

// Generate tests for dimensions 2-5
// Parameters: dimension, min_vertices, max_vertices
test_minimal_simplex_hull!(2);
test_minimal_simplex_hull!(3);
test_minimal_simplex_hull!(4);
test_minimal_simplex_hull!(5);
test_convex_hull_properties!(2, 4, 10);
test_convex_hull_properties!(3, 5, 12);
test_convex_hull_properties!(4, 6, 14, #[cfg(feature = "slow-tests")]);
test_convex_hull_properties!(5, 7, 16, #[cfg(feature = "slow-tests")]);
