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

#[path = "common/full_dimensional_vertices.rs"]
mod full_dimensional_vertices;

#[macro_use]
#[path = "common/proptest_config.rs"]
mod proptest_config;

use delaunay::assert_jaccard_gte;
use delaunay::prelude::construction::{DelaunayTriangulation, TopologyGuarantee};
use delaunay::prelude::query::*;
use delaunay::try_vertices_from_points;
use full_dimensional_vertices::full_dimensional_vertices;
use proptest::prelude::*;
use proptest::test_runner::TestCaseError;
use std::collections::{HashMap, HashSet};

// =============================================================================
// TEST CONFIGURATION
// =============================================================================

/// Strategy for generating finite f64 coordinates
fn finite_coordinate() -> impl Strategy<Value = f64> {
    (-100.0..100.0).prop_filter("must be finite", |x: &f64| x.is_finite())
}

fn count_boundary_facets<K, U, V, const D: usize>(dt: &DelaunayTriangulation<K, U, V, D>) -> usize {
    dt.boundary_facets()
        .expect("boundary facets should be queryable for valid triangulations")
        .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))
        .expect("boundary facet handles should resolve")
}

// =============================================================================
// DIMENSIONAL TEST GENERATION MACROS
// =============================================================================

/// Macro to generate minimal simplex hull property tests for a given dimension
macro_rules! test_minimal_simplex_hull {
    ($dim:literal $(, #[$attr:meta])*) => {
        pastey::paste! {
            repo_proptest! {
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

                    let dt_result = DelaunayTriangulation::builder(&vertices).topology_guarantee(TopologyGuarantee::PLManifold).build();
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
            repo_proptest! {
                /// Property: Convex hull can be constructed from valid triangulation
                $(#[$attr])*
                #[test]
                fn [<prop_hull_construction_ $dim d>](
                    vertices in full_dimensional_vertices::<$dim>($min_vertices, $max_vertices)
                ) {
                    let dt = DelaunayTriangulation::builder(&vertices)
                        .topology_guarantee(TopologyGuarantee::PLManifold)
                        .build()
                        .map_err(|error| TestCaseError::fail(format!(
                            "{}D construction failed for admitted vertices {vertices:?}: {error:?}", $dim
                        )))?;

                    let boundary_count = count_boundary_facets(&dt);
                    prop_assert!(boundary_count > 0, "{}D full-dimensional cloud must have a boundary", $dim);

                    // Should be able to construct hull from valid triangulation
                    let hull_result = ConvexHull::try_from_triangulation(dt.as_triangulation());
                    prop_assert!(
                        hull_result.is_ok(),
                        "{}D convex hull construction should succeed for valid triangulation: {:?}",
                        $dim,
                        hull_result.as_ref().err()
                    );
                    let hull = hull_result.expect("checked random hull construction");

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
                    vertices in full_dimensional_vertices::<$dim>($min_vertices, $max_vertices)
                ) {
                    let dt = DelaunayTriangulation::builder(&vertices)
                        .topology_guarantee(TopologyGuarantee::PLManifold)
                        .build()
                        .map_err(|error| TestCaseError::fail(format!(
                            "{}D construction failed for admitted vertices {vertices:?}: {error:?}", $dim
                        )))?;
                    let hull = ConvexHull::try_from_triangulation(dt.as_triangulation())
                        .map_err(|error| TestCaseError::fail(format!(
                            "{}D hull extraction failed for {vertices:?}: {error:?}", $dim
                        )))?;

                    let facet_count = hull.number_of_facets();
                    let vertex_count = dt.number_of_vertices();

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

                /// Property: An owned hull remains unchanged after source mutation
                $(#[$attr])*
                #[test]
                fn [<prop_hull_snapshot_independence_ $dim d>](
                    initial_vertices in full_dimensional_vertices::<$dim>($min_vertices, $max_vertices),
                    new_point in prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates"))
                ) {
                    // Only raw coordinate equality can exclude an insertion case.
                    prop_assume!(initial_vertices.iter().all(|vertex| {
                        !vertex.point().coords().iter().eq(new_point.coords().iter())
                    }));
                    let mut dt = DelaunayTriangulation::builder(&initial_vertices)
                        .topology_guarantee(TopologyGuarantee::PLManifold)
                        .build()
                        .map_err(|error| TestCaseError::fail(format!(
                            "{}D construction failed for admitted vertices {initial_vertices:?}: {error:?}", $dim
                        )))?;

                    let initial_boundary_count = count_boundary_facets(&dt);
                    prop_assert!(initial_boundary_count > 0);

                    let hull = ConvexHull::try_from_triangulation(dt.as_triangulation())
                        .map_err(|error| TestCaseError::fail(format!(
                            "{}D initial hull extraction failed for {initial_vertices:?}: {error:?}", $dim
                        )))?;

                    let original_facets = extract_hull_facet_set(&hull);

                    // Modify the triangulation by inserting a new vertex
                    let new_vertex =
                        try_vertices_from_points(&[new_point]).expect("finite point coordinates");
                    dt.insert_vertex(new_vertex[0])
                        .map_err(|error| TestCaseError::fail(format!(
                            "{}D insertion failed for {initial_vertices:?}, point {new_point:?}: {error:?}", $dim
                        )))?;
                    prop_assert_eq!(dt.number_of_vertices(), initial_vertices.len() + 1);

                    let modified_boundary_count = count_boundary_facets(&dt);
                    prop_assert!(modified_boundary_count > 0);

                    prop_assert_eq!(
                        extract_hull_facet_set(&hull),
                        original_facets,
                        "{}D owned hull should not change with its source triangulation",
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

                    prop_assert!(new_hull.number_of_facets() > 0);
                }

                /// Property: Hull vertices are a subset of triangulation vertices
                $(#[$attr])*
                #[test]
                fn [<prop_hull_vertices_subset_ $dim d>](
                    vertices in full_dimensional_vertices::<$dim>($min_vertices, $max_vertices)
                ) {
                    let dt = DelaunayTriangulation::builder(&vertices)
                        .topology_guarantee(TopologyGuarantee::PLManifold)
                        .build()
                        .map_err(|error| TestCaseError::fail(format!(
                            "{}D construction failed for admitted vertices {vertices:?}: {error:?}", $dim
                        )))?;
                    let hull = ConvexHull::try_from_triangulation(dt.as_triangulation())
                        .map_err(|error| TestCaseError::fail(format!(
                            "{}D hull extraction failed for {vertices:?}: {error:?}", $dim
                        )))?;

                    let source_vertices: HashMap<_, _> = dt.vertices()
                        .map(|(_, vertex)| (vertex.uuid(), vertex.point().coords().map(f64::to_bits)))
                        .collect();
                    let mut hull_vertices = HashSet::new();
                    for facet in hull.facets() {
                        let mut facet_vertices = HashSet::new();
                        for vertex in facet.vertices() {
                            let uuid = vertex.uuid();
                            let coordinates = vertex.point().coords().map(f64::to_bits);
                            prop_assert_eq!(
                                source_vertices.get(&uuid), Some(&coordinates),
                                "{}D hull vertex {} must preserve source identity and coordinates", $dim, uuid
                            );
                            prop_assert!(facet_vertices.insert(uuid), "{}D facet must not repeat a vertex", $dim);
                            hull_vertices.insert(uuid);
                        }
                        prop_assert_eq!(facet_vertices.len(), $dim);
                    }
                    prop_assert!(
                        hull_vertices.len() > $dim && hull_vertices.len() <= source_vertices.len(),
                        "{}D hull must contain between {} and {} distinct source vertices, got {}",
                        $dim, $dim + 1, source_vertices.len(), hull_vertices.len()
                    );
                }

                /// Property: Reconstructing hull from same TDS gives consistent facet count
                $(#[$attr])*
                #[test]
                fn [<prop_hull_reconstruction_consistency_ $dim d>](
                    vertices in full_dimensional_vertices::<$dim>($min_vertices, $max_vertices)
                ) {
                    let dt = DelaunayTriangulation::builder(&vertices)
                        .topology_guarantee(TopologyGuarantee::PLManifold)
                        .build()
                        .map_err(|error| TestCaseError::fail(format!(
                            "{}D construction failed for admitted vertices {vertices:?}: {error:?}", $dim
                        )))?;
                    let hull1 = ConvexHull::try_from_triangulation(dt.as_triangulation())
                        .map_err(|error| TestCaseError::fail(format!(
                            "{}D first hull extraction failed for {vertices:?}: {error:?}", $dim
                        )))?;
                    let hull2 = ConvexHull::try_from_triangulation(dt.as_triangulation())
                        .map_err(|error| TestCaseError::fail(format!(
                            "{}D second hull extraction failed for {vertices:?}: {error:?}", $dim
                        )))?;

                    // Both hulls should have the same facet count
                    prop_assert_eq!(
                        hull1.number_of_facets(),
                        hull2.number_of_facets(),
                        "{}D reconstructing hull from same triangulation should give same facet count",
                        $dim
                    );

                    // Extract facet sets and compare via Jaccard similarity
                    // Should be exactly identical (Jaccard = 1.0) since same triangulation
                    let facets1 = extract_hull_facet_set(&hull1);
                    let facets2 = extract_hull_facet_set(&hull2);
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
test_convex_hull_properties!(4, 6, 14);
test_convex_hull_properties!(5, 7, 16, #[cfg(feature = "slow-tests")]);
