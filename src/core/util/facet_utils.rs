//! Facet-related utility helpers.

#![forbid(unsafe_code)]

use crate::core::collections::VertexUuidBuffer;
use crate::core::facet::FacetView;
use crate::core::traits::data_type::DataType;
use crate::core::vertex::Vertex;

/// Determines if two facet views are adjacent by comparing their vertices.
///
/// Two facets are considered adjacent if they contain the same set of vertices.
///
/// # Arguments
///
/// * `facet1` - The first facet view to compare
/// * `facet2` - The second facet view to compare
///
/// # Returns
///
/// `true` if the facets share the same vertices, or `false` if they have
/// different vertices.
///
/// # Examples
///
/// ```rust,no_run
/// use delaunay::prelude::tds::{FacetView, FacetError};
/// use delaunay::prelude::tds::facet_views_are_adjacent;
/// use delaunay::prelude::tds::Tds;
///
/// // This is a conceptual example - in practice you would get these from a real TDS
/// fn example(tds: &Tds<(), (), 3>) -> Result<bool, FacetError> {
///     let simplex_keys: Vec<_> = tds.simplex_keys().take(2).collect();
///     if simplex_keys.len() >= 2 {
///         let facet1 = FacetView::try_new(tds, simplex_keys[0], 0)?;
///         let facet2 = FacetView::try_new(tds, simplex_keys[1], 0)?;
///
///         let adjacent = facet_views_are_adjacent(&facet1, &facet2);
///         match adjacent {
///             true => println!("Facets are adjacent"),
///             false => println!("Facets are not adjacent"),
///         }
///         Ok(adjacent)
///     } else {
///         Ok(false)
///     }
/// }
/// ```
#[must_use]
pub fn facet_views_are_adjacent<U, V, const D: usize>(
    facet1: &FacetView<'_, U, V, D>,
    facet2: &FacetView<'_, U, V, D>,
) -> bool
where
    U: DataType,
    V: DataType,
{
    let vertices1 = sorted_facet_vertex_uuids(facet1);
    let vertices2 = sorted_facet_vertex_uuids(facet2);

    vertices1 == vertices2
}

/// Canonicalizes facet vertex UUIDs so facet comparison stays allocation-light
/// while remaining independent of local vertex order.
fn sorted_facet_vertex_uuids<U, V, const D: usize>(
    facet: &FacetView<'_, U, V, D>,
) -> VertexUuidBuffer
where
    U: DataType,
    V: DataType,
{
    let mut vertices: VertexUuidBuffer = facet.vertices().map(Vertex::uuid).collect();
    vertices.sort_unstable();
    vertices
}

/// Extracts owned vertices from a `FacetView` as a `Vec<Vertex>`.
///
/// This is a convenience utility that creates owned copies of the facet's vertices.
/// Since `Vertex` implements `Copy`, this operation is efficient and avoids the need
/// for manual `.copied().collect()` boilerplate throughout the codebase.
///
/// # Arguments
///
/// * `facet_view` - The facet view to extract vertices from
///
/// # Returns
///
/// A `Vec` of owned `Vertex` objects.
///
/// # Examples
///
/// ```rust,no_run
/// use delaunay::prelude::tds::FacetView;
/// use delaunay::prelude::tds::facet_view_to_vertices;
/// use delaunay::prelude::tds::{FacetError, Tds};
///
/// #[derive(Debug, thiserror::Error)]
/// enum ExampleError {
///     #[error(transparent)]
///     Facet(#[from] FacetError),
///     #[error("triangulation unexpectedly contains no simplices")]
///     MissingSimplex,
/// }
///
/// fn extract_vertices_example(
///     tds: &Tds<(), (), 3>,
/// ) -> Result<(), ExampleError> {
///     let simplex_key = tds.simplex_keys().next().ok_or(ExampleError::MissingSimplex)?;
///     let facet_view = FacetView::try_new(tds, simplex_key, 0)?;
///     
///     // Extract owned vertices
///     let vertices = facet_view_to_vertices(&facet_view);
///     println!("Facet has {} vertices", vertices.len());
///     Ok(())
/// }
/// ```
///
/// # Performance
///
/// - Time Complexity: O(D) where D is the dimension (number of vertices in facet)
/// - Space Complexity: O(D) for the returned vector
/// - Uses `Copy` semantics so this is as efficient as possible for owned vertices
#[must_use]
pub fn facet_view_to_vertices<U, V, const D: usize>(
    facet_view: &FacetView<'_, U, V, D>,
) -> Vec<Vertex<U, D>>
where
    U: DataType,
    V: DataType,
{
    facet_view.vertices().copied().collect()
}

/// Generates all unique combinations of `k` vertices for local regression tests.
///
/// Keeping this helper test-only avoids exposing a nested-`Vec` API for a
/// simplex-sized operation that production code models with small buffers.
///
/// # Arguments
///
/// * `vertices` - A slice of vertices from which to generate combinations.
/// * `k` - The size of each combination.
///
#[cfg(test)]
fn generate_combinations<U, const D: usize>(
    vertices: &[Vertex<U, D>],
    k: usize,
) -> Vec<Vec<Vertex<U, D>>>
where
    U: DataType,
{
    let mut combinations = Vec::new();

    if k == 0 {
        combinations.push(Vec::new());
        return combinations;
    }

    if k > vertices.len() {
        return combinations;
    }

    if k == vertices.len() {
        combinations.push(vertices.to_vec());
        return combinations;
    }

    // Generate combinations using iterative approach
    let n = vertices.len();
    let mut indices = (0..k).collect::<Vec<_>>();

    loop {
        // Add current combination
        let combination = indices.iter().map(|i| vertices[*i]).collect();
        combinations.push(combination);

        // Find next combination
        let mut i = k;
        loop {
            if i == 0 {
                return combinations;
            }
            i -= 1;
            if indices[i] != i + n - k {
                break;
            }
        }

        indices[i] += 1;
        for j in (i + 1)..k {
            indices[j] = indices[j - 1] + 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vertex;

    use crate::core::collections::FastHashSet;
    use crate::triangulation::DelaunayTriangulation;
    use std::time::Instant;

    #[test]
    fn test_generate_combinations_comprehensive() {
        // Test basic functionality with 4 vertices
        let vertices: Vec<Vertex<(), 1>> = vec![
            vertex!([0.0]).unwrap(),
            vertex!([1.0]).unwrap(),
            vertex!([2.0]).unwrap(),
            vertex!([3.0]).unwrap(),
        ];

        // Combinations of 2 from 4 - should be C(4,2) = 6
        let combinations_2 = generate_combinations(&vertices, 2);
        assert_eq!(combinations_2.len(), 6, "C(4,2) should equal 6");

        // Combinations of 3 from 4 - should be C(4,3) = 4
        let combinations_3 = generate_combinations(&vertices, 3);
        assert_eq!(combinations_3.len(), 4, "C(4,3) should equal 4");
        assert!(
            combinations_3.contains(&vec![vertices[0], vertices[1], vertices[2]]),
            "Should contain specific combination"
        );

        // Single vertex combinations (k=1) - should be C(4,1) = 4
        let combinations_1 = generate_combinations(&vertices, 1);
        assert_eq!(combinations_1.len(), 4, "C(4,1) should equal 4");
        assert!(
            combinations_1.contains(&vec![vertices[0]]),
            "Should contain first vertex"
        );
        assert!(
            combinations_1.contains(&vec![vertices[1]]),
            "Should contain second vertex"
        );
        assert!(
            combinations_1.contains(&vec![vertices[2]]),
            "Should contain third vertex"
        );
        assert!(
            combinations_1.contains(&vec![vertices[3]]),
            "Should contain fourth vertex"
        );

        // Edge case: k=0 - should return one empty combination
        let combinations_0 = generate_combinations(&vertices, 0);
        assert_eq!(combinations_0.len(), 1, "C(4,0) should equal 1");
        assert!(
            combinations_0[0].is_empty(),
            "k=0 should produce empty combination"
        );

        // Edge case: k > len - should return empty result
        let combinations_5 = generate_combinations(&vertices, 5);
        assert!(
            combinations_5.is_empty(),
            "k > n should return no combinations"
        );

        // Edge case: k == len - should return all vertices as single combination
        let combinations_4 = generate_combinations(&vertices, 4);
        assert_eq!(combinations_4.len(), 1, "C(4,4) should equal 1");
        assert_eq!(
            combinations_4[0], vertices,
            "k=n should return all vertices"
        );

        // Test with different size - 3 vertices, choose 2
        let small_vertices: Vec<Vertex<(), 1>> = vec![
            vertex!([1.0]).unwrap(),
            vertex!([2.0]).unwrap(),
            vertex!([3.0]).unwrap(),
        ];
        let combinations_small = generate_combinations(&small_vertices, 2);
        assert_eq!(combinations_small.len(), 3, "C(3,2) should equal 3");

        // Test larger case - 5 vertices, choose 3 to exercise inner loops
        let large_vertices: Vec<Vertex<(), 1>> = vec![
            vertex!([1.0]).unwrap(),
            vertex!([2.0]).unwrap(),
            vertex!([3.0]).unwrap(),
            vertex!([4.0]).unwrap(),
            vertex!([5.0]).unwrap(),
        ];
        let combinations_large = generate_combinations(&large_vertices, 3);
        assert_eq!(combinations_large.len(), 10, "C(5,3) should equal 10");

        // Verify some specific combinations exist in large case
        assert!(
            combinations_large.contains(&vec![
                large_vertices[0],
                large_vertices[1],
                large_vertices[2]
            ]),
            "Should contain first combination"
        );
        assert!(
            combinations_large.contains(&vec![
                large_vertices[2],
                large_vertices[3],
                large_vertices[4]
            ]),
            "Should contain last combination"
        );

        // Test empty input edge cases
        let empty_vertices: Vec<Vertex<(), 1>> = vec![];
        let combinations_empty_k1 = generate_combinations(&empty_vertices, 1);
        assert!(
            combinations_empty_k1.is_empty(),
            "Empty input with k>0 should return no combinations"
        );

        let combinations_empty_k0 = generate_combinations(&empty_vertices, 0);
        assert_eq!(
            combinations_empty_k0.len(),
            1,
            "Empty input with k=0 should return one empty combination"
        );
        assert!(
            combinations_empty_k0[0].is_empty(),
            "Empty input k=0 combination should be empty"
        );
    }

    #[test]
    fn test_facet_views_are_adjacent_comprehensive() {
        // Test 1: Adjacent facets in 3D (tetrahedra sharing a triangular face)
        tracing::debug!("Test 1: Adjacent facets in 3D");

        // Create two tetrahedra that share 3 vertices (forming a shared triangular face)
        let shared_vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(), // v0
            vertex!([1.0, 0.0, 0.0]).unwrap(), // v1
            vertex!([0.5, 1.0, 0.0]).unwrap(), // v2
        ];

        let vertex_a = vertex!([0.5, 0.5, 1.0]).unwrap(); // Above the shared triangle
        let vertex_b = vertex!([0.5, 0.5, -1.0]).unwrap(); // Below the shared triangle

        // Tetrahedron 1: shared triangle + vertex_a
        let mut vertices1 = shared_vertices.clone();
        vertices1.push(vertex_a);

        // Tetrahedron 2: shared triangle + vertex_b
        let mut vertices2 = shared_vertices;
        vertices2.push(vertex_b);

        let dt1 = DelaunayTriangulation::builder(&vertices1).build().unwrap();
        let dt2 = DelaunayTriangulation::builder(&vertices2).build().unwrap();
        let tds1 = dt1.tds();
        let tds2 = dt2.tds();

        let simplex1_key = tds1.simplex_keys().next().unwrap();
        let simplex2_key = tds2.simplex_keys().next().unwrap();

        // Find any pair of facets that share the 3 vertices from shared triangle
        // Can't assume insertion order, so check all facet pairs
        let mut found_adjacent = false;
        let mut facet_view1_adj = None;

        for facet_idx1 in 0..4 {
            for facet_idx2 in 0..4 {
                let fv1 = FacetView::try_new(tds1, simplex1_key, facet_idx1).unwrap();
                let fv2 = FacetView::try_new(tds2, simplex2_key, facet_idx2).unwrap();
                if facet_views_are_adjacent(&fv1, &fv2) {
                    found_adjacent = true;
                    facet_view1_adj = Some(fv1);
                    break;
                }
            }
            if found_adjacent {
                break;
            }
        }

        assert!(
            found_adjacent,
            "Facets representing the same shared triangle should be adjacent"
        );
        tracing::debug!("  ✓ Adjacent facets correctly identified");

        // Test 2: Non-adjacent facets from the same tetrahedra
        tracing::debug!("Test 2: Non-adjacent facets from same tetrahedra");

        // Find two facets that are NOT adjacent
        let mut found_non_adjacent = false;
        for facet_idx1 in 0..4 {
            for facet_idx2 in 0..4 {
                let fv1 = FacetView::try_new(tds1, simplex1_key, facet_idx1).unwrap();
                let fv2 = FacetView::try_new(tds2, simplex2_key, facet_idx2).unwrap();
                if !facet_views_are_adjacent(&fv1, &fv2) {
                    found_non_adjacent = true;
                    break;
                }
            }
            if found_non_adjacent {
                break;
            }
        }

        assert!(
            found_non_adjacent,
            "Should be able to find non-adjacent facets"
        );
        tracing::debug!("  ✓ Non-adjacent facets correctly identified");

        // Test 3: Same facet should be adjacent to itself
        tracing::debug!("Test 3: Facet adjacent to itself");

        let facet_view1 = facet_view1_adj.unwrap();
        assert!(
            facet_views_are_adjacent(&facet_view1, &facet_view1),
            "A facet should be adjacent to itself"
        );
        tracing::debug!("  ✓ Self-adjacency works correctly");
    }

    #[test]
    fn test_facet_views_are_adjacent_2d_cases() {
        tracing::debug!("Test 2D facet adjacency");

        // Create two 2D triangles that share an edge (2 vertices)
        let shared_edge = vec![
            vertex!([0.0, 0.0]).unwrap(), // v0
            vertex!([1.0, 0.0]).unwrap(), // v1
        ];

        let vertex_c = vertex!([0.5, 1.0]).unwrap(); // Above the shared edge
        let vertex_d = vertex!([0.5, -1.0]).unwrap(); // Below the shared edge

        // Triangle 1: shared edge + vertex_c
        let mut vertices1 = shared_edge.clone();
        vertices1.push(vertex_c);

        // Triangle 2: shared edge + vertex_d
        let mut vertices2 = shared_edge;
        vertices2.push(vertex_d);

        let dt1 = DelaunayTriangulation::builder(&vertices1).build().unwrap();
        let dt2 = DelaunayTriangulation::builder(&vertices2).build().unwrap();
        let tds1 = dt1.tds();
        let tds2 = dt2.tds();

        let simplex1_key = tds1.simplex_keys().next().unwrap();
        let simplex2_key = tds2.simplex_keys().next().unwrap();

        // In 2D, facets are edges. Find ANY pair of facets that share the 2 vertices from shared_edge
        // We can't assume insertion order, so check all facet pairs
        let mut found_adjacent = false;
        for facet_idx1 in 0..3 {
            for facet_idx2 in 0..3 {
                let facet_view1 = FacetView::try_new(tds1, simplex1_key, facet_idx1).unwrap();
                let facet_view2 = FacetView::try_new(tds2, simplex2_key, facet_idx2).unwrap();
                if facet_views_are_adjacent(&facet_view1, &facet_view2) {
                    found_adjacent = true;
                    break;
                }
            }
            if found_adjacent {
                break;
            }
        }

        assert!(
            found_adjacent,
            "2D facets (edges) sharing vertices should be adjacent"
        );

        tracing::debug!("  ✓ 2D facet adjacency works correctly");
    }

    #[test]
    fn test_facet_views_are_adjacent_1d_cases() {
        tracing::debug!("Test 1D facet adjacency");

        // In 1D, simplices are edges and facets are vertices (0D)
        // Two edges sharing a vertex have adjacent facets

        let shared_vertex = vertex!([0.0]).unwrap();
        let vertex_left = vertex!([-1.0]).unwrap();
        let vertex_right = vertex!([1.0]).unwrap();

        // Edge 1: shared_vertex to vertex_left
        let vertices1 = vec![shared_vertex, vertex_left];
        // Edge 2: shared_vertex to vertex_right
        let vertices2 = vec![shared_vertex, vertex_right];

        let dt1 = DelaunayTriangulation::builder(&vertices1).build().unwrap();
        let dt2 = DelaunayTriangulation::builder(&vertices2).build().unwrap();
        let tds1 = dt1.tds();
        let tds2 = dt2.tds();

        let simplex1_key = tds1.simplex_keys().next().unwrap();
        let simplex2_key = tds2.simplex_keys().next().unwrap();

        // In 1D, the facets are the individual vertices (0D)
        // Both edges contain the shared so find which facet pairs are adjacent
        let mut found_adjacent = false;
        let mut found_non_adjacent = false;

        for facet_idx1 in 0..2 {
            for facet_idx2 in 0..2 {
                let fv1 = FacetView::try_new(tds1, simplex1_key, facet_idx1).unwrap();
                let fv2 = FacetView::try_new(tds2, simplex2_key, facet_idx2).unwrap();
                if facet_views_are_adjacent(&fv1, &fv2) {
                    found_adjacent = true;
                } else {
                    found_non_adjacent = true;
                }
            }
        }

        // The facets containing the shared vertex should be adjacent
        assert!(
            found_adjacent,
            "1D facets (vertices) that are the same should be adjacent"
        );

        // The facets containing different vertices should not be adjacent
        assert!(
            found_non_adjacent,
            "1D facets with different vertices should not be adjacent"
        );

        tracing::debug!("  ✓ 1D facet adjacency works correctly");
    }

    #[test]
    fn test_facet_views_are_adjacent_edge_cases() {
        tracing::debug!("Test facet adjacency edge cases");

        // Test with minimal triangulation (single tetrahedron)
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];

        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let tds = dt.tds();
        let simplex_key = tds.simplex_keys().next().unwrap();

        // All facets of the same tetrahedron should be different from each other
        let facet0 = FacetView::try_new(tds, simplex_key, 0).unwrap();
        let facet1 = FacetView::try_new(tds, simplex_key, 1).unwrap();
        let facet2 = FacetView::try_new(tds, simplex_key, 2).unwrap();
        let facet3 = FacetView::try_new(tds, simplex_key, 3).unwrap();

        // Each facet should be adjacent to itself
        assert!(facet_views_are_adjacent(&facet0, &facet0));
        assert!(facet_views_are_adjacent(&facet1, &facet1));
        assert!(facet_views_are_adjacent(&facet2, &facet2));
        assert!(facet_views_are_adjacent(&facet3, &facet3));

        // Different facets of the same tetrahedron should not be adjacent
        // (they have different sets of vertices)
        assert!(!facet_views_are_adjacent(&facet0, &facet1));
        assert!(!facet_views_are_adjacent(&facet0, &facet2));
        assert!(!facet_views_are_adjacent(&facet0, &facet3));
        assert!(!facet_views_are_adjacent(&facet1, &facet2));
        assert!(!facet_views_are_adjacent(&facet1, &facet3));
        assert!(!facet_views_are_adjacent(&facet2, &facet3));

        tracing::debug!("  ✓ Single tetrahedron facet relationships correct");
    }

    #[test]
    fn test_facet_views_are_adjacent_performance() {
        tracing::debug!("Test facet adjacency performance");

        // Create a moderately complex case to test performance
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([2.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 2.0, 0.0]).unwrap(),
            vertex!([1.0, 1.0, 2.0]).unwrap(),
        ];

        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let tds = dt.tds();
        let simplex_key = tds.simplex_keys().next().unwrap();

        let facet1 = FacetView::try_new(tds, simplex_key, 0).unwrap();
        let facet2 = FacetView::try_new(tds, simplex_key, 1).unwrap();

        // Run the adjacency check many times to measure performance
        let start = Instant::now();
        let iterations = 10000;

        for _ in 0..iterations {
            // This should be very fast since it just compares UUID sets
            let _result = facet_views_are_adjacent(&facet1, &facet2);
        }

        let duration = start.elapsed();
        tracing::debug!("  ✓ {iterations} adjacency checks completed in {duration:?}");

        // Performance info: each check is just UUID set comparison
        // Note: Timing can vary significantly based on build type and CI environment
        if duration.as_millis() > 500 {
            tracing::warn!("  ⚠️  Performance warning: adjacency checks took {duration:?}");
            tracing::warn!("     This may indicate debug build or slower CI environment");
        }
    }

    #[test]
    fn test_facet_views_are_adjacent_different_geometries() {
        tracing::debug!("Test facet adjacency with different geometries");

        // Create vertices with different coordinates to ensure different UUIDs
        let vertices1 = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];

        let vertices2 = vec![
            vertex!([10.0, 10.0, 10.0]).unwrap(),
            vertex!([11.0, 10.0, 10.0]).unwrap(),
            vertex!([10.0, 11.0, 10.0]).unwrap(),
            vertex!([10.0, 10.0, 11.0]).unwrap(),
        ];

        let dt1 = DelaunayTriangulation::builder(&vertices1).build().unwrap();
        let dt2 = DelaunayTriangulation::builder(&vertices2).build().unwrap();
        let tds1 = dt1.tds();
        let tds2 = dt2.tds();

        let simplex1_key = tds1.simplex_keys().next().unwrap();
        let simplex2_key = tds2.simplex_keys().next().unwrap();

        let facet1 = FacetView::try_new(tds1, simplex1_key, 0).unwrap();
        let facet2 = FacetView::try_new(tds2, simplex2_key, 0).unwrap();

        // Facets from completely different geometries should not be adjacent
        assert!(
            !facet_views_are_adjacent(&facet1, &facet2),
            "Facets from different geometries should not be adjacent"
        );

        tracing::debug!("  ✓ Different geometries correctly distinguished");
    }

    #[test]
    fn test_facet_views_are_adjacent_uuid_based_comparison() {
        tracing::debug!("Test that adjacency is purely UUID-based");

        // Create identical geometry in separate TDS instances
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];

        let dt1 = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let dt2 = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let tds1 = dt1.tds();
        let tds2 = dt2.tds();

        let simplex1_key = tds1.simplex_keys().next().unwrap();
        let simplex2_key = tds2.simplex_keys().next().unwrap();

        let facet1 = FacetView::try_new(tds1, simplex1_key, 0).unwrap();
        let facet2 = FacetView::try_new(tds2, simplex2_key, 0).unwrap();

        // Check if the UUID generation is deterministic based on coordinates
        let facet1_vertex_uuids: FastHashSet<_> = facet1.vertices().map(Vertex::uuid).collect();
        let facet2_vertex_uuids: FastHashSet<_> = facet2.vertices().map(Vertex::uuid).collect();

        let uuids_are_same = facet1_vertex_uuids == facet2_vertex_uuids;
        let facets_are_adjacent = facet_views_are_adjacent(&facet1, &facet2);

        if uuids_are_same != facets_are_adjacent {
            let mut facet1_uuid_list: Vec<_> = facet1_vertex_uuids.iter().copied().collect();
            facet1_uuid_list.sort_unstable();
            let mut facet2_uuid_list: Vec<_> = facet2_vertex_uuids.iter().copied().collect();
            facet2_uuid_list.sort_unstable();
            tracing::debug!(
                "  ⚠️ UUID mismatch: facet1={facet1_uuid_list:?}, facet2={facet2_uuid_list:?}"
            );
        }

        // The adjacency should match the UUID equality
        assert_eq!(
            uuids_are_same, facets_are_adjacent,
            "Facet adjacency should exactly match vertex UUID equality"
        );

        if uuids_are_same {
            tracing::debug!(
                "  ✓ Identical coordinates produce identical UUIDs - facets are adjacent"
            );
        } else {
            tracing::debug!(
                "  ✓ Different UUIDs for identical coordinates - facets are not adjacent"
            );
        }
    }

    #[test]
    fn test_facet_views_are_adjacent_4d_cases() {
        tracing::debug!("Test 4D facet adjacency");

        // Create two 4D simplices (5-vertices each) that share a 3D facet (4 vertices)
        let shared_tetrahedron = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]).unwrap(), // v0
            vertex!([1.0, 0.0, 0.0, 0.0]).unwrap(), // v1
            vertex!([0.0, 1.0, 0.0, 0.0]).unwrap(), // v2
            vertex!([0.0, 0.0, 1.0, 0.0]).unwrap(), // v3
        ];

        let vertex_e = vertex!([0.25, 0.25, 0.25, 1.0]).unwrap(); // Above in 4th dimension
        let vertex_f = vertex!([0.25, 0.25, 0.25, -1.0]).unwrap(); // Below in 4th dimension

        // 4D Simplex 1: shared tetrahedron + vertex_e
        let mut vertices1 = shared_tetrahedron.clone();
        vertices1.push(vertex_e);

        // 4D Simplex 2: shared tetrahedron + vertex_f
        let mut vertices2 = shared_tetrahedron;
        vertices2.push(vertex_f);

        let dt1 = DelaunayTriangulation::builder(&vertices1).build().unwrap();
        let dt2 = DelaunayTriangulation::builder(&vertices2).build().unwrap();
        let tds1 = dt1.tds();
        let tds2 = dt2.tds();

        let simplex1_key = tds1.simplex_keys().next().unwrap();
        let simplex2_key = tds2.simplex_keys().next().unwrap();

        // In 4D, facets are tetrahedra. Find any pair that shares the 4 vertices from shared tetrahedron
        let mut found_adjacent = false;
        let mut found_non_adjacent = false;

        for facet_idx1 in 0..5 {
            for facet_idx2 in 0..5 {
                let fv1 = FacetView::try_new(tds1, simplex1_key, facet_idx1).unwrap();
                let fv2 = FacetView::try_new(tds2, simplex2_key, facet_idx2).unwrap();
                if facet_views_are_adjacent(&fv1, &fv2) {
                    found_adjacent = true;
                } else {
                    found_non_adjacent = true;
                }
            }
        }

        assert!(
            found_adjacent,
            "4D facets (tetrahedra) sharing vertices should be adjacent"
        );

        assert!(
            found_non_adjacent,
            "4D facets with different vertices should not be adjacent"
        );

        tracing::debug!("  ✓ 4D facet adjacency works correctly");
    }

    #[test]
    fn test_facet_views_are_adjacent_5d_cases() {
        tracing::debug!("Test 5D facet adjacency");

        // Create two 5D simplices (6-vertices each) that share a 4D facet (5 vertices)
        let shared_4d_simplex = vec![
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0]).unwrap(), // v0
            vertex!([1.0, 0.0, 0.0, 0.0, 0.0]).unwrap(), // v1
            vertex!([0.0, 1.0, 0.0, 0.0, 0.0]).unwrap(), // v2
            vertex!([0.0, 0.0, 1.0, 0.0, 0.0]).unwrap(), // v3
            vertex!([0.0, 0.0, 0.0, 1.0, 0.0]).unwrap(), // v4
        ];

        let vertex_g = vertex!([0.2, 0.2, 0.2, 0.2, 1.0]).unwrap(); // Above in 5th dimension
        let vertex_h = vertex!([0.2, 0.2, 0.2, 0.2, -1.0]).unwrap(); // Below in 5th dimension

        // 5D Simplex 1: shared 4D simplex + vertex_g
        let mut vertices1 = shared_4d_simplex.clone();
        vertices1.push(vertex_g);

        // 5D Simplex 2: shared 4D simplex + vertex_h
        let mut vertices2 = shared_4d_simplex;
        vertices2.push(vertex_h);

        let dt1 = DelaunayTriangulation::builder(&vertices1).build().unwrap();
        let dt2 = DelaunayTriangulation::builder(&vertices2).build().unwrap();
        let tds1 = dt1.tds();
        let tds2 = dt2.tds();

        let simplex1_key = tds1.simplex_keys().next().unwrap();
        let simplex2_key = tds2.simplex_keys().next().unwrap();

        // In 5D, facets are 4D simplices. Find any pair that shares the 5 vertices from shared 4D simplex
        let mut found_adjacent = false;
        let mut found_non_adjacent = false;

        for facet_idx1 in 0..6 {
            for facet_idx2 in 0..6 {
                let fv1 = FacetView::try_new(tds1, simplex1_key, facet_idx1).unwrap();
                let fv2 = FacetView::try_new(tds2, simplex2_key, facet_idx2).unwrap();
                if facet_views_are_adjacent(&fv1, &fv2) {
                    found_adjacent = true;
                } else {
                    found_non_adjacent = true;
                }
            }
        }

        assert!(
            found_adjacent,
            "5D facets (4D simplices) sharing vertices should be adjacent"
        );

        assert!(
            found_non_adjacent,
            "5D facets with different vertices should not be adjacent"
        );

        tracing::debug!("  ✓ 5D facet adjacency works correctly");
    }
}
