#![forbid(unsafe_code)]

//! Integration tests for `DelaunayTriangulation` incremental insertion.
//!
//! These tests focus on the incremental insertion workflow and features
//! specific to `DelaunayTriangulation`, including:
//! - Sequential vertex insertion
//! - Various point distributions
//! - Different dimensions (2D-5D)
//! - Different kernels (Fast vs Robust)

use approx::assert_relative_eq;
use delaunay::geometry::kernel::RobustKernel;
use delaunay::prelude::algorithms::{LocateResult, find_conflict_region, locate};
use delaunay::prelude::collections::MAX_PRACTICAL_DIMENSION_SIZE;
use delaunay::prelude::geometry::{AdaptiveKernel, Coordinate, Point};
use delaunay::prelude::tds::{
    Simplex, SimplexKey, SmallBuffer, VertexKey, facet_key_from_vertices,
};
use delaunay::prelude::triangulation::construction::{
    ConstructionOptions, DedupPolicy, DelaunayTriangulation, TopologyGuarantee, Vertex, vertex,
};
use uuid::Uuid;

/// Build the canonical facet key used to compare neighbor mirror facets in tests.
fn facet_key_for_simplex<const D: usize>(
    simplex: &Simplex<f64, (), (), D>,
    facet_idx: usize,
) -> u64 {
    let mut facet_vertices = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
    for (idx, &vertex_key) in simplex.vertices().iter().enumerate() {
        if idx != facet_idx {
            facet_vertices.push(vertex_key);
        }
    }
    facet_key_from_vertices(&facet_vertices)
}

/// Assert that every populated neighbor slot references an existing simplex that points back.
fn assert_neighbors_valid_and_symmetric<const D: usize>(
    dt: &DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>,
) {
    for (simplex_key, simplex) in dt.simplices() {
        let Some(neighbors) = simplex.neighbors() else {
            continue;
        };
        for (facet_idx, neighbor_opt) in neighbors.into_iter().enumerate() {
            let Some(neighbor_key) = neighbor_opt else {
                continue;
            };
            let neighbor_simplex = dt.tds().simplex(neighbor_key).unwrap_or_else(|| {
                panic!("neighbor {neighbor_key:?} for {simplex_key:?} is missing")
            });
            let facet_key = facet_key_for_simplex(simplex, facet_idx);
            let mirror_idx = (0..neighbor_simplex.number_of_vertices())
                .find(|&idx| facet_key_for_simplex(neighbor_simplex, idx) == facet_key)
                .unwrap_or_else(|| {
                    panic!(
                        "neighbor {neighbor_key:?} does not share facet {facet_idx} with {simplex_key:?}"
                    )
                });
            let neighbor_back = neighbor_simplex.neighbor_key(mirror_idx).flatten();
            assert_eq!(
                neighbor_back,
                Some(simplex_key),
                "neighbor symmetry failed for {simplex_key:?} facet {facet_idx}"
            );
        }
    }
}

/// Return a query point strictly inside the first simplex so locate traversal has a stable target.
fn centroid_of_first_simplex<const D: usize>(
    dt: &DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>,
) -> (SimplexKey, Point<f64, D>) {
    let (simplex_key, simplex) = dt
        .simplices()
        .next()
        .expect("test triangulation should contain at least one simplex");
    let mut coords = [0.0; D];
    for &vertex_key in simplex.vertices() {
        let vertex = dt
            .tds()
            .vertex(vertex_key)
            .expect("simplex vertex should exist");
        for (coord, &value) in coords.iter_mut().zip(vertex.point().coords()) {
            *coord += value;
        }
    }
    let mut vertex_count = 0.0;
    for _ in simplex.vertices() {
        vertex_count += 1.0;
    }
    let scale = 1.0 / vertex_count;
    for coord in &mut coords {
        *coord *= scale;
    }
    (simplex_key, Point::new(coords))
}

/// Verify repaired neighbor pointers support hinted locate and conflict-region traversal.
fn assert_locate_and_conflict_traversal<const D: usize>(
    dt: &DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>,
) {
    let kernel = AdaptiveKernel::<f64>::new();
    let (hint_simplex, query) = centroid_of_first_simplex(dt);

    let no_hint = locate(dt.tds(), &kernel, &query, None).expect("locate without hint failed");
    let with_hint =
        locate(dt.tds(), &kernel, &query, Some(hint_simplex)).expect("locate with hint failed");

    let start_simplex = match with_hint {
        LocateResult::InsideSimplex(simplex_key) => simplex_key,
        other => panic!("centroid should locate inside a simplex with hint, got {other:?}"),
    };
    assert!(
        matches!(no_hint, LocateResult::InsideSimplex(_)),
        "centroid should locate inside a simplex without hint, got {no_hint:?}"
    );

    let conflict_simplices = find_conflict_region(dt.tds(), &kernel, &query, start_simplex)
        .expect("conflict traversal failed");
    assert!(!conflict_simplices.is_empty());
    for &simplex_key in &conflict_simplices {
        assert!(dt.tds().contains_simplex(simplex_key));
    }
}

// =========================================================================
// Basic Incremental Insertion Tests (using macros for 2D-5D)
// =========================================================================

/// Macro to generate single point insertion tests across dimensions.
macro_rules! test_insert_single_point {
    ($dim:expr, [$($simplex:expr),+ $(,)?], $point:expr, $expected_simplices:expr) => {
        pastey::paste! {
            #[test]
            fn [<test_insert_single_point_ $dim d>]() {
                let vertices = vec![
                    $(vertex!($simplex)),+
                ];

                let mut dt: DelaunayTriangulation<_, (), (), $dim> =
                    DelaunayTriangulation::new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    )
                    .unwrap();

                let initial_vertices = vertices.len();
                assert_eq!(dt.number_of_simplices(), 1);

                dt.insert(vertex!($point)).unwrap();

                assert_eq!(dt.number_of_vertices(), initial_vertices + 1);
                assert_eq!(dt.number_of_simplices(), $expected_simplices);
            }
        }
    };
}

// Generate tests for 2D-5D
test_insert_single_point!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [0.3, 0.3], 3);
test_insert_single_point!(
    3,
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ],
    [0.25, 0.25, 0.25],
    4
);
test_insert_single_point!(
    4,
    [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ],
    [0.15, 0.15, 0.15, 0.15],
    5
);
test_insert_single_point!(
    5,
    [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0]
    ],
    [0.1, 0.1, 0.1, 0.1, 0.1],
    6
);

/// Macro to generate 5-point insertion tests across dimensions.
macro_rules! test_insert_5_points {
    ($dim:expr, [$($simplex:expr),+ $(,)?], [$($point:expr),+ $(,)?]) => {
        pastey::paste! {
            #[test]
            fn [<test_insert_5_points_ $dim d>]() {
                let vertices = vec![
                    $(vertex!($simplex)),+
                ];

                let mut dt: DelaunayTriangulation<_, (), (), $dim> =
                    DelaunayTriangulation::new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    )
                    .unwrap();

                let initial_vertices = vertices.len();
                assert_eq!(dt.number_of_simplices(), 1);

                // Insert 5 well-spaced interior points
                let points = vec![$(vertex!($point)),+];
                for (i, point) in points.iter().enumerate() {
                    dt.insert(*point).unwrap();
                    assert_eq!(dt.number_of_vertices(), initial_vertices + i + 1);
                }

                assert_eq!(dt.number_of_vertices(), initial_vertices + 5);
                assert!(dt.number_of_simplices() > 1);
            }
        }
    };
}

// Generate 5-point tests for 2D-5D
test_insert_5_points!(
    2,
    [[0.0, 0.0], [4.0, 0.0], [2.0, 4.0]],
    [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [1.5, 2.0], [2.5, 2.0]]
);

test_insert_5_points!(
    3,
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ],
    [
        [0.2, 0.2, 0.2],
        [0.3, 0.15, 0.15],
        [0.15, 0.3, 0.15],
        [0.15, 0.15, 0.3],
        [0.25, 0.25, 0.25]
    ]
);

test_insert_5_points!(
    4,
    [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ],
    [
        [0.1, 0.1, 0.1, 0.1],
        [0.15, 0.1, 0.1, 0.1],
        [0.1, 0.15, 0.1, 0.1],
        [0.1, 0.1, 0.15, 0.1],
        [0.12, 0.12, 0.12, 0.12]
    ]
);

test_insert_5_points!(
    5,
    [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0]
    ],
    [
        [0.1, 0.1, 0.1, 0.1, 0.1],
        [0.15, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.15, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.15, 0.1, 0.1],
        [0.12, 0.12, 0.12, 0.12, 0.12]
    ]
);

macro_rules! test_local_neighbor_repair_guardrails {
    ($dim:expr, [$($simplex:expr),+ $(,)?], [$($point:expr),+ $(,)?]) => {
        pastey::paste! {
            #[test]
            fn [<test_local_neighbor_repair_guardrails_ $dim d>]() {
                let vertices = vec![
                    $(vertex!($simplex)),+
                ];
                let mut dt: DelaunayTriangulation<AdaptiveKernel<f64>, (), (), $dim> =
                    DelaunayTriangulation::with_topology_guarantee(
                        &AdaptiveKernel::new(),
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    )
                    .unwrap();

                for point in [$(vertex!($point)),+] {
                    dt.insert(point).unwrap();
                    assert_neighbors_valid_and_symmetric(&dt);
                    assert_locate_and_conflict_traversal(&dt);
                }
            }
        }
    };
}

test_local_neighbor_repair_guardrails!(
    2,
    [[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]],
    [
        [0.8, 0.8],
        [1.6, 0.7],
        [0.7, 1.5],
        [1.2, 1.2],
        [1.0, 1.0e-9]
    ]
);

test_local_neighbor_repair_guardrails!(
    3,
    [
        [0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 4.0]
    ],
    [
        [0.8, 0.8, 0.8],
        [1.2, 0.6, 0.7],
        [0.7, 1.1, 0.9],
        [1.0, 0.9, 1.2],
        [1.0e-9, 0.8, 0.8]
    ]
);

test_local_neighbor_repair_guardrails!(
    4,
    [
        [0.0, 0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0, 0.0],
        [0.0, 0.0, 4.0, 0.0],
        [0.0, 0.0, 0.0, 4.0]
    ],
    [
        [0.5, 0.5, 0.5, 0.5],
        [0.8, 0.4, 0.5, 0.6],
        [0.4, 0.8, 0.6, 0.5],
        [0.6, 0.5, 0.8, 0.4],
        [1.0e-9, 0.6, 0.6, 0.6]
    ]
);

test_local_neighbor_repair_guardrails!(
    5,
    [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 4.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 4.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 4.0]
    ],
    [
        [0.4, 0.4, 0.4, 0.4, 0.4],
        [0.7, 0.3, 0.4, 0.5, 0.4],
        [0.3, 0.7, 0.5, 0.4, 0.4],
        [0.4, 0.5, 0.7, 0.3, 0.4],
        [1.0e-9, 0.5, 0.5, 0.5, 0.5]
    ]
);

// =========================================================================
// Kernel Comparison Tests
// =========================================================================

#[test]
fn test_adaptive_kernel_vs_robust_kernel_2d() {
    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
        vertex!([0.5, 0.5]),
    ];

    let dt_adaptive: DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 2> =
        DelaunayTriangulation::with_topology_guarantee(
            &AdaptiveKernel::new(),
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

    let dt_robust: DelaunayTriangulation<RobustKernel<f64>, (), (), 2> =
        DelaunayTriangulation::with_topology_guarantee(
            &RobustKernel::new(),
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

    // Both should produce same vertex count
    assert_eq!(
        dt_adaptive.number_of_vertices(),
        dt_robust.number_of_vertices()
    );
    assert_eq!(dt_adaptive.number_of_vertices(), 4);

    // Both should create valid triangulations
    assert!(dt_adaptive.number_of_simplices() > 0);
    assert!(dt_robust.number_of_simplices() > 0);
}

#[test]
fn test_robust_kernel_incremental_insertion() {
    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
    ];

    let mut dt: DelaunayTriangulation<RobustKernel<f64>, (), (), 2> =
        DelaunayTriangulation::with_topology_guarantee(
            &RobustKernel::new(),
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

    // Insert points with robust kernel
    dt.insert(vertex!([0.3, 0.3])).unwrap();
    dt.insert(vertex!([0.5, 0.3])).unwrap();
    dt.insert(vertex!([0.4, 0.5])).unwrap();

    assert_eq!(dt.number_of_vertices(), 6);
    assert!(dt.number_of_simplices() > 1);
}

// =========================================================================
// Point Distribution Tests
// =========================================================================

#[test]
fn test_clustered_points_2d() {
    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([10.0, 0.0]),
        vertex!([5.0, 10.0]),
    ];

    let mut dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

    // Insert 5 points clustered around (3.0, 3.0)
    dt.insert(vertex!([3.0, 3.0])).unwrap();
    dt.insert(vertex!([3.1, 3.0])).unwrap();
    dt.insert(vertex!([3.0, 3.1])).unwrap();
    dt.insert(vertex!([3.1, 3.1])).unwrap();
    dt.insert(vertex!([3.05, 3.05])).unwrap();

    assert_eq!(dt.number_of_vertices(), 8);
    assert!(dt.number_of_simplices() > 3);
}

#[test]
fn test_grid_pattern_2d() {
    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([3.0, 0.0]),
        vertex!([1.5, 3.0]),
    ];

    let mut dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

    // Insert 4 points in a grid
    dt.insert(vertex!([1.0, 1.0])).unwrap();
    dt.insert(vertex!([2.0, 1.0])).unwrap();
    dt.insert(vertex!([1.0, 1.5])).unwrap();
    dt.insert(vertex!([2.0, 1.5])).unwrap();

    assert_eq!(dt.number_of_vertices(), 7);
    assert!(dt.number_of_simplices() > 4);
}

// =========================================================================
// Batch vs Incremental Construction Tests
// =========================================================================

#[test]
fn test_batch_vs_incremental_same_vertex_count() {
    let all_vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([4.0, 0.0]),
        vertex!([2.0, 4.0]),
        vertex!([1.0, 1.0]),
        vertex!([2.0, 1.0]),
        vertex!([3.0, 1.0]),
    ];

    // Batch construction
    let dt_batch: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::new_with_topology_guarantee(
            &all_vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

    // Incremental construction
    let initial = vec![
        vertex!([0.0, 0.0]),
        vertex!([4.0, 0.0]),
        vertex!([2.0, 4.0]),
    ];
    let mut dt_incremental: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::new_with_topology_guarantee(&initial, TopologyGuarantee::PLManifold)
            .unwrap();

    dt_incremental.insert(vertex!([1.0, 1.0])).unwrap();
    dt_incremental.insert(vertex!([2.0, 1.0])).unwrap();
    dt_incremental.insert(vertex!([3.0, 1.0])).unwrap();

    // Both should have same vertex count
    assert_eq!(
        dt_batch.number_of_vertices(),
        dt_incremental.number_of_vertices()
    );
    assert_eq!(dt_batch.number_of_vertices(), 6);

    // Both should produce valid triangulations
    assert!(dt_batch.number_of_simplices() > 0);
    assert!(dt_incremental.number_of_simplices() > 0);
}

#[test]
fn test_bulk_construction_skips_near_duplicate_3d() {
    // Use bulk construction with a near-duplicate inside the tolerance (1e-10).
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
        vertex!([0.2, 0.2, 0.2]),
        vertex!([0.2 + 1e-11, 0.2, 0.2]),
    ];

    let opts =
        ConstructionOptions::default().with_dedup_policy(DedupPolicy::Epsilon { tolerance: 1e-10 });
    let dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::new_with_options(&vertices, opts).unwrap();

    // The near-duplicate should be skipped, so only 5 unique vertices remain.
    assert_eq!(dt.number_of_vertices(), 5);
    assert!(dt.number_of_simplices() > 0);
}
// =========================================================================
// Edge Cases
// =========================================================================

#[test]
fn test_insert_at_centroid() {
    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([3.0, 0.0]),
        vertex!([1.5, 3.0]),
    ];

    let mut dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

    // Insert point at approximate centroid
    dt.insert(vertex!([1.5, 1.0])).unwrap();

    assert_eq!(dt.number_of_vertices(), 4);
    assert_eq!(dt.number_of_simplices(), 3);
}

#[test]
fn test_minimal_simplex_then_insert() {
    // Test with exactly D+1 vertices initially
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];

    let mut dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

    assert_eq!(dt.number_of_vertices(), 4);
    assert_eq!(dt.number_of_simplices(), 1);

    // Insert one point
    dt.insert(vertex!([0.25, 0.25, 0.25])).unwrap();

    assert_eq!(dt.number_of_vertices(), 5);
    assert!(dt.number_of_simplices() > 1);
}

// =========================================================================
// Coordinate Type Tests
// =========================================================================

#[test]
fn test_f32_coordinates() {
    let vertices = vec![
        vertex!([0.0f32, 0.0f32]),
        vertex!([1.0f32, 0.0f32]),
        vertex!([0.0f32, 1.0f32]),
    ];

    let mut dt: DelaunayTriangulation<AdaptiveKernel<f32>, (), (), 2> =
        DelaunayTriangulation::with_topology_guarantee(
            &AdaptiveKernel::new(),
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .unwrap();

    dt.insert(vertex!([0.3f32, 0.3f32])).unwrap();

    assert_eq!(dt.number_of_vertices(), 4);
    assert_eq!(dt.number_of_simplices(), 3);
}

// =========================================================================
// Bootstrap Phase Tests (empty() → incremental insertion)
// =========================================================================

/// Macro to generate bootstrap key stability tests across dimensions.
///
/// This verifies that `VertexKeys` returned during incremental construction
/// from an empty triangulation remain valid after the initial simplex is
/// created at D+1 vertices.
macro_rules! test_bootstrap_key_stability {
    ($dim:expr, [$($point:expr),+ $(,)?]) => {
        pastey::paste! {
            #[test]
            fn [<test_bootstrap_key_stability_ $dim d>]() {
                let mut dt: DelaunayTriangulation<_, (), (), $dim> =
                    DelaunayTriangulation::empty_with_topology_guarantee(
                        TopologyGuarantee::PLManifold,
                    );

                // Collect the test points
                let points = vec![$(vertex!($point)),+];
                assert_eq!(points.len(), $dim + 1, "Must provide exactly D+1 points");

                // Insert vertices incrementally and collect keys
                let mut keys = Vec::new();
                for (i, point) in points.iter().enumerate() {
                    let key = dt.insert(*point).unwrap();
                    keys.push(key);
                    assert_eq!(dt.number_of_vertices(), i + 1);

                    // Before D+1 vertices: no simplices
                    if i < $dim {
                        assert_eq!(dt.number_of_simplices(), 0);
                    } else {
                        // At D+1: simplex created
                        assert_eq!(dt.number_of_simplices(), 1);
                    }
                }

                // Verify all keys remain valid after simplex creation
                for (i, &key) in keys.iter().enumerate() {
                    let vertex = dt.tds().vertex(key);
                    assert!(vertex.is_some(),
                        "Key {} should remain valid after simplex creation", i);

                    // Verify vertex has correct coordinates (using approx for float comparison)
                    let actual_coords = vertex.unwrap().point().coords();
                    let expected_coords = points[i].point().coords();
                    for (&actual, &expected) in actual_coords.iter().zip(expected_coords.iter()) {
                        assert_relative_eq!(actual, expected, epsilon = 1e-12);
                    }
                }
            }
        }
    };
}

// Generate bootstrap tests for 2D-5D
test_bootstrap_key_stability!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);

test_bootstrap_key_stability!(
    3,
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]
);

test_bootstrap_key_stability!(
    4,
    [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]
);

test_bootstrap_key_stability!(
    5,
    [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0]
    ]
);

/// Regression test for stale `VertexKey` bug in initial simplex construction.
///
/// This test catches a bug where `Triangulation::insert()` returned a stale
/// `VertexKey` from the old `TDS` after the initial simplex construction replaced
/// the `TDS`. The bug was masked because `SlotMap` keys generated in sequence happen
/// to have the same values across different `SlotMap` instances.
///
/// This test breaks that lucky coincidence by:
/// 1. Creating vertices with explicit UUIDs
/// 2. Inserting them in a specific order during bootstrap
/// 3. Creating a reference TDS with the same vertices in a DIFFERENT order
/// 4. Verifying that the returned keys are actually valid in the final TDS
///
/// Without the fix, the returned key from the D+1 insertion would be from the
/// old TDS and lookups would fail or return wrong data.
#[test]
fn test_bootstrap_returns_valid_key_after_tds_rebuild() {
    // Create vertices with explicit UUIDs so we can track them
    let uuid1 = Uuid::new_v4();
    let uuid2 = Uuid::new_v4();
    let uuid3 = Uuid::new_v4();

    let v1 = Vertex::new_with_uuid(Point::try_from([0.0, 0.0]).unwrap(), uuid1, None);
    let v2 = Vertex::new_with_uuid(Point::try_from([1.0, 0.0]).unwrap(), uuid2, None);
    let v3 = Vertex::new_with_uuid(Point::try_from([0.0, 1.0]).unwrap(), uuid3, None);

    // Bootstrap insertion: vertices inserted in order v1, v2, v3
    let mut dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::empty_with_topology_guarantee(TopologyGuarantee::PLManifold);

    let key1 = dt.insert(v1).unwrap();
    assert_eq!(dt.number_of_vertices(), 1);
    assert_eq!(dt.number_of_simplices(), 0);

    let key2 = dt.insert(v2).unwrap();
    assert_eq!(dt.number_of_vertices(), 2);
    assert_eq!(dt.number_of_simplices(), 0);

    // D+1 vertex triggers TDS rebuild - this is where the bug occurred
    let key3 = dt.insert(v3).unwrap();
    assert_eq!(dt.number_of_vertices(), 3);
    assert_eq!(dt.number_of_simplices(), 1);

    // Verify all returned keys are valid in the final TDS
    let vertex1 = dt.tds().vertex(key1);
    assert!(
        vertex1.is_some(),
        "First key should be valid after simplex creation"
    );
    assert_eq!(
        vertex1.unwrap().uuid(),
        uuid1,
        "First key should map to correct vertex UUID"
    );

    let vertex2 = dt.tds().vertex(key2);
    assert!(
        vertex2.is_some(),
        "Second key should be valid after simplex creation"
    );
    assert_eq!(
        vertex2.unwrap().uuid(),
        uuid2,
        "Second key should map to correct vertex UUID"
    );

    let vertex3 = dt.tds().vertex(key3);
    assert!(
        vertex3.is_some(),
        "Third key (D+1) should be valid after simplex creation"
    );
    assert_eq!(
        vertex3.unwrap().uuid(),
        uuid3,
        "Third key should map to correct vertex UUID"
    );

    // Extra verification: keys should map to correct coordinates
    let coords1 = vertex1.unwrap().point().coords();
    assert_relative_eq!(coords1[0], 0.0, epsilon = 1e-12);
    assert_relative_eq!(coords1[1], 0.0, epsilon = 1e-12);

    let coords2 = vertex2.unwrap().point().coords();
    assert_relative_eq!(coords2[0], 1.0, epsilon = 1e-12);
    assert_relative_eq!(coords2[1], 0.0, epsilon = 1e-12);

    let coords3 = vertex3.unwrap().point().coords();
    assert_relative_eq!(coords3[0], 0.0, epsilon = 1e-12);
    assert_relative_eq!(coords3[1], 1.0, epsilon = 1e-12);
}
