//! Storage Backend Compatibility Tests
//!
//! This test suite verifies that both `DenseSlotMap` (default) and `SlotMap`
//! (when built with `--no-default-features`) storage backends work correctly with
//! the triangulation data structure.
//!
//! **NOTE**: All tests in this file are ignored by default because they are Phase 4
//! evaluation tests (not regression tests) and take ~92 seconds to run.
//!
//! ## Running Tests
//!
//! Test with default `DenseSlotMap` backend:
//! ```bash
//! cargo test --test storage_backend_compatibility -- --ignored
//! ```
//!
//! Test with `SlotMap` backend:
//! ```bash
//! cargo test --no-default-features --test storage_backend_compatibility -- --ignored
//! ```
//!
//! For faster validation in release mode:
//! ```bash
//! cargo test --release --test storage_backend_compatibility -- --ignored
//! ```
//!
//! ## Large-Scale Tests
//!
//! Large-Scale tests (Phase 4) can be gated with environment variables:
//! ```bash
//! RUN_LARGE_SCALE_TESTS=1 cargo test --test storage_backend_compatibility -- --ignored
//! ```
//!
//! Expected resource usage:
//! - 2D: ~900 vertices, <1s runtime, ~10MB memory
//! - 3D: ~900 vertices, ~2s runtime, ~50MB memory
//! - 4D: ~500 vertices, ~5s runtime, ~100MB memory
//! - 5D: ~256 vertices, ~10s runtime, ~150MB memory
//!
//! ## Purpose
//!
//! These tests ensure API compatibility and correctness across storage backends.
//! Tests cover all supported dimensions (2D-5D) with focus on operations where
//! storage backend differences matter most:
//! - Construction
//! - Vertex/cell iteration (where `DenseSlotMap` should excel)
//! - Neighbor relationships (pointer chasing performance)
//! - Large-scale operations
//! - Serialization/deserialization

use delaunay::assert_jaccard_gte;
use delaunay::core::util::extract_edge_set;
use delaunay::geometry::kernel::FastKernel;
use delaunay::prelude::triangulation::*;

// =============================================================================
// TEST GENERATION MACROS (reduces duplication across 2D-5D)
// =============================================================================

/// Generate basic construction tests for a given dimension.
macro_rules! test_construction {
    ($dim:expr, $name:ident, $vertices:expr) => {
        #[test]
        #[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
        fn $name() {
            let vertices: Vec<_> = $vertices;
            let dt = DelaunayTriangulation::<_, (), (), $dim>::new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ).unwrap();
            let tds = dt.tds();

            assert_eq!(tds.number_of_vertices(), vertices.len());
            assert_eq!(tds.number_of_cells(), 1);
        }
    };
}

/// Generate vertex iteration tests for a given dimension.
macro_rules! test_vertex_iteration {
    ($dim:expr, $name:ident, $vertices:expr) => {
        #[test]
        #[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
        fn $name() {
            let vertices: Vec<_> = $vertices;
            let dt = DelaunayTriangulation::<_, (), (), $dim>::new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ).unwrap();
            let tds = dt.tds();

            let vertex_count = tds.vertices().count();
            assert_eq!(vertex_count, vertices.len());

            for (_key, vertex) in tds.vertices() {
                let _point = vertex.point();
            }
        }
    };
}

/// Generate cell iteration tests for a given dimension.
macro_rules! test_cell_iteration {
    ($dim:expr, $name:ident, $vertices:expr, $expected_vertices_per_cell:expr) => {
        #[test]
        #[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
        fn $name() {
            let vertices: Vec<_> = $vertices;
            let dt = DelaunayTriangulation::<_, (), (), $dim>::new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ).unwrap();
            let tds = dt.tds();

            let cell_count = tds.cells().count();
            assert!(cell_count > 0);

            for (_key, cell) in tds.cells() {
                let vertex_keys = cell.vertices();
                assert_eq!(vertex_keys.len(), $expected_vertices_per_cell);
            }
        }
    };
}

/// Generate neighbor access tests with dimensional invariants.
macro_rules! test_neighbor_access {
    ($dim:expr, $name:ident, $vertices:expr) => {
        #[test]
        #[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
        fn $name() {
            let vertices: Vec<_> = $vertices;
            let dt = DelaunayTriangulation::<_, (), (), $dim>::new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ).unwrap();
            let tds = dt.tds();

            for (_key, cell) in tds.cells() {
                if let Some(neighbors) = cell.neighbors() {
                    // Count non-None neighbors
                    let neighbor_count = neighbors.iter().flatten().count();

                    // Verify all neighbors are valid
                    for neighbor_key in neighbors.iter().flatten() {
                        assert!(tds.get_cell(*neighbor_key).is_some(),
                                "Invalid neighbor reference in {}D", $dim);
                        assert_ne!(*neighbor_key, _key,
                                   "Self-referential neighbor in {}D", $dim);
                    }

                    // In a D-dimensional simplex, there are at most D+1 facets (neighbors)
                    assert!(neighbor_count <= $dim + 1,
                            "Too many neighbors: {} > {} in {}D", neighbor_count, $dim + 1, $dim);
                }
            }
        }
    };
}

/// Generate serialization tests for a given dimension.
///
/// Uses Jaccard similarity to verify edge topology preservation across serialization,
/// providing robust validation that handles potential ordering variations.
macro_rules! test_serialization {
    ($dim:expr, $name:ident, $vertices:expr) => {
        #[test]
        #[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
        fn $name() {
            let vertices: Vec<_> = $vertices;
            let dt = DelaunayTriangulation::<_, (), (), $dim>::new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ).unwrap();
            let tds = dt.tds();

            // Extract edge topology before serialization
            let edges_before = extract_edge_set(&tds).expect("edge extraction should not fail");

            let serialized = serde_json::to_string(&tds).expect("serialization failed");
            let deserialized: Tds<f64, (), (), $dim> =
                serde_json::from_str(&serialized).expect("deserialization failed");

            // Verify counts
            assert_eq!(deserialized.number_of_vertices(), tds.number_of_vertices());
            assert_eq!(deserialized.number_of_cells(), tds.number_of_cells());

            // Verify edge topology preservation via Jaccard similarity
            // Use strict threshold (0.999) as topology should be fully preserved
            let edges_after = extract_edge_set(&deserialized).expect("edge extraction should not fail");
            assert_jaccard_gte!(
                &edges_before,
                &edges_after,
                0.999,
                "{}D edge topology preservation through serialization (backend: {})",
                $dim,
                if cfg!(feature = "dense-slotmap") { "DenseSlotMap" } else { "SlotMap" }
            );
        }
    };
}

/// Generate vertex data storage tests.
macro_rules! test_vertex_data {
    ($dim:expr, $name:ident, $vertices:expr) => {
        #[test]
        #[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
        fn $name() {
            let vertices: Vec<_> = $vertices;
            let dt = DelaunayTriangulation::<FastKernel<f64>, Option<i32>, (), $dim>::with_topology_guarantee(
                &FastKernel::default(),
                &vertices,
                TopologyGuarantee::PLManifold,
            )
            .unwrap();
            let tds = dt.tds();

            for (_key, vertex) in tds.vertices() {
                assert!(vertex.data.is_some());
            }
        }
    };
}

/// Generate cell data storage tests.
macro_rules! test_cell_data {
    ($dim:expr, $name:ident, $vertices:expr) => {
        #[test]
        #[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
        fn $name() {
            let vertices: Vec<_> = $vertices;
            let dt = DelaunayTriangulation::<FastKernel<f64>, (), i32, $dim>::with_topology_guarantee(
                &FastKernel::default(),
                &vertices,
                TopologyGuarantee::PLManifold,
            )
            .unwrap();
            let mut tds = dt.tds().clone();

            // Collect cell keys first to avoid borrow checker issues
            let cell_keys: Vec<_> = tds.cells().map(|(key, _)| key).collect();
            for key in cell_keys {
                if let Some(cell) = tds.get_cell_by_key_mut(key) {
                    cell.data = Some(42);
                }
            }

            for (_key, cell) in tds.cells() {
                assert_eq!(cell.data, Some(42));
            }
        }
    };
}

// =============================================================================
// BASIC CONSTRUCTION TESTS (2D-5D)
// =============================================================================

test_construction!(
    2,
    test_storage_backend_construction_2d,
    vec![
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.5, 1.0]),
    ]
);

test_construction!(
    3,
    test_storage_backend_construction_3d,
    vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ]
);

test_construction!(
    4,
    test_storage_backend_construction_4d,
    vec![
        vertex!([0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0]),
    ]
);

test_construction!(
    5,
    test_storage_backend_construction_5d,
    vec![
        vertex!([0.0, 0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 0.0, 1.0]),
    ]
);

// =============================================================================
// ITERATION TESTS (where DenseSlotMap should excel)
// =============================================================================

test_vertex_iteration!(
    2,
    test_storage_backend_vertex_iteration_2d,
    vec![
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.5, 1.0]),
        vertex!([0.5, 0.5]),
    ]
);

test_cell_iteration!(
    3,
    test_storage_backend_cell_iteration_3d,
    vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
        vertex!([0.5, 0.5, 0.5]),
    ],
    4
);

test_vertex_iteration!(
    4,
    test_storage_backend_vertex_iteration_4d,
    vec![
        vertex!([0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0]),
        vertex!([0.5, 0.5, 0.5, 0.5]),
    ]
);

test_cell_iteration!(
    5,
    test_storage_backend_cell_iteration_5d,
    vec![
        vertex!([0.0, 0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 0.0, 1.0]),
        vertex!([0.3, 0.3, 0.3, 0.3, 0.3]),
    ],
    6
);

// =============================================================================
// NEIGHBOR ACCESS TESTS (2D-5D - tests pointer chasing performance)
// =============================================================================

test_neighbor_access!(
    2,
    test_storage_backend_neighbor_access_2d,
    vec![
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.5, 1.0]),
        vertex!([0.5, 0.5]),
    ]
);

test_neighbor_access!(
    3,
    test_storage_backend_neighbor_access_3d,
    vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
        vertex!([0.5, 0.5, 0.5]),
    ]
);

test_neighbor_access!(
    4,
    test_storage_backend_neighbor_access_4d,
    vec![
        vertex!([0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0]),
        vertex!([0.5, 0.5, 0.5, 0.5]),
    ]
);

test_neighbor_access!(
    5,
    test_storage_backend_neighbor_access_5d,
    vec![
        vertex!([0.0, 0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 0.0, 1.0]),
        vertex!([0.3, 0.3, 0.3, 0.3, 0.3]),
    ]
);

// =============================================================================
// LARGE-SCALE TESTS (stress storage backends across 2D-5D)
// =============================================================================
//
// These tests can be gated with RUN_LARGE_SCALE_TESTS=1 environment variable.
// Expected resource usage is documented in the module header.

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_large_scale_2d() {
    // Check environment gate (optional, for extra safety)
    if std::env::var("RUN_LARGE_SCALE_TESTS").ok().as_deref() != Some("1") {
        eprintln!("⚠️  Large-scale test skipped (set RUN_LARGE_SCALE_TESTS=1 to enable)");
        eprintln!("   Expected: ~900 vertices, <1s runtime, ~10MB memory");
        return;
    }

    let mut vertices = Vec::new();
    for i in 0..30 {
        for j in 0..30 {
            vertices.push(vertex!([f64::from(i) * 0.1, f64::from(j) * 0.1]));
        }
    }

    let dt = DelaunayTriangulation::<_, (), (), 2>::new_with_topology_guarantee(
        &vertices,
        TopologyGuarantee::PLManifold,
    )
    .unwrap();
    let tds = dt.tds();

    assert_eq!(tds.number_of_vertices(), 900);
    assert!(tds.number_of_cells() > 0);

    let vertex_count = tds.vertices().count();
    assert_eq!(vertex_count, 900);
}

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_large_scale_3d() {
    // Check environment gate (optional, for extra safety)
    if std::env::var("RUN_LARGE_SCALE_TESTS").ok().as_deref() != Some("1") {
        eprintln!("⚠️  Large-scale test skipped (set RUN_LARGE_SCALE_TESTS=1 to enable)");
        eprintln!("   Expected: ~900 vertices, ~2s runtime, ~50MB memory");
        return;
    }

    let mut vertices = Vec::new();
    for i in 0..15 {
        for j in 0..15 {
            for k in 0..4 {
                vertices.push(vertex!([
                    f64::from(i) * 0.1,
                    f64::from(j) * 0.1,
                    f64::from(k) * 0.1
                ]));
            }
        }
    }

    let dt = DelaunayTriangulation::<_, (), (), 3>::new_with_topology_guarantee(
        &vertices,
        TopologyGuarantee::PLManifold,
    )
    .unwrap();
    let tds = dt.tds();

    assert_eq!(tds.number_of_vertices(), 900);
    assert!(tds.number_of_cells() > 0);

    let vertex_count = tds.vertices().count();
    assert_eq!(vertex_count, 900);
}

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_large_scale_4d() {
    // Check environment gate (optional, for extra safety)
    if std::env::var("RUN_LARGE_SCALE_TESTS").ok().as_deref() != Some("1") {
        eprintln!("⚠️  Large-scale test skipped (set RUN_LARGE_SCALE_TESTS=1 to enable)");
        eprintln!("   Expected: ~500 vertices, ~5s runtime, ~100MB memory");
        return;
    }

    let mut vertices = Vec::new();
    for i in 0..10 {
        for j in 0..10 {
            for k in 0..5 {
                vertices.push(vertex!([
                    f64::from(i) * 0.1,
                    f64::from(j) * 0.1,
                    f64::from(k) * 0.1,
                    f64::from(i + j + k) * 0.05
                ]));
            }
        }
    }

    let dt = DelaunayTriangulation::<_, (), (), 4>::new_with_topology_guarantee(
        &vertices,
        TopologyGuarantee::PLManifold,
    )
    .unwrap();
    let tds = dt.tds();

    assert_eq!(tds.number_of_vertices(), 500);
    assert!(tds.number_of_cells() > 0);

    let vertex_count = tds.vertices().count();
    assert_eq!(vertex_count, 500);
}

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_large_scale_5d() {
    // Check environment gate (optional, for extra safety)
    if std::env::var("RUN_LARGE_SCALE_TESTS").ok().as_deref() != Some("1") {
        eprintln!("⚠️  Large-scale test skipped (set RUN_LARGE_SCALE_TESTS=1 to enable)");
        eprintln!("   Expected: ~256 vertices, ~10s runtime, ~150MB memory");
        return;
    }

    let mut vertices = Vec::new();
    for i in 0..8 {
        for j in 0..8 {
            for k in 0..4 {
                vertices.push(vertex!([
                    f64::from(i) * 0.1,
                    f64::from(j) * 0.1,
                    f64::from(k) * 0.1,
                    f64::from(i + j) * 0.05,
                    f64::from(i + k) * 0.05
                ]));
            }
        }
    }

    let dt = DelaunayTriangulation::<_, (), (), 5>::new_with_topology_guarantee(
        &vertices,
        TopologyGuarantee::PLManifold,
    )
    .unwrap();
    let tds = dt.tds();

    assert_eq!(tds.number_of_vertices(), 256);
    assert!(tds.number_of_cells() > 0);

    let vertex_count = tds.vertices().count();
    assert_eq!(vertex_count, 256);
}

// =============================================================================
// SERIALIZATION TESTS (2D-5D)
// =============================================================================

test_serialization!(
    2,
    test_storage_backend_serialization_2d,
    vec![
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.5, 1.0]),
    ]
);

test_serialization!(
    3,
    test_storage_backend_serialization_3d,
    vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ]
);

test_serialization!(
    4,
    test_storage_backend_serialization_4d,
    vec![
        vertex!([0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0]),
    ]
);

test_serialization!(
    5,
    test_storage_backend_serialization_5d,
    vec![
        vertex!([0.0, 0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 0.0, 1.0]),
    ]
);

// =============================================================================
// DATA STORAGE TESTS (2D-5D)
// =============================================================================

test_vertex_data!(
    2,
    test_storage_backend_vertex_data_2d,
    vec![
        vertex!([0.0, 0.0], Some(1)),
        vertex!([1.0, 0.0], Some(2)),
        vertex!([0.5, 1.0], Some(3)),
    ]
);

test_vertex_data!(
    3,
    test_storage_backend_vertex_data_3d,
    vec![
        vertex!([0.0, 0.0, 0.0], Some(1)),
        vertex!([1.0, 0.0, 0.0], Some(2)),
        vertex!([0.0, 1.0, 0.0], Some(3)),
        vertex!([0.0, 0.0, 1.0], Some(4)),
    ]
);

test_vertex_data!(
    4,
    test_storage_backend_vertex_data_4d,
    vec![
        vertex!([0.0, 0.0, 0.0, 0.0], Some(1)),
        vertex!([1.0, 0.0, 0.0, 0.0], Some(2)),
        vertex!([0.0, 1.0, 0.0, 0.0], Some(3)),
        vertex!([0.0, 0.0, 1.0, 0.0], Some(4)),
        vertex!([0.0, 0.0, 0.0, 1.0], Some(5)),
    ]
);

test_vertex_data!(
    5,
    test_storage_backend_vertex_data_5d,
    vec![
        vertex!([0.0, 0.0, 0.0, 0.0, 0.0], Some(1)),
        vertex!([1.0, 0.0, 0.0, 0.0, 0.0], Some(2)),
        vertex!([0.0, 1.0, 0.0, 0.0, 0.0], Some(3)),
        vertex!([0.0, 0.0, 1.0, 0.0, 0.0], Some(4)),
        vertex!([0.0, 0.0, 0.0, 1.0, 0.0], Some(5)),
        vertex!([0.0, 0.0, 0.0, 0.0, 1.0], Some(6)),
    ]
);

test_cell_data!(
    2,
    test_storage_backend_cell_data_2d,
    vec![
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.5, 1.0]),
    ]
);

test_cell_data!(
    3,
    test_storage_backend_cell_data_3d,
    vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ]
);

test_cell_data!(
    4,
    test_storage_backend_cell_data_4d,
    vec![
        vertex!([0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0]),
    ]
);

test_cell_data!(
    5,
    test_storage_backend_cell_data_5d,
    vec![
        vertex!([0.0, 0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 0.0, 1.0]),
    ]
);

// =============================================================================
// FEATURE FLAG VERIFICATION TESTS
// =============================================================================

#[cfg(feature = "dense-slotmap")]
#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_dense_slotmap_backend_active() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0]),
    ];

    let dt = DelaunayTriangulation::<_, (), (), 4>::new_with_topology_guarantee(
        &vertices,
        TopologyGuarantee::PLManifold,
    )
    .unwrap();
    let tds = dt.tds();

    assert_eq!(tds.number_of_vertices(), 5);
    assert_eq!(tds.number_of_cells(), 1);

    eprintln!("✓ DenseSlotMap backend test passed (4D)");
}

#[cfg(not(feature = "dense-slotmap"))]
#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_slotmap_backend_active() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0]),
    ];

    let dt = DelaunayTriangulation::<_, (), (), 4>::new_with_topology_guarantee(
        &vertices,
        TopologyGuarantee::PLManifold,
    )
    .unwrap();
    let tds = dt.tds();

    assert_eq!(tds.number_of_vertices(), 5);
    assert_eq!(tds.number_of_cells(), 1);

    eprintln!("✓ SlotMap backend test passed (4D)");
}
