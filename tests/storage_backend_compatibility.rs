//! Storage Backend Compatibility Tests
//!
//! This test suite verifies that both `SlotMap` (default) and `DenseSlotMap` (feature flag)
//! storage backends work correctly with the triangulation data structure.
//!
//! **NOTE**: All tests in this file are ignored by default because they are Phase 4
//! evaluation tests (not regression tests) and take ~92 seconds to run.
//!
//! ## Running Tests
//!
//! Test with default `SlotMap` backend:
//! ```bash
//! cargo test --test storage_backend_compatibility -- --ignored
//! ```
//!
//! Test with `DenseSlotMap` backend:
//! ```bash
//! cargo test --test storage_backend_compatibility --features dense-slotmap -- --ignored
//! ```
//!
//! For faster validation in release mode:
//! ```bash
//! cargo test --release --test storage_backend_compatibility -- --ignored
//! ```
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

use delaunay::prelude::*;
use delaunay::vertex;

// =============================================================================
// BASIC CONSTRUCTION TESTS (2D-5D)
// =============================================================================

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_construction_2d() {
    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.5, 1.0]),
    ];

    let tds: Tds<f64, (), (), 2> = Tds::new(&vertices).unwrap();

    assert_eq!(tds.number_of_vertices(), 3);
    assert_eq!(tds.number_of_cells(), 1);
}

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_construction_3d() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];

    let tds: Tds<f64, (), (), 3> = Tds::new(&vertices).unwrap();

    assert_eq!(tds.number_of_vertices(), 4);
    assert_eq!(tds.number_of_cells(), 1);
}

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_construction_4d() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0]),
    ];

    let tds: Tds<f64, (), (), 4> = Tds::new(&vertices).unwrap();

    assert_eq!(tds.number_of_vertices(), 5);
    assert_eq!(tds.number_of_cells(), 1);
}

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_construction_5d() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 0.0, 1.0]),
    ];

    let tds: Tds<f64, (), (), 5> = Tds::new(&vertices).unwrap();

    assert_eq!(tds.number_of_vertices(), 6);
    assert_eq!(tds.number_of_cells(), 1);
}

// =============================================================================
// ITERATION TESTS (where DenseSlotMap should excel)
// =============================================================================

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_vertex_iteration_2d() {
    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.5, 1.0]),
        vertex!([0.5, 0.5]),
    ];

    let tds: Tds<f64, (), (), 2> = Tds::new(&vertices).unwrap();

    let vertex_count = tds.vertices().iter().count();
    assert_eq!(vertex_count, 4);

    for (_key, vertex) in tds.vertices() {
        let _point = vertex.point();
    }
}

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_cell_iteration_3d() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
        vertex!([0.5, 0.5, 0.5]),
    ];

    let tds: Tds<f64, (), (), 3> = Tds::new(&vertices).unwrap();

    let cell_count = tds.cells().iter().count();
    assert!(cell_count > 0);

    for (_key, cell) in tds.cells() {
        let vertex_keys = cell.vertices();
        assert_eq!(vertex_keys.len(), 4); // 3D simplices have 4 vertices
    }
}

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_vertex_iteration_4d() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0]),
        vertex!([0.5, 0.5, 0.5, 0.5]),
    ];

    let tds: Tds<f64, (), (), 4> = Tds::new(&vertices).unwrap();

    let vertex_count = tds.vertices().iter().count();
    assert_eq!(vertex_count, 6);

    for (_key, vertex) in tds.vertices() {
        let _point = vertex.point();
    }
}

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_cell_iteration_5d() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 0.0, 1.0]),
        vertex!([0.3, 0.3, 0.3, 0.3, 0.3]),
    ];

    let tds: Tds<f64, (), (), 5> = Tds::new(&vertices).unwrap();

    let cell_count = tds.cells().iter().count();
    assert!(cell_count > 0);

    for (_key, cell) in tds.cells() {
        let vertex_keys = cell.vertices();
        assert_eq!(vertex_keys.len(), 6); // 5D simplices have 6 vertices
    }
}

// =============================================================================
// NEIGHBOR ACCESS TESTS (2D-5D - tests pointer chasing performance)
// =============================================================================

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_neighbor_access_2d() {
    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.5, 1.0]),
        vertex!([0.5, 0.5]),
    ];

    let tds: Tds<f64, (), (), 2> = Tds::new(&vertices).unwrap();

    for (_key, cell) in tds.cells() {
        if let Some(neighbors) = cell.neighbors() {
            for neighbor_key in neighbors.iter().flatten() {
                assert!(tds.cells().get(*neighbor_key).is_some());
            }
        }
    }
}

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_neighbor_access_3d() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
        vertex!([0.5, 0.5, 0.5]),
    ];

    let tds: Tds<f64, (), (), 3> = Tds::new(&vertices).unwrap();

    for (_key, cell) in tds.cells() {
        if let Some(neighbors) = cell.neighbors() {
            for neighbor_key in neighbors.iter().flatten() {
                assert!(tds.cells().get(*neighbor_key).is_some());
            }
        }
    }
}

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_neighbor_access_4d() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0]),
        vertex!([0.5, 0.5, 0.5, 0.5]),
    ];

    let tds: Tds<f64, (), (), 4> = Tds::new(&vertices).unwrap();

    for (_key, cell) in tds.cells() {
        if let Some(neighbors) = cell.neighbors() {
            for neighbor_key in neighbors.iter().flatten() {
                assert!(tds.cells().get(*neighbor_key).is_some());
            }
        }
    }
}

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_neighbor_access_5d() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 0.0, 1.0]),
        vertex!([0.3, 0.3, 0.3, 0.3, 0.3]),
    ];

    let tds: Tds<f64, (), (), 5> = Tds::new(&vertices).unwrap();

    for (_key, cell) in tds.cells() {
        if let Some(neighbors) = cell.neighbors() {
            for neighbor_key in neighbors.iter().flatten() {
                assert!(tds.cells().get(*neighbor_key).is_some());
            }
        }
    }
}

// =============================================================================
// LARGE-SCALE TESTS (stress storage backends across 2D-5D)
// =============================================================================

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_large_scale_2d() {
    let mut vertices = Vec::new();
    for i in 0..30 {
        for j in 0..30 {
            vertices.push(vertex!([f64::from(i) * 0.1, f64::from(j) * 0.1]));
        }
    }

    let tds: Tds<f64, (), (), 2> = Tds::new(&vertices).unwrap();

    assert_eq!(tds.number_of_vertices(), 900);
    assert!(tds.number_of_cells() > 0);

    let vertex_count = tds.vertices().iter().count();
    assert_eq!(vertex_count, 900);
}

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_large_scale_3d() {
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

    let tds: Tds<f64, (), (), 3> = Tds::new(&vertices).unwrap();

    assert_eq!(tds.number_of_vertices(), 900);
    assert!(tds.number_of_cells() > 0);

    let vertex_count = tds.vertices().iter().count();
    assert_eq!(vertex_count, 900);
}

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_large_scale_4d() {
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

    let tds: Tds<f64, (), (), 4> = Tds::new(&vertices).unwrap();

    assert_eq!(tds.number_of_vertices(), 500);
    assert!(tds.number_of_cells() > 0);

    let vertex_count = tds.vertices().iter().count();
    assert_eq!(vertex_count, 500);
}

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_large_scale_5d() {
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

    let tds: Tds<f64, (), (), 5> = Tds::new(&vertices).unwrap();

    assert_eq!(tds.number_of_vertices(), 256);
    assert!(tds.number_of_cells() > 0);

    let vertex_count = tds.vertices().iter().count();
    assert_eq!(vertex_count, 256);
}

// =============================================================================
// SERIALIZATION TESTS (2D-5D)
// =============================================================================

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_serialization_2d() {
    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.5, 1.0]),
    ];

    let tds: Tds<f64, (), (), 2> = Tds::new(&vertices).unwrap();

    let serialized = serde_json::to_string(&tds).expect("serialization failed");
    let deserialized: Tds<f64, (), (), 2> =
        serde_json::from_str(&serialized).expect("deserialization failed");

    assert_eq!(deserialized.number_of_vertices(), tds.number_of_vertices());
    assert_eq!(deserialized.number_of_cells(), tds.number_of_cells());
}

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_serialization_3d() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];

    let tds: Tds<f64, (), (), 3> = Tds::new(&vertices).unwrap();

    let serialized = serde_json::to_string(&tds).expect("serialization failed");
    let deserialized: Tds<f64, (), (), 3> =
        serde_json::from_str(&serialized).expect("deserialization failed");

    assert_eq!(deserialized.number_of_vertices(), tds.number_of_vertices());
    assert_eq!(deserialized.number_of_cells(), tds.number_of_cells());
}

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_serialization_4d() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0]),
    ];

    let tds: Tds<f64, (), (), 4> = Tds::new(&vertices).unwrap();

    let serialized = serde_json::to_string(&tds).expect("serialization failed");
    let deserialized: Tds<f64, (), (), 4> =
        serde_json::from_str(&serialized).expect("deserialization failed");

    assert_eq!(deserialized.number_of_vertices(), tds.number_of_vertices());
    assert_eq!(deserialized.number_of_cells(), tds.number_of_cells());
}

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_serialization_5d() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 0.0, 1.0]),
    ];

    let tds: Tds<f64, (), (), 5> = Tds::new(&vertices).unwrap();

    let serialized = serde_json::to_string(&tds).expect("serialization failed");
    let deserialized: Tds<f64, (), (), 5> =
        serde_json::from_str(&serialized).expect("deserialization failed");

    assert_eq!(deserialized.number_of_vertices(), tds.number_of_vertices());
    assert_eq!(deserialized.number_of_cells(), tds.number_of_cells());
}

// =============================================================================
// DATA STORAGE TESTS (2D-5D)
// =============================================================================

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_vertex_data_2d() {
    let vertices = vec![
        vertex!([0.0, 0.0], Some(1)),
        vertex!([1.0, 0.0], Some(2)),
        vertex!([0.5, 1.0], Some(3)),
    ];

    let tds: Tds<f64, Option<i32>, (), 2> = Tds::new(&vertices).unwrap();

    for (_key, vertex) in tds.vertices() {
        assert!(vertex.data.is_some());
    }
}

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_vertex_data_3d() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0], Some(1)),
        vertex!([1.0, 0.0, 0.0], Some(2)),
        vertex!([0.0, 1.0, 0.0], Some(3)),
        vertex!([0.0, 0.0, 1.0], Some(4)),
    ];

    let tds: Tds<f64, Option<i32>, (), 3> = Tds::new(&vertices).unwrap();

    for (_key, vertex) in tds.vertices() {
        assert!(vertex.data.is_some());
    }
}

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_vertex_data_4d() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0], Some(1)),
        vertex!([1.0, 0.0, 0.0, 0.0], Some(2)),
        vertex!([0.0, 1.0, 0.0, 0.0], Some(3)),
        vertex!([0.0, 0.0, 1.0, 0.0], Some(4)),
        vertex!([0.0, 0.0, 0.0, 1.0], Some(5)),
    ];

    let tds: Tds<f64, Option<i32>, (), 4> = Tds::new(&vertices).unwrap();

    for (_key, vertex) in tds.vertices() {
        assert!(vertex.data.is_some());
    }
}

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_vertex_data_5d() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0, 0.0], Some(1)),
        vertex!([1.0, 0.0, 0.0, 0.0, 0.0], Some(2)),
        vertex!([0.0, 1.0, 0.0, 0.0, 0.0], Some(3)),
        vertex!([0.0, 0.0, 1.0, 0.0, 0.0], Some(4)),
        vertex!([0.0, 0.0, 0.0, 1.0, 0.0], Some(5)),
        vertex!([0.0, 0.0, 0.0, 0.0, 1.0], Some(6)),
    ];

    let tds: Tds<f64, Option<i32>, (), 5> = Tds::new(&vertices).unwrap();

    for (_key, vertex) in tds.vertices() {
        assert!(vertex.data.is_some());
    }
}

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_cell_data_2d() {
    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.5, 1.0]),
    ];

    let mut tds: Tds<f64, (), i32, 2> = Tds::new(&vertices).unwrap();

    for (_key, cell) in tds.cells_mut() {
        cell.data = Some(42);
    }

    for (_key, cell) in tds.cells() {
        assert_eq!(cell.data, Some(42));
    }
}

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_cell_data_3d() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];

    let mut tds: Tds<f64, (), i32, 3> = Tds::new(&vertices).unwrap();

    for (_key, cell) in tds.cells_mut() {
        cell.data = Some(42);
    }

    for (_key, cell) in tds.cells() {
        assert_eq!(cell.data, Some(42));
    }
}

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_cell_data_4d() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0]),
    ];

    let mut tds: Tds<f64, (), i32, 4> = Tds::new(&vertices).unwrap();

    for (_key, cell) in tds.cells_mut() {
        cell.data = Some(42);
    }

    for (_key, cell) in tds.cells() {
        assert_eq!(cell.data, Some(42));
    }
}

#[test]
#[ignore = "Phase 4 storage backend evaluation test - run with: cargo test --test storage_backend_compatibility -- --ignored"]
fn test_storage_backend_cell_data_5d() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 0.0, 1.0]),
    ];

    let mut tds: Tds<f64, (), i32, 5> = Tds::new(&vertices).unwrap();

    for (_key, cell) in tds.cells_mut() {
        cell.data = Some(42);
    }

    for (_key, cell) in tds.cells() {
        assert_eq!(cell.data, Some(42));
    }
}

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

    let tds: Tds<f64, (), (), 4> = Tds::new(&vertices).unwrap();

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

    let tds: Tds<f64, (), (), 4> = Tds::new(&vertices).unwrap();

    assert_eq!(tds.number_of_vertices(), 5);
    assert_eq!(tds.number_of_cells(), 1);

    eprintln!("✓ SlotMap backend test passed (4D)");
}
