//! Integration tests for coherent-orientation invariants in the TDS layer.

#![forbid(unsafe_code)]

use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
use delaunay::core::triangulation_data_structure::{Tds, TdsValidationError};
use delaunay::vertex;

#[test]
fn test_tds_is_coherently_oriented_2d() {
    let vertices = [
        vertex!([0.0, 0.0]),
        vertex!([2.0, 0.0]),
        vertex!([0.0, 1.0]),
        vertex!([1.5, 1.2]),
    ];

    let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&vertices).unwrap();
    assert!(dt.tds().is_coherently_oriented());
    assert!(dt.tds().is_valid().is_ok());
}

#[test]
fn test_tds_is_coherently_oriented_3d() {
    let vertices = [
        vertex!([0.0, 0.0, 0.0]),
        vertex!([2.0, 0.0, 0.0]),
        vertex!([0.0, 2.0, 0.0]),
        vertex!([0.0, 0.0, 2.0]),
        vertex!([0.4, 0.4, 0.4]),
    ];

    let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    assert!(dt.tds().number_of_cells() >= 2);
    assert!(dt.tds().is_coherently_oriented());
    assert!(dt.tds().is_valid().is_ok());

    let mut serialized = serde_json::to_value(dt.tds()).unwrap();
    let cell_vertices_map = serialized
        .get_mut("cell_vertices")
        .and_then(serde_json::Value::as_object_mut)
        .unwrap();
    let first_cell_vertices = cell_vertices_map
        .values_mut()
        .next()
        .and_then(serde_json::Value::as_array_mut)
        .unwrap();
    assert!(first_cell_vertices.len() >= 4);
    first_cell_vertices.swap(0, 1);

    let tampered_json = serde_json::to_string(&serialized).unwrap();
    let tampered_tds: Tds<f64, (), (), 3> = serde_json::from_str(&tampered_json).unwrap();
    assert!(!tampered_tds.is_coherently_oriented());
    assert!(matches!(
        tampered_tds.is_valid(),
        Err(TdsValidationError::OrientationViolation { .. })
    ));
}

#[test]
fn test_tds_is_coherently_oriented_4d() {
    let vertices = [
        vertex!([0.0, 0.0, 0.0, 0.0]),
        vertex!([2.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 2.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 2.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 2.0]),
        vertex!([0.3, 0.3, 0.3, 0.3]),
    ];

    let dt: DelaunayTriangulation<_, (), (), 4> = DelaunayTriangulation::new(&vertices).unwrap();
    assert!(dt.tds().number_of_cells() >= 2);
    assert!(dt.tds().is_coherently_oriented());
    assert!(dt.tds().is_valid().is_ok());

    let mut serialized = serde_json::to_value(dt.tds()).unwrap();
    let cell_vertices_map = serialized
        .get_mut("cell_vertices")
        .and_then(serde_json::Value::as_object_mut)
        .unwrap();
    let first_cell_vertices = cell_vertices_map
        .values_mut()
        .next()
        .and_then(serde_json::Value::as_array_mut)
        .unwrap();
    assert!(first_cell_vertices.len() >= 5);
    first_cell_vertices.swap(0, 1);

    let tampered_json = serde_json::to_string(&serialized).unwrap();
    let tampered_tds: Tds<f64, (), (), 4> = serde_json::from_str(&tampered_json).unwrap();
    assert!(!tampered_tds.is_coherently_oriented());
    assert!(matches!(
        tampered_tds.is_valid(),
        Err(TdsValidationError::OrientationViolation { .. })
    ));
}

#[test]
fn test_tds_orientation_violation_detected_via_validation() {
    let vertices = [
        vertex!([0.0, 0.0]),
        vertex!([2.0, 0.0]),
        vertex!([0.0, 1.0]),
        vertex!([1.5, 1.2]),
    ];
    let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&vertices).unwrap();
    assert!(dt.tds().number_of_cells() >= 2);

    // Tamper with serialized cell vertex ordering to flip one cell orientation
    // while preserving the same combinatorial adjacency.
    let mut serialized = serde_json::to_value(dt.tds()).unwrap();
    let cell_vertices_map = serialized
        .get_mut("cell_vertices")
        .and_then(serde_json::Value::as_object_mut)
        .unwrap();
    let first_cell_vertices = cell_vertices_map
        .values_mut()
        .next()
        .and_then(serde_json::Value::as_array_mut)
        .unwrap();
    assert!(first_cell_vertices.len() >= 3);
    first_cell_vertices.swap(0, 1);

    let tampered_json = serde_json::to_string(&serialized).unwrap();
    let tampered_tds: Tds<f64, (), (), 2> = serde_json::from_str(&tampered_json).unwrap();
    assert!(!tampered_tds.is_coherently_oriented());
    assert!(matches!(
        tampered_tds.is_valid(),
        Err(TdsValidationError::OrientationViolation { .. })
    ));
}
