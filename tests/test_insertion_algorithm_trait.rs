//! Integration tests for `InsertionAlgorithm` trait public API
//!
//! This test module focuses on PUBLIC API methods that are not covered by unit tests
//! in the source module. The source module tests use private field access, while these
//! integration tests verify the public accessor methods work correctly.
//!
//! Tests cover:
//! - `InsertionBuffers` public accessor methods and compatibility helpers
//! - Error type conversions and display formatting through public interfaces
//!
//! Note: Basic functionality (new, reset, record methods) is covered by comprehensive
//! unit tests in `src/core/traits/insertion_algorithm.rs` and is not duplicated here.

use approx::assert_relative_eq;
use delaunay::core::algorithms::bowyer_watson::IncrementalBowyerWatson;
use delaunay::core::facet::FacetHandle;
use delaunay::core::traits::insertion_algorithm::{
    BadCellsError, InsertionAlgorithm, InsertionBuffers, InsertionError, InsertionStrategy,
};
use delaunay::core::triangulation_data_structure::{CellKey, Tds, TriangulationValidationError};
use delaunay::geometry::Coordinate;
use delaunay::vertex;
// These tests focus on public accessor methods that are not tested by unit tests
// in the source module (which use private field access)

// ============================================================================
// InsertionBuffers Tests
// ============================================================================

#[test]
fn test_insertion_buffers_new() {
    let buffers: InsertionBuffers<f64, (), (), 3> = InsertionBuffers::new();
    assert_eq!(buffers.bad_cells_buffer().len(), 0);
    assert_eq!(buffers.boundary_facets_buffer().len(), 0);
    assert_eq!(buffers.vertex_points_buffer().len(), 0);
    assert_eq!(buffers.visible_facets_buffer().len(), 0);
}

#[test]
fn test_insertion_buffers_default() {
    let buffers: InsertionBuffers<f64, (), (), 3> = InsertionBuffers::default();
    assert_eq!(buffers.bad_cells_buffer().len(), 0);
    assert_eq!(buffers.boundary_facets_buffer().len(), 0);
    assert_eq!(buffers.vertex_points_buffer().len(), 0);
    assert_eq!(buffers.visible_facets_buffer().len(), 0);
}

#[test]
fn test_insertion_buffers_with_capacity() {
    let buffers: InsertionBuffers<f64, (), (), 3> = InsertionBuffers::with_capacity(10);
    // Capacity is pre-allocated but length is 0
    assert_eq!(buffers.bad_cells_buffer().len(), 0);
    assert_eq!(buffers.boundary_facets_buffer().len(), 0);
    assert_eq!(buffers.vertex_points_buffer().len(), 0);
    assert_eq!(buffers.visible_facets_buffer().len(), 0);

    // Verify capacity is at least the requested amount (buffers may allocate more)
    assert!(buffers.bad_cells_buffer().capacity() >= 10);
    assert!(buffers.boundary_facets_buffer().capacity() >= 10);
    assert!(buffers.vertex_points_buffer().capacity() >= 10 * (3 + 1)); // D+1 points
}

#[test]
fn test_insertion_buffers_clear_all() {
    let mut buffers: InsertionBuffers<f64, (), (), 3> = InsertionBuffers::new();

    // Add some data using mutable accessors
    buffers.bad_cells_buffer_mut().push(CellKey::default());
    let dummy_handle = FacetHandle::new(CellKey::default(), 0);
    buffers.boundary_facets_buffer_mut().push(dummy_handle);
    buffers.visible_facets_buffer_mut().push(dummy_handle);

    assert_eq!(buffers.bad_cells_buffer().len(), 1);
    assert_eq!(buffers.boundary_facets_buffer().len(), 1);
    assert_eq!(buffers.visible_facets_buffer().len(), 1);

    buffers.clear_all();

    assert_eq!(buffers.bad_cells_buffer().len(), 0);
    assert_eq!(buffers.boundary_facets_buffer().len(), 0);
    assert_eq!(buffers.vertex_points_buffer().len(), 0);
    assert_eq!(buffers.visible_facets_buffer().len(), 0);
}

#[test]
fn test_insertion_buffers_prepare_methods() {
    let mut buffers: InsertionBuffers<f64, (), (), 3> = InsertionBuffers::new();

    // Add data to all buffers
    buffers.bad_cells_buffer_mut().push(CellKey::default());
    let dummy_handle = FacetHandle::new(CellKey::default(), 0);
    buffers.boundary_facets_buffer_mut().push(dummy_handle);
    buffers.visible_facets_buffer_mut().push(dummy_handle);

    // Prepare methods should clear and return mutable references
    let bad_cells = buffers.prepare_bad_cells_buffer();
    assert_eq!(bad_cells.len(), 0);
    bad_cells.push(CellKey::default());

    let boundary = buffers.prepare_boundary_facets_buffer();
    assert_eq!(boundary.len(), 0);
    let dummy_handle = FacetHandle::new(CellKey::default(), 0);
    boundary.push(dummy_handle);

    let points = buffers.prepare_vertex_points_buffer();
    assert_eq!(points.len(), 0);

    let visible = buffers.prepare_visible_facets_buffer();
    assert_eq!(visible.len(), 0);
}

#[test]
fn test_insertion_buffers_bad_cells_as_vec() {
    let mut buffers: InsertionBuffers<f64, (), (), 3> = InsertionBuffers::new();

    let empty_vec = buffers.bad_cells_as_vec();
    assert!(empty_vec.is_empty());

    buffers.bad_cells_buffer_mut().push(CellKey::default());
    buffers.bad_cells_buffer_mut().push(CellKey::default());

    let vec = buffers.bad_cells_as_vec();
    assert_eq!(vec.len(), 2);
}

#[test]
fn test_insertion_buffers_set_bad_cells_from_vec() {
    let mut buffers: InsertionBuffers<f64, (), (), 3> = InsertionBuffers::new();

    let cells = vec![CellKey::default(), CellKey::default()];
    buffers.set_bad_cells_from_vec(&cells);

    assert_eq!(buffers.bad_cells_buffer().len(), 2);
}

#[test]
fn test_insertion_buffers_boundary_facet_handles() {
    let mut buffers: InsertionBuffers<f64, (), (), 3> = InsertionBuffers::new();

    let empty_handles = buffers.boundary_facet_handles();
    assert!(empty_handles.is_empty());

    let dummy_handle = FacetHandle::new(CellKey::default(), 0);
    buffers.boundary_facets_buffer_mut().push(dummy_handle);

    let handles = buffers.boundary_facet_handles();
    assert_eq!(handles.len(), 1);
}

#[test]
fn test_insertion_buffers_set_boundary_facet_handles() {
    let mut buffers: InsertionBuffers<f64, (), (), 3> = InsertionBuffers::new();

    let dummy_handle = FacetHandle::new(CellKey::default(), 0);
    let handles = vec![dummy_handle, dummy_handle];
    buffers.set_boundary_facet_handles(&handles);

    assert_eq!(buffers.boundary_facets_buffer().len(), 2);
}

#[test]
fn test_insertion_buffers_boundary_facets_as_views() {
    let buffers: InsertionBuffers<f64, (), (), 3> = InsertionBuffers::new();
    // Create a TDS with minimal vertices for testing
    let vertices = delaunay::core::vertex::Vertex::from_points(&[
        delaunay::geometry::point::Point::new([0.0, 0.0, 0.0]),
        delaunay::geometry::point::Point::new([1.0, 0.0, 0.0]),
        delaunay::geometry::point::Point::new([0.0, 1.0, 0.0]),
        delaunay::geometry::point::Point::new([0.0, 0.0, 1.0]),
    ]);
    let tds: Tds<f64, (), (), 3> = Tds::new(&vertices).expect("valid tetrahedron");

    // With empty buffers, should return empty vec
    let result = buffers.boundary_facets_as_views(&tds);
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
}

#[test]
fn test_insertion_buffers_visible_facet_handles() {
    let mut buffers: InsertionBuffers<f64, (), (), 3> = InsertionBuffers::new();

    let empty_handles = buffers.visible_facet_handles();
    assert!(empty_handles.is_empty());

    let dummy_handle = FacetHandle::new(CellKey::default(), 0);
    buffers.visible_facets_buffer_mut().push(dummy_handle);

    let handles = buffers.visible_facet_handles();
    assert_eq!(handles.len(), 1);
}

#[test]
fn test_insertion_buffers_set_visible_facet_handles() {
    let mut buffers: InsertionBuffers<f64, (), (), 3> = InsertionBuffers::new();

    let dummy_handle = FacetHandle::new(CellKey::default(), 0);
    let handles = vec![dummy_handle];
    buffers.set_visible_facet_handles(&handles);

    assert_eq!(buffers.visible_facets_buffer().len(), 1);
}

#[test]
fn test_insertion_buffers_visible_facets_as_views() {
    let buffers: InsertionBuffers<f64, (), (), 3> = InsertionBuffers::new();
    // Create a TDS with minimal vertices for testing
    let vertices = delaunay::core::vertex::Vertex::from_points(&[
        delaunay::geometry::point::Point::new([0.0, 0.0, 0.0]),
        delaunay::geometry::point::Point::new([1.0, 0.0, 0.0]),
        delaunay::geometry::point::Point::new([0.0, 1.0, 0.0]),
        delaunay::geometry::point::Point::new([0.0, 0.0, 1.0]),
    ]);
    let tds: Tds<f64, (), (), 3> = Tds::new(&vertices).expect("valid tetrahedron");

    // With empty buffers, should return empty vec
    let result = buffers.visible_facets_as_views(&tds);
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
}

// ============================================================================
// Error Type Tests
// ============================================================================

#[test]
fn test_insertion_error_geometric_failure() {
    let error = InsertionError::geometric_failure(
        "Cannot determine orientation",
        InsertionStrategy::CavityBased,
    );

    match &error {
        InsertionError::GeometricFailure {
            message,
            strategy_attempted,
        } => {
            assert_eq!(message, "Cannot determine orientation");
            assert_eq!(*strategy_attempted, InsertionStrategy::CavityBased);
        }
        _ => panic!("Expected GeometricFailure"),
    }

    assert!(error.is_recoverable());
    assert_eq!(
        error.attempted_strategy(),
        Some(InsertionStrategy::CavityBased)
    );
}

#[test]
fn test_insertion_error_invalid_vertex() {
    let error = InsertionError::invalid_vertex("Duplicate point");

    match &error {
        InsertionError::InvalidVertex { reason } => {
            assert_eq!(reason, "Duplicate point");
        }
        _ => panic!("Expected InvalidVertex"),
    }

    assert!(!error.is_recoverable());
    assert_eq!(error.attempted_strategy(), None);
}

#[test]
fn test_insertion_error_precision_failure() {
    let error = InsertionError::precision_failure(1e-10, 5);

    match error {
        InsertionError::PrecisionFailure {
            tolerance,
            perturbation_attempts,
        } => {
            assert_relative_eq!(tolerance, 1e-10);
            assert_eq!(perturbation_attempts, 5);
        }
        _ => panic!("Expected PrecisionFailure"),
    }

    assert!(error.is_recoverable());
    assert_eq!(error.attempted_strategy(), None);
}

#[test]
fn test_insertion_error_hull_extension_failure() {
    let error = InsertionError::hull_extension_failure("No visible facets");

    match &error {
        InsertionError::HullExtensionFailure { reason } => {
            assert_eq!(reason, "No visible facets");
        }
        _ => panic!("Expected HullExtensionFailure"),
    }

    assert!(!error.is_recoverable());
    assert_eq!(error.attempted_strategy(), None);
}

#[test]
fn test_insertion_error_from_bad_cells_error() {
    let bad_cells_error = BadCellsError::NoCells;
    let insertion_error: InsertionError = bad_cells_error.into();

    match insertion_error {
        InsertionError::BadCellsDetection(BadCellsError::NoCells) => {}
        _ => panic!("Expected BadCellsDetection(NoCells)"),
    }

    assert!(insertion_error.is_recoverable());
}

#[test]
fn test_insertion_error_from_triangulation_validation_error() {
    let validation_error = TriangulationValidationError::InconsistentDataStructure {
        message: "Invalid neighbor".to_string(),
    };
    let insertion_error: InsertionError = validation_error.into();

    match insertion_error {
        InsertionError::TriangulationState(_) => {}
        _ => panic!("Expected TriangulationState"),
    }

    assert!(!insertion_error.is_recoverable());
}

#[test]
fn test_insertion_error_is_recoverable() {
    // Recoverable errors
    let geom_error = InsertionError::geometric_failure("test", InsertionStrategy::Standard);
    assert!(geom_error.is_recoverable());

    let precision_error = InsertionError::precision_failure(1e-10, 3);
    assert!(precision_error.is_recoverable());

    let bad_cells_error: InsertionError = BadCellsError::NoCells.into();
    assert!(bad_cells_error.is_recoverable());

    // Non-recoverable errors
    let invalid_vertex = InsertionError::invalid_vertex("test");
    assert!(!invalid_vertex.is_recoverable());

    let hull_failure = InsertionError::hull_extension_failure("test");
    assert!(!hull_failure.is_recoverable());
}

#[test]
fn test_incremental_bowyer_watson_insert_vertex_is_transactional_on_duplicate() {
    // Build a simple 3D triangulation
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];
    let mut tds: Tds<f64, (), (), 3> = Tds::new(&vertices).unwrap();

    let mut algorithm = IncrementalBowyerWatson::<f64, (), (), 3>::new();

    let vertices_before = tds.number_of_vertices();
    let cells_before = tds.number_of_cells();

    // Attempt to insert a duplicate vertex (same coordinates as an existing vertex).
    let duplicate = vertex!([0.0, 0.0, 0.0]);
    let result = algorithm.insert_vertex(&mut tds, duplicate);

    // The algorithm should report a useful InvalidVertex error and leave the TDS unchanged.
    match result {
        Err(InsertionError::InvalidVertex { reason }) => {
            assert!(reason.contains("duplicate") || reason.contains("Duplicate"));
        }
        other => panic!("Expected InvalidVertex error for duplicate insertion, got {other:?}"),
    }

    assert_eq!(
        tds.number_of_vertices(),
        vertices_before,
        "insert_vertex must not change the number of vertices on error",
    );
    assert_eq!(
        tds.number_of_cells(),
        cells_before,
        "insert_vertex must not change the number of cells on error",
    );
    assert!(
        tds.is_valid().is_ok(),
        "TDS should remain structurally valid after a failed insertion",
    );
}

// Note: BadCellsError and TooManyDegenerateCellsError display tests are covered
// by unit tests in the source module and are not duplicated here

// ============================================================================
// InsertionInfo and InsertionStrategy Tests
// ============================================================================

// Note: InsertionStrategy and InsertionInfo tests are simple struct validation
// covered adequately by usage in other tests
