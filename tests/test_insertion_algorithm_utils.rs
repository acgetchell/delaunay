//! Integration tests for insertion algorithm utility types.
//!
//! Tests the `InsertionBuffers` and `InsertionStatistics` helper types that support
//! insertion algorithms in the Delaunay triangulation.

use approx::assert_relative_eq;
use delaunay::core::facet::FacetHandle;
use delaunay::core::traits::insertion_algorithm::{InsertionBuffers, InsertionStatistics};
use delaunay::core::triangulation_data_structure::{CellKey, Tds};
use delaunay::geometry::point::Point;
use delaunay::geometry::traits::coordinate::Coordinate;
use delaunay::vertex;

// ====================================================================================
// InsertionBuffers Tests
// ====================================================================================

#[test]
fn test_insertion_buffers_new() {
    let buffers: InsertionBuffers<f64, Option<()>, Option<()>, 3> = InsertionBuffers::new();

    // All buffers should be empty
    assert!(buffers.bad_cells_buffer().is_empty());
    assert!(buffers.boundary_facets_buffer().is_empty());
    assert!(buffers.vertex_points_buffer().is_empty());
    assert!(buffers.visible_facets_buffer().is_empty());
}

#[test]
fn test_insertion_buffers_default() {
    let buffers: InsertionBuffers<f64, Option<()>, Option<()>, 3> = InsertionBuffers::default();

    // Default should behave like new()
    assert!(buffers.bad_cells_buffer().is_empty());
    assert!(buffers.boundary_facets_buffer().is_empty());
    assert!(buffers.vertex_points_buffer().is_empty());
    assert!(buffers.visible_facets_buffer().is_empty());
}

#[test]
fn test_insertion_buffers_with_capacity() {
    let buffers: InsertionBuffers<f64, Option<()>, Option<()>, 3> =
        InsertionBuffers::with_capacity(10);

    // All buffers should be empty but have capacity
    assert!(buffers.bad_cells_buffer().is_empty());
    assert!(buffers.bad_cells_buffer().capacity() >= 10);

    assert!(buffers.boundary_facets_buffer().is_empty());
    assert!(buffers.boundary_facets_buffer().capacity() >= 10);

    assert!(buffers.vertex_points_buffer().is_empty());
    // Vertex points buffer has larger capacity (capacity * (D + 1))
    assert!(buffers.vertex_points_buffer().capacity() >= 40); // 10 * (3 + 1)

    assert!(buffers.visible_facets_buffer().is_empty());
    // Visible facets buffer has smaller capacity (capacity / 2, min 1)
    assert!(buffers.visible_facets_buffer().capacity() >= 5);
}

#[test]
fn test_insertion_buffers_clear_all() {
    let mut buffers: InsertionBuffers<f64, Option<()>, Option<()>, 3> = InsertionBuffers::new();

    // Add some data to all buffers
    buffers.bad_cells_buffer_mut().push(CellKey::default());
    buffers
        .boundary_facets_buffer_mut()
        .push(FacetHandle::new(CellKey::default(), 0));
    buffers
        .vertex_points_buffer_mut()
        .push(Point::new([0.0, 0.0, 0.0]));
    buffers
        .visible_facets_buffer_mut()
        .push(FacetHandle::new(CellKey::default(), 0));

    // Verify data was added
    assert!(!buffers.bad_cells_buffer().is_empty());
    assert!(!buffers.boundary_facets_buffer().is_empty());
    assert!(!buffers.vertex_points_buffer().is_empty());
    assert!(!buffers.visible_facets_buffer().is_empty());

    // Clear all buffers
    buffers.clear_all();

    // All buffers should be empty again
    assert!(buffers.bad_cells_buffer().is_empty());
    assert!(buffers.boundary_facets_buffer().is_empty());
    assert!(buffers.vertex_points_buffer().is_empty());
    assert!(buffers.visible_facets_buffer().is_empty());
}

#[test]
fn test_insertion_buffers_prepare_methods() {
    let mut buffers: InsertionBuffers<f64, Option<()>, Option<()>, 3> = InsertionBuffers::new();

    // Add data to all buffers
    buffers.bad_cells_buffer_mut().push(CellKey::default());
    buffers
        .boundary_facets_buffer_mut()
        .push(FacetHandle::new(CellKey::default(), 0));
    buffers
        .vertex_points_buffer_mut()
        .push(Point::new([0.0, 0.0, 0.0]));
    buffers
        .visible_facets_buffer_mut()
        .push(FacetHandle::new(CellKey::default(), 0));

    // Prepare methods should clear and return mutable references
    {
        let bad_cells_buf = buffers.prepare_bad_cells_buffer();
        assert!(bad_cells_buf.is_empty());
        bad_cells_buf.push(CellKey::default());
    }

    {
        let boundary_buf = buffers.prepare_boundary_facets_buffer();
        assert!(boundary_buf.is_empty());
        boundary_buf.push(FacetHandle::new(CellKey::default(), 0));
    }

    {
        let vertex_points_buf = buffers.prepare_vertex_points_buffer();
        assert!(vertex_points_buf.is_empty());
        vertex_points_buf.push(Point::new([1.0, 1.0, 1.0]));
    }

    {
        let visible_buf = buffers.prepare_visible_facets_buffer();
        assert!(visible_buf.is_empty());
        visible_buf.push(FacetHandle::new(CellKey::default(), 1));
    }

    // Verify data was added through prepare methods
    assert_eq!(buffers.bad_cells_buffer().len(), 1);
    assert_eq!(buffers.boundary_facets_buffer().len(), 1);
    assert_eq!(buffers.vertex_points_buffer().len(), 1);
    assert_eq!(buffers.visible_facets_buffer().len(), 1);
}

#[test]
fn test_insertion_buffers_vec_compatibility() {
    let mut buffers: InsertionBuffers<f64, Option<()>, Option<()>, 3> = InsertionBuffers::new();

    // Test bad_cells_as_vec and set_bad_cells_from_vec
    let cell_keys = vec![CellKey::default(), CellKey::default()];
    buffers.set_bad_cells_from_vec(cell_keys);
    let retrieved = buffers.bad_cells_as_vec();
    assert_eq!(retrieved.len(), 2);

    // Test boundary_facet_handles and set_boundary_facet_handles
    let facet_handles = vec![
        FacetHandle::new(CellKey::default(), 0),
        FacetHandle::new(CellKey::default(), 1),
    ];
    buffers.set_boundary_facet_handles(facet_handles);
    let retrieved_facets = buffers.boundary_facet_handles();
    assert_eq!(retrieved_facets.len(), 2);

    // Test vertex_points_as_vec and set_vertex_points_from_vec
    let points = vec![Point::new([0.0, 0.0, 0.0]), Point::new([1.0, 1.0, 1.0])];
    buffers.set_vertex_points_from_vec(points);
    let retrieved_points = buffers.vertex_points_as_vec();
    assert_eq!(retrieved_points.len(), 2);

    // Test visible_facet_handles and set_visible_facet_handles
    let visible_handles = vec![
        FacetHandle::new(CellKey::default(), 0),
        FacetHandle::new(CellKey::default(), 2),
    ];
    buffers.set_visible_facet_handles(visible_handles);
    let retrieved_visible = buffers.visible_facet_handles();
    assert_eq!(retrieved_visible.len(), 2);
}

#[test]
fn test_insertion_buffers_facet_views() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];
    let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

    let mut buffers: InsertionBuffers<f64, Option<()>, Option<()>, 3> = InsertionBuffers::new();

    // Get a valid cell key from the triangulation
    let (cell_key, _cell) = tds.cells().next().unwrap();

    // Test boundary_facets_as_views
    let facet_handle = FacetHandle::new(cell_key, 0);
    buffers.set_boundary_facet_handles(vec![facet_handle]);

    let facet_views = buffers.boundary_facets_as_views(&tds).unwrap();
    assert_eq!(facet_views.len(), 1);
    assert_eq!(facet_views[0].cell_key(), cell_key);
    assert_eq!(facet_views[0].facet_index(), 0);

    // Test visible_facets_as_views
    let visible_handle = FacetHandle::new(cell_key, 1);
    buffers.set_visible_facet_handles(vec![visible_handle]);

    let visible_views = buffers.visible_facets_as_views(&tds).unwrap();
    assert_eq!(visible_views.len(), 1);
    assert_eq!(visible_views[0].cell_key(), cell_key);
    assert_eq!(visible_views[0].facet_index(), 1);
}

// ====================================================================================
// InsertionStatistics Tests
// ====================================================================================

#[test]
fn test_insertion_statistics_new() {
    let stats = InsertionStatistics::new();

    let (processed, created, removed) = stats.as_basic_tuple();
    assert_eq!(processed, 0);
    assert_eq!(created, 0);
    assert_eq!(removed, 0);

    assert_relative_eq!(stats.fallback_usage_rate(), 0.0);
}

#[test]
fn test_insertion_statistics_record_vertex_insertion() {
    use delaunay::core::traits::insertion_algorithm::{InsertionInfo, InsertionStrategy};
    let mut stats = InsertionStatistics::new();

    // Record an insertion with 5 cells created, 2 cells removed
    let info1 = InsertionInfo {
        strategy: InsertionStrategy::CavityBased,
        cells_removed: 2,
        cells_created: 5,
        success: true,
        degenerate_case_handled: false,
    };
    stats.record_vertex_insertion(&info1);

    let (processed, created, removed) = stats.as_basic_tuple();
    assert_eq!(processed, 1);
    assert_eq!(created, 5);
    assert_eq!(removed, 2);

    // Record another insertion
    let info2 = InsertionInfo {
        strategy: InsertionStrategy::CavityBased,
        cells_removed: 1,
        cells_created: 3,
        success: true,
        degenerate_case_handled: false,
    };
    stats.record_vertex_insertion(&info2);

    let (processed, created, removed) = stats.as_basic_tuple();
    assert_eq!(processed, 2);
    assert_eq!(created, 8); // 5 + 3
    assert_eq!(removed, 3); // 2 + 1
}

#[test]
fn test_insertion_statistics_record_fallback() {
    use delaunay::core::traits::insertion_algorithm::{InsertionInfo, InsertionStrategy};
    let mut stats = InsertionStatistics::new();

    // Record some insertions
    let info1 = InsertionInfo {
        strategy: InsertionStrategy::CavityBased,
        cells_removed: 2,
        cells_created: 5,
        success: true,
        degenerate_case_handled: false,
    };
    stats.record_vertex_insertion(&info1);

    let info2 = InsertionInfo {
        strategy: InsertionStrategy::CavityBased,
        cells_removed: 1,
        cells_created: 3,
        success: true,
        degenerate_case_handled: false,
    };
    stats.record_vertex_insertion(&info2);

    // Initially no fallbacks
    assert_relative_eq!(stats.fallback_usage_rate(), 0.0);

    // Record a fallback
    stats.record_fallback_usage();

    // Fallback rate should be 1/2 = 0.5
    assert_relative_eq!(stats.fallback_usage_rate(), 0.5);

    // Record another insertion without fallback
    let info3 = InsertionInfo {
        strategy: InsertionStrategy::CavityBased,
        cells_removed: 1,
        cells_created: 2,
        success: true,
        degenerate_case_handled: false,
    };
    stats.record_vertex_insertion(&info3);

    // Fallback rate should be 1/3 ≈ 0.333...
    assert_relative_eq!(stats.fallback_usage_rate(), 1.0 / 3.0, epsilon = 1e-10);
}

#[test]
fn test_insertion_statistics_fallback_usage_rate_zero_processed() {
    let stats = InsertionStatistics::new();

    // With zero vertices processed, fallback rate should be 0.0
    assert_relative_eq!(stats.fallback_usage_rate(), 0.0);
}

#[test]
fn test_insertion_statistics_multiple_fallbacks() {
    use delaunay::core::traits::insertion_algorithm::{InsertionInfo, InsertionStrategy};
    let mut stats = InsertionStatistics::new();

    // Record 10 insertions
    for _ in 0..10 {
        let info = InsertionInfo {
            strategy: InsertionStrategy::CavityBased,
            cells_removed: 0,
            cells_created: 1,
            success: true,
            degenerate_case_handled: false,
        };
        stats.record_vertex_insertion(&info);
    }

    // Record 3 fallbacks
    stats.record_fallback_usage();
    stats.record_fallback_usage();
    stats.record_fallback_usage();

    // Fallback rate should be 3/10 = 0.3
    assert_relative_eq!(stats.fallback_usage_rate(), 0.3);
}

#[test]
fn test_insertion_statistics_get_individual_values() {
    use delaunay::core::traits::insertion_algorithm::{InsertionInfo, InsertionStrategy};
    let mut stats = InsertionStatistics::new();

    let info1 = InsertionInfo {
        strategy: InsertionStrategy::CavityBased,
        cells_removed: 5,
        cells_created: 10,
        success: true,
        degenerate_case_handled: false,
    };
    stats.record_vertex_insertion(&info1);

    let info2 = InsertionInfo {
        strategy: InsertionStrategy::CavityBased,
        cells_removed: 3,
        cells_created: 8,
        success: true,
        degenerate_case_handled: false,
    };
    stats.record_vertex_insertion(&info2);

    stats.record_fallback_usage();

    let (processed, created, removed) = stats.as_basic_tuple();

    assert_eq!(processed, 2);
    assert_eq!(created, 18);
    assert_eq!(removed, 8);
}
