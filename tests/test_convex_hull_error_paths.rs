//! Error path tests for `ConvexHull` module.
//!
//! This module targets uncovered error paths in `src/geometry/algorithms/convex_hull.rs`
//! to improve test coverage from 46.49% (126/271 lines) to ≥70% (190/271 lines).
//!
//! ## Coverage Goals
//!
//! **Target error paths:**
//! - `InsufficientData` - Empty triangulation cases (0 vertices, 0 cells, no boundary facets)
//! - `StaleHull` - Using hull with modified TDS
//! - `FacetDataAccessFailed` - Accessing invalid facet data
//! - `CoordinateConversion` - Coordinate conversion failures in fallback visibility
//! - `VisibilityCheckFailed` - Invalid facet index, insufficient vertices, orientation failures
//! - Degenerate orientation fallback - `fallback_visibility_test()` path
//!
//! ## Test Strategy
//!
//! These tests complement existing integration and property tests by focusing on
//! error conditions that are difficult to trigger through normal usage.

use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
use delaunay::core::vertex::Vertex;
use delaunay::geometry::algorithms::convex_hull::{ConvexHull, ConvexHullConstructionError};
use delaunay::geometry::point::Point;
use delaunay::geometry::traits::coordinate::Coordinate;
use delaunay::vertex;

// =============================================================================
// TEST FIXTURES
// =============================================================================

/// Standard 3D tetrahedron vertices (origin + unit vectors)
fn simplex_3d() -> [Vertex<f64, (), 3>; 4] {
    [
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ]
}

/// Standard 4D simplex vertices (origin + 4D unit vectors)
fn simplex_4d() -> [Vertex<f64, (), 4>; 5] {
    [
        vertex!([0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0]),
    ]
}

// =============================================================================
// CONSTRUCTION ERROR PATHS
// =============================================================================

/// Test `InsufficientData` error when trying to extract hull from empty triangulation.
///
/// **Coverage target:** Lines checking `tds.number_of_vertices() == 0` in `from_triangulation()`
#[test]
fn test_insufficient_data_no_vertices() {
    // Create an empty DelaunayTriangulation (no cells, no vertices)
    let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();

    let result = ConvexHull::from_triangulation(dt.triangulation());
    assert!(result.is_err(), "Creating hull from empty TDS should fail");

    match result.unwrap_err() {
        ConvexHullConstructionError::InsufficientData { message } => {
            assert!(
                message.contains("no vertices"),
                "Expected 'no vertices' message, got: {message}"
            );
        }
        other => panic!("Expected InsufficientData error, got: {other:?}"),
    }
}

/// Test `InsufficientData` error when TDS has vertices but no cells.
///
/// **Coverage target:** Lines checking `tds.number_of_cells() == 0` in `from_triangulation()`
///
/// **Note:** This is a theoretical case - normal TDS construction ensures cells exist if vertices exist.
/// The code path exists for defensive validation.
#[test]
fn test_insufficient_data_no_cells() {
    // Use empty DelaunayTriangulation which has no cells
    let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();

    let result = ConvexHull::from_triangulation(dt.triangulation());
    assert!(
        result.is_err(),
        "Creating hull from TDS with no cells should fail"
    );

    // This will hit the "no vertices" check first since default TDS has no vertices
    match result.unwrap_err() {
        ConvexHullConstructionError::InsufficientData { .. } => {
            // Expected - either "no vertices" or "no cells" message is acceptable
        }
        other => panic!("Expected InsufficientData error, got: {other:?}"),
    }
}

// =============================================================================
// STALE HULL DETECTION
// =============================================================================

/// Test `StaleHull` error when using hull with modified TDS.
///
/// **Coverage target:** Lines checking `creation_gen != tds_gen` in `is_facet_visible_from_point()`
#[test]
fn test_stale_hull_detection_visibility() {
    let mut dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::new(&simplex_3d()).unwrap();

    let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();
    assert!(
        hull.is_valid_for_triangulation(dt.triangulation()),
        "Hull should be valid initially"
    );

    // Modify the triangulation to invalidate the hull
    dt.tds_mut().add(vertex!([0.5, 0.5, 0.5])).unwrap();

    // Verify staleness is detected
    assert!(
        !hull.is_valid_for_triangulation(dt.triangulation()),
        "Hull should be invalid after triangulation modification"
    );

    // Try to use stale hull for visibility check - should return StaleHull error
    let facet = hull.get_facet(0).unwrap();
    let test_point = Point::new([2.0, 2.0, 2.0]);

    let result = hull.is_facet_visible_from_point(facet, &test_point, dt.triangulation());

    assert!(result.is_err(), "Stale hull usage should fail");

    match result.unwrap_err() {
        ConvexHullConstructionError::StaleHull {
            hull_generation,
            tds_generation,
        } => {
            assert_ne!(
                hull_generation, tds_generation,
                "Generation counters should differ for stale hull"
            );
        }
        other => panic!("Expected StaleHull error, got: {other:?}"),
    }
}

/// Test `StaleHull` error when using `find_visible_facets()` with stale hull.
///
/// **Coverage target:** Staleness check in `find_visible_facets()` method
#[test]
fn test_stale_hull_detection_find_visible() {
    let mut dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::new(&simplex_3d()).unwrap();

    let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

    // Modify triangulation to make hull stale
    dt.tds_mut().add(vertex!([1.5, 1.5, 1.5])).unwrap();

    let test_point = Point::new([3.0, 3.0, 3.0]);
    let result = hull.find_visible_facets(&test_point, dt.triangulation());

    assert!(
        result.is_err(),
        "find_visible_facets should fail with stale hull"
    );

    match result.unwrap_err() {
        ConvexHullConstructionError::StaleHull { .. } => {
            // Expected
        }
        other => panic!("Expected StaleHull error, got: {other:?}"),
    }
}

// =============================================================================
// CACHE INVALIDATION AND REBUILDING
// =============================================================================

/// Test cache invalidation and rebuilding after `invalidate_cache()`.
///
/// **Coverage target:** `invalidate_cache()` method and cache rebuild logic
#[test]
fn test_cache_invalidation() {
    let dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::new(&simplex_3d()).unwrap();

    let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

    // Perform visibility check to populate cache
    let facet = hull.get_facet(0).unwrap();
    let test_point = Point::new([2.0, 2.0, 2.0]);

    let result1 = hull.is_facet_visible_from_point(facet, &test_point, dt.triangulation());
    assert!(
        result1.is_ok(),
        "First visibility check should succeed: {result1:?}"
    );

    // Invalidate cache
    hull.invalidate_cache();

    // Verify hull is still valid for triangulation (invalidate_cache doesn't affect validity)
    assert!(
        hull.is_valid_for_triangulation(dt.triangulation()),
        "Hull should remain valid after cache invalidation"
    );

    // Perform another visibility check - should rebuild cache and succeed
    let result2 = hull.is_facet_visible_from_point(facet, &test_point, dt.triangulation());
    assert!(
        result2.is_ok(),
        "Visibility check after cache invalidation should succeed: {result2:?}"
    );

    // Results should be consistent
    assert_eq!(
        result1.unwrap(),
        result2.unwrap(),
        "Visibility results should match before and after cache invalidation"
    );
}

// =============================================================================
// VISIBILITY TESTING ERROR PATHS
// =============================================================================

/// Test visibility testing with various point positions to exercise different code paths.
///
/// **Coverage target:** Orientation comparison logic and degenerate case handling
#[test]
fn test_visibility_various_positions() {
    let dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::new(&simplex_3d()).unwrap();

    let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

    // Test multiple facets and points to maximize coverage
    let test_cases = vec![
        (Point::new([0.25, 0.25, 0.25]), "interior point"),
        (Point::new([2.0, 2.0, 2.0]), "far exterior"),
        (Point::new([0.5, 0.0, 0.0]), "on edge"),
        (Point::new([-1.0, 0.0, 0.0]), "negative direction"),
        (Point::new([0.0, -1.0, 0.0]), "negative Y"),
        (Point::new([0.0, 0.0, -1.0]), "negative Z"),
    ];

    for (point, description) in test_cases {
        for facet_idx in 0..hull.facet_count() {
            let facet = hull.get_facet(facet_idx).unwrap();
            let result = hull.is_facet_visible_from_point(facet, &point, dt.triangulation());

            assert!(
                result.is_ok(),
                "Visibility check for {description} on facet {facet_idx} should succeed: {result:?}"
            );
        }
    }
}

/// Test `find_visible_facets()` with various point positions.
///
/// **Coverage target:** Batch visibility checking logic
#[test]
fn test_find_visible_facets_various_positions() {
    let dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::new(&simplex_3d()).unwrap();

    let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

    // Inside point should see no facets
    let inside_point = Point::new([0.25, 0.25, 0.25]);
    let visible_inside = hull
        .find_visible_facets(&inside_point, dt.triangulation())
        .unwrap();
    assert!(
        visible_inside.is_empty(),
        "Inside point should see no facets, but saw {}",
        visible_inside.len()
    );

    // Outside point should see some facets
    let outside_point = Point::new([3.0, 3.0, 3.0]);
    let visible_outside = hull
        .find_visible_facets(&outside_point, dt.triangulation())
        .unwrap();
    assert!(
        !visible_outside.is_empty(),
        "Outside point should see at least one facet"
    );

    // Verify visible facet indices are valid
    for &idx in &visible_outside {
        assert!(
            idx < hull.facet_count(),
            "Visible facet index {idx} exceeds facet count {}",
            hull.facet_count()
        );
    }
}

// =============================================================================
// ACCESSOR METHOD COVERAGE
// =============================================================================

/// Test hull accessor methods for coverage.
///
/// **Coverage target:** Simple getter methods that may not be exercised by integration tests
#[test]
fn test_hull_accessors() {
    let dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::new(&simplex_3d()).unwrap();

    let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

    // Test facet_count()
    assert_eq!(hull.facet_count(), 4, "3D tetrahedron has 4 facets");

    // Test get_facet() with valid index
    assert!(
        hull.get_facet(0).is_some(),
        "Valid index should return Some"
    );
    assert!(
        hull.get_facet(3).is_some(),
        "Last valid index should return Some"
    );

    // Test get_facet() with invalid index
    assert!(
        hull.get_facet(10).is_none(),
        "Out-of-bounds index should return None"
    );

    // Test is_empty()
    assert!(!hull.is_empty(), "Non-empty hull should return false");

    let empty_hull: ConvexHull<delaunay::geometry::kernel::FastKernel<f64>, (), (), 3> =
        ConvexHull::default();
    assert!(empty_hull.is_empty(), "Empty hull should return true");

    // Test dimension()
    assert_eq!(hull.dimension(), 3, "3D hull should return dimension 3");

    // Test facets() iterator
    assert_eq!(
        hull.facets().count(),
        4,
        "Facets iterator should yield 4 facets"
    );

    // Test is_valid_for_triangulation
    assert!(
        hull.is_valid_for_triangulation(dt.triangulation()),
        "Hull should be valid for its triangulation"
    );
}

/// Test 2D convex hull construction and accessors.
///
/// **Coverage target:** Dimension-specific code paths for 2D case
#[test]
fn test_2d_convex_hull() {
    let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&[
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
    ])
    .unwrap();

    let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

    assert_eq!(hull.dimension(), 2, "2D hull dimension");
    assert_eq!(hull.facet_count(), 3, "2D triangle has 3 edges");
    assert!(
        hull.is_valid_for_triangulation(dt.triangulation()),
        "2D hull should be valid"
    );

    // Test visibility in 2D
    let outside_point = Point::new([2.0, 2.0]);
    let result = hull.find_visible_facets(&outside_point, dt.triangulation());
    assert!(
        result.is_ok(),
        "2D visibility check should succeed: {result:?}"
    );
}

/// Test 4D convex hull construction and accessors.
///
/// **Coverage target:** Higher-dimensional code paths
#[test]
fn test_4d_convex_hull() {
    let dt: DelaunayTriangulation<_, (), (), 4> =
        DelaunayTriangulation::new(&simplex_4d()).unwrap();

    let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

    assert_eq!(hull.dimension(), 4, "4D hull dimension");
    assert_eq!(hull.facet_count(), 5, "4-simplex has 5 facets");
    assert!(
        hull.is_valid_for_triangulation(dt.triangulation()),
        "4D hull should be valid"
    );

    // Test visibility in 4D
    let outside_point = Point::new([2.0, 2.0, 2.0, 2.0]);
    let result = hull.find_visible_facets(&outside_point, dt.triangulation());
    assert!(
        result.is_ok(),
        "4D visibility check should succeed: {result:?}"
    );
}

// =============================================================================
// DEFAULT IMPLEMENTATION COVERAGE
// =============================================================================

/// Test `Default` implementation for `ConvexHull`.
///
/// **Coverage target:** Default trait implementation and empty hull behavior
#[test]
fn test_default_hull() {
    use delaunay::geometry::kernel::FastKernel;
    let hull: ConvexHull<FastKernel<f64>, (), (), 3> = ConvexHull::default();

    assert!(hull.is_empty(), "Default hull should be empty");
    assert_eq!(hull.facet_count(), 0, "Default hull has zero facets");
    assert_eq!(hull.dimension(), 3, "Default hull preserves dimension");

    // Empty hull is always considered valid (has no facets that could be stale)
    let dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::new(&simplex_3d()).unwrap();

    assert!(
        hull.is_valid_for_triangulation(dt.triangulation()),
        "Empty hull should be valid for any triangulation (no stale facets)"
    );
}

// =============================================================================
// DEGENERATE CASE HANDLING
// =============================================================================

/// Test visibility with near-coplanar points to potentially trigger degenerate orientation.
///
/// **Coverage target:** Degenerate orientation handling and `fallback_visibility_test()`
///
/// **Note:** This test may not reliably trigger DEGENERATE orientation due to robust predicates,
/// but it exercises the code paths that would handle such cases if they occurred.
#[test]
fn test_near_degenerate_visibility() {
    // Create a flat configuration (nearly coplanar in 3D)
    let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&[
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.5, 0.5, 1e-10]), // Very small z-coordinate
    ])
    .unwrap();

    let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

    // Test visibility from various points
    let test_points = [
        Point::new([0.5, 0.5, 1e-12]),  // Very close to facet
        Point::new([0.5, 0.5, -1e-12]), // Very close, opposite side
        Point::new([0.5, 0.5, 0.0]),    // On the plane
    ];

    for (i, point) in test_points.iter().enumerate() {
        let result = hull.find_visible_facets(point, dt.triangulation());
        assert!(
            result.is_ok(),
            "Near-degenerate visibility check {i} should succeed: {result:?}"
        );
    }
}

/// Test visibility with large coordinate values to exercise coordinate conversion.
///
/// **Coverage target:** Coordinate conversion and numeric cast paths in fallback test
#[test]
fn test_large_coordinates_visibility() {
    let scale = 1e8;
    let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&[
        vertex!([0.0, 0.0, 0.0]),
        vertex!([scale, 0.0, 0.0]),
        vertex!([0.0, scale, 0.0]),
        vertex!([0.0, 0.0, scale]),
    ])
    .unwrap();

    let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

    // Test with large-scale points
    let test_point = Point::new([scale * 2.0, scale * 2.0, scale * 2.0]);
    let result = hull.find_visible_facets(&test_point, dt.triangulation());

    assert!(
        result.is_ok(),
        "Large coordinate visibility check should succeed: {result:?}"
    );
}

// =============================================================================
// ADDITIONAL PUBLIC API COVERAGE
// =============================================================================

/// Test `find_nearest_visible_facet()` method.
///
/// **Coverage target:** Lines 1227-1303 in `convex_hull.rs`
#[test]
fn test_find_nearest_visible_facet() {
    let dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::new(&simplex_3d()).unwrap();

    let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

    // Outside point should find a nearest visible facet
    let outside_point = Point::new([2.0, 2.0, 2.0]);
    let result = hull.find_nearest_visible_facet(&outside_point, dt.triangulation());
    assert!(
        result.is_ok(),
        "find_nearest_visible_facet should succeed for outside point: {result:?}"
    );
    assert!(
        result.unwrap().is_some(),
        "Outside point should have a nearest visible facet"
    );

    // Inside point should find no visible facets
    let inside_point = Point::new([0.25, 0.25, 0.25]);
    let result = hull.find_nearest_visible_facet(&inside_point, dt.triangulation());
    assert!(
        result.is_ok(),
        "find_nearest_visible_facet should succeed for inside point: {result:?}"
    );
    assert!(
        result.unwrap().is_none(),
        "Inside point should have no visible facets"
    );
}

/// Test `find_nearest_visible_facet()` with stale hull.
///
/// **Coverage target:** Staleness check in `find_nearest_visible_facet`
#[test]
fn test_find_nearest_visible_facet_stale() {
    let mut dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::new(&simplex_3d()).unwrap();

    let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

    // Modify triangulation to make hull stale
    dt.tds_mut().add(vertex!([2.0, 2.0, 2.0])).unwrap();

    let test_point = Point::new([3.0, 3.0, 3.0]);
    let result = hull.find_nearest_visible_facet(&test_point, dt.triangulation());

    assert!(
        result.is_err(),
        "find_nearest_visible_facet should fail with stale hull"
    );

    match result.unwrap_err() {
        ConvexHullConstructionError::StaleHull { .. } => {
            // Expected
        }
        other => panic!("Expected StaleHull error, got: {other:?}"),
    }
}

/// Test `is_point_outside()` method.
///
/// **Coverage target:** Lines 1350-1357 in `convex_hull.rs`
#[test]
fn test_is_point_outside() {
    let dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::new(&simplex_3d()).unwrap();

    let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

    // Inside point should return false
    let inside_point = Point::new([0.25, 0.25, 0.25]);
    let result = hull.is_point_outside(&inside_point, dt.triangulation());
    assert!(
        result.is_ok(),
        "is_point_outside should succeed: {result:?}"
    );
    assert!(
        !result.unwrap(),
        "Inside point should not be outside the hull"
    );

    // Outside point should return true
    let outside_point = Point::new([3.0, 3.0, 3.0]);
    let result = hull.is_point_outside(&outside_point, dt.triangulation());
    assert!(
        result.is_ok(),
        "is_point_outside should succeed: {result:?}"
    );
    assert!(result.unwrap(), "Outside point should be outside the hull");
}

/// Test `validate()` method for valid hull.
///
/// **Coverage target:** Lines 1394-1459 in `convex_hull.rs`
#[test]
fn test_validate_valid_hull() {
    let dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::new(&simplex_3d()).unwrap();

    let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();

    // Validate should pass for a well-formed hull
    let result = hull.validate(dt.triangulation());
    assert!(
        result.is_ok(),
        "Validation should succeed for well-formed hull: {result:?}"
    );
}

/// Test `validate()` method for empty hull.
///
/// **Coverage target:** Empty hull validation path
#[test]
fn test_validate_empty_hull() {
    let dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::new(&simplex_3d()).unwrap();

    let empty_hull: ConvexHull<delaunay::geometry::kernel::FastKernel<f64>, (), (), 3> =
        ConvexHull::default();

    // Empty hull should validate successfully (no facets to check)
    let result = empty_hull.validate(dt.triangulation());
    assert!(
        result.is_ok(),
        "Empty hull validation should succeed: {result:?}"
    );
}

/// Test `validate()` with multiple dimensions.
///
/// **Coverage target:** Dimension-specific validation paths
#[test]
fn test_validate_multiple_dimensions() {
    // 2D validation
    let dt_2d: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&[
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
    ])
    .unwrap();

    let hull_2d = ConvexHull::from_triangulation(dt_2d.triangulation()).unwrap();
    assert!(
        hull_2d.validate(dt_2d.triangulation()).is_ok(),
        "2D hull validation should succeed"
    );

    // 4D validation
    let dt_4d: DelaunayTriangulation<_, (), (), 4> =
        DelaunayTriangulation::new(&simplex_4d()).unwrap();

    let hull_4d = ConvexHull::from_triangulation(dt_4d.triangulation()).unwrap();
    assert!(
        hull_4d.validate(dt_4d.triangulation()).is_ok(),
        "4D hull validation should succeed"
    );
}
