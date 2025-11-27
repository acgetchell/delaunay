#![expect(deprecated)]
//! Configuration API tests for `RobustBowyerWatson` algorithm.
//!
//! This module documents and tests the configuration API for `RobustBowyerWatson`,
//! including different constructor variants and predicate configuration presets.
//!
//! ## Focus
//!
//! These tests serve as API documentation and contract validation:
//! - Constructor variants (`new()`, `with_config()`, `for_degenerate_cases()`)
//! - Configuration preset usage (`general_triangulation`, `high_precision`, `degenerate_robust`)
//! - Algorithm state management (`reset()`, statistics accumulation)
//! - Multi-dimensional configuration behavior
//!
//! **Note**: Coverage impact is minimal (+0%) as existing integration/property tests already
//! cover the implementation. These tests focus on documenting public API contracts.

use delaunay::core::algorithms::robust_bowyer_watson::RobustBowyerWatson;
use delaunay::core::traits::insertion_algorithm::InsertionAlgorithm;
use delaunay::core::triangulation_data_structure::Tds;
use delaunay::geometry::robust_predicates::config_presets;
use delaunay::vertex;

// =============================================================================
// PREDICATE CONFIGURATION TESTS
// =============================================================================

/// Test default constructor uses `general_triangulation` config.
///
/// **Coverage target:** `new()` method and default config initialization
#[test]
fn test_default_constructor_uses_general_config() {
    let algorithm: RobustBowyerWatson<f64, Option<()>, Option<()>, 3> = RobustBowyerWatson::new();

    // Verify initialization
    let (processed, created, removed) = algorithm.get_statistics();
    assert_eq!(
        processed, 0,
        "New algorithm should have 0 processed vertices"
    );
    assert_eq!(created, 0, "New algorithm should have 0 created cells");
    assert_eq!(removed, 0, "New algorithm should have 0 removed cells");
}

/// Test `with_config()` custom configuration constructor.
///
/// **Coverage target:** `with_config()` method with various config presets
#[test]
fn test_with_config_constructor() {
    // Test with high_precision config
    let config_hp = config_presets::high_precision::<f64>();
    let algorithm_hp: RobustBowyerWatson<f64, Option<()>, Option<()>, 3> =
        RobustBowyerWatson::with_config(config_hp);

    let (processed, created, removed) = algorithm_hp.get_statistics();
    assert_eq!(
        processed, 0,
        "Algorithm with high_precision config should start with 0 processed"
    );
    assert_eq!(
        created, 0,
        "Algorithm with high_precision config should start with 0 created"
    );
    assert_eq!(
        removed, 0,
        "Algorithm with high_precision config should start with 0 removed"
    );

    // Test with general_triangulation config
    let config_gen = config_presets::general_triangulation::<f64>();
    let algorithm_gen: RobustBowyerWatson<f64, Option<()>, Option<()>, 3> =
        RobustBowyerWatson::with_config(config_gen);

    let stats_gen = algorithm_gen.get_statistics();
    assert_eq!(
        stats_gen.0, 0,
        "Algorithm with general config should start with 0 processed"
    );
}

/// Test `for_degenerate_cases()` constructor.
///
/// **Coverage target:** `for_degenerate_cases()` method and `degenerate_robust` config
#[test]
fn test_for_degenerate_cases_constructor() {
    let algorithm: RobustBowyerWatson<f64, Option<()>, Option<()>, 3> =
        RobustBowyerWatson::for_degenerate_cases();

    let (processed, created, removed) = algorithm.get_statistics();
    assert_eq!(
        processed, 0,
        "Degenerate-optimized algorithm should start with 0 processed"
    );
    assert_eq!(
        created, 0,
        "Degenerate-optimized algorithm should start with 0 created"
    );
    assert_eq!(
        removed, 0,
        "Degenerate-optimized algorithm should start with 0 removed"
    );
}

/// Test algorithm with custom tolerance configuration.
///
/// **Coverage target:** Custom `RobustPredicateConfig` usage
#[test]
fn test_custom_tolerance_configuration() {
    let mut custom_config = config_presets::general_triangulation::<f64>();

    // Modify tolerance for testing
    custom_config.base_tolerance = 1e-10;
    custom_config.relative_tolerance_factor = 1e-12;

    let mut algorithm: RobustBowyerWatson<f64, Option<()>, Option<()>, 3> =
        RobustBowyerWatson::with_config(custom_config);

    // Create simple TDS
    let initial_vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];
    let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();

    // Insert a test vertex
    let test_vertex = vertex!([0.5, 0.5, 0.5]);
    let result = algorithm.insert_vertex(&mut tds, test_vertex);

    assert!(
        result.is_ok(),
        "Insertion with custom tolerance should succeed: {result:?}"
    );
    assert!(tds.is_valid().is_ok(), "TDS should remain valid");
}

// =============================================================================
// MULTI-DIMENSIONAL CONFIGURATION TESTS
// =============================================================================

/// Test degenerate handling across dimensions with `for_degenerate_cases()`.
///
/// **Coverage target:** Degenerate config usage in various dimensions
#[test]
fn test_degenerate_config_2d() {
    let mut algorithm = RobustBowyerWatson::for_degenerate_cases();

    let initial_vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
    ];
    let mut tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&initial_vertices).unwrap();

    // Insert points with varying precision
    let test_points = [
        vertex!([0.5, 0.5]),
        vertex!([0.25, 0.25]),
        vertex!([0.75, 0.75]),
        vertex!([0.5 + 1e-14, 0.5 + 1e-14]), // Near-duplicate
    ];

    for (i, vertex) in test_points.iter().enumerate() {
        let _ = algorithm.insert_vertex(&mut tds, *vertex);

        assert!(
            tds.is_valid().is_ok(),
            "2D TDS should remain valid after degenerate insertion {i}"
        );
    }
}

/// Test `high_precision` config in 4D.
///
/// **Coverage target:** High precision config usage in higher dimensions
#[test]
fn test_high_precision_config_4d() {
    let config = config_presets::high_precision::<f64>();
    let mut algorithm: RobustBowyerWatson<f64, Option<()>, Option<()>, 4> =
        RobustBowyerWatson::with_config(config);

    let initial_vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0]),
    ];
    let mut tds: Tds<f64, Option<()>, Option<()>, 4> = Tds::new(&initial_vertices).unwrap();

    // Insert test vertices
    let test_vertex = vertex!([0.25, 0.25, 0.25, 0.25]);
    let result = algorithm.insert_vertex(&mut tds, test_vertex);

    assert!(
        result.is_ok(),
        "4D insertion with high_precision config should succeed: {result:?}"
    );
    assert!(tds.is_valid().is_ok(), "4D TDS should remain valid");
}

// =============================================================================
// ALGORITHM RESET AND REUSE TESTS
// =============================================================================

/// Test algorithm reset clears statistics but preserves configuration.
///
/// **Coverage target:** `reset()` method and state management
#[test]
fn test_algorithm_reset_preserves_config() {
    let mut algorithm = RobustBowyerWatson::for_degenerate_cases();

    let initial_vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];
    let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();

    // Insert some vertices
    let test_vertices = [vertex!([0.5, 0.5, 0.5]), vertex!([1.5, 1.5, 1.5])];

    for vertex in test_vertices {
        let _ = algorithm.insert_vertex(&mut tds, vertex);
    }

    let (processed_before, _, _) = algorithm.get_statistics();
    assert!(
        processed_before > 0,
        "Should have processed vertices before reset"
    );

    // Reset the algorithm
    algorithm.reset();

    let (processed_after, created_after, removed_after) = algorithm.get_statistics();
    assert_eq!(
        processed_after, 0,
        "Processed count should be 0 after reset"
    );
    assert_eq!(created_after, 0, "Created count should be 0 after reset");
    assert_eq!(removed_after, 0, "Removed count should be 0 after reset");

    // Verify algorithm still works with same config after reset
    let mut new_tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
    let test_vertex = vertex!([2.0, 2.0, 2.0]);
    let result = algorithm.insert_vertex(&mut new_tds, test_vertex);

    assert!(
        result.is_ok(),
        "Algorithm should work after reset: {result:?}"
    );
}

/// Test algorithm reuse with different TDS instances.
///
/// **Coverage target:** Algorithm state management across multiple TDS
#[test]
fn test_algorithm_reuse_different_tds() {
    let mut algorithm = RobustBowyerWatson::new();

    // First TDS
    let vertices1 = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];
    let mut tds1: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices1).unwrap();

    let _ = algorithm.insert_vertex(&mut tds1, vertex!([0.5, 0.5, 0.5]));

    let (processed1, _, _) = algorithm.get_statistics();
    assert!(
        processed1 > 0,
        "Should have processed vertices in first TDS"
    );

    // Second TDS without reset (same dimension)
    let vertices2 = vec![
        vertex!([10.0, 10.0, 10.0]),
        vertex!([11.0, 10.0, 10.0]),
        vertex!([10.0, 11.0, 10.0]),
        vertex!([10.0, 10.0, 11.0]),
    ];
    let mut tds2: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices2).unwrap();

    let _ = algorithm.insert_vertex(&mut tds2, vertex!([10.5, 10.5, 10.5]));

    let (processed2, _, _) = algorithm.get_statistics();
    assert!(
        processed2 > processed1,
        "Statistics should accumulate across TDS instances"
    );
}
