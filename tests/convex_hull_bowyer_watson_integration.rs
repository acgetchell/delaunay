//! Integration tests for `ConvexHull` and Bowyer-Watson algorithm integration
//!
//! This module contains focused integration tests that verify the proper
//! integration between the `ConvexHull` module and the `IncrementalBowyerWatson`
//! algorithm using only the public API. Tests focus on:
//!
//! - Hull extension execution and correctness
//! - Mixed insertion strategies behavior
//! - Triangulation validity after hull extensions
//! - Algorithm statistics and cache behavior
//! - Performance characteristics of the integration

use delaunay::core::{
    algorithms::bowyer_watson::IncrementalBowyerWatson,
    traits::insertion_algorithm::{InsertionAlgorithm, InsertionStrategy},
    triangulation_data_structure::Tds,
};
use delaunay::vertex;

/// Helper function to analyze facet sharing from a single map computation
/// Returns (`boundary_count`, `internal_count`, `invalid_count`)
fn analyze_facet_sharing(tds: &Tds<f64, (), (), 3>) -> (usize, usize, usize) {
    let facet_to_cells = tds
        .build_facet_to_cells_map()
        .expect("facet map should build successfully in integration test");

    let mut boundary_count = 0;
    let mut internal_count = 0;
    let mut invalid_count = 0;

    for cells in facet_to_cells.values() {
        match cells.len() {
            1 => boundary_count += 1,
            2 => internal_count += 1,
            n if n > 2 => invalid_count += 1,
            _ => {} // 0 cells should not happen in a valid triangulation
        }
    }

    (boundary_count, internal_count, invalid_count)
}

/// Helper function to analyze triangulation state
fn analyze_triangulation_state(tds: &Tds<f64, (), (), 3>, label: &str) {
    // Use the optimized function that computes the map once
    let (boundary_count, internal_count, invalid_count) = analyze_facet_sharing(tds);

    println!(
        "  {} - Vertices: {}, Cells: {}",
        label,
        tds.number_of_vertices(),
        tds.number_of_cells()
    );
    println!(
        "  {label} - Boundary: {boundary_count}, Internal: {internal_count}, Invalid: {invalid_count}"
    );
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "Comprehensive integration scenario; length is deliberate for coverage and diagnostics"
)]
fn test_comprehensive_hull_extension_execution() {
    println!("=== COMPREHENSIVE HULL EXTENSION EXECUTION TEST ===");

    let initial_vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];

    let mut tds: Tds<f64, (), (), 3> = Tds::new(&initial_vertices).unwrap();
    let mut algorithm = IncrementalBowyerWatson::new();

    // Verify initial state is valid and analyze facet structure
    assert!(
        tds.is_valid().is_ok(),
        "Initial tetrahedron should be valid"
    );
    let (initial_boundary_count, initial_internal_count, initial_invalid_count) =
        analyze_facet_sharing(&tds);
    assert_eq!(
        initial_boundary_count, 4,
        "Tetrahedron should have 4 boundary facets"
    );
    assert_eq!(
        initial_internal_count, 0,
        "Tetrahedron should have 0 internal facets"
    );
    assert_eq!(
        initial_invalid_count, 0,
        "Tetrahedron should have 0 invalid facets"
    );

    analyze_triangulation_state(&tds, "initial");

    // Test multiple exterior vertices to comprehensively test hull extensions
    let exterior_vertices = [
        (vertex!([2.0, 0.0, 0.0]), "+X direction"),
        (vertex!([0.0, 2.0, 0.0]), "+Y direction"),
        (vertex!([0.0, 0.0, 2.0]), "+Z direction"),
        (vertex!([-2.0, 0.0, 0.0]), "-X direction"),
        (vertex!([3.0, 3.0, 3.0]), "far exterior"),
    ];

    let mut total_cells_created = 0;
    let mut hull_extensions = 0;
    let initial_cell_count = tds.number_of_cells();

    for (i, (vertex, description)) in exterior_vertices.iter().enumerate() {
        println!(
            "Inserting exterior vertex {}: {} at {:?}",
            i + 1,
            description,
            vertex.point()
        );

        let cells_before = tds.number_of_cells();

        match algorithm.insert_vertex(&mut tds, *vertex) {
            Ok(info) => {
                println!("  Hull extension result: {info:?}");

                // Verify strategy and cell operations
                assert_eq!(
                    info.strategy,
                    InsertionStrategy::HullExtension,
                    "Should use hull extension for exterior vertices"
                );
                assert!(
                    info.cells_created > 0,
                    "Should create cells during hull extension"
                );
                assert_eq!(
                    info.cells_removed, 0,
                    "Hull extension should not remove cells"
                );

                total_cells_created += info.cells_created;
                hull_extensions += 1;

                // Comprehensive validity checks after each insertion
                match tds.is_valid() {
                    Ok(()) => println!("  ✅ Triangulation valid after insertion {}", i + 1),
                    Err(e) => panic!(
                        "❌ Triangulation invalid after insertion {}: {:?}",
                        i + 1,
                        e
                    ),
                }

                // Check facet sharing invariants
                let (boundary_count, _internal_count, invalid_count) = analyze_facet_sharing(&tds);
                assert_eq!(
                    invalid_count,
                    0,
                    "Should have no invalid facets after hull extension {}",
                    i + 1
                );
                assert!(
                    boundary_count > 0,
                    "Should have boundary facets after hull extension {}",
                    i + 1
                );

                let cells_after = tds.number_of_cells();
                println!(
                    "  Cells: {} -> {} (+{})",
                    cells_before,
                    cells_after,
                    cells_after - cells_before
                );
            }
            Err(e) => panic!("Failed to insert exterior vertex {}: {:?}", i + 1, e),
        }

        analyze_triangulation_state(&tds, &format!("after_extension_{}", i + 1));
    }

    // Final comprehensive analysis
    println!("Final comprehensive analysis:");
    let (final_boundary_count, final_internal_count, final_invalid_count) =
        analyze_facet_sharing(&tds);
    let final_cell_count = tds.number_of_cells();
    let (insertions, created, removed) = algorithm.get_statistics();

    println!("  Hull extensions performed: {hull_extensions}");
    println!("  Total cells created: {total_cells_created}");
    println!("  Initial -> Final cells: {initial_cell_count} -> {final_cell_count}");
    println!("  Algorithm stats: {insertions} insertions, {created} created, {removed} removed");
    println!("  Boundary facets: {initial_boundary_count} -> {final_boundary_count}");
    println!("  Internal facets: {initial_internal_count} -> {final_internal_count}");
    println!("  Invalid facets: {final_invalid_count}");

    // Comprehensive assertions
    assert_eq!(
        hull_extensions,
        exterior_vertices.len(),
        "Should have performed one hull extension per exterior vertex"
    );
    assert!(
        total_cells_created > 0,
        "Should have created cells during hull extensions"
    );
    assert_eq!(
        removed, 0,
        "Hull extensions should not have removed any cells"
    );
    assert!(
        final_cell_count > initial_cell_count,
        "Final cell count should exceed initial"
    );
    assert!(
        final_boundary_count >= initial_boundary_count,
        "Boundary facets should not decrease after hull extensions"
    );
    assert_eq!(
        final_invalid_count, 0,
        "Final triangulation should have no invalid facets"
    );
    assert!(
        tds.is_valid().is_ok(),
        "Final triangulation should be valid"
    );

    println!("✅ Comprehensive hull extension execution completed successfully");
}

#[test]
fn test_mixed_insertion_strategies() {
    println!("=== MIXED INSERTION STRATEGIES TEST ===");

    let initial_vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];

    let mut tds: Tds<f64, (), (), 3> = Tds::new(&initial_vertices).unwrap();
    let mut algorithm = IncrementalBowyerWatson::new();

    // Focused test with minimal vertices to verify both main insertion strategies
    let test_vertices = vec![
        (vertex!([0.25, 0.25, 0.25]), "interior"),
        (vertex!([10.0, 10.0, 10.0]), "exterior"),
        (vertex!([0.4, 0.3, 0.2]), "interior large"),
    ];

    let mut cavity_insertions = 0;
    let mut hull_insertions = 0;

    for (vertex, description) in test_vertices {
        match algorithm.insert_vertex(&mut tds, vertex) {
            Ok(info) => {
                println!(
                    "Inserted {description} vertex: strategy={:?}",
                    info.strategy
                );

                // Count strategies used
                match info.strategy {
                    InsertionStrategy::CavityBased | InsertionStrategy::Standard => {
                        cavity_insertions += 1;
                    }
                    InsertionStrategy::HullExtension => hull_insertions += 1,
                    _ => {} // Other strategies not expected in this focused test
                }
            }
            Err(e) => println!("Insertion failed (acceptable): {e:?}"),
        }
    }

    // Essential verification: both main strategies should be exercised
    assert!(
        cavity_insertions > 0,
        "Should perform cavity-based insertions for interior vertices"
    );
    assert!(
        hull_insertions > 0,
        "Should perform hull extensions for exterior vertices"
    );
    assert!(
        tds.is_valid().is_ok(),
        "Final triangulation should be valid"
    );

    println!("✅ Mixed insertion strategies: {cavity_insertions} cavity, {hull_insertions} hull");
}

#[test]
fn test_algorithm_state_management() {
    println!("=== ALGORITHM STATE MANAGEMENT TEST ===");

    let initial_vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];

    let mut tds: Tds<f64, (), (), 3> = Tds::new(&initial_vertices).unwrap();
    let mut algorithm = IncrementalBowyerWatson::new();

    // Perform insertions to populate internal state and statistics
    let exterior_vertices = vec![vertex!([2.0, 0.0, 0.0]), vertex!([0.0, 2.0, 0.0])];

    for vertex in exterior_vertices {
        match algorithm.insert_vertex(&mut tds, vertex) {
            Ok(info) => {
                println!("Insertion successful: {info:?}");
                assert_eq!(info.strategy, InsertionStrategy::HullExtension);
            }
            Err(e) => panic!("Hull extension failed: {e:?}"),
        }
    }

    // Get statistics before reset
    let (insertions_before, cells_created_before, cells_removed_before) =
        algorithm.get_statistics();
    println!(
        "Statistics before reset: {insertions_before} insertions, {cells_created_before} created, {cells_removed_before} removed"
    );
    assert!(insertions_before > 0, "Should have performed insertions");
    assert!(cells_created_before > 0, "Should have created cells");

    // Reset the algorithm state
    algorithm.reset();

    // Verify statistics are reset
    let (insertions_after, cells_created_after, cells_removed_after) = algorithm.get_statistics();
    println!(
        "Statistics after reset: {insertions_after} insertions, {cells_created_after} created, {cells_removed_after} removed"
    );
    assert_eq!(insertions_after, 0, "Insertion count should be reset");
    assert_eq!(
        cells_created_after, 0,
        "Cells created count should be reset"
    );
    assert_eq!(
        cells_removed_after, 0,
        "Cells removed count should be reset"
    );

    // Verify algorithm functionality after reset
    let test_vertex = vertex!([0.0, 0.0, 2.0]);
    match algorithm.insert_vertex(&mut tds, test_vertex) {
        Ok(info) => {
            println!("Post-reset insertion successful: {info:?}");
            let (new_insertions, new_created, _new_removed) = algorithm.get_statistics();
            assert!(new_insertions > 0, "Should track insertions after reset");
            assert!(new_created > 0, "Should track cell creation after reset");
        }
        Err(e) => panic!("Post-reset insertion failed: {e:?}"),
    }

    println!("✅ Algorithm state management working correctly");
}
