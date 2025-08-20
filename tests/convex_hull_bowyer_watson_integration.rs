//! Integration tests for `ConvexHull` and Bowyer-Watson algorithm integration
//!
//! This module contains focused integration tests that verify the proper
//! integration between the `ConvexHull` module and the `IncrementalBoyerWatson`
//! algorithm using only the public API. Tests focus on:
//!
//! - Hull extension execution and correctness
//! - Mixed insertion strategies behavior
//! - Triangulation validity after hull extensions
//! - Algorithm statistics and cache behavior
//! - Performance characteristics of the integration

use delaunay::core::{
    algorithms::bowyer_watson::IncrementalBoyerWatson,
    traits::insertion_algorithm::{InsertionAlgorithm, InsertionStrategy},
    triangulation_data_structure::Tds,
};
use delaunay::vertex;

/// Helper function to count boundary facets (shared by 1 cell)
fn count_boundary_facets(tds: &Tds<f64, Option<()>, Option<()>, 3>) -> usize {
    tds.build_facet_to_cells_hashmap()
        .values()
        .filter(|cells| cells.len() == 1)
        .count()
}

/// Helper function to count internal facets (shared by 2 cells)
fn count_internal_facets(tds: &Tds<f64, Option<()>, Option<()>, 3>) -> usize {
    tds.build_facet_to_cells_hashmap()
        .values()
        .filter(|cells| cells.len() == 2)
        .count()
}

/// Helper function to count invalid facets (shared by 3+ cells)
fn count_invalid_facets(tds: &Tds<f64, Option<()>, Option<()>, 3>) -> usize {
    tds.build_facet_to_cells_hashmap()
        .values()
        .filter(|cells| cells.len() > 2)
        .count()
}

/// Helper function to analyze triangulation state
fn analyze_triangulation_state(tds: &Tds<f64, Option<()>, Option<()>, 3>, label: &str) {
    let boundary_count = count_boundary_facets(tds);
    let internal_count = count_internal_facets(tds);
    let invalid_count = count_invalid_facets(tds);

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
fn test_basic_hull_extension_execution() {
    println!("=== BASIC HULL EXTENSION EXECUTION TEST ===");

    let initial_vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];

    let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
    let mut algorithm = IncrementalBoyerWatson::new();

    analyze_triangulation_state(&tds, "before_hull_extension");

    // Verify initial state is valid
    assert!(
        tds.is_valid().is_ok(),
        "Initial tetrahedron should be valid"
    );

    let _initial_boundary_count = count_boundary_facets(&tds);
    let initial_cell_count = tds.number_of_cells();

    // Add a vertex that should trigger hull extension
    let exterior_vertex = vertex!([2.0, 0.0, 0.0]); // Far along X-axis
    println!("Adding exterior vertex: {:?}", exterior_vertex.point());

    // Execute the insertion
    let insertion_result = algorithm.insert_vertex(&mut tds, exterior_vertex);

    match insertion_result {
        Ok(info) => {
            println!("Hull extension successful:");
            println!("  Strategy used: {:?}", info.strategy);
            println!("  Cells created: {}", info.cells_created);
            println!("  Cells removed: {}", info.cells_removed);

            // Verify the strategy used
            assert_eq!(
                info.strategy,
                InsertionStrategy::HullExtension,
                "Should have used hull extension strategy"
            );

            // Verify cells were created but none removed (hull extension adds to boundary)
            assert!(info.cells_created > 0, "Should have created new cells");
            assert_eq!(
                info.cells_removed, 0,
                "Hull extension shouldn't remove cells"
            );

            // Verify triangulation is still valid
            assert!(
                tds.is_valid().is_ok(),
                "Triangulation should remain valid after hull extension"
            );

            // Verify cell count increased
            let final_cell_count = tds.number_of_cells();
            assert!(
                final_cell_count > initial_cell_count,
                "Cell count should increase after hull extension"
            );

            // Verify no invalid facet sharing
            let invalid_facets = count_invalid_facets(&tds);
            assert_eq!(
                invalid_facets, 0,
                "Should have no invalid facets after hull extension"
            );

            analyze_triangulation_state(&tds, "after_hull_extension");
        }
        Err(e) => panic!("Hull extension insertion failed: {e:?}"),
    }

    println!("✅ Basic hull extension executed successfully");
}

#[test]
fn test_multiple_hull_extensions() {
    println!("=== MULTIPLE HULL EXTENSIONS TEST ===");

    let initial_vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];

    let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
    let mut algorithm = IncrementalBoyerWatson::new();

    analyze_triangulation_state(&tds, "initial");

    // Add multiple exterior vertices to test sequential hull extensions
    let exterior_vertices = vec![
        (vertex!([2.0, 0.0, 0.0]), "+X direction"),
        (vertex!([0.0, 2.0, 0.0]), "+Y direction"),
        (vertex!([0.0, 0.0, 2.0]), "+Z direction"),
        (vertex!([-1.0, 0.0, 0.0]), "-X direction"),
    ];

    let mut total_cells_created = 0;
    let mut hull_extensions = 0;

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
                println!("  Result: {info:?}");

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

                // Verify triangulation remains valid after each insertion
                assert!(
                    tds.is_valid().is_ok(),
                    "Triangulation should remain valid after hull extension {}",
                    i + 1
                );

                // Verify no invalid facet sharing
                let invalid_facets = count_invalid_facets(&tds);
                assert_eq!(
                    invalid_facets,
                    0,
                    "Should have no invalid facets after hull extension {}",
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

    println!("Final statistics:");
    println!("  Hull extensions performed: {hull_extensions}");
    println!("  Total cells created: {total_cells_created}");
    println!("  Final triangulation: {} cells", tds.number_of_cells());

    let (insertions, created, removed) = algorithm.get_statistics();
    println!("  Algorithm stats: {insertions} insertions, {created} created, {removed} removed");

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

    println!("✅ Multiple hull extensions completed successfully");
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_mixed_insertion_strategies() {
    println!("=== MIXED INSERTION STRATEGIES TEST ===");

    let initial_vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];

    let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
    let mut algorithm = IncrementalBoyerWatson::new();

    // Mix of interior and exterior vertices
    let test_vertices = vec![
        (
            vertex!([0.25, 0.25, 0.25]),
            "interior",
            InsertionStrategy::CavityBased,
        ),
        (
            vertex!([2.0, 0.0, 0.0]),
            "exterior +X",
            InsertionStrategy::HullExtension,
        ),
        (
            vertex!([0.1, 0.1, 0.1]),
            "interior small",
            InsertionStrategy::CavityBased,
        ),
        (
            vertex!([0.0, 2.0, 0.0]),
            "exterior +Y",
            InsertionStrategy::HullExtension,
        ),
        (
            vertex!([0.4, 0.3, 0.2]),
            "interior large",
            InsertionStrategy::CavityBased,
        ),
        (
            vertex!([-1.0, -1.0, -1.0]),
            "exterior -XYZ",
            InsertionStrategy::HullExtension,
        ),
    ];

    let mut cavity_insertions = 0;
    let mut hull_insertions = 0;
    let mut fallback_insertions = 0;

    for (i, (vertex, description, _expected_strategy)) in test_vertices.iter().enumerate() {
        println!(
            "Inserting {} vertex {}: {} at {:?}",
            description,
            i + 1,
            description,
            vertex.point()
        );

        // We can't test strategy determination directly since it's private,
        // but we can verify it through the insertion result
        match algorithm.insert_vertex(&mut tds, *vertex) {
            Ok(info) => {
                println!("  Insertion result: {info:?}");

                // Count strategies used
                match info.strategy {
                    InsertionStrategy::CavityBased | InsertionStrategy::Standard => {
                        cavity_insertions += 1;
                    } // Standard is like cavity-based
                    InsertionStrategy::HullExtension => hull_insertions += 1,
                    InsertionStrategy::Fallback | InsertionStrategy::Perturbation => {
                        fallback_insertions += 1;
                    } // Count as fallback
                    InsertionStrategy::Skip => {} // Don't count skipped insertions
                }

                // Verify triangulation remains valid after each insertion
                assert!(
                    tds.is_valid().is_ok(),
                    "Triangulation should remain valid after insertion {}",
                    i + 1
                );

                // Verify no invalid facet sharing
                let invalid_facets = count_invalid_facets(&tds);
                assert_eq!(
                    invalid_facets,
                    0,
                    "Should have no invalid facets after insertion {}",
                    i + 1
                );
            }
            Err(e) => {
                // Some insertions might fail due to geometric constraints,
                // but we should still test that the algorithm handles this gracefully
                println!("  Insertion failed (this may be expected): {e:?}");
            }
        }

        analyze_triangulation_state(&tds, &format!("after_mixed_{}", i + 1));
    }

    println!("Final mixed strategy statistics:");
    println!("  Cavity-based insertions: {cavity_insertions}");
    println!("  Hull extension insertions: {hull_insertions}");
    println!("  Fallback insertions: {fallback_insertions}");
    println!("  Total cells: {}", tds.number_of_cells());

    // We should have used both strategies
    assert!(
        cavity_insertions > 0,
        "Should have performed cavity-based insertions"
    );
    assert!(
        hull_insertions > 0,
        "Should have performed hull extension insertions"
    );

    // Final triangulation should be valid
    assert!(
        tds.is_valid().is_ok(),
        "Final triangulation should be valid"
    );

    println!("✅ Mixed insertion strategies executed successfully");
}

#[test]
fn test_hull_cache_and_reset_behavior() {
    println!("=== HULL CACHE AND RESET BEHAVIOR TEST ===");

    let initial_vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];

    let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
    let mut algorithm = IncrementalBoyerWatson::new();

    // Perform some insertions to potentially populate internal caches
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

    println!("Statistics before reset:");
    println!("  Insertions: {insertions_before}");
    println!("  Cells created: {cells_created_before}");
    println!("  Cells removed: {cells_removed_before}");

    assert!(insertions_before > 0, "Should have performed insertions");
    assert!(cells_created_before > 0, "Should have created cells");

    // Reset the algorithm
    algorithm.reset();

    // Verify statistics are reset
    let (insertions_after, cells_created_after, cells_removed_after) = algorithm.get_statistics();

    println!("Statistics after reset:");
    println!("  Insertions: {insertions_after}");
    println!("  Cells created: {cells_created_after}");
    println!("  Cells removed: {cells_removed_after}");

    assert_eq!(insertions_after, 0, "Insertion count should be reset");
    assert_eq!(
        cells_created_after, 0,
        "Cells created count should be reset"
    );
    assert_eq!(
        cells_removed_after, 0,
        "Cells removed count should be reset"
    );

    // Verify we can still perform operations after reset
    let test_vertex = vertex!([0.0, 0.0, 2.0]);
    match algorithm.insert_vertex(&mut tds, test_vertex) {
        Ok(info) => {
            println!("Post-reset insertion successful: {info:?}");

            // Verify statistics are tracking again
            let (new_insertions, new_created, _new_removed) = algorithm.get_statistics();
            assert!(new_insertions > 0, "Should track insertions after reset");
            assert!(new_created > 0, "Should track cell creation after reset");
        }
        Err(e) => panic!("Post-reset insertion failed: {e:?}"),
    }

    println!("✅ Hull cache and reset behavior working correctly");
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_triangulation_validity_after_hull_extensions() {
    println!("=== TRIANGULATION VALIDITY AFTER HULL EXTENSIONS TEST ===");

    let initial_vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];

    let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
    let mut algorithm = IncrementalBoyerWatson::new();

    // Verify initial state is valid
    assert!(
        tds.is_valid().is_ok(),
        "Initial tetrahedron should be valid"
    );

    let initial_boundary_count = count_boundary_facets(&tds);
    let initial_internal_count = count_internal_facets(&tds);
    let initial_invalid_count = count_invalid_facets(&tds);

    println!("Initial state:");
    println!("  Boundary facets: {initial_boundary_count}");
    println!("  Internal facets: {initial_internal_count}");
    println!("  Invalid facets: {initial_invalid_count}");

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

    // Add several exterior vertices and check validity after each
    let exterior_vertices = vec![
        vertex!([1.5, 0.5, 0.5]), // Outside but close
        vertex!([3.0, 0.0, 0.0]), // Far along +X
        vertex!([0.0, 3.0, 0.0]), // Far along +Y
        vertex!([0.0, 0.0, 3.0]), // Far along +Z
    ];

    for (i, vertex) in exterior_vertices.iter().enumerate() {
        println!("Adding exterior vertex {}: {:?}", i + 1, vertex.point());

        match algorithm.insert_vertex(&mut tds, *vertex) {
            Ok(info) => {
                println!("  Hull extension result: {info:?}");

                // Verify strategy used
                assert_eq!(
                    info.strategy,
                    InsertionStrategy::HullExtension,
                    "Should use hull extension for exterior vertices"
                );

                // Check triangulation validity immediately after insertion
                match tds.is_valid() {
                    Ok(()) => println!("  ✅ Triangulation valid after insertion {}", i + 1),
                    Err(e) => panic!(
                        "❌ Triangulation invalid after insertion {}: {:?}",
                        i + 1,
                        e
                    ),
                }

                // Check facet sharing invariants
                let boundary_count = count_boundary_facets(&tds);
                let internal_count = count_internal_facets(&tds);
                let invalid_count = count_invalid_facets(&tds);

                println!("  Post-insertion state:");
                println!("    Boundary facets: {boundary_count}");
                println!("    Internal facets: {internal_count}");
                println!("    Invalid facets: {invalid_count}");

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

                // Verify cell count increased (hull extension adds cells)
                assert!(
                    info.cells_created > 0,
                    "Hull extension {} should create cells",
                    i + 1
                );
                assert_eq!(
                    info.cells_removed,
                    0,
                    "Hull extension {} should not remove cells",
                    i + 1
                );
            }
            Err(e) => panic!("Hull extension {} failed: {:?}", i + 1, e),
        }
    }

    // Final comprehensive validity check
    println!("Performing final comprehensive validity check...");

    match tds.is_valid() {
        Ok(()) => {
            println!("✅ Final triangulation is valid");

            let final_boundary_count = count_boundary_facets(&tds);
            let final_internal_count = count_internal_facets(&tds);
            let final_invalid_count = count_invalid_facets(&tds);

            println!("Final statistics:");
            println!("  Total vertices: {}", tds.number_of_vertices());
            println!("  Total cells: {}", tds.number_of_cells());
            println!("  Boundary facets: {final_boundary_count}");
            println!("  Internal facets: {final_internal_count}");
            println!("  Invalid facets: {final_invalid_count}");

            assert_eq!(
                final_invalid_count, 0,
                "Final triangulation should have no invalid facets"
            );
            assert!(
                final_boundary_count > initial_boundary_count,
                "Should have more boundary facets after hull extensions"
            );
        }
        Err(e) => panic!("❌ Final triangulation is invalid: {e:?}"),
    }

    println!("✅ All triangulation validity checks passed");
}
