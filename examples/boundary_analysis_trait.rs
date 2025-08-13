//! Demonstration of the `BoundaryAnalysis` trait for triangulation boundary operations.
//!
//! This example shows how the boundary analysis functions have been cleanly separated
//! from the Tds struct using a trait-based approach, making the code more modular
//! and extensible.

use delaunay::delaunay_core::{
    traits::boundary_analysis::BoundaryAnalysis, triangulation_data_structure::Tds,
};
use delaunay::vertex;

fn main() {
    println!("🔺 Boundary Analysis Trait Demonstration\n");

    // Create a simple 3D triangulation (tetrahedron)
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];

    let tds: Tds<f64, Option<()>, Option<()>, 3> =
        Tds::new(&vertices).expect("Failed to create triangulation");

    println!("Created triangulation with:");
    println!("  - {} vertices", tds.number_of_vertices());
    println!("  - {} cells", tds.number_of_cells());
    println!("  - dimension: {}", tds.dim());

    // Now using the BoundaryAnalysis trait methods
    // Note: The trait is automatically in scope due to the prelude

    println!("\n📊 Boundary Analysis:");

    // Method 1: Get all boundary facets
    let boundary_facets = tds.boundary_facets();
    println!("  - Found {} boundary facets", boundary_facets.len());

    // Method 2: Count boundary facets efficiently (without creating the full vector)
    let boundary_count = tds.number_of_boundary_facets();
    println!("  - Boundary facet count: {boundary_count}");

    // Method 3: Check if specific facets are boundary facets
    if let Some(cell) = tds.cells().values().next() {
        let facets = cell.facets();
        println!("  - Testing individual facets from the first cell:");

        for (i, facet) in facets.iter().enumerate() {
            let is_boundary = tds.is_boundary_facet(facet);
            println!(
                "    • Facet {}: {} boundary",
                i,
                if is_boundary { "IS" } else { "NOT" }
            );
        }
    }

    println!("\n✨ Benefits of the trait-based approach:");
    println!("  1. Clean separation of concerns");
    println!("  2. The Tds struct is no longer cluttered with boundary-specific methods");
    println!("  3. Easy to extend with additional boundary analysis algorithms");
    println!("  4. Traits can be implemented for other data structures");
    println!("  5. Better testability and modularity");

    // Demonstrate with a more complex triangulation
    println!("\n🔸 Complex Triangulation Example:");

    let complex_vertices = vec![
        vertex!([0.0, 0.0, 0.0]),  // A
        vertex!([1.0, 0.0, 0.0]),  // B
        vertex!([0.5, 1.0, 0.0]),  // C - forms base triangle ABC
        vertex!([0.5, 0.5, 1.0]),  // D - above base
        vertex!([0.5, 0.5, -1.0]), // E - below base
    ];

    let complex_tds: Tds<f64, Option<()>, Option<()>, 3> =
        Tds::new(&complex_vertices).expect("Failed to create complex triangulation");

    println!("Complex triangulation:");
    println!("  - {} vertices", complex_tds.number_of_vertices());
    println!("  - {} cells", complex_tds.number_of_cells());
    println!(
        "  - {} boundary facets",
        complex_tds.number_of_boundary_facets()
    );

    // The two tetrahedra share one facet, so:
    // Total facets = 2 * 4 = 8
    // Shared facets = 1 (counted twice above, so subtract 1)
    // Boundary facets = 8 - 1 - 1 = 6
    println!("  - Expected: 6 boundary facets (2 tetrahedra × 4 facets - 2 shared)");

    println!("\n🎯 The BoundaryAnalysis trait makes boundary operations:");
    println!("  • More discoverable (IDE autocompletion)");
    println!("  • Easier to test in isolation");
    println!("  • Extensible for future algorithms");
    println!("  • Consistent across different triangulation types");
}
