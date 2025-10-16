#!/usr/bin/env cargo
//! # Zero-Allocation Iterator Demo
//!
//! This example demonstrates the performance benefits of using the new
//! zero-allocation `vertex_uuid_iter()` method compared to the traditional
//! `vertex_uuids()` method that allocates a Vec.

use delaunay::geometry::util::generate_random_triangulation;
use delaunay::prelude::*;
use std::hint::black_box;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=================================================================");
    println!("Zero-Allocation Iterator Demo");
    println!("=================================================================\n");

    // Create a triangulation and get a cell from it to demonstrate the iterator
    let tds: Tds<f64, (), (), 4> = generate_random_triangulation(
        50,          // Number of points (fewer for faster demo)
        (-5.0, 5.0), // Coordinate bounds
        None,        // No vertex data
        Some(42),    // Fixed seed for reproducibility
    )?;

    // Get the first cell from the triangulation
    let Some(cell) = tds.cells().values().next() else {
        eprintln!("No cells in triangulation; nothing to demo.");
        return Ok(());
    };
    println!(
        "Using a 4D cell with {} vertices from triangulation\n",
        cell.number_of_vertices()
    );

    // Demonstrate functional equivalence
    println!("Functional Equivalence Test:");
    println!("===========================");

    let vec_uuids = cell.vertex_uuids(&tds);
    let iter_uuids: Vec<_> = cell.vertex_uuid_iter(&tds).collect();

    println!("  vertex_uuids() returned {} UUIDs", vec_uuids.len());
    println!(
        "  vertex_uuid_iter().collect() returned {} UUIDs",
        iter_uuids.len()
    );
    println!("  Results are identical: {}", vec_uuids == iter_uuids);
    println!();

    // Performance comparison
    println!("Performance Comparison:");
    println!("======================");

    let iterations = 10_000;

    // Method 1: Allocating Vec (traditional)
    let start = Instant::now();
    let mut total_count = 0;
    for _ in 0..iterations {
        let uuids = cell.vertex_uuids(&tds); // Allocates Vec
        total_count += black_box(uuids.len()); // Prevent optimization
    }
    let vec_duration = start.elapsed();

    // Method 2: Zero-allocation iterator
    let start = Instant::now();
    let mut total_count_iter = 0;
    for _ in 0..iterations {
        let count = cell.vertex_uuid_iter(&tds).count(); // No allocation
        total_count_iter += black_box(count);
    }
    let iter_duration = start.elapsed();

    println!("  Method 1 (vertex_uuids): {vec_duration:>8.2?} ({iterations} iterations)");
    println!("  Method 2 (vertex_uuid_iter): {iter_duration:>8.2?} ({iterations} iterations)");

    // Handle speedup calculation with edge case for very fast operations
    if iter_duration.as_nanos() < 1000 || vec_duration.as_nanos() < 1000 {
        println!("  Speedup: N/A (iteration time too small to measure reliably)");
    } else {
        let speedup = vec_duration.as_secs_f64() / iter_duration.as_secs_f64();
        println!("  Speedup: {speedup:.2}x faster");
    }
    println!("  Counts match: {}", total_count == total_count_iter);
    println!();

    // Demonstrate iterator capabilities
    println!("Iterator Capabilities:");
    println!("=====================");

    // ExactSizeIterator
    let iter = cell.vertex_uuid_iter(&tds);
    println!("  Length via ExactSizeIterator: {}", iter.len());

    // Can be used in for loops
    let mut count = 0;
    for uuid in cell.vertex_uuid_iter(&tds) {
        if !uuid.is_nil() {
            count += 1;
        }
    }
    println!("  Non-nil UUIDs via for loop: {count}");

    // Can be chained with other iterator methods
    let valid_uuid_count = cell
        .vertex_uuid_iter(&tds)
        .filter(|uuid| !uuid.is_nil())
        .count();
    println!("  Valid UUIDs via iterator chain: {valid_uuid_count}");

    // Can be used with iterator combinators
    let first_few_count = cell.vertex_uuid_iter(&tds).take(3).count();
    println!("  First 3 UUIDs: {first_few_count} collected");

    println!("\n=================================================================");
    println!("Key Benefits of vertex_uuid_iter():");
    println!("- Zero heap allocations (no Vec created)");
    println!("- Implements ExactSizeIterator (O(1) len())");
    println!("- Full iterator trait support (map, filter, etc.)");
    println!("- Lazy evaluation (only compute what you need)");
    println!("- Better performance for iteration-only use cases");
    println!("=================================================================");

    Ok(())
}
