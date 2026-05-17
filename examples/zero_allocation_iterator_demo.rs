#!/usr/bin/env cargo
//! # Zero-Allocation Iterator Demo
//!
//! This example demonstrates the performance benefits of using the new
//! zero-allocation `vertex_uuid_iter()` method compared to the traditional
//! `vertex_uuids()` method that allocates a Vec.

use delaunay::prelude::generators::generate_random_triangulation;
use delaunay::prelude::tds::SimplexValidationError;
use delaunay::prelude::triangulation::construction::{
    DelaunayTriangulation, DelaunayTriangulationConstructionError,
};
use std::hint::black_box;
use std::time::Instant;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
enum DemoError {
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    Simplex(#[from] SimplexValidationError),
}

fn main() -> Result<(), DemoError> {
    println!("=================================================================");
    println!("Zero-Allocation Iterator Demo");
    println!("=================================================================\n");

    // Create a triangulation and get a simplex from it to demonstrate the iterator.
    // NOTE: The (n_points, bounds, seed) triple matches the 4D configuration used in
    // `test_generate_random_triangulation_dimensions` so that this example exercises
    // a realistic 4D triangulation without triggering extreme Delaunay repair in CI.
    let dt: DelaunayTriangulation<_, (), (), 4> = generate_random_triangulation(
        12,          // Number of points (fewer for faster demo)
        (-1.0, 1.0), // Coordinate bounds
        None,        // No vertex data
        Some(777),   // Fixed seed for reproducibility (matches tested configuration)
    )?;

    // Get the first simplex from the triangulation
    let Some(simplex) = dt.tds().simplices().map(|(_, simplex)| simplex).next() else {
        println!("No simplices in triangulation; nothing to demo.");
        return Ok(());
    };
    println!(
        "Using a 4D simplex with {} vertices from triangulation\n",
        simplex.number_of_vertices()
    );

    // Demonstrate functional equivalence
    println!("Functional Equivalence Test:");
    println!("===========================");

    let vec_uuids = simplex.vertex_uuids(dt.tds())?;
    let iter_uuids: Vec<_> = simplex
        .vertex_uuid_iter(dt.tds())
        .collect::<Result<Vec<_>, _>>()?;

    println!("  vertex_uuids() returned {} UUIDs", vec_uuids.len());
    println!(
        "  vertex_uuid_iter().collect() returned {} UUIDs",
        iter_uuids.len()
    );
    // Compare as slices since vertex_uuids() now returns SmallVec
    println!(
        "  Results are identical: {}",
        vec_uuids.as_slice() == iter_uuids.as_slice()
    );
    println!();

    // Performance comparison
    println!("Performance Comparison:");
    println!("======================");

    let iterations = 10_000;

    // Method 1: Allocating Vec (traditional)
    let start = Instant::now();
    let mut total_count = 0;
    for _ in 0..iterations {
        // Note: Errors are intentionally ignored in this timing loop for fair comparison.
        // In a well-formed triangulation with valid simplex references, errors should not occur.
        if let Ok(uuids) = simplex.vertex_uuids(dt.tds()) {
            // Allocates Vec
            total_count += black_box(uuids.len()); // Prevent optimization
        }
    }
    let vec_duration = start.elapsed();

    // Method 2: Zero-allocation iterator
    let start = Instant::now();
    let mut total_count_iter = 0;
    for _ in 0..iterations {
        // Match Method 1 by counting only successfully resolved UUIDs.
        let count = simplex
            .vertex_uuid_iter(dt.tds())
            .filter_map(Result::ok)
            .count(); // No allocation
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
    // Note: len() returns the total number of items (including any errors in Results)
    let iter = simplex.vertex_uuid_iter(dt.tds());
    println!("  Length via ExactSizeIterator: {}", iter.len());

    // Can be used in for loops
    let mut count = 0;
    for uuid_result in simplex.vertex_uuid_iter(dt.tds()) {
        let uuid = uuid_result?;
        if !uuid.is_nil() {
            count += 1;
        }
    }
    println!("  Non-nil UUIDs via for loop: {count}");

    // Can be chained with other iterator methods
    let valid_uuid_count = simplex.vertex_uuid_iter(dt.tds()).try_fold(
        0usize,
        |count, uuid| -> Result<usize, SimplexValidationError> {
            let uuid = uuid?;
            Ok(count + usize::from(!uuid.is_nil()))
        },
    )?;
    println!("  Valid UUIDs via iterator chain: {valid_uuid_count}");

    // Can be used with iterator combinators
    // Note: take(3).count() counts items in the iterator (including Results with Err if any)
    let first_few_count = simplex.vertex_uuid_iter(dt.tds()).take(3).count();
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
