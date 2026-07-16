#![forbid(unsafe_code)]

//! # Dynamic Vertex Lifecycle
//!
//! This example builds a triangulation, inserts a vertex with statistics,
//! locates the inserted point, deletes the vertex transactionally, and checks
//! the final invariant stack.
//!
//! Run it with:
//!
//! ```bash
//! cargo run --release --example dynamic_lifecycle
//! ```

use delaunay::prelude::construction::{DelaunayResult, DelaunayTriangulationBuilder, vertex};
use delaunay::prelude::insertion::InsertionOutcome;

/// Demonstrates the successful insert-query-delete lifecycle.
fn main() -> DelaunayResult<()> {
    let vertices = [vertex![0.0, 0.0]?, vertex![1.0, 0.0]?, vertex![0.0, 1.0]?];
    let mut triangulation = DelaunayTriangulationBuilder::new(&vertices).build()?;

    let candidate = vertex![0.25, 0.25]?;
    let query_point = *candidate.point();
    let (outcome, statistics) = triangulation.insert_best_effort_with_statistics(candidate)?;
    let inserted_key = match outcome {
        InsertionOutcome::Inserted { vertex_key, hint } => {
            println!("Inserted vertex {vertex_key:?} with next-location hint {hint:?}");
            vertex_key
        }
        InsertionOutcome::Skipped { error } => return Err(error.into()),
    };

    println!("  attempts: {}", statistics.attempts);
    println!("  perturbation used: {}", statistics.used_perturbation());
    println!(
        "  simplices removed during repair: {}",
        statistics.simplices_removed_during_repair
    );

    let location = triangulation.locate(&query_point, None)?;
    println!("  point location after insertion: {location:?}");

    let simplices_removed = triangulation.delete_vertex(inserted_key)?;
    triangulation.validate()?;
    println!("Deleted the vertex and {simplices_removed} incident simplices");
    println!(
        "  final state: {} vertices, {} simplices",
        triangulation.number_of_vertices(),
        triangulation.number_of_simplices()
    );
    Ok(())
}
