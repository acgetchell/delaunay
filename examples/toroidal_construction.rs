#![forbid(unsafe_code)]

//! # Toroidal Construction
//!
//! This example builds and validates periodic image-point quotients of `T^2`
//! and `T^3`.
//!
//! Run it with:
//!
//! ```bash
//! cargo run --release --example toroidal_construction
//! ```

use delaunay::prelude::construction::{
    DelaunayResult, DelaunayTriangulationBuilder, Vertex, vertex,
};
use delaunay::prelude::geometry::RobustKernel;

/// Demonstrates periodic `T^2` and `T^3` construction.
fn main() -> DelaunayResult<()> {
    demonstrate_periodic_t2()?;
    demonstrate_periodic_t3()?;
    Ok(())
}

/// Builds and validates a true periodic quotient of `T^2`.
fn demonstrate_periodic_t2() -> DelaunayResult<()> {
    let kernel = RobustKernel::new();
    let vertices = periodic_fixture_t2()?;
    let periodic = DelaunayTriangulationBuilder::new(&vertices)
        .try_toroidal([1.0, 1.0])?
        .build_with_kernel(&kernel)?;
    periodic.validate()?;

    let boundary_facets = periodic
        .boundary_facets()?
        .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))?;

    println!("\nPeriodic T^2 image-point quotient:");
    println!("  vertices: {}", periodic.number_of_vertices());
    println!("  triangular simplices: {}", periodic.number_of_simplices());
    println!("  boundary facets: {boundary_facets}");
    println!(
        "  periodic quotient: {}",
        periodic.global_topology().is_periodic()
    );
    Ok(())
}

/// Builds and validates a true periodic quotient of `T^3`.
fn demonstrate_periodic_t3() -> DelaunayResult<()> {
    let kernel = RobustKernel::new();
    let vertices = periodic_fixture_t3()?;
    let periodic = DelaunayTriangulationBuilder::new(&vertices)
        .try_toroidal([1.0, 1.0, 1.0])?
        .build_with_kernel(&kernel)?;
    periodic.validate()?;

    let boundary_facets = periodic
        .boundary_facets()?
        .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))?;

    println!("\nPeriodic T^3 image-point quotient:");
    println!("  vertices: {}", periodic.number_of_vertices());
    println!(
        "  tetrahedral simplices: {}",
        periodic.number_of_simplices()
    );
    println!("  boundary facets: {boundary_facets}");
    println!(
        "  periodic quotient: {}",
        periodic.global_topology().is_periodic()
    );
    Ok(())
}

/// Returns a compact, deterministic point set known to form a valid `T^2` quotient.
fn periodic_fixture_t2() -> DelaunayResult<Vec<Vertex<(), 2>>> {
    Ok(vec![
        vertex![0.20, 0.30]?,
        vertex![0.80, 0.10]?,
        vertex![0.50, 0.70]?,
        vertex![0.10, 0.90]?,
        vertex![0.60, 0.40]?,
        vertex![0.30, 0.50]?,
        vertex![0.90, 0.20]?,
    ])
}

/// Returns a compact, deterministic point set known to form a valid `T^3` quotient.
fn periodic_fixture_t3() -> DelaunayResult<Vec<Vertex<(), 3>>> {
    Ok(vec![
        vertex![0.20, 0.30, 0.40]?,
        vertex![0.80, 0.10, 0.20]?,
        vertex![0.50, 0.70, 0.60]?,
        vertex![0.10, 0.90, 0.30]?,
        vertex![0.60, 0.40, 0.80]?,
        vertex![0.30, 0.50, 0.90]?,
        vertex![0.90, 0.20, 0.60]?,
    ])
}
