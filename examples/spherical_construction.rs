#![forbid(unsafe_code)]

//! # Spherical Construction
//!
//! This example uses the direct Rust builder for the supported `S^2` and `S^3`
//! spherical Delaunay prototypes. The companion spherical notebook is the
//! primary visual interpretation workflow.
//!
//! Run it with:
//!
//! ```bash
//! cargo run --release --example spherical_construction
//! ```

use delaunay::prelude::construction::{
    SphericalDelaunayBuilder, SphericalDelaunayConstructionError, SphericalDelaunayValidationError,
};

#[derive(Debug, thiserror::Error)]
enum SphericalExampleError {
    #[error(transparent)]
    Construction(#[from] SphericalDelaunayConstructionError),
    #[error(transparent)]
    Validation(#[from] SphericalDelaunayValidationError),
}

/// Builds and validates the two currently supported intrinsic dimensions.
fn main() -> Result<(), SphericalExampleError> {
    demonstrate_s2()?;
    demonstrate_s3()?;
    Ok(())
}

/// Builds the boundary of a tetrahedron as a triangulation of `S^2`.
fn demonstrate_s2() -> Result<(), SphericalExampleError> {
    let points = [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ];
    let triangulation = SphericalDelaunayBuilder::<2>::try_new(points)?.build()?;
    triangulation.validate()?;

    println!("S^2 Delaunay triangulation:");
    println!("  ambient dimension: {}", triangulation.ambient_dimension());
    println!("  vertices: {}", triangulation.number_of_vertices());
    println!(
        "  triangular simplices: {}",
        triangulation.number_of_simplices()
    );
    Ok(())
}

/// Builds the boundary of a 4-simplex as a triangulation of `S^3`.
fn demonstrate_s3() -> Result<(), SphericalExampleError> {
    let points = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [-1.0, -1.0, -1.0, -1.0],
    ];
    let triangulation = SphericalDelaunayBuilder::<3>::try_new(points)?.build()?;
    triangulation.validate()?;

    println!("\nS^3 Delaunay triangulation:");
    println!("  ambient dimension: {}", triangulation.ambient_dimension());
    println!("  vertices: {}", triangulation.number_of_vertices());
    println!(
        "  tetrahedral simplices: {}",
        triangulation.number_of_simplices()
    );
    Ok(())
}
