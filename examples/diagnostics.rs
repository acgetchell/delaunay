//! # Diagnostics Example
//!
//! This example demonstrates the opt-in `diagnostics` feature.
//!
//! Run with:
//!
//! ```bash
//! cargo run --features diagnostics --example diagnostics
//! ```

#[cfg(feature = "diagnostics")]
use delaunay::prelude::DelaunayValidationError;
#[cfg(feature = "diagnostics")]
use delaunay::prelude::construction::{
    ConstructionOptions, DelaunayTriangulation, DelaunayTriangulationBuilder,
    DelaunayTriangulationConstructionError, vertex,
};
#[cfg(feature = "diagnostics")]
use delaunay::prelude::diagnostics::{
    debug_print_first_delaunay_violation, delaunay_violation_report,
};
#[cfg(feature = "diagnostics")]
use delaunay::prelude::geometry::{AdaptiveKernel, CoordinateConversionError};
#[cfg(feature = "diagnostics")]
use delaunay::prelude::tds::InvariantError;
#[cfg(feature = "diagnostics")]
#[derive(Debug, thiserror::Error)]
enum DiagnosticsExampleError {
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    DelaunayValidation(#[from] DelaunayValidationError),
    #[error(transparent)]
    CoordinateConversion(#[from] CoordinateConversionError),
    #[error(transparent)]
    Invariant(#[from] InvariantError),
}

#[cfg(feature = "diagnostics")]
fn main() -> Result<(), DiagnosticsExampleError> {
    init_tracing();

    println!("Diagnostics feature example");
    println!("===========================\n");

    report_valid_triangulation()?;
    println!();
    report_non_delaunay_triangulation()?;

    println!("\nDone. Set RUST_LOG=delaunay=debug to see verbose tracing output.");
    Ok(())
}

#[cfg(not(feature = "diagnostics"))]
fn main() {
    println!("This example requires the diagnostics feature.");
    println!("Run: cargo run --features diagnostics --example diagnostics");
}

/// Installs a tracing subscriber so users can opt into verbose helper output with `RUST_LOG`.
#[cfg(feature = "diagnostics")]
fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

/// Shows the shape of an empty diagnostics report for a valid triangulation.
#[cfg(feature = "diagnostics")]
fn report_valid_triangulation() -> Result<(), DiagnosticsExampleError> {
    let vertices = vec![
        vertex![0.0, 0.0, 0.0]?,
        vertex![1.0, 0.0, 0.0]?,
        vertex![0.0, 1.0, 0.0]?,
        vertex![0.0, 0.0, 1.0]?,
    ];
    let dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulationBuilder::new(&vertices).build()?;

    let report = delaunay_violation_report(dt.tds(), None)?;

    println!("Valid 3D triangulation:");
    println!("  vertices: {}", report.number_of_vertices);
    println!("  simplices: {}", report.number_of_simplices);
    println!("  checked simplices: {}", report.checked_simplices);
    println!("  Delaunay valid: {}", report.is_valid());
    assert!(report.is_valid());
    Ok(())
}

/// Imports valid explicit connectivity that is not Delaunay, then reports the violation.
#[cfg(feature = "diagnostics")]
fn report_non_delaunay_triangulation() -> Result<(), DiagnosticsExampleError> {
    let dt = build_non_delaunay_triangulation_2d()?;
    let report = delaunay_violation_report(dt.tds(), None)?;

    println!("Explicit non-Delaunay 2D triangulation:");
    println!("  vertices: {}", report.number_of_vertices);
    println!("  simplices: {}", report.number_of_simplices);
    println!(
        "  violating simplices: {}",
        report.violating_simplices.len()
    );
    assert!(!report.is_valid());

    if let Some(detail) = report.first_violation() {
        println!("  first violating simplex: {:?}", detail.simplex_key);
        println!("  simplex vertex count: {}", detail.simplex_vertices.len());
        println!("  offending external vertex: {:?}", detail.offending_vertex);
    }

    debug_print_first_delaunay_violation(dt.tds(), None);
    Ok(())
}

/// Builds a valid Levels 1-4 triangulation whose prescribed diagonal violates Delaunayness.
#[cfg(feature = "diagnostics")]
fn build_non_delaunay_triangulation_2d()
-> Result<DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 2>, DiagnosticsExampleError> {
    let vertices = vec![
        vertex![0.0, 0.0]?,
        vertex![4.0, 0.0]?,
        vertex![4.0, 2.0]?,
        vertex![1.0, 2.0]?,
    ];
    let simplices = vec![vec![0, 1, 2], vec![0, 2, 3]];
    let dt = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
        .map_err(DelaunayTriangulationConstructionError::from)?
        .construction_options(ConstructionOptions::default().without_final_delaunay_enforcement())
        .build()?;

    dt.as_triangulation().validate()?;
    assert!(dt.is_valid_delaunay().is_err());
    Ok(dt)
}
