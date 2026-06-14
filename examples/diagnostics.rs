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
    DelaunayTriangulation, DelaunayTriangulationConstructionError,
};
#[cfg(feature = "diagnostics")]
use delaunay::prelude::diagnostics::{
    debug_print_first_delaunay_violation, delaunay_violation_report,
};
#[cfg(feature = "diagnostics")]
use delaunay::prelude::flips::*;
#[cfg(feature = "diagnostics")]
use delaunay::prelude::geometry::{AdaptiveKernel, CoordinateConversionError};
#[cfg(feature = "diagnostics")]
use delaunay::prelude::validation::DelaunayTriangulationValidationError;
#[cfg(feature = "diagnostics")]
#[derive(Debug, thiserror::Error)]
enum DiagnosticsExampleError {
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    DelaunayValidation(#[from] DelaunayValidationError),
    #[error(transparent)]
    CoordinateConversion(#[from] CoordinateConversionError),
    #[error("expected at least one public k=2 flip to produce a Delaunay violation")]
    NoDelaunayViolatingFlip,
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
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    ];
    let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices)?;

    let report = delaunay_violation_report(dt.tds(), None)?;

    println!("Valid 3D triangulation:");
    println!("  vertices: {}", report.number_of_vertices);
    println!("  simplices: {}", report.number_of_simplices);
    println!("  checked simplices: {}", report.checked_simplices);
    println!("  Delaunay valid: {}", report.is_valid());
    assert!(report.is_valid());
    Ok(())
}

/// Applies a public topology edit that breaks Delaunayness, then reports the violation.
#[cfg(feature = "diagnostics")]
fn report_non_delaunay_triangulation() -> Result<(), DiagnosticsExampleError> {
    let dt = build_non_delaunay_triangulation_2d()?;
    let report = delaunay_violation_report(dt.tds(), None)?;

    println!("Non-Delaunay 2D triangulation after an explicit k=2 flip:");
    println!("  vertices: {}", report.number_of_vertices);
    println!("  simplices: {}", report.number_of_simplices);
    println!(
        "  violating simplices: {}",
        report.violating_simplices.len()
    );
    assert!(!report.is_valid());

    if let Some(detail) = &report.first_violation {
        println!("  first violating simplex: {:?}", detail.simplex_key);
        println!("  simplex vertex count: {}", detail.simplex_vertices.len());
        println!("  offending external vertex: {:?}", detail.offending_vertex);
    }

    debug_print_first_delaunay_violation(dt.tds(), None);
    Ok(())
}

/// Builds a valid 2D triangulation and returns a clone after a k=2 flip creates a violation.
#[cfg(feature = "diagnostics")]
fn build_non_delaunay_triangulation_2d()
-> Result<DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 2>, DiagnosticsExampleError> {
    let vertices = vec![
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([4.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([4.0, 4.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 4.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([2.0, 2.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([1.0, 1.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([3.0, 1.0])?,
    ];
    let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&vertices)?;

    for (simplex_key, simplex) in dt.simplices() {
        if let Some(neighbors) = simplex.neighbors() {
            for (facet_index, neighbor) in neighbors.enumerate() {
                if neighbor.is_none() {
                    continue;
                }

                let Ok(facet_index) = u8::try_from(facet_index) else {
                    continue;
                };
                let facet = FacetHandle::new(simplex_key, facet_index);
                let mut trial = dt.clone();
                if trial.flip_k2(facet).is_ok()
                    && trial.as_triangulation().validate().is_ok()
                    && matches!(
                        trial.is_valid(),
                        Err(DelaunayTriangulationValidationError::VerificationFailed { .. })
                    )
                {
                    return Ok(trial);
                }
            }
        }
    }

    Err(DiagnosticsExampleError::NoDelaunayViolatingFlip)
}
