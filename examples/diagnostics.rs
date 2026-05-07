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
use delaunay::prelude::diagnostics::{
    debug_print_first_delaunay_violation, delaunay_violation_report,
};
#[cfg(feature = "diagnostics")]
use delaunay::prelude::geometry::AdaptiveKernel;
#[cfg(feature = "diagnostics")]
use delaunay::prelude::triangulation::{
    DelaunayTriangulation, DelaunayTriangulationValidationError, flips::*,
};
#[cfg(feature = "diagnostics")]
use delaunay::vertex;

#[cfg(feature = "diagnostics")]
fn main() {
    init_tracing();

    println!("Diagnostics feature example");
    println!("===========================\n");

    report_valid_triangulation();
    println!();
    report_non_delaunay_triangulation();

    println!("\nDone. Set RUST_LOG=delaunay=debug to see verbose tracing output.");
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
fn report_valid_triangulation() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];
    let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();

    let report = delaunay_violation_report(dt.tds(), None).unwrap();

    println!("Valid 3D triangulation:");
    println!("  vertices: {}", report.number_of_vertices);
    println!("  cells: {}", report.number_of_cells);
    println!("  checked cells: {}", report.checked_cells);
    println!("  Delaunay valid: {}", report.is_valid());
    assert!(report.is_valid());
}

/// Applies a public topology edit that breaks Delaunayness, then reports the violation.
#[cfg(feature = "diagnostics")]
fn report_non_delaunay_triangulation() {
    let dt = build_non_delaunay_triangulation_2d();
    let report = delaunay_violation_report(dt.tds(), None).unwrap();

    println!("Non-Delaunay 2D triangulation after an explicit k=2 flip:");
    println!("  vertices: {}", report.number_of_vertices);
    println!("  cells: {}", report.number_of_cells);
    println!("  violating cells: {}", report.violating_cells.len());
    assert!(!report.is_valid());

    if let Some(detail) = &report.first_violation {
        println!("  first violating cell: {:?}", detail.cell_key);
        println!("  cell vertex count: {}", detail.cell_vertices.len());
        println!("  offending external vertex: {:?}", detail.offending_vertex);
    }

    debug_print_first_delaunay_violation(dt.tds(), None);
}

/// Builds a valid 2D triangulation and returns a clone after a k=2 flip creates a violation.
#[cfg(feature = "diagnostics")]
fn build_non_delaunay_triangulation_2d() -> DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 2> {
    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([4.0, 0.0]),
        vertex!([4.0, 4.0]),
        vertex!([0.0, 4.0]),
        vertex!([2.0, 2.0]),
        vertex!([1.0, 1.0]),
        vertex!([3.0, 1.0]),
    ];
    let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&vertices).unwrap();

    for (cell_key, cell) in dt.cells() {
        if let Some(neighbors) = cell.neighbors() {
            for (facet_index, neighbor) in neighbors.iter().enumerate() {
                if neighbor.is_none() {
                    continue;
                }

                let Ok(facet_index) = u8::try_from(facet_index) else {
                    continue;
                };
                let facet = FacetHandle::new(cell_key, facet_index);
                let mut trial = dt.clone();
                if trial.flip_k2(facet).is_ok()
                    && trial.as_triangulation().validate().is_ok()
                    && matches!(
                        trial.is_valid(),
                        Err(DelaunayTriangulationValidationError::VerificationFailed { .. })
                    )
                {
                    return trial;
                }
            }
        }
    }

    panic!("expected at least one public k=2 flip to produce a Delaunay violation");
}
