#![forbid(unsafe_code)]

//! # Delaunayize-by-Flips Repair Example
//!
//! This example demonstrates the **delaunayize-by-flips** workflow that
//! performs bounded topology repair followed by flip-based Delaunay repair.
//!
//! The workflow has three steps:
//!
//! 1. **PL-manifold topology repair** — removes simplices that cause facet
//!    over-sharing (codimension-1 facet degree > 2).
//! 2. **Delaunay flip repair** — restores the empty-circumsphere property
//!    via k=2/k=3 bistellar flips.
//! 3. **Optional fallback rebuild** — rebuilds from the vertex set if both
//!    repair passes fail.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example delaunayize_repair
//! ```

use delaunay::prelude::construction::DelaunayTriangulationConstructionError;
use delaunay::prelude::delaunayize::*;
use delaunay::prelude::flips::*;
use delaunay::prelude::geometry::CoordinateConversionError;
use delaunay::prelude::tds::FacetError;
use delaunay::prelude::validation::DelaunayTriangulationValidationError;

// For the generic print_outcome helper.
use delaunay::prelude::DataType;

#[derive(Debug, thiserror::Error)]
enum DelaunayizeRepairExampleError {
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    Delaunayize(#[from] DelaunayizeError),
    #[error(transparent)]
    Validation(#[from] DelaunayTriangulationValidationError),
    #[error(transparent)]
    Flip(#[from] FlipError),
    #[error(transparent)]
    Facet(#[from] FacetError),
    #[error(transparent)]
    CoordinateConversion(#[from] CoordinateConversionError),
}

#[expect(
    clippy::result_large_err,
    reason = "example preserves the crate's typed repair errors instead of erasing them"
)]
fn main() -> Result<(), DelaunayizeRepairExampleError> {
    println!("============================================================");
    println!("Delaunayize-by-Flips Repair Workflow");
    println!("============================================================\n");

    already_delaunay_3d()?;
    println!("\n------------------------------------------------------------\n");
    already_delaunay_4d()?;
    println!("\n------------------------------------------------------------\n");
    flip_then_repair_2d()?;
    println!("\n------------------------------------------------------------\n");
    custom_config_2d()?;

    println!("\n============================================================");
    println!("Example completed successfully!");
    println!("============================================================");
    Ok(())
}

/// A 3D triangulation that is already Delaunay — delaunayize is a no-op.
#[expect(
    clippy::result_large_err,
    reason = "example preserves the crate's typed repair errors instead of erasing them"
)]
fn already_delaunay_3d() -> Result<(), DelaunayizeRepairExampleError> {
    println!("1. Already-Delaunay 3D triangulation (no-op)");
    println!("--------------------------------------------\n");

    let vertices = vec![
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.5, 0.5, 0.5])?,
    ];
    let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::try_new(&vertices)?;

    println!(
        "  Built 3D triangulation: {} vertices, {} simplices",
        dt.number_of_vertices(),
        dt.number_of_simplices()
    );

    let outcome = delaunayize_by_flips(&mut dt, DelaunayizeConfig::default())?;
    print_outcome(&outcome);

    dt.validate()?;
    println!("  ✓ Full validation (Levels 1–4) passed");
    Ok(())
}

/// A 4D triangulation — shows the workflow is dimension-generic.
#[expect(
    clippy::result_large_err,
    reason = "example preserves the crate's typed repair errors instead of erasing them"
)]
fn already_delaunay_4d() -> Result<(), DelaunayizeRepairExampleError> {
    println!("2. Already-Delaunay 4D triangulation (no-op)");
    println!("--------------------------------------------\n");

    let vertices = vec![
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 1.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.25, 0.25, 0.25, 0.25])?,
    ];
    let mut dt: DelaunayTriangulation<_, (), (), 4> = DelaunayTriangulation::try_new(&vertices)?;

    println!(
        "  Built 4D triangulation: {} vertices, {} simplices",
        dt.number_of_vertices(),
        dt.number_of_simplices()
    );

    let outcome = delaunayize_by_flips(&mut dt, DelaunayizeConfig::default())?;
    print_outcome(&outcome);

    dt.validate()?;
    println!("  ✓ Full validation (Levels 1–4) passed");
    Ok(())
}

/// Apply a k=2 flip in 2D to break the Delaunay property, then repair.
///
/// 2D with 7 points guarantees interior facets that are flippable.
#[expect(
    clippy::result_large_err,
    reason = "example preserves the crate's typed repair errors instead of erasing them"
)]
fn flip_then_repair_2d() -> Result<(), DelaunayizeRepairExampleError> {
    println!("3. Flip breaks Delaunay in 2D → delaunayize restores it");
    println!("-------------------------------------------------------\n");

    let vertices = vec![
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([4.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([4.0, 4.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 4.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([2.0, 2.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([1.0, 1.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([3.0, 1.0])?,
    ];
    let mut dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::try_new(&vertices)?;

    println!(
        "  Initial: {} vertices, {} simplices",
        dt.number_of_vertices(),
        dt.number_of_simplices()
    );
    assert!(dt.validate().is_ok());
    println!("  ✓ Initially Delaunay");

    // Collect interior facets and find one whose k=2 flip actually breaks Delaunay.
    let mut facets: Vec<_> = Vec::new();
    for (ck, simplex) in dt.simplices() {
        if let Some(neighbors) = simplex.neighbors() {
            for (i, n) in neighbors.enumerate() {
                if let (Some(_), Ok(idx)) = (n, u8::try_from(i)) {
                    facets.push(FacetHandle::try_new(dt.tds(), ck, idx)?);
                }
            }
        }
    }

    let mut violating_facet = None;
    for facet in facets {
        let mut trial = dt.clone();
        if trial.flip_k2(facet).is_ok() && trial.is_valid().is_err() {
            violating_facet = Some(facet);
            break;
        }
    }

    let Some(facet) = violating_facet else {
        println!("  (No k=2 flip produced a non-Delaunay state — skipping repair demonstration)");
        return Ok(());
    };

    dt.flip_k2(facet)?;
    match dt.is_valid() {
        Ok(()) => {
            println!(
                "  Applied selected k=2 flip, but Delaunay property remained satisfied (unexpected)"
            );
            return Ok(());
        }
        Err(err) => {
            println!("  Applied k=2 flip; post-flip check confirms Delaunay violation: {err}");
        }
    }

    // Repair.
    let outcome = delaunayize_by_flips(&mut dt, DelaunayizeConfig::default())?;
    print_outcome(&outcome);

    dt.validate()?;
    println!("  ✓ Delaunay property restored");
    Ok(())
}

/// Custom configuration with tight budgets and fallback enabled.
#[expect(
    clippy::result_large_err,
    reason = "example preserves the crate's typed repair errors instead of erasing them"
)]
fn custom_config_2d() -> Result<(), DelaunayizeRepairExampleError> {
    println!("4. Custom configuration (2D, fallback enabled)");
    println!("----------------------------------------------\n");

    let vertices = vec![
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([1.0, 1.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.5, 0.5])?,
    ];
    let mut dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::try_new(&vertices)?;

    let config = DelaunayizeConfig {
        topology_max_iterations: 10,
        topology_max_simplices_removed: 100,
        fallback_rebuild: true,
        delaunay_max_flips: None,
    };

    println!(
        "  Config: max_iterations={}, max_simplices_removed={}, fallback={}",
        config.topology_max_iterations,
        config.topology_max_simplices_removed,
        config.fallback_rebuild,
    );

    let outcome = delaunayize_by_flips(&mut dt, config)?;
    print_outcome(&outcome);

    dt.validate()?;
    println!(
        "  ✓ Valid 2D triangulation: {} vertices, {} simplices",
        dt.number_of_vertices(),
        dt.number_of_simplices(),
    );
    Ok(())
}

fn print_outcome<U: DataType, V: DataType, const D: usize>(outcome: &DelaunayizeOutcome<U, V, D>) {
    println!(
        "  Topology repair: succeeded={}, iterations={}, simplices_removed={}",
        outcome.topology_repair.succeeded,
        outcome.topology_repair.iterations,
        outcome.topology_repair.simplices_removed,
    );
    println!(
        "  Delaunay repair: facets_checked={}, flips_performed={}",
        outcome.delaunay_repair.facets_checked, outcome.delaunay_repair.flips_performed,
    );
    println!("  Fallback rebuild used: {}", outcome.used_fallback_rebuild);
}
