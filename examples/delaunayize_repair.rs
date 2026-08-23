#![forbid(unsafe_code)]

//! # Delaunay Refinement Repair Example
//!
//! This example demonstrates the flip-repair mode of [`DelaunayRefinementBuilder`], which
//! consumes a Levels 1–4 triangulation and performs bounded flip-based
//! Delaunay repair before Level 5 certification.
//!
//! The workflow has three steps:
//!
//! 1. **Levels 1–4 proof consumption** — accepts the validated
//!    `Triangulation` domain value without repeating its encoded checks.
//! 2. **Delaunay flip repair** — preserves Levels 1–4 transactionally while
//!    restoring the empty-circumsphere property via k=2/k=3 bistellar flips.
//! 3. **Optional fallback rebuild** — rebuilds from the vertex set if flip
//!    repair fails.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example delaunayize_repair
//! ```

use delaunay::RefinementError;
use delaunay::prelude::construction::{
    DelaunayTriangulation, DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError,
    vertex,
};
use delaunay::prelude::delaunayize::*;
use delaunay::prelude::geometry::CoordinateConversionError;
use delaunay::prelude::pachner::{FacetError, FlipError, PachnerMove, PachnerMoves};
use delaunay::prelude::validation::DelaunayTriangulationValidationError;

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
        vertex![0.0, 0.0, 0.0]?,
        vertex![1.0, 0.0, 0.0]?,
        vertex![0.0, 1.0, 0.0]?,
        vertex![0.0, 0.0, 1.0]?,
        vertex![0.5, 0.5, 0.5]?,
    ];
    let dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulationBuilder::new(&vertices).build()?;

    println!(
        "  Built 3D triangulation: {} vertices, {} simplices",
        dt.number_of_vertices(),
        dt.number_of_simplices()
    );

    let result = DelaunayRefinementBuilder::new(dt.into_triangulation())
        .repair_by_flips()
        .build()
        .map_err(RefinementError::into_reason)?;
    print_outcome(&result.outcome);

    result.triangulation.validate()?;
    println!("  ✓ Full validation (Levels 1–5) passed");
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
        vertex![0.0, 0.0, 0.0, 0.0]?,
        vertex![1.0, 0.0, 0.0, 0.0]?,
        vertex![0.0, 1.0, 0.0, 0.0]?,
        vertex![0.0, 0.0, 1.0, 0.0]?,
        vertex![0.0, 0.0, 0.0, 1.0]?,
    ];
    let dt: DelaunayTriangulation<_, (), (), 4> =
        DelaunayTriangulationBuilder::new(&vertices).build()?;

    println!(
        "  Built 4D triangulation: {} vertices, {} simplices",
        dt.number_of_vertices(),
        dt.number_of_simplices()
    );

    let result = DelaunayRefinementBuilder::new(dt.into_triangulation())
        .repair_by_flips()
        .build()
        .map_err(RefinementError::into_reason)?;
    print_outcome(&result.outcome);

    result.triangulation.validate()?;
    println!("  ✓ Full validation (Levels 1–5) passed");
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
        vertex![0.0, 0.0]?,
        vertex![4.0, 0.0]?,
        vertex![4.0, 4.0]?,
        vertex![0.0, 4.0]?,
        vertex![2.0, 2.0]?,
        vertex![1.0, 1.0]?,
        vertex![3.0, 1.0]?,
    ];
    let dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulationBuilder::new(&vertices).build()?;

    println!(
        "  Initial: {} vertices, {} simplices",
        dt.number_of_vertices(),
        dt.number_of_simplices()
    );
    assert!(dt.validate().is_ok());
    println!("  ✓ Initially Delaunay");

    // Topology-changing primitives require an explicit demotion to the
    // Levels 1–4 owner.
    let mut tri = dt.into_triangulation();

    // Collect interior facets and find one whose k=2 flip actually breaks Delaunay.
    let mut facets: Vec<_> = Vec::new();
    for (ck, simplex) in tri.simplices() {
        if let Some(neighbors) = simplex.neighbors() {
            for (i, n) in neighbors.enumerate() {
                if let (Some(_), Ok(idx)) = (n, u8::try_from(i)) {
                    facets.push(tri.facet_handle(ck, idx)?);
                }
            }
        }
    }

    let mut violating_facet = None;
    for facet in facets {
        let mut trial = tri.clone();
        let Ok(proposal) = trial.propose_pachner(PachnerMove::K2 { facet }) else {
            continue;
        };
        if proposal.attempt_on(&mut trial).is_ok()
            && DelaunayRefinementBuilder::new(trial).build().is_err()
        {
            violating_facet = Some(facet);
            break;
        }
    }

    let Some(facet) = violating_facet else {
        println!("  (No k=2 flip produced a non-Delaunay state — skipping repair demonstration)");
        return Ok(());
    };

    let selected_flip = tri
        .propose_pachner(PachnerMove::K2 { facet })?
        .attempt_on(&mut tri)?;
    assert!(!selected_flip.new_simplices.is_empty());
    tri = match DelaunayRefinementBuilder::new(tri).build() {
        Ok(_) => {
            println!(
                "  Applied selected k=2 flip, but Delaunay property remained satisfied (unexpected)"
            );
            return Ok(());
        }
        Err(failure) => {
            println!(
                "  Applied k=2 flip; post-flip check confirms Delaunay violation: {}",
                failure.reason()
            );
            failure.into_owner()
        }
    };

    // Repair.
    let result = DelaunayRefinementBuilder::new(tri)
        .repair_by_flips()
        .build()
        .map_err(RefinementError::into_reason)?;
    print_outcome(&result.outcome);

    result.triangulation.validate()?;
    println!("  ✓ Delaunay property restored");
    Ok(())
}

/// Custom configuration with an explicit flip budget and fallback enabled.
#[expect(
    clippy::result_large_err,
    reason = "example preserves the crate's typed repair errors instead of erasing them"
)]
fn custom_config_2d() -> Result<(), DelaunayizeRepairExampleError> {
    println!("4. Custom configuration (2D, fallback enabled)");
    println!("----------------------------------------------\n");

    let vertices = vec![
        vertex![0.0, 0.0]?,
        vertex![1.0, 0.0]?,
        vertex![0.0, 1.0]?,
        vertex![1.0, 1.0]?,
        vertex![0.5, 0.5]?,
    ];
    let dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulationBuilder::new(&vertices).build()?;

    println!("  Config: max_flips=100, fallback=true");

    let result = DelaunayRefinementBuilder::new(dt.into_triangulation())
        .repair_by_flips()
        .max_flips(100)
        .fallback_rebuild(true)
        .build()
        .map_err(RefinementError::into_reason)?;
    print_outcome(&result.outcome);

    result.triangulation.validate()?;
    println!(
        "  ✓ Valid 2D triangulation: {} vertices, {} simplices",
        result.triangulation.number_of_vertices(),
        result.triangulation.number_of_simplices(),
    );
    Ok(())
}

fn print_outcome(outcome: &DelaunayizeOutcome) {
    println!(
        "  Delaunay repair: facets_checked={}, flips_performed={}",
        outcome.delaunay_repair.facets_checked, outcome.delaunay_repair.flips_performed,
    );
    println!("  Fallback rebuild used: {}", outcome.used_fallback_rebuild);
}
