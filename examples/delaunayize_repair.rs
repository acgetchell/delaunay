//! # Delaunayize-by-Flips Repair Example
//!
//! This example demonstrates the **delaunayize-by-flips** workflow that
//! performs bounded topology repair followed by flip-based Delaunay repair.
//!
//! The workflow has three steps:
//!
//! 1. **PL-manifold topology repair** — removes cells that cause facet
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

use delaunay::prelude::triangulation::delaunayize::*;
use delaunay::prelude::triangulation::flips::*;

fn main() {
    println!("============================================================");
    println!("Delaunayize-by-Flips Repair Workflow");
    println!("============================================================\n");

    already_delaunay_3d();
    println!("\n------------------------------------------------------------\n");
    already_delaunay_4d();
    println!("\n------------------------------------------------------------\n");
    flip_then_repair_2d();
    println!("\n------------------------------------------------------------\n");
    custom_config_2d();

    println!("\n============================================================");
    println!("Example completed successfully!");
    println!("============================================================");
}

/// A 3D triangulation that is already Delaunay — delaunayize is a no-op.
fn already_delaunay_3d() {
    println!("1. Already-Delaunay 3D triangulation (no-op)");
    println!("--------------------------------------------\n");

    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
        vertex!([0.5, 0.5, 0.5]),
    ];
    let mut dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::new(&vertices).unwrap();

    println!(
        "  Built 3D triangulation: {} vertices, {} cells",
        dt.number_of_vertices(),
        dt.number_of_cells()
    );

    let outcome = delaunayize_by_flips(&mut dt, DelaunayizeConfig::default()).unwrap();
    print_outcome(&outcome);

    dt.validate().unwrap();
    println!("  ✓ Full validation (Levels 1–4) passed");
}

/// A 4D triangulation — shows the workflow is dimension-generic.
fn already_delaunay_4d() {
    println!("2. Already-Delaunay 4D triangulation (no-op)");
    println!("--------------------------------------------\n");

    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0]),
        vertex!([0.25, 0.25, 0.25, 0.25]),
    ];
    let mut dt: DelaunayTriangulation<_, (), (), 4> =
        DelaunayTriangulation::new(&vertices).unwrap();

    println!(
        "  Built 4D triangulation: {} vertices, {} cells",
        dt.number_of_vertices(),
        dt.number_of_cells()
    );

    let outcome = delaunayize_by_flips(&mut dt, DelaunayizeConfig::default()).unwrap();
    print_outcome(&outcome);

    dt.validate().unwrap();
    println!("  ✓ Full validation (Levels 1–4) passed");
}

/// Apply a k=2 flip in 2D to break the Delaunay property, then repair.
///
/// 2D with 7 points guarantees interior facets that are flippable.
fn flip_then_repair_2d() {
    println!("3. Flip breaks Delaunay in 2D → delaunayize restores it");
    println!("-------------------------------------------------------\n");

    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([4.0, 0.0]),
        vertex!([4.0, 4.0]),
        vertex!([0.0, 4.0]),
        vertex!([2.0, 2.0]),
        vertex!([1.0, 1.0]),
        vertex!([3.0, 1.0]),
    ];
    let mut dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::new(&vertices).unwrap();

    println!(
        "  Initial: {} vertices, {} cells",
        dt.number_of_vertices(),
        dt.number_of_cells()
    );
    assert!(dt.validate().is_ok());
    println!("  ✓ Initially Delaunay");

    // Collect interior facets and find one whose k=2 flip actually breaks Delaunay.
    let mut facets: Vec<_> = Vec::new();
    for (ck, cell) in dt.cells() {
        if let Some(neighbors) = cell.neighbors() {
            for (i, n) in neighbors.iter().enumerate() {
                if let (Some(_), Ok(idx)) = (n, u8::try_from(i)) {
                    facets.push(FacetHandle::new(ck, idx));
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
        return;
    };

    dt.flip_k2(facet).unwrap();
    match dt.is_valid() {
        Ok(()) => {
            println!(
                "  Applied selected k=2 flip, but Delaunay property remained satisfied (unexpected)"
            );
            return;
        }
        Err(err) => {
            println!("  Applied k=2 flip; post-flip check confirms Delaunay violation: {err}");
        }
    }

    // Repair.
    let outcome = delaunayize_by_flips(&mut dt, DelaunayizeConfig::default()).unwrap();
    print_outcome(&outcome);

    dt.validate().unwrap();
    println!("  ✓ Delaunay property restored");
}

/// Custom configuration with tight budgets and fallback enabled.
fn custom_config_2d() {
    println!("4. Custom configuration (2D, fallback enabled)");
    println!("----------------------------------------------\n");

    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
        vertex!([1.0, 1.0]),
        vertex!([0.5, 0.5]),
    ];
    let mut dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::new(&vertices).unwrap();

    let config = DelaunayizeConfig {
        topology_max_iterations: 10,
        topology_max_cells_removed: 100,
        fallback_rebuild: true,
    };

    println!(
        "  Config: max_iterations={}, max_cells_removed={}, fallback={}",
        config.topology_max_iterations, config.topology_max_cells_removed, config.fallback_rebuild,
    );

    let outcome = delaunayize_by_flips(&mut dt, config).unwrap();
    print_outcome(&outcome);

    dt.validate().unwrap();
    println!(
        "  ✓ Valid 2D triangulation: {} vertices, {} cells",
        dt.number_of_vertices(),
        dt.number_of_cells(),
    );
}

fn print_outcome(outcome: &DelaunayizeOutcome) {
    println!(
        "  Topology repair: succeeded={}, iterations={}, cells_removed={}",
        outcome.topology_repair.succeeded,
        outcome.topology_repair.iterations,
        outcome.topology_repair.cells_removed,
    );
    println!(
        "  Delaunay repair: facets_checked={}, flips_performed={}",
        outcome.delaunay_repair.facets_checked, outcome.delaunay_repair.flips_performed,
    );
    println!("  Fallback rebuild used: {}", outcome.used_fallback_rebuild);
}
