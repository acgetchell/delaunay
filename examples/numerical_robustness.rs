//! # Numerical Robustness Example
//!
//! This example accompanies `docs/numerical_robustness_guide.md`.
//!
//! It compares kernel behavior on degenerate predicate inputs and shows the
//! default adaptive construction path on a small point set.

use delaunay::prelude::geometry::{
    AdaptiveKernel, Coordinate, FastKernel, Kernel, Point, RobustKernel, robust_insphere,
    robust_orientation,
};
use delaunay::prelude::triangulation::DelaunayTriangulation;
use delaunay::vertex;

fn main() {
    println!("Numerical robustness example");
    println!("============================\n");

    compare_orientation_kernels();
    println!();
    compare_insphere_boundary_handling();
    println!();
    build_with_adaptive_kernel();
}

/// Compares orientation predicate behavior on a degenerate collinear simplex.
fn compare_orientation_kernels() {
    let collinear = [
        Point::new([0.0, 0.0]),
        Point::new([1.0, 1.0]),
        Point::new([2.0, 2.0]),
    ];

    let fast = FastKernel::<f64>::new();
    let robust = RobustKernel::<f64>::new();
    let adaptive = AdaptiveKernel::<f64>::new();

    let direct_robust = robust_orientation(&collinear).unwrap();
    let fast_sign = fast.orientation(&collinear).unwrap();
    let robust_sign = robust.orientation(&collinear).unwrap();
    let adaptive_sign = adaptive.orientation(&collinear).unwrap();

    println!("Collinear orientation:");
    println!("  robust_orientation: {direct_robust:?}");
    println!("  FastKernel sign: {fast_sign}");
    println!("  RobustKernel sign: {robust_sign}");
    println!("  AdaptiveKernel sign with SoS tie-break: {adaptive_sign}");

    assert_eq!(robust_sign, 0);
    assert_ne!(adaptive_sign, 0);
}

/// Compares explicit boundary reporting with adaptive `SoS` tie-breaking.
fn compare_insphere_boundary_handling() {
    let simplex = [
        Point::new([0.0, 0.0]),
        Point::new([1.0, 0.0]),
        Point::new([0.0, 1.0]),
    ];
    let boundary_point = Point::new([1.0, 1.0]);

    let robust = RobustKernel::<f64>::new();
    let adaptive = AdaptiveKernel::<f64>::new();

    let direct_robust = robust_insphere(&simplex, &boundary_point).unwrap();
    let robust_sign = robust.in_sphere(&simplex, &boundary_point).unwrap();
    let adaptive_sign = adaptive.in_sphere(&simplex, &boundary_point).unwrap();

    println!("Cospherical insphere query:");
    println!("  robust_insphere: {direct_robust:?}");
    println!("  RobustKernel sign: {robust_sign}");
    println!("  AdaptiveKernel sign with SoS tie-break: {adaptive_sign}");

    assert_eq!(robust_sign, 0);
    assert_ne!(adaptive_sign, 0);
}

/// Builds a small triangulation with the default exact adaptive kernel and validates it.
fn build_with_adaptive_kernel() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
        vertex!([0.25, 0.25, 0.25]),
    ];

    let dt: DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 3> =
        DelaunayTriangulation::new(&vertices).unwrap();

    dt.validate().unwrap();
    println!(
        "Adaptive construction: {} vertices, {} cells, full validation passed",
        dt.number_of_vertices(),
        dt.number_of_cells()
    );
}
