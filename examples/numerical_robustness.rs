#![forbid(unsafe_code)]

//! # Numerical Robustness Example
//!
//! This example accompanies `docs/numerical_robustness_guide.md`.
//!
//! It compares kernel behavior on degenerate predicate inputs and shows the
//! default adaptive construction path on a small point set.

use delaunay::prelude::construction::{
    DelaunayTriangulation, DelaunayTriangulationConstructionError,
};
use delaunay::prelude::geometry::{
    AdaptiveKernel, CircumcenterError, CoordinateConversionError, CoordinateValidationError,
    FastKernel, Kernel, Point, RobustKernel, robust_insphere, robust_orientation,
};
use delaunay::prelude::validation::DelaunayTriangulationValidationError;

#[derive(Debug, thiserror::Error)]
enum NumericalRobustnessExampleError {
    #[error(transparent)]
    Predicate(#[from] CircumcenterError),
    #[error(transparent)]
    CoordinateConversion(#[from] CoordinateConversionError),
    #[error(transparent)]
    CoordinateValidation(#[from] CoordinateValidationError),
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    Validation(#[from] DelaunayTriangulationValidationError),
}

fn main() -> Result<(), NumericalRobustnessExampleError> {
    println!("Numerical robustness example");
    println!("============================\n");

    compare_orientation_kernels()?;
    println!();
    compare_insphere_boundary_handling()?;
    println!();
    build_with_adaptive_kernel()?;
    Ok(())
}

/// Compares orientation predicate behavior on a degenerate collinear simplex.
fn compare_orientation_kernels() -> Result<(), NumericalRobustnessExampleError> {
    let collinear = [
        Point::try_new([0.0, 0.0])?,
        Point::try_new([1.0, 1.0])?,
        Point::try_new([2.0, 2.0])?,
    ];

    let fast = FastKernel::<f64>::new();
    let robust = RobustKernel::<f64>::new();
    let adaptive = AdaptiveKernel::<f64>::new();

    let direct_robust = robust_orientation(&collinear)?;
    let fast_sign = fast.orientation(&collinear)?;
    let robust_sign = robust.orientation(&collinear)?;
    let adaptive_sign = adaptive.orientation(&collinear)?;

    println!("Collinear orientation:");
    println!("  robust_orientation: {direct_robust:?}");
    println!("  FastKernel sign: {fast_sign}");
    println!("  RobustKernel sign: {robust_sign}");
    println!("  AdaptiveKernel sign with SoS tie-break: {adaptive_sign}");

    assert_eq!(robust_sign, 0);
    assert_ne!(adaptive_sign, 0);
    Ok(())
}

/// Compares explicit boundary reporting with adaptive `SoS` tie-breaking.
fn compare_insphere_boundary_handling() -> Result<(), NumericalRobustnessExampleError> {
    let simplex = [
        Point::try_new([0.0, 0.0])?,
        Point::try_new([1.0, 0.0])?,
        Point::try_new([0.0, 1.0])?,
    ];
    let boundary_point = Point::try_new([1.0, 1.0])?;

    let robust = RobustKernel::<f64>::new();
    let adaptive = AdaptiveKernel::<f64>::new();

    let direct_robust = robust_insphere(&simplex, &boundary_point)?;
    let robust_sign = robust.in_sphere(&simplex, &boundary_point)?;
    let adaptive_sign = adaptive.in_sphere(&simplex, &boundary_point)?;

    println!("Cospherical insphere query:");
    println!("  robust_insphere: {direct_robust:?}");
    println!("  RobustKernel sign: {robust_sign}");
    println!("  AdaptiveKernel sign with SoS tie-break: {adaptive_sign}");

    assert_eq!(robust_sign, 0);
    assert_ne!(adaptive_sign, 0);
    Ok(())
}

/// Builds a small triangulation with the default exact adaptive kernel and validates it.
fn build_with_adaptive_kernel() -> Result<(), NumericalRobustnessExampleError> {
    let vertices = vec![
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.25, 0.25, 0.25])?,
    ];

    let dt: DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 3> =
        DelaunayTriangulation::try_new(&vertices)?;

    dt.validate()?;
    println!(
        "Adaptive construction: {} vertices, {} simplices, full validation passed",
        dt.number_of_vertices(),
        dt.number_of_simplices()
    );
    Ok(())
}
