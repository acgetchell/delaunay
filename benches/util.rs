//! Shared helper functions for benchmarks
//!
//! This module contains utility functions that are commonly used across multiple
//! benchmark files to avoid code duplication and improve maintainability.

// Import from prelude which already includes all necessary types
use delaunay::prelude::*;

// Additional imports needed for the trait bounds
use nalgebra as na;
use rand::Rng;
use std::iter::Sum;
use std::ops::{AddAssign, Div, SubAssign};

/// Helper function to clear neighbors from all cells in a triangulation.
///
/// This is useful for benchmarking neighbor assignment in isolation by ensuring
/// all cells start without existing neighbor relationships.
///
/// # Arguments
///
/// * `tds` - Mutable reference to the triangulation data structure
///
/// # Examples
///
/// ```ignore
/// use delaunay::prelude::*;
/// use crate::helpers::clear_all_neighbors;
///
/// let mut tds = Tds::new(&vertices).unwrap();
/// clear_all_neighbors(&mut tds);  // All cells now have neighbors = None
/// ```
#[allow(dead_code)]
pub fn clear_all_neighbors<T, U, V, const D: usize>(tds: &mut Tds<T, U, V, D>)
where
    T: CoordinateScalar
        + AddAssign<T>
        + na::ComplexField<RealField = T>
        + SubAssign<T>
        + Sum
        + From<f64>,
    U: DataType,
    V: DataType,
    f64: From<T>,
    for<'a> &'a T: Div<T>,
    [T; D]: Copy + Default + serde::de::DeserializeOwned + serde::Serialize + Sized,
    ordered_float::OrderedFloat<f64>: From<T>,
{
    for cell in tds.cells_mut().values_mut() {
        cell.clear_neighbors();
    }
}

/// Generic function to generate random points for benchmarking across any dimension.
///
/// This function provides a consistent way to generate random points for benchmarks
/// across all supported dimensions (2D-5D) with a fixed coordinate range.
///
/// # Arguments
///
/// * `n_points` - Number of points to generate
///
/// # Returns
///
/// Vector of random points with coordinates in the range [-100.0, 100.0]
///
/// # Examples
///
/// ```ignore
/// use crate::helpers::generate_random_points;
///
/// let points_2d: Vec<Point<f64, 2>> = generate_random_points(100);
/// let points_3d: Vec<Point<f64, 3>> = generate_random_points(100);
/// let points_4d: Vec<Point<f64, 4>> = generate_random_points(100);
/// let points_5d: Vec<Point<f64, 5>> = generate_random_points(100);
/// ```
#[must_use]
#[allow(dead_code)]
pub fn generate_random_points<const D: usize>(n_points: usize) -> Vec<Point<f64, D>> {
    let mut rng = rand::rng();
    (0..n_points)
        .map(|_| {
            let coords = [0.0; D].map(|_| rng.random_range(-100.0..100.0));
            Point::new(coords)
        })
        .collect()
}

/// Generate random points with a seeded RNG for reproducible benchmarks.
///
/// This function is useful when you need consistent point generation across
/// multiple benchmark runs for fair comparison.
///
/// # Arguments
///
/// * `n_points` - Number of points to generate
/// * `seed` - Seed for the random number generator
///
/// # Returns
///
/// Vector of random points with coordinates in the range [-100.0, 100.0]
#[must_use]
#[allow(dead_code)]
pub fn generate_random_points_seeded<const D: usize>(
    n_points: usize,
    seed: u64,
) -> Vec<Point<f64, D>> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n_points)
        .map(|_| {
            let coords = [0.0; D].map(|_| rng.random_range(-100.0..100.0));
            Point::new(coords)
        })
        .collect()
}

/// Macro to generate dimension-specific convenience functions.
///
/// This macro creates functions like `generate_random_points_2d`, `generate_random_points_3d`, etc.
/// for backward compatibility and clearer code in benchmarks that work with specific dimensions.
macro_rules! generate_dimension_specific_fns {
    ($($dim:literal),+) => {
        $(
            pastey::paste! {
                #[doc = "Generate random " $dim "D points for benchmarking."]
                #[must_use]
                #[allow(dead_code)]
                pub fn [<generate_random_points_ $dim d>](n_points: usize) -> Vec<Point<f64, $dim>> {
                    generate_random_points::<$dim>(n_points)
                }

                #[doc = "Generate random " $dim "D points with a seeded RNG."]
                #[must_use]
                #[allow(dead_code)]
                pub fn [<generate_random_points_ $dim d_seeded>](n_points: usize, seed: u64) -> Vec<Point<f64, $dim>> {
                    generate_random_points_seeded::<$dim>(n_points, seed)
                }
            }
        )+
    };
}

// Generate convenience functions for dimensions 2-5
generate_dimension_specific_fns!(2, 3, 4, 5);

// Tests for this module are located in tests/bench_util_test.rs
