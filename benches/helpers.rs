//! Shared helper functions for benchmarks
//!
//! This module contains utility functions that are commonly used across multiple
//! benchmark files to avoid code duplication and improve maintainability.

// Import from prelude which already includes all necessary types
use delaunay::prelude::*;

// Additional imports needed for the trait bounds
use nalgebra as na;
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
pub fn clear_all_neighbors<T, U, V, const D: usize>(tds: &mut Tds<T, U, V, D>)
where
    T: CoordinateScalar + AddAssign<T> + na::ComplexField<RealField = T> + SubAssign<T> + Sum,
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

// Tests for this module are located in tests/bench_helpers_test.rs
