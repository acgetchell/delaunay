//! Spherical space topology implementation.
//!
//! This module provides topological analysis for triangulations
//! embedded on spherical manifolds (e.g., triangulations on S²).

#![forbid(unsafe_code)]

use crate::topology::traits::topological_space::{TopologicalSpace, TopologyKind};

/// Projects finite, nonzero coordinate vectors onto the unit sphere.
///
/// Returns `true` when the slice was normalized in place. Returns `false`
/// without modifying `coords` when any coordinate is non-finite or the vector
/// has zero length. The implementation scales by the largest absolute
/// coordinate before summing squares so very large and very small finite vectors
/// can be normalized without overflow or underflow.
pub(crate) fn normalize_unit_sphere_coordinates(coords: &mut [f64]) -> bool {
    let mut max_abs = 0.0_f64;
    for coord in coords.iter().copied() {
        if !coord.is_finite() {
            return false;
        }
        max_abs = max_abs.max(coord.abs());
    }

    if max_abs == 0.0 {
        return false;
    }

    let sum_scaled_squares = coords.iter().fold(0.0, |acc, &coord| {
        let scaled = coord / max_abs;
        scaled.mul_add(scaled, acc)
    });
    let scale = sum_scaled_squares.sqrt();

    for coord in coords.iter_mut() {
        *coord = (*coord / max_abs) / scale;
    }
    true
}

/// Represents spherical topological space.
///
/// Spherical spaces are closed manifolds. For example, a 2-sphere (S²)
/// has Euler characteristic χ = 2.
///
/// [`TopologicalSpace::canonicalize_point`] projects finite nonzero coordinate
/// slices onto the unit sphere. Because the trait method has no error channel,
/// zero-length and non-finite slices are left unchanged. Fallible topology-model
/// adapters report typed failures such as
/// [`crate::topology::traits::GlobalTopologyModelError::ZeroSphericalPointNorm`].
///
/// The dimension `D` is a const generic parameter that must match the
/// dimension of the associated triangulation.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::topology::spaces::SphericalSpace;
/// use delaunay::prelude::topology::spaces::TopologicalSpace;
///
/// let space = SphericalSpace::<2>::new();
/// assert!(!space.allows_boundary());
///
/// let mut coords = [3.0, 4.0];
/// space.canonicalize_point(&mut coords);
/// let norm_sq: f64 = coords.iter().map(|&coord| coord * coord).sum();
/// assert!((norm_sq - 1.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct SphericalSpace<const D: usize>;

impl<const D: usize> SphericalSpace<D> {
    /// Creates a new spherical space instance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::topology::spaces::{
    ///     SphericalSpace, TopologicalSpace, TopologyKind,
    /// };
    ///
    /// let space = SphericalSpace::<3>::new();
    /// assert_eq!(space.kind(), TopologyKind::Spherical);
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl<const D: usize> TopologicalSpace for SphericalSpace<D> {
    const DIM: usize = D;

    fn kind(&self) -> TopologyKind {
        TopologyKind::Spherical
    }

    fn allows_boundary(&self) -> bool {
        false
    }

    fn canonicalize_point(&self, coords: &mut [f64]) {
        normalize_unit_sphere_coordinates(coords);
    }

    fn fundamental_domain(&self) -> Option<&[f64]> {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_1_SQRT_2;

    use approx::{assert_abs_diff_eq, assert_relative_eq};

    use super::*;

    fn squared_norm(coords: &[f64]) -> f64 {
        coords
            .iter()
            .fold(0.0, |acc, &coord| coord.mul_add(coord, acc))
    }

    fn assert_unit_norm(coords: &[f64]) {
        assert_relative_eq!(squared_norm(coords), 1.0, epsilon = 1e-12);
    }

    fn assert_normalizes_to_unit_norm<const D: usize>(mut coords: [f64; D]) {
        let space = SphericalSpace::<D>::new();
        space.canonicalize_point(&mut coords);
        assert_unit_norm(&coords);
    }

    #[test]
    fn test_new() {
        let _space = SphericalSpace::<3>::new();
        assert_eq!(SphericalSpace::<3>::DIM, 3);
    }

    #[test]
    fn test_default() {
        // Test that Default trait is implemented
        fn assert_default<T: Default>() {}
        assert_default::<SphericalSpace<3>>();
    }

    #[test]
    fn test_kind() {
        let space = SphericalSpace::<3>::new();
        assert_eq!(space.kind(), TopologyKind::Spherical);
    }

    #[test]
    fn test_allows_boundary() {
        let space = SphericalSpace::<3>::new();
        assert!(
            !space.allows_boundary(),
            "Spherical space is a closed manifold"
        );
    }

    #[test]
    fn test_canonicalize_point() {
        let space = SphericalSpace::<3>::new();
        let mut coords = [3.0, 4.0, 0.0];
        space.canonicalize_point(&mut coords);
        assert_relative_eq!(coords[0], 0.6);
        assert_relative_eq!(coords[1], 0.8);
        assert_relative_eq!(coords[2], 0.0);
        assert_unit_norm(&coords);
    }

    #[test]
    fn test_canonicalize_point_normalizes_dimensions_2_to_5() {
        assert_normalizes_to_unit_norm([3.0, 4.0]);
        assert_normalizes_to_unit_norm([1.5, 2.5, 3.5]);
        assert_normalizes_to_unit_norm([1.0, 2.0, 3.0, 4.0]);
        assert_normalizes_to_unit_norm([1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_canonicalize_point_handles_near_zero_vector() {
        let space = SphericalSpace::<3>::new();
        let mut coords = [f64::MIN_POSITIVE, f64::MIN_POSITIVE, 0.0];
        space.canonicalize_point(&mut coords);
        assert_relative_eq!(coords[0], FRAC_1_SQRT_2);
        assert_relative_eq!(coords[1], FRAC_1_SQRT_2);
        assert_relative_eq!(coords[2], 0.0);
        assert_unit_norm(&coords);
    }

    #[test]
    fn test_canonicalize_point_handles_large_vector_without_overflow() {
        let space = SphericalSpace::<3>::new();
        let mut coords = [f64::MAX, -f64::MAX, 0.0];
        space.canonicalize_point(&mut coords);
        assert_relative_eq!(coords[0], FRAC_1_SQRT_2);
        assert_relative_eq!(coords[1], -FRAC_1_SQRT_2);
        assert_relative_eq!(coords[2], 0.0);
        assert_unit_norm(&coords);
    }

    #[test]
    fn test_canonicalize_point_is_idempotent() {
        let space = SphericalSpace::<3>::new();
        let mut coords = [2.0, -3.0, 6.0];
        space.canonicalize_point(&mut coords);
        let once = coords;
        space.canonicalize_point(&mut coords);

        for (after_once, after_twice) in once.iter().zip(coords) {
            assert_relative_eq!(*after_once, after_twice);
        }
        assert_unit_norm(&coords);
    }

    #[test]
    fn test_canonicalize_point_leaves_zero_vector_unchanged() {
        let space = SphericalSpace::<3>::new();
        let mut coords = [0.0, 0.0, 0.0];
        space.canonicalize_point(&mut coords);
        assert_abs_diff_eq!(coords[0], 0.0);
        assert_abs_diff_eq!(coords[1], 0.0);
        assert_abs_diff_eq!(coords[2], 0.0);
    }

    #[test]
    fn test_canonicalize_point_leaves_non_finite_vector_unchanged() {
        let space = SphericalSpace::<3>::new();
        let mut coords = [1.0, f64::INFINITY, 0.0];
        space.canonicalize_point(&mut coords);
        assert_relative_eq!(coords[0], 1.0);
        assert!(coords[1].is_infinite());
        assert_relative_eq!(coords[2], 0.0);

        let mut coords = [1.0, f64::NAN, -2.0];
        space.canonicalize_point(&mut coords);
        assert_relative_eq!(coords[0], 1.0);
        assert!(coords[1].is_nan());
        assert_relative_eq!(coords[2], -2.0);
    }

    #[test]
    fn test_fundamental_domain() {
        let space = SphericalSpace::<3>::new();
        assert_eq!(space.fundamental_domain(), None);
    }

    #[test]
    fn test_dimension_consistency() {
        assert_eq!(SphericalSpace::<2>::DIM, 2);
        assert_eq!(SphericalSpace::<3>::DIM, 3);
        assert_eq!(SphericalSpace::<4>::DIM, 4);
        assert_eq!(SphericalSpace::<5>::DIM, 5);
    }
}
