//! Toroidal space topology implementation.
//!
//! This module provides topological analysis for triangulations
//! on toroidal manifolds with periodic boundary conditions.

#![forbid(unsafe_code)]

use crate::geometry::traits::coordinate::CoordinateScalar;
use crate::topology::traits::topological_space::{TopologicalSpace, TopologyKind};
use num_traits::NumCast;

/// Represents toroidal topological space with periodic boundaries.
///
/// Toroidal spaces have periodic boundary conditions defined by a
/// fundamental domain. For example, a 2-torus (T²) has Euler
/// characteristic χ = 0.
///
/// The dimension `D` is a const generic parameter that must match the
/// dimension of the associated triangulation.
///
/// # Examples
///
/// ```rust
/// use delaunay::topology::spaces::ToroidalSpace;
///
/// let space = ToroidalSpace::<2>::new([1.0, 2.0]);
/// assert_eq!(space.domain, [1.0, 2.0]);
/// ```
#[derive(Debug, Clone)]
pub struct ToroidalSpace<const D: usize> {
    /// The fundamental domain defining the period of each dimension.
    pub domain: [f64; D],
}

impl<const D: usize> ToroidalSpace<D> {
    /// Creates a new toroidal space with the given fundamental domain.
    ///
    /// # Arguments
    ///
    /// * `domain` - The period of each dimension for periodic boundary conditions
    #[must_use]
    pub const fn new(domain: [f64; D]) -> Self {
        Self { domain }
    }

    /// Creates a unit toroidal space where every dimension has period 1.0.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::topology::spaces::ToroidalSpace;
    ///
    /// let space = ToroidalSpace::<3>::unit();
    /// assert_eq!(space.domain, [1.0, 1.0, 1.0]);
    /// ```
    #[must_use]
    pub const fn unit() -> Self {
        Self { domain: [1.0; D] }
    }

    /// Wraps a single coordinate value into the fundamental domain `[0, L_axis)`
    /// using `rem_euclid` arithmetic.
    ///
    /// Converts `value` to `f64`, applies `rem_euclid(domain[axis])`, then converts
    /// back to `T`. Returns `None` if either conversion fails (e.g. the input is not
    /// finite, or the result is not representable in `T`).
    ///
    /// # Arguments
    ///
    /// * `axis` - The dimension index (must be `< D`; returns `None` if out of range).
    /// * `value` - The coordinate value to wrap.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::topology::spaces::ToroidalSpace;
    ///
    /// let space = ToroidalSpace::<2>::new([1.0, 2.0]);
    ///
    /// // Positive out-of-range
    /// assert_eq!(space.wrap_coord::<f64>(0, 1.7), Some(0.7));
    ///
    /// // Negative wraps to positive
    /// assert_eq!(space.wrap_coord::<f64>(1, -0.5), Some(1.5));
    ///
    /// // Out-of-range axis returns None
    /// assert_eq!(space.wrap_coord::<f64>(5, 0.3), None);
    /// ```
    #[must_use]
    pub fn wrap_coord<T: CoordinateScalar>(&self, axis: usize, value: T) -> Option<T> {
        let period = *self.domain.get(axis)?;
        let v_f64 = value.to_f64()?;
        if !v_f64.is_finite() {
            return None;
        }
        let wrapped = v_f64.rem_euclid(period);
        <T as NumCast>::from(wrapped)
    }
}

impl<const D: usize> TopologicalSpace for ToroidalSpace<D> {
    const DIM: usize = D;

    fn kind(&self) -> TopologyKind {
        TopologyKind::Toroidal
    }

    fn allows_boundary(&self) -> bool {
        false
    }

    fn canonicalize_point(&self, coords: &mut [f64]) {
        for (coord, &period) in coords.iter_mut().zip(self.domain.iter()) {
            *coord = coord.rem_euclid(period);
        }
    }

    fn fundamental_domain(&self) -> Option<&[f64]> {
        Some(&self.domain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_new() {
        let space = ToroidalSpace::<3>::new([1.0, 2.0, 3.0]);
        assert_eq!(ToroidalSpace::<3>::DIM, 3);
        assert_relative_eq!(space.domain[0], 1.0);
        assert_relative_eq!(space.domain[1], 2.0);
        assert_relative_eq!(space.domain[2], 3.0);
    }

    #[test]
    fn test_kind() {
        let space = ToroidalSpace::<3>::new([1.0, 1.0, 1.0]);
        assert_eq!(space.kind(), TopologyKind::Toroidal);
    }

    #[test]
    fn test_allows_boundary() {
        let space = ToroidalSpace::<3>::new([1.0, 1.0, 1.0]);
        assert!(
            !space.allows_boundary(),
            "Toroidal space is a closed manifold with periodic boundaries"
        );
    }

    #[test]
    fn test_canonicalize_point() {
        let space = ToroidalSpace::<3>::new([2.0, 3.0, 4.0]);
        let mut coords = [2.5, -1.0, 5.5];
        space.canonicalize_point(&mut coords);
        // 2.5 rem_euclid 2.0 = 0.5
        assert_relative_eq!(coords[0], 0.5);
        // -1.0 rem_euclid 3.0 = 2.0
        assert_relative_eq!(coords[1], 2.0);
        // 5.5 rem_euclid 4.0 = 1.5
        assert_relative_eq!(coords[2], 1.5);
    }

    #[test]
    fn test_canonicalize_point_idempotent() {
        let space = ToroidalSpace::<2>::new([1.0, 1.0]);
        let mut coords = [0.3, 0.7];
        space.canonicalize_point(&mut coords);
        // Already in [0, 1), should be unchanged
        assert_relative_eq!(coords[0], 0.3);
        assert_relative_eq!(coords[1], 0.7);
        // Applying again should be unchanged (idempotent)
        space.canonicalize_point(&mut coords);
        assert_relative_eq!(coords[0], 0.3);
        assert_relative_eq!(coords[1], 0.7);
    }

    #[test]
    fn test_canonicalize_point_boundary() {
        let space = ToroidalSpace::<2>::new([1.0, 1.0]);
        // Exactly on boundary: 1.0 rem_euclid 1.0 = 0.0
        let mut coords = [1.0, 2.0];
        space.canonicalize_point(&mut coords);
        assert_relative_eq!(coords[0], 0.0);
        assert_relative_eq!(coords[1], 0.0);
    }

    #[test]
    fn test_canonicalize_point_negative() {
        let space = ToroidalSpace::<3>::new([1.0, 1.0, 1.0]);
        // Negative coordinates should wrap into [0, 1)
        let mut coords = [-0.1, -1.0, -2.5];
        space.canonicalize_point(&mut coords);
        assert_relative_eq!(coords[0], 0.9);
        assert_relative_eq!(coords[1], 0.0);
        assert_relative_eq!(coords[2], 0.5);
    }

    #[test]
    fn test_fundamental_domain() {
        let domain = [2.0, 3.0, 4.0];
        let space = ToroidalSpace::<3>::new(domain);
        assert_eq!(space.fundamental_domain(), Some(&domain[..]));
    }

    #[test]
    fn test_different_domains() {
        // 2D unit square torus
        let unit_torus = ToroidalSpace::<2>::new([1.0, 1.0]);
        assert_eq!(unit_torus.fundamental_domain(), Some(&[1.0, 1.0][..]));

        // 2D rectangular torus
        let rect_torus = ToroidalSpace::<2>::new([2.0, 3.0]);
        assert_eq!(rect_torus.fundamental_domain(), Some(&[2.0, 3.0][..]));

        // 3D cube torus
        let cube_torus = ToroidalSpace::<3>::new([1.0, 1.0, 1.0]);
        assert_eq!(cube_torus.fundamental_domain(), Some(&[1.0, 1.0, 1.0][..]));
    }

    #[test]
    fn test_dimension_consistency() {
        assert_eq!(ToroidalSpace::<2>::DIM, 2);
        assert_eq!(ToroidalSpace::<3>::DIM, 3);
        assert_eq!(ToroidalSpace::<4>::DIM, 4);
        assert_eq!(ToroidalSpace::<5>::DIM, 5);
    }

    #[test]
    fn test_unit() {
        let space = ToroidalSpace::<3>::unit();
        assert_relative_eq!(space.domain[0], 1.0);
        assert_relative_eq!(space.domain[1], 1.0);
        assert_relative_eq!(space.domain[2], 1.0);
        let space2d = ToroidalSpace::<2>::unit();
        assert_relative_eq!(space2d.domain[0], 1.0);
        assert_relative_eq!(space2d.domain[1], 1.0);
    }

    #[test]
    fn test_wrap_coord_positive_out_of_range() {
        let space = ToroidalSpace::<2>::new([1.0, 2.0]);
        let wrapped = space.wrap_coord::<f64>(0, 1.7);
        assert!(wrapped.is_some());
        assert_relative_eq!(wrapped.unwrap(), 0.7);
    }

    #[test]
    fn test_wrap_coord_negative() {
        let space = ToroidalSpace::<2>::new([1.0, 2.0]);
        // -0.5 rem_euclid 2.0 = 1.5
        let wrapped = space.wrap_coord::<f64>(1, -0.5);
        assert!(wrapped.is_some());
        assert_relative_eq!(wrapped.unwrap(), 1.5);
    }

    #[test]
    fn test_wrap_coord_in_range_unchanged() {
        let space = ToroidalSpace::<2>::new([1.0, 1.0]);
        let wrapped = space.wrap_coord::<f64>(0, 0.3);
        assert!(wrapped.is_some());
        assert_relative_eq!(wrapped.unwrap(), 0.3);
    }

    #[test]
    fn test_wrap_coord_boundary() {
        let space = ToroidalSpace::<2>::new([1.0, 1.0]);
        // Exactly at period boundary wraps to 0
        let wrapped = space.wrap_coord::<f64>(0, 1.0);
        assert!(wrapped.is_some());
        assert_relative_eq!(wrapped.unwrap(), 0.0);
    }

    #[test]
    fn test_wrap_coord_out_of_range_axis() {
        let space = ToroidalSpace::<2>::new([1.0, 1.0]);
        assert!(space.wrap_coord::<f64>(5, 0.3).is_none());
    }

    #[test]
    fn test_wrap_coord_f32() {
        let space = ToroidalSpace::<2>::new([1.0, 1.0]);
        let wrapped = space.wrap_coord::<f32>(0, 1.5_f32);
        assert!(wrapped.is_some());
        assert_relative_eq!(wrapped.unwrap(), 0.5_f32, epsilon = 1e-6);
    }

    #[test]
    fn test_wrap_coord_non_finite() {
        let space = ToroidalSpace::<2>::new([1.0, 1.0]);
        assert!(space.wrap_coord::<f64>(0, f64::NAN).is_none());
        assert!(space.wrap_coord::<f64>(0, f64::INFINITY).is_none());
    }
}
