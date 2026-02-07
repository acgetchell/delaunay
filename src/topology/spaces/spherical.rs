//! Spherical space topology implementation.
//!
//! This module provides topological analysis for triangulations
//! embedded on spherical manifolds (e.g., triangulations on S²).

#![forbid(unsafe_code)]

use crate::topology::traits::topological_space::{TopologicalSpace, TopologyKind};

/// Represents spherical topological space.
///
/// Spherical spaces are closed manifolds. For example, a 2-sphere (S²)
/// has Euler characteristic χ = 2.
///
/// The dimension `D` is a const generic parameter that must match the
/// dimension of the associated triangulation.
///
/// # Examples
///
/// ```rust
/// use delaunay::topology::spaces::SphericalSpace;
/// use delaunay::topology::traits::topological_space::TopologicalSpace;
///
/// let space = SphericalSpace::<2>::new();
/// assert!(!space.allows_boundary());
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct SphericalSpace<const D: usize>;

impl<const D: usize> SphericalSpace<D> {
    /// Creates a new spherical space instance.
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

    fn canonicalize_point(&self, _coords: &mut [f64]) {
        // TODO: Implement unit-sphere normalization.
        // This should normalize the input coordinates to lie on the unit sphere (radius = 1).
        // Use the L2 norm of the coordinate array and divide each coordinate by it.
        // Currently a no-op; see test_canonicalize_point() which documents this behavior.
        // Tracking issue: https://github.com/acgetchell/delaunay/issues/188
    }

    fn fundamental_domain(&self) -> Option<&[f64]> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

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
        let mut coords = [1.5, 2.5, 3.5];
        space.canonicalize_point(&mut coords);
        // TODO: Currently a no-op, will normalize to unit sphere in future
        assert_relative_eq!(coords[0], 1.5);
        assert_relative_eq!(coords[1], 2.5);
        assert_relative_eq!(coords[2], 3.5);
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
