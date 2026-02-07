//! Euclidean space topology implementation.
//!
//! This module provides topological analysis for triangulations
//! embedded in flat Euclidean space.

#![forbid(unsafe_code)]

use crate::topology::traits::topological_space::{TopologicalSpace, TopologyKind};

/// Represents Euclidean (flat) topological space.
///
/// Euclidean spaces have flat topology where D-dimensional balls have
/// Euler characteristic χ = 1.
///
/// The dimension `D` is a const generic parameter that must match the
/// dimension of the associated triangulation.
///
/// # Examples
///
/// ```rust
/// use delaunay::topology::spaces::EuclideanSpace;
/// use delaunay::topology::traits::topological_space::TopologicalSpace;
///
/// let space = EuclideanSpace::<3>::new();
/// assert!(space.allows_boundary());
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct EuclideanSpace<const D: usize>;

impl<const D: usize> EuclideanSpace<D> {
    /// Creates a new Euclidean space instance.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl<const D: usize> TopologicalSpace for EuclideanSpace<D> {
    const DIM: usize = D;

    fn kind(&self) -> TopologyKind {
        TopologyKind::Euclidean
    }

    fn allows_boundary(&self) -> bool {
        true
    }

    fn canonicalize_point(&self, _coords: &mut [f64]) {
        // No-op for Euclidean space
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
        let _space = EuclideanSpace::<3>::new();
        assert_eq!(EuclideanSpace::<3>::DIM, 3);
    }

    #[test]
    fn test_default() {
        // Test that Default trait is implemented
        fn assert_default<T: Default>() {}
        assert_default::<EuclideanSpace<3>>();
    }

    #[test]
    fn test_kind() {
        let space = EuclideanSpace::<3>::new();
        assert_eq!(space.kind(), TopologyKind::Euclidean);
    }

    #[test]
    fn test_allows_boundary() {
        let space = EuclideanSpace::<3>::new();
        assert!(space.allows_boundary());
    }

    #[test]
    fn test_canonicalize_point() {
        let space = EuclideanSpace::<3>::new();
        let mut coords = [1.5, 2.5, 3.5];
        space.canonicalize_point(&mut coords);
        // Euclidean space doesn't modify coordinates
        assert_relative_eq!(coords[0], 1.5);
        assert_relative_eq!(coords[1], 2.5);
        assert_relative_eq!(coords[2], 3.5);
    }

    #[test]
    fn test_fundamental_domain() {
        let space = EuclideanSpace::<3>::new();
        assert_eq!(space.fundamental_domain(), None);
    }

    #[test]
    fn test_dimension_consistency() {
        assert_eq!(EuclideanSpace::<2>::DIM, 2);
        assert_eq!(EuclideanSpace::<3>::DIM, 3);
        assert_eq!(EuclideanSpace::<4>::DIM, 4);
        assert_eq!(EuclideanSpace::<5>::DIM, 5);
    }
}
