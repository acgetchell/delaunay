//! Toroidal space topology implementation.
//!
//! This module provides topological analysis for triangulations
//! on toroidal manifolds with periodic boundary conditions.

use crate::topology::traits::topological_space::{TopologicalSpace, TopologyKind};

/// Represents toroidal topological space with periodic boundaries.
///
/// Toroidal spaces have periodic boundary conditions defined by a
/// fundamental domain. For example, a 2-torus (T²) has Euler
/// characteristic χ = 0.
///
/// The dimension `D` is a const generic parameter that must match the
/// dimension of the associated triangulation.
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
}

impl<const D: usize> TopologicalSpace for ToroidalSpace<D> {
    const DIM: usize = D;

    fn kind(&self) -> TopologyKind {
        TopologyKind::Toroidal
    }

    fn allows_boundary(&self) -> bool {
        false
    }

    fn canonicalize_point(&self, _coords: &mut [f64]) {
        // TODO: wrap coords into domain
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
        // TODO: Currently a no-op, will wrap into domain in future
        assert_relative_eq!(coords[0], 2.5);
        assert_relative_eq!(coords[1], -1.0);
        assert_relative_eq!(coords[2], 5.5);
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
}
