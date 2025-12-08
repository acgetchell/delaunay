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
