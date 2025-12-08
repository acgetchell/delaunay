//! Spherical space topology implementation.
//!
//! This module provides topological analysis for triangulations
//! embedded on spherical manifolds (e.g., triangulations on S²).

use crate::topology::traits::topological_space::{TopologicalSpace, TopologyKind};

/// Represents spherical topological space.
///
/// Spherical spaces are closed manifolds. For example, a 2-sphere (S²)
/// has Euler characteristic χ = 2.
///
/// The dimension `D` is a const generic parameter that must match the
/// dimension of the associated triangulation.
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
        // TODO: normalize coords to unit sphere
    }

    fn fundamental_domain(&self) -> Option<&[f64]> {
        None
    }
}
