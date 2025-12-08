//! Spherical space topology implementation.
//!
//! This module provides topological analysis for triangulations
//! embedded on spherical manifolds (e.g., triangulations on S²).

use crate::topology::traits::topological_space::{TopologicalSpace, TopologyKind};

/// Represents spherical topological space.
///
/// Spherical spaces are closed manifolds. For example, a 2-sphere (S²)
/// has Euler characteristic χ = 2.
#[derive(Debug, Clone, Copy, Default)]
pub struct SphericalSpace;

impl SphericalSpace {
    /// Creates a new spherical space instance.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl TopologicalSpace for SphericalSpace {
    fn kind(&self) -> TopologyKind {
        TopologyKind::Spherical
    }

    fn allows_boundary(&self) -> bool {
        false
    }

    fn canonicalize_point<const D: usize>(&self, _coords: &mut [f64; D]) {
        // TODO: normalize coords to unit sphere
    }

    fn fundamental_domain<const D: usize>(&self) -> Option<[f64; D]> {
        None
    }
}
