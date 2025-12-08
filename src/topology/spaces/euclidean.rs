//! Euclidean space topology implementation.
//!
//! This module provides topological analysis for triangulations
//! embedded in flat Euclidean space.

use crate::topology::traits::topological_space::{TopologicalSpace, TopologyKind};

/// Represents Euclidean (flat) topological space.
///
/// Euclidean spaces have flat topology where D-dimensional balls have
/// Euler characteristic χ = 1.
///
/// The dimension `D` is a const generic parameter that must match the
/// dimension of the associated triangulation.
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
