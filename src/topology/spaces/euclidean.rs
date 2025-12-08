//! Euclidean space topology implementation.
//!
//! This module provides topological analysis for triangulations
//! embedded in flat Euclidean space.

use crate::topology::traits::topological_space::{TopologicalSpace, TopologyKind};

/// Represents Euclidean (flat) topological space.
///
/// Euclidean spaces have flat topology where D-dimensional balls have
/// Euler characteristic χ = 1.
#[derive(Debug, Clone, Copy, Default)]
pub struct EuclideanSpace;

impl EuclideanSpace {
    /// Creates a new Euclidean space instance.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl TopologicalSpace for EuclideanSpace {
    fn kind(&self) -> TopologyKind {
        TopologyKind::Euclidean
    }

    fn allows_boundary(&self) -> bool {
        true
    }

    fn canonicalize_point<const D: usize>(&self, _coords: &mut [f64; D]) {}

    fn fundamental_domain<const D: usize>(&self) -> Option<[f64; D]> {
        None
    }
}
