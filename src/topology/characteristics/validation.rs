//! Topology validation functions for triangulated spaces.
//!
//! This module provides high-level validation functions that combine
//! simplex counting, classification, and Euler characteristic checking.

#![forbid(unsafe_code)]

use crate::core::{collections::FacetToSimplicesMap, tds::Tds};
use crate::topology::{
    characteristics::euler::{
        FVector, TopologyClassification, count_simplices_with_facet_to_simplices_map,
        euler_characteristic, expected_chi_for,
    },
    traits::topological_space::TopologyError,
};

/// Result of Euler characteristic validation.
///
/// Contains the computed Euler characteristic, expected value based on
/// topological classification, and diagnostic information.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::*;
/// use delaunay::prelude::topology::validation;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Topology(#[from] delaunay::topology::TopologyError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).expect("finite vertex coordinates"),
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
///
/// let result = validation::validate_triangulation_euler(dt.tds())?;
/// assert_eq!(result.chi, 1);
/// assert!(result.is_valid());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologyCheckResult {
    /// Computed Euler characteristic.
    pub chi: isize,

    /// Expected χ based on classification (None if unknown).
    pub expected: Option<isize>,

    /// Topological classification.
    pub classification: TopologyClassification,

    /// Full simplex counts (f-vector).
    pub counts: FVector,

    /// Diagnostic notes or warnings.
    pub notes: Vec<String>,
}

impl TopologyCheckResult {
    /// Returns `true` if χ matches expectation.
    ///
    /// If expected is None (unknown classification), returns `true`
    /// since we cannot determine if it's valid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::topology::validation::{euler::*, TopologyCheckResult};
    ///
    /// let valid_result = TopologyCheckResult {
    ///     chi: 1,
    ///     expected: Some(1),
    ///     classification: TopologyClassification::Ball(3),
    ///     counts: FVector { by_dim: vec![4, 6, 4, 1] },
    ///     notes: vec![],
    /// };
    /// assert!(valid_result.is_valid());
    ///
    /// let invalid_result = TopologyCheckResult {
    ///     chi: 0,
    ///     expected: Some(1),
    ///     classification: TopologyClassification::Ball(3),
    ///     counts: FVector { by_dim: vec![4, 6, 4, 1] },
    ///     notes: vec!["Mismatch".to_string()],
    /// };
    /// assert!(!invalid_result.is_valid());
    /// ```
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.expected.is_none_or(|exp| self.chi == exp)
    }
}

/// Validate triangulation Euler characteristic.
///
/// Combines simplex counting, classification, and validation into a
/// comprehensive topology check.
///
/// # Returns
///
/// A `TopologyCheckResult` containing:
/// - Computed Euler characteristic
/// - Expected value (if determinable)
/// - Topological classification
/// - Complete simplex counts
/// - Diagnostic notes
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::*;
/// use delaunay::prelude::topology::validation;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Topology(#[from] delaunay::topology::TopologyError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.5, 1.0]).expect("finite vertex coordinates"),
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
///
/// let result = validation::validate_triangulation_euler(dt.tds())?;
/// assert_eq!(result.chi, 1);
/// assert_eq!(result.counts.count(0), 3);  // 3 vertices
/// assert_eq!(result.counts.count(1), 3);  // 3 edges
/// assert_eq!(result.counts.count(2), 1);  // 1 face
/// assert!(result.is_valid());
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`TopologyError`] if topology validation support data cannot be built.
pub fn validate_triangulation_euler<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
) -> Result<TopologyCheckResult, TopologyError> {
    // Precompute the facet map once and reuse it for both counting and classification.
    //
    // Avoid building the map for empty triangulations.
    let facet_to_simplices = if tds.number_of_simplices() == 0 {
        FacetToSimplicesMap::default()
    } else {
        tds.build_facet_to_simplices_map()
            .map_err(|source| TopologyError::FacetMapBuild { source })?
    };

    Ok(validate_triangulation_euler_with_facet_to_simplices_map(
        tds,
        &facet_to_simplices,
    ))
}

pub(crate) fn validate_triangulation_euler_with_facet_to_simplices_map<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    facet_to_simplices: &FacetToSimplicesMap,
) -> TopologyCheckResult {
    let counts = count_simplices_with_facet_to_simplices_map(tds, facet_to_simplices);
    let chi = euler_characteristic(&counts);

    let num_simplices = tds.number_of_simplices();
    let classification = if num_simplices == 0 {
        TopologyClassification::Empty
    } else if num_simplices == 1 {
        TopologyClassification::SingleSimplex(D)
    } else if facet_to_simplices
        .values()
        .any(|simplices| simplices.len() == 1)
    {
        TopologyClassification::Ball(D)
    } else {
        TopologyClassification::ClosedSphere(D)
    };

    let expected = expected_chi_for(&classification);

    let mut notes = Vec::new();

    // Add diagnostic notes
    if let Some(exp) = expected.filter(|&exp| chi != exp) {
        notes.push(format!(
            "Euler characteristic mismatch: computed {chi}, expected {exp}"
        ));
    }

    TopologyCheckResult {
        chi,
        expected,
        classification,
        counts,
        notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::characteristics::euler::TopologyClassification;

    #[test]
    fn test_topology_check_result_is_valid() {
        let valid_result = TopologyCheckResult {
            chi: 1,
            expected: Some(1),
            classification: TopologyClassification::Ball(3),
            counts: FVector {
                by_dim: vec![4, 6, 4, 1],
            },
            notes: vec![],
        };
        assert!(valid_result.is_valid());

        let invalid_result = TopologyCheckResult {
            chi: 0,
            expected: Some(1),
            classification: TopologyClassification::Ball(3),
            counts: FVector {
                by_dim: vec![4, 6, 4, 1],
            },
            notes: vec!["Mismatch".to_string()],
        };
        assert!(!invalid_result.is_valid());

        let unknown_result = TopologyCheckResult {
            chi: 42,
            expected: None,
            classification: TopologyClassification::Unknown,
            counts: FVector { by_dim: vec![1] },
            notes: vec![],
        };
        assert!(unknown_result.is_valid()); // Unknown classification is considered valid
    }
}
