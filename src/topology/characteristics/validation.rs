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
    manifold::{ValidatedFacetDegreeMap, has_boundary_facets_in_validated_facet_map},
    traits::topological_space::{GlobalTopology, TopologyError},
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
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Topology(#[from] delaunay::topology::TopologyError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0, 1.0]?,
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
///
/// let result = dt.euler_check()?;
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
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Topology(#[from] delaunay::topology::TopologyError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0]?,
///     delaunay::vertex![0.5, 1.0]?,
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
///
/// let result = dt.euler_check()?;
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
/// Returns [`TopologyError::FacetMapBuild`] if the TDS facet-incidence map
/// cannot be built. Returns [`TopologyError::BoundaryClassification`] if the
/// declared [`GlobalTopology`] is incompatible with the observed facet
/// incidences, such as an open one-sided facet in a closed topology.
pub fn validate_triangulation_euler<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    global_topology: GlobalTopology<D>,
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

    let facet_to_simplices = ValidatedFacetDegreeMap::try_from_facet_map(&facet_to_simplices)
        .map_err(|source| TopologyError::BoundaryClassification {
            source: Box::new(source),
        })?;
    validate_triangulation_euler_from_validated_facet_map(tds, facet_to_simplices, global_topology)
}

/// Computes the Euler check while reusing an already-validated facet-degree map.
///
/// This keeps [`Triangulation`](crate::prelude::triangulation::Triangulation)
/// Level-3 validation from rebuilding the same incidence map while preserving
/// the public boundary contract: one-sided incidence is classified against
/// [`GlobalTopology`] before it affects the expected χ.
///
/// # Errors
///
/// Returns [`TopologyError::BoundaryClassification`] if one-sided incidence is
/// incompatible with `global_topology`.
pub(crate) fn validate_triangulation_euler_from_validated_facet_map<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    facet_to_simplices: ValidatedFacetDegreeMap<'_>,
    global_topology: GlobalTopology<D>,
) -> Result<TopologyCheckResult, TopologyError> {
    let counts = count_simplices_with_facet_to_simplices_map(tds, facet_to_simplices.as_map());
    let chi = euler_characteristic(&counts);

    let num_simplices = tds.number_of_simplices();
    let has_boundary = num_simplices != 0
        && has_boundary_facets_in_validated_facet_map(tds, facet_to_simplices, global_topology)
            .map_err(|source| TopologyError::BoundaryClassification {
                source: Box::new(source),
            })?;

    let classification = if num_simplices == 0 {
        TopologyClassification::Empty
    } else if num_simplices == 1 && has_boundary {
        TopologyClassification::SingleSimplex(D)
    } else if has_boundary {
        TopologyClassification::Ball(D)
    } else if global_topology.is_toroidal() {
        TopologyClassification::ClosedToroid(D)
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

    Ok(TopologyCheckResult {
        chi,
        expected,
        classification,
        counts,
        notes,
    })
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
