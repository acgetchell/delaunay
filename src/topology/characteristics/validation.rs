//! Topology validation functions for triangulated spaces.
//!
//! This module provides high-level validation functions that combine
//! simplex counting, classification, and Euler characteristic checking.

use crate::core::{
    collections::FacetToCellsMap, traits::DataType, triangulation_data_structure::Tds,
};
use crate::geometry::traits::coordinate::CoordinateScalar;
use crate::topology::{
    characteristics::euler::{
        SimplexCounts, TopologyClassification, count_simplices_with_facet_to_cells_map,
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
/// use delaunay::topology::characteristics::validation;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let dt = DelaunayTriangulation::new(&vertices).unwrap();
///
/// let result = validation::validate_triangulation_euler(dt.tds()).unwrap();
/// assert_eq!(result.chi, 1);
/// assert!(result.is_valid());
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
    pub counts: SimplexCounts,

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
    /// use delaunay::topology::characteristics::{euler::*, validation::TopologyCheckResult};
    ///
    /// let valid_result = TopologyCheckResult {
    ///     chi: 1,
    ///     expected: Some(1),
    ///     classification: TopologyClassification::Ball(3),
    ///     counts: SimplexCounts { by_dim: vec![4, 6, 4, 1] },
    ///     notes: vec![],
    /// };
    /// assert!(valid_result.is_valid());
    ///
    /// let invalid_result = TopologyCheckResult {
    ///     chi: 0,
    ///     expected: Some(1),
    ///     classification: TopologyClassification::Ball(3),
    ///     counts: SimplexCounts { by_dim: vec![4, 6, 4, 1] },
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
/// use delaunay::topology::characteristics::validation;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0]),
///     vertex!([1.0, 0.0]),
///     vertex!([0.5, 1.0]),
/// ];
/// let dt = DelaunayTriangulation::new(&vertices).unwrap();
///
/// let result = validation::validate_triangulation_euler(dt.tds()).unwrap();
/// assert_eq!(result.chi, 1);
/// assert_eq!(result.counts.count(0), 3);  // 3 vertices
/// assert_eq!(result.counts.count(1), 3);  // 3 edges
/// assert_eq!(result.counts.count(2), 1);  // 1 face
/// assert!(result.is_valid());
/// ```
///
/// # Errors
///
/// Returns `TopologyError::Counting` or `TopologyError::Classification`
/// if the underlying operations fail.
pub fn validate_triangulation_euler<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
) -> Result<TopologyCheckResult, TopologyError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    // Precompute the facet map once and reuse it for both counting and classification.
    //
    // Avoid building the map for empty triangulations.
    let facet_to_cells = if tds.number_of_cells() == 0 {
        FacetToCellsMap::default()
    } else {
        tds.build_facet_to_cells_map()
            .map_err(|e| TopologyError::Counting(format!("Failed to build facet map: {e}")))?
    };

    Ok(validate_triangulation_euler_with_facet_to_cells_map(
        tds,
        &facet_to_cells,
    ))
}

pub(crate) fn validate_triangulation_euler_with_facet_to_cells_map<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    facet_to_cells: &FacetToCellsMap,
) -> TopologyCheckResult
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let counts = count_simplices_with_facet_to_cells_map(tds, facet_to_cells);
    let chi = euler_characteristic(&counts);

    let num_cells = tds.number_of_cells();
    let classification = if num_cells == 0 {
        TopologyClassification::Empty
    } else if num_cells == 1 {
        TopologyClassification::SingleSimplex(D)
    } else if facet_to_cells.values().any(|cells| cells.len() == 1) {
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
            counts: SimplexCounts {
                by_dim: vec![4, 6, 4, 1],
            },
            notes: vec![],
        };
        assert!(valid_result.is_valid());

        let invalid_result = TopologyCheckResult {
            chi: 0,
            expected: Some(1),
            classification: TopologyClassification::Ball(3),
            counts: SimplexCounts {
                by_dim: vec![4, 6, 4, 1],
            },
            notes: vec!["Mismatch".to_string()],
        };
        assert!(!invalid_result.is_valid());

        let unknown_result = TopologyCheckResult {
            chi: 42,
            expected: None,
            classification: TopologyClassification::Unknown,
            counts: SimplexCounts { by_dim: vec![1] },
            notes: vec![],
        };
        assert!(unknown_result.is_valid()); // Unknown classification is considered valid
    }
}
