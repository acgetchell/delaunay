//! TDS-level facet-incidence analysis functions.
//!
//! This module implements the `FacetIncidenceAnalysis` trait for triangulation
//! data structures, providing methods to identify and analyze one-sided facet
//! incidence in d-dimensional triangulations.

#![forbid(unsafe_code)]

#[cfg(test)]
use super::collections::FacetToSimplicesMap;
use super::{
    facet::{FacetError, FacetToSimplicesIndex, FacetView, OneSidedFacetsIter},
    tds::{Tds, TdsError},
    traits::facet_incidence_analysis::FacetIncidenceAnalysis,
};
use std::ptr;

/// Counts one-sided raw facet incidences and rejects non-manifold multiplicities.
///
/// This test helper exercises raw multiplicity parsing only. Production
/// topology-aware boundary classification uses [`FacetToSimplicesIndex`] so
/// admissible periodic self-identifications remain closed topology instead of
/// being counted as boundary.
#[cfg(test)]
fn number_of_one_sided_facets_in_map(
    facet_to_simplices: &FacetToSimplicesMap,
) -> Result<usize, TdsError> {
    let mut count = 0usize;
    for (&facet_key, simplices) in facet_to_simplices {
        match simplices.len() {
            1 => count = count.saturating_add(1),
            2 => {}
            found => {
                return Err(FacetError::InvalidFacetMultiplicity { facet_key, found }.into());
            }
        }
    }
    Ok(count)
}

/// Implementation of `FacetIncidenceAnalysis` trait for `Tds`.
///
/// This implementation provides efficient one-sided facet incidence analysis
/// for d-dimensional triangulations using the triangulation data structure.
impl<U, V, const D: usize> FacetIncidenceAnalysis<U, V, D> for Tds<U, V, D> {
    /// Identifies all one-sided facet incidences in the TDS.
    ///
    /// This is incidence analysis only: a one-sided facet can be a Euclidean
    /// boundary facet, but topology-aware callers must still decide whether it
    /// is true boundary or a closed periodic self-identification.
    ///
    /// # Triangulation Invariant
    ///
    /// This method relies on the fundamental invariant of Delaunay triangulations:
    /// **every facet is one-sided or two-sided.** Any facet shared by 0, 3, or
    /// more simplices indicates a structural/topological error in the
    /// triangulation.
    ///
    /// For a comprehensive discussion of all topological invariants in Delaunay triangulations,
    /// see the [Topological Invariants](crate::tds::Tds#topological-invariants)
    /// section in the triangulation data structure documentation.
    ///
    /// # Returns
    ///
    /// A `Result<OneSidedFacetsIter<'_, U, V, D>, TdsError>` containing an
    /// iterator over one-sided facet incidences. The iterator owns a sorted
    /// handle list derived from the facet-incidence index and yields
    /// `Result<FacetView, FacetError>` items while still surfacing corrupted
    /// facet views during iteration.
    ///
    /// # Errors
    ///
    /// Returns a [`TdsError`] (typically
    /// [`crate::prelude::tds::FacetError`]) if:
    /// - The one-sided-facet iterator cannot be constructed.
    /// - A facet index is out of bounds (indicates data corruption)
    /// - A referenced simplex is not found in the triangulation (indicates data corruption)
    ///
    /// Individual iterator items return [`FacetError`] if a facet view cannot
    /// be created or keyed from the simplices.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Query(#[from] delaunay::query::QueryError),
    /// #     #[error(transparent)]
    /// #     Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)]
    /// #     Facet(#[from] delaunay::prelude::tds::FacetError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// // Create a simple 3D triangulation (single tetrahedron)
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    ///
    /// // High-level API returns `QueryError` if the underlying TDS is corrupted.
    /// let high_level_count = dt
    ///     .boundary_facets()?
    ///     .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))?;
    /// assert_eq!(high_level_count, 4);
    ///
    /// // TDS-level API reports raw one-sided incidence.
    /// let count = dt
    ///     .tds()
    ///     .one_sided_facets()?
    ///     .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))?;
    /// assert_eq!(count, 4);
    /// # Ok(())
    /// # }
    /// ```
    fn one_sided_facets(&self) -> Result<OneSidedFacetsIter<'_, U, V, D>, TdsError> {
        // Build an owner-bound index from facet keys to the simplices that contain them.
        let facet_to_simplices = self.build_facet_to_simplices_index()?;

        // Create the one-sided facets iterator.
        OneSidedFacetsIter::try_new(&facet_to_simplices).map_err(TdsError::from)
    }

    /// Checks if a specific facet has one-sided incidence.
    ///
    /// This does not classify manifold boundary. Use
    /// [`Triangulation::boundary_facets`](crate::Triangulation::boundary_facets)
    /// or
    /// [`DelaunayTriangulation::boundary_facets`](crate::DelaunayTriangulation::boundary_facets)
    /// when the declared global topology matters.
    ///
    /// # Performance Note
    ///
    /// This method rebuilds the facet-to-simplices index on every call, which has O(N·F) complexity.
    /// For checking multiple facets in hot paths, prefer using
    /// [`FacetIncidenceAnalysis::is_one_sided_facet_with_index`] with a
    /// precomputed index to avoid recomputation.
    ///
    /// # Arguments
    ///
    /// * `facet` - The facet to check.
    ///
    /// # Returns
    ///
    /// `Ok(true)` if the facet has one-sided incidence, `Ok(false)` if it is
    /// two-sided or absent from the facet index.
    ///
    /// # Errors
    ///
    /// Returns a [`TdsError`] if the facet index cannot be built or the facet
    /// view belongs to a different TDS.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Query(#[from] delaunay::query::QueryError),
    /// #     #[error(transparent)]
    /// #     Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)]
    /// #     Facet(#[from] delaunay::prelude::tds::FacetError),
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
    /// // Get one-sided facets using the TDS incidence API.
    /// let Some(first_facet) = dt.tds().one_sided_facets()?.next().transpose()? else {
    ///     return Ok(());
    /// };
    /// assert!(dt.tds().is_one_sided_facet(&first_facet)?);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn is_one_sided_facet(&self, facet: &FacetView<'_, U, V, D>) -> Result<bool, TdsError> {
        ensure_facet_view_owner(self, facet)?;
        let facet_to_simplices = self.build_facet_to_simplices_index()?;
        self.is_one_sided_facet_with_index(facet, &facet_to_simplices)
    }

    /// Checks if a specific facet has one-sided incidence using a precomputed facet index.
    ///
    /// This is an optimized version of
    /// [`FacetIncidenceAnalysis::is_one_sided_facet`] that accepts a prebuilt
    /// owner-bound facet-to-simplices index to avoid recomputation in tight
    /// loops.
    ///
    /// # Arguments
    ///
    /// * `facet` - The facet to check.
    /// * `facet_to_simplices` - Precomputed index from facet keys to simplices containing them.
    ///
    /// # Returns
    ///
    /// `Ok(true)` if the facet has one-sided incidence, `Ok(false)` if it is
    /// two-sided or absent from the facet index.
    ///
    /// # Errors
    ///
    /// Returns a [`TdsError`] if the supplied index or facet view belongs to a
    /// different TDS. A facet view borrowed from a different TDS is rejected
    /// before its key is compared with the supplied index.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Query(#[from] delaunay::query::QueryError),
    /// #     #[error(transparent)]
    /// #     Tds(#[from] delaunay::prelude::tds::TdsError),
    /// #     #[error(transparent)]
    /// #     Facet(#[from] delaunay::prelude::tds::FacetError),
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
    /// // Build the facet index once for multiple queries
    /// let facet_to_simplices = dt.tds().build_facet_to_simplices_index()?;
    ///
    /// // Check one-sided incidence efficiently using the iterator API.
    /// for facet in dt.tds().one_sided_facets()? {
    ///     let facet = facet?;
    ///     let is_one_sided = dt.tds().is_one_sided_facet_with_index(&facet, &facet_to_simplices)?;
    ///     println!("Facet is one-sided: {is_one_sided}");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn is_one_sided_facet_with_index(
        &self,
        facet: &FacetView<'_, U, V, D>,
        facet_to_simplices: &FacetToSimplicesIndex<'_, U, V, D>,
    ) -> Result<bool, TdsError> {
        ensure_facet_view_owner(self, facet)?;
        ensure_facet_index_owner(self, facet_to_simplices)?;
        // Use FacetView's cached key path.
        Ok(facet_to_simplices.is_one_sided_facet_key(&facet.key()))
    }

    /// Returns the number of one-sided facet incidences in the TDS.
    ///
    /// This method efficiently counts one-sided facets from the derived facet-incidence map
    /// without allocating or cloning `Facet` objects, making it O(|facets|) with
    /// no per-simplex `facets()` calls.
    ///
    /// # Returns
    ///
    /// A `Result` containing the number of one-sided facet incidences,
    /// or a [`TdsError`] if the facet index cannot be built.
    ///
    /// # Errors
    ///
    /// Returns a [`TdsError`] if the facet-to-simplices index cannot be built.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Query(#[from] delaunay::query::QueryError),
    /// #     #[error(transparent)]
    /// #     Tds(#[from] delaunay::prelude::tds::TdsError),
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
    /// // A single Euclidean tetrahedron has 4 one-sided facets.
    /// assert_eq!(dt.tds().number_of_one_sided_facets()?, 4);
    /// # Ok(())
    /// # }
    /// ```
    fn number_of_one_sided_facets(&self) -> Result<usize, TdsError> {
        let facet_to_simplices = self.build_facet_to_simplices_index()?;
        Ok(facet_to_simplices
            .iter()
            .filter(|incidence| incidence.is_one_sided())
            .count())
    }
}

fn ensure_facet_view_owner<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    facet: &FacetView<'_, U, V, D>,
) -> Result<(), TdsError> {
    if ptr::eq(facet.tds(), tds) {
        Ok(())
    } else {
        Err(FacetError::FacetOwnerMismatch {
            simplex_key: facet.simplex_key(),
            facet_index: facet.facet_index(),
        }
        .into())
    }
}

/// Rejects owner-bound facet indexes from another TDS before their keys are queried.
fn ensure_facet_index_owner<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    facet_to_simplices: &FacetToSimplicesIndex<'_, U, V, D>,
) -> Result<(), TdsError> {
    if ptr::eq(facet_to_simplices.tds(), tds) {
        Ok(())
    } else {
        Err(FacetError::FacetIndexOwnerMismatch.into())
    }
}

#[cfg(test)]
mod tests {
    use super::{FacetIncidenceAnalysis, number_of_one_sided_facets_in_map};
    use crate::core::collections::{FacetToSimplicesMap, SmallBuffer};
    use crate::core::facet::{FacetError, FacetHandle, FacetToSimplicesIndex, FacetView};
    use crate::core::query::QueryError;
    use crate::core::simplex::Simplex;
    use crate::core::tds::{SimplexKey, Tds, TdsError};
    use crate::geometry::point::Point;
    use crate::triangulation::DelaunayTriangulation;
    use crate::try_vertices_from_points;
    use crate::vertex;
    use std::assert_matches;

    #[cfg(feature = "diagnostics")]
    macro_rules! test_debug {
        ($($arg:tt)*) => {{
            tracing::debug!($($arg)*);
        }};
    }

    #[cfg(not(feature = "diagnostics"))]
    macro_rules! test_debug {
        ($($arg:tt)*) => {{
            let _ = core::format_args!($($arg)*);
        }};
    }

    // =============================================================================
    // SINGLE SIMPLEX TESTS
    // =============================================================================

    #[expect(
        clippy::too_many_lines,
        reason = "one-sided incidence regression test keeps setup and assertions together"
    )]
    #[test]
    fn test_one_sided_facets_single_simplices() {
        // Test one-sided incidence analysis for single simplices in different dimensions.

        // Test Case 1: 2D triangle - all 3 edges should be one-sided facets.
        {
            let points = vec![
                Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.5, 1.0]).expect("finite point coordinates"),
            ];
            let vertices = try_vertices_from_points(&points).expect("finite point coordinates");
            let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();

            assert_eq!(
                dt.number_of_simplices(),
                1,
                "2D triangle should have 1 simplex"
            );
            assert_eq!(dt.dim(), 2, "Should be 2-dimensional");

            let one_sided_count = dt
                .tds()
                .one_sided_facets()
                .unwrap()
                .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))
                .unwrap();
            assert_eq!(
                one_sided_count, 3,
                "2D triangle should have 3 one-sided facets"
            );

            // Verify all facets are one-sided using cached index.
            let facet_to_simplices = dt
                .tds()
                .build_facet_to_simplices_index()
                .expect("Should build facet index");
            assert!(dt.tds().one_sided_facets().unwrap().all(|f| {
                let f = f.expect("valid one-sided facet");
                dt.tds()
                    .is_one_sided_facet_with_index(&f, &facet_to_simplices)
                    .expect("Should not fail for valid facets")
            }));
        }

        // Test Case 2: 3D tetrahedron - all 4 faces should be one-sided facets.
        {
            let points = vec![
                Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
            ];
            let vertices = try_vertices_from_points(&points).expect("finite point coordinates");
            let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();

            assert_eq!(
                dt.number_of_simplices(),
                1,
                "3D tetrahedron should have 1 simplex"
            );
            assert_eq!(dt.dim(), 3, "Should be 3-dimensional");

            let one_sided_count = dt
                .tds()
                .one_sided_facets()
                .unwrap()
                .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))
                .unwrap();
            assert_eq!(
                one_sided_count, 4,
                "3D tetrahedron should have 4 one-sided facets"
            );

            // Verify all facets are one-sided.
            let facet_to_simplices = dt
                .tds()
                .build_facet_to_simplices_index()
                .expect("Should build facet index");
            assert!(dt.tds().one_sided_facets().unwrap().all(|f| {
                let f = f.expect("valid one-sided facet");
                dt.tds()
                    .is_one_sided_facet_with_index(&f, &facet_to_simplices)
                    .expect("Should not fail for valid facets")
            }));
        }

        // Test Case 3: 4D simplex - all 5 tetrahedra should be one-sided facets.
        {
            let points = vec![
                Point::try_new([0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 0.0, 0.0, 1.0]).expect("finite point coordinates"),
            ];
            let vertices = try_vertices_from_points(&points).expect("finite point coordinates");
            let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();

            assert_eq!(
                dt.number_of_simplices(),
                1,
                "4D simplex should have 1 simplex"
            );
            assert_eq!(dt.dim(), 4, "Should be 4-dimensional");

            let one_sided_count = dt
                .tds()
                .one_sided_facets()
                .unwrap()
                .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))
                .unwrap();
            assert_eq!(
                one_sided_count, 5,
                "4D simplex should have 5 one-sided facets"
            );

            // Verify all facets are one-sided.
            let facet_to_simplices = dt
                .tds()
                .build_facet_to_simplices_index()
                .expect("Should build facet index");
            let confirmed_one_sided = dt
                .tds()
                .one_sided_facets()
                .unwrap()
                .filter(|f| {
                    let f = f.as_ref().expect("valid one-sided facet");
                    dt.tds()
                        .is_one_sided_facet_with_index(f, &facet_to_simplices)
                        .expect("Should not fail for valid facets")
                })
                .count();
            assert_eq!(confirmed_one_sided, 5, "All facets should be one-sided");
        }

        // Test Case 4: Empty triangulation
        {
            let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();
            assert_eq!(
                dt.number_of_simplices(),
                0,
                "Empty triangulation should have no simplices"
            );

            let one_sided_count = dt
                .tds()
                .one_sided_facets()
                .unwrap()
                .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))
                .unwrap();
            assert_eq!(
                one_sided_count, 0,
                "Empty triangulation should have no one-sided facets"
            );
        }

        // Test Case 5: 5D simplex - all 6 facets should be one-sided facets.
        {
            let points = vec![
                Point::try_new([0.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([1.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 0.0, 0.0, 0.0, 1.0]).expect("finite point coordinates"),
            ];
            let vertices = try_vertices_from_points(&points).expect("finite point coordinates");
            let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();

            assert_eq!(
                dt.number_of_simplices(),
                1,
                "5D simplex should have 1 simplex"
            );
            assert_eq!(dt.dim(), 5, "Should be 5-dimensional");

            let one_sided_count = dt
                .tds()
                .one_sided_facets()
                .unwrap()
                .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))
                .unwrap();
            assert_eq!(
                one_sided_count, 6,
                "5D simplex should have 6 one-sided facets"
            );

            let facet_to_simplices = dt
                .tds()
                .build_facet_to_simplices_index()
                .expect("Should build facet index");
            let confirmed_one_sided = dt
                .tds()
                .one_sided_facets()
                .unwrap()
                .filter(|f| {
                    let f = f.as_ref().expect("valid one-sided facet");
                    dt.tds()
                        .is_one_sided_facet_with_index(f, &facet_to_simplices)
                        .expect("Should not fail for valid facets")
                })
                .count();
            assert_eq!(
                confirmed_one_sided, 6,
                "All 5D simplex facets should be one-sided"
            );
        }

        test_debug!(
            "✓ Single simplex one-sided incidence analysis works correctly in 2D, 3D, 4D, 5D, and empty cases"
        );
    }

    #[test]
    fn test_one_sided_facets_method_coverage() {
        // Test method delegation and implementation path coverage

        // Test case 1: Basic method delegation and error propagation
        {
            let points = vec![
                Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
            ];
            let vertices = try_vertices_from_points(&points).expect("finite point coordinates");
            let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();

            // Test one_sided_facets() normal path.
            let one_sided_count = dt
                .tds()
                .one_sided_facets()
                .unwrap()
                .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))
                .unwrap();
            assert_eq!(
                one_sided_count, 4,
                "Single tetrahedron has 4 one-sided facets"
            );

            // Test is_one_sided_facet() delegation (builds facet index internally).
            if let Some(facet) = dt.tds().one_sided_facets().unwrap().next() {
                let facet = facet.unwrap();
                let result = dt.tds().is_one_sided_facet(&facet);
                assert!(result.is_ok(), "Should not error on valid facet");
                assert!(result.unwrap(), "Facet should be one-sided");
            }
        }

        // Test case 2: Capacity allocation and vector operations
        {
            let points = vec![
                Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
                Point::try_new([0.5, 0.5, 0.5]).expect("finite point coordinates"), // Interior point
            ];
            let vertices = try_vertices_from_points(&points).expect("finite point coordinates");
            let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();

            // After robust cleanup and facet-sharing filtering, we may end up with a single simplex
            assert!(
                dt.number_of_simplices() >= 1,
                "Should have at least one simplex for this test"
            );

            // Exercise capacity allocation, cache initialization, and vector push operations
            let one_sided_count = dt
                .tds()
                .one_sided_facets()
                .unwrap()
                .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))
                .unwrap();
            assert!(one_sided_count > 0, "Should have one-sided facets");
            assert!(
                one_sided_count >= 4,
                "Should have at least 4 one-sided facets"
            );
        }

        test_debug!("✓ One-sided facets method coverage and delegation work correctly");
    }

    // =============================================================================
    // ADDITIONAL TESTS FOR UNCOVERED ERROR PATHS
    // =============================================================================

    #[test]
    fn test_boundary_facets_invalid_facet_index_error() {
        test_debug!("Testing boundary_facets with invalid facet index error path");

        // Note: This error path (InvalidFacetIndex) is difficult to trigger in practice
        // because the facet-to-simplices mapping is built from valid facets.
        // We test this by confirming the error structure exists and can be created.

        // Test that the error can be created and has correct structure
        let error = TdsError::FacetError(FacetError::InvalidFacetIndex {
            index: 42,
            facet_count: 4,
        });

        // Verify error display includes useful information
        let error_string = format!("{error}");
        assert!(
            error_string.contains("42"),
            "Error should contain the invalid index"
        );
        assert!(
            error_string.contains('4'),
            "Error should contain the facet count"
        );

        test_debug!("  Error structure: {error}");
        test_debug!("  ✓ InvalidFacetIndex error path structure verified");
    }

    #[test]
    fn test_boundary_facets_simplex_not_found_error() {
        test_debug!("Testing boundary_facets with simplex not found error path");

        // Note: This error path (SimplexNotFoundInTriangulation) is also difficult to trigger
        // in practice because the mapping is built from existing simplices.
        // We test the error structure.

        // Test that the error can be created
        let error = TdsError::FacetError(FacetError::SimplexNotFoundInTriangulation);

        // Verify error display is meaningful
        let error_string = format!("{error}");
        assert!(
            error_string.contains("Simplex") || error_string.contains("simplex"),
            "Error should mention simplex: {error_string}"
        );

        test_debug!("  Error structure: {error}");
        test_debug!("  ✓ SimplexNotFoundInTriangulation error path structure verified");
    }

    #[test]
    fn test_is_one_sided_facet_with_index_consistency() {
        test_debug!("Testing is_one_sided_facet_with_index consistency with one_sided_facets");

        // Create a valid triangulation
        let points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        let vertices = try_vertices_from_points(&points).expect("finite point coordinates");
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();

        // Build facet index
        let facet_to_simplices = dt
            .tds()
            .build_facet_to_simplices_index()
            .expect("Should build index");

        // Get all one-sided facets and verify they are correctly identified.
        let mut one_sided_count = 0;

        for facet in dt.tds().one_sided_facets().unwrap() {
            let facet = facet.unwrap();
            let is_one_sided = dt
                .tds()
                .is_one_sided_facet_with_index(&facet, &facet_to_simplices)
                .expect("Should successfully check one-sided status");

            assert!(
                is_one_sided,
                "All facets returned by one_sided_facets() should be one-sided"
            );
            one_sided_count += 1;
        }

        assert_eq!(
            one_sided_count, 4,
            "Single tetrahedron should have 4 one-sided facets"
        );

        let reported_count = dt
            .tds()
            .one_sided_facets()
            .unwrap()
            .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))
            .unwrap();
        assert_eq!(
            one_sided_count, reported_count,
            "One-sided facet count should be consistent"
        );

        test_debug!("  ✓ All {one_sided_count} one-sided facets correctly identified");
        test_debug!("  ✓ is_one_sided_facet_with_index consistency verified");
    }

    #[test]
    fn facet_index_rejects_invalid_multiplicity() {
        let points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        let vertices = try_vertices_from_points(&points).expect("finite point coordinates");
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();
        let facet = dt.boundary_facets().unwrap().next().unwrap().unwrap();
        let facet_key = facet.key();

        let mut facet_to_simplices = dt.tds().build_facet_to_simplices_map().unwrap();
        facet_to_simplices.remove(&facet_key);
        let facet_index =
            FacetToSimplicesIndex::try_from_map(dt.tds(), &facet_to_simplices).unwrap();
        assert!(
            !dt.tds()
                .is_one_sided_facet_with_index(&facet, &facet_index)
                .unwrap()
        );

        facet_to_simplices.insert(facet_key, SmallBuffer::new());
        let err = FacetToSimplicesIndex::try_from_map(dt.tds(), &facet_to_simplices).unwrap_err();
        assert_matches!(err, FacetError::InvalidFacetMultiplicity { found: 0, .. });

        facet_to_simplices.insert(
            facet_key,
            [
                FacetHandle::from_validated(facet.simplex_key(), 0),
                FacetHandle::from_validated(facet.simplex_key(), 1),
                FacetHandle::from_validated(facet.simplex_key(), 2),
            ]
            .into_iter()
            .collect(),
        );

        let err = FacetToSimplicesIndex::try_from_map(dt.tds(), &facet_to_simplices).unwrap_err();

        assert_matches!(err, FacetError::InvalidFacetMultiplicity { found: 3, .. });
    }

    #[test]
    fn one_sided_facet_query_rejects_foreign_facet_view() {
        let points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        let vertices = try_vertices_from_points(&points).expect("finite point coordinates");
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let foreign_dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let foreign_facet = foreign_dt
            .boundary_facets()
            .unwrap()
            .next()
            .unwrap()
            .unwrap();

        let err = dt.tds().is_one_sided_facet(&foreign_facet).unwrap_err();

        assert_matches!(
            err,
            TdsError::FacetError(FacetError::FacetOwnerMismatch { .. })
        );
    }

    #[test]
    fn one_sided_facet_query_rejects_foreign_facet_index() {
        let points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        let vertices = try_vertices_from_points(&points).expect("finite point coordinates");
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let foreign_dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let facet = dt.boundary_facets().unwrap().next().unwrap().unwrap();
        let foreign_index = foreign_dt.tds().build_facet_to_simplices_index().unwrap();

        let err = dt
            .tds()
            .is_one_sided_facet_with_index(&facet, &foreign_index)
            .unwrap_err();

        assert_matches!(
            err,
            TdsError::FacetError(FacetError::FacetIndexOwnerMismatch)
        );
    }

    #[test]
    fn test_one_sided_facet_count_rejects_invalid_multiplicity() {
        let mut facet_to_simplices = FacetToSimplicesMap::default();
        facet_to_simplices.insert(
            0xCAFE,
            [
                FacetHandle::from_validated(SimplexKey::default(), 0),
                FacetHandle::from_validated(SimplexKey::default(), 1),
                FacetHandle::from_validated(SimplexKey::default(), 2),
            ]
            .into_iter()
            .collect(),
        );

        let err = number_of_one_sided_facets_in_map(&facet_to_simplices).unwrap_err();

        assert_matches!(
            err,
            TdsError::FacetError(FacetError::InvalidFacetMultiplicity {
                facet_key: 0xCAFE,
                found: 3
            })
        );
    }

    #[test]
    fn test_boundary_facets_error_propagation_from_build_map() {
        let points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        let vertices = try_vertices_from_points(&points).expect("finite point coordinates");
        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let (simplex_key, _) = dt.tri.tds.simplices().next().unwrap();
        let first_vertex = dt.tri.tds.simplex(simplex_key).unwrap().vertices()[0];

        {
            let simplex = dt.tri.tds.simplex_mut(simplex_key).unwrap();
            while simplex.number_of_vertices() <= usize::from(u8::MAX) + 1 {
                simplex.push_vertex_key(first_vertex);
            }
        }

        match dt.boundary_facets() {
            Ok(_) => panic!("corrupted facet map should return a query error"),
            Err(QueryError::TriangulationCorrupted { source })
                if matches!(*source, TdsError::IndexOutOfBounds { .. }) => {}
            Err(err) => panic!("expected index-out-of-bounds query error, got {err:?}"),
        }
    }

    #[test]
    fn test_number_of_one_sided_facets_delegation() {
        test_debug!("Testing number_of_one_sided_facets delegation to one_sided_facets");

        // This test exercises the delegation to one_sided_facets() and result
        // transformation, ensuring the method properly delegates and transforms
        // the result.

        let points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        let vertices = try_vertices_from_points(&points).expect("finite point coordinates");
        let dt = DelaunayTriangulation::builder(&vertices).build().unwrap();

        // Test both methods return consistent results
        let one_sided_facets_count = dt
            .tds()
            .one_sided_facets()
            .unwrap()
            .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))
            .unwrap();
        let one_sided_count = dt
            .tds()
            .number_of_one_sided_facets()
            .expect("Should get one-sided count");

        assert_eq!(
            one_sided_facets_count, one_sided_count,
            "number_of_one_sided_facets should equal one_sided_facets().count()"
        );

        assert_eq!(
            one_sided_count, 4,
            "Single tetrahedron should have 4 one-sided facets"
        );

        test_debug!("  ✓ number_of_one_sided_facets delegation working correctly");
        test_debug!("    - one_sided_facets().count(): {one_sided_facets_count}");
        test_debug!("    - number_of_one_sided_facets(): {one_sided_count}");
    }

    #[test]
    fn periodic_self_identified_one_sided_facet_is_not_boundary() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();

        let mut simplex = Simplex::try_new(vec![v0, v1, v2]).unwrap();
        simplex
            .set_periodic_vertex_offsets(vec![[0, 0], [0, 0], [1, 0]])
            .unwrap();
        let simplex_key = tds.insert_simplex_with_mapping(simplex).unwrap();
        tds.simplex_mut(simplex_key)
            .unwrap()
            .set_neighbors_from_keys([Some(simplex_key), None, None])
            .unwrap();

        let facet_index = tds.build_facet_to_simplices_index().unwrap();
        let self_identified_facet = FacetView::try_new(&tds, simplex_key, 0).unwrap();
        let self_identified_key = self_identified_facet.key();
        let boundary_keys: Vec<_> = tds
            .one_sided_facets()
            .unwrap()
            .map(|facet| facet.unwrap().key())
            .collect();

        assert!(
            tds.is_one_sided_facet_with_index(&self_identified_facet, &facet_index)
                .unwrap()
        );
        assert!(boundary_keys.contains(&self_identified_key));
        assert_eq!(boundary_keys.len(), 3);
        assert_eq!(tds.number_of_one_sided_facets().unwrap(), 3);
    }

    #[test]
    fn test_invalid_facet_multiplicity_error_creation() {
        test_debug!("Testing InvalidFacetMultiplicity error creation and formatting");

        // Test that the error can be created with various multiplicity values
        let test_cases = [
            (0, "zero multiplicity"),
            (3, "triple multiplicity"),
            (5, "excessive multiplicity"),
        ];

        for &(multiplicity, description) in &test_cases {
            let facet_key = 0x1234_5678_9ABC_DEF0_u64; // Example facet key
            let error = TdsError::FacetError(FacetError::InvalidFacetMultiplicity {
                facet_key,
                found: multiplicity,
            });

            // Verify error display includes all necessary information
            let error_string = format!("{error}");
            assert!(
                error_string.contains(&multiplicity.to_string()),
                "Error should contain multiplicity {multiplicity}: {error_string}"
            );
            assert!(
                error_string.contains(&format!("{facet_key:016x}")),
                "Error should contain facet key in hex: {error_string}"
            );
            assert!(
                error_string.contains("expected 1 (one-sided) or 2 (two-sided)"),
                "Error should explain valid multiplicities: {error_string}"
            );

            test_debug!("  ✓ {description}: {error}");
        }

        test_debug!("  ✓ InvalidFacetMultiplicity error creation and formatting verified");
    }
}
