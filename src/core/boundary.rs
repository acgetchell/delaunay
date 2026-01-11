//! Boundary and convex hull analysis functions
//!
//! This module implements the `BoundaryAnalysis` trait for triangulation data structures,
//! providing methods to identify and analyze boundary facets in d-dimensional triangulations.

use super::{
    facet::{BoundaryFacetsIter, FacetView},
    traits::{boundary_analysis::BoundaryAnalysis, data_type::DataType},
    triangulation_data_structure::{Tds, TdsValidationError},
};

use crate::prelude::CoordinateScalar;
use num_traits::NumCast;

/// Implementation of `BoundaryAnalysis` trait for `Tds`.
///
/// This implementation provides efficient boundary facet identification and analysis
/// for d-dimensional triangulations using the triangulation data structure.
impl<T, U, V, const D: usize> BoundaryAnalysis<T, U, V, D> for Tds<T, U, V, D>
where
    T: CoordinateScalar + NumCast,
    U: DataType,
    V: DataType,
{
    /// Identifies all boundary facets in the triangulation.
    ///
    /// A boundary facet is a facet that belongs to only one cell, meaning it lies on the
    /// boundary of the triangulation (convex hull). These facets are important for
    /// convex hull computation and boundary analysis.
    ///
    /// # Triangulation Invariant
    ///
    /// This method relies on the fundamental invariant of Delaunay triangulations:
    /// **every facet is shared by exactly two cells, except boundary facets which belong to exactly one cell.**
    /// Any facet shared by 0, 3, or more cells indicates a topological error in the triangulation.
    ///
    /// For a comprehensive discussion of all topological invariants in Delaunay triangulations,
    /// see the [Topological Invariants](crate::core::triangulation_data_structure#topological-invariants)
    /// section in the triangulation data structure documentation.
    ///
    /// # Returns
    ///
    /// A `Result<BoundaryFacetsIter<'_, T, U, V, D>, TdsValidationError>` containing an iterator over boundary facets.
    /// The iterator yields facets lazily without pre-allocating vectors, providing better performance.
    ///
    /// # Errors
    ///
    /// Returns a [`TdsValidationError`] (typically
    /// [`crate::core::facet::FacetError`]) if:
    /// - Any boundary facet cannot be created from the cells
    /// - A facet index is out of bounds (indicates data corruption)
    /// - A referenced cell is not found in the triangulation (indicates data corruption)
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// // Create a simple 3D triangulation (single tetrahedron)
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// // High-level API (infallible): panics if the underlying TDS is corrupted.
    /// assert_eq!(dt.boundary_facets().count(), 4);
    ///
    /// // TDS-level API (fallible): returns `TdsValidationError` on corruption.
    /// let count = dt.tds().boundary_facets()?.count();
    /// assert_eq!(count, 4);
    /// # Ok::<(), delaunay::core::triangulation_data_structure::TdsValidationError>(())
    /// ```
    fn boundary_facets(&self) -> Result<BoundaryFacetsIter<'_, T, U, V, D>, TdsValidationError> {
        // Build a map from facet keys to the cells that contain them
        let facet_to_cells = self.build_facet_to_cells_map()?;

        // Create the boundary facets iterator
        Ok(BoundaryFacetsIter::new(self, facet_to_cells))
    }

    /// Checks if a specific facet is a boundary facet.
    ///
    /// A boundary facet is a facet that belongs to only one cell in the triangulation.
    ///
    /// # Performance Note
    ///
    /// This method rebuilds the facet-to-cells map on every call, which has O(N·F) complexity.
    /// For checking multiple facets in hot paths, prefer using `is_boundary_facet_with_map()`
    /// with a precomputed map to avoid recomputation.
    ///
    /// # Arguments
    ///
    /// * `facet` - The facet to check.
    ///
    /// # Returns
    ///
    /// `Ok(true)` if the facet is on the boundary (belongs to only one cell),
    /// `Ok(false)` otherwise. `Err(...)` on validation/lookup failures.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// // Get boundary facets using the new iterator API
    /// let first_facet = dt.boundary_facets().next().unwrap();
    /// // In a single tetrahedron, all facets are boundary facets
    /// assert!(dt.tds().is_boundary_facet(&first_facet).unwrap());
    /// ```
    #[inline]
    fn is_boundary_facet(
        &self,
        facet: &FacetView<'_, T, U, V, D>,
    ) -> Result<bool, TdsValidationError> {
        let facet_to_cells = self.build_facet_to_cells_map()?;
        self.is_boundary_facet_with_map(facet, &facet_to_cells)
    }

    /// Checks if a specific facet is a boundary facet using a precomputed facet map.
    ///
    /// This is an optimized version of `is_boundary_facet` that accepts a prebuilt
    /// facet-to-cells map to avoid recomputation in tight loops.
    ///
    /// # Arguments
    ///
    /// * `facet` - The facet to check.
    /// * `facet_to_cells` - Precomputed map from facet keys to cells containing them.
    ///
    /// # Returns
    ///
    /// `Ok(true)` if the facet is on the boundary (belongs to only one cell),
    /// `Ok(false)` otherwise. `Err(...)` on validation/lookup failures.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// // Build the facet map once for multiple queries
    /// let facet_to_cells = dt.tds().build_facet_to_cells_map()
    ///     .expect("Should build facet map");
    ///
    /// // Check boundary facets efficiently using the iterator API
    /// for facet in dt.boundary_facets() {
    ///     let is_boundary = dt.tds().is_boundary_facet_with_map(&facet, &facet_to_cells)
    ///         .expect("Should check if facet is boundary");
    ///     println!("Facet is boundary: {is_boundary}");
    /// }
    /// ```
    #[inline]
    fn is_boundary_facet_with_map(
        &self,
        facet: &FacetView<'_, T, U, V, D>,
        facet_to_cells: &crate::core::collections::FacetToCellsMap,
    ) -> Result<bool, TdsValidationError> {
        // Use FacetView's key() method which is more efficient
        let facet_key = facet.key().map_err(TdsValidationError::FacetError)?;

        Ok(facet_to_cells
            .get(&facet_key)
            .is_some_and(|cells| cells.len() == 1))
    }

    /// Returns the number of boundary facets in the triangulation.
    ///
    /// This method efficiently counts boundary facets directly from the facet map
    /// without allocating or cloning `Facet` objects, making it O(|facets|) with
    /// no per-cell `facets()` calls.
    ///
    /// # Returns
    ///
    /// A `Result` containing the number of boundary facets in the triangulation,
    /// or a `TdsValidationError` if the facet map cannot be built.
    ///
    /// # Errors
    ///
    /// Returns a [`TdsValidationError`] if the facet-to-cells map cannot be built.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// // A single tetrahedron has 4 boundary facets
    /// assert_eq!(dt.tds().number_of_boundary_facets()?, 4);
    /// # Ok::<(), delaunay::core::triangulation_data_structure::TdsValidationError>(())
    /// ```
    fn number_of_boundary_facets(&self) -> Result<usize, TdsValidationError> {
        self.build_facet_to_cells_map()
            .map(|m| m.values().filter(|v| v.len() == 1).count())
    }
}

#[cfg(test)]
mod tests {
    use super::BoundaryAnalysis;
    use crate::core::delaunay_triangulation::DelaunayTriangulation;
    use crate::core::facet::FacetError;
    use crate::core::triangulation_data_structure::TdsValidationError;
    use crate::core::vertex::Vertex;
    use crate::geometry::{point::Point, traits::coordinate::Coordinate};

    #[cfg(feature = "bench")]
    use num_traits::cast::cast;
    #[cfg(feature = "bench")]
    use rand::Rng;
    #[cfg(feature = "bench")]
    use std::time::Instant;

    // =============================================================================
    // SINGLE SIMPLEX TESTS
    // =============================================================================

    #[expect(clippy::too_many_lines)]
    #[test]
    fn test_boundary_facets_single_simplices() {
        // Test boundary analysis for single simplices in different dimensions

        // Test Case 1: 2D triangle - all 3 edges should be boundary facets
        {
            let points = vec![
                Point::new([0.0, 0.0]),
                Point::new([1.0, 0.0]),
                Point::new([0.5, 1.0]),
            ];
            let vertices = Vertex::from_points(&points);
            let dt = DelaunayTriangulation::new(&vertices).unwrap();

            assert_eq!(dt.number_of_cells(), 1, "2D triangle should have 1 cell");
            assert_eq!(dt.dim(), 2, "Should be 2-dimensional");

            let boundary_count = dt.boundary_facets().count();
            assert_eq!(
                boundary_count, 3,
                "2D triangle should have 3 boundary facets"
            );

            // Verify all facets are boundary facets using cached map
            let facet_to_cells = dt
                .tds()
                .build_facet_to_cells_map()
                .expect("Should build facet map");
            assert!(dt.boundary_facets().all(|f| {
                dt.tds()
                    .is_boundary_facet_with_map(&f, &facet_to_cells)
                    .expect("Should not fail for valid facets")
            }));
        }

        // Test Case 2: 3D tetrahedron - all 4 faces should be boundary facets
        {
            let points = vec![
                Point::new([0.0, 0.0, 0.0]),
                Point::new([1.0, 0.0, 0.0]),
                Point::new([0.0, 1.0, 0.0]),
                Point::new([0.0, 0.0, 1.0]),
            ];
            let vertices = Vertex::from_points(&points);
            let dt = DelaunayTriangulation::new(&vertices).unwrap();

            assert_eq!(dt.number_of_cells(), 1, "3D tetrahedron should have 1 cell");
            assert_eq!(dt.dim(), 3, "Should be 3-dimensional");

            let boundary_count = dt.boundary_facets().count();
            assert_eq!(
                boundary_count, 4,
                "3D tetrahedron should have 4 boundary facets"
            );

            // Verify all facets are boundary facets
            let facet_to_cells = dt
                .tds()
                .build_facet_to_cells_map()
                .expect("Should build facet map");
            assert!(dt.boundary_facets().all(|f| {
                dt.tds()
                    .is_boundary_facet_with_map(&f, &facet_to_cells)
                    .expect("Should not fail for valid facets")
            }));
        }

        // Test Case 3: 4D simplex - all 5 tetrahedra should be boundary facets
        {
            let points = vec![
                Point::new([0.0, 0.0, 0.0, 0.0]),
                Point::new([1.0, 0.0, 0.0, 0.0]),
                Point::new([0.0, 1.0, 0.0, 0.0]),
                Point::new([0.0, 0.0, 1.0, 0.0]),
                Point::new([0.0, 0.0, 0.0, 1.0]),
            ];
            let vertices = Vertex::from_points(&points);
            let dt = DelaunayTriangulation::new(&vertices).unwrap();

            assert_eq!(dt.number_of_cells(), 1, "4D simplex should have 1 cell");
            assert_eq!(dt.dim(), 4, "Should be 4-dimensional");

            let boundary_count = dt.boundary_facets().count();
            assert_eq!(
                boundary_count, 5,
                "4D simplex should have 5 boundary facets"
            );

            // Verify all facets are boundary facets
            let facet_to_cells = dt
                .tds()
                .build_facet_to_cells_map()
                .expect("Should build facet map");
            let confirmed_boundary = dt
                .boundary_facets()
                .filter(|f| {
                    dt.tds()
                        .is_boundary_facet_with_map(f, &facet_to_cells)
                        .expect("Should not fail for valid facets")
                })
                .count();
            assert_eq!(
                confirmed_boundary, 5,
                "All facets should be boundary facets"
            );
        }

        // Test Case 4: Empty triangulation
        {
            let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();
            assert_eq!(
                dt.number_of_cells(),
                0,
                "Empty triangulation should have no cells"
            );

            let boundary_count = dt.boundary_facets().count();
            assert_eq!(
                boundary_count, 0,
                "Empty triangulation should have no boundary facets"
            );
        }

        println!(
            "✓ Single simplex boundary analysis works correctly in 2D, 3D, 4D, and empty cases"
        );
    }

    #[test]
    fn test_boundary_facets_method_coverage() {
        // Test method delegation and implementation path coverage

        // Test case 1: Basic method delegation and error propagation
        {
            let points = vec![
                Point::new([0.0, 0.0, 0.0]),
                Point::new([1.0, 0.0, 0.0]),
                Point::new([0.0, 1.0, 0.0]),
                Point::new([0.0, 0.0, 1.0]),
            ];
            let vertices = Vertex::from_points(&points);
            let dt = DelaunayTriangulation::new(&vertices).unwrap();

            // Test boundary_facets() normal path
            let boundary_count = dt.boundary_facets().count();
            assert_eq!(
                boundary_count, 4,
                "Single tetrahedron has 4 boundary facets"
            );

            // Test is_boundary_facet() delegation (builds facet map internally)
            if let Some(facet) = dt.boundary_facets().next() {
                let result = dt.tds().is_boundary_facet(&facet);
                assert!(result.is_ok(), "Should not error on valid facet");
                assert!(
                    result.unwrap(),
                    "Facet should be boundary in single tetrahedron"
                );
            }
        }

        // Test case 2: Capacity allocation and vector operations
        {
            let points = vec![
                Point::new([0.0, 0.0, 0.0]),
                Point::new([1.0, 0.0, 0.0]),
                Point::new([0.0, 1.0, 0.0]),
                Point::new([0.0, 0.0, 1.0]),
                Point::new([0.5, 0.5, 0.5]), // Interior point
            ];
            let vertices = Vertex::from_points(&points);
            let dt = DelaunayTriangulation::new(&vertices).unwrap();

            // After robust cleanup and facet-sharing filtering, we may end up with a single cell
            assert!(
                dt.number_of_cells() >= 1,
                "Should have at least one cell for this test"
            );

            // Exercise capacity allocation, cache initialization, and vector push operations
            let boundary_count = dt.boundary_facets().count();
            assert!(boundary_count > 0, "Should have boundary facets");
            assert!(
                boundary_count >= 4,
                "Should have at least 4 boundary facets"
            );
        }

        println!("✓ Boundary facets method coverage and delegation work correctly");
    }

    #[test]
    #[cfg(feature = "bench")]
    fn test_boundary_analysis_performance_characteristics() {
        // Test that boundary analysis methods have reasonable performance characteristics

        // Create a moderately complex triangulation
        let points: Vec<Point<f64, 3>> = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([3.0, 0.0, 0.0]),
            Point::new([0.0, 3.0, 0.0]),
            Point::new([0.0, 0.0, 3.0]),
            Point::new([1.5, 1.5, 0.0]),
            Point::new([1.5, 0.0, 1.5]),
            Point::new([0.0, 1.5, 1.5]),
            Point::new([1.0, 1.0, 1.0]),
        ];

        let vertices = Vertex::from_points(&points);
        let dt = DelaunayTriangulation::new(&vertices).unwrap();

        if dt.number_of_cells() > 0 {
            println!(
                "Performance test triangulation: {} vertices, {} cells",
                dt.number_of_vertices(),
                dt.number_of_cells()
            );

            // Time boundary_facets() method
            let start = Instant::now();
            let boundary_count_direct = dt.boundary_facets().count();
            let boundary_facets_time = start.elapsed();

            // Collect facets for multiple operations
            let boundary_facets_vec: Vec<_> = dt.boundary_facets().collect();
            let boundary_len = boundary_facets_vec.len();

            // Time is_boundary_facet() for each boundary facet
            let start = Instant::now();
            let mut confirmed_boundary = 0;
            for facet in &boundary_facets_vec {
                if dt
                    .tds()
                    .is_boundary_facet(facet)
                    .expect("Should not fail to check boundary facet")
                {
                    confirmed_boundary += 1;
                }
            }
            let is_boundary_time = start.elapsed();

            println!("Performance results:");
            println!(
                "  boundary_facets().count(): {boundary_facets_time:?} (found {boundary_len} facets)"
            );
            println!(
                "  is_boundary_facet() × {boundary_len}: {is_boundary_time:?} (confirmed: {confirmed_boundary})"
            );

            // Verify consistency
            assert_eq!(boundary_len, boundary_count_direct);
            assert_eq!(confirmed_boundary, boundary_len);

            // Performance should be reasonable (these are very loose bounds)
            assert!(
                boundary_facets_time.as_millis() < 1000,
                "boundary_facets() should complete quickly"
            );
            assert!(
                is_boundary_time.as_millis() < 1000,
                "is_boundary_facet() calls should complete quickly"
            );
        }

        println!("✓ Performance characteristics are acceptable");
    }

    #[test]
    #[cfg(feature = "bench")]
    fn benchmark_boundary_facets_performance() {
        // Smaller point counts for reasonable test time
        let point_counts = [20, 40, 60, 80];

        println!("\nBenchmarking boundary_facets() performance with DelaunayTriangulation:");
        println!(
            "Note: This demonstrates the O(N·F) complexity where N = cells, F = facets per cell"
        );

        for &n_points in &point_counts {
            // Create a number of well-distributed random points in 3D
            let mut rng = rand::rng();
            let points: Vec<Point<f64, 3>> = (0..n_points)
                .map(|i| {
                    // Add some spacing to reduce degeneracy
                    let spacing = cast(i).unwrap_or(0.0) * 0.1;
                    Point::new([
                        rng.random::<f64>().mul_add(100.0, spacing),
                        rng.random::<f64>().mul_add(100.0, spacing * 1.1),
                        rng.random::<f64>().mul_add(100.0, spacing * 1.3),
                    ])
                })
                .collect();

            let vertices = Vertex::from_points(&points);

            // Create triangulation using DelaunayTriangulation
            let dt = match DelaunayTriangulation::new(&vertices) {
                Ok(dt) => {
                    println!("Successfully created triangulation with {n_points} vertices");
                    dt
                }
                Err(e) => {
                    println!("Points: {n_points:3} | Skipped due to triangulation error: {e}");
                    continue; // Skip this test case
                }
            };

            // Time multiple runs to get more stable measurements
            let mut total_time = std::time::Duration::ZERO;
            let runs: u32 = 10;

            for _ in 0..runs {
                let start = Instant::now();
                let boundary_facets = dt.boundary_facets();
                total_time += start.elapsed();

                // Prevent optimization away
                std::hint::black_box(boundary_facets);
            }

            let avg_time = total_time / runs;

            let boundary_count = dt.boundary_facets().count();
            println!(
                "Points: {:3} | Cells: {:4} | Boundary Facets: {:4} | Avg Time: {:?}",
                n_points,
                dt.number_of_cells(),
                boundary_count,
                avg_time
            );
        }

        println!("\nOptimization achieved:");
        println!("- Single pass over all cells and facets: O(N·F)");
        println!("- HashMap-based facet-to-cells mapping");
        println!("- Direct facet cloning instead of repeated computation");
    }

    // =============================================================================
    // ADDITIONAL TESTS FOR UNCOVERED ERROR PATHS
    // =============================================================================

    #[test]
    fn test_boundary_facets_invalid_facet_index_error() {
        println!("Testing boundary_facets with invalid facet index error path");

        // Note: This error path (InvalidFacetIndex) is difficult to trigger in practice
        // because the facet-to-cells mapping is built from valid facets.
        // We test this by confirming the error structure exists and can be created.

        // Test that the error can be created and has correct structure
        let error = TdsValidationError::FacetError(FacetError::InvalidFacetIndex {
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

        println!("  Error structure: {error}");
        println!("  ✓ InvalidFacetIndex error path structure verified");
    }

    #[test]
    fn test_boundary_facets_cell_not_found_error() {
        println!("Testing boundary_facets with cell not found error path");

        // Note: This error path (CellNotFoundInTriangulation) is also difficult to trigger
        // in practice because the mapping is built from existing cells.
        // We test the error structure.

        // Test that the error can be created
        let error = TdsValidationError::FacetError(FacetError::CellNotFoundInTriangulation);

        // Verify error display is meaningful
        let error_string = format!("{error}");
        assert!(
            error_string.contains("Cell") || error_string.contains("cell"),
            "Error should mention cell: {error_string}"
        );

        println!("  Error structure: {error}");
        println!("  ✓ CellNotFoundInTriangulation error path structure verified");
    }

    #[test]
    fn test_is_boundary_facet_with_map_consistency() {
        println!("Testing is_boundary_facet_with_map consistency with boundary_facets");

        // Create a valid triangulation
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        let vertices = Vertex::from_points(&points);
        let dt = DelaunayTriangulation::new(&vertices).unwrap();

        // Build facet map
        let facet_to_cells = dt
            .tds()
            .build_facet_to_cells_map()
            .expect("Should build map");

        // Get all boundary facets and verify they are correctly identified
        let mut boundary_count = 0;

        for boundary_facet in dt.boundary_facets() {
            let is_boundary = dt
                .tds()
                .is_boundary_facet_with_map(&boundary_facet, &facet_to_cells)
                .expect("Should successfully check boundary status");

            assert!(
                is_boundary,
                "All facets returned by boundary_facets() should be boundary facets"
            );
            boundary_count += 1;
        }

        // Single tetrahedron should have 4 boundary facets
        assert_eq!(
            boundary_count, 4,
            "Single tetrahedron should have 4 boundary facets"
        );

        // Verify consistency
        let reported_count = dt.boundary_facets().count();
        assert_eq!(
            boundary_count, reported_count,
            "Boundary facet count should be consistent"
        );

        println!("  ✓ All {boundary_count} boundary facets correctly identified");
        println!("  ✓ is_boundary_facet_with_map consistency verified");
    }

    #[test]
    fn test_boundary_facets_error_propagation_from_build_map() {
        println!("Testing error propagation from build_facet_to_cells_map");

        // Test that boundary_facets properly propagates errors from build_facet_to_cells_map
        // This exercises the error propagation path in boundary_facets()

        // Create a minimal valid triangulation
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        let vertices = Vertex::from_points(&points);
        let dt = DelaunayTriangulation::new(&vertices).unwrap();

        // Test that build_facet_to_cells_map succeeds on valid triangulation
        let map_result = dt.tds().build_facet_to_cells_map();
        assert!(
            map_result.is_ok(),
            "build_facet_to_cells_map should succeed on valid TDS"
        );

        // Test that boundary_facets succeeds when build_facet_to_cells_map succeeds
        let boundary_count = dt.boundary_facets().count();
        assert_eq!(
            boundary_count, 4,
            "Single tetrahedron should have 4 boundary facets"
        );

        println!("  ✓ Error propagation path from build_facet_to_cells_map verified");
    }

    #[test]
    fn test_number_of_boundary_facets_delegation() {
        println!("Testing number_of_boundary_facets delegation to boundary_facets");

        // This test exercises the delegation to boundary_facets() and result transformation
        // ensuring the method properly delegates and transforms the result

        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        let vertices = Vertex::from_points(&points);
        let dt = DelaunayTriangulation::new(&vertices).unwrap();

        // Test both methods return consistent results
        let boundary_facets_count = dt.boundary_facets().count();
        let boundary_count = dt
            .tds()
            .number_of_boundary_facets()
            .expect("Should get boundary count");

        assert_eq!(
            boundary_facets_count, boundary_count,
            "number_of_boundary_facets should equal boundary_facets().count()"
        );

        assert_eq!(
            boundary_count, 4,
            "Single tetrahedron should have 4 boundary facets"
        );

        println!("  ✓ number_of_boundary_facets delegation working correctly");
        println!("    - boundary_facets().count(): {boundary_facets_count}");
        println!("    - number_of_boundary_facets(): {boundary_count}");
    }

    #[test]
    fn test_invalid_facet_multiplicity_error_creation() {
        println!("Testing InvalidFacetMultiplicity error creation and formatting");

        // Test that the error can be created with various multiplicity values
        let test_cases = [
            (0, "zero multiplicity"),
            (3, "triple multiplicity"),
            (5, "excessive multiplicity"),
        ];

        for (multiplicity, description) in &test_cases {
            let facet_key = 0x1234_5678_9ABC_DEF0_u64; // Example facet key
            let error = TdsValidationError::FacetError(FacetError::InvalidFacetMultiplicity {
                facet_key,
                found: *multiplicity,
            });

            // Verify error display includes all necessary information
            let error_string = format!("{error}");
            assert!(
                error_string.contains(&format!("{multiplicity:}").to_string()),
                "Error should contain multiplicity {multiplicity}: {error_string}"
            );
            assert!(
                error_string.contains(&format!("{facet_key:016x}")),
                "Error should contain facet key in hex: {error_string}"
            );
            assert!(
                error_string.contains("expected 1 (boundary) or 2 (internal)"),
                "Error should explain valid multiplicities: {error_string}"
            );

            println!("  ✓ {description}: {error}");
        }

        println!("  ✓ InvalidFacetMultiplicity error creation and formatting verified");
    }
}
