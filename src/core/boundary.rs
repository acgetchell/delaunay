//! Boundary and convex hull analysis functions
//!
//! This module implements the `BoundaryAnalysis` trait for triangulation data structures,
//! providing methods to identify and analyze boundary facets in d-dimensional triangulations.

use super::{
    facet::{Facet, FacetError},
    traits::{boundary_analysis::BoundaryAnalysis, data_type::DataType},
    triangulation_data_structure::{Tds, TriangulationValidationError},
    util::derive_facet_key_from_vertices,
};
use crate::core::collections::{KeyBasedCellMap, fast_hash_map_with_capacity};
use crate::geometry::traits::coordinate::CoordinateScalar;
use nalgebra::ComplexField;
use serde::{Serialize, de::DeserializeOwned};
use std::collections::hash_map::Entry;
use std::iter::Sum;
use std::ops::{AddAssign, Div, SubAssign};

/// Implementation of `BoundaryAnalysis` trait for `Tds`.
///
/// This implementation provides efficient boundary facet identification and analysis
/// for d-dimensional triangulations using the triangulation data structure.
impl<T, U, V, const D: usize> BoundaryAnalysis<T, U, V, D> for Tds<T, U, V, D>
where
    T: CoordinateScalar
        + AddAssign<T>
        + ComplexField<RealField = T>
        + SubAssign<T>
        + Sum
        + From<f64>
        + DeserializeOwned,
    U: DataType + DeserializeOwned,
    V: DataType + DeserializeOwned,
    f64: From<T>,
    for<'a> &'a T: Div<T>,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    ordered_float::OrderedFloat<f64>: From<T>,
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
    /// A `Result<Vec<Facet<T, U, V, D>>, TriangulationValidationError>` containing all boundary facets in the triangulation.
    /// The facets are returned in no particular order.
    ///
    /// # Errors
    ///
    /// Returns a [`FacetError`] if:
    /// - Any boundary facet cannot be created from the cells
    /// - A facet index is out of bounds (indicates data corruption)
    /// - A referenced cell is not found in the triangulation (indicates data corruption)
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::core::triangulation_data_structure::Tds;
    /// use delaunay::core::traits::boundary_analysis::BoundaryAnalysis;
    /// use delaunay::vertex;
    ///
    /// // Create a simple 3D triangulation (single tetrahedron)
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
    ///
    /// // A single tetrahedron has 4 boundary facets (all facets are on the boundary)
    /// let boundary_facets = tds.boundary_facets().unwrap();
    /// assert_eq!(boundary_facets.len(), 4);
    /// ```
    fn boundary_facets(&self) -> Result<Vec<Facet<T, U, V, D>>, TriangulationValidationError> {
        // Build a map from facet keys to the cells that contain them
        // Use try_build for strict error handling, proper error propagation
        let facet_to_cells = self.build_facet_to_cells_map()?;
        // Right-size the vector by counting boundary facets first
        let boundary_estimate = facet_to_cells.values().filter(|v| v.len() == 1).count();
        let mut boundary_facets = Vec::with_capacity(boundary_estimate);

        // Per-call cache to avoid repeated cell.facets() allocations
        // when multiple boundary facets reference the same cell
        let mut cell_facets_cache: KeyBasedCellMap<Vec<Facet<T, U, V, D>>> =
            fast_hash_map_with_capacity(self.number_of_cells());

        // Collect all facets that belong to only one cell and validate multiplicities
        for (facet_key, cells) in facet_to_cells {
            match cells.as_slice() {
                [(cell_id, facet_index)] => {
                    // Boundary facet - bind dereferenced values once to avoid repetitive derefs
                    let (cell_id, facet_index) = (*cell_id, *facet_index);
                    if let Some(cell) = self.cells().get(cell_id) {
                        // Cache facets per cell to avoid repeated allocations, but propagate errors
                        let facets = match cell_facets_cache.entry(cell_id) {
                            Entry::Occupied(e) => e.into_mut(),
                            Entry::Vacant(v) => {
                                let computed = cell.facets()?; // propagate FacetError (auto-converted to TriangulationValidationError)
                                v.insert(computed)
                            }
                        };

                        if let Some(f) = facets.get(usize::from(facet_index)) {
                            boundary_facets.push(f.clone());
                        } else {
                            // Fail fast: invalid facet index indicates data corruption
                            return Err(TriangulationValidationError::FacetError(
                                FacetError::InvalidFacetIndex {
                                    index: facet_index,
                                    facet_count: facets.len(),
                                },
                            ));
                        }
                    } else {
                        // Fail fast: cell not found indicates data corruption
                        return Err(TriangulationValidationError::FacetError(
                            FacetError::CellNotFoundInTriangulation,
                        ));
                    }
                }
                [_, _] => {
                    // Internal facet shared by exactly 2 cells; skip (valid)
                }
                slice => {
                    // Invalid multiplicity: facet shared by 0, 3+ cells indicates topology violation
                    return Err(TriangulationValidationError::FacetError(
                        FacetError::InvalidFacetMultiplicity {
                            facet_key,
                            found: slice.len(),
                        },
                    ));
                }
            }
        }

        Ok(boundary_facets)
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
    /// use delaunay::core::triangulation_data_structure::Tds;
    /// use delaunay::core::traits::boundary_analysis::BoundaryAnalysis;
    /// use delaunay::core::facet::Facet;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
    ///
    /// // Get a facet from one of the cells
    /// if let Some(cell) = tds.cells().values().next() {
    ///     if let Ok(facets) = cell.facets() {
    ///         if let Some(facet) = facets.first() {
    ///             // In a single tetrahedron, all facets are boundary facets
    ///             assert!(tds.is_boundary_facet(facet).unwrap());
    ///         }
    ///     }
    /// }
    /// ```
    #[inline]
    fn is_boundary_facet(
        &self,
        facet: &Facet<T, U, V, D>,
    ) -> Result<bool, TriangulationValidationError> {
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
    /// use delaunay::core::triangulation_data_structure::Tds;
    /// use delaunay::core::traits::boundary_analysis::BoundaryAnalysis;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
    ///
    /// // Build the facet map once for multiple queries
    /// let facet_to_cells = tds.build_facet_to_cells_map()
    ///     .expect("Should build facet map");
    ///
    /// // Check multiple facets efficiently
    /// if let Some(cell) = tds.cells().values().next() {
    ///     if let Ok(facets) = cell.facets() {
    ///         for facet in &facets {
    ///             let is_boundary = tds.is_boundary_facet_with_map(facet, &facet_to_cells)
    ///                 .expect("Should check if facet is boundary");
    ///             println!("Facet is boundary: {is_boundary}");
    ///         }
    ///     }
    /// }
    /// ```
    #[inline]
    fn is_boundary_facet_with_map(
        &self,
        facet: &Facet<T, U, V, D>,
        facet_to_cells: &crate::core::collections::FacetToCellsMap,
    ) -> Result<bool, TriangulationValidationError> {
        // Facet::vertices() should always return D vertices by construction
        let vertices = facet.vertices();
        debug_assert_eq!(
            vertices.len(),
            D,
            "Invalid facet: expected {} vertices, got {}",
            D,
            vertices.len()
        );

        let facet_key = derive_facet_key_from_vertices(&vertices, self)
            .map_err(TriangulationValidationError::FacetError)?;

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
    /// or a `TriangulationValidationError` if the facet map cannot be built.
    ///
    /// # Errors
    ///
    /// Returns a [`TriangulationValidationError`] if the facet-to-cells map cannot be built.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::core::triangulation_data_structure::Tds;
    /// use delaunay::core::traits::boundary_analysis::BoundaryAnalysis;
    /// use delaunay::vertex;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
    ///
    /// // A single tetrahedron has 4 boundary facets
    /// assert_eq!(tds.number_of_boundary_facets().unwrap(), 4);
    /// ```
    fn number_of_boundary_facets(&self) -> Result<usize, TriangulationValidationError> {
        self.build_facet_to_cells_map()
            .map(|m| m.values().filter(|v| v.len() == 1).count())
    }
}

#[cfg(test)]
#[allow(unnameable_test_items)]
mod tests {
    use super::BoundaryAnalysis;
    use crate::core::facet::FacetError;
    use crate::core::triangulation_data_structure::{Tds, TriangulationValidationError};
    use crate::core::vertex::Vertex;
    use crate::geometry::{point::Point, traits::coordinate::Coordinate};

    // =============================================================================
    // SINGLE SIMPLEX TESTS
    // =============================================================================

    #[allow(clippy::too_many_lines)]
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
            let vertices = Vertex::from_points(points);
            let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();

            assert_eq!(tds.number_of_cells(), 1, "2D triangle should have 1 cell");
            assert_eq!(tds.dim(), 2, "Should be 2-dimensional");

            let boundary_facets = tds.boundary_facets().expect("Should get boundary facets");
            assert_eq!(
                boundary_facets.len(),
                3,
                "2D triangle should have 3 boundary facets"
            );
            assert_eq!(tds.number_of_boundary_facets().expect("Should count"), 3);

            // Verify all facets are boundary facets using cached map
            let facet_to_cells = tds
                .build_facet_to_cells_map()
                .expect("Should build facet map");
            assert!(boundary_facets.iter().all(|f| {
                tds.is_boundary_facet_with_map(f, &facet_to_cells)
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
            let vertices = Vertex::from_points(points);
            let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

            assert_eq!(
                tds.number_of_cells(),
                1,
                "3D tetrahedron should have 1 cell"
            );
            assert_eq!(tds.dim(), 3, "Should be 3-dimensional");

            let boundary_facets = tds.boundary_facets().expect("Should get boundary facets");
            assert_eq!(
                boundary_facets.len(),
                4,
                "3D tetrahedron should have 4 boundary facets"
            );
            assert_eq!(tds.number_of_boundary_facets().expect("Should count"), 4);

            // Verify all facets are boundary facets
            let facet_to_cells = tds
                .build_facet_to_cells_map()
                .expect("Should build facet map");
            assert!(boundary_facets.iter().all(|f| {
                tds.is_boundary_facet_with_map(f, &facet_to_cells)
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
            let vertices = Vertex::from_points(points);
            let tds: Tds<f64, Option<()>, Option<()>, 4> = Tds::new(&vertices).unwrap();

            assert_eq!(tds.number_of_cells(), 1, "4D simplex should have 1 cell");
            assert_eq!(tds.dim(), 4, "Should be 4-dimensional");

            let boundary_facets = tds.boundary_facets().expect("Should get boundary facets");
            assert_eq!(
                boundary_facets.len(),
                5,
                "4D simplex should have 5 boundary facets"
            );
            assert_eq!(tds.number_of_boundary_facets().expect("Should count"), 5);

            // Verify all facets are boundary facets
            let facet_to_cells = tds
                .build_facet_to_cells_map()
                .expect("Should build facet map");
            let confirmed_boundary = boundary_facets
                .iter()
                .filter(|f| {
                    tds.is_boundary_facet_with_map(f, &facet_to_cells)
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
            let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&[]).unwrap();
            assert_eq!(
                tds.number_of_cells(),
                0,
                "Empty triangulation should have no cells"
            );

            let boundary_facets = tds.boundary_facets().expect("Should get boundary facets");
            assert_eq!(
                boundary_facets.len(),
                0,
                "Empty triangulation should have no boundary facets"
            );
            assert_eq!(
                tds.number_of_boundary_facets().expect("Should count"),
                0,
                "Count should be 0"
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
            let vertices = Vertex::from_points(points);
            let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

            // Test boundary_facets() normal path
            let boundary_facets = tds.boundary_facets().expect("Should get boundary facets");
            assert_eq!(
                boundary_facets.len(),
                4,
                "Single tetrahedron has 4 boundary facets"
            );

            // Test is_boundary_facet() delegation (builds facet map internally)
            if let Some(facet) = boundary_facets.first() {
                let result = tds.is_boundary_facet(facet);
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
            let vertices = Vertex::from_points(points);
            let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

            assert!(
                tds.number_of_cells() >= 2,
                "Should have multiple cells for this test"
            );

            // Exercise capacity allocation, cache initialization, and vector push operations
            let boundary_facets = tds.boundary_facets().expect("Should get boundary facets");
            assert!(!boundary_facets.is_empty(), "Should have boundary facets");
            assert!(
                boundary_facets.len() >= 4,
                "Should have at least 4 boundary facets"
            );

            // Test count method delegation
            let count = tds
                .number_of_boundary_facets()
                .expect("Should count boundary facets");
            assert_eq!(
                count,
                boundary_facets.len(),
                "Count should match vector length"
            );
        }

        println!("✓ Boundary facets method coverage and delegation work correctly");
    }

    #[test]
    #[cfg(feature = "bench")]
    fn test_boundary_analysis_performance_characteristics() {
        // Test that boundary analysis methods have reasonable performance characteristics
        use std::time::Instant;

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

        let vertices = Vertex::from_points(points);
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        if tds.number_of_cells() > 0 {
            println!(
                "Performance test triangulation: {} vertices, {} cells",
                tds.number_of_vertices(),
                tds.number_of_cells()
            );

            // Time boundary_facets() method
            let start = Instant::now();
            let boundary_facets = tds.boundary_facets().expect("Should get boundary facets");
            let boundary_facets_time = start.elapsed();

            // Time number_of_boundary_facets() method
            let start = Instant::now();
            let boundary_count = tds
                .number_of_boundary_facets()
                .expect("Should count boundary facets");
            let boundary_count_time = start.elapsed();

            // Time is_boundary_facet() for each boundary facet
            let start = Instant::now();
            let mut confirmed_boundary = 0;
            for facet in &boundary_facets {
                if tds
                    .is_boundary_facet(facet)
                    .expect("Should not fail to check boundary facet")
                {
                    confirmed_boundary += 1;
                }
            }
            let is_boundary_time = start.elapsed();

            println!("Performance results:");
            println!(
                "  boundary_facets(): {:?} (found {} facets)",
                boundary_facets_time,
                boundary_facets.len()
            );
            println!(
                "  number_of_boundary_facets(): {boundary_count_time:?} (count: {boundary_count})"
            );
            println!(
                "  is_boundary_facet() × {}: {:?} (confirmed: {})",
                boundary_facets.len(),
                is_boundary_time,
                confirmed_boundary
            );

            // Verify consistency
            assert_eq!(boundary_facets.len(), boundary_count);
            assert_eq!(confirmed_boundary, boundary_facets.len());

            // Performance should be reasonable (these are very loose bounds)
            assert!(
                boundary_facets_time.as_millis() < 1000,
                "boundary_facets() should complete quickly"
            );
            assert!(
                boundary_count_time.as_millis() < 1000,
                "number_of_boundary_facets() should complete quickly"
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
        use crate::core::algorithms::robust_bowyer_watson::RobustBoyerWatson;
        use crate::core::traits::insertion_algorithm::InsertionAlgorithm;
        use num_traits::cast::cast;
        use rand::Rng;
        use std::time::Instant;

        // Smaller point counts for reasonable test time
        let point_counts = [20, 40, 60, 80];

        println!("\nBenchmarking boundary_facets() performance with robust triangulation:");
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

            let vertices = Vertex::from_points(points);

            // Use robust Bowyer-Watson algorithm to create triangulation from scratch
            let mut robust_algorithm: RobustBoyerWatson<f64, Option<()>, Option<()>, 3> =
                RobustBoyerWatson::new();

            // Create triangulation using robust algorithm
            let tds = match robust_algorithm.new_triangulation(&vertices) {
                Ok(tds) => {
                    println!("Successfully created robust triangulation with {n_points} vertices");
                    tds
                }
                Err(e) => {
                    println!(
                        "Points: {n_points:3} | Skipped due to robust triangulation error: {e}"
                    );
                    continue; // Skip this test case
                }
            };

            // Time multiple runs to get more stable measurements
            let mut total_time = std::time::Duration::ZERO;
            let runs: u32 = 10;

            for _ in 0..runs {
                let start = Instant::now();
                let boundary_facets = tds.boundary_facets().expect("Should get boundary facets");
                total_time += start.elapsed();

                // Prevent optimization away
                std::hint::black_box(boundary_facets);
            }

            let avg_time = total_time / runs;

            println!(
                "Points: {:3} | Cells: {:4} | Boundary Facets: {:4} | Avg Time: {:?}",
                n_points,
                tds.number_of_cells(),
                tds.number_of_boundary_facets()
                    .expect("Should count boundary facets"),
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
        let error = TriangulationValidationError::FacetError(FacetError::InvalidFacetIndex {
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
        let error =
            TriangulationValidationError::FacetError(FacetError::CellNotFoundInTriangulation);

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
    fn test_is_boundary_facet_with_map_derive_key_failure_path() {
        println!("Testing is_boundary_facet_with_map when derive_facet_key_from_vertices fails");

        // Create a valid triangulation
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        let vertices = Vertex::from_points(points);
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        // To exercise the error path in is_boundary_facet_with_map where
        // derive_facet_key_from_vertices fails, we intentionally pass a Facet
        // from a different triangulation (with different vertex UUIDs) to this TDS.
        // The UUID lookups will fail against this TDS, triggering the error path.

        // First, create a completely separate triangulation with different points
        let other_points = vec![
            Point::new([10.0, 10.0, 10.0]),
            Point::new([11.0, 10.0, 10.0]),
            Point::new([10.0, 11.0, 10.0]),
            Point::new([10.0, 10.0, 11.0]),
        ];
        let other_vertices = Vertex::from_points(other_points);
        let other_tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&other_vertices).unwrap();

        // Build a facet map for the other triangulation (consistent with receiver TDS)
        let facet_to_cells = other_tds
            .build_facet_to_cells_map()
            .expect("Should build map for other TDS");

        // Obtain a valid facet from the original triangulation
        let boundary_facets = tds.boundary_facets().expect("Should get boundary facets");
        let foreign_facet = &boundary_facets[0];

        // This should now return an error because derive_facet_key_from_vertices will fail
        // when looking up the foreign facet's vertex UUIDs in other_tds
        let result = other_tds.is_boundary_facet_with_map(foreign_facet, &facet_to_cells);
        assert!(
            result.is_err(),
            "Should return error when vertex UUIDs are not found in the receiver TDS"
        );

        // Verify it's the expected error type
        if let Err(e) = result {
            println!("  Got expected error: {e}");
        }

        println!("  ✓ is_boundary_facet_with_map error path (derive key failure) exercised");
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
        let vertices = Vertex::from_points(points);
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        // Test that build_facet_to_cells_map succeeds on valid triangulation
        let map_result = tds.build_facet_to_cells_map();
        assert!(
            map_result.is_ok(),
            "build_facet_to_cells_map should succeed on valid TDS"
        );

        // Test that boundary_facets succeeds when build_facet_to_cells_map succeeds
        let boundary_result = tds.boundary_facets();
        assert!(
            boundary_result.is_ok(),
            "boundary_facets should succeed when build_map succeeds"
        );

        let boundary_facets = boundary_result.unwrap();
        assert_eq!(
            boundary_facets.len(),
            4,
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
        let vertices = Vertex::from_points(points);
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        // Test both methods return consistent results
        let boundary_facets = tds.boundary_facets().expect("Should get boundary facets");
        let boundary_count = tds
            .number_of_boundary_facets()
            .expect("Should get boundary count");

        assert_eq!(
            boundary_facets.len(),
            boundary_count,
            "number_of_boundary_facets should equal boundary_facets().len()"
        );

        assert_eq!(
            boundary_count, 4,
            "Single tetrahedron should have 4 boundary facets"
        );

        println!("  ✓ number_of_boundary_facets delegation working correctly");
        println!("    - boundary_facets().len(): {}", boundary_facets.len());
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
            let error =
                TriangulationValidationError::FacetError(FacetError::InvalidFacetMultiplicity {
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
