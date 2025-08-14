//! Boundary and convex hull analysis functions
//!
//! This module implements the `BoundaryAnalysis` trait for triangulation data structures,
//! providing methods to identify and analyze boundary facets in d-dimensional triangulations.

use super::{
    facet::Facet,
    traits::{boundary_analysis::BoundaryAnalysis, data_type::DataType},
    triangulation_data_structure::Tds,
};
use crate::geometry::traits::coordinate::CoordinateScalar;
use nalgebra::ComplexField;
use serde::{Serialize, de::DeserializeOwned};
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
    /// # Returns
    ///
    /// A `Vec<Facet<T, U, V, D>>` containing all boundary facets in the triangulation.
    /// The facets are returned in no particular order.
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
    /// let boundary_facets = tds.boundary_facets();
    /// assert_eq!(boundary_facets.len(), 4);
    /// ```
    fn boundary_facets(&self) -> Vec<Facet<T, U, V, D>> {
        // Build a map from facet keys to the cells that contain them
        let facet_to_cells = self.build_facet_to_cells_hashmap();
        // Upper bound on the number of boundary facets is the map size
        let mut boundary_facets = Vec::with_capacity(facet_to_cells.len());

        // Collect all facets that belong to only one cell
        for (_facet_key, cells) in facet_to_cells {
            if cells.len() == 1 {
                let cell_id = cells[0].0;
                let facet_index = cells[0].1;
                if let Some(cell) = self.cells().get(cell_id) {
                    boundary_facets.push(cell.facets()[facet_index].clone());
                }
            }
        }

        boundary_facets
    }

    /// Checks if a specific facet is a boundary facet.
    ///
    /// A boundary facet is a facet that belongs to only one cell in the triangulation.
    ///
    /// # Arguments
    ///
    /// * `facet` - The facet to check.
    ///
    /// # Returns
    ///
    /// `true` if the facet is on the boundary (belongs to only one cell), `false` otherwise.
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
    ///     let facets = cell.facets();
    ///     if let Some(facet) = facets.first() {
    ///         // In a single tetrahedron, all facets are boundary facets
    ///         assert!(tds.is_boundary_facet(facet));
    ///     }
    /// }
    /// ```
    fn is_boundary_facet(&self, facet: &Facet<T, U, V, D>) -> bool {
        // Build the facet-to-cells map to check if the facet belongs to only one cell
        // Note: This recomputes the map per call; for repeated queries, compute once and reuse.
        let facet_to_cells = self.build_facet_to_cells_hashmap();
        facet_to_cells
            .get(&facet.key())
            .is_some_and(|cells| cells.len() == 1)
    }

    /// Returns the number of boundary facets in the triangulation.
    ///
    /// This is a more efficient way to count boundary facets without creating
    /// the full vector of facets.
    ///
    /// # Returns
    ///
    /// The number of boundary facets in the triangulation.
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
    /// assert_eq!(tds.number_of_boundary_facets(), 4);
    /// ```
    fn number_of_boundary_facets(&self) -> usize {
        // Count facets that belong to exactly one cell
        self.build_facet_to_cells_hashmap()
            .values()
            .filter(|cells| cells.len() == 1)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::BoundaryAnalysis;
    use crate::core::triangulation_data_structure::{Tds, TriangulationValidationError};
    use crate::core::vertex::Vertex;
    use crate::geometry::{point::Point, traits::coordinate::Coordinate};
    use std::collections::HashMap;
    use uuid::Uuid;

    // =============================================================================
    // SINGLE SIMPLEX TESTS
    // =============================================================================

    #[test]
    fn test_boundary_facets_2d_triangle() {
        // Test 2D triangulation (triangle) - all 3 edges should be boundary facets
        let points = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.5, 1.0]),
        ];
        let vertices = Vertex::from_points(points);
        let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();

        assert_eq!(tds.number_of_cells(), 1, "2D triangle should have 1 cell");
        assert_eq!(tds.dim(), 2, "Should be 2-dimensional");

        // A 2D triangle has 3 boundary facets (edges)
        let boundary_facets = tds.boundary_facets();
        assert_eq!(
            boundary_facets.len(),
            3,
            "2D triangle should have 3 boundary facets (edges)"
        );

        assert_eq!(
            tds.number_of_boundary_facets(),
            3,
            "Count should match vector length"
        );

        // All facets should be boundary facets
        for boundary_facet in &boundary_facets {
            assert!(
                tds.is_boundary_facet(boundary_facet),
                "All facets should be boundary facets in single triangle"
            );
        }

        println!("✓ 2D triangle boundary analysis works correctly");
    }

    #[test]
    fn test_boundary_facets_3d_tetrahedron() {
        // Test 3D triangulation (single tetrahedron) - all 4 facets should be boundary facets
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

        // A 3D tetrahedron has 4 boundary facets (triangular faces)
        let boundary_facets = tds.boundary_facets();
        assert_eq!(
            boundary_facets.len(),
            4,
            "3D tetrahedron should have 4 boundary facets"
        );

        assert_eq!(
            tds.number_of_boundary_facets(),
            4,
            "Count should match vector length"
        );

        // All facets should be boundary facets
        for boundary_facet in &boundary_facets {
            assert!(
                tds.is_boundary_facet(boundary_facet),
                "All facets should be boundary facets in single tetrahedron"
            );
        }

        println!("✓ 3D tetrahedron boundary analysis works correctly");
    }

    #[test]
    fn test_boundary_facets_4d_simplex() {
        // Test 4D triangulation (4-simplex) - all 5 facets should be boundary facets
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

        // A 4D simplex has 5 boundary facets (3D tetrahedra)
        let boundary_facets = tds.boundary_facets();
        assert_eq!(
            boundary_facets.len(),
            5,
            "4D simplex should have 5 boundary facets"
        );

        assert_eq!(
            tds.number_of_boundary_facets(),
            5,
            "Count should match vector length"
        );

        // All facets should be boundary facets
        for boundary_facet in &boundary_facets {
            assert!(
                tds.is_boundary_facet(boundary_facet),
                "All facets should be boundary facets in single 4D simplex"
            );
        }

        println!("✓ 4D simplex boundary analysis works correctly");
    }

    #[test]
    fn test_boundary_facets_empty_triangulation() {
        // Test boundary analysis on empty triangulation
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&[]).unwrap();

        assert_eq!(
            tds.number_of_cells(),
            0,
            "Empty triangulation should have no cells"
        );

        // Empty triangulation should have no boundary facets
        let boundary_facets = tds.boundary_facets();
        assert_eq!(
            boundary_facets.len(),
            0,
            "Empty triangulation should have no boundary facets"
        );

        assert_eq!(
            tds.number_of_boundary_facets(),
            0,
            "Count should be 0 for empty triangulation"
        );

        println!("✓ Empty triangulation boundary analysis works correctly");
    }

    // =============================================================================
    // MULTI-CELL TESTS
    // =============================================================================

    #[test]
    fn test_boundary_facets_3d_two_tetrahedra() {
        // Test 3D triangulation with two adjacent tetrahedra sharing one facet
        // This should result in 6 boundary facets and 1 internal (shared) facet
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),  // A
            Point::new([1.0, 0.0, 0.0]),  // B
            Point::new([0.5, 1.0, 0.0]),  // C - forms base triangle ABC
            Point::new([0.5, 0.5, 1.0]),  // D - above base
            Point::new([0.5, 0.5, -1.0]), // E - below base
        ];
        let vertices = Vertex::from_points(points);
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        assert_eq!(
            tds.number_of_cells(),
            2,
            "Should have exactly two tetrahedra"
        );

        // Get all boundary facets
        let boundary_facets = tds.boundary_facets();
        assert_eq!(
            boundary_facets.len(),
            6,
            "Two adjacent tetrahedra should have 6 boundary facets"
        );

        // Test that all facets from boundary_facets() are indeed boundary facets
        for boundary_facet in &boundary_facets {
            assert!(
                tds.is_boundary_facet(boundary_facet),
                "All facets from boundary_facets() should be boundary facets"
            );
        }

        // Test the count method
        assert_eq!(
            tds.number_of_boundary_facets(),
            6,
            "Count should match the vector length"
        );

        // Build a map of facet keys to the cells that contain them for detailed verification
        let mut facet_map: HashMap<u64, Vec<Uuid>> = HashMap::new();
        for cell in tds.cells().values() {
            for facet in cell.facets() {
                facet_map.entry(facet.key()).or_default().push(cell.uuid());
            }
        }

        // Count boundary and shared facets
        let mut boundary_count = 0;
        let mut shared_count = 0;

        for (_, cells) in facet_map {
            if cells.len() == 1 {
                boundary_count += 1;
            } else if cells.len() == 2 {
                shared_count += 1;
            } else {
                panic!(
                    "Facet should be shared by at most 2 cells, found {}",
                    cells.len()
                );
            }
        }

        // Two tetrahedra should have 6 boundary facets and 1 shared facet
        assert_eq!(boundary_count, 6, "Should have 6 boundary facets");
        assert_eq!(shared_count, 1, "Should have 1 shared (internal) facet");

        // Verify that we can find the internal facet using is_boundary_facet
        let mut found_internal_facet = false;
        for cell in tds.cells().values() {
            for facet in cell.facets() {
                if !tds.is_boundary_facet(&facet) {
                    found_internal_facet = true;
                    break;
                }
            }
            if found_internal_facet {
                break;
            }
        }
        assert!(
            found_internal_facet,
            "Should find at least one internal facet"
        );

        println!("✓ 3D two-tetrahedra boundary analysis works correctly");
    }

    #[test]
    fn test_is_boundary_facet_mixed_cases() {
        // Test is_boundary_facet with a mix of boundary and internal facets
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),  // A
            Point::new([2.0, 0.0, 0.0]),  // B
            Point::new([1.0, 2.0, 0.0]),  // C - forms base triangle ABC
            Point::new([1.0, 1.0, 2.0]),  // D - above base
            Point::new([1.0, 1.0, -2.0]), // E - below base
        ];
        let vertices = Vertex::from_points(points);
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        if tds.number_of_cells() >= 2 {
            let boundary_facets = tds.boundary_facets();
            let total_facets_count = tds
                .cells()
                .values()
                .map(|cell| cell.facets().len())
                .sum::<usize>();

            println!("Total facets in all cells: {total_facets_count}");
            println!("Boundary facets: {}", boundary_facets.len());
            println!(
                "Expected internal (shared) facets: {}",
                (total_facets_count - boundary_facets.len()) / 2
            );

            // Test each boundary facet
            for boundary_facet in &boundary_facets {
                assert!(
                    tds.is_boundary_facet(boundary_facet),
                    "Facet from boundary_facets() should be identified as boundary"
                );
            }

            // Check if we can find any internal (non-boundary) facets
            let mut found_internal_facet = false;
            for cell in tds.cells().values() {
                for facet in cell.facets() {
                    if !tds.is_boundary_facet(&facet) {
                        found_internal_facet = true;
                        println!("Found internal facet: key = {}", facet.key());
                        break;
                    }
                }
                if found_internal_facet {
                    break;
                }
            }

            if tds.number_of_cells() > 1 && !found_internal_facet {
                println!(
                    "Warning: Expected to find internal facets with {} cells, but found none",
                    tds.number_of_cells()
                );
            }
        }

        println!("✓ Mixed boundary/internal facet identification works correctly");
    }

    // =============================================================================
    // CONSISTENCY AND VALIDATION TESTS
    // =============================================================================

    #[test]
    fn test_boundary_facets_consistency() {
        // Test that boundary_facets(), is_boundary_facet(), and number_of_boundary_facets() are consistent
        let test_cases = vec![
            // Single tetrahedron
            vec![
                Point::new([0.0, 0.0, 0.0]),
                Point::new([1.0, 0.0, 0.0]),
                Point::new([0.0, 1.0, 0.0]),
                Point::new([0.0, 0.0, 1.0]),
            ],
            // Well-separated points (avoids degenerate configurations)
            vec![
                Point::new([0.0, 0.0, 0.0]),
                Point::new([2.0, 0.0, 0.0]),
                Point::new([0.0, 2.0, 0.0]),
                Point::new([0.0, 0.0, 2.0]),
                Point::new([1.0, 1.0, 1.0]),
            ],
        ];

        for (i, points) in test_cases.into_iter().enumerate() {
            let vertices = Vertex::from_points(points);
            let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

            println!(
                "Test case {}: {} vertices, {} cells",
                i + 1,
                tds.number_of_vertices(),
                tds.number_of_cells()
            );

            // Get boundary facets using the vector method
            let boundary_facets = tds.boundary_facets();
            let boundary_count_from_vector = boundary_facets.len();

            // Get count using the efficient counting method
            let boundary_count_from_count = tds.number_of_boundary_facets();

            // These should be equal
            assert_eq!(
                boundary_count_from_vector, boundary_count_from_count,
                "boundary_facets().len() should equal number_of_boundary_facets()"
            );

            // Each facet from boundary_facets() should be identified as boundary by is_boundary_facet()
            let mut boundary_facets_confirmed = 0;
            for facet in &boundary_facets {
                assert!(
                    tds.is_boundary_facet(facet),
                    "Facet from boundary_facets() should be confirmed as boundary by is_boundary_facet()"
                );
                boundary_facets_confirmed += 1;
            }

            assert_eq!(
                boundary_facets_confirmed, boundary_count_from_vector,
                "All facets should be confirmed as boundary"
            );

            // Count boundary facets by checking each facet individually
            let mut boundary_count_from_individual_checks = 0;
            for cell in tds.cells().values() {
                for facet in cell.facets() {
                    if tds.is_boundary_facet(&facet) {
                        boundary_count_from_individual_checks += 1;
                    }
                }
            }

            assert_eq!(
                boundary_count_from_individual_checks, boundary_count_from_vector,
                "Individual facet checks should match boundary_facets() count"
            );

            println!(
                "  ✓ All {boundary_count_from_vector} boundary facets are consistent across all methods"
            );
        }

        println!("✓ All boundary analysis methods are consistent");
    }

    #[test]
    fn test_boundary_facets_large_triangulation() {
        // Test with a larger triangulation to ensure scalability
        use rand::Rng;

        let mut rng = rand::rng();
        let points: Vec<Point<f64, 3>> = (0..15)
            .map(|_| {
                Point::new([
                    rng.random::<f64>() * 10.0,
                    rng.random::<f64>() * 10.0,
                    rng.random::<f64>() * 10.0,
                ])
            })
            .collect();

        let vertices = Vertex::from_points(points);
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        println!(
            "Large triangulation: {} vertices, {} cells",
            tds.number_of_vertices(),
            tds.number_of_cells()
        );

        if tds.number_of_cells() > 0 {
            let boundary_facets = tds.boundary_facets();
            let boundary_count = tds.number_of_boundary_facets();

            // Consistency check
            assert_eq!(
                boundary_facets.len(),
                boundary_count,
                "boundary_facets().len() should equal number_of_boundary_facets()"
            );

            // Each boundary facet should be confirmed by is_boundary_facet()
            for facet in &boundary_facets {
                assert!(
                    tds.is_boundary_facet(facet),
                    "Each facet from boundary_facets() should be confirmed as boundary"
                );
            }

            println!("  ✓ {boundary_count} boundary facets identified and verified");

            // Basic sanity checks
            assert!(
                boundary_count > 0,
                "Non-empty triangulation should have boundary facets"
            );

            // In a convex triangulation, we expect boundary facets to exist
            let total_facets: usize = tds.cells().values().map(|cell| cell.facets().len()).sum();

            println!("  Total facet instances: {total_facets}, Boundary facets: {boundary_count}");

            // Internal facets are counted twice (once per adjacent cell), boundary facets once
            // So: total_facets = boundary_facets + 2 * internal_facets
            let internal_facets = (total_facets - boundary_count) / 2;
            println!("  Calculated internal facets: {internal_facets}");

            // Verify this makes sense
            assert_eq!(
                boundary_count + 2 * internal_facets,
                total_facets,
                "Facet accounting should be correct: boundary + 2*internal = total"
            );
        }

        println!("✓ Large triangulation boundary analysis completed successfully");
    }

    #[test]
    fn test_boundary_facets_edge_cases() {
        // Test various edge cases for boundary analysis

        // Test 1: Minimal triangulation (single point - should fail with InsufficientVertices)
        let single_point = vec![Point::new([0.0, 0.0, 0.0])];
        let vertices_single = Vertex::from_points(single_point);
        let result_single = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices_single);

        // Single vertex should fail with InsufficientVertices error since 1 < 4 (D+1 for 3D)
        assert!(matches!(
            result_single,
            Err(TriangulationValidationError::InsufficientVertices { .. })
        ));

        // Test 2: Collinear points (should fail with InsufficientVertices)
        let collinear_points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([2.0, 0.0, 0.0]),
        ];
        let vertices_collinear = Vertex::from_points(collinear_points);
        let result_collinear = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices_collinear);

        // Collinear points should fail with InsufficientVertices error since 3 < 4 (D+1 for 3D)
        assert!(matches!(
            result_collinear,
            Err(TriangulationValidationError::InsufficientVertices { .. })
        ));

        // Test 3: Coplanar points in 3D (should fail with InsufficientVertices)
        let coplanar_points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.5, 0.5, 0.0]),
        ];
        let vertices_coplanar = Vertex::from_points(coplanar_points);
        let tds_coplanar: Tds<f64, Option<()>, Option<()>, 3> =
            Tds::new(&vertices_coplanar).unwrap();

        // With exactly D+1=4 vertices, this should succeed even if coplanar
        assert_eq!(tds_coplanar.number_of_cells(), 1); // Should create 1 degenerate cell
        assert_eq!(tds_coplanar.boundary_facets().len(), 4); // Should have 4 boundary facets
        assert_eq!(tds_coplanar.number_of_boundary_facets(), 4);

        println!("✓ Edge cases handled correctly:");
        println!("  - Single point: correctly fails with InsufficientVertices");
        println!("  - Collinear points: correctly fails with InsufficientVertices");
        println!("  - Minimum vertices (4) for 3D: creates triangulation with boundary facets");
    }

    #[test]
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
            let boundary_facets = tds.boundary_facets();
            let boundary_facets_time = start.elapsed();

            // Time number_of_boundary_facets() method
            let start = Instant::now();
            let boundary_count = tds.number_of_boundary_facets();
            let boundary_count_time = start.elapsed();

            // Time is_boundary_facet() for each boundary facet
            let start = Instant::now();
            let mut confirmed_boundary = 0;
            for facet in &boundary_facets {
                if tds.is_boundary_facet(facet) {
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
    #[ignore = "Benchmark test is time-consuming and not suitable for regular test runs"]
    fn benchmark_boundary_facets_performance() {
        use rand::Rng;
        use std::time::Instant;

        // Smaller point counts for reasonable test time
        let point_counts = [20, 40, 60, 80];

        println!("\nBenchmarking boundary_facets() performance:");
        println!(
            "Note: This demonstrates the O(N·F) complexity where N = cells, F = facets per cell"
        );

        for &n_points in &point_counts {
            // Create a number of random points in 3D
            let mut rng = rand::rng();
            let points: Vec<Point<f64, 3>> = (0..n_points)
                .map(|_| {
                    Point::new([
                        rng.random::<f64>() * 100.0,
                        rng.random::<f64>() * 100.0,
                        rng.random::<f64>() * 100.0,
                    ])
                })
                .collect();

            let vertices = Vertex::from_points(points);
            let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

            // Time multiple runs to get more stable measurements
            let mut total_time = std::time::Duration::ZERO;
            let runs: u32 = 10;

            for _ in 0..runs {
                let start = Instant::now();
                let boundary_facets = tds.boundary_facets();
                total_time += start.elapsed();

                // Prevent optimization away
                std::hint::black_box(boundary_facets);
            }

            let avg_time = total_time / runs;

            println!(
                "Points: {:3} | Cells: {:4} | Boundary Facets: {:4} | Avg Time: {:?}",
                n_points,
                tds.number_of_cells(),
                tds.number_of_boundary_facets(),
                avg_time
            );
        }

        println!("\nOptimization achieved:");
        println!("- Single pass over all cells and facets: O(N·F)");
        println!("- HashMap-based facet-to-cells mapping");
        println!("- Direct facet cloning instead of repeated computation");
    }
}
