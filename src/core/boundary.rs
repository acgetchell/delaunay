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
use std::collections::HashMap;
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
        let mut boundary_facets = Vec::new();

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
        // Use the precomputed facet-to-cells map to check if the facet belongs to only one cell
        // This avoids recomputing the facet's neighbors and is more efficient
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
        // Build a map from facet keys to count of cells that contain them
        let mut facet_counts: HashMap<u64, usize> = HashMap::new();

        for cell in self.cells().values() {
            for facet in cell.facets() {
                *facet_counts.entry(facet.key()).or_insert(0) += 1;
            }
        }

        // Count facets that appear in only one cell
        facet_counts.values().filter(|&&count| count == 1).count()
    }
}

#[cfg(test)]
mod tests {
    use super::BoundaryAnalysis;
    use crate::core::{triangulation_data_structure::Tds, vertex::Vertex};
    use crate::geometry::{point::Point, traits::coordinate::Coordinate};
    use crate::vertex;
    use std::collections::HashMap;
    use uuid::Uuid;

    #[test]
    fn test_boundary_facets_single_cell() {
        // Create a single tetrahedron - all its facets should be boundary facets
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        let vertices = Vertex::from_points(points);
        let tds: Tds<f64, usize, usize, 3> = Tds::new(&vertices).unwrap();

        assert_eq!(tds.number_of_cells(), 1, "Should contain one cell");

        // All 4 facets of the tetrahedron should be on the boundary
        let boundary_facets = tds.boundary_facets();
        assert_eq!(
            boundary_facets.len(),
            4,
            "A single tetrahedron should have 4 boundary facets"
        );

        // Also test the count method for efficiency
        assert_eq!(
            tds.number_of_boundary_facets(),
            4,
            "Count of boundary facets should be 4"
        );
    }

    #[test]
    fn test_is_boundary_facet() {
        // Create a triangulation with two adjacent tetrahedra sharing one facet
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

        println!("Created triangulation with {} cells", tds.number_of_cells());
        for (i, cell) in tds.cells().values().enumerate() {
            println!(
                "Cell {}: vertices = {:?}",
                i,
                cell.vertices()
                    .iter()
                    .map(|v| v.point().to_array())
                    .collect::<Vec<_>>()
            );
        }

        assert_eq!(tds.number_of_cells(), 2, "Should have exactly two cells");

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

        // Build a map of facet keys to the cells that contain them
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

        // Verify neighbors are correctly assigned
        let all_cells: Vec<_> = tds.cells().values().collect();
        let first_cell = all_cells[0];
        let second_cell = all_cells[1];

        // Each cell should have exactly one neighbor (the other cell)
        assert!(
            first_cell.neighbors.is_some(),
            "Cell 1 should have neighbors"
        );
        assert!(
            second_cell.neighbors.is_some(),
            "Cell 2 should have neighbors"
        );

        let neighbors1 = first_cell.neighbors.as_ref().unwrap();
        let neighbors2 = second_cell.neighbors.as_ref().unwrap();

        assert_eq!(neighbors1.len(), 1, "Cell 1 should have exactly 1 neighbor");
        assert_eq!(neighbors2.len(), 1, "Cell 2 should have exactly 1 neighbor");

        assert!(
            neighbors1.contains(&second_cell.uuid()),
            "Cell 1 should have Cell 2 as neighbor"
        );
        assert!(
            neighbors2.contains(&first_cell.uuid()),
            "Cell 2 should have Cell 1 as neighbor"
        );
    }

    #[test]
    fn test_find_bad_cells_and_boundary_facets() {
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        let vertices = Vertex::from_points(points);
        let tds: Tds<f64, usize, usize, 3> = Tds::new(&vertices).unwrap();
        // Triangulation is automatically done in Tds::new
        let mut result = tds;

        if result.number_of_cells() > 0 {
            // Create a test vertex that might be inside/outside existing cells
            let test_vertex = vertex!([0.25, 0.25, 0.25]);

            // Test the bad cells and boundary facets detection
            let bad_cells_result = result.find_bad_cells_and_boundary_facets(&test_vertex);
            assert!(bad_cells_result.is_ok());

            let (bad_cells, boundary_facets) = bad_cells_result.unwrap();
            println!(
                "Found {} bad cells and {} boundary facets",
                bad_cells.len(),
                boundary_facets.len()
            );
        }
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
