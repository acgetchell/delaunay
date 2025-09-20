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

        // Collect all facets that belong to only one cell
        for (_facet_key, cells) in facet_to_cells {
            if let [(cell_id, facet_index)] = cells.as_slice() {
                // Bind dereferenced values once to avoid repetitive derefs
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
        Ok(self.is_boundary_facet_with_map(facet, &facet_to_cells))
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
    /// `true` if the facet is on the boundary (belongs to only one cell), `false` otherwise.
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
    ///             let is_boundary = tds.is_boundary_facet_with_map(facet, &facet_to_cells);
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
    ) -> bool {
        // Facet::vertices() should always return D vertices by construction
        let vertices = facet.vertices();
        debug_assert_eq!(
            vertices.len(),
            D,
            "Invalid facet: expected {} vertices, got {}",
            D,
            vertices.len()
        );

        match derive_facet_key_from_vertices(&vertices, self) {
            Ok(facet_key) => facet_to_cells
                .get(&facet_key)
                .is_some_and(|cells| cells.len() == 1),
            Err(e) => {
                debug_assert!(false, "derive_facet_key_from_vertices failed: {e:?}");
                // Option: flip to logging + false, or plumb a Result in the non-cached path.
                false
            }
        }
    }

    /// Returns the number of boundary facets in the triangulation.
    ///
    /// This delegates to `boundary_facets()` for consistent error handling.
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
        // Delegate to boundary_facets() for consistent error handling
        self.boundary_facets().map(|v| v.len())
    }
}

#[cfg(test)]
mod tests {
    use super::BoundaryAnalysis;
    use crate::core::collections::{FastHashMap, fast_hash_map_with_capacity};
    use crate::core::triangulation_data_structure::{Tds, TriangulationConstructionError};
    use crate::core::util::derive_facet_key_from_vertices;
    use crate::core::vertex::Vertex;
    use crate::geometry::{point::Point, traits::coordinate::Coordinate};

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
        let boundary_facets = tds.boundary_facets().expect("Should get boundary facets");
        assert_eq!(
            boundary_facets.len(),
            3,
            "2D triangle should have 3 boundary facets (edges)"
        );

        assert_eq!(
            tds.number_of_boundary_facets()
                .expect("Should get boundary facet count"),
            3,
            "Count should match vector length"
        );

        // All facets should be boundary facets
        let facet_to_cells = tds
            .build_facet_to_cells_map()
            .expect("Should build facet map in test");
        assert!(
            boundary_facets
                .iter()
                .all(|f| tds.is_boundary_facet_with_map(f, &facet_to_cells)),
            "All facets should be boundary facets in single triangle"
        );

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
        let boundary_facets = tds.boundary_facets().expect("Should get boundary facets");
        assert_eq!(
            boundary_facets.len(),
            4,
            "3D tetrahedron should have 4 boundary facets"
        );

        assert_eq!(
            tds.number_of_boundary_facets()
                .expect("Should get boundary facet count"),
            4,
            "Count should match vector length"
        );

        // All facets should be boundary facets
        let facet_to_cells = tds
            .build_facet_to_cells_map()
            .expect("Should build facet map in test");
        assert!(
            boundary_facets
                .iter()
                .all(|f| tds.is_boundary_facet_with_map(f, &facet_to_cells)),
            "All facets should be boundary facets in single tetrahedron"
        );

        println!("✓ 3D tetrahedron boundary analysis works correctly");
    }

    #[test]
    fn test_is_boundary_facet_with_map_cached_version() {
        // Test the cached version of is_boundary_facet for better performance in tight loops
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        let vertices = Vertex::from_points(points);
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        // Build the facet map once for efficiency
        let facet_to_cells = tds
            .build_facet_to_cells_map()
            .expect("Should build facet map in test");

        // Get all facets from the single tetrahedron
        let cell = tds.cells().values().next().expect("Should have one cell");
        let facets = cell.facets().expect("Should get facets from cell");

        // Test that both methods give the same results
        for facet in &facets {
            let is_boundary_regular = tds
                .is_boundary_facet(facet)
                .expect("Should not fail to check boundary facet");
            let is_boundary_cached = tds.is_boundary_facet_with_map(facet, &facet_to_cells);

            assert_eq!(
                is_boundary_regular, is_boundary_cached,
                "Regular and cached methods should give same result"
            );

            // In a single tetrahedron, all facets are boundary facets
            assert!(
                is_boundary_cached,
                "All facets in single tetrahedron should be boundary facets"
            );
        }

        println!("✓ Cached boundary facet method works correctly and matches regular method");
    }

    #[test]
    fn test_derive_facet_key_from_vertices_integration() {
        // Test derive_facet_key_from_vertices integration to ensure proper facet key computation
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        let vertices = Vertex::from_points(points);
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        // Get a facet from the single tetrahedron
        let cell = tds.cells().values().next().expect("Should have one cell");
        let facets = cell.facets().expect("Should get facets from cell");
        let test_facet = &facets[0];

        // Test that the utility function returns Ok(key) for valid facets
        let facet_vertices = test_facet.vertices();
        let facet_key_result = derive_facet_key_from_vertices(&facet_vertices, &tds);
        assert!(
            facet_key_result.is_ok(),
            "derive_facet_key_from_vertices should return Ok(key) for valid facet"
        );

        let facet_key = facet_key_result.unwrap();

        // Test that the same facet produces the same key (deterministic)
        let facet_vertices_2 = test_facet.vertices();
        let facet_key_2 = derive_facet_key_from_vertices(&facet_vertices_2, &tds).unwrap();
        assert_eq!(
            facet_key, facet_key_2,
            "Same facet should produce same key (deterministic)"
        );

        // Test consistency with build_facet_to_cells_map
        let boundary_facets_from_map = tds
            .build_facet_to_cells_map()
            .expect("Should build facet map in test");
        assert!(
            boundary_facets_from_map.contains_key(&facet_key),
            "Key from utility function should exist in facet_to_cells map"
        );

        // Verify the facet is correctly identified as boundary using the computed key
        let cells_for_facet = &boundary_facets_from_map[&facet_key];
        assert_eq!(
            cells_for_facet.len(),
            1,
            "Facet in single tetrahedron should belong to exactly 1 cell"
        );

        // Additional verification: ensure the (cell_id, facet_index) from the map
        // corresponds to test_facet by reconstructing and comparing keys
        if let Some((cell_id, facet_index)) = cells_for_facet.first().copied() {
            if let Some(mapped_cell) = tds.cells().get(cell_id) {
                let mapped_facets = mapped_cell
                    .facets()
                    .expect("Should get facets from mapped cell");
                if let Some(mapped_facet) = mapped_facets.get(usize::from(facet_index)) {
                    let mapped_facet_vertices = mapped_facet.vertices();
                    let mapped_facet_key =
                        derive_facet_key_from_vertices(&mapped_facet_vertices, &tds)
                            .expect("Should derive key from mapped facet");
                    assert_eq!(
                        facet_key, mapped_facet_key,
                        "Facet key from test_facet should match key from (cell_id, facet_index) mapping"
                    );

                    // Verify the facets represent the same geometric entity
                    assert_eq!(
                        test_facet.key(),
                        mapped_facet.key(),
                        "test_facet and mapped facet should have the same facet key"
                    );
                } else {
                    panic!("facet_index {facet_index} out of bounds for cell {cell_id:?}");
                }
            } else {
                panic!("cell_id {cell_id:?} not found in triangulation");
            }
        }

        println!("✓ derive_facet_key_from_vertices integration works correctly and consistently");
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
        let boundary_facets = tds.boundary_facets().expect("Should get boundary facets");
        assert_eq!(
            boundary_facets.len(),
            5,
            "4D simplex should have 5 boundary facets"
        );

        assert_eq!(
            tds.number_of_boundary_facets()
                .expect("Should count boundary facets"),
            5,
            "Count should match vector length"
        );

        // All facets should be boundary facets
        // Cache once
        let facet_to_cells = tds
            .build_facet_to_cells_map()
            .expect("Should build facet map in test");
        let mut confirmed_boundary = 0;
        for boundary_facet in &boundary_facets {
            if tds.is_boundary_facet_with_map(boundary_facet, &facet_to_cells) {
                confirmed_boundary += 1;
            }
        }
        assert_eq!(
            confirmed_boundary, 5,
            "All facets should be boundary facets in single 4D simplex"
        );

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
        let boundary_facets = tds.boundary_facets().expect("Should get boundary facets");
        assert_eq!(
            boundary_facets.len(),
            0,
            "Empty triangulation should have no boundary facets"
        );

        assert_eq!(
            tds.number_of_boundary_facets()
                .expect("Should count boundary facets"),
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
        let boundary_facets = tds.boundary_facets().expect("Should get boundary facets");
        assert_eq!(
            boundary_facets.len(),
            6,
            "Two adjacent tetrahedra should have 6 boundary facets"
        );

        // Test that all facets from boundary_facets() are indeed boundary facets
        for boundary_facet in &boundary_facets {
            assert!(
                tds.is_boundary_facet(boundary_facet)
                    .expect("Should not fail to check boundary facet"),
                "All facets from boundary_facets() should be boundary facets"
            );
        }

        // Test the count method
        assert_eq!(
            tds.number_of_boundary_facets()
                .expect("Should count boundary facets"),
            6,
            "Count should match vector length"
        );

        // Build a map of facet keys to the cells that contain them for detailed verification
        let dim_usize = usize::try_from(tds.dim().max(0)).unwrap_or(0);
        let mut facet_map: FastHashMap<u64, Vec<_>> =
            fast_hash_map_with_capacity(tds.number_of_cells() * (dim_usize + 1));
        for (cell_key, cell) in tds.cells() {
            for facet in cell.facets().expect("Should get cell facets") {
                let facet_vertices = facet.vertices();
                if let Ok(fk) = derive_facet_key_from_vertices(&facet_vertices, &tds) {
                    facet_map.entry(fk).or_default().push(cell_key);
                }
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
            for facet in cell.facets().expect("Should get cell facets") {
                if !tds
                    .is_boundary_facet(&facet)
                    .expect("Should not fail to check boundary facet")
                {
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
            let boundary_facets = tds.boundary_facets().expect("Should get boundary facets");
            let total_facets_count = tds
                .cells()
                .values()
                .map(|cell| cell.facets().expect("Should get cell facets").len())
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
                    tds.is_boundary_facet(boundary_facet)
                        .expect("Should not fail to check boundary facet"),
                    "Facet from boundary_facets() should be identified as boundary"
                );
            }

            // Check if we can find any internal (non-boundary) facets
            let mut found_internal_facet = false;
            for cell in tds.cells().values() {
                for facet in cell.facets().expect("Should get cell facets") {
                    if !tds
                        .is_boundary_facet(&facet)
                        .expect("Should not fail to check boundary facet")
                    {
                        found_internal_facet = true;
                        let facet_vertices = facet.vertices();
                        let facet_key =
                            derive_facet_key_from_vertices(&facet_vertices, &tds).unwrap_or(0); // Use 0 as fallback for debug output
                        println!("Found internal facet: key = {facet_key}");
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
            let boundary_facets = tds.boundary_facets().expect("Should get boundary facets");
            let boundary_count_from_vector = boundary_facets.len();

            // Get count using the efficient counting method
            let boundary_count_from_count = tds
                .number_of_boundary_facets()
                .expect("Should count boundary facets");

            // These should be equal
            assert_eq!(
                boundary_count_from_vector, boundary_count_from_count,
                "boundary_facets().len() should equal number_of_boundary_facets()"
            );

            // Each facet from boundary_facets() should be identified as boundary by is_boundary_facet()
            for facet in &boundary_facets {
                assert!(
                    tds.is_boundary_facet(facet)
                        .expect("Should not fail to check boundary facet"),
                    "Facet from boundary_facets() should be confirmed as boundary by is_boundary_facet()"
                );
            }

            // Count boundary facets by checking each facet individually
            let mut boundary_count_from_individual_checks = 0;
            for cell in tds.cells().values() {
                for facet in cell.facets().expect("Should get cell facets") {
                    if tds
                        .is_boundary_facet(&facet)
                        .expect("Should not fail to check boundary facet")
                    {
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
    #[cfg(feature = "bench")]
    fn test_boundary_facets_large_triangulation() {
        // Test with a larger triangulation to ensure scalability
        // This test is marked as benchmark-only due to its performance-sensitive nature
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
            let boundary_facets = tds.boundary_facets().expect("Should get boundary facets");
            let boundary_count = tds
                .number_of_boundary_facets()
                .expect("Should count boundary facets");

            // Consistency check
            assert_eq!(
                boundary_facets.len(),
                boundary_count,
                "boundary_facets().len() should equal number_of_boundary_facets()"
            );

            // Each boundary facet should be confirmed by is_boundary_facet()
            for facet in &boundary_facets {
                assert!(
                    tds.is_boundary_facet(facet)
                        .expect("Should not fail to check boundary facet"),
                    "Each facet from boundary_facets() should be confirmed as boundary"
                );
            }

            // Basic sanity checks
            assert!(
                boundary_count > 0,
                "Non-empty triangulation should have boundary facets"
            );

            // In a convex triangulation, we expect boundary facets to exist
            let total_facets: usize = tds
                .cells()
                .values()
                .map(|cell| cell.facets().expect("Should get cell facets").len())
                .sum();

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
            Err(TriangulationConstructionError::InsufficientVertices { .. })
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
            Err(TriangulationConstructionError::InsufficientVertices { .. })
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
        assert_eq!(
            tds_coplanar
                .boundary_facets()
                .expect("Should get boundary facets")
                .len(),
            4
        ); // Should have 4 boundary facets
        assert_eq!(
            tds_coplanar
                .number_of_boundary_facets()
                .expect("Should count boundary facets"),
            4
        );

        println!("✓ Edge cases handled correctly:");
        println!("  - Single point: correctly fails with InsufficientVertices");
        println!("  - Collinear points: correctly fails with InsufficientVertices");
        println!("  - Minimum vertices (4) for 3D: creates triangulation with boundary facets");
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
}
