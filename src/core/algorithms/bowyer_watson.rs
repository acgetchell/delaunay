//! Incremental Bowyer-Watson algorithm for Delaunay triangulation.
//!
//! This module implements a robust, incremental approach to Delaunay triangulation
//! construction using the Bowyer-Watson algorithm. The implementation focuses on
//! clean separation of concerns and efficient vertex insertion without supercells.
//!
//! # Algorithm Overview
//!
//! The incremental Bowyer-Watson algorithm works as follows:
//!
//! 1. **Initialization**: Create initial simplex from first D+1 vertices
//! 2. **Incremental insertion**: For each subsequent vertex:
//!    - Locate the vertex relative to existing triangulation
//!    - Use cavity-based insertion for interior vertices
//!    - Use convex hull extension for exterior vertices
//!    - Maintain Delaunay property through circumsphere tests
//! 3. **Cleanup**: Remove degenerate cells and establish neighbor relationships
//!
//! # Triangulation Invariant Enforcement
//!
//! The Delaunay property and other triangulation invariants are enforced in specific locations:
//!
//! | Invariant Type | Enforcement Location | Method |
//! |---|---|---|
//! | **Delaunay Property** | `find_bad_cells()` | Empty circumsphere test using `insphere()` |
//! | **Facet Sharing** | `validate_facet_sharing()` | Each facet shared by ≤ 2 cells |
//! | **No Duplicate Cells** | `validate_no_duplicate_cells()` | No cells with identical vertex sets |
//! | **Neighbor Consistency** | `validate_neighbors()` | Mutual neighbor relationships |
//! | **Cell Validity** | `CellBuilder::validate()` (vertex count) + `cell.is_valid()` (comprehensive) | Construction + runtime validation |
//! | **Vertex Validity** | `Point::from()` (coordinates) + UUID auto-gen + `vertex.is_valid()` | Construction + runtime validation |
//!
//! The Delaunay property (empty circumsphere) is enforced **proactively** during construction in
//! `find_bad_cells()`, while structural invariants are enforced **reactively** through validation
//! in `is_valid()` and related methods.
//!
//! # Key Features
//!
//! - **Pure Incremental**: No supercells or batch processing
//! - **Robust**: Handles degenerate cases and numerical precision issues
//! - **Flexible**: Supports both interior and exterior vertex insertion
//! - **Efficient**: Expected O(1) amortized time per insertion
//! - **Maintainable**: Clean separation of algorithm logic and data structures
//!
//! # References
//!
//! The Bowyer-Watson algorithm was independently developed by Adrian Bowyer and David Watson
//! in 1981. This implementation draws from both foundational papers and modern computational
//! geometry literature:
//!
//! ## Original Papers
//!
//! - **Bowyer, A.** "Computing Dirichlet tessellations." *The Computer Journal* 24.2 (1981): 162-166.
//!   DOI: [10.1093/comjnl/24.2.162](https://doi.org/10.1093/comjnl/24.2.162)
//!
//! - **Watson, D.F.** "Computing the n-dimensional Delaunay tessellation with application to
//!   Voronoi polytopes." *The Computer Journal* 24.2 (1981): 167-172.
//!   DOI: [10.1093/comjnl/24.2.167](https://doi.org/10.1093/comjnl/24.2.167)
//!
//! ## Modern References
//!
//! - **de Berg, M., Cheong, O., van Kreveld, M., and Overmars, M.**
//!   *Computational Geometry: Algorithms and Applications.* 3rd ed. Springer-Verlag, 2008.
//!   Chapter 9: Delaunay Triangulations. ISBN: 978-3-540-77973-5
//!
//! - **Fortune, S.** "A sweepline algorithm for Voronoi diagrams."
//!   *Algorithmica* 2.1-4 (1987): 153-174. DOI: [10.1007/BF01840357](https://doi.org/10.1007/BF01840357)
//!
//! - **Guibas, L. and Stolfi, J.** "Primitives for the manipulation of general subdivisions
//!   and the computation of Voronoi diagrams." *ACM Transactions on Graphics* 4.2 (1985): 74-123.
//!   DOI: [10.1145/282918.282923](https://doi.org/10.1145/282918.282923)
//!
//! ## Implementation References
//!
//! - **CGAL Editorial Board.** *CGAL User and Reference Manual.* 5.6 edition, 2023.
//!   Available: [https://doc.cgal.org/latest/Triangulation_2/](https://doc.cgal.org/latest/Triangulation_2/)
//!
//! - **Shewchuk, J.R.** "Delaunay refinement algorithms for triangular mesh generation."
//!   *Computational Geometry* 22.1-3 (2002): 21-74.
//!   DOI: [10.1016/S0925-7721(01)00047-5](https://doi.org/10.1016/S0925-7721(01)00047-5)
//!
//! - **Edelsbrunner, H.** *Geometry and Topology for Mesh Generation.*
//!   Cambridge University Press, 2001. ISBN: 978-0-521-79309-4
//!
//! ## Numerical Stability References
//!
//! - **Shewchuk, J.R.** "Adaptive precision floating-point arithmetic and fast robust
//!   geometric predicates." *Discrete & Computational Geometry* 18.3 (1997): 305-363.
//!   DOI: [10.1007/PL00009321](https://doi.org/10.1007/PL00009321)
//!
//! - **Fortune, S. and Van Wyk, C.J.** "Efficient exact arithmetic for computational geometry."
//!   *Proceedings of the ninth annual symposium on Computational geometry* (1993): 163-172.
//!   DOI: [10.1145/160985.161140](https://doi.org/10.1145/160985.161140)

use crate::core::{
    collections::FacetToCellsMap,
    traits::{
        data_type::DataType,
        facet_cache::FacetCacheProvider,
        insertion_algorithm::{
            InsertionAlgorithm, InsertionBuffers, InsertionError, InsertionInfo,
            InsertionStatistics, InsertionStrategy,
        },
    },
    triangulation_data_structure::Tds,
    vertex::Vertex,
};
use crate::geometry::traits::coordinate::CoordinateScalar;
use arc_swap::ArcSwapOption;
use num_traits::NumCast;
use std::{
    iter::Sum,
    ops::{AddAssign, SubAssign},
    sync::{Arc, atomic::AtomicU64},
};

// InsertionStrategy and InsertionInfo are now imported from traits::insertion_algorithm

/// Incremental Bowyer-Watson algorithm implementation
///
/// This struct provides a clean interface for constructing Delaunay triangulations
/// using the incremental Bowyer-Watson algorithm. It operates on triangulation
/// data structures and maintains the Delaunay property throughout construction.
pub struct IncrementalBowyerWatson<T, U, V, const D: usize>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    /// Unified statistics tracking
    stats: InsertionStatistics,

    /// Reusable buffers for performance
    buffers: InsertionBuffers<T, U, V, D>,

    /// Cache for facet-to-cells mapping
    facet_to_cells_cache: ArcSwapOption<FacetToCellsMap>,

    /// Generation counter for cache invalidation
    cached_generation: Arc<AtomicU64>,
}

/// Deprecated alias for backward compatibility.
///
/// # Deprecated
/// Use [`IncrementalBowyerWatson`] instead. This alias will be removed in a future version.
#[deprecated(
    since = "0.5.1",
    note = "Use `IncrementalBowyerWatson` instead (correct spelling). This alias will be removed in v0.6.0."
)]
pub type IncrementalBoyerWatson<T, U, V, const D: usize> = IncrementalBowyerWatson<T, U, V, D>;

impl<T, U, V, const D: usize> IncrementalBowyerWatson<T, U, V, D>
where
    T: CoordinateScalar + AddAssign<T> + SubAssign<T> + Sum,
    U: DataType,
    V: DataType,
{
    /// Creates a new incremental Bowyer-Watson algorithm instance
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::core::algorithms::bowyer_watson::IncrementalBowyerWatson;
    /// use delaunay::core::traits::insertion_algorithm::InsertionAlgorithm;
    ///
    /// // Create a new 3D Bowyer-Watson algorithm instance
    /// let algorithm: IncrementalBowyerWatson<f64, (), (), 3> =
    ///     IncrementalBowyerWatson::new();
    ///
    /// // Verify initial statistics using the trait method
    /// let (insertions, created, removed) = algorithm.get_statistics();
    /// assert_eq!(insertions, 0);
    /// assert_eq!(created, 0);
    /// assert_eq!(removed, 0);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            stats: InsertionStatistics::new(),
            // Scale buffer capacity with dimension for better performance
            buffers: InsertionBuffers::with_capacity(D * 10),
            facet_to_cells_cache: ArcSwapOption::empty(),
            cached_generation: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Returns a snapshot of the internal insertion statistics.
    ///
    /// This method is public to support advanced diagnostics and benchmarking.
    #[must_use]
    pub const fn statistics(&self) -> &InsertionStatistics {
        &self.stats
    }

    /// Determines the appropriate insertion strategy for a vertex
    ///
    /// # Arguments
    ///
    /// * `tds` - Reference to the triangulation data structure
    /// * `vertex` - The vertex to be inserted
    ///
    /// # Returns
    ///
    /// The recommended insertion strategy.
    ///
    /// # Implementation Note
    ///
    /// If interior testing fails (geometry error), falls back to hull extension strategy
    /// rather than masking the error. This ensures that geometric failures are visible
    /// through the insertion attempt rather than being silently ignored.
    fn determine_insertion_strategy(
        &self,
        tds: &Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> InsertionStrategy {
        // Check if vertex is inside any existing cell's circumsphere
        match <Self as InsertionAlgorithm<T, U, V, D>>::is_vertex_interior(self, tds, vertex) {
            Ok(true) => InsertionStrategy::CavityBased,
            Ok(false) => InsertionStrategy::HullExtension,
            Err(_) => {
                // On geometry error, use Fallback strategy which tries multiple approaches
                // This prevents silently treating errors as "exterior" and gives fallback
                // mechanisms a chance to handle degenerate cases
                InsertionStrategy::Fallback
            }
        }
    }
}

impl<T, U, V, const D: usize> Default for IncrementalBowyerWatson<T, U, V, D>
where
    T: CoordinateScalar + AddAssign<T> + SubAssign<T> + Sum,
    U: DataType,
    V: DataType,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, U, V, const D: usize> FacetCacheProvider<T, U, V, D> for IncrementalBowyerWatson<T, U, V, D>
where
    T: CoordinateScalar + AddAssign<T> + SubAssign<T> + Sum + NumCast,
    U: DataType,
    V: DataType,
{
    fn facet_cache(&self) -> &ArcSwapOption<FacetToCellsMap> {
        &self.facet_to_cells_cache
    }

    fn cached_generation(&self) -> &AtomicU64 {
        // Return inner &AtomicU64 from Arc explicitly
        self.cached_generation.as_ref()
    }
}

// Implementation of the InsertionAlgorithm trait
impl<T, U, V, const D: usize> InsertionAlgorithm<T, U, V, D> for IncrementalBowyerWatson<T, U, V, D>
where
    T: CoordinateScalar + AddAssign<T> + SubAssign<T> + Sum + NumCast,
    U: DataType,
    V: DataType,
{
    /// Insert a single vertex into the triangulation (implementation)
    fn insert_vertex_impl(
        &mut self,
        tds: &mut Tds<T, U, V, D>,
        vertex: Vertex<T, U, D>,
    ) -> Result<InsertionInfo, InsertionError> {
        // Duplicate detection already handled by default insert_vertex
        // Determine insertion strategy
        let strategy = self.determine_insertion_strategy(tds, &vertex);

        // Try the primary strategy first using trait methods
        let mut result = match strategy {
            InsertionStrategy::Standard => {
                // For Standard strategy, use CavityBased as default
                <Self as InsertionAlgorithm<T, U, V, D>>::insert_vertex_cavity_based(
                    self, tds, &vertex,
                )
            }
            InsertionStrategy::CavityBased => {
                <Self as InsertionAlgorithm<T, U, V, D>>::insert_vertex_cavity_based(
                    self, tds, &vertex,
                )
            }
            InsertionStrategy::HullExtension => {
                <Self as InsertionAlgorithm<T, U, V, D>>::insert_vertex_hull_extension(
                    self, tds, &vertex,
                )
            }
            InsertionStrategy::Fallback => {
                <Self as InsertionAlgorithm<T, U, V, D>>::insert_vertex_fallback(self, tds, &vertex)
            }
            InsertionStrategy::Perturbation | InsertionStrategy::Skip => {
                // These strategies are for robust algorithms - fallback to basic methods
                <Self as InsertionAlgorithm<T, U, V, D>>::insert_vertex_fallback(self, tds, &vertex)
            }
        };

        // If the primary strategy failed, try fallback strategies using trait methods
        if result.is_err() {
            match strategy {
                InsertionStrategy::CavityBased => {
                    // If cavity-based failed, try hull extension
                    result = <Self as InsertionAlgorithm<T, U, V, D>>::insert_vertex_hull_extension(
                        self, tds, &vertex,
                    );
                    if result.is_err() {
                        // If hull extension also failed, try fallback
                        result = <Self as InsertionAlgorithm<T, U, V, D>>::insert_vertex_fallback(
                            self, tds, &vertex,
                        );
                    }
                }
                InsertionStrategy::HullExtension => {
                    // If hull extension failed, try fallback
                    result = <Self as InsertionAlgorithm<T, U, V, D>>::insert_vertex_fallback(
                        self, tds, &vertex,
                    );
                }
                _ => {
                    // For other strategies that already failed, no additional fallback
                }
            }
        }

        // Update statistics on successful insertion
        if let Ok(ref info) = result {
            self.stats.vertices_processed += 1;
            self.stats.total_cells_created += info.cells_created;
            self.stats.total_cells_removed += info.cells_removed;
        }

        result
    }

    /// Get statistics about the insertion algorithm's performance
    fn get_statistics(&self) -> (usize, usize, usize) {
        // Use the standardized statistics method
        self.stats.as_basic_tuple()
    }

    /// Reset the algorithm state for reuse
    fn reset(&mut self) {
        // Reset statistics and clear buffers
        self.stats.reset();
        self.buffers.clear_all();
        // Clear facet cache to prevent serving stale mappings across runs
        self.invalidate_facet_cache();
    }

    /// Update the cell creation counter
    fn increment_cells_created(&mut self, count: usize) {
        self.stats.total_cells_created += count;
    }

    /// Update the cell removal counter
    fn increment_cells_removed(&mut self, count: usize) {
        self.stats.total_cells_removed += count;
    }

    /// Determine the appropriate insertion strategy for a given vertex
    fn determine_strategy(
        &self,
        tds: &Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> InsertionStrategy {
        // Use the existing determine_insertion_strategy implementation
        self.determine_insertion_strategy(tds, vertex)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::boundary_analysis::BoundaryAnalysis;
    use crate::core::traits::insertion_algorithm::InsertionAlgorithm;
    use crate::core::vertex::Vertex;
    use crate::geometry::point::Point;
    use crate::geometry::traits::coordinate::Coordinate;
    use crate::vertex;

    /// Macro to generate dimension-specific Bowyer-Watson algorithm tests for dimensions 2D-5D.
    ///
    /// This macro reduces test duplication by generating consistent tests across
    /// multiple dimensions for the `IncrementalBowyerWatson` algorithm. It creates tests for:
    /// - Algorithm construction and initialization
    /// - Vertex insertion with cavity-based approach
    /// - Statistics tracking across insertions
    /// - TDS validation after operations
    ///
    /// # Usage
    ///
    /// ```ignore
    /// test_bowyer_watson_dimensions! {
    ///     bw_2d => 2 => "triangle" => vec![vertex!([0.0, 0.0]), ...], vertex!([1.0, 0.5])
    /// }
    /// ```
    macro_rules! test_bowyer_watson_dimensions {
        ($(
            $test_name:ident => $dim:expr => $desc:expr => $initial_vertices:expr, $test_vertex:expr
        ),+ $(,)?) => {
            $(
                #[test]
                fn $test_name() {
                    // Test basic algorithm functionality in this dimension
                    let mut algorithm = IncrementalBowyerWatson::<f64, (), (), $dim>::new();
                    let initial_vertices = $initial_vertices;
                    let mut tds: Tds<f64, (), (), $dim> = Tds::new(&initial_vertices).unwrap();

                    assert!(tds.is_valid().is_ok(), "{}D initial TDS should be valid", $dim);
                    assert_eq!(tds.dim(), $dim as i32, "{}D TDS should have dimension {}", $dim, $dim);

                    // Verify initial statistics
                    let (initial_processed, initial_created, initial_removed) = algorithm.get_statistics();
                    assert_eq!(initial_processed, 0, "{}D: Should have 0 vertices processed initially", $dim);
                    assert_eq!(initial_created, 0, "{}D: Should have 0 cells created initially", $dim);
                    assert_eq!(initial_removed, 0, "{}D: Should have 0 cells removed initially", $dim);

                    // Test vertex insertion
                    let test_vertex = $test_vertex;
                    let result = algorithm.insert_vertex(&mut tds, test_vertex);
                    assert!(result.is_ok(), "{}D: {} insertion should succeed", $dim, $desc);

                    let info = result.unwrap();
                    assert!(info.success, "{}D: Insertion should be successful", $dim);
                    assert!(info.cells_created > 0, "{}D: Should create at least one cell", $dim);

                    // Verify statistics were updated
                    let (processed, created, removed) = algorithm.get_statistics();
                    assert_eq!(processed, 1, "{}D: Should have processed 1 vertex", $dim);
                    assert_eq!(created, info.cells_created, "{}D: Created cells should match", $dim);
                    assert_eq!(removed, info.cells_removed, "{}D: Removed cells should match", $dim);

                    // Verify TDS remains valid after insertion
                    assert!(tds.is_valid().is_ok(), "{}D: TDS should remain valid after insertion", $dim);
                }

                pastey::paste! {
                    #[test]
                    fn [<$test_name _reset>]() {
                        // Test algorithm reset functionality
                        let mut algorithm = IncrementalBowyerWatson::<f64, (), (), $dim>::new();
                        let initial_vertices = $initial_vertices;
                        let mut tds: Tds<f64, (), (), $dim> = Tds::new(&initial_vertices).unwrap();

                        // Insert a vertex
                        let test_vertex = $test_vertex;
                        let _ = algorithm.insert_vertex(&mut tds, test_vertex);

                        // Verify statistics are non-zero
                        let (processed, _, _) = algorithm.get_statistics();
                        assert!(processed > 0, "{}D: Should have processed vertices before reset", $dim);

                        // Reset the algorithm
                        algorithm.reset();

                        // Verify statistics are reset
                        let (processed, created, removed) = algorithm.get_statistics();
                        assert_eq!(processed, 0, "{}D: Processed should be 0 after reset", $dim);
                        assert_eq!(created, 0, "{}D: Created should be 0 after reset", $dim);
                        assert_eq!(removed, 0, "{}D: Removed should be 0 after reset", $dim);
                    }

                    #[test]
                    fn [<$test_name _multiple_insertions>]() {
                        // Test multiple vertex insertions
                        let mut algorithm = IncrementalBowyerWatson::<f64, (), (), $dim>::new();
                        let initial_vertices = $initial_vertices;
                        let mut tds: Tds<f64, (), (), $dim> = Tds::new(&initial_vertices).unwrap();

                        let initial_vertex_count = tds.number_of_vertices();

                        // Insert test vertex
                        let test_vertex = $test_vertex;
                        let result1 = algorithm.insert_vertex(&mut tds, test_vertex);
                        assert!(result1.is_ok(), "{}D: First insertion should succeed", $dim);

                        // Verify vertex count increased
                        assert_eq!(tds.number_of_vertices(), initial_vertex_count + 1,
                            "{}D: Vertex count should increase after insertion", $dim);

                        // Verify TDS is still valid
                        assert!(tds.is_valid().is_ok(),
                            "{}D: TDS should be valid after insertion", $dim);

                        // Verify statistics tracking
                        let (processed, _, _) = algorithm.get_statistics();
                        assert_eq!(processed, 1, "{}D: Should have processed 1 vertex", $dim);
                    }
                }
            )+
        };
    }

    // Generate tests for dimensions 2D through 5D using the macro
    test_bowyer_watson_dimensions! {
        bw_2d_insertion => 2 => "interior point" =>
            vec![
                vertex!([0.0, 0.0]),
                vertex!([2.0, 0.0]),
                vertex!([1.0, 2.0]),
            ],
            vertex!([1.0, 0.5]),

        bw_3d_insertion => 3 => "interior point" =>
            vec![
                vertex!([0.0, 0.0, 0.0]),
                vertex!([2.0, 0.0, 0.0]),
                vertex!([0.0, 2.0, 0.0]),
                vertex!([0.0, 0.0, 2.0]),
            ],
            vertex!([0.5, 0.5, 0.5]),

        bw_4d_insertion => 4 => "interior point" =>
            vec![
                vertex!([0.0, 0.0, 0.0, 0.0]),
                vertex!([2.0, 0.0, 0.0, 0.0]),
                vertex!([0.0, 2.0, 0.0, 0.0]),
                vertex!([0.0, 0.0, 2.0, 0.0]),
                vertex!([0.0, 0.0, 0.0, 2.0]),
            ],
            vertex!([0.5, 0.5, 0.5, 0.5]),

        bw_5d_insertion => 5 => "interior point" =>
            vec![
                vertex!([0.0, 0.0, 0.0, 0.0, 0.0]),
                vertex!([2.0, 0.0, 0.0, 0.0, 0.0]),
                vertex!([0.0, 2.0, 0.0, 0.0, 0.0]),
                vertex!([0.0, 0.0, 2.0, 0.0, 0.0]),
                vertex!([0.0, 0.0, 0.0, 2.0, 0.0]),
                vertex!([0.0, 0.0, 0.0, 0.0, 2.0]),
            ],
            vertex!([0.5, 0.5, 0.5, 0.5, 0.5]),
    }

    // ============================================================================
    // TEST HELPERS
    // ============================================================================

    /// Count boundary facets (shared by 1 cell)
    fn count_boundary_facets(tds: &Tds<f64, (), (), 3>) -> usize {
        tds.build_facet_to_cells_map().ok().map_or(0, |map| {
            map.values().filter(|cells| cells.len() == 1).count()
        })
    }

    /// Count internal facets (shared by 2 cells)
    fn count_internal_facets(tds: &Tds<f64, (), (), 3>) -> usize {
        tds.build_facet_to_cells_map().ok().map_or(0, |map| {
            map.values().filter(|cells| cells.len() == 2).count()
        })
    }

    /// Count invalid facets (shared by 3+ cells)
    fn count_invalid_facets(tds: &Tds<f64, (), (), 3>) -> usize {
        tds.build_facet_to_cells_map().ok().map_or(0, |map| {
            map.values().filter(|cells| cells.len() > 2).count()
        })
    }

    // ============================================================================
    // DIAGNOSTIC AND INVARIANT TESTS
    // ============================================================================

    #[test]
    #[expect(clippy::too_many_lines)]
    fn diagnose_triangulation_invariant_violations() {
        #[cfg(debug_assertions)]
        eprintln!("=== DELAUNAY TRIANGULATION DIAGNOSTIC ===\n");

        // The problematic point configuration from failing tests
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),  // A - origin
            Point::new([1.0, 0.0, 0.0]),  // B - on x-axis
            Point::new([0.5, 1.0, 0.0]),  // C - forms triangle ABC in xy-plane
            Point::new([0.5, 0.5, 1.0]),  // D - above the triangle
            Point::new([0.5, 0.5, -1.0]), // E - below the triangle
        ];

        #[cfg(debug_assertions)]
        {
            eprintln!("Input Points:");
            for (i, point) in points.iter().enumerate() {
                eprintln!("  {}: {:?}", i, point.coords());
            }
        }

        let vertices = Vertex::from_points(&points);
        let tds: Tds<f64, (), (), 3> = Tds::new(&vertices).unwrap();

        #[cfg(debug_assertions)]
        {
            eprintln!("\n=== BASIC STATISTICS ===");
            eprintln!("Vertices: {}", tds.number_of_vertices());
            eprintln!("Cells: {}", tds.number_of_cells());
            eprintln!("Dimension: {}", tds.dim());
        }

        // Analyze each cell
        #[cfg(debug_assertions)]
        {
            eprintln!("\n=== CELL ANALYSIS ===");
            for (i, cell) in tds.cells().map(|(_, cell)| cell).enumerate() {
                let vertex_coords: Vec<&[f64; 3]> = cell
                    .vertices()
                    .iter()
                    .map(|vk| {
                        let v = &tds
                            .get_vertex_by_key(*vk)
                            .expect("Cell should reference valid vertex key");
                        v.point().coords()
                    })
                    .collect();
                eprintln!("Cell {i}: {vertex_coords:?}");

                // Check if cell has neighbors
                match &cell.neighbors {
                    Some(neighbors) => {
                        eprintln!("  Neighbors: {} cells", neighbors.len());
                    }
                    None => {
                        eprintln!("  Neighbors: None");
                    }
                }
            }

            // Detailed facet sharing analysis
            eprintln!("\n=== FACET SHARING ANALYSIS ===");
            let facet_to_cells = tds
                .build_facet_to_cells_map()
                .expect("facet map should build");

            let mut invalid_sharing = 0;
            let mut boundary_facets = 0;
            let mut internal_facets = 0;

            for (facet_key, cells) in &facet_to_cells {
                let cell_count = cells.len();
                match cell_count {
                    1 => boundary_facets += 1,
                    2 => internal_facets += 1,
                    _ => {
                        invalid_sharing += 1;
                        eprintln!("❌ INVALID: Facet {facet_key} shared by {cell_count} cells");
                        for facet_handle in cells {
                            if let Some(cell) = tds.get_cell(facet_handle.cell_key()) {
                                eprintln!(
                                    "   Cell {:?} at facet index {}",
                                    cell.uuid(),
                                    facet_handle.facet_index()
                                );
                            }
                        }
                    }
                }
            }

            eprintln!("Boundary facets (1 cell): {boundary_facets}");
            eprintln!("Internal facets (2 cells): {internal_facets}");
            eprintln!("Invalid facets (3+ cells): {invalid_sharing}");

            // Boundary analysis
            eprintln!("\n=== BOUNDARY ANALYSIS ===");
            match tds.boundary_facets() {
                Ok(bf) => {
                    let bf_count = bf.count();
                    eprintln!("Boundary facets found: {bf_count}");
                    if bf_count != boundary_facets {
                        eprintln!(
                            "❌ MISMATCH: Direct count ({boundary_facets}) vs boundary_facets() ({bf_count})"
                        );
                    }
                }
                Err(e) => {
                    eprintln!("❌ ERROR: Failed to get boundary facets: {e:?}");
                }
            }

            // Validation check
            eprintln!("\n=== VALIDATION CHECK ===");
            match tds.is_valid() {
                Ok(()) => eprintln!("✅ Triangulation reports as valid"),
                Err(e) => eprintln!("❌ Triangulation is invalid: {e:?}"),
            }

            // Expected vs Actual analysis
            eprintln!("\n=== GEOMETRIC ANALYSIS ===");
            eprintln!("Expected for 5 points in general position in 3D:");
            eprintln!("  - Convex hull should have 6-8 triangular faces");
            eprintln!("  - Should create 2-3 tetrahedra for optimal triangulation");
            eprintln!("  - Each internal facet shared by exactly 2 cells");
            eprintln!("  - Boundary facets shared by exactly 1 cell");

            eprintln!("\nActual results:");
            eprintln!("  - {} tetrahedra created", tds.number_of_cells());
            eprintln!("  - {invalid_sharing} invalid facet sharings (3+ cells)");
            eprintln!("  - {boundary_facets} boundary facets");
            eprintln!("  - {internal_facets} internal facets");
        }

        // Critical issue detection
        let facet_to_cells = tds
            .build_facet_to_cells_map()
            .expect("facet map should build");
        let mut invalid_sharing = 0;
        let mut boundary_facets = 0;

        for cells in facet_to_cells.values() {
            let cell_count = cells.len();
            match cell_count {
                1 => boundary_facets += 1,
                2 => {} // internal facets
                _ => invalid_sharing += 1,
            }
        }

        let mut issues = Vec::new();

        if invalid_sharing > 0 {
            issues.push("Facet sharing invariant violated");
        }

        if boundary_facets == 0 && tds.number_of_cells() > 0 {
            issues.push("No boundary facets (impossible for finite convex hull)");
        }

        if tds.number_of_cells() > 6 {
            issues.push("Too many cells created (over-triangulation)");
        }

        #[cfg(debug_assertions)]
        {
            if issues.is_empty() {
                eprintln!("\n✅ All triangulation invariants appear to be satisfied");
            } else {
                eprintln!("\n🚨 CRITICAL ISSUES DETECTED:");
                for issue in &issues {
                    eprintln!("  - {issue}");
                }
                eprintln!("  - Bowyer-Watson algorithm implementation has serious bugs");
            }
        }

        assert!(
            issues.is_empty(),
            "Triangulation violates Delaunay invariants: {} issues found: {issues:?}",
            issues.len()
        );
    }

    /// Test basic triangulation invariants with a simple tetrahedron
    #[test]
    fn test_simple_tetrahedron_invariants() {
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        let vertices = Vertex::from_points(&points);
        let tds: Tds<f64, (), (), 3> = Tds::new(&vertices).unwrap();

        // For a single tetrahedron, we expect:
        // - 1 cell, 4 boundary facets, 0 internal facets
        assert_eq!(tds.number_of_cells(), 1, "Should have 1 cell");
        assert_eq!(
            count_boundary_facets(&tds),
            4,
            "Should have 4 boundary facets"
        );
        assert_eq!(
            count_internal_facets(&tds),
            0,
            "Should have 0 internal facets"
        );
        assert_eq!(
            count_invalid_facets(&tds),
            0,
            "Should have 0 invalid facets"
        );

        // Boundary analysis API should work
        let boundary_count = tds
            .boundary_facets()
            .expect("Should get boundary facets")
            .count();
        assert_eq!(
            boundary_count, 4,
            "boundary_facets() should return 4 facets"
        );
    }

    /// Test incremental insertion maintains triangulation invariants
    #[test]
    fn test_incremental_insertion_diagnostic() {
        // Start with simple tetrahedron
        let mut points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        // Add point outside to trigger Bowyer-Watson
        points.push(Point::new([0.5, 0.5, 2.0]));
        let all_vertices = Vertex::from_points(&points);
        let tds: Tds<f64, (), (), 3> = Tds::new(&all_vertices).unwrap();

        // Verify triangulation invariants are maintained
        let boundary_count = count_boundary_facets(&tds);
        let invalid_count = count_invalid_facets(&tds);

        assert!(
            boundary_count > 0,
            "Should have boundary facets after insertion"
        );
        assert_eq!(invalid_count, 0, "Should have no invalid facet sharing");
        assert!(
            tds.number_of_cells() <= 3,
            "Should not over-triangulate (expected 2-3 cells)"
        );
    }

    /// Test algorithm internals step by step for debugging
    #[test]
    fn test_bowyer_watson_step_by_step() {
        let initial_points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        let vertices = Vertex::from_points(&initial_points);
        let mut tds: Tds<f64, (), (), 3> = Tds::new(&vertices).unwrap();
        let mut algorithm = IncrementalBowyerWatson::new();
        let new_vertex = vertex!([0.5, 0.5, 2.0]);

        // Test strategy determination
        let strategy = algorithm.determine_insertion_strategy(&tds, &new_vertex);
        assert_eq!(
            strategy,
            InsertionStrategy::HullExtension,
            "Exterior point should use hull extension"
        );

        // Test bad cell detection (may be empty for exterior points with hull extension)
        let _bad_cells = <IncrementalBowyerWatson<f64, (), (), 3> as InsertionAlgorithm<
            f64,
            (),
            (),
            3,
        >>::find_bad_cells(&mut algorithm, &tds, &new_vertex)
        .expect("Bad cell detection should succeed");

        // Test full insertion
        let result = algorithm.insert_vertex(&mut tds, new_vertex);
        assert!(result.is_ok(), "Insertion should succeed");
        assert!(
            tds.is_valid().is_ok(),
            "TDS should remain valid after insertion"
        );
    }

    // ============================================================================
    // TRAIT IMPLEMENTATION TESTS
    // ============================================================================

    #[test]
    fn test_finalize_triangulation_success() {
        println!("Testing finalize_triangulation with valid triangulation");

        // Create a triangulation that needs finalization
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, (), (), 3> = Tds::new(&initial_vertices).unwrap();
        let _algorithm = IncrementalBowyerWatson::<f64, (), (), 3>::new();

        // Add some additional complexity to the triangulation
        // (In a real scenario, this would be done through the Bowyer-Watson process)

        // Test finalization using trait method
        let result = <IncrementalBowyerWatson<f64, (), (), 3> as InsertionAlgorithm<
            f64,
            (),
            (),
            3,
        >>::finalize_triangulation(&mut tds);

        assert!(
            result.is_ok(),
            "Finalization should succeed for valid triangulation"
        );

        // Verify the triangulation is valid after finalization
        assert!(
            tds.is_valid().is_ok(),
            "Triangulation should be valid after finalization"
        );

        println!("✓ Triangulation finalization works correctly");
    }

    #[test]
    fn test_default_implementation() {
        println!("Testing Default trait implementation");

        // Test that Default::default() creates the same instance as new()
        let algorithm_new = IncrementalBowyerWatson::<f64, (), (), 3>::new();
        let algorithm_default = IncrementalBowyerWatson::<f64, (), (), 3>::default();

        // Both should have the same initial statistics
        let (insertions_new, created_new, removed_new) = algorithm_new.get_statistics();
        let (insertions_default, created_default, removed_default) =
            algorithm_default.get_statistics();

        assert_eq!(
            insertions_new, insertions_default,
            "Default should match new() - insertions"
        );
        assert_eq!(
            created_new, created_default,
            "Default should match new() - created"
        );
        assert_eq!(
            removed_new, removed_default,
            "Default should match new() - removed"
        );

        // All should be zero initially
        assert_eq!(insertions_new, 0, "Initial insertions should be 0");
        assert_eq!(created_new, 0, "Initial created should be 0");
        assert_eq!(removed_new, 0, "Initial removed should be 0");

        println!("✓ Default trait implementation works correctly");
    }

    // ============================================================================
    // REGRESSION TESTS
    // ============================================================================

    /// Regression test for cell counting bug (double counting fix)
    #[test]
    fn test_simple_double_counting_fix() {
        let mut algorithm = IncrementalBowyerWatson::<f64, (), (), 3>::new();
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, (), (), 3> = Tds::empty();

        algorithm.triangulate(&mut tds, &vertices).unwrap();
        let (insertions, created, removed) = algorithm.get_statistics();

        // For a minimal tetrahedron with 4 vertices:
        // - No incremental insertions (all 4 used for initial simplex)
        // - 1 cell created, 0 cells removed
        assert_eq!(
            insertions, 0,
            "No incremental insertions for minimal tetrahedron"
        );
        assert_eq!(created, 1, "Exactly one cell should be created");
        assert_eq!(removed, 0, "No cells should be removed");
        assert_eq!(tds.number_of_vertices(), 4, "Should contain all 4 vertices");
        assert_eq!(
            tds.number_of_cells(),
            1,
            "Should contain exactly 1 tetrahedron"
        );
    }

    /// Test `FacetCacheProvider` trait implementation
    #[test]
    fn test_facet_cache_provider_methods() {
        let algorithm = IncrementalBowyerWatson::<f64, (), (), 3>::new();

        let cache_ref = algorithm.facet_cache();
        assert!(cache_ref.load().is_none(), "Initial cache should be empty");

        let generation_ref = algorithm.cached_generation();
        let initial_generation = generation_ref.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(initial_generation, 0, "Initial generation should be 0");
    }

    /// Test `InsertionAlgorithm` trait methods (statistics, strategy, reset)
    #[test]
    fn test_insertion_algorithm_trait_methods() {
        let mut algorithm = IncrementalBowyerWatson::<f64, (), (), 3>::new();

        // Test increment methods
        let initial_stats = algorithm.get_statistics();
        algorithm.increment_cells_created(5);
        algorithm.increment_cells_removed(3);
        let after_increment = algorithm.get_statistics();
        assert_eq!(
            after_increment.1,
            initial_stats.1 + 5,
            "increment_cells_created should work"
        );
        assert_eq!(
            after_increment.2,
            initial_stats.2 + 3,
            "increment_cells_removed should work"
        );

        // Test strategy determination
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, (), (), 3> = Tds::new(&vertices).unwrap();

        let interior_vertex = vertex!([0.25, 0.25, 0.25]);
        let strategy = algorithm.determine_strategy(&tds, &interior_vertex);
        assert_eq!(
            strategy,
            InsertionStrategy::CavityBased,
            "Interior vertex should use cavity-based"
        );

        let exterior_vertex = vertex!([5.0, 5.0, 5.0]);
        let strategy = algorithm.determine_strategy(&tds, &exterior_vertex);
        assert_eq!(
            strategy,
            InsertionStrategy::HullExtension,
            "Exterior vertex should use hull extension"
        );

        // Test reset
        algorithm.reset();
        assert_eq!(
            algorithm.get_statistics(),
            (0, 0, 0),
            "Reset should clear all statistics"
        );
    }

    /// Test error handling paths in insertion algorithm
    #[test]
    fn test_error_handling_in_insertion() {
        let mut algorithm = IncrementalBowyerWatson::<f64, (), (), 3>::new();
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, (), (), 3> = Tds::new(&vertices).unwrap();

        // Test multiple insertions (interior and exterior points)
        let test_vertices = vec![
            vertex!([0.1, 0.1, 0.1]), // Interior
            vertex!([2.0, 0.0, 0.0]), // Exterior
            vertex!([0.0, 2.0, 0.0]), // Exterior different direction
            vertex!([0.0, 0.0, 2.0]), // Exterior another direction
        ];

        for test_vertex in test_vertices {
            // Either succeeds or fails gracefully - both are acceptable
            let _ = algorithm.insert_vertex(&mut tds, test_vertex);
        }
    }
}
