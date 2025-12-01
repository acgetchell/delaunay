//! # delaunay
//!
//! This is a library for computing the Delaunay triangulation of a set of n-dimensional points
//! in a [simplicial complex](https://en.wikipedia.org/wiki/Simplicial_complex)
//! inspired by [CGAL](https://www.cgal.org).
//!
//! # Features
//!
//! - d-dimensional Delaunay triangulations
//! - d-dimensional convex hulls
//! - Generic floating-point coordinate types (supports `f32`, `f64`, and other types implementing `CoordinateScalar`)
//! - Copy-able data types associated with vertices and cells (see [`DataType`](core::traits::DataType) for constraints)
//! - Serialization/Deserialization with [serde](https://serde.rs)
//!
//! # Basic Usage
//!
//! This library handles **arbitrary dimensions** (subject to numerical issues). Here's a 4D triangulation example:
//!
//! ```rust
//! use delaunay::prelude::*;
//!
//! // Create a 4D Delaunay triangulation (4-dimensional space!)
//! let vertices = vec![
//!     vertex!([0.0, 0.0, 0.0, 0.0]),
//!     vertex!([1.0, 0.0, 0.0, 0.0]),
//!     vertex!([0.0, 1.0, 0.0, 0.0]),
//!     vertex!([0.0, 0.0, 1.0, 0.0]),
//!     vertex!([0.0, 0.0, 0.0, 1.0]),  // 5 vertices (D+1) creates first 4-simplex
//!     vertex!([0.2, 0.2, 0.2, 0.2]),  // Additional vertex uses incremental insertion
//! ];
//!
//! let dt: DelaunayTriangulation<_, (), (), 4> =
//!     DelaunayTriangulation::new(&vertices).unwrap();
//!
//! assert_eq!(dt.number_of_vertices(), 6);
//! assert_eq!(dt.dim(), 4);                    // Full 4D triangulation
//! assert!(dt.number_of_cells() > 1);          // Cavity-based insertion creates additional 4-simplices
//! ```
//!
//! **Key insight**: The triangulation uses efficient incremental cavity-based insertion after
//! building an initial simplex from the first D+1 vertices (5 vertices for 4D).
//!
//! # Convex Hull Extraction
//!
//! Extract d-dimensional convex hulls from Delaunay triangulations:
//!
//! ```rust
//! use delaunay::prelude::*;
//!
//! // Create two tetrahedrons sharing a triangular facet (double tetrahedron)
//! let vertices: Vec<_> = vec![
//!     // Shared triangular facet vertices (forms base of both tetrahedrons)
//!     vertex!([0.0, 0.0, 0.0]),    // Shared vertex A
//!     vertex!([2.0, 0.0, 0.0]),    // Shared vertex B
//!     vertex!([1.0, 2.0, 0.0]),    // Shared vertex C
//!     // Apex of first tetrahedron (above the shared facet)
//!     vertex!([1.0, 0.7, 1.5]),    // First tet apex
//!     // Apex of second tetrahedron (below the shared facet)
//!     vertex!([1.0, 0.7, -1.5]),   // Second tet apex
//! ];
//!
//! let dt: DelaunayTriangulation<_, (), (), 3> =
//!     DelaunayTriangulation::new(&vertices).unwrap();
//!
//! // Extract the convex hull (boundary facets of the triangulation)
//! let hull = ConvexHull::from_triangulation(dt.triangulation()).unwrap();
//!
//! println!("Convex hull has {} facets in {}D", hull.facet_count(), hull.dimension());
//!
//! // Test point containment
//! let inside_point = Point::new([1.0, 0.5, 0.5]);
//! let outside_point = Point::new([3.0, 3.0, 3.0]);
//!
//! assert!(!hull.is_point_outside(&inside_point, dt.triangulation()).unwrap());  // Inside the hull
//! assert!(hull.is_point_outside(&outside_point, dt.triangulation()).unwrap());   // Outside the hull
//!
//! // Find visible facets from an external point (useful for incremental construction)
//! let visible_facets = hull.find_visible_facets(&outside_point, dt.triangulation()).unwrap();
//! println!("Point sees {} out of {} facets", visible_facets.len(), hull.facet_count());
//!
//! // Works in any dimension!
//! let vertices_4d: Vec<_> = vec![
//!     vertex!([0.0, 0.0, 0.0, 0.0]),
//!     vertex!([1.0, 0.0, 0.0, 0.0]),
//!     vertex!([0.0, 1.0, 0.0, 0.0]),
//!     vertex!([0.0, 0.0, 1.0, 0.0]),
//!     vertex!([0.0, 0.0, 0.0, 1.0]),
//! ];
//! let dt_4d: DelaunayTriangulation<_, (), (), 4> =
//!     DelaunayTriangulation::new(&vertices_4d).unwrap();
//! let hull_4d = ConvexHull::from_triangulation(dt_4d.triangulation()).unwrap();
//!
//! assert_eq!(hull_4d.facet_count(), 5);  // 4-simplex has 5 boundary facets
//! assert_eq!(hull_4d.dimension(), 4);     // 4D convex hull
//! ```
//!
//! # Triangulation Invariants
//!
//! The triangulation data structure maintains a set of **structural** and **geometric** invariants
//! that are checked by [`DelaunayTriangulation::is_valid`](core::delaunay_triangulation::DelaunayTriangulation::is_valid) and
//! [`DelaunayTriangulation::validation_report`](core::delaunay_triangulation::DelaunayTriangulation::validation_report):
//!
//! - **Vertex mappings** – every vertex UUID has a corresponding key and vice versa.
//! - **Cell mappings** – every cell UUID has a corresponding key and vice versa.
//! - **No duplicate cells** – no two maximal cells share the same vertex set.
//! - **Cell validity** – each cell has the correct number of vertices and passes internal
//!   consistency checks.
//! - **Facet sharing** – each facet is shared by at most 2 cells (1 on the boundary, 2 in the interior).
//! - **Neighbor consistency** – neighbor relationships are mutual and reference a shared facet.
//! - **Delaunay property** – triangulations constructed via incremental insertion
//!   maintain the empty circumsphere invariant; no vertex lies strictly
//!   inside the circumsphere of any maximal cell.
//!
//! ## Validation helpers
//!
//! These invariants are exposed through focused validation helpers on
//! [`core::delaunay_triangulation::DelaunayTriangulation`]:
//!
//! | Invariant | Helper method | Notes |
//! |---|---|---|
//! | Vertex mappings | [`DelaunayTriangulation::validate_vertex_mappings`](core::delaunay_triangulation::DelaunayTriangulation::validate_vertex_mappings) | Ensures UUID↔key consistency for all vertices. |
//! | Cell mappings | [`DelaunayTriangulation::validate_cell_mappings`](core::delaunay_triangulation::DelaunayTriangulation::validate_cell_mappings) | Ensures UUID↔key consistency for all cells. |
//! | Duplicate cells | [`DelaunayTriangulation::validate_no_duplicate_cells`](core::delaunay_triangulation::DelaunayTriangulation::validate_no_duplicate_cells) | Detects maximal cells with identical vertex sets. |
//! | Cell validity | [`Cell::is_valid`](core::cell::Cell::is_valid) (aggregated via [`DelaunayTriangulation::validation_report`](core::delaunay_triangulation::DelaunayTriangulation::validation_report)) | Per-cell structural checks. |
//! | Facet sharing | [`DelaunayTriangulation::validate_facet_sharing`](core::delaunay_triangulation::DelaunayTriangulation::validate_facet_sharing) | Verifies that each facet is shared by ≤ 2 cells. |
//! | Neighbor consistency | [`DelaunayTriangulation::validate_neighbors`](core::delaunay_triangulation::DelaunayTriangulation::validate_neighbors) | Verifies neighbor topology and mutual relationships. |
//! | Delaunay property | [`DelaunayTriangulation::validate_delaunay`](core::delaunay_triangulation::DelaunayTriangulation::validate_delaunay) | Expensive global empty-circumsphere check (optional). |
//!
//! [`DelaunayTriangulation::is_valid`](core::delaunay_triangulation::DelaunayTriangulation::is_valid) runs all **structural**
//! invariants (mappings, duplicates, per-cell validity, facet sharing, neighbors) and returns
//! only the first failure for convenience. For full diagnostics or to include the Delaunay
//! invariant, use [`core::delaunay_triangulation::DelaunayTriangulation::validation_report`]
//! with
//! [`core::triangulation_data_structure::ValidationOptions::check_delaunay`]
//! set to `true`.
//!
//! For detailed information, see:
//! - [`core::algorithms::incremental_insertion`] - Primary invariant enforcement during triangulation construction
//! - [`core::delaunay_triangulation::DelaunayTriangulation::validation_report`] - Comprehensive validation of all invariants
//!
//! # Correctness Guarantees and Limitations
//!
//! The library provides strong correctness guarantees for vertex insertion operations while being
//! transparent about edge cases and limitations.
//!
//! ## Guarantees
//!
//! When using [`DelaunayTriangulation::insert()`](core::delaunay_triangulation::DelaunayTriangulation::insert) or the underlying
//! insertion algorithms:
//!
//! 1. **Successful insertions maintain ALL invariants** - If insertion succeeds (`Ok(_)`), the
//!    triangulation is guaranteed to satisfy all structural and topological invariants, including
//!    the Delaunay property. The incremental cavity-based insertion algorithm maintains
//!    these invariants at all times.
//!
//! 2. **Failed insertions leave triangulation in valid state** - If insertion fails (`Err(_)`),
//!    the triangulation remains in a valid state with all invariants maintained. No partial or
//!    corrupted state is possible.
//!
//! 3. **Clear error messages** - Insertion failures include detailed error messages specifying
//!    which constraint or invariant was violated, along with context about what went wrong.
//!
//! 4. **No silent failures** - The library never silently produces incorrect triangulations.
//!    Operations either succeed with guarantees or fail with explicit errors.
//!
//! 5. **Duplicate vertex detection** - Duplicate and near-duplicate vertices (within `1e-10`
//!    epsilon) are automatically detected and rejected with
//!    [`InsertionError::InvalidVertex`](core::algorithms::incremental_insertion::InsertionError), preventing
//!    numerical instabilities.
//!
//! When constructing a triangulation from a batch of vertices using
//! [`DelaunayTriangulation::new`](core::delaunay_triangulation::DelaunayTriangulation::new):
//!
//! - Successful construction yields a triangulation that passes both
//!   `dt.is_valid()` and `dt.validate_delaunay()`.
//! - Duplicate coordinates are automatically detected and rejected.
//!
//! Incremental construction via [`DelaunayTriangulation::insert`](core::delaunay_triangulation::DelaunayTriangulation::insert)
//! follows the same invariant rules on each insertion: on success the triangulation
//! remains structurally valid and Delaunay; on failure the data structure is rolled
//! back to its previous state.
//!
//! ## Incremental insertion algorithm
//!
//! Triangulations are built using an efficient incremental cavity-based insertion algorithm:
//!
//! - **Initial simplex construction** - The first D+1 affinely independent vertices are used
//!   to create an initial valid simplex using robust orientation predicates. If no
//!   non-degenerate simplex can be formed, construction fails with
//!   [`TriangulationConstructionError::GeometricDegeneracy`](core::triangulation_data_structure::TriangulationConstructionError::GeometricDegeneracy).
//!
//! - **Incremental insertion** - Each subsequent vertex is inserted using a cavity-based
//!   algorithm that:
//!   1. Locates the vertex using efficient point location
//!   2. Identifies conflicting cells (those whose circumsphere contains the new vertex)
//!   3. Removes conflicting cells to create a cavity
//!   4. Fills the cavity with new cells connecting the cavity boundary to the new vertex
//!   5. Wires neighbor relationships locally without global recomputation
//!
//! The incremental insertion algorithm maintains all structural invariants and the
//! Delaunay property throughout construction. Vertices are only rejected if they would
//! violate fundamental geometric constraints (duplicates, near-duplicates, or degenerate
//! configurations).
//!
//! ## Delaunay validation
//!
//! The incremental insertion algorithm maintains the Delaunay property by construction,
//! ensuring that the empty circumsphere property holds after each insertion. Global
//! Delaunay validation can be performed explicitly using
//! [`DelaunayTriangulation::validate_delaunay`](core::delaunay_triangulation::DelaunayTriangulation::validate_delaunay)
//! when additional verification is needed.
//!
//! For construction from a batch of vertices using
//! [`DelaunayTriangulation::new`](core::delaunay_triangulation::DelaunayTriangulation::new),
//! the resulting triangulation is guaranteed to satisfy the Delaunay property.
//!
//! ## Error handling
//!
//! The incremental insertion algorithm provides clear error reporting for vertices that
//! cannot be inserted:
//!
//! - **Duplicate detection** - Exact and near-duplicate vertices are detected and rejected
//!   with [`InsertionError::InvalidVertex`](core::algorithms::incremental_insertion::InsertionError)
//! - **Geometric failures** - Degenerate configurations that would violate the Delaunay
//!   property are rejected with appropriate error messages
//! - **Validation failures** - If insertion would break structural invariants, the operation
//!   fails and the triangulation is left in its previous valid state
//!
//! ```rust
//! use delaunay::prelude::*;
//!
//! let vertices = vec![
//!     vertex!([0.0, 0.0, 0.0]),
//!     vertex!([1.0, 0.0, 0.0]),
//!     vertex!([0.0, 1.0, 0.0]),
//!     vertex!([0.0, 0.0, 1.0]),
//! ];
//!
//! let dt: DelaunayTriangulation<_, (), (), 3> =
//!     DelaunayTriangulation::new(&vertices).unwrap();
//!
//! assert_eq!(dt.number_of_vertices(), 4);
//! assert!(dt.is_valid().is_ok());
//! assert!(dt.validate_delaunay().is_ok());
//! ```
//!
//! ### Degenerate input handling
//!
//! When the input vertices cannot form a non-degenerate simplex (for example, when all points
//! are collinear in 2D), construction fails with
//! [`TriangulationConstructionError::GeometricDegeneracy`](core::triangulation_data_structure::TriangulationConstructionError::GeometricDegeneracy).
//!
//! ```rust
//! use delaunay::prelude::*;
//!
//! // All points lie on a line in 2D: no non-degenerate simplex exists.
//! let degenerate = vec![
//!     vertex!([0.0, 0.0]),
//!     vertex!([1.0, 0.0]),
//!     vertex!([2.0, 0.0]),
//!     vertex!([3.0, 0.0]),
//! ];
//!
//! let result: Result<DelaunayTriangulation<_, (), (), 2>, _> =
//!     DelaunayTriangulation::new(&degenerate);
//!
//! // Collinear points fail during incremental insertion due to degenerate simplices
//! assert!(matches!(
//!     result,
//!     Err(TriangulationConstructionError::FailedToAddVertex { .. })
//! ));
//! ```
//!
//! ## Limitations
//!
//! 1. **Degenerate geometry in higher dimensions** - Highly degenerate point configurations (e.g.,
//!    many nearly collinear or coplanar points) in 4D and 5D may cause insertion to fail gracefully
//!    with [`InsertionError`](core::algorithms::incremental_insertion::InsertionError).
//!    This is a known limitation of incremental algorithms in high-dimensional spaces with
//!    degenerate inputs.
//!
//! 2. **Iterative refinement constraints** - The cavity-based insertion algorithm uses iterative
//!    refinement to maintain the Delaunay property. In rare cases with complex geometries,
//!    refinement may hit topological constraints and fail gracefully rather than producing an
//!    invalid triangulation.
//!
//! 3. **Numerical precision** - Like all computational geometry libraries, numerical precision can
//!    affect results near floating-point boundaries. The library uses robust predicates to minimize
//!    these issues, but extreme coordinate values or ill-conditioned point sets may still cause
//!    problems.
//!
//! ## Simple API Usage
//!
//! ```rust
//! use delaunay::prelude::*;
//!
//! // Create 4D triangulation - uses fast predicates by default (f64)
//! let vertices = vec![
//!     vertex!([0.0, 0.0, 0.0, 0.0]),
//!     vertex!([1.0, 0.0, 0.0, 0.0]),
//!     vertex!([0.0, 1.0, 0.0, 0.0]),
//!     vertex!([0.0, 0.0, 1.0, 0.0]),
//!     vertex!([0.0, 0.0, 0.0, 1.0]),
//! ];
//!
//! let dt: DelaunayTriangulation<_, (), (), 4> =
//!     DelaunayTriangulation::new(&vertices).unwrap();
//!
//! assert_eq!(dt.number_of_vertices(), 5);
//! assert_eq!(dt.dim(), 4);
//! assert_eq!(dt.number_of_cells(), 1);  // Single 4-simplex
//!
//! // Also works in 2D
//! let vertices_2d = vec![
//!     vertex!([0.0, 0.0]),
//!     vertex!([1.0, 0.0]),
//!     vertex!([0.5, 1.0]),
//! ];
//!
//! let dt_2d: DelaunayTriangulation<_, (), (), 2> =
//!     DelaunayTriangulation::new(&vertices_2d).unwrap();
//!
//! assert_eq!(dt_2d.number_of_vertices(), 3);
//! assert_eq!(dt_2d.dim(), 2);
//! ```
//!
//! For implementation details on invariant validation and error handling, see
//! [`core::algorithms::incremental_insertion`].
//!
//! # References
//!
//! The algorithms and geometric predicates in this library are based on established computational
//! geometry literature. For a comprehensive list of academic references and bibliographic citations,
//! see the [REFERENCES.md](https://github.com/acgetchell/delaunay/blob/main/REFERENCES.md) file in the repository.
//!
//! ## Project History
//! Versions ≤ 0.1.0 were maintained at [old repo](https://github.com/oovm/shape-rs).
//! Versions ≥ 0.3.4 are maintained [here](https://github.com/acgetchell/delaunay).
//!
//! See <https://docs.rs/delaunay/0.1.0> for historical documentation.
//! See <https://docs.rs/delaunay> for the latest documentation.

// Allow multiple crate versions due to transitive dependencies
#![expect(clippy::multiple_crate_versions)]
// Temporarily allow deprecated warnings during API migration (v0.6.0)
// - Facet -> FacetView migration
// - Tds::new()/add() -> DelaunayTriangulation::new()/insert()
// Forbid unsafe code throughout the entire crate
#![forbid(unsafe_code)]

#[macro_use]
extern crate derive_builder;

/// The `core` module contains the primary data structures and algorithms for building and manipulating Delaunay triangulations.
///
/// It includes the `Tds` struct, which represents the triangulation, as well as `Cell`, `Facet`, and `Vertex` components.
/// This module provides traits for customizing vertex and cell data. The crate also includes a `prelude` module for convenient access to commonly used types.
pub mod core {
    /// Triangulation algorithms for construction, maintenance, and querying
    pub mod algorithms {
        /// Bistellar flip operations - Phase 3 TODO
        pub mod flips;
        /// Incremental cavity-based insertion (Phase 3.6)
        pub mod incremental_insertion;
        /// Point location algorithms (facet walking)
        pub mod locate;
    }
    pub mod boundary;
    pub mod cell;
    /// High-performance collection types optimized for computational geometry
    pub mod collections;
    /// Delaunay triangulation layer with incremental insertion - Phase 3 TODO
    pub mod delaunay_triangulation;
    pub mod facet;
    /// Generic triangulation combining kernel + Tds - Phase 2 TODO
    pub mod triangulation;
    pub mod triangulation_data_structure;
    pub mod util;
    pub mod vertex;
    /// Traits for Delaunay triangulation data structures.
    pub mod traits {
        pub mod boundary_analysis;
        pub mod data_type;
        pub mod facet_cache;
        pub use boundary_analysis::*;
        pub use data_type::*;
        pub use facet_cache::*;
    }
    // Re-export the `core` modules.
    pub use cell::*;
    pub use delaunay_triangulation::*;
    pub use facet::*;
    pub use traits::*;
    pub use triangulation_data_structure::*;
    pub use util::*;
    pub use vertex::*;
    // Note: collections module not re-exported here to avoid namespace pollution
    // Import specific types via prelude or use crate::core::collections::
}

/// Contains geometric types including the `Point` struct and geometry predicates.
///
/// The geometry module provides a coordinate abstraction through the `Coordinate` trait
/// that unifies coordinate operations across different storage mechanisms. The `Point`
/// type implements this abstraction, providing generic floating-point coordinate support
/// (for `f32`, `f64`, and other types implementing `CoordinateScalar`) with proper NaN
/// handling, validation, and hashing.
pub mod geometry {
    /// Geometric algorithms for triangulations and spatial data structures
    pub mod algorithms {
        /// Convex hull operations on d-dimensional triangulations
        pub mod convex_hull;
        pub use convex_hull::*;
    }
    /// Geometric kernel abstraction (CGAL-style) - Phase 2 TODO
    pub mod kernel;
    pub mod matrix;
    pub mod point;
    pub mod predicates;
    /// Geometric quality measures for d-dimensional simplicial cells
    pub mod quality;
    /// Enhanced predicates with improved numerical robustness
    pub mod robust_predicates;
    /// Geometric utility functions for d-dimensional geometry calculations
    pub mod util;
    /// Traits module containing coordinate abstractions and reusable trait definitions.
    ///
    /// This module contains the core `Coordinate` trait that abstracts coordinate
    /// operations, along with supporting traits for validation (`FiniteCheck`),
    /// equality comparison (`OrderedEq`), and hashing (`HashCoordinate`) of
    /// floating-point coordinate values.
    pub mod traits {
        pub mod coordinate;
        pub use coordinate::*;
    }
    pub use algorithms::*;
    pub use matrix::*;
    pub use point::*;
    pub use predicates::*;
    pub use quality::*;
    pub use traits::*;
    pub use util::*;
}

/// A prelude module that re-exports commonly used types and macros.
/// This makes it easier to import the most commonly used items from the crate.
pub mod prelude {
    // Re-export from core
    pub use crate::core::{
        cell::*,
        delaunay_triangulation::*,
        facet::*,
        traits::{boundary_analysis::*, data_type::*},
        triangulation_data_structure::*,
        util::*,
        vertex::*,
    };

    // Re-export commonly used collection types from core::collections
    // These are frequently used in advanced examples and downstream code
    pub use crate::core::collections::{
        CellNeighborsMap, FacetToCellsMap, FastHashMap, FastHashSet, SmallBuffer, VertexToCellsMap,
        fast_hash_map_with_capacity, fast_hash_set_with_capacity,
    };

    // Re-export from geometry
    pub use crate::geometry::{
        algorithms::*, kernel::*, matrix::*, point::*, predicates::*, quality::*,
        robust_predicates::*, traits::coordinate::*, util::*,
    };

    // Convenience macros
    pub use crate::vertex;
}

/// The function `is_normal` checks that structs implement `auto` traits.
/// Traits are checked at compile time, so this function is only used for
/// testing.
#[must_use]
pub const fn is_normal<T: Sized + Send + Sync + Unpin>() -> bool {
    true
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use crate::{
        core::{cell::Cell, triangulation_data_structure::Tds, vertex::Vertex},
        geometry::Point,
        is_normal,
    };

    // =============================================================================
    // TYPE SAFETY TESTS
    // =============================================================================

    #[test]
    fn normal_types() {
        assert!(is_normal::<Point<f64, 3>>());
        assert!(is_normal::<Point<f32, 3>>());
        assert!(is_normal::<Vertex<f64, (), 3>>());
        assert!(is_normal::<Cell<f64, (), (), 4>>());
        assert!(is_normal::<Tds<f64, (), (), 4>>());
    }

    #[test]
    fn test_prelude_collections_exports() {
        use crate::prelude::*;

        // Test that we can use the collections from the prelude
        let mut map: FastHashMap<u64, usize> = FastHashMap::default();
        map.insert(123, 456);
        assert_eq!(map.get(&123), Some(&456));

        let mut set: FastHashSet<u64> = FastHashSet::default();
        set.insert(789);
        assert!(set.contains(&789));

        let mut buffer: SmallBuffer<i32, 8> = SmallBuffer::new();
        buffer.push(42);
        assert_eq!(buffer.len(), 1);

        // Test capacity helpers
        let map_with_cap = fast_hash_map_with_capacity::<u64, usize>(100);
        assert!(map_with_cap.capacity() >= 100);

        let set_with_cap = fast_hash_set_with_capacity::<u64>(50);
        assert!(set_with_cap.capacity() >= 50);

        // Test domain-specific types can be instantiated
        let _facet_map: FacetToCellsMap = FacetToCellsMap::default();
        let _neighbors: CellNeighborsMap = CellNeighborsMap::default();
        let _vertex_cells: VertexToCellsMap = VertexToCellsMap::default();
    }

    #[test]
    fn test_prelude_quality_exports() {
        use crate::prelude::*;

        // Test that quality functions are accessible from prelude
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Get a cell to test quality functions
        let (cell_key, _) = dt.cells().next().unwrap();

        // Test that quality functions are accessible
        let ratio = radius_ratio(dt.triangulation(), cell_key).unwrap();
        assert!(ratio > 0.0);

        let norm_vol = normalized_volume(dt.triangulation(), cell_key).unwrap();
        assert!(norm_vol > 0.0);
    }

    // =============================================================================
    // ALLOCATION COUNTING TESTS
    // =============================================================================

    /// Run these with `cargo test allocation_counting --features count-allocations`
    #[cfg(feature = "count-allocations")]
    #[test]
    fn test_basic_allocation_counting() {
        use allocation_counter::measure;

        // Test a trivial operation that should not allocate
        let result = measure(|| {
            let x = 1 + 1;
            assert_eq!(x, 2);
        });

        // Assert that the returned struct has the expected fields
        // Available fields: count_total, count_current, count_max, bytes_total, bytes_current, bytes_max
        // For a trivial operation, we expect zero allocations
        assert_eq!(
            result.count_total, 0,
            "Expected zero total allocations for trivial operation, found: {}",
            result.count_total
        );
        assert_eq!(
            result.bytes_total, 0,
            "Expected zero total bytes allocated for trivial operation, found: {}",
            result.bytes_total
        );

        // Also check that current allocations are zero (no leaked allocations)
        assert_eq!(
            result.count_current, 0,
            "Expected zero current allocations after trivial operation, found: {}",
            result.count_current
        );
        assert_eq!(
            result.bytes_current, 0,
            "Expected zero current bytes allocated after trivial operation, found: {}",
            result.bytes_current
        );
    }

    #[cfg(feature = "count-allocations")]
    #[test]
    fn test_allocation_counting_with_allocating_operation() {
        use allocation_counter::measure;

        // Test an operation that does allocate memory
        let result = measure(|| {
            let _vec: Vec<i32> = vec![1, 2, 3, 4, 5];
        });

        // For this operation, we expect some allocations
        assert!(
            result.count_total > 0,
            "Expected some allocations for Vec creation, found: {}",
            result.count_total
        );
        assert!(
            result.bytes_total > 0,
            "Expected some bytes allocated for Vec creation, found: {}",
            result.bytes_total
        );

        // After the operation, current allocations should be zero (Vec was dropped)
        assert_eq!(
            result.count_current, 0,
            "Expected zero current allocations after Vec drop, found: {}",
            result.count_current
        );
        assert_eq!(
            result.bytes_current, 0,
            "Expected zero current bytes after Vec drop, found: {}",
            result.bytes_current
        );

        // Max values should be at least as large as total (they track peak usage)
        assert!(
            result.count_max >= result.count_total,
            "Max count should be >= total count"
        );
        assert!(
            result.bytes_max >= result.bytes_total,
            "Max bytes should be >= total bytes"
        );
    }
}
