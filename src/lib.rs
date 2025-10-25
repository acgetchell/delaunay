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
//! use delaunay::core::triangulation_data_structure::Tds;
//! use delaunay::vertex;
//!
//! // Create a 4D triangulation (4-dimensional space!)
//! let mut tds: Tds<f64, Option<()>, Option<()>, 4> = Tds::empty();
//!
//! // Add vertices incrementally - triangulation evolves automatically
//! tds.add(vertex!([0.0, 0.0, 0.0, 0.0])).unwrap();  // 1 vertex, 0 cells
//! tds.add(vertex!([1.0, 0.0, 0.0, 0.0])).unwrap();  // 2 vertices, 0 cells
//! tds.add(vertex!([0.0, 1.0, 0.0, 0.0])).unwrap();  // 3 vertices, 0 cells
//! tds.add(vertex!([0.0, 0.0, 1.0, 0.0])).unwrap();  // 4 vertices, 0 cells
//! assert_eq!(tds.number_of_cells(), 0);
//! tds.add(vertex!([0.0, 0.0, 0.0, 1.0])).unwrap();  // 5 vertices, 1 cell (first 4-simplex!)
//! tds.add(vertex!([0.2, 0.2, 0.2, 0.2])).unwrap();  // 6 vertices, multiple cells
//!
//! assert_eq!(tds.number_of_vertices(), 6);
//! assert_eq!(tds.dim(), 4);                    // Full 4D triangulation
//! assert!(tds.number_of_cells() > 1);          // Bowyer-Watson creates additional 4-simplices
//! assert!(tds.is_valid().is_ok());             // Maintains Delaunay property in 4D
//! ```
//!
//! **Key insight**: The transition happens at D+1 vertices (5 vertices for 4D), where the first
//! 4-simplex (5-vertex cell) is created. Additional vertices trigger the Bowyer-Watson algorithm
//! to maintain the 4D Delaunay triangulation.
//!
//! # Convex Hull Extraction
//!
//! Extract d-dimensional convex hulls from Delaunay triangulations:
//!
//! ```rust
//! use delaunay::core::triangulation_data_structure::Tds;
//! use delaunay::geometry::algorithms::convex_hull::ConvexHull;
//! use delaunay::geometry::point::Point;
//! use delaunay::geometry::traits::coordinate::Coordinate;
//! use delaunay::vertex;
//!
//! // Create two tetrahedrons sharing a triangular facet (double tetrahedron)
//! let vertices = vec![
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
//! let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
//!
//! // Extract the convex hull (boundary facets of the triangulation)
//! let hull: ConvexHull<f64, Option<()>, Option<()>, 3> =
//!     ConvexHull::from_triangulation(&tds).unwrap();
//!
//! println!("Convex hull has {} facets in {}D", hull.facet_count(), hull.dimension());
//!
//! // Test point containment
//! let inside_point = Point::new([1.0, 0.5, 0.5]);
//! let outside_point = Point::new([3.0, 3.0, 3.0]);
//!
//! assert!(!hull.is_point_outside(&inside_point, &tds).unwrap());  // Inside the hull
//! assert!(hull.is_point_outside(&outside_point, &tds).unwrap());   // Outside the hull
//!
//! // Find visible facets from an external point (useful for incremental construction)
//! let visible_facets = hull.find_visible_facets(&outside_point, &tds).unwrap();
//! println!("Point sees {} out of {} facets", visible_facets.len(), hull.facet_count());
//!
//! // Works in any dimension!
//! let vertices_4d = vec![
//!     vertex!([0.0, 0.0, 0.0, 0.0]),
//!     vertex!([1.0, 0.0, 0.0, 0.0]),
//!     vertex!([0.0, 1.0, 0.0, 0.0]),
//!     vertex!([0.0, 0.0, 1.0, 0.0]),
//!     vertex!([0.0, 0.0, 0.0, 1.0]),
//! ];
//! let tds_4d: Tds<f64, Option<()>, Option<()>, 4> = Tds::new(&vertices_4d).unwrap();
//! let hull_4d: ConvexHull<f64, Option<()>, Option<()>, 4> =
//!     ConvexHull::from_triangulation(&tds_4d).unwrap();
//!
//! assert_eq!(hull_4d.facet_count(), 5);  // 4-simplex has 5 boundary facets
//! assert_eq!(hull_4d.dimension(), 4);     // 4D convex hull
//! ```
//!
//! # Triangulation Invariants
//!
//! The library maintains several critical triangulation invariants that ensure geometric and topological correctness:
//!
//! ## Invariant Enforcement
//!
//! | Invariant Type | Enforcement Location | Method | Timing |
//! |---|---|---|---|
//! | **Delaunay Property** | `bowyer_watson::find_bad_cells()` | Empty circumsphere test using `insphere()` | **Proactive** (during construction) |
//! | **Facet Sharing** | `validate_facet_sharing()` | Each facet shared by ≤ 2 cells | **Reactive** (via validation) |
//! | **No Duplicate Cells** | `validate_no_duplicate_cells()` | No cells with identical vertex sets | **Reactive** (via validation) |
//! | **Neighbor Consistency** | `validate_neighbors_internal()` | Mutual neighbor relationships | **Reactive** (via validation) |
//! | **Cell Validity** | `CellBuilder::validate()` (vertex count) + [`cell.is_valid()`](core::cell::Cell::is_valid) (comprehensive) | Construction + runtime validation | **Both** (construction + validation) |
//! | **Vertex Validity** | `Point::from()` (coordinates) + UUID auto-gen + `vertex.is_valid()` | Construction + runtime validation | **Both** (construction + validation) |
//!
//! The **Delaunay property** (empty circumsphere) is enforced **proactively** during construction by removing
//! violating cells, while **structural invariants** are enforced **reactively** through validation methods.
//!
//! For detailed information, see:
//! - [`core::algorithms::bowyer_watson`] - Primary invariant enforcement during triangulation construction
//! - [`core::triangulation_data_structure::Tds::is_valid`] - Comprehensive validation of all invariants
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
#![allow(clippy::multiple_crate_versions)]
// Temporarily allow deprecated warnings during Facet -> FacetView migration
#![allow(deprecated)]
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
        /// Pure incremental Bowyer-Watson algorithm for Delaunay triangulation
        pub mod bowyer_watson;
        /// Robust Bowyer-Watson implementation with enhanced numerical stability
        pub mod robust_bowyer_watson;
        pub use bowyer_watson::*;
        pub use robust_bowyer_watson::*;
    }
    pub mod boundary;
    pub mod cell;
    /// High-performance collection types optimized for computational geometry
    pub mod collections;
    pub mod facet;
    pub mod triangulation_data_structure;
    pub mod util;
    pub mod vertex;
    /// Traits for Delaunay triangulation data structures.
    pub mod traits {
        pub mod boundary_analysis;
        pub mod data_type;
        pub mod facet_cache;
        pub mod insertion_algorithm;
        pub use boundary_analysis::*;
        pub use data_type::*;
        pub use facet_cache::*;
        pub use insertion_algorithm::*;
    }
    // Re-export the `core` modules.
    pub use cell::*;
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
        facet::*,
        traits::{boundary_analysis::*, data_type::*, insertion_algorithm::*},
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
        algorithms::*, matrix::*, point::*, predicates::*, quality::*, robust_predicates::*,
        traits::coordinate::*, util::*,
    };

    // Convenience macros
    pub use crate::{cell, vertex};
}

/// The function `is_normal` checks that structs implement `auto` traits.
/// Traits are checked at compile time, so this function is only used for
/// testing.
#[allow(clippy::extra_unused_type_parameters)]
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
        assert!(is_normal::<Vertex<f64, Option<()>, 3>>());
        assert!(is_normal::<Cell<f64, Option<()>, Option<()>, 4>>());
        assert!(is_normal::<Tds<f64, Option<()>, Option<()>, 4>>());
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
        // Create a simple 2D triangle
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();

        // Get a cell to test quality functions
        let (cell_key, _) = tds.cells().next().unwrap();

        // Test that quality functions are accessible
        let ratio = radius_ratio(&tds, cell_key).unwrap();
        assert!(ratio > 0.0);

        let norm_vol = normalized_volume(&tds, cell_key).unwrap();
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
