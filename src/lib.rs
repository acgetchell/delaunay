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
//! - Arbitrary data types associated with vertices and cells
//! - Serialization/Deserialization with [serde](https://serde.rs)
//!
//! ## Project History
//! Versions ≤ 0.1.0 were maintained at [old repo](https://github.com/oovm/shape-rs).
//! Versions ≥ 0.3.4 are maintained [here](https://github.com/acgetchell/delaunay).
//!
//! See <https://docs.rs/delaunay/0.1.0> for historical documentation.
//! See <https://docs.rs/delaunay> for the latest documentation.

// Allow multiple crate versions due to transitive dependencies
#![allow(clippy::multiple_crate_versions)]

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
        pub use bowyer_watson::*;
    }
    pub mod boundary;
    pub mod cell;
    pub mod facet;
    pub mod triangulation_data_structure;
    pub mod utilities;
    pub mod vertex;
    /// Traits for Delaunay triangulation data structures.
    pub mod traits {
        pub mod boundary_analysis;
        pub mod data_type;
        pub use boundary_analysis::*;
        pub use data_type::*;
    }
    // Re-export the `core` modules.
    pub use cell::*;
    pub use facet::*;
    pub use traits::*;
    pub use triangulation_data_structure::*;
    pub use utilities::*;
    pub use vertex::*;
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
    /// Traits module containing coordinate abstractions and reusable trait definitions.
    ///
    /// This module contains the core `Coordinate` trait that abstracts coordinate
    /// operations, along with supporting traits for validation (`FiniteCheck`),
    /// equality comparison (`OrderedEq`), and hashing (`HashCoordinate`) of
    /// floating-point coordinate values.
    pub mod traits {
        pub mod coordinate;
        pub mod finitecheck;
        pub mod hashcoordinate;
        pub mod orderedeq;
        pub use coordinate::*;
        pub use finitecheck::*;
        pub use hashcoordinate::*;
        pub use orderedeq::*;
    }
    pub use algorithms::*;
    pub use matrix::*;
    pub use point::*;
    pub use predicates::*;
    pub use traits::*;
}

/// A prelude module that re-exports commonly used types and macros.
/// This makes it easier to import the most commonly used items from the crate.
pub mod prelude {
    // Re-export from core
    pub use crate::core::{
        cell::*,
        facet::*,
        traits::{boundary_analysis::*, data_type::*},
        triangulation_data_structure::*,
        utilities::*,
        vertex::*,
    };

    // Re-export from geometry
    pub use crate::geometry::{
        algorithms::*,
        matrix::*,
        point::*,
        predicates::*,
        traits::{coordinate::*, finitecheck::*, hashcoordinate::*, orderedeq::*},
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
        core::{cell::Cell, facet::Facet, triangulation_data_structure::Tds, vertex::Vertex},
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
        assert!(is_normal::<Facet<f64, Option<()>, Option<()>, 3>>());
        assert!(is_normal::<Cell<f64, Option<()>, Option<()>, 4>>());
        assert!(is_normal::<Tds<f64, Option<()>, Option<()>, 4>>());
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
