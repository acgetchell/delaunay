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
//! assert!(tds.is_valid().is_ok());             // Structural invariants hold for the triangulation
//! // Optional (expensive): validate global Delaunay property
//! // tds.validate_delaunay().unwrap();
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
//! The triangulation data structure maintains a set of **structural** and **geometric** invariants
//! that are checked by [`Tds::is_valid`](core::triangulation_data_structure::Tds::is_valid) and
//! [`Tds::validation_report`](core::triangulation_data_structure::Tds::validation_report):
//!
//! - **Vertex mappings** – every vertex UUID has a corresponding key and vice versa.
//! - **Cell mappings** – every cell UUID has a corresponding key and vice versa.
//! - **No duplicate cells** – no two maximal cells share the same vertex set.
//! - **Cell validity** – each cell has the correct number of vertices and passes internal
//!   consistency checks.
//! - **Facet sharing** – each facet is shared by at most 2 cells (1 on the boundary, 2 in the interior).
//! - **Neighbor consistency** – neighbor relationships are mutual and reference a shared facet.
//! - **Delaunay property** – triangulations constructed via the Bowyer–Watson
//!   pipeline maintain the empty circumsphere invariant; no vertex lies strictly
//!   inside the circumsphere of any maximal cell.
//!
//! ## Validation helpers
//!
//! These invariants are exposed through focused validation helpers on
//! [`core::triangulation_data_structure::Tds`]:
//!
//! | Invariant | Helper method | Notes |
//! |---|---|---|
//! | Vertex mappings | [`Tds::validate_vertex_mappings`](core::triangulation_data_structure::Tds::validate_vertex_mappings) | Ensures UUID↔key consistency for all vertices. |
//! | Cell mappings | [`Tds::validate_cell_mappings`](core::triangulation_data_structure::Tds::validate_cell_mappings) | Ensures UUID↔key consistency for all cells. |
//! | Duplicate cells | [`Tds::validate_no_duplicate_cells`](core::triangulation_data_structure::Tds::validate_no_duplicate_cells) | Detects maximal cells with identical vertex sets. |
//! | Cell validity | [`Cell::is_valid`](core::cell::Cell::is_valid) (aggregated via [`Tds::validation_report`](core::triangulation_data_structure::Tds::validation_report)) | Per-cell structural checks. |
//! | Facet sharing | [`Tds::validate_facet_sharing`](core::triangulation_data_structure::Tds::validate_facet_sharing) | Verifies that each facet is shared by ≤ 2 cells. |
//! | Neighbor consistency | [`Tds::validate_neighbors`](core::triangulation_data_structure::Tds::validate_neighbors) | Verifies neighbor topology and mutual relationships. |
//! | Delaunay property | [`Tds::validate_delaunay`](core::triangulation_data_structure::Tds::validate_delaunay) | Expensive global empty-circumsphere check (optional). |
//!
//! [`Tds::is_valid`](core::triangulation_data_structure::Tds::is_valid) runs all **structural**
//! invariants (mappings, duplicates, per-cell validity, facet sharing, neighbors) and returns
//! only the first failure for convenience. For full diagnostics or to include the Delaunay
//! invariant, use [`core::triangulation_data_structure::Tds::validation_report`]
//! with
//! [`core::triangulation_data_structure::ValidationOptions::check_delaunay`]
//! set to `true`.
//!
//! For detailed information, see:
//! - [`core::algorithms::bowyer_watson`] - Primary invariant enforcement during triangulation construction
//! - [`core::triangulation_data_structure::Tds::validation_report`] - Comprehensive validation of all invariants
//!
//! # Correctness Guarantees and Limitations
//!
//! The library provides strong correctness guarantees for vertex insertion operations while being
//! transparent about edge cases and limitations.
//!
//! ## Guarantees
//!
//! When using [`Tds::add()`](core::triangulation_data_structure::Tds::add) or the underlying
//! insertion algorithms:
//!
//! 1. **Successful insertions maintain ALL invariants** - If insertion succeeds (`Ok(_)`), the
//!    triangulation is guaranteed to satisfy all structural and topological invariants, including
//!    the Delaunay property. The unified fast+robust Bowyer–Watson pipeline may skip
//!    unsalvageable vertices, but it never leaves a non-Delaunay triangulation.
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
//!    [`InsertionError::InvalidVertex`](core::traits::InsertionError::InvalidVertex), preventing
//!    numerical instabilities.
//!
//! When constructing a triangulation from a batch of vertices using
//! [`Tds::new`](core::triangulation_data_structure::Tds::new):
//!
//! - Successful construction yields a triangulation that passes both
//!   `tds.is_valid()` and `tds.validate_delaunay()`.
//! - Duplicate coordinates are silently filtered during construction.
//! - Any vertices that cannot be inserted without breaking invariants are reported via
//!   [`TriangulationDiagnostics::unsalvageable_vertices`](core::triangulation_data_structure::TriangulationDiagnostics).
//!
//! Incremental construction via [`Tds::add`](core::triangulation_data_structure::Tds::add)
//! follows the same invariant rules on each insertion: on success the triangulation
//! remains structurally valid and Delaunay; on failure the data structure is rolled
//! back to its previous state.
//!
//! ## Two-stage insertion pipeline and diagnostics
//!
//! Triangulations are built by a unified Bowyer–Watson pipeline with two conceptual
//! stages:
//!
//! - **Stage 1 – robust initial simplex search** filters exact and near-duplicate
//!   coordinates, then searches for D+1 affinely independent vertices using robust
//!   orientation predicates. If such a simplex is found it seeds the triangulation.
//!   If no non-degenerate simplex exists, construction fails with
//!   [`TriangulationConstructionError::GeometricDegeneracy`](core::triangulation_data_structure::TriangulationConstructionError::GeometricDegeneracy)
//!   and, when possible, leaves behind a **zero-cell triangulation**: all unique
//!   input vertices are present, but no cells are created.
//!
//! - **Stage 2 – per-vertex fast → robust → skip insertion** classifies the remaining
//!   vertices (unique, duplicate, near-duplicate, or degenerate) and inserts them via
//!   a shared fast/robust pipeline. The fast path uses an incremental Bowyer–Watson
//!   implementation; if it encounters a recoverable geometric failure, a robust
//!   fallback is tried. If both paths fail, the vertex is marked *unsalvageable* and
//!   skipped without modifying the triangulation, and processing continues with later
//!   vertices.
//!
//! After Stage 2 the triangulation is finalized (duplicate cells removed, facet
//! sharing repaired, neighbors and incident cells assigned) and a global Delaunay
//! validation/repair pass is run. A successful construction therefore satisfies both
//! the structural invariants described above and the global Delaunay empty-
//! circumsphere property.
//!
//! ## Delaunay validation cadence
//!
//! Global Delaunay checks can be expensive, so the pipeline exposes a policy type
//! [`DelaunayCheckPolicy`](core::traits::insertion_algorithm::DelaunayCheckPolicy) to
//! control how often they run:
//!
//! - `DelaunayCheckPolicy::EndOnly` (the default) runs validation once at the end of
//!   triangulation. This matches the legacy behavior used by most callers.
//! - `DelaunayCheckPolicy::EveryN(k)` runs validation after every `k` successful
//!   vertex insertions *in addition* to the final pass, which is useful for tests and
//!   debug builds.
//!
//! Zero-cell triangulations are a special case: when `number_of_cells() == 0` there
//! is nothing to validate, so the policy-based validator is a no-op.
//!
//! You can select a policy explicitly via
//! [`Tds::bowyer_watson_with_diagnostics_and_policy`](core::triangulation_data_structure::Tds::bowyer_watson_with_diagnostics_and_policy);
//! higher-level constructors such as
//! [`Tds::new`](core::triangulation_data_structure::Tds::new) use
//! `DelaunayCheckPolicy::EndOnly` internally.
//!
//! ## Observability, statistics, and unsalvageable vertices
//!
//! To inspect how a triangulation was constructed, use
//! [`Tds::bowyer_watson_with_diagnostics`](core::triangulation_data_structure::Tds::bowyer_watson_with_diagnostics),
//! which returns a [`TriangulationDiagnostics`](core::triangulation_data_structure::TriangulationDiagnostics)
//! value containing:
//!
//! - `unsalvageable_vertices`: a list of
//!   [`UnsalvageableVertexReport`](core::traits::insertion_algorithm::UnsalvageableVertexReport)
//!   entries, each with the original vertex, its classification (duplicate,
//!   near-duplicate, or degenerate), the sequence of insertion strategies that were
//!   attempted, and the corresponding error chain.
//! - `statistics`: a
//!   [`TriangulationStatistics`](core::triangulation_data_structure::TriangulationStatistics)
//!   record aggregating Stage 1 + Stage 2 behavior (fast/robust attempts and
//!   successes, how many vertices were skipped as duplicates vs genuinely
//!   unsalvageable, and how many global Delaunay validation runs occurred under the
//!   selected
//!   [`DelaunayCheckPolicy`](core::traits::insertion_algorithm::DelaunayCheckPolicy)).
//!
//! Vertices that appear in `unsalvageable_vertices` are guaranteed not to appear in
//! any triangulation cell: the unified pipeline fully skips them, so the final
//! triangulation is always described entirely by the kept subset of vertices.
//!
//! ```rust
//! use delaunay::core::triangulation_data_structure::{Tds, TriangulationStatistics};
//! use delaunay::vertex;
//!
//! let vertices = vec![
//!     vertex!([0.0, 0.0, 0.0]),
//!     vertex!([1.0, 0.0, 0.0]),
//!     vertex!([0.0, 1.0, 0.0]),
//!     vertex!([0.0, 0.0, 1.0]),
//! ];
//!
//! let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
//! let diagnostics = tds.bowyer_watson_with_diagnostics().unwrap();
//!
//! // Aggregated statistics for this triangulation run.
//! let stats: &TriangulationStatistics = &diagnostics.statistics;
//! assert!(stats.fast_path_successes + stats.robust_path_successes >= 0);
//! assert!(
//!     stats.global_delaunay_validation_runs >= 1,
//!     "Bowyer–Watson pipeline should run at least one global Delaunay validation",
//! );
//!
//! // Convenience: access the last triangulation statistics directly from the TDS.
//! if let Some(last) = tds.last_triangulation_statistics() {
//!     assert_eq!(last.insertion.vertices_processed, stats.insertion.vertices_processed);
//!     assert!(last.global_delaunay_validation_runs >= 1);
//! }
//! ```
//!
//! The accessor
//! [`Tds::last_triangulation_statistics`](core::triangulation_data_structure::Tds::last_triangulation_statistics)
//! always returns the statistics for the most recent successful Bowyer–Watson-based
//! construction (`Tds::new` or a subsequent call to one of the `bowyer_watson_*` helpers).
//!
//! ### Zero-cell triangulations and recovery
//!
//! When Stage 1 cannot find a non-degenerate simplex (for example, when all points
//! are collinear in 2D), the library reports geometric degeneracy but leaves behind
//! a valid zero-cell triangulation that still contains the unique input vertices.
//! Callers can then recover by incrementally adding additional vertices.
//!
//! ```no_run
//! use delaunay::core::algorithms::bowyer_watson::IncrementalBowyerWatson;
//! use delaunay::core::traits::insertion_algorithm::InsertionAlgorithm;
//! use delaunay::core::triangulation_data_structure::{
//!     Tds, TriangulationConstructionError,
//! };
//! use delaunay::vertex;
//!
//! type Alg = IncrementalBowyerWatson<f64, Option<()>, Option<()>, 2>;
//!
//! // All points lie on a line in 2D: no non-degenerate simplex exists.
//! let degenerate = vec![
//!     vertex!([0.0, 0.0]),
//!     vertex!([1.0, 0.0]),
//!     vertex!([2.0, 0.0]),
//!     vertex!([3.0, 0.0]),
//! ];
//!
//! let mut tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::empty();
//! let mut algorithm: Alg = Alg::new();
//!
//! let result = <Alg as InsertionAlgorithm<_, _, _, 2>>::triangulate(
//!     &mut algorithm,
//!     &mut tds,
//!     &degenerate,
//! );
//!
//! if let Err(TriangulationConstructionError::GeometricDegeneracy { .. }) = result {
//!     // Zero-cell fallback: vertices are retained, no cells are created, and the
//!     // triangulation remains structurally valid.
//!     assert_eq!(tds.number_of_cells(), 0);
//!     assert_eq!(tds.number_of_vertices(), degenerate.len());
//!     assert!(tds.is_valid().is_ok());
//! }
//! ```
//!
//! From this state you can continue building the triangulation incrementally with
//! [`Tds::add`](core::triangulation_data_structure::Tds::add); additional
//! non-degenerate vertices can be inserted without rebuilding from scratch, and the
//! TDS remains valid throughout.
//!
//! ## Limitations
//!
//! 1. **Degenerate geometry in higher dimensions** - Highly degenerate point configurations (e.g.,
//!    many nearly collinear or coplanar points) in 4D and 5D may cause insertion to fail gracefully
//!    with [`InsertionError::GeometricFailure`](core::traits::InsertionError::GeometricFailure).
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
//! ## Error Handling
//!
//! ```rust
//! use delaunay::core::triangulation_data_structure::{Tds, TriangulationConstructionError};
//! use delaunay::vertex;
//!
//! let mut tds: Tds<f64, Option<()>, Option<()>, 4> = Tds::empty();
//!
//! // Add initial vertices
//! tds.add(vertex!([0.0, 0.0, 0.0, 0.0])).unwrap();
//! tds.add(vertex!([1.0, 0.0, 0.0, 0.0])).unwrap();
//!
//! // Attempt to add a duplicate vertex - will fail gracefully
//! match tds.add(vertex!([1.0, 0.0, 0.0, 0.0])) {
//!     Ok(_) => println!("Insertion succeeded"),
//!     Err(TriangulationConstructionError::DuplicateCoordinates { coordinates }) => {
//!         println!("Duplicate vertex detected: {}", coordinates);
//!         // Triangulation remains valid - can continue with other vertices
//!     }
//!     Err(TriangulationConstructionError::GeometricDegeneracy { message }) => {
//!         println!("Geometry too degenerate: {}", message);
//!         // Triangulation remains valid - insertion was rejected
//!     }
//!     Err(e) => println!("Other error: {}", e),
//! }
//!
//! // Triangulation remains valid regardless of insertion outcome
//! assert!(tds.is_valid().is_ok());
//! ```
//!
//! For implementation details on invariant validation and error handling, see
//! [`core::traits::insertion_algorithm`].
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
// Temporarily allow deprecated warnings during Facet -> FacetView migration
#![expect(deprecated)]
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
        /// Internal unified fast+robust insertion pipeline for Stage 2
        pub mod unified_insertion_pipeline;
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
