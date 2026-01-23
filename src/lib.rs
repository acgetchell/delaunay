//! # delaunay
//!
//! This is a library for computing the Delaunay triangulation of a set of n-dimensional points
//! in a [simplicial complex](https://grokipedia.com/page/Simplicial_complex)
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
//! let hull = ConvexHull::from_triangulation(dt.as_triangulation()).unwrap();
//!
//! println!("Convex hull has {} facets in {}D", hull.number_of_facets(), hull.dimension());
//!
//! // Test point containment
//! let inside_point = Point::new([1.0, 0.5, 0.5]);
//! let outside_point = Point::new([3.0, 3.0, 3.0]);
//!
//! assert!(!hull.is_point_outside(&inside_point, dt.as_triangulation()).unwrap());  // Inside the hull
//! assert!(hull.is_point_outside(&outside_point, dt.as_triangulation()).unwrap());   // Outside the hull
//!
//! // Find visible facets from an external point (useful for incremental construction)
//! let visible_facets = hull.find_visible_facets(&outside_point, dt.as_triangulation()).unwrap();
//! println!("Point sees {} out of {} facets", visible_facets.len(), hull.number_of_facets());
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
//! let hull_4d = ConvexHull::from_triangulation(dt_4d.as_triangulation()).unwrap();
//!
//! assert_eq!(hull_4d.number_of_facets(), 5);  // 4-simplex has 5 boundary facets
//! assert_eq!(hull_4d.dimension(), 4);     // 4D convex hull
//! ```
//!
//! # Triangulation invariants and validation hierarchy
//!
//! The crate is organized as a small **validation stack**, where each layer adds additional
//! invariants on top of the preceding one:
//!
//! - [`Tds`](crate::core::triangulation_data_structure::Tds) (Triangulation Data Structure)
//!   stores the **combinatorial / structural** representation.
//!   Level 2 (structural) validation checks invariants such as:
//!   - **Vertex mappings** – every vertex UUID has a corresponding key and vice versa.
//!   - **Cell mappings** – every cell UUID has a corresponding key and vice versa.
//!   - **No duplicate cells** – no two maximal cells share the same vertex set.
//!   - **Facet sharing** – each facet is shared by at most 2 cells (1 on the boundary, 2 in the interior).
//!   - **Neighbor consistency** – neighbor relationships are mutual and reference a shared facet.
//!
//!   These checks are surfaced via [`Tds::is_valid`](crate::core::triangulation_data_structure::Tds::is_valid)
//!   (structural only) and [`Tds::validate`](crate::core::triangulation_data_structure::Tds::validate)
//!   (Levels 1–2, elements + structural). For cumulative diagnostics across the full stack,
//!   use [`DelaunayTriangulation::validation_report`](core::delaunay_triangulation::DelaunayTriangulation::validation_report).
//!
//! - [`Triangulation`](crate::core::triangulation::Triangulation) builds on the TDS and validates
//!   **manifold topology**.
//!   Level 3 (topology) validation is performed by
//!   [`Triangulation::is_valid`](crate::core::triangulation::Triangulation::is_valid) (Level 3 only) and
//!   [`Triangulation::validate`](crate::core::triangulation::Triangulation::validate) (Levels 1–3), which:
//!   - Strengthens facet sharing to the **manifold facet property**: each facet belongs to
//!     exactly 1 cell (boundary) or exactly 2 cells (interior).
//!   - Checks the **Euler characteristic** of the triangulation (using the topology module).
//!
//! - [`DelaunayTriangulation`](crate::core::delaunay_triangulation::DelaunayTriangulation) builds on
//!   `Triangulation` and validates the **geometric** Delaunay condition.
//!   Level 4 (Delaunay property) validation is performed by
//!   [`DelaunayTriangulation::is_valid`](core::delaunay_triangulation::DelaunayTriangulation::is_valid) (Level 4 only) and
//!   [`DelaunayTriangulation::validate`](core::delaunay_triangulation::DelaunayTriangulation::validate) (Levels 1–4).
//!   Construction is designed to satisfy the Delaunay property, but in rare cases it may be violated for
//!   near-degenerate inputs (see [Issue #120](https://github.com/acgetchell/delaunay/issues/120)).
//!
//! ## Validation
//!
//! The crate exposes four validation levels (element → structural → manifold → Delaunay). The
//! canonical guide (when to use each level, complexity, examples, troubleshooting) lives in
//! `docs/validation.md`:
//! <https://github.com/acgetchell/delaunay/blob/main/docs/validation.md>
//!
//! In brief:
//! - Level 2 (structural / `Tds`): `dt.tds().is_valid()` for a quick check, or `dt.tds().validate()` for
//!   Levels 1–2.
//! - Level 3 (topology / `Triangulation`): `dt.as_triangulation().is_valid()` for topology-only checks, or
//!   `dt.as_triangulation().validate()` for Levels 1–3.
//! - Level 4 (Delaunay / `DelaunayTriangulation`): `dt.is_valid()` for the empty-circumsphere property, or
//!   `dt.validate()` for Levels 1–4.
//! - Full diagnostics: `dt.validation_report()` returns all violated invariants across Levels 1–4.
//!
//! ### Automatic topology validation during insertion (`ValidationPolicy`)
//!
//! In addition to explicit validation calls, incremental construction (`new()` / `insert*()`) can run an
//! automatic **Level 3** topology validation pass after insertion, controlled by
//! [`ValidationPolicy`](crate::core::triangulation::ValidationPolicy).
//!
//! The default is [`ValidationPolicy::OnSuspicion`](crate::core::triangulation::ValidationPolicy::OnSuspicion):
//! Level 3 validation runs only when insertion takes a suspicious path (e.g. perturbation retries,
//! repair loops, or neighbor-pointer repairs that actually changed pointers).
//!
//! This automatic pass only runs Level 3 (`Triangulation::is_valid()`). It does **not** run Level 4.
//!
//! ```rust
//! use delaunay::prelude::*;
//! # let vertices = vec![
//! #     vertex!([0.0, 0.0, 0.0]),
//! #     vertex!([1.0, 0.0, 0.0]),
//! #     vertex!([0.0, 1.0, 0.0]),
//! #     vertex!([0.0, 0.0, 1.0]),
//! # ];
//! let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
//!
//! // Default:
//! assert_eq!(dt.validation_policy(), ValidationPolicy::OnSuspicion);
//!
//! // Tests/debugging:
//! dt.set_validation_policy(ValidationPolicy::Always);
//!
//! // Max performance (you can still validate explicitly when desired):
//! dt.set_validation_policy(ValidationPolicy::Never);
//! ```
//!
//! ### Choosing Level 3 topology guarantee (`TopologyGuarantee`)
//!
//! Level 3 topology validation is parameterized by
//! [`TopologyGuarantee`](crate::core::triangulation::TopologyGuarantee). This is separate from
//! `ValidationPolicy`: it controls *what* invariants Level 3 enforces, not *when* automatic
//! validation runs.
//!
//! - [`TopologyGuarantee::PLManifold`](crate::core::triangulation::TopologyGuarantee::PLManifold)
//!   (default): facet degree + closed boundary + connectedness + isolated-vertex + Euler characteristic
//!   checks **plus** strict vertex-link validation.
//! - [`TopologyGuarantee::Pseudomanifold`](crate::core::triangulation::TopologyGuarantee::Pseudomanifold):
//!   skips vertex-link validation (may be faster), but bistellar flip convergence is not guaranteed and
//!   you may want to validate the Delaunay property explicitly for near-degenerate inputs.
//!
//! ```rust
//! use delaunay::prelude::*;
//! # let vertices = vec![
//! #     vertex!([0.0, 0.0, 0.0]),
//! #     vertex!([1.0, 0.0, 0.0]),
//! #     vertex!([0.0, 1.0, 0.0]),
//! #     vertex!([0.0, 0.0, 1.0]),
//! # ];
//! let mut dt: DelaunayTriangulation<_, (), (), 3> =
//!     DelaunayTriangulation::new(&vertices).unwrap();
//!
//! assert_eq!(dt.topology_guarantee(), TopologyGuarantee::PLManifold);
//!
//! // Optional: relax topology checks for speed (weaker guarantees).
//! dt.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);
//!
//! // Now Level 3 skips vertex-link validation.
//! dt.as_triangulation().is_valid().unwrap();
//! ```
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
//! let dt = DelaunayTriangulation::new(&vertices).unwrap();
//! assert!(dt.tds().is_valid().is_ok());
//! assert!(dt.as_triangulation().is_valid().is_ok());
//! assert!(dt.is_valid().is_ok());
//! assert!(dt.validate().is_ok());
//! ```
//!
//! For implementation details on invariant enforcement, see [`core::algorithms::incremental_insertion`].
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
//! 1. **Successful insertions are designed to maintain all invariants** - If insertion succeeds
//!    (`Ok(_)`), the triangulation is expected to satisfy all structural and topological invariants.
//!    The incremental cavity-based insertion algorithm is designed to maintain these invariants.
//!    For applications requiring strict guarantees, use `DelaunayTriangulation::validate()` (Levels 1–4)
//!    or `DelaunayTriangulation::is_valid()` (Level 4 only) to verify the Delaunay property.
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
//!    [`InsertionError::DuplicateCoordinates`](core::algorithms::incremental_insertion::InsertionError::DuplicateCoordinates)
//!    or [`InsertionError::DuplicateUuid`](core::algorithms::incremental_insertion::InsertionError::DuplicateUuid),
//!    preventing numerical instabilities.
//!
//! When constructing a triangulation from a batch of vertices using
//! [`DelaunayTriangulation::new`](core::delaunay_triangulation::DelaunayTriangulation::new):
//!
//! - Successful construction yields a triangulation that is designed to satisfy the Delaunay property.
//!   Use `dt.validate()` (Levels 1–4) for cumulative verification.
//! - Duplicate coordinates are automatically detected and rejected.
//!
//! Incremental construction via [`DelaunayTriangulation::insert`](core::delaunay_triangulation::DelaunayTriangulation::insert)
//! follows the same invariant rules on each insertion: on success the triangulation remains
//! structurally valid; on failure the data structure is rolled back to its previous state.
//! Use `DelaunayTriangulation::is_valid()` (Level 4) if you need explicit verification of the Delaunay property.
//!
//! ## Incremental insertion algorithm
//!
//! Triangulations are built using an efficient incremental cavity-based insertion algorithm:
//!
//! - **Initial simplex construction** - The first D+1 affinely independent vertices are used
//!   to create an initial valid simplex using robust orientation predicates. If no
//!   non-degenerate simplex can be formed, construction fails with
//!   [`TriangulationConstructionError::GeometricDegeneracy`](core::triangulation::TriangulationConstructionError::GeometricDegeneracy).
//!
//! - **Incremental insertion** - Each subsequent vertex is inserted using a cavity-based
//!   algorithm that:
//!   1. Locates the vertex using efficient point location
//!   2. Identifies conflicting cells (those whose circumsphere contains the new vertex)
//!   3. Removes conflicting cells to create a cavity
//!   4. Fills the cavity with new cells connecting the cavity boundary to the new vertex
//!   5. Wires neighbor relationships locally without global recomputation
//!
//! The incremental insertion algorithm maintains all structural invariants throughout construction,
//! and is designed to satisfy the Delaunay (empty-circumsphere) property. Vertices are only rejected
//! if they would violate fundamental geometric constraints (duplicates, near-duplicates, or degenerate
//! configurations).
//!
//! ## Delaunay validation
//!
//! The incremental insertion algorithm is designed to maintain the Delaunay property,
//! aiming to ensure that the empty circumsphere property holds after each insertion.
//! Global Delaunay validation can be performed explicitly using
//! [`DelaunayTriangulation::is_valid`](core::delaunay_triangulation::DelaunayTriangulation::is_valid)
//! when verification is needed (see [Issue #120](https://github.com/acgetchell/delaunay/issues/120)
//! for rare edge cases where validation may be necessary).
//!
//! For construction from a batch of vertices using
//! [`DelaunayTriangulation::new`](core::delaunay_triangulation::DelaunayTriangulation::new),
//! the resulting triangulation is constructed to satisfy the Delaunay property. Call
//! `DelaunayTriangulation::is_valid()` if you need explicit verification.
//!
//! ## Error handling
//!
//! The incremental insertion algorithm provides clear error reporting for vertices that
//! cannot be inserted:
//!
//! - **Duplicate detection** - Exact and near-duplicate vertices are detected and rejected
//!   with [`InsertionError::DuplicateCoordinates`](core::algorithms::incremental_insertion::InsertionError::DuplicateCoordinates)
//!   or [`InsertionError::DuplicateUuid`](core::algorithms::incremental_insertion::InsertionError::DuplicateUuid)
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
//! assert!(dt.validate().is_ok());
//! ```
//!
//! ### Degenerate input handling
//!
//! When the input vertices cannot form a non-degenerate simplex (for example, when all points
//! are collinear in 2D), construction fails during initial simplex construction with
//! [`TriangulationConstructionError::GeometricDegeneracy`](core::triangulation::TriangulationConstructionError::GeometricDegeneracy).
//! This occurs because degenerate simplices (collinear in 2D, coplanar in 3D, etc.) are detected
//! early using robust orientation predicates before any topology is built.
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
//! // Collinear points fail during initial simplex construction due to degeneracy
//! assert!(matches!(
//!     result,
//!     Err(DelaunayTriangulationConstructionError::Triangulation(
//!         TriangulationConstructionError::GeometricDegeneracy { .. },
//!     ))
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

    pub mod adjacency;
    pub mod boundary;
    pub mod cell;
    /// High-performance collection types optimized for computational geometry operations.
    ///
    /// This module provides centralized type aliases for performance-critical data structures
    /// used throughout the delaunay triangulation library. These aliases allow for easy
    /// future optimization and maintenance by providing a single location to change
    /// the underlying implementation.
    ///
    /// # Performance Rationale
    ///
    /// The type aliases in this module are optimized based on the specific usage patterns
    /// in computational geometry algorithms:
    ///
    /// ## Hash-based Collections
    ///
    /// - **FastHashMap/FastHashSet**: Uses `FastHasher`, a non-cryptographic hasher
    ///   that is 2-3x faster than `SipHash` for trusted data. Perfect for internal data
    ///   where collision resistance against adversarial input is not required.
    ///
    /// ### ⚠️ Security Warning: `DoS` Resistance
    ///
    /// **The hasher used in these collections is NOT DoS-resistant.** It should only be
    /// used with trusted input data. Do not use `FastHashMap` or `FastHashSet` with
    /// attacker-controlled keys, as this could lead to hash collision attacks that
    /// degrade performance to O(n) worst-case behavior.
    ///
    /// **Safe usage patterns:**
    /// - Internal geometric computations with generated/computed keys
    /// - Trusted coordinate data from known sources
    /// - UUID-based keys generated by the library itself
    ///
    /// **Unsafe usage patterns:**
    /// - Processing untrusted coordinate data from external sources
    /// - Using user-provided keys without validation
    /// - Network-facing applications with external input
    ///
    /// ## Small Collections
    ///
    /// - **`SmallVec`**: Uses stack allocation for small collections, avoiding heap
    ///   allocations for the common case where collections remain small. This is
    ///   particularly effective for:
    ///   - Vertex neighbor lists (typically D+1 neighbors)
    ///   - Facet-to-cell mappings (typically 1-2 cells per facet)
    ///   - Temporary collections during geometric operations
    ///
    /// # Usage Patterns
    ///
    /// The size parameters for `SmallVec` are chosen based on empirical analysis of
    /// typical triangulation patterns:
    ///
    /// - **2 elements**: Facet sharing (boundary facets = 1 cell, interior facets = 2 cells)
    /// - **4 elements**: Small temporary collections during geometric operations
    /// - **8 elements**: Vertex degrees and cell neighbor counts in typical triangulations
    /// - **16 elements**: Larger temporary buffers for batch operations
    ///
    /// # Future Optimization
    ///
    /// This centralized approach allows for easy experimentation with different
    /// high-performance data structures:
    /// - Alternative hash functions (ahash, seahash)
    /// - Specialized geometric data structures
    /// - SIMD-optimized containers
    /// - Memory pool allocators
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::collections::{FastHashMap, FacetToCellsMap, SmallBuffer};
    ///
    /// // Use optimized HashMap for temporary mappings
    /// let mut temp_map: FastHashMap<u64, usize> = FastHashMap::default();
    ///
    /// // Use stack-allocated buffer for small collections
    /// let mut small_list: SmallBuffer<i32, 8> = SmallBuffer::new();
    /// small_list.push(1);
    /// small_list.push(2);
    ///
    /// // Use domain-specific optimized collections
    /// let facet_map: FacetToCellsMap = FacetToCellsMap::default();
    /// ```
    ///
    /// ## Phase 1 Migration: Key-Based Internal Operations
    ///
    /// Phase 1 of the UUID-to-Key migration provides optimized collections for internal operations:
    ///
    /// ```rust
    /// use delaunay::core::collections::{CellKeySet, KeyBasedCellMap, VertexKeySet};
    ///
    /// // Phase 1: Direct key-based collections for internal algorithms
    /// let mut internal_cells: CellKeySet = CellKeySet::default();
    /// let mut internal_vertices: VertexKeySet = VertexKeySet::default();
    /// let mut key_mappings: KeyBasedCellMap<String> = KeyBasedCellMap::default();
    /// ```
    pub mod collections {
        mod aliases;
        mod buffers;
        mod helpers;
        mod key_maps;
        mod secondary_maps;
        mod triangulation_maps;

        pub(crate) mod spatial_hash_grid;

        pub(crate) use aliases::StorageMap;
        pub use aliases::{
            Entry, FacetIndex, FastBuildHasher, FastHashMap, FastHashSet, FastHasher,
            MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer, Uuid,
        };

        pub use buffers::*;
        pub use helpers::*;
        pub use key_maps::*;
        pub use secondary_maps::*;
        pub use triangulation_maps::*;
    }
    /// Delaunay triangulation layer with incremental insertion - Phase 3 TODO
    pub mod delaunay_triangulation;
    pub mod edge;
    pub mod facet;
    /// Semantic classification and telemetry for topological operations
    pub mod operations;
    /// Generic triangulation combining kernel + Tds - Phase 2 TODO
    pub mod triangulation;
    pub mod triangulation_data_structure;

    /// General utility functions organized by functionality.
    pub mod util {
        pub mod deduplication;
        pub mod delaunay_validation;
        pub mod facet_keys;
        pub mod facet_utils;
        pub mod hashing;
        pub mod hilbert;
        pub mod jaccard;
        pub mod measurement;
        pub mod uuid;

        // Re-export public items for ergonomic `crate::core::util::*` access.
        pub use deduplication::*;
        pub use delaunay_validation::*;
        pub use facet_keys::*;
        pub use facet_utils::*;
        pub use hashing::*;
        pub use hilbert::*;
        pub use jaccard::*;
        pub use measurement::*;
        pub use uuid::*;
    }

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
    pub use adjacency::*;
    pub use cell::*;
    pub use delaunay_triangulation::*;
    pub use edge::*;
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
    #[macro_use]
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

/// Topology analysis and validation for triangulated spaces.
///
/// This module provides traits, algorithms, and data structures for analyzing
/// and validating the topological properties of triangulations.
///
/// # Features
///
/// - **Euler Characteristic Calculation**: Compute topological invariants
/// - **Topology Classification**: Classify triangulations (Ball, Sphere, etc.)
/// - **Validation Framework**: Verify triangulation topological correctness
/// - **Dimensional Generic**: Works across all supported dimensions
///
/// # Applicability
///
/// These tools work for **any triangulation** (not just Delaunay triangulations).
/// The Euler characteristic and topological properties are combinatorial invariants
/// that depend only on the connectivity structure, not on geometric properties.
///
/// # Example
///
/// ```rust
/// use delaunay::prelude::*;
/// use delaunay::topology::characteristics::validation;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let dt = DelaunayTriangulation::new(&vertices).unwrap();
///
/// let result = validation::validate_triangulation_euler(dt.tds()).unwrap();
/// assert_eq!(result.chi, 1);  // Tetrahedron has χ = 1
/// assert!(result.is_valid());
/// ```
pub mod topology {
    /// Topology editing operations (bistellar flips).
    pub mod edit;
    /// Traits for topological spaces and error types
    pub mod traits {
        pub mod topological_space;
        pub use topological_space::*;
    }
    /// Topological invariants and their computation
    pub mod characteristics {
        pub mod euler;
        pub mod validation;
        pub use euler::*;
        pub use validation::*;
    }

    /// Manifold / simplicial-complex validity checks (topology-only).
    pub mod manifold;

    /// Concrete topological space implementations (future work).
    ///
    /// This module will contain specialized implementations for different
    /// topological spaces (Euclidean, spherical, toroidal) once the
    /// [`TopologicalSpace`] trait is stabilized.
    pub mod spaces {
        /// Euclidean space topology
        pub mod euclidean;
        /// Spherical space topology
        pub mod spherical;
        /// Toroidal space topology
        pub mod toroidal;

        pub use euclidean::EuclideanSpace;
        pub use spherical::SphericalSpace;
        pub use toroidal::ToroidalSpace;
    }

    // Re-export commonly used types
    pub use crate::core::triangulation::TopologyGuarantee;
    pub use characteristics::*;
    pub use manifold::{
        ManifoldError, validate_closed_boundary, validate_facet_degree, validate_ridge_links,
        validate_vertex_links,
    };
    pub use traits::*;

    /// Prelude modules for topology editing and validation.
    pub mod prelude {
        /// High-level topology edit API (bistellar flips).
        pub mod edit {
            pub use crate::topology::edit::*;
        }

        /// Topology validation & analysis utilities.
        pub mod validation {
            pub use crate::core::triangulation::TopologyGuarantee;
            pub use crate::topology::characteristics::*;
            pub use crate::topology::manifold::{
                ManifoldError, validate_closed_boundary, validate_facet_degree,
                validate_ridge_links, validate_vertex_links,
            };
            pub use crate::topology::traits::*;
        }
    }
}

/// A prelude module that re-exports commonly used types and macros.
/// This makes it easier to import the most commonly used items from the crate.
pub mod prelude {
    // Re-export from core
    pub use crate::core::{
        adjacency::*,
        cell::*,
        delaunay_triangulation::*,
        edge::*,
        facet::*,
        traits::{boundary_analysis::*, data_type::*},
        triangulation::*,
        triangulation_data_structure::*,
        vertex::*,
    };

    // Re-export utility items, but avoid exporting the util module names themselves.
    //
    // In particular, exporting `core::util::uuid` as `uuid` conflicts with the external `uuid`
    // crate name, making `use uuid::Uuid;` ambiguous for downstream users.
    pub use crate::core::util::{
        deduplication::*, delaunay_validation::*, facet_keys::*, facet_utils::*, hashing::*,
        hilbert::*, jaccard::*, measurement::*, uuid::*,
    };

    // Re-export point location algorithms from core::algorithms
    pub use crate::core::algorithms::locate::{
        LocateError, LocateFallback, LocateFallbackReason, LocateResult, LocateStats, locate,
        locate_with_stats,
    };

    // Re-export incremental insertion types
    pub use crate::core::algorithms::incremental_insertion::InsertionError;
    pub use crate::core::operations::{InsertionOutcome, InsertionStatistics, SuspicionFlags};

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

    /// Convenience re-exports for common **read-only** workflows (topology traversal, adjacency,
    /// convex-hull extraction, and common input types).
    ///
    /// This is useful if you want a smaller import surface than `delaunay::prelude::*`,
    /// while still having access to the key public APIs typically used in docs/tests/examples/benches.
    ///
    /// Note: `query` currently also re-exports a few helpers commonly used in
    /// docs/tests/examples/benches (e.g., random generators). If this grows over time, it may be
    /// split into more focused modules (e.g., `prelude::generators`).
    ///
    /// Includes:
    /// - Topology traversal: [`DelaunayTriangulation::edges`], [`DelaunayTriangulation::incident_edges`],
    ///   [`DelaunayTriangulation::cell_neighbors`]
    /// - Fast repeated queries: [`DelaunayTriangulation::build_adjacency_index`] and [`AdjacencyIndex`]
    /// - Zero-allocation geometry accessors: [`DelaunayTriangulation::vertex_coords`],
    ///   [`DelaunayTriangulation::cell_vertices`]
    /// - Convex hull extraction: [`ConvexHull::from_triangulation`]
    /// - Test/example helpers: [`generate_random_triangulation`], [`generate_random_points_seeded`]
    pub mod query {
        // Core read-only traversal / adjacency
        pub use crate::core::adjacency::{AdjacencyIndex, AdjacencyIndexBuildError};
        pub use crate::core::delaunay_triangulation::DelaunayTriangulation;
        pub use crate::core::edge::EdgeKey;
        pub use crate::core::triangulation::Triangulation;
        pub use crate::core::triangulation_data_structure::{CellKey, VertexKey};

        // Common input/output types (kept intentionally small)
        pub use crate::core::facet::FacetView;
        pub use crate::core::traits::boundary_analysis::BoundaryAnalysis;
        pub use crate::core::traits::data_type::DataType;
        pub use crate::core::{Cell, Vertex};
        pub use crate::geometry::Point;
        pub use crate::geometry::kernel::{FastKernel, Kernel};
        pub use crate::geometry::traits::coordinate::Coordinate;

        // Read-only predicates (useful in benchmarks / lightweight geometry checks)
        pub use crate::geometry::{insphere, insphere_distance, insphere_lifted};

        // Read-only algorithms
        pub use crate::geometry::algorithms::convex_hull::ConvexHull;

        // Convenience generators (commonly used in docs/tests/examples/benches)
        pub use crate::geometry::util::{
            generate_random_points_seeded, generate_random_triangulation,
        };

        // Instrumentation helpers (no-op unless features enable extra tracking)
        pub use crate::core::util::measure_with_result;

        // Convenience macro (commonly used in docs/tests/examples) without importing full `prelude::*`.
        pub use crate::vertex;
    }
    /// Topology edit API (bistellar flips).
    pub mod edit {
        pub use crate::topology::prelude::edit::*;
    }

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
        core::{
            adjacency::AdjacencyIndex, cell::Cell, delaunay_triangulation::DelaunayTriangulation,
            edge::EdgeKey, triangulation::Triangulation, triangulation_data_structure::Tds,
            vertex::Vertex,
        },
        geometry::{Point, algorithms::convex_hull::ConvexHull, kernel::FastKernel},
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
        assert!(is_normal::<Triangulation<FastKernel<f64>, (), (), 3>>());
        assert!(is_normal::<DelaunayTriangulation<FastKernel<f64>, (), (), 3>>());
        assert!(is_normal::<ConvexHull<FastKernel<f64>, (), (), 3>>());
        assert!(is_normal::<EdgeKey>());
        assert!(is_normal::<AdjacencyIndex>());
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
        let ratio = radius_ratio(dt.as_triangulation(), cell_key).unwrap();
        assert!(ratio > 0.0);

        let norm_vol = normalized_volume(dt.as_triangulation(), cell_key).unwrap();
        assert!(norm_vol > 0.0);
    }

    #[test]
    fn test_prelude_kernel_exports() {
        use crate::prelude::*;

        // Test that kernel types and predicates are accessible from prelude
        let fast_kernel = FastKernel::<f64>::new();
        let robust_kernel = RobustKernel::<f64>::new();

        // Test 2D orientation predicate
        let triangle = [
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.0, 1.0]),
        ];

        let fast_orientation = fast_kernel.orientation(&triangle).unwrap();
        assert_ne!(fast_orientation, 0, "Triangle should be non-degenerate");

        let robust_orientation = robust_kernel.orientation(&triangle).unwrap();
        assert_eq!(
            fast_orientation, robust_orientation,
            "Both kernels should agree"
        );

        // Test collinear detection
        let collinear = [
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([2.0, 0.0]),
        ];
        assert_eq!(
            fast_kernel.orientation(&collinear).unwrap(),
            0,
            "Collinear points should have zero orientation"
        );

        // Test in_sphere predicate
        let inside_point = Point::new([0.25, 0.25]);
        let result = fast_kernel.in_sphere(&triangle, &inside_point).unwrap();
        assert_eq!(result, 1, "Point should be inside circumcircle");

        let outside_point = Point::new([2.0, 2.0]);
        let result = fast_kernel.in_sphere(&triangle, &outside_point).unwrap();
        assert_eq!(result, -1, "Point should be outside circumcircle");
    }

    #[test]
    fn test_prelude_core_types() {
        use crate::prelude::*;

        // Test that core types are accessible and work from prelude
        // Point construction
        let p1 = Point::new([0.0, 0.0, 0.0]);
        let p2 = Point::new([1.0, 0.0, 0.0]);
        assert_ne!(p1, p2);

        // Vertex construction via macro and builder
        let v1: Vertex<f64, (), 3> = vertex!([0.0, 0.0, 0.0]);
        let v2: Vertex<f64, (), 3> = vertex!([1.0, 0.0, 0.0]);
        assert_ne!(v1.point(), v2.point());

        // DelaunayTriangulation construction
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
        assert_eq!(dt.number_of_vertices(), 4);
        assert_eq!(dt.number_of_cells(), 1);

        // Access Triangulation, Tds, Cell types
        let tri = dt.as_triangulation();
        assert_eq!(tri.number_of_vertices(), 4);

        let tds = &tri.tds;
        assert_eq!(tds.number_of_cells(), 1);

        // Iterate over cells
        for (cell_key, _cell) in tri.cells() {
            assert!(tds.get_cell(cell_key).is_some());
        }
    }

    #[test]
    fn test_prelude_point_location() {
        use crate::prelude::*;

        // Test that point location algorithms are accessible
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // Test locate function with kernel
        let kernel = FastKernel::<f64>::new();
        let query_point = Point::new([0.3, 0.3]);
        let result = locate(dt.tds(), &kernel, &query_point, None);
        assert!(result.is_ok());

        // Result should be a LocateResult
        match result.unwrap() {
            LocateResult::InsideCell(_)
            | LocateResult::OnFacet { .. }
            | LocateResult::OnEdge { .. }
            | LocateResult::OnVertex(_) => { /* expected or acceptable */ }
            LocateResult::Outside => panic!("Point should be inside triangulation"),
        }

        // Test outside point
        let outside_point = Point::new([10.0, 10.0]);
        let result = locate(dt.tds(), &kernel, &outside_point, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_prelude_geometry_types() {
        use crate::prelude::*;

        // Test Point with Coordinate trait
        let p = Point::new([1.0_f64, 2.0_f64, 3.0_f64]);
        assert!((p.coords()[0] - 1.0_f64).abs() < f64::EPSILON);
        assert!((p.coords()[1] - 2.0_f64).abs() < f64::EPSILON);
        assert!((p.coords()[2] - 3.0_f64).abs() < f64::EPSILON);

        // Test predicates are accessible
        let triangle = [
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.0, 1.0]),
        ];

        // simplex_orientation is exported from predicates
        let orientation = simplex_orientation(&triangle).unwrap();
        assert_ne!(orientation, Orientation::DEGENERATE);

        // Test insphere predicate
        let test_point = Point::new([0.25, 0.25]);
        let result = insphere(&triangle, test_point).unwrap();
        assert_eq!(result, InSphere::INSIDE);
    }

    #[test]
    fn test_prelude_convex_hull() {
        use crate::prelude::*;

        // Test that convex hull operations are accessible
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();

        // ConvexHull type should be accessible
        let hull = ConvexHull::from_triangulation(dt.as_triangulation()).unwrap();
        assert_eq!(hull.number_of_facets(), 4); // Tetrahedron has 4 faces

        // Test point visibility
        let outside_point = Point::new([2.0, 2.0, 2.0]);
        let is_outside = hull
            .is_point_outside(&outside_point, dt.as_triangulation())
            .unwrap();
        assert!(is_outside);

        let inside_point = Point::new([0.25, 0.25, 0.25]);
        let is_outside = hull
            .is_point_outside(&inside_point, dt.as_triangulation())
            .unwrap();
        assert!(!is_outside);
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
